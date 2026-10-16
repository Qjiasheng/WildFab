from utils import field_query_with_grads, field_querying
import time
import torch
from torch.nn import functional as F
from vis import vis_grad, vis_pcd_fields
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import Voronoi, ConvexHull
from sklearn.decomposition import PCA
import numpy as np
from joblib import Parallel, delayed
import octree_nt_clouds_wn


def load_pcd_update(pcd_fn, n_neighbors=8, remove_duplicates=True, cut_idx=None):
    """estimate areas after cutting"""

    pcd = np.genfromtxt(pcd_fn)
    coords = pcd[:, :3]
    normals = pcd[:, 3:6]
    print(f'Loaded {coords.shape[0]} waypoints from {pcd_fn}')

    if cut_idx is not None:
        coords = coords[:int(cut_idx)]
        normals = normals[:int(cut_idx)]
        print(f'After cutting, {coords.shape[0]} waypoints remain.')

    if remove_duplicates:
        coords_unique, unique_indices = np.unique(coords, axis=0, return_index=True)
        order = np.argsort(unique_indices)          # restore original order
        coords  = coords_unique[order]
        normals = normals[unique_indices[order]]
        print(f'After removing duplicates, {coords.shape[0]} waypoints remain.')

        # if dist < 1e-6, consider as duplicate
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        duplicate_mask = distances[:, 1] < 1e-6
        non_duplicate_indices = np.where(~duplicate_mask)[0]
        coords = coords[non_duplicate_indices]
        normals = normals[non_duplicate_indices]
        print(f'After removing near-duplicates, {coords.shape[0]} waypoints remain.')

    # build areas
    areas = estimate_areas(coords, n_neighbors=n_neighbors)

    vis_grad(coords, normals, arrow_scale=0.03)
    vis_pcd_fields(coords, areas, label='KNN Voronoi Area')

    return coords, normals, areas


def estimate_areas(points, n_neighbors=8):
    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(points)

    indices = knn.kneighbors(return_distance=False)

    def worker(i):
        pca = PCA(2)
        idx = indices[i]
        idx = np.insert(idx, 0, i)

        nbs = points[idx]
        nbs_new = pca.fit_transform(nbs)

        try:
            vor = Voronoi(nbs_new)
            reg = vor.regions[vor.point_region[0]]
            if -1 in reg:
                return 0.
            hull = ConvexHull(vor.vertices[reg])
            area = hull.volume
        except Exception as e:
            return 0.

        return area

    print(f'Estimating areas for {len(points)} points')
    areas = Parallel(32)(delayed(worker)(i) for i in range(len(points)))
    areas = np.array(areas)

    zero_area_indices = np.where(areas == 0)[0]
    for i in zero_area_indices:
        neighbor_indices = indices[i]
        non_zero_area_indices = np.where(areas[neighbor_indices] != 0)[0]
        if len(non_zero_area_indices) == 0:
            continue
        mean_area = np.mean(areas[neighbor_indices[non_zero_area_indices]])
        areas[i] = mean_area

    perc_98 = np.percentile(areas, 98)
    areas = np.clip(areas, 0., perc_98)

    return areas.reshape(-1, 1).squeeze()


def tracing_min_udf_level(
    udf_decoder,
    coords,                 # (B,3)
    max_iters       = 8,
    ray_sample_num  = 32,
    max_init_step   = 0.05,
    eps_hit         = 1.0e-5
):
    """
    Ray-tracing that repeatedly samples along the gradient-descent
    direction of the UDF and jumps to the sample with the smallest
    |UDF|.  The distance for every ray is *always* the distance from
    the start point to its current best-UDF point, not only for hits.

    Returns
    -------
    P_best : (B,3)  point with the overall smallest |UDF| found
    dists  : (B,)   Euclidean distance ||P_best - P0||
    hit    : (B,)   bool, True if |UDF|<eps_hit was ever reached
    """

    # --------------------------------------------------------- helpers
    def _query_udf(dec, pts):
        _, v, g = field_query_with_grads(
            dec, pts.clone().detach(), no_print=True, return_on_cuda=True
        )
        return v.detach(), g.detach()

    # --------------------------------------------------------- init
    device, dtype = coords.device, coords.dtype
    B             = coords.shape[0]

    udf0, grad0   = _query_udf(udf_decoder, coords)
    ray_dir       = F.normalize(-grad0, dim=-1, eps=1e-12)          # march ↓UDF
    step          = torch.clamp(2.0 * udf0.abs(), 1e-3, max_init_step)

    P0            = coords.clone()
    P_best        = coords.clone()                                  # current best
    best_udf      = udf0.abs()                                      # (B,)

    hit           = best_udf < eps_hit                              # early hits
    dists         = torch.zeros_like(best_udf)                      # to be filled

    t_lin = torch.linspace(-0.5, 1.0, ray_sample_num,
                           device=device, dtype=dtype)              # (S,)

    torch.cuda.synchronize(); t0 = time.time()

    # ========================================================= loop
    for _ in range(max_iters):
        active = ~hit
        if not active.any():
            break

        # sample ----------------------------------------------------------------
        P_smpl = (
            P_best[active, None, :] +
            ray_dir[active, None, :] * (step[active, None, None] * t_lin[None, :, None])
        )                                                           # (A,S,3)
        A, S, _ = P_smpl.shape
        udf_smpl, grad_smpl = _query_udf(udf_decoder, P_smpl.reshape(-1, 3))
        udf_smpl  = udf_smpl.view(A, S)
        grad_smpl = grad_smpl.view(A, S, 3)

        # choose minimal |UDF| sample ------------------------------------------
        abs_udf      = udf_smpl.abs()
        idx_min      = abs_udf.argmin(dim=1)                        # (A,)
        new_best_udf = abs_udf[torch.arange(A, device=device), idx_min]
        new_pts      = P_smpl [torch.arange(A, device=device), idx_min]
        new_grads    = grad_smpl[torch.arange(A, device=device), idx_min]

        # overwrite state where improvement is achieved ------------------------
        g_idx                   = torch.nonzero(active, as_tuple=False).squeeze(1)
        improved                = new_best_udf < best_udf[g_idx]

        imp_idx                 = g_idx[improved]
        if imp_idx.numel() > 0:
            P_best  [imp_idx]   = new_pts [improved]
            best_udf[imp_idx]   = new_best_udf[improved]
            ray_dir [imp_idx]   = F.normalize(-new_grads[improved], dim=-1, eps=1e-12)
            step    [imp_idx]   = torch.clamp(2.0 * best_udf[imp_idx], 1e-3, max_init_step)

        # mark hits ------------------------------------------------------------
        hit = hit | (best_udf < eps_hit)

    # --------------------------------------------------------- final distance
    dists = (P_best - P0).norm(dim=-1)

    torch.cuda.synchronize()
    print(f"[I] tracing_min_udf_level: {time.time()-t0:.3f}s  "
          f"({hit.sum().item()}/{B} reached |udf|<{eps_hit})")

    return P_best, dists, hit


def tracing_min_udf_gradnorm(
    udf_decoder,
    coords,                 # (B,3)
    max_iters       = 8,
    ray_sample_num  = 32,
    max_init_step   = 0.05,
    eps_hit         = 1.0e-4
):
    """
    Marches along ∇UDF, but decides the next jump by selecting the
    sample with the *smallest gradient norm* on each iteration.

    A ray is finished when min‖∇UDF‖ < eps_hit.

    Returns
    -------
    P_best : (B,3)   point with the overall smallest grad-norm found
    dists  : (B,)    ||P_best - P0||
    hit    : (B,)    bool, True where min‖∇UDF‖ < eps_hit
    """

    # ----------------------------------------------------- helper
    def _query(dec, pts):
        _, v, g = field_query_with_grads(
            dec, pts.clone().detach(), no_print=True, return_on_cuda=True
        )
        return v.detach(), g.detach()          # udf value, grad

    # ----------------------------------------------------- init
    device, dtype = coords.device, coords.dtype
    B             = coords.shape[0]

    udf0, grad0   = _query(udf_decoder, coords)
    grad_norm0    = grad0.norm(dim=-1)                         # (B,)

    ray_dir       = F.normalize(-grad0, dim=-1, eps=1e-12)     # go ↓UDF
    step          = torch.clamp(2.0 * udf0.abs(), 1e-3, max_init_step)

    P0            = coords.clone()
    P_best        = coords.clone()
    best_grad_norm = grad_norm0.clone()                        # (B,)

    hit           = best_grad_norm < eps_hit
    dists         = torch.zeros_like(best_grad_norm)           # filled later

    t_lin = torch.linspace(-0.5, 1.0, ray_sample_num,
                           device=device, dtype=dtype)

    torch.cuda.synchronize(); t0 = time.time()

    # ===================================================== loop
    for _ in range(max_iters):
        active = ~hit
        if not active.any():
            break

        # ------------------------------------------------ sample points
        P_smpl = (
            P_best[active, None, :] +
            ray_dir[active, None, :] *
            (step[active, None, None] * t_lin[None, :, None])
        )                                                      # (A,S,3)
        A, S, _ = P_smpl.shape

        udf_smpl, grad_smpl = _query(udf_decoder, P_smpl.reshape(-1, 3))
        grad_smpl     = grad_smpl.view(A, S, 3)
        udf_smpl      = udf_smpl.view(A, S)

        grad_norm_smpl = grad_smpl.norm(dim=-1)                # (A,S)

        # pick sample of minimal grad norm -------------------
        idx_min          = grad_norm_smpl.argmin(dim=1)
        new_grad_norm    = grad_norm_smpl[torch.arange(A, device=device), idx_min]
        new_pts          = P_smpl [torch.arange(A, device=device), idx_min]
        new_grads        = grad_smpl[torch.arange(A, device=device), idx_min]
        new_udf_abs      = udf_smpl.abs()[torch.arange(A, device=device), idx_min]

        # update where improvement ---------------------------
        g_idx       = torch.nonzero(active, as_tuple=False).squeeze(1)
        improved    = new_grad_norm < best_grad_norm[g_idx]
        imp_idx     = g_idx[improved]

        if imp_idx.numel() > 0:
            P_best [imp_idx]    = new_pts [improved]
            best_grad_norm[imp_idx] = new_grad_norm[improved]

            # new marching direction – if grad is almost zero, keep old dir
            dir_new = -new_grads[improved]
            zero_mask = dir_new.norm(dim=-1) < 1e-8
            dir_new[zero_mask] = ray_dir[imp_idx][zero_mask]
            ray_dir[imp_idx] = F.normalize(dir_new, dim=-1, eps=1e-12)

            # step size from udf magnitude at the new position
            step[imp_idx] = torch.clamp(2.0 * new_udf_abs[improved], 1e-3, max_init_step)

        # mark hits ------------------------------------------
        hit = hit | (best_grad_norm < eps_hit)

    # ------------------------------------------------ final distance
    dists = (P_best - P0).norm(dim=-1)

    torch.cuda.synchronize()
    print(f"[I] tracing_min_grad_level: {time.time()-t0:.3f}s  "
          f"({hit.sum().item()}/{B} reached ‖∇udf‖<{eps_hit})")

    return P_best, dists, hit


def fast_octree_wn(coords, octree_wn_dict):
    """octree WN query for filtering, return -1 for out of cube queries"""
    # torch.cuda.synchronize()
    # start_time = time.time()
    wns, wn_grads, _ = octree_nt_clouds_wn.batch_query_wn_clouds(coords, 
    octree_wn_dict.get('coords'), octree_wn_dict.get('normals'), 
    octree_wn_dict.get('areas'), octree_wn_dict.get('point_indices'), 
    octree_wn_dict.get('children'), octree_wn_dict.get('cell_centers'),
    octree_wn_dict.get('cell_normals'), octree_wn_dict.get('cell_radii'),
    octree_wn_dict.get('high_inv_delta'))
    # print(f"octree WN query time: {time.time() - start_time:.4f} seconds for {coords.shape[0]} points")
    return wns, wn_grads


def once_tracing_max_wngrad_octree(queries, udf_decoder, octree_wn_dict, ray_sample_num=64, ori_inv_delta=32.0, 
                                   peak_eps=1e-4, max_init_step=0.5, min_step=1e-2, use_quadratic_refine=True, 
                                   dist_valid_mode="ori_delta", wn_gradnorm_valid_th=10.0,):
    """
    One-shot ray sampling + optional local quadratic peak refinement.

    This is a more robust replacement for the 3-point line-search version.
        Revised version:
            Samples `ray_sample_num` points along:
                [-0.5 * step, 1.0 * step]

            Finds the strongest WN-gradient sample.
            Optionally applies one local 3-point quadratic interpolation
            around the best sample and its two neighbors.
    This still performs only one tracing pass. There is no iterative loop.

    Args:
        queries: Tensor, shape (B, 3)
        udf_decoder:
            UDF decoder.
        octree_wn_dict:
            Prebuilt octree WN dictionary used by `fast_octree_wn`.
        ray_sample_num: int
            Number of samples along each ray. Good practical values:
            32, 64, 128. If your peak is very sharp, use 128 or 256.
        ori_inv_delta: float
            Original grid inverse delta.
        peak_eps: float
            Minimum WN-gradient norm threshold.
        max_init_step: float
            Maximum ray interval scale.
        min_step: float
            Minimum ray interval scale.
        use_quadratic_refine: bool
            If True, use one local parabolic interpolation around the best sampled point.

        dist_valid_mode: str
            Controls distance validity check.
            "ori_delta":
                valid if hit_dist < 1.0 / ori_inv_delta.
            "step":
                valid if hit_dist < step.
            "none":
                only require peak_val > peak_eps.

    Returns:
        P_peak: Tensor, shape (B, 3)
        dists:  Tensor, shape (B,)
        valid:  BoolTensor, shape (B,)
    """

    device, dtype = queries.device, queries.dtype
    B = queries.shape[0]

    assert ray_sample_num >= 3, "ray_sample_num must be >= 3 for local quadratic refinement."

    # ------------------------------------------------------------
    # Evaluate UDF once
    # ------------------------------------------------------------
    _, udf_val, udf_grad = field_query_with_grads(udf_decoder, queries.clone(), no_print=True, return_on_cuda=True,)

    udf_val = udf_val.detach()
    udf_grad = udf_grad.detach()

    if udf_val.ndim > 1:
        udf_val = udf_val.squeeze(-1)

    # ------------------------------------------------------------
    # Ray direction
    # ------------------------------------------------------------
    ray_dir = F.normalize(-udf_grad, dim=-1, eps=1e-12,)  # (B, 3)

    # ------------------------------------------------------------
    # Adaptive search step
    # ------------------------------------------------------------
    step = torch.clamp(2.0 * udf_val.abs(), min=min_step, max=max_init_step,)  # (B,)
    t_lin = torch.linspace(-0.5, 1.0, ray_sample_num, device=device, dtype=dtype,)  # (S,)

    S = ray_sample_num
    t_all = step[:, None] * t_lin[None, :]  # (B, S)
    P_smpl = (queries[:, None, :] + ray_dir[:, None, :] * t_all[:, :, None])  # (B, S, 3)
    P_flat = P_smpl.reshape(-1, 3).contiguous()

    # ------------------------------------------------------------
    # Fast WN octree query
    # ------------------------------------------------------------
    _, wn_grads = fast_octree_wn(P_flat, octree_wn_dict,)
    grad_norm = wn_grads.norm(dim=-1).view(B, S)  # (B, S)

    # ------------------------------------------------------------
    # Best sampled point
    # ------------------------------------------------------------
    batch_idx = torch.arange(B, device=device)

    best_idx = grad_norm.argmax(dim=1)  # (B,)

    peak_val_sample = grad_norm[batch_idx, best_idx]  # (B,)
    t_best_sample = t_all[batch_idx, best_idx]        # (B,)
    P_best_sample = P_smpl[batch_idx, best_idx]       # (B, 3)

    # Default output is sampled best.
    t_peak = t_best_sample.clone()
    P_peak = P_best_sample.clone()
    peak_val = peak_val_sample.clone()

    if use_quadratic_refine:
        has_neighbor = (best_idx > 0) & (best_idx < S - 1)

        if has_neighbor.any():
            rows = batch_idx[has_neighbor]
            mids = best_idx[has_neighbor]

            f_left = grad_norm[rows, mids - 1]
            f_mid = grad_norm[rows, mids]
            f_right = grad_norm[rows, mids + 1]

            denom = f_left - 2.0 * f_mid + f_right
            numer = 0.5 * (f_left - f_right)

            eps = torch.finfo(dtype).eps

            # For a valid maximum, parabola should open downward.
            safe = denom.abs() > eps
            opens_down = denom < 0.0

            delta = torch.zeros_like(f_mid)
            valid_quad = safe & opens_down

            delta[valid_quad] = numer[valid_quad] / denom[valid_quad]

            # Avoid extrapolation beyond neighboring samples.
            delta = torch.clamp(delta, min=-1.0, max=1.0)
            # Sample spacing in normalized interval [-0.5, 1.0].
            dt_lin = t_lin[1] - t_lin[0]
            # Convert local offset to actual ray distance.
            t_quad = t_best_sample[has_neighbor] + delta * step[has_neighbor] * dt_lin
            P_quad = queries[has_neighbor] + ray_dir[has_neighbor] * t_quad[:, None]
            # Re-query the quadratic-estimated peak.
            _, wn_grads_quad = fast_octree_wn(P_quad.contiguous(), octree_wn_dict,)
            peak_val_quad = wn_grads_quad.norm(dim=-1)

            # Accept quadratic point only if it improves over sampled best.
            improve = peak_val_quad > peak_val_sample[has_neighbor]

            if improve.any():
                accept_rows = rows[improve]

                P_peak[accept_rows] = P_quad[improve]
                t_peak[accept_rows] = t_quad[improve]
                peak_val[accept_rows] = peak_val_quad[improve]

    # ------------------------------------------------------------
    # Validity and distance
    # ------------------------------------------------------------
    hit_dist = (P_peak - queries).norm(dim=-1)
    found = peak_val > peak_eps

    if dist_valid_mode == "ori_delta":
        dist_ok = hit_dist < 1.0 / ori_inv_delta
    elif dist_valid_mode == "step":
        dist_ok = hit_dist < step
    elif dist_valid_mode == "none":
        dist_ok = torch.ones_like(found, dtype=torch.bool, device=device)
    else:
        raise ValueError(f"Unknown dist_valid_mode: {dist_valid_mode}. " f"Expected 'ori_delta', 'step', or 'none'.")

    # Optionally filter by WN-gradient norm threshold, which can help reject some bad hits.
    if wn_gradnorm_valid_th is not None and wn_gradnorm_valid_th > 0:
        wn_grad_ok = peak_val >= wn_gradnorm_valid_th
    else:
        wn_grad_ok = torch.ones_like(found, dtype=torch.bool, device=device)

    valid = found & dist_ok & wn_grad_ok
    dists = torch.full((B,), float("inf"), device=device, dtype=dtype,)
    dists[valid] = hit_dist[valid]

    return P_peak, dists, valid


def batch_once_project_diff_field(queries, slice_decoder, target_iso_value, max_step=0.6, grad_eps=1e-12, 
                                  damping=1e-3, max_backtracks=8, search_radius=0.05, residual_tol=1e-6, surface_normals=None):
    """
    NOTE USE surface normals to constrain motion to the tangent plane, surface normal use wn gradient
    One safeguarded slice projection; supports scalar or per-point iso-values.
    original newton projection performs poorly on some points with vanishing gradients, 
    Backtracking and local probing handle stalls; surface normals constrain motion
    to the tangent plane so the following surface projection cancels less progress.
    """
    if max_step <= 0 or grad_eps <= 0 or damping < 0 or max_backtracks < 1 or search_radius < 0 or residual_tol < 0:
        raise ValueError("Invalid projection step, damping, search, or tolerance parameter.")
    device, dtype = queries.device, queries.dtype
    B = queries.shape[0]
    if B == 0:
        return queries.clone()

    target = torch.as_tensor(target_iso_value, device=device, dtype=dtype).reshape(-1)
    if target.numel() == 1:
        target = target.expand(B)
    if target.numel() != B:
        raise ValueError(f"target_iso_value must be scalar or have {B} entries.")

    _, field_val, field_grad = field_query_with_grads(slice_decoder, queries.clone(), no_print=True, return_on_cuda=True)
    residual = field_val.detach().reshape(-1) - target
    grad = field_grad.detach()
    normals = None
    if surface_normals is not None:
        normals = torch.as_tensor(surface_normals, device=device, dtype=dtype).detach()
        if normals.shape != queries.shape:
            raise ValueError("surface_normals must have the same shape as queries.")
        normals = torch.where(torch.isfinite(normals).all(dim=-1, keepdim=True), normals, torch.zeros_like(normals))
        normals = F.normalize(normals, dim=-1, eps=grad_eps)
        # Use the surface gradient, not motion normal to the surface that gets undone.
        grad = grad - (grad * normals).sum(dim=-1, keepdim=True) * normals
        if search_radius > 0:
            max_step = min(max_step, search_radius)
    error = residual.abs()
    grad_norm_sq = grad.square().sum(dim=-1)
    delta = -residual[:, None] * grad / ((1.0 + damping) * grad_norm_sq + grad_eps)[:, None]
    delta_norm = delta.norm(dim=-1, keepdim=True)
    delta = delta * torch.clamp(max_step / delta_norm.clamp_min(grad_eps), max=1.0)
    projected = queries.detach().clone()
    best_error = error.clone()
    active = torch.isfinite(error) & torch.isfinite(delta).all(dim=-1) & (error > residual_tol)

    def candidate_error(points, targets):
        values = field_querying(slice_decoder, points.detach().clone(), no_print=True, return_on_cuda=True)
        return (values.detach().reshape(-1) - targets).abs()

    # Backtrack damped steps; locally probe stalled points to escape vanishing gradients.
    for _ in range(max_backtracks):
        indices = torch.where(active)[0]
        if indices.numel() == 0:
            break
        candidates = queries[indices] + delta[indices]
        errors = candidate_error(candidates, target[indices])
        accept = torch.isfinite(errors) & (errors < best_error[indices])
        accepted = indices[accept]
        projected[accepted] = candidates[accept].detach()
        best_error[accepted] = errors[accept]
        active[accepted] = False
        delta[indices[~accept]] *= 0.5

    stalled = torch.isfinite(error) & (best_error > residual_tol) & ((error - best_error) <= 1e-3 * error)
    stalled_indices = torch.where(stalled)[0]
    radius = min(search_radius, max_step)
    if radius > 0 and stalled_indices.numel() > 0:
        axis = torch.tensor([-1., 0., 1.], device=device, dtype=dtype)
        directions = torch.cartesian_prod(axis, axis, axis)
        directions = directions[directions.square().sum(dim=-1) > 0]
        directions = F.normalize(directions, dim=-1)
        # Diagonals also escape saddle points whose coordinate-axis probes are flat.
        for chunk in stalled_indices.split(4096):
            search_directions = directions[None, :, :].expand(len(chunk), -1, -1)
            if normals is not None:
                local_normals = normals[chunk, None, :]
                search_directions = search_directions - (search_directions * local_normals).sum(dim=-1, keepdim=True) * local_normals
                search_directions = F.normalize(search_directions, dim=-1, eps=grad_eps)
            for fraction in (0.0625, 0.25, 1.0):
                candidates = queries[chunk, None, :] + (radius * fraction) * search_directions
                targets = target[chunk, None].expand(-1, directions.shape[0]).reshape(-1)
                errors = candidate_error(candidates.reshape(-1, 3), targets).reshape(len(chunk), -1)
                errors = torch.where(torch.isfinite(errors), errors, torch.full_like(errors, float('inf')))
                minimum, choice = errors.min(dim=1)
                accept = minimum < best_error[chunk]
                rows = torch.arange(len(chunk), device=device)
                projected[chunk[accept]] = candidates[rows[accept], choice[accept]].detach()
                best_error[chunk[accept]] = minimum[accept]

    return projected

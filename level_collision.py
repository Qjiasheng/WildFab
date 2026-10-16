import numpy as np
import torch
import torch.nn.functional as F
import os

import igl
from utils import *
from vis import *




def load_mesh_pcd_coll(mesh_fn):
    """load mesh barycenter as pcd, and areas"""
    V, F = igl.read_triangle_mesh(mesh_fn)
    V = np.ascontiguousarray(V, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.int32)
    pcd = (V[F[:, 0]] + V[F[:, 1]] + V[F[:, 2]]) / 3.0
    double_areas = igl.doublearea(V, F)        
    areas = 0.5 * double_areas
    normals = igl.per_face_normals(V, F)

    edges = igl.edges(F) 
    edge_vecs = V[edges[:, 0]] - V[edges[:, 1]]
    edge_lengths = np.linalg.norm(edge_vecs, axis=1)
    avg_edge_length = edge_lengths.mean()

    return pcd, normals, areas, avg_edge_length    

    
def build_init_octree_with_mesh(surf_path):
    import octree_nt_clouds_wn
    coords, normals, areas, avg_edge_len = load_mesh_pcd_coll(surf_path)
    coords = torch.from_numpy(coords).float()
    normals = torch.from_numpy(normals).float()
    areas = torch.from_numpy(areas).float()

    torch.cuda.synchronize(); t0 = time.time()
    point_indices, children = octree_nt_clouds_wn.py_build_octree(coords, max_points_per_leaf=100, max_depth=16)
    mass_centers, mass_normals, mass_radii = octree_nt_clouds_wn.py_initialize_octree(
    coords, normals, areas, point_indices, children)

    children = children.to(dtype=torch.int64).cuda().contiguous()
    cell_centers = mass_centers.float().cuda().contiguous()
    cell_normals = mass_normals.float().cuda().contiguous()
    cell_radii = mass_radii.float().cuda().contiguous()
    coords = coords.float().cuda().contiguous()
    normals = normals.float().cuda().contiguous()
    areas = areas.float().cuda().contiguous()
    torch.cuda.synchronize()
    print(f"[I] Octree build and init time: {time.time()-t0:.8f}s")

    high_inv_delta = 2.5 / avg_edge_len   # adaptive 
    octree_wn_dict = {'coords': coords, 'normals': normals, 'areas': areas,
                      'point_indices': point_indices, 'children': children, 
                      'cell_centers': cell_centers, 'cell_normals': cell_normals, 'cell_radii': cell_radii,
                      'high_inv_delta': high_inv_delta}
    return octree_wn_dict



def build_init_octree_with_pcd(pcd_path, high_inv_delta=32.0, use_adaptive_delta=True):
    import octree_nt_clouds_wn
    from wn_utils import load_pcd_update
    coords, normals, areas = load_pcd_update(pcd_path, n_neighbors=20, remove_duplicates=True, cut_idx=None)

    if use_adaptive_delta:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(coords)
        distances, _ = nbrs.kneighbors()  # Query fitted points with self excluded.
        avg_edge_len = distances.mean()

        high_inv_delta = 2.5 / avg_edge_len
        print(f"[I] Adaptive high_inv_delta: {high_inv_delta:.4f} based on avg edge length: 1/ {int(1/avg_edge_len)}")

    coords = torch.from_numpy(coords).float()
    normals = torch.from_numpy(normals).float()
    areas = torch.from_numpy(areas).float()

    torch.cuda.synchronize(); t0 = time.time()
    point_indices, children = octree_nt_clouds_wn.py_build_octree(coords, max_points_per_leaf=100, max_depth=16)
    mass_centers, mass_normals, mass_radii = octree_nt_clouds_wn.py_initialize_octree(
    coords, normals, areas, point_indices, children)

    children = children.to(dtype=torch.int64).cuda().contiguous()
    cell_centers = mass_centers.float().cuda().contiguous()
    cell_normals = mass_normals.float().cuda().contiguous()
    cell_radii = mass_radii.float().cuda().contiguous()
    coords = coords.float().cuda().contiguous()
    normals = normals.float().cuda().contiguous()
    areas = areas.float().cuda().contiguous()
    torch.cuda.synchronize()
    print(f"[I] Octree build and init time: {time.time()-t0:.8f}s")

    octree_wn_dict = {'coords': coords, 'normals': normals, 'areas': areas,
                      'point_indices': point_indices, 'children': children, 
                      'cell_centers': cell_centers, 'cell_normals': cell_normals, 'cell_radii': cell_radii,
                      'high_inv_delta': high_inv_delta}
    return octree_wn_dict




def build_init_time_octree_with_pcd(clean_waypts_dict, high_inv_delta=32.0, use_adaptive_delta=True):
    """
    clean waypts means each timestep (using index) has valid waypts wrt. the waypts in octree
    """

    import octree_clouds_wn
    coords, normals, areas = clean_waypts_dict['coords'], clean_waypts_dict['normals'], clean_waypts_dict['areas']

    if use_adaptive_delta:
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='kd_tree').fit(coords)
        distances, _ = nbrs.kneighbors()  # Query fitted points with self excluded.
        avg_edge_len = distances.mean()

        high_inv_delta = 2.5 / avg_edge_len
        print(f"[I] Adaptive high_inv_delta: {high_inv_delta:.4f} based on avg edge length: 1/ {int(1/avg_edge_len)}")

    coords = torch.from_numpy(coords).float()
    normals = torch.from_numpy(normals).float()
    areas = torch.from_numpy(areas).float()

    torch.cuda.synchronize(); t0 = time.time()
    point_indices, children, time_intervals = octree_clouds_wn.py_build_octree(coords)
    mass_centers, mass_normals, mass_radii = octree_clouds_wn.py_initialize_octree(
    coords, normals, areas, point_indices, children)

    children = children.to(dtype=torch.int64).cuda().contiguous()
    cell_centers = mass_centers.float().cuda().contiguous()
    cell_normals = mass_normals.float().cuda().contiguous()
    cell_radii = mass_radii.float().cuda().contiguous()
    cell_minmax_range = time_intervals.to(dtype=torch.int32).cuda().contiguous()
    coords = coords.float().cuda().contiguous()
    normals = normals.float().cuda().contiguous()
    areas = areas.float().cuda().contiguous()
    torch.cuda.synchronize()
    print(f"[I] Octree build and init time: {time.time()-t0:.8f}s")

    time_octree_wn_dict = {'coords': coords, 'normals': normals, 'areas': areas,
                      'point_indices': point_indices, 'children': children, 
                      'cell_centers': cell_centers, 'cell_normals': cell_normals, 
                      'cell_radii': cell_radii, 'cell_minmax_range': cell_minmax_range,
                      'high_inv_delta': high_inv_delta}
    
    return time_octree_wn_dict


def fast_time_octree_wn(queries, ts, toctree_wn_dict):
    """
    fast winding number with timestep octree query.
    queries: (B, M, 3), ts: (B, ) int32 for each batch's timestep
    """
    import octree_clouds_wn
    wns, wn_grads = octree_clouds_wn.batch_query_wn_clouds(queries, ts,
    toctree_wn_dict.get('coords'), toctree_wn_dict.get('normals'), toctree_wn_dict.get('areas'),
    toctree_wn_dict.get('point_indices'), toctree_wn_dict.get('children'),
    toctree_wn_dict.get('cell_centers'), toctree_wn_dict.get('cell_normals'), toctree_wn_dict.get('cell_radii'),
    toctree_wn_dict.get('cell_minmax_range'), toctree_wn_dict.get('high_inv_delta'))

    return wns, wn_grads



def batch_alter_project( samples, udf_decoder, slice_decoder, iso_value, octree_wn_dict, 
                        order_type='udf-slice', wn_gradnorm_valid_th=None, ):
    """
    Alter the projection order of UDF/GWN and slice projection.
    Corresponding to paper [algorithm 1 Tracing Toolpath Points on Guidance Level Sets and Ridge Surface]
    """
    from wn_utils import batch_once_project_diff_field, once_tracing_max_wngrad_octree, fast_octree_wn

    if order_type == 'udf-slice':
        udf_projected, udf_dists, udf_valid = \
            once_tracing_max_wngrad_octree(samples, udf_decoder, octree_wn_dict, wn_gradnorm_valid_th=wn_gradnorm_valid_th, )

        # For invalid UDF projections, keep original samples.
        mid_points = samples.clone()
        mid_points[udf_valid] = udf_projected[udf_valid]

        _, surface_normals = fast_octree_wn(mid_points.contiguous(), octree_wn_dict)
        slice_projected = batch_once_project_diff_field(mid_points, slice_decoder, iso_value, surface_normals=surface_normals)

        projected = slice_projected
        valid = udf_valid
        dists = udf_dists

    elif order_type == 'slice-udf':
        _, surface_normals = fast_octree_wn(samples.contiguous(), octree_wn_dict)
        slice_projected = batch_once_project_diff_field(samples, slice_decoder, iso_value, surface_normals=surface_normals)

        udf_projected, udf_dists, udf_valid = \
            once_tracing_max_wngrad_octree(slice_projected, udf_decoder, octree_wn_dict, wn_gradnorm_valid_th=wn_gradnorm_valid_th, )

        # For invalid UDF projections, keep slice-projected points.
        projected = slice_projected.clone()
        projected[udf_valid] = udf_projected[udf_valid]

        valid = udf_valid
        dists = udf_dists

    else:
        raise ValueError( f"Unknown order_type: {order_type}. " f"Expected 'udf-slice' or 'slice-udf'." )

    return projected, dists, valid



class CollUDF_WN(torch.autograd.Function):
    """
    Specific implementation of Algorithm 2 in paper.

    Custom autograd function for collision evaluation.
    shell with grad direction crossing + GWN gradnorm coeff of variation rechecking
    infill uses wn > 0.5 as in infill part collision.

    GWN collision for shell and infill are time-varying, with octree acceleration.
    """

    @staticmethod
    def collect_edges(coords_flat, point_mask_flat, local_map, M):
        device = coords_flat.device
        N = point_mask_flat.numel() // M
        E = local_map.size(0)

        # ---------- rebuild (N,M) mask
        point_mask = point_mask_flat.view(N, M)            # (N,M)

        # ---------- which edges survive?  (N,E) bool
        left_ok  = point_mask[:, local_map[:, 0]]
        right_ok = point_mask[:, local_map[:, 1]]
        edge_mask = left_ok & right_ok                     # (N,E)

        # ---------- collect valid edge indices
        wp_idx, e_idx = torch.nonzero(edge_mask, as_tuple=True)   # (K,), (K,)
        i_local = local_map[e_idx, 0]
        j_local = local_map[e_idx, 1]

        # flattened point indices
        flat_i = wp_idx * M + i_local                       # (K,)
        flat_j = wp_idx * M + j_local                       # (K,)

        edge_indices = torch.stack((flat_i, flat_j), dim=1) # (K,2)
        edges_coords = torch.stack(
            (coords_flat[flat_i], coords_flat[flat_j]), dim=1)     # (K,2,3)

        wp_edge_idx = torch.stack((wp_idx, i_local, j_local), dim=1)  # (K,3)

        return edges_coords, edge_indices, wp_edge_idx

    @staticmethod
    def recover_point_mask(edge_mask, edge_indices, total_points):
        if edge_mask.sum() == 0:
            return torch.zeros(total_points, dtype=torch.bool,
                            device=edge_mask.device)

        kept_edges = edge_indices[edge_mask]               # (K',2)
        pts = kept_edges.flatten()                         # (2*K',)

        mask = torch.zeros(total_points, dtype=torch.bool,
                        device=edge_mask.device)
        mask[pts] = True
        return mask

    @staticmethod
    def forward(ctx, nozzle_pcd, nozzle_grads, local_map, udf_decoder, udf_init_margin, base_th,
                shell_ts, outer_shells_ts_tagdict, st_octree_wn_dict, ost_octree_wn_dict_tagdict,
                edge_sample_num=20, cv_th=1.0, max_batch=64**3):
        """
        Check shell crossings and infill interiors at each waypoint's clean timestep.

        max_batch limits query points per call
        complete edge (edge_sample_num points) per call.

        larger cv_th can be set for more conservative collision detection, 
        but may miss some crossing.
        """
        N, M, _ = nozzle_pcd.shape   # N-waypt batch, M-nozzle pcd queries
        device = nozzle_pcd.device
        coords = nozzle_pcd.reshape(-1, 3)
        grads = nozzle_grads.reshape(-1, 3)
        local_map = local_map.to(device=device, dtype=torch.long)
        shell_ts = shell_ts.to(device=device, dtype=torch.long).reshape(-1)

        coll_depths = coords.new_zeros(N * M)
        coll_grads = torch.zeros_like(coords)
        coll_mask = torch.zeros(N * M, dtype=torch.bool, device=device)
        coll_pts_edge_mask = torch.zeros_like(coll_mask)
        infill_mask = torch.zeros_like(coll_mask)

        # Base collisions are independent of the volume query bounds.
        base_coll_mask = coords[:, 2] < base_th
        coll_depths[base_coll_mask] = 0.1        # any arbitrary negative value
        coll_grads[base_coll_mask, 2] = -1.0      # up-z coll grad
        coll_mask |= base_coll_mask

        cube_mask = ((coords >= -1.0) & (coords <= 1.0)).all(dim=-1)
        original_udfs = coords.new_full((N * M,), 10.0)
        if cube_mask.any():
            cube_udfs = field_querying(udf_decoder, coords[cube_mask], no_print=True,
                                      return_on_cuda=True, max_batch=max_batch)
            original_udfs[cube_mask] = cube_udfs.reshape(-1)

        # Shell: UDF margin, opposing endpoint gradients, then time-varying GWN recheck.
        within_margin_mask = cube_mask & (original_udfs < udf_init_margin)
        if within_margin_mask.any():
            _, _, wm_udf_grads = field_query_with_grads_no_higher_order(
                udf_decoder, coords[within_margin_mask], no_print=True,
                return_on_cuda=True, max_batch=max_batch)
            udf_grads = torch.zeros_like(coords)
            udf_grads[within_margin_mask] = F.normalize(wm_udf_grads, dim=-1, p=2)
            edge_grads, edge_indices, _ = CollUDF_WN.collect_edges(
                udf_grads, within_margin_mask, local_map, M)
            edge_mask = (edge_grads[:, 0] * edge_grads[:, 1]).sum(dim=-1) < 0.0
            edge_indices = edge_indices[edge_mask]   # (K,2) flattened indices in 0 ... N*M-1

            # Each candidate edge is a query batch with its own waypoint timestep.
            edge_ts = shell_ts[edge_indices[:, 0] // M]
            printed = edge_ts >= 0
            edge_indices, edge_ts = edge_indices[printed], edge_ts[printed]
            crossing_mask = torch.zeros(len(edge_indices), dtype=torch.bool, device=device)
            sample_t = torch.linspace(0, 1, edge_sample_num, device=device, dtype=coords.dtype)
            edge_batch = max(1, max_batch // edge_sample_num)
            for start in range(0, len(edge_indices), edge_batch):
                end = min(start + edge_batch, len(edge_indices))
                edges = coords[edge_indices[start:end]]
                # sample points along edges and query TV-GWN for wn_gradnorms coeff of variation
                queries = edges[:, :1] + sample_t[None, :, None] * (edges[:, 1:] - edges[:, :1])
                _, wn_grads = fast_time_octree_wn(
                    queries.contiguous(), edge_ts[start:end].contiguous(), st_octree_wn_dict)
                grad_norms = torch.norm(wn_grads, dim=-1)
                cv = grad_norms.std(dim=1, unbiased=False) / (grad_norms.mean(dim=1) + 1e-8)
                crossing_mask[start:end] = cv > cv_th
            coll_pts_edge_mask = CollUDF_WN.recover_point_mask(
                crossing_mask, edge_indices, total_points=N * M)
            # info recheck with TV-GWN after UDF crossing.
            # print(f'TV-GWN rechecked: {crossing_mask.sum().item()} / {len(edge_indices)} UDF-crossing.')

        # Infill: test points against each printed outer shell, without a UDF-margin gate.
        if outer_shells_ts_tagdict:
            for shell_tag, outer_ts in outer_shells_ts_tagdict.items():
                outer_ts = outer_ts.to(device=device, dtype=torch.long).reshape(-1)
                point_ts = outer_ts.repeat_interleave(M)
                point_ids = torch.nonzero(
                    cube_mask & ~base_coll_mask & ~infill_mask & (point_ts >= 0),
                    as_tuple=True)[0]
                for start in range(0, len(point_ids), max_batch):
                    ids = point_ids[start:start + max_batch]
                    # One query point per batch preserves the point's original waypoint time.
                    wns, _ = fast_time_octree_wn(
                        coords[ids].unsqueeze(1).contiguous(), point_ts[ids].contiguous(),
                        ost_octree_wn_dict_tagdict[shell_tag])
                    infill_mask[ids] |= wns.reshape(-1) > 0.5   # inside tv-printed infill volume

        object_coll_mask = (coll_pts_edge_mask | infill_mask) & ~base_coll_mask
        coll_mask |= object_coll_mask
        coll_depths[object_coll_mask] = original_udfs[object_coll_mask]
        coll_grads[object_coll_mask] = grads[object_coll_mask]
        ctx.save_for_backward(coll_grads.reshape(N, M, 3))
        coll_mask = coll_mask.reshape(N, M)
        ctx.mark_non_differentiable(coll_mask)
        return coll_depths.reshape(N, M), coll_mask

    @staticmethod
    def backward(ctx, dl_dcoll_depths, dl_dcoll_mask=None):
        coll_grads, = ctx.saved_tensors
        N, M, _ = coll_grads.shape
        dl_dpcd = None
        if ctx.needs_input_grad[0] and dl_dcoll_depths is not None:
            coll_grads = F.normalize(coll_grads, dim=-1, eps=1e-8)
            dl_dpcd = coll_grads.mul_(dl_dcoll_depths.view(N, -1, 1))  # (N, M, 3)
        return (dl_dpcd,) + (None,) * (len(ctx.needs_input_grad) - 1)



def collision_evaluation_with_infill(nozzle_pcd, nozzle_grads, local_map, udf_decoder, 
                                     base_th=None, udf_init_margin=None, 
                                    shell_ts=None, outer_shells_ts_tagdict=None,
                                    st_octree_wn_dict=None, ost_octree_wn_dict_tagdict=None,
                                    max_batch=64**3, edge_sample_num=20, cv_th=1.0):
    """
    Algorithm 2 in paper
    evaluate collision for both shells and infills.
    -------------------------------
    nozzle_pcd: (num_waypts, num_nozzle_pts, 3)
    nozzle_grads: (num_waypts, num_nozzle_pts, 3)
    local_map: to recover nozzle pcd to edges using pcd indices

    base_th: base collision threshold, nozzle pcd below this z is coll
    udf_init_margin: initial filter margin for UDF collision evaluation

    shell_ts: (num_waypts, ) pre-removed timestep wrt. shell octree, for shell collision 
    outer_shells_ts_tagdict: dict of (tag, ts) for each outer shell wrt. each outer shell octree, for infill collision 
    st_octree_wn_dict: octree dict for shell octree
    ost_octree_wn_dict_tagdict: dict of (tag, octree_dict) for each outer shell octree
    -------------------------------
    NOTE query amount can be further significantly reduced with a sequence field (level+guidance) as in INF-3DP,
    where seq-field provides a printing-sequence info for initial query filtering.
    """

    coll_depths, coll_mask = CollUDF_WN.apply(
        nozzle_pcd, nozzle_grads, local_map, udf_decoder, udf_init_margin, base_th, 
        shell_ts, outer_shells_ts_tagdict, st_octree_wn_dict, ost_octree_wn_dict_tagdict,
        edge_sample_num, cv_th, max_batch) # (num_waypts, num_nozzle_pts)
    
    waypt_coll_mask = coll_mask.any(dim=-1)  # (num_waypts, )

    return coll_depths, coll_mask, waypt_coll_mask




def batch_collision_querying(udf_decoder, waypts, quats, nozzle_pcd, nozzle_grads, local_map, base_coll_th, udf_init_margin,
                             shell_ts, outer_shells_ts_tagdict, st_octree_wn_dict, ost_octree_wn_dict_tagdict,
                             batch_size=10000, max_batch=64**3):
    """Return CPU collision mask and concatenated nozzle data for all waypoints.

    Nozzle arrays have shape (N, M, 3), in the original waypoint order.
    Gradients are collision-loss gradients, not rotated nozzle normals.
    """
    N = waypts.shape[0]
    device = nozzle_pcd.device
    local_map = local_map.to(device)
    full_waypt_coll_mask = torch.zeros(N, dtype=torch.bool)
    coll_pcd_list = []
    coll_grads_list = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        with torch.no_grad():
            batch_waypts = waypts[start:end].to(device)
            batch_quats = quats[start:end].to(device)
            transed_nozzle_pcd = transform_nozzle_pcd_with_quaternion(nozzle_pcd, batch_waypts, batch_quats)
            transed_nozzle_grads = transform_nozzle_grads_with_quaternion(nozzle_grads, batch_waypts, batch_quats)
        batch_shell_ts = shell_ts[start:end].to(device=device, dtype=torch.long).contiguous()
        batch_outer_ts = {
            tag: ts[start:end].to(device=device, dtype=torch.long).contiguous()
            for tag, ts in outer_shells_ts_tagdict.items()
        }

        batch_start_time = time.time()
        with torch.enable_grad():
            transed_nozzle_pcd = transed_nozzle_pcd.detach().requires_grad_(True)
            coll_depths, coll_mask, waypt_coll_mask = collision_evaluation_with_infill(
                transed_nozzle_pcd, transed_nozzle_grads, local_map, udf_decoder,
                base_th=base_coll_th, udf_init_margin=udf_init_margin,
                shell_ts=batch_shell_ts, outer_shells_ts_tagdict=batch_outer_ts,
                st_octree_wn_dict=st_octree_wn_dict, ost_octree_wn_dict_tagdict=ost_octree_wn_dict_tagdict,
                max_batch=max_batch)
            coll_grads, = torch.autograd.grad(coll_depths.sum(), transed_nozzle_pcd)

        full_waypt_coll_mask[start:end] = waypt_coll_mask.detach().cpu()
        coll_pcd_list.append(transed_nozzle_pcd.detach().cpu())
        coll_grads_list.append(coll_grads.detach().cpu())

        del transed_nozzle_pcd, transed_nozzle_grads, coll_depths, coll_grads, coll_mask, waypt_coll_mask
        print(f'Collision evaluation: {end}/{N}, time: {time.time()-batch_start_time:.4f}s')

    empty_shape = (0, nozzle_pcd.shape[0], 3)
    coll_pcd = torch.cat(coll_pcd_list, dim=0) if coll_pcd_list else nozzle_pcd.new_empty(empty_shape, device='cpu')
    coll_grads = torch.cat(coll_grads_list, dim=0) if coll_grads_list else nozzle_grads.new_empty(empty_shape, device='cpu')
    return full_waypt_coll_mask, coll_pcd, coll_grads


def quat_collision_test(udf_decoder, quat_decoder, dataset_paras=None, loss_paras=None, 
                        save_path=None, batch_size=15000, max_batch=96**3):
    """Single collision test using predicted quaternions or original print-direction frames.

    Save waypoints, quaternions, the collision mask and collision-only nozzle data.
    """
    from dataio import WayPts
    from utils import load_nozzle_data
    from scipy.spatial.transform import Rotation

    object_scale = loss_paras.get('object_scale')
    udf_init_margin = 5.0 / object_scale

    # less edge will not reduce pcd grads, but reduce gwn querying number and memory
    nozzle_coords, nozzle_grads, local_map = load_nozzle_data(
        dataset_paras.get('nozzle_nodes_path'), dataset_paras.get('nozzle_elems_path'),
        object_scale=object_scale, sample_num=5000, device='cuda')

    waydata = WayPts(waypoint_path=dataset_paras.get('waypoint_path'))
    seq_timesteps = np.arange(waydata.waypt_len, dtype=np.int64)  # all pts
    coords, normals, print_dirs, shell_infill_tags, shell_ts, outer_shells_ts_tagdict = \
        waydata.get_bacth_waydata(seq_timesteps)

    # Construct an orthonormal frame, including print directions parallel to the reference.
    if quat_decoder is None:
        # no optimized quat
        print_dirs = F.normalize(print_dirs, dim=-1)
        ref_dirs = torch.zeros_like(print_dirs)
        ref_dirs[:, 0] = -1.0
        parallel = print_dirs[:, 0].abs() > 0.99
        ref_dirs[parallel] = torch.tensor([0., 1., 0.], dtype=print_dirs.dtype)
        frame_dirs = F.normalize(torch.cross(ref_dirs, print_dirs, dim=-1), dim=-1)
        y_dirs = torch.cross(print_dirs, frame_dirs, dim=-1)
        matrices = torch.stack((frame_dirs, y_dirs, print_dirs), dim=-1)
        # scipy uses xyzw; the nozzle transforms use wxyz.
        quats_xyzw = Rotation.from_matrix(matrices.numpy()).as_quat()
        quaternions = torch.from_numpy(quats_xyzw[:, [3, 0, 1, 2]].copy()).float()
    else:
        with torch.no_grad():
            quaternions = quat_field_querying(quat_decoder, coords, no_print=True, return_on_cuda=False)
        quaternions = F.normalize(quaternions, dim=-1)

    base_coll_th = waydata.min_z - 2.0 / object_scale
    
    start = time.time()
    waypt_coll_mask, transed_nozzle_pcd, transed_nozzle_grads = batch_collision_querying(
        udf_decoder, coords, quaternions, nozzle_coords, nozzle_grads, local_map,
        base_coll_th, udf_init_margin, shell_ts, outer_shells_ts_tagdict,
        waydata.shell_time_octree_wn_dict, waydata.outer_shells_time_octree_wn_dict,
        batch_size=batch_size, max_batch=max_batch)
    print(f'Collision evaluation for {waydata.waypt_len} waypoints, Total time: {time.time()-start:.4f}s')

    num_coll = waypt_coll_mask.sum().item()
    print(f'Collision ratio: {num_coll / waydata.waypt_len * 100:.2f}%, {num_coll} / {waydata.waypt_len}')
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        np.savez_compressed(
            os.path.join(save_path, 'quat_collision_test.npz'),
            waypts=coords.numpy(), quaternions=quaternions.detach().cpu().numpy(),
            waypt_coll_mask=waypt_coll_mask.numpy(),
            coll_transed_nozzle_pcd=transed_nozzle_pcd[waypt_coll_mask].numpy(),
            coll_transed_nozzle_grads=transed_nozzle_grads[waypt_coll_mask].numpy())
        print(f'Saved collision test data to {save_path}')
    

    # if having collision, pick one to show
    if num_coll > 0:
        coll_index = torch.nonzero(waypt_coll_mask, as_tuple=True)[0][0].item()
        nozzle_mesh = load_nozzle_mesh(dataset_paras.get('nozzle_mesh_path'), object_scale=object_scale)
        vertices = torch.from_numpy(np.asarray(nozzle_mesh.vertices).copy()).float()
        with torch.no_grad():
            transed_vertices = transform_nozzle_pcd_with_quaternion(
                vertices, coords[coll_index:coll_index + 1], quaternions[coll_index:coll_index + 1])
        nozzle_mesh.vertices = transed_vertices[0].cpu().numpy()
        vis_print_nozzle_mesh_comparison(
            mesh=nozzle_mesh, waypoints=coords[:coll_index + 1].numpy(), add_cube=True)

    return waypt_coll_mask

import numpy as np
import time
import torch
from scipy.spatial import KDTree, cKDTree
import igl
from collections import defaultdict
from qupkg.extract_isocontours import extract_isocontours
from sdf_meshing import gen_voxle_queries
from utils import field_querying
from vis import visualize_merged_iso_pts_mul


def cuda_extract_iso_contours(vertices, faces, fields, iso_values, len_lim=50, connect_broken=False, connect_broken_len=None):
    """extract iso-contours with marching triangles in cuda, then split into a dict in python.
    all are numpy arrays"""

    vertices = torch.from_numpy(vertices).float().cuda().contiguous()
    faces = torch.from_numpy(faces).int().cuda().contiguous() # int type
    fields = torch.from_numpy(fields).float().cuda().contiguous()
    iso_values = torch.from_numpy(iso_values).float().cuda().contiguous()
    
    # first get merged contours in cuda, then split them in python
    contour_merged_points, contour_nums = extract_isocontours(vertices, faces, fields, iso_values)
    contours = split_merged_contours(contour_merged_points, contour_nums, iso_values, 
                                                      len_lim=len_lim,connect_broken=connect_broken,
                                                    connect_broken_len=connect_broken_len)
    
    return contours


def split_merged_contours(contour_points, contour_nums, iso_values, len_lim=50, connect_broken=False, connect_broken_len=None):
    """Split CUDA point chains, optionally reconnect endpoints, then filter length.

    Each chain ends with a (-1e9, -1e9, -1e9) separator. Output capacity
    can exceed num_faces because each chain also stores its start and separator.
    connect_broken_len is the maximum distance for joining separate open chains.
    """
    contour_points = contour_points.cpu().numpy()
    contour_nums = contour_nums.cpu().numpy()
    iso_values = iso_values.cpu().numpy()
    contours_dict = {}

    def is_closed(contour):
        return len(contour) > 2 and np.linalg.norm(contour[0] - contour[-1]) < 1e-4

    for i, iso_value in enumerate(iso_values):
        merged_points = contour_points[i]
        # One index per separator row, not one per coordinate.
        separators = np.flatnonzero(np.all(merged_points == -1e9, axis=1))
        contours = []
        broken = []
        start = 0
        for separator in separators[:int(contour_nums[i])]:
            contour = merged_points[start:separator]
            start = separator + 1
            if not len(contour):
                continue
            if connect_broken and not is_closed(contour):
                broken.append(contour)
            else:
                contours.append(contour)

        # Collect every fragment before looking for connections. Short fragments
        # may be necessary bridges, so length filtering must happen afterwards.
        while broken:
            current = broken.pop(0)
            while not is_closed(current):
                best_distance = connect_broken_len if connect_broken_len is not None else float('inf')
                best = None
                for idx, candidate in enumerate(broken):
                    for distance, reverse, prepend in (
                        (np.linalg.norm(current[-1] - candidate[0]), False, False),
                        (np.linalg.norm(current[-1] - candidate[-1]), True, False),
                        (np.linalg.norm(current[0] - candidate[-1]), False, True),
                        (np.linalg.norm(current[0] - candidate[0]), True, True),
                    ):
                        if distance <= best_distance and (best is None or distance < best_distance):
                            best_distance = distance
                            best = idx, reverse, prepend
                if best is None:
                    break
                idx, reverse, prepend = best
                candidate = broken.pop(idx)
                if reverse:
                    candidate = candidate[::-1]
                left, right = (candidate, current) if prepend else (current, candidate)
                if np.array_equal(left[-1], right[0]):
                    right = right[1:]
                current = np.vstack((left, right))
            contours.append(current)

        contours_dict[iso_value] = [contour.tolist() for contour in contours if len(contour) > len_lim]
    return contours_dict


def get_avg_edge_length(verts, faces):
    edges = igl.edges(faces) 
    edge_vecs = verts[edges[:, 0]] - verts[edges[:, 1]]
    edge_lengths = np.linalg.norm(edge_vecs, axis=1)
    avg_edge_length = edge_lengths.mean()
    return avg_edge_length


def load_igl_mesh(mesh_fn):
    return igl.read_triangle_mesh(mesh_fn)


def get_single_shell_contours(slice_decoder, slice_num, mesh_fn, len_lim=0,
                              connect_broken=False, discard_fisrt_num=0):
    """
    mesh iso-contours extraction in cuda.
    -------------------- 
    len_lim: minimum number of points for a subcontour, realated with mesh resolution
    ----------------------
    connect broken len use avg_edge_len * 2 as default
    """

    vertices, faces = load_igl_mesh(mesh_fn)  
    avg_edge_len = get_avg_edge_length(vertices, faces)
    print(f"average edge length: {avg_edge_len:.6f}")

    slice_fields = field_querying(slice_decoder, vertices, no_print=True, return_on_cuda=False)
    
    # collect slice iso-values
    slice_fields_np = slice_fields.cpu().numpy()
    field_min, field_max = np.min(slice_fields_np), np.max(slice_fields_np)

    slice_iso_values = np.linspace(field_min, field_max, slice_num)
    print(f'Generated {len(slice_iso_values)} uniform slice iso-values.')

    if discard_fisrt_num > 0:
        slice_iso_values = slice_iso_values[discard_fisrt_num:]
        print(f"Discarded the first {discard_fisrt_num} slice iso-values, remaining {len(slice_iso_values)} iso-values.")
    
    # parallel iso-contours extraction in cuda
    start = time.time()
    torch.cuda.synchronize()
    contours = cuda_extract_iso_contours(vertices, faces, slice_fields_np, 
                                        slice_iso_values, len_lim=len_lim, connect_broken=connect_broken, 
                                        connect_broken_len=avg_edge_len*2.0)
    torch.cuda.synchronize()
    print(f"CUDA iso-contours extraction takes: {time.time() - start:.6f} seconds")

    # convert them to numpy for later use
    contours = {iso_val: [np.array(subcontour) for subcontour in subcontours] for iso_val, subcontours in contours.items()}

    return contours, avg_edge_len


def get_single_shell_contours_field_projection(udf_decoder, slice_decoder, slice_num, pcd_fn, len_lim=0,
                                            connect_broken=False, discard_fisrt_num=0, resolution=128,
                                            max_batch_points=None, max_batch_contouring=500, contour_backend="auto", 
                                            contour_voxel_size=None, contour_search_radius=None, contour_query_batch_size=32**3,
                                            tracing_max_iter=10):
    """
    extract iso-contours through field projection with UDF/GWN guidance field.
    --------------------
    two stages: 1. tracing iso-contours clouds
                2. neighborhood contouring with cached surface normals and field tangents
    max_batch_contouring limits iso-value clouds per CUDA batch in stage 2.
    contour_query_batch_size independently limits points per geometry query.
    contour_voxel_size=0 disables downsampling; backend can be auto, cuda, or cpu.
    Open chains remain open; connect_broken does not force endpoint closure in this path.
    """
    from dataio import get_avg_dist
    from level_collision import build_init_octree_with_pcd

    vertices = np.loadtxt(pcd_fn)[:, :3]  # only use xyz 
    avg_edge_len = get_avg_dist(coords=vertices, k=10)
    print(f"average edge length: {avg_edge_len:.6f}")
    slice_fields = field_querying(slice_decoder, vertices, no_print=True, return_on_cuda=False)
    
    # collect slice iso-values
    slice_fields_np = slice_fields.cpu().numpy()
    field_min, field_max = np.min(slice_fields_np), np.max(slice_fields_np)

    slice_iso_values = np.linspace(field_min, field_max, slice_num)
    print(f'Generated {len(slice_iso_values)} uniform slice iso-values.')

    if discard_fisrt_num > 0:
        slice_iso_values = slice_iso_values[discard_fisrt_num:]
        print(f"Discarded the first {discard_fisrt_num} slice iso-values, remaining {len(slice_iso_values)} iso-values.")
    
    # ----------- iso-contours tracing through alternating UDF/GWN and guidance field projection
    octree_wn_dict = build_init_octree_with_pcd(pcd_path=pcd_fn)
    # stage 1: tracing iso-contours clouds, less iteration is enough 
    start = time.time()
    torch.cuda.synchronize()
    contours_pts = alter_project_extract_iso_pts_batch(udf_decoder, slice_decoder, slice_iso_values, 
                                                       octree_wn_dict, init_N=resolution, order_type='slice-udf',
                                                       max_batch_points=max_batch_points, max_iter=tracing_max_iter)
    torch.cuda.synchronize()
    print(f"Stage 1: iso-contours clouds tracing takes: {time.time() - start:.6f} seconds")
    
    # stage 2: parallel neighborhood contouring (connecting points into subcontours)
    start = time.time()
    torch.cuda.synchronize()
    contours = connect_projected_iso_points(contours_pts, slice_decoder, octree_wn_dict, 
                                            spacing=avg_edge_len, len_lim=len_lim, 
                                            query_batch_size=contour_query_batch_size, 
                                            max_batch_contouring=max_batch_contouring, backend=contour_backend, 
                                            voxel_size=contour_voxel_size, search_radius=contour_search_radius,
                                            profile=False)
    
    torch.cuda.synchronize()
    print(f"Stage 2: iso-contours clouds connection takes: {time.time() - start:.6f} seconds")

    visualize_merged_iso_pts_mul(contours_pts, start=None, end=None, random_color=True, 
                                 mesh_fn=None, point_size=12, label='alter tracing iso-contours clouds')

    # convert them to numpy for later use
    contours = {iso_val: [np.array(subcontour) for subcontour in subcontours] for iso_val, subcontours in contours.items()}

    return contours, avg_edge_len


def alter_project_extract_iso_pts_batch(udf_decoder, slice_decoder, slice_iso_values, octree_wn_dict, init_N=256, 
                                        order_type='udf-slice', udf_margin=1.0 / 128.0, slice_margin=0.02, max_iter=10, 
                                        max_batch_points=None, wn_gradnorm_valid_th=10.0,):
    """Extract iso points for multiple slice iso-values using one concatenated batch.
    This version concatenates queries from all iso-values:

        queries:       Tensor, shape (total_num, 3)
        iso_per_query: Tensor, shape (total_num,)

    Then each projection iteration is performed once on the concatenated tensor.
    At the end, projected points are recovered back into:
        iso_pts_dict[iso_value] = np.ndarray, shape (Ni, 3)
    
    ----------------------------------- 
    init_N: initial voxel grid resolution for UDF/GWN and slice field querying
    udf_margin: initial UDF margin for selecting iso-candidates
    slice_margin: initial slice margin for selecting iso-candidates

    wn_gradnorm_valid_th: threshold for valid wn gradnorm. useful to cut off capping of open surfaces.
    """

    from level_collision import batch_alter_project

    start_all = time.time()

    # Prepare volume data of both udf and slice fields
    samples_cpu, _, _ = gen_voxle_queries( N=init_N, cube_size=1.0, )
    slice_fields_cpu = field_querying( slice_decoder, samples_cpu, no_print=True, return_on_cuda=False, )
    udf_fields_cpu = field_querying( udf_decoder, samples_cpu, no_print=True, return_on_cuda=False, )

    if slice_fields_cpu.ndim > 1:
        slice_fields_cpu = slice_fields_cpu.squeeze(-1)

    if udf_fields_cpu.ndim > 1:
        udf_fields_cpu = udf_fields_cpu.squeeze(-1)

    udf_base_mask = udf_fields_cpu < udf_margin

    if torch.is_tensor(slice_iso_values):
        iso_values = [ float(v) for v in slice_iso_values.detach().cpu().tolist() ]
    else:
        iso_values = [ float(v) for v in slice_iso_values ]

    # Build concatenated initial queries
    pts_list = []
    iso_list = []

    iso_pts_dict = { iso_value: np.zeros((0, 3), dtype=np.float32) for iso_value in iso_values }
    for iso_value in iso_values:
        start_iso_init = time.time()
        mask = ( udf_base_mask & (slice_fields_cpu >= iso_value - slice_margin) & (slice_fields_cpu <= iso_value + slice_margin) )
        filtered_samples = samples_cpu[mask]
        init_count = filtered_samples.shape[0]
        if init_count == 0:
            print(f"[iso={iso_value:.6f}] init=0, skipped")
            continue

        pts_list.append(filtered_samples.detach())
        iso_list.append( torch.full( (init_count,), iso_value, dtype=filtered_samples.dtype, device=filtered_samples.device, ) )
        # print( f"[iso={iso_value:.6f}] " f"init={init_count}, " f"init_time={time.time() - start_iso_init:.4f}s" )

    if len(pts_list) == 0:
        print( f"[multi-iso concat] all empty. " f"total_time={time.time() - start_all:.4f}s" )
        return iso_pts_dict

    queries = torch.cat(pts_list, dim=0).contiguous()
    iso_per_query = torch.cat(iso_list, dim=0).contiguous()

    queries = queries.cuda(non_blocking=True)
    iso_per_query = iso_per_query.cuda(non_blocking=True)
    print( f"[multi-iso concat] total_init={queries.shape[0]}, " f"num_iso={len(iso_values)}" )

    # Iterative projection on concatenated queries
    for i in range(max_iter):
        iter_start = time.time()
        if queries.shape[0] == 0:
            break

        if max_batch_points is None:
            pro_coords, dists, valid = \
                batch_alter_project(queries, udf_decoder, slice_decoder, iso_per_query, 
                                    octree_wn_dict, order_type, wn_gradnorm_valid_th=wn_gradnorm_valid_th, )
            if valid is not None:
                valid = valid.detach()

                if valid.any():
                    queries = pro_coords[valid].detach()
                    iso_per_query = iso_per_query[valid].detach()
                else:
                    queries = pro_coords.detach()
                    iso_per_query = iso_per_query.detach()
            else:
                queries = pro_coords.detach()
                iso_per_query = iso_per_query.detach()

        else:
            new_queries_list = []
            new_iso_list = []

            total_num = queries.shape[0]
            for st in range(0, total_num, max_batch_points):
                ed = min(st + max_batch_points, total_num)

                q_chunk = queries[st:ed]
                iso_chunk = iso_per_query[st:ed]
                pro_coords, dists, valid = \
                    batch_alter_project(q_chunk, udf_decoder, slice_decoder, iso_chunk, octree_wn_dict, 
                                        order_type, wn_gradnorm_valid_th=wn_gradnorm_valid_th, )

                if valid is not None:
                    valid = valid.detach()

                    if valid.any():
                        new_queries_list.append(pro_coords[valid].detach())
                        new_iso_list.append(iso_chunk[valid].detach())
                    else:
                        new_queries_list.append(pro_coords.detach())
                        new_iso_list.append(iso_chunk.detach())
                else:
                    new_queries_list.append(pro_coords.detach())
                    new_iso_list.append(iso_chunk.detach())

            if len(new_queries_list) == 0:
                queries = torch.empty( (0, 3), device=queries.device, dtype=queries.dtype, )

                iso_per_query = torch.empty( (0,), device=queries.device, dtype=iso_per_query.dtype, )
            else:
                queries = torch.cat(new_queries_list, dim=0).contiguous()
                iso_per_query = torch.cat(new_iso_list, dim=0).contiguous()

        print( f"[multi-iso concat] " f"iter={i + 1}/{max_iter}, " f"active={queries.shape[0]}, " f"time={time.time() - iter_start:.4f}s" )

    # Recover points back to each iso-value
    queries_np = ( queries .detach() .cpu() .numpy() .astype(np.float32) )
    iso_per_query_np = ( iso_per_query .detach() .cpu() .numpy() )

    for iso_value in iso_values:
        keep = np.isclose( iso_per_query_np, iso_value, rtol=0.0, atol=1e-7, )
        pts_np = queries_np[keep]
        iso_pts_dict[iso_value] = pts_np
        # print( f"[iso={iso_value:.6f}] " f"final={pts_np.shape[0]}" )
    print( f"[multi-iso concat] all done. " f"total_time={time.time() - start_all:.4f}s" )

    return iso_pts_dict


def compute_subcontour_centers(contours):
    centers = {}
    for iso_value, subcontours in contours.items():
        centers[iso_value] = [np.mean(subcontour, axis=0) for subcontour in subcontours]
    return centers


def get_nearest_normals(points, verts, faces):
    """use igl pseudo normals as approximated point normal.
    sign_type set as pseudonormal"""
    pseudo_normals = igl.signed_distance(points, verts, faces, sign_type=0)[-1]
    return pseudo_normals / np.linalg.norm(pseudo_normals, axis=1, keepdims=True)


def get_nearest_normals_pcd(points, pcd_coords, pcd_normals, k=5):
    """use point cloud nearest neighbors as approximated point normal."""
    tree = KDTree(pcd_coords)
    dists, idxs = tree.query(points, k=k)
    neighbor_normals = pcd_normals[idxs]  
    avg_normals = np.mean(neighbor_normals, axis=1)  
    return avg_normals / np.linalg.norm(avg_normals, axis=1, keepdims=True)


def _flatten_contours(contours):
    pts_all = []
    layout = []          # (iso, list_idx, L)

    for iso, polys in contours.items():
        for i, p in enumerate(polys):
            p = np.asarray(p, float)
            pts_all.append(p)
            layout.append((iso, i, len(p)))

    return np.vstack(pts_all), layout


def _restore_nested(flat_values, layout):
    out = defaultdict(list)
    cursor = 0
    for iso, li, L in layout:
        block = flat_values[cursor:cursor + L]
        out[iso].append(block)
        cursor += L
    return dict(out)


def compute_approximate_normals(contours, mesh_fn):
    """approximate normals using face pseudo normals"""

    v, f = igl.read_triangle_mesh(mesh_fn)
    pts_flat, layout = _flatten_contours(contours)          

    n_flat = get_nearest_normals(pts_flat, v, f)           
    normals_dict = _restore_nested(n_flat, layout)
    
    return normals_dict


def compute_approximate_normals_pcd(contours, pcd_fn, k=5):
    """approximate normals using point cloud nearest neighbors"""

    from dataio import load_udf_pcd
    pcd_data = load_udf_pcd(pcd_fn)
    coords, normals = pcd_data[0], pcd_data[1]
    pts_flat, layout = _flatten_contours(contours)

    n_flat = get_nearest_normals_pcd(pts_flat, coords, normals, k=k)
    normals_dict = _restore_nested(n_flat, layout)

    return normals_dict


def del_non_enclosed_subcontours(contours):
    """del if subcontour is not enclosed"""
    def _is_closed_loop(contour, tol=1e-5):
        return np.linalg.norm(np.asarray(contour[0]) - np.asarray(contour[-1])) < tol
    
    filtered_contours = {}
    for iso, subs in contours.items():
        filtered_subs = []
        for sub in subs:
            if _is_closed_loop(sub):
                filtered_subs.append(sub)
        if filtered_subs:
            filtered_contours[iso] = filtered_subs
    return filtered_contours


def connect_projected_iso_points(iso_points, slice_decoder, octree_wn_dict, spacing, len_lim=0, voxel_size=None, 
                                 search_radius=None, query_batch_size=32768, backend='auto', return_packed=False, 
                                 max_batch_contouring=300, graph_backend="auto", profile=True):
    """
    Stage 2: cache field/surface geometry once, then connect points independently per level.
    using neighborhood search and noraml and tangents 
    """
    import octree_nt_clouds_wn

    # Flatten and upload the immutable octree leaf structure once for all query batches.
    point_indices = octree_wn_dict['point_indices']
    leaf_offsets = torch.tensor(np.r_[0, np.cumsum([len(indices) for indices in point_indices])], dtype=torch.int64, device=octree_wn_dict['coords'].device)
    leaf_indices = torch.tensor([index for indices in point_indices for index in indices], dtype=torch.int64, device=leaf_offsets.device)
    slice_decoder.eval()

    def geometry_query(points):
        samples = torch.from_numpy(points).float().cuda().contiguous()
        
        with torch.enable_grad():
            samples.requires_grad_(True)
            output = slice_decoder(samples)
            gradients = torch.autograd.grad(output['model_out'], output['model_in'], 
                                            grad_outputs=torch.ones_like(output['model_out']), create_graph=False)[0]
        _, normals, _ = octree_nt_clouds_wn._C.fast_wn_clouds(samples.detach(), octree_wn_dict['coords'], 
                                                              octree_wn_dict['normals'], octree_wn_dict['areas'], 
                                                              leaf_offsets, leaf_indices, octree_wn_dict['children'], 
                                                              octree_wn_dict['cell_centers'], octree_wn_dict['cell_radii'], 
                                                              octree_wn_dict['cell_normals'], octree_wn_dict['high_inv_delta'])
        return gradients.detach().cpu().numpy(), normals.detach().cpu().numpy()

    return reconstruct_iso_contours(iso_points, geometry_query, spacing, voxel_size=voxel_size, 
                                    search_radius=search_radius, min_points=max(3, len_lim + 1), 
                                    backend=backend, query_batch_size=query_batch_size, return_packed=return_packed, 
                                    max_batch_contouring=max_batch_contouring, graph_backend=graph_backend, profile=profile,)


def voxel_representatives(points, cell_size):
    """Retain an original point closest to each occupied cell's centroid.

    Use a conservative cell size below branch separation, or zero to disable.
    Returns indices; representatives are never averaged off the isocontour.
    """
    if len(points) == 0:
        return np.empty(0, dtype=np.int64)
    _, unique = np.unique(points, axis=0, return_index=True)
    unique.sort()
    if cell_size <= 0:
        return unique
    p = points[unique]
    cells = np.floor((p - p.min(axis=0)) / cell_size).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    centers = np.zeros((len(counts), 3))
    np.add.at(centers, inverse, p)
    centers /= counts[:, None]
    distance = np.square(p - centers[inverse]).sum(axis=1)
    order = np.lexsort((unique, distance, inverse))
    first = np.r_[True, inverse[order[1:]] != inverse[order[:-1]]]
    return np.sort(unique[order[first]])


def contour_tangents(points, gradients, normals, tangent_ratio=0.05):
    """Field tangents with PCA fallback only in confidently curve-like neighborhoods."""
    normals = np.asarray(normals, dtype=np.float64)
    normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    tangents = np.cross(normals, gradients)
    lengths = np.linalg.norm(tangents, axis=1)
    weak = (lengths <= 1e-10) | (lengths < tangent_ratio * np.linalg.norm(gradients, axis=1))
    if weak.any() and len(points) >= 3:
        tree = cKDTree(points)
        _, neighbors = tree.query(points[weak], k=min(8, len(points)))
        weak_indices = np.flatnonzero(weak)
        tangents[weak] = 0
        # Batched eigensolves replace a Python loop over every weak-gradient point.
        for start in range(0, len(weak_indices), 32768):
            indices = weak_indices[start:start + 32768]
            local = points[neighbors[start:start + 32768]]
            local = local - local.mean(axis=1, keepdims=True)
            values, vectors = np.linalg.eigh(local.transpose(0, 2, 1) @ local)
            confident = values[:, -1] > 4 * np.maximum(values[:, -2], 1e-20)
            selected = indices[confident]
            tangent = vectors[confident, :, -1]
            tangent = tangent - np.sum(tangent * normals[selected], axis=1, keepdims=True) * normals[selected]
            tangents[selected] = tangent
    lengths = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = np.divide(tangents, lengths, out=np.zeros_like(tangents), where=lengths > 1e-12)
    return tangents.astype(np.float32), normals.astype(np.float32)


def neighbor_candidates(points, tangents, normals, radius, k=4, min_alignment=0.5, min_normal_alignment=0.5, backend='auto', iso_ids=None):
    """Return Nx2xK candidate IDs and scores, separated by tangent half-space."""
    if radius <= 0 or not np.isfinite(radius) or not 1 <= k <= 16 or not 0 <= min_alignment <= 1 or not 0 <= min_normal_alignment <= 1:
        raise ValueError('Invalid neighbor search parameters.')
    iso_ids = np.zeros(len(points), dtype=np.int64) if iso_ids is None else np.asarray(iso_ids, dtype=np.int64)
    if iso_ids.shape != (len(points),) or np.any(iso_ids < 0):
        raise ValueError('iso_ids must be a nonnegative N-vector.')
    if backend not in ('auto', 'cpu', 'cuda'):
        raise ValueError('backend must be auto, cpu, or cuda')
    if backend == 'cuda' or (backend == 'auto' and torch.cuda.is_available()):
        try:
            from contour_connect import score_neighbors
        except ImportError as error:
            raise ImportError('Build the extension: cd submodules/contour_connect && python -m pip install . --no-build-isolation; or use backend="cpu".') from error
        p, t, n = [torch.as_tensor(a, dtype=torch.float32, device='cuda').contiguous() for a in (points, tangents, normals)]
        if not len(p):
            return np.empty((0, 2, k), dtype=np.int64), np.empty((0, 2, k), dtype=np.float32)
        cells = torch.floor((p - p.amin(dim=0)) / radius).long()
        dims = cells.amax(dim=0) + 1
        if np.prod(dims.cpu().numpy().astype(object)) * (int(iso_ids.max()) + 1) >= 2**63:
            raise ValueError('Search grid exceeds int64 key capacity.')
        levels = torch.as_tensor(iso_ids, dtype=torch.int64, device=p.device).contiguous()
        keys = ((levels * dims[0] + cells[:, 0]) * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
        keys, order = keys.sort(stable=True)
        ids, scores = score_neighbors(p, t, n, cells.contiguous(), keys, order, dims, levels, radius, min_alignment, min_normal_alignment, k)
        return ids.cpu().numpy(), scores.cpu().numpy()
    ids = np.full((len(points), 2, k), -1, dtype=np.int64)
    scores = np.full(ids.shape, np.inf, dtype=np.float32)
    if not len(points):
        return ids, scores
    tree = cKDTree(points)
    for i, neighbors in enumerate(tree.query_ball_point(points, radius)):
        candidates = [[], []]
        for j in neighbors:
            if i == j or iso_ids[i] != iso_ids[j]:
                continue
            d = points[j] - points[i]
            distance = np.linalg.norm(d)
            if distance == 0:
                continue
            dot_i, dot_j = np.dot(d, tangents[i]) / distance, np.dot(d, tangents[j]) / distance
            if abs(dot_i) < min_alignment or abs(dot_j) < min_alignment or abs(np.dot(normals[i], normals[j])) < min_normal_alignment:
                continue
            score = distance / radius + 2 * (1 - abs(dot_i)) + 2 * (1 - abs(dot_j))
            candidates[int(dot_i >= 0)].append((score, j))
        for side in range(2):
            for rank, (score, j) in enumerate(sorted(candidates[side])[:k]):
                ids[i, side, rank], scores[i, side, rank] = j, score
    return ids, scores


def trace_candidate_graph(points, tangents, ids, scores, min_points=3):
    """Resolve proposals into degree-two components; trace each component once."""
    proposals = {}
    for i in range(len(points)):
        for j, score in zip(ids[i].ravel(), scores[i].ravel()):
            if j >= 0 and np.isfinite(score):
                proposals[i, int(j)] = min(float(score), proposals.get((i, int(j)), np.inf))
    edges = []
    for i, j in sorted({tuple(sorted(edge)) for edge in proposals}):
        mutual = (i, j) in proposals and (j, i) in proposals
        score = max(proposals.get((i, j), -np.inf), proposals.get((j, i), -np.inf))
        edges.append((not mutual, score, i, j))
    slots = np.full((len(points), 2), -1, dtype=np.int64)
    parent = np.arange(len(points))
    size = np.ones(len(points), dtype=np.int64)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for _, _, i, j in sorted(edges):
        d = points[j] - points[i]
        si, sj = int(np.dot(d, tangents[i]) >= 0), int(np.dot(-d, tangents[j]) >= 0)
        if slots[i, si] >= 0 or slots[j, sj] >= 0:
            continue
        ri, rj = root(i), root(j)
        if ri == rj and size[ri] < max(3, min_points):
            continue
        slots[i, si], slots[j, sj] = j, i
        if ri != rj:
            parent[rj] = ri
            size[ri] += size[rj]
    degree = (slots >= 0).sum(axis=1)
    starts = np.r_[np.flatnonzero(degree == 1), np.flatnonzero(degree == 2)]
    visited = np.zeros(len(points), dtype=bool)
    chains, closed = [], []
    for start in starts:
        if visited[start]:
            continue
        chain, previous, current = [], -1, int(start)
        is_closed = False
        while not visited[current]:
            visited[current] = True
            chain.append(current)
            next_points = [int(j) for j in slots[current] if j >= 0 and j != previous]
            if not next_points:
                break
            following = next_points[0]
            if following == start:
                is_closed = True
                break
            previous, current = current, following
        if len(chain) >= min_points:
            if is_closed:
                chain.append(int(start))
            chains.append(points[chain].copy())
            closed.append(is_closed)
    return chains, closed, int((degree == 0).sum())


def trace_graphs_batch(points, tangents, ids, scores, offsets, min_points=3, graph_backend='auto'):
    """Native per-level parallel graph pass, with an explicit Python reference fallback."""
    if graph_backend not in ('auto', 'native', 'python'):
        raise ValueError('graph_backend must be auto, native, or python')
    native = None
    if graph_backend != 'python':
        try:
            from contour_connect import trace_graph_batch as native
        except ImportError:
            if graph_backend == 'native':
                raise ImportError('Rebuild submodules/contour_connect to enable native graph tracing.')
    if native is not None:
        tensors = [torch.from_numpy(np.ascontiguousarray(a, dtype=dtype)) for a, dtype in ((points, np.float32), (tangents, np.float32), (ids, np.int64), (scores, np.float32), (offsets, np.int64))]
        indices, contour_offsets, iso_offsets, closed, isolated = [a.numpy() for a in native(*tensors, min_points)]
        ordered = points[indices]
        result = []
        for level in range(len(offsets) - 1):
            start, end = iso_offsets[level:level + 2]
            chains = [ordered[contour_offsets[c]:contour_offsets[c + 1]] for c in range(start, end)]
            result.append((chains, closed[start:end].tolist(), int(isolated[level])))
        return result
    result = []
    for start, end in zip(offsets[:-1], offsets[1:]):
        local_ids = ids[start:end].copy()
        present = local_ids >= 0
        if np.any((local_ids[present] < start) | (local_ids[present] >= end)):
            raise RuntimeError('Neighbor candidate crossed iso-value boundaries.')
        local_ids[present] -= start
        result.append(trace_candidate_graph(points[start:end], tangents[start:end], local_ids, scores[start:end], min_points))
    return result


def pack_contours(contours, closure):
    """Ragged storage: points + contour offsets + iso offsets; loops repeat start."""
    arrays, offsets, iso_offsets, flags = [], [0], [0], []
    for iso, chains in contours.items():
        for chain, closed in zip(chains, closure[iso]):
            arrays.append(chain)
            offsets.append(offsets[-1] + len(chain))
            flags.append(closed)
        iso_offsets.append(len(arrays))
    return {'points': np.concatenate(arrays).astype(np.float32) if arrays else np.empty((0, 3), np.float32), 'contour_offsets': np.asarray(offsets, np.int64), 'iso_offsets': np.asarray(iso_offsets, np.int64), 'iso_values': np.asarray(list(contours)), 'closed': np.asarray(flags, bool)}


def reconstruct_iso_contours(iso_points, geometry_query, spacing, voxel_size=None, search_radius=None, 
                             min_points=3, backend='auto', query_batch_size=32768, min_alignment=0.5, 
                             min_normal_alignment=0.5, candidates_per_side=4, return_packed=False, 
                             max_batch_contouring=300, graph_backend="auto", profile=False):
    """Pack up to max_batch_contouring iso clouds into each neighbor-scoring launch.

    Geometry is queried once per retained point, in independent query_batch_size chunks.
    CUDA scores points across packed levels; native CPU workers trace independent graphs in parallel.
    """
    if not np.isfinite(spacing) or spacing <= 0 or query_batch_size < 1 or min_points < 2:
        raise ValueError('spacing, query_batch_size, and min_points must be positive (min_points >= 2).')
    if not isinstance(max_batch_contouring, (int, np.integer)) or max_batch_contouring < 1:
        raise ValueError('max_batch_contouring must be a positive integer number of iso clouds.')
    voxel_size = spacing / 4 if voxel_size is None else voxel_size
    search_radius = 2.5 * spacing if search_radius is None else search_radius
    if not np.isfinite(voxel_size) or voxel_size < 0:
        raise ValueError('voxel_size must be finite and nonnegative.')
    contours = {iso: [] for iso in iso_points}
    closure = {iso: [] for iso in iso_points}
    items = list(iso_points.items())
    timings = dict(thinning=0., geometry=0., tangents=0., neighbors=0., graph=0.)
    for batch_start in range(0, len(items), max_batch_contouring):
        phase_start = time.perf_counter()
        clouds = []
        for iso, cloud in items[batch_start:batch_start + max_batch_contouring]:
            points = np.asarray(cloud, dtype=np.float32).reshape(-1, 3)
            points = points[np.isfinite(points).all(axis=1)]
            points = points[voxel_representatives(points, voxel_size)]
            if len(points) >= min_points:
                clouds.append((iso, points))
        if not clouds:
            continue
        packed = np.concatenate([points for _, points in clouds])
        timings["thinning"] += time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        gradients, normals = [], []
        for start in range(0, len(packed), query_batch_size):
            grad, normal = geometry_query(packed[start:start + query_batch_size])
            gradients.append(np.asarray(grad)); normals.append(np.asarray(normal))
        gradients, normals = np.concatenate(gradients), np.concatenate(normals)
        if gradients.shape != packed.shape or normals.shape != packed.shape:
            raise ValueError('geometry_query must return two Nx3 arrays.')
        timings["geometry"] += time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        prepared = []
        start = 0
        for iso, points in clouds:
            end = start + len(points)
            grad, normal = gradients[start:end], normals[start:end]
            start = end
            valid = np.isfinite(grad).all(axis=1) & np.isfinite(normal).all(axis=1) & (np.linalg.norm(normal, axis=1) > 1e-12)
            points, grad, normal = points[valid], grad[valid], normal[valid]
            if len(points) >= min_points:
                tangent, normal = contour_tangents(points, grad, normal)
                prepared.append((iso, points, tangent, normal))
        if not prepared:
            continue
        packed_points = np.concatenate([c[1] for c in prepared])
        packed_tangents = np.concatenate([c[2] for c in prepared])
        packed_normals = np.concatenate([c[3] for c in prepared])
        iso_ids = np.repeat(np.arange(len(prepared), dtype=np.int64), [len(c[1]) for c in prepared])
        timings["tangents"] += time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        ids, scores = neighbor_candidates(packed_points, packed_tangents, packed_normals, search_radius, candidates_per_side, min_alignment, min_normal_alignment, backend, iso_ids=iso_ids)
        timings["neighbors"] += time.perf_counter() - phase_start
        phase_start = time.perf_counter()
        offsets = np.r_[0, np.cumsum([len(c[1]) for c in prepared])].astype(np.int64)
        resolved_backend = 'native' if graph_backend == 'auto' and (backend == 'cuda' or (backend == 'auto' and torch.cuda.is_available())) else graph_backend
        graphs = trace_graphs_batch(packed_points, packed_tangents, ids, scores, offsets, min_points, resolved_backend)
        for (iso, _, _, _), (chains, closed, isolated) in zip(prepared, graphs):
            contours[iso], closure[iso] = chains, closed
        timings["graph"] += time.perf_counter() - phase_start
    if profile:
        print('[contouring timing] ' + ', '.join(f'{name}={seconds:.3f}s' for name, seconds in timings.items()))
    return pack_contours(contours, closure) if return_packed else contours

import os
import logging
import numpy as np
import time
import torch
import torch.nn.functional as F
import igl

from utils import field_query_with_grads, field_querying, load_udf_pcd
from vis import *


def gen_voxle_queries(N=128, cube_size=1.0):
    """generate voxel queries in [-cube_size, cube_size]^3 cube"""

    voxel_origin = np.array([-cube_size, -cube_size, -cube_size])  # shape (3,)
    voxel_size = cube_size * 2.0 / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)
    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index // N) % N
    samples[:, 0] = (overall_index // (N * N)) % N
    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    return samples, voxel_origin, voxel_size


    
def create_udf_volume(decoder, save_path, dataset_paras, N=256, max_batch=48 ** 3):
    """
    Original training pcd UDF value distribution
    querying UDF and UDF grads in volume.
    """
    # load pcd 
    surf_coords = load_udf_pcd(dataset_paras.get('point_cloud_path'))[0]
    surf_coords = torch.from_numpy(surf_coords).float()

    surf_fields = field_querying(decoder, surf_coords, max_batch=max_batch, no_print=False, return_on_cuda=True)
    hist_item(surf_fields.cpu().numpy(), bins=100)


    # generate voxel queries
    samples, voxel_origin, voxel_size = gen_voxle_queries(N, cube_size=1.0)
    _, udf_fields, udf_grads = field_query_with_grads(decoder, samples, max_batch=max_batch, no_print=True, return_on_cuda=False)
    grad_norms = torch.norm(udf_grads, dim=-1)

    udf_volume = udf_fields.reshape(N, N, N).cpu().numpy()
    udf_gradnorm_volume = grad_norms.reshape(N, N, N).cpu().numpy()

    udf_volume_fn = os.path.join(save_path, 'nn_udf_volume')
    udf_gradnorm_fn = os.path.join(save_path, 'nn_udf_gradnorm_volume')
    np.savez(udf_volume_fn+'.npz', sdf=udf_volume, voxel_grid_origin=voxel_origin, voxel_size=voxel_size)
    np.savez(udf_gradnorm_fn+'.npz', sdf=udf_gradnorm_volume, voxel_grid_origin=voxel_origin, voxel_size=voxel_size)
    print(f'saved udf volume and udf gradnorm volume...')



    
def create_wn_volume(save_path, dataset_paras, N=256, max_batch=48 ** 3):
    """
    Original training pcd UDF value distribution
    querying UDF and UDF grads in volume.
    """
    # load pcd 
    from level_collision import build_init_octree_with_mesh, build_init_octree_with_pcd
    from wn_utils import fast_octree_wn
    
    mesh_path = dataset_paras.get('mesh_path', None)
    if mesh_path is not None and os.path.exists(mesh_path):
        octree_wn_dict = build_init_octree_with_mesh(mesh_path)

    elif dataset_paras.get('point_cloud_path', None) is not None:
        pcd_path = dataset_paras.get('point_cloud_path')
        octree_wn_dict = build_init_octree_with_pcd(pcd_path)

    # volume queries
    N = dataset_paras.get('resolution')
    samples, voxel_origin, voxel_size = gen_voxle_queries(N=N, cube_size=1.0)
    samples = samples.cuda()

    wn_fields, wn_grads = fast_octree_wn(samples, octree_wn_dict,)
    wn_gradnorms = torch.norm(wn_grads, dim=-1)

    wn_volume = wn_fields.reshape(N, N, N).cpu().numpy()
    wn_gradnorm_volume = wn_gradnorms.reshape(N, N, N).cpu().numpy()

    wn_volume_fn = os.path.join(save_path, 'wn_volume')
    wn_gradnorm_fn = os.path.join(save_path, 'wn_gradnorm_volume')
    np.savez(wn_volume_fn+'.npz', sdf=wn_volume, voxel_grid_origin=voxel_origin, voxel_size=voxel_size)
    np.savez(wn_gradnorm_fn+'.npz', sdf=wn_gradnorm_volume, voxel_grid_origin=voxel_origin, voxel_size=voxel_size)
    print(f'saved wn volume and wn gradnorm volume...')


def create_tv_wn_volume(waydata=None, batch_size=None, dataset_paras=None, save_path=None, N=256, max_batch=48 ** 3):
    """
    time-varying wn volume and wn gradnorm volume at the middle batch timestep.
    also save waypt coords through the current timestep, inclusive.
    """
    from level_collision import fast_time_octree_wn

    # volume queries
    samples, voxel_origin, voxel_size = gen_voxle_queries(N=N, cube_size=1.0)

    # pick original timestep and map to clean shell timestep
    waypt_len = waydata.waypt_len
    coords = waydata.coords

    batch_queries = samples.unsqueeze(0).repeat(batch_size, 1, 1).float().cuda().contiguous()
    batch_indices = torch.linspace(1, waypt_len - 1, steps=batch_size).round().long()

    shell_ts= waydata.get_bacth_waydata(batch_indices.numpy())[-2]
    shell_ts = shell_ts.cuda().contiguous()
    time_octree_wn_dict = move_dict_to_device(waydata.shell_time_octree_wn_dict, device='cuda')

    start = time.time()
    wn_fields, wn_grads = fast_time_octree_wn(
        batch_queries, shell_ts, time_octree_wn_dict
    )
    print(f'fast_time_octree_wn {batch_size}*{N}^3 in {time.time() - start:.6f} seconds')

    # NOTE dirty quick check for infill outer shell tv-wn
    # test infill outer shell in tree with only one infill region
    # outer_shell_ts_dict = waydata.get_bacth_waydata(batch_indices.numpy())[-1]
    # shell_ts = outer_shell_ts_dict[2].cuda().contiguous()
    # time_octree_wn_dict = move_dict_to_device(waydata.outer_shells_time_octree_wn_dict_tuple[2][0], device='cuda')

    # start = time.time()
    # wn_fields, wn_grads = fast_time_octree_wn(
    #     batch_queries, shell_ts, time_octree_wn_dict
    # )
    # print(f'fast_time_octree_wn {batch_size}*{N}^3 in {time.time() - start:.6f} seconds')


    # ! grab one to save for vis check
    pick_idx = batch_size // 3
    wn_volume = wn_fields[pick_idx].reshape(N, N, N).cpu().numpy()
    wn_gradnorm_volume = torch.norm(wn_grads[pick_idx], dim=-1).reshape(N, N, N).cpu().numpy()
    till_coords = coords[:batch_indices[pick_idx].item()+1]

    wn_volume_fn = os.path.join(save_path, 'tv_wn_volume')
    wn_gradnorm_fn = os.path.join(save_path, 'tv_wn_gradnorm_volume')
    waypt_coords_fn = os.path.join(save_path, 'tv_waypt_coords.xyz')
    np.savez(wn_volume_fn+'.npz', sdf=wn_volume, voxel_grid_origin=voxel_origin, voxel_size=voxel_size)
    np.savez(wn_gradnorm_fn+'.npz', sdf=wn_gradnorm_volume, voxel_grid_origin=voxel_origin, voxel_size=voxel_size)
    np.savetxt(waypt_coords_fn, till_coords, fmt='%.6f')

    vis_pcd_fields(till_coords, label='till waypt coords')




def move_dict_to_device(data, device):
    """Recursively move tensors in a nested structure to a specified device."""
    if isinstance(data, dict):
        return {key: move_dict_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_dict_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_dict_to_device(item, device) for item in data)
    elif hasattr(data, 'to'):
        return data.to(device=device)
    else:
        return data


def surf_extraction_volume_wn(decoder, save_path=None, dataset_paras=None,
                              spec_N=128, sample_num=100000, 
                              init_small_level=1e-2, max_iter=10, ray_sample_num=100,
                              wn_gradnorm_valid_th=200.0/12.0,
                              enable_eval_metrics=False):
    
    """
    extract surface pts from
    1. tracing min udf level
    2. tracing min udf gradnorm
    3. tracing max wn gradnorm

    use Chamfer distance and Hausdorff distance to evaluate the traced points to gt mesh (resampled)
    if only pcd, no metrics evaluation as interpolation is not accurate 

    wn_gradnorm_valid_th: threshold when using max wn gradnorm tracing, 
    in most cases, small value is enough, larger value to avoid open-bn capping points and skeleton points
    """
    from level_collision import build_init_octree_with_mesh, build_init_octree_with_pcd
    from wn_utils import tracing_min_udf_level, tracing_min_udf_gradnorm, once_tracing_max_wngrad_octree
    
    mesh_path = dataset_paras.get('mesh_path', None)
    if mesh_path is not None and os.path.exists(mesh_path):
        verts, faces = igl.read_triangle_mesh(mesh_path)
        octree_wn_dict = build_init_octree_with_mesh(mesh_path)
        gt_pts = sample_pts_from_mesh(verts, faces, num_pts=sample_num*2)

    elif dataset_paras.get('point_cloud_path', None) is not None:
        pcd_path = dataset_paras.get('point_cloud_path')
        octree_wn_dict = build_init_octree_with_pcd(pcd_path)
        gt_pts = None  # no gt mesh, so no evaluation metrics

    # initial samples
    if spec_N is not None:
        N = spec_N
    else:   
        N = dataset_paras.get('resolution')
    samples, voxel_origin, voxel_size = gen_voxle_queries(N=N, cube_size=1.0)

    udf_fields = field_querying(decoder, samples, max_batch=48 ** 3, no_print=False, return_on_cuda=False).squeeze()
    
    # get init points for bn tracing with udf level method
    init_mask = udf_fields < init_small_level
    init_points = samples[init_mask].detach().cuda()
    max_init_len = init_small_level * 2.0


    # different tracing methods
    # 1 tracing min udf level
    traced_coords_udf, udf_level_dists, valid_mask_udf_level = \
        tracing_min_udf_level(decoder, init_points, max_iter, ray_sample_num, max_init_len, eps_hit=1.0e-3)
    
    valid_mask_udf_level = valid_mask_udf_level.cpu().numpy()
    udf_level_dists = udf_level_dists.cpu().numpy()
    udf_level_dists[~valid_mask_udf_level] = np.nan
    non_nan_mask = ~np.isnan(udf_level_dists)
    coords_udf = traced_coords_udf.detach().cpu().numpy()[non_nan_mask]

    print(f'mean UDF level tracing non-nan ratio {np.sum(non_nan_mask)/len(non_nan_mask):.2%}')


    # 2 tracing min udf gradnorm
    traced_coords_udfgrad, udfgrad_level_dists, valid_mask_udfgrad_level = \
        tracing_min_udf_gradnorm(decoder, init_points, max_iter, ray_sample_num, max_init_len, eps_hit=1.0e-1)
    valid_mask_udfgrad_level = valid_mask_udfgrad_level.cpu().numpy()
    
    udfgrad_level_dists = udfgrad_level_dists.cpu().numpy()
    traced_coords_udfs = field_querying(decoder, traced_coords_udfgrad, max_batch=48 ** 3, 
                                        no_print=True, return_on_cuda=False).squeeze().cpu().numpy()
    # filter out points whose udf is too large, seemingly close to skeleton
    udfgrad_level_dists[traced_coords_udfs > 0.001] = np.nan  
    udfgrad_level_dists[~valid_mask_udfgrad_level] = np.nan
    non_nan_mask = ~np.isnan(udfgrad_level_dists)
    coords_udfgrad = traced_coords_udfgrad.detach().cpu().numpy()[non_nan_mask]

    print(f'mean UDF gradnorm tracing dists non-nan ratio {np.sum(non_nan_mask)/len(non_nan_mask):.2%}')


    # 3 tracing max wn gradnorm
    coords_wn = init_points.clone()
    valid_mask_total = torch.zeros(init_points.shape[0], dtype=torch.bool, device=init_points.device,)
    for i in range(max_iter):
        coords_wn_tmp, _, valid_mask_wn = \
            once_tracing_max_wngrad_octree(init_points, decoder, octree_wn_dict, ray_sample_num=ray_sample_num,
                                           ori_inv_delta=32.0, use_quadratic_refine=True,
                                           wn_gradnorm_valid_th=wn_gradnorm_valid_th,)  
        coords_wn = coords_wn_tmp.clone()
        valid_mask_total |= valid_mask_wn

        # Keep valid points fixed next round
        init_points = coords_wn.clone()
        if valid_mask_total.all():
            break
        print(f'iter {i} valid_mask_total.sum() {valid_mask_total.sum().item()} / {valid_mask_total.shape[0]}')
         
    coords_wn = coords_wn.detach().cpu().numpy()
    valid_mask_total = valid_mask_total.cpu().numpy()
    coords_wn = coords_wn[valid_mask_total]

    # render extracted pcd
    from vis import render_pcd, vis_two_set_pcd
    render_pcd(coords_udf, label='min udf level tracing')
    render_pcd(coords_udfgrad, label='min udfgrad level tracing')
    render_pcd(coords_wn, label='max wngrad tracing')

    # evaluate metrics if gt mesh is available
    if enable_eval_metrics and gt_pts is not None:

        # since training normalization max dim not touches [-1, 1]
        # do normalization first with max touches [-1, 1] then evaluate metrics
        # ! espeically compare with other baselines 
        train_pcd = np.genfromtxt(dataset_paras.get('point_cloud_path'))[:, :3]
        coords_udf = normalize_to_unit(coords_udf, ref_pcd=train_pcd)
        coords_udfgrad = normalize_to_unit(coords_udfgrad, ref_pcd=train_pcd)
        coords_wn = normalize_to_unit(coords_wn, ref_pcd=train_pcd)
        gt_pts = normalize_to_unit(gt_pts, ref_pcd=train_pcd)

        if gt_pts is not None:
            vis_two_set_pcd(coords_udf, gt_pts, pcd1_size=10, pcd2_size=10, title='traced udf level points vs gt pts')
            vis_two_set_pcd(coords_udfgrad, gt_pts, pcd1_size=10, pcd2_size=10, title='traced udf grad level points vs gt pts')
            vis_two_set_pcd(coords_wn, gt_pts, pcd1_size=10, pcd2_size=10, title='traced wn grad points vs gt pts')

        cd1_lvl, cd2_lvl, hd_lvl, hd_percent_lvl, f1_score_lvl, f2_score_lvl = \
            eval_dists(coords_udf, gt_pts, f1_th=5e-3, f2_th=1e-2, hd_percentile=99.0, sample_num=sample_num)
        print(f'UDF level tracing --- cd1: {cd1_lvl:.6f}, cd2: {cd2_lvl:.6f}, hd: {hd_lvl:.6f}, f_score_5e-3: {f1_score_lvl:.4f}, f_score_1e-2: {f2_score_lvl:.4f}')

        cd1_grad, cd2_grad, hd_grad, hd_percent_grad, f1_score_grad, f2_score_grad = \
            eval_dists(coords_udfgrad, gt_pts, f1_th=5e-3, f2_th=1e-2, hd_percentile=99.0, sample_num=sample_num)
        print(f'UDF grad tracing --- cd1: {cd1_grad:.6f}, cd2: {cd2_grad:.6f}, hd: {hd_grad:.6f}, f_score_5e-3: {f1_score_grad:.4f}, f_score_1e-2: {f2_score_grad:.4f}')

        cd1_wn, cd2_wn, hd_wn, hd_percent_wn, f1_score_wn, f2_score_wn = \
            eval_dists(coords_wn, gt_pts, f1_th=5e-3, f2_th=1e-2, hd_percentile=99.0, sample_num=sample_num)
        print(f'WN grad tracing --- cd1: {cd1_wn:.6f}, cd2: {cd2_wn:.6f}, hd: {hd_wn:.6f}, f_score_5e-3: {f1_score_wn:.4f}, f_score_1e-2: {f2_score_wn:.4f}')

       
    
def normalize_to_unit(pts, ref_pcd=None, eps=1e-12):
    """
    Normalize points to unit space [-1, 1]^3 using a bounding box.

    The largest bounding-box dimension is scaled to span [-1, 1],
    while preserving the aspect ratio. If ref_pcd is provided, its
    bounding box is used to normalize to [-1, 1]^3; otherwise, use own pts
    """
    pts = np.asarray(pts)

    if ref_pcd is not None:
        ref_pcd = np.asarray(ref_pcd)
        bbx_min = ref_pcd.min(axis=0)
        bbx_max = ref_pcd.max(axis=0)
    else:
        bbx_min = pts.min(axis=0)
        bbx_max = pts.max(axis=0)

    center = (bbx_min + bbx_max) / 2.0

    bbox_size = bbx_max - bbx_min
    scale = bbox_size.max() / 2.0
    scale = max(scale, eps)

    normalized_pts = (pts - center) / scale

    return normalized_pts

    

def sample_pts_from_mesh(verts, faces, num_pts=10000):
    """sample points from mesh surface for evaluation, using area-weighted sampling"""
    # compute face areas
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    face_probs = face_areas / np.sum(face_areas)

    # sample faces according to area
    sampled_face_indices = np.random.choice(len(faces), size=num_pts, p=face_probs)
    sampled_faces = faces[sampled_face_indices]

    # sample points on the selected faces using barycentric coordinates
    r1 = np.random.rand(num_pts)
    r2 = np.random.rand(num_pts)
    sqrt_r1 = np.sqrt(r1)
    bary_coords = np.stack([1 - sqrt_r1, sqrt_r1 * (1 - r2), sqrt_r1 * r2], axis=1)  # (num_pts, 3)

    sampled_points = (
        verts[sampled_faces[:, 0]] * bary_coords[:, 0:1] +
        verts[sampled_faces[:, 1]] * bary_coords[:, 1:2] +
        verts[sampled_faces[:, 2]] * bary_coords[:, 2:3]
    )

    return sampled_points




def eval_dists(pred_pts, gt_pts, f1_th=5e-3, f2_th=1e-2, hd_percentile=95.0, sample_num=None):
    """compute chamfer distance L1, L2, Hausdorff distance, percentile
    Hausdorff distance, and F score between predicted points and gt points.
    
    unit_normalize: True to scale max dim to [-1, 1]. as original normlaized_len may be smaller"""

    import numpy as np
    from scipy.spatial import cKDTree

    pred_pts = np.asarray(pred_pts)
    gt_pts = np.asarray(gt_pts)

    print(f'[i] eval pred pts shape: {pred_pts.shape}, gt pts shape: {gt_pts.shape}')
        
    # print both bbx
    # print('---------------------------------------------')
    # print('gt bbx', gt_pts.min(axis=0), gt_pts.max(axis=0))    
    # print('pred bbx', pred_pts.min(axis=0), pred_pts.max(axis=0))
        
    # randomly sample both sets 
    if sample_num is not None:
        if sample_num > pred_pts.shape[0] or sample_num > gt_pts.shape[0]:
            raise ValueError(f'sample num larger than base numbers')
        pred_pts = pred_pts[np.random.choice(pred_pts.shape[0], size=min(sample_num, pred_pts.shape[0]), replace=False)]
        gt_pts = gt_pts[np.random.choice(gt_pts.shape[0], size=min(sample_num, gt_pts.shape[0]), replace=False)]
        print(f'[i] after sampling {sample_num} points, eval pred pts shape: {pred_pts.shape}, gt pts shape: {gt_pts.shape}')
        print('-------------------------------------')

        # this may have better metrics
        # import point_cloud_utils as pcu 
        # pred_pts_idx = pcu.downsample_point_cloud_poisson_disk(
        #     pred_pts, 0.01,
        #     target_num_samples=sample_num,
        # )
        # gt_pts_idx = pcu.downsample_point_cloud_poisson_disk(
        #     gt_pts, 0.01,
        #     target_num_samples=sample_num,
        # )
        # pred_pts, gt_pts = pred_pts[pred_pts_idx], gt_pts[gt_pts_idx]
        # print(f'[i] after poisson disk sampling {sample_num} points, eval pred pts shape: {pred_pts.shape}, gt pts shape: {gt_pts.shape}')
        # print('-------------------------------------')

    pred_tree = cKDTree(pred_pts)
    gt_tree = cKDTree(gt_pts)

    pred_to_gt, _ = gt_tree.query(pred_pts, k=1)
    gt_to_pred, _ = pred_tree.query(gt_pts, k=1)

    chamfer_l1 = np.mean(pred_to_gt) + np.mean(gt_to_pred)
    chamfer_l2 = np.mean(pred_to_gt ** 2) + np.mean(gt_to_pred ** 2)

    hausdorff = max(np.max(pred_to_gt), np.max(gt_to_pred))

    hausdorff_percentile = max(
        np.percentile(pred_to_gt, hd_percentile),
        np.percentile(gt_to_pred, hd_percentile)
    )

    precision = np.mean(pred_to_gt <= f1_th)
    recall = np.mean(gt_to_pred <= f1_th)
    if precision + recall > 0:
        f1_score = 2 * precision * recall / (precision + recall)
    else:
        f1_score = 0.0

    precision_f2 = np.mean(pred_to_gt <= f2_th)
    recall_f2 = np.mean(gt_to_pred <= f2_th)
    if precision_f2 + recall_f2 > 0:
        f2_score = 2 * precision_f2 * recall_f2 / (precision_f2 + recall_f2)
    else:
        f2_score = 0.0    

    return chamfer_l1, chamfer_l2, hausdorff, hausdorff_percentile, f1_score, f2_score




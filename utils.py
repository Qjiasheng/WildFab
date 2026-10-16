import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import random
import yaml
import time
import diff_operators
import skimage.measure
import torch.nn.functional as F
import trimesh
from vis import vis_edges, vis_elems


def set_seed(seed):
    # Set the seed for PyTorch (both CPU and CUDA)
    torch.manual_seed(seed)
    # If you are using CUDA, make the GPU deterministic
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU
        # Ensure that CUDA operations are deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set the seed for numpy (if you're using it)
    np.random.seed(seed)
    # Set the seed for Python's built-in random module
    random.seed(seed)
    print(f"Seed set to {seed}")


def load_configs(config_path, exp_name):
    # load training parameters
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    # dataset, optim, loss parameters
    # return configs[exp_name]['dataset'], configs[exp_name]['optim'], configs[exp_name]['parameters']
    return configs[exp_name]


def put_configs(config_path, paras):
    with open(config_path, 'w') as f:
        yaml.dump(paras, f)
    print(f"Configs saved to {config_path}")


def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords


def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


def make_scatter_plot(array_2d, draw_contours=True):
    """
    Draw the SDF slice and its contours with the image and the contour
    grid in the *same* (x, y) coordinate system.

    The only changes compared with your original version are:
      1. build the mesh-grid as  (X = columns,  Y = rows)
      2. flip the y-axis so that the first array row appears at the top
         (same convention that `scatter` and most image viewers use)
    """
    height, width = array_2d.shape

    # --- 1. correct axis ordering ------------------------------------
    X, Y = np.meshgrid(np.arange(width),        # X  ← columns
                       np.arange(height))       # Y  ← rows

    X_flat, Y_flat = X.ravel(), Y.ravel()
    intensities     = array_2d.ravel()

    fig, ax = plt.subplots(figsize=(2.75, 2.75), dpi=300)

    scatter = ax.scatter(X_flat, Y_flat,
                         c=intensities,
                         cmap='coolwarm',
                         s=4)

    # blue for negative, red for positive, black for zero
    if draw_contours:
        ax.contour(X, Y, array_2d,
                levels=np.arange(-2.0, 0.0, 0.03),
                colors='b', linewidths=0.4)
        ax.contour(X, Y, array_2d,
                levels=np.arange(0.0, 2.0, 0.03),
                colors='r', linewidths=0.4)
        ax.contour(X, Y, array_2d,
                levels=[0], colors='k', linewidths=1.0)
    

    cbar = fig.colorbar(scatter, ax=ax, label='SDF Value')

    # --- 2. put row 0 at the top so pixels & contours coincide --------
    ax.set_aspect('equal')
    ax.invert_yaxis()          # ← crucial for perfect overlap

    return fig


def write_sdf_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train_'):
    slice_coords_2d = get_mgrid(512)

    with torch.no_grad():
        x_cross, y_cross, z_cross = 0.0, 0.0, 0.0
        yz_slice_coords = torch.cat((torch.ones_like(slice_coords_2d[:, :1]) * x_cross, slice_coords_2d), dim=-1)
        yz_slice_model_input = {'coords': yz_slice_coords.cuda()[None, ...]}

        yz_model_out = model(yz_slice_model_input)
        sdf_values = yz_model_out['model_out']
        sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
        # fig = make_contour_plot(sdf_values)
        fig = make_scatter_plot(sdf_values)
        writer.add_figure(prefix + 'yz_sdf_slice', fig, global_step=total_steps)

        xz_slice_coords = torch.cat((slice_coords_2d[:,:1],
                                     torch.ones_like(slice_coords_2d[:, :1]) * y_cross,
                                     slice_coords_2d[:,-1:]), dim=-1)
        xz_slice_model_input = {'coords': xz_slice_coords.cuda()[None, ...]}

        xz_model_out = model(xz_slice_model_input)
        sdf_values = xz_model_out['model_out']
        sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
        # fig = make_contour_plot(sdf_values)
        fig = make_scatter_plot(sdf_values)
        writer.add_figure(prefix + 'xz_sdf_slice', fig, global_step=total_steps)

        xy_slice_coords = torch.cat((slice_coords_2d[:,:2],
                                     z_cross*torch.ones_like(slice_coords_2d[:, :1])), dim=-1)
        xy_slice_model_input = {'coords': xy_slice_coords.cuda()[None, ...]}

        xy_model_out = model(xy_slice_model_input)
        sdf_values = xy_model_out['model_out']
        sdf_values = lin2img(sdf_values).squeeze().cpu().numpy()
        # fig = make_contour_plot(sdf_values)
        fig = make_scatter_plot(sdf_values)
        writer.add_figure(prefix + 'xy_sdf_slice', fig, global_step=total_steps)


def write_collision_summary(model, model_input, gt, model_output, writer, total_steps, prefix='train_', out_index = 0):
    pass


def field_querying(decoder, samples, max_batch=32 ** 3, no_print=False, return_on_cuda=False, set_grad_flase=True):
    """querying field with given samples
    samples: N x 3 torch tensor
    return fields: N x 1 torch tensor"""

    start = time.time()
    decoder.eval()

    if not isinstance(samples, torch.Tensor):
        samples = torch.from_numpy(samples).float()
    else:
        samples = samples.float()
    
    num_samples = samples.shape[0]
    fields = torch.zeros(num_samples)

    if set_grad_flase:
        samples.requires_grad = False

    head = 0
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), :].cuda()

        sub_output = decoder(sample_subset)
        sub_coords = sub_output['model_in']
        sub_fields = sub_output['model_out']

        fields[head : min(head + max_batch, num_samples)] = sub_fields.squeeze().detach().cpu()
        head += max_batch

    end = time.time()
    if not no_print:
        print("field querying takes: %f" % (end - start))

    if return_on_cuda:
        return fields.cuda()
    return fields


def quat_field_querying(decoder, samples, max_batch=32 ** 3, no_print=False, return_on_cuda=False, set_grad_flase=True):
    """querying field with given samples
    samples: N x 3 torch tensor
    return fields: N x 4 torch tensor"""

    start = time.time()
    decoder.eval()

    if not isinstance(samples, torch.Tensor):
        samples = torch.from_numpy(samples).float()
    else:
        samples = samples.float()
    
    num_samples = samples.shape[0]
    fields = torch.zeros(num_samples, 4)

    if set_grad_flase:
        samples.requires_grad = False

    head = 0
    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), :].cuda()

        sub_output = decoder(sample_subset)
        sub_coords = sub_output['model_in']
        sub_fields = sub_output['model_out']

        fields[head : min(head + max_batch, num_samples), :] = sub_fields.squeeze().detach().cpu()
        head += max_batch

    end = time.time()
    if not no_print:
        print("field querying takes: %f" % (end - start))

    if return_on_cuda:
        return fields.cuda()
    return fields


def field_query_with_grads_no_higher_order(decoder, coords, max_batch=32 ** 3, no_print=False, return_on_cuda=False):
    """Query detached fields and first derivatives for collision evaluation only."""

    start = time.time()
    # check coords on cuda tensor
    if not isinstance(coords, torch.Tensor):
        samples = torch.from_numpy(coords).float()
    else:
        samples = coords.detach().float()
    samples.requires_grad = True

    decoder.eval()

    head = 0
    num_samples = samples.shape[0]
    fields = torch.zeros(num_samples)
    grads = torch.zeros(num_samples, 3)

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), :].cuda()
        
        with torch.enable_grad():   # must enable for custom autograd function
            sub_output = decoder(sample_subset)
            sub_coords = sub_output['model_in']
            sub_fields = sub_output['model_out']

            fields[head : min(head + max_batch, num_samples)] = (
                sub_fields
                .squeeze()
                .detach()
            )

            grads[head : min(head + max_batch, num_samples)] = (
                torch.autograd.grad(sub_fields, sub_coords,
                                    grad_outputs=torch.ones_like(sub_fields), create_graph=False)[0] 
                .squeeze()
                .detach()
            )

        head += max_batch

    if not no_print:
        print('Elapsed time: ', time.time() - start)

    if return_on_cuda:
        return samples.cuda(), fields.cuda(), grads.cuda()

    samples = samples.detach().cpu()
    fields = fields.detach().cpu()
    grads = grads.detach().cpu()
    return samples, fields, grads


def field_query_with_grads(decoder, coords, max_batch=32 ** 3, no_print=False, return_on_cuda=False):
    """seperate func to query field and grads"""

    start = time.time()
    # check coords on cuda tensor
    if not isinstance(coords, torch.Tensor):
        samples = torch.from_numpy(coords).float()
    else:
        samples = coords.float()
    samples.requires_grad = True

    decoder.eval()

    head = 0
    num_samples = samples.shape[0]
    fields = torch.zeros(num_samples)
    grads = torch.zeros(num_samples, 3)

    while head < num_samples:
        sample_subset = samples[head : min(head + max_batch, num_samples), :].cuda()
        
        with torch.enable_grad():   # must enable for custom autograd function
            sub_output = decoder(sample_subset)
            sub_coords = sub_output['model_in']
            sub_fields = sub_output['model_out']

            fields[head : min(head + max_batch, num_samples)] = (
                sub_fields
                .squeeze()
                .detach()
            )

            grads[head : min(head + max_batch, num_samples)] = (
                diff_operators.gradient(sub_fields, sub_coords) 
                .squeeze()
                .detach()
            )

        head += max_batch

    if not no_print:
        print('Elapsed time: ', time.time() - start)

    if return_on_cuda:
        return samples.cuda(), fields.cuda(), grads.cuda()

    samples = samples.detach().cpu()
    fields = fields.detach().cpu()
    grads = grads.detach().cpu()
    return samples, fields, grads


def gen_iso_layers(fields, num_iso_layers, mask, N=128):
    """generate iso-layers from volume fields, mask is to get only interior layers"""

    masked_fields = torch.where(mask, fields, torch.tensor(float('nan')))
    valid_values = masked_fields[~torch.isnan(masked_fields)]  # Extract only valid (non-NaN) values
    min_iso_value = valid_values.min().item()
    max_iso_value = valid_values.max().item()

    # Generate a range of iso-values evenly 
    iso_values = np.linspace(min_iso_value, max_iso_value, num_iso_layers)

    # extract iso-layers
    iso_layers = {}
    for iso_value in iso_values:
        try:
            verts, faces, _, _ = skimage.measure.marching_cubes(
                masked_fields.numpy(), level=iso_value, spacing=(2.0 / N, 2.0 / N, 2.0 / N) # follow our voxel size
            )
            iso_layers[iso_value] = (verts, faces)  
            print(f"Extracted iso-layer for iso-value: {iso_value:.6f}")
        except Exception as e:
            print(f"Failed to extract iso-layer for iso-value: {iso_value:.6f}, Error: {e}")

    return iso_layers


def load_nozzle_pcd(node_fn, elem_fn, object_scale, sample_num=10000, elem_type='edge'):
    """
    load nozzle pcd (physical scale in mm) with origin at tooltip.
    scale into unit cube based on object scale.

    node -- (xyz, sdf_grads) 
    elems -- (node idx 1, 2, 3, 4) for a tet elem. convert into edges for futhur using
    """
    node_data = np.loadtxt(node_fn, dtype=np.float32)
    nodes, node_grads = node_data[:, :3], node_data[:, 3:6]

    extrusion_thickness = 2.0  # mm
    nodes[:, 2] = nodes[:, 2] + extrusion_thickness 

    from vis import vis_grad
    vis_grad(nodes, node_grads, arrow_scale=10.0)

    elems = np.loadtxt(elem_fn, skiprows=1, usecols=(1, 2, 3, 4), dtype=np.int32)
    nodes_cp, node_grads_cp = nodes.copy(), node_grads.copy()
    print(f"Original nozzle nodes: {nodes.shape[0]}, elems: {elems.shape[0]}")

    nodes = nodes / object_scale  # scale to [-1, 1] cube

    if elem_type == 'edge':
        # ----------- edges, from tet elems (has 6 edges), obtain unique edges 
        edge_tpl = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]], dtype=np.int64)
        edges = elems[:, edge_tpl].reshape(-1, 2)
        edges = np.sort(edges, axis=1)
        edges_unique, edge_inverse = np.unique(edges, axis=0, return_inverse=True)  # get unique edges first

        n_unique_edges = edges_unique.shape[0]
        print(f"Unique edges: {n_unique_edges}")
        unique_edge_idx = np.random.choice(n_unique_edges, min(sample_num, n_unique_edges), replace=False)
        chosen_edges = edges_unique[unique_edge_idx]

        chosen_ids, inverse = np.unique(chosen_edges.ravel(), return_inverse=True)
        edges_local = inverse.reshape(chosen_edges.shape)
        chosen_nodes = nodes[chosen_ids]
        chosen_nodes_grads = node_grads[chosen_ids]
        print(f"Unique nodes: {chosen_nodes.shape[0]}, avoid {min(sample_num, n_unique_edges)*2 - chosen_nodes.shape[0]} repetitions.")

        vis_edges(nodes, chosen_edges, down=1, nodes_ori=nodes_cp/ object_scale)
        return chosen_nodes, chosen_nodes_grads, edges_local
    
    elif elem_type == 'tet':   
        # ----------- tets, random sample, unique pcd for wn field querying, elems_local for ele reconvery
        n_elems = elems.shape[0]
        print(f"Total elems: {n_elems}")
        elem_idx = np.random.choice(n_elems, min(sample_num, n_elems), replace=False)
        chosen_elems = elems[elem_idx]
        chosen_ids, inverse = np.unique(chosen_elems.ravel(), return_inverse=True)
        elems_local = inverse.reshape(chosen_elems.shape)
        chosen_nodes = nodes[chosen_ids]
        chosen_nodes_grads = node_grads[chosen_ids]
        print(f"Unique nodes: {chosen_nodes.shape[0]}, avoid {min(sample_num, n_elems)*4 - chosen_nodes.shape[0]} repetitions.")

        vis_elems(nodes, chosen_elems, down=1, nodes_ori=nodes_cp/ object_scale)  
        return chosen_nodes, chosen_nodes_grads, elems_local
    
    else:
        raise NotImplementedError(f"elem_type {elem_type} not implemented. Only support 'edge' and 'tet'.")


def load_nozzle_data(nozzle_nodes_path, nozzle_elems_path, object_scale, sample_num=50000, device='cuda'):
    nozzle_coords, nozzle_grads, local_map = \
        load_nozzle_pcd(nozzle_nodes_path, nozzle_elems_path, object_scale=object_scale, sample_num=sample_num)
    
    nozzle_coords = torch.from_numpy(nozzle_coords).float().to(device)
    nozzle_grads = torch.from_numpy(nozzle_grads).float().to(device)
    local_map = torch.from_numpy(local_map).long().to(device)     # MUST be long for indexing later

    nozzle_grads = F.normalize(nozzle_grads, dim=-1, p=2)

    # vis check
    # from vis import vis_grad
    # vis_grad(nozzle_coords.cpu().numpy(), nozzle_grads.cpu().numpy(), label='Nozzle Grads')
    
    return nozzle_coords, nozzle_grads, local_map


def load_nozzle_mesh(nozzle_path, object_scale):
    mesh = trimesh.load(nozzle_path)
    # mesh.apply_scale(10.0 / object_scale)  # cm
    mesh.apply_scale(1.0 / object_scale) # mm
    return mesh


def quaternion_rotate(q, v):
    """Rotate vector v using quaternion q (supports broadcasting)."""
    q_vec = q[..., 1:]  # Extract vector part (x, y, z) -> shape (..., 3)
    q_scalar = q[..., :1]  # Extract scalar part (w) -> shape (..., 1)
    # Ensure q and v have the same batch dimensions
    t = 2 * torch.cross(q_vec, v, dim=-1)
    v_rot = v + q_scalar * t + torch.cross(q_vec, t, dim=-1)
    return v_rot


def transform_nozzle_pcd_with_quaternion(nozzle_pcd, waypts, quaternions):
    """
    transform nozzle pcd at each waypoint given quaternions (M, 4).
    quaternions as rotation for local frame to each waypoint.
    """
    # normalize quaternions
    quaternions = F.normalize(quaternions, dim=-1)
    transed_nozzle_pcd = quaternion_rotate(quaternions[:, None, :], nozzle_pcd[None, :, :]) 
    transed_nozzle_pcd += waypts[:, None, :]  # (M, N, 3)
    return transed_nozzle_pcd


def transform_nozzle_grads_with_quaternion(nozzle_grads, waypts, quaternions):
    """
    transform nozzle grads at each waypoint given quaternions (M, 4).
    quaternions as rotation for local frame to each waypoint.
    NOTE grads only rotate. no translation 
    """
    # normalize quaternions
    quaternions = F.normalize(quaternions, dim=-1)
    transed_nozzle_grad = quaternion_rotate(quaternions[:, None, :], nozzle_grads[None, :, :])
    # transed_nozzle_pcd += waypts[:, None, :]  # (M, N, 3)
    return transed_nozzle_grad


def quaternion_para_axes(quaternions):
    """get print directions and frame axes from quaternions
    quaternions as rotation for local frame to each waypoint.
    print_dirs ---- nozzle z+
    frame_dirs ---- nozzle x+
    """
    N = quaternions.shape[0]
    x_ref = torch.tensor([1., 0., 0.], device=quaternions.device).unsqueeze(0).expand(N, -1)
    z_ref = torch.tensor([0., 0., 1.], device=quaternions.device).unsqueeze(0).expand(N, -1)

    # normalize quaternions
    quaternions = F.normalize(quaternions, dim=-1)
    # get print directions and frame axes from local frames
    waypt_dirs = quaternion_rotate(quaternions, z_ref) 
    frame_dirs = quaternion_rotate(quaternions, x_ref) 
    waypt_dirs = F.normalize(waypt_dirs, dim=-1) 
    frame_dirs = F.normalize(frame_dirs, dim=-1)

    return waypt_dirs, frame_dirs


def load_udf_pcd(pcd_fn, down_ratio=1):
    """
    [xyz, nxnynz, curv_dir1, curv_dir2, heat_grad, curv_feature_tag, bn_tag, nm_tag, app_placeholder_tag]
    """
    data = np.genfromtxt(pcd_fn)
    coords = data[:, :3]
    normals = data[:, 3:6]
    pd1s = data[:, 6:9]                     # curv vec1
    pd2s = data[:, 9:12]                    # curv vec2
    heat_grads = data[:, 12:15]             # heat grad
    feature_tags = data[:, 15].astype(int)  # 0: regular, 1: high curvature
    boundary_tags = data[:, 16].astype(int) # 0: non-boundary, 1, 2, 3, ... boundary in different loops
    nm_tags = data[:, 17].astype(int)       # 0: manifold, 1, 2, 3, ... non-manifold in different loops
    app_tags = data[:, 18].astype(int)      # application tags place holder

    if down_ratio > 1:
        down_ratio = int(down_ratio)
        coords = coords[::down_ratio, :]
        normals = normals[::down_ratio, :]
        pd1s = pd1s[::down_ratio, :]
        pd2s = pd2s[::down_ratio, :]
        heat_grads = heat_grads[::down_ratio, :]
        feature_tags = feature_tags[::down_ratio]
        boundary_tags = boundary_tags[::down_ratio]
        nm_tags = nm_tags[::down_ratio]
        app_tags = app_tags[::down_ratio]
        print(f'Downsampled UDF pcd by ratio {down_ratio}, new num points: {coords.shape[0]}')

    return coords, normals, pd1s, pd2s, heat_grads, feature_tags, boundary_tags, nm_tags


def move_dict_to_device(data, device, dtype=None):
    """Move nested tensors; preserve unchanged containers and integer dtypes.

    Keeping unchanged lists avoids invalidating cached octree leaf tensors.
    dtype, when provided, applies only to floating-point tensors.
    """
    if isinstance(data, dict):
        moved = {key: move_dict_to_device(value, device, dtype) for key, value in data.items()}
        return data if all(moved[key] is value for key, value in data.items()) else moved
    elif isinstance(data, (list, tuple)):
        moved = [move_dict_to_device(item, device, dtype) for item in data]
        if all(new is old for new, old in zip(moved, data)):
            return data
        return tuple(moved) if isinstance(data, tuple) else moved
    elif torch.is_tensor(data):
        target_dtype = dtype if dtype is not None and data.is_floating_point() else data.dtype
        return data.to(device=device, dtype=target_dtype)
    else:
        return data

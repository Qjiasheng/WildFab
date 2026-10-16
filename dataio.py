import numpy as np
import torch
from torch.utils.data import Dataset
import diff_operators
import torch.nn.functional as F
from utils import move_dict_to_device, load_udf_pcd, field_querying
import simple_dists
import os



class HeightPCD(Dataset):
    """Unifotm height filed as scalar field initialization, training not uses sdf values but only normals"""
    def __init__(self, sample_number=1000, num_batches=10, ref_dir='z'):
        super().__init__()
        self.sample_number = sample_number
        self.num_batches = num_batches

        self.ref_dir = ref_dir  # 'x', 'y', 'z'

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        coords = torch.empty((self.sample_number, 3), device='cuda').uniform_(-1, 1)

        if self.ref_dir == 'z':
            # height field
            sdf = coords[:, 2].reshape(-1, 1)
            normals = torch.zeros_like(coords)
            normals[:, 2] = 1
        elif self.ref_dir == 'x':
            # x fields
            sdf = coords[:, 0].reshape(-1, 1)
            normals = torch.zeros_like(coords)
            normals[:, 0] = 1
        else:
            # y fields
            sdf = coords[:, 1].reshape(-1, 1)
            normals = torch.zeros_like(coords)
            normals[:, 1] = 1

        # model input and gt
        return {'coords': coords}, {'sdf': sdf, 'normals': normals}




def gen_bn_pca_bbx_set(surf_coords, boundary_ids, near_dist_th, far_dist_th):
    """
    for each boundary loop, generate more points in each PCA bbx, and sampling off-surf points
    """

    def single_bbx_sample(surf_coords, bn_coords, near_dist_th, far_dist_th, scale_factor=1.5):
        def sample_uniform_pca_bbx(num_candidates, center, axes, half_size, device):
            local = torch.rand((num_candidates, 3), device=device) * 2.0 - 1.0
            local = local * half_size.view(1, 3)
            candidate_coords = local @ axes.T + center.view(1, 3)
            return candidate_coords

        device = surf_coords.device
        center = bn_coords.mean(dim=0)
        centered = bn_coords - center.view(1, 3)

        cov = centered.T @ centered / max(bn_coords.shape[0] - 1, 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)

        order = torch.argsort(eigvals, descending=True)
        axes = eigvecs[:, order]

        if torch.linalg.det(axes) < 0:
            axes[:, -1] *= -1

        local_bn = centered @ axes
        local_min = local_bn.min(dim=0).values
        local_max = local_bn.max(dim=0).values

        local_center = (local_min + local_max) * 0.5
        half_size = (local_max - local_min) * 0.5 * scale_factor
        half_size = torch.clamp(half_size, min=1e-6)

        center = center + local_center @ axes.T

        num_candidates = bn_coords.shape[0] * 200
        candidate_coords = sample_uniform_pca_bbx(num_candidates, center, axes, half_size, device,)

        far_mask = simple_dists._C.far_dist_check(surf_coords, candidate_coords, far_dist_th,)
        near_far_mask = simple_dists._C.far_dist_check(surf_coords,candidate_coords, near_dist_th,)
        near_mask = (~far_mask) & near_far_mask

        valid_near_coords = candidate_coords[near_mask]
        valid_far_coords = candidate_coords[far_mask]

        return valid_near_coords.cpu(), valid_far_coords.cpu()

    surf_coords_cuda = torch.from_numpy(surf_coords).float().to("cuda")

    near_coords_list, far_coords_list = [], []
    unique_bn_ids = torch.unique(boundary_ids)

    for bn_id in unique_bn_ids:
        if bn_id == 0:
            continue

        bn_coords = surf_coords_cuda[boundary_ids == bn_id]
        near_coords, far_coords = single_bbx_sample(surf_coords_cuda, bn_coords, near_dist_th, far_dist_th,)
        near_coords_list.append(near_coords)
        far_coords_list.append(far_coords)

    return torch.cat(near_coords_list, dim=0), torch.cat(far_coords_list, dim=0)


def gen_near_pcd(surf_coords_cuda, near_dist_th, far_dist_th):
    """pre-generate large set of near-surface points, to avoid far-surface points too close to surface."""
    device = surf_coords_cuda.device
    candidate_coords = torch.empty((surf_coords_cuda.shape[0] * 3000, 3), device=device).uniform_(-1, 1)
    far_mask = simple_dists._C.far_dist_check(surf_coords_cuda, candidate_coords, far_dist_th)
    near_far_mask = simple_dists._C.far_dist_check(surf_coords_cuda, candidate_coords, near_dist_th)
    near_mask = (~far_mask) & near_far_mask  
    valid_near_coords = candidate_coords[near_mask]
    del candidate_coords, far_mask, near_far_mask, near_mask
    print(f'[I] Pre-generated {valid_near_coords.shape[0]} near-surface points for training.')
    return valid_near_coords.cpu()


def far_surf_gen_cuda(surf_coords, dist_th, far_batch_size):
    """
    generate far-surface points by random sampling and filtering based on distance to surface points.
    already on cuda, use customized cuda function for distance computation.
    """
    device = surf_coords.device
    far_coords = torch.empty((far_batch_size * 3, 3), device=device).uniform_(-1, 1) 
    # start = time.time()
    far_mask = simple_dists._C.far_dist_check(surf_coords, far_coords, dist_th)
    # print(f"far_surf_gen_cuda distance check time: {time.time() - start:.8f} seconds")
    valid_far_coords = far_coords[far_mask]

    return valid_far_coords[:far_batch_size]

def get_avg_dist(coords, k=10):
    """Mean distance to each point's k nearest neighbors, excluding itself."""
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(coords)
    distances, _ = nbrs.kneighbors()  # Query fitted points with self excluded.
    avg_edge_len = distances.mean()
    return avg_edge_len

    
class UDFDatasetDiffuse(Dataset):
    """dataset for UDF training. already in [-1, 1] cube.
    Use off surface points in near and far regions."""

    def __init__(self, pointcloud_path, more_smpl_times=0, near_surf_times=10, near_inv_delta=256, 
                 far_surf_times=10, far_inv_delta=128, use_adaptive_near_far=True, 
                 batch_size=2000, down_ratio=1, high_feature_ratio=0.7, 
                 sample_type=None):
        super().__init__()

        print(f'[I] Sample type: {sample_type}') 

        # load training pcd
        self.surf_coords, self.surf_normals, _, _, _, high_curv, boundary, nm = load_udf_pcd(pointcloud_path, down_ratio=down_ratio)
        print(f'bn points {(boundary!=0).sum()}, nm points {(nm!=0).sum()}, total surface points {self.surf_coords.shape[0]}')

        if use_adaptive_near_far:
            # off-surf sampling is important for udf training.
            # Sparse data with too near off-surf samples can damage surface level set
            # While insufficient near samples have less surface supervision, make zero-level-set thicker. 
            # when spatial ghost not well eliminated, check the sampling.
            # adaptively set near/far thresholds based on point cloud avg distance

            # NOTE tune them for impovement
            avg_dist = get_avg_dist(self.surf_coords, k=10)
            self.far_dist_th = avg_dist * 1.5
            self.near_dist_th = avg_dist * 0.5
            print(f'[I] avg edge inv: {int(1/avg_dist)}, near_inv_delta {int(2.0/self.near_dist_th)}, far_inv_delta {int(2.0/self.far_dist_th)}')
        else:
            # near_th < dist < far_th refs to near regions
            self.far_dist_th = 2.0 / far_inv_delta
            self.near_dist_th = 2.0 / near_inv_delta  

        # treat nm loops as extra bn for more sampling on nm and open-bn regions
        more_smpl_tags = boundary.copy()
        more_smpl_tags[nm!=0] = boundary.max() + nm[nm!=0]
        more_smpl_tensor = torch.from_numpy(more_smpl_tags).long()

        batch_size = min(batch_size, self.surf_coords.shape[0])  
        self.surf_batch_size = batch_size
        self.far_batch_size = int(batch_size * far_surf_times) 
        self.near_batch_size = int(batch_size * near_surf_times)
        
        # vis check
        from vis import vis_grad, vis_pcd_fields
        vis_grad(self.surf_coords, self.surf_normals, label='Surface Normals')
        vis_pcd_fields(self.surf_coords, high_curv, label='High Curvature')
        vis_pcd_fields(self.surf_coords, more_smpl_tags!=0, label='more sampling tags')

        if boundary.sum() > 0:
            vis_grad(self.surf_coords[boundary!=0], self.surf_normals[boundary!=0])
        if nm.sum() > 0:
            vis_grad(self.surf_coords[nm!=0], self.surf_normals[nm!=0])

        # prepare sampling
        self.boundary_ids = np.where(boundary!=0)[0]
        self.high_curv_ids = np.where(high_curv!=0)[0]
        self.nm_ids = np.where(nm!=0)[0]
        self.other_ids = np.where((boundary==0) & (high_curv==0) & (nm==0))[0]

        # coordinate all samples
        self.boundary_size = min(self.boundary_ids.shape[0], self.surf_batch_size // 3)
        self.nm_size = min(self.nm_ids.shape[0], self.surf_batch_size // 3)
        fix_size = self.boundary_size + self.nm_size
        if sample_type=='random':
            # random based on actual high_curv/ other ratio
            high_curv_ratio = self.high_curv_ids.shape[0] / (self.surf_coords.shape[0] - self.boundary_ids.shape[0])
            self.high_curv_size = min(self.high_curv_ids.shape[0], int((batch_size - fix_size) * high_curv_ratio))
        else:
            # fixed ratio for high_curv/ other
            self.high_curv_size = min(self.high_curv_ids.shape[0], int((batch_size - fix_size) * high_feature_ratio))
        self.other_size = batch_size - fix_size - self.high_curv_size

        # pre-generate large set near pcd regions
        self.surf_coords_cuda = torch.from_numpy(self.surf_coords).float().to('cuda')
        self.near_coords_set = gen_near_pcd(self.surf_coords_cuda, self.near_dist_th, self.far_dist_th)

        if self.near_coords_set.shape[0] < self.surf_coords.shape[0] * near_surf_times:
            raise ValueError(f'[E] Not enough near-surface points generated: {self.near_coords_set.shape[0]} < {self.near_batch_size}.')

        # if have, need for more bn near samples
        if more_smpl_times > 0 and more_smpl_tensor.sum() > 0:
            self.bn_far_batch_size = int(batch_size * more_smpl_times)
            self.bn_near_coords_set, self.bn_far_coords_set = \
            gen_bn_pca_bbx_set(self.surf_coords, more_smpl_tensor, self.near_dist_th, self.far_dist_th)

            print(f'[I] Ask for more bn off surf samples, times x{more_smpl_times}')
            vis_pcd_fields(self.bn_far_coords_set.cpu().numpy())
        else:
            self.bn_far_batch_size = 0
            self.bn_near_coords_set = torch.empty((0, 3), device=self.surf_coords_cuda.device)
            self.bn_far_coords_set = torch.empty((0, 3), device=self.surf_coords_cuda.device)
            print(f'[I] No extra bn off surf samples')
        
    
    def __len__(self):
        return self.surf_coords.shape[0] // self.surf_batch_size
    
    def __getitem__(self, idx):
        # surface points, all boundary, random sample from high-curv and other
        boundary_ids = np.random.choice(self.boundary_ids, size=min(self.boundary_size, self.boundary_ids.shape[0]), replace=False)
        nm_ids = np.random.choice(self.nm_ids, size=min(self.nm_size, self.nm_ids.shape[0]), replace=False)
        high_curv_ids = np.random.choice(self.high_curv_ids, size=min(self.high_curv_size, self.high_curv_ids.shape[0]), replace=False)
        other_ids = np.random.choice(self.other_ids, size=min(self.other_size, self.other_ids.shape[0]), replace=False)

        surf_ids = np.concatenate((boundary_ids, nm_ids, high_curv_ids, other_ids), axis=0)
        surf_tags = np.zeros((surf_ids.shape[0],), dtype=np.int32)         # 0 for surface points
        surf_tags[:self.boundary_size] = 1                                 # 1 for boundary points
        surf_tags[self.boundary_size:self.boundary_size+self.nm_size] = 3  # 3 for non-manifold points

        surf_coords = torch.from_numpy(self.surf_coords[surf_ids]).float().to('cuda')
        surf_normals = torch.from_numpy(self.surf_normals[surf_ids]).float().to('cuda')
        surf_tags = torch.from_numpy(surf_tags).long().to('cuda')

        # near/ far-off-surface points
        near_idx = torch.randperm(self.near_coords_set.shape[0])[:self.near_batch_size]
        near_coords = self.near_coords_set[near_idx].to('cuda')
        far_coords = far_surf_gen_cuda(self.surf_coords_cuda, self.far_dist_th, self.far_batch_size) 

        # same surf bacth, for bn near-far coords
        if self.bn_far_batch_size > 0 and self.bn_far_coords_set.shape[0] > 0:
            bn_far_coords = self.bn_far_coords_set[torch.randperm(self.bn_far_coords_set.shape[0])[:self.surf_batch_size]].to('cuda')
            off_surf_coords = torch.cat((near_coords, far_coords, bn_far_coords), dim=0) 
        else:
            off_surf_coords = torch.cat((near_coords, far_coords), dim=0)

        # tag=2 for near-far-off-surface points, normals not used in loss
        off_surf_normals = torch.ones((off_surf_coords.shape[0], 3), device=off_surf_coords.device) 
        off_surf_tags = torch.ones((off_surf_coords.shape[0],), device=off_surf_coords.device, dtype=torch.long) * 2  

        # concatenate all points
        coords = torch.cat((surf_coords, off_surf_coords), dim=0)
        normals = torch.cat((surf_normals, off_surf_normals), dim=0)
        tags = torch.cat((surf_tags, off_surf_tags), dim=0)

        # vis check
        # from vis import vis_pcd_fields, vis_grad
        # from toolpath_utils import vis_two_set_pcd
        # vis_two_set_pcd(surf_coords.cpu().numpy(), near_coords.cpu().numpy(), pcd2_size=20)
        # vis_pcd_fields(coords.cpu().numpy()[tags.cpu().numpy()==0], fields=tags.cpu().numpy()[tags.cpu().numpy()==0])
        # vis_grad(coords.cpu().numpy(), normals.cpu().numpy())

        return {'coords': coords}, {'normals':normals, 'tags': tags.squeeze()}        




def load_train_pcd(pcd_path, down_ratio=1, base_threshold=None):
    """load prepared pcd for training.
    coords already in [-1, 1] cube.
    all directions are unit vectors.
    NOTE heat_dirs could be all zeros if not set.
    """
    # load udf data, process to set base and heat dirs
    coords, normals, pd1s, pd2s, heat_dirs, feature_tags, boundary_tags, nm_tags = load_udf_pcd(pcd_path, down_ratio=down_ratio)
    min_dirs = pd2s
    max_dirs = pd1s 
    
    if base_threshold is not None:
        # 1 for base points, 0 for non-base points
        base_tags = np.where(coords[:, 2] < base_threshold, 1, 0)  
    else:
        # all non-base points
        base_tags = np.zeros((coords.shape[0],), dtype=np.int32) 

    return coords, normals, min_dirs, max_dirs, heat_dirs, base_tags, boundary_tags



    
class SliceSurfPCDHeat(Dataset):
    """
    train surafce vector field (grad of a scalar field). 
    using heat grad is optional to guide overhang regions
    training data pcd has already been prepared in `test_sdf.py`, in [-1, 1] cube.
    """
    def __init__(self, pointcloud_path, sample_number=2000, num_batches=10, down_ratio=1, base_threshold=None,
                 stage='init'):
        super().__init__()
    
        self.sample_number = sample_number
        self.num_batches = num_batches
        self.device = 'cuda'

        self.coords, self.normals, self.min_dirs, self.max_dirs, self.heat_dirs, self.base_tags, self.bn_tags = \
            load_train_pcd(pointcloud_path, down_ratio=down_ratio, base_threshold=base_threshold)
        
        # vis check
        from vis import vis_pcd_fields, vis_grad
        vis_pcd_fields(self.coords, self.base_tags, label='Base Tags')
        vis_pcd_fields(self.coords, self.bn_tags, label='Boundary Tags')
        vis_grad(self.coords, self.normals, label='Normals')
        vis_grad(self.coords, self.min_dirs, label='Min Curv Dirs')
        vis_grad(self.coords, self.heat_dirs, label='Heat Dirs')

        self.base_ids = np.where(self.base_tags == 1)[0]
        self.boundary_ids = np.where(self.bn_tags != 0)[0]
        self.other_ids = np.where((self.bn_tags == 0) & (self.base_tags == 0))[0]

        if stage == 'init':
            self.boundary_size = 0   # first stage not need bn constraints
        elif stage == 'final':
            self.boundary_size = min(self.boundary_ids.shape[0], self.sample_number // 5)
        else:
            raise ValueError(f"Invalid stage: {stage}. Expected 'init' or 'final'.")

        self.base_size = min(self.base_ids.shape[0], self.sample_number // 10)
        fix_size = self.boundary_size + self.base_size
        self.other_size = self.sample_number - fix_size       
    

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        # boundary points, base points, other points
        boundary_ids = np.random.choice(self.boundary_ids, size=self.boundary_size, replace=False)
        base_ids = np.random.choice(self.base_ids, size=self.base_size, replace=False)
        other_ids = np.random.choice(self.other_ids, size=self.other_size, replace=False)

        all_ids = np.concatenate((boundary_ids, base_ids, other_ids), axis=0)

        all_bn_tags = np.zeros((all_ids.shape[0],), dtype=np.int32)    # bn loops
        all_bn_tags[:self.boundary_size] = self.bn_tags[boundary_ids]  # keep original boundary tags 

        all_base_tags = np.zeros((all_ids.shape[0],), dtype=np.int32)  # base/ non-base
        all_base_tags[self.boundary_size: self.boundary_size + self.base_size] = 1  # 1 for base points, 0 for non-base points

        # collect data
        coords = torch.from_numpy(self.coords[all_ids]).float().to(self.device)
        normals = torch.from_numpy(self.normals[all_ids]).float().to(self.device)
        min_dirs = torch.from_numpy(self.min_dirs[all_ids]).float().to(self.device)
        heat_dirs = torch.from_numpy(self.heat_dirs[all_ids]).float().to(self.device)
        all_bn_tags = torch.from_numpy(all_bn_tags).long().to(self.device)
        all_base_tags = torch.from_numpy(all_base_tags).long().to(self.device)

        # vis check
        # from vis import vis_pcd_fields, vis_grad
        # vis_pcd_fields(coords.cpu().numpy(), all_bn_tags.cpu().numpy())
        # vis_pcd_fields(coords.cpu().numpy(), all_base_tags.cpu().numpy(), label='Base Tags')
        # vis_grad(coords.cpu().numpy(), normals.cpu().numpy(), label='normals')
        # vis_grad(coords.cpu().numpy(), min_dirs.cpu().numpy(), label='min_curv_dir')
        # vis_grad(coords.cpu().numpy(), heat_dirs.cpu().numpy(), label='heat_dir')

        # All on GPU
        return {'coords': coords}, {'normals': normals, 'min_dirs': min_dirs, 'heat_dirs': heat_dirs, 
                                    'bn_tags': all_bn_tags.squeeze(), 'base_tags': all_base_tags.squeeze()}


def get_infill_pcd(udf_decoder, infill_surf_path, base_infill_number, eps=0.005, device='cuda'):
    """generate infill pcd by sampling and filtering with udf decoder and wn field.
    already in [-1, 1] cube, clamped by model's UDF.
    """
    # import clouds_wn
    import igl
    # prepare a large set of infill pcd
    base_coords = np.random.uniform(-1, 1, size=(base_infill_number, 3)).astype(np.float32)

    base_udf = field_querying(udf_decoder, base_coords, no_print=True, return_on_cuda=False)
    base_udf = base_udf.detach().cpu().numpy()

    # wn field, .xyz -- dipolesum or .obj -- igl fast wn
    if infill_surf_path.endswith('.xyz'):
        # use dipolesum wn field, need to estimate areas 
        pass
    elif infill_surf_path.endswith('.obj'):
        # igl fast winding number
        verts, faces = igl.read_triangle_mesh(infill_surf_path)
        wn = igl.fast_winding_number(verts, faces, base_coords)
    else:
        raise ValueError(f"Unsupported infill_surf_path format: {infill_surf_path}. Expected .xyz or .obj")
    
    mask = (base_udf.squeeze() > eps) & (wn.squeeze() > 0.5)  
    if mask.sum() > 0:
        base_infill_coords = base_coords[mask]

    return base_infill_coords



class FinalSlicePCD(Dataset):
    """second stage to train final guidance field, pcd contains surface, inside, base pcd.
    interior is clamped by udf_eps=0.005 inside by wn field from clouds or soups.
    
    init_slice directions are projected onto tangents, prepare all data in init"""

    def __init__(self, udf_decoder, preslice_decoder, pointcloud_path, infill_surf_path, 
                 base_infill_number=5000000, sample_number=2000, infill_times=1.0, num_batches=10, down_ratio=1, 
                 base_threshold=None, large_th_pcd_index_path=None, stage='final'):
        super().__init__()

        self.sample_number = sample_number
        self.num_batches = num_batches
        self.device = 'cuda'

        # load surface/base pcd
        self.coords, self.normals, _, _, _, self.base_tags, self.bn_tags = \
            load_train_pcd(pointcloud_path, down_ratio=down_ratio, base_threshold=base_threshold)
        
        # slice gradient on tangents 
        samples = torch.from_numpy(self.coords).float().to(self.device)
        sample_normals = torch.from_numpy(self.normals).float().to(self.device)
        samples.requires_grad = True
        slice_model_output = preslice_decoder(samples)
        model_coords = slice_model_output['model_in']
        model_slice_value = slice_model_output['model_out']
        slice_grads = diff_operators.gradient(model_slice_value, model_coords).detach()
        slice_grads_prj = slice_grads - (slice_grads * sample_normals).sum(dim=1, keepdim=True) * sample_normals 
        slice_dirs = F.normalize(slice_grads_prj, dim=1, p=2)
        self.slice_dirs = slice_dirs.cpu().numpy()
        # base and bn no need to align slice dirs, set to all ones
        self.slice_dirs[(self.base_tags==1) | (self.bn_tags!=0)] = np.array([0, 0, 1], dtype=np.float32)    

        # vis check
        from vis import vis_pcd_fields, vis_grad, visvis_two_set_pcd
        vis_pcd_fields(self.coords, self.base_tags, label='Base Tags')
        vis_pcd_fields(self.coords, self.bn_tags, label='Boundary Tags')
        vis_grad(self.coords, self.normals, label='Normals')
        vis_grad(self.coords, self.slice_dirs, label='slice_dirs')

        # for data sampling 
        self.base_ids = np.where(self.base_tags == 1)[0]
        self.boundary_ids = np.where(self.bn_tags != 0)[0]
        self.other_ids = np.where((self.bn_tags == 0) & (self.base_tags == 0))[0]

        # append large th pcd in training
        if large_th_pcd_index_path is not None and os.path.exists(large_th_pcd_index_path):  
            self.large_th_ids = np.load(large_th_pcd_index_path).astype(np.int32)
            # not repeated with base and boundary points
            self.large_th_ids = np.setdiff1d(self.large_th_ids, np.concatenate((self.base_ids, self.boundary_ids), axis=0))
            print(f"Loaded {self.large_th_ids.shape[0]} large_th_ids")
            visvis_two_set_pcd(self.coords, self.coords[self.large_th_ids])
        else:
            self.large_th_ids = np.array([], dtype=np.int32)

        # infill pcd generation with udf decoder and wn 
        if infill_surf_path is None:
             print("No infill_surf_path provided, skipping infill pcd generation.")
             self.infill_coords = np.empty((0, 3), dtype=np.float32)
             self.infill_ids = np.empty((0,), dtype=np.int32)
        else:
            self.infill_coords = get_infill_pcd(udf_decoder, infill_surf_path, base_infill_number, eps=0.005, device=self.device)
            if self.infill_coords.shape[0] < self.sample_number * 2:
                print(f"Warning: infill coords only {self.infill_coords.shape[0]}, less than expected {self.sample_number * 2}")
            self.infill_ids = np.arange(self.infill_coords.shape[0], dtype=np.int32)

            vis_pcd_fields(self.infill_coords, np.ones((self.infill_coords.shape[0],), dtype=np.int32), label='Infill Points')

        if stage == 'init':
            self.boundary_size = 0   # first stage not need bn constraints
            self.large_th_size = 0
        elif stage == 'final':
            self.boundary_size = min(self.boundary_ids.shape[0], self.sample_number // 5)
            self.large_th_size = min(self.large_th_ids.shape[0], self.sample_number // 3)
        else:
            raise ValueError(f"Invalid stage: {stage}. Expected 'init' or 'final'.")

        self.base_size = min(self.base_ids.shape[0], self.sample_number // 8)
        fix_size = self.boundary_size + self.base_size + self.large_th_size
        self.other_size = self.sample_number - fix_size
        self.infill_size = min(self.infill_coords.shape[0], int(self.sample_number * infill_times))  # model varying

    
    def __len__(self):
        return self.num_batches 

    def __getitem__(self, idx):
        """ sample from on-surface points, also set a base field"""
        boundary_ids = np.random.choice(self.boundary_ids, size=self.boundary_size, replace=False)
        base_ids = np.random.choice(self.base_ids, size=self.base_size, replace=False) 
        largeth_ids = np.random.choice(self.large_th_ids, size=self.large_th_size, replace=False)
        other_ids = np.random.choice(self.other_ids, size=self.other_size, replace=False)
        infill_ids = np.random.choice(self.infill_ids, size=self.infill_size, replace=False)

        all_surf_ids = np.concatenate((boundary_ids, base_ids, largeth_ids, other_ids), axis=0)
        all_bn_tags = np.zeros((all_surf_ids.shape[0],), dtype=np.int32)    
        all_bn_tags[:self.boundary_size] = self.bn_tags[boundary_ids]      # bn loops, 0 for non-bn, keep original bn tags for diff loops

        all_base_tags = np.zeros((all_surf_ids.shape[0],), dtype=np.int32)          # base/ non-base
        all_base_tags[self.boundary_size: self.boundary_size + self.base_size] = 1  # 1 for base points, 0 for non-base points

        surf_coords = torch.from_numpy(self.coords[all_surf_ids]).float().to(self.device)
        surf_normals = F.normalize(torch.from_numpy(self.normals[all_surf_ids]).float().to(self.device), dim=-1, p=2)
        surf_dirs = F.normalize(torch.from_numpy(self.slice_dirs[all_surf_ids]).float().to(self.device), dim=-1, p=2)
        surf_bn_tags = torch.from_numpy(all_bn_tags).long().to(self.device)
        surf_base_tags = torch.from_numpy(all_base_tags).long().to(self.device)

        infill_coords = torch.from_numpy(self.infill_coords[infill_ids]).float().to(self.device)
        infill_normals = F.normalize(torch.ones_like(infill_coords), dim=-1, p=2)  
        infill_dirs = F.normalize(torch.ones_like(infill_coords), dim=-1, p=2)  
        infill_bn_tags = torch.zeros(infill_coords.shape[0], dtype=torch.long).to(self.device)       # infill no bn, set 0
        infill_base_tags = torch.ones(infill_coords.shape[0], dtype=torch.long).to(self.device) * 2  # NOTE infill 2 in base tags

        total_coords = torch.cat((surf_coords, infill_coords), dim=0)
        total_normals = torch.cat((surf_normals, infill_normals), dim=0)
        total_dirs = torch.cat((surf_dirs, infill_dirs), dim=0)
        total_bn_tags = torch.cat((surf_bn_tags, infill_bn_tags), dim=0)
        total_base_tags = torch.cat((surf_base_tags, infill_base_tags), dim=0)
        
        # vis check
        # from vis import vis_pcd_fields, vis_grad
        # vis_pcd_fields(total_coords.cpu().numpy(), total_bn_tags.cpu().numpy(), label='Boundary Tags')
        # vis_pcd_fields(total_coords.cpu().numpy(), total_base_tags.cpu().numpy(), label='Base Tags')
        # vis_grad(surf_coords.cpu().numpy(), surf_dirs.cpu().numpy(), label='surf slice_dirs')
        # vis_grad(surf_coords.cpu().numpy(), surf_normals.cpu().numpy(), label='surf normals')

        # bn_tags: 0 for non-bn(both surf/inifll), >0 for different bn loops; 
        # base_tags: 0 for non-base, 1 for base, 2 for infill
        return {'coords': total_coords}, {'normals': total_normals.squeeze().detach(),'dirs':total_dirs.squeeze().detach(), 
                                          'bn_tags': total_bn_tags.squeeze(), 'base_tags': total_base_tags.squeeze()}
    


class CollisionDataset(Dataset):
    """
    collision dataset for quaternion field training.
    quaternion as a rotation to transform base frame at each waypoint.
    put nozzle data out of dataloader

    shell_infill_ratio is shell / infill; None uses natural proportions.
    Each batch is sampled independently using original waypoint indices.
    """
    def __init__(self, waypoint_path, object_scale, batch_size=10000, shell_infill_ratio=2.0):
        super().__init__()

        self.batch_size = batch_size
        self.device = 'cuda'

        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        if shell_infill_ratio is not None and (not np.isfinite(shell_infill_ratio) or shell_infill_ratio < 0):
            raise ValueError('shell_infill_ratio must be nonnegative or None')

        self.waydata = WayPts(waypoint_path=waypoint_path)
        self.num_waypts = self.waydata.waypt_len

        # 2.0 mm below as collision of base platform
        self.base_coll_th = self.waydata.min_z  - 2.0 / object_scale

        # grab built shell and infill outer octrees
        self.shell_time_octree_wn_dict = move_dict_to_device(self.waydata.shell_time_octree_wn_dict, device=self.device)
        self.outer_shells_time_octree_wn_dict_tagdict = \
            move_dict_to_device(self.waydata.outer_shells_time_octree_wn_dict, device=self.device)

        # shell/ infill sampling ratio
        self.shell_infill_ratio = shell_infill_ratio
        tags = self.waydata.shell_infill_tags
        self.shell_ids = np.flatnonzero(tags != 1)
        self.infill_ids = np.flatnonzero(tags == 1)
        self.waypt_ids = np.arange(self.num_waypts, dtype=np.int64)
        self.infill_size = None
        self.shell_size = None
        if shell_infill_ratio is not None and len(self.shell_ids) and len(self.infill_ids):
            self.infill_size = round(batch_size / (1 + shell_infill_ratio))
            self.shell_size = batch_size - self.infill_size
        
    
    def __len__(self):
        # Epoch sampling budget; batches may reuse waypoints across calls.
        return max(1, (self.num_waypts + self.batch_size - 1) // self.batch_size)

    def _sample_batch_indices(self):
        if self.infill_size is None:
            # Natural sampling, including shell-only data where no ratio applies.
            return np.random.choice(self.waypt_ids, size=self.batch_size,
                                    replace=self.num_waypts < self.batch_size)

        shell_ids = np.random.choice(self.shell_ids, size=self.shell_size,
                                     replace=len(self.shell_ids) < self.shell_size)
        infill_ids = np.random.choice(self.infill_ids, size=self.infill_size,
                                      replace=len(self.infill_ids) < self.infill_size)
        batch_indices = np.concatenate((shell_ids, infill_ids))
        np.random.shuffle(batch_indices)
        return batch_indices
    

    def __getitem__(self, idx):

        # ------ waypts data indices for this batch
        batch_indices = self._sample_batch_indices()
        
        # grab batch waypt data
        coords, normals, print_dirs, shell_infill_tags, shell_ts, outer_shells_ts_tagdict = \
            self.waydata.get_bacth_waydata(batch_indices)
        
        # move to cuda
        waypt_payload = {'coords': coords, 'normals': normals, 'print_dirs': print_dirs, 'shell_infill_tags': shell_infill_tags,
                         'shell_ts': shell_ts, 'outer_shells_ts_tagdict': outer_shells_ts_tagdict}
        waypt_payload = move_dict_to_device(waypt_payload, device=self.device)
        
        # vis check of waypt batch
        # from vis import vis_grad, vis_pcd_fields
        # vis_pcd_fields(coords.cpu().numpy(), shell_infill_tags.cpu().numpy(), label='Waypt Shell/Infill Tags')
        # vis_grad(coords.cpu().numpy(), normals.cpu().numpy(), label='Waypt Normals')
        # vis_grad(coords.cpu().numpy(), print_dirs.cpu().numpy(), label='Waypt Dirs')

        return {'coords': coords}, {"waypt_payload": waypt_payload,
                                    "st_octree_wn_dict": self.shell_time_octree_wn_dict,
                                    "ost_octree_wn_dict_tagdict": self.outer_shells_time_octree_wn_dict_tagdict,
                                    "base_coll_th": self.base_coll_th}
        



from level_collision import build_init_time_octree_with_pcd
from wn_utils import estimate_areas

class WayPts(object):
    """
    to load planned waypoints and preprocess for collision dataset.
    waypts data format: [coords, normals, print_dirs, thickness, level_tags, shell_infill_tags, marching_dirs]
    
    normals: shell normals use surface normals, infill normals set [111] placeholder.
    print_dirs: shell use project slice grad onto tangent, infill directly use slice grad.
    level_tags: follow INF-3DP different printing part levels.
    shell_infill_tags: 0 for shell, 1 for infill, 2, 3, ... for different infill outer shells.
    marching_dirs: for each waypoint, the direction of the next waypoint, optional set for motion constraints.

    --------------------------
    Build octree for (0) pure shell (1) infill (2, 3, ...) different infill outer shell
    octree building needs clean only surface pts. first duplicate removal, then infill removal.
    for time-varying GWN, needing to each pre-remove wrt. clean timesteps in each octree, during batch sampling.
    --------------------------
    data in dict for convenient access
    """

    def __init__(self, waypoint_path):

        self.coords, self.normals, self.print_dirs, _, _, self.shell_infill_tags, _ = \
            self.load_waypts(waypoint_path)
        
        self.waypt_len = self.coords.shape[0]
        self.min_z = self.coords[:, 2].min()   # for base collision 

        # build shell octree with pre-remove wrt. original timesteps
        self.shell_time_octree_wn_dict, self.shell_pre_remove = self.build_shell_octree()

        # build infill outer shells octree with pre-remove wrt. original timesteps
        self.outer_shells_time_octree_wn_dict_tuple, self.outer_shells_time_octree_wn_dict \
            = self.build_infill_outer_shells_octree()


    
    def get_bacth_waydata(self, batch_indices):
        """
        get batch waypt data, including coords, normals, print_dirs
        batch_indices: np.ndarray of shape (batch_size,)
        as original timestep, pre_remove based on shell_clean and outer_shells_clean, for time-varying GWN.
        Returns CPU float32 waypoint tensors and CPU long clean timesteps.
        A timestep of -1 means no point in that shell has been printed yet.
        """
        batch_indices = np.asarray(batch_indices)
        batch_indices = np.ascontiguousarray(batch_indices, dtype=np.int64)

        batch_coords = torch.from_numpy(self.coords[batch_indices]).float()
        batch_normals = torch.from_numpy(self.normals[batch_indices]).float()
        batch_print_dirs = torch.from_numpy(self.print_dirs[batch_indices]).float()
        batch_shell_infill_tags = torch.from_numpy(self.shell_infill_tags[batch_indices]).long()

        # move to clean timestep for selected batched data
        original_ts = torch.from_numpy(batch_indices)
        shell_ts = original_ts - self.shell_pre_remove[original_ts]

        if len(self.outer_shells_time_octree_wn_dict_tuple) == 0:
            outer_shells_ts_dict = {}
        else:
            outer_shells_ts_dict = {
                shell_tag: original_ts - pre_remove[original_ts]
                for shell_tag, (_, pre_remove) in self.outer_shells_time_octree_wn_dict_tuple.items()
            }

        return batch_coords, batch_normals, batch_print_dirs, batch_shell_infill_tags, shell_ts, outer_shells_ts_dict


    def build_shell_octree(self):
        """
        use clean shell waypts payload to build time octree.
        pre_remove is to shift original timestep to current clean timestep, for time-varying GWN.
        """

        shell_clean_waypts, shell_pre_remove = self.build_shell_clean(remove_duplicates=True)
        shell_time_octree_wn_dict = build_init_time_octree_with_pcd(shell_clean_waypts)

        return shell_time_octree_wn_dict, shell_pre_remove

    def build_infill_outer_shells_octree(self):
        """
        use clean infill outer shells waypts payload to build time octree.
        pre_remove is to shift original timestep to current clean timestep, for time-varying GWN.
        each shell tag >= 2 has its own octree and pre_remove.
        Return two empty dictionaries when no infill outer shells exist.
        """
        outer_shells_dict = self.build_infill_outer_shells(remove_duplicates=True)
        if len(outer_shells_dict) == 0:
            return {}, {}

        outer_shells_time_octree_wn_dict_tuple = {}
        outer_shells_time_octree_wn_dict = {}

        for shell_tag, (clean_waypts, pre_remove) in outer_shells_dict.items():
            octree = build_init_time_octree_with_pcd(clean_waypts)
            outer_shells_time_octree_wn_dict_tuple[shell_tag] = \
                (octree, pre_remove)
            
            outer_shells_time_octree_wn_dict[shell_tag] = octree
        return outer_shells_time_octree_wn_dict_tuple, outer_shells_time_octree_wn_dict


    def build_shell_clean(self, remove_duplicates=True):
        """Return ordered shell arrays and an inclusive removal count (CPU torch.long).

        Keep tags != 1, including infill outer shells. Exact duplicate coordinates
        keep their first shell occurrence; infill points cannot suppress shell points.
        For original timestep t, clean_ts = t - pre_remove[t]; 
        -1 means no shell yet for no wn contribution in cuda
        """
        

        coords = np.asarray(self.coords)
        normals = np.asarray(self.normals)
        tags = np.asarray(self.shell_infill_tags)

        retained = np.flatnonzero(tags != 1)
        if remove_duplicates and len(retained):
            _, first = np.unique(coords[retained], axis=0, return_index=True)
            retained = retained[np.sort(first)]  # Preserve printing order, not coordinate order.

        keep = np.zeros(len(coords), dtype=bool)
        keep[retained] = True
        pre_remove = torch.from_numpy((~keep).astype(np.int64)).cumsum(dim=0)
        clean_coords = np.ascontiguousarray(coords[retained], dtype=np.float32)
        clean_normals = np.ascontiguousarray(normals[retained], dtype=np.float32)
        
        areas = np.asarray(estimate_areas(clean_coords, n_neighbors=min(20, len(retained) - 1)), dtype=np.float32).reshape(-1)
        
        clean_waypts = {'coords': clean_coords, 'normals': clean_normals, 'areas': np.ascontiguousarray(areas)}
        print(f"Shell points after cleaning: {len(retained)} / {np.count_nonzero(tags != 1)}")
        return clean_waypts, pre_remove

    def build_infill_outer_shells(self, remove_duplicates=True):
        """Return {shell_tag: (clean_waypts, pre_remove)} for tags >= 2.

        Each shell preserves printing order and removes duplicates independently.
        pre_remove counts removed points through each original timestep t;
        t - pre_remove[t] is the last printed clean index (-1 before this shell).
        """
        coords = np.asarray(self.coords)
        normals = np.asarray(self.normals)
        tags = np.asarray(self.shell_infill_tags)
        outer_shells = {}

        outer_shell_tags = np.unique(tags[tags >= 2])
        if outer_shell_tags.size == 0:
            print("No infill outer shells found in the waypoint data.")
            return {}

        for shell_tag in outer_shell_tags:
            retained = np.flatnonzero(tags == shell_tag)
            if remove_duplicates:
                _, first = np.unique(coords[retained], axis=0, return_index=True)
                retained = retained[np.sort(first)]  # Preserve printing order within this shell.

            keep = np.zeros(len(coords), dtype=bool)
            keep[retained] = True
            pre_remove = torch.from_numpy((~keep).astype(np.int64)).cumsum(dim=0)
            clean_coords = np.ascontiguousarray(coords[retained], dtype=np.float32)
            clean_normals = np.ascontiguousarray(normals[retained], dtype=np.float32)

            # Too few points cannot define a 2D Voronoi area.
            if len(retained) < 4:
                areas = np.zeros(len(retained), dtype=np.float32)
            else:
                areas = np.asarray(estimate_areas(clean_coords, n_neighbors=min(20, len(retained) - 1)), dtype=np.float32).reshape(-1)

            clean_waypts = {'coords': clean_coords, 'normals': clean_normals, 'areas': np.ascontiguousarray(areas)}
            outer_shells[int(shell_tag)] = (clean_waypts, pre_remove)
            print(f"Infill outer shell tag {shell_tag} points after cleaning: {len(retained)} / {np.count_nonzero(tags == shell_tag)}")

        return outer_shells
    

    @staticmethod
    def load_waypts(fn):
        """load waypts data"""
        data = np.genfromtxt(fn)
        coords = data[:, :3].astype(np.float32)
        normals = data[:, 3:6].astype(np.float32)
        print_dirs = data[:, 6:9].astype(np.float32)
        thickness = data[:, 9].astype(np.float32)
        level_tags = data[:, 10].astype(np.int32)
        shell_infill_tags = data[:, 11].astype(np.int32)
        march_dirs = data[:, 12:15].astype(np.float32)

        for tag in np.unique(shell_infill_tags):
            count = np.sum(shell_infill_tags == tag)
            print(f"Shell/ Infill tag {tag}: {count} points")

        return coords, normals, print_dirs, thickness, level_tags, shell_infill_tags, march_dirs
        
        


        


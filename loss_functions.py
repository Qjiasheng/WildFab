import torch
import torch.nn.functional as F
import diff_operators
from utils import transform_nozzle_pcd_with_quaternion, quaternion_para_axes, transform_nozzle_grads_with_quaternion
from level_collision import collision_evaluation_with_infill


def get_linear_weight(epoch, start_epoch, end_epoch, start_weight, end_weight):
    """linear weight tuning along training epochs"""
    return start_weight + (end_weight - start_weight) * (epoch - start_epoch) / (end_epoch - start_epoch)

def height(model_output, gt, epoch, epochs, params):

    gt_sdf = gt['sdf']  # not use gt sdf values
    gt_normals = gt['normals']

    coords = model_output['model_in']
    pred_sdf = model_output['model_out']

    # field gradient align directions
    gradient = diff_operators.gradient(pred_sdf, coords)
    normal_constraint = 1 - F.cosine_similarity(gradient, gt_normals, dim=-1)[..., None]
    grad_constraint = torch.abs(gradient.norm(dim=-1) - 1)

    # add harmonic energy
    laplace = diff_operators.laplace(pred_sdf, coords)

    # values not outside [-1, 1]
    delta_lower, delta_upper = torch.clamp(-1 - pred_sdf, min=0), torch.clamp(pred_sdf - 1, min=0)
    sdf_pen = delta_lower + delta_upper

    return {'sdf_pen': sdf_pen.mean() * params.get('sdf_pen'), 
            'normal': normal_constraint.mean() * params.get('normal'), 
            'grad': grad_constraint.mean() * params.get('grad'),  
            'laplace': torch.abs(laplace).mean() * params.get('laplace')} 



def udf_scalar_diffuse_dirichlet(model_output, gt, epoch, epochs, params):
    """
    udf training with scalar heat diffusion, smoothness use scalar dirichlet energy
    """
    
    # 0 -- non-boundary surface, 1 -- boundary surface, 2 -- off-surface, 3 -- non-manifold surface
    normals = gt['normals']
    tags = gt['tags'] 

    coords = model_output['model_in']
    pred_udf = model_output['model_out']

    pred_udf = pred_udf.squeeze()

    gradient = diff_operators.gradient(pred_udf, coords)
    grad_norm = gradient.norm(dim=-1)

    heat_field = torch.exp(-pred_udf)

    # surface
    surf_udf = torch.where(tags!=2, pred_udf, torch.zeros_like(pred_udf))

    # eikonal 
    eikonal = torch.where(tags==2, torch.abs(grad_norm - 1.0), torch.zeros_like(grad_norm))

    # ----------------------- scalar difffusion
    start_epoch = 0
    end_epoch = int(epochs * params.get('diffuse_epoch_portion', 0.1) )

    # scalar dirichlet for smoothness 
    if epoch >= start_epoch and epoch < end_epoch:
        smooth_energy = diff_operators.scalar_dirichlet(heat_field, coords)   
        smooth_weight = get_linear_weight(epoch, start_epoch=start_epoch, end_epoch=end_epoch, 
                                           start_weight=params.get('smooth_init'), 
                                           end_weight=params.get('smooth_end'))
        smooth_loss = smooth_energy.mean() * smooth_weight
    else:
        smooth_loss = torch.tensor(0.0)
        
    
    # eikonal weight. optional weight schedule
    if epoch >= start_epoch and epoch < end_epoch:
        eikonal_weight = get_linear_weight(epoch, start_epoch=start_epoch, end_epoch=end_epoch, 
                                           start_weight=params.get('eikonal_init'), 
                                           end_weight=params.get('eikonal_end'))
    elif epoch >= end_epoch:
        eikonal_weight = params.get('eikonal_end')
    else:
        eikonal_weight = params.get('eikonal_init')


    # open boundary Neumann constraint, soft orthogonal constraint on boundary normals
    if epoch > end_epoch:
        if params.get('bound_normal_ortho') is not None and params.get('bound_normal_ortho') > 0.0 and torch.sum(tags==1) > 0:
            bound_normal_ortho = torch.relu(torch.abs(F.cosine_similarity(gradient[tags==1], normals[tags==1], dim=-1)) - 0.7)
            bound_loss = bound_normal_ortho.mean() * params.get('bound_normal_ortho')
        else:
            bound_loss = torch.tensor(0.0)
    else:
        bound_loss = torch.tensor(0.0)

    
    # surface udf value. after diffusion stage, set very large zero surface
    surf_udf_weight = params.get('surf_udf') 
    if epoch > end_epoch:
        surf_udf_weight = params.get('surf_udf') * 10.0

    return {'surf_udf': surf_udf.mean() * surf_udf_weight,
            'eikonal': eikonal.mean() * eikonal_weight,
            'bound_normal_ortho': bound_loss,
            'smoothness': smooth_loss}



def init_slice_field(model_output, gt, epoch, epochs, params):
    """
    first stage for guidance field training. vec field on surface
    """

    use_heat_grad = params.get('use_heat_grad', False)
    normals = gt['normals']
    min_dirs = gt['min_dirs']
    heat_dirs = gt['heat_dirs']

    # NOTE first stage not constrain boundary values.
    bn_tags = gt['bn_tags']  # 0- non-boundary, 1, 2, 3.. different bn loops
    base_tags = gt['base_tags']  # 1-base, 0-surface

    coords = model_output['model_in']
    slice_fields = model_output['model_out']

    # vector laplace (care about directions not norms)
    slice_grad = diff_operators.gradient(slice_fields, coords)
    slice_vec_laplace, _ = diff_operators.surface_vector_laplace(slice_grad, coords, normals=normals, norm=True)
    laplace_constraint = torch.abs(slice_vec_laplace.norm(dim=-1)) ** 2

    # manifold grad norm 1
    slice_grad_prj = slice_grad - torch.sum(slice_grad * normals, dim=-1, keepdim=True) * normals
    grad_constraint = torch.abs(slice_grad_prj.norm(dim=-1)-1) ** 2

    # base pcd gradient up to z
    if (base_tags == 1).sum() > 0:
        base_grad = slice_grad[base_tags == 1]
        base_grad = F.normalize(base_grad, dim=-1)
        base_grad_constraint = torch.abs(base_grad[..., 2] - 1)
    else:
        base_grad_constraint = torch.tensor(0.0).cuda()

    # set guidance direction (min curv dirs) aligned
    slice_prj_dir = F.normalize(slice_grad_prj, dim=-1)
    align_loss = (torch.norm(torch.cross(slice_prj_dir, min_dirs, dim=-1), dim=-1)) **2 

    # align with heat dir (optional)
    if use_heat_grad:
        heat_align_loss_temp = 1 - F.cosine_similarity(slice_prj_dir, heat_dirs, dim=-1)
        heat_align_loss = torch.where(base_tags==0, heat_align_loss_temp, torch.zeros_like(heat_align_loss_temp))
    else:
        heat_align_loss = torch.tensor(0.0).cuda()

    # lie on tangent plane
    print_dir = F.normalize(slice_grad, dim=-1)
    tangent_loss = torch.abs(torch.sum(print_dir * normals, dim=-1)) ** 2

    return {'grad': grad_constraint.mean() * params.get('grad'),  
            'base_grad': base_grad_constraint.mean() * params.get('base_grad'), 
            'laplace': laplace_constraint.mean() * params.get('laplace'),
            'curv_align_loss': align_loss.mean() * params.get('curv_align'), 
            'heat_align_loss': heat_align_loss.mean() * params.get('heat_align'),
            'tangent': tangent_loss.mean() * params.get('tangent')}



def final_slice_field(model_output, gt, epoch, epochs, paras):
    """
    second stage for final guidance feild training, for surface, base, inside points.
    vec laplace lift to 3D space, interior is singularity-free.
    NOTE not necessary to set Support-free loss on surface points, since further collision-stage will set it.
    """
    gt_normals = gt['normals']
    gt_dirs = gt['dirs']            # surf dir already projected to tangent plane
    gt_base_tags = gt['base_tags']  # 0-surface, 1-base, 2-inside
    gt_bn_tags = gt['bn_tags']      # 0-non-boundary, 1, 2, 3.. different bn loops
    
    coords = model_output['model_in']
    slice_fields = model_output['model_out']

    slice_grad = diff_operators.gradient(slice_fields, coords)
    slice_veclap = diff_operators.vector_laplace_new(slice_fields, coords, norm=True).squeeze(-1) 
    
    # vec lap in 3d space, not surface manifold. For non-bn pcd
    veclap_constraint = torch.abs(slice_veclap.norm(dim=-1)) ** 2
    veclap_loss = torch.where(gt_bn_tags == 0, veclap_constraint, torch.zeros_like(veclap_constraint))
 
    # Surface grad_len close to 1. MUST on surface -- project to tangent plane
    slice_grad_prj = slice_grad - torch.sum(slice_grad * gt_normals, dim=-1, keepdim=True) * gt_normals
    surf_grad_len = torch.abs(slice_grad_prj.norm(dim=-1)-1) ** 2
    surf_grad_len_loss = torch.where(gt_base_tags!= 2, surf_grad_len, torch.zeros_like(surf_grad_len))

    # inside grad_len close to 1. no projection
    inside_grad_len = torch.abs(slice_grad.norm(dim=-1)-1) ** 2
    inside_grad_len_loss = torch.where(gt_base_tags == 2, inside_grad_len, torch.zeros_like(inside_grad_len))

    # surface dirs align with gt dirs
    slice_grad_dir = F.normalize(slice_grad, dim=-1)
    grad_dir_loss = (1 - F.cosine_similarity(slice_grad_dir, gt_dirs, dim=-1)) ** 2
    surf_dir_loss = torch.where(gt_base_tags != 2, grad_dir_loss, torch.zeros_like(grad_dir_loss))

    # whatever base fields not be curved
    if (gt_base_tags == 1).sum() > 0:
        base_grad = slice_grad[gt_base_tags == 1]
        base_grad_dir = F.normalize(base_grad, dim=-1)
        base_dir_loss = torch.abs(base_grad_dir[..., 2] - 1.0)
    else:
        base_dir_loss = torch.tensor(0.0).cuda()

    # inside hessian F norm zero, regularize interior field
    hessian_weight = paras.get('hessian', 0.0)
    if hessian_weight > 0.0:
        slice_hessian = diff_operators.hessian(slice_fields, coords)[0].squeeze(2)
        hessian = slice_hessian[gt_base_tags == 2]
        hessian_loss = torch.sum(hessian ** 2, dim=(1, 2))
    else:
        hessian_loss = torch.tensor(0.0).cuda()

    # each boundary loop has same slice value
    if paras.get('bn_loop') is not None and paras.get('bn_loop') > 0.0 and torch.sum(gt_bn_tags > 0) > 0:
        unique_bn_tags = torch.unique(gt_bn_tags[gt_bn_tags > 0])
        bn_loss = 0.0
        for tag in unique_bn_tags:
            mask = (gt_bn_tags == tag)
            if mask.sum() > 0:
                bn_slice_value = slice_fields[mask]
                bn_loss = bn_loss + torch.mean((bn_slice_value - bn_slice_value.T) ** 2)
        bn_loss = bn_loss.mean() * paras.get('bn_loop')
    else:
        bn_loss = torch.tensor(0.0).cuda()


    return {'surf_grad_len': surf_grad_len_loss.mean() * paras.get('surf_grad_len'), 
            'inside_grad_len': inside_grad_len_loss.mean() * paras.get('inside_grad_len'),
            'surf_dir': surf_dir_loss.mean() * paras.get('surf_dir'), 
            'base_dir': base_dir_loss.mean() * paras.get('base_dir'), 
            'veclap': veclap_loss.mean() * paras.get('veclap'),
            'hessian': hessian_loss.mean() * paras.get('hessian'),
            'bn_loop': bn_loss}  




def cal_sf_loss_(normal, print_dir, alpha_degree=45, expansion=3e1):
    """inner/ outer surface are sf. 
    (45, 135) degree is the range of sf, with 90 degree as the center"""
    device = normal.device
    dtype = normal.dtype

    alpha = torch.as_tensor(alpha_degree, device=device, dtype=dtype)
    cos_alpha = torch.cos(torch.deg2rad(alpha))

    cos_theta = torch.sum(normal * print_dir, dim=-1).clamp(-1.0, 1.0)
    abs_cos_theta = cos_theta.abs()

    sf_loss_term = torch.sigmoid(expansion * (abs_cos_theta - cos_alpha))
    sf_mask = abs_cos_theta <= cos_alpha
    sf_ratio = sf_mask.float().mean().item() * 100.0

    return sf_loss_term, sf_ratio


def collision_loss(model_output, gt, epoch, epochs, paras):
    """
    collision loss for quaternion field optimization.
    infill normals are invalid and not used, print_dirs directly use guidance gradients
    
    set base_coll_th. under this response up-z
    """

    # collect decoders and other paras
    udf_decoder = paras.get('udf_decoder')
    udf_init_margin = paras.get('udf_init_margin')
    base_threshold = paras.get('base_threshold') 

    # collect nozzle data. already cuda 
    nozzle_dict = paras['nozzle_dict']
    nozzle_pcd = nozzle_dict['nozzle_coords']
    nozzle_grads = nozzle_dict['nozzle_grads']
    local_map = nozzle_dict['local_map']

    # collect gt data
    # waypt_si_tags: 0-pure shell, 1-infill, 2,3,4... different infill outer shells
    payload = gt['waypt_payload']
    waypt_normals = F.normalize(payload['normals'].reshape(-1, 3), dim=-1)
    waypt_print_dirs = payload['print_dirs'].reshape(-1, 3)
    waypt_si_tags = payload['shell_infill_tags'].reshape(-1)

    # clean timestep wrt built octree for tv-GWN. outer_shell is empty when no infill outers
    shell_ts = payload['shell_ts']
    outer_shells_ts_tagdict = payload['outer_shells_ts_tagdict']

    # other gt data
    st_octree_wn_dict = gt['st_octree_wn_dict']
    ost_octree_wn_dict_tagdict = gt['ost_octree_wn_dict_tagdict']
    base_coll_th = gt['base_coll_th']


    # model output
    quaternions = model_output['model_out']  
    waypts = model_output['model_in'] 

    # quaternion must be normalized, smoothness on para axes
    quaternions = F.normalize(quaternions, dim=-1).reshape(-1, 4) 
    print_dirs, frame_dirs = quaternion_para_axes(quaternions)  

    # print dir close to orginal print dir
    print_dir_align_loss = (1 - F.cosine_similarity(print_dirs, waypt_print_dirs, dim=-1)).mean()

    # sf free loss and metrics (optional) 
    sf_valid = waypt_si_tags != 1
    if base_threshold is not None:
        sf_valid &= waypts.reshape(-1, 3)[:, 2] > base_threshold
    if sf_valid.any():
        sf_loss_term, sf_ratio = cal_sf_loss_(waypt_normals[sf_valid], print_dirs[sf_valid])
        sf_loss = sf_loss_term.mean()
    else:
        sf_loss = print_dirs.sum() * 0.0
        sf_ratio = 100.0  # No eligible shell points violate the SF condition.

    # axes smoothness
    print_smooth_loss = print_dirs.sum() * 0.0
    if paras.get('print_dir_vec', 0.0) != 0:
        print_smooth_loss = diff_operators.vector_dirichlet(print_dirs, waypts).mean()

    frame_smooth_loss = frame_dirs.sum() * 0.0
    if paras.get('frame_dir_vec', 0.0) != 0:
        frame_smooth_loss = diff_operators.vector_dirichlet(frame_dirs, waypts).mean()

    # collision evaluation 
    waypts_cp = waypts.detach().reshape(-1, 3) 
    transed_nozzle_pcd = transform_nozzle_pcd_with_quaternion(nozzle_pcd, waypts_cp, quaternions)
    transed_nozzle_grads = transform_nozzle_grads_with_quaternion(nozzle_grads, waypts_cp, quaternions) # no translation

    coll_depths, _, waypt_coll_mask = collision_evaluation_with_infill(
                transed_nozzle_pcd, transed_nozzle_grads, local_map, udf_decoder,
                base_th=base_coll_th, udf_init_margin=udf_init_margin,
                shell_ts=shell_ts, outer_shells_ts_tagdict=outer_shells_ts_tagdict,
                st_octree_wn_dict=st_octree_wn_dict, ost_octree_wn_dict_tagdict=ost_octree_wn_dict_tagdict,)
        
    coll_loss = coll_depths.sum()  # total number as a weight
    coll_ratio = waypt_coll_mask.float().mean().item() * 100.0


    losses = {
        'coll': coll_loss * paras.get('coll'), 
        'sf': sf_loss * paras.get('sf'), 
        'print_dir_align': print_dir_align_loss * paras.get('print_dir_align'),
        'print_dir_vec': print_smooth_loss * paras.get('print_dir_vec'),
        'frame_dir_vec': frame_smooth_loss * paras.get('frame_dir_vec'),
    }

    metrics = {'collision_rate': coll_ratio, 'sf_rate': sf_ratio}

    return losses, metrics






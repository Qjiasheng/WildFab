import torch
from torch.autograd import grad
import torch.nn.functional as F


def hessian(y, x):
    ''' hessian of y wrt x
    y: shape (meta_batch_size, num_observations, channels)
    x: shape (meta_batch_size, num_observations, 2)
    '''
    meta_batch_size, num_observations = y.shape[:2]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)
    h = torch.zeros(meta_batch_size, num_observations, y.shape[-1], x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(y.shape[-1]):
        # calculate dydx over batches for each feature value of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]

        # calculate hessian on y for each x value
        for j in range(x.shape[-1]):
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y, create_graph=True)[0][..., :]

    status = 0
    if torch.any(torch.isnan(h)):
        status = -1
    return h, status

def hessian3d(y, x):
    '''
    Compute the Hessian of y with respect to x (3D coordinates).
    y: shape (num_observations, channels), channels is 1
    x: shape (num_observations, 3)
    '''
    num_observations = y.shape[0]
    channels = y.shape[1]
    grad_y = torch.ones_like(y[..., 0]).to(y.device)

    h = torch.zeros(num_observations, channels, x.shape[-1], x.shape[-1]).to(y.device)
    for i in range(channels):  # Loop over channels of y
        dydx = grad(y[..., i], x, grad_y, create_graph=True)[0]
    
        for j in range(x.shape[-1]):  # Loop over the 3 coordinates (x1, x2, x3)
            # Compute the second derivative of dydx[..., j] w.r.t. the input x
            h[..., i, j, :] = grad(dydx[..., j], x, grad_y, create_graph=True)[0][..., :]
    return h  # (num_observations, channels, 3, 3)


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def vector_laplace_new(y, x, norm=False, min_norm=1e-6):
    grad = gradient(y, x)
    if norm:
        normas = torch.norm(grad, dim=-1, keepdim=True).clamp(min=min_norm)  # Compute the original norms
        total_norm = torch.sum(normas)  # Total sum of original norms
        scale = grad.shape[1] / (total_norm + 1e-6)  # Compute the scaling factor to ensure the sum of lengths equals N
        grad = grad * scale  # Scale the vectors proportionally
    div = divergence(grad, x)
    vec_lap = gradient(div, x)
    return vec_lap


def scalar_dirichlet(y, x):
    grad = gradient(y, x)
    loss = torch.sum(grad ** 2, dim=-1)
    return loss

def vector_dirichlet(y, x):
    """Pointwise Dirichlet energy; normalize y beforehand for a unit-vector field."""
    Jv, _ = jacobian(y, x)
    loss = torch.sum(Jv ** 2, dim=(-1, -2))  # Frobenius norm squared
    return loss


def surface_vector_laplace(v, x, normals, norm=False):
    """
    \Delta F = \nabla( \nabla \cdot F) - \nabla \times (\nabla \times F) 
    approximate surface vector laplace by projection noto tangent plane. 
    rescale vector norms to total length. Direct set all vectors as unit length causing inf energy at singularities.
    see paper --- Globally Optimal Direction Fields, Felix Knöppel et al., TOG 2013.
    https://www.cs.cmu.edu/~kmcrane/Projects/GloballyOptimalDirectionFields/
    """

    def project_onto_tangent(v, normals):
        dot = torch.sum(v * normals, dim=-1, keepdim=True)  
        v_proj = v - dot * normals  
        return v_proj
    
    def surface_divergence(v, x, normals):
        v_proj = project_onto_tangent(v, normals)  # Ensure the vector field is tangential
        div = divergence(v_proj, x)  # Compute the divergence in 3D
        return div
    
    def surface_gradient(y, x, normals):
        grad = gradient(y, x)  # Full 3D gradient
        dot = torch.sum(grad * normals, dim=-1, keepdim=True)  # Component along normal
        grad_surf = grad - dot * normals  # Project onto tangent plane
        return grad_surf
    
    def normalize_proj(v):
        norms = torch.norm(v, dim=-1, keepdim=True).clamp(min=0.5)  # Compute the original norms
        total_norm = torch.sum(norms)  # Total sum of original norms
        scale = v.shape[1] / (total_norm + 1e-6)  # Compute the scaling factor to ensure the sum of lengths equals N
        v_normalized = v * scale  # Scale the vectors proportionally
        return v_normalized

    if norm:
        v = F.normalize(v, p=2, dim=-1)
   
    # \Delta F = \nabla( \nabla \cdot F) - \nabla \times (\nabla \times F) 
    # gradient, thus curl is zero. D
    v_proj = project_onto_tangent(v, normals) 
    v_proj = normalize_proj(v_proj)  # Normalize the projected vector field
    div_v = surface_divergence(v_proj, x, normals)
    laplacian = surface_gradient(div_v, x, normals)

    return laplacian.squeeze(-1), v_proj


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def jacobian(y, x):
    """Pointwise Jacobian for independent samples: (..., C) wrt (..., D).

    Returns (..., C, D) and a status flag. Leading dimensions must match.
    Keep the derivative graph so losses on the Jacobian can train the field.
    """
    rows = []
    for i in range(y.shape[-1]):
        component = y[..., i]
        row = None
        if component.requires_grad:
            row = torch.autograd.grad(
                component, x, torch.ones_like(component),
                create_graph=True, retain_graph=True, allow_unused=True)[0]
        if row is None:
            row = x * 0.0 + component.unsqueeze(-1) * 0.0
        rows.append(row)
    jac = torch.stack(rows, dim=-2)

    status = 0
    if torch.any(torch.isnan(jac)):
        status = -1

    return jac, status

def curl(y, x):
    """curl of y wrt x
    curl of gradient is always zero"""
    pass

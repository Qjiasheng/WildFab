import torch
import torch.nn.functional as F
from . import _C

# Fast winding number field from clouds with Octree acceleration structure.
# No time dimension in octree cell, purely spatial.



def py_build_octree(points, max_points_per_leaf=40, max_depth=16):
    """build octree from points on CPU
    return
    ----------------------
    point_indices: [[],[],[]...] list of lists of point indices in each node. empty sublist means internal node.
                    non-empty sublist means leaf node. number of non-empty sublists = number of leaf nodes containing points.
                    determined by max_points_per_leaf and max_depth in the C++ code. The finest case is one point per leaf.
                    Different from libigl (internal node also has all contained clouds indices).

    child_nodes: tensor of shape (num_nodes, 8), each row indicates the indices of the 8 child nodes. 
                 all -1 means no child, i.e., leaf node, if its point_indices[node] is empty, empty leaf node.

    """ 
    return _C.build_octree(points, max_points_per_leaf, max_depth)

def py_initialize_octree(points, normals, areas, point_indices, children):
    """initialize octree nodes from leaf to root on CPU
    ----------------------
    return
    centers: tensor of shape (num_nodes, 3), mass center of each node cell once has points. empty cell has (0.,0.,0.)
    normals: tensor of shape (num_nodes, 3), mass normal (area-weighted average) of each node cell once has points. empty cell has (0.,0.,0.)
    radii: empty cell, both internal/ leaf (no clouds included) has radius 0., single point leaf has radius 0.
    """ 
    return _C.initialize_octree(points, normals, areas, point_indices, children)


def _get_leaf_tensors(points, point_indices):
    """Cache immutable octree topology on its owning points tensor, per device.

    Replacing point_indices rebuilds the cache. Do not mutate its lists in place.
    The cache is released with points; no global cache retains old CUDA trees.
    """
    cache = getattr(points, '_leaf_index_cache', None)
    if cache is None or cache[0] is not point_indices or cache[1].device != points.device:
        offsets = [0]
        indices = []
        for leaf in point_indices:
            indices.extend(leaf)
            offsets.append(len(indices))
        leaf_offsets = torch.tensor(offsets, dtype=torch.int64, device=points.device)
        leaf_indices = torch.tensor(indices, dtype=torch.int64, device=points.device)
        cache = (point_indices, leaf_offsets, leaf_indices)
        points._leaf_index_cache = cache
    return cache[1], cache[2]


def batch_query_wn_clouds(queries, points, normals, areas, point_indices,          
        children, cell_centers, cell_normals, cell_radii, inv_delta=128.0):
    """all tensors are in cuda.
    inv_delta: 2.0/ inv_delta is the distance threshold for switching from exact WN to regularized WN
    return
    fields: tensor of shape (M), WN field values at query points
    field_grads: tensor of shape (M,3), WN field gradients at query points
    near_tags: tensor of shape (M), 1 if query is near the clouds, 0 otherwise. 
    """
    
    leaf_offsets, leaf_indices = _get_leaf_tensors(points, point_indices)

    return _C.fast_wn_clouds(queries, points, normals, areas, leaf_offsets, leaf_indices, 
                             children, cell_centers, cell_radii, cell_normals, inv_delta)
    
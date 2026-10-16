"""Point-cloud heat distance and tangent gradient helpers for pre_data.py."""

import numpy as np


def normalize_rows(values, eps=1e-12):
    out = np.asarray(values, dtype=np.float64).copy()
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > eps)
    out[valid] /= norms[valid]
    out[~valid] = 0.0
    return out


def get_source_idx_axis_band(coords, tags=None, axis=2, side="min", width=0.02, exclude_tag=None):
    """Select an inclusive band in coordinate units (normalized units in pre_data)."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or not len(coords) or not np.all(np.isfinite(coords)):
        raise ValueError("Heat sources require a nonempty finite (N, 3) cloud")
    if isinstance(axis, bool) or axis not in (0, 1, 2):
        raise ValueError("source_axis must be 0, 1, or 2")
    axis = int(axis)
    width = float(width)
    if not np.isfinite(width) or width < 0:
        raise ValueError("source_width must be finite and nonnegative")
    eligible = np.ones(len(coords), dtype=bool)
    if exclude_tag is not None:
        if tags is None:
            raise ValueError("tags are required with exclude_tag")
        tag_values = np.asarray(tags)
        if tag_values.ndim == 2:
            tag_values = tag_values[:, -1]
        if tag_values.shape != (len(coords),):
            raise ValueError("tags must have one row per point")
        eligible &= tag_values != exclude_tag
    if not np.any(eligible):
        raise ValueError("No eligible heat source points")
    values = coords[:, axis]
    if side == "min":
        condition = values <= np.min(values[eligible]) + width
    elif side == "max":
        condition = values >= np.max(values[eligible]) - width
    else:
        raise ValueError("source_side must be 'min' or 'max'")
    return np.flatnonzero(condition & eligible).tolist()


def heat_pcd(coords, source_idx, t_coef=1.0):
    """Compute heat distance; the native solver expects an integer for one source."""
    import potpourri3d as pp3d
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) < 3 or not np.all(np.isfinite(coords)):
        raise ValueError("Heat computation requires at least 3 finite 3D points")
    if np.linalg.matrix_rank(coords - coords.mean(axis=0)) < 2:
        raise ValueError("Heat computation requires a surface cloud, not collinear points")
    if not np.isfinite(t_coef) or t_coef <= 0:
        raise ValueError("heat_t_coef must be finite and positive")
    sources = np.asarray(source_idx, dtype=np.int64)
    if sources.ndim != 1 or not len(sources) or np.any(sources < 0) or np.any(sources >= len(coords)):
        raise ValueError("Heat computation requires valid source indices")
    # OBJ seams can contain coincident vertices. Solve once per position and
    # map distances back without changing exported rows or their normals.
    coords, inverse = np.unique(coords, axis=0, return_inverse=True)
    sources = np.unique(inverse[sources])
    if len(sources) == len(coords):
        return np.zeros(len(inverse))
    solver = pp3d.PointCloudHeatSolver(coords, t_coef=t_coef)
    distances = solver.compute_distance(int(sources[0])) if len(sources) == 1 else solver.compute_distance_multisource(sources.tolist())
    distances = np.asarray(distances)
    if distances.shape != (len(coords),) or not np.all(np.isfinite(distances)):
        raise ValueError("Heat solver returned invalid distances; check point-cloud connectivity and sampling")
    return np.maximum(distances, 0.0)[inverse]


def get_heat_grad(points, distances, k_neighbors=20, normals=None):
    """Local least-squares gradients, optionally constrained to each tangent plane."""
    from scipy.spatial import cKDTree
    points = np.asarray(points, dtype=np.float64)
    distances = np.asarray(distances, dtype=np.float64)
    if len(points) < 3 or int(k_neighbors) < 3:
        raise ValueError("Heat gradients require at least 3 points and heat_k_neighbors >= 3")
    if distances.shape != (len(points),) or not np.all(np.isfinite(distances)):
        raise ValueError("Expected one finite heat distance per point")
    if normals is not None:
        normals = normalize_rows(normals)
        if normals.shape != points.shape or np.any(np.linalg.norm(normals, axis=1) < 1e-12):
            raise ValueError("Tangent gradients require nonzero normals for every point")
    tree = cKDTree(points)
    gradients = np.zeros_like(points)
    for i in range(len(points)):
        _, ids = tree.query(points[i], k=min(int(k_neighbors) + 1, len(points)))
        delta = points[ids] - points[i]
        values = distances[ids] - distances[i]
        if normals is None:
            gradients[i] = np.linalg.lstsq(delta, values, rcond=None)[0]
        else:
            normal = normals[i]
            reference = np.array([1., 0., 0.]) if abs(normal[0]) < 0.9 else np.array([0., 1., 0.])
            t1 = np.cross(normal, reference)
            t1 /= np.linalg.norm(t1)
            basis = np.column_stack((t1, np.cross(normal, t1)))
            gradient, _, rank, _ = np.linalg.lstsq(delta @ basis, values, rcond=None)
            if rank == 2:
                gradients[i] = basis @ gradient
    return gradients

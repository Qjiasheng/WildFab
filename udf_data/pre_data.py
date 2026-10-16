"""Normalize mesh/PCD data, compute curvature and heat, and export 19 columns."""

import argparse
from pathlib import Path

import numpy as np
import yaml

if __package__:
    from .heat_pcd import get_source_idx_axis_band, heat_pcd, get_heat_grad, normalize_rows
else:
    from heat_pcd import get_source_idx_axis_band, heat_pcd, get_heat_grad, normalize_rows


def get_igl():
    import igl
    return igl


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def obj_vertex_index(token, vertex_count):
    raw = token.split("/")[0]
    index = int(raw)
    if index < 0:
        return vertex_count + index
    return index - 1


def read_obj_triangle_mesh(path):
    vertices = []
    faces = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                face = [obj_vertex_index(token, len(vertices)) for token in parts[1:]]
                for i in range(1, len(face) - 1):
                    faces.append([face[0], face[i], face[i + 1]])
    if not vertices or not faces:
        raise ValueError(f"{path} does not contain vertices and triangle faces")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)


def read_triangle_mesh(path):
    path = Path(path)
    if path.suffix.lower() == ".obj":
        return read_obj_triangle_mesh(path)
    return get_igl().read_triangle_mesh(str(path))


def load_config(path):
    raw = read_yaml(path)
    defaults = dict(raw.get("defaults") or {})

    models = dict(raw.get("models_data") or {})

    default_model = raw.get("default_model")
    return defaults, models, default_model


def merge_params(defaults, params, args):
    merged = dict(defaults)
    merged.update(params)
    if args.ele_edge_len is not None:
        merged["ele_edge_len"] = args.ele_edge_len
    if args.save_path is not None:
        merged["save_path"] = args.save_path
    if args.out_pcd_fn is not None:
        merged["out_pcd_fn"] = args.out_pcd_fn
    if args.out_mesh_fn is not None:
        merged["out_mesh_fn"] = args.out_mesh_fn
    if args.mesh_path is not None:
        merged["mesh_path"] = args.mesh_path
        merged.pop("pcd_path", None)
    if args.pcd_path is not None:
        merged["pcd_path"] = args.pcd_path
        merged.pop("mesh_path", None)
    if args.no_sample:
        merged["sample_surface"] = False
    if args.no_downsample:
        merged["downsample_ratio"] = 1.0
    if args.downsample_ratio is not None:
        merged["downsample_ratio"] = args.downsample_ratio
    if args.vis:
        merged["visualize"] = True
    if args.no_vis:
        merged["visualize"] = False
    return merged


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def get_pyvista():
    try:
        import pyvista as pv
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("pyvista is required when visualize is true or --vis is used") from exc

    return pv


def pv_mesh(v, f):
    pv = get_pyvista()
    faces = np.insert(f, 0, 3, axis=1).astype(np.int64).ravel()
    return pv.PolyData(v, faces)


def vis_mesh(v, f, label=None):
    pv = get_pyvista()
    mesh = pv_mesh(v, f)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, smooth_shading=True, color="lightblue", opacity=0.35)
    if label is not None:
        plotter.add_text(label, position="upper_left", font_size=18, color="black")
    plotter.add_axes()
    plotter.show()


def vis_dirs(v, f, dirs, mag=0.1, label=None):
    pv = get_pyvista()
    mesh = pv_mesh(v, f)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, smooth_shading=True, color="lightblue", opacity=0.25)
    plotter.add_arrows(v, dirs, color="green", mag=mag)
    if label is not None:
        plotter.add_text(label, position="upper_left", font_size=18, color="black")
    plotter.add_axes()
    plotter.show()


def vis_pcd_dirs(coords, dirs, mag=0.03, label=None, point_size=4):
    if len(coords) == 0:
        return
    pv = get_pyvista()
    point_cloud = pv.PolyData(coords)
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, color="lightblue", point_size=point_size, render_points_as_spheres=True, opacity=0.35)
    plotter.add_arrows(coords, dirs, color="green", mag=mag)
    if label is not None:
        plotter.add_text(label, position="upper_left", font_size=18, color="black")
    plotter.add_axes()
    plotter.show()


def vis_pcd_fields(coords, fields, label=None, cmap="seismic", point_size=10):
    if len(coords) == 0:
        return
    pv = get_pyvista()
    point_cloud = pv.PolyData(coords)
    point_cloud["field"] = np.asarray(fields)
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, scalars="field", point_size=point_size, render_points_as_spheres=True, cmap=cmap)
    if label is not None:
        plotter.add_text(label, position="upper_left", font_size=18, color="black")
    plotter.add_axes()
    plotter.show()


def vis_boundary_loops(v, f, loops, label=None, show_loop_ids=False, interactive=True, auto_close=True, interactive_update=False):
    pv = get_pyvista()
    mesh = pv_mesh(v, f)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, show_edges=True, smooth_shading=True, color="lightblue", opacity=0.35)
    label_points = []
    labels = []
    for loop_id, loop in enumerate(loops, start=1):
        loop = np.asarray(loop, dtype=np.int64)
        if loop.size == 0:
            continue
        color = np.random.rand(3)
        plotter.add_mesh(pv.PolyData(v[loop]), color=color, point_size=15, render_points_as_spheres=True)
        if show_loop_ids:
            loop_points = v[loop]
            center = np.mean(loop_points, axis=0)
            label_points.append(loop_points[np.argmin(np.linalg.norm(loop_points - center, axis=1))])
            labels.append(str(loop_id))
    if show_loop_ids and label_points:
        plotter.add_point_labels(
            np.asarray(label_points),
            labels,
            font_size=50,
            point_size=0,
            shape_opacity=0.65,
            text_color="black",
            shape_color="white",
        )
    if label is not None:
        plotter.add_text(label, position="upper_left", font_size=18, color="black")
    plotter.add_axes()
    plotter.show(interactive=interactive, interactive_update=interactive_update, auto_close=auto_close)
    return plotter


def normalization_params(v, scale_range):
    if scale_range is None:
        return None
    scale_range = float(scale_range)
    if not np.isfinite(scale_range) or scale_range <= 0:
        raise ValueError("scale_range must be finite and positive")
    lower, upper = np.min(v, axis=0), np.max(v, axis=0)
    span = np.max(upper - lower)
    if span <= 0:
        raise ValueError("Cannot normalize a point cloud with zero extent")
    return (lower + upper) / 2, scale_range / span


def apply_normalization(v, params):
    out = v.astype(np.float64, copy=True)
    if params is not None:
        center, scale = params
        out = (out - center) * scale
    return out


def normalize_vectors(values, fallback):
    out = np.asarray(values, dtype=np.float64).copy()
    norms = np.linalg.norm(out, axis=1)
    invalid = (~np.isfinite(norms)) | (norms < 1e-12)
    out[invalid] = fallback
    norms = np.linalg.norm(out, axis=1)
    norms[norms < 1e-12] = 1.0
    out /= norms[:, None]
    return out, invalid


def tangent_basis(normals):
    normals, _ = normalize_vectors(normals, np.array([0.0, 0.0, 1.0]))
    ref = np.tile(np.array([[1.0, 0.0, 0.0]]), (len(normals), 1))
    near_parallel = np.abs(np.einsum("ij,ij->i", normals, ref)) > 0.9
    ref[near_parallel] = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(normals, ref)
    t1, _ = normalize_vectors(t1, np.array([1.0, 0.0, 0.0]))
    t2 = np.cross(normals, t1)
    t2, _ = normalize_vectors(t2, np.array([0.0, 1.0, 0.0]))
    return t1, t2


def project_to_tangent(values, normals):
    projected = values - np.sum(values * normals, axis=1, keepdims=True) * normals
    out, invalid = normalize_vectors(projected, np.array([1.0, 0.0, 0.0]))
    if np.any(invalid):
        t1, _ = tangent_basis(normals)
        out[invalid] = t1[invalid]
    return out, invalid


def face_edges(face):
    return ((int(face[0]), int(face[1])), (int(face[1]), int(face[2])), (int(face[2]), int(face[0])))


def edge_key(a, b):
    a = int(a)
    b = int(b)
    return (a, b) if a < b else (b, a)


def build_edge_faces(f):
    edge_faces = {}
    for fi, face in enumerate(f):
        for a, b in face_edges(face):
            edge_faces.setdefault(edge_key(a, b), []).append(fi)
    return edge_faces


def boundary_loops_from_edges(edge_faces):
    boundary_edges = [key for key, faces in edge_faces.items() if len(faces) == 1]
    adjacency = {}
    for a, b in boundary_edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)

    visited_edges = set()
    loops = []
    for start_edge in boundary_edges:
        if start_edge in visited_edges:
            continue

        start, current = start_edge
        previous = start
        loop = [start]
        while True:
            loop.append(current)
            visited_edges.add(edge_key(previous, current))

            next_vertices = [
                v
                for v in adjacency.get(current, [])
                if v != previous and edge_key(current, v) not in visited_edges
            ]
            if not next_vertices:
                if start in adjacency.get(current, []) and edge_key(current, start) not in visited_edges:
                    next_vertices = [start]

            if not next_vertices:
                break

            next_vertex = next_vertices[0]
            previous, current = current, next_vertex
            if current == start:
                visited_edges.add(edge_key(previous, current))
                break

        if loop:
            loops.append(np.asarray(loop, dtype=np.int64))

    return loops


def boundary_tags(edge_faces, n_vertices, enabled):
    tags = np.zeros(n_vertices, dtype=np.int32)
    boundary_edge_tags = {}
    if not enabled:
        return tags, boundary_edge_tags, []

    loops = boundary_loops_from_edges(edge_faces)
    for loop_id, loop in enumerate(loops, start=1):
        loop = np.asarray(loop, dtype=np.int64)
        if loop.size == 0:
            continue
        tags[loop] = loop_id
        if loop.size > 1:
            rolled = np.roll(loop, -1)
            for a, b in zip(loop, rolled):
                key = edge_key(a, b)
                if len(edge_faces.get(key, ())) == 1:
                    boundary_edge_tags[key] = loop_id

    return tags, boundary_edge_tags, loops


def nonmanifold_tags(f, n_vertices, edge_faces, enabled):
    """Label connected non-manifold edge groups with IDs 1, 2, ... (0 elsewhere).

    Connectivity uses shared mesh vertex indices, not coordinate proximity.
    A group can be a closed loop, an open chain, or a branched network. Groups
    meeting at a vertex share one ID because each exported vertex has one tag.
    Returns vertex tags and an edge-to-group-ID map for surface sampling.
    """
    tags = np.zeros(n_vertices, dtype=np.int32)
    nm_edges = set()
    if not enabled:
        return tags, {}

    ok, bf, e, emap, be = get_igl().is_edge_manifold(f)
    e = np.asarray(e, dtype=np.int64)
    be = np.asarray(be, dtype=bool)
    if e.size and be.size:
        for a, b in e[~be]:
            nm_edges.add(edge_key(a, b))

    for key, faces in edge_faces.items():
        if len(faces) > 2:
            nm_edges.add(key)

    adjacency = {}
    for a, b in nm_edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    # Sorted roots make IDs deterministic, independent of face/edge ordering.
    group_id = 0
    for root in sorted(adjacency):
        if tags[root]:
            continue
        group_id += 1
        tags[root] = group_id
        pending = [root]
        while pending:
            vertex = pending.pop()
            for neighbor in adjacency[vertex]:
                if not tags[neighbor]:
                    tags[neighbor] = group_id
                    pending.append(neighbor)

    return tags, {key: int(tags[key[0]]) for key in sorted(nm_edges)}


def compute_normals(v, f):
    normals = get_igl().per_vertex_normals(v, f)
    return normalize_vectors(normals, np.array([0.0, 0.0, 1.0]))


def vis_heat_sources(coords, source_idx):
    pv = get_pyvista()
    plotter = pv.Plotter()
    plotter.add_points(coords, color="lightgray", point_size=4, opacity=0.3)
    plotter.add_points(coords[source_idx], color="red", point_size=12,
                       render_points_as_spheres=True, label="Heat sources")
    plotter.add_text(f"Selected heat sources: {len(source_idx)}", font_size=16)
    plotter.add_legend()
    plotter.add_axes()
    plotter.show()


def compute_heat_directions(coords, normals, params):
    source_idx = get_source_idx_axis_band(
        coords, axis=params.get("source_axis", 2),
        side=params.get("source_side", "min"), width=params.get("source_width", 0.02))
    distances = heat_pcd(coords, source_idx, float(params.get("heat_t_coef", 1.0)))
    gradients = get_heat_grad(coords, distances, int(params.get("heat_k_neighbors", 20)), normals)
    directions = normalize_rows(gradients)
    invalid = np.linalg.norm(directions, axis=1) < 1e-12
    return directions, invalid, source_idx, distances


def curvature_features(mean_curv, invalid, feature_percent):
    tags = np.zeros(len(mean_curv), dtype=np.int32)
    threshold = np.inf
    if feature_percent is not None:
        fraction = float(feature_percent)
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("feature_percent must be a fraction in [0, 1]")
        valid = ~invalid & np.isfinite(mean_curv)
        if fraction > 0 and np.any(valid):
            threshold = np.percentile(mean_curv[valid], (1 - fraction) * 100)
            tags[valid] = (mean_curv[valid] >= threshold).astype(np.int32)
    return tags, threshold


def compute_curvature(v, f, normals, radius, feature_percent):
    """Return directions in libigl order: larger, then smaller signed curvature."""
    if int(radius) < 1:
        raise ValueError("curv_radius must be at least 1")
    pd1, pd2, pv1, pv2, badv = get_igl().principal_curvature(v, f, radius=int(radius), useKring=True)
    mean_curv = (np.abs(pv1) + np.abs(pv2)) * 0.5
    pd1, invalid1 = project_to_tangent(pd1, normals)
    pd2, invalid2 = project_to_tangent(pd2, normals)
    invalid = invalid1 | invalid2 | ~np.isfinite(mean_curv)
    if badv is not None:
        invalid[np.asarray(badv, dtype=np.int64)] = True
    flat = np.isfinite(mean_curv) & np.isclose(mean_curv, 0)
    t1, t2 = tangent_basis(normals)
    pd1[flat], pd2[flat] = t1[flat], t2[flat]
    invalid[flat] = False
    tags, threshold = curvature_features(mean_curv, invalid, feature_percent)
    return pd1, pd2, mean_curv, tags, invalid, threshold


def compute_pcd_curvature(coords, normals, k_neighbors, feature_percent):
    """Fit a symmetric shape operator to local normal changes in a tangent plane.

    Uses supplied oriented normals; no mesh reconstruction. Neighborhood size is
    a point count, independent of the mesh's curv_radius (k-ring count).
    Match libigl's sign convention: outward convex surfaces have positive
    curvature. Direction 1 has the larger signed curvature, direction 2 the
    smaller. Eigenvector signs themselves are arbitrary in both methods.
    """
    from scipy.spatial import cKDTree
    if int(k_neighbors) < 3 or len(coords) < 3:
        raise ValueError("PCD curvature requires at least 3 points and curv_k_neighbors >= 3")
    tree = cKDTree(coords)
    t1, t2 = tangent_basis(normals)
    pd1, pd2 = t1.copy(), t2.copy()
    mean_curv = np.zeros(len(coords))
    invalid = np.zeros(len(coords), dtype=bool)
    for i in range(len(coords)):
        distances, ids = tree.query(coords[i], k=min(int(k_neighbors) + 1, len(coords)))
        keep = distances > 1e-12
        ids, distances = ids[keep], distances[keep]
        if len(ids) < 2:
            invalid[i] = True
            continue
        basis = np.column_stack((t1[i], t2[i]))
        uv = (coords[ids] - coords[i]) @ basis
        dn = (normals[ids] - normals[i]) @ basis
        weights = np.exp(-0.5 * (distances / distances[-1]) ** 2)
        # libigl uses +dN here; -dN would reverse the signed eigenvalue order
        # and swap the principal directions relative to the mesh path.
        derivative, _, rank, _ = np.linalg.lstsq(uv * weights[:, None], dn * weights[:, None], rcond=None)
        if rank < 2:
            invalid[i] = True
            continue
        values, vectors = np.linalg.eigh((derivative + derivative.T) / 2)
        pd1[i], pd2[i] = basis @ vectors[:, 1], basis @ vectors[:, 0]
        mean_curv[i] = np.mean(np.abs(values))
    tags, threshold = curvature_features(mean_curv, invalid, feature_percent)
    return pd1, pd2, mean_curv, tags, invalid, threshold


def read_point_cloud(path):
    data = np.loadtxt(path, ndmin=2)
    if data.shape[1] != 6 or len(data) == 0 or not np.all(np.isfinite(data)):
        raise ValueError(f"{path}: expected finite XYZ rows with exactly 6 columns [xyz, nxnynz]")
    normals, invalid = normalize_vectors(data[:, 3:6], np.array([0., 0., 1.]))
    if np.any(invalid):
        raise ValueError(f"{path}: point-cloud normals must be nonzero")
    return data[:, :3], normals


def bary_grid(n):
    values = [[], [], [], []]
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            zeros = (i == 0) + (j == 0) + (k == 0)
            if zeros >= 2:
                continue
            w = (i / n, j / n, k / n)
            if zeros == 0:
                values[0].append(w)
            elif i == 0:
                values[1].append(w)
            elif j == 0:
                values[2].append(w)
            else:
                values[3].append(w)
    return [np.asarray(value, dtype=np.float64).reshape((-1, 3)) for value in values]


def sample_edge_from_zero(face, zero_index):
    if zero_index == 0:
        return edge_key(face[1], face[2])
    if zero_index == 1:
        return edge_key(face[2], face[0])
    return edge_key(face[0], face[1])


def weighted_values(weights, values):
    return np.einsum("wk,fk->fw", weights, values).reshape(-1)


def weighted_vectors(weights, values):
    return np.einsum("wk,fkc->fwc", weights, values).reshape(-1, 3)


def append_weighted_samples(coords, normal_values, pd1_values, pd2_values, curv_values, bn_values, nm_values, v, f, face_ids, weights, normals, pd1, pd2, mean_curv, curv_threshold, curv_enabled, face_bn_tags, face_nm_tags, sample_block_size):
    if len(face_ids) == 0 or weights.size == 0:
        return
    weight_count = len(weights)
    chunk_size = max(1, int(sample_block_size) // max(1, weight_count))
    for start in range(0, len(face_ids), chunk_size):
        ids = face_ids[start:start + chunk_size]
        faces = f[ids]
        tri = v[faces]
        count = len(ids) * weight_count
        coords.append(weighted_vectors(weights, tri))
        normal_values.append(normalize_vectors(weighted_vectors(weights, normals[faces]), np.array([0.0, 0.0, 1.0]))[0])
        pd1_values.append(normalize_vectors(weighted_vectors(weights, pd1[faces]), np.array([1.0, 0.0, 0.0]))[0])
        pd2_values.append(normalize_vectors(weighted_vectors(weights, pd2[faces]), np.array([0.0, 1.0, 0.0]))[0])
        if curv_enabled:
            curv_values.append((weighted_values(weights, mean_curv[faces]) >= curv_threshold).astype(np.int32))
        else:
            curv_values.append(np.zeros(count, dtype=np.int32))
        bn_values.append(np.repeat(face_bn_tags[start:start + len(ids)], weight_count).astype(np.int32))
        nm_values.append(np.repeat(face_nm_tags[start:start + len(ids)], weight_count).astype(np.int32))


def downsample_by_ratio(coords, normals, pd1, pd2, curv_tags, bn_tags, nm_tags, ratio, seed):
    ratio = float(ratio)
    if ratio >= 1.0 or len(coords) == 0:
        return coords, normals, pd1, pd2, curv_tags, bn_tags, nm_tags
    if ratio <= 0.0:
        keep = np.empty(0, dtype=np.int64)
    else:
        target = max(1, int(np.floor(len(coords) * ratio)))
        rng = np.random.default_rng(int(seed))
        keep = np.sort(rng.choice(len(coords), size=target, replace=False))
    return coords[keep], normals[keep], pd1[keep], pd2[keep], curv_tags[keep], bn_tags[keep], nm_tags[keep]


def finalize_samples(coords, normals, pd1, pd2, curv_tags, bn_tags, nm_tags, downsample_ratio, downsample_seed):
    samples = stack_samples(coords, normals, pd1, pd2, curv_tags, bn_tags, nm_tags)
    before_count = len(samples[0])
    samples = downsample_by_ratio(*samples, downsample_ratio, downsample_seed)
    return samples, before_count, len(samples[0])


def collect_samples(v, f, normals, pd1, pd2, mean_curv, curv_threshold, curv_enabled, bn_vertex_tags, nm_vertex_tags, edge_faces, boundary_edge_tags, nm_edge_tags, ele_edge_len, sample_surface, downsample_ratio, downsample_seed, sample_block_size, valid):
    coords = [v[valid]]
    normal_values = [normals[valid]]
    pd1_values = [pd1[valid]]
    pd2_values = [pd2[valid]]
    curv_values = [(mean_curv[valid] >= curv_threshold).astype(np.int32) if curv_enabled else np.zeros(np.sum(valid), dtype=np.int32)]
    bn_values = [bn_vertex_tags[valid]]
    nm_values = [nm_vertex_tags[valid]]

    if not sample_surface or ele_edge_len is None or float(ele_edge_len) <= 0:
        return finalize_samples(coords, normal_values, pd1_values, pd2_values, curv_values, bn_values, nm_values, downsample_ratio, downsample_seed)

    ele_edge_len = float(ele_edge_len)
    edge_owner = {key: min(faces) for key, faces in edge_faces.items()}
    grid_cache = {}

    tri_all = v[f]
    edge_lengths = np.linalg.norm(tri_all[:, [1, 2, 0]] - tri_all[:, [0, 1, 2]], axis=2)
    face_steps = np.maximum(1, np.ceil(np.amax(edge_lengths, axis=1) / ele_edge_len).astype(np.int64))
    face_valid = np.all(valid[f], axis=1)
    large_faces = np.flatnonzero(face_valid & (face_steps > 1))
    large_steps = face_steps[large_faces]

    for n in np.unique(large_steps):
        if n not in grid_cache:
            grid_cache[n] = bary_grid(n)
        face_ids = large_faces[large_steps == n]
        weights = grid_cache[n][0]
        append_weighted_samples(
            coords,
            normal_values,
            pd1_values,
            pd2_values,
            curv_values,
            bn_values,
            nm_values,
            v,
            f,
            face_ids,
            weights,
            normals,
            pd1,
            pd2,
            mean_curv,
            curv_threshold,
            curv_enabled,
            np.zeros(len(face_ids), dtype=np.int32),
            np.zeros(len(face_ids), dtype=np.int32),
            sample_block_size,
        )
        for zero_index, weights in enumerate(grid_cache[n][1:]):
            if weights.size == 0:
                continue
            edge_keys = [sample_edge_from_zero(face, zero_index) for face in f[face_ids]]
            owner_mask = np.asarray([edge_owner.get(key) == face_id for key, face_id in zip(edge_keys, face_ids)], dtype=bool)
            if not np.any(owner_mask):
                continue
            selected_keys = [key for key, keep in zip(edge_keys, owner_mask) if keep]
            selected_faces = face_ids[owner_mask]
            face_bn = np.asarray([boundary_edge_tags.get(key, 0) for key in selected_keys], dtype=np.int32)
            face_nm = np.asarray([nm_edge_tags.get(key, 0) for key in selected_keys], dtype=np.int32)
            append_weighted_samples(
                coords,
                normal_values,
                pd1_values,
                pd2_values,
                curv_values,
                bn_values,
                nm_values,
                v,
                f,
                selected_faces,
                weights,
                normals,
                pd1,
                pd2,
                mean_curv,
                curv_threshold,
                curv_enabled,
                face_bn,
                face_nm,
                sample_block_size,
            )

    return finalize_samples(coords, normal_values, pd1_values, pd2_values, curv_values, bn_values, nm_values, downsample_ratio, downsample_seed)


def stack_samples(coords, normals, pd1, pd2, curv_tags, bn_tags, nm_tags):
    return (
        np.vstack(coords),
        np.vstack(normals),
        np.vstack(pd1),
        np.vstack(pd2),
        np.concatenate(curv_tags),
        np.concatenate(bn_tags),
        np.concatenate(nm_tags),
    )


def print_run_config(model_name, params):
    print(f"Config for {model_name}:")
    for key in sorted(params):
        print(f"  {key}: {params[key]}")


def process_model(model_name, params):
    print_run_config(model_name, params)
    mesh_path, pcd_path = params.get("mesh_path"), params.get("pcd_path")
    if bool(mesh_path) == bool(pcd_path):
        raise ValueError(f"{model_name}: configure exactly one of mesh_path or pcd_path")
    visualize = as_bool(params.get("visualize", False))
    enable_curv = as_bool(params.get("enable_curv", True))
    enable_heat = as_bool(params.get("enable_heat", False))
    ratio = float(params.get("downsample_ratio", 1.0))
    if not np.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("downsample_ratio must be in (0, 1]")
    if mesh_path:
        v, f = read_triangle_mesh(mesh_path)
        if not np.all(np.isfinite(v)) or np.any(f < 0) or np.any(f >= len(v)):
            raise ValueError("Mesh contains nonfinite coordinates or invalid vertex indices")
    else:
        v, normals = read_point_cloud(pcd_path)
        f = None
    norm_params = normalization_params(v, params.get("scale_range", 1.9)) if as_bool(params.get("enable_norm", True)) else None
    v = apply_normalization(v, norm_params)
    bn_tags = np.zeros(len(v), dtype=np.int32)
    nm_tags = np.zeros(len(v), dtype=np.int32)
    if f is not None:
        normals, normal_invalid = compute_normals(v, f)
        edge_faces = build_edge_faces(f)
        bn_tags, boundary_edge_tags, loops = boundary_tags(edge_faces, len(v), as_bool(params.get("enable_bn", True)))
        nm_tags, nm_edge_tags = nonmanifold_tags(f, len(v), edge_faces, as_bool(params.get("enable_nm", False)))
        if visualize:
            vis_mesh(v, f, f"{model_name} normalized mesh")
            if loops:
                vis_boundary_loops(v, f, loops, "Boundary Loop IDs", show_loop_ids=True)
            if nm_edge_tags:
                groups = [np.flatnonzero(nm_tags == group_id)
                          for group_id in range(1, int(nm_tags.max()) + 1)]
                vis_boundary_loops(v, f, groups, "Non-Manifold Group IDs", show_loop_ids=True)
    else:
        normal_invalid = np.zeros(len(v), dtype=bool)
        if as_bool(params.get("enable_bn", False)) or as_bool(params.get("enable_nm", False)):
            print("PCD input: mesh boundary/non-manifold detection is unavailable; exporting zero tags.")

    if enable_curv:
        if f is not None:
            curvature = compute_curvature(v, f, normals, params.get("curv_radius", 5), params.get("feature_percent", 0.5))
        else:
            curvature = compute_pcd_curvature(v, normals, params.get("curv_k_neighbors", 30), params.get("feature_percent", 0.5))
        pd1, pd2, mean_curv, curv_tags, curv_invalid, threshold = curvature
        if visualize:
            vis_pcd_fields(v, mean_curv, label="Mean Absolute Principal Curvature", cmap="coolwarm")
            vis_pcd_dirs(v, pd1, label="Principal Direction 1")
            vis_pcd_dirs(v, pd2, label="Principal Direction 2")
    else:
        pd1, pd2 = tangent_basis(normals)
        mean_curv, curv_tags = np.zeros(len(v)), np.zeros(len(v), dtype=np.int32)
        curv_invalid, threshold = np.zeros(len(v), dtype=bool), np.inf
    valid = ~(normal_invalid | curv_invalid)
    if f is not None:
        samples, before, _ = collect_samples(
            v, f, normals, pd1, pd2, mean_curv, threshold, enable_curv,
            bn_tags, nm_tags, edge_faces, boundary_edge_tags, nm_edge_tags,
            params.get("ele_edge_len"), as_bool(params.get("sample_surface", True)),
            ratio, params.get("downsample_seed", 0), params.get("sample_block_size", 1000000), valid)
    else:
        before = int(np.sum(valid))
        samples = downsample_by_ratio(v[valid], normals[valid], pd1[valid], pd2[valid],
                                     curv_tags[valid], bn_tags[valid], nm_tags[valid],
                                     ratio, params.get("downsample_seed", 0))
    coords, normals_s, pd1_s, pd2_s, curv_s, bn_s, nm_s = samples
    if not len(coords):
        raise ValueError(f"{model_name}: no valid samples remain after preprocessing")
    # Interpolated mesh directions must also lie in the sampled tangent plane.
    pd1_s, _ = project_to_tangent(pd1_s, normals_s)
    pd2_s = normalize_rows(np.cross(normals_s, pd1_s))
    heat_dirs = np.zeros_like(coords)
    source_idx = []
    if enable_heat:
        # Solve on the final normalized cloud so every exported row has a field.
        heat_dirs, heat_invalid, source_idx, distances = compute_heat_directions(coords, normals_s, params)
        print(f"Heat sources: {len(source_idx)}; undefined/zero gradients: {np.sum(heat_invalid)}")
        if visualize:
            vis_heat_sources(coords, source_idx)
            vis_pcd_fields(coords, distances, label="Heat Distance", cmap="seismic_r")
            vis_pcd_dirs(coords, heat_dirs, mag=0.05, label="Heat Gradient Directions")
    if visualize:
        vis_pcd_fields(coords, curv_s, label="Curvature Feature Tags", cmap="coolwarm")
        vis_pcd_fields(coords, bn_s, label="Boundary Loop Tags", cmap="coolwarm")
        vis_pcd_fields(coords, nm_s, label="Non-Manifold Group IDs", cmap="coolwarm")
        vis_pcd_dirs(coords, normals_s, label="Sample Normals")
    save_path = Path(params.get("save_path") or Path("logs") / model_name)
    save_path.mkdir(parents=True, exist_ok=True)
    out_pcd = save_path / params.get("out_pcd_fn", "mesh_data.xyz")
    app_tags = np.zeros(len(coords), dtype=np.int32)
    data = np.column_stack((coords, normals_s, pd1_s, pd2_s, heat_dirs, curv_s, bn_s, nm_s, app_tags))
    if not np.all(np.isfinite(data)):
        raise ValueError("Preprocessing produced nonfinite export values")
    np.savetxt(out_pcd, data, fmt=["%.6f"] * 15 + ["%d"] * 4)
    if f is not None:
        get_igl().write_triangle_mesh(str(save_path / params.get("out_mesh_fn", "mesh_data.obj")), v, f)
    print(f"{model_name}: valid vertices={np.sum(valid)}/{len(v)}, samples={before} -> {len(coords)}, output={out_pcd}")
    return data


def selected_model_names(args, models, default_model):
    if args.all:
        return list(models.keys())
    if args.model:
        names = args.model
    elif default_model:
        names = [default_model]
    else:
        names = [next(iter(models))]
    missing = [name for name in names if name not in models]
    if missing:
        raise ValueError(f"unknown model config: {', '.join(missing)}")
    return names


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("data_config.yaml")))
    parser.add_argument("--model", nargs="+")
    parser.add_argument("--all", action="store_true")
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--mesh-path")
    input_group.add_argument("--pcd-path")
    parser.add_argument("--save-path")
    parser.add_argument("--out-pcd-fn")
    parser.add_argument("--out-mesh-fn")
    parser.add_argument("--ele-edge-len", type=float)
    parser.add_argument("--no-sample", action="store_true")
    parser.add_argument("--downsample-ratio", type=float)
    parser.add_argument("--no-downsample", action="store_true")
    vis_group = parser.add_mutually_exclusive_group()
    vis_group.add_argument("--vis", action="store_true")
    vis_group.add_argument("--no-vis", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).resolve()
    defaults, models, default_model = load_config(config_path)
    # Config paths are relative to YAML; command-line paths are relative to cwd.
    for params in [defaults, *models.values()]:
        for key in ("mesh_path", "pcd_path", "save_path"):
            if params.get(key):
                params[key] = str(config_path.parent / params[key])
    if args.mesh_path or args.pcd_path:
        if args.all or args.model:
            raise ValueError("Use a direct input path or configured model selection, not both")
        input_path = args.mesh_path or args.pcd_path
        model_name = Path(input_path).stem
        params = merge_params(defaults, {"save_path": args.save_path or str(config_path.parent / "logs" / model_name)}, args)
        process_model(model_name, params)
        return
    if not models:
        raise ValueError("no models found in config")
    for model_name in selected_model_names(args, models, default_model):
        if args.all and not (models[model_name].get("mesh_path") or models[model_name].get("pcd_path")):
            print(f"Skipping {model_name}: input path is blank")
            continue
        params = merge_params(defaults, models[model_name], args)
        process_model(model_name, params)


if __name__ == "__main__":
    main()

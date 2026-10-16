import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import random
from matplotlib import cm
from matplotlib.colors import Normalize
pv.global_theme.allow_empty_mesh = True


def vis_grad(coords, grads, arrow_scale=0.05, label=None):
    """all np arrays.
    segments show dir, length shows its magnitude"""

    grad_norms = np.linalg.norm(grads, axis=1)

    normalized_grads = np.where(grad_norms[:, np.newaxis] > 0, grads / (grad_norms[:, np.newaxis]+1e-8), 0)
    scaled_grads = normalized_grads * grad_norms[:, np.newaxis] * arrow_scale
    
    end_points = coords + scaled_grads
    points = np.vstack([coords, end_points])  # Combine start and end points into one array
    lines = np.hstack([[2, i, i + len(coords)] for i in range(len(coords))])  # Connectivity for line segments
    
    poly_lines = pv.PolyData()
    poly_lines.points = points
    poly_lines.lines = lines  
    plotter = pv.Plotter()
    plotter.add_points(coords, color="grey", point_size=7, render_points_as_spheres=True)
    plotter.add_mesh(poly_lines, color="blue", line_width=2)
    if label is not None:
        plotter.add_text(label, position='upper_left', font_size=18, color='black')  
    plotter.show_axes()
    plotter.show()


def hist_item(arr, bins=100, x_range=None, title=None):
    plt.hist(arr, bins=bins)
    if x_range is not None:
        plt.xlim(x_range)
    plt.xlabel("Values")
    plt.ylabel("Frequency")
    plt.title(title if title is not None else "Histogram of Values")
    plt.grid()
    plt.show()


def vis_pcd_fields(coords, fields=None, label=None, cmap="seismic"):
    point_cloud = pv.PolyData(coords)
    if fields is None:
        fields = np.zeros(coords.shape[0])
    point_cloud["intensity"] = fields
    plotter = pv.Plotter()
    plotter.add_mesh(point_cloud, scalars="intensity", point_size=10, render_points_as_spheres=True, cmap=cmap)
    if label is not None:
        plotter.add_text(label, position='upper_left', font_size=18, color='black')
    plotter.show_axes()
    plotter.show()


def visvis_two_set_pcd(pcd1, pcd2, pcd1_size=20, pcd2_size=50):

    cloud1 = pv.PolyData(pcd1)  
    cloud2 = pv.PolyData(pcd2)  
    plotter = pv.Plotter()

    plotter.add_mesh(cloud1, color='gray', point_size=pcd1_size, render_points_as_spheres=True)
    plotter.add_mesh(cloud2, color='red', point_size=pcd2_size, render_points_as_spheres=True)
    plotter.add_axes()  
    plotter.show()


def vis_print_nozzle_mesh_comparison(raw_mesh=None, mesh=None, waypoints=None, add_cube=False):
    """
    vis raw nozzle mesh and transformed nozzle mesh at waypoints.
    """
    
    plotter = pv.Plotter()
    if raw_mesh is not None:
        plotter.add_mesh(raw_mesh, color='whitesmoke', opacity=0.5, 
        show_edges=False, metallic=1.0, roughness=0.3, specular=0.5, specular_power=50)
    if mesh is not None:
        plotter.add_mesh(mesh, color='whitesmoke', opacity=1.0, 
        show_edges=False, metallic=1.0, roughness=0.3, specular=0.5, specular_power=50)

    if waypoints is not None:
        # waypoints_cloud = pv.PolyData(waypoints)
        # plotter.add_mesh(waypoints_cloud, color="gray", point_size=5, render_points_as_spheres=True)
        num_waypoints = waypoints.shape[0]
        waypoint_scalars = np.linspace(0, 1, num_waypoints) 
        waypoints_cloud = pv.PolyData(waypoints)
        waypoints_cloud["scalars"] = waypoint_scalars 
        plotter.add_mesh(
            waypoints_cloud, cmap="coolwarm", scalars="scalars",
            point_size=8, render_points_as_spheres=True, label="Waypoints", show_scalar_bar=False
        )

    if add_cube:
        cube = pv.Cube(center=(0, 0, 0), x_length=2, y_length=2, z_length=2)
        plotter.add_mesh(cube, color="grey", opacity=0.1, show_edges=True)

    plotter.show_axes()
    plotter.show()


def vis_contours_normals(contours, normal_dict, start=None, end=None):
    sorted_keys = sorted(contours.keys())
    if start is None:
        start = 0
    if end is None:
        end = len(sorted_keys)
    sliced_keys = sorted_keys[start:end]

    all_points = []  
    all_normals = []  
    for iso_value in sliced_keys:
        subcontours = contours[iso_value]
        subnormals = normal_dict.get(iso_value, [])
        for subcontour, subnormal in zip(subcontours, subnormals):
            all_points.append(np.array(subcontour)) 
            all_normals.append(np.array(subnormal))  

    all_points_np = np.vstack(all_points)  
    all_normals_np = np.vstack(all_normals)  

    vis_grad(all_points_np, all_normals_np)


def random_color():
    return [random.random(), random.random(), random.random()]


def vis_isolayers(iso_layers):
    """visualize iso-layers"""
    plotter = pv.Plotter()
    for iso_value, (verts, faces) in iso_layers.items():
        # Convert vertices and faces to a PyVista-compatible mesh
        mesh = pv.PolyData(verts, np.c_[np.full(len(faces), 3), faces])
        plotter.add_mesh(mesh, color=random_color(), opacity=0.5, show_edges=False)
    plotter.add_axes()
    plotter.show()


def render_pcd(coords, label=None):
    coords = np.asarray(coords, dtype=np.float64)

    pcd = pv.PolyData(coords)

    pl = pv.Plotter()
    pl.add_mesh(
        pcd,
        color="lightgrey",
        point_size=12,
        render_points_as_spheres=True,
        lighting=True,
        ambient=0.35,
        diffuse=0.75,
        specular=0.25,
    )
    if label is not None:
        pl.add_text(label, font_size=15)

    pl.enable_eye_dome_lighting()
    pl.add_axes()
    pl.show()


def visualize_merged_iso_contours_mul(contours, start=None, end=None, line_width=7, random_color=True):
    """ Visualize merged iso-contours.
    Handles multiple contours (>1 sublists) for each iso-value. """

    def get_random_color(index, seed=0):
        random.seed(seed + index)  # combine seed with index for reproducibility
        return (random.random(), random.random(), random.random())

    sorted_keys = sorted(contours.keys())  
    if start is None:
        start = 0  # Default first index
    if end is None:
        end = len(sorted_keys)  # Default last index
    color_range = np.linspace(1, len(sorted_keys) , len(sorted_keys)  - 1) 

    plotter = pv.Plotter()
    # for iso_value in sliced_keys:
    for idx, iso_value in enumerate(sorted_keys):
        points = contours[iso_value]  
        if not points:
            continue
        if random_color:
            color = get_random_color(index=idx)  
        else: 
            color = get_value_color(color_range, idx, colormap='seismic')
        for contour in points:
            contour = np.array(contour) if isinstance(contour, list) else contour
            if len(contour) < 2:
                continue
            lines = pv.lines_from_points(contour)
            if idx >= start and idx <= end:
                plotter.add_mesh(lines, color=color, line_width=line_width)
            else:
                plotter.add_mesh(lines, color='lightgray', line_width=5, opacity=0.2)
    plotter.add_axes()
    # set_view(plotter)
    plotter.show()


def get_value_color(values, value, colormap='seismic'):
    norm = Normalize(vmin=min(values), vmax=max(values))
    cmap = cm.get_cmap(colormap)
    color = cmap(norm(value))[:3]  # Extract RGB values
    return color


def vis_two_set_pcd(pcd1, pcd2, pcd1_size=20, pcd2_size=50, title=None):

    cloud1 = pv.PolyData(pcd1)  
    cloud2 = pv.PolyData(pcd2)  
    plotter = pv.Plotter()

    plotter.add_mesh(cloud1, color='gray', point_size=pcd1_size, render_points_as_spheres=True)
    plotter.add_mesh(cloud2, color='red', point_size=pcd2_size, render_points_as_spheres=True)
    # add unit cube filled color
    cube = pv.Cube(center=(0, 0, 0), x_length=2, y_length=2, z_length=2)
    plotter.add_mesh(cube, color='lightgrey', opacity=0.1, show_edges=True)

    if title is not None:
        plotter.add_text(title, font_size=12)
    plotter.add_axes()  
    plotter.show()


def visualize_subcontour_centers(centers, mesh_fn=None, point_size=15):
    if mesh_fn is not None:
        mesh = pv.read(mesh_fn)

    plotter = pv.Plotter()
    
    for iso_value, center_points in centers.items():
        if not len(center_points):
            continue  # Skip if no centers for the iso-value
        color = get_value_color(list(centers.keys()), iso_value)

        point_cloud = pv.PolyData(np.array(center_points))
        plotter.add_mesh(
            point_cloud,
            color=color,
            point_size=point_size,
            render_points_as_spheres=True,
            # label=f"Iso {iso_value:.2f}"
        )

    if mesh_fn is not None:
        plotter.add_mesh(
        mesh,
        color="ghostwhite",  # Mesh color
        opacity=0.5,  # Transparency level
        show_edges=False,  # Optionally show edges
        label="Mesh",
        smooth_shading=True,
        lighting=True,
        specular=0.8,  # Specular reflection
        specular_power=50,
    )
    plotter.add_axes()
    plotter.show()


def vis_elems(nodes, elems, nodes_ori=None, down=1):
    # ------------------------------------------------ tetra connectivity
    elems = elems[::down]
    if len(elems) == 0:
        raise ValueError('no elements after down-sampling')

    n_tet = len(elems)
    cells = np.hstack(np.c_[np.full((n_tet, 1), 4, np.int64), elems]).ravel()
    celltypes = np.full(n_tet, 10, np.uint8)               # VTK_TETRA (10)
    grid = pv.UnstructuredGrid(cells, celltypes, nodes)

    # ------------------------------------------------ plot
    pl = pv.Plotter()
    pl.add_mesh(grid, color='dodgerblue', opacity=0.2, show_edges=True)

    if nodes_ori is not None:
        ori_grid = pv.UnstructuredGrid(cells, celltypes, nodes_ori)
        pl.add_mesh(ori_grid, color='orangered', opacity=0.8, show_edges=True)

    pl.show_axes()
    pl.show()


def vis_edges(nodes, edges, nodes_ori=None, down=1):
    # ------------------------------------------------ line connectivity
    edges = edges[::down]
    if len(edges) == 0:
        raise ValueError('no edges after down-sampling')

    n_edge = len(edges)
    lines = np.hstack(np.c_[np.full((n_edge, 1), 2, np.int64), edges]).ravel()
    grid = pv.PolyData(nodes, lines=lines)

    # ------------------------------------------------ plot
    pl = pv.Plotter()
    pl.add_mesh(grid, color='dodgerblue', line_width=1.5)

    if nodes_ori is not None:
        ori_lines = np.hstack(np.c_[np.full((n_edge, 1), 2, np.int64), edges]).ravel()
        ori_grid = pv.PolyData(nodes_ori, lines=ori_lines)
        pl.add_mesh(ori_grid, color='orangered', line_width=2.5)

    pl.show_axes()
    pl.show()


def visualize_merged_iso_pts_mul(init_contour_pts, start=None, end=None, random_color=True, 
                                 point_size=8, opacity=1.0, mesh_fn=None, label=None):
    def get_random_color(index, seed=0):
        random.seed(seed + index)
        return (random.random(), random.random(), random.random())

    sorted_keys = sorted(init_contour_pts.keys())

    if start is None:
        start = 0

    if end is None:
        end = len(sorted_keys) - 1

    if len(sorted_keys) > 1:
        color_range = np.linspace(1, len(sorted_keys), len(sorted_keys))
    else:
        color_range = np.array([1])

    plotter = pv.Plotter()

    for idx, iso_value in enumerate(sorted_keys):
        points = init_contour_pts[iso_value]

        if points is None:
            continue

        # --------------------------------------------------------
        # Normalize input to a list of point arrays
        # --------------------------------------------------------
        if isinstance(points, np.ndarray):
            contour_list = [points]
        elif isinstance(points, list):
            if len(points) == 0:
                continue

            # Case: points is directly a list of XYZ points
            arr = np.asarray(points)

            if arr.ndim == 2 and arr.shape[1] == 3:
                contour_list = [arr]
            else:
                # Case: list of multiple contours / point arrays
                contour_list = points
        else:
            continue

        # --------------------------------------------------------
        # Choose color
        # --------------------------------------------------------
        if random_color:
            color = get_random_color(index=idx)
        else:
            color = get_value_color(color_range, idx, colormap='Set3')

        # --------------------------------------------------------
        # Highlight selected iso range, gray out others
        # --------------------------------------------------------
        if start <= idx <= end:
            render_color = color
            render_opacity = opacity
            render_point_size = point_size
        else:
            render_color = 'lightgray'
            render_opacity = 0.15
            render_point_size = max(point_size * 0.6, 2)

        # --------------------------------------------------------
        # Add only points, no connectivity
        # --------------------------------------------------------
        for contour in contour_list:
            contour = np.asarray(contour, dtype=np.float32)

            if contour.ndim != 2 or contour.shape[1] != 3:
                continue

            if contour.shape[0] == 0:
                continue

            point_cloud = pv.PolyData(contour)

            # render_points_as_spheres=True,
            plotter.add_mesh(point_cloud, color=render_color, point_size=render_point_size, opacity=render_opacity, 
                             render_points_as_spheres=True)

    plotter.add_axes()

    if mesh_fn is not None:
        try:
            overlay_mesh = pv.read(mesh_fn)
            plotter.add_mesh(overlay_mesh, color='white', opacity=0.4, show_edges=False)
        except Exception as e:
            print(f"Error loading mesh from {mesh_fn}: {e}")

    if label is not None:
        plotter.add_text(label, position='upper_left', font_size=14, color='black', shadow=True)
    
    plotter.enable_anti_aliasing()
    plotter.show()

"""Export TetGen node coordinates and outward signed-distance gradients."""
from pathlib import Path

import igl
import numpy as np


DATA_DIR = Path(__file__).resolve().parent
MESH_PATH = DATA_DIR / "rect_nozzle.obj"
NODE_PATH = DATA_DIR / "rect_nozzle_var.1.node"
OUTPUT_PATH = DATA_DIR / "rect_nozzle_xyz_normal.xyz"
# OBJ and TetGen coordinates differ slightly through decimal rounding.
SURFACE_REL_TOL = 1e-7
SHOW_VISUALIZATION = True
ARROW_LENGTH = 4.0  # In the same units as the nozzle coordinates.


def visualize_coords_gradients(coords, gradients, vertices, faces, arrow_length=ARROW_LENGTH, show=True, off_screen=False):
    """Display node positions and gradient directions over a transparent nozzle."""
    import pyvista as pv

    mesh_faces = np.column_stack((np.full(len(faces), 3), faces))
    mesh = pv.PolyData(vertices, mesh_faces)
    plotter = pv.Plotter(off_screen=off_screen)
    plotter.set_background("white")
    plotter.add_mesh(mesh, color="lightgray", opacity=0.2)
    plotter.add_points(coords, color="royalblue", point_size=5, render_points_as_spheres=True)
    plotter.add_arrows(coords, gradients, mag=arrow_length, color="crimson")
    plotter.add_text("Nozzle nodes (blue) and distance gradients (red)", color="black", font_size=11)
    plotter.add_axes()
    plotter.view_isometric()
    plotter.reset_camera()
    if show:
        plotter.show()
    return plotter


def main():
    with NODE_PATH.open() as stream:
        for line in stream:
            header = line.partition("#")[0].strip()
            if header:
                break
        else:
            raise ValueError("The node file is empty.")
        count, dimension, attributes, markers = map(int, header.split())
        if dimension != 3:
            raise ValueError("Expected three-dimensional TetGen nodes.")
        data = np.loadtxt(stream, comments="#", ndmin=2)
    if data.shape != (count, 4 + attributes + markers):
        raise ValueError("Node records do not match the TetGen header.")
    ids = data[:, 0]
    if ids[0] not in (0, 1) or not np.array_equal(ids, np.arange(count) + ids[0]):
        raise ValueError("Node IDs must be sequential in file order, starting at zero or one.")
    coords = np.ascontiguousarray(data[:, 1:4], dtype=np.float64)
    if not np.isfinite(coords).all():
        raise ValueError("Node coordinates must be finite.")

    vertices, faces = igl.read_triangle_mesh(str(MESH_PATH))
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("The nozzle mesh is empty.")
    distances, _, closest, pseudonormals = igl.signed_distance(coords, vertices, faces, sign_type=igl.SIGNED_DISTANCE_TYPE_PSEUDONORMAL)
    displacement = coords - closest
    lengths = np.linalg.norm(displacement, axis=1)
    tolerance = SURFACE_REL_TOL * np.linalg.norm(np.ptp(vertices, axis=0))
    away = lengths > tolerance

    # grad(SDF) points outward on both sides; surface nodes use the mesh pseudonormal.
    gradients = np.asarray(pseudonormals, dtype=np.float64).copy()
    gradients[away] = np.sign(distances[away, None]) * displacement[away] / lengths[away, None]
    norms = np.linalg.norm(gradients, axis=1)
    if not np.isfinite(gradients).all() or np.any(norms <= 1e-12):
        raise ValueError("The mesh produced invalid distance gradients.")
    gradients /= norms[:, None]

    np.savetxt(OUTPUT_PATH, np.column_stack((coords, gradients)), fmt="%.9f")
    print(f"Saved {count} rows (x y z gx gy gz) to {OUTPUT_PATH}")
    print(f"Distance gradients: {away.sum()}; surface pseudonormals: {(~away).sum()}")
    if SHOW_VISUALIZATION:
        visualize_coords_gradients(coords, gradients, vertices, faces)


if __name__ == "__main__":
    main()

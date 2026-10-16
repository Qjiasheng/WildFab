import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt5 import QtWidgets, QtCore
import sys
from pathlib import Path
# Enable global theme to allow empty meshes (prevents crashes)
pv.global_theme.allow_empty_mesh = True



def update_plot(self, axis: str, slice_position: float) -> None:
    """
    Draw the scalar slice, its iso-contours *and* the intersection of the mesh
    with that slice plane.

    Parameters
    ----------
    axis : {'x', 'y', 'z'}
        Axis normal of the slice.
    slice_position : float
        World-space position of the slice plane along the chosen axis.
    """
    slice_position = float(slice_position)
    self.slice_positions[axis] = slice_position

    # ----------------------------- housekeeping -----------------------------
    # remove everything that belongs to the *previous* slice on this axis
    for suffix in (
        "_slice",
        "_contour",
        "_mesh_xsec",
    ):
        self.plotter.remove_actor(f"{axis}{suffix}", render=False)

    # --------------------------- compute the slice --------------------------
    if axis == "x":
        normal, origin = (1, 0, 0), (slice_position, 0, 0)
    elif axis == "y":
        normal, origin = (0, 1, 0), (0, slice_position, 0)
    else:  # 'z'
        normal, origin = (0, 0, 1), (0, 0, slice_position)

    slice_ = grid.slice(normal=normal, origin=origin)

    # ----------------------- add the scalar colour map ----------------------
    if slice_.n_points and slice_.active_scalars is not None:
        valid = ~np.isnan(slice_.active_scalars)
        valid_slice = slice_.extract_points(valid, adjacent_cells=False)

        self.plotter.add_mesh(
            valid_slice,
            name=f"{axis}_slice",
            cmap="coolwarm",
            nan_color=(1, 1, 1),
        )

    # ------------------------- spacing contours ----------------------------
    if len(contour_values) and slice_.n_points and slice_.active_scalars is not None:
        contours = slice_.contour(isosurfaces=contour_values)
        if contours.n_points:
            actor = self.plotter.add_mesh(
                contours,
                name=f"{axis}_contour",
                color="black",
                line_width=3,
                render=False,
            )
            actor.SetVisibility(self.contours_visible)

    # ---------------------- NEW : mesh cross-section -----------------------
    if hasattr(self, "mesh") and self.mesh is not None:
        mesh_section = self.mesh.slice(normal=normal, origin=origin)
        if mesh_section.n_points:
            tube = mesh_section.tube(radius=voxel_size * 0.5)   # tweak thickness

            # 3. show it
            actor = self.plotter.add_mesh(
                tube,
                name=f"{axis}_mesh_xsec",
                color="yellow",
                smooth_shading=True,
                render=False,
            )
            actor.SetVisibility(self.mesh_visible)
    self.update_iso_contour(axis, slice_)
    # -----------------------------------------------------------------------
    self.plotter.render()



# PyQt Application setup
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.mesh = None
        self.pcd = None
        self.coords = None
        self.slice_positions = {}
        self.iso_contour_visible = False
        self.setWindowTitle("3D Volume Cross-section Viewer")

        # Create the PyVista plotter
        self.plotter = QtInteractor(self)
        self.setCentralWidget(self.plotter.interactor)

        # Create a dock widget for the sidebar
        dock = QtWidgets.QDockWidget("Slice & Bounding Box Controls", self)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock)

        # Create a widget to house the controls
        self.controls = QtWidgets.QWidget()
        dock.setWidget(self.controls)
        layout = QtWidgets.QVBoxLayout()

        # Add group box for slice controls
        slice_group_box = QtWidgets.QGroupBox("Slice Controls")
        slice_layout = QtWidgets.QVBoxLayout()

        # Create slice sliders for X, Y, Z cross-section adjustment
        self.x_slice_label = QtWidgets.QLabel("X Slice: 0.00")
        self.x_slider = self.create_slider('X Slice', N, slice_layout)
        self.x_slider.valueChanged.connect(self.on_x_slider_moved)

        self.y_slice_label = QtWidgets.QLabel("Y Slice: 0.00")
        self.y_slider = self.create_slider('Y Slice', N, slice_layout)
        self.y_slider.valueChanged.connect(self.on_y_slider_moved)

        self.z_slice_label = QtWidgets.QLabel("Z Slice: 0.00")
        self.z_slider = self.create_slider('Z Slice', N, slice_layout)
        self.z_slider.valueChanged.connect(self.on_z_slider_moved)

        slice_group_box.setLayout(slice_layout)
        layout.addWidget(slice_group_box)

        # Add group box for bounding box controls
        bbx_group_box = QtWidgets.QGroupBox("Bounding Box Controls")
        bbx_layout = QtWidgets.QVBoxLayout()

        # Create bounding box sliders for X, Y, Z axes
        self.x1_slider, self.x1_label = self.create_bbx_slider('X1', N, bbx_layout)
        self.x1_slider.valueChanged.connect(self.on_bbx_slider_moved)

        self.x2_slider, self.x2_label = self.create_bbx_slider('X2', N, bbx_layout)
        self.x2_slider.valueChanged.connect(self.on_bbx_slider_moved)

        self.y1_slider, self.y1_label = self.create_bbx_slider('Y1', N, bbx_layout)
        self.y1_slider.valueChanged.connect(self.on_bbx_slider_moved)

        self.y2_slider, self.y2_label = self.create_bbx_slider('Y2', N, bbx_layout)
        self.y2_slider.valueChanged.connect(self.on_bbx_slider_moved)

        self.z1_slider, self.z1_label = self.create_bbx_slider('Z1', N, bbx_layout)
        self.z1_slider.valueChanged.connect(self.on_bbx_slider_moved)

        self.z2_slider, self.z2_label = self.create_bbx_slider('Z2', N, bbx_layout)
        self.z2_slider.valueChanged.connect(self.on_bbx_slider_moved)

        bbx_group_box.setLayout(bbx_layout)
        layout.addWidget(bbx_group_box)

        # Add a toggle button for point cloud visibility
        self.toggle_pcd_button = QtWidgets.QPushButton("Toggle Point Cloud")
        layout.addWidget(self.toggle_pcd_button)
        self.toggle_pcd_button.clicked.connect(self.toggle_pcd_visibility)

        # Add a toggle button for mesh visibility
        self.toggle_mesh_button = QtWidgets.QPushButton("Toggle Mesh")
        layout.addWidget(self.toggle_mesh_button)
        self.toggle_mesh_button.clicked.connect(self.toggle_mesh_visibility)

        self.toggle_contours_button = QtWidgets.QPushButton("Toggle Spacing Contours")
        layout.addWidget(self.toggle_contours_button)
        self.toggle_contours_button.clicked.connect(self.toggle_contours_visibility)

        layout.addWidget(QtWidgets.QLabel("Iso Contour Value"))
        self.iso_value_input = QtWidgets.QDoubleSpinBox()
        self.iso_value_input.setDecimals(6)
        self.iso_value_input.setRange(-1e12, 1e12)
        self.iso_value_input.setSingleStep(0.01)
        self.iso_value_input.setValue(0.5)
        self.iso_value_input.setKeyboardTracking(False)
        layout.addWidget(self.iso_value_input)
        self.iso_value_input.valueChanged.connect(self.refresh_iso_contours)

        self.toggle_iso_contour_button = QtWidgets.QPushButton("Show Iso Contour")
        self.toggle_iso_contour_button.setCheckable(True)
        layout.addWidget(self.toggle_iso_contour_button)
        self.toggle_iso_contour_button.toggled.connect(self.toggle_iso_contour_visibility)

        self.controls.setLayout(layout)

        self.mesh_visible = True
        self.contours_visible = False
        

        # Initialize the plot with the volume and the first X slice
        self.add_volume_with_gpu()

        # Show the first slice
        # self.update_x_slice(N // 2)
        # self.update_y_slice(N // 2)

        # Show the coordinate axes
        self.plotter.show_axes()

        # Mesh and point cloud are optional; the volume can be viewed alone.
        self.pcd_visible = bool(pcd_fn) and Path(pcd_fn).is_file()
        self.mesh_visible = bool(mesh_fn) and Path(mesh_fn).is_file()
        self.toggle_pcd_button.setEnabled(self.pcd_visible)
        self.toggle_mesh_button.setEnabled(self.mesh_visible)
        if self.pcd_visible:
            self.load_and_add_pcd(hide=False)
        if self.mesh_visible:
            self.load_and_add_mesh(hide=False)

        self.update_y_slice(N // 2)

        # Initial state
        self.streamlines_visible = False
        # self.mesh = None
        self.gradients = None
        self.sing_centers = None
        self.sing_types = None


        # Initialize the plot with the visualization mesh
        self.plotter.show_axes()


    def create_slider(self, label_text, max_value, layout):
        """Helper to create a slider with a label."""
        label = QtWidgets.QLabel(label_text)
        layout.addWidget(label)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(max_value - 1)
        slider.setValue(max_value // 2)
        layout.addWidget(slider)
        return slider

    def create_bbx_slider(self, label_text, max_value, layout):
        """Helper to create a bounding box slider with a label and a value display."""
        label = QtWidgets.QLabel(f'{label_text}: 0.00')
        layout.addWidget(label)
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(max_value - 1)
        slider.setValue(max_value // 2)
        layout.addWidget(slider)
        return slider, label

    def add_volume_with_gpu(self):
        """Add the volume to the plot using GPU-based volume rendering."""
        volume_mapper = 'gpu'  # Set GPU-based volume rendering
        # self.plotter.add_volume(
        #     grid, 
        #     scalars="SDF", 
        #     cmap="coolwarm", 
        #     # opacity="sigmoid_1",
        #     opacity="foreground",  # Use simpler opacity mapping for performance
        #     n_colors=256, 
        #     diffuse=0.0, 
        #     specular=0.0, 
        #     mapper=volume_mapper  # Use GPU-based mapper
        # )


    def on_x_slider_moved(self):
        """Update the X cross-section when the slider is moved."""
        x_value = self.x_slider.value()
        slice_position = voxel_grid_origin[0] + voxel_size * x_value
        self.x_slice_label.setText(f'X Slice: {slice_position:.2f}')
        self.update_x_slice(x_value)

    def on_y_slider_moved(self):
        """Update the Y cross-section when the slider is moved."""
        y_value = self.y_slider.value()
        slice_position = voxel_grid_origin[1] + voxel_size * y_value
        self.y_slice_label.setText(f'Y Slice: {slice_position:.2f}')
        self.update_y_slice(y_value)

    def on_z_slider_moved(self):
        """Update the Z cross-section when the slider is moved."""
        z_value = self.z_slider.value()
        slice_position = voxel_grid_origin[2] + voxel_size * z_value
        self.z_slice_label.setText(f'Z Slice: {slice_position:.2f}')
        self.update_z_slice(z_value)

    def update_x_slice(self, value):
        """Update the plot with the X slice."""
        slice_position = voxel_grid_origin[0] + voxel_size * value
        update_plot(self, 'x', slice_position)

    def update_y_slice(self, value):
        """Update the plot with the Y slice."""
        slice_position = voxel_grid_origin[1] + voxel_size * value
        update_plot(self, 'y', slice_position)

    def update_z_slice(self, value):
        """Update the plot with the Z slice."""
        slice_position = voxel_grid_origin[2] + voxel_size * value
        update_plot(self, 'z', slice_position)

    def load_and_add_pcd(self, hide=False):
        """Load the point cloud from a .xyz file and add it to the plot."""
        # NOTE following same normalize params as in training
        pcd_data = np.loadtxt(pcd_fn)  # Assuming .xyz format: [x, y, z]
        coords = pcd_data[:, :3]

        if scale_coords:
            coords -= np.mean(coords, axis=0, keepdims=True)
            coord_max = np.amax(coords)
            coord_min = np.amin(coords)

            coords = (coords - coord_min) / (coord_max - coord_min)
            coords -= 0.5
            coords *= 1.95  # Scale so that the surface doesn't touch the box boundary

        self.coords = coords  # Store the coordinates for easy access
        # Initial update of the point cloud color based on the bounding box
        self.update_pcd_color()

        if hide:
            self.plotter.remove_actor("point_cloud")


    def load_and_add_mesh(self, hide=False):
        """Load the mesh from a file and add it to the plot."""
        global mesh_fn
        mesh = pv.read(mesh_fn)  # Load the mesh file
        # smooth mesh
        # mesh = mesh.smooth(n_iter=100, relaxation_factor=0.5)

        self.mesh = mesh  # Store the mesh for toggling visibility

        # Add the mesh to the plot with initial visibility
        self.plotter.add_mesh(
            mesh,
            name="mesh",
            color="ghostwhite",  # Mesh color
            opacity=0.4,  # Transparency level
            show_edges=False,
            smooth_shading=True,
            lighting=True,
            specular=0.8,  # Specular reflection
            specular_power=50,
        )
        if hide:
            self.plotter.remove_actor("mesh") 


    def update_pcd_color(self):
        """Update the point cloud color based on the bounding box."""
        if self.coords is None:
            return
        # Get bounding box limits from sliders
        x1 = voxel_grid_origin[0] + voxel_size * self.x1_slider.value()
        x2 = voxel_grid_origin[0] + voxel_size * self.x2_slider.value()
        y1 = voxel_grid_origin[1] + voxel_size * self.y1_slider.value()
        y2 = voxel_grid_origin[1] + voxel_size * self.y2_slider.value()
        z1 = voxel_grid_origin[2] + voxel_size * self.z1_slider.value()
        z2 = voxel_grid_origin[2] + voxel_size * self.z2_slider.value()

        # Update slider labels with the current positions (formatted to 2 decimal places)
        self.x1_label.setText(f'X1: {x1:.2f}')
        self.x2_label.setText(f'X2: {x2:.2f}')
        self.y1_label.setText(f'Y1: {y1:.2f}')
        self.y2_label.setText(f'Y2: {y2:.2f}')
        self.z1_label.setText(f'Z1: {z1:.2f}')
        self.z2_label.setText(f'Z2: {z2:.2f}')

        # Update point cloud colors based on bounding box
        coords = self.coords
        colors = np.zeros((coords.shape[0], 3))  # Prepare an array for RGB colors

        # Points within the bounding box will be red, others will be gray
        inside_bbx = (
            (coords[:, 0] >= x1) & (coords[:, 0] <= x2) &
            (coords[:, 1] >= y1) & (coords[:, 1] <= y2) &
            (coords[:, 2] >= z1) & (coords[:, 2] <= z2)
        )
        colors[inside_bbx] = [1, 0, 0]  # Red for points inside the bounding box
        colors[~inside_bbx] = [0.9, 0.9, 0.9]  # Gray for points outside the bounding box

        # Update the point cloud with the new colors
        self.pcd = pv.PolyData(coords)
        self.pcd['colors'] = colors  # Add the color array to the PolyData

        # Re-add the point cloud to the plot with updated colors
        self.plotter.add_mesh(self.pcd, scalars="colors", rgb=True, point_size=5.0, name="point_cloud", 
                              reset_camera=False, render_points_as_spheres=True)
        self.plotter.render()

    def on_bbx_slider_moved(self):
        """Update the bounding box and point cloud colors when a bounding box slider is moved."""
        self.update_pcd_color()

    def toggle_pcd_visibility(self):
        """Toggle the visibility of the point cloud."""
        if self.pcd is None:
            return
        self.pcd_visible = not self.pcd_visible
        if self.pcd_visible:
            # self.plotter.add_mesh(self.pcd, color="white", point_size=3.0, name="point_cloud")
            self.plotter.add_mesh(self.pcd, color='gray', point_size=3.0, name="point_cloud")
        else:
            self.plotter.remove_actor("point_cloud")
        self.plotter.render()

        
    def toggle_contours_visibility(self):
        """Toggle spacing contours on all displayed slices independently of the mesh."""
        self.contours_visible = not self.contours_visible
        for axis in ("x", "y", "z"):
            actor = self.plotter.actors.get(f"{axis}_contour")
            if actor is not None:
                actor.SetVisibility(self.contours_visible)
        self.plotter.render()

    def update_iso_contour(self, axis, slice_):
        """Draw the selected scalar level independently of the other contours."""
        self.plotter.remove_actor(f"{axis}_iso_contour", render=False)
        if self.iso_contour_visible and slice_.n_points and slice_.active_scalars is not None:
            contour = slice_.contour(isosurfaces=[self.iso_value_input.value()])
            if contour.n_points:
                tube = contour.tube(radius=voxel_size * 0.5)
                self.plotter.add_mesh(
                    tube,
                    name=f"{axis}_iso_contour",
                    color="black",
                    smooth_shading=True,
                    reset_camera=False,
                    render=False,
                )

    def refresh_iso_contours(self):
        """Refresh only the iso overlay on slices already displayed."""
        for axis, position in self.slice_positions.items():
            origin = [0, 0, 0]
            origin['xyz'.index(axis)] = position
            slice_ = grid.slice(normal=axis, origin=origin)
            self.update_iso_contour(axis, slice_)
        self.plotter.render()

    def toggle_iso_contour_visibility(self, visible):
        self.iso_contour_visible = visible
        self.toggle_iso_contour_button.setText("Hide Iso Contour" if visible else "Show Iso Contour")
        self.refresh_iso_contours()

    def toggle_mesh_visibility(self):
        """Toggle the mesh and all slice-intersection tubes together."""
        if self.mesh is None:
            return
        self.mesh_visible = not self.mesh_visible
        if self.mesh_visible:
            self.plotter.add_mesh(
            self.mesh,
            name="mesh",
            color="ghostwhite",  # Mesh color
            opacity=0.5,  # Transparency level
            show_edges=False,
            smooth_shading=True,
            lighting=True,
            specular=0.8,  # Specular reflection
            specular_power=50,
        )
        else:
            self.plotter.remove_actor("mesh", render=False)
        for axis in ("x", "y", "z"):
            actor = self.plotter.actors.get(f"{axis}_mesh_xsec")
            if actor is not None:
                actor.SetVisibility(self.mesh_visible)
        self.plotter.render()



# PyQt Application Main Loop
if __name__ == "__main__":
    # Global contour levels: edit start, stop, and spacing here (or supply a list).
    contour_values = np.arange(-1.5, 1.5, 0.04)

    # Load the SDF data from a .npz file
    global pcd_fn
    global mesh_fn
    global scale_coords
    scale_coords = False

    exp = 'klein'
    sdf_volume_fn = f'logs/udf/{exp}/nn_udf_volume.npz'
    # sdf_volume_fn = f'logs/udf/{exp}/nn_udf_gradnorm_volume.npz'

    # sdf_volume_fn = f'logs/udf/{exp}/wn_volume.npz'
    # sdf_volume_fn = f'logs/udf/{exp}/wn_gradnorm_volume.npz'

    pcd_fn = f'udf_data/logs/{exp}/mesh_data.xyz'
    mesh_fn = f'udf_data/logs/{exp}/mesh_data.obj'



    # check tv wn and wn gradnorm
    # exp = 'klein_single_shell'
    # exp = 'tree_lattice'
    # sdf_volume_fn = f'logs/collision/{exp}/tv_wn_volume.npz'
    # sdf_volume_fn = f'logs/collision/{exp}/tv_wn_gradnorm_volume.npz'
    # pcd_fn = f'logs/collision/{exp}/tv_waypt_coords.xyz'

    data = np.load(sdf_volume_fn)
    sdf_values = data['sdf']
    sdf_values = np.abs(sdf_values) 
    voxel_grid_origin = data['voxel_grid_origin']
    voxel_size = data['voxel_size']
    N = sdf_values.shape[0]

    # Create a 3D grid of voxel positions
    x = np.linspace(voxel_grid_origin[0], voxel_grid_origin[0] + voxel_size * (N - 1), N)
    y = np.linspace(voxel_grid_origin[1], voxel_grid_origin[1] + voxel_size * (N - 1), N)
    z = np.linspace(voxel_grid_origin[2], voxel_grid_origin[2] + voxel_size * (N - 1), N)
    Z, Y, X = np.meshgrid(x, y, z, indexing="ij")
    points = np.c_[X.flatten(), Y.flatten(), Z.flatten()]

    # Create structured grid for PyVista
    grid = pv.StructuredGrid()
    grid.points = points
    grid.dimensions = [N, N, N]
    grid["SDF"] = sdf_values.flatten(order="F")  # Fortran-style flattening

    # Setup the PyQt application
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()

    # Start the event loop
    sys.exit(app.exec_())


# cmd
# python viewers/slice_view_volume.py
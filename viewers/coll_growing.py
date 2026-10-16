# -*- coding: utf-8 -*-
"""
Single-pose collision timeline for quat_collision_test.npz.
Collision samples are compact: row k belongs to the kth True waypoint mask entry.
"""

import argparse, os, sys
from pathlib import Path
import yaml
import numpy as np
import pyvista as pv
import matplotlib
import vtk
from PyQt5 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor

# --------------------------------------------------------------------------- PyVista
pv.set_plot_theme('document')
pv.global_theme.allow_empty_mesh = True          # silence empty-mesh warnings


# --------------------------------------------------------------------------- helpers
def coolwarm_rgb(n: int) -> np.ndarray:
    lut = matplotlib.colormaps['coolwarm'](np.linspace(0, 1, n))[:, :3]
    return (lut * 255).astype(np.uint8)


def q_to_R(q):
    """Quaternion (w,x,y,z) → 3×3 rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]],
                    np.float32)


def transform(pts, R, t):
    return pts @ R.T + t


def load_collision_data(npz_path):
    with np.load(npz_path, allow_pickle=False) as data:
        wp = data['waypts'].astype(np.float32)
        quats = data['quaternions'].astype(np.float32)
        mask = data['waypt_coll_mask'].astype(bool)
        points = data['coll_transed_nozzle_pcd'].astype(np.float32)
        grads = data['coll_transed_nozzle_grads'].astype(np.float32)
    if len(wp) == 0 or quats.shape != (len(wp), 4) or mask.shape != (len(wp),):
        raise ValueError('Expected nonempty waypts (N,3), quaternions (N,4), and waypt_coll_mask (N,)')
    if points.ndim != 3 or points.shape != grads.shape or points.shape[0] != mask.sum() or points.shape[-1] != 3:
        raise ValueError('Collision points and gradients must have shape (waypt_coll_mask.sum(), M, 3)')
    compact_idx = np.full(len(wp), -1, dtype=np.int64)
    compact_idx[mask] = np.arange(mask.sum())
    return wp, quats, mask, points, grads, compact_idx


def collision_frame(points, grads, compact_index):
    if compact_index < 0:
        return np.empty((0, 3), np.float32), np.empty((0, 3), np.float32)
    pts, vectors = points[compact_index], grads[compact_index]
    valid = np.isfinite(pts).all(axis=1) & np.isfinite(vectors).all(axis=1)
    valid &= np.linalg.norm(vectors, axis=1) > 1e-6
    return pts[valid], vectors[valid]


# --------------------------------------------------------------------------- tiny axes actor
def axes_actor(scale=1.0):
    """Return a label-free RGB triad axes actor of the requested length."""
    ax = vtk.vtkAxesActor()
    ax.SetTotalLength(scale, scale, scale)
    ax.SetCylinderRadius(scale * .1)
    ax.AxisLabelsOff()
    return ax


def vtk_mat_from_RT(R: np.ndarray, t: np.ndarray):
    """NumPy R(3×3), t(3,) → vtkMatrix4x4."""
    M = vtk.vtkMatrix4x4()
    for r in range(3):
        for c in range(3):
            M.SetElement(r, c, float(R[r, c]))
        M.SetElement(r, 3, float(t[r]))
    M.SetElement(3, 3, 1.0)
    return M


# --------------------------------------------------------------------------- timeline widget
class CollisionTimeline(QtWidgets.QWidget):
    def __init__(self, mask):
        super().__init__()
        self.N = len(mask)
        self.idx = 0
        self.setMinimumHeight(26)
        edges = np.diff(np.r_[False, mask, False].astype(np.int8))
        self.runs = list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))

    def set_index(self, i):
        self.idx = int(i)
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QtGui.QColor('whitesmoke'))
        # Keep isolated collisions visible when the timeline is compressed.
        for start, stop in self.runs:
            left = int(start * w / self.N)
            right = int(np.ceil(stop * w / self.N))
            width = min(w, max(7, right - left))
            left = max(0, min(w - width, (left + right - width) // 2))
            p.fillRect(left, 0, width, h, QtGui.QColor('red'))
        x = int(self.idx / max(1, self.N - 1) * max(0, w - 1))
        pen = QtGui.QPen(QtGui.QColor('black'))
        pen.setWidth(3)
        p.setPen(pen)
        p.drawLine(x, 0, x, h)
        p.end()


# --------------------------------------------------------------------------- main window
class Viewer(QtWidgets.QMainWindow):
    def __init__(self, npz_path, obj_mesh=None, nozzle_mesh=None, object_scale=1.0):
        super().__init__()
        self.setWindowTitle('Collision viewer – ' + os.path.basename(npz_path))

        # -------------------- load data
        self.wp, self.quats, self.wp_mask, self.coll_points, self.coll_grads, self.compact_idx = load_collision_data(npz_path)
        self.collision_indices = np.flatnonzero(self.wp_mask)

        self.N = self.wp.shape[0]
        self.col_path = coolwarm_rgb(self.N)

        # -------------------- 3-D view
        self.plot = QtInteractor(self)
        self.setCentralWidget(self.plot)
        self.plot.set_background('white')
        self.plot.add_axes(line_width=1)

        # ---- way-point path
        self.path_pd = pv.PolyData(self.wp[:1]); self.path_pd['rgb'] = self.col_path[:1]
        self.act_path = self.plot.add_points(self.path_pd, rgb=True,
                                             render_points_as_spheres=True, point_size=10)

        # ---- colliding nozzle points
        self.act_points = self.plot.add_points(pv.PolyData(), color='red',
                                               point_size=15, render_points_as_spheres=True)
        self.act_arrows = self.plot.add_mesh(pv.PolyData(), color='orange',
                                             render_lines_as_tubes=True, line_width=3)
        self.arrow_scale = 0.08

        self.vtk_axes = axes_actor(scale=0.1)
        self.plot.add_actor(self.vtk_axes)

        self.act_mesh = None
        if nozzle_mesh and os.path.exists(nozzle_mesh):
            mesh = pv.read(nozzle_mesh)
            mesh.points = mesh.points / object_scale
            self.act_mesh = self.plot.add_mesh(mesh, color='lightgray', opacity=.5)

        self.act_obj = None
        if obj_mesh and os.path.exists(obj_mesh):
            try:
                self.act_obj = self.plot.add_mesh(pv.read(obj_mesh), color='silver', opacity=.15)
            except Exception as e:
                print('cannot read object mesh:', e)

        self.act_wp = self.plot.add_mesh(pv.Sphere(radius=0.01), color='yellow')

        # -------------------- timeline & controls
        vlay = QtWidgets.QVBoxLayout()
        self.timeline = CollisionTimeline(self.wp_mask)
        vlay.addWidget(self.timeline)
        self.status_label = QtWidgets.QLabel()
        vlay.addWidget(self.status_label)

        hl = QtWidgets.QHBoxLayout()
        self.btn_play = QtWidgets.QPushButton('Play'); self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self.toggle_play); hl.addWidget(self.btn_play)
        self.btn_back = QtWidgets.QPushButton('◀'); self.btn_back.clicked.connect(
            lambda: self.btn_step_back()); hl.addWidget(self.btn_back)
        self.btn_fwd = QtWidgets.QPushButton('▶'); self.btn_fwd.clicked.connect(
            lambda: self.btn_step_fwd()); hl.addWidget(self.btn_fwd)
        self.spin_step = QtWidgets.QSpinBox(); self.spin_step.setRange(1, self.N); self.spin_step.setValue(1)
        hl.addWidget(QtWidgets.QLabel('Step')); hl.addWidget(self.spin_step)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.slider.setRange(0, self.N - 1)
        self.slider.valueChanged.connect(self.show_idx); hl.addWidget(self.slider)
        vlay.addLayout(hl)
        collision_nav = QtWidgets.QHBoxLayout()
        for label, direction in [('Previous collision', -1), ('Next collision', 1)]:
            button = QtWidgets.QPushButton(label)
            button.setEnabled(bool(len(self.collision_indices)))
            button.clicked.connect(lambda checked=False, d=direction: self.jump_collision(d))
            collision_nav.addWidget(button)
        vlay.addLayout(collision_nav)

        w_bottom = QtWidgets.QWidget(); w_bottom.setLayout(vlay)
        dock = QtWidgets.QDockWidget(); dock.setWidget(w_bottom)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock)

        # -------------------- display toggles (left)
        grid, r = QtWidgets.QGridLayout(), 0

        def tgl(text, actor):
            nonlocal r
            btn = QtWidgets.QPushButton(text); btn.setCheckable(True); btn.setChecked(True)
            def set_visibility(on):
                actor.SetVisibility(on)
                self.plot.render()
            btn.toggled.connect(set_visibility)
            grid.addWidget(btn, r, 0); r += 1

        tgl('Collision points', self.act_points)
        tgl('Collision gradient arrows', self.act_arrows)
        if self.act_mesh is not None:
            tgl('Nozzle mesh', self.act_mesh)
        if self.act_obj is not None:
            tgl('Object mesh', self.act_obj)
        tgl('Frame axes', self.vtk_axes)
        tgl('Printed waypoints', self.act_path)
        grid.setRowStretch(r, 1)
        w_left = QtWidgets.QWidget(); w_left.setLayout(grid)
        dockL = QtWidgets.QDockWidget('Display'); dockL.setWidget(w_left)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dockL)

        # -------------------- timer & shortcuts
        self.timer = QtCore.QTimer(); self.timer.setInterval(33); self.timer.timeout.connect(self.timer_step)
        QtWidgets.QShortcut(QtGui.QKeySequence('Space'), self, activated=self.btn_play.toggle)
        QtWidgets.QShortcut(QtGui.QKeySequence('Left'), self, activated=self.btn_step_back)
        QtWidgets.QShortcut(QtGui.QKeySequence('Right'), self, activated=self.btn_step_fwd)
        QtWidgets.QShortcut(QtGui.QKeySequence('Shift+Left'), self,
                            activated=lambda: self.btn_step_back(mult=10))
        QtWidgets.QShortcut(QtGui.QKeySequence('Shift+Right'), self,
                            activated=lambda: self.btn_step_fwd(mult=10))

        # -------------------- initialise
        self.show_idx(0)
        self.plot.reset_camera()

    # ----------------------------------------------------------------- navigation
    def current_step(self, mult=1):
        return mult * self.spin_step.value()

    def step_back(self, mult=1):
        self.slider.setValue((self.slider.value() - self.current_step(mult)) % self.N)

    def step_fwd(self, mult=1):
        self.slider.setValue((self.slider.value() + self.current_step(mult)) % self.N)

    def btn_step_back(self, mult=1):
        if self.timer.isActive(): self.btn_play.toggle()
        self.step_back(mult)

    def btn_step_fwd(self, mult=1):
        if self.timer.isActive(): self.btn_play.toggle()
        self.step_fwd(mult)

    def jump_collision(self, direction):
        self.btn_play.setChecked(False)
        current = self.slider.value()
        if direction > 0:
            k = np.searchsorted(self.collision_indices, current, side='right')
        else:
            k = np.searchsorted(self.collision_indices, current, side='left') - 1
        self.slider.setValue(int(self.collision_indices[k % len(self.collision_indices)]))

    # ----------------------------------------------------------------- play / pause
    def toggle_play(self, playing: bool):
        self.btn_play.setText('Pause' if playing else 'Play')
        (self.timer.start if playing else self.timer.stop)()

    def timer_step(self):
        self.step_fwd()

    # ----------------------------------------------------------------- per-frame update
    @QtCore.pyqtSlot(int)
    def show_idx(self, i: int):
        i = int(i)
        # ---- path & current wp
        self.path_pd.points = self.wp[:i + 1]
        self.path_pd['rgb'] = self.col_path[:i + 1]
        self.path_pd.verts = np.c_[np.ones(i + 1, np.int64), np.arange(i + 1)]
        self.path_pd.Modified()
        self.act_wp.SetPosition(*self.wp[i])

        R = q_to_R(self.quats[i])
        t = self.wp[i]
        matrix = vtk_mat_from_RT(R, t)
        # Mesh is local/scaled; saved collision samples already use world coordinates.
        if self.act_mesh is not None:
            self.act_mesh.SetUserMatrix(matrix)
        self.vtk_axes.SetUserMatrix(matrix)

        pts, grads = collision_frame(self.coll_points, self.coll_grads, self.compact_idx[i])
        self.act_points.GetMapper().SetInputData(pv.PolyData(pts))
        if len(pts):
            # Limit glyph count, keeping all collision points visible.
            stride = max(1, int(np.ceil(len(pts) / 3000)))
            src = pv.PolyData(pts[::stride])
            src['g'] = grads[::stride]
            glyphs = src.glyph(orient='g', scale=False, factor=self.arrow_scale)
        else:
            glyphs = pv.PolyData()
        self.act_arrows.GetMapper().SetInputData(glyphs)
        state = 'COLLISION' if self.wp_mask[i] else 'clear'
        self.status_label.setText(f'Waypoint {i} / {self.N - 1} — {state} | '
                                 f'{len(self.collision_indices)} colliding waypoints (red)')

        # ---- timeline & render
        self.timeline.set_index(i)
        self.plot.render()


# --------------------------------------------------------------------------- entry point
def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('npz_path', nargs='?', help='Saved quat_collision_test.npz')
    parser.add_argument('--config_path', default=str(repo / 'configs/collision_config.yaml'))
    parser.add_argument('--experiment_name', default='tree_lattice')
    parser.add_argument('--object-mesh')
    parser.add_argument('--nozzle-mesh')
    parser.add_argument('--object-scale', type=float, help='Divide nozzle mesh vertices by this scale; 1 for an already scaled mesh')
    args = parser.parse_args()
    with open(args.config_path) as stream:
        config = yaml.safe_load(stream)[args.experiment_name]
    dataset = config['dataset']

    def resolve(path):
        if not path:
            return None
        path = Path(path)
        return str(path if path.is_absolute() else repo / path)

    npz_path = args.npz_path or os.path.join(dataset['save_path'], args.experiment_name, 'quat_collision_test.npz')
    obj_mesh = args.object_mesh or dataset.get('mesh_path')
    nozzle_mesh = args.nozzle_mesh or dataset.get('nozzle_mesh_path')
    object_scale = args.object_scale if args.object_scale is not None else config['parameters']['object_scale']
    app = QtWidgets.QApplication(sys.argv)
    win = Viewer(resolve(npz_path), resolve(obj_mesh), resolve(nozzle_mesh), object_scale)
    win.resize(1500, 900)
    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()


# cmd
# python viewers/coll_growing.py --experiment_name=klein_single_shell --config_path=configs/collision_config.yaml

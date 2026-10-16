#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive waypoint viewer with two colour modes:
    • level  – each level label has its own colour            (default)
    • type   – shell / infill shown in blue / orange
Left-dock button “Colour mode” toggles the scheme on the fly.
"""

import sys, os, numpy as np, pyvista as pv, matplotlib
from PyQt5 import QtWidgets, QtCore, QtGui
from pyvistaqt import QtInteractor

# ------------------------------------------------------------------ colour helpers
def level_to_rgb(level_vals):
    """Integer level labels → RGB uint8 colours."""
    uniq = np.unique(level_vals)
    lut  = matplotlib.colormaps['coolwarm'].resampled(len(uniq))(np.arange(len(uniq)))[:, :3]
    lut  = (lut * 255).astype(np.uint8)
    tbl  = {lvl: lut[i] for i, lvl in enumerate(uniq)}
    return np.vstack([tbl[l] for l in level_vals])

TAG_COLOURS = np.array([[ 40, 170, 255],   # shell  (blue)
                        [255, 140,  40],   # infill (orange)
                        [  0,   255,   0]    # outershell green
                        ],  
                       dtype=np.uint8)
def tag_to_rgb(tags):
    return TAG_COLOURS[tags.astype(int)]

# ------------------------------------------------------------------ I/O
def load_waypoints(path):
    """
    Returns
        coords : (N,3) float32
        levels : (N,) int
        tags   : (N,) int   (shell=0, infill=1)
    The file is assumed to have floats except the last two integer columns.
    """
    raw = np.loadtxt(path, dtype=float)
    if raw.ndim == 1:
        raw = raw[None, :]
    coords = raw[:, :3].astype(np.float32)
    levels = raw[:, 10].astype(np.int32)
    tags   = raw[:, 11].astype(np.int32)
    print(f'Loaded {len(coords)} waypoints from {path}')
    return coords, levels, tags

# ------------------------------------------------------------------ tiny timeline widget
class TimelineWidget(QtWidgets.QWidget):
    def __init__(self, colours, parent=None):
        super().__init__(parent)
        self.setFixedHeight(14)
        self.set_colours(colours)
        self.idx = 0

    def set_colours(self, cols):
        self.N = len(cols)
        img = cols.astype(np.uint8)[None, :, :]
        self.qimg = QtGui.QImage(img.data, self.N, 1,
                                 QtGui.QImage.Format_RGB888).copy()
        self.update()

    def set_index(self, i):
        self.idx = int(i); self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        w, h = self.width(), self.height()
        p.drawImage(QtCore.QRect(0, 0, w, h), self.qimg)
        x = int(self.idx / max(1, self.N - 1) * w)
        pen = QtGui.QPen(QtGui.QColor('white')); pen.setWidth(4)
        p.setPen(pen); p.drawLine(x, 0, x, h); p.end()

# ------------------------------------------------------------------ main window
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, wp_file, mesh_file=None):
        super().__init__()
        self.setWindowTitle(f'Waypoint viewer – {os.path.basename(wp_file)}')

        # ----------------------- data
        coords, levels, tags = load_waypoints(wp_file)
        self.coords, self.levels, self.tags = coords, levels, tags
        self.N = len(coords)
        self.level_cols = level_to_rgb(levels)
        self.tag_cols   = tag_to_rgb(tags)
        self.colours = self.level_cols.copy()
        self.colour_mode = 'level'
        self.last_idx = 0

        # ----------------------- 3-D view
        self.plotter = QtInteractor(self)
        self.setCentralWidget(self.plotter)
        self.plotter.set_background('white')
        self.plotter.add_axes(line_width=2)

        self.path_poly = pv.PolyData(coords[:1])
        self.path_poly['rgb'] = self.colours[:1]
        self.path_actor = self.plotter.add_points(self.path_poly,
                                                  render_points_as_spheres=True,
                                                  point_size=12, scalars='rgb', rgb=True)

        self.cur_sphere = self.plotter.add_mesh(pv.Sphere(radius=0.02),color='yellow')
        self.cur_sphere.SetVisibility(False)

        # optional mesh
        self.mesh_actor = None
        if mesh_file and os.path.exists(mesh_file):
            try:
                self.mesh_actor = self.plotter.add_mesh(
                    pv.read(mesh_file), color='lightgray', opacity=0.1,)
            except Exception as e:
                print('Cannot read mesh:', e)

        # ----------------------- timeline + controls
        vb_btm = QtWidgets.QVBoxLayout()
        self.timeline = TimelineWidget(self.colours)
        vb_btm.addWidget(self.timeline)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, self.N - 1)
        self.slider.valueChanged.connect(self.update_frame)
        self.step_spin = QtWidgets.QSpinBox(); self.step_spin.setRange(1, self.N); self.step_spin.setValue(1)
        btn_back = QtWidgets.QPushButton('◀'); btn_back.clicked.connect(self.step_back)
        btn_fwd  = QtWidgets.QPushButton('▶'); btn_fwd .clicked.connect(self.step_fwd)
        self.btn_play = QtWidgets.QPushButton('Play'); self.btn_play.setCheckable(True)
        self.btn_play.toggled.connect(self.toggle_play)
        hb_ctrl = QtWidgets.QHBoxLayout()
        for w in (self.btn_play, btn_back, btn_fwd,
                  QtWidgets.QLabel('step'), self.step_spin, self.slider):
            hb_ctrl.addWidget(w)
        vb_btm.addLayout(hb_ctrl)
        dock_btm = QtWidgets.QDockWidget(); w_btm = QtWidgets.QWidget(); w_btm.setLayout(vb_btm)
        dock_btm.setWidget(w_btm); self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, dock_btm)

        # ----------------------- left dock (display)
        vb_l = QtWidgets.QVBoxLayout()

        # colour-mode switch
        self.btn_mode = QtWidgets.QPushButton('Colour mode: level')
        self.btn_mode.setCheckable(True); self.btn_mode.setChecked(False)
        self.btn_mode.toggled.connect(self.switch_colour_mode)
        vb_l.addWidget(self.btn_mode)

        if self.mesh_actor:
            btn_mesh = QtWidgets.QPushButton('Mesh'); btn_mesh.setCheckable(True); btn_mesh.setChecked(True)
            btn_mesh.toggled.connect(lambda on: self.mesh_actor.SetVisibility(on))
            vb_l.addWidget(btn_mesh)

        vb_l.addStretch()
        w_disp = QtWidgets.QWidget(); w_disp.setLayout(vb_l)
        dock_l = QtWidgets.QDockWidget('Display'); dock_l.setWidget(w_disp)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, dock_l)

        # ----------------------- timer
        self.timer = QtCore.QTimer(); self.timer.setInterval(33); self.timer.timeout.connect(self.timer_step)

        # init
        self.slider.setValue(0)
        self.update_frame(0)
        self.plotter.reset_camera()

    # ------------------------------------------------------------------ frame ops
    def update_frame(self, idx):
        idx = int(idx)
        pts = self.coords[:idx + 1]
        self.path_poly.points = pts
        self.path_poly['rgb'] = self.colours[:idx + 1]
        if len(pts):
            self.path_poly.verts = np.c_[np.ones(len(pts), np.int64),
                                         np.arange(len(pts), dtype=np.int64)]
        self.path_poly.Modified()
        if len(pts):
            self.cur_sphere.SetPosition(*pts[-1]); self.cur_sphere.SetVisibility(True)
        else:
            self.cur_sphere.SetVisibility(False)
        self.timeline.set_index(idx)
        self.plotter.render()
        self.last_idx = idx

    def switch_colour_mode(self, on):
        self.colour_mode = 'type' if on else 'level'
        self.btn_mode.setText(f'Colour mode: {self.colour_mode}')
        self.colours = self.tag_cols if on else self.level_cols
        self.timeline.set_colours(self.colours)
        self.update_frame(self.last_idx)

    def step_back(self):
        s = self.step_spin.value(); self.slider.setValue((self.slider.value() - s) % self.N)
    def step_fwd(self):
        s = self.step_spin.value(); self.slider.setValue((self.slider.value() + s) % self.N)
    def toggle_play(self, on):
        self.btn_play.setText('Pause' if on else 'Play')
        (self.timer.start if on else self.timer.stop)()
    def timer_step(self):
        self.step_fwd()

# ------------------------------------------------------------------ main
def main():
    exp_name = 'klein'     # change to your experiment name
    waypt_fn = f'logs/toolpath/{exp_name}_single_shell/waypoints.xyz'
    mesh_fn  = f'udf_data/logs/{exp_name}/mesh_data.obj'

    exp_name = 'tree'     # change to your experiment name
    waypt_fn = f'logs/toolpath/{exp_name}_lattice/waypoints.xyz'
    mesh_fn  = f'udf_data/logs/{exp_name}/mesh_data.obj'

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(waypt_fn, mesh_fn)
    win.resize(1200, 800)
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()


# cmd
# python viewers/waypt_growing.py
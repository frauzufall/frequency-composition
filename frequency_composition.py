"""
Three oscillating colour frequencies (x, y and z) are blended into a 20x20x20 voxel
volume, controlled by a Korg MIDI controller or the GUI, and sent to the voxelbox /
simulator via UDP.

    python frequency_composition.py [--host 127.0.0.1] [--port 5005] [--midi nanokontrol]

Needs: numpy, PyQt5, mido, python-rtmidi, moderngl
"""
import argparse
import socket
import sys

import moderngl
import mido
import numpy as np
from pathlib import Path

from PyQt5 import QtCore, QtGui, QtWidgets

SHADERS = Path(__file__).parent / "shaders"

N = 20
SMOOTH_N = 60  # resolution of the continuous preview
WAVE_N = 30  # ... of the single frequencies
FPS = 30

# MIDI CC numbers of a Korg nanoKONTROL: knobs for x and y and pixelate,
# the first three faders for z
CC = {
    "x": {"wavelength": 14, "amplitude": 15, "frequency": 16},
    "y": {"wavelength": 17, "amplitude": 18, "frequency": 19},
    "z": {"wavelength": 2, "amplitude": 3, "frequency": 4},
    "pixelate_h": 21,
    "pixelate_v": 22,
}

# parameter ranges
BACKGROUND = (22, 22, 28)
RANGES = {"wavelength": (0.01, 2.0), "amplitude": (0.0, 1.0), "frequency": (-2.0, 2.0)}


# ---------------------------------------------------------------- model
# blendmode numbers of blending.frag
BLEND_MODES = [
    "Normal", "Lighten", "Darken", "Multiply", "Average", "Add", "Subtract", "Difference",
    "Negation", "Exclusion", "Screen", "Overlay", "SoftLight", "HardLight", "ColorDodge",
    "ColorBurn", "LinearDodge", "LinearBurn", "LinearLight", "VividLight", "PinLight",
    "HardMix", "Reflect", "Glow", "Phoenix", "NoBlack",
]


class Frequency:
    """One oscillation along an axis (0=x, 1=y, 2=z of the led data)."""

    def __init__(self, axis, enabled, color1, color2):
        self.axis = axis
        self.enabled = enabled
        self.wavelength = 1.0
        self.amplitude = 1.0
        self.frequency = 0.0
        self.phase = 0.0
        self.color1 = np.array(color1, dtype=float) / 255
        self.color2 = np.array(color2, dtype=float) / 255

    def advance(self, dt):
        self.phase += dt * self.frequency * (2 * np.pi / self.wavelength)

    def values(self, n=N):
        """oscillation value 0..1 for each of the n positions along the axis (for the plot)"""
        pos = (np.arange(n) + 0.5) / n
        return np.cos(pos * 2 * np.pi / self.wavelength + self.phase) * 0.5 * self.amplitude + 0.5

    def colors(self, n=N):
        v = self.values(n)[:, None]
        return self.color1 + (self.color2 - self.color1) * v  # (n, 3)


class Composition:
    """The waves, blended and pixelated by the shaders on the GPU."""

    def __init__(self):
        # one frequency per axis of the led data: x, y (index grows downwards) and z (depth)
        self.waves = [Frequency(axis, True, (0, 0, 0), (255, 255, 255)) for axis in (0, 1, 2)]
        self.blend = "Multiply"
        self.pixelate_h = 0.0  # 0..1
        self.pixelate_v = 0.0

        # The volume lives in a 2D texture, n wide and n*n high: row = z*n + y.
        # Rendered twice per frame: at N (what the voxelbox shows) and at SMOOTH_N (preview).
        self.ctx = moderngl.create_standalone_context(backend="egl")
        self.targets = {n: self._make_targets(n) for n in (N, SMOOTH_N, WAVE_N)}
        vert = (SHADERS / "fullscreen.vert").read_text()
        self.progs = {
            name: self.ctx.program(vertex_shader=vert, fragment_shader=(SHADERS / f"{name}.frag").read_text())
            for name in ("oscillation", "blending", "pixelate")
        }
        self.vao = {name: self.ctx.vertex_array(p, []) for name, p in self.progs.items()}

    def _make_targets(self, n):
        # 0/1: ping-pong blend result, 2: the frequency being drawn, 3: pixelated output
        tex = [self.ctx.texture((n, n * n), 3, dtype="f4") for _ in range(4)]
        for t in tex:
            t.filter = (moderngl.NEAREST, moderngl.NEAREST)
        return tex, [self.ctx.framebuffer([t]) for t in tex]

    def _pass(self, n, name, target, textures=(), **uniforms):
        tex, fbo = self.targets[n]
        prog = self.progs[name]
        for i, i_tex in enumerate(textures):
            tex[i_tex].use(i)
        for key, val in uniforms.items():
            if key in prog:  # unused uniforms get optimised away by the compiler
                prog[key].value = val
        fbo[target].use()
        self.vao[name].render(moderngl.TRIANGLES, vertices=3)

    def _compose(self, n):
        """run the shader chain for an n*n*n volume, returns [z, y, x, rgb] floats"""
        active = [w for w in self.waves if w.enabled]
        if not active:
            return np.zeros((n, n, n, 3), dtype="f4")

        def oscillate(w, target):
            self._pass(n, "oscillation", target, amplitude=w.amplitude, wavelength=w.wavelength,
                       phase=w.phase, size=float(n), axis=w.axis,
                       color1=(*w.color1, 1.0), color2=(*w.color2, 1.0))

        cur = 0
        oscillate(active[0], cur)
        for w in active[1:]:
            oscillate(w, 2)
            nxt = 1 - cur
            self._pass(n, "blending", nxt, (cur, 2), tex0=0, texBlend=1,
                       contrast=1.0, brightness=0.0, saturation=1.0, blendmix=1.0,
                       blendmode=BLEND_MODES.index(self.blend))
            cur = nxt
        # blocks are measured in voxels of the N grid
        bh = (1 + self.pixelate_h * (N - 1)) * n / N
        bv = (1 + self.pixelate_v * (N - 1)) * n / N
        if n == N:
            bh, bv = round(bh), round(bv)
        self._pass(n, "pixelate", 3, (cur,), tex0=0, size=float(n), block=(bh, bv))

        data = np.frombuffer(self.targets[n][0][3].read(), dtype="f4").reshape(n, n, n, 3)
        return np.clip(data, 0, 1)

    def _wave_volume(self, w, n):
        """one frequency on its own, as it goes into the blending"""
        self._pass(n, "oscillation", 2, amplitude=w.amplitude, wavelength=w.wavelength,
                   phase=w.phase, size=float(n), axis=w.axis,
                   color1=(*w.color1, 1.0), color2=(*w.color2, 1.0))
        return np.clip(np.frombuffer(self.targets[n][0][2].read(), dtype="f4").reshape(n, n, n, 3), 0, 1)

    def render(self, dt):
        """-> led volume (N^3), smooth result, [smooth volume of each frequency]"""
        for w in self.waves:
            w.advance(dt)
        waves = [self._wave_volume(w, WAVE_N) for w in self.waves]
        return self._compose(N), self._compose(SMOOTH_N), waves


# ---------------------------------------------------------------- widgets
def to_qcolor(rgb):
    r, g, b = (int(round(c * 255)) for c in rgb)
    return QtGui.QColor(r, g, b)


class ColorButton(QtWidgets.QPushButton):
    changed = QtCore.pyqtSignal()

    def __init__(self, rgb):
        super().__init__()
        self.rgb = np.array(rgb, dtype=float)
        self.setFixedSize(36, 24)
        self.clicked.connect(self.pick)
        self._style()

    def _style(self):
        self.setStyleSheet(f"background-color: {to_qcolor(self.rgb).name()}")

    def pick(self):
        c = QtWidgets.QColorDialog.getColor(to_qcolor(self.rgb), self)
        if c.isValid():
            self.set_rgb((c.red() / 255, c.green() / 255, c.blue() / 255))
            self.changed.emit()

    def set_rgb(self, rgb):
        self.rgb = np.array(rgb, dtype=float)
        self._style()


class ParamSlider(QtWidgets.QWidget):
    """float slider with label and value read-out; set() moves it without feedback"""
    valueChanged = QtCore.pyqtSignal(float)

    def __init__(self, label, lo, hi, value):
        super().__init__()
        self.lo, self.hi = lo, hi
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.label = QtWidgets.QLabel(label)
        self.label.setFixedWidth(100)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.readout = QtWidgets.QLabel()
        self.readout.setFixedWidth(50)
        lay.addWidget(self.label)
        lay.addWidget(self.slider)
        lay.addWidget(self.readout)
        self.slider.valueChanged.connect(self._moved)
        self.set(value)

    def _moved(self, i):
        v = self.lo + (self.hi - self.lo) * i / 1000
        self.readout.setText(f"{v:.2f}")
        self.valueChanged.emit(v)

    def set(self, v):
        self.slider.setValue(int(round((v - self.lo) / (self.hi - self.lo) * 1000)))


class FrequencyPanel(QtWidgets.QFrame):
    """one frequency: its own 3D image, colours and sliders"""

    def __init__(self, name, wave, camera):
        super().__init__()
        self.setObjectName("card")
        self.wave = wave
        self.sliders = {}
        lay = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        self.enable = QtWidgets.QCheckBox(name)
        self.enable.setStyleSheet("font-weight: bold")
        self.enable.setChecked(wave.enabled)
        self.enable.toggled.connect(lambda on: setattr(wave, "enabled", on))
        self.c1 = ColorButton(wave.color1)
        self.c2 = ColorButton(wave.color2)
        self.c1.changed.connect(self._colors)
        self.c2.changed.connect(self._colors)
        top.addWidget(self.enable)
        top.addStretch()
        top.addWidget(self.c1)
        top.addWidget(self.c2)
        lay.addLayout(top)
        self.view = SmoothView(camera, 250)
        lay.addWidget(self.view, 0, QtCore.Qt.AlignHCenter)
        for key, label in (("wavelength", "Wavelength"), ("amplitude", "Amplitude"), ("frequency", "Speed")):
            lo, hi = RANGES[key]
            s = ParamSlider(label, lo, hi, getattr(wave, key))
            s.valueChanged.connect(lambda v, k=key: setattr(wave, k, v))
            self.sliders[key] = s
            lay.addWidget(s)

    def _colors(self):
        self.wave.color1 = self.c1.rgb
        self.wave.color2 = self.c2.rgb


class Camera:
    """rotation shared by both 3D views"""

    def __init__(self):
        self.yaw, self.pitch = 0.5, 0.4

    def rotation(self):
        cy, sy, cp, sp = np.cos(self.yaw), np.sin(self.yaw), np.cos(self.pitch), np.sin(self.pitch)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
        return ry.T @ rx.T  # row vectors: p @ rotation


class View3D(QtWidgets.QLabel):
    """drag to rotate (both views together)"""

    def __init__(self, camera, px):
        super().__init__()
        self.camera = camera
        self.px = px
        self.setFixedSize(px, px)
        self._drag = None

    def mousePressEvent(self, e):
        self._drag = e.pos()

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            d = e.pos() - self._drag
            self._drag = e.pos()
            self.camera.yaw += d.x() * 0.01
            self.camera.pitch = float(np.clip(self.camera.pitch + d.y() * 0.01, -1.5, 1.5))

    def to_screen(self, p):
        """rotated points (..., 3) -> pixel coordinates; screen convention of the simulator:
        x to the right, y index grows downwards, z away from the viewer"""
        scale = self.px / (N * 1.9)
        return np.stack([p[..., 0] * scale + self.px / 2, -p[..., 1] * scale + self.px / 2], -1)


class VoxelView(View3D):
    """the led volume; points are splatted into an image with numpy (no OpenGL needed)"""

    def __init__(self, camera, px):
        super().__init__(camera, px)
        z, y, x = np.meshgrid(np.arange(N), np.arange(N), np.arange(N), indexing="ij")
        self.pts = np.stack([x - (N - 1) / 2, -(y - (N - 1) / 2), z - (N - 1) / 2], -1).reshape(-1, 3)

    def show_volume(self, vol):
        p = self.pts @ self.camera.rotation()
        cols = (vol.reshape(-1, 3) * 255).astype(np.uint8)
        keep = cols.max(axis=1) > 12  # dark voxels would only hide the lit ones behind
        p, cols = p[keep], cols[keep]
        order = np.argsort(-p[:, 2])  # far to near, near ones overwrite
        p, cols = p[order], cols[order]
        sx, sy = self.to_screen(p).astype(int).T
        img = np.zeros((self.px, self.px, 3), np.uint8)
        img[:] = BACKGROUND
        for dx in (0, 1, 2):
            for dy in (0, 1, 2):
                xi = np.clip(sx + dx - 1, 0, self.px - 1)
                yi = np.clip(sy + dy - 1, 0, self.px - 1)
                img[yi, xi] = cols
        qimg = QtGui.QImage(img.data, self.px, self.px, self.px * 3, QtGui.QImage.Format_RGB888)
        self.setPixmap(QtGui.QPixmap.fromImage(qimg))


class SmoothView(View3D):
    """the same composition as a continuous box: the six faces of a high-res volume,
    drawn as affinely transformed images (orthographic projection keeps faces affine)"""

    def show_volume(self, vol, dim=False):
        m = vol.shape[0]
        h = N / 2
        # (image [rows, cols, rgb], origin, vector along columns, vector along rows) in world coordinates
        faces = []
        for z, zi in ((-h, 0), (h, m - 1)):
            faces.append((vol[zi], (-h, h, z), (2 * h, 0, 0), (0, -2 * h, 0)))
        for y, yi in ((h, 0), (-h, m - 1)):
            faces.append((vol[:, yi], (-h, y, -h), (2 * h, 0, 0), (0, 0, 2 * h)))
        for x, xi in ((-h, 0), (h, m - 1)):
            faces.append((vol[:, :, xi].transpose(1, 0, 2), (x, h, -h), (0, 0, 2 * h), (0, -2 * h, 0)))

        rot = self.camera.rotation()
        pm = QtGui.QPixmap(self.px, self.px)
        pm.fill(QtGui.QColor(*BACKGROUND))
        painter = QtGui.QPainter(pm)
        painter.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)
        for img, o, u, v in faces:
            o, u, v = np.array(o, float), np.array(u, float), np.array(v, float)
            if ((o + (u + v) / 2) @ rot)[2] >= 0:  # face turned away from the viewer
                continue
            s0, su, sv = self.to_screen(np.stack([o, o + u, o + v]) @ rot)
            rows, cols = img.shape[:2]
            a, b = (su - s0) / cols, (sv - s0) / rows
            qimg = QtGui.QImage(np.ascontiguousarray((img * 255).astype(np.uint8)).data,
                                cols, rows, cols * 3, QtGui.QImage.Format_RGB888)
            painter.setTransform(QtGui.QTransform(a[0], a[1], b[0], b[1], s0[0], s0[1]))
            painter.drawImage(0, 0, qimg)
        painter.resetTransform()
        if dim:  # switched off
            painter.fillRect(pm.rect(), QtGui.QColor(*BACKGROUND, 190))
        painter.end()
        self.setPixmap(pm)


# ---------------------------------------------------------------- main window
class Window(QtWidgets.QWidget):
    midi_cc = QtCore.pyqtSignal(int, int)

    def __init__(self, host, port, midi_name):
        super().__init__()
        self.setWindowTitle("VoxelBox frequency composition")
        self.comp = Composition()
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.midi_in = None

        self.camera = Camera()
        root = QtWidgets.QVBoxLayout(self)

        # row 1: the frequencies, each with its own image, blended into the result below
        waves = QtWidgets.QHBoxLayout()
        self.op_labels = []
        self.panels = []
        for wave, axis in zip(self.comp.waves, "XYZ"):
            panel = FrequencyPanel(f"Frequency {axis}", wave, self.camera)
            self.panels.append(panel)
            if axis != "X":
                op = QtWidgets.QLabel()
                op.setObjectName("op")
                self.op_labels.append(op)
                waves.addWidget(op)
            waves.addWidget(panel)
        root.addLayout(waves)

        arrow = QtWidgets.QLabel("\u25bc")
        arrow.setObjectName("op")
        arrow.setAlignment(QtCore.Qt.AlignCenter)
        root.addWidget(arrow)

        # row 2: blend + pixelate controls -> result (continuous and as leds)
        bottom = QtWidgets.QHBoxLayout()
        ctrl = QtWidgets.QFrame()
        ctrl.setObjectName("card")
        cl = QtWidgets.QVBoxLayout(ctrl)
        blend = self.blend_box = QtWidgets.QComboBox()
        blend.addItems(BLEND_MODES)
        blend.setCurrentText(self.comp.blend)
        blend.currentTextChanged.connect(self.set_blend)
        cl.addWidget(QtWidgets.QLabel("Blend frequencies"))
        cl.addWidget(blend)
        self.pix_h = ParamSlider("Pixelate x/z", 0, 1, 0)
        self.pix_v = ParamSlider("Pixelate y", 0, 1, 0)
        self.pix_h.valueChanged.connect(lambda v: setattr(self.comp, "pixelate_h", v))
        self.pix_v.valueChanged.connect(lambda v: setattr(self.comp, "pixelate_v", v))
        cl.addWidget(self.pix_h)
        cl.addWidget(self.pix_v)
        self.midi_box = QtWidgets.QComboBox()
        self.midi_box.addItem("(no midi)")
        self.midi_box.addItems(mido.get_input_names())
        self.midi_box.currentTextChanged.connect(self.open_midi)
        cl.addWidget(QtWidgets.QLabel("MIDI input"))
        cl.addWidget(self.midi_box)
        self.status = QtWidgets.QLabel(f"sending to {host}:{port}")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #888")
        cl.addWidget(self.status)
        cl.addStretch()
        bottom.addWidget(ctrl, 1)
        eq = QtWidgets.QLabel("=")
        eq.setObjectName("op")
        bottom.addWidget(eq)
        self.smooth_view = SmoothView(self.camera, 340)
        self.view = VoxelView(self.camera, 340)
        bottom.addWidget(self.smooth_view)
        bottom.addWidget(self.view)
        root.addLayout(bottom)
        self.set_blend(self.comp.blend)

        self.midi_cc.connect(self.on_cc)
        self.open_midi_by_name(midi_name)

        self.clock = QtCore.QElapsedTimer()
        self.clock.start()
        self.last = self.clock.elapsed()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000 // FPS)

    def set_blend(self, name):
        self.comp.blend = name
        for op in self.op_labels:
            op.setText(name)
            op.setToolTip(f"blend mode: {name}")

    # --- midi
    def open_midi_by_name(self, name):
        names = mido.get_input_names()
        match = [n for n in names if name and name.lower() in n.lower() and "through" not in n.lower()]
        if match:
            self.midi_box.setCurrentText(match[0])  # triggers open_midi
        else:
            print(f"midi: no input matching '{name}' found, available: {names}")

    def open_midi(self, name):
        if self.midi_in is not None:
            self.midi_in.close()
            self.midi_in = None
        if name == "(no midi)":
            return
        # callback runs in rtmidi's thread -> hop to the gui thread via signal
        self.midi_in = mido.open_input(name, callback=self._midi_msg)
        print(f"midi: listening on {name}")
        self.status.setText(f"sending to {self.addr[0]}:{self.addr[1]}  |  midi: {name}")

    def _midi_msg(self, msg):
        if msg.type == "control_change":
            self.midi_cc.emit(msg.control, msg.value)

    def on_cc(self, cc, value):
        f = value / 127
        self.status.setText(f"sending to {self.addr[0]}:{self.addr[1]}  |  midi: CC {cc} = {value}")
        for i, key in enumerate(("x", "y", "z")):
            for param, num in CC[key].items():
                if num == cc:
                    lo, hi = RANGES[param]
                    self.panels[i].sliders[param].set(lo + (hi - lo) * f)
        if cc == CC["pixelate_h"]:
            self.pix_h.set(f)
        elif cc == CC["pixelate_v"]:
            self.pix_v.set(f)

    # --- frame
    def tick(self):
        now = self.clock.elapsed()
        dt = (now - self.last) / 1000
        self.last = now
        vol, smooth, waves = self.comp.render(dt)
        self.view.show_volume(vol)
        self.smooth_view.show_volume(smooth)
        for panel, wave_vol in zip(self.panels, waves):
            panel.view.show_volume(wave_vol, dim=not panel.wave.enabled)
        try:
            # index = (z*N + y)*N + x -> exactly the C order of vol[z, y, x, rgb]
            self.sock.sendto((vol * 255).astype(np.uint8).tobytes(), self.addr)
        except OSError as e:
            self.status.setText(f"send failed: {e}")

    def closeEvent(self, e):
        if self.midi_in is not None:
            self.midi_in.close()
        super().closeEvent(e)


def apply_style(app):
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QWidget { background: #2b2b33; color: #ddd; font-size: 13px; }
        QFrame#card { background: #33333d; border-radius: 8px; }
        QLabel#op { font-size: 15px; font-weight: bold; color: #aaa; background: transparent; }
        QLabel { background: transparent; }
        QPushButton { border: 1px solid #666; border-radius: 4px; }
        QComboBox { background: #444450; padding: 3px; }
        QSlider::groove:horizontal { height: 4px; background: #555; border-radius: 2px; }
        QSlider::handle:horizontal { width: 14px; margin: -6px 0; border-radius: 7px; background: #e0a030; }
        QSlider::sub-page:horizontal { background: #e0a030; border-radius: 2px; }
    """)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="voxelbox or simulator host")
    ap.add_argument("--port", type=int, default=5005)
    ap.add_argument("--midi", default="nanokontrol", help="substring of the MIDI input name to open automatically")
    args = ap.parse_args()
    app = QtWidgets.QApplication(sys.argv)
    apply_style(app)
    win = Window(args.host, args.port, args.midi)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

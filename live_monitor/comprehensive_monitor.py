#!/usr/bin/env python3
"""
PRISM-NHS Comprehensive Live Monitor
Full dashboard showing ALL engine metrics in real-time.
"""

import argparse
import socket
import struct
import sys
import threading
import time
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from PyQt5 import QtCore, QtWidgets, QtGui
    from PyQt5.QtCore import Qt
except ImportError:
    print("Install: pip3 install numpy pyqtgraph PyQt5 PyOpenGL")
    sys.exit(1)

@dataclass
class LiveFrame:
    frame_id: int = 0
    time_ps: float = 0.0
    temperature: float = 0.0
    potential_energy: float = 0.0
    kinetic_energy: float = 0.0
    n_atoms: int = 0
    spike_count: int = 0
    grid_dim: int = 0
    probe_id: int = 0
    sequence_score: float = 0.0
    # Extended
    target_temperature: float = 0.0
    uv_burst_active: bool = False
    differential_score: float = 0.0
    n_resonances: int = 0
    pocket_open_fraction: float = 0.0

    # Header format matches Rust fused_engine.rs build_monitor_frame() (60 bytes)
    # Q=u64, f=f32, I=u32, i=i32
    HEADER_FORMAT = '<QffffIIIif16s'
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 60 bytes

    @classmethod
    def from_bytes(cls, data):
        u = struct.unpack(cls.HEADER_FORMAT, data[:cls.HEADER_SIZE])
        return cls(frame_id=u[0], time_ps=u[1], temperature=u[2],
                   potential_energy=u[3], kinetic_energy=u[4],
                   n_atoms=u[5], spike_count=u[6], grid_dim=u[7],
                   probe_id=u[8], sequence_score=u[9])

@dataclass
class LiveSnapshot:
    header: LiveFrame
    positions: np.ndarray
    exclusion_field: np.ndarray
    spikes: List[Tuple[int, float]]
    aromatic_excitation: np.ndarray
    resonance_spectrum: Optional[np.ndarray] = None
    binding_candidates: List[Tuple[float, float, float, float]] = field(default_factory=list)

    @classmethod
    def from_bytes(cls, data):
        offset = 0
        header = LiveFrame.from_bytes(data)
        offset += LiveFrame.HEADER_SIZE

        # Positions
        n_pos = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
        positions = np.frombuffer(data[offset:offset+n_pos*4], dtype=np.float32).copy()
        positions = positions.reshape(-1, 3) if n_pos > 0 else np.zeros((0, 3))
        offset += n_pos * 4

        # Exclusion field
        n_excl = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
        exclusion_field = np.frombuffer(data[offset:offset+n_excl*4], dtype=np.float32).copy()
        dim = header.grid_dim
        if n_excl == dim**3 and dim > 0:
            exclusion_field = exclusion_field.reshape(dim, dim, dim)
        offset += n_excl * 4

        # Spikes
        n_spikes = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
        spikes = []
        for _ in range(n_spikes):
            idx = struct.unpack('<I', data[offset:offset+4])[0]
            intensity = struct.unpack('<f', data[offset+4:offset+8])[0]
            spikes.append((idx, intensity)); offset += 8

        # Aromatic excitation
        n_arom = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
        aromatic_excitation = np.frombuffer(data[offset:offset+n_arom*4], dtype=np.float32).copy() if n_arom > 0 else np.array([])
        offset += n_arom * 4

        # Extended data (optional)
        resonance_spectrum = None
        binding_candidates = []

        try:
            if offset < len(data) - 4:
                # Skip LIF potential
                n_lif = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
                offset += n_lif * 4

            if offset < len(data) - 4:
                # Resonance spectrum
                n_res = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
                if n_res > 0:
                    resonance_spectrum = np.frombuffer(data[offset:offset+n_res*4], dtype=np.float32).copy()
                offset += n_res * 4

            if offset < len(data) - 4:
                # Skip differential matrix
                n_diff = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
                offset += n_diff * 4

            if offset < len(data) - 4:
                # Binding candidates
                n_bind = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
                for _ in range(n_bind):
                    if offset + 16 <= len(data):
                        x, y, z, conf = struct.unpack('<ffff', data[offset:offset+16])
                        binding_candidates.append((x, y, z, conf))
                        offset += 16
        except:
            pass

        return cls(header, positions, exclusion_field, spikes, aromatic_excitation,
                   resonance_spectrum, binding_candidates)

class LiveMonitorReceiver:
    def __init__(self, host='127.0.0.1', port=9999):
        self.host, self.port = host, port
        self.socket = self.conn = self.thread = None
        self.running = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.frames_received = 0
        self.start_time = None

    def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        self.socket.settimeout(1.0)
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print(f"Monitor listening on {self.host}:{self.port}")

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=2)
        if self.conn: self.conn.close()
        if self.socket: self.socket.close()

    def _loop(self):
        while self.running:
            if self.conn is None:
                try:
                    self.conn, addr = self.socket.accept()
                    self.conn.settimeout(0.5)
                    print(f"Connected: {addr}")
                    self.start_time = time.time()
                    self.frames_received = 0
                except: continue
            try:
                len_data = self._recv(4)
                if not len_data: self.conn.close(); self.conn = None; continue
                frame_len = struct.unpack('<I', len_data)[0]
                frame_data = self._recv(frame_len)
                if not frame_data: self.conn.close(); self.conn = None; continue
                snapshot = LiveSnapshot.from_bytes(frame_data)
                with self.frame_lock: self.latest_frame = snapshot
                self.frames_received += 1
            except socket.timeout: continue
            except Exception as e:
                print(f"Error: {e}")
                if self.conn: self.conn.close()
                self.conn = None

    def _recv(self, n):
        data = b''
        while len(data) < n and self.running:
            try:
                chunk = self.conn.recv(n - len(data))
                if not chunk: return None
                data += chunk
            except socket.timeout: continue
        return data if len(data) == n else None

    def get_frame(self):
        with self.frame_lock: return self.latest_frame
    def get_fps(self):
        if not self.start_time: return 0
        return self.frames_received / max(time.time() - self.start_time, 0.001)

class MetricWidget(QtWidgets.QFrame):
    def __init__(self, label, unit="", color="#00ff88"):
        super().__init__()
        self.setStyleSheet(f"background:#1a1a2e;border:1px solid {color}40;border-radius:4px;")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(0)
        self.lbl = QtWidgets.QLabel(label)
        self.lbl.setStyleSheet(f"color:{color};font-size:10px;")
        layout.addWidget(self.lbl)
        self.val = QtWidgets.QLabel("--")
        self.val.setStyleSheet("color:white;font-size:14px;font-weight:bold;")
        layout.addWidget(self.val)
        if unit:
            u = QtWidgets.QLabel(unit)
            u.setStyleSheet("color:#666;font-size:9px;")
            layout.addWidget(u)
    def set_value(self, v): self.val.setText(v)
    def set_color(self, c): self.val.setStyleSheet(f"color:{c};font-size:14px;font-weight:bold;")

class MetricGroup(QtWidgets.QGroupBox):
    def __init__(self, title, metrics):
        super().__init__(title)
        self.setStyleSheet("QGroupBox{color:#00ff88;border:1px solid #00ff8840;border-radius:4px;margin-top:8px;}QGroupBox::title{subcontrol-origin:margin;left:8px;}")
        layout = QtWidgets.QGridLayout(self)
        layout.setSpacing(4)
        self.widgets = {}
        for i, (key, label, unit) in enumerate(metrics):
            w = MetricWidget(label, unit)
            layout.addWidget(w, i // 2, i % 2)
            self.widgets[key] = w
    def update_metric(self, key, value, color=None):
        if key in self.widgets:
            self.widgets[key].set_value(value)
            if color: self.widgets[key].set_color(color)

class MonitorWindow(QtWidgets.QMainWindow):
    def __init__(self, receiver):
        super().__init__()
        self.receiver = receiver
        self.setWindowTitle("PRISM-NHS Live Monitor")
        self.setGeometry(50, 50, 1800, 1000)
        self.history = {k: deque(maxlen=500) for k in ['time','temp','pe','ke','spikes','seq_score','diff_score']}
        self._setup_ui()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_all)
        self.timer.start(33)

    def _setup_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet("background-color:#0a0a1a;")
        main = QtWidgets.QHBoxLayout(central)
        main.setContentsMargins(5,5,5,5)

        # Left: 3D
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0,0,0,0)
        self.status = QtWidgets.QLabel("Waiting...")
        self.status.setStyleSheet("color:#00ffaa;font-size:14px;font-weight:bold;background:#1a1a2e;padding:6px;border-radius:4px;")
        ll.addWidget(self.status)
        self.gl = gl.GLViewWidget()
        self.gl.setCameraPosition(distance=60, elevation=25, azimuth=45)
        self.gl.setBackgroundColor('#0a0a1a')
        ll.addWidget(self.gl, stretch=1)
        axis = gl.GLAxisItem(); axis.setSize(20,20,20); self.gl.addItem(axis)
        grid = gl.GLGridItem(); grid.setSize(60,60); grid.translate(0,0,-25); self.gl.addItem(grid)
        self.protein = self.aromatics = self.spikes_gl = None
        self.binding_markers = []
        main.addWidget(left, stretch=2)

        # Middle: Metrics
        mid = QtWidgets.QWidget()
        ml = QtWidgets.QVBoxLayout(mid)
        ml.setContentsMargins(0,0,0,0)
        ml.setSpacing(4)

        self.physics = MetricGroup("PHYSICS", [('temp','Temperature','K'),('pe','Potential E','kcal/mol'),('ke','Kinetic E','kcal/mol'),('friction','Friction','ps^-1')])
        ml.addWidget(self.physics)
        self.uv = MetricGroup("UV EXCITATION", [('probe','Probe Target',''),('burst','Burst Active',''),('excited','Excited','aromatics')])
        ml.addWidget(self.uv)
        self.neuro = MetricGroup("NEUROMORPHIC", [('spikes','Spikes','/step'),('total','Total',''),('rate','Rate','/ps')])
        ml.addWidget(self.neuro)
        self.sensing = MetricGroup("ACTIVE SENSING", [('seq_score','Sequence Score',''),('diff','Differential',''),('best_pair','Best Pair','')])
        ml.addWidget(self.sensing)
        self.binding = MetricGroup("BINDING SITES", [('candidates','Candidates',''),('confidence','Best Conf.',''),('pocket','Pocket Open','%')])
        ml.addWidget(self.binding)
        self.perf = MetricGroup("PERFORMANCE", [('frame','Frame',''),('time','Sim Time','ps'),('fps','FPS','')])
        ml.addWidget(self.perf)
        ml.addStretch()
        main.addWidget(mid, stretch=1)

        # Right: Plots
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0,0,0,0)

        self.temp_plot = pg.PlotWidget(title="Temperature"); self.temp_plot.setBackground('#0a0a1a')
        self.temp_curve = self.temp_plot.plot(pen=pg.mkPen('#ffaa00',width=2))
        rl.addWidget(self.temp_plot)

        self.energy_plot = pg.PlotWidget(title="Energy"); self.energy_plot.setBackground('#0a0a1a')
        self.pe_curve = self.energy_plot.plot(pen=pg.mkPen('#ff4444',width=2),name='PE')
        self.ke_curve = self.energy_plot.plot(pen=pg.mkPen('#44aaff',width=2),name='KE')
        rl.addWidget(self.energy_plot)

        self.spike_plot = pg.PlotWidget(title="Spikes"); self.spike_plot.setBackground('#0a0a1a')
        self.spike_curve = self.spike_plot.plot(pen=pg.mkPen('#ff6600',width=2),fillLevel=0,brush='#ff660040')
        rl.addWidget(self.spike_plot)

        self.seq_plot = pg.PlotWidget(title="Sequence Score"); self.seq_plot.setBackground('#0a0a1a')
        self.seq_plot.setYRange(0,1)
        self.seq_curve = self.seq_plot.plot(pen=pg.mkPen('#aa44ff',width=2))
        self.seq_plot.addLine(y=0.7,pen=pg.mkPen('#ff0000',width=1,style=Qt.DashLine))
        rl.addWidget(self.seq_plot)

        self.res_plot = pg.PlotWidget(title="Resonance Spectrum"); self.res_plot.setBackground('#0a0a1a')
        self.res_bars = pg.BarGraphItem(x=[],height=[],width=0.05,brush='#4488ff')
        self.res_plot.addItem(self.res_bars)
        rl.addWidget(self.res_plot)

        main.addWidget(right, stretch=1)

    def update_all(self):
        frame = self.receiver.get_frame()
        if not frame: return
        h = frame.header
        fps = self.receiver.get_fps()

        uv_ind = "ON" if h.uv_burst_active else "OFF"
        self.status.setText(f"Frame {h.frame_id:,} | {h.time_ps:.2f}ps | {h.temperature:.1f}K | UV:{uv_ind} | Spikes:{h.spike_count} | Seq:{h.sequence_score:.2f}")

        self.history['time'].append(h.time_ps)
        self.history['temp'].append(h.temperature)
        self.history['pe'].append(h.potential_energy)
        self.history['ke'].append(h.kinetic_energy)
        self.history['spikes'].append(h.spike_count)
        self.history['seq_score'].append(h.sequence_score)

        t = np.array(self.history['time'])
        self.temp_curve.setData(t, np.array(self.history['temp']))
        self.pe_curve.setData(t, np.array(self.history['pe']))
        self.ke_curve.setData(t, np.array(self.history['ke']))
        self.spike_curve.setData(t, np.array(self.history['spikes']))
        self.seq_curve.setData(t, np.array(self.history['seq_score']))

        if frame.resonance_spectrum is not None and len(frame.resonance_spectrum) > 0:
            freqs = np.linspace(0.05, 2.0, len(frame.resonance_spectrum))
            self.res_bars.setOpts(x=freqs, height=frame.resonance_spectrum, width=0.03)

        # Metrics
        t_color = "#44aaff" if h.temperature < 100 else "#ffaa00" if h.temperature < 200 else "#ff4444"
        self.physics.update_metric('temp', f"{h.temperature:.1f}", t_color)
        self.physics.update_metric('pe', f"{h.potential_energy:.0f}")
        self.physics.update_metric('ke', f"{h.kinetic_energy:.0f}")

        self.uv.update_metric('probe', f"#{h.probe_id}")
        self.uv.update_metric('burst', "ON" if h.uv_burst_active else "OFF", "#ffff00" if h.uv_burst_active else "#666")
        n_exc = np.sum(frame.aromatic_excitation > 0.3) if len(frame.aromatic_excitation) > 0 else 0
        self.uv.update_metric('excited', f"{n_exc}")

        self.neuro.update_metric('spikes', f"{h.spike_count}", "#ff4444" if h.spike_count > 100 else "#44ff44")
        self.neuro.update_metric('total', f"{sum(self.history['spikes']):,}")

        seq_color = "#44ff44" if h.sequence_score > 0.7 else "#ffaa00" if h.sequence_score > 0.4 else "#ff4444"
        self.sensing.update_metric('seq_score', f"{h.sequence_score:.3f}", seq_color)
        self.sensing.update_metric('diff', f"{h.differential_score:.3f}")

        n_cand = len(frame.binding_candidates)
        best_conf = max([c[3] for c in frame.binding_candidates]) if frame.binding_candidates else 0
        self.binding.update_metric('candidates', f"{n_cand}")
        self.binding.update_metric('confidence', f"{best_conf:.2f}")
        self.binding.update_metric('pocket', f"{h.pocket_open_fraction*100:.0f}")

        self.perf.update_metric('frame', f"{h.frame_id:,}")
        self.perf.update_metric('time', f"{h.time_ps:.2f}")
        self.perf.update_metric('fps', f"{fps:.1f}")

        # 3D
        self._update_3d(frame)

    def _update_3d(self, frame):
        if len(frame.positions) == 0: return
        center = frame.positions.mean(axis=0)
        pos = frame.positions - center

        if self.protein: self.gl.removeItem(self.protein)
        colors = np.ones((len(pos),4)) * [0.5,0.7,0.9,0.6]
        self.protein = gl.GLScatterPlotItem(pos=pos, color=colors, size=3, pxMode=True)
        self.gl.addItem(self.protein)

        if self.aromatics: self.gl.removeItem(self.aromatics)
        if len(frame.aromatic_excitation) > 0:
            n = min(len(frame.aromatic_excitation), len(pos)//20)
            idx = np.linspace(0, len(pos)-1, n, dtype=int)
            apos = pos[idx]
            acol = []
            asz = []
            for i, exc in enumerate(frame.aromatic_excitation[:n]):
                if exc > 0.5: acol.append([1,1,0,1]); asz.append(15+exc*20)
                else: acol.append([0.7,0.3,0.8,0.8]); asz.append(10)
            self.aromatics = gl.GLScatterPlotItem(pos=apos, color=np.array(acol), size=np.array(asz), pxMode=True)
            self.gl.addItem(self.aromatics)

        if self.spikes_gl: self.gl.removeItem(self.spikes_gl)
        if frame.spikes and frame.header.grid_dim > 0:
            dim = frame.header.grid_dim
            spos, scol = [], []
            for idx, intensity in frame.spikes[:200]:
                z,y,x = idx//(dim*dim), (idx//dim)%dim, idx%dim
                spos.append([x-dim/2, y-dim/2, z-dim/2])
                scol.append([1, 1-intensity*0.7, 0, 0.9])
            if spos:
                self.spikes_gl = gl.GLScatterPlotItem(pos=np.array(spos), color=np.array(scol), size=8, pxMode=True)
                self.gl.addItem(self.spikes_gl)

        for m in self.binding_markers: self.gl.removeItem(m)
        self.binding_markers.clear()
        for x,y,z,conf in frame.binding_candidates[:5]:
            sphere = gl.MeshData.sphere(rows=8,cols=8,radius=3+conf*3)
            color = [1,0,0.5,0.6] if conf > 0.7 else [1,0.5,0,0.4]
            m = gl.GLMeshItem(meshdata=sphere, smooth=True, color=color, shader='shaded', glOptions='translucent')
            m.translate(x,y,z)
            self.gl.addItem(m)
            self.binding_markers.append(m)

    def closeEvent(self, e):
        self.timer.stop()
        self.receiver.stop()
        e.accept()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()

    print("PRISM-NHS Comprehensive Live Monitor")
    print("=" * 40)

    receiver = LiveMonitorReceiver(args.host, args.port)
    receiver.start()

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle('Fusion')
    p = QtGui.QPalette()
    p.setColor(QtGui.QPalette.Window, QtGui.QColor(10,10,26))
    p.setColor(QtGui.QPalette.WindowText, Qt.white)
    app.setPalette(p)

    w = MonitorWindow(receiver)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""PRISM-NHS Live Monitor Client"""
import argparse, socket, struct, sys, threading, time
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np

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
    HEADER_FORMAT = '<QfffIIIif16s'
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    
    @classmethod
    def from_bytes(cls, data):
        u = struct.unpack(cls.HEADER_FORMAT, data[:cls.HEADER_SIZE])
        return cls(u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9])

@dataclass  
class LiveSnapshot:
    header: LiveFrame
    positions: np.ndarray
    exclusion_field: np.ndarray
    spikes: List[Tuple[int, float]]
    aromatic_excitation: np.ndarray
    
    @classmethod
    def from_bytes(cls, data):
        offset = 0
        header = LiveFrame.from_bytes(data)
        offset += LiveFrame.HEADER_SIZE
        
        n_pos = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
        positions = np.frombuffer(data[offset:offset+n_pos*4], dtype=np.float32).copy().reshape(-1, 3) if n_pos else np.zeros((0,3))
        offset += n_pos * 4
        
        n_excl = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
        exclusion_field = np.frombuffer(data[offset:offset+n_excl*4], dtype=np.float32).copy()
        dim = header.grid_dim
        if n_excl == dim**3 and dim > 0: exclusion_field = exclusion_field.reshape(dim,dim,dim)
        offset += n_excl * 4
        
        n_spikes = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
        spikes = []
        for _ in range(n_spikes):
            idx = struct.unpack('<I', data[offset:offset+4])[0]
            intensity = struct.unpack('<f', data[offset+4:offset+8])[0]
            spikes.append((idx, intensity)); offset += 8
        
        n_arom = struct.unpack('<I', data[offset:offset+4])[0]; offset += 4
        aromatic_excitation = np.frombuffer(data[offset:offset+n_arom*4], dtype=np.float32).copy()
        return cls(header, positions, exclusion_field, spikes, aromatic_excitation)

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
        if self.thread: self.thread.join(timeout=2.0)
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
        if not self.start_time or not self.frames_received: return 0.0
        return self.frames_received / max(time.time() - self.start_time, 0.001)

# Try PyQt5
try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from PyQt5 import QtCore, QtWidgets, QtGui
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    print("PyQt5 not available, using matplotlib fallback")

if HAS_PYQT:
    class Window(QtWidgets.QMainWindow):
        def __init__(self, receiver):
            super().__init__()
            self.receiver = receiver
            self.setWindowTitle("PRISM-NHS Live Monitor")
            self.setGeometry(100, 100, 1400, 900)
            
            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            layout = QtWidgets.QHBoxLayout(central)
            splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            layout.addWidget(splitter)
            
            self.gl = gl.GLViewWidget()
            self.gl.setCameraPosition(distance=50)
            self.gl.setBackgroundColor('k')
            splitter.addWidget(self.gl)
            
            right = QtWidgets.QWidget()
            rl = QtWidgets.QVBoxLayout(right)
            splitter.addWidget(right)
            splitter.setSizes([900, 500])
            
            self.info = QtWidgets.QLabel("Waiting...")
            self.info.setStyleSheet("font-size:14px;font-weight:bold;color:#0f0;")
            rl.addWidget(self.info)
            
            self.ep = pg.PlotWidget(title="Energy"); self.ep.addLegend(); rl.addWidget(self.ep)
            self.pe = self.ep.plot(pen='r', name='PE')
            self.ke = self.ep.plot(pen='c', name='KE')
            
            self.tp = pg.PlotWidget(title="Temperature"); rl.addWidget(self.tp)
            self.tc = self.tp.plot(pen='y')
            
            self.sp = pg.PlotWidget(title="Spikes"); rl.addWidget(self.sp)
            self.sc = self.sp.plot(pen='orange', fillLevel=0, brush=(255,165,0,80))
            
            self.qp = pg.PlotWidget(title="Sequence Score"); self.qp.setYRange(0,1); rl.addWidget(self.qp)
            self.qc = self.qp.plot(pen='m')
            
            self.hist = {k: deque(maxlen=1000) for k in ['t','pe','ke','temp','spike','seq']}
            self.atoms = self.spikes_gl = None
            
            axis = gl.GLAxisItem(); axis.setSize(30,30,30); self.gl.addItem(axis)
            grid = gl.GLGridItem(); grid.setSize(60,60); self.gl.addItem(grid)
            
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update)
            self.timer.start(33)
            
        def update(self):
            f = self.receiver.get_frame()
            if not f: return
            h = f.header
            self.info.setText(f"Frame:{h.frame_id} Time:{h.time_ps:.2f}ps T:{h.temperature:.1f}K Spikes:{h.spike_count} FPS:{self.receiver.get_fps():.1f}")
            
            for k,v in [('t',h.time_ps),('pe',h.potential_energy),('ke',h.kinetic_energy),
                       ('temp',h.temperature),('spike',h.spike_count),('seq',h.sequence_score)]:
                self.hist[k].append(v)
            
            t = np.array(self.hist['t'])
            self.pe.setData(t, np.array(self.hist['pe']))
            self.ke.setData(t, np.array(self.hist['ke']))
            self.tc.setData(t, np.array(self.hist['temp']))
            self.sc.setData(t, np.array(self.hist['spike']))
            self.qc.setData(t, np.array(self.hist['seq']))
            
            if len(f.positions) > 0:
                pos = f.positions - f.positions.mean(axis=0)
                colors = np.ones((len(pos),4)) * [0.7,0.8,1.0,0.6]
                if self.atoms: self.gl.removeItem(self.atoms)
                self.atoms = gl.GLScatterPlotItem(pos=pos, color=colors, size=2, pxMode=True)
                self.gl.addItem(self.atoms)
            
            if self.spikes_gl: self.gl.removeItem(self.spikes_gl); self.spikes_gl = None
            if f.spikes and h.grid_dim > 0:
                pos, col, sz = [], [], []
                for idx, intensity in f.spikes[:500]:
                    z,y,x = idx//(h.grid_dim**2), (idx//h.grid_dim)%h.grid_dim, idx%h.grid_dim
                    pos.append([x-h.grid_dim/2, y-h.grid_dim/2, z-h.grid_dim/2])
                    col.append([1,1-intensity*0.8,0,0.9])
                    sz.append(8+intensity*15)
                if pos:
                    self.spikes_gl = gl.GLScatterPlotItem(pos=np.array(pos), color=np.array(col), size=np.array(sz), pxMode=True)
                    self.gl.addItem(self.spikes_gl)
                    
        def closeEvent(self, e): self.timer.stop(); self.receiver.stop(); e.accept()

def run_matplotlib(receiver):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8)); plt.ion()
    hist = {k: deque(maxlen=500) for k in ['t','pe','temp','spike']}
    print("Matplotlib monitor. Ctrl+C to stop.")
    try:
        while True:
            f = receiver.get_frame()
            if f:
                h = f.header
                hist['t'].append(h.time_ps); hist['pe'].append(h.potential_energy)
                hist['temp'].append(h.temperature); hist['spike'].append(h.spike_count)
                t = np.array(hist['t'])
                for ax in axes.flat: ax.clear()
                axes[0,0].plot(t, hist['pe'], 'r-'); axes[0,0].set_title(f'Frame {h.frame_id}')
                axes[0,1].plot(t, hist['temp'], 'g-')
                axes[1,0].fill_between(t, hist['spike'], alpha=0.5, color='orange')
                if len(f.positions) > 0:
                    axes[1,1].scatter(f.positions[:,0], f.positions[:,1], s=1, alpha=0.5)
                plt.tight_layout()
            plt.pause(0.05)
    except KeyboardInterrupt: pass
    finally: receiver.stop(); plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--matplotlib", action="store_true")
    args = parser.parse_args()
    
    receiver = LiveMonitorReceiver(args.host, args.port)
    receiver.start()
    
    if args.matplotlib or not HAS_PYQT:
        run_matplotlib(receiver)
    else:
        app = QtWidgets.QApplication(sys.argv)
        app.setStyle('Fusion')
        p = QtGui.QPalette()
        p.setColor(QtGui.QPalette.Window, QtGui.QColor(30,30,30))
        p.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)
        app.setPalette(p)
        w = Window(receiver)
        w.show()
        sys.exit(app.exec_())

if __name__ == "__main__": main()

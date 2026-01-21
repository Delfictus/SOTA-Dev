#!/usr/bin/env python3
"""
PRISM-NHS Blind Validation Monitor

STRICT ISOLATION ARCHITECTURE:
- This monitor runs in a SEPARATE PROCESS from the engine
- Loads known site coordinates from EXTERNAL JSON file at startup
- Engine has ZERO access to this file or its contents
- Comparison happens AFTER receiving engine output
- NO feedback is sent back to the engine

When the engine finds a known site, it's a genuine blind rediscovery
through physics, not through any prior knowledge.
"""

import socket
import struct
import json
import sys
import argparse
import math
import threading
from datetime import datetime
from collections import deque

import numpy as np

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                  QHBoxLayout, QLabel, QGroupBox, QGridLayout,
                                  QTableWidget, QTableWidgetItem, QHeaderView,
                                  QProgressBar, QTextEdit)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
    from PyQt5.QtGui import QFont, QColor
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    HAS_GUI = True
except ImportError:
    HAS_GUI = False
    print("WARNING: PyQt5/pyqtgraph not available. Running in headless mode.")


class KnownSiteDatabase:
    """Loads and manages known binding site reference data."""

    def __init__(self, json_path: str):
        self.json_path = json_path
        self.sites = []
        self.criteria = {}
        self.thresholds = {}
        self.structure_id = ""
        self._load()

    def _load(self):
        """Load known sites from JSON file."""
        print(f"[VALIDATION] Loading reference data from: {self.json_path}")

        with open(self.json_path, 'r') as f:
            data = json.load(f)

        self.structure_id = data.get("structure_id", "unknown")
        self.criteria = data.get("discovery_criteria", {})
        self.thresholds = data.get("validation_thresholds", {})

        for site_data in data.get("known_sites", []):
            site = {
                "id": site_data["site_id"],
                "name": site_data["name"],
                "center": np.array(site_data["center_angstrom"], dtype=np.float32),
                "radius": site_data["radius_angstrom"],
                "druggability": site_data.get("druggability", "unknown"),
                "residues": site_data.get("defining_residues", []),
                "volume_threshold": site_data.get("pocket_volume_angstrom3", {}).get(
                    "threshold_for_detection", 100)
            }
            self.sites.append(site)

        print(f"[VALIDATION] Loaded {len(self.sites)} known sites for {self.structure_id}")
        for site in self.sites:
            print(f"  - {site['name']}: center={site['center']}, radius={site['radius']}A")

    def compute_rmsd(self, candidate_xyz: np.ndarray, site_idx: int = 0) -> float:
        """Compute RMSD between candidate and known site center."""
        if site_idx >= len(self.sites):
            return float('inf')

        known_center = self.sites[site_idx]["center"]
        diff = candidate_xyz - known_center
        return float(np.sqrt(np.sum(diff**2)))

    def classify_match(self, rmsd: float) -> str:
        """Classify match quality based on RMSD."""
        if rmsd <= self.thresholds.get("rmsd_excellent", 2.0):
            return "REDISCOVERY"
        elif rmsd <= self.thresholds.get("rmsd_good", 3.5):
            return "STRONG"
        elif rmsd <= self.thresholds.get("rmsd_acceptable", 5.0):
            return "GOOD"
        elif rmsd <= self.thresholds.get("rmsd_weak", 8.0):
            return "WEAK"
        else:
            return "NONE"


class ValidationState:
    """Tracks validation state across frames."""

    def __init__(self, database: KnownSiteDatabase):
        self.database = database
        self.frame_count = 0
        self.best_rmsd = {site["id"]: float('inf') for site in database.sites}
        self.best_candidate = {site["id"]: None for site in database.sites}
        self.match_history = {site["id"]: deque(maxlen=100) for site in database.sites}
        self.sustained_match = {site["id"]: 0 for site in database.sites}
        self.rediscovery_declared = {site["id"]: False for site in database.sites}
        self.rediscovery_frame = {site["id"]: None for site in database.sites}

        # Time series for plotting
        self.rmsd_history = {site["id"]: deque(maxlen=500) for site in database.sites}
        self.confidence_history = deque(maxlen=500)
        self.time_history = deque(maxlen=500)

    def update(self, candidates: list, time_ps: float) -> dict:
        """
        Update validation state with new candidates from engine.

        candidates: list of (x, y, z, confidence) tuples
        Returns: dict with validation results for each known site
        """
        self.frame_count += 1
        results = {}

        for site in self.database.sites:
            site_id = site["id"]
            best_rmsd_this_frame = float('inf')
            best_candidate_this_frame = None

            # Find closest candidate to this known site
            for cand in candidates:
                if len(cand) >= 4:
                    xyz = np.array([cand[0], cand[1], cand[2]], dtype=np.float32)
                    confidence = cand[3]
                    rmsd = self.database.compute_rmsd(xyz,
                        self.database.sites.index(site))

                    if rmsd < best_rmsd_this_frame:
                        best_rmsd_this_frame = rmsd
                        best_candidate_this_frame = (xyz, confidence)

            # Update tracking
            self.rmsd_history[site_id].append(best_rmsd_this_frame)

            if best_rmsd_this_frame < self.best_rmsd[site_id]:
                self.best_rmsd[site_id] = best_rmsd_this_frame
                self.best_candidate[site_id] = best_candidate_this_frame

            # Check for sustained match
            match_quality = self.database.classify_match(best_rmsd_this_frame)
            is_matching = match_quality in ["GOOD", "STRONG", "REDISCOVERY"]

            self.match_history[site_id].append(is_matching)

            if is_matching:
                self.sustained_match[site_id] += 1
            else:
                self.sustained_match[site_id] = max(0, self.sustained_match[site_id] - 1)

            # Declare rediscovery if sustained
            min_sustained = self.database.criteria.get("min_sustained_frames", 10)
            if (not self.rediscovery_declared[site_id] and
                self.sustained_match[site_id] >= min_sustained and
                match_quality in ["STRONG", "REDISCOVERY"]):
                self.rediscovery_declared[site_id] = True
                self.rediscovery_frame[site_id] = self.frame_count
                print(f"\n{'='*60}")
                print(f"  REDISCOVERY CONFIRMED: {site['name']}")
                print(f"  Frame: {self.frame_count}, Time: {time_ps:.1f} ps")
                print(f"  RMSD: {best_rmsd_this_frame:.2f} A")
                print(f"  Sustained for {self.sustained_match[site_id]} frames")
                print(f"{'='*60}\n")

            results[site_id] = {
                "rmsd": best_rmsd_this_frame,
                "best_rmsd": self.best_rmsd[site_id],
                "match_quality": match_quality,
                "sustained": self.sustained_match[site_id],
                "rediscovered": self.rediscovery_declared[site_id],
                "candidate": best_candidate_this_frame
            }

        self.time_history.append(time_ps)

        return results


class EngineDataReceiver(QObject if HAS_GUI else object):
    """Receives data from PRISM-NHS engine via TCP."""

    if HAS_GUI:
        data_received = pyqtSignal(dict)

    def __init__(self, port: int = 9999):
        if HAS_GUI:
            super().__init__()
        self.port = port
        self.running = False
        self.socket = None
        self.conn = None

    def start(self):
        """Start listening for engine connections."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('127.0.0.1', self.port))
        self.socket.listen(1)
        self.running = True

        self.thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()

        print(f"[RECEIVER] Listening on 127.0.0.1:{self.port}")

    def _listen_loop(self):
        """Main receive loop."""
        while self.running:
            try:
                self.socket.settimeout(1.0)
                try:
                    self.conn, addr = self.socket.accept()
                    print(f"[RECEIVER] Engine connected from {addr}")
                    self._receive_frames()
                except socket.timeout:
                    continue
            except Exception as e:
                if self.running:
                    print(f"[RECEIVER] Error: {e}")

    def _receive_frames(self):
        """Receive frames from connected engine."""
        self.conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        while self.running:
            try:
                # Read frame length
                length_data = self._recv_exact(4)
                if not length_data:
                    break
                frame_length = struct.unpack('<I', length_data)[0]

                # Read frame data
                frame_data = self._recv_exact(frame_length)
                if not frame_data:
                    break

                # Parse frame
                parsed = self._parse_frame(frame_data)
                if parsed and HAS_GUI:
                    self.data_received.emit(parsed)
                elif parsed:
                    # Headless mode - just print
                    self._print_update(parsed)

            except Exception as e:
                print(f"[RECEIVER] Frame error: {e}")
                break

        if self.conn:
            self.conn.close()
            print("[RECEIVER] Engine disconnected")

    def _recv_exact(self, n: int) -> bytes:
        """Receive exactly n bytes."""
        data = b''
        while len(data) < n:
            chunk = self.conn.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data

    def _parse_frame(self, data: bytes) -> dict:
        """Parse frame from engine (same protocol as comprehensive_monitor)."""
        try:
            # 60-byte header (matches Rust fused_engine.rs)
            # Q=u64, ffff=4*f32, III=3*u32, i=i32, f=f32, 16s=padding
            header = struct.unpack('<QffffIIIif16s', data[:60])
            frame_id, time_ps, temperature, pe, ke = header[:5]
            n_atoms, n_spikes, grid_dim = header[5:8]
            current_probe = header[8]
            sequence_score = header[9]

            offset = 60  # Header is 60 bytes

            # Positions
            n_pos = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            positions = []
            for _ in range(n_pos):
                positions.append(struct.unpack('<f', data[offset:offset+4])[0])
                offset += 4

            # Exclusion field
            n_excl = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            offset += n_excl * 4  # Skip

            # Spikes
            n_spk = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            offset += n_spk * 8  # Skip (idx + intensity)

            # Excitation
            n_exc = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            offset += n_exc * 4  # Skip

            # LIF
            n_lif = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            offset += n_lif * 4  # Skip

            # Resonance
            n_res = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            offset += n_res * 4  # Skip

            # Differential
            n_diff = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            offset += n_diff * 8  # Skip

            # Binding candidates - THIS IS WHAT WE CARE ABOUT
            n_bind = struct.unpack('<I', data[offset:offset+4])[0]
            offset += 4
            candidates = []
            for _ in range(n_bind):
                x, y, z, conf = struct.unpack('<ffff', data[offset:offset+16])
                candidates.append((x, y, z, conf))
                offset += 16

            return {
                "frame_id": frame_id,
                "time_ps": time_ps,
                "temperature": temperature,
                "n_spikes": n_spikes,
                "sequence_score": sequence_score,
                "candidates": candidates,
                "n_atoms": n_atoms
            }

        except Exception as e:
            print(f"[PARSE] Error: {e}")
            return None

    def _print_update(self, data: dict):
        """Print update in headless mode."""
        print(f"\rFrame {data['frame_id']:>6} | "
              f"t={data['time_ps']:>7.1f}ps | "
              f"T={data['temperature']:>5.0f}K | "
              f"Candidates: {len(data['candidates'])}", end='', flush=True)

    def stop(self):
        """Stop receiver."""
        self.running = False
        if self.socket:
            self.socket.close()


class ValidationMonitorWindow(QMainWindow):
    """Main validation monitor GUI."""

    def __init__(self, database: KnownSiteDatabase):
        super().__init__()
        self.database = database
        self.state = ValidationState(database)
        self.receiver = EngineDataReceiver(port=9999)

        self.setWindowTitle(f"PRISM-NHS Blind Validation - {database.structure_id}")
        self.setGeometry(100, 100, 1400, 900)

        self._setup_ui()
        self._connect_signals()

        # Start receiver
        self.receiver.start()

    def _setup_ui(self):
        """Setup the UI."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        # Left panel: 3D view
        left_panel = QVBoxLayout()

        # 3D visualization
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setMinimumSize(600, 500)
        self.gl_widget.setCameraPosition(distance=50)

        # Add coordinate axes
        ax = gl.GLAxisItem()
        ax.setSize(20, 20, 20)
        self.gl_widget.addItem(ax)

        # Add known site spheres (wireframe - green)
        self.known_site_meshes = []
        for site in self.database.sites:
            mesh = gl.GLMeshItem(
                meshdata=self._create_sphere_mesh(site["radius"]),
                smooth=True,
                drawEdges=True,
                edgeColor=(0, 1, 0, 0.5),
                drawFaces=False
            )
            mesh.translate(*site["center"])
            self.gl_widget.addItem(mesh)
            self.known_site_meshes.append(mesh)

        # Candidate markers (will be updated)
        self.candidate_scatter = gl.GLScatterPlotItem()
        self.gl_widget.addItem(self.candidate_scatter)

        left_panel.addWidget(self.gl_widget)

        # Status panel
        status_box = QGroupBox("Validation Status")
        status_layout = QGridLayout(status_box)

        self.status_labels = {}
        for i, site in enumerate(self.database.sites):
            name_label = QLabel(f"<b>{site['name']}</b>")
            rmsd_label = QLabel("RMSD: --")
            match_label = QLabel("Match: --")
            status_label = QLabel("Searching...")

            status_layout.addWidget(name_label, i, 0)
            status_layout.addWidget(rmsd_label, i, 1)
            status_layout.addWidget(match_label, i, 2)
            status_layout.addWidget(status_label, i, 3)

            self.status_labels[site["id"]] = {
                "rmsd": rmsd_label,
                "match": match_label,
                "status": status_label
            }

        left_panel.addWidget(status_box)
        layout.addLayout(left_panel, 2)

        # Right panel: plots and log
        right_panel = QVBoxLayout()

        # RMSD plot
        rmsd_box = QGroupBox("RMSD vs Known Sites")
        rmsd_layout = QVBoxLayout(rmsd_box)

        self.rmsd_plot = pg.PlotWidget()
        self.rmsd_plot.setLabel('left', 'RMSD', units='A')
        self.rmsd_plot.setLabel('bottom', 'Time', units='ps')
        self.rmsd_plot.addLegend()
        self.rmsd_plot.setYRange(0, 20)

        # Add threshold lines
        self.rmsd_plot.addLine(y=5.0, pen=pg.mkPen('g', width=2, style=Qt.DashLine))
        self.rmsd_plot.addLine(y=2.0, pen=pg.mkPen('c', width=2, style=Qt.DashLine))

        self.rmsd_curves = {}
        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
        for i, site in enumerate(self.database.sites):
            curve = self.rmsd_plot.plot(
                pen=pg.mkPen(colors[i % len(colors)], width=2),
                name=site["name"][:20]
            )
            self.rmsd_curves[site["id"]] = curve

        rmsd_layout.addWidget(self.rmsd_plot)
        right_panel.addWidget(rmsd_box)

        # Match strength indicator
        match_box = QGroupBox("Match Strength")
        match_layout = QVBoxLayout(match_box)

        self.match_bars = {}
        for site in self.database.sites:
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat(f"{site['name'][:15]}: %p%")
            match_layout.addWidget(bar)
            self.match_bars[site["id"]] = bar

        right_panel.addWidget(match_box)

        # Event log
        log_box = QGroupBox("Validation Log")
        log_layout = QVBoxLayout(log_box)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)

        right_panel.addWidget(log_box)

        layout.addLayout(right_panel, 1)

        # Initial log entry
        self._log(f"Validation monitor started for {self.database.structure_id}")
        self._log(f"Loaded {len(self.database.sites)} known sites")
        self._log("Waiting for engine connection on port 9999...")

    def _create_sphere_mesh(self, radius: float, rows=12, cols=12):
        """Create sphere mesh data."""
        verts = []
        faces = []

        for i in range(rows + 1):
            theta = i * np.pi / rows
            for j in range(cols):
                phi = j * 2 * np.pi / cols
                x = radius * np.sin(theta) * np.cos(phi)
                y = radius * np.sin(theta) * np.sin(phi)
                z = radius * np.cos(theta)
                verts.append([x, y, z])

        for i in range(rows):
            for j in range(cols):
                p1 = i * cols + j
                p2 = i * cols + (j + 1) % cols
                p3 = (i + 1) * cols + j
                p4 = (i + 1) * cols + (j + 1) % cols
                faces.append([p1, p2, p4])
                faces.append([p1, p4, p3])

        return gl.MeshData(vertexes=np.array(verts), faces=np.array(faces))

    def _connect_signals(self):
        """Connect receiver signals."""
        self.receiver.data_received.connect(self._on_data_received)

    def _on_data_received(self, data: dict):
        """Handle data from engine."""
        candidates = data.get("candidates", [])
        time_ps = data.get("time_ps", 0)

        # Update validation state
        results = self.state.update(candidates, time_ps)

        # Update 3D view with candidates
        if candidates:
            pos = np.array([[c[0], c[1], c[2]] for c in candidates])
            conf = np.array([c[3] for c in candidates])

            # Color by confidence (red=low, green=high)
            colors = np.zeros((len(candidates), 4))
            colors[:, 0] = 1 - conf  # Red
            colors[:, 1] = conf      # Green
            colors[:, 3] = 0.8       # Alpha

            self.candidate_scatter.setData(pos=pos, color=colors, size=8)

        # Update status labels and plots
        for site_id, result in results.items():
            labels = self.status_labels[site_id]

            # RMSD
            rmsd = result["rmsd"]
            labels["rmsd"].setText(f"RMSD: {rmsd:.2f} A")

            # Match quality with color
            quality = result["match_quality"]
            color = {
                "REDISCOVERY": "green",
                "STRONG": "lime",
                "GOOD": "yellow",
                "WEAK": "orange",
                "NONE": "gray"
            }.get(quality, "gray")
            labels["match"].setText(f"<span style='color:{color}'>{quality}</span>")

            # Status
            if result["rediscovered"]:
                labels["status"].setText("<b style='color:green'>REDISCOVERED!</b>")
            else:
                sustained = result["sustained"]
                labels["status"].setText(f"Sustained: {sustained}")

            # Match bar
            bar = self.match_bars[site_id]
            strength = {
                "REDISCOVERY": 100,
                "STRONG": 85,
                "GOOD": 65,
                "WEAK": 40,
                "NONE": 0
            }.get(quality, 0)
            bar.setValue(strength)

            # Update RMSD curve
            if len(self.state.time_history) > 0:
                times = list(self.state.time_history)
                rmsds = list(self.state.rmsd_history[site_id])
                self.rmsd_curves[site_id].setData(times, rmsds)

        # Log significant events
        for site_id, result in results.items():
            if result["rediscovered"] and self.state.rediscovery_frame[site_id] == self.state.frame_count:
                site_name = next(s["name"] for s in self.database.sites if s["id"] == site_id)
                self._log(f"REDISCOVERY: {site_name} at t={time_ps:.1f}ps, RMSD={result['rmsd']:.2f}A")

    def _log(self, msg: str):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {msg}")

    def closeEvent(self, event):
        """Cleanup on close."""
        self.receiver.stop()
        super().closeEvent(event)


def run_headless(database: KnownSiteDatabase):
    """Run in headless mode (no GUI)."""
    state = ValidationState(database)
    receiver = EngineDataReceiver(port=9999)

    def on_data(data):
        candidates = data.get("candidates", [])
        time_ps = data.get("time_ps", 0)
        results = state.update(candidates, time_ps)

        # Print status
        best_matches = []
        for site_id, result in results.items():
            best_matches.append(f"{site_id[:10]}={result['rmsd']:.1f}A")

        print(f"\rFrame {data['frame_id']:>6} | t={time_ps:>7.1f}ps | " +
              " | ".join(best_matches), end='', flush=True)

    # Monkey-patch for headless
    original_print = receiver._print_update
    receiver._print_update = lambda d: on_data(d)

    receiver.start()

    print("Running in headless mode. Press Ctrl+C to stop.")
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        receiver.stop()


def main():
    parser = argparse.ArgumentParser(
        description="PRISM-NHS Blind Validation Monitor",
        epilog="This monitor validates engine output against known binding sites "
               "loaded from an EXTERNAL JSON file. The engine has ZERO access to "
               "this reference data."
    )
    parser.add_argument("--reference", "-r", required=True,
                        help="Path to known sites JSON file")
    parser.add_argument("--port", "-p", type=int, default=9999,
                        help="TCP port to listen on (default: 9999)")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI")

    args = parser.parse_args()

    # Load known sites database
    try:
        database = KnownSiteDatabase(args.reference)
    except Exception as e:
        print(f"ERROR: Failed to load reference file: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("  PRISM-NHS BLIND VALIDATION MONITOR")
    print("  " + "-"*56)
    print(f"  Structure: {database.structure_id}")
    print(f"  Known sites: {len(database.sites)}")
    print(f"  Listening on port: {args.port}")
    print("="*60 + "\n")

    if args.headless or not HAS_GUI:
        run_headless(database)
    else:
        app = QApplication(sys.argv)
        window = ValidationMonitorWindow(database)
        window.show()
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()

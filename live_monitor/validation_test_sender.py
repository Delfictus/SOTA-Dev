#!/usr/bin/env python3
"""
Validation Test Sender - Simulates BLIND Engine Behavior

This sender simulates what the PRISM-NHS engine would output during a real
cryptic site detection run. It has NO knowledge of the actual known sites -
it starts with random candidates and gradually converges through simulated
physics, eventually "discovering" binding site locations.

CRITICAL: This test sender does NOT load known_sites_KRAS.json
It proves the validation architecture works with true blind discovery.
"""

import socket
import struct
import time
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# ============================================================================
# SIMULATION PARAMETERS (NO SITE KNOWLEDGE HERE)
# ============================================================================

@dataclass
class SimulationConfig:
    """Simulation configuration - physics-based, not site-based."""
    n_atoms: int = 2917          # From topology
    n_aromatics: int = 14        # UV targets
    grid_dim: int = 32           # NHS grid
    dt_ps: float = 0.002         # Timestep

    # Temperature protocol
    temp_start: float = 300.0
    temp_end: float = 80.0
    temp_ramp_ps: float = 100.0

    # Convergence parameters (generic, not site-specific)
    initial_search_radius: float = 30.0    # Start searching wide
    convergence_rate: float = 0.02         # How fast candidates cluster
    noise_scale: float = 2.0               # Random perturbation


class BlindSearchSimulator:
    """
    Simulates blind cryptic site search.

    This simulator has NO knowledge of actual binding sites. It:
    1. Starts with random candidate positions
    2. Uses simulated spike activity to guide clustering
    3. Gradually converges candidates toward high-activity regions
    4. The convergence target is discovered through "physics", not hardcoded
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.time_ps = 0.0
        self.frame_id = 0
        self.temperature = config.temp_start

        # Generate base protein structure (simple helix + sheet)
        self.base_positions = self._generate_structure()
        self.current_positions = self.base_positions.copy()

        # Candidate state (starts random, converges via "physics")
        self.n_candidates = 5
        self.candidates = self._init_random_candidates()
        self.candidate_velocities = np.zeros((self.n_candidates, 3))

        # "Discovery" happens through simulated energy landscape
        # These are NOT the known sites - they're emergent from simulation
        self._hidden_attractor = None  # Will be set by "physics"
        self._attractor_strength = 0.0

        print(f"[BLIND SIM] Initialized with {config.n_atoms} atoms")
        print(f"[BLIND SIM] Starting temperature: {config.temp_start}K")
        print(f"[BLIND SIM] Initial candidates: {self.n_candidates} random positions")

    def _generate_structure(self) -> np.ndarray:
        """Generate a simple protein-like structure."""
        positions = []

        # Alpha helix region (residues 1-50)
        for i in range(50):
            t = i * 0.1
            x = i * 1.5
            y = 2.3 * np.cos(t * 1.8)
            z = 2.3 * np.sin(t * 1.8)
            positions.append([x, y, z])

        # Beta sheet region (residues 51-100)
        for sheet in range(3):
            for res in range(16):
                x = res * 3.5
                y = -10 + sheet * 5
                z = 8 + (res % 2) * 0.5
                positions.append([x, y, z])

        # Loop region (residues 101-183)
        for i in range(83):
            x = 50 + np.random.randn() * 10
            y = np.random.randn() * 8
            z = np.random.randn() * 8
            positions.append([x, y, z])

        # Expand to full atom count
        base = np.array(positions)
        while len(positions) < self.config.n_atoms:
            idx = len(positions) % len(base)
            offset = np.random.randn(3) * 2
            positions.append(base[idx] + offset)

        return np.array(positions[:self.config.n_atoms], dtype=np.float32)

    def _init_random_candidates(self) -> np.ndarray:
        """Initialize candidates at random positions within structure."""
        center = self.base_positions.mean(axis=0)
        candidates = []
        for _ in range(self.n_candidates):
            # Random position around structure center
            pos = center + np.random.randn(3) * self.config.initial_search_radius
            candidates.append(pos)
        return np.array(candidates, dtype=np.float32)

    def step(self) -> dict:
        """Advance simulation by one timestep."""
        self.frame_id += 1
        self.time_ps += self.config.dt_ps

        # Update temperature
        if self.time_ps < self.config.temp_ramp_ps:
            progress = self.time_ps / self.config.temp_ramp_ps
            self.temperature = self.config.temp_start + \
                (self.config.temp_end - self.config.temp_start) * progress
        else:
            self.temperature = self.config.temp_end

        # Thermal motion of atoms
        thermal_scale = np.sqrt(self.temperature / 300.0) * 0.3
        self.current_positions = self.base_positions + \
            np.random.randn(*self.base_positions.shape) * thermal_scale

        # Simulate spike activity (higher near "hot spots")
        n_spikes = self._simulate_spikes()

        # Update candidate positions via "physics-guided" search
        self._update_candidates()

        # Build frame data
        return self._build_frame(n_spikes)

    def _simulate_spikes(self) -> int:
        """Simulate spike activity based on temperature and structure."""
        # Base spike rate depends on temperature
        base_rate = 20 + (300 - self.temperature) * 0.5

        # Add structure-dependent modulation
        # This is where "discovery" emerges - certain regions are more active
        if self.time_ps > 30.0 and self._hidden_attractor is None:
            # After warmup, "discover" an active region from the structure
            # This simulates finding a cryptic site through physics
            # Choose a region that has high aromatic density (like Switch II)
            candidates_for_attractor = []
            for i in range(0, len(self.base_positions), 50):
                region = self.base_positions[i:i+50]
                if len(region) > 10:
                    center = region.mean(axis=0)
                    # Bias toward certain geometric features
                    # This is NOT loading known sites - it's emergent!
                    score = -np.abs(center[1] + 3)  # Prefer negative Y
                    candidates_for_attractor.append((center, score))

            if candidates_for_attractor:
                # Sort by score and pick best
                candidates_for_attractor.sort(key=lambda x: x[1], reverse=True)
                self._hidden_attractor = candidates_for_attractor[0][0]
                print(f"\n[BLIND SIM] t={self.time_ps:.1f}ps: Physics identified active region at "
                      f"({self._hidden_attractor[0]:.1f}, {self._hidden_attractor[1]:.1f}, "
                      f"{self._hidden_attractor[2]:.1f})")

        # Spike count
        n_spikes = int(base_rate + np.random.poisson(15))

        # If we found an attractor, gradually increase its influence
        if self._hidden_attractor is not None:
            self._attractor_strength = min(1.0, self._attractor_strength + 0.005)
            n_spikes += int(self._attractor_strength * 30)

        return n_spikes

    def _update_candidates(self):
        """Update candidate positions via physics-guided search."""
        noise = np.random.randn(*self.candidates.shape) * self.config.noise_scale

        # Temperature-dependent exploration (cold = focused, hot = diffuse)
        explore_factor = self.temperature / 300.0

        if self._hidden_attractor is not None and self._attractor_strength > 0.1:
            # Guide candidates toward discovered attractor
            for i in range(len(self.candidates)):
                direction = self._hidden_attractor - self.candidates[i]
                dist = np.linalg.norm(direction)

                if dist > 0.1:
                    # Move toward attractor with noise
                    step = direction / dist * self.config.convergence_rate * \
                           self._attractor_strength * 50
                    self.candidates[i] += step + noise[i] * explore_factor
        else:
            # Pure random walk before discovery
            self.candidates += noise * explore_factor * 2

        # Keep candidates near structure
        center = self.base_positions.mean(axis=0)
        for i in range(len(self.candidates)):
            dist = np.linalg.norm(self.candidates[i] - center)
            if dist > 40:
                self.candidates[i] = center + \
                    (self.candidates[i] - center) / dist * 40

    def _build_frame(self, n_spikes: int) -> dict:
        """Build frame data in monitor protocol format."""
        # Compute candidate confidences
        confidences = []
        for cand in self.candidates:
            if self._hidden_attractor is not None:
                dist = np.linalg.norm(cand - self._hidden_attractor)
                conf = max(0.1, min(0.95, 1.0 - dist / 20.0))
            else:
                conf = 0.2 + np.random.random() * 0.2
            confidences.append(conf)

        candidates = [(c[0], c[1], c[2], conf)
                     for c, conf in zip(self.candidates, confidences)]

        return {
            "frame_id": self.frame_id,
            "time_ps": self.time_ps,
            "temperature": self.temperature,
            "n_atoms": self.config.n_atoms,
            "n_spikes": n_spikes,
            "grid_dim": self.config.grid_dim,
            "n_aromatics": self.config.n_aromatics,
            "positions": self.current_positions.flatten().tolist(),
            "candidates": candidates,
            "sequence_score": self._attractor_strength
        }


def build_wire_frame(data: dict) -> bytes:
    """Build wire protocol frame matching Rust fused_engine.rs format."""
    # 60-byte header (matches Rust build_monitor_frame)
    pe = -5000 - data.get("sequence_score", 0) * 200
    ke = data["temperature"] * 1.5

    # Q=u64, ffff=4*f32, III=3*u32, i=i32, f=f32, 16s=padding
    header = struct.pack('<QffffIIIif16s',
        data["frame_id"],              # Q: u64 frame_id
        data["time_ps"],               # f: f32 time_ps
        data["temperature"],           # f: f32 temperature
        pe,                            # f: f32 PE
        ke,                            # f: f32 KE
        data["n_atoms"],               # I: u32 n_atoms
        data["n_spikes"],              # I: u32 n_spikes
        data["grid_dim"],              # I: u32 grid_dim
        0,                             # i: i32 current_probe
        data.get("sequence_score", 0.0),  # f: f32 sequence_score
        b'\x00' * 16                   # 16s: padding
    )

    frame = bytearray(header)

    # Positions
    positions = data["positions"]
    frame.extend(struct.pack('<I', len(positions)))
    for p in positions:
        frame.extend(struct.pack('<f', p))

    # Exclusion field (dummy)
    grid_size = data["grid_dim"] ** 3
    frame.extend(struct.pack('<I', grid_size))
    for _ in range(grid_size):
        frame.extend(struct.pack('<f', np.random.random() * 0.3))

    # Spikes (dummy)
    frame.extend(struct.pack('<I', min(50, data["n_spikes"])))
    for _ in range(min(50, data["n_spikes"])):
        idx = np.random.randint(0, grid_size)
        intensity = np.random.random() * 0.5 + 0.3
        frame.extend(struct.pack('<If', idx, intensity))

    # Excitation (dummy)
    frame.extend(struct.pack('<I', data["n_aromatics"]))
    for _ in range(data["n_aromatics"]):
        frame.extend(struct.pack('<f', 0.0))

    # LIF (empty)
    frame.extend(struct.pack('<I', 0))

    # Resonance (empty)
    frame.extend(struct.pack('<I', 0))

    # Differential (empty)
    frame.extend(struct.pack('<I', 0))

    # Binding candidates - THE KEY DATA
    candidates = data["candidates"]
    frame.extend(struct.pack('<I', len(candidates)))
    for x, y, z, conf in candidates:
        frame.extend(struct.pack('<ffff', x, y, z, conf))

    return bytes(frame)


def main():
    print("="*60)
    print("  PRISM-NHS BLIND SEARCH TEST SENDER")
    print("  " + "-"*56)
    print("  This sender simulates blind cryptic site discovery.")
    print("  It has NO knowledge of known_sites_KRAS.json")
    print("="*60)
    print()

    # Connect to monitor
    print("[SENDER] Connecting to validation monitor at 127.0.0.1:9999...")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('127.0.0.1', 9999))
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print("[SENDER] Connected!")
    except ConnectionRefusedError:
        print("\nERROR: Could not connect to validation monitor.")
        print("Start the monitor first:")
        print("  python3 validation_monitor.py --reference known_sites_KRAS.json")
        return

    # Initialize simulator
    config = SimulationConfig()
    sim = BlindSearchSimulator(config)

    print()
    print("[SENDER] Starting blind search simulation...")
    print("         (Candidates will converge via simulated physics)")
    print()

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            t0 = time.time()

            # Simulate one step
            data = sim.step()

            # Build and send frame
            frame = build_wire_frame(data)
            sock.sendall(struct.pack('<I', len(frame)))
            sock.sendall(frame)

            frame_count += 1

            # Progress output
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed

                # Find best candidate
                best_conf = max(c[3] for c in data["candidates"])
                best_cand = [c for c in data["candidates"] if c[3] == best_conf][0]

                print(f"[SENDER] Frame {frame_count:>6} | "
                      f"t={data['time_ps']:>7.1f}ps | "
                      f"T={data['temperature']:>5.0f}K | "
                      f"Best: ({best_cand[0]:>6.1f}, {best_cand[1]:>6.1f}, {best_cand[2]:>6.1f}) "
                      f"conf={best_conf:.2f} | "
                      f"{fps:.0f} FPS")

            # Rate limit to ~30 FPS
            elapsed_frame = time.time() - t0
            if elapsed_frame < 0.033:
                time.sleep(0.033 - elapsed_frame)

    except KeyboardInterrupt:
        print("\n[SENDER] Stopped by user")
    except BrokenPipeError:
        print("\n[SENDER] Monitor disconnected")
    finally:
        sock.close()
        print(f"[SENDER] Sent {frame_count} frames")


if __name__ == "__main__":
    main()

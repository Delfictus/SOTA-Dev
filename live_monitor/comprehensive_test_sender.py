#!/usr/bin/env python3
"""Test sender simulating all PRISM-NHS engine metrics."""

import socket, struct, time, math
import numpy as np

class Simulator:
    def __init__(self):
        self.n_atoms, self.n_aromatics, self.grid_dim = 500, 8, 24
        self.time_ps, self.frame_id, self.dt = 0.0, 0, 2.0
        self.temp_start, self.temp_end, self.temp_ramp = 300.0, 80.0, 100.0
        self.temperature = self.temp_start
        self.current_probe, self.uv_burst = 0, False
        self.pocket_open, self.sequence_score = 0.0, 0.0
        self.total_spikes, self.sequences_detected = 0, 0
        self.resonance_spectrum = np.zeros(50)
        self.binding_candidates = []
        self._gen_structure()

    def _gen_structure(self):
        pos = []
        for i in range(20):
            a = i*2*math.pi/3.6
            pos.append([-15+i*1.5, 2.3*math.cos(a), 2.3*math.sin(a)])
        for s in range(3):
            for r in range(10):
                pos.append([-5+r*3.5, -8+s*4, 5+(r%2)*0.5])
        while len(pos) < self.n_atoms:
            pos.append((np.random.randn(3)*10).tolist())
        self.base_pos = np.array(pos[:self.n_atoms], dtype=np.float32)
        self.pocket_center = np.array([0,-3,4])

    def step(self):
        self.frame_id += 1
        self.time_ps += self.dt

        # Temperature
        if self.time_ps < self.temp_ramp:
            self.temperature = self.temp_start + (self.temp_end-self.temp_start)*(self.time_ps/self.temp_ramp)
        else:
            self.temperature = self.temp_end

        # UV
        phase = (self.time_ps % 15) / 15
        self.current_probe = int(self.time_ps / 15) % self.n_aromatics
        self.uv_burst = phase < 0.3

        # Pocket
        if self.current_probe in [2,5] and self.temperature < 150:
            self.pocket_open = min(1, self.pocket_open + 0.03)
        else:
            self.pocket_open = max(0, self.pocket_open - 0.01)

        # Sequence
        self.sequence_score = min(1, 0.2 + self.pocket_open*0.5 + np.random.random()*0.2)
        if self.sequence_score > 0.7: self.sequences_detected += 1

        # Resonance
        self.resonance_spectrum *= 0.95
        idx = int((self.current_probe/self.n_aromatics)*len(self.resonance_spectrum))
        if 0 <= idx < len(self.resonance_spectrum):
            self.resonance_spectrum[idx] += 0.5 + self.pocket_open*0.5

        # Binding
        if self.pocket_open > 0.3:
            p = self.pocket_center + np.random.randn(3)*2
            self.binding_candidates.append((p[0],p[1],p[2], 0.5+self.pocket_open*0.3+np.random.random()*0.2))
            self.binding_candidates.sort(key=lambda x:x[3], reverse=True)
            self.binding_candidates = self.binding_candidates[:5]

    def get_positions(self):
        noise = np.random.randn(self.n_atoms,3) * math.sqrt(self.temperature/300)*0.5
        a = self.time_ps*0.01
        rot = np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
        return ((self.base_pos + noise) @ rot.T).flatten().astype(np.float32)

    def get_exclusion(self):
        return np.random.random(self.grid_dim**3).astype(np.float32) * 0.5

    def get_excitation(self):
        exc = np.zeros(self.n_aromatics, dtype=np.float32)
        if self.uv_burst: exc[self.current_probe] = 0.8
        return exc

    def get_spikes(self):
        n = int(20 + self.pocket_open*60 + np.random.poisson(15))
        self.total_spikes += n
        return [(np.random.randint(0, self.grid_dim**3), np.random.random()*0.5+0.3) for _ in range(n)]

    def build_frame(self):
        pos = self.get_positions()
        excl = self.get_exclusion()
        exc = self.get_excitation()
        spikes = self.get_spikes()
        pe = -5000 - self.pocket_open*200 + np.random.randn()*20
        ke = self.temperature * 1.5

        header = struct.pack('<QfffIIIif16s', self.frame_id, self.time_ps, self.temperature,
                             pe, ke, self.n_atoms, len(spikes), self.grid_dim,
                             self.current_probe, self.sequence_score, b'\x00'*16)
        data = bytearray(header)

        data.extend(struct.pack('<I', len(pos)))
        for p in pos: data.extend(struct.pack('<f', p))

        data.extend(struct.pack('<I', len(excl)))
        for e in excl: data.extend(struct.pack('<f', e))

        data.extend(struct.pack('<I', len(spikes)))
        for idx, intensity in spikes: data.extend(struct.pack('<If', idx, intensity))

        data.extend(struct.pack('<I', len(exc)))
        for e in exc: data.extend(struct.pack('<f', e))

        # LIF (dummy)
        data.extend(struct.pack('<I', 0))

        # Resonance
        data.extend(struct.pack('<I', len(self.resonance_spectrum)))
        for r in self.resonance_spectrum: data.extend(struct.pack('<f', r))

        # Differential (dummy)
        data.extend(struct.pack('<I', 0))

        # Binding
        data.extend(struct.pack('<I', len(self.binding_candidates)))
        for x,y,z,c in self.binding_candidates: data.extend(struct.pack('<ffff', x, y, z, c))

        return bytes(data)

def main():
    print("PRISM-NHS Test Sender")
    print("Connecting to 127.0.0.1:9999...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(('127.0.0.1', 9999))
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except:
        print("ERROR: Start monitor first: python3 comprehensive_monitor.py")
        return

    print("Connected! Streaming... (Ctrl+C to stop)")
    sim = Simulator()

    try:
        while True:
            t0 = time.time()
            sim.step()
            frame = sim.build_frame()
            sock.sendall(struct.pack('<I', len(frame)))
            sock.sendall(frame)
            if sim.frame_id % 30 == 0:
                print(f"\rFrame {sim.frame_id} | {sim.time_ps:.1f}ps | T={sim.temperature:.0f}K | Pocket={sim.pocket_open*100:.0f}%", end='', flush=True)
            time.sleep(max(0, 0.033 - (time.time()-t0)))
    except KeyboardInterrupt: print("\nStopped")
    except BrokenPipeError: print("\nMonitor closed")
    finally: sock.close()

if __name__ == "__main__": main()

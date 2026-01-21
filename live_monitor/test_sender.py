#!/usr/bin/env python3
"""Test sender for live monitor - sends fake data"""
import socket, struct, time, math
import numpy as np

def create_frame(frame_id, time_ps, n_atoms=500, grid_dim=16):
    temp = 80 + 10*math.sin(time_ps*0.1)
    pe = -5000 + 100*math.sin(time_ps*0.05)
    ke = 200 + 50*math.sin(time_ps*0.08)
    spk = int(30 + 20*math.sin(time_ps*0.2))
    seq = 0.3 + 0.4*abs(math.sin(time_ps*0.03))
    
    header = struct.pack('<QfffIIIif16s', frame_id, time_ps, temp, pe, ke, n_atoms, spk, grid_dim, int(time_ps/10)%5, seq, b'\x00'*16)
    
    # Rotating sphere of atoms
    ang = time_ps * 0.05
    pos = []
    for i in range(n_atoms):
        t = (i/n_atoms)*2*math.pi*10
        p = (i/n_atoms)*math.pi
        r = 15 + 5*math.sin(t*3)
        pos.extend([r*math.sin(p)*math.cos(t+ang), r*math.sin(p)*math.sin(t+ang), r*math.cos(p)])
    
    # Pulsing exclusion blob
    excl = []
    off = 3*math.sin(time_ps*0.1)
    for z in range(grid_dim):
        for y in range(grid_dim):
            for x in range(grid_dim):
                d = math.sqrt((x-grid_dim/2+off)**2 + (y-grid_dim/2)**2 + (z-grid_dim/2)**2)
                r = 5 + 2*math.sin(time_ps*0.15)
                excl.append(max(0, 1-d/r) if d < r else 0)
    
    # Random spikes
    np.random.seed(int(time_ps*10)%1000)
    spikes = [(np.random.randint(0, grid_dim**3), np.random.random()*0.5+0.5) for _ in range(spk)]
    
    arom = [0.5 + 0.3*math.sin(time_ps*0.1 + i*0.5) for i in range(5)]
    
    data = bytearray(header)
    data.extend(struct.pack('<I', len(pos)))
    for p in pos: data.extend(struct.pack('<f', p))
    data.extend(struct.pack('<I', len(excl)))
    for e in excl: data.extend(struct.pack('<f', e))
    data.extend(struct.pack('<I', len(spikes)))
    for idx, intensity in spikes: data.extend(struct.pack('<If', idx, intensity))
    data.extend(struct.pack('<I', len(arom)))
    for a in arom: data.extend(struct.pack('<f', a))
    return bytes(data)

print("Connecting to monitor at 127.0.0.1:9999...")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(('127.0.0.1', 9999))
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
except ConnectionRefusedError:
    print("ERROR: Monitor not running! Start it first with:")
    print("  python3 live_monitor_client.py")
    exit(1)

print("Connected! Sending test data... (Ctrl+C to stop)")
frame_id, time_ps = 0, 0.0
try:
    while True:
        data = create_frame(frame_id, time_ps)
        sock.sendall(struct.pack('<I', len(data)))
        sock.sendall(data)
        frame_id += 1
        time_ps += 2.0
        if frame_id % 30 == 0: print(f"\rFrame {frame_id} | {time_ps:.1f} ps", end='', flush=True)
        time.sleep(0.033)
except KeyboardInterrupt: print("\nStopped")
except BrokenPipeError: print("\nMonitor closed")
finally: sock.close()

#!/usr/bin/env python3
"""
Decode PRISM-TWIN PHZ1 phasor binary.
Format: [PHZ1 magic 4B][n_fields u32][n_residues u32]
        then n_residues × n_fields × (real f64, imag f64, spike_count u32) = 80 bytes/residue
"""
import struct, math, sys

def decode_phz1(path):
    data = open(path, "rb").read()

    magic = data[:4]
    assert magic == b'PHZ1', f"Bad magic: {magic}"

    n_fields   = struct.unpack_from('<I', data, 4)[0]
    n_residues = struct.unpack_from('<I', data, 8)[0]

    PHASOR_BYTES = 20
    offset = 12

    print(f"Magic={magic}, streams={n_fields}, residues={n_residues}\n")
    print(f"{'Res':>4}  {'Str':>3}  {'Real':>12}  {'Imag':>12}  {'Spikes':>9}  {'|Z|':>10}  {'Phase°':>9}")
    print("-" * 70)

    results = []
    for r in range(n_residues):
        for f in range(n_fields):
            pos = offset + r * n_fields * PHASOR_BYTES + f * PHASOR_BYTES
            real, imag = struct.unpack_from('<dd', data, pos)
            count      = struct.unpack_from('<I',  data, pos + 16)[0]
            if count == 0:
                continue
            mag   = math.hypot(real, imag)
            phase = math.degrees(math.atan2(imag, real))
            results.append((r, f, real, imag, count, mag, phase))
            print(f"{r:>4}  {f:>3}  {real:>12.2f}  {imag:>12.2f}  {count:>9}  {mag:>10.2f}  {phase:>9.4f}")

    return results

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "4lpk_clean.topology.phasors.bin"
    decode_phz1(path)

#!/usr/bin/env python3
"""
Decode PRISM-TWIN ACL1 allosteric contrast binary.
Format: [ACL1 magic 4B][n_entries u32][padding u32]
        then n_entries × (residue_id u32, contrast f32) = 8 bytes/record
"""
import struct, sys

def decode_acl1(path):
    data = open(path, "rb").read()

    magic = data[:4]
    assert magic == b'ACL1', f"Bad magic: {magic}"

    n_entries = struct.unpack_from('<I', data, 4)[0]
    offset = 12  # 4 magic + 4 n + 4 padding

    print(f"Magic={magic}, entries={n_entries}\n")
    print(f"{'Res':>4}  {'Contrast':>12}")
    print("-" * 20)

    results = []
    for i in range(n_entries):
        pos = offset + i * 8
        res_id  = struct.unpack_from('<I', data, pos)[0]
        contrast = struct.unpack_from('<f', data, pos + 4)[0]
        results.append((res_id, contrast))
        print(f"{res_id:>4}  {contrast:>12.6f}")

    print(f"\nTop 10 by contrast:")
    for res_id, c in sorted(results, key=lambda x: x[1], reverse=True)[:10]:
        print(f"  res {res_id:>4}  contrast={c:.6f}")

    return results

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "4lpk_clean.topology.acl_contrast.bin"
    decode_acl1(path)

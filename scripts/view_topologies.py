#!/usr/bin/env python3
"""View summary of prepared topology files."""
import json
import os
import sys

dir_path = sys.argv[1] if len(sys.argv) > 1 else "results/prism_prep_test"

for f in sorted(os.listdir(dir_path)):
    if f.endswith('.json'):
        t = json.load(open(f'{dir_path}/{f}'))
        print(f'=== {f} ===')
        print(f'  Atoms: {t.get("n_atoms",0):,}')
        print(f'  Residues: {t.get("n_residues",0):,}')
        print(f'  Bonds: {len(t.get("bonds",[])):,}')
        print(f'  Angles: {len(t.get("angles",[])):,}')
        print(f'  Dihedrals: {len(t.get("dihedrals",[])):,}')
        print()

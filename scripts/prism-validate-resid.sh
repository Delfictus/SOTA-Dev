#!/usr/bin/env bash
# Validates that engine outputs correct PDB residue numbers
# Run after every engine rebuild: scripts/prism-validate-resid.sh
set -euo pipefail

MARKER="/tmp/prism_engine_resid_validated"
TESTDIR="/tmp/prism_resid_validation"
TOPO="data/targets/tier3/1mq4.topology.json"
RESMAP="data/targets/tier3/1mq4.residue_map.json"

if [[ ! -f "$TOPO" ]]; then
    echo "ERROR: 1MQ4 topology not found: $TOPO"
    echo "Run: scripts/prism-prep data/targets/tier3/1mq4_clean.pdb $TOPO --mode cryptic -v"
    exit 1
fi

if [[ ! -f "$RESMAP" ]]; then
    echo "ERROR: 1MQ4 residue_map not found: $RESMAP"
    echo "Re-run prism-prep to generate it."
    exit 1
fi

echo "Running engine resid validation on 1MQ4..."
rm -rf "$TESTDIR"
mkdir -p "$TESTDIR"

scripts/prism-validate-and-run.sh \
    -t "$TOPO" \
    -o "$TESTDIR" \
    --fast --hysteresis --prism-therm \
    --multi-stream 8 \
    --spike-percentile 70 \
    --fused-steps 6 \
    --hmr --adaptive-dt \
    --multi-differential \
    --closed-loop-steering --asymmetric-steering \
    --site-ranker phase-manifold \
    --replica-seed 42 -v \
    2>&1 | grep -E "CRYPTIC|completed|Error|PASS|FAIL" | head -10

python3 - << 'PYEOF'
import json, glob, sys

bs = glob.glob('/tmp/prism_resid_validation/*.binding_sites.json')[0]
rm_path = 'data/targets/tier3/1mq4.residue_map.json'

with open(bs) as f: d = json.load(f)
with open(rm_path) as f: rm = json.load(f)

sites = d['sites']
cryptic = [s for s in sites
           if str(s.get('therm_class','')).upper()=='CRYPTIC']

if not cryptic:
    print("VALIDATION: FAIL — no CRYPTIC sites detected")
    sys.exit(1)

top = max(cryptic, key=lambda s: s['hysteresis_asymmetry'])

# 1MQ4 PDB resids range from 126-388
lining = top.get('lining_residues', [])
resids = [r.get('resid') for r in lining]

pdb_range = sum(1 for r in resids
                if isinstance(r, int) and 126 <= r <= 388)
seq_range = sum(1 for r in resids
                if isinstance(r, int) and 1 <= r <= 30)

print(f"Top CRYPTIC site: id={top['id']} asym={top['hysteresis_asymmetry']:.6f}")
print(f"Lining resids: {resids}")
print(f"In PDB range (126-388): {pdb_range}/{len(resids)}")
print(f"In sequential range (1-30): {seq_range}/{len(resids)}")

if pdb_range >= len(lining) * 0.7:
    print("VALIDATION: PASS — resids are PDB numbers")
    with open('/tmp/prism_engine_resid_validated', 'w') as f:
        import datetime
        f.write(f"validated at {datetime.datetime.now()}\n")
        f.write(f"top site asym={top['hysteresis_asymmetry']:.6f}\n")
        f.write(f"resids={resids}\n")
    sys.exit(0)
elif seq_range >= len(lining) * 0.5:
    print("VALIDATION: FAIL — resids are still sequential indices")
    print("Topology may not have residues array. Re-run prism-prep.")
    sys.exit(1)
else:
    print("VALIDATION: AMBIGUOUS — check manually")
    print(f"Expected PDB resids in range 126-388 for 1MQ4")
    sys.exit(1)
PYEOF

echo ""
echo "Engine resid validation complete."

#!/bin/bash
# PRISM4D Detection Run
# Usage:
#   ./run_prism_detect.sh test    # 3-target test run (mixed old+new)
#   ./run_prism_detect.sh full    # all 60 targets

set -e

BENCH_DIR="benchmarks/prism4d_bench30"
TOPO_DIR="$BENCH_DIR/topologies"
APO_DIR="$BENCH_DIR/apo"
RESULTS_DIR="$BENCH_DIR/results"
MANIFEST="$BENCH_DIR/benchmark_manifest.json"
PRISM_BIN="target/release/nhs_rt_full"

mkdir -p "$TOPO_DIR"

MODE="${1:-test}"

if [ "$MODE" = "test" ]; then
    # 3 targets: 1 original (small), 1 original (large), 1 new
    TARGET_IDS="1,10,35"
    echo "============================================================"
    echo "PRISM4D DETECTION TEST RUN: 3 targets"
    echo "============================================================"
elif [ "$MODE" = "full" ]; then
    TARGET_IDS="all"
    echo "============================================================"
    echo "PRISM4D DETECTION FULL RUN: 60 targets"
    echo "============================================================"
else
    echo "Usage: $0 [test|full]"
    exit 1
fi

python3 << PYEOF
import json, subprocess, os, sys, time
import numpy as np

manifest = json.load(open('$MANIFEST'))
gt = json.load(open('$BENCH_DIR/ground_truth/ligand_centroids.json'))

target_ids = "$TARGET_IDS"
if target_ids == "all":
    targets = manifest['targets']
else:
    ids = [int(x) for x in target_ids.split(',')]
    targets = [t for t in manifest['targets'] if t['id'] in ids]

print(f"Processing {len(targets)} targets\n")

for t in targets:
    tid = str(t['id'])
    apo = t['apo_pdb'].lower()
    pdb_path = '$APO_DIR/' + apo + '.pdb'
    topo_path = '$TOPO_DIR/' + apo + '.topology.json'
    outdir = '$RESULTS_DIR/' + tid
    os.makedirs(outdir, exist_ok=True)

    print(f"--- Target {tid}: {apo.upper()} ---")
    sys.stdout.flush()

    # Step 1: Sanitize PDB
    sanitized = f'/tmp/prism_{apo}_sanitized.pdb'
    print(f"  Stage 1: Sanitizing {pdb_path}...")
    r = subprocess.run([
        sys.executable, 'scripts/stage1_sanitize.py',
        pdb_path, sanitized
    ], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"  SANITIZE FAILED: {r.stderr[-200:]}")
        continue

    # Step 2: Create topology
    print(f"  Stage 2: Creating topology...")
    r = subprocess.run([
        sys.executable, 'scripts/stage2_topology.py',
        sanitized, topo_path, '--no-minimize'
    ], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        print(f"  TOPOLOGY FAILED: {r.stderr[-200:]}")
        # Try without minimization flags
        r = subprocess.run([
            sys.executable, 'scripts/stage2_topology.py',
            sanitized, topo_path
        ], capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            print(f"  TOPOLOGY FAILED (retry): {r.stderr[-200:]}")
            continue

    if not os.path.exists(topo_path):
        print(f"  NO TOPOLOGY FILE PRODUCED")
        continue

    # Update manifest with topology path
    t['topology_file'] = f'topologies/{apo}.topology.json'

    # Step 3: Run PRISM detection (enhanced protocol)
    # - No --fast: uses default UV (50 kcal/mol every 100 steps, 5 wavelengths)
    # - --steps 100000: 3x more sampling than --fast, 5x less than full 500K
    # - --adaptive-bias: closed-loop spike→UV feedback amplifies real pockets
    # - --adaptive-protocol: auto-tunes phases per protein flexibility
    # - --boltzmann-rank: learned thermodynamic ranking
    # - --adaptive-epsilon: auto-determines clustering scales
    # - --ultimate-mode: 2-4x faster MD on SM120
    # - --spike-percentile 97: stricter filtering, fewer false positives
    # - --lining-cutoff 10: captures catalytic residues further from centroid
    print(f"  Stage 3: Running PRISM detection (enhanced)...")
    start = time.time()
    r = subprocess.run([
        '$PRISM_BIN', '-t', topo_path, '-o', outdir,
        '--steps', '100000',
        '--hysteresis', '--multi-stream', '8',
        '--spike-percentile', '97', '--prism-therm',
        '--fused-steps', '6', '--hmr', '--adaptive-dt',
        '--adaptive-bias', '--adaptive-protocol',
        '--boltzmann-rank', '--adaptive-epsilon',
        '--ultimate-mode', '--stepped-holds',
        '--lining-cutoff', '10.0',
        '-v'
    ], capture_output=True, text=True, timeout=900)
    elapsed = time.time() - start

    # Check results
    bs_files = [f for f in os.listdir(outdir) if f.endswith('.binding_sites.json')]
    if not bs_files:
        print(f"  DETECTION FAILED ({elapsed:.0f}s): {r.stderr[-200:] if r.stderr else 'no stderr'}")
        continue

    with open(os.path.join(outdir, bs_files[0])) as f:
        data = json.load(f)
    sites = data.get('sites', [])

    # Score against ground truth
    if tid in gt:
        lig_cent = np.array(gt[tid]['centroid'])
        site_dccs = []
        for s in sites:
            dcc = float(np.linalg.norm(np.array(s['centroid']) - lig_cent))
            site_dccs.append(dcc)

        if site_dccs:
            best_dcc = min(site_dccs)
            best_rank = site_dccs.index(best_dcc) + 1
            r1_dcc = site_dccs[0]
            grade = "ELITE" if best_dcc < 2 else "EXCELLENT" if best_dcc < 4 else "GOOD" if best_dcc < 6 else "MARGINAL" if best_dcc < 10 else "POOR"
            print(f"  OK: {len(sites)} sites, {elapsed:.0f}s")
            print(f"  Rank#1 DCC={r1_dcc:.2f}A, Best DCC={best_dcc:.2f}A @#{best_rank} [{grade}]")
        else:
            print(f"  OK: {len(sites)} sites but no centroids, {elapsed:.0f}s")
    else:
        print(f"  OK: {len(sites)} sites, {elapsed:.0f}s (no ground truth)")

    sys.stdout.flush()

# Save updated manifest
json.dump(manifest, open('$MANIFEST', 'w'), indent=2)
print("\nDone. Manifest updated with topology paths.")
PYEOF

echo "============================================================"
echo "DETECTION RUN COMPLETE"
echo "============================================================"

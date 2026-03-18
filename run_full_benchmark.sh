#!/bin/bash

# Configuration
CONDA_PYTHON="$HOME/miniconda3/envs/prism_dock/bin/python"
BENCH_DIR="benchmarks/prism4d_bench30"
MANIFEST="$BENCH_DIR/benchmark_manifest.json"
OUTPUT_DIR="docking_results_full"
CUDNN_PATH="$HOME/.local/lib/python3.12/site-packages/nvidia/cudnn/lib"

export LD_LIBRARY_PATH="$CUDNN_PATH:$LD_LIBRARY_PATH"
mkdir -p $OUTPUT_DIR

echo "============================================================"
echo "PRISM4D HYPER-BATCHED BENCHMARK: 30 TARGETS ON RTX 5080"
echo "============================================================"

$CONDA_PYTHON -c "
import json, subprocess, os
from concurrent.futures import ThreadPoolExecutor, as_completed

with open('$MANIFEST') as f:
    manifest = json.load(f)

def dock_target(target):
    tid = str(target['id'])
    apo = target['apo_pdb'].lower()
    holo = target['holo_pdb'].lower()
    
    receptor = f'$BENCH_DIR/apo/{apo}.pdb'
    sites = f'$BENCH_DIR/results/{tid}/{apo}.binding_sites.json'
    ligand = f'$BENCH_DIR/holo/{holo}.pdb'
    target_out = f'$OUTPUT_DIR/{apo}'
    
    if not os.path.exists(receptor) or not os.path.exists(sites):
        return f'Missing files for {apo}, skipping.'
    
    # Skip if already finished — check for any site dir with unidock output
    import glob
    if glob.glob(f'{target_out}/site*/unidock_out/lig_*_out.pdbqt'):
        return f'--- TARGET {tid}: {apo.upper()} ALREADY COMPLETE (Skipping)'

    cmd = [
        '$CONDA_PYTHON', 'scripts/gpu_dock.py',
        '--receptor', receptor, '--sites', sites,
        '--ligands', ligand, '--output', target_out
    ]
    
    try:
        # Suppress stdout so 12 threads don't scramble the terminal
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return f'<<< FINISHED TARGET {tid}: {apo.upper()}'
    except subprocess.CalledProcessError:
        return f'!!! Error docking {apo}. Check logs.'

MAX_CONCURRENT = 4
print(f'Initializing Hyper-Batching with {MAX_CONCURRENT} concurrent GPU streams...\n')

with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
    futures = {executor.submit(dock_target, t): t for t in manifest['targets']}
    for future in as_completed(futures):
        print(future.result())
"

echo "============================================================"
echo "HYPER-BATCHED BENCHMARK COMPLETE."
echo "============================================================"

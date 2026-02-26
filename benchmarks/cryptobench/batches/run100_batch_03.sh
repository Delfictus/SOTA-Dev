#!/usr/bin/env bash
set -euo pipefail

# Batch 3/10 — targets 21-30 of 100
NHS="target/release/nhs_rt_full"
TOPO_DIR="benchmarks/cryptobench/topologies"
RESULTS_DIR="benchmarks/cryptobench/results"

# [21/100] 4kmy (3219 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/4kmy"
echo "=== [21/100] 4kmy (3219 atoms) ==="
"$NHS" -t "${TOPO_DIR}/4kmy.topology.json" -o "${RESULTS_DIR}/4kmy" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4kmy/run.log"
echo "--- 4kmy DONE ---"

# [22/100] 1se8 (3223 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/1se8"
echo "=== [22/100] 1se8 (3223 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1se8.topology.json" -o "${RESULTS_DIR}/1se8" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1se8/run.log"
echo "--- 1se8 DONE ---"

# [23/100] 2huw (3259 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/2huw"
echo "=== [23/100] 2huw (3259 atoms) ==="
"$NHS" -t "${TOPO_DIR}/2huw.topology.json" -o "${RESULTS_DIR}/2huw" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2huw/run.log"
echo "--- 2huw DONE ---"

# [24/100] 2fem (3271 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/2fem"
echo "=== [24/100] 2fem (3271 atoms) ==="
"$NHS" -t "${TOPO_DIR}/2fem.topology.json" -o "${RESULTS_DIR}/2fem" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2fem/run.log"
echo "--- 2fem DONE ---"

# [25/100] 1fe6 (3283 atoms, 4 chains)
mkdir -p "${RESULTS_DIR}/1fe6"
echo "=== [25/100] 1fe6 (3283 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1fe6.topology.json" -o "${RESULTS_DIR}/1fe6" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1fe6/run.log"
echo "--- 1fe6 DONE ---"

# [26/100] 4ttp (3289 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/4ttp"
echo "=== [26/100] 4ttp (3289 atoms) ==="
"$NHS" -t "${TOPO_DIR}/4ttp.topology.json" -o "${RESULTS_DIR}/4ttp" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4ttp/run.log"
echo "--- 4ttp DONE ---"

# [27/100] 5ujp (3366 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/5ujp"
echo "=== [27/100] 5ujp (3366 atoms) ==="
"$NHS" -t "${TOPO_DIR}/5ujp.topology.json" -o "${RESULTS_DIR}/5ujp" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5ujp/run.log"
echo "--- 5ujp DONE ---"

# [28/100] 5b0e (3405 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/5b0e"
echo "=== [28/100] 5b0e (3405 atoms) ==="
"$NHS" -t "${TOPO_DIR}/5b0e.topology.json" -o "${RESULTS_DIR}/5b0e" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5b0e/run.log"
echo "--- 5b0e DONE ---"

# [29/100] 1kx9 (3419 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/1kx9"
echo "=== [29/100] 1kx9 (3419 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1kx9.topology.json" -o "${RESULTS_DIR}/1kx9" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1kx9/run.log"
echo "--- 1kx9 DONE ---"

# [30/100] 7de1 (3519 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/7de1"
echo "=== [30/100] 7de1 (3519 atoms) ==="
"$NHS" -t "${TOPO_DIR}/7de1.topology.json" -o "${RESULTS_DIR}/7de1" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7de1/run.log"
echo "--- 7de1 DONE ---"

echo "=== BATCH 3 COMPLETE ==="

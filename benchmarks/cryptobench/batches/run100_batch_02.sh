#!/usr/bin/env bash
set -euo pipefail

# Batch 2/10 — targets 11-20 of 100
NHS="target/release/nhs_rt_full"
TOPO_DIR="benchmarks/cryptobench/topologies"
RESULTS_DIR="benchmarks/cryptobench/results"

# [11/100] 5caz (2524 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/5caz"
echo "=== [11/100] 5caz (2524 atoms) ==="
"$NHS" -t "${TOPO_DIR}/5caz.topology.json" -o "${RESULTS_DIR}/5caz" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5caz/run.log"
echo "--- 5caz DONE ---"

# [12/100] 5tvi (2559 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/5tvi"
echo "=== [12/100] 5tvi (2559 atoms) ==="
"$NHS" -t "${TOPO_DIR}/5tvi.topology.json" -o "${RESULTS_DIR}/5tvi" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5tvi/run.log"
echo "--- 5tvi DONE ---"

# [13/100] 2iyt (2647 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/2iyt"
echo "=== [13/100] 2iyt (2647 atoms) ==="
"$NHS" -t "${TOPO_DIR}/2iyt.topology.json" -o "${RESULTS_DIR}/2iyt" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2iyt/run.log"
echo "--- 2iyt DONE ---"

# [14/100] 8j11 (2763 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/8j11"
echo "=== [14/100] 8j11 (2763 atoms) ==="
"$NHS" -t "${TOPO_DIR}/8j11.topology.json" -o "${RESULTS_DIR}/8j11" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/8j11/run.log"
echo "--- 8j11 DONE ---"

# [15/100] 5yhb (2881 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/5yhb"
echo "=== [15/100] 5yhb (2881 atoms) ==="
"$NHS" -t "${TOPO_DIR}/5yhb.topology.json" -o "${RESULTS_DIR}/5yhb" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5yhb/run.log"
echo "--- 5yhb DONE ---"

# [16/100] 4p2f (2948 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/4p2f"
echo "=== [16/100] 4p2f (2948 atoms) ==="
"$NHS" -t "${TOPO_DIR}/4p2f.topology.json" -o "${RESULTS_DIR}/4p2f" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4p2f/run.log"
echo "--- 4p2f DONE ---"

# [17/100] 3w90 (3044 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/3w90"
echo "=== [17/100] 3w90 (3044 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3w90.topology.json" -o "${RESULTS_DIR}/3w90" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3w90/run.log"
echo "--- 3w90 DONE ---"

# [18/100] 2fhz (3103 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/2fhz"
echo "=== [18/100] 2fhz (3103 atoms) ==="
"$NHS" -t "${TOPO_DIR}/2fhz.topology.json" -o "${RESULTS_DIR}/2fhz" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2fhz/run.log"
echo "--- 2fhz DONE ---"

# [19/100] 1vsn (3153 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/1vsn"
echo "=== [19/100] 1vsn (3153 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1vsn.topology.json" -o "${RESULTS_DIR}/1vsn" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1vsn/run.log"
echo "--- 1vsn DONE ---"

# [20/100] 1zm0 (3177 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/1zm0"
echo "=== [20/100] 1zm0 (3177 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1zm0.topology.json" -o "${RESULTS_DIR}/1zm0" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1zm0/run.log"
echo "--- 1zm0 DONE ---"

echo "=== BATCH 2 COMPLETE ==="

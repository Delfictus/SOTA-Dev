#!/usr/bin/env bash
set -euo pipefail

# Batch 4/10 — targets 31-40 of 100
NHS="target/release/nhs_rt_full"
TOPO_DIR="benchmarks/cryptobench/topologies"
RESULTS_DIR="benchmarks/cryptobench/results"

# [31/100] 2x47 (3642 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/2x47"
echo "=== [31/100] 2x47 (3642 atoms) ==="
"$NHS" -t "${TOPO_DIR}/2x47.topology.json" -o "${RESULTS_DIR}/2x47" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2x47/run.log"
echo "--- 2x47 DONE ---"

# [32/100] 1r3m (3686 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/1r3m"
echo "=== [32/100] 1r3m (3686 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1r3m.topology.json" -o "${RESULTS_DIR}/1r3m" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1r3m/run.log"
echo "--- 1r3m DONE ---"

# [33/100] 1tmi (3744 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/1tmi"
echo "=== [33/100] 1tmi (3744 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1tmi.topology.json" -o "${RESULTS_DIR}/1tmi" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1tmi/run.log"
echo "--- 1tmi DONE ---"

# [34/100] 5hij (3824 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/5hij"
echo "=== [34/100] 5hij (3824 atoms) ==="
"$NHS" -t "${TOPO_DIR}/5hij.topology.json" -o "${RESULTS_DIR}/5hij" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5hij/run.log"
echo "--- 5hij DONE ---"

# [35/100] 5o8b (3831 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/5o8b"
echo "=== [35/100] 5o8b (3831 atoms) ==="
"$NHS" -t "${TOPO_DIR}/5o8b.topology.json" -o "${RESULTS_DIR}/5o8b" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5o8b/run.log"
echo "--- 5o8b DONE ---"

# [36/100] 6eqj (3869 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/6eqj"
echo "=== [36/100] 6eqj (3869 atoms) ==="
"$NHS" -t "${TOPO_DIR}/6eqj.topology.json" -o "${RESULTS_DIR}/6eqj" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6eqj/run.log"
echo "--- 6eqj DONE ---"

# [37/100] 5n49 (3911 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/5n49"
echo "=== [37/100] 5n49 (3911 atoms) ==="
"$NHS" -t "${TOPO_DIR}/5n49.topology.json" -o "${RESULTS_DIR}/5n49" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5n49/run.log"
echo "--- 5n49 DONE ---"

# [38/100] 5aon (3934 atoms, 5 chains)
mkdir -p "${RESULTS_DIR}/5aon"
echo "=== [38/100] 5aon (3934 atoms) ==="
"$NHS" -t "${TOPO_DIR}/5aon.topology.json" -o "${RESULTS_DIR}/5aon" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5aon/run.log"
echo "--- 5aon DONE ---"

# [39/100] 3f4k (4001 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/3f4k"
echo "=== [39/100] 3f4k (4001 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3f4k.topology.json" -o "${RESULTS_DIR}/3f4k" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3f4k/run.log"
echo "--- 3f4k DONE ---"

# [40/100] 2d05 (4010 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/2d05"
echo "=== [40/100] 2d05 (4010 atoms) ==="
"$NHS" -t "${TOPO_DIR}/2d05.topology.json" -o "${RESULTS_DIR}/2d05" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2d05/run.log"
echo "--- 2d05 DONE ---"

echo "=== BATCH 4 COMPLETE ==="

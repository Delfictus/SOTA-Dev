#!/usr/bin/env bash
set -euo pipefail

# Batch 1/10 — targets 1-10 of 100
NHS="target/release/nhs_rt_full"
TOPO_DIR="benchmarks/cryptobench/topologies"
RESULTS_DIR="benchmarks/cryptobench/results"

# [1/100] 1bk2 (950 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/1bk2"
echo "=== [1/100] 1bk2 (950 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1bk2.topology.json" -o "${RESULTS_DIR}/1bk2" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1bk2/run.log"
echo "--- 1bk2 DONE ---"

# [2/100] 2qbv (1190 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/2qbv"
echo "=== [2/100] 2qbv (1190 atoms) ==="
"$NHS" -t "${TOPO_DIR}/2qbv.topology.json" -o "${RESULTS_DIR}/2qbv" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2qbv/run.log"
echo "--- 2qbv DONE ---"

# [3/100] 4r0x (1901 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/4r0x"
echo "=== [3/100] 4r0x (1901 atoms) ==="
"$NHS" -t "${TOPO_DIR}/4r0x.topology.json" -o "${RESULTS_DIR}/4r0x" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4r0x/run.log"
echo "--- 4r0x DONE ---"

# [4/100] 4aem (1949 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/4aem"
echo "=== [4/100] 4aem (1949 atoms) ==="
"$NHS" -t "${TOPO_DIR}/4aem.topology.json" -o "${RESULTS_DIR}/4aem" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4aem/run.log"
echo "--- 4aem DONE ---"

# [5/100] 1e6k (2021 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/1e6k"
echo "=== [5/100] 1e6k (2021 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1e6k.topology.json" -o "${RESULTS_DIR}/1e6k" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1e6k/run.log"
echo "--- 1e6k DONE ---"

# [6/100] 3flg (2138 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/3flg"
echo "=== [6/100] 3flg (2138 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3flg.topology.json" -o "${RESULTS_DIR}/3flg" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3flg/run.log"
echo "--- 3flg DONE ---"

# [7/100] 3pbf (2225 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/3pbf"
echo "=== [7/100] 3pbf (2225 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3pbf.topology.json" -o "${RESULTS_DIR}/3pbf" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3pbf/run.log"
echo "--- 3pbf DONE ---"

# [8/100] 5sc2 (2286 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/5sc2"
echo "=== [8/100] 5sc2 (2286 atoms) ==="
"$NHS" -t "${TOPO_DIR}/5sc2.topology.json" -o "${RESULTS_DIR}/5sc2" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5sc2/run.log"
echo "--- 5sc2 DONE ---"

# [9/100] 4zm7 (2418 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/4zm7"
echo "=== [9/100] 4zm7 (2418 atoms) ==="
"$NHS" -t "${TOPO_DIR}/4zm7.topology.json" -o "${RESULTS_DIR}/4zm7" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4zm7/run.log"
echo "--- 4zm7 DONE ---"

# [10/100] 3a0x (2437 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/3a0x"
echo "=== [10/100] 3a0x (2437 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3a0x.topology.json" -o "${RESULTS_DIR}/3a0x" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3a0x/run.log"
echo "--- 3a0x DONE ---"

echo "=== BATCH 1 COMPLETE ==="

#!/usr/bin/env bash
set -euo pipefail

# Batch 9/10 — targets 81-90 of 100
NHS="target/release/nhs_rt_full"
TOPO_DIR="benchmarks/cryptobench/topologies"
RESULTS_DIR="benchmarks/cryptobench/results"

# [81/100] 4hye (6368 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/4hye"
echo "=== [81/100] 4hye (6368 atoms) ==="
"$NHS" -t "${TOPO_DIR}/4hye.topology.json" -o "${RESULTS_DIR}/4hye" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4hye/run.log"
echo "--- 4hye DONE ---"

# [82/100] 3ve9 (6424 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/3ve9"
echo "=== [82/100] 3ve9 (6424 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3ve9.topology.json" -o "${RESULTS_DIR}/3ve9" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3ve9/run.log"
echo "--- 3ve9 DONE ---"

# [83/100] 2czd (6507 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/2czd"
echo "=== [83/100] 2czd (6507 atoms) ==="
"$NHS" -t "${TOPO_DIR}/2czd.topology.json" -o "${RESULTS_DIR}/2czd" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2czd/run.log"
echo "--- 2czd DONE ---"

# [84/100] 6fgj (6592 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/6fgj"
echo "=== [84/100] 6fgj (6592 atoms) ==="
"$NHS" -t "${TOPO_DIR}/6fgj.topology.json" -o "${RESULTS_DIR}/6fgj" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6fgj/run.log"
echo "--- 6fgj DONE ---"

# [85/100] 3la7 (6651 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/3la7"
echo "=== [85/100] 3la7 (6651 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3la7.topology.json" -o "${RESULTS_DIR}/3la7" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3la7/run.log"
echo "--- 3la7 DONE ---"

# [86/100] 3rwv (6700 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/3rwv"
echo "=== [86/100] 3rwv (6700 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3rwv.topology.json" -o "${RESULTS_DIR}/3rwv" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3rwv/run.log"
echo "--- 3rwv DONE ---"

# [87/100] 6w10 (6740 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/6w10"
echo "=== [87/100] 6w10 (6740 atoms) ==="
"$NHS" -t "${TOPO_DIR}/6w10.topology.json" -o "${RESULTS_DIR}/6w10" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6w10/run.log"
echo "--- 6w10 DONE ---"

# [88/100] 1dq2 (6745 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/1dq2"
echo "=== [88/100] 1dq2 (6745 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1dq2.topology.json" -o "${RESULTS_DIR}/1dq2" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1dq2/run.log"
echo "--- 1dq2 DONE ---"

# [89/100] 3tpo (6772 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/3tpo"
echo "=== [89/100] 3tpo (6772 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3tpo.topology.json" -o "${RESULTS_DIR}/3tpo" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3tpo/run.log"
echo "--- 3tpo DONE ---"

# [90/100] 3t8b (6821 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/3t8b"
echo "=== [90/100] 3t8b (6821 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3t8b.topology.json" -o "${RESULTS_DIR}/3t8b" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3t8b/run.log"
echo "--- 3t8b DONE ---"

echo "=== BATCH 9 COMPLETE ==="

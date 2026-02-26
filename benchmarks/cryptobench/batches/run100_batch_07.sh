#!/usr/bin/env bash
set -euo pipefail

# Batch 7/10 — targets 61-70 of 100
NHS="target/release/nhs_rt_full"
TOPO_DIR="benchmarks/cryptobench/topologies"
RESULTS_DIR="benchmarks/cryptobench/results"

# [61/100] 3n4u (4962 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/3n4u"
echo "=== [61/100] 3n4u (4962 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3n4u.topology.json" -o "${RESULTS_DIR}/3n4u" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3n4u/run.log"
echo "--- 3n4u DONE ---"

# [62/100] 6ksc (5019 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/6ksc"
echo "=== [62/100] 6ksc (5019 atoms) ==="
"$NHS" -t "${TOPO_DIR}/6ksc.topology.json" -o "${RESULTS_DIR}/6ksc" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6ksc/run.log"
echo "--- 6ksc DONE ---"

# [63/100] 1ksg (5030 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/1ksg"
echo "=== [63/100] 1ksg (5030 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1ksg.topology.json" -o "${RESULTS_DIR}/1ksg" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1ksg/run.log"
echo "--- 1ksg DONE ---"

# [64/100] 1xgd (5036 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/1xgd"
echo "=== [64/100] 1xgd (5036 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1xgd.topology.json" -o "${RESULTS_DIR}/1xgd" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1xgd/run.log"
echo "--- 1xgd DONE ---"

# [65/100] 6g6y (5081 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/6g6y"
echo "=== [65/100] 6g6y (5081 atoms) ==="
"$NHS" -t "${TOPO_DIR}/6g6y.topology.json" -o "${RESULTS_DIR}/6g6y" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6g6y/run.log"
echo "--- 6g6y DONE ---"

# [66/100] 3nx1 (5105 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/3nx1"
echo "=== [66/100] 3nx1 (5105 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3nx1.topology.json" -o "${RESULTS_DIR}/3nx1" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3nx1/run.log"
echo "--- 3nx1 DONE ---"

# [67/100] 3ly8 (5121 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/3ly8"
echo "=== [67/100] 3ly8 (5121 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3ly8.topology.json" -o "${RESULTS_DIR}/3ly8" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3ly8/run.log"
echo "--- 3ly8 DONE ---"

# [68/100] 1evy (5321 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/1evy"
echo "=== [68/100] 1evy (5321 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1evy.topology.json" -o "${RESULTS_DIR}/1evy" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1evy/run.log"
echo "--- 1evy DONE ---"

# [69/100] 2rfj (5353 atoms, 3 chains)
mkdir -p "${RESULTS_DIR}/2rfj"
echo "=== [69/100] 2rfj (5353 atoms) ==="
"$NHS" -t "${TOPO_DIR}/2rfj.topology.json" -o "${RESULTS_DIR}/2rfj" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2rfj/run.log"
echo "--- 2rfj DONE ---"

# [70/100] 6du4 (5574 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/6du4"
echo "=== [70/100] 6du4 (5574 atoms) ==="
"$NHS" -t "${TOPO_DIR}/6du4.topology.json" -o "${RESULTS_DIR}/6du4" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6du4/run.log"
echo "--- 6du4 DONE ---"

echo "=== BATCH 7 COMPLETE ==="

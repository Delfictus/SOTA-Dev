#!/usr/bin/env bash
set -euo pipefail

# Batch 10/10 — targets 91-100 of 100
NHS="target/release/nhs_rt_full"
TOPO_DIR="benchmarks/cryptobench/topologies"
RESULTS_DIR="benchmarks/cryptobench/results"

# [91/100] 6hei (6842 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/6hei"
echo "=== [91/100] 6hei (6842 atoms) ==="
"$NHS" -t "${TOPO_DIR}/6hei.topology.json" -o "${RESULTS_DIR}/6hei" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6hei/run.log"
echo "--- 6hei DONE ---"

# [92/100] 5acv (6892 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/5acv"
echo "=== [92/100] 5acv (6892 atoms) ==="
"$NHS" -t "${TOPO_DIR}/5acv.topology.json" -o "${RESULTS_DIR}/5acv" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5acv/run.log"
echo "--- 5acv DONE ---"

# [93/100] 2xsa (6903 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/2xsa"
echo "=== [93/100] 2xsa (6903 atoms) ==="
"$NHS" -t "${TOPO_DIR}/2xsa.topology.json" -o "${RESULTS_DIR}/2xsa" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2xsa/run.log"
echo "--- 2xsa DONE ---"

# [94/100] 3idh (7050 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/3idh"
echo "=== [94/100] 3idh (7050 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3idh.topology.json" -o "${RESULTS_DIR}/3idh" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3idh/run.log"
echo "--- 3idh DONE ---"

# [95/100] 9atc (7145 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/9atc"
echo "=== [95/100] 9atc (7145 atoms) ==="
"$NHS" -t "${TOPO_DIR}/9atc.topology.json" -o "${RESULTS_DIR}/9atc" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/9atc/run.log"
echo "--- 9atc DONE ---"

# [96/100] 2vl2 (7229 atoms, 3 chains)
mkdir -p "${RESULTS_DIR}/2vl2"
echo "=== [96/100] 2vl2 (7229 atoms) ==="
"$NHS" -t "${TOPO_DIR}/2vl2.topology.json" -o "${RESULTS_DIR}/2vl2" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2vl2/run.log"
echo "--- 2vl2 DONE ---"

# [97/100] 7yjc (7256 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/7yjc"
echo "=== [97/100] 7yjc (7256 atoms) ==="
"$NHS" -t "${TOPO_DIR}/7yjc.topology.json" -o "${RESULTS_DIR}/7yjc" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7yjc/run.log"
echo "--- 7yjc DONE ---"

# [98/100] 1a8d (7257 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/1a8d"
echo "=== [98/100] 1a8d (7257 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1a8d.topology.json" -o "${RESULTS_DIR}/1a8d" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1a8d/run.log"
echo "--- 1a8d DONE ---"

# [99/100] 2w8n (7258 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/2w8n"
echo "=== [99/100] 2w8n (7258 atoms) ==="
"$NHS" -t "${TOPO_DIR}/2w8n.topology.json" -o "${RESULTS_DIR}/2w8n" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2w8n/run.log"
echo "--- 2w8n DONE ---"

# [100/100] 4p32 (7284 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/4p32"
echo "=== [100/100] 4p32 (7284 atoms) ==="
"$NHS" -t "${TOPO_DIR}/4p32.topology.json" -o "${RESULTS_DIR}/4p32" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4p32/run.log"
echo "--- 4p32 DONE ---"

echo "=== BATCH 10 COMPLETE ==="

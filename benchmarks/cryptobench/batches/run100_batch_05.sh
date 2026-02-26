#!/usr/bin/env bash
set -euo pipefail

# Batch 5/10 — targets 41-50 of 100
NHS="target/release/nhs_rt_full"
TOPO_DIR="benchmarks/cryptobench/topologies"
RESULTS_DIR="benchmarks/cryptobench/results"

# [41/100] 6bty (4087 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/6bty"
echo "=== [41/100] 6bty (4087 atoms) ==="
"$NHS" -t "${TOPO_DIR}/6bty.topology.json" -o "${RESULTS_DIR}/6bty" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6bty/run.log"
echo "--- 6bty DONE ---"

# [42/100] 1rtc (4215 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/1rtc"
echo "=== [42/100] 1rtc (4215 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1rtc.topology.json" -o "${RESULTS_DIR}/1rtc" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1rtc/run.log"
echo "--- 1rtc DONE ---"

# [43/100] 1uka (4226 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/1uka"
echo "=== [43/100] 1uka (4226 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1uka.topology.json" -o "${RESULTS_DIR}/1uka" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1uka/run.log"
echo "--- 1uka DONE ---"

# [44/100] 3fzo (4270 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/3fzo"
echo "=== [44/100] 3fzo (4270 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3fzo.topology.json" -o "${RESULTS_DIR}/3fzo" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3fzo/run.log"
echo "--- 3fzo DONE ---"

# [45/100] 2phz (4306 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/2phz"
echo "=== [45/100] 2phz (4306 atoms) ==="
"$NHS" -t "${TOPO_DIR}/2phz.topology.json" -o "${RESULTS_DIR}/2phz" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2phz/run.log"
echo "--- 2phz DONE ---"

# [46/100] 7nlx (4323 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/7nlx"
echo "=== [46/100] 7nlx (4323 atoms) ==="
"$NHS" -t "${TOPO_DIR}/7nlx.topology.json" -o "${RESULTS_DIR}/7nlx" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7nlx/run.log"
echo "--- 7nlx DONE ---"

# [47/100] 3hrm (4336 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/3hrm"
echo "=== [47/100] 3hrm (4336 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3hrm.topology.json" -o "${RESULTS_DIR}/3hrm" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3hrm/run.log"
echo "--- 3hrm DONE ---"

# [48/100] 4mwi (4350 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/4mwi"
echo "=== [48/100] 4mwi (4350 atoms) ==="
"$NHS" -t "${TOPO_DIR}/4mwi.topology.json" -o "${RESULTS_DIR}/4mwi" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4mwi/run.log"
echo "--- 4mwi DONE ---"

# [49/100] 7c48 (4448 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/7c48"
echo "=== [49/100] 7c48 (4448 atoms) ==="
"$NHS" -t "${TOPO_DIR}/7c48.topology.json" -o "${RESULTS_DIR}/7c48" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7c48/run.log"
echo "--- 7c48 DONE ---"

# [50/100] 1xxo (4458 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/1xxo"
echo "=== [50/100] 1xxo (4458 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1xxo.topology.json" -o "${RESULTS_DIR}/1xxo" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1xxo/run.log"
echo "--- 1xxo DONE ---"

echo "=== BATCH 5 COMPLETE ==="

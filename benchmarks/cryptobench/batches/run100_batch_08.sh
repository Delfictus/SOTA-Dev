#!/usr/bin/env bash
set -euo pipefail

# Batch 8/10 — targets 71-80 of 100
NHS="target/release/nhs_rt_full"
TOPO_DIR="benchmarks/cryptobench/topologies"
RESULTS_DIR="benchmarks/cryptobench/results"

# [71/100] 3jzg (5773 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/3jzg"
echo "=== [71/100] 3jzg (5773 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3jzg.topology.json" -o "${RESULTS_DIR}/3jzg" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3jzg/run.log"
echo "--- 3jzg DONE ---"

# [72/100] 3k01 (5838 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/3k01"
echo "=== [72/100] 3k01 (5838 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3k01.topology.json" -o "${RESULTS_DIR}/3k01" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3k01/run.log"
echo "--- 3k01 DONE ---"

# [73/100] 5yqp (5846 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/5yqp"
echo "=== [73/100] 5yqp (5846 atoms) ==="
"$NHS" -t "${TOPO_DIR}/5yqp.topology.json" -o "${RESULTS_DIR}/5yqp" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5yqp/run.log"
echo "--- 5yqp DONE ---"

# [74/100] 3ugk (5887 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/3ugk"
echo "=== [74/100] 3ugk (5887 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3ugk.topology.json" -o "${RESULTS_DIR}/3ugk" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3ugk/run.log"
echo "--- 3ugk DONE ---"

# [75/100] 3v55 (5897 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/3v55"
echo "=== [75/100] 3v55 (5897 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3v55.topology.json" -o "${RESULTS_DIR}/3v55" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3v55/run.log"
echo "--- 3v55 DONE ---"

# [76/100] 6isu (6031 atoms, 3 chains)
mkdir -p "${RESULTS_DIR}/6isu"
echo "=== [76/100] 6isu (6031 atoms) ==="
"$NHS" -t "${TOPO_DIR}/6isu.topology.json" -o "${RESULTS_DIR}/6isu" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6isu/run.log"
echo "--- 6isu DONE ---"

# [77/100] 1nd7 (6169 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/1nd7"
echo "=== [77/100] 1nd7 (6169 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1nd7.topology.json" -o "${RESULTS_DIR}/1nd7" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1nd7/run.log"
echo "--- 1nd7 DONE ---"

# [78/100] 4fkm (6198 atoms, 2 chains)
mkdir -p "${RESULTS_DIR}/4fkm"
echo "=== [78/100] 4fkm (6198 atoms) ==="
"$NHS" -t "${TOPO_DIR}/4fkm.topology.json" -o "${RESULTS_DIR}/4fkm" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4fkm/run.log"
echo "--- 4fkm DONE ---"

# [79/100] 1h13 (6282 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/1h13"
echo "=== [79/100] 1h13 (6282 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1h13.topology.json" -o "${RESULTS_DIR}/1h13" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1h13/run.log"
echo "--- 1h13 DONE ---"

# [80/100] 7c63 (6331 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/7c63"
echo "=== [80/100] 7c63 (6331 atoms) ==="
"$NHS" -t "${TOPO_DIR}/7c63.topology.json" -o "${RESULTS_DIR}/7c63" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7c63/run.log"
echo "--- 7c63 DONE ---"

echo "=== BATCH 8 COMPLETE ==="

#!/usr/bin/env bash
set -euo pipefail

# Batch 6/10 — targets 51-60 of 100
NHS="target/release/nhs_rt_full"
TOPO_DIR="benchmarks/cryptobench/topologies"
RESULTS_DIR="benchmarks/cryptobench/results"

# [51/100] 3vgm (4518 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/3vgm"
echo "=== [51/100] 3vgm (4518 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3vgm.topology.json" -o "${RESULTS_DIR}/3vgm" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3vgm/run.log"
echo "--- 3vgm DONE ---"

# [52/100] 5igh (4530 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/5igh"
echo "=== [52/100] 5igh (4530 atoms) ==="
"$NHS" -t "${TOPO_DIR}/5igh.topology.json" -o "${RESULTS_DIR}/5igh" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5igh/run.log"
echo "--- 5igh DONE ---"

# [53/100] 1xqv (4625 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/1xqv"
echo "=== [53/100] 1xqv (4625 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1xqv.topology.json" -o "${RESULTS_DIR}/1xqv" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1xqv/run.log"
echo "--- 1xqv DONE ---"

# [54/100] 3bjp (4683 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/3bjp"
echo "=== [54/100] 3bjp (4683 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3bjp.topology.json" -o "${RESULTS_DIR}/3bjp" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3bjp/run.log"
echo "--- 3bjp DONE ---"

# [55/100] 1ute (4756 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/1ute"
echo "=== [55/100] 1ute (4756 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1ute.topology.json" -o "${RESULTS_DIR}/1ute" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1ute/run.log"
echo "--- 1ute DONE ---"

# [56/100] 1rjb (4781 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/1rjb"
echo "=== [56/100] 1rjb (4781 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1rjb.topology.json" -o "${RESULTS_DIR}/1rjb" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1rjb/run.log"
echo "--- 1rjb DONE ---"

# [57/100] 5uxa (4809 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/5uxa"
echo "=== [57/100] 5uxa (4809 atoms) ==="
"$NHS" -t "${TOPO_DIR}/5uxa.topology.json" -o "${RESULTS_DIR}/5uxa" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5uxa/run.log"
echo "--- 5uxa DONE ---"

# [58/100] 1bzj (4811 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/1bzj"
echo "=== [58/100] 1bzj (4811 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1bzj.topology.json" -o "${RESULTS_DIR}/1bzj" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1bzj/run.log"
echo "--- 1bzj DONE ---"

# [59/100] 1ak1 (4880 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/1ak1"
echo "=== [59/100] 1ak1 (4880 atoms) ==="
"$NHS" -t "${TOPO_DIR}/1ak1.topology.json" -o "${RESULTS_DIR}/1ak1" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1ak1/run.log"
echo "--- 1ak1 DONE ---"

# [60/100] 3uyi (4888 atoms, 1 chains)
mkdir -p "${RESULTS_DIR}/3uyi"
echo "=== [60/100] 3uyi (4888 atoms) ==="
"$NHS" -t "${TOPO_DIR}/3uyi.topology.json" -o "${RESULTS_DIR}/3uyi" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3uyi/run.log"
echo "--- 3uyi DONE ---"

echo "=== BATCH 6 COMPLETE ==="

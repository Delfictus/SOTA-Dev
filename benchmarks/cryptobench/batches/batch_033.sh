#!/usr/bin/env bash
# CryptoBench Batch 33/39 — 5 structures
# Structures: 6jq9, 6ksc, 6n5j, 6nei, 6syh
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 33/39'
echo '=========================================='
echo 'Structures: 6jq9 6ksc 6n5j 6nei 6syh'
echo ''

# --- [1/5] 6jq9 ---
echo "[1/5] Running 6jq9..."
mkdir -p "${RESULTS_DIR}/6jq9"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6jq9.topology.json" -o "${RESULTS_DIR}/6jq9" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6jq9/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6jq9: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6jq9,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_033_timing.csv"

# --- [2/5] 6ksc ---
echo "[2/5] Running 6ksc..."
mkdir -p "${RESULTS_DIR}/6ksc"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6ksc.topology.json" -o "${RESULTS_DIR}/6ksc" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6ksc/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6ksc: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6ksc,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_033_timing.csv"

# --- [3/5] 6n5j ---
echo "[3/5] Running 6n5j..."
mkdir -p "${RESULTS_DIR}/6n5j"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6n5j.topology.json" -o "${RESULTS_DIR}/6n5j" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6n5j/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6n5j: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6n5j,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_033_timing.csv"

# --- [4/5] 6nei ---
echo "[4/5] Running 6nei..."
mkdir -p "${RESULTS_DIR}/6nei"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6nei.topology.json" -o "${RESULTS_DIR}/6nei" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6nei/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6nei: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6nei,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_033_timing.csv"

# --- [5/5] 6syh ---
echo "[5/5] Running 6syh..."
mkdir -p "${RESULTS_DIR}/6syh"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6syh.topology.json" -o "${RESULTS_DIR}/6syh" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6syh/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6syh: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6syh,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_033_timing.csv"

echo ''
echo 'Batch 33 complete.'
echo ''

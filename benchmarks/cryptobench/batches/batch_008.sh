#!/usr/bin/env bash
# CryptoBench Batch 8/39 — 5 structures
# Structures: 1zm0, 2aka, 2czd, 2d05, 2dfp
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 8/39'
echo '=========================================='
echo 'Structures: 1zm0 2aka 2czd 2d05 2dfp'
echo ''

# --- [1/5] 1zm0 ---
echo "[1/5] Running 1zm0..."
mkdir -p "${RESULTS_DIR}/1zm0"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1zm0.topology.json" -o "${RESULTS_DIR}/1zm0" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1zm0/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1zm0: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1zm0,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_008_timing.csv"

# --- [2/5] 2aka ---
echo "[2/5] Running 2aka..."
mkdir -p "${RESULTS_DIR}/2aka"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2aka.topology.json" -o "${RESULTS_DIR}/2aka" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2aka/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2aka: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2aka,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_008_timing.csv"

# --- [3/5] 2czd ---
echo "[3/5] Running 2czd..."
mkdir -p "${RESULTS_DIR}/2czd"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2czd.topology.json" -o "${RESULTS_DIR}/2czd" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2czd/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2czd: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2czd,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_008_timing.csv"

# --- [4/5] 2d05 ---
echo "[4/5] Running 2d05..."
mkdir -p "${RESULTS_DIR}/2d05"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2d05.topology.json" -o "${RESULTS_DIR}/2d05" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2d05/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2d05: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2d05,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_008_timing.csv"

# --- [5/5] 2dfp ---
echo "[5/5] Running 2dfp..."
mkdir -p "${RESULTS_DIR}/2dfp"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2dfp.topology.json" -o "${RESULTS_DIR}/2dfp" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2dfp/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2dfp: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2dfp,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_008_timing.csv"

echo ''
echo 'Batch 8 complete.'
echo ''

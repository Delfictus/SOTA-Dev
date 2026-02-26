#!/usr/bin/env bash
# CryptoBench Batch 32/39 — 5 structures
# Structures: 6fc2, 6fgj, 6g6y, 6hei, 6isu
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 32/39'
echo '=========================================='
echo 'Structures: 6fc2 6fgj 6g6y 6hei 6isu'
echo ''

# --- [1/5] 6fc2 ---
echo "[1/5] Running 6fc2..."
mkdir -p "${RESULTS_DIR}/6fc2"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6fc2.topology.json" -o "${RESULTS_DIR}/6fc2" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6fc2/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6fc2: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6fc2,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_032_timing.csv"

# --- [2/5] 6fgj ---
echo "[2/5] Running 6fgj..."
mkdir -p "${RESULTS_DIR}/6fgj"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6fgj.topology.json" -o "${RESULTS_DIR}/6fgj" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6fgj/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6fgj: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6fgj,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_032_timing.csv"

# --- [3/5] 6g6y ---
echo "[3/5] Running 6g6y..."
mkdir -p "${RESULTS_DIR}/6g6y"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6g6y.topology.json" -o "${RESULTS_DIR}/6g6y" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6g6y/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6g6y: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6g6y,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_032_timing.csv"

# --- [4/5] 6hei ---
echo "[4/5] Running 6hei..."
mkdir -p "${RESULTS_DIR}/6hei"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6hei.topology.json" -o "${RESULTS_DIR}/6hei" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6hei/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6hei: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6hei,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_032_timing.csv"

# --- [5/5] 6isu ---
echo "[5/5] Running 6isu..."
mkdir -p "${RESULTS_DIR}/6isu"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6isu.topology.json" -o "${RESULTS_DIR}/6isu" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6isu/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6isu: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6isu,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_032_timing.csv"

echo ''
echo 'Batch 32 complete.'
echo ''

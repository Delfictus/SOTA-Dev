#!/usr/bin/env bash
# CryptoBench Batch 34/39 — 5 structures
# Structures: 6tx0, 6vle, 6w10, 7c48, 7c63
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 34/39'
echo '=========================================='
echo 'Structures: 6tx0 6vle 6w10 7c48 7c63'
echo ''

# --- [1/5] 6tx0 ---
echo "[1/5] Running 6tx0..."
mkdir -p "${RESULTS_DIR}/6tx0"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6tx0.topology.json" -o "${RESULTS_DIR}/6tx0" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6tx0/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6tx0: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6tx0,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_034_timing.csv"

# --- [2/5] 6vle ---
echo "[2/5] Running 6vle..."
mkdir -p "${RESULTS_DIR}/6vle"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6vle.topology.json" -o "${RESULTS_DIR}/6vle" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6vle/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6vle: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6vle,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_034_timing.csv"

# --- [3/5] 6w10 ---
echo "[3/5] Running 6w10..."
mkdir -p "${RESULTS_DIR}/6w10"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6w10.topology.json" -o "${RESULTS_DIR}/6w10" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6w10/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6w10: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6w10,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_034_timing.csv"

# --- [4/5] 7c48 ---
echo "[4/5] Running 7c48..."
mkdir -p "${RESULTS_DIR}/7c48"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/7c48.topology.json" -o "${RESULTS_DIR}/7c48" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7c48/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  7c48: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "7c48,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_034_timing.csv"

# --- [5/5] 7c63 ---
echo "[5/5] Running 7c63..."
mkdir -p "${RESULTS_DIR}/7c63"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/7c63.topology.json" -o "${RESULTS_DIR}/7c63" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7c63/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  7c63: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "7c63,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_034_timing.csv"

echo ''
echo 'Batch 34 complete.'
echo ''

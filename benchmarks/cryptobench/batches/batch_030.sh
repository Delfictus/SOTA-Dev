#!/usr/bin/env bash
# CryptoBench Batch 30/39 — 5 structures
# Structures: 5yhb, 5yqp, 5ysb, 5zj4, 6a98
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 30/39'
echo '=========================================='
echo 'Structures: 5yhb 5yqp 5ysb 5zj4 6a98'
echo ''

# --- [1/5] 5yhb ---
echo "[1/5] Running 5yhb..."
mkdir -p "${RESULTS_DIR}/5yhb"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5yhb.topology.json" -o "${RESULTS_DIR}/5yhb" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5yhb/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5yhb: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5yhb,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_030_timing.csv"

# --- [2/5] 5yqp ---
echo "[2/5] Running 5yqp..."
mkdir -p "${RESULTS_DIR}/5yqp"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5yqp.topology.json" -o "${RESULTS_DIR}/5yqp" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5yqp/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5yqp: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5yqp,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_030_timing.csv"

# --- [3/5] 5ysb ---
echo "[3/5] Running 5ysb..."
mkdir -p "${RESULTS_DIR}/5ysb"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5ysb.topology.json" -o "${RESULTS_DIR}/5ysb" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5ysb/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5ysb: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5ysb,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_030_timing.csv"

# --- [4/5] 5zj4 ---
echo "[4/5] Running 5zj4..."
mkdir -p "${RESULTS_DIR}/5zj4"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5zj4.topology.json" -o "${RESULTS_DIR}/5zj4" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5zj4/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5zj4: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5zj4,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_030_timing.csv"

# --- [5/5] 6a98 ---
echo "[5/5] Running 6a98..."
mkdir -p "${RESULTS_DIR}/6a98"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6a98.topology.json" -o "${RESULTS_DIR}/6a98" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6a98/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6a98: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6a98,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_030_timing.csv"

echo ''
echo 'Batch 30 complete.'
echo ''

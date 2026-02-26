#!/usr/bin/env bash
# CryptoBench Batch 37/39 — 5 structures
# Structures: 7x0i, 7xgf, 7yjc, 8aeq, 8aqi
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 37/39'
echo '=========================================='
echo 'Structures: 7x0i 7xgf 7yjc 8aeq 8aqi'
echo ''

# --- [1/5] 7x0i ---
echo "[1/5] Running 7x0i..."
mkdir -p "${RESULTS_DIR}/7x0i"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/7x0i.topology.json" -o "${RESULTS_DIR}/7x0i" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7x0i/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  7x0i: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "7x0i,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_037_timing.csv"

# --- [2/5] 7xgf ---
echo "[2/5] Running 7xgf..."
mkdir -p "${RESULTS_DIR}/7xgf"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/7xgf.topology.json" -o "${RESULTS_DIR}/7xgf" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7xgf/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  7xgf: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "7xgf,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_037_timing.csv"

# --- [3/5] 7yjc ---
echo "[3/5] Running 7yjc..."
mkdir -p "${RESULTS_DIR}/7yjc"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/7yjc.topology.json" -o "${RESULTS_DIR}/7yjc" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7yjc/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  7yjc: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "7yjc,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_037_timing.csv"

# --- [4/5] 8aeq ---
echo "[4/5] Running 8aeq..."
mkdir -p "${RESULTS_DIR}/8aeq"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/8aeq.topology.json" -o "${RESULTS_DIR}/8aeq" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/8aeq/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  8aeq: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "8aeq,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_037_timing.csv"

# --- [5/5] 8aqi ---
echo "[5/5] Running 8aqi..."
mkdir -p "${RESULTS_DIR}/8aqi"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/8aqi.topology.json" -o "${RESULTS_DIR}/8aqi" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/8aqi/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  8aqi: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "8aqi,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_037_timing.csv"

echo ''
echo 'Batch 37 complete.'
echo ''

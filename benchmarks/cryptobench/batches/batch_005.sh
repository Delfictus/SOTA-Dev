#!/usr/bin/env bash
# CryptoBench Batch 5/39 — 5 structures
# Structures: 1q4k, 1r3m, 1rjb, 1rtc, 1se8
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 5/39'
echo '=========================================='
echo 'Structures: 1q4k 1r3m 1rjb 1rtc 1se8'
echo ''

# --- [1/5] 1q4k ---
echo "[1/5] Running 1q4k..."
mkdir -p "${RESULTS_DIR}/1q4k"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1q4k.topology.json" -o "${RESULTS_DIR}/1q4k" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1q4k/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1q4k: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1q4k,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_005_timing.csv"

# --- [2/5] 1r3m ---
echo "[2/5] Running 1r3m..."
mkdir -p "${RESULTS_DIR}/1r3m"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1r3m.topology.json" -o "${RESULTS_DIR}/1r3m" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1r3m/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1r3m: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1r3m,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_005_timing.csv"

# --- [3/5] 1rjb ---
echo "[3/5] Running 1rjb..."
mkdir -p "${RESULTS_DIR}/1rjb"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1rjb.topology.json" -o "${RESULTS_DIR}/1rjb" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1rjb/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1rjb: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1rjb,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_005_timing.csv"

# --- [4/5] 1rtc ---
echo "[4/5] Running 1rtc..."
mkdir -p "${RESULTS_DIR}/1rtc"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1rtc.topology.json" -o "${RESULTS_DIR}/1rtc" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1rtc/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1rtc: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1rtc,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_005_timing.csv"

# --- [5/5] 1se8 ---
echo "[5/5] Running 1se8..."
mkdir -p "${RESULTS_DIR}/1se8"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1se8.topology.json" -o "${RESULTS_DIR}/1se8" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1se8/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1se8: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1se8,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_005_timing.csv"

echo ''
echo 'Batch 5 complete.'
echo ''

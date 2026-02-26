#!/usr/bin/env bash
# CryptoBench Batch 36/39 — 5 structures
# Structures: 7o1i, 7qoq, 7w19, 7x0f, 7x0g
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 36/39'
echo '=========================================='
echo 'Structures: 7o1i 7qoq 7w19 7x0f 7x0g'
echo ''

# --- [1/5] 7o1i ---
echo "[1/5] Running 7o1i..."
mkdir -p "${RESULTS_DIR}/7o1i"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/7o1i.topology.json" -o "${RESULTS_DIR}/7o1i" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7o1i/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  7o1i: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "7o1i,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_036_timing.csv"

# --- [2/5] 7qoq ---
echo "[2/5] Running 7qoq..."
mkdir -p "${RESULTS_DIR}/7qoq"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/7qoq.topology.json" -o "${RESULTS_DIR}/7qoq" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7qoq/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  7qoq: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "7qoq,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_036_timing.csv"

# --- [3/5] 7w19 ---
echo "[3/5] Running 7w19..."
mkdir -p "${RESULTS_DIR}/7w19"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/7w19.topology.json" -o "${RESULTS_DIR}/7w19" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7w19/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  7w19: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "7w19,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_036_timing.csv"

# --- [4/5] 7x0f ---
echo "[4/5] Running 7x0f..."
mkdir -p "${RESULTS_DIR}/7x0f"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/7x0f.topology.json" -o "${RESULTS_DIR}/7x0f" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7x0f/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  7x0f: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "7x0f,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_036_timing.csv"

# --- [5/5] 7x0g ---
echo "[5/5] Running 7x0g..."
mkdir -p "${RESULTS_DIR}/7x0g"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/7x0g.topology.json" -o "${RESULTS_DIR}/7x0g" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7x0g/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  7x0g: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "7x0g,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_036_timing.csv"

echo ''
echo 'Batch 36 complete.'
echo ''

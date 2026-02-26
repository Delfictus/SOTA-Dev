#!/usr/bin/env bash
# CryptoBench Batch 4/39 — 5 structures
# Structures: 1lbe, 1nd7, 1p4o, 1p9o, 1pu5
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 4/39'
echo '=========================================='
echo 'Structures: 1lbe 1nd7 1p4o 1p9o 1pu5'
echo ''

# --- [1/5] 1lbe ---
echo "[1/5] Running 1lbe..."
mkdir -p "${RESULTS_DIR}/1lbe"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1lbe.topology.json" -o "${RESULTS_DIR}/1lbe" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1lbe/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1lbe: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1lbe,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_004_timing.csv"

# --- [2/5] 1nd7 ---
echo "[2/5] Running 1nd7..."
mkdir -p "${RESULTS_DIR}/1nd7"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1nd7.topology.json" -o "${RESULTS_DIR}/1nd7" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1nd7/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1nd7: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1nd7,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_004_timing.csv"

# --- [3/5] 1p4o ---
echo "[3/5] Running 1p4o..."
mkdir -p "${RESULTS_DIR}/1p4o"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1p4o.topology.json" -o "${RESULTS_DIR}/1p4o" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1p4o/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1p4o: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1p4o,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_004_timing.csv"

# --- [4/5] 1p9o ---
echo "[4/5] Running 1p9o..."
mkdir -p "${RESULTS_DIR}/1p9o"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1p9o.topology.json" -o "${RESULTS_DIR}/1p9o" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1p9o/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1p9o: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1p9o,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_004_timing.csv"

# --- [5/5] 1pu5 ---
echo "[5/5] Running 1pu5..."
mkdir -p "${RESULTS_DIR}/1pu5"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1pu5.topology.json" -o "${RESULTS_DIR}/1pu5" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1pu5/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1pu5: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1pu5,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_004_timing.csv"

echo ''
echo 'Batch 4 complete.'
echo ''

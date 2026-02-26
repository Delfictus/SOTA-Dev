#!/usr/bin/env bash
# CryptoBench Batch 25/39 — 5 structures
# Structures: 4x19, 4zm7, 4zoe, 5acv, 5aon
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 25/39'
echo '=========================================='
echo 'Structures: 4x19 4zm7 4zoe 5acv 5aon'
echo ''

# --- [1/5] 4x19 ---
echo "[1/5] Running 4x19..."
mkdir -p "${RESULTS_DIR}/4x19"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4x19.topology.json" -o "${RESULTS_DIR}/4x19" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4x19/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4x19: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4x19,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_025_timing.csv"

# --- [2/5] 4zm7 ---
echo "[2/5] Running 4zm7..."
mkdir -p "${RESULTS_DIR}/4zm7"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4zm7.topology.json" -o "${RESULTS_DIR}/4zm7" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4zm7/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4zm7: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4zm7,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_025_timing.csv"

# --- [3/5] 4zoe ---
echo "[3/5] Running 4zoe..."
mkdir -p "${RESULTS_DIR}/4zoe"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4zoe.topology.json" -o "${RESULTS_DIR}/4zoe" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4zoe/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4zoe: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4zoe,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_025_timing.csv"

# --- [4/5] 5acv ---
echo "[4/5] Running 5acv..."
mkdir -p "${RESULTS_DIR}/5acv"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5acv.topology.json" -o "${RESULTS_DIR}/5acv" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5acv/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5acv: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5acv,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_025_timing.csv"

# --- [5/5] 5aon ---
echo "[5/5] Running 5aon..."
mkdir -p "${RESULTS_DIR}/5aon"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5aon.topology.json" -o "${RESULTS_DIR}/5aon" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5aon/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5aon: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5aon,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_025_timing.csv"

echo ''
echo 'Batch 25 complete.'
echo ''

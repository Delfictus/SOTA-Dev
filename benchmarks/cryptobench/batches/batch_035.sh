#!/usr/bin/env bash
# CryptoBench Batch 35/39 — 5 structures
# Structures: 7de1, 7e5q, 7f2m, 7nc8, 7nlx
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 35/39'
echo '=========================================='
echo 'Structures: 7de1 7e5q 7f2m 7nc8 7nlx'
echo ''

# --- [1/5] 7de1 ---
echo "[1/5] Running 7de1..."
mkdir -p "${RESULTS_DIR}/7de1"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/7de1.topology.json" -o "${RESULTS_DIR}/7de1" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7de1/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  7de1: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "7de1,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_035_timing.csv"

# --- [2/5] 7e5q ---
echo "[2/5] Running 7e5q..."
mkdir -p "${RESULTS_DIR}/7e5q"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/7e5q.topology.json" -o "${RESULTS_DIR}/7e5q" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7e5q/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  7e5q: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "7e5q,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_035_timing.csv"

# --- [3/5] 7f2m ---
echo "[3/5] Running 7f2m..."
mkdir -p "${RESULTS_DIR}/7f2m"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/7f2m.topology.json" -o "${RESULTS_DIR}/7f2m" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7f2m/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  7f2m: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "7f2m,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_035_timing.csv"

# --- [4/5] 7nc8 ---
echo "[4/5] Running 7nc8..."
mkdir -p "${RESULTS_DIR}/7nc8"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/7nc8.topology.json" -o "${RESULTS_DIR}/7nc8" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7nc8/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  7nc8: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "7nc8,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_035_timing.csv"

# --- [5/5] 7nlx ---
echo "[5/5] Running 7nlx..."
mkdir -p "${RESULTS_DIR}/7nlx"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/7nlx.topology.json" -o "${RESULTS_DIR}/7nlx" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/7nlx/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  7nlx: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "7nlx,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_035_timing.csv"

echo ''
echo 'Batch 35 complete.'
echo ''

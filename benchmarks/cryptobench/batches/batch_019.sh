#!/usr/bin/env bash
# CryptoBench Batch 19/39 — 5 structures
# Structures: 3uyi, 3v55, 3ve9, 3vgm, 3w90
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 19/39'
echo '=========================================='
echo 'Structures: 3uyi 3v55 3ve9 3vgm 3w90'
echo ''

# --- [1/5] 3uyi ---
echo "[1/5] Running 3uyi..."
mkdir -p "${RESULTS_DIR}/3uyi"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3uyi.topology.json" -o "${RESULTS_DIR}/3uyi" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3uyi/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3uyi: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3uyi,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_019_timing.csv"

# --- [2/5] 3v55 ---
echo "[2/5] Running 3v55..."
mkdir -p "${RESULTS_DIR}/3v55"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3v55.topology.json" -o "${RESULTS_DIR}/3v55" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3v55/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3v55: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3v55,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_019_timing.csv"

# --- [3/5] 3ve9 ---
echo "[3/5] Running 3ve9..."
mkdir -p "${RESULTS_DIR}/3ve9"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3ve9.topology.json" -o "${RESULTS_DIR}/3ve9" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3ve9/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3ve9: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3ve9,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_019_timing.csv"

# --- [4/5] 3vgm ---
echo "[4/5] Running 3vgm..."
mkdir -p "${RESULTS_DIR}/3vgm"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3vgm.topology.json" -o "${RESULTS_DIR}/3vgm" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3vgm/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3vgm: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3vgm,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_019_timing.csv"

# --- [5/5] 3w90 ---
echo "[5/5] Running 3w90..."
mkdir -p "${RESULTS_DIR}/3w90"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3w90.topology.json" -o "${RESULTS_DIR}/3w90" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3w90/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3w90: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3w90,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_019_timing.csv"

echo ''
echo 'Batch 19 complete.'
echo ''

#!/usr/bin/env bash
# CryptoBench Batch 20/39 — 5 structures
# Structures: 3wb9, 4aem, 4amv, 4bg8, 4cmw
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 20/39'
echo '=========================================='
echo 'Structures: 3wb9 4aem 4amv 4bg8 4cmw'
echo ''

# --- [1/5] 3wb9 ---
echo "[1/5] Running 3wb9..."
mkdir -p "${RESULTS_DIR}/3wb9"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3wb9.topology.json" -o "${RESULTS_DIR}/3wb9" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3wb9/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3wb9: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3wb9,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_020_timing.csv"

# --- [2/5] 4aem ---
echo "[2/5] Running 4aem..."
mkdir -p "${RESULTS_DIR}/4aem"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4aem.topology.json" -o "${RESULTS_DIR}/4aem" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4aem/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4aem: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4aem,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_020_timing.csv"

# --- [3/5] 4amv ---
echo "[3/5] Running 4amv..."
mkdir -p "${RESULTS_DIR}/4amv"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4amv.topology.json" -o "${RESULTS_DIR}/4amv" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4amv/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4amv: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4amv,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_020_timing.csv"

# --- [4/5] 4bg8 ---
echo "[4/5] Running 4bg8..."
mkdir -p "${RESULTS_DIR}/4bg8"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4bg8.topology.json" -o "${RESULTS_DIR}/4bg8" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4bg8/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4bg8: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4bg8,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_020_timing.csv"

# --- [5/5] 4cmw ---
echo "[5/5] Running 4cmw..."
mkdir -p "${RESULTS_DIR}/4cmw"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4cmw.topology.json" -o "${RESULTS_DIR}/4cmw" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4cmw/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4cmw: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4cmw,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_020_timing.csv"

echo ''
echo 'Batch 20 complete.'
echo ''

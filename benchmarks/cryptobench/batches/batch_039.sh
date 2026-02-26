#!/usr/bin/env bash
# CryptoBench Batch 39/39 — 4 structures
# Structures: 8j11, 8onn, 8vxu, 9atc
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 39/39'
echo '=========================================='
echo 'Structures: 8j11 8onn 8vxu 9atc'
echo ''

# --- [1/4] 8j11 ---
echo "[1/4] Running 8j11..."
mkdir -p "${RESULTS_DIR}/8j11"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/8j11.topology.json" -o "${RESULTS_DIR}/8j11" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/8j11/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  8j11: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "8j11,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_039_timing.csv"

# --- [2/4] 8onn ---
echo "[2/4] Running 8onn..."
mkdir -p "${RESULTS_DIR}/8onn"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/8onn.topology.json" -o "${RESULTS_DIR}/8onn" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/8onn/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  8onn: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "8onn,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_039_timing.csv"

# --- [3/4] 8vxu ---
echo "[3/4] Running 8vxu..."
mkdir -p "${RESULTS_DIR}/8vxu"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/8vxu.topology.json" -o "${RESULTS_DIR}/8vxu" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/8vxu/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  8vxu: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "8vxu,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_039_timing.csv"

# --- [4/4] 9atc ---
echo "[4/4] Running 9atc..."
mkdir -p "${RESULTS_DIR}/9atc"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/9atc.topology.json" -o "${RESULTS_DIR}/9atc" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/9atc/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  9atc: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "9atc,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_039_timing.csv"

echo ''
echo 'Batch 39 complete.'
echo ''

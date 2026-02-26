#!/usr/bin/env bash
# CryptoBench Batch 7/39 — 5 structures
# Structures: 1xgd, 1xjf, 1xqv, 1xtc, 1xxo
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 7/39'
echo '=========================================='
echo 'Structures: 1xgd 1xjf 1xqv 1xtc 1xxo'
echo ''

# --- [1/5] 1xgd ---
echo "[1/5] Running 1xgd..."
mkdir -p "${RESULTS_DIR}/1xgd"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1xgd.topology.json" -o "${RESULTS_DIR}/1xgd" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1xgd/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1xgd: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1xgd,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_007_timing.csv"

# --- [2/5] 1xjf ---
echo "[2/5] Running 1xjf..."
mkdir -p "${RESULTS_DIR}/1xjf"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1xjf.topology.json" -o "${RESULTS_DIR}/1xjf" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1xjf/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1xjf: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1xjf,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_007_timing.csv"

# --- [3/5] 1xqv ---
echo "[3/5] Running 1xqv..."
mkdir -p "${RESULTS_DIR}/1xqv"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1xqv.topology.json" -o "${RESULTS_DIR}/1xqv" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1xqv/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1xqv: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1xqv,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_007_timing.csv"

# --- [4/5] 1xtc ---
echo "[4/5] Running 1xtc..."
mkdir -p "${RESULTS_DIR}/1xtc"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1xtc.topology.json" -o "${RESULTS_DIR}/1xtc" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1xtc/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1xtc: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1xtc,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_007_timing.csv"

# --- [5/5] 1xxo ---
echo "[5/5] Running 1xxo..."
mkdir -p "${RESULTS_DIR}/1xxo"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1xxo.topology.json" -o "${RESULTS_DIR}/1xxo" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1xxo/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1xxo: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1xxo,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_007_timing.csv"

echo ''
echo 'Batch 7 complete.'
echo ''

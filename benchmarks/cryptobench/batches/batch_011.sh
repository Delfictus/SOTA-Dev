#!/usr/bin/env bash
# CryptoBench Batch 11/39 — 5 structures
# Structures: 2pwz, 2qbv, 2rfj, 2v6m, 2vl2
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 11/39'
echo '=========================================='
echo 'Structures: 2pwz 2qbv 2rfj 2v6m 2vl2'
echo ''

# --- [1/5] 2pwz ---
echo "[1/5] Running 2pwz..."
mkdir -p "${RESULTS_DIR}/2pwz"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2pwz.topology.json" -o "${RESULTS_DIR}/2pwz" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2pwz/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2pwz: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2pwz,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_011_timing.csv"

# --- [2/5] 2qbv ---
echo "[2/5] Running 2qbv..."
mkdir -p "${RESULTS_DIR}/2qbv"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2qbv.topology.json" -o "${RESULTS_DIR}/2qbv" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2qbv/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2qbv: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2qbv,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_011_timing.csv"

# --- [3/5] 2rfj ---
echo "[3/5] Running 2rfj..."
mkdir -p "${RESULTS_DIR}/2rfj"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2rfj.topology.json" -o "${RESULTS_DIR}/2rfj" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2rfj/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2rfj: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2rfj,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_011_timing.csv"

# --- [4/5] 2v6m ---
echo "[4/5] Running 2v6m..."
mkdir -p "${RESULTS_DIR}/2v6m"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2v6m.topology.json" -o "${RESULTS_DIR}/2v6m" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2v6m/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2v6m: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2v6m,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_011_timing.csv"

# --- [5/5] 2vl2 ---
echo "[5/5] Running 2vl2..."
mkdir -p "${RESULTS_DIR}/2vl2"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2vl2.topology.json" -o "${RESULTS_DIR}/2vl2" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2vl2/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2vl2: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2vl2,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_011_timing.csv"

echo ''
echo 'Batch 11 complete.'
echo ''

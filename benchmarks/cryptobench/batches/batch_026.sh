#!/usr/bin/env bash
# CryptoBench Batch 26/39 — 5 structures
# Structures: 5b0e, 5caz, 5dy9, 5e0v, 5ey7
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 26/39'
echo '=========================================='
echo 'Structures: 5b0e 5caz 5dy9 5e0v 5ey7'
echo ''

# --- [1/5] 5b0e ---
echo "[1/5] Running 5b0e..."
mkdir -p "${RESULTS_DIR}/5b0e"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5b0e.topology.json" -o "${RESULTS_DIR}/5b0e" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5b0e/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5b0e: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5b0e,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_026_timing.csv"

# --- [2/5] 5caz ---
echo "[2/5] Running 5caz..."
mkdir -p "${RESULTS_DIR}/5caz"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5caz.topology.json" -o "${RESULTS_DIR}/5caz" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5caz/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5caz: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5caz,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_026_timing.csv"

# --- [3/5] 5dy9 ---
echo "[3/5] Running 5dy9..."
mkdir -p "${RESULTS_DIR}/5dy9"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5dy9.topology.json" -o "${RESULTS_DIR}/5dy9" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5dy9/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5dy9: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5dy9,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_026_timing.csv"

# --- [4/5] 5e0v ---
echo "[4/5] Running 5e0v..."
mkdir -p "${RESULTS_DIR}/5e0v"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5e0v.topology.json" -o "${RESULTS_DIR}/5e0v" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5e0v/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5e0v: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5e0v,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_026_timing.csv"

# --- [5/5] 5ey7 ---
echo "[5/5] Running 5ey7..."
mkdir -p "${RESULTS_DIR}/5ey7"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5ey7.topology.json" -o "${RESULTS_DIR}/5ey7" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5ey7/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5ey7: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5ey7,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_026_timing.csv"

echo ''
echo 'Batch 26 complete.'
echo ''

#!/usr/bin/env bash
# CryptoBench Batch 3/39 — 5 structures
# Structures: 1h13, 1i7n, 1ksg, 1kx9, 1kxr
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 3/39'
echo '=========================================='
echo 'Structures: 1h13 1i7n 1ksg 1kx9 1kxr'
echo ''

# --- [1/5] 1h13 ---
echo "[1/5] Running 1h13..."
mkdir -p "${RESULTS_DIR}/1h13"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1h13.topology.json" -o "${RESULTS_DIR}/1h13" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1h13/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1h13: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1h13,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_003_timing.csv"

# --- [2/5] 1i7n ---
echo "[2/5] Running 1i7n..."
mkdir -p "${RESULTS_DIR}/1i7n"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1i7n.topology.json" -o "${RESULTS_DIR}/1i7n" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1i7n/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1i7n: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1i7n,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_003_timing.csv"

# --- [3/5] 1ksg ---
echo "[3/5] Running 1ksg..."
mkdir -p "${RESULTS_DIR}/1ksg"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1ksg.topology.json" -o "${RESULTS_DIR}/1ksg" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1ksg/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1ksg: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1ksg,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_003_timing.csv"

# --- [4/5] 1kx9 ---
echo "[4/5] Running 1kx9..."
mkdir -p "${RESULTS_DIR}/1kx9"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1kx9.topology.json" -o "${RESULTS_DIR}/1kx9" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1kx9/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1kx9: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1kx9,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_003_timing.csv"

# --- [5/5] 1kxr ---
echo "[5/5] Running 1kxr..."
mkdir -p "${RESULTS_DIR}/1kxr"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1kxr.topology.json" -o "${RESULTS_DIR}/1kxr" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1kxr/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1kxr: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1kxr,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_003_timing.csv"

echo ''
echo 'Batch 3 complete.'
echo ''

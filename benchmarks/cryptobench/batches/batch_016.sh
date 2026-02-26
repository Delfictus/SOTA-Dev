#!/usr/bin/env bash
# CryptoBench Batch 16/39 — 5 structures
# Structures: 3k01, 3kjr, 3la7, 3lnz, 3ly8
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 16/39'
echo '=========================================='
echo 'Structures: 3k01 3kjr 3la7 3lnz 3ly8'
echo ''

# --- [1/5] 3k01 ---
echo "[1/5] Running 3k01..."
mkdir -p "${RESULTS_DIR}/3k01"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3k01.topology.json" -o "${RESULTS_DIR}/3k01" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3k01/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3k01: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3k01,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_016_timing.csv"

# --- [2/5] 3kjr ---
echo "[2/5] Running 3kjr..."
mkdir -p "${RESULTS_DIR}/3kjr"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3kjr.topology.json" -o "${RESULTS_DIR}/3kjr" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3kjr/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3kjr: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3kjr,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_016_timing.csv"

# --- [3/5] 3la7 ---
echo "[3/5] Running 3la7..."
mkdir -p "${RESULTS_DIR}/3la7"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3la7.topology.json" -o "${RESULTS_DIR}/3la7" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3la7/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3la7: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3la7,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_016_timing.csv"

# --- [4/5] 3lnz ---
echo "[4/5] Running 3lnz..."
mkdir -p "${RESULTS_DIR}/3lnz"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3lnz.topology.json" -o "${RESULTS_DIR}/3lnz" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3lnz/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3lnz: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3lnz,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_016_timing.csv"

# --- [5/5] 3ly8 ---
echo "[5/5] Running 3ly8..."
mkdir -p "${RESULTS_DIR}/3ly8"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3ly8.topology.json" -o "${RESULTS_DIR}/3ly8" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3ly8/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3ly8: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3ly8,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_016_timing.csv"

echo ''
echo 'Batch 16 complete.'
echo ''

#!/usr/bin/env bash
# CryptoBench Batch 29/39 — 5 structures
# Structures: 5tvi, 5ujp, 5uxa, 5wbm, 5wm9
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 29/39'
echo '=========================================='
echo 'Structures: 5tvi 5ujp 5uxa 5wbm 5wm9'
echo ''

# --- [1/5] 5tvi ---
echo "[1/5] Running 5tvi..."
mkdir -p "${RESULTS_DIR}/5tvi"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5tvi.topology.json" -o "${RESULTS_DIR}/5tvi" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5tvi/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5tvi: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5tvi,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_029_timing.csv"

# --- [2/5] 5ujp ---
echo "[2/5] Running 5ujp..."
mkdir -p "${RESULTS_DIR}/5ujp"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5ujp.topology.json" -o "${RESULTS_DIR}/5ujp" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5ujp/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5ujp: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5ujp,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_029_timing.csv"

# --- [3/5] 5uxa ---
echo "[3/5] Running 5uxa..."
mkdir -p "${RESULTS_DIR}/5uxa"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5uxa.topology.json" -o "${RESULTS_DIR}/5uxa" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5uxa/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5uxa: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5uxa,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_029_timing.csv"

# --- [4/5] 5wbm ---
echo "[4/5] Running 5wbm..."
mkdir -p "${RESULTS_DIR}/5wbm"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5wbm.topology.json" -o "${RESULTS_DIR}/5wbm" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5wbm/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5wbm: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5wbm,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_029_timing.csv"

# --- [5/5] 5wm9 ---
echo "[5/5] Running 5wm9..."
mkdir -p "${RESULTS_DIR}/5wm9"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5wm9.topology.json" -o "${RESULTS_DIR}/5wm9" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5wm9/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5wm9: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5wm9,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_029_timing.csv"

echo ''
echo 'Batch 29 complete.'
echo ''

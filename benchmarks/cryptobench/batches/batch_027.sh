#!/usr/bin/env bash
# CryptoBench Batch 27/39 — 5 structures
# Structures: 5gmc, 5hij, 5igh, 5kcg, 5loc
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 27/39'
echo '=========================================='
echo 'Structures: 5gmc 5hij 5igh 5kcg 5loc'
echo ''

# --- [1/5] 5gmc ---
echo "[1/5] Running 5gmc..."
mkdir -p "${RESULTS_DIR}/5gmc"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5gmc.topology.json" -o "${RESULTS_DIR}/5gmc" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5gmc/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5gmc: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5gmc,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_027_timing.csv"

# --- [2/5] 5hij ---
echo "[2/5] Running 5hij..."
mkdir -p "${RESULTS_DIR}/5hij"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5hij.topology.json" -o "${RESULTS_DIR}/5hij" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5hij/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5hij: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5hij,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_027_timing.csv"

# --- [3/5] 5igh ---
echo "[3/5] Running 5igh..."
mkdir -p "${RESULTS_DIR}/5igh"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5igh.topology.json" -o "${RESULTS_DIR}/5igh" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5igh/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5igh: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5igh,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_027_timing.csv"

# --- [4/5] 5kcg ---
echo "[4/5] Running 5kcg..."
mkdir -p "${RESULTS_DIR}/5kcg"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5kcg.topology.json" -o "${RESULTS_DIR}/5kcg" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5kcg/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5kcg: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5kcg,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_027_timing.csv"

# --- [5/5] 5loc ---
echo "[5/5] Running 5loc..."
mkdir -p "${RESULTS_DIR}/5loc"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5loc.topology.json" -o "${RESULTS_DIR}/5loc" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5loc/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5loc: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5loc,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_027_timing.csv"

echo ''
echo 'Batch 27 complete.'
echo ''

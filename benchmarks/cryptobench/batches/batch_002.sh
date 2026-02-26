#!/usr/bin/env bash
# CryptoBench Batch 2/39 — 5 structures
# Structures: 1e6k, 1evy, 1fd4, 1fe6, 1g1m
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 2/39'
echo '=========================================='
echo 'Structures: 1e6k 1evy 1fd4 1fe6 1g1m'
echo ''

# --- [1/5] 1e6k ---
echo "[1/5] Running 1e6k..."
mkdir -p "${RESULTS_DIR}/1e6k"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1e6k.topology.json" -o "${RESULTS_DIR}/1e6k" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1e6k/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1e6k: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1e6k,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_002_timing.csv"

# --- [2/5] 1evy ---
echo "[2/5] Running 1evy..."
mkdir -p "${RESULTS_DIR}/1evy"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1evy.topology.json" -o "${RESULTS_DIR}/1evy" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1evy/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1evy: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1evy,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_002_timing.csv"

# --- [3/5] 1fd4 ---
echo "[3/5] Running 1fd4..."
mkdir -p "${RESULTS_DIR}/1fd4"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1fd4.topology.json" -o "${RESULTS_DIR}/1fd4" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1fd4/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1fd4: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1fd4,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_002_timing.csv"

# --- [4/5] 1fe6 ---
echo "[4/5] Running 1fe6..."
mkdir -p "${RESULTS_DIR}/1fe6"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1fe6.topology.json" -o "${RESULTS_DIR}/1fe6" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1fe6/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1fe6: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1fe6,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_002_timing.csv"

# --- [5/5] 1g1m ---
echo "[5/5] Running 1g1m..."
mkdir -p "${RESULTS_DIR}/1g1m"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1g1m.topology.json" -o "${RESULTS_DIR}/1g1m" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1g1m/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1g1m: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1g1m,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_002_timing.csv"

echo ''
echo 'Batch 2 complete.'
echo ''

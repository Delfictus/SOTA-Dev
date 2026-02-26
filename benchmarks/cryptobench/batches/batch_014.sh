#!/usr/bin/env bash
# CryptoBench Batch 14/39 — 5 structures
# Structures: 3bjp, 3f4k, 3flg, 3fzo, 3gdg
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 14/39'
echo '=========================================='
echo 'Structures: 3bjp 3f4k 3flg 3fzo 3gdg'
echo ''

# --- [1/5] 3bjp ---
echo "[1/5] Running 3bjp..."
mkdir -p "${RESULTS_DIR}/3bjp"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3bjp.topology.json" -o "${RESULTS_DIR}/3bjp" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3bjp/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3bjp: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3bjp,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_014_timing.csv"

# --- [2/5] 3f4k ---
echo "[2/5] Running 3f4k..."
mkdir -p "${RESULTS_DIR}/3f4k"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3f4k.topology.json" -o "${RESULTS_DIR}/3f4k" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3f4k/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3f4k: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3f4k,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_014_timing.csv"

# --- [3/5] 3flg ---
echo "[3/5] Running 3flg..."
mkdir -p "${RESULTS_DIR}/3flg"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3flg.topology.json" -o "${RESULTS_DIR}/3flg" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3flg/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3flg: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3flg,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_014_timing.csv"

# --- [4/5] 3fzo ---
echo "[4/5] Running 3fzo..."
mkdir -p "${RESULTS_DIR}/3fzo"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3fzo.topology.json" -o "${RESULTS_DIR}/3fzo" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3fzo/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3fzo: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3fzo,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_014_timing.csv"

# --- [5/5] 3gdg ---
echo "[5/5] Running 3gdg..."
mkdir -p "${RESULTS_DIR}/3gdg"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3gdg.topology.json" -o "${RESULTS_DIR}/3gdg" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3gdg/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3gdg: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3gdg,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_014_timing.csv"

echo ''
echo 'Batch 14 complete.'
echo ''

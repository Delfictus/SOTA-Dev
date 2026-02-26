#!/usr/bin/env bash
# CryptoBench Batch 1/39 — 5 structures
# Structures: 1arl, 1bk2, 1bzj, 1cwq, 1dq2
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 1/39'
echo '=========================================='
echo 'Structures: 1arl 1bk2 1bzj 1cwq 1dq2'
echo ''

# --- [1/5] 1arl ---
echo "[1/5] Running 1arl..."
mkdir -p "${RESULTS_DIR}/1arl"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1arl.topology.json" -o "${RESULTS_DIR}/1arl" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1arl/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1arl: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1arl,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_001_timing.csv"

# --- [2/5] 1bk2 ---
echo "[2/5] Running 1bk2..."
mkdir -p "${RESULTS_DIR}/1bk2"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1bk2.topology.json" -o "${RESULTS_DIR}/1bk2" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1bk2/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1bk2: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1bk2,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_001_timing.csv"

# --- [3/5] 1bzj ---
echo "[3/5] Running 1bzj..."
mkdir -p "${RESULTS_DIR}/1bzj"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1bzj.topology.json" -o "${RESULTS_DIR}/1bzj" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1bzj/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1bzj: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1bzj,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_001_timing.csv"

# --- [4/5] 1cwq ---
echo "[4/5] Running 1cwq..."
mkdir -p "${RESULTS_DIR}/1cwq"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1cwq.topology.json" -o "${RESULTS_DIR}/1cwq" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1cwq/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1cwq: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1cwq,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_001_timing.csv"

# --- [5/5] 1dq2 ---
echo "[5/5] Running 1dq2..."
mkdir -p "${RESULTS_DIR}/1dq2"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1dq2.topology.json" -o "${RESULTS_DIR}/1dq2" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1dq2/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1dq2: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1dq2,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_001_timing.csv"

echo ''
echo 'Batch 1 complete.'
echo ''

#!/usr/bin/env bash
# CryptoBench Batch 38/39 — 5 structures
# Structures: 8b9p, 8bre, 8gxj, 8h27, 8h49
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 38/39'
echo '=========================================='
echo 'Structures: 8b9p 8bre 8gxj 8h27 8h49'
echo ''

# --- [1/5] 8b9p ---
echo "[1/5] Running 8b9p..."
mkdir -p "${RESULTS_DIR}/8b9p"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/8b9p.topology.json" -o "${RESULTS_DIR}/8b9p" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/8b9p/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  8b9p: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "8b9p,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_038_timing.csv"

# --- [2/5] 8bre ---
echo "[2/5] Running 8bre..."
mkdir -p "${RESULTS_DIR}/8bre"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/8bre.topology.json" -o "${RESULTS_DIR}/8bre" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/8bre/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  8bre: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "8bre,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_038_timing.csv"

# --- [3/5] 8gxj ---
echo "[3/5] Running 8gxj..."
mkdir -p "${RESULTS_DIR}/8gxj"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/8gxj.topology.json" -o "${RESULTS_DIR}/8gxj" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/8gxj/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  8gxj: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "8gxj,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_038_timing.csv"

# --- [4/5] 8h27 ---
echo "[4/5] Running 8h27..."
mkdir -p "${RESULTS_DIR}/8h27"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/8h27.topology.json" -o "${RESULTS_DIR}/8h27" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/8h27/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  8h27: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "8h27,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_038_timing.csv"

# --- [5/5] 8h49 ---
echo "[5/5] Running 8h49..."
mkdir -p "${RESULTS_DIR}/8h49"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/8h49.topology.json" -o "${RESULTS_DIR}/8h49" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/8h49/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  8h49: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "8h49,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_038_timing.csv"

echo ''
echo 'Batch 38 complete.'
echo ''

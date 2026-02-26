#!/usr/bin/env bash
# CryptoBench Batch 12/39 — 5 structures
# Structures: 2vqz, 2vyr, 2w8n, 2x47, 2xdo
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 12/39'
echo '=========================================='
echo 'Structures: 2vqz 2vyr 2w8n 2x47 2xdo'
echo ''

# --- [1/5] 2vqz ---
echo "[1/5] Running 2vqz..."
mkdir -p "${RESULTS_DIR}/2vqz"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2vqz.topology.json" -o "${RESULTS_DIR}/2vqz" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2vqz/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2vqz: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2vqz,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_012_timing.csv"

# --- [2/5] 2vyr ---
echo "[2/5] Running 2vyr..."
mkdir -p "${RESULTS_DIR}/2vyr"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2vyr.topology.json" -o "${RESULTS_DIR}/2vyr" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2vyr/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2vyr: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2vyr,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_012_timing.csv"

# --- [3/5] 2w8n ---
echo "[3/5] Running 2w8n..."
mkdir -p "${RESULTS_DIR}/2w8n"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2w8n.topology.json" -o "${RESULTS_DIR}/2w8n" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2w8n/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2w8n: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2w8n,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_012_timing.csv"

# --- [4/5] 2x47 ---
echo "[4/5] Running 2x47..."
mkdir -p "${RESULTS_DIR}/2x47"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2x47.topology.json" -o "${RESULTS_DIR}/2x47" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2x47/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2x47: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2x47,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_012_timing.csv"

# --- [5/5] 2xdo ---
echo "[5/5] Running 2xdo..."
mkdir -p "${RESULTS_DIR}/2xdo"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2xdo.topology.json" -o "${RESULTS_DIR}/2xdo" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2xdo/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2xdo: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2xdo,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_012_timing.csv"

echo ''
echo 'Batch 12 complete.'
echo ''

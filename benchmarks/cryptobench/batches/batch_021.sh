#!/usr/bin/env bash
# CryptoBench Batch 21/39 — 5 structures
# Structures: 4dnc, 4e1y, 4fkm, 4gpi, 4hye
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 21/39'
echo '=========================================='
echo 'Structures: 4dnc 4e1y 4fkm 4gpi 4hye'
echo ''

# --- [1/5] 4dnc ---
echo "[1/5] Running 4dnc..."
mkdir -p "${RESULTS_DIR}/4dnc"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4dnc.topology.json" -o "${RESULTS_DIR}/4dnc" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4dnc/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4dnc: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4dnc,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_021_timing.csv"

# --- [2/5] 4e1y ---
echo "[2/5] Running 4e1y..."
mkdir -p "${RESULTS_DIR}/4e1y"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4e1y.topology.json" -o "${RESULTS_DIR}/4e1y" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4e1y/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4e1y: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4e1y,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_021_timing.csv"

# --- [3/5] 4fkm ---
echo "[3/5] Running 4fkm..."
mkdir -p "${RESULTS_DIR}/4fkm"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4fkm.topology.json" -o "${RESULTS_DIR}/4fkm" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4fkm/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4fkm: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4fkm,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_021_timing.csv"

# --- [4/5] 4gpi ---
echo "[4/5] Running 4gpi..."
mkdir -p "${RESULTS_DIR}/4gpi"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4gpi.topology.json" -o "${RESULTS_DIR}/4gpi" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4gpi/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4gpi: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4gpi,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_021_timing.csv"

# --- [5/5] 4hye ---
echo "[5/5] Running 4hye..."
mkdir -p "${RESULTS_DIR}/4hye"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4hye.topology.json" -o "${RESULTS_DIR}/4hye" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4hye/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4hye: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4hye,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_021_timing.csv"

echo ''
echo 'Batch 21 complete.'
echo ''

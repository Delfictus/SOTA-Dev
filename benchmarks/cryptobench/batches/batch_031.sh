#!/usr/bin/env bash
# CryptoBench Batch 31/39 — 5 structures
# Structures: 6bty, 6cqe, 6du4, 6eqj, 6f52
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 31/39'
echo '=========================================='
echo 'Structures: 6bty 6cqe 6du4 6eqj 6f52'
echo ''

# --- [1/5] 6bty ---
echo "[1/5] Running 6bty..."
mkdir -p "${RESULTS_DIR}/6bty"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6bty.topology.json" -o "${RESULTS_DIR}/6bty" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6bty/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6bty: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6bty,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_031_timing.csv"

# --- [2/5] 6cqe ---
echo "[2/5] Running 6cqe..."
mkdir -p "${RESULTS_DIR}/6cqe"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6cqe.topology.json" -o "${RESULTS_DIR}/6cqe" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6cqe/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6cqe: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6cqe,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_031_timing.csv"

# --- [3/5] 6du4 ---
echo "[3/5] Running 6du4..."
mkdir -p "${RESULTS_DIR}/6du4"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6du4.topology.json" -o "${RESULTS_DIR}/6du4" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6du4/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6du4: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6du4,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_031_timing.csv"

# --- [4/5] 6eqj ---
echo "[4/5] Running 6eqj..."
mkdir -p "${RESULTS_DIR}/6eqj"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6eqj.topology.json" -o "${RESULTS_DIR}/6eqj" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6eqj/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6eqj: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6eqj,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_031_timing.csv"

# --- [5/5] 6f52 ---
echo "[5/5] Running 6f52..."
mkdir -p "${RESULTS_DIR}/6f52"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/6f52.topology.json" -o "${RESULTS_DIR}/6f52" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/6f52/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  6f52: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "6f52,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_031_timing.csv"

echo ''
echo 'Batch 31 complete.'
echo ''

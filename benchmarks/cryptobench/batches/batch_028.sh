#!/usr/bin/env bash
# CryptoBench Batch 28/39 — 5 structures
# Structures: 5m7r, 5n49, 5o8b, 5sc2, 5tc0
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 28/39'
echo '=========================================='
echo 'Structures: 5m7r 5n49 5o8b 5sc2 5tc0'
echo ''

# --- [1/5] 5m7r ---
echo "[1/5] Running 5m7r..."
mkdir -p "${RESULTS_DIR}/5m7r"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5m7r.topology.json" -o "${RESULTS_DIR}/5m7r" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5m7r/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5m7r: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5m7r,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_028_timing.csv"

# --- [2/5] 5n49 ---
echo "[2/5] Running 5n49..."
mkdir -p "${RESULTS_DIR}/5n49"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5n49.topology.json" -o "${RESULTS_DIR}/5n49" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5n49/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5n49: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5n49,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_028_timing.csv"

# --- [3/5] 5o8b ---
echo "[3/5] Running 5o8b..."
mkdir -p "${RESULTS_DIR}/5o8b"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5o8b.topology.json" -o "${RESULTS_DIR}/5o8b" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5o8b/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5o8b: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5o8b,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_028_timing.csv"

# --- [4/5] 5sc2 ---
echo "[4/5] Running 5sc2..."
mkdir -p "${RESULTS_DIR}/5sc2"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5sc2.topology.json" -o "${RESULTS_DIR}/5sc2" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5sc2/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5sc2: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5sc2,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_028_timing.csv"

# --- [5/5] 5tc0 ---
echo "[5/5] Running 5tc0..."
mkdir -p "${RESULTS_DIR}/5tc0"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/5tc0.topology.json" -o "${RESULTS_DIR}/5tc0" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/5tc0/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  5tc0: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "5tc0,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_028_timing.csv"

echo ''
echo 'Batch 28 complete.'
echo ''

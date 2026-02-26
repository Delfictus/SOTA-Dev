#!/usr/bin/env bash
# CryptoBench Batch 22/39 — 5 structures
# Structures: 4ikv, 4ilg, 4j4e, 4jfr, 4kmy
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 22/39'
echo '=========================================='
echo 'Structures: 4ikv 4ilg 4j4e 4jfr 4kmy'
echo ''

# --- [1/5] 4ikv ---
echo "[1/5] Running 4ikv..."
mkdir -p "${RESULTS_DIR}/4ikv"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4ikv.topology.json" -o "${RESULTS_DIR}/4ikv" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4ikv/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4ikv: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4ikv,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_022_timing.csv"

# --- [2/5] 4ilg ---
echo "[2/5] Running 4ilg..."
mkdir -p "${RESULTS_DIR}/4ilg"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4ilg.topology.json" -o "${RESULTS_DIR}/4ilg" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4ilg/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4ilg: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4ilg,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_022_timing.csv"

# --- [3/5] 4j4e ---
echo "[3/5] Running 4j4e..."
mkdir -p "${RESULTS_DIR}/4j4e"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4j4e.topology.json" -o "${RESULTS_DIR}/4j4e" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4j4e/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4j4e: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4j4e,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_022_timing.csv"

# --- [4/5] 4jfr ---
echo "[4/5] Running 4jfr..."
mkdir -p "${RESULTS_DIR}/4jfr"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4jfr.topology.json" -o "${RESULTS_DIR}/4jfr" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4jfr/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4jfr: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4jfr,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_022_timing.csv"

# --- [5/5] 4kmy ---
echo "[5/5] Running 4kmy..."
mkdir -p "${RESULTS_DIR}/4kmy"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4kmy.topology.json" -o "${RESULTS_DIR}/4kmy" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4kmy/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4kmy: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4kmy,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_022_timing.csv"

echo ''
echo 'Batch 22 complete.'
echo ''

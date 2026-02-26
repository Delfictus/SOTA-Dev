#!/usr/bin/env bash
# CryptoBench Batch 23/39 — 5 structures
# Structures: 4mwi, 4nzv, 4oqo, 4p2f, 4p32
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 23/39'
echo '=========================================='
echo 'Structures: 4mwi 4nzv 4oqo 4p2f 4p32'
echo ''

# --- [1/5] 4mwi ---
echo "[1/5] Running 4mwi..."
mkdir -p "${RESULTS_DIR}/4mwi"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4mwi.topology.json" -o "${RESULTS_DIR}/4mwi" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4mwi/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4mwi: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4mwi,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_023_timing.csv"

# --- [2/5] 4nzv ---
echo "[2/5] Running 4nzv..."
mkdir -p "${RESULTS_DIR}/4nzv"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4nzv.topology.json" -o "${RESULTS_DIR}/4nzv" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4nzv/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4nzv: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4nzv,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_023_timing.csv"

# --- [3/5] 4oqo ---
echo "[3/5] Running 4oqo..."
mkdir -p "${RESULTS_DIR}/4oqo"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4oqo.topology.json" -o "${RESULTS_DIR}/4oqo" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4oqo/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4oqo: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4oqo,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_023_timing.csv"

# --- [4/5] 4p2f ---
echo "[4/5] Running 4p2f..."
mkdir -p "${RESULTS_DIR}/4p2f"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4p2f.topology.json" -o "${RESULTS_DIR}/4p2f" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4p2f/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4p2f: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4p2f,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_023_timing.csv"

# --- [5/5] 4p32 ---
echo "[5/5] Running 4p32..."
mkdir -p "${RESULTS_DIR}/4p32"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4p32.topology.json" -o "${RESULTS_DIR}/4p32" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4p32/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4p32: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4p32,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_023_timing.csv"

echo ''
echo 'Batch 23 complete.'
echo ''

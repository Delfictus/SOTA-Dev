#!/usr/bin/env bash
# CryptoBench Batch 13/39 — 5 structures
# Structures: 2xsa, 2zcg, 2zj7, 3a0x, 3b1o
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 13/39'
echo '=========================================='
echo 'Structures: 2xsa 2zcg 2zj7 3a0x 3b1o'
echo ''

# --- [1/5] 2xsa ---
echo "[1/5] Running 2xsa..."
mkdir -p "${RESULTS_DIR}/2xsa"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2xsa.topology.json" -o "${RESULTS_DIR}/2xsa" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2xsa/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2xsa: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2xsa,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_013_timing.csv"

# --- [2/5] 2zcg ---
echo "[2/5] Running 2zcg..."
mkdir -p "${RESULTS_DIR}/2zcg"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2zcg.topology.json" -o "${RESULTS_DIR}/2zcg" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2zcg/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2zcg: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2zcg,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_013_timing.csv"

# --- [3/5] 2zj7 ---
echo "[3/5] Running 2zj7..."
mkdir -p "${RESULTS_DIR}/2zj7"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2zj7.topology.json" -o "${RESULTS_DIR}/2zj7" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2zj7/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2zj7: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2zj7,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_013_timing.csv"

# --- [4/5] 3a0x ---
echo "[4/5] Running 3a0x..."
mkdir -p "${RESULTS_DIR}/3a0x"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3a0x.topology.json" -o "${RESULTS_DIR}/3a0x" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3a0x/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3a0x: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3a0x,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_013_timing.csv"

# --- [5/5] 3b1o ---
echo "[5/5] Running 3b1o..."
mkdir -p "${RESULTS_DIR}/3b1o"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3b1o.topology.json" -o "${RESULTS_DIR}/3b1o" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3b1o/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3b1o: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3b1o,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_013_timing.csv"

echo ''
echo 'Batch 13 complete.'
echo ''

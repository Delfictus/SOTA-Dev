#!/usr/bin/env bash
# CryptoBench Batch 6/39 — 5 structures
# Structures: 1tmi, 1uka, 1ute, 1vsn, 1x2g
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 6/39'
echo '=========================================='
echo 'Structures: 1tmi 1uka 1ute 1vsn 1x2g'
echo ''

# --- [1/5] 1tmi ---
echo "[1/5] Running 1tmi..."
mkdir -p "${RESULTS_DIR}/1tmi"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1tmi.topology.json" -o "${RESULTS_DIR}/1tmi" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1tmi/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1tmi: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1tmi,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_006_timing.csv"

# --- [2/5] 1uka ---
echo "[2/5] Running 1uka..."
mkdir -p "${RESULTS_DIR}/1uka"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1uka.topology.json" -o "${RESULTS_DIR}/1uka" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1uka/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1uka: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1uka,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_006_timing.csv"

# --- [3/5] 1ute ---
echo "[3/5] Running 1ute..."
mkdir -p "${RESULTS_DIR}/1ute"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1ute.topology.json" -o "${RESULTS_DIR}/1ute" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1ute/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1ute: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1ute,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_006_timing.csv"

# --- [4/5] 1vsn ---
echo "[4/5] Running 1vsn..."
mkdir -p "${RESULTS_DIR}/1vsn"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1vsn.topology.json" -o "${RESULTS_DIR}/1vsn" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1vsn/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1vsn: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1vsn,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_006_timing.csv"

# --- [5/5] 1x2g ---
echo "[5/5] Running 1x2g..."
mkdir -p "${RESULTS_DIR}/1x2g"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/1x2g.topology.json" -o "${RESULTS_DIR}/1x2g" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/1x2g/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  1x2g: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "1x2g,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_006_timing.csv"

echo ''
echo 'Batch 6 complete.'
echo ''

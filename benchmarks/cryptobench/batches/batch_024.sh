#!/usr/bin/env bash
# CryptoBench Batch 24/39 — 5 structures
# Structures: 4qvk, 4r0x, 4rvt, 4ttp, 4uum
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 24/39'
echo '=========================================='
echo 'Structures: 4qvk 4r0x 4rvt 4ttp 4uum'
echo ''

# --- [1/5] 4qvk ---
echo "[1/5] Running 4qvk..."
mkdir -p "${RESULTS_DIR}/4qvk"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4qvk.topology.json" -o "${RESULTS_DIR}/4qvk" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4qvk/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4qvk: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4qvk,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_024_timing.csv"

# --- [2/5] 4r0x ---
echo "[2/5] Running 4r0x..."
mkdir -p "${RESULTS_DIR}/4r0x"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4r0x.topology.json" -o "${RESULTS_DIR}/4r0x" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4r0x/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4r0x: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4r0x,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_024_timing.csv"

# --- [3/5] 4rvt ---
echo "[3/5] Running 4rvt..."
mkdir -p "${RESULTS_DIR}/4rvt"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4rvt.topology.json" -o "${RESULTS_DIR}/4rvt" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4rvt/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4rvt: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4rvt,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_024_timing.csv"

# --- [4/5] 4ttp ---
echo "[4/5] Running 4ttp..."
mkdir -p "${RESULTS_DIR}/4ttp"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4ttp.topology.json" -o "${RESULTS_DIR}/4ttp" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4ttp/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4ttp: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4ttp,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_024_timing.csv"

# --- [5/5] 4uum ---
echo "[5/5] Running 4uum..."
mkdir -p "${RESULTS_DIR}/4uum"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/4uum.topology.json" -o "${RESULTS_DIR}/4uum" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/4uum/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  4uum: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "4uum,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_024_timing.csv"

echo ''
echo 'Batch 24 complete.'
echo ''

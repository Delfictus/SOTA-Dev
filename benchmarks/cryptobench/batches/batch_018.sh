#!/usr/bin/env bash
# CryptoBench Batch 18/39 — 5 structures
# Structures: 3rwv, 3st6, 3t8b, 3tpo, 3ugk
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 18/39'
echo '=========================================='
echo 'Structures: 3rwv 3st6 3t8b 3tpo 3ugk'
echo ''

# --- [1/5] 3rwv ---
echo "[1/5] Running 3rwv..."
mkdir -p "${RESULTS_DIR}/3rwv"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3rwv.topology.json" -o "${RESULTS_DIR}/3rwv" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3rwv/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3rwv: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3rwv,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_018_timing.csv"

# --- [2/5] 3st6 ---
echo "[2/5] Running 3st6..."
mkdir -p "${RESULTS_DIR}/3st6"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3st6.topology.json" -o "${RESULTS_DIR}/3st6" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3st6/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3st6: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3st6,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_018_timing.csv"

# --- [3/5] 3t8b ---
echo "[3/5] Running 3t8b..."
mkdir -p "${RESULTS_DIR}/3t8b"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3t8b.topology.json" -o "${RESULTS_DIR}/3t8b" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3t8b/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3t8b: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3t8b,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_018_timing.csv"

# --- [4/5] 3tpo ---
echo "[4/5] Running 3tpo..."
mkdir -p "${RESULTS_DIR}/3tpo"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3tpo.topology.json" -o "${RESULTS_DIR}/3tpo" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3tpo/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3tpo: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3tpo,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_018_timing.csv"

# --- [5/5] 3ugk ---
echo "[5/5] Running 3ugk..."
mkdir -p "${RESULTS_DIR}/3ugk"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3ugk.topology.json" -o "${RESULTS_DIR}/3ugk" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3ugk/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3ugk: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3ugk,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_018_timing.csv"

echo ''
echo 'Batch 18 complete.'
echo ''

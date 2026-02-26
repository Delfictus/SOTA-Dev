#!/usr/bin/env bash
# CryptoBench Batch 15/39 — 5 structures
# Structures: 3h8a, 3hrm, 3i8s, 3idh, 3jzg
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 15/39'
echo '=========================================='
echo 'Structures: 3h8a 3hrm 3i8s 3idh 3jzg'
echo ''

# --- [1/5] 3h8a ---
echo "[1/5] Running 3h8a..."
mkdir -p "${RESULTS_DIR}/3h8a"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3h8a.topology.json" -o "${RESULTS_DIR}/3h8a" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3h8a/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3h8a: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3h8a,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_015_timing.csv"

# --- [2/5] 3hrm ---
echo "[2/5] Running 3hrm..."
mkdir -p "${RESULTS_DIR}/3hrm"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3hrm.topology.json" -o "${RESULTS_DIR}/3hrm" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3hrm/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3hrm: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3hrm,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_015_timing.csv"

# --- [3/5] 3i8s ---
echo "[3/5] Running 3i8s..."
mkdir -p "${RESULTS_DIR}/3i8s"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3i8s.topology.json" -o "${RESULTS_DIR}/3i8s" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3i8s/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3i8s: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3i8s,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_015_timing.csv"

# --- [4/5] 3idh ---
echo "[4/5] Running 3idh..."
mkdir -p "${RESULTS_DIR}/3idh"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3idh.topology.json" -o "${RESULTS_DIR}/3idh" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3idh/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3idh: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3idh,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_015_timing.csv"

# --- [5/5] 3jzg ---
echo "[5/5] Running 3jzg..."
mkdir -p "${RESULTS_DIR}/3jzg"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3jzg.topology.json" -o "${RESULTS_DIR}/3jzg" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3jzg/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3jzg: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3jzg,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_015_timing.csv"

echo ''
echo 'Batch 15 complete.'
echo ''

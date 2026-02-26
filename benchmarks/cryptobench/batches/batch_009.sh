#!/usr/bin/env bash
# CryptoBench Batch 9/39 — 5 structures
# Structures: 2fem, 2fhz, 2h7s, 2huw, 2i3a
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 9/39'
echo '=========================================='
echo 'Structures: 2fem 2fhz 2h7s 2huw 2i3a'
echo ''

# --- [1/5] 2fem ---
echo "[1/5] Running 2fem..."
mkdir -p "${RESULTS_DIR}/2fem"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2fem.topology.json" -o "${RESULTS_DIR}/2fem" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2fem/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2fem: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2fem,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_009_timing.csv"

# --- [2/5] 2fhz ---
echo "[2/5] Running 2fhz..."
mkdir -p "${RESULTS_DIR}/2fhz"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2fhz.topology.json" -o "${RESULTS_DIR}/2fhz" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2fhz/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2fhz: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2fhz,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_009_timing.csv"

# --- [3/5] 2h7s ---
echo "[3/5] Running 2h7s..."
mkdir -p "${RESULTS_DIR}/2h7s"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2h7s.topology.json" -o "${RESULTS_DIR}/2h7s" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2h7s/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2h7s: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2h7s,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_009_timing.csv"

# --- [4/5] 2huw ---
echo "[4/5] Running 2huw..."
mkdir -p "${RESULTS_DIR}/2huw"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2huw.topology.json" -o "${RESULTS_DIR}/2huw" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2huw/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2huw: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2huw,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_009_timing.csv"

# --- [5/5] 2i3a ---
echo "[5/5] Running 2i3a..."
mkdir -p "${RESULTS_DIR}/2i3a"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2i3a.topology.json" -o "${RESULTS_DIR}/2i3a" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2i3a/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2i3a: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2i3a,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_009_timing.csv"

echo ''
echo 'Batch 9 complete.'
echo ''

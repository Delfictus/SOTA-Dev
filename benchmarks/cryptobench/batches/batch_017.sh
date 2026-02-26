#!/usr/bin/env bash
# CryptoBench Batch 17/39 — 5 structures
# Structures: 3mwg, 3n4u, 3nx1, 3pbf, 3pfp
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 17/39'
echo '=========================================='
echo 'Structures: 3mwg 3n4u 3nx1 3pbf 3pfp'
echo ''

# --- [1/5] 3mwg ---
echo "[1/5] Running 3mwg..."
mkdir -p "${RESULTS_DIR}/3mwg"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3mwg.topology.json" -o "${RESULTS_DIR}/3mwg" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3mwg/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3mwg: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3mwg,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_017_timing.csv"

# --- [2/5] 3n4u ---
echo "[2/5] Running 3n4u..."
mkdir -p "${RESULTS_DIR}/3n4u"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3n4u.topology.json" -o "${RESULTS_DIR}/3n4u" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3n4u/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3n4u: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3n4u,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_017_timing.csv"

# --- [3/5] 3nx1 ---
echo "[3/5] Running 3nx1..."
mkdir -p "${RESULTS_DIR}/3nx1"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3nx1.topology.json" -o "${RESULTS_DIR}/3nx1" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3nx1/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3nx1: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3nx1,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_017_timing.csv"

# --- [4/5] 3pbf ---
echo "[4/5] Running 3pbf..."
mkdir -p "${RESULTS_DIR}/3pbf"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3pbf.topology.json" -o "${RESULTS_DIR}/3pbf" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3pbf/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3pbf: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3pbf,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_017_timing.csv"

# --- [5/5] 3pfp ---
echo "[5/5] Running 3pfp..."
mkdir -p "${RESULTS_DIR}/3pfp"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/3pfp.topology.json" -o "${RESULTS_DIR}/3pfp" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/3pfp/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  3pfp: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "3pfp,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_017_timing.csv"

echo ''
echo 'Batch 17 complete.'
echo ''

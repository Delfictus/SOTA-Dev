#!/usr/bin/env bash
# CryptoBench Batch 10/39 — 5 structures
# Structures: 2i3r, 2idj, 2iyt, 2phz, 2pkf
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

echo '=========================================='
echo '  CryptoBench Batch 10/39'
echo '=========================================='
echo 'Structures: 2i3r 2idj 2iyt 2phz 2pkf'
echo ''

# --- [1/5] 2i3r ---
echo "[1/5] Running 2i3r..."
mkdir -p "${RESULTS_DIR}/2i3r"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2i3r.topology.json" -o "${RESULTS_DIR}/2i3r" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2i3r/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2i3r: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2i3r,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_010_timing.csv"

# --- [2/5] 2idj ---
echo "[2/5] Running 2idj..."
mkdir -p "${RESULTS_DIR}/2idj"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2idj.topology.json" -o "${RESULTS_DIR}/2idj" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2idj/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2idj: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2idj,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_010_timing.csv"

# --- [3/5] 2iyt ---
echo "[3/5] Running 2iyt..."
mkdir -p "${RESULTS_DIR}/2iyt"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2iyt.topology.json" -o "${RESULTS_DIR}/2iyt" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2iyt/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2iyt: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2iyt,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_010_timing.csv"

# --- [4/5] 2phz ---
echo "[4/5] Running 2phz..."
mkdir -p "${RESULTS_DIR}/2phz"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2phz.topology.json" -o "${RESULTS_DIR}/2phz" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2phz/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2phz: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2phz,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_010_timing.csv"

# --- [5/5] 2pkf ---
echo "[5/5] Running 2pkf..."
mkdir -p "${RESULTS_DIR}/2pkf"
START=$(date +%s)
"$NHS" -t "${TOPO_DIR}/2pkf.topology.json" -o "${RESULTS_DIR}/2pkf" --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v 2>&1 | tee "${RESULTS_DIR}/2pkf/run.log"
EXIT_CODE=${PIPESTATUS[0]}
END=$(date +%s)
ELAPSED=$((END - START))
echo "  2pkf: exit=${EXIT_CODE}, time=${ELAPSED}s"
echo "2pkf,${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_010_timing.csv"

echo ''
echo 'Batch 10 complete.'
echo ''

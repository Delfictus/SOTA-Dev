#!/usr/bin/env bash
# CryptoBench Master Runner — All batches sequentially
# Total: 194 structures in 39 batches of 5
set -euo pipefail

BENCH_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench"
BATCHES_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/batches"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"

echo '============================================='
echo '  CryptoBench Full Benchmark Run'
echo '  194 structures / 39 batches'
echo '============================================='
echo ''

START_ALL=$(date +%s)

echo "=== Batch 1/39 ==="
bash "${BATCHES_DIR}/batch_001.sh"

echo "=== Batch 2/39 ==="
bash "${BATCHES_DIR}/batch_002.sh"

echo "=== Batch 3/39 ==="
bash "${BATCHES_DIR}/batch_003.sh"

echo "=== Batch 4/39 ==="
bash "${BATCHES_DIR}/batch_004.sh"

echo "=== Batch 5/39 ==="
bash "${BATCHES_DIR}/batch_005.sh"

echo "=== Batch 6/39 ==="
bash "${BATCHES_DIR}/batch_006.sh"

echo "=== Batch 7/39 ==="
bash "${BATCHES_DIR}/batch_007.sh"

echo "=== Batch 8/39 ==="
bash "${BATCHES_DIR}/batch_008.sh"

echo "=== Batch 9/39 ==="
bash "${BATCHES_DIR}/batch_009.sh"

echo "=== Batch 10/39 ==="
bash "${BATCHES_DIR}/batch_010.sh"

echo "=== Batch 11/39 ==="
bash "${BATCHES_DIR}/batch_011.sh"

echo "=== Batch 12/39 ==="
bash "${BATCHES_DIR}/batch_012.sh"

echo "=== Batch 13/39 ==="
bash "${BATCHES_DIR}/batch_013.sh"

echo "=== Batch 14/39 ==="
bash "${BATCHES_DIR}/batch_014.sh"

echo "=== Batch 15/39 ==="
bash "${BATCHES_DIR}/batch_015.sh"

echo "=== Batch 16/39 ==="
bash "${BATCHES_DIR}/batch_016.sh"

echo "=== Batch 17/39 ==="
bash "${BATCHES_DIR}/batch_017.sh"

echo "=== Batch 18/39 ==="
bash "${BATCHES_DIR}/batch_018.sh"

echo "=== Batch 19/39 ==="
bash "${BATCHES_DIR}/batch_019.sh"

echo "=== Batch 20/39 ==="
bash "${BATCHES_DIR}/batch_020.sh"

echo "=== Batch 21/39 ==="
bash "${BATCHES_DIR}/batch_021.sh"

echo "=== Batch 22/39 ==="
bash "${BATCHES_DIR}/batch_022.sh"

echo "=== Batch 23/39 ==="
bash "${BATCHES_DIR}/batch_023.sh"

echo "=== Batch 24/39 ==="
bash "${BATCHES_DIR}/batch_024.sh"

echo "=== Batch 25/39 ==="
bash "${BATCHES_DIR}/batch_025.sh"

echo "=== Batch 26/39 ==="
bash "${BATCHES_DIR}/batch_026.sh"

echo "=== Batch 27/39 ==="
bash "${BATCHES_DIR}/batch_027.sh"

echo "=== Batch 28/39 ==="
bash "${BATCHES_DIR}/batch_028.sh"

echo "=== Batch 29/39 ==="
bash "${BATCHES_DIR}/batch_029.sh"

echo "=== Batch 30/39 ==="
bash "${BATCHES_DIR}/batch_030.sh"

echo "=== Batch 31/39 ==="
bash "${BATCHES_DIR}/batch_031.sh"

echo "=== Batch 32/39 ==="
bash "${BATCHES_DIR}/batch_032.sh"

echo "=== Batch 33/39 ==="
bash "${BATCHES_DIR}/batch_033.sh"

echo "=== Batch 34/39 ==="
bash "${BATCHES_DIR}/batch_034.sh"

echo "=== Batch 35/39 ==="
bash "${BATCHES_DIR}/batch_035.sh"

echo "=== Batch 36/39 ==="
bash "${BATCHES_DIR}/batch_036.sh"

echo "=== Batch 37/39 ==="
bash "${BATCHES_DIR}/batch_037.sh"

echo "=== Batch 38/39 ==="
bash "${BATCHES_DIR}/batch_038.sh"

echo "=== Batch 39/39 ==="
bash "${BATCHES_DIR}/batch_039.sh"

END_ALL=$(date +%s)
TOTAL_TIME=$((END_ALL - START_ALL))
echo "============================================="
echo "  CryptoBench complete: 194 structures"
echo "  Total time: ${TOTAL_TIME}s ($(( TOTAL_TIME / 60 ))m $(( TOTAL_TIME % 60 ))s)"
echo "============================================="

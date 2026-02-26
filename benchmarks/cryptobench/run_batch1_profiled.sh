#!/usr/bin/env bash
# CryptoBench Batch 1 — 5 structures with full nsys kernel tracing
set -euo pipefail

TOPO_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/topologies"
RESULTS_DIR="/home/diddy/Desktop/Prism4D-bio/benchmarks/cryptobench/results"
NHS="/home/diddy/Desktop/Prism4D-bio/target/release/nhs_rt_full"

# nhs_rt_full looks for PTX kernels via relative paths from cwd
cd /home/diddy/Desktop/Prism4D-bio

NSYS_FLAGS="--trace=cuda,nvtx,osrt,cudnn,cublas --cuda-memory-usage=true --cudabacktrace=all --stats=true --export=sqlite --force-overwrite true"

for PDB in 1arl 1bk2 1bzj 1cwq 1dq2; do
    echo "=========================================="
    echo "  Running ${PDB} with nsys profiling"
    echo "=========================================="
    mkdir -p "${RESULTS_DIR}/${PDB}"
    START=$(date +%s)
    nsys profile ${NSYS_FLAGS} \
        -o "${RESULTS_DIR}/${PDB}/${PDB}_profile" \
        "${NHS}" \
        -t "${TOPO_DIR}/${PDB}.topology.json" \
        -o "${RESULTS_DIR}/${PDB}" \
        --fast --hysteresis --multi-stream 8 \
        --spike-percentile 95 --rt-clustering -v
    EXIT_CODE=$?
    END=$(date +%s)
    ELAPSED=$((END - START))
    echo "  ${PDB}: exit=${EXIT_CODE}, time=${ELAPSED}s"
    echo "${PDB},${EXIT_CODE},${ELAPSED}" >> "${RESULTS_DIR}/batch_001_timing.csv"
    echo ""
done

echo "Batch 1 complete."

#!/usr/bin/env bash
# Nsight Compute kernel profile for 1arl — 10 kernel samples after 1000 warmup
set -euo pipefail

cd /home/diddy/Desktop/Prism4D-bio

NCU="/opt/nvidia/nsight-compute/2025.4.1/ncu"
NHS="./target/release/nhs_rt_full"
TOPO="benchmarks/cryptobench/topologies/1arl.topology.json"
OUT_DIR="benchmarks/cryptobench/results/1arl"
PROFILE_OUT="${OUT_DIR}/1arl_kernel_profile"

mkdir -p "${OUT_DIR}"

"${NCU}" --set basic \
    --replay-mode application \
    --launch-skip 1000 --launch-count 5 \
    -o "${PROFILE_OUT}" \
    "${NHS}" \
    -t "${TOPO}" \
    -o "${OUT_DIR}/ncu_test" \
    --fast --hysteresis --multi-stream 8 \
    --spike-percentile 95 --rt-clustering -v

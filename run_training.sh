#!/bin/bash
# PRISM-Zero v3.1 Training Runner
# Usage: ./run_training.sh [manifest] [output_dir]

set -e

# Environment setup for libtorch
export LIBTORCH=/home/diddy/libtorch
export LD_LIBRARY_PATH=/home/diddy/libtorch/lib:$LD_LIBRARY_PATH

# Defaults
MANIFEST="${1:-data/manifests/combined_calibration.json}"
OUTPUT="${2:-training_output}"

cd "$(dirname "$0")"

echo "=============================================="
echo "  PRISM-Zero v3.1 - Self-Calibrating Engine"
echo "=============================================="
echo "Manifest: $MANIFEST"
echo "Output:   $OUTPUT"
echo ""

./target/release/prism-train \
    --manifest "$MANIFEST" \
    --output "$OUTPUT" \
    --macro-steps \
    --parallel \
    --parallel-jobs 4 \
    --checkpoint-interval 50 \
    --verbose

echo ""
echo "Training complete! Results saved to: $OUTPUT"

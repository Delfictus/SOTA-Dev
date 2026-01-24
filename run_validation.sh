#!/bin/bash
# PRISM-Zero v3.1 Discovery Validation Runner
# Usage: ./run_validation.sh [holdout_manifest] [output_dir]

set -e

# Environment setup for libtorch
export LIBTORCH=/home/diddy/libtorch
export LD_LIBRARY_PATH=/home/diddy/libtorch/lib:$LD_LIBRARY_PATH

# Defaults
MANIFEST="${1:-data/manifests/spike_calibration.json}"
OUTPUT="${2:-validation_output}"
RESULTS="${3:-training_output/training_results.json}"

cd "$(dirname "$0")"

echo "=============================================="
echo "  PRISM-Zero v3.1 - Discovery Validation"
echo "=============================================="
echo "Holdout Manifest: $MANIFEST"
echo "Output Directory: $OUTPUT"
echo "Results File:     $RESULTS"
echo ""

./target/release/prism-validate \
    --manifest "$MANIFEST" \
    --output "$OUTPUT" \
    --results "$RESULTS" \
    --steps 100000 \
    --temperature 1.5 \
    --friction 0.1 \
    --spring-k 5.0 \
    --bias-strength 0.5 \
    --verbose

echo ""
echo "=============================================="
echo "  Validation Complete!"
echo "=============================================="
echo "Outputs saved to: $OUTPUT"
echo ""
echo "Per-target artifacts:"
echo "  - {target}_relaxed.pdb  (Digital Twin)"
echo "  - residue_scores.csv    (Treasure Map)"
echo "  - validation_report.json"

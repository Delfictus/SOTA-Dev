#!/bin/bash
# Example workflow for PRISM-CryoUV cryptic site detection

set -e

# Configuration
PDB_FILE="${1:-example.pdb}"
OUTPUT_DIR="${2:-./results}"

# Check if input exists
if [ ! -f "$PDB_FILE" ]; then
    echo "Usage: $0 <input.pdb> [output_dir]"
    echo "Error: PDB file not found: $PDB_FILE"
    exit 1
fi

BASENAME=$(basename "$PDB_FILE" .pdb)
TOPOLOGY="${OUTPUT_DIR}/${BASENAME}_topology.json"

echo "=============================================="
echo "PRISM-CryoUV Cryptic Site Detection Pipeline"
echo "=============================================="
echo "Input:  $PDB_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Preprocessing
echo "[1/3] Preprocessing with prism-prep..."
prism-prep "$PDB_FILE" "$TOPOLOGY" \
    --use-amber \
    --mode cryptic \
    --strict \
    -v

echo ""
echo "[2/3] Running Cryo-UV simulation..."
nhs-adaptive \
    --topology "$TOPOLOGY" \
    --output "$OUTPUT_DIR" \
    --survey-steps 500000 \
    --convergence-steps 250000 \
    --precision-steps 250000 \
    --temperature 300.0 \
    --cryo-temp 100.0

echo ""
echo "[3/3] Complete!"
echo ""
echo "Results written to: $OUTPUT_DIR"
echo "  - summary.json: Detection summary"
echo "  - spikes.json: Spike events"
echo "  - ensemble.pdb: Conformations"
echo ""

#!/bin/bash
# PRISM-Cryptic Accelerated Mode Example
# Uses HMR topology with 4 replicas and 4fs timestep

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(dirname "$SCRIPT_DIR")"

echo "PRISM-Cryptic Accelerated Mode"
echo "=============================="
echo ""

if [ -z "$1" ]; then
    echo "Usage: $0 <input.pdb>"
    exit 1
fi

INPUT_PDB="$1"
BASENAME=$(basename "$INPUT_PDB" .pdb)

# Create HMR topology (required for 4fs timestep)
echo "Step 1: Creating HMR topology..."
"$RELEASE_DIR/scripts/prism-prep" "$INPUT_PDB" "${BASENAME}_hmr_topology.json" \
    --use-amber --mode cryptic --hmr --strict
echo "  ✓ HMR topology generated"

# Run accelerated detection
echo "Step 2: Running accelerated detection (4 replicas, 4fs)..."
mkdir -p results_accelerated
"$RELEASE_DIR/bin/prism-cryptic" detect \
    --topology "${BASENAME}_hmr_topology.json" \
    --output-dir results_accelerated/ \
    --accelerated \
    --verbose

echo ""
echo "Results saved to: results_accelerated/"

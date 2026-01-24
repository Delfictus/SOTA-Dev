#!/bin/bash
# PRISM-Cryptic Quick Start Example
# Downloads and analyzes TEM-1 β-lactamase (1BTL)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(dirname "$SCRIPT_DIR")"

echo "PRISM-Cryptic Quick Start"
echo "========================="
echo ""

# Check if prism-cryptic is available
if [ ! -f "$RELEASE_DIR/bin/prism-cryptic" ]; then
    echo "Error: prism-cryptic not found in $RELEASE_DIR/bin/"
    exit 1
fi

# Download test structure
echo "Step 1: Downloading 1BTL.pdb..."
mkdir -p data
wget -q "https://files.rcsb.org/download/1BTL.pdb" -O data/1BTL.pdb
echo "  ✓ Downloaded"

# Preprocess
echo "Step 2: Preprocessing with prism-prep..."
"$RELEASE_DIR/scripts/prism-prep" data/1BTL.pdb data/1BTL_topology.json \
    --use-amber --mode cryptic --strict
echo "  ✓ Topology generated"

# Run detection
echo "Step 3: Running cryptic site detection..."
mkdir -p results/1BTL
"$RELEASE_DIR/bin/prism-cryptic" detect \
    --topology data/1BTL_topology.json \
    --output-dir results/1BTL/
echo "  ✓ Detection complete"

# Show results
echo ""
echo "Results:"
echo "========"
cat results/1BTL/*_cryptic_sites.txt
echo ""
echo "Full results saved to: results/1BTL/"

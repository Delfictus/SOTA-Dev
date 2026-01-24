#!/bin/bash
# PRISM-Cryptic Batch Processing Example

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RELEASE_DIR="$(dirname "$SCRIPT_DIR")"

echo "PRISM-Cryptic Batch Processing Example"
echo "======================================="
echo ""

# Create manifest
cat > batch_manifest.txt << MANIFEST
# List of topology files (one per line)
data/1BTL_topology.json
data/1A9U_topology.json
data/1M47_topology.json
MANIFEST

echo "Manifest created: batch_manifest.txt"
echo ""

# Run batch (assuming topologies already exist)
"$RELEASE_DIR/bin/prism-cryptic" batch \
    --manifest batch_manifest.txt \
    --output-dir results_batch/ \
    --verbose \
    --continue-on-error

echo ""
echo "Batch processing complete. Results in: results_batch/"

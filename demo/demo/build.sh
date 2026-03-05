#!/bin/bash
set -euo pipefail

# PRISM-4D Demo Container Build Script
# Run from: ~/Desktop/Prism4D-bio/demo/

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== PRISM-4D Demo Container Build ==="
echo "Project root: $PROJECT_ROOT"
echo "Build dir:    $SCRIPT_DIR"
echo ""

# Stage binary
echo "[1/4] Staging nhs_rt_full binary..."
cp "$PROJECT_ROOT/target/release/nhs_rt_full" "$SCRIPT_DIR/nhs_rt_full"

# Stage prism-prep
echo "[2/4] Staging prism-prep script..."
cp "$PROJECT_ROOT/scripts/prism-prep" "$SCRIPT_DIR/prism-prep"

# Stage sample PDB files
echo "[3/4] Staging sample PDB files..."
mkdir -p "$SCRIPT_DIR/samples"
for pdb in 1btl.pdb 1ade.pdb 3k5v.pdb 3l15_chainA.pdb; do
    if [ -f "$PROJECT_ROOT/$pdb" ]; then
        cp "$PROJECT_ROOT/$pdb" "$SCRIPT_DIR/samples/"
        echo "  - $pdb"
    fi
done

# Build Docker image
echo "[4/4] Building Docker image..."
docker compose build

echo ""
echo "=== Build complete ==="
echo ""
echo "To start:   docker compose up -d"
echo "To test:    ssh demo@localhost -p 2222"
echo "To stop:    docker compose down"
echo "Password:   prism4d-demo (change in docker-compose.yml DEMO_PASSWORD)"

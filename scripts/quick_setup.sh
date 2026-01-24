#!/bin/bash
# PRISM Quick Setup - Run this first!
#
# This script:
# 1. Checks for conda/mamba
# 2. Installs OpenMM and pdbfixer (for proper structure preparation)
# 3. Downloads test structures
# 4. Prepares 1L2Y (tiny) and 1UBQ (small) for immediate testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PRISM_DATA:-$HOME/prism_data}"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                  PRISM-4D Quick Setup                            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "This prepares protein structures with OpenMM/pdbfixer:"
echo "  - Adds ALL missing atoms including hydrogens"
echo "  - Proper solvation with TIP3P water"
echo "  - Complete AMBER ff14SB topology with all parameters"
echo "  - H-bond cluster detection for analytic constraints"
echo ""

# Check for conda/mamba
if command -v mamba &> /dev/null; then
    PKG_MGR="mamba"
elif command -v conda &> /dev/null; then
    PKG_MGR="conda"
else
    echo "❌ Neither conda nor mamba found!"
    echo ""
    echo "   Install Miniforge (recommended):"
    echo "   https://github.com/conda-forge/miniforge"
    echo ""
    echo "   Or install Miniconda:"
    echo "   https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "Using package manager: $PKG_MGR"

# Check if OpenMM is installed
if python -c "import openmm" 2>/dev/null; then
    echo "✓ OpenMM already installed"
else
    echo ""
    echo "📦 Installing OpenMM and dependencies..."
    $PKG_MGR install -y -c conda-forge openmm pdbfixer
fi

# Create data directory
mkdir -p "$DATA_DIR"
echo ""
echo "Data directory: $DATA_DIR"

# Download test structures
echo ""
echo "📥 Downloading test structures..."
python "$SCRIPT_DIR/download_and_setup.py" --structures --data-dir "$DATA_DIR"

# Prepare smallest structure for testing
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "🔧 Preparing 1L2Y (Trp-cage, smallest test)..."
echo "═══════════════════════════════════════════════════════════════"
python "$SCRIPT_DIR/download_and_setup.py" --prepare 1l2y --data-dir "$DATA_DIR"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "🔧 Preparing 1UBQ (Ubiquitin, standard benchmark)..."
echo "═══════════════════════════════════════════════════════════════"
python "$SCRIPT_DIR/download_and_setup.py" --prepare 1ubq --data-dir "$DATA_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                       Setup Complete!                            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Prepared files (with ALL hydrogens and proper solvation):"
echo "  $DATA_DIR/prepared/1l2y_topology.json  (tiny: ~300 atoms)"
echo "  $DATA_DIR/prepared/1ubq_topology.json  (small: ~600 atoms)"
echo ""
echo "To prepare more structures:"
echo "  python $SCRIPT_DIR/download_and_setup.py --prepare 4ake"
echo ""
echo "To prepare without solvent (vacuum):"
echo "  python $SCRIPT_DIR/download_and_setup.py --prepare 1ubq --no-solvent"
echo ""
echo "To run MD test with PRISM:"
echo "  cargo run --release -p prism-validation --features cryptic-gpu \\"
echo "      --bin test_openmm_topology -- $DATA_DIR/prepared/1l2y_topology.json"
echo ""

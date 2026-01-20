#!/bin/bash
# ============================================================================
# PRISM-Cryptic Release Packaging Script
# ============================================================================
#
# This script builds and packages the official PRISM-Cryptic release.
#
# Usage:
#   ./scripts/package_cryptic_release.sh
#
# Output:
#   release_package/PRISM-Cryptic-v1.0.0-linux-x86_64.tar.gz
# ============================================================================

set -e

VERSION="1.0.0"
RELEASE_NAME="PRISM-Cryptic-v${VERSION}-linux-x86_64"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    PRISM-Cryptic Release Packaging                           ║"
echo "║                         Version ${VERSION}                                         ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

cd "$PROJECT_ROOT"

# ============================================================================
# Step 1: Build Release Binaries
# ============================================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 1: Building release binaries..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "  Building prism-cryptic..."
cargo build --release -p prism-validation --features cryptic-gpu --bin prism-cryptic

if [ ! -f "target/release/prism-cryptic" ]; then
    echo "ERROR: prism-cryptic binary not found!"
    exit 1
fi
echo "  ✓ prism-cryptic built successfully"

# ============================================================================
# Step 2: Create Release Directory Structure
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 2: Creating release directory structure..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

RELEASE_DIR="release_package/${RELEASE_NAME}"
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR/bin"
mkdir -p "$RELEASE_DIR/scripts"
mkdir -p "$RELEASE_DIR/docs"
mkdir -p "$RELEASE_DIR/examples"
mkdir -p "$RELEASE_DIR/ptx"

echo "  Created: $RELEASE_DIR"

# ============================================================================
# Step 3: Copy Binaries
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 3: Copying binaries..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cp target/release/prism-cryptic "$RELEASE_DIR/bin/"
echo "  ✓ prism-cryptic"

# Copy prism-prep and dependencies
cp scripts/prism-prep "$RELEASE_DIR/scripts/"
cp scripts/multichain_preprocessor.py "$RELEASE_DIR/scripts/"
cp scripts/stage1_sanitize.py "$RELEASE_DIR/scripts/"
cp scripts/stage1_sanitize_hybrid.py "$RELEASE_DIR/scripts/"
cp scripts/stage2_topology.py "$RELEASE_DIR/scripts/"
cp scripts/verify_topology.py "$RELEASE_DIR/scripts/"
cp scripts/glycan_preprocessor.py "$RELEASE_DIR/scripts/"
cp scripts/combine_chain_topologies.py "$RELEASE_DIR/scripts/"
echo "  ✓ prism-prep + dependencies"

# Copy PTX files
if [ -d "target/ptx" ]; then
    cp target/ptx/*.ptx "$RELEASE_DIR/ptx/" 2>/dev/null || true
    echo "  ✓ PTX kernels"
fi

# ============================================================================
# Step 4: Copy Documentation
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 4: Copying documentation..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cp README_CRYPTIC_RELEASE.md "$RELEASE_DIR/README.md"
cp docs/PRISM_CRYPTIC.md "$RELEASE_DIR/docs/"
cp docs/PRISM_PREP.md "$RELEASE_DIR/docs/"
cp LICENSE* "$RELEASE_DIR/" 2>/dev/null || echo "  (No LICENSE file found)"
echo "  ✓ Documentation copied"

# ============================================================================
# Step 5: Create Example Scripts
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 5: Creating example scripts..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Quick start script
cat > "$RELEASE_DIR/examples/quick_start.sh" << 'EOF'
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
EOF
chmod +x "$RELEASE_DIR/examples/quick_start.sh"
echo "  ✓ quick_start.sh"

# Batch processing example
cat > "$RELEASE_DIR/examples/batch_example.sh" << 'EOF'
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
EOF
chmod +x "$RELEASE_DIR/examples/batch_example.sh"
echo "  ✓ batch_example.sh"

# Accelerated mode example
cat > "$RELEASE_DIR/examples/accelerated_mode.sh" << 'EOF'
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
EOF
chmod +x "$RELEASE_DIR/examples/accelerated_mode.sh"
echo "  ✓ accelerated_mode.sh"

# ============================================================================
# Step 6: Create Install Script
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 6: Creating install script..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cat > "$RELEASE_DIR/install.sh" << 'EOF'
#!/bin/bash
# PRISM-Cryptic Installation Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "PRISM-Cryptic Installation"
echo "=========================="
echo ""

# Check for sudo
SUDO=""
if [ "$EUID" -ne 0 ]; then
    if command -v sudo &> /dev/null; then
        SUDO="sudo"
    else
        echo "Warning: Not running as root and sudo not available."
        echo "Will install to user directories instead."
    fi
fi

# Determine install location
if [ -n "$SUDO" ]; then
    INSTALL_BIN="/usr/local/bin"
else
    INSTALL_BIN="$HOME/.local/bin"
    mkdir -p "$INSTALL_BIN"
fi

# Install prism-cryptic
echo "Installing prism-cryptic to $INSTALL_BIN..."
$SUDO cp "$SCRIPT_DIR/bin/prism-cryptic" "$INSTALL_BIN/"
$SUDO chmod +x "$INSTALL_BIN/prism-cryptic"
echo "  ✓ prism-cryptic installed"

# Install prism-prep
echo "Installing prism-prep..."
$SUDO cp "$SCRIPT_DIR/scripts/prism-prep" "$INSTALL_BIN/"
$SUDO chmod +x "$INSTALL_BIN/prism-prep"
echo "  ✓ prism-prep installed"

# Create scripts directory for helper scripts
SCRIPTS_DIR="${INSTALL_BIN%/bin}/share/prism-cryptic/scripts"
$SUDO mkdir -p "$SCRIPTS_DIR"
$SUDO cp "$SCRIPT_DIR/scripts/"*.py "$SCRIPTS_DIR/"
echo "  ✓ Helper scripts installed to $SCRIPTS_DIR"

# Install PTX files
PTX_DIR="${INSTALL_BIN%/bin}/share/prism-cryptic/ptx"
if [ -d "$SCRIPT_DIR/ptx" ] && [ "$(ls -A "$SCRIPT_DIR/ptx" 2>/dev/null)" ]; then
    $SUDO mkdir -p "$PTX_DIR"
    $SUDO cp "$SCRIPT_DIR/ptx/"*.ptx "$PTX_DIR/"
    echo "  ✓ PTX kernels installed to $PTX_DIR"
fi

echo ""
echo "Installation complete!"
echo ""
echo "Verify installation:"
echo "  prism-cryptic --version"
echo "  prism-prep --check-deps"
echo ""

# Add to PATH if needed
if [ -z "$SUDO" ]; then
    if [[ ":$PATH:" != *":$INSTALL_BIN:"* ]]; then
        echo "Note: Add $INSTALL_BIN to your PATH:"
        echo "  echo 'export PATH=\"\$PATH:$INSTALL_BIN\"' >> ~/.bashrc"
        echo "  source ~/.bashrc"
    fi
fi
EOF
chmod +x "$RELEASE_DIR/install.sh"
echo "  ✓ install.sh"

# ============================================================================
# Step 7: Create Package
# ============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Step 7: Creating release package..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd release_package
tar -czf "${RELEASE_NAME}.tar.gz" "$RELEASE_NAME"
cd ..

echo "  ✓ Created: release_package/${RELEASE_NAME}.tar.gz"

# Calculate size and checksum
PACKAGE_SIZE=$(du -h "release_package/${RELEASE_NAME}.tar.gz" | cut -f1)
CHECKSUM=$(sha256sum "release_package/${RELEASE_NAME}.tar.gz" | cut -d' ' -f1)

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         Release Package Complete                             ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "  Package:  release_package/${RELEASE_NAME}.tar.gz"
echo "  Size:     $PACKAGE_SIZE"
echo "  SHA256:   $CHECKSUM"
echo ""
echo "  Contents:"
echo "    bin/prism-cryptic     - Main detection binary"
echo "    scripts/prism-prep    - PDB preprocessing tool"
echo "    scripts/*.py          - Helper scripts"
echo "    docs/                 - Documentation"
echo "    examples/             - Example scripts"
echo "    install.sh            - Installation script"
echo "    README.md             - Quick start guide"
echo ""
echo "  To install:"
echo "    tar -xzf ${RELEASE_NAME}.tar.gz"
echo "    cd ${RELEASE_NAME}"
echo "    ./install.sh"
echo ""

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

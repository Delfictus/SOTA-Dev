#!/bin/bash
# Install libtorch system-wide
# Run with: sudo ./install_libtorch_system.sh

set -e

LIBTORCH_SRC="/home/diddy/libtorch"
LIBTORCH_DST="/usr/local/lib/libtorch"
INCLUDE_DST="/usr/local/include/libtorch"

echo "=============================================="
echo "  Installing libtorch system-wide"
echo "=============================================="

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo $0"
    exit 1
fi

# Check source exists
if [ ! -d "$LIBTORCH_SRC/lib" ]; then
    echo "Error: $LIBTORCH_SRC/lib not found"
    exit 1
fi

echo "[1/5] Creating directories..."
mkdir -p "$LIBTORCH_DST"
mkdir -p "$INCLUDE_DST"

echo "[2/5] Copying libraries (~2.3GB)..."
cp -av "$LIBTORCH_SRC/lib/"* "$LIBTORCH_DST/"

echo "[3/5] Copying headers..."
cp -av "$LIBTORCH_SRC/include/"* "$INCLUDE_DST/"

echo "[4/5] Configuring ldconfig..."
echo "$LIBTORCH_DST" > /etc/ld.so.conf.d/libtorch.conf

echo "[5/5] Updating library cache..."
ldconfig

echo ""
echo "=============================================="
echo "  Installation complete!"
echo "=============================================="
echo ""
echo "Add these to your ~/.bashrc:"
echo ""
echo "  export LIBTORCH=/usr/local/lib/libtorch"
echo "  export LIBTORCH_INCLUDE=/usr/local/include/libtorch"
echo "  export LIBTORCH_LIB=/usr/local/lib/libtorch"
echo ""
echo "Then run: source ~/.bashrc"
echo ""
echo "Verify with: ldconfig -p | grep torch"

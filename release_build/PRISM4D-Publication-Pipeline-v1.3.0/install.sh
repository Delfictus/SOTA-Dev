#!/bin/bash
# PRISM4D Installation Helper

set -e

echo "============================================================"
echo "PRISM4D Publication Pipeline - Installation Check"
echo "============================================================"

# Check OS
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "WARNING: This package is designed for Linux systems"
    echo "Current OS: $OSTYPE"
fi

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "x86_64" ]; then
    echo "ERROR: Unsupported architecture: $ARCH"
    echo "This package requires x86_64"
    exit 1
fi

echo "✓ Architecture: $ARCH"

# Check CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found"
    echo "Please install NVIDIA drivers and CUDA toolkit"
    echo "https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo "✓ NVIDIA driver detected"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Check CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
    echo "✓ CUDA Toolkit: $CUDA_VERSION"
else
    echo "WARNING: nvcc not found - CUDA toolkit may not be installed"
    echo "Binaries include CUDA runtime, but toolkit recommended for debugging"
fi

# Check Python (optional)
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo "✓ Python: $PYTHON_VERSION"

    # Check matplotlib
    if python3 -c "import matplotlib" 2>/dev/null; then
        echo "✓ matplotlib installed"
    else
        echo "⚠ matplotlib not found (needed for visualization)"
        echo "  Install: pip3 install matplotlib numpy"
    fi
else
    echo "⚠ Python3 not found (needed for visualization)"
fi

# Check PyMOL (optional)
if command -v pymol &> /dev/null; then
    echo "✓ PyMOL installed (movies enabled)"
else
    echo "⚠ PyMOL not found (optional - needed for movie generation)"
    echo "  Install: sudo apt-get install pymol"
fi

# Add to PATH recommendation
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo ""
echo "============================================================"
echo "Installation Complete!"
echo "============================================================"
echo ""
echo "To use PRISM4D from anywhere, add to PATH:"
echo "  export PATH=\"$INSTALL_DIR/bin:\$PATH\""
echo ""
echo "Or add this line to ~/.bashrc:"
echo "  echo 'export PATH=\"$INSTALL_DIR/bin:\$PATH\"' >> ~/.bashrc"
echo ""
echo "Quick test:"
echo "  ./bin/nhs-cryo-probe --version"
echo "  ./bin/nhs-analyze-pro --version"
echo ""

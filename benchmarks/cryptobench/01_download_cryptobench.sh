#!/usr/bin/env bash
# =============================================================================
# CryptoBench Dataset Download
# Downloads the CryptoBench dataset (1,107 apo structures with cryptic pockets)
# Reference: Skrhak et al., Bioinformatics 2025, 41(1):btae745
# =============================================================================
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${BENCH_DIR}/data"

echo "============================================="
echo "  CryptoBench Dataset Download"
echo "============================================="
echo ""

mkdir -p "${DATA_DIR}"

# Download dataset.json (8.4 MB) — main ground truth file
# OSF verified URL: https://osf.io/download/ta2ju/
if [ ! -f "${DATA_DIR}/dataset.json" ]; then
    echo "Downloading dataset.json (8.4 MB)..."
    wget -q --show-progress -O "${DATA_DIR}/dataset.json" \
        "https://osf.io/download/ta2ju/"
    echo "  OK"
else
    echo "dataset.json already exists, skipping"
fi

# Download folds.json (18 KB) — cross-validation splits
# OSF verified URL: https://osf.io/download/5s93p/
if [ ! -f "${DATA_DIR}/folds.json" ]; then
    echo "Downloading folds.json..."
    wget -q --show-progress -O "${DATA_DIR}/folds.json" \
        "https://osf.io/download/5s93p/"
    echo "  OK"
else
    echo "folds.json already exists, skipping"
fi

# Download CIF structure files (1.15 GB)
# OSF verified URL: https://osf.io/download/c5vp8/
if [ ! -d "${DATA_DIR}/cif_files" ]; then
    echo "Downloading CIF structure files (1.15 GB)..."
    echo "  This may take several minutes..."
    wget -q --show-progress -O "${DATA_DIR}/cif-files.zip" \
        "https://osf.io/download/c5vp8/"
    echo "Extracting CIF files..."
    mkdir -p "${DATA_DIR}/cif_files"
    unzip -q "${DATA_DIR}/cif-files.zip" -d "${DATA_DIR}/cif_files/"
    rm "${DATA_DIR}/cif-files.zip"
    echo "  OK"
else
    echo "CIF files already exist, skipping"
fi

# Verify
N_CIF=$(find "${DATA_DIR}/cif_files" -name "*.cif" | wc -l)
echo ""
echo "============================================="
echo "  Download complete"
echo "  CIF files: ${N_CIF}"
echo "  Dataset: ${DATA_DIR}/dataset.json"
echo "  Folds: ${DATA_DIR}/folds.json"
echo "============================================="
echo ""
echo "Next step: run 02_prepare_topologies.py"

#!/usr/bin/env bash
set -euo pipefail
#
# PRISM4D — Full Developer Release Builder (YOUR EYES ONLY)
# Complete source code archive with all CUDA kernels, PTX, Rust source,
# Python pipeline, benchmarks, and build infrastructure.
#
# Usage:  bash release/build_developer_release.sh [output_dir]
# Output: prism4d-dev-<date>.tar.gz.enc (AES-256 encrypted, MANDATORY)
#

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATE_TAG=$(date +%Y%m%d)
OUT_DIR="${1:-${PROJ_ROOT}/release/dist}"
STAGE="${OUT_DIR}/prism4d-dev-${DATE_TAG}"

echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  PRISM4D DEVELOPER Release Builder (CONFIDENTIAL)               ║"
echo "║  Date: ${DATE_TAG}                                              ║"
echo "║  THIS ARCHIVE CONTAINS ALL SOURCE CODE AND IP                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

# ── 0. Pre-flight ─────────────────────────────────────────────────────
echo "[0/10] Pre-flight checks..."
command -v openssl >/dev/null || { echo "ERROR: openssl required for mandatory encryption"; exit 1; }
[ -f "${PROJ_ROOT}/Cargo.toml" ] || { echo "ERROR: Not in project root"; exit 1; }

# ── 1. Create directory structure ─────────────────────────────────────
echo "[1/10] Creating release tree..."
rm -rf "${STAGE}"
mkdir -p "${STAGE}"

# ── 2. Copy entire Rust workspace source ──────────────────────────────
echo "[2/10] Packaging Rust workspace (all 26 crates)..."

# Workspace root files
cp "${PROJ_ROOT}/Cargo.toml" "${STAGE}/"
cp "${PROJ_ROOT}/Cargo.lock" "${STAGE}/"
[ -d "${PROJ_ROOT}/.cargo" ] && cp -r "${PROJ_ROOT}/.cargo" "${STAGE}/" 2>/dev/null || true

# All crates with full source
mkdir -p "${STAGE}/crates"
for crate_dir in "${PROJ_ROOT}/crates"/*/; do
    CRATE_NAME=$(basename "$crate_dir")
    echo "  + crate: ${CRATE_NAME}"

    # Copy source, Cargo.toml, build.rs — but NOT target/ (build artifacts)
    mkdir -p "${STAGE}/crates/${CRATE_NAME}"

    # Cargo.toml & build.rs
    [ -f "${crate_dir}/Cargo.toml" ] && cp "${crate_dir}/Cargo.toml" "${STAGE}/crates/${CRATE_NAME}/"
    [ -f "${crate_dir}/build.rs" ] && cp "${crate_dir}/build.rs" "${STAGE}/crates/${CRATE_NAME}/"

    # Source directory
    if [ -d "${crate_dir}/src" ]; then
        cp -r "${crate_dir}/src" "${STAGE}/crates/${CRATE_NAME}/src"
    fi

    # Kernel directories (for crates that have them)
    if [ -d "${crate_dir}/kernels" ]; then
        cp -r "${crate_dir}/kernels" "${STAGE}/crates/${CRATE_NAME}/kernels"
    fi

    # Tests
    if [ -d "${crate_dir}/tests" ]; then
        cp -r "${crate_dir}/tests" "${STAGE}/crates/${CRATE_NAME}/tests"
    fi
done

# ── 3. Copy ALL CUDA kernel sources (.cu) ─────────────────────────────
echo "[3/10] Packaging CUDA kernel sources..."

CU_COUNT=$(find "${PROJ_ROOT}/crates" -name "*.cu" -not -path "*/target/*" | wc -l)
echo "  ${CU_COUNT} CUDA kernel source files (.cu)"

# Already copied via crate src/ directories above
# Verify they're present
STAGED_CU=$(find "${STAGE}/crates" -name "*.cu" | wc -l)
echo "  Verified: ${STAGED_CU} .cu files in staging"

# ── 4. Copy precompiled PTX and OptiX IR ──────────────────────────────
echo "[4/10] Packaging compiled PTX and OptiX IR..."

# PTX from build output
mkdir -p "${STAGE}/crates/prism-gpu/target/ptx"
for ptx in "${PROJ_ROOT}/crates/prism-gpu/target/ptx"/*.ptx; do
    [ -f "$ptx" ] && cp "$ptx" "${STAGE}/crates/prism-gpu/target/ptx/"
done
for sig in "${PROJ_ROOT}/crates/prism-gpu/target/ptx"/*.sha256; do
    [ -f "$sig" ] && cp "$sig" "${STAGE}/crates/prism-gpu/target/ptx/"
done
echo "  $(find "${STAGE}/crates/prism-gpu/target/ptx" -name "*.ptx" | wc -l) PTX files"

# NHS PTX
if [ -d "${PROJ_ROOT}/crates/prism-nhs/target/ptx" ]; then
    mkdir -p "${STAGE}/crates/prism-nhs/target/ptx"
    cp "${PROJ_ROOT}/crates/prism-nhs/target/ptx"/*.ptx "${STAGE}/crates/prism-nhs/target/ptx/" 2>/dev/null || true
fi

# OptiX IR (already in src/kernels, but also check for loose copies)
echo "  $(find "${STAGE}" -name "*.optixir" | wc -l) OptiX IR files"

# ── 5. Copy precompiled binaries ──────────────────────────────────────
echo "[5/10] Packaging precompiled binaries (NOT stripped — debug-capable)..."

mkdir -p "${STAGE}/target/release"
for bin in "${PROJ_ROOT}/target/release"/*; do
    [ -f "$bin" ] && [ -x "$bin" ] && [ ! -d "$bin" ] || continue
    BASENAME=$(basename "$bin")
    # Skip .d files and internal artifacts
    [[ "$BASENAME" == *.d ]] && continue
    [[ "$BASENAME" == *.rlib ]] && continue
    cp "$bin" "${STAGE}/target/release/${BASENAME}"
    echo "  + ${BASENAME} ($(du -h "${STAGE}/target/release/${BASENAME}" | cut -f1))"
done

# ── 6. Copy Python pipeline (FULL source) ─────────────────────────────
echo "[6/10] Packaging Python pipeline (full source)..."

mkdir -p "${STAGE}/scripts"
# Copy all scripts
for f in "${PROJ_ROOT}/scripts"/*; do
    BASENAME=$(basename "$f")
    [ -d "$f" ] && continue
    [[ "$BASENAME" == __pycache__ ]] && continue
    cp "$f" "${STAGE}/scripts/${BASENAME}"
done

# Copy scripts subdirectories
for subdir in pipeline explicit_solvent preprocessing interfaces; do
    if [ -d "${PROJ_ROOT}/scripts/${subdir}" ]; then
        cp -r "${PROJ_ROOT}/scripts/${subdir}" "${STAGE}/scripts/${subdir}"
    fi
done

echo "  $(find "${STAGE}/scripts" -type f | wc -l) script files"

# ── 7. Copy environments ──────────────────────────────────────────────
echo "[7/10] Packaging conda environments..."

mkdir -p "${STAGE}/envs"
cp "${PROJ_ROOT}/envs"/*.yml "${STAGE}/envs/" 2>/dev/null || true
echo "  $(ls "${STAGE}/envs"/*.yml 2>/dev/null | wc -l) environment definitions"

# ── 8. Copy benchmarks ────────────────────────────────────────────────
echo "[8/10] Packaging benchmark infrastructure..."

mkdir -p "${STAGE}/benchmarks/cryptobench"
# Copy benchmark scripts and ground truth
for item in \
    01_download_cryptobench.sh \
    02_prepare_topologies.py \
    03_generate_run_commands.py \
    04_evaluate_results.py \
    05_comprehensive_analysis.py \
    run_all_batches.sh \
    README.md \
    BATCH_COMMANDS.md \
    ; do
    SRC="${PROJ_ROOT}/benchmarks/cryptobench/${item}"
    [ -f "$SRC" ] && cp "$SRC" "${STAGE}/benchmarks/cryptobench/"
done

# Batch scripts
if [ -d "${PROJ_ROOT}/benchmarks/cryptobench/batches" ]; then
    cp -r "${PROJ_ROOT}/benchmarks/cryptobench/batches" "${STAGE}/benchmarks/cryptobench/batches"
fi

# Ground truth
if [ -d "${PROJ_ROOT}/benchmarks/cryptobench/ground_truth" ]; then
    cp -r "${PROJ_ROOT}/benchmarks/cryptobench/ground_truth" "${STAGE}/benchmarks/cryptobench/ground_truth"
fi

# ── 9. Copy documentation ─────────────────────────────────────────────
echo "[9/10] Packaging documentation..."

mkdir -p "${STAGE}/docs"
[ -d "${PROJ_ROOT}/docs" ] && cp -r "${PROJ_ROOT}/docs"/* "${STAGE}/docs/" 2>/dev/null || true
[ -f "${PROJ_ROOT}/README.md" ] && cp "${PROJ_ROOT}/README.md" "${STAGE}/"
[ -f "${PROJ_ROOT}/CLAUDE.md" ] && cp "${PROJ_ROOT}/CLAUDE.md" "${STAGE}/"

# ── 10. Create manifest and encrypted archive ─────────────────────────
echo "[10/10] Creating manifest and encrypted archive..."

# Compute file tree statistics
TOTAL_RS=$(find "${STAGE}/crates" -name "*.rs" | wc -l)
TOTAL_CU=$(find "${STAGE}" -name "*.cu" | wc -l)
TOTAL_PTX=$(find "${STAGE}" -name "*.ptx" | wc -l)
TOTAL_OPTIXIR=$(find "${STAGE}" -name "*.optixir" | wc -l)
TOTAL_PY=$(find "${STAGE}" -name "*.py" | wc -l)
TOTAL_BINS=$(find "${STAGE}/target/release" -maxdepth 1 -type f -executable 2>/dev/null | wc -l)

cat > "${STAGE}/MANIFEST.json" << MANIFEST
{
    "product": "PRISM4D",
    "version": "1.0.0-dev",
    "build_date": "${DATE_TAG}",
    "type": "FULL_DEVELOPER_RELEASE",
    "classification": "CONFIDENTIAL — YOUR EYES ONLY",
    "gpu_target": "sm_120 (Blackwell)",
    "cuda_toolkit": "$(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',')",
    "platform": "linux-x86_64",
    "contents": {
        "rust_source_files": ${TOTAL_RS},
        "cuda_kernel_sources": ${TOTAL_CU},
        "ptx_compiled_kernels": ${TOTAL_PTX},
        "optixir_files": ${TOTAL_OPTIXIR},
        "python_scripts": ${TOTAL_PY},
        "precompiled_binaries": ${TOTAL_BINS},
        "rust_crates": $(ls -d "${STAGE}/crates"/*/ 2>/dev/null | wc -l)
    },
    "build_instructions": "See BUILD.md",
    "license": "PROPRIETARY — All rights reserved"
}
MANIFEST

# Build instructions
cat > "${STAGE}/BUILD.md" << 'BUILD'
# PRISM4D Build Instructions

## Prerequisites
- Linux x86_64
- Rust 1.75+ (2021 edition)
- CUDA Toolkit 12.6+ (nvcc in PATH)
- OptiX SDK 9.1.0 (set OPTIX_ROOT)
- NVIDIA Driver 590+
- RTX GPU (Turing+ minimum, Blackwell optimized)

## Environment Setup
```bash
export CUDA_HOME=/usr/local/cuda-12.6
export OPTIX_ROOT=$HOME/.local/opt/optix-9.1.0
```

## Build from Source
```bash
cargo build --release
```

## Python Environment
```bash
conda env create -f envs/preprocessing.yml
conda activate prism4d-preprocessing
```

## End-to-End Pipeline
```bash
# 1. Prepare structure
python scripts/prism-prep input.pdb output.topology.json

# 2. Detect cryptic pockets
./target/release/nhs_rt_full -t output.topology.json -o results/ \
    --fast --hysteresis --multi-stream 8 --spike-percentile 95 --rt-clustering -v

# 3. Re-rank sites
python scripts/rerank_sites.py results/ --in-place

# 4. Evaluate (if ground truth available)
python benchmarks/cryptobench/04_evaluate_results.py
```
BUILD

# Create tarball
TARBALL="${OUT_DIR}/prism4d-dev-${DATE_TAG}.tar.gz"
cd "${OUT_DIR}"
tar czf "$(basename "$TARBALL")" "$(basename "$STAGE")"

echo ""
echo "Unencrypted archive: ${TARBALL} ($(du -h "$TARBALL" | cut -f1))"
echo ""

# MANDATORY encryption for developer release
echo "This archive contains ALL source code and trade secrets."
echo "Encryption is MANDATORY for the developer release."
echo ""
echo "To encrypt:"
echo "  openssl enc -aes-256-cbc -salt -pbkdf2 -iter 100000 \\"
echo "    -in ${TARBALL} -out ${TARBALL}.enc"
echo ""
echo "To decrypt:"
echo "  openssl enc -d -aes-256-cbc -pbkdf2 -iter 100000 \\"
echo "    -in ${TARBALL}.enc -out ${TARBALL}"
echo ""
echo "After encryption, DELETE the unencrypted tarball:"
echo "  shred -u ${TARBALL}"
echo ""

# Summary
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  DEVELOPER RELEASE COMPLETE                                     ║"
echo "╠═══════════════════════════════════════════════════════════════════╣"
printf "║  Rust source:     %-5d .rs files                               ║\n" "${TOTAL_RS}"
printf "║  CUDA kernels:    %-5d .cu files (FULL SOURCE)                 ║\n" "${TOTAL_CU}"
printf "║  Compiled PTX:    %-5d .ptx files                              ║\n" "${TOTAL_PTX}"
printf "║  OptiX IR:        %-5d .optixir files                          ║\n" "${TOTAL_OPTIXIR}"
printf "║  Python scripts:  %-5d .py files                               ║\n" "${TOTAL_PY}"
printf "║  Binaries:        %-5d (NOT stripped — debuggable)             ║\n" "${TOTAL_BINS}"
echo "║                                                                 ║"
echo "║  ⚠  ENCRYPT BEFORE STORING OR TRANSFERRING                     ║"
echo "║  ⚠  DELETE UNENCRYPTED COPY AFTER ENCRYPTION                   ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"

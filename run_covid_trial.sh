#!/bin/bash
set -e

echo "=================================================="
echo "🦠 PRISM-ZERO: COVID-19 VRAM TURBO PROTOCOL"
echo "=================================================="

# 1. COMPILE KERNEL
echo "⚙️  [1/6] Compiling Holographic Kernel..."
nvcc -ptx crates/prism-gpu/kernels/holographic_langevin.cu -o crates/prism-gpu/kernels/holographic_langevin.ptx

# 2. TARGETING SWAP
echo "🎯 [2/6] Retargeting Engine to SARS-CoV-2..."
sed -i 's|data/processed/2VWD.ptb|data/processed/6VXX.ptb|g' crates/prism-physics/src/bin/prism-niv-bench.rs

# 3. THE PURGE
echo "🧹 [3/6] Cleaning workspace..."
rm -f data/processed/6VXX.ptb data/processed/nipah_relaxed.ptb data/processed/covid_relaxed.pdb

# 4. THE FUEL
echo "🧬 [4/6] Ingesting SARS-CoV-2 Spike (6VXX)..."
cargo run --quiet --release -p prism-io --bin prism-ingest -- data/raw/6VXX.pdb data/processed/6VXX.ptb

# 5. THE ENGINE
echo "🚀 [5/6] Igniting VRAM Turbo Engine (1M Steps)..."
RUST_LOG=info cargo run --release --bin prism-niv-bench --features "cuda telemetry" -p prism-physics || true

# 6. THE ARTIFACT
echo "💾 [6/6] Exporting Visual Proof..."
if [ -f "data/processed/nipah_relaxed.ptb" ]; then
    cargo run --quiet --release -p prism-io --bin prism-export -- \
      data/processed/nipah_relaxed.ptb \
      data/processed/covid_relaxed.pdb \
      --template data/raw/6VXX.pdb
    
    echo "✅ SUCCESS: Artifact Generated."
    ls -lh data/processed/covid_relaxed.pdb
else
    echo "❌ FAILURE: Simulation did not produce an output file."
fi

# 7. CLEANUP
sed -i 's|data/processed/6VXX.ptb|data/processed/2VWD.ptb|g' crates/prism-physics/src/bin/prism-niv-bench.rs
echo "🔄 System Reset to Default."

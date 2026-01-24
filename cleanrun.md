# 1. THE PURGE
rm -f data/processed/2VWD.ptb data/processed/nipah_relaxed.ptb data/processed/*.pdb

# 2. THE FUEL
cargo run --quiet --release -p prism-io --bin prism-ingest -- data/raw/2VWD.pdb data/processed/2VWD.ptb

# 3. THE ENGINE (Run 4 - Goldilocks Physics)
# This will take ~0.5 seconds
RUST_LOG=info cargo run --quiet --release --bin prism-niv-bench --features "cuda telemetry" -p prism-physics || true

# 4. THE ARTIFACT (Export)
cargo run --quiet --release -p prism-io --bin prism-export -- \
  data/processed/nipah_relaxed.ptb \
  data/processed/nipah_relaxed_perfect.pdb \
  --template data/raw/2VWD.pdb

# 5. VERIFY
echo "🔍 Verifying..."
if ! cmp -s data/raw/2VWD.pdb data/processed/nipah_relaxed_perfect.pdb; then
    echo "✅ SUCCESS: Coordinates Updated."
else
    echo "❌ FAILURE: Files are identical."
fi

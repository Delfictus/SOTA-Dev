#!/bin/bash
# Run this LOCALLY to package your BENCH60 data for upload to RunPod.
# Creates bench60_data.tar.gz that you scp to the RunPod instance.

set -e

BENCH_DIR="benchmarks/prism4d_bench30"
OUT_DIR="runpod_training/bench60_data"

echo "Packaging BENCH60 data for RunPod..."

mkdir -p "$OUT_DIR/apo" "$OUT_DIR/results"

# Copy manifest and ground truth
cp "$BENCH_DIR/benchmark_manifest.json" "$OUT_DIR/"
cp "$BENCH_DIR/ground_truth/ligand_centroids.json" "$OUT_DIR/"

# Copy apo PDBs
cp "$BENCH_DIR/apo/"*.pdb "$OUT_DIR/apo/"

# Copy detection results (binding_sites.json only, skip large PDB/trajectory files)
for d in "$BENCH_DIR/results"/*/; do
    tid=$(basename "$d")
    mkdir -p "$OUT_DIR/results/$tid"
    cp "$d"/*.binding_sites.json "$OUT_DIR/results/$tid/" 2>/dev/null || true
done

# Create tarball
cd runpod_training
tar czf bench60_data.tar.gz bench60_data/
echo "Created: runpod_training/bench60_data.tar.gz"
echo "Upload to RunPod: scp runpod_training/bench60_data.tar.gz root@<runpod-ip>:~/runpod_training/"
echo "Then on RunPod: cd ~/runpod_training && tar xzf bench60_data.tar.gz"

#!/bin/bash
# Run 3 concurrent replicas for each target using AmberSimdBatch
# This gives BETTER conformational sampling than single trajectory!

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  CONCURRENT REPLICA VALIDATION - AmberSimdBatch              ║"
echo "║  3 replicas per target (better conformational sampling!)    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

TARGETS=(
  "07_FructoseAldolase_apo"
  "11_HCV_NS5B_palm_holo"
  "16_GBA_apo"
)

for target in "${TARGETS[@]}"; do
  echo "═══════════════════════════════════════════════════════════════"
  echo "🔥 Processing $target with 3 CONCURRENT REPLICAS..."
  echo "═══════════════════════════════════════════════════════════════"

  # Run prism4d with 3 replicates (should use AmberSimdBatch internally!)
  target/release/prism4d run \
    --topology "production_test/targets/${target}.topology.json" \
    --pdb "production_test/targets/${target}.pdb" \
    --out "production_test/replicas_${target}" \
    --replicates 3 \
    --steps 500000 \
    --skip-ablation \
    --cold-hold-steps 100000 \
    --ramp-steps 125000 \
    --warm-hold-steps 125000

  echo "✅ $target replicas complete!"
  echo ""
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              REPLICA VALIDATION COMPLETE!                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Results:"
for target in "${TARGETS[@]}"; do
  echo "  production_test/replicas_${target}/summary.json"
done

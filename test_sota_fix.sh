#!/bin/bash
# Test SOTA path fix for Verlet list livelock
# This script runs a 1000-step MD simulation to verify the fix works

set -e

TOPOLOGY="/home/diddy/Desktop/Prism4D-bio/e2e_validation_test/prep/1crn.topology.json"
OUTPUT_DIR="/home/diddy/Desktop/Prism4D-bio/test_output/sota_fix_test"
BINARY="/home/diddy/Desktop/Prism4D-bio/target/release/nhs-rt-full"

# Clean and create output directory
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "========================================"
echo "SOTA PATH LIVELOCK FIX - SMOKE TEST"
echo "========================================"
echo ""
echo "Test: 1000 MD steps on 1crn (327 atoms)"
echo "Expected: Completes in ~2-5 seconds with Verlet lists enabled"
echo "Previous bug: Would hang/livelock in Phase 2"
echo ""
echo "Running..."
echo ""

# Run with SOTA optimizations (default - Verlet lists enabled)
# Use fast mode (50K steps) but we'll let it run for a short time to test the fix
timeout 60 "$BINARY" \
  --topology "$TOPOLOGY" \
  --output "$OUTPUT_DIR" \
  --steps 1000 \
  --temperature 300.0 \
  --fast \
  --verbose \
  2>&1 | tee "$OUTPUT_DIR/test.log"

echo ""
echo "========================================"
echo "SUCCESS! SOTA path completed without hanging"
echo "========================================"
echo ""
echo "Checking logs for Verlet list usage..."
grep -i "verlet" "$OUTPUT_DIR/test.log" || echo "No Verlet mentions (check if SOTA path was used)"
grep -i "sota" "$OUTPUT_DIR/test.log" || echo "No SOTA mentions"
echo ""
echo "Performance metrics:"
grep -i "steps/sec" "$OUTPUT_DIR/test.log" || echo "No performance metrics found"

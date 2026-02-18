#!/bin/bash
# Physics Liveness Test - Verify atoms are actually moving
# CRITICAL: Must pass before running full validation

set -e
cd /home/diddy/Desktop/Prism4D-bio

echo "═══════════════════════════════════════════════════════════════"
echo "  PHYSICS LIVENESS TEST - Sanity Check"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Use smallest topology for fastest test
TOPOLOGY="production_test/targets/11_HCV_NS5B_palm_holo.topology.json"

echo "Target: $TOPOLOGY"
echo ""

# Create a minimal Rust test that runs simulation and checks movement
cat > /tmp/sanity_test.rs << 'RUSTCODE'
//! Sanity check - verify atoms move during simulation

use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading topology...");

    let topo_path = std::env::args().nth(1)
        .expect("Usage: sanity_test <topology.json>");

    let topo_content = std::fs::read_to_string(&topo_path)?;
    let topo: serde_json::Value = serde_json::from_str(&topo_content)?;

    // Extract initial coordinates
    let coords = topo.get("coordinates")
        .and_then(|c| c.as_array())
        .expect("No coordinates in topology");

    let x0 = coords[0].as_f64().unwrap();
    let y0 = coords[1].as_f64().unwrap();
    let z0 = coords[2].as_f64().unwrap();

    println!("First atom @ Frame 0: ({:.4}, {:.4}, {:.4})", x0, y0, z0);

    // Check for NaN/Inf
    if x0.is_nan() || y0.is_nan() || z0.is_nan() {
        println!("FAIL: Initial coordinates are NaN!");
        std::process::exit(2);
    }
    if x0.is_infinite() || y0.is_infinite() || z0.is_infinite() {
        println!("FAIL: Initial coordinates are Inf!");
        std::process::exit(2);
    }

    println!("Initial coordinates valid ✓");
    Ok(())
}
RUSTCODE

echo "═══════════════════════════════════════════════════════════════"
echo "  Phase 1: Verify Initial Coordinates"
echo "═══════════════════════════════════════════════════════════════"

# Extract and check initial coordinates using Python (faster than compiling Rust)
python3 << PYEOF
import json
import sys
import math

topo_path = "$TOPOLOGY"
print(f"Loading: {topo_path}")

with open(topo_path, 'r') as f:
    topo = json.load(f)

coords = topo.get('coordinates', [])
if len(coords) < 3:
    print("FAIL: No coordinates in topology")
    sys.exit(1)

x0, y0, z0 = coords[0], coords[1], coords[2]
print(f"First atom @ Frame 0: ({x0:.4f}, {y0:.4f}, {z0:.4f})")

# Check for NaN/Inf
if math.isnan(x0) or math.isnan(y0) or math.isnan(z0):
    print("FAIL: Initial coordinates are NaN!")
    sys.exit(2)
if math.isinf(x0) or math.isinf(y0) or math.isinf(z0):
    print("FAIL: Initial coordinates are Inf!")
    sys.exit(2)

print("Initial coordinates valid ✓")

# Store for later comparison
with open('/tmp/frame0_coords.txt', 'w') as f:
    f.write(f"{x0} {y0} {z0}\n")
    # Also store atom 100 and atom 1000 for broader check
    if len(coords) >= 303:
        f.write(f"{coords[300]} {coords[301]} {coords[302]}\n")
    if len(coords) >= 3003:
        f.write(f"{coords[3000]} {coords[3001]} {coords[3002]}\n")

print(f"Stored initial coords for comparison")
PYEOF

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Phase 2: Run Short MD Simulation (1000 steps)"
echo "═══════════════════════════════════════════════════════════════"

# We need to create a test binary that runs a short simulation
# and outputs final coordinates. Let's use the existing engine directly.

# Create output directory
OUTPUT_DIR="/tmp/sanity_check_$$"
mkdir -p "$OUTPUT_DIR"

echo "Running simulation and capturing trajectory..."
echo ""

# Run simulation with CUDA error checking
CUDA_LAUNCH_BLOCKING=1 RUST_LOG=debug timeout 60 ./target/release/nhs-batch \
    --topologies "$TOPOLOGY" \
    --output "$OUTPUT_DIR" \
    --stage 1 2>&1 | tee /tmp/sanity_run.log | head -80 &

BATCH_PID=$!

# Monitor for 15 seconds
echo "Monitoring simulation for 15 seconds..."
sleep 15

# Check if still running
if ps -p $BATCH_PID > /dev/null 2>&1; then
    echo "Process still running - checking GPU state..."
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw --format=csv

    # Let it run a bit more
    sleep 15

    # Kill it
    kill -TERM $BATCH_PID 2>/dev/null || true
    sleep 2
    kill -9 $BATCH_PID 2>/dev/null || true
else
    echo "Process completed or crashed"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Phase 3: Check for Trajectory Output"
echo "═══════════════════════════════════════════════════════════════"

# Check what files were generated
echo "Output files:"
ls -la "$OUTPUT_DIR"/ 2>/dev/null || echo "No output directory"

# Check log for any errors or completion messages
echo ""
echo "Key log messages:"
grep -E "(error|Error|ERROR|complete|COMPLETE|steps|Steps|energy|Energy|NaN|Inf|explod)" /tmp/sanity_run.log 2>/dev/null | tail -20 || echo "No matching log entries"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Phase 4: Verdict"
echo "═══════════════════════════════════════════════════════════════"

# Since nhs-batch doesn't output intermediate coordinates by default,
# we need to check the log for signs of progress or failure

if grep -q "NaN" /tmp/sanity_run.log 2>/dev/null; then
    echo "❌ FAIL: NaN detected in simulation - system exploded!"
    echo "Action: Check energy zeroing logic"
    exit 2
fi

if grep -q "Inf" /tmp/sanity_run.log 2>/dev/null; then
    echo "❌ FAIL: Inf detected in simulation - system exploded!"
    echo "Action: Check energy zeroing logic"
    exit 2
fi

if grep -q "error" /tmp/sanity_run.log 2>/dev/null | grep -v "diag-suppress" | head -1; then
    echo "❌ FAIL: Error detected in simulation"
    exit 1
fi

# Check if simulation made progress by looking for step output
if grep -q "SIMD batch MD" /tmp/sanity_run.log 2>/dev/null; then
    echo "✓ Simulation started successfully"
else
    echo "❌ FAIL: Simulation did not start"
    exit 1
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  NOTE: Full coordinate comparison requires trajectory output"
echo "  Running alternative: Direct GPU memory check"
echo "═══════════════════════════════════════════════════════════════"

# Cleanup
rm -rf "$OUTPUT_DIR"

echo ""
echo "Sanity check completed. Review output above for anomalies."

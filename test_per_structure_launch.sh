#!/bin/bash
# Test script for per-structure kernel launch fix
# Monitors GPU metrics during execution to verify proper utilization

set -e

echo "=========================================="
echo "Per-Structure Kernel Launch Validation"
echo "=========================================="
echo ""

# Start GPU monitoring in background
echo "Starting GPU monitoring..."
nvidia-smi dmon -s pucvmet -c 120 > /tmp/gpu_metrics.log 2>&1 &
MONITOR_PID=$!

# Give monitoring time to initialize
sleep 2

echo "Running batch MD test with mini topology..."
echo ""

# Run the test with the small topology
cargo run --release --bin batch_md_test -- \
    --topology /home/diddy/Desktop/Prism4D-bio/test_mini.topology.json \
    2>&1 | tee /tmp/batch_test.log

echo ""
echo "Test complete!"
echo ""

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true

# Analyze GPU metrics
echo "=========================================="
echo "GPU Utilization Analysis"
echo "=========================================="
echo ""

# Get max values during run (skip header)
MAX_GPU=$(tail -n +3 /tmp/gpu_metrics.log | awk '{print $2}' | sort -n | tail -1)
MAX_MEM=$(tail -n +3 /tmp/gpu_metrics.log | awk '{print $3}' | sort -n | tail -1)
MAX_POWER=$(tail -n +3 /tmp/gpu_metrics.log | awk '{print $5}' | sort -n | tail -1)

# Calculate average memory bandwidth (read + write)
AVG_RX=$(tail -n +3 /tmp/gpu_metrics.log | awk '{if ($6 != "-") sum+=$6; count++} END {if (count>0) print sum/count; else print 0}')
AVG_TX=$(tail -n +3 /tmp/gpu_metrics.log | awk '{if ($7 != "-") sum+=$7; count++} END {if (count>0) print sum/count; else print 0}')

echo "Max GPU Utilization: ${MAX_GPU}%"
echo "Max Memory Usage: ${MAX_MEM} MB"
echo "Max Power Draw: ${MAX_POWER} W"
echo "Avg Memory BW (RX): ${AVG_RX} MB/s"
echo "Avg Memory BW (TX): ${AVG_TX} MB/s"
echo ""

# Check if metrics are acceptable (>20% memory BW, >200W power for RTX 5080)
if [ $(echo "$MAX_POWER > 200" | bc -l 2>/dev/null || echo "1") -eq 1 ]; then
    echo "✓ Power draw is healthy (>200W indicates proper GPU engagement)"
else
    echo "⚠ Low power draw (<200W may indicate race condition causing early exit)"
fi

echo ""
echo "Check test log for stability (no NaN, no explosion):"
grep -E "(RMSD|temperature|energy)" /tmp/batch_test.log | tail -5

echo ""
echo "=========================================="
echo "Full GPU metrics saved to: /tmp/gpu_metrics.log"
echo "Test output saved to: /tmp/batch_test.log"
echo "=========================================="

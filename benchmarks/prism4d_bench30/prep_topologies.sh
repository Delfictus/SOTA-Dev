#!/bin/bash
# Batch topology preparation for BENCH30 new targets
# Usage: bash prep_topologies.sh

SCRIPTS=/home/diddy/Desktop/Prism4D-bio/scripts
BENCH=/home/diddy/Desktop/Prism4D-bio/benchmarks/prism4d_bench30
APO=$BENCH/apo
SANITIZED=$BENCH/sanitized
TOPO=$BENCH/topologies

mkdir -p "$SANITIZED"

# New targets (20) - format: pdb_id chain
TARGETS=(
    "1jwp A"
    "2npq A"
    "1kv1 A"
    "1my0 A"
    "2w9t A"
    "4ey4 A"
    "2hnp A"
    "1m47 A"
    "2cpl A"
    "1ake A"
    "1hcl A"
    "1yes A"
    "2oss A"
    "1fkg A"
    "1hpv A"
    "1stp A"
    "3ert A"
    "1ohr A"
    "1r1w A"
    "2rh1 A"
)

SUCCESS=0
FAIL=0
FAILED_LIST=""

for entry in "${TARGETS[@]}"; do
    pdb=$(echo $entry | cut -d' ' -f1)
    chain=$(echo $entry | cut -d' ' -f2)

    # Skip if topology already exists
    if [ -f "$TOPO/${pdb}.topology.json" ]; then
        echo "SKIP $pdb (topology exists)"
        SUCCESS=$((SUCCESS + 1))
        continue
    fi

    echo "=== Processing $pdb chain $chain ==="

    # Stage 1: Sanitize
    python3 "$SCRIPTS/stage1_sanitize.py" \
        "$APO/${pdb}.pdb" \
        "$SANITIZED/${pdb}_clean.pdb" \
        --chain "$chain" -q 2>&1

    if [ $? -ne 0 ]; then
        echo "FAIL $pdb at stage1"
        FAIL=$((FAIL + 1))
        FAILED_LIST="$FAILED_LIST $pdb(s1)"
        continue
    fi

    # Stage 2: Topology
    python3 "$SCRIPTS/stage2_topology.py" \
        "$SANITIZED/${pdb}_clean.pdb" \
        "$TOPO/${pdb}.topology.json" \
        --hmr -q 2>&1

    if [ $? -ne 0 ]; then
        echo "FAIL $pdb at stage2"
        FAIL=$((FAIL + 1))
        FAILED_LIST="$FAILED_LIST $pdb(s2)"
        continue
    fi

    echo "OK $pdb"
    SUCCESS=$((SUCCESS + 1))
done

echo ""
echo "=== SUMMARY ==="
echo "Success: $SUCCESS / $((SUCCESS + FAIL))"
echo "Failed: $FAIL"
if [ -n "$FAILED_LIST" ]; then
    echo "Failed targets:$FAILED_LIST"
fi

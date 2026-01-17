#!/bin/bash
# Generate AMBER topologies using tleap
# Uses ff14SB force field for proteins

set -e

# Use miniconda tleap
TLEAP="/home/diddy/miniconda3/bin/tleap"

SANITIZED_DIR="sanitized"
TOPO_DIR="topologies"

mkdir -p "$TOPO_DIR"

# List of structures
STRUCTURES="6M0J 7WHH 7K45 6WPS 6W41 8SGU 7JJI"

echo "=============================================="
echo "Generating AMBER topologies with tleap"
echo "=============================================="

for PDB in $STRUCTURES; do
    INPUT_PDB="${SANITIZED_DIR}/${PDB}_amber.pdb"
    OUTPUT_PRMTOP="${TOPO_DIR}/${PDB}.prmtop"
    OUTPUT_INPCRD="${TOPO_DIR}/${PDB}.inpcrd"
    TLEAP_IN="${TOPO_DIR}/${PDB}_tleap.in"

    if [ ! -f "$INPUT_PDB" ]; then
        echo "SKIP $PDB: $INPUT_PDB not found"
        continue
    fi

    echo ""
    echo "[$PDB] Generating topology..."

    # Create tleap input script
    cat > "$TLEAP_IN" << EOF
# Load AMBER ff14SB force field
source leaprc.protein.ff14SB

# Load the fixed PDB
mol = loadpdb $INPUT_PDB

# Check for any issues
check mol

# Save topology and coordinates
saveamberparm mol $OUTPUT_PRMTOP $OUTPUT_INPCRD

quit
EOF

    # Run tleap
    if $TLEAP -f "$TLEAP_IN" > "${TOPO_DIR}/${PDB}_tleap.log" 2>&1; then
        if [ -f "$OUTPUT_PRMTOP" ] && [ -f "$OUTPUT_INPCRD" ]; then
            PRMTOP_SIZE=$(ls -lh "$OUTPUT_PRMTOP" | awk '{print $5}')
            echo "  OK: $OUTPUT_PRMTOP ($PRMTOP_SIZE)"
        else
            echo "  FAILED: Output files not created"
            cat "${TOPO_DIR}/${PDB}_tleap.log"
        fi
    else
        echo "  FAILED: tleap error"
        tail -20 "${TOPO_DIR}/${PDB}_tleap.log"
    fi
done

echo ""
echo "=============================================="
echo "Summary"
echo "=============================================="
ls -la "$TOPO_DIR"/*.prmtop 2>/dev/null || echo "No topology files generated"

#!/bin/bash
# PRISM4D Cryo-UV Ensemble Generation Script
# Usage: ./run_cryo_uv_ensemble.sh <topology.json> <output_dir>

set -e

# Configuration
SURVEY_STEPS=${SURVEY_STEPS:-50000}
CONVERGENCE_STEPS=${CONVERGENCE_STEPS:-25000}
PRECISION_STEPS=${PRECISION_STEPS:-25000}
CRYO_TEMP=${CRYO_TEMP:-100.0}
TARGET_TEMP=${TARGET_TEMP:-300.0}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     PRISM4D Cryo-UV Ensemble Generation                        ║"
echo "║     Cryptic Binding Site Discovery Pipeline                    ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check arguments
if [ $# -lt 2 ]; then
    echo -e "${RED}Usage: $0 <topology.json> <output_dir>${NC}"
    echo ""
    echo "Environment variables:"
    echo "  SURVEY_STEPS      - Survey phase steps (default: 50000)"
    echo "  CONVERGENCE_STEPS - Convergence phase steps (default: 25000)"
    echo "  PRECISION_STEPS   - Precision phase steps (default: 25000)"
    echo "  CRYO_TEMP         - Starting temperature in K (default: 100.0)"
    echo "  TARGET_TEMP       - Final temperature in K (default: 300.0)"
    exit 1
fi

TOPOLOGY=$1
OUTPUT_DIR=$2

# Validate input
if [ ! -f "$TOPOLOGY" ]; then
    echo -e "${RED}Error: Topology file not found: $TOPOLOGY${NC}"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}Configuration:${NC}"
echo "  Topology:     $TOPOLOGY"
echo "  Output:       $OUTPUT_DIR"
echo "  Survey:       $SURVEY_STEPS steps"
echo "  Convergence:  $CONVERGENCE_STEPS steps"
echo "  Precision:    $PRECISION_STEPS steps"
echo "  Temperature:  ${CRYO_TEMP}K → ${TARGET_TEMP}K"
echo ""

# Find PRISM4D directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRISM_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Check for binary
NHS_ADAPTIVE="$PRISM_DIR/target/release/nhs-adaptive"
if [ ! -f "$NHS_ADAPTIVE" ]; then
    echo -e "${YELLOW}Building nhs-adaptive...${NC}"
    cd "$PRISM_DIR"
    cargo build --release -p prism-nhs --features gpu --bin nhs-adaptive
    cd -
fi

# Run ensemble generation
echo -e "${GREEN}Starting Cryo-UV ensemble generation...${NC}"
echo ""

START_TIME=$(date +%s)

"$NHS_ADAPTIVE" "$TOPOLOGY" \
    -o "$OUTPUT_DIR" \
    --survey-steps $SURVEY_STEPS \
    --convergence-steps $CONVERGENCE_STEPS \
    --precision-steps $PRECISION_STEPS \
    --cryo \
    --cryo-temp $CRYO_TEMP \
    --temperature $TARGET_TEMP \
    -v

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Ensemble generation complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Elapsed time: ${ELAPSED}s"
echo "  Output files:"
ls -lh "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.pdb 2>/dev/null || true
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Extract stable frames: python3 scripts/extract_stable_frames.py $OUTPUT_DIR"
echo "  2. Run correlation: python3 scripts/correlate_rmsf_spikes.py $OUTPUT_DIR"

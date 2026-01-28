#!/bin/bash
# =============================================================================
# PRISM4D Complete Comprehensive Package Generator
# =============================================================================
# Usage: bash generate_complete_package.sh <topology.json> <output_dir> [frames]
# =============================================================================

set -e

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TOPOLOGY="$1"
OUTPUT_DIR="$2"
FRAMES="${3:-200}"  # Default 200 frames

if [ -z "$TOPOLOGY" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <topology.json> <output_dir> [frames]"
    echo ""
    echo "Example:"
    echo "  $0 data/production/topologies_json/6M0J_topology.json /tmp/6M0J_complete 200"
    echo ""
    echo "This generates:"
    echo "  - Trajectory (ensemble PDB + frames JSON) via nhs-cryo-probe"
    echo "  - JSON outputs (cryptic_sites.json, comprehensive_report.json) via nhs-analyze-pro"
    echo "  - 5 PNG figures (Figure 11-13 + bonuses)"
    echo "  - 8 PyMOL scripts (master session, movies, pharma analysis)"
    echo "  - 4 movies (rotation, tour, reveal, wavelengths) [requires PyMOL]"
    exit 1
fi

echo "============================================================"
echo "PRISM4D Complete Package Generator"
echo "============================================================"
echo "Topology:      $TOPOLOGY"
echo "Output:        $OUTPUT_DIR"
echo "Frames:        $FRAMES"
echo ""

# Verify topology exists
if [ ! -f "$TOPOLOGY" ]; then
    echo "ERROR: Topology file not found: $TOPOLOGY"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Extract PDB ID from topology
PDB_ID=$(jq -r '.pdb_id // "unknown"' "$TOPOLOGY" 2>/dev/null || echo "unknown")
echo "Detected PDB ID: $PDB_ID"
echo ""

# ============================================================================
# STEP 1: Generate Trajectory with UV Spectroscopy
# ============================================================================
echo "[1/4] Generating trajectory with nhs-cryo-probe..."
echo "       (This may take 30-120 seconds depending on structure size)"
echo ""

TRAJ_DIR="${OUTPUT_DIR}/trajectory"
mkdir -p "$TRAJ_DIR"

cd "$REPO_ROOT"
cargo run --release -p prism-nhs --bin nhs-cryo-probe -- \
    --topology "$TOPOLOGY" \
    --output "$TRAJ_DIR" \
    --frames "$FRAMES" \
    --temperature 300.0 \
    --spectroscopy \
    --verbose

if [ $? -ne 0 ]; then
    echo "ERROR: Trajectory generation failed"
    exit 1
fi

# Find generated files
ENSEMBLE_PDB=$(find "$TRAJ_DIR" -name "*ensemble.pdb" -type f | head -1)
FRAMES_JSON=$(find "$TRAJ_DIR" -name "*frames.json" -type f | head -1)

if [ -z "$ENSEMBLE_PDB" ] || [ ! -f "$ENSEMBLE_PDB" ]; then
    echo "ERROR: Ensemble PDB not found in $TRAJ_DIR"
    ls -la "$TRAJ_DIR"
    exit 1
fi

if [ -z "$FRAMES_JSON" ] || [ ! -f "$FRAMES_JSON" ]; then
    echo "WARNING: Frames JSON not found - continuing without wavelength data"
    FRAMES_ARG=""
else
    FRAMES_ARG="--frames-json $FRAMES_JSON"
    echo "Found frames JSON: $FRAMES_JSON"
fi

echo "Found ensemble PDB: $ENSEMBLE_PDB"
echo ""

# ============================================================================
# STEP 2: Analyze Trajectory for Cryptic Sites
# ============================================================================
echo "[2/4] Analyzing trajectory with nhs-analyze-pro..."
echo "       (GPU-accelerated neural spike detection)"
echo ""

cd "$REPO_ROOT"
cargo run --release -p prism-nhs --bin nhs-analyze-pro -- \
    --topology "$TOPOLOGY" \
    --output "$OUTPUT_DIR" \
    $FRAMES_ARG \
    --verbose \
    "$ENSEMBLE_PDB"

if [ $? -ne 0 ]; then
    echo "ERROR: NHS analysis failed"
    exit 1
fi

# Verify outputs
if [ ! -f "$OUTPUT_DIR/cryptic_sites.json" ]; then
    echo "ERROR: cryptic_sites.json not generated"
    exit 1
fi

if [ ! -f "$OUTPUT_DIR/comprehensive_report.json" ]; then
    echo "ERROR: comprehensive_report.json not generated"
    exit 1
fi

# ============================================================================
# STEP 3: Generate Figures + PyMOL Scripts
# ============================================================================
echo ""
echo "[3/4] Generating Figures + PyMOL Scripts..."

python3 "$SCRIPT_DIR/generate_comprehensive_figures.py" "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: Figure generation failed"
    exit 1
fi

# ============================================================================
# STEP 4: Generate Movies (Optional - requires PyMOL)
# ============================================================================
echo ""
echo "[4/4] Generating Movies (requires PyMOL)..."

cd "$OUTPUT_DIR"

if command -v pymol &> /dev/null; then
    DETECTED_PDB_ID=$(jq -r '.pdb_id' comprehensive_report.json 2>/dev/null || echo "$PDB_ID")

    if [ -f "${DETECTED_PDB_ID}_generate_movies.sh" ]; then
        echo "       Running PyMOL movie generator..."
        echo "       (This will take several minutes for ray-traced quality)"
        echo ""
        bash "${DETECTED_PDB_ID}_generate_movies.sh" || {
            echo "WARNING: Movie generation had errors (continuing anyway)"
        }
    else
        echo "WARNING: Movie script not found: ${DETECTED_PDB_ID}_generate_movies.sh"
    fi
else
    echo "INFO: PyMOL not found. Skipping movie generation."
    echo ""
    echo "To install PyMOL:"
    echo "  Ubuntu/Debian: sudo apt-get install pymol"
    echo "  Conda:         conda install -c conda-forge pymol-open-source"
    echo ""
    echo "To generate movies later:"
    DETECTED_PDB_ID=$(jq -r '.pdb_id' comprehensive_report.json 2>/dev/null || echo "$PDB_ID")
    echo "  cd $OUTPUT_DIR && bash ${DETECTED_PDB_ID}_generate_movies.sh"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo "COMPLETE PACKAGE GENERATED!"
echo "============================================================"

cd "$OUTPUT_DIR"
DETECTED_PDB_ID=$(jq -r '.pdb_id' comprehensive_report.json 2>/dev/null || echo "$PDB_ID")

echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Count outputs
JSON_COUNT=$(ls -1 *.json 2>/dev/null | wc -l)
PNG_COUNT=$(ls -1 Figure*.png 2>/dev/null | wc -l)
PML_COUNT=$(ls -1 *.pml 2>/dev/null | wc -l)
MOVIE_COUNT=$(ls -1 ${DETECTED_PDB_ID}_movies/*.mp4 2>/dev/null | wc -l || echo 0)

echo "Generated:"
echo "  - $JSON_COUNT JSON outputs"
echo "  - $PNG_COUNT PNG figures"
echo "  - $PML_COUNT PyMOL scripts"
echo "  - $MOVIE_COUNT movies"
echo ""

# Show key metrics
if [ -f "comprehensive_report.json" ]; then
    echo "Analysis Summary:"
    jq -r '"  PDB: \(.pdb_id)"' comprehensive_report.json 2>/dev/null
    jq -r '"  Sites found: \(.summary.sites_found)"' comprehensive_report.json 2>/dev/null
    jq -r '"  HIGH confidence: \(.summary.high_confidence)"' comprehensive_report.json 2>/dev/null
    jq -r '"  MEDIUM confidence: \(.summary.medium_confidence)"' comprehensive_report.json 2>/dev/null
    jq -r '"  Total spikes: \(.summary.total_spikes)"' comprehensive_report.json 2>/dev/null
    echo ""
fi

echo "Quick Start:"
echo "  View figures:       eog Figure*.png"
echo "  PyMOL master:       pymol ${DETECTED_PDB_ID}_PRISM4D_master.pml"
echo "  Pharma analysis:    pymol ${DETECTED_PDB_ID}_pharma_actionable.pml"
echo "  Play movies:        vlc ${DETECTED_PDB_ID}_movies/*.mp4"
echo ""

echo "All files:"
ls -lh *.json *.png *.pml 2>/dev/null | tail -n +2 || ls -lh

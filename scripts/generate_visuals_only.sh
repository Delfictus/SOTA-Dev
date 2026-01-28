#!/bin/bash
# =============================================================================
# PRISM4D Visualization Generator (For Existing Analysis Results)
# =============================================================================
# Usage: bash generate_visuals_only.sh <output_dir_with_json>
#
# Use this when you ALREADY have cryptic_sites.json and comprehensive_report.json
# and just want to generate figures + PyMOL scripts + movies
# =============================================================================

set -e

# Get script directory and repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

OUTPUT_DIR="$1"

if [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: $0 <output_dir>"
    echo ""
    echo "Example:"
    echo "  $0 /tmp/6M0J_comprehensive"
    echo ""
    echo "This script generates visualizations from EXISTING analysis results."
    echo "It expects these files to already exist in the output directory:"
    echo "  - cryptic_sites.json"
    echo "  - comprehensive_report.json"
    echo ""
    echo "Generates:"
    echo "  - 5 PNG figures (Figure 11-13 + bonuses)"
    echo "  - 8 PyMOL scripts (master session, movies, pharma analysis)"
    echo "  - 4 movies (if PyMOL installed)"
    exit 1
fi

echo "============================================================"
echo "PRISM4D Visualization Generator"
echo "============================================================"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Verify output directory exists
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: Directory not found: $OUTPUT_DIR"
    exit 1
fi

# Verify required files exist
if [ ! -f "$OUTPUT_DIR/cryptic_sites.json" ]; then
    echo "ERROR: cryptic_sites.json not found in $OUTPUT_DIR"
    echo "This script requires existing analysis results."
    echo "Run nhs-analyze-pro first, or use generate_complete_package.sh"
    exit 1
fi

if [ ! -f "$OUTPUT_DIR/comprehensive_report.json" ]; then
    echo "ERROR: comprehensive_report.json not found in $OUTPUT_DIR"
    echo "This script requires existing analysis results."
    exit 1
fi

echo "✓ Found existing analysis results"
echo ""

# Get PDB ID
PDB_ID=$(jq -r '.pdb_id // "unknown"' "$OUTPUT_DIR/comprehensive_report.json" 2>/dev/null)
echo "PDB ID: $PDB_ID"
echo ""

# ============================================================================
# Generate Figures + PyMOL Scripts
# ============================================================================
echo "[1/2] Generating Figures + PyMOL Scripts..."

python3 "$SCRIPT_DIR/generate_comprehensive_figures.py" "$OUTPUT_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: Figure generation failed"
    exit 1
fi

# ============================================================================
# Generate Movies (Optional)
# ============================================================================
echo ""
echo "[2/2] Generating Movies (requires PyMOL)..."

cd "$OUTPUT_DIR"

if command -v pymol &> /dev/null; then
    if [ -f "${PDB_ID}_generate_movies.sh" ]; then
        echo "       Running PyMOL movie generator..."
        echo "       (This will take several minutes for ray-traced quality)"
        echo ""

        # Ask user if they want to generate movies (they take time)
        echo "Generate movies now? This takes 5-10 minutes. [y/N]"
        read -t 10 -n 1 RESPONSE || RESPONSE="n"
        echo ""

        if [[ "$RESPONSE" =~ ^[Yy]$ ]]; then
            bash "${PDB_ID}_generate_movies.sh" || {
                echo "WARNING: Movie generation had errors (continuing anyway)"
            }
        else
            echo "Skipping movie generation."
            echo "To generate movies later, run:"
            echo "  cd $OUTPUT_DIR && bash ${PDB_ID}_generate_movies.sh"
        fi
    else
        echo "WARNING: Movie script not found: ${PDB_ID}_generate_movies.sh"
    fi
else
    echo "INFO: PyMOL not found. Skipping movie generation."
    echo ""
    echo "To install PyMOL:"
    echo "  Ubuntu/Debian: sudo apt-get install pymol"
    echo "  Conda:         conda install -c conda-forge pymol-open-source"
    echo ""
    echo "To generate movies later:"
    echo "  cd $OUTPUT_DIR && bash ${PDB_ID}_generate_movies.sh"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================================"
echo "VISUALIZATIONS GENERATED!"
echo "============================================================"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo ""

# Count outputs
cd "$OUTPUT_DIR"
JSON_COUNT=$(ls -1 *.json 2>/dev/null | wc -l)
PNG_COUNT=$(ls -1 Figure*.png 2>/dev/null | wc -l)
PML_COUNT=$(ls -1 *.pml 2>/dev/null | wc -l)
MOVIE_COUNT=$(ls -1 ${PDB_ID}_movies/*.mp4 2>/dev/null | wc -l || echo 0)

echo "Generated:"
echo "  - $PNG_COUNT PNG figures"
echo "  - $PML_COUNT PyMOL scripts"
echo "  - $MOVIE_COUNT movies"
echo "  - $JSON_COUNT JSON files (existing)"
echo ""

# Show key metrics
if [ -f "comprehensive_report.json" ]; then
    echo "Analysis Summary:"
    jq -r '"  PDB: \(.pdb_id)"' comprehensive_report.json 2>/dev/null
    jq -r '"  Sites found: \(.summary.sites_found)"' comprehensive_report.json 2>/dev/null
    jq -r '"  HIGH confidence: \(.summary.high_confidence)"' comprehensive_report.json 2>/dev/null
    jq -r '"  MEDIUM confidence: \(.summary.medium_confidence)"' comprehensive_report.json 2>/dev/null
    echo ""
fi

echo "Quick Start:"
echo "  View figures:       eog Figure*.png"
echo "  PyMOL master:       pymol ${PDB_ID}_PRISM4D_master.pml"
echo "  Pharma analysis:    pymol ${PDB_ID}_pharma_actionable.pml"
if [ $MOVIE_COUNT -gt 0 ]; then
    echo "  Play movies:        vlc ${PDB_ID}_movies/*.mp4"
fi
echo ""

echo "All visualization files:"
ls -lh Figure*.png *.pml 2>/dev/null | grep -v "^total"

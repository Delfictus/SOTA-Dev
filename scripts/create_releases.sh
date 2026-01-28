#!/bin/bash
# PRISM4D Release Creation Script
# Creates git tags and GitHub releases for all stable modules

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DATE=$(date +%Y-%m-%d)
VERSION_SUFFIX=${1:-"v1.0.0"}

echo "=== PRISM4D Release Script ==="
echo "Date: $DATE"
echo "Version suffix: $VERSION_SUFFIX"
echo ""

# Function to create a tag
create_tag() {
    local name=$1
    local message=$2
    local tag="${name}-${VERSION_SUFFIX}"

    if git rev-parse "$tag" >/dev/null 2>&1; then
        echo "  [SKIP] Tag $tag already exists"
    else
        echo "  [CREATE] $tag"
        git tag -a "$tag" -m "$message"
    fi
}

# Function to create GitHub release
create_release() {
    local tag=$1
    local title=$2
    local binary=$3
    local notes=$4

    if gh release view "$tag" >/dev/null 2>&1; then
        echo "  [SKIP] Release $tag already exists"
    else
        echo "  [CREATE] GitHub release $tag"
        if [ -f "$binary" ]; then
            gh release create "$tag" \
                --title "$title" \
                --notes "$notes" \
                "$binary"
        else
            gh release create "$tag" \
                --title "$title" \
                --notes "$notes"
        fi
    fi
}

echo "=== Creating Git Tags ==="
echo ""

echo "Cryo-UV Pipeline:"
create_tag "nhs-adaptive" "Cryo-UV spike detection - 200K spikes, 231 sites, 2001 steps/sec"
create_tag "nhs-batch" "Batched MD processing - 315K atoms/sec"
create_tag "nhs-cryo-probe" "Cryo-UV pump-probe protocol - 100K cryo temp"
create_tag "nhs-detect" "Static neuromorphic detection"
create_tag "nhs-diagnose" "NHS diagnostic tool"

echo ""
echo "Ensemble Generation:"
create_tag "ensemble-simd" "SIMD batched ensemble generation - 315K atoms/sec"
create_tag "ensemble-standard" "Standard ensemble generation"

echo ""
echo "Preprocessing:"
create_tag "prism-prep" "Production PDB preprocessing - AMBER reduce, ff14SB"

echo ""
echo "Benchmarking:"
create_tag "cryptobench" "CryptoBench v2 validation - ROC 0.445"
create_tag "atlas-bench" "ATLAS benchmark - 96.3% pass rate"
create_tag "apo-holo-bench" "Apo-holo benchmark suite"

echo ""
echo "Core Libraries:"
create_tag "prism-gpu" "GPU kernels - CUDA MD, TDA, NHS"
create_tag "prism-nhs" "NHS Rust library"
create_tag "prism-physics" "Physics engines - AMBER, Langevin"
create_tag "prism-io" "I/O utilities - PDB, topology"
create_tag "prism-core" "Core data structures"
create_tag "prism-validation" "Validation framework"

echo ""
echo "=== Pushing Tags to Remote ==="
read -p "Push all tags to origin? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push origin --tags
    echo "Tags pushed successfully"
else
    echo "Tags created locally only. Push with: git push origin --tags"
fi

echo ""
echo "=== Creating GitHub Releases ==="
read -p "Create GitHub releases with binaries? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    RELEASE_DIR="$REPO_ROOT/target/release"

    create_release "nhs-adaptive-${VERSION_SUFFIX}" \
        "NHS Adaptive ${VERSION_SUFFIX} - Cryo-UV Spike Detection" \
        "$RELEASE_DIR/nhs-adaptive" \
        "## NHS Adaptive ${VERSION_SUFFIX}

### Cryo-UV Cryptic Site Detection

**Key Metrics:**
- 200,000+ raw spikes detected
- 35,100 weighted spikes
- 231 correlated druggable sites
- 2,001 steps/sec on RTX 3060

**Features:**
- Cryo-UV pump-probe protocol (100K → 300K)
- LIF neuromorphic spike detection
- RMSF × spike correlation analysis

**Usage:**
\`\`\`bash
nhs-adaptive --topology input.json --output results/ --cryo-temp 100
\`\`\`
"

    create_release "prism-prep-${VERSION_SUFFIX}" \
        "PRISM-PREP ${VERSION_SUFFIX} - Production Preprocessing" \
        "$RELEASE_DIR/../scripts/prism-prep" \
        "## PRISM-PREP ${VERSION_SUFFIX}

### Production PDB Preprocessing

**Features:**
- AMBER reduce for H-bond optimization
- Cryptic/escape mode routing
- ff14SB topology generation
- Strict validation mode

**Usage:**
\`\`\`bash
prism-prep input.pdb output.json --use-amber --mode cryptic --strict
\`\`\`
"

    echo "GitHub releases created"
else
    echo "Skipping GitHub releases"
fi

echo ""
echo "=== Release Summary ==="
echo "Tags created: $(git tag | wc -l)"
git tag | tail -20

echo ""
echo "Done! See RELEASES.md for full documentation."

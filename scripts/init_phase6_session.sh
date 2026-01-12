#!/bin/bash
# Phase 6 Session Initialization Script
# Run this BEFORE starting a new Claude Code session

set -e

echo "========================================"
echo "PRISM Phase 6 Session Initialization"
echo "========================================"
echo ""

# Check current phase
echo "1. Checking current checkpoint status..."
if [[ -f "./scripts/phase6_checkpoint.sh" ]]; then
    ./scripts/phase6_checkpoint.sh auto 2>/dev/null || true
else
    echo "   [WARN] Checkpoint script not found"
fi
echo ""

# Verify critical files exist
echo "2. Verifying critical files..."
CRITICAL_FILES=(
    "CLAUDE.md"
    "results/phase6_sota_plan.md"
    "results/phase7_8_sota_plan.md"
    "scripts/phase6_compliance_check.sh"
)

for file in "${CRITICAL_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo "   [OK] $file"
    else
        echo "   [MISSING] $file"
    fi
done
echo ""

# Show implementation status
echo "3. Implementation status..."
PHASE6_FILES=(
    "crates/prism-validation/src/cryptic_features.rs"
    "crates/prism-validation/src/gpu_zro_cryptic_scorer.rs"
    "crates/prism-validation/src/pdb_sanitizer.rs"
    "crates/prism-validation/src/sampling/contract.rs"
    "crates/prism-validation/src/sampling/paths/nova_path.rs"
    "crates/prism-validation/src/sampling/paths/amber_path.rs"
    "crates/prism-validation/src/cryptobench_dataset.rs"
    "crates/prism-validation/src/ablation.rs"
)

IMPLEMENTED=0
TOTAL=${#PHASE6_FILES[@]}

for file in "${PHASE6_FILES[@]}"; do
    if [[ -f "$file" ]]; then
        echo "   [DONE] $(basename $file)"
        ((IMPLEMENTED++))
    else
        echo "   [TODO] $(basename $file)"
    fi
done
echo ""
echo "   Progress: $IMPLEMENTED / $TOTAL files"
echo ""

# Determine next action
echo "4. Next implementation target..."
for file in "${PHASE6_FILES[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "   -> $(basename $file)"
        break
    fi
done
echo ""

echo "========================================"
echo "SESSION READY"
echo "========================================"
echo ""
echo "Copy this prompt to start Claude Code:"
echo ""
echo "----------------------------------------"
cat << 'PROMPT'
Begin Phase 6 implementation session.

1. Read CLAUDE.md for project instructions
2. Read results/phase6_sota_plan.md for implementation plan
3. Run: ./scripts/phase6_checkpoint.sh auto
4. Identify and implement the next file in order
5. Follow atomic implementation units (one file at a time)
6. Run tests after each file
7. Commit after each successful implementation

What is the current checkpoint status and next implementation target?
PROMPT
echo "----------------------------------------"

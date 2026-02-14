#!/bin/bash
# Phase 6 Session Initialization Script
# Run this BEFORE starting a new Claude Code session

set -e

VAULT_DIR=".obsidian/vault"
STATUS_FILE="$VAULT_DIR/Progress/implementation_status.json"
SESSION_FILE="$VAULT_DIR/Sessions/Current Session.md"

echo "========================================"
echo "PRISM Phase 6 Session Initialization"
echo "========================================"
echo ""

# Check vault exists
if [[ ! -d "$VAULT_DIR" ]]; then
    echo "[ERROR] Obsidian vault not found at $VAULT_DIR"
    exit 1
fi

# Read progress from vault
echo "1. Reading progress from Obsidian vault..."
if [[ -f "$STATUS_FILE" ]]; then
    CURRENT_PHASE=$(jq -r '.current_phase' "$STATUS_FILE")
    CURRENT_WEEK=$(jq -r '.current_week' "$STATUS_FILE")
    PROGRESS=$(jq -r '.overall_progress_percent' "$STATUS_FILE")
    NEXT_FILE=$(jq -r '.next_action.file' "$STATUS_FILE")
    NEXT_SECTION=$(jq -r '.next_action.plan_section' "$STATUS_FILE")

    echo "   Phase: $CURRENT_PHASE"
    echo "   Week: $CURRENT_WEEK"
    echo "   Progress: ${PROGRESS}%"
    echo "   Next: $NEXT_FILE"
else
    echo "   [WARN] Status file not found, using defaults"
    NEXT_FILE="cryptic_features.rs"
    NEXT_SECTION="4.1"
fi
echo ""

# Count completed files
echo "2. Implementation status..."
TOTAL_FILES=15
COMPLETED=0

if [[ -f "$STATUS_FILE" ]]; then
    COMPLETED=$(jq '[.files[][] | select(.status == "completed")] | length' "$STATUS_FILE")
fi

echo "   Completed: $COMPLETED / $TOTAL_FILES files"
echo ""

# Check current checkpoint
echo "3. Checkpoint status..."
if [[ -f "./scripts/phase6_checkpoint.sh" ]]; then
    ./scripts/phase6_checkpoint.sh auto 2>&1 | head -10 || true
fi
echo ""

# Read last session context
echo "4. Last session context..."
if [[ -f "$SESSION_FILE" ]]; then
    echo "   Found: $SESSION_FILE"
    # Extract key context
    if grep -q "Blocking issues:" "$SESSION_FILE"; then
        echo "   Blocking issues:"
        sed -n '/Blocking issues:/,/^$/p' "$SESSION_FILE" | head -5
    fi
else
    echo "   [NEW] No previous session found"
fi
echo ""

echo "========================================"
echo "SESSION READY"
echo "========================================"
echo ""
echo "Next target: $NEXT_FILE"
echo "Plan section: $NEXT_SECTION"
echo ""
echo "Copy this prompt to start Claude Code:"
echo ""
echo "----------------------------------------"
cat << PROMPT
Continue Phase 6 implementation.

**Read these files first:**
1. .obsidian/vault/Sessions/Current Session.md
2. .obsidian/vault/Progress/implementation_status.json
3. CLAUDE.md

**Current State:**
- Phase: $CURRENT_PHASE
- Week: $CURRENT_WEEK
- Progress: ${PROGRESS}%
- Completed: $COMPLETED / $TOTAL_FILES files

**Next Target:** \`$NEXT_FILE\`
**Plan Section:** $NEXT_SECTION

Read the vault files, confirm the current state, and proceed with implementation.
Update vault files as you work.
PROMPT
echo "----------------------------------------"
echo ""
echo "Or for a fresh start:"
echo ""
echo "----------------------------------------"
cat << 'FRESH'
Begin fresh Phase 6 implementation session.

1. Read CLAUDE.md for all constraints
2. Read .obsidian/vault/Progress/implementation_status.json for current state
3. Read .obsidian/vault/Sessions/Current Session.md for context
4. Identify next file from status JSON
5. Implement, test, commit
6. Update vault files after each file completion

What is the current state and next implementation target?
FRESH
echo "----------------------------------------"

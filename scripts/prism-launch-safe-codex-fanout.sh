#!/usr/bin/env bash
# scripts/prism-launch-safe-codex-fanout.sh
#
# Validate that all gates are satisfied for the safe Codex fan-out subset
# and emit FANOUT_READY (or FANOUT_BLOCKED) to the orchestration event log.
#
# This script DOES NOT spawn agents. It only:
#   1. checks APPROVALS/FANOUT_APPROVED is active (approved: true)
#   2. checks no high-risk tracked runtime files are dirty
#   3. checks active runtime blockers = 0
#   4. prints the approved safe-lane ticket prompts
#   5. appends FANOUT_READY or FANOUT_BLOCKED to EVENTS.ndjson
#   6. updates DASHBOARD.md with a fan-out status section
#
# Approved safe lanes (only set):
#   - F2-SCOUT-001         (Codex-A, read-only)
#   - DAG-SCHEMA-001       (Codex-B)
#   - CONTROLTRACE-001     (Codex-C)
#   - TEST-AUDIT-RUNNER-001 (Codex-H)
#
# Excluded regardless of approval:
#   F1, WHILE, ASC runtime changes, ClusterEvent touching nhs_rt_full.rs,
#   site materialization, runtime topology changes.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
ORCH_DIR="$REPO_ROOT/.prism_orchestration"
APPROVALS="$ORCH_DIR/APPROVALS"
EVENTS="$ORCH_DIR/EVENTS.ndjson"
DASHBOARD="$ORCH_DIR/DASHBOARD.md"
REPORTS="$ORCH_DIR/reports"

mkdir -p "$REPORTS"
cd "$REPO_ROOT"

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

emit_event() {
    printf '%s\n' "$1" >> "$EVENTS"
}

approval_active() {
    local marker="$1"
    [[ -f "$marker" ]] && grep -Eq '^approved:[[:space:]]*true[[:space:]]*$' "$marker"
}

reasons=()

# -- Check 1: FANOUT_APPROVED active ----------------------------------------
if approval_active "$APPROVALS/FANOUT_APPROVED"; then
    fanout_ok=1
else
    fanout_ok=0
    reasons+=("FANOUT_APPROVED is not active (file missing or 'approved: true' not set)")
fi

# -- Check 2: no high-risk tracked runtime files dirty ----------------------
HIGH_RISK_FILES=(
    crates/prism-nhs/src/bin/nhs_rt_full.rs
    crates/prism-nhs/src/captured_pipeline.rs
    crates/prism-nhs/src/graph_capture.rs
    crates/prism-nhs/src/graph_node.rs
    crates/prism-nhs/src/cuda/adjudicator.cu
    crates/prism-nhs/src/cuda/gearbox.cu
    crates/prism-nhs/src/cuda/graph_node.cu
    crates/prism-nhs/build.rs
)
high_risk_dirty="$(git status --short --untracked-files=no -- "${HIGH_RISK_FILES[@]}" 2>/dev/null || true)"
if [[ -z "$high_risk_dirty" ]]; then
    high_risk_clean=1
else
    high_risk_clean=0
    reasons+=("high-risk tracked runtime files are dirty: $(echo "$high_risk_dirty" | tr '\n' ';')")
fi

# -- Check 3: active runtime blockers = 0 -----------------------------------
ACTIVE_HITS_FILE="$REPORTS/active_run_cuda_failure_hits.txt"
if [[ ! -f "$ACTIVE_HITS_FILE" ]] || [[ ! -s "$ACTIVE_HITS_FILE" ]]; then
    active_blockers_zero=1
    active_hits_count=0
else
    active_hits_count="$(wc -l < "$ACTIVE_HITS_FILE" | tr -d ' ')"
    if [[ "$active_hits_count" == "0" ]]; then
        active_blockers_zero=1
    else
        active_blockers_zero=0
        reasons+=("active runtime blockers detected: $active_hits_count hit(s) in $ACTIVE_HITS_FILE")
    fi
fi

# -- Verdict ----------------------------------------------------------------
if (( fanout_ok && high_risk_clean && active_blockers_zero )); then
    verdict="READY"
else
    verdict="BLOCKED"
fi

# -- Write/print ticket prompts (always emit, even if blocked, as documentation)
TICKET_PROMPT_FILE="$REPORTS/safe_lane_ticket_prompts.md"
{
    echo "# Safe-Lane Ticket Prompts"
    echo
    echo "Generated: $(ts)"
    echo "Verdict: $verdict"
    echo
    if (( !fanout_ok || !high_risk_clean || !active_blockers_zero )); then
        echo "## BLOCKERS"
        for r in "${reasons[@]}"; do
            echo "- $r"
        done
        echo
        echo "Tickets below are documented but MUST NOT be launched until all blockers clear."
        echo
    fi
    cat <<'TICKETS'
## F2-SCOUT-001 — Codex-A — read-only

Lane:  read-only F2 evidence-plane preparation
Allowed reads:
  crates/prism-nhs/src/zstr.rs
  crates/prism-nhs/src/ghost_tile.rs
  crates/prism-nhs/src/bin/nhs_rt_full.rs (read-only)
Forbidden writes: all runtime files.
Output: docs/F2_PREP_NOTES.md (under your AGENT_OUTBOX), no code changes.
Acceptance: read-only audit complete; emit AGENT_COMPLETE with status=READONLY_AUDIT_DONE.

## DAG-SCHEMA-001 — Codex-B — bounded implementation

Lane: green
Allowed files:
  crates/prism-core/src/dag.rs
  crates/prism-core/src/lib.rs
Forbidden: any file under crates/prism-nhs/, build.rs, *.cu, *.cuh.
Acceptance:
  cargo check -p prism-core
  cargo test  -p prism-core dag
  no runtime files touched
  emit AGENT_COMPLETE with files_touched listed

## CONTROLTRACE-001 — Codex-C — bounded implementation

Lane: green
Allowed files:
  crates/prism-nhs/src/control_trace.rs
  crates/prism-nhs/src/lib.rs (single `pub mod` line only)
Forbidden: any other prism-nhs file, no runtime topology changes.
Acceptance:
  cargo check -p prism-nhs
  cargo test  -p prism-nhs control_trace
  emit AGENT_COMPLETE

## TEST-AUDIT-RUNNER-001 — Codex-H — bounded implementation

Lane: green/medium
Allowed files:
  scripts/prism-test-audit-runner.sh   (NEW)
  scripts/lib/prism-audit-helpers.sh   (NEW, optional)
Forbidden: any crates/ file, any *.cu/*.cuh, any build.rs.
Acceptance:
  bash -n on the new script
  emit AGENT_COMPLETE with test_summary attached
TICKETS
} | tee "$TICKET_PROMPT_FILE" >/dev/null

# -- Append FANOUT_READY or FANOUT_BLOCKED event ---------------------------
if [[ "$verdict" == "READY" ]]; then
    emit_event "{\"ts\":\"$(ts)\",\"agent\":\"safe-fanout\",\"event\":\"FANOUT_READY\",\"approved_lanes\":[\"F2-SCOUT-001\",\"DAG-SCHEMA-001\",\"CONTROLTRACE-001\",\"TEST-AUDIT-RUNNER-001\"],\"prompts\":\"$TICKET_PROMPT_FILE\"}"
else
    # Build a single reason string (escape quotes)
    reason_str="$(printf '%s; ' "${reasons[@]}" | sed 's/"/\\"/g')"
    emit_event "{\"ts\":\"$(ts)\",\"agent\":\"safe-fanout\",\"event\":\"FANOUT_BLOCKED\",\"reason\":\"${reason_str%; }\"}"
fi

# -- Update DASHBOARD.md with a fan-out status block (idempotent: replace marker block)
DASH_TMP="$(mktemp)"
trap 'rm -f "$DASH_TMP"' EXIT

if [[ -f "$DASHBOARD" ]]; then
    awk '
        /^<!-- FANOUT-STATUS-BEGIN -->$/ {skip=1; next}
        /^<!-- FANOUT-STATUS-END -->$/   {skip=0; next}
        !skip
    ' "$DASHBOARD" > "$DASH_TMP"
    cp "$DASH_TMP" "$DASHBOARD"
fi

{
    echo
    echo "<!-- FANOUT-STATUS-BEGIN -->"
    echo "## Fan-out Status"
    echo
    echo "- Verdict: **$verdict**"
    echo "- Generated: $(ts)"
    echo "- FANOUT_APPROVED active: $([ "$fanout_ok" = "1" ] && echo yes || echo no)"
    echo "- High-risk tracked dirty: $([ "$high_risk_clean" = "1" ] && echo none || echo yes)"
    echo "- Active runtime blockers: $active_hits_count"
    if [[ "$verdict" == "BLOCKED" ]]; then
        echo "- Reasons:"
        for r in "${reasons[@]}"; do
            echo "  - $r"
        done
    fi
    echo "- Approved safe-lane subset (when ready):"
    echo "  - F2-SCOUT-001 (Codex-A, read-only)"
    echo "  - DAG-SCHEMA-001 (Codex-B)"
    echo "  - CONTROLTRACE-001 (Codex-C)"
    echo "  - TEST-AUDIT-RUNNER-001 (Codex-H)"
    echo "- Ticket prompts: \`.prism_orchestration/reports/safe_lane_ticket_prompts.md\`"
    echo "<!-- FANOUT-STATUS-END -->"
} >> "$DASHBOARD"

# -- Stdout summary ---------------------------------------------------------
echo "[safe-fanout] verdict: $verdict"
if [[ "$verdict" == "BLOCKED" ]]; then
    for r in "${reasons[@]}"; do
        echo "[safe-fanout] blocker: $r"
    done
fi
echo "[safe-fanout] ticket prompts written to: $TICKET_PROMPT_FILE"
echo "[safe-fanout] dashboard updated: $DASHBOARD"
echo "[safe-fanout] event appended: $EVENTS"

# Exit 0 even when BLOCKED — this script is informational; gates are the
# event/approval state, not the script exit code.
exit 0

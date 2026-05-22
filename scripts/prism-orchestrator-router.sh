#!/usr/bin/env bash
# scripts/prism-orchestrator-router.sh
#
# Active orchestration layer. Reads .prism_orchestration/EVENTS.ndjson and
# routes events to agent inboxes. Idempotent via router_state.txt.
#
# SAFETY INVARIANTS (also documented in RUNBOOK.yaml):
#   - NEVER edits files outside .prism_orchestration/
#   - NEVER spawns agents
#   - NEVER auto-launches F1 / WHILE / ASC / ClusterEvent / site_materialization
#   - NEVER mutates BASELINE.yaml / STATE.yaml / EVENTS.ndjson except for append
#
# Event types handled (see RUNBOOK.yaml event_schemas):
#   GATE_PASS                — emit FANOUT_PREPARED if gate=PATH_A_BASELINE
#   GATE_FAIL                — block fan-out, route to claude inbox
#   AGENT_STARTED            — log to operator inbox
#   AGENT_COMPLETE           — emit CLAUDE_REVIEW_REQUIRED
#   HIGH_RISK_TOUCH_REQUEST  — emit RUNTIME_BLOCKER, route to claude inbox
#   FANOUT_APPROVED          — append routing note to codex inbox (still no launch)
#   RUNTIME_BLOCKER          — route to claude+operator inbox

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
ORCH_DIR="$REPO_ROOT/.prism_orchestration"
EVENTS="$ORCH_DIR/EVENTS.ndjson"
STATE_FILE="$ORCH_DIR/router_state.txt"
APPROVALS="$ORCH_DIR/APPROVALS"
CLAUDE_INBOX="$ORCH_DIR/AGENT_INBOX/claude.md"
CODEX_INBOX="$ORCH_DIR/AGENT_INBOX/codex.md"
OPERATOR_INBOX="$ORCH_DIR/AGENT_INBOX/operator.md"

mkdir -p "$ORCH_DIR" "$APPROVALS" "$ORCH_DIR/AGENT_INBOX" \
    "$ORCH_DIR/AGENT_OUTBOX/claude" "$ORCH_DIR/AGENT_OUTBOX/codex"

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

json_escape() {
    sed 's/\\/\\\\/g; s/"/\\"/g' <<<"${1:-}"
}

emit_event() {
    # Append a single ndjson event line to EVENTS. Caller provides full JSON.
    printf '%s\n' "$1" >> "$EVENTS"
}

append_once() {
    # Append text to a markdown file only if marker substring is not already present.
    local file="$1"
    local marker="$2"
    local text="$3"
    if ! rg -q --fixed-strings "$marker" "$file" 2>/dev/null; then
        {
            echo
            echo "$text"
        } >> "$file"
    fi
}

approval_active() {
    local marker="$1"
    [[ -f "$marker" ]] && grep -Eq '^approved:[[:space:]]*true[[:space:]]*$' "$marker"
}

already_emitted() {
    # Check whether an event with given event-name and gate (or ticket) has already
    # been emitted. Cheap idempotence guard for router-emitted echo events.
    local event_name="$1"
    local key_field="$2"   # e.g. gate or ticket
    local key_value="$3"
    rg -q "\"event\":\"${event_name}\".*\"${key_field}\":\"${key_value}\"" "$EVENTS" 2>/dev/null
}

extract_field() {
    # Field extraction without jq. Reads single ndjson line on stdin, prints field value.
    local field="$1"
    sed -n "s/.*\"${field}\"[[:space:]]*:[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p"
}

# Determine which event lines are new since last router pass.
LAST_PROCESSED=0
if [[ -f "$STATE_FILE" ]]; then
    LAST_PROCESSED="$(awk -F'=' '$1=="last_line"{print $2}' "$STATE_FILE" | tail -1)"
    LAST_PROCESSED="${LAST_PROCESSED:-0}"
fi

if [[ ! -f "$EVENTS" ]]; then
    echo "[router] no events file yet; nothing to do"
    exit 0
fi

CURRENT_LINES="$(wc -l < "$EVENTS" | tr -d ' ')"
NEW_FROM=$((LAST_PROCESSED + 1))

# Defensive: if events file was truncated (we never want this, but handle it),
# replay from the beginning rather than skip.
if (( CURRENT_LINES < LAST_PROCESSED )); then
    NEW_FROM=1
fi

if (( CURRENT_LINES < NEW_FROM )); then
    echo "[router] no new events since line $LAST_PROCESSED"
    exit 0
fi

routed_count=0

while IFS= read -r line; do
    [[ -z "$line" ]] && continue

    event="$(printf '%s\n' "$line" | extract_field event)"
    gate="$(printf '%s\n' "$line" | extract_field gate)"
    agent="$(printf '%s\n' "$line" | extract_field agent)"
    ticket="$(printf '%s\n' "$line" | extract_field ticket)"
    status="$(printf '%s\n' "$line" | extract_field status)"
    reason="$(printf '%s\n' "$line" | extract_field reason)"
    file_field="$(printf '%s\n' "$line" | extract_field file)"

    case "$event" in
        GATE_PASS)
            if [[ "$gate" == "PATH_A_BASELINE" ]] && ! already_emitted "FANOUT_PREPARED" "gate" "PATH_A_BASELINE"; then
                append_once "$CODEX_INBOX" "FANOUT_PREPARED PATH_A_BASELINE" \
"## FANOUT_PREPARED — Path A baseline gate passed

Gate: PATH_A_BASELINE
Status: PASS

Codex fan-out lanes are PREPARED but NOT launched. The router does not
auto-launch.

Required next step: operator places the approval file at
\`.prism_orchestration/APPROVALS/FANOUT_APPROVED\` with content
\`approved: true\` (see APPROVALS/README.md for the full template).

Until then all candidate lanes (DAG-SCHEMA-001, CONTROLTRACE-001,
F2-SCOUT-001, TEST-AUDIT-RUNNER-001) remain WAITING_FOR_OPERATOR_GO.

After approval, only the directive's safe-lane subset may begin. F1,
WHILE, ASC runtime changes, ClusterEvent touching \`nhs_rt_full.rs\`,
and site materialization remain blocked regardless of FANOUT_APPROVED."

                emit_event "{\"ts\":\"$(ts)\",\"agent\":\"router\",\"event\":\"FANOUT_PREPARED\",\"gate\":\"PATH_A_BASELINE\",\"status\":\"PENDING_APPROVAL\",\"approval_path\":\".prism_orchestration/APPROVALS/FANOUT_APPROVED\"}"

                if approval_active "$APPROVALS/FANOUT_APPROVED"; then
                    append_once "$CODEX_INBOX" "FANOUT_APPROVED PATH_A_BASELINE detected" \
"## FANOUT_APPROVED — Codex lanes may begin

The operator has placed an active FANOUT_APPROVED approval. Codex Lead
may launch the approved safe-lane subset:

- F2-SCOUT-001 (Codex-A, read-only)
- DAG-SCHEMA-001 (Codex-B)
- CONTROLTRACE-001 (Codex-C)
- TEST-AUDIT-RUNNER-001 (Codex-H)

Reminder: F1, WHILE, ASC runtime changes, ClusterEvent touching
\`nhs_rt_full.rs\`, and site materialization remain blocked. Any agent
attempting these MUST emit HIGH_RISK_TOUCH_REQUEST first and wait for
Claude approval (drops APPROVALS/HIGH_RISK_APPROVED) before staging.

Each agent MUST emit AGENT_STARTED with its ticket id when beginning."
                fi
                routed_count=$((routed_count + 1))
            fi
            ;;

        GATE_FAIL)
            if ! already_emitted "RUNTIME_BLOCKER" "reason" "GATE_FAIL gate=${gate:-unknown}"; then
                append_once "$CLAUDE_INBOX" "GATE_FAIL ${gate:-unknown}" \
"## GATE_FAIL — gate=${gate:-unknown}

Status: FAIL
Reason: ${reason:-(unspecified)}

Implementation fan-out is blocked. Claude (gatekeeper) must investigate
and clear the gate before any Codex lane begins."

                emit_event "{\"ts\":\"$(ts)\",\"agent\":\"router\",\"event\":\"RUNTIME_BLOCKER\",\"reason\":\"GATE_FAIL gate=${gate:-unknown}\",\"source_event\":\"GATE_FAIL\"}"
                routed_count=$((routed_count + 1))
            fi
            ;;

        AGENT_STARTED)
            append_once "$OPERATOR_INBOX" "AGENT_STARTED ${agent:-?}/${ticket:-?}" \
"## AGENT_STARTED — ${agent:-unknown} / ${ticket:-unknown}

An implementation agent has started its assigned ticket. Operator
visibility only — no action required unless the ticket touches red-lane
files."
            routed_count=$((routed_count + 1))
            ;;

        AGENT_COMPLETE)
            if ! already_emitted "CLAUDE_REVIEW_REQUIRED" "ticket" "${ticket:-unknown}"; then
                append_once "$CLAUDE_INBOX" "AGENT_COMPLETE ${agent:-?}/${ticket:-?}" \
"## CLAUDE_REVIEW_REQUIRED — ${agent:-unknown} / ${ticket:-unknown}

An implementation agent has completed its ticket. Claude (gatekeeper)
review required before merge:

- Ticket: ${ticket:-unknown}
- Agent: ${agent:-unknown}
- Status: ${status:-unknown}

Apply the Codex review rubric (10 fields). Reject if a high-risk file
was touched without active APPROVALS/HIGH_RISK_APPROVED, runtime Python
introduced, deferred drains hidden, fake sites emitted, or CaptureGuard
removed."

                emit_event "{\"ts\":\"$(ts)\",\"agent\":\"router\",\"event\":\"CLAUDE_REVIEW_REQUIRED\",\"ticket\":\"${ticket:-unknown}\",\"upstream_agent\":\"${agent:-unknown}\"}"
                routed_count=$((routed_count + 1))
            fi
            ;;

        HIGH_RISK_TOUCH_REQUEST)
            if ! already_emitted "RUNTIME_BLOCKER" "reason" "HIGH_RISK_TOUCH_REQUEST agent=${agent:-unknown}"; then
                append_once "$CLAUDE_INBOX" "HIGH_RISK_TOUCH_REQUEST ${agent:-?}/${file_field:-?}" \
"## HIGH_RISK_TOUCH_REQUEST — ${agent:-unknown}

An agent is requesting permission to touch a high-risk runtime file:
- Agent: ${agent:-unknown}
- File:  ${file_field:-(unspecified)}
- Reason: ${reason:-(unspecified)}

Claude (gatekeeper) must:
1. Inspect the requested file path against the red-lane list (lanes.red
   in STATE.yaml).
2. Verify the request matches a directive-approved commit in §6.
3. Either: write APPROVALS/HIGH_RISK_APPROVED with 'approved: true', OR
   reject and emit RUNTIME_BLOCKER.

Until explicit approval lands, the agent MUST NOT touch the file."

                emit_event "{\"ts\":\"$(ts)\",\"agent\":\"router\",\"event\":\"RUNTIME_BLOCKER\",\"reason\":\"HIGH_RISK_TOUCH_REQUEST agent=${agent:-unknown}\",\"source_event\":\"HIGH_RISK_TOUCH_REQUEST\"}"
                routed_count=$((routed_count + 1))
            fi
            ;;

        FANOUT_APPROVED)
            append_once "$CODEX_INBOX" "FANOUT_APPROVED via_event" \
"## FANOUT_APPROVED — operator authorization (via event)

The operator has approved fan-out by emitting FANOUT_APPROVED directly.
Codex Lead may launch the approved safe-lane subset (same restrictions
as the file-based approval path)."
            routed_count=$((routed_count + 1))
            ;;

        RUNTIME_BLOCKER)
            append_once "$CLAUDE_INBOX" "RUNTIME_BLOCKER ${reason:-unspecified}" \
"## RUNTIME_BLOCKER

Reason: ${reason:-unspecified}

All Codex implementation lanes are frozen. Claude must clear the
blocker before any work resumes."
            append_once "$OPERATOR_INBOX" "RUNTIME_BLOCKER ${reason:-unspecified}" \
"## RUNTIME_BLOCKER

Reason: ${reason:-unspecified}

Implementation work is frozen. Claude (gatekeeper) is investigating."
            routed_count=$((routed_count + 1))
            ;;

        FANOUT_PREPARED|CLAUDE_REVIEW_REQUIRED|TRIGGER_GATE_PASS|TRIGGER_GATE_FAIL|TRIGGER_CLAUDE_REVIEW|TRIGGER_RUNTIME_BLOCK|TRIGGER_OPERATOR_CHECK|ORCHESTRATION_SCAFFOLD_CREATED)
            # Router-emitted echoes or watcher/pre-commit lifecycle events; ignore.
            ;;

        *)
            # Unknown / informational event; ignore silently.
            ;;
    esac
done < <(sed -n "${NEW_FROM},${CURRENT_LINES}p" "$EVENTS")

# Update state to reflect the LATEST line count (including any router-emitted echoes).
NEW_LINE_COUNT="$(wc -l < "$EVENTS" | tr -d ' ')"
{
    echo "last_line=${NEW_LINE_COUNT}"
    echo "last_run_utc=$(ts)"
    echo "events_routed_this_run=${routed_count}"
} > "$STATE_FILE"

echo "[router] processed $((CURRENT_LINES - NEW_FROM + 1)) new line(s); routed ${routed_count} action(s)"
echo "[router] next run starts at line $((NEW_LINE_COUNT + 1))"

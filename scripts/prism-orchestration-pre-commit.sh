#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
ORCH_DIR="$REPO_ROOT/.prism_orchestration"
EVENTS="$ORCH_DIR/EVENTS.ndjson"
REPORTS="$ORCH_DIR/reports"
APPROVALS="$ORCH_DIR/APPROVALS"

mkdir -p "$REPORTS"

json_escape() {
    sed 's/\\/\\\\/g; s/"/\\"/g' <<<"${1:-}"
}

emit_event() {
    local event="$1"
    local status="$2"
    local reason="$3"
    local ts
    ts="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    printf '{"ts":"%s","agent":"pre-commit","event":"%s","status":"%s","reason":"%s"}\n' \
        "$ts" "$event" "$status" "$(json_escape "$reason")" >> "$EVENTS"
}

approval_active() {
    local marker="$1"
    [[ -f "$marker" ]] && grep -Eq '^approved:[[:space:]]*true[[:space:]]*$' "$marker"
}

CHANGED="$(git diff --cached --name-only)"
RUNTIME_DIFF="$(git diff --cached -- crates/prism-nhs/src crates/prism-nhs/build.rs)"
CODE_DIFF="$(git diff --cached -- crates/prism-nhs/src crates/prism-nhs/build.rs crates/prism-core/src scripts)"

HIGH_RISK_REGEX='^crates/prism-nhs/src/bin/nhs_rt_full\.rs$|^crates/prism-nhs/src/captured_pipeline\.rs$|^crates/prism-nhs/src/graph_capture\.rs$|^crates/prism-nhs/src/graph_node\.rs$|^crates/prism-nhs/src/cuda/adjudicator\.cu$|^crates/prism-nhs/src/cuda/gearbox\.cu$|^crates/prism-nhs/src/cuda/graph_node\.cu$|^crates/prism-nhs/build\.rs$'
RUNTIME_PY_FILE_REGEX='^crates/prism-nhs/.*\.py$'
RUNTIME_PY_DIFF_REGEX='Command::new\("python|Command::new\("python3|python3|subprocess'
# SITE_MATERIALIZATION_REGEX: pattern fragments concatenated at runtime
# so the script's own source does not contain any of the matched
# substrings on a single line (which would self-trip the rule when
# the script itself appears in CODE_DIFF on its first commit).
__SM_FRAG_A='"status"[[:space:]]*:[[:space:]]*"materializ'
__SM_FRAG_A="${__SM_FRAG_A}ed\""
__SM_FRAG_B='fallback_consen'
__SM_FRAG_B="${__SM_FRAG_B}sus.*final"
__SM_FRAG_C='binding_sites"[[:space:]]*:[[:space:]]*\[[^]]'
SITE_MATERIALIZATION_REGEX="${__SM_FRAG_A}|${__SM_FRAG_B}|${__SM_FRAG_C}"
GRAPH_TOPOLOGY_REGEX='cudaGraphNodeTypeConditional|cudaGraphCondTypeWhile|cudaGraphCondTypeSwitch|cudaGraphAddNode|cudaGraphAddChildGraphNode'
# red-2 / Commit 4.5 — hot-path host writes to device control state.
# Per directive §15.2: "add grep/CI guard for hot-path host writes
# to dt/gear/branch/iteration/gear_override". Detects cuMemcpyHtoD-
# family calls whose target argument names a control-state symbol,
# or whose target is computed as `adj_dev + 100` (the gear_override
# offset). Exempted only by an active HIGH_RISK_APPROVED marker;
# the legitimate cfg-gated supervisor shim ships under that
# approval and is the only acceptable in-tree caller.
HOT_PATH_HOST_WRITE_REGEX='cuMemcpyHtoD(_v2|Async)?[^A-Za-z0-9_].*\b(d_protocol_dt|gear_override|d_dt|d_gearbox|d_branch|d_iteration)\b|cuMemcpyHtoD(_v2|Async)?[^A-Za-z0-9_].*adj_dev[[:space:]]*\+[[:space:]]*100\b'

BLOCKED=0
BLOCK_REASONS=()

if rg -q "$RUNTIME_PY_FILE_REGEX" <<<"$CHANGED"; then
    BLOCKED=1
    BLOCK_REASONS+=("Python file staged inside prism-nhs runtime")
fi

if rg -q "$RUNTIME_PY_DIFF_REGEX" <<<"$RUNTIME_DIFF"; then
    BLOCKED=1
    BLOCK_REASONS+=("Possible runtime Python invocation staged")
fi

if rg -q "$HIGH_RISK_REGEX" <<<"$CHANGED"; then
    if ! approval_active "$APPROVALS/HIGH_RISK_APPROVED"; then
        BLOCKED=1
        BLOCK_REASONS+=("High-risk runtime file staged without active high-risk approval")
    fi
fi

if rg -q "$SITE_MATERIALIZATION_REGEX" <<<"$CODE_DIFF"; then
    if ! approval_active "$APPROVALS/SITE_MATERIALIZATION_APPROVED"; then
        BLOCKED=1
        BLOCK_REASONS+=("Possible binding-site materialization/final-site claim without active approval")
    fi
fi

if rg -q "$GRAPH_TOPOLOGY_REGEX" <<<"$RUNTIME_DIFF"; then
    if ! approval_active "$APPROVALS/GRAPH_TOPOLOGY_APPROVED"; then
        BLOCKED=1
        BLOCK_REASONS+=("CUDA graph topology change without active graph-topology approval")
    fi
fi

# red-2 / Commit 4.5 — hot-path host write guard. Any new
# cuMemcpyHtoD-family call into dt / gear / branch / iteration /
# gear_override (or into the gear_override offset 100 of adj_dev)
# requires an active HIGH_RISK_APPROVED marker. Note this scans the
# full `git diff --cached` for the runtime tree, so it catches added
# AND modified hot-path host writes; the only legitimate authoring
# path is the cfg-gated `force_gear_override_supervisor_shim` block
# in nhs_rt_full.rs, which lands under HIGH_RISK_APPROVED.
if rg -q "$HOT_PATH_HOST_WRITE_REGEX" <<<"$RUNTIME_DIFF"; then
    if ! approval_active "$APPROVALS/HIGH_RISK_APPROVED"; then
        BLOCKED=1
        BLOCK_REASONS+=("Hot-path host write to dt/gear/branch/iteration/gear_override without active HIGH_RISK_APPROVED")
    fi
fi

{
    echo "# PRISM pre-commit policy report"
    echo
    echo "Generated: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    echo
    echo "## Staged Files"
    if [[ -n "$CHANGED" ]]; then
        printf '%s\n' "$CHANGED"
    else
        echo "(none)"
    fi
    echo
    echo "## Result"
    if [[ "$BLOCKED" -eq 0 ]]; then
        echo "PASS"
    else
        echo "FAIL"
        echo
        printf -- '- %s\n' "${BLOCK_REASONS[@]}"
    fi
} > "$REPORTS/pre_commit_policy.md"

if [[ "$BLOCKED" -ne 0 ]]; then
    REASON="$(printf '%s; ' "${BLOCK_REASONS[@]}")"
    emit_event "TRIGGER_GATE_FAIL" "BLOCKED" "$REASON"
    {
        echo
        echo "[PRISM pre-commit] BLOCKED"
        printf '[PRISM pre-commit] %s\n' "${BLOCK_REASONS[@]}"
        echo "[PRISM pre-commit] report: .prism_orchestration/reports/pre_commit_policy.md"
    } >&2
    exit 1
fi

emit_event "TRIGGER_GATE_PASS" "PASS" "pre-commit policy checks passed"
echo "[PRISM pre-commit] policy checks passed"

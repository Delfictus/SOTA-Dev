#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
ORCH_DIR="$REPO_ROOT/.prism_orchestration"
EVENTS="$ORCH_DIR/EVENTS.ndjson"
REPORTS="$ORCH_DIR/reports"
CLAUDE_INBOX="$ORCH_DIR/AGENT_INBOX/claude.md"
OPERATOR_INBOX="$ORCH_DIR/AGENT_INBOX/operator.md"
BASELINE="$ORCH_DIR/BASELINE.yaml"

INTERVAL_SECONDS="${PRISM_WATCH_INTERVAL_SECONDS:-10}"
HEARTBEAT_TIMEOUT_SECONDS="${PRISM_HEARTBEAT_TIMEOUT_SECONDS:-1200}"
ONCE=false
EXPLICIT_LOGS=()

for arg in "$@"; do
    case "$arg" in
        --once)
            ONCE=true
            ;;
        *)
            EXPLICIT_LOGS+=("$arg")
            ;;
    esac
done

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
    printf '{"ts":"%s","agent":"watcher","event":"%s","status":"%s","reason":"%s"}\n' \
        "$ts" "$event" "$status" "$(json_escape "$reason")" >> "$EVENTS"
}

append_once() {
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

yaml_scalar() {
    local key="$1"
    local file="$2"
    awk -F':[[:space:]]*' -v key="$key" '$1 == key {print $2; exit}' "$file" 2>/dev/null \
        | sed 's/^"//; s/"$//'
}

active_run_dir() {
    local value
    value="$(yaml_scalar "active_run_dir" "$BASELINE")"
    if [[ -z "$value" || "$value" == "null" ]]; then
        return 0
    fi
    printf '%s\n' "$value"
}

runtime_logs_to_scan() {
    if (( ${#EXPLICIT_LOGS[@]} > 0 )); then
        printf '%s\n' "${EXPLICIT_LOGS[@]}"
        return 0
    fi

    local run_dir
    run_dir="$(active_run_dir)"
    if [[ -n "$run_dir" && -d "$run_dir" ]]; then
        find "$run_dir" -type f -name 'run.log' -print
    fi
}

snapshot_and_trigger() {
    cd "$REPO_ROOT"

    git status --short > "$REPORTS/git_status.txt"

    local high_risk
    high_risk="$(git status --short -- \
        crates/prism-nhs/src/bin/nhs_rt_full.rs \
        crates/prism-nhs/src/captured_pipeline.rs \
        crates/prism-nhs/src/graph_capture.rs \
        crates/prism-nhs/src/graph_node.rs \
        crates/prism-nhs/src/cuda/adjudicator.cu \
        crates/prism-nhs/src/cuda/gearbox.cu \
        crates/prism-nhs/src/cuda/graph_node.cu \
        crates/prism-nhs/build.rs || true)"

    printf '%s\n' "$high_risk" > "$REPORTS/high_risk_dirty.txt"

    local last_high_risk="$REPORTS/last_high_risk_dirty.txt"
    if [[ -n "$high_risk" ]] && ! cmp -s "$REPORTS/high_risk_dirty.txt" "$last_high_risk" 2>/dev/null; then
        emit_event "TRIGGER_CLAUDE_REVIEW" "BLOCKED" "high-risk dirty file detected"
        append_once "$CLAUDE_INBOX" "WATCHER: high-risk dirty file detected" "## WATCHER: high-risk dirty file detected

Review .prism_orchestration/reports/high_risk_dirty.txt before approving any red-lane work."
    fi
    cp "$REPORTS/high_risk_dirty.txt" "$last_high_risk"

    local cuda_hits="$REPORTS/active_run_cuda_failure_hits.txt"
    local scanned_logs="$REPORTS/active_run_logs_scanned.txt"
    : > "$cuda_hits"
    : > "$scanned_logs"

    mapfile -t logs_to_scan < <(runtime_logs_to_scan)
    if (( ${#logs_to_scan[@]} > 0 )); then
        printf '%s\n' "${logs_to_scan[@]}" > "$scanned_logs"
        rg -n "CUDA_ERROR|rc=801|rc=900|rc=901|STREAM_CAPTURE_INVALIDATED" "${logs_to_scan[@]}" \
            > "$cuda_hits" 2>/dev/null || true
    fi

    local last_cuda="$REPORTS/last_active_run_cuda_failure_hits.txt"
    if [[ -s "$cuda_hits" ]] && ! cmp -s "$cuda_hits" "$last_cuda" 2>/dev/null; then
        emit_event "TRIGGER_RUNTIME_BLOCK" "BLOCKED" "CUDA failure detected in active run or explicit log"
        append_once "$CLAUDE_INBOX" "WATCHER: active-run CUDA failure detected" "## WATCHER: active-run CUDA failure detected

Runtime fan-out should freeze. Inspect .prism_orchestration/reports/active_run_cuda_failure_hits.txt."
    fi
    cp "$cuda_hits" "$last_cuda"

    if [[ -f "$EVENTS" ]]; then
        local last_heartbeat_epoch
        last_heartbeat_epoch="$(rg '"heartbeat"' "$EVENTS" | tail -1 | sed -n 's/.*"heartbeat":"\([^"]*\)".*/\1/p' | xargs -r -I{} date -u -d {} +%s 2>/dev/null || true)"
        if [[ -n "$last_heartbeat_epoch" ]]; then
            local now_epoch
            now_epoch="$(date -u +%s)"
            if (( now_epoch - last_heartbeat_epoch > HEARTBEAT_TIMEOUT_SECONDS )); then
                local stamp="$REPORTS/heartbeat_stale_reported"
                if [[ ! -f "$stamp" ]]; then
                    emit_event "TRIGGER_OPERATOR_CHECK" "STALE" "agent heartbeat older than threshold"
                    append_once "$OPERATOR_INBOX" "WATCHER: stale heartbeat" "## WATCHER: stale heartbeat

No agent heartbeat has been observed within the configured threshold."
                    date -u +'%Y-%m-%dT%H:%M:%SZ' > "$stamp"
                fi
            else
                rm -f "$REPORTS/heartbeat_stale_reported"
            fi
        fi
    fi
}

if [[ "$ONCE" == "true" ]]; then
    echo "[PRISM watcher] one-shot scan"
    snapshot_and_trigger
    echo "[PRISM watcher] one-shot scan complete"
    exit 0
fi

echo "[PRISM watcher] monitoring every ${INTERVAL_SECONDS}s; Ctrl-C to stop"
while true; do
    snapshot_and_trigger
    sleep "$INTERVAL_SECONDS"
done

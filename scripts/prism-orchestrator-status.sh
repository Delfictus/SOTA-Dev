#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
ORCH_DIR="$REPO_ROOT/.prism_orchestration"
REPORTS="$ORCH_DIR/reports"
DASHBOARD="$ORCH_DIR/DASHBOARD.md"
BASELINE="$ORCH_DIR/BASELINE.yaml"

mkdir -p "$REPORTS"

cd "$REPO_ROOT"

TS="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
CUDA_PATTERN="CUDA_ERROR|rc=801|rc=900|rc=901|STREAM_CAPTURE_INVALIDATED"

yaml_scalar() {
    local key="$1"
    local file="$2"
    awk -F':[[:space:]]*' -v key="$key" '$1 == key {print $2; exit}' "$file" 2>/dev/null \
        | sed 's/^"//; s/"$//'
}

yaml_list() {
    local key="$1"
    local file="$2"
    awk -v key="$key" '
        $0 == key ":" {in_list=1; next}
        in_list && /^  - / {sub(/^  - /, ""); gsub(/^"|"$/, ""); print; next}
        in_list && /^[^[:space:]]/ {in_list=0}
    ' "$file" 2>/dev/null
}

normalized_active_run_dir() {
    local value
    value="$(yaml_scalar "active_run_dir" "$BASELINE")"
    if [[ -z "$value" || "$value" == "null" ]]; then
        return 0
    fi
    printf '%s\n' "$value"
}

git status --short > "$REPORTS/git_status.txt"
git log --oneline -5 > "$REPORTS/recent_commits.txt"

git status --short -- \
    crates/prism-nhs/src/bin/nhs_rt_full.rs \
    crates/prism-nhs/src/captured_pipeline.rs \
    crates/prism-nhs/src/graph_capture.rs \
    crates/prism-nhs/src/graph_node.rs \
    crates/prism-nhs/src/cuda/adjudicator.cu \
    crates/prism-nhs/src/cuda/gearbox.cu \
    crates/prism-nhs/src/cuda/graph_node.cu \
    crates/prism-nhs/build.rs > "$REPORTS/high_risk_dirty.txt" || true

git status --short --untracked-files=no -- \
    crates/prism-nhs/src/bin/nhs_rt_full.rs \
    crates/prism-nhs/src/captured_pipeline.rs \
    crates/prism-nhs/src/graph_capture.rs \
    crates/prism-nhs/src/graph_node.rs \
    crates/prism-nhs/src/cuda/adjudicator.cu \
    crates/prism-nhs/src/cuda/gearbox.cu \
    crates/prism-nhs/src/cuda/graph_node.cu \
    crates/prism-nhs/build.rs > "$REPORTS/high_risk_tracked_dirty.txt" || true

awk '/^\?\?/ {print substr($0, 4)}' "$REPORTS/git_status.txt" > "$REPORTS/untracked_operator_files.txt"

awk '
    /^\.prism_orchestration\// {counts["orchestration_scaffold"]++; next}
    /^scripts\// {counts["scripts"]++; next}
    /^docs\// {counts["docs"]++; next}
    /^data\// {counts["data"]++; next}
    /^crates\// {counts["crates"]++; next}
    /\.py$/ {counts["root_python_or_script"]++; next}
    /\.json$/ {counts["json_data"]++; next}
    {counts["other"]++}
    END {
        cats[1]="orchestration_scaffold";
        cats[2]="scripts";
        cats[3]="docs";
        cats[4]="data";
        cats[5]="crates";
        cats[6]="root_python_or_script";
        cats[7]="json_data";
        cats[8]="other";
        for (i=1; i<=8; i++) {
            cat=cats[i];
            if (counts[cat] > 0) {
                print cat ": " counts[cat];
            }
        }
    }
' "$REPORTS/untracked_operator_files.txt" > "$REPORTS/untracked_categories.txt"

ACTIVE_RUN_DIR="$(normalized_active_run_dir)"
ACTIVE_RUN_LABEL="${ACTIVE_RUN_DIR:-none}"
ACTIVE_LOGS="$REPORTS/active_run_logs_scanned.txt"
ACTIVE_HITS="$REPORTS/active_run_cuda_failure_hits.txt"
: > "$ACTIVE_LOGS"
: > "$ACTIVE_HITS"

if [[ -n "$ACTIVE_RUN_DIR" && -d "$ACTIVE_RUN_DIR" ]]; then
    find "$ACTIVE_RUN_DIR" -type f -name 'run.log' -print > "$ACTIVE_LOGS"
    if [[ -s "$ACTIVE_LOGS" ]]; then
        mapfile -t active_logs < "$ACTIVE_LOGS"
        rg -n "$CUDA_PATTERN" "${active_logs[@]}" > "$ACTIVE_HITS" 2>/dev/null || true
    fi
fi

HISTORICAL_LOGS="$REPORTS/historical_logs_scanned.txt"
HISTORICAL_HITS="$REPORTS/historical_cuda_failure_hits.txt"
: > "$HISTORICAL_LOGS"
: > "$HISTORICAL_HITS"

while IFS= read -r root; do
    [[ -z "$root" || ! -d "$root" ]] && continue
    find "$root" -maxdepth 2 -type f -name 'run.log' -print >> "$HISTORICAL_LOGS"
done < <(yaml_list "historical_log_roots" "$BASELINE")

if [[ -s "$HISTORICAL_LOGS" ]]; then
    mapfile -t historical_logs < "$HISTORICAL_LOGS"
    rg -n "$CUDA_PATTERN" "${historical_logs[@]}" > "$HISTORICAL_HITS" 2>/dev/null || true
fi

git grep -n -E "python|python3|subprocess|Command::new" -- crates/prism-nhs/src \
    > "$REPORTS/runtime_python_scan.txt" 2>/dev/null || true

git grep -n -E "deferred" -- crates/prism-nhs/src \
    | head -50 > "$REPORTS/deferred_drain_refs.txt" 2>/dev/null || true

git grep -n -E "cudaGraphNodeTypeConditional|cudaGraphCondTypeSwitch|cudaGraphCondTypeWhile" -- \
    crates/prism-nhs/src crates/prism-nhs/src/cuda \
    > "$REPORTS/conditional_graph_refs.txt" 2>/dev/null || true

{
    echo "# PRISM Orchestration Dashboard"
    echo
    echo "Generated: $TS"
    echo
    echo "## Current Phase"
    echo
    echo "Parent-Control Supergraph Wave 1."
    echo
    echo "## Runtime Baseline"
    echo
    echo "- TIER 8: PASS"
    echo "- TIER 8.1: GREEN with deferred-drain caveat"
    echo "- Baseline commit: \`737b273b\`"
    echo "- G21 hygiene commit: \`d7301243\`"
    echo "- TIER 8 freeze commit: \`8ca26189\`"
    echo "- Runtime log scan mode: \`active_run_only\`"
    echo
    echo "## Active Run"
    echo
    echo "$ACTIVE_RUN_LABEL"
    echo
    echo "## Git Status"
    echo
    if [[ ! -s "$REPORTS/git_status.txt" ]]; then
        echo "Clean."
    else
        STATUS_COUNT="$(wc -l < "$REPORTS/git_status.txt" | tr -d ' ')"
        echo "$STATUS_COUNT changed paths detected."
        echo
        echo "Full status: \`.prism_orchestration/reports/git_status.txt\`"
    fi
    echo
    echo "## High-Risk Tracked Dirty Files"
    echo
    if [[ -s "$REPORTS/high_risk_tracked_dirty.txt" ]]; then
        sed 's/^/- `/' "$REPORTS/high_risk_tracked_dirty.txt" | sed 's/$/`/'
    else
        echo "None."
    fi
    echo
    echo "## Untracked Operator Files"
    echo
    UNTRACKED_COUNT="$(wc -l < "$REPORTS/untracked_operator_files.txt" | tr -d ' ')"
    echo "$UNTRACKED_COUNT untracked paths detected. Non-blocking unless assigned by a ticket."
    if [[ -s "$REPORTS/untracked_categories.txt" ]]; then
        sed 's/^/- `/' "$REPORTS/untracked_categories.txt" | sed 's/$/`/'
    fi
    echo
    echo "## Runtime Log Findings"
    echo
    ACTIVE_HIT_COUNT="$(wc -l < "$ACTIVE_HITS" | tr -d ' ')"
    HISTORICAL_HIT_COUNT="$(wc -l < "$HISTORICAL_HITS" | tr -d ' ')"
    echo "- Runtime blockers from active run: $ACTIVE_HIT_COUNT"
    echo "- Active-run report: \`.prism_orchestration/reports/active_run_cuda_failure_hits.txt\`"
    echo "- Historical blockers ignored: $HISTORICAL_HIT_COUNT"
    echo "- Historical report: \`.prism_orchestration/reports/historical_cuda_failure_hits.txt\`"
    if [[ "$ACTIVE_HIT_COUNT" == "0" ]]; then
        echo "- Blocking status: no active-run runtime block"
    else
        echo "- Blocking status: active-run runtime block present"
    fi
    echo
    echo "## Active Tickets"
    echo
    echo "None. Implementation fan-out is waiting for operator GO."
    echo
    echo "## Candidate Tickets"
    echo
    echo "- DAG-SCHEMA-001 — Codex-B — WAITING_FOR_OPERATOR_GO"
    echo "- CONTROLTRACE-001 — Codex-C — WAITING_FOR_OPERATOR_GO"
    echo "- F2-SCOUT-001 — Codex-A — WAITING_FOR_OPERATOR_GO, read-only"
    echo "- TEST-AUDIT-RUNNER-001 — Codex-H — WAITING_FOR_OPERATOR_GO"
    echo
    echo "## Last Gate"
    echo
    echo "G0_RUNTIME_BASELINE: PASS."
    echo
    echo "## Recent Events"
    echo
    echo "Events are append-only. Runtime-block events emitted before baseline-aware active-run scanning are historical unless active-run blockers above are nonzero."
    echo
    if [[ -s "$ORCH_DIR/EVENTS.ndjson" ]]; then
        tail -5 "$ORCH_DIR/EVENTS.ndjson" | sed 's/^/- `/' | sed 's/$/`/'
    else
        echo "No events recorded."
    fi
    echo
    echo "## Next Operator Decision"
    echo
    echo "Hold implementation tickets. Have Claude reconcile tracked high-risk dirty files against the sealed baseline/current Claude state."
} > "$DASHBOARD"

cat <<EOF
=== PRISM ORCHESTRATION STATUS ===
$TS

Dashboard: .prism_orchestration/DASHBOARD.md
Reports:
  .prism_orchestration/reports/git_status.txt
  .prism_orchestration/reports/high_risk_tracked_dirty.txt
  .prism_orchestration/reports/untracked_categories.txt
  .prism_orchestration/reports/active_run_cuda_failure_hits.txt
  .prism_orchestration/reports/historical_cuda_failure_hits.txt
  .prism_orchestration/reports/runtime_python_scan.txt
  .prism_orchestration/reports/deferred_drain_refs.txt
  .prism_orchestration/reports/conditional_graph_refs.txt
EOF

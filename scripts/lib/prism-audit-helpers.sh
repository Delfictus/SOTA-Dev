#!/usr/bin/env bash
# Shared helpers for PRISM-4D test/audit command runners.
#
# This file is intentionally inert when sourced. Callers opt in to execution.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "prism-audit-helpers.sh is a library; source it from a runner script." >&2
    exit 2
fi

prism_audit_repo_root() {
    local helper_dir
    helper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "$helper_dir/../.." && pwd
}

prism_audit_timestamp() {
    date +%Y%m%d_%H%M%S
}

prism_audit_print_header() {
    local title="$1"
    printf '\n== %s ==\n\n' "$title"
}

prism_audit_instantiation_pattern() {
    printf '%s\n' 'V2-INSTANTIATE-COMPLETE stream|MONO-FUSE stream .*monolithic exec instantiated'
}

prism_audit_cuda_failure_pattern() {
    printf '%s\n' 'CUDA_ERROR|rc=801|rc=900|rc=901|STREAM_CAPTURE|STREAM_CAPTURE_INVALIDATED|INVALID_VALUE|NOT_SUPPORTED|NotSupported|not supported'
}

prism_audit_conditional_graph_pattern() {
    printf '%s\n' 'child_conditional_nodes|parent_conditional_nodes'
}

prism_audit_deferred_drain_pattern() {
    printf '%s\n' 'TIER8-CAPTURE deferred-error-state|TIER8-CAPTURE deferred-summary|deferred-summary|deferred_drain_count|post_t7_sync|post-gasp-sync|subsequent_build_succeeded|failure_within_3_milestones'
}

prism_audit_deferred_drain_risk_pattern() {
    printf '%s\n' 'deferred-summary.*subsequent_build_succeeded=false|deferred-summary.*failure_within_3_milestones=true'
}

prism_audit_log_milestone_pattern() {
    printf '%s\n' 'V2-INSTANTIATE-COMPLETE stream|MONO-FUSE stream .*waiting|MONO-FUSE stream .*acquired|MONO-FUSE stream .*monolithic exec instantiated|TIER8-CAPTURE deferred-error-state|TIER8-CAPTURE deferred-summary|TIER8-DIAG post-gasp-sync|post_t7_sync|G21|capture-guard|build failed|MULTI-STREAM PIPELINE COMPLETE'
}

prism_audit_high_risk_paths() {
    cat <<'EOF'
crates/prism-nhs/src/bin/nhs_rt_full.rs
crates/prism-nhs/src/captured_pipeline.rs
crates/prism-nhs/src/graph_capture.rs
crates/prism-nhs/src/graph_node.rs
crates/prism-nhs/src/cuda/adjudicator.cu
crates/prism-nhs/src/cuda/gearbox.cu
crates/prism-nhs/src/cuda/graph_node.cu
crates/prism-nhs/build.rs
EOF
}

prism_audit_require_file_arg() {
    local path="${1:-}"
    local label="${2:-file}"

    if [[ -z "$path" ]]; then
        echo "ERROR: missing $label path" >&2
        return 2
    fi
    if [[ ! -f "$path" ]]; then
        echo "ERROR: $label not found: $path" >&2
        return 1
    fi
}

prism_audit_scan_instantiation_milestones() {
    local run_log="$1"

    prism_audit_require_file_arg "$run_log" "run log" || return $?
    rg -n "$(prism_audit_instantiation_pattern)" "$run_log"
}

prism_audit_scan_cuda_failures() {
    local run_log="$1"

    prism_audit_require_file_arg "$run_log" "run log" || return $?
    rg -n "$(prism_audit_cuda_failure_pattern)" "$run_log" || true
}

prism_audit_scan_conditional_graph_refs() {
    local run_log="$1"

    prism_audit_require_file_arg "$run_log" "run log" || return $?
    rg -n "$(prism_audit_conditional_graph_pattern)" "$run_log"
}

prism_audit_scan_deferred_drains() {
    local run_log="$1"

    prism_audit_require_file_arg "$run_log" "run log" || return $?
    rg -n "$(prism_audit_deferred_drain_pattern)" "$run_log" || true
}

prism_audit_scan_deferred_drain_risks() {
    local run_log="$1"

    prism_audit_require_file_arg "$run_log" "run log" || return $?
    rg -n "$(prism_audit_deferred_drain_risk_pattern)" "$run_log" || true
}

prism_audit_extract_log_milestones() {
    local run_log="$1"

    prism_audit_require_file_arg "$run_log" "run log" || return $?
    rg -n "$(prism_audit_log_milestone_pattern)" "$run_log" || true
}

prism_audit_extract_log_milestone_window() {
    local run_log="${1:-}"
    local anchor_line="${2:-}"
    local radius="${3:-3}"

    prism_audit_require_file_arg "$run_log" "run log" || return $?
    if [[ ! "$anchor_line" =~ ^[0-9]+$ ]]; then
        echo "ERROR: anchor line must be a positive integer" >&2
        return 2
    fi
    if [[ ! "$radius" =~ ^[0-9]+$ ]]; then
        echo "ERROR: radius must be a non-negative integer" >&2
        return 2
    fi

    prism_audit_extract_log_milestones "$run_log" | awk -F: -v anchor="$anchor_line" -v radius="$radius" '
        $1 ~ /^[0-9]+$/ {
            rows[++n] = $0
            if ($1 <= anchor) {
                before = n
            }
            if ($1 >= anchor && after == 0) {
                after = n
            }
        }
        END {
            if (n == 0) {
                exit 0
            }
            center = after ? after : before
            if (center == 0) {
                center = 1
            }
            start = center - radius
            stop = center + radius
            if (start < 1) {
                start = 1
            }
            if (stop > n) {
                stop = n
            }
            for (i = start; i <= stop; i++) {
                print rows[i]
            }
        }
    '
}

prism_audit_scan_high_risk_dirty() {
    local repo_root="${1:-}"
    local -a high_risk_paths=()

    if [[ -z "$repo_root" ]]; then
        repo_root="$(prism_audit_repo_root)"
    fi
    prism_audit_require_repo_root "$repo_root" || return $?

    mapfile -t high_risk_paths < <(prism_audit_high_risk_paths)
    git -C "$repo_root" status --short -- "${high_risk_paths[@]}"
}

prism_audit_assert_no_high_risk_dirty() {
    local repo_root="${1:-}"
    local dirty

    dirty="$(prism_audit_scan_high_risk_dirty "$repo_root")"
    if [[ -n "$dirty" ]]; then
        printf '%s\n' "$dirty"
        return 1
    fi
}

prism_audit_check_artifacts() {
    local artifact_root="${1:-}"
    local status=0
    local spec candidate_pattern artifact size
    local -a matches=()

    if [[ -z "$artifact_root" ]]; then
        echo "ERROR: artifact root is required" >&2
        return 2
    fi
    if [[ ! -d "$artifact_root" ]]; then
        echo "ERROR: artifact root not found: $artifact_root" >&2
        return 1
    fi
    shift || true
    if [[ $# -eq 0 ]]; then
        set -- run.log
    fi

    for spec in "$@"; do
        matches=()
        if [[ "$spec" == /* ]]; then
            candidate_pattern="$spec"
        else
            candidate_pattern="$artifact_root/$spec"
        fi

        if [[ "$spec" == *[\*\?\[]* ]]; then
            mapfile -t matches < <(compgen -G "$candidate_pattern" || true)
        else
            matches=("$candidate_pattern")
        fi

        if [[ ${#matches[@]} -eq 0 ]]; then
            printf 'MISSING artifact %s\n' "$spec"
            status=1
            continue
        fi

        for artifact in "${matches[@]}"; do
            if [[ ! -e "$artifact" ]]; then
                printf 'MISSING artifact %s\n' "$artifact"
                status=1
            elif [[ -d "$artifact" ]]; then
                printf 'OK artifact %s directory\n' "$artifact"
            elif [[ -s "$artifact" ]]; then
                size="$(wc -c < "$artifact" | tr -d '[:space:]')"
                printf 'OK artifact %s bytes=%s\n' "$artifact" "$size"
            else
                printf 'EMPTY artifact %s bytes=0\n' "$artifact"
                status=1
            fi
        done
    done

    return "$status"
}

prism_audit_print_build_commands() {
    cat <<'EOF'
cargo check -p prism-core
cargo check -p prism-nhs --bin nhs_rt_full --features=v2_ignition
cargo build --release -p prism-nhs --bin nhs_rt_full --features=v2_ignition
EOF
}

prism_audit_print_tier81_smoke_command() {
    cat <<'EOF'
OUT=/mnt/storage/prism_tier8_1_smoke_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"

timeout --signal=INT --kill-after=30s 600s scripts/prism-validate-and-run.sh   -t data/targets/mpro_monomer.topology.json -o "$OUT"   --fast --hysteresis --prism-therm --multi-stream 8 --spike-percentile 70   --fused-steps 6 --hmr --adaptive-dt --multi-differential   --closed-loop-steering --asymmetric-steering --m1-monolithic-discovery   --no-autonomous-rescue --replica-seed 42 -v 2>&1 | tee "$OUT/run.log"
EOF
}

prism_audit_print_all8_gate_command() {
    cat <<'EOF'
OUT=/mnt/storage/prism_tier8_all8_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"

timeout --signal=INT --kill-after=30s 4800s scripts/prism-validate-and-run.sh   -t data/targets/mpro_monomer.topology.json -o "$OUT"   --fast --hysteresis --prism-therm --multi-stream 8 --spike-percentile 70   --fused-steps 6 --hmr --adaptive-dt --multi-differential   --closed-loop-steering --asymmetric-steering --m1-monolithic-discovery   --no-autonomous-rescue --replica-seed 42 -v 2>&1 | tee "$OUT/run.log"
EOF
}

prism_audit_print_log_scan_commands() {
    printf 'rg -n "%s" "$OUT/run.log"\n' "$(prism_audit_instantiation_pattern)"
    printf 'rg -n "%s" "$OUT/run.log" || true\n' "$(prism_audit_cuda_failure_pattern)"
    printf 'rg -n "%s" "$OUT/run.log"\n' "$(prism_audit_conditional_graph_pattern)"
    printf 'rg -n "%s" "$OUT/run.log" || true\n' "$(prism_audit_deferred_drain_pattern)"
    printf 'rg -n "%s" "$OUT/run.log" || true\n' "$(prism_audit_log_milestone_pattern)"
    cat <<'EOF'
git status --short -- \
    crates/prism-nhs/src/bin/nhs_rt_full.rs \
    crates/prism-nhs/src/captured_pipeline.rs \
    crates/prism-nhs/src/graph_capture.rs \
    crates/prism-nhs/src/graph_node.rs \
    crates/prism-nhs/src/cuda/adjudicator.cu \
    crates/prism-nhs/src/cuda/gearbox.cu \
    crates/prism-nhs/src/cuda/graph_node.cu \
    crates/prism-nhs/build.rs
# Optional artifact check after a run:
# source scripts/lib/prism-audit-helpers.sh
# prism_audit_check_artifacts "$OUT" run.log binding_sites.json kcc_visualization.json transform_dag.json
EOF
}

prism_audit_print_standard_report_template() {
    cat <<'EOF'
Ticket: TEST-AUDIT-RUNNER-001
Files touched:
- scripts/prism-test-audit-runner.sh
- scripts/lib/prism-audit-helpers.sh

Summary:
- 

Commands run and results:
- 

Artifacts produced:
- 

Acceptance status:
- 

CUDA errors observed: none expected
Deferred drains observed: none expected
Runtime Python introduced: no
Fake scientific fields emitted: no

Rollback path:
- Remove scripts/prism-test-audit-runner.sh and scripts/lib/prism-audit-helpers.sh, or revert the TEST-AUDIT-RUNNER-001 commit.

Remaining risks:
- 
EOF
}

prism_audit_print_rollback_report_template() {
    cat <<'EOF'
Rollback Report

Trigger:
- 

Observed regression:
- CUDA hard errors / stream capture invalidation:
- Deferred drains hidden or normalized:
- Unexpected runtime Python:
- Fake scientific fields:

Commands used to confirm:
- PRISM_TIER8_VERBOSE_DIAG=1 <focused gate command>
- <required rg scans>

Rollback action:
- Revert the TEST-AUDIT-RUNNER-001 commit, or remove:
  - scripts/prism-test-audit-runner.sh
  - scripts/lib/prism-audit-helpers.sh

Post-rollback verification:
- bash -n scripts/prism-test-audit-runner.sh scripts/lib/prism-audit-helpers.sh
- Re-run the same focused/all-8 gate command used to expose the regression.
EOF
}

prism_audit_require_repo_root() {
    local repo_root="$1"
    if [[ ! -f "$repo_root/Cargo.toml" || ! -d "$repo_root/scripts" ]]; then
        echo "ERROR: expected PRISM repo root, got: $repo_root" >&2
        return 1
    fi
}

prism_audit_run_build_command() {
    local command_name="$1"

    case "$command_name" in
        cargo-check-core)
            cargo check -p prism-core
            ;;
        cargo-check-nhs)
            cargo check -p prism-nhs --bin nhs_rt_full --features=v2_ignition
            ;;
        cargo-build-nhs-release)
            cargo build --release -p prism-nhs --bin nhs_rt_full --features=v2_ignition
            ;;
        *)
            echo "ERROR: unknown build command: $command_name" >&2
            return 2
            ;;
    esac
}

prism_audit_run_tier81_smoke() {
    local out
    out="/mnt/storage/prism_tier8_1_smoke_$(prism_audit_timestamp)"
    mkdir -p "$out"
    echo "OUT=$out"
    timeout --signal=INT --kill-after=30s 600s scripts/prism-validate-and-run.sh \
        -t data/targets/mpro_monomer.topology.json -o "$out" \
        --fast --hysteresis --prism-therm --multi-stream 8 --spike-percentile 70 \
        --fused-steps 6 --hmr --adaptive-dt --multi-differential \
        --closed-loop-steering --asymmetric-steering --m1-monolithic-discovery \
        --no-autonomous-rescue --replica-seed 42 -v 2>&1 | tee "$out/run.log"
}

prism_audit_run_all8_gate() {
    local out
    out="/mnt/storage/prism_tier8_all8_$(prism_audit_timestamp)"
    mkdir -p "$out"
    echo "OUT=$out"
    timeout --signal=INT --kill-after=30s 4800s scripts/prism-validate-and-run.sh \
        -t data/targets/mpro_monomer.topology.json -o "$out" \
        --fast --hysteresis --prism-therm --multi-stream 8 --spike-percentile 70 \
        --fused-steps 6 --hmr --adaptive-dt --multi-differential \
        --closed-loop-steering --asymmetric-steering --m1-monolithic-discovery \
        --no-autonomous-rescue --replica-seed 42 -v 2>&1 | tee "$out/run.log"
}

prism_audit_run_log_scans() {
    local run_log="$1"

    if [[ -z "$run_log" ]]; then
        echo "ERROR: --log <path/to/run.log> is required for log-scans --execute" >&2
        return 2
    fi
    if [[ ! -f "$run_log" ]]; then
        echo "ERROR: log not found: $run_log" >&2
        return 1
    fi

    prism_audit_scan_instantiation_milestones "$run_log"
    prism_audit_scan_cuda_failures "$run_log"
    prism_audit_scan_conditional_graph_refs "$run_log"
    prism_audit_scan_deferred_drains "$run_log"
    prism_audit_scan_deferred_drain_risks "$run_log"
    prism_audit_extract_log_milestones "$run_log"
    prism_audit_scan_high_risk_dirty
}

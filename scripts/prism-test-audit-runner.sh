#!/usr/bin/env bash
# PRISM-4D reusable test/audit runner.
#
# Default behavior is inert: print command blocks/templates only. Potentially
# long runtime gates execute only when the caller passes --execute.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/prism-audit-helpers.sh
source "$SCRIPT_DIR/lib/prism-audit-helpers.sh"

REPO_ROOT="$(prism_audit_repo_root)"
EXECUTE=false
EXECUTION_MODE_SET=false
LOG_PATH=""

usage() {
    cat <<'EOF'
Usage:
  scripts/prism-test-audit-runner.sh [--dry-run|--execute] [mode] [options]

Modes:
  help                         Show this help.
  dry-run                      Print all directive command blocks and templates.
  commands                     Print all directive command blocks.
  build-commands               Print cargo check/build commands.
  cargo-check-core             Print or execute: cargo check -p prism-core.
  cargo-check-nhs              Print or execute nhs_rt_full cargo check.
  cargo-build-nhs-release      Print or execute nhs_rt_full release build.
  tier81-smoke                 Print or execute focused TIER 8.1 smoke.
  all8-gate                    Print or execute all-8 instantiate gate.
  scans                        Print rg scans, or execute with --log <run.log>.
  log-scans                    Print rg scans, or execute with --log <run.log>.
  templates                    Print standard and rollback report templates.
  report-template              Print the standard final report template.
  rollback-template            Print the rollback report template.

Options:
  --dry-run                    Print commands/templates only. This is the default.
  --execute                    Execute the selected runnable mode.
  --log <path>                 run.log path for: log-scans --execute.

Examples:
  scripts/prism-test-audit-runner.sh
  scripts/prism-test-audit-runner.sh dry-run
  scripts/prism-test-audit-runner.sh commands
  scripts/prism-test-audit-runner.sh scans
  scripts/prism-test-audit-runner.sh templates
  scripts/prism-test-audit-runner.sh --execute cargo-check-core
  scripts/prism-test-audit-runner.sh log-scans
  scripts/prism-test-audit-runner.sh --execute log-scans --log /mnt/storage/run/run.log

Default mode is dry-run. Long runtime campaigns are never launched unless the
caller passes --execute.
EOF
}

parse_args() {
    MODE="dry-run"
    local mode_seen=false

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)
                if [[ "$EXECUTION_MODE_SET" == "true" && "$EXECUTE" == "true" ]]; then
                    echo "ERROR: --dry-run and --execute are mutually exclusive" >&2
                    exit 2
                fi
                EXECUTE=false
                EXECUTION_MODE_SET=true
                shift
                ;;
            --execute)
                if [[ "$EXECUTION_MODE_SET" == "true" && "$EXECUTE" == "false" ]]; then
                    echo "ERROR: --dry-run and --execute are mutually exclusive" >&2
                    exit 2
                fi
                EXECUTE=true
                EXECUTION_MODE_SET=true
                shift
                ;;
            --log)
                if [[ $# -lt 2 ]]; then
                    echo "ERROR: --log requires a path" >&2
                    exit 2
                fi
                LOG_PATH="$2"
                shift 2
                ;;
            -h|--help)
                MODE="help"
                shift
                ;;
            -*)
                echo "ERROR: unknown option: $1" >&2
                exit 2
                ;;
            *)
                if [[ "$mode_seen" == "true" ]]; then
                    echo "ERROR: unexpected positional argument: $1" >&2
                    exit 2
                fi
                MODE="$1"
                mode_seen=true
                shift
                ;;
        esac
    done
}

print_single_build_command() {
    local command_name="$1"

    case "$command_name" in
        cargo-check-core)
            echo "cargo check -p prism-core"
            ;;
        cargo-check-nhs)
            echo "cargo check -p prism-nhs --bin nhs_rt_full --features=v2_ignition"
            ;;
        cargo-build-nhs-release)
            echo "cargo build --release -p prism-nhs --bin nhs_rt_full --features=v2_ignition"
            ;;
        *)
            echo "ERROR: unknown build command: $command_name" >&2
            return 2
            ;;
    esac
}

run_or_print_build_command() {
    local command_name="$1"

    if [[ "$EXECUTE" == "true" ]]; then
        cd "$REPO_ROOT"
        prism_audit_run_build_command "$command_name"
    else
        print_single_build_command "$command_name"
    fi
}

run_or_print_tier81_smoke() {
    if [[ "$EXECUTE" == "true" ]]; then
        cd "$REPO_ROOT"
        prism_audit_run_tier81_smoke
    else
        prism_audit_print_tier81_smoke_command
    fi
}

run_or_print_all8_gate() {
    if [[ "$EXECUTE" == "true" ]]; then
        cd "$REPO_ROOT"
        prism_audit_run_all8_gate
    else
        prism_audit_print_all8_gate_command
    fi
}

run_or_print_log_scans() {
    if [[ "$EXECUTE" == "true" ]]; then
        cd "$REPO_ROOT"
        prism_audit_run_log_scans "$LOG_PATH"
    else
        prism_audit_print_log_scan_commands
    fi
}

print_all_commands() {
    prism_audit_print_header "Build"
    prism_audit_print_build_commands
    prism_audit_print_header "Focused TIER 8.1 Smoke"
    prism_audit_print_tier81_smoke_command
    prism_audit_print_header "All-8 Instantiate Gate"
    prism_audit_print_all8_gate_command
    prism_audit_print_header "Required Log Scans"
    prism_audit_print_log_scan_commands
}

print_templates() {
    prism_audit_print_header "Standard Report Template"
    prism_audit_print_standard_report_template
    prism_audit_print_header "Rollback Report Template"
    prism_audit_print_rollback_report_template
}

print_dry_run() {
    print_all_commands
    print_templates
}

main() {
    parse_args "$@"
    prism_audit_require_repo_root "$REPO_ROOT"

    case "$MODE" in
        help)
            usage
            ;;
        dry-run)
            if [[ "$EXECUTE" == "true" ]]; then
                echo "ERROR: dry-run mode cannot be combined with --execute" >&2
                exit 2
            fi
            print_dry_run
            ;;
        commands)
            print_all_commands
            ;;
        build-commands)
            prism_audit_print_build_commands
            ;;
        cargo-check-core|cargo-check-nhs|cargo-build-nhs-release)
            run_or_print_build_command "$MODE"
            ;;
        tier81-smoke)
            run_or_print_tier81_smoke
            ;;
        all8-gate)
            run_or_print_all8_gate
            ;;
        scans|log-scans)
            run_or_print_log_scans
            ;;
        templates)
            print_templates
            ;;
        report-template)
            prism_audit_print_standard_report_template
            ;;
        rollback-template)
            prism_audit_print_rollback_report_template
            ;;
        *)
            echo "ERROR: unknown mode: $MODE" >&2
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"

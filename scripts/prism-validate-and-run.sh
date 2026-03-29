#!/usr/bin/env bash
# PRISM-4D Validated Run Pipeline
#
# THE ONLY PERMITTED WAY TO INVOKE THE ENGINE.
# Direct invocation of nhs_rt_full is prohibited.
#
# Usage:
#   scripts/prism-validate-and-run.sh \
#       -t <topology.json> \
#       -o <output_dir> \
#       [all other engine flags passed through]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENGINE="$PROJECT_DIR/target/release/nhs_rt_full"
PREFLIGHT="$SCRIPT_DIR/prism-preflight.py"
POSTFLIGHT="$SCRIPT_DIR/prism-postflight.py"

# Parse -t, -o, --chain-map, collect remaining flags
TOPOLOGY=""
OUTPUT_DIR=""
CHAIN_MAP=""
ENGINE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--topology)
            TOPOLOGY="$2"
            ENGINE_ARGS+=("$1" "$2")
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            ENGINE_ARGS+=("$1" "$2")
            shift 2
            ;;
        --chain-map)
            CHAIN_MAP="$2"
            shift 2
            ;;
        *)
            ENGINE_ARGS+=("$1")
            shift
            ;;
    esac
done

# Validate arguments
if [[ -z "$TOPOLOGY" ]]; then
    echo "Usage: prism-validate-and-run.sh -t <topology.json> -o <output_dir> [engine flags]"
    echo ""
    echo "This is the ONLY permitted way to invoke the PRISM-4D engine."
    echo "All engine flags (--fast, --hysteresis, --multi-stream, etc.) are passed through."
    exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    echo "ERROR: -o <output_dir> is required"
    exit 1
fi

if [[ ! -f "$TOPOLOGY" ]]; then
    echo "ERROR: Topology file not found: $TOPOLOGY"
    exit 1
fi

if [[ ! -f "$ENGINE" ]]; then
    echo "ERROR: Engine binary not found: $ENGINE"
    echo "Build with: cargo build --release --features gpu -p prism-nhs --bin nhs_rt_full"
    exit 1
fi

# Derive prefix from topology filename
PREFIX=$(basename "$TOPOLOGY" | sed 's/_clean\.topology\.json$//' | sed 's/\.topology\.json$//')

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         PRISM-4D PRE-RUN VALIDATION GATE                ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Phase 1: Preflight
PREFLIGHT_ARGS=("$TOPOLOGY")
if [[ -n "$CHAIN_MAP" ]]; then
    PREFLIGHT_ARGS+=("--chain-map" "$CHAIN_MAP")
fi
echo "[Phase 1] Preflight validation: $TOPOLOGY"
if ! python3 "$PREFLIGHT" "${PREFLIGHT_ARGS[@]}"; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║         PREFLIGHT FAILED — RUN ABORTED                  ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    exit 1
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         RUNNING PRISM-4D ENGINE                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Phase 2: Create output dir and run engine
mkdir -p "$OUTPUT_DIR"
ENGINE_EXIT=0
RUST_LOG=info "$ENGINE" "${ENGINE_ARGS[@]}" || ENGINE_EXIT=$?

# Engine exit 134/139 = CUDA teardown segfault — output is valid
if [[ $ENGINE_EXIT -ne 0 && $ENGINE_EXIT -ne 134 && $ENGINE_EXIT -ne 139 ]]; then
    echo ""
    echo "ERROR: Engine exited with code $ENGINE_EXIT"
    exit $ENGINE_EXIT
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         POST-RUN VALIDATION                             ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Phase 3: Postflight
POSTFLIGHT_ARGS=("$OUTPUT_DIR" "$PREFIX")
if [[ -n "$CHAIN_MAP" ]]; then
    POSTFLIGHT_ARGS+=("--chain-map" "$CHAIN_MAP")
fi
python3 "$POSTFLIGHT" "${POSTFLIGHT_ARGS[@]}"
POST_EXIT=$?

if [[ $POST_EXIT -ne 0 ]]; then
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║         POSTFLIGHT FAILED — CHECK OUTPUT                ║"
    echo "╚══════════════════════════════════════════════════════════╝"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         PRISM-4D RUN COMPLETE                           ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo "  Topology: $TOPOLOGY"
echo "  Output:   $OUTPUT_DIR"
echo "  Prefix:   $PREFIX"
echo "  Engine exit: $ENGINE_EXIT"
echo "  Postflight: $([ $POST_EXIT -eq 0 ] && echo PASS || echo FAIL)"

exit $ENGINE_EXIT

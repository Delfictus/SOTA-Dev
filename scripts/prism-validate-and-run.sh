#!/usr/bin/env bash
# PRISM-4D Validated Run Pipeline
#
# THE ONLY PERMITTED WAY TO INVOKE THE ENGINE.
# Direct invocation of nhs_rt_full is prohibited.
# The engine checks for PRISM_VALIDATED=1 and exits with code 2 if unset.
#
# Usage:
#   scripts/prism-validate-and-run.sh \
#       -t <topology.json> \
#       -o <output_dir> \
#       [all other engine flags passed through]
#
# CANONICAL PRODUCTION RUN (as of 2026-03-29, Tier 1 targets confirmed):
#
#   scripts/prism-validate-and-run.sh \
#       -t data/targets/<pdb>.topology.json \
#       -o /tmp/prism_<pdb> \
#       --fast --hysteresis \
#       --multi-stream 20 \
#       --spike-percentile 95 \
#       --prism-therm \
#       --fused-steps 4 \
#       --hmr \
#       --adaptive-dt \
#       --replica-seed 42 \
#       --boltzmann-rank \
#       -v
#
# For smaller targets (<200 residues): --multi-stream 8
# For large targets (>400 residues):   --multi-stream 20

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENGINE="$PROJECT_DIR/target/release/nhs_rt_full"
PREFLIGHT="$SCRIPT_DIR/prism-preflight.py"
POSTFLIGHT="$SCRIPT_DIR/prism-postflight.py"

# ─────────────────────────────────────────────────────────────────────────────
# TWIN PTX bootstrap — fresh clones won't have target/ptx/ populated.
# find_twin_ptx (in twin_kernels.rs) searches target/ptx/ FIRST, so seeding it
# from the vendored bundle takes precedence over anything cargo build produces.
#
# IMPORTANT: this is conditional. If you're iterating on .cu files and your
# freshly-built target/ptx/protocol_director.ptx is already present, it is
# NOT touched. Only seeds when target/ptx/protocol_director.ptx is missing.
# ─────────────────────────────────────────────────────────────────────────────
PTX_BUNDLE="$PROJECT_DIR/vendor/working_ptx_2026-04-10"
PTX_TARGET="$PROJECT_DIR/target/ptx"
if [[ -d "$PTX_BUNDLE" && ! -f "$PTX_TARGET/protocol_director.ptx" ]]; then
    mkdir -p "$PTX_TARGET"
    cp "$PTX_BUNDLE"/*.ptx "$PTX_TARGET/" 2>/dev/null || true
    echo "[PTX bootstrap] Seeded $PTX_TARGET from $PTX_BUNDLE (vendored TWIN bundle)"
fi

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

# Derive prefix from topology filename. The engine derives its file_stem from the
# topology path: for `4lpk_clean.topology.json` → file_stem `4lpk_clean.topology`,
# then `with_extension("binding_sites.json")` strips `.topology` and produces
# `4lpk_clean.binding_sites.json`. So PREFIX must equal `4lpk_clean`, not `4lpk`.
# Strip ONLY `.topology.json` — never `_clean.topology.json` (the previous version
# stripped both, which made postflight look for `4lpk.binding_sites.json` while the
# engine wrote `4lpk_clean.binding_sites.json`, failing the validation gate).
PREFIX=$(basename "$TOPOLOGY" | sed 's/\.topology\.json$//')

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

# Phase 1.5: Auto-generate NMA modes for TWIN (Layer 4 differential perturbation)
# When --coupled-twin is set, Group B needs NMA modes for barrier measurement.
# If no --nma-perturb was explicitly provided and no modes file exists, generate one.
COUPLED_TWIN_ACTIVE=false
NMA_PROVIDED=false
for arg in "${ENGINE_ARGS[@]}"; do
    if [[ "$arg" == "--coupled-twin" ]]; then COUPLED_TWIN_ACTIVE=true; fi
    if [[ "$arg" == "--nma-perturb" ]]; then NMA_PROVIDED=true; fi
done

if [[ "$COUPLED_TWIN_ACTIVE" == "true" && "$NMA_PROVIDED" == "false" ]]; then
    TOPO_DIR=$(dirname "$TOPOLOGY")
    NMA_FILE="${TOPO_DIR}/${PREFIX}_nma_modes.json"
    if [[ -f "$NMA_FILE" ]]; then
        echo "[Phase 1.5] NMA modes found: $NMA_FILE"
        ENGINE_ARGS+=("--nma-perturb" "$NMA_FILE")
    else
        echo "[Phase 1.5] Auto-generating NMA modes for TWIN Layer 4..."
        # Find the source PDB for this topology
        CLEAN_PDB="${TOPO_DIR}/${PREFIX}_clean.pdb"
        RAW_PDB="${TOPO_DIR}/${PREFIX}.pdb"
        SOURCE_PDB=""
        if [[ -f "$CLEAN_PDB" ]]; then
            SOURCE_PDB="$CLEAN_PDB"
        elif [[ -f "$RAW_PDB" ]]; then
            SOURCE_PDB="$RAW_PDB"
        fi

        if [[ -n "$SOURCE_PDB" ]]; then
            if python3 scripts/prism-prep "$SOURCE_PDB" "$TOPOLOGY" --nma-modes 10 2>&1 | tail -5; then
                if [[ -f "$NMA_FILE" ]]; then
                    echo "  ✓ NMA modes generated: $NMA_FILE"
                    ENGINE_ARGS+=("--nma-perturb" "$NMA_FILE")
                else
                    echo "  ✗ NMA generation completed but modes file not found"
                    echo "    TWIN will run without NMA (Layer 4 differential features = zero)"
                fi
            else
                echo "  ✗ NMA generation failed (ProDy error?)"
                echo "    TWIN will run without NMA (Layer 4 differential features = zero)"
            fi
        else
            echo "  ✗ Source PDB not found for NMA generation"
            echo "    Looked for: $CLEAN_PDB and $RAW_PDB"
            echo "    TWIN will run without NMA (Layer 4 differential features = zero)"
        fi
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         RUNNING PRISM-4D ENGINE                         ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Phase 2: Create output dir and run engine
mkdir -p "$OUTPUT_DIR"
ENGINE_EXIT=0
PRISM_VALIDATED=1 RUST_LOG=info "$ENGINE" "${ENGINE_ARGS[@]}" || ENGINE_EXIT=$?

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

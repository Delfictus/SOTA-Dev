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
# CANONICAL PRODUCTION RUN (2026-05-20 red-flag revision):
#
# Source of truth: crates/prism-nhs/src/bin/nhs_rt_full.rs (see docs/CANONICAL_PROVENANCE.md)
# Sizing rule: <200 residues: --multi-stream 8  |  200-400: --multi-stream 8  |  >400: --multi-stream 20
#
# RED FLAG (operator-locked 2026-05-20): every invocation MUST include
# --md-only-evidence + --path-a-production-profile + --path-a-max-wall-seconds.
# The engine's internal post-MD CCL union-find clustering is FORBIDDEN
# across the board, forever.  Phase-manifold ranking REPLACES it (does
# not run in addition).  Stevens GLP1R aleniglipron canonical (40 replicas,
# 3.046B spikes, 194s mean replica wall) used this exact shape -- see
# prism-glp1r-aleniglipron-workspace/.../02_RUNTIME_CONFIG/glp1r_runtime.env
#
#   scripts/prism-validate-and-run.sh \
#       -t <topology.json> \
#       -o <output_dir> \
#       --fast --hysteresis --prism-therm \
#       --multi-stream 8 \
#       --spike-percentile 70 \
#       --fused-steps 6 \
#       --hmr --adaptive-dt \
#       --multi-differential \
#       --closed-loop-steering --asymmetric-steering \
#       --site-ranker phase-manifold \
#       --md-only-evidence \
#       --path-a-production-profile \
#       --path-a-max-wall-seconds 180 \
#       --uv-wavelengths 280,274,258,254,211 \
#       --nma-amplification 3.0 --nma-scan-fraction 0.3 \
#       --replica-seed 42 -v
#
# DO NOT add: --use-xgb-ranker, --boltzmann-rank, --cascade, monolithic
# --replicas.  N-replicate consensus goes through prism_replicate.py only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
ENGINE="$PROJECT_DIR/target/release/nhs_rt_full"
PREFLIGHT="$SCRIPT_DIR/prism-preflight.py"
GROUND_TRUTH="$SCRIPT_DIR/prism-ground-truth.py"
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
    echo "Build with: cargo build --release --features v2_ignition -p prism-nhs --bin nhs_rt_full"
    exit 1
fi

engine_has_symbol() {
    grep -a -q -- "$1" "$ENGINE"
}

if ! engine_has_symbol "producer_frames_enqueued" \
    || ! engine_has_symbol "frames_written" \
    || ! engine_has_symbol "all_hashes_match" \
    || ! engine_has_symbol "lossless backpressure"; then
    echo "ERROR: Engine binary does not contain the V2 lossless trajectory audit path."
    echo "Rebuild with: cargo build --release --features v2_ignition -p prism-nhs --bin nhs_rt_full"
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

# Phase 1.2: Ground truth resolution
# Resolves the holo PDB on RCSB (cached), classifies the deposit, and
# writes <output_dir>/<prefix>_ground_truth.json. This sidecar is read
# by Phase 3 postflight to compute DCC validation against the published
# ligand centroid. Filters out PanDDA fragment hits and templated
# ternary complexes — those are not appropriate orthosteric ground
# truths and would produce false-negative validation results.
#
# This phase NEVER fails the run. If the holo PDB is unavailable or
# the deposit is filtered, ground truth becomes "skip" and DCC
# validation is bypassed in postflight. The engine still runs.
mkdir -p "$OUTPUT_DIR"
echo ""
echo "[Phase 1.2] Ground truth resolution"
python3 "$GROUND_TRUTH" "$TOPOLOGY" "$OUTPUT_DIR" || true

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

# Phase 2: Run engine (output dir was created in Phase 1.2)
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

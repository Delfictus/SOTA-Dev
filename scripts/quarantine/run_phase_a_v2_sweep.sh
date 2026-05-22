#!/usr/bin/env bash
# Phase-A V2 capture-window sweep — fixed mpro_monomer/seed/topology baseline,
# varies --path-a-v2-trigger-steps. Run sequentially (GPU is shared).
# Output: per-trigger run dirs under /mnt/storage/PHASE_A_V2_SWEEP_<utc>/.
#
# Required: patched binary at target/release/nhs_rt_full (with V2 CAPTURE GUARD
# at nhs_rt_full.rs:6587-6650 + cargo build --release --features v2_ignition).

set -u
ROOT="/home/diddy/Desktop/Prism4D-bio"
TOPO="${ROOT}/data/targets/mpro_monomer.topology.json"
SWEEP_ROOT="/mnt/storage/PHASE_A_V2_SWEEP_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$SWEEP_ROOT"
echo "$SWEEP_ROOT" > /tmp/phase_a_sweep_root.txt

TRIGGERS=("${@:-2000 10000 30000}")
# allow comma- or space-separated list as one arg, or multiple args
if [ "$#" -eq 1 ]; then
  IFS=', ' read -r -a TRIGGERS <<< "$1"
fi

cd "$ROOT" || exit 2

for T in "${TRIGGERS[@]}"; do
  RUNDIR="${SWEEP_ROOT}/trigger_${T}"
  mkdir -p "$RUNDIR"
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "═══ V2 sweep: trigger=${T}  rundir=${RUNDIR}"
  echo "═══════════════════════════════════════════════════════════════"
  ./scripts/prism-validate-and-run.sh \
    -t "$TOPO" \
    -o "$RUNDIR" \
    --fast --hysteresis --prism-therm \
    --multi-stream 4 \
    --spike-percentile 70 \
    --fused-steps 6 \
    --hmr --adaptive-dt \
    --m1-monolithic-discovery \
    --mar-v2-telemetry \
    --ghost-diagnostic-firehose \
    --path-a-v2-trigger-steps "$T" \
    --path-a-t7-max-chunks 24 \
    --path-a-evidence-exit \
    --path-a-max-wall-seconds 1800 \
    --replica-seed 42 \
    -v \
    > "$RUNDIR/run.log" 2>&1
  EX=$?
  echo "trigger=${T} exit=${EX}"
  echo "  ghost_tiles.bin: $(ls "$RUNDIR"/*ghost_tiles.bin 2>/dev/null | wc -l)"
  echo "  v2_frames.bin:   $(ls "$RUNDIR"/*v2_frames.bin 2>/dev/null | wc -l)"
  echo "  V2 markers:      $(grep -cE 'V2-BUILD|MONO-FUSE|TIER8-PREFLIGHT|HARD.GATE|CapturedAdjudicationPipeline' "$RUNDIR/run.log" 2>/dev/null)"
  echo "  GUARD-BLOCKS:    $(grep -cE 'V2 CAPTURE GUARD blocks early-exit' "$RUNDIR/run.log" 2>/dev/null)"
done

echo ""
echo "Sweep complete. Root: $SWEEP_ROOT"

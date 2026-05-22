#!/usr/bin/env bash
set -u
mode="${1:-}"
case "$mode" in
  single|twin|md_asym) ;;
  *)
    echo "usage: $0 {single|twin|md_asym}" >&2
    exit 1
    ;;
esac

ROOT="/home/diddy/Desktop/Prism4D-bio"
BASE="/home/diddy/Desktop/TEST 5-8/6OIM"
TOPO="${BASE}/prep/6oim_chainA.topology.json"
OUT="${BASE}/${mode}"
LOG="${OUT}/run.log"

mkdir -p "$OUT"

CMN=( --fast --hysteresis --prism-therm
      --multi-stream 8
      --spike-percentile 70
      --fused-steps 6
      --hmr --adaptive-dt
      -v )

case "$mode" in
  single)
    EXTRA=()
    ;;
  twin)
    EXTRA=( --coupled-twin --graph-coupling )
    ;;
  md_asym)
    EXTRA=( --multi-differential --closed-loop-steering --asymmetric-steering )
    ;;
esac

cd "$ROOT" || exit 2
export RUST_LOG="${RUST_LOG:-info}"

nohup ./scripts/prism-validate-and-run.sh \
    -t "$TOPO" -o "$OUT" \
    "${CMN[@]}" "${EXTRA[@]}" \
    > "$LOG" 2>&1 &

PID=$!
echo "mode=$mode  PID=$PID"
echo "log:  $LOG"
echo "watch: tail -f \"$LOG\""

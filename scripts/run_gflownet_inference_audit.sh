#!/usr/bin/env bash
# Phase 2-9 chain orchestrator. Waits for Phase 2's parquet to appear
# (since Phase 2 may already be running in another shell), then runs
# Phases 3-9 in order.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG=/tmp/gflownet_inference_chain.log
PHASE2_PARQUET=campaigns/glp1r_aleniglipron/track_a_generative/gflownet_raw_policy_samples.parquet

echo "[chain] start $(date -u +%H:%M:%S)" | tee -a "$LOG"

# --- Wait for Phase 2 to finish (parquet appears) ----------------------
while [ ! -f "$PHASE2_PARQUET" ]; do
  echo "[chain] $(date -u +%H:%M:%S) Phase 2 not yet — waiting…" | tee -a "$LOG"
  sleep 30
done
echo "[chain] $(date -u +%H:%M:%S) Phase 2 parquet detected ($(stat -c%s $PHASE2_PARQUET) B)" | tee -a "$LOG"

# Also need the summary JSON before progressing.
SUMM=campaigns/glp1r_aleniglipron/track_a_generative/gflownet_raw_policy_samples_summary.json
while [ ! -f "$SUMM" ]; do
  echo "[chain] $(date -u +%H:%M:%S) waiting for Phase 2 summary JSON…" | tee -a "$LOG"
  sleep 15
done

run_phase () {
  local name="$1"; shift
  echo "[chain] $(date -u +%H:%M:%S) starting $name" | tee -a "$LOG"
  if ! "$@" >> "$LOG" 2>&1; then
    echo "[chain] $(date -u +%H:%M:%S) FAILED $name (see $LOG)" | tee -a "$LOG"
    return 1
  fi
  echo "[chain] $(date -u +%H:%M:%S) OK $name" | tee -a "$LOG"
}

run_phase phase3_oracle  python3 scripts/rescore_gflownet_samples.py
run_phase phase4_baselines python3 scripts/evaluate_gflownet_vs_baselines.py
run_phase phase5_medchem  python3 scripts/filter_gflownet_medchem_plausibility.py
run_phase phase6_topk     python3 scripts/select_gflownet_diverse_top_candidates.py
run_phase phase7_audit    python3 scripts/audit_gflownet_failure_modes.py
run_phase phase8_artifacts python3 scripts/build_gflownet_review_artifacts.py
run_phase phase9_validate  python3 scripts/validate_and_package_gflownet_inference.py

echo "[chain] $(date -u +%H:%M:%S) all phases complete" | tee -a "$LOG"
echo "[chain] final tarball:" | tee -a "$LOG"
ls -la "$ROOT"/PRISM_TRACK_A_GFLOWNET_V1_INFERENCE_AUDIT_v1.0.tar.gz* | tee -a "$LOG"

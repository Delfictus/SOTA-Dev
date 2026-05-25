#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRACK_A="campaigns/glp1r_aleniglipron/track_a_generative"
LOG_DIR="$TRACK_A/fullscale_logs"
mkdir -p "$LOG_DIR"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="$LOG_DIR/vspace_smarts_full_${RUN_ID}.log"
LOCK_FILE="$LOG_DIR/vspace_smarts_full.lock"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "full_vspace_already_running lock=$LOCK_FILE" | tee -a "$RUN_LOG"
  exit 1
fi

echo "full_vspace_launch run_id=$RUN_ID utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUN_LOG"

REGISTRY="00_registry/chemistry/reaction_rules.v1.yml"
SOURCE_CSV="$TRACK_A/115k_curated_anchors.csv"
FULL_SYNTHONS="$TRACK_A/enamine_115k_synthons_3d.parquet"
FULL_REPORT="$TRACK_A/enamine_115k_synthons_ingest_report.json"
PLAN="$TRACK_A/vspace_fullscale_dry_run_plan.json"
OUTPUT="$TRACK_A/vspace_survivors_smarts_full.parquet"
TELEMETRY="$TRACK_A/vspace_smarts_full_telemetry.json"
LIGAND="$TRACK_A/ALENI-PARENT_6XOX_frame_o3a_relaxed.sdf"
VOXEL_THRESHOLDS="$TRACK_A/voxel_thresholds.json"

if [[ ! -s "$SOURCE_CSV" ]]; then
  echo "missing_source_csv path=$SOURCE_CSV" | tee -a "$RUN_LOG"
  exit 2
fi

python3 scripts/validate_reaction_registry.py 2>&1 | tee -a "$RUN_LOG"

if [[ ! -s "$FULL_SYNTHONS" ]]; then
  TMP_SYNTHONS="${FULL_SYNTHONS}.tmp.${RUN_ID}"
  TMP_REPORT="${FULL_REPORT}.tmp.${RUN_ID}"
  echo "smart_ingest_start source=$SOURCE_CSV output_tmp=$TMP_SYNTHONS" | tee -a "$RUN_LOG"
  python3 scripts/ingest_vspace_synthons.py \
    --input "$SOURCE_CSV" \
    --reaction-registry "$REGISTRY" \
    --output "$TMP_SYNTHONS" \
    --report "$TMP_REPORT" 2>&1 | tee -a "$RUN_LOG"
  mv "$TMP_SYNTHONS" "$FULL_SYNTHONS"
  mv "$TMP_REPORT" "$FULL_REPORT"
  echo "smart_ingest_committed output=$FULL_SYNTHONS report=$FULL_REPORT" | tee -a "$RUN_LOG"
else
  echo "smart_ingest_reuse_existing output=$FULL_SYNTHONS" | tee -a "$RUN_LOG"
fi

cargo build --release -p prism-forge --bin vspace_pruner 2>&1 | tee -a "$RUN_LOG"

echo "dry_run_plan_start path=$PLAN" | tee -a "$RUN_LOG"
target/release/vspace_pruner \
  --full-scale true \
  --dry-run-plan true \
  --force true \
  --reaction-registry "$REGISTRY" \
  --synthon-parquet "$FULL_SYNTHONS" \
  --ligand-sdf "$LIGAND" \
  --voxel-thresholds "$VOXEL_THRESHOLDS" \
  --assembly-mode smarts_zmatrix \
  --fullscale-dry-run-plan "$PLAN" \
  --output "$OUTPUT" \
  --telemetry-json "$TELEMETRY" 2>&1 | tee -a "$RUN_LOG"

TMP_OUTPUT="${OUTPUT}.tmp.${RUN_ID}"
TMP_TELEMETRY="${TELEMETRY}.tmp.${RUN_ID}"

echo "full_vspace_execute_start output_tmp=$TMP_OUTPUT telemetry_tmp=$TMP_TELEMETRY" | tee -a "$RUN_LOG"
target/release/vspace_pruner \
  --max-pairs 100000000000 \
  --reaction-registry "$REGISTRY" \
  --synthon-parquet "$FULL_SYNTHONS" \
  --ligand-sdf "$LIGAND" \
  --voxel-thresholds "$VOXEL_THRESHOLDS" \
  --assembly-mode smarts_zmatrix \
  --real-anchors-only true \
  --survivor-limit 1000000 \
  --output "$TMP_OUTPUT" \
  --telemetry-json "$TMP_TELEMETRY" 2>&1 | tee -a "$RUN_LOG"

mv "$TMP_OUTPUT" "$OUTPUT"
mv "$TMP_TELEMETRY" "$TELEMETRY"

python3 - <<'PY' 2>&1 | tee -a "$RUN_LOG"
import json
from pathlib import Path
import polars as pl

track = Path("campaigns/glp1r_aleniglipron/track_a_generative")
telemetry = json.loads((track / "vspace_smarts_full_telemetry.json").read_text())
survivors = pl.scan_parquet(track / "vspace_survivors_smarts_full.parquet").select(
    pl.len().alias("rows"),
    pl.col("canonical_smiles").n_unique().alias("unique_smiles"),
).collect()
print(
    "full_vspace_complete "
    f"attempted_pairs={telemetry.get('attempted_pairs')} "
    f"rotamers_evaluated={telemetry.get('rotamers_evaluated')} "
    f"survivors={int(survivors['rows'][0])} "
    f"unique_smiles={int(survivors['unique_smiles'][0])}"
)
PY

echo "full_vspace_done utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$RUN_LOG"

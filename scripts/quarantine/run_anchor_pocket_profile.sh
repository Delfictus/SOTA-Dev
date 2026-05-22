#!/usr/bin/env bash
# Run anchor_point_map.py + pocket_profile_builder.py for all 10 blind validation targets.
# Inputs: frozen binding_sites.json + per-target run/ spike_events dir
# Output: per-target anchor_points.json + pocket_profiles.json → OUTROOT/

set -euo pipefail

REPO="/home/diddy/Desktop/Prism4D-bio"
BLIND="/mnt/storage/prism-outputs/blind_validation"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
OUTROOT="${BLIND}/anchor_pocket_profiles_${TIMESTAMP}"
MANIFEST="${REPO}/docs/blind_validation/frozen_predictions/sha256_manifest.txt"

declare -A TARGETS=(
  [B01_HRAS_Q61H]="4L9S"
  [B02_CDK2_allosteric]="1HCL"
  [B03_Kv1.2]="3LUT"
  [B04_MDM2]="1YCR"
  [B05_TP53_R175H]="2OCJ"
  [B06_cGAS]="4KM5"
  [B07_TEAD1]="3KYS"
  [B08_CRBN]="4TZ4_chainC"
  [B09_Thrombin_exosite]="1PPB"
  [B10_ADRB2]="2RH1"
)

mkdir -p "${OUTROOT}"
echo "Output root: ${OUTROOT}"
echo ""

echo "# anchor_points.json + pocket_profiles.json — ${TIMESTAMP}" >> "${MANIFEST}"

for TARGET in "${!TARGETS[@]}"; do
  PDB="${TARGETS[$TARGET]}"
  BS="${BLIND}/${TARGET}/frozen/${PDB}.binding_sites.json"
  SPIKES_DIR="${BLIND}/${TARGET}/run"
  OUTDIR="${OUTROOT}/${TARGET}"
  mkdir -p "${OUTDIR}"

  if [[ ! -f "${BS}" ]]; then
    echo "MISSING binding_sites: ${TARGET} ${BS}" >&2
    continue
  fi

  echo "[$(date -u +%H:%M:%SZ)] ${TARGET} — anchor_point_map..."
  python3 -m scripts.anchor_point_map \
    --binding-sites "${BS}" \
    --spike-events "${SPIKES_DIR}" \
    --out "${OUTDIR}/anchor_points.json"

  echo "[$(date -u +%H:%M:%SZ)] ${TARGET} — pocket_profile_builder..."
  python3 -m scripts.pocket_profile_builder \
    --binding-sites "${BS}" \
    --out "${OUTDIR}/pocket_profiles.json"

  # Hash outputs into freeze manifest
  sha256sum "${OUTDIR}/anchor_points.json" >> "${MANIFEST}"
  sha256sum "${OUTDIR}/pocket_profiles.json" >> "${MANIFEST}"

  echo "[$(date -u +%H:%M:%SZ)] ${TARGET} DONE"
  echo ""
done

echo "=== ALL TARGETS COMPLETE ==="
echo "Results: ${OUTROOT}"

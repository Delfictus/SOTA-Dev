#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <target-bank-run-dir>" >&2
  exit 2
fi

RUN_DIR="$1"
CURATE_SUBDIR="${PRISM5000_CURATE_SUBDIR:-curate_full}"
PREP_SUBDIR="${PRISM5000_PREP_SUBDIR:-prepared_full}"
MANIFEST="${RUN_DIR}/${CURATE_SUBDIR}/prism5000_chain_manifest.jsonl"
REPORT="${RUN_DIR}/${CURATE_SUBDIR}/prism5000_category_report.json"
PREP_OUT="${RUN_DIR}/${PREP_SUBDIR}"
PARALLEL="${PRISM5000_PREP_PARALLEL:-4}"
TIMEOUT_SEC="${PRISM5000_PREP_TIMEOUT_SEC:-1800}"

echo "[prep-wait] run_dir=${RUN_DIR}"
echo "[prep-wait] waiting for ${MANIFEST}"

while [[ ! -s "${MANIFEST}" || ! -s "${REPORT}" ]]; do
  sleep 60
done

echo "[prep-wait] manifest is ready; starting strict PRISM prep"
exec "$(dirname "$0")/prepare_prism5000_chain_targets.py" \
  --manifest "${MANIFEST}" \
  --out-dir "${PREP_OUT}" \
  --parallel "${PARALLEL}" \
  --timeout-sec "${TIMEOUT_SEC}"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <target-bank-run-dir>" >&2
  exit 2
fi

RUN_DIR="$1"
CURATE_SUBDIR="${PRISM5000_CURATE_SUBDIR:-curate_balanced_6200}"
PREP_SUBDIR="${PRISM5000_PREP_SUBDIR:-prepared_balanced_6200}"
FINAL_SUBDIR="${PRISM5000_FINAL_SUBDIR:-final_ready_5000}"
N_TARGETS="${PRISM5000_FINAL_N_TARGETS:-5000}"

CURATED="${RUN_DIR}/${CURATE_SUBDIR}/prism5000_chain_manifest.jsonl"
READY="${RUN_DIR}/${PREP_SUBDIR}/ready_manifest.jsonl"
REPORT="${RUN_DIR}/${PREP_SUBDIR}/prep_report.json"
OUT_DIR="${RUN_DIR}/${FINAL_SUBDIR}"

echo "[finalize-wait] run_dir=${RUN_DIR}"
echo "[finalize-wait] waiting for ${READY}"

while [[ ! -s "${READY}" || ! -s "${REPORT}" ]]; do
  sleep 120
done

echo "[finalize-wait] ready manifest found; producing exact ${N_TARGETS} target queue"
exec "$(dirname "$0")/finalize_prism5000_ready_set.py" \
  --curated-manifest "${CURATED}" \
  --ready-manifest "${READY}" \
  --out-dir "${OUT_DIR}" \
  --n-targets "${N_TARGETS}"

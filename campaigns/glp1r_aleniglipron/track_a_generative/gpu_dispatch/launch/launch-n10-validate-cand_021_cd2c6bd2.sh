#!/usr/bin/env bash
#SBATCH --job-name=prism-cand_021_cd2c6bd2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

REPO_ROOT=/home/diddy/Desktop/Prism4D-bio
cd "$REPO_ROOT"

CANDIDATE_ID=cand_021_cd2c6bd2
SDF_PATH=/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/gpu_dispatch/sdf/cand_021_cd2c6bd2.sdf
TOPOLOGY_JSON=/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/gpu_dispatch/topologies/cand_021_cd2c6bd2.json
REPLICAS="10"
PROTOCOL=ccns_5phase
DISPATCH_ROOT="$(dirname "$SDF_PATH")/.."
VALIDATION_MANIFEST_DIR="${PRISM_VALIDATION_MANIFEST_DIR:-${DISPATCH_ROOT}/validation_manifests}"
RESULTS_ROOT="${PRISM_GPU_RESULTS_ROOT:-${DISPATCH_ROOT}/results}"
OUTPUT_DIR="${RESULTS_ROOT}/${CANDIDATE_ID}"
CHANNEL_DIR="${OUTPUT_DIR}/channels"
DEFAULT_PROTOCOL_STATE_SUMMARY=/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/protocol_state_summary.parquet
REFERENCE_RAW_ROOT=/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/20260518T031002Z/05_RESULTS/glp1r_aleniglipron_risk_map
DEFAULT_TOPOLOGY_ROOT=/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/20260518T031002Z/04_TOPOLOGIES
PROTOCOL_STATE_SUMMARY="${PRISM_PROTOCOL_STATE_SUMMARY:-$DEFAULT_PROTOCOL_STATE_SUMMARY}"
CANDIDATE_RAW_ROOT="${PRISM_CANDIDATE_RAW_ROOT:-${OUTPUT_DIR}/ccns_raw}"
RAW_ROOT="${PRISM_RAW_ROOT:-$CANDIDATE_RAW_ROOT}"
TOPOLOGY_ROOT="${PRISM_TOPOLOGY_ROOT:-$DEFAULT_TOPOLOGY_ROOT}"
ALLOW_SHARED_RAW_ROOT="${PRISM_ALLOW_SHARED_RAW_ROOT:-0}"
BIFURCATE_CHANNELS="pocket,lock,pathway"
SIGNAL_GRID_DIR="${CHANNEL_DIR}/channel_1_signal_grid"
WARP_JACOBIAN_DIR="${CHANNEL_DIR}/channel_2_warp_jacobian"
HYSTERESIS_DIR="${CHANNEL_DIR}/channel_3_hysteresis"
PATHWAY_DIR="${CHANNEL_DIR}/channel_4_pathway"
TIMESTEP_JSON="${OUTPUT_DIR}/protocol_timesteps.json"
SIGNAL_GRID_PARQUET="${SIGNAL_GRID_DIR}/signal_grid_variance_channel.parquet"
WARP_JACOBIAN_PARQUET="${WARP_JACOBIAN_DIR}/shear_stress_field.parquet"
HYSTERESIS_JSON="${HYSTERESIS_DIR}/hysteresis_analysis.json"
PATHWAY_JSON="${PATHWAY_DIR}/pathway_analysis.json"
TRIPARTITE_UPGRADE_JSON="${OUTPUT_DIR}/tripartite_upgrade.json"

mkdir -p "$VALIDATION_MANIFEST_DIR" "$SIGNAL_GRID_DIR" "$WARP_JACOBIAN_DIR" "$HYSTERESIS_DIR" "$PATHWAY_DIR"

assert_candidate_raw_root() {
  if [ "$ALLOW_SHARED_RAW_ROOT" != "1" ]; then
    if [ "$RAW_ROOT" = "$REFERENCE_RAW_ROOT" ]; then
      echo "PRISM_RAW_ROOT points at shared campaign raw root; set PRISM_CANDIDATE_RAW_ROOT/PRISM_RAW_ROOT to candidate-specific CCNS output or PRISM_ALLOW_SHARED_RAW_ROOT=1" >&2
      return 2
    fi
    case "$RAW_ROOT" in
      *"$CANDIDATE_ID"*) ;;
      *) echo "RAW_ROOT must include candidate id ($CANDIDATE_ID) unless PRISM_ALLOW_SHARED_RAW_ROOT=1: $RAW_ROOT" >&2; return 2 ;;
    esac
  fi
  if ! find "$RAW_ROOT" -name signal_grid.bin -print -quit 2>/dev/null | grep -q .; then
    echo "candidate-specific RAW_ROOT has no signal_grid.bin: $RAW_ROOT" >&2
    return 2
  fi
  if ! find "$RAW_ROOT" -name warp_matrix.bin -print -quit 2>/dev/null | grep -q .; then
    echo "candidate-specific RAW_ROOT has no warp_matrix.bin: $RAW_ROOT" >&2
    return 2
  fi
}

python3 scripts/process_gpu_dispatch_results.py \
  --mode timestep_extraction \
  --protocol-state-summary "$PROTOCOL_STATE_SUMMARY" \
  --output "$TIMESTEP_JSON"

extract_json_list() {
  local json_path="$1"
  local key="$2"
  python3 - "$json_path" "$key" <<'PY'
import json
import sys
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
values = payload.get(sys.argv[2], [])
if isinstance(values, list):
    print(",".join(str(int(value)) for value in values))
else:
    print("")
PY
}

EQUILIBRATED_FRAMES="$(extract_json_list "$TIMESTEP_JSON" equilibrated_frames)"
RAMP_PHASE_FRAMES="$(extract_json_list "$TIMESTEP_JSON" ramp_frames)"

run_prism_nhs_bin() {
  local bin_name="$1"
  shift
  cargo run -p prism-nhs --bin "$bin_name" -- "$@"
}

signal_grid_differential() {
  local raw_root=""
  local out_dir=""
  local protocol_state_summary=""
  local frame_scope=""
  local bifurcate=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --raw-root) raw_root="$2"; shift 2 ;;
      --out-dir) out_dir="$2"; shift 2 ;;
      --protocol-state-summary) protocol_state_summary="$2"; shift 2 ;;
      --frame-scope) frame_scope="$2"; shift 2 ;;
      --bifurcate) bifurcate="$2"; shift 2 ;;
      *) echo "signal_grid_differential unsupported arg: $1" >&2; return 2 ;;
    esac
  done
  run_prism_nhs_bin signal_grid_differential \
    --raw-root "$raw_root" \
    --out-dir "$out_dir" \
    --protocol-state-summary "$protocol_state_summary" \
    --frame-scope "$frame_scope" \
    --bifurcate "$bifurcate"
  python3 scripts/process_gpu_dispatch_results.py \
    --mode channel_metadata \
    --channel signal_grid_differential \
    --protocol-state-summary "$protocol_state_summary" \
    --frame-scope "$frame_scope" \
    --bifurcate "$bifurcate" \
    --output "${out_dir}/dispatch_channel_metadata.json"
}

warp_jacobian() {
  local raw_root=""
  local out_dir=""
  local topology_root=""
  local protocol_state_summary=""
  local frames=""
  local bifurcate=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --raw-root) raw_root="$2"; shift 2 ;;
      --out-dir) out_dir="$2"; shift 2 ;;
      --topology-root) topology_root="$2"; shift 2 ;;
      --protocol-state-summary) protocol_state_summary="$2"; shift 2 ;;
      --frames) frames="$2"; shift 2 ;;
      --bifurcate) bifurcate="$2"; shift 2 ;;
      *) echo "warp_jacobian unsupported arg: $1" >&2; return 2 ;;
    esac
  done
  run_prism_nhs_bin warp_jacobian \
    --raw-root "$raw_root" \
    --out-dir "$out_dir" \
    --topology-root "$topology_root" \
    --protocol-state-summary "$protocol_state_summary" \
    --frames "$frames" \
    --bifurcate "$bifurcate"
  python3 scripts/process_gpu_dispatch_results.py \
    --mode channel_metadata \
    --channel warp_jacobian \
    --frames "$frames" \
    --bifurcate "$bifurcate" \
    --output "${out_dir}/dispatch_channel_metadata.json"
}

hysteresis_analysis() {
  run_prism_nhs_bin hysteresis_analysis "$@"
}

pathway_analysis() {
  run_prism_nhs_bin pathway_analysis "$@"
}

echo "gpu_validation_start candidate=${CANDIDATE_ID} replicas=${REPLICAS} sdf=${SDF_PATH} protocol=${PROTOCOL}"
python3 scripts/run_ccns_validation_md.py \
  --candidate-id "${CANDIDATE_ID}" \
  --sdf "${SDF_PATH}" \
  --replicas "${REPLICAS}" \
  --protocol "${PROTOCOL}" \
  --output-dir "${VALIDATION_MANIFEST_DIR}" \
  --raw-output-dir "${CANDIDATE_RAW_ROOT}"

assert_candidate_raw_root

echo "gpu_dispatch_channel_1_start candidate=${CANDIDATE_ID}"
signal_grid_differential \
  --raw-root "${RAW_ROOT}" \
  --out-dir "${SIGNAL_GRID_DIR}" \
  --protocol-state-summary "${PROTOCOL_STATE_SUMMARY}" \
  --frame-scope "all" \
  --bifurcate "${BIFURCATE_CHANNELS}"

echo "gpu_dispatch_channel_2_start candidate=${CANDIDATE_ID}"
warp_jacobian \
  --raw-root "${RAW_ROOT}" \
  --out-dir "${WARP_JACOBIAN_DIR}" \
  --topology-root "${TOPOLOGY_ROOT}" \
  --protocol-state-summary "${PROTOCOL_STATE_SUMMARY}" \
  --frames "${EQUILIBRATED_FRAMES}" \
  --bifurcate "${BIFURCATE_CHANNELS}"

echo "gpu_dispatch_channel_3_start candidate=${CANDIDATE_ID}"
hysteresis_analysis \
  --candidate-id "${CANDIDATE_ID}" \
  --signal-grid "${SIGNAL_GRID_PARQUET}" \
  --protocol-state-summary "${PROTOCOL_STATE_SUMMARY}" \
  --output "${HYSTERESIS_JSON}" \
  --bifurcate "${BIFURCATE_CHANNELS}"

echo "gpu_dispatch_channel_4_start candidate=${CANDIDATE_ID}"
pathway_analysis \
  --candidate-id "${CANDIDATE_ID}" \
  --signal-grid "${SIGNAL_GRID_PARQUET}" \
  --warp-jacobian "${WARP_JACOBIAN_PARQUET}" \
  --protocol-state-summary "${PROTOCOL_STATE_SUMMARY}" \
  --phase-filter "${RAMP_PHASE_FRAMES}" \
  --output "${PATHWAY_JSON}" \
  --bifurcate "${BIFURCATE_CHANNELS}"

python3 scripts/process_gpu_dispatch_results.py \
  --mode tripartite_upgrade \
  --candidate-id "${CANDIDATE_ID}" \
  --dispatch-dir "${DISPATCH_ROOT}" \
  --result-dir "${OUTPUT_DIR}" \
  --signal-grid "${SIGNAL_GRID_PARQUET}" \
  --warp-jacobian "${WARP_JACOBIAN_PARQUET}" \
  --hysteresis "${HYSTERESIS_JSON}" \
  --pathway "${PATHWAY_JSON}" \
  --output "${TRIPARTITE_UPGRADE_JSON}" \
  --bifurcate "${BIFURCATE_CHANNELS}"
echo "gpu_validation_complete candidate=${CANDIDATE_ID}"

#!/usr/bin/env bash
#SBATCH --job-name=prism-cand_021_cd2c6bd2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

CANDIDATE_ID="cand_021_cd2c6bd2"
SDF_PATH="/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/gpu_dispatch/sdf/cand_021_cd2c6bd2.sdf"
TOPOLOGY_JSON="campaigns/glp1r_aleniglipron/track_a_generative/gpu_dispatch/topologies/cand_021_cd2c6bd2.json"
REPLICAS="10"
PROTOCOL="ccns_5phase"
ENGINE="${PRISM_NHS_ENGINE:-demo/nhs_rt_full}"
OUTPUT_DIR="$(dirname "$TOPOLOGY_JSON")/../results/${CANDIDATE_ID}"
mkdir -p "$OUTPUT_DIR"

echo "gpu_validation_start candidate=${CANDIDATE_ID} replicas=${REPLICAS} engine=${ENGINE}"
"$ENGINE" \
  --topology "$TOPOLOGY_JSON" \
  --ligand-sdf "$SDF_PATH" \
  --protocol "$PROTOCOL" \
  --n-replicas "$REPLICAS" \
  --multi-differential \
  --hysteresis \
  --prism-therm \
  --save-trajectory-interval 50 \
  --output-dir "$OUTPUT_DIR"
echo "gpu_validation_complete candidate=${CANDIDATE_ID} output_dir=${OUTPUT_DIR}"

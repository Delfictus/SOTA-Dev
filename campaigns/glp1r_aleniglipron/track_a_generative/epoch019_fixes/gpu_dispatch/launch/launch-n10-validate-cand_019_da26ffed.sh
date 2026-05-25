#!/usr/bin/env bash
#SBATCH --job-name=prism-cand_019_da26ffed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

CANDIDATE_ID="cand_019_da26ffed"
SDF_PATH="campaigns/glp1r_aleniglipron/track_a_generative/epoch019_fixes/gpu_dispatch/sdf/cand_019_da26ffed.sdf"
REPLICAS="10"
PROTOCOL="ccns_5phase"

echo "gpu_validation_start candidate=${CANDIDATE_ID} replicas=${REPLICAS} sdf=${SDF_PATH} protocol=${PROTOCOL}"
python3 scripts/run_ccns_validation_md.py \
  --candidate-id "${CANDIDATE_ID}" \
  --sdf "${SDF_PATH}" \
  --replicas "${REPLICAS}" \
  --protocol "${PROTOCOL}"
echo "gpu_validation_complete candidate=${CANDIDATE_ID}"

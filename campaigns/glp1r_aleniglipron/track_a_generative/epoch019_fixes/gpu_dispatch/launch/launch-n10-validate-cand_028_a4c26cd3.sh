#!/usr/bin/env bash
#SBATCH --job-name=prism-cand_028_a4c26cd3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

CANDIDATE_ID="cand_028_a4c26cd3"
SDF_PATH="campaigns/glp1r_aleniglipron/track_a_generative/epoch019_fixes/gpu_dispatch/sdf/cand_028_a4c26cd3.sdf"
REPLICAS="10"
PROTOCOL="ccns_5phase"

echo "gpu_validation_start candidate=${CANDIDATE_ID} replicas=${REPLICAS} sdf=${SDF_PATH} protocol=${PROTOCOL}"
python3 scripts/run_ccns_validation_md.py \
  --candidate-id "${CANDIDATE_ID}" \
  --sdf "${SDF_PATH}" \
  --replicas "${REPLICAS}" \
  --protocol "${PROTOCOL}"
echo "gpu_validation_complete candidate=${CANDIDATE_ID}"

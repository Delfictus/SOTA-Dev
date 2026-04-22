#!/bin/bash
# PRISM4D EGNN v4 Training on RunPod B200
# =========================================
# RunPod template: runpod/pytorch:2.8.0 (B200, sm_120 native)
#
# Setup:
#   1. Deploy B200 pod with runpod-torch-v280 template
#   2. SSH in: ssh root@<pod-ip> -p <ssh-port> -i ~/.ssh/id_ed25519
#   3. Upload this directory:
#      scp -P <port> -r runpod_training/ root@<pod-ip>:/workspace/
#   4. Run: cd /workspace/runpod_training && bash setup_and_train.sh
#
# B200 specs: 192GB HBM3e, sm_120, ~2.5 PFLOPS FP16
# Expected time: scPDB pretrain ~45min, BENCH30 fine-tune ~15min

set -e

WORKDIR="/workspace/runpod_training"
cd "$WORKDIR"

echo "============================================================"
echo "PRISM4D EGNN v4 Training Pipeline (B200)"
echo "============================================================"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# ── Step 1: Install dependencies ──
echo "[1/5] Installing dependencies..."
pip install -q fair-esm torch-geometric biopython pandas scikit-learn 2>&1 | tail -3

# Verify GPU works with PyTorch
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_mem // 1024**3}GB')
    x = torch.randn(1000, 1000, device='cuda')
    y = x @ x.T
    print(f'GPU compute test: OK ({y.shape})')
"

# ── Step 2: Download scPDB / PDBBind structures ──
echo "[2/5] Downloading training structures..."
python3 download_scpdb.py

# ── Step 3: Pretrain on scPDB ──
echo "[3/5] Pretraining on scPDB..."
# B200 has 192GB — can fit ESM-2 (2.5GB) + large batch comfortably
python3 egnn_pocket_ranker_v4.py train \
    --manifest scpdb_data/scpdb_manifest.json \
    --gt scpdb_data/scpdb_ground_truth.json \
    --sites-dir scpdb_data/sites \
    --apo-dir scpdb_data/pdbs \
    --epochs 100 \
    --lr 5e-4 \
    --dcc-threshold 4.0 \
    --n-models 5 \
    --model-dir pretrained_models \
    --esm-model esm2_t33_650M_UR50D 2>&1 | tee pretrain.log

# ── Step 4: Fine-tune on BENCH30 ──
echo "[4/5] Fine-tuning on BENCH30..."
if [ -d "bench30_data" ]; then
    # Initialize from pretrained weights if available
    if [ -d "pretrained_models" ]; then
        echo "  Loading pretrained weights..."
        cp pretrained_models/egnn_ranker_v4_m*.pt final_models/ 2>/dev/null || mkdir -p final_models
    fi

    python3 egnn_pocket_ranker_v4.py train \
        --manifest bench30_data/benchmark_manifest.json \
        --gt bench30_data/ligand_centroids.json \
        --sites-dir bench30_data/results \
        --apo-dir bench30_data/apo \
        --epochs 300 \
        --lr 1e-4 \
        --dcc-threshold 8.0 \
        --n-models 5 \
        --model-dir final_models \
        --esm-model esm2_t33_650M_UR50D 2>&1 | tee finetune.log
else
    echo "  SKIP: bench30_data/ not found."
    echo "  Package it locally: bash package_bench30.sh"
    echo "  Then upload: scp -P <port> bench30_data.tar.gz root@<pod-ip>:/workspace/runpod_training/"
    echo "  Then: cd /workspace/runpod_training && tar xzf bench30_data.tar.gz && bash setup_and_train.sh"
fi

# ── Step 5: Package results for download ──
echo "[5/5] Packaging results..."
tar czf /workspace/trained_models.tar.gz \
    pretrained_models/ \
    final_models/ \
    pretrain.log \
    finetune.log 2>/dev/null || true

echo ""
echo "============================================================"
echo "TRAINING COMPLETE"
echo "============================================================"
echo ""
echo "Download trained models:"
echo "  scp -P <port> root@<pod-ip>:/workspace/trained_models.tar.gz ."
echo ""
echo "Pretrained models: pretrained_models/"
echo "Fine-tuned models: final_models/"
echo "Logs: pretrain.log, finetune.log"
echo ""
echo "To use locally:"
echo "  tar xzf trained_models.tar.gz"
echo "  cp final_models/egnn_ranker_v4_m*.pt ~/Desktop/Prism4D-bio/models/"
echo "============================================================"

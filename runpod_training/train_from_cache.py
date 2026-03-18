#!/usr/bin/env python3
"""Train from pre-built dataset. Just: python3 train_from_cache.py"""
import sys, torch
sys.path.insert(0, '/workspace/prism4d/runpod_training')
from egnn_pocket_ranker_v4 import train_ensemble

d = torch.load('/workspace/training_dataset.pt', weights_only=False)
print(f"{len(d['graphs'])} graphs, {int(sum(d['labels']))} hits")
train_ensemble(d['graphs'], d['labels'], d['meta'],
               n_models=5, epochs=100, lr=3e-4,
               model_dir='/workspace/models')

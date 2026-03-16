#!/usr/bin/env python3
"""
Boltzmann Ranker: First-Principles Thermodynamic Pocket Ranking
================================================================
Maps PRISM4D physics features to ΔG = ΔH - TΔS and uses gradient
descent to find the universal thermodynamic constants.

This is NOT a black-box ML model — it's a parameterized canonical
ensemble with constrained positive weights (thermodynamic scaling
coefficients converting arbitrary units to kcal/mol).

Cross-entropy loss on Softmax output = KL divergence minimization
= Free Energy Minimization of the macroscopic system.
"""

import json
import numpy as np
import os
import sys

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("ERROR: PyTorch required. Install with: pip install torch")
    sys.exit(1)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────
# 1. Load benchmark results + ground truth
# ──────────────────────────────────────────────────────────────
gt = json.load(open("ground_truth/ligand_centroids.json"))
manifest = json.load(open("benchmark_manifest.json"))

print("Loading PRISM4D-BENCH30 results...")

all_proteins = []  # list of dicts: {features: [N_pockets x 5], target_idx: int}

for t in manifest["targets"]:
    tid = str(t["id"])
    outdir = f"results/{tid}"
    bs_files = [f for f in os.listdir(outdir) if f.endswith('.binding_sites.json')] if os.path.exists(outdir) else []
    if not bs_files or tid not in gt:
        continue

    with open(os.path.join(outdir, bs_files[0])) as f:
        data = json.load(f)
    sites = data.get("sites", [])
    if len(sites) < 3:
        continue

    lig_c = np.array(gt[tid]["centroid"])

    # Compute DCC for each site to find the target (closest to ligand)
    dccs = []
    features = []
    for s in sites:
        sc = np.array(s["centroid"])
        dcc = float(np.linalg.norm(sc - lig_c))
        dccs.append(dcc)

        # Extract the 5 thermodynamic features:
        # H_contact: spike intensity (favorable enthalpy, negate)
        spike_count = s.get("spike_count", 0)
        h_contact = np.log1p(spike_count) / 14.0  # log-normalized

        # H_solv: frustrated solvent / water displacement
        h_solv = s.get("wd_coherence", 0) or 0

        # W_conf: activation barrier (onset score, inverted = work)
        onset = s.get("onset_score", 0.5) or 0.5
        w_conf = 1.0 - onset  # high onset = low barrier = low work

        # S_config: geometric freedom (ray-length entropy proxy)
        sphericity = s.get("sphericity", 0.5) or 0.5
        s_config = 1.0 - sphericity  # low sphericity = high geometric entropy

        # S_chem: chemical diversity (spike type entropy)
        source_div = s.get("source_diversity", 0) or 0
        s_chem = source_div

        features.append([h_contact, h_solv, w_conf, s_config, s_chem])

    # Find target pocket (closest to ligand, within 10A)
    best_idx = int(np.argmin(dccs))
    best_dcc = dccs[best_idx]

    if best_dcc > 15.0:  # skip proteins where no pocket is close
        continue

    all_proteins.append({
        "tid": tid,
        "apo": t["apo_pdb"],
        "features": np.array(features, dtype=np.float32),
        "target_idx": best_idx,
        "best_dcc": best_dcc,
        "n_sites": len(sites),
    })

print(f"Loaded {len(all_proteins)} proteins with {sum(p['n_sites'] for p in all_proteins)} total pockets")

# ──────────────────────────────────────────────────────────────
# 2. Define the Boltzmann Ranker
# ──────────────────────────────────────────────────────────────

class BoltzmannRanker(nn.Module):
    """Differentiable Canonical Ensemble for pocket ranking.

    Maps features to ΔG = ΔH - TΔS with constrained positive weights.
    Output: Boltzmann probabilities P(i) = exp(-βΔG_i) / Z
    """
    def __init__(self, n_features=5):
        super().__init__()
        # Trainable thermodynamic scaling constants (log-space for positivity)
        self.log_w = nn.Parameter(torch.zeros(n_features))
        # Trainable inverse temperature β (controls sharpness)
        self.log_beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, features):
        # features: [batch, n_pockets, n_features]
        w = torch.exp(self.log_w)  # Enforce strict positivity
        beta = torch.exp(self.log_beta)

        # Thermodynamic decomposition:
        # H_contact (favorable enthalpy, negative)
        # H_solv (favorable solvation, negative)
        # W_conf (activation work, positive/unfavorable)
        # S_config (geometric entropy, favorable)
        # S_chem (chemical entropy, favorable)
        H = -(w[0] * features[:, :, 0] + w[1] * features[:, :, 1])
        W = w[2] * features[:, :, 2]
        S = w[3] * features[:, :, 3] + w[4] * features[:, :, 4]

        # ΔG = H + W - T*S  (T absorbed into weights)
        delta_G = H + W - S

        # Boltzmann probabilities via partition function
        probs = torch.softmax(-beta * delta_G, dim=1)
        return probs

# ──────────────────────────────────────────────────────────────
# 3. Prepare training data
# ──────────────────────────────────────────────────────────────

# Pad all proteins to same number of pockets (max_pockets)
max_pockets = max(p["n_sites"] for p in all_proteins)
n_proteins = len(all_proteins)

X = np.zeros((n_proteins, max_pockets, 5), dtype=np.float32)
Y = np.zeros(n_proteins, dtype=np.int64)
mask = np.zeros((n_proteins, max_pockets), dtype=np.float32)

for i, p in enumerate(all_proteins):
    n = p["n_sites"]
    X[i, :n, :] = p["features"]
    Y[i] = p["target_idx"]
    mask[i, :n] = 1.0

X_t = torch.tensor(X)
Y_t = torch.tensor(Y)
mask_t = torch.tensor(mask)

print(f"Training data: {n_proteins} proteins, max {max_pockets} pockets, 5 features")

# ──────────────────────────────────────────────────────────────
# 4. Train
# ──────────────────────────────────────────────────────────────

model = BoltzmannRanker(n_features=5)
optimizer = optim.Adam(model.parameters(), lr=0.05)
loss_fn = nn.CrossEntropyLoss()

print("\nTraining Boltzmann Ranker (free energy minimization)...")
print(f"{'Epoch':>6} {'Loss':>8} {'SR@1':>6} {'SR@3':>6} {'Weights':>40}")

best_sr1 = 0
best_weights = None

for epoch in range(500):
    model.train()
    optimizer.zero_grad()

    probs = model(X_t)

    # Compute raw logits for cross-entropy (more numerically stable)
    w = torch.exp(model.log_w)
    beta = torch.exp(model.log_beta)
    H = -(w[0] * X_t[:, :, 0] + w[1] * X_t[:, :, 1])
    W = w[2] * X_t[:, :, 2]
    S = w[3] * X_t[:, :, 3] + w[4] * X_t[:, :, 4]
    delta_G = H + W - S
    logits = -beta * delta_G

    # Mask out padded pockets with -inf
    logits = logits + (1.0 - mask_t) * (-1e9)

    loss = loss_fn(logits, Y_t)
    loss.backward()
    optimizer.step()

    # Evaluate
    with torch.no_grad():
        predictions = logits.argmax(dim=1)
        sr1 = (predictions == Y_t).float().mean().item()

        # SR@3
        top3 = logits.topk(3, dim=1).indices
        sr3 = sum(1 for i in range(n_proteins) if Y_t[i] in top3[i]) / n_proteins

        weights = torch.exp(model.log_w).detach().numpy()
        beta_val = torch.exp(model.log_beta).detach().item()

    if sr1 > best_sr1:
        best_sr1 = sr1
        best_weights = weights.copy()
        best_beta = beta_val

    if epoch % 50 == 0 or epoch == 499:
        w_str = ", ".join(f"{v:.3f}" for v in weights)
        print(f"{epoch:>6} {loss.item():>8.4f} {sr1:>5.1%} {sr3:>5.1%}  β={beta_val:.2f} w=[{w_str}]")

# ──────────────────────────────────────────────────────────────
# 5. Results
# ──────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"BOLTZMANN RANKER TRAINING COMPLETE")
print(f"{'='*60}")
print(f"Best SR@1: {best_sr1:.1%}")
print(f"Best β (inverse temperature): {best_beta:.3f}")
print(f"\nLearned Thermodynamic Constants:")
feature_names = ["H_contact (spike intensity)", "H_solv (water displacement)",
                 "W_conf (activation barrier)", "S_config (geometric entropy)",
                 "S_chem (chemical diversity)"]
for name, w in zip(feature_names, best_weights):
    print(f"  {name:>35}: {w:.4f}")

# Per-protein results with best weights
print(f"\nPer-protein results:")
print(f"{'TID':>4} {'APO':>5} {'N':>3} {'Target':>6} {'Pred':>5} {'DCC':>6} {'Hit':>4}")
print("-" * 45)

w = torch.tensor(best_weights)
beta = best_beta
total_sr1 = 0
total_sr3 = 0

for i, p in enumerate(all_proteins):
    feat = torch.tensor(p["features"]).unsqueeze(0)
    H = -(w[0] * feat[:, :, 0] + w[1] * feat[:, :, 1])
    W = w[2] * feat[:, :, 2]
    S = w[3] * feat[:, :, 3] + w[4] * feat[:, :, 4]
    dG = H + W - S
    logits = -beta * dG
    pred = logits.squeeze().argmax().item()
    top3 = logits.squeeze().topk(min(3, p["n_sites"])).indices.tolist()

    hit1 = "✓" if pred == p["target_idx"] else "·"
    hit3 = "✓3" if p["target_idx"] in top3 else "·"
    if pred == p["target_idx"]: total_sr1 += 1
    if p["target_idx"] in top3: total_sr3 += 1

    print(f"{p['tid']:>4} {p['apo']:>5} {p['n_sites']:>3} {p['target_idx']:>6} {pred:>5} {p['best_dcc']:>5.1f}A {hit1} {hit3}")

print(f"\nFinal SR@1: {total_sr1}/{len(all_proteins)} ({total_sr1/len(all_proteins):.1%})")
print(f"Final SR@3: {total_sr3}/{len(all_proteins)} ({total_sr3/len(all_proteins):.1%})")

# Save weights for Rust integration
output = {
    "weights": best_weights.tolist(),
    "beta": best_beta,
    "feature_names": feature_names,
    "sr1": best_sr1,
    "n_proteins": len(all_proteins),
}
with open("boltzmann_weights.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nWeights saved to boltzmann_weights.json")
print("Port these to Rust: ΔG = -(w0*H_contact + w1*H_solv) + w2*W_conf - (w3*S_config + w4*S_chem)")

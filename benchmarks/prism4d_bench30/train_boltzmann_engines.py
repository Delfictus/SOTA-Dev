#!/usr/bin/env python3
"""
Boltzmann Engine Ranker: Train on 4 Cobb-Douglas engine scores.
Instead of 15 raw features, train on the pre-computed engine outputs:
  geo_score, chem_score, phys_score, vcs_score
These already capture nonlinear interactions via the multi-head
Cobb-Douglas engines. The Boltzmann weights learn the optimal
WEIGHTING of the engines, not the raw features.
"""
import json, os, sys, time
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from jax import grad, jit, vmap
    from jax.scipy.special import logsumexp
except ImportError:
    print("ERROR: JAX required. Install with: pip install jax[cuda12]")
    sys.exit(1)

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print(f"JAX {jax.__version__}, {jax.devices()}")

gt = json.load(open("ground_truth/ligand_centroids.json"))
manifest = json.load(open("benchmark_manifest.json"))

print("Loading engine scores from V9 Stage 1 results...")
proteins = []
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
    dccs, feats = [], []
    for s in sites:
        sc = np.array(s["centroid"])
        dccs.append(float(np.linalg.norm(sc - lig_c)))
        # 4 engine scores — use defaults if not exported yet
        geo = s.get("engine_geo", 0.5)
        chem = s.get("engine_chem", 0.5)
        phys = s.get("engine_phys", 0.5)
        vcs = s.get("engine_vcs", 0.3)
        feats.append([geo or 0.5, chem or 0.5, phys or 0.5, vcs or 0.3])

    best_idx = int(np.argmin(dccs))
    if dccs[best_idx] > 15.0:
        continue
    proteins.append({
        "tid": tid, "apo": t["apo_pdb"],
        "features": np.array(feats, dtype=np.float32),
        "target_idx": best_idx, "best_dcc": dccs[best_idx],
        "n_sites": len(sites),
    })

N_FEAT = 4
max_p = max(p["n_sites"] for p in proteins)
n_prot = len(proteins)
X = np.zeros((n_prot, max_p, N_FEAT), dtype=np.float32)
Y = np.zeros(n_prot, dtype=np.int32)
mask = np.full((n_prot, max_p), -1e9, dtype=np.float32)
for i, p in enumerate(proteins):
    n = p["n_sites"]
    X[i, :n, :] = p["features"]
    Y[i] = p["target_idx"]
    mask[i, :n] = 0.0

X_j, Y_j, mask_j = jnp.array(X), jnp.array(Y), jnp.array(mask)
print(f"Loaded {n_prot} proteins, max {max_p} pockets, {N_FEAT} engine features")

# Check if engines are exported
has_engines = any(
    proteins[0]["features"][0][0] != 0.5 or
    proteins[0]["features"][0][1] != 0.5
    for _ in [1]
)
if not has_engines:
    print("\nWARNING: engine_geo/chem/phys/vcs not found in JSON — using defaults (0.5).")
    print("Run benchmark first with the updated binary to export engine scores.")
    print("Proceeding with proxy values...\n")

def init_params():
    return {"log_w": jnp.zeros(N_FEAT), "log_beta": jnp.array(0.5)}

@jit
def loss_fn(params, X, Y, mask):
    w = jnp.exp(params["log_w"])
    beta = jnp.exp(params["log_beta"])
    # ΔG = -(w0*geo + w1*chem + w2*phys + w3*vcs)
    # More negative = more favorable (higher engines = better pocket)
    delta_G = -jnp.sum(w * X, axis=2)
    logits = -beta * delta_G + mask
    log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
    nll = -jnp.mean(log_probs[jnp.arange(len(Y)), Y])
    reg = 0.001 * jnp.sum(jnp.exp(params["log_w"]) ** 2)
    return nll + reg

grad_fn = jit(grad(loss_fn))

from jax.example_libraries.optimizers import adam
opt_init, opt_update, get_params = adam(0.05)
params = init_params()
opt_state = opt_init(params)

print(f"\nTraining 4-Engine Boltzmann Ranker...")
print(f"{'Ep':>5} {'Loss':>7} {'SR@1':>6} {'SR@3':>6} {'SR@5':>6} {'w_geo':>6} {'w_chem':>7} {'w_phys':>7} {'w_vcs':>6}")

best_sr1 = 0
best_params = None
t0 = time.time()

for epoch in range(2000):
    grads = grad_fn(params, X_j, Y_j, mask_j)
    opt_state = opt_update(epoch, grads, opt_state)
    params = get_params(opt_state)

    if epoch % 100 == 0 or epoch == 1999:
        loss = float(loss_fn(params, X_j, Y_j, mask_j))
        w = jnp.exp(params["log_w"])
        beta = jnp.exp(params["log_beta"])
        delta_G = -jnp.sum(w * X_j, axis=2)
        logits = -beta * delta_G + mask_j
        preds = jnp.argmax(logits, axis=1)
        sr1 = float(jnp.mean(preds == Y_j))
        top3 = jnp.argsort(-logits, axis=1)[:, :3]
        sr3 = sum(1 for i in range(n_prot) if Y_j[i] in top3[i]) / n_prot
        top5 = jnp.argsort(-logits, axis=1)[:, :5]
        sr5 = sum(1 for i in range(n_prot) if Y_j[i] in top5[i]) / n_prot
        wn = np.array(w)
        print(f"{epoch:>5} {loss:>7.3f} {sr1:>5.1%} {sr3:>5.1%} {sr5:>5.1%} {wn[0]:>6.3f} {wn[1]:>7.3f} {wn[2]:>7.3f} {wn[3]:>6.3f}")
        if sr1 >= best_sr1:
            best_sr1 = sr1
            best_params = jax.tree.map(lambda x: np.array(x), params)

print(f"\nTrained in {time.time()-t0:.1f}s")

w_best = np.exp(best_params["log_w"])
beta_best = float(np.exp(best_params["log_beta"]))

print(f"\n{'='*50}")
print(f"4-ENGINE BOLTZMANN WEIGHTS")
print(f"{'='*50}")
print(f"  Geo:  {w_best[0]:.4f}")
print(f"  Chem: {w_best[1]:.4f}")
print(f"  Phys: {w_best[2]:.4f}")
print(f"  VCS:  {w_best[3]:.4f}")
print(f"  β:    {beta_best:.4f}")
print(f"  SR@1: {best_sr1:.1%}")

# Per-protein breakdown
print(f"\n{'TID':>4} {'APO':>5} {'Tgt':>4} {'Pred':>5} {'DCC':>6} {'@1':>3} {'@3':>3}")
w = jnp.array(w_best)
beta = beta_best
total_sr1 = total_sr3 = 0
for i, p in enumerate(proteins):
    feat = jnp.array(p["features"]).reshape(1, -1, N_FEAT)
    m = jnp.array(np.where(np.arange(max_p) < p["n_sites"], 0.0, -1e9).astype(np.float32)).reshape(1, -1)
    dG = -jnp.sum(w * feat, axis=2)
    logits = (-beta * dG + m).squeeze()
    pred = int(jnp.argmax(logits))
    top3 = set(jnp.argsort(-logits)[:3].tolist())
    h1 = "✓" if pred == p["target_idx"] else "·"
    h3 = "✓" if p["target_idx"] in top3 else "·"
    if pred == p["target_idx"]: total_sr1 += 1
    if p["target_idx"] in top3: total_sr3 += 1
    print(f"{p['tid']:>4} {p['apo']:>5} {p['target_idx']:>4} {pred:>5} {p['best_dcc']:>5.1f}A  {h1}   {h3}")

print(f"\nFinal: SR@1={total_sr1}/{n_prot} ({total_sr1/n_prot:.1%}), SR@3={total_sr3}/{n_prot} ({total_sr3/n_prot:.1%})")

output = {
    "engine_weights": {"geo": float(w_best[0]), "chem": float(w_best[1]),
                       "phys": float(w_best[2]), "vcs": float(w_best[3])},
    "beta": beta_best, "sr1": best_sr1, "n_proteins": n_prot,
}
with open("boltzmann_engine_weights.json", "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved to boltzmann_engine_weights.json")

"""
Tier 2: Wasserstein-2 distribution alignment via Sinkhorn-regularized
optimal transport. Replaces single-frame Kabsch with measure-theoretic
alignment of the full spike-density cloud against reference backbone.

Pure post-processor module. NO_POST_MD_LOOPS-compliant.

Mathematical formulation:
  Find rotation R, translation t minimizing W_2^2(T_{R,t}#mu_spike, mu_ref)
  where mu_spike is the intensity-weighted spike point cloud and mu_ref is
  the reference protein backbone (CA atoms or all heavy atoms).

Solved by alternating: (1) Sinkhorn on optimal transport plan given (R,t),
(2) Procrustes on the transport plan to update (R,t). Converges in 5-10
outer iterations for typical Prism-4D substrate sizes.
"""
import numpy as np
import ot  # POT library


def downsample_spike_cloud(spike_xyz, spike_weights, n_target=2000, seed=42):
    """
    Importance-sample n_target spikes weighted by intensity.
    Sinkhorn complexity is O(n_spike * n_ref); cap n_spike at 2000 for
    tractability on 5M-spike Prism-4D substrates.
    """
    rng = np.random.default_rng(seed)
    n = len(spike_xyz)
    if n <= n_target:
        # Even when not downsampling, return normalized uniform weights for
        # Sinkhorn (POT requires distributions that sum to 1).
        if spike_weights.sum() <= 0:
            w = np.full(n, 1.0 / max(n, 1))
        else:
            w = spike_weights / spike_weights.sum()
        return spike_xyz, w
    p = spike_weights / spike_weights.sum()
    idx = rng.choice(n, size=n_target, replace=False, p=p)
    return spike_xyz[idx], np.full(n_target, 1.0 / n_target)


def sinkhorn_procrustes_align(spike_xyz, spike_weights, ref_xyz,
                               n_iter=10, sinkhorn_reg=1.0,
                               sinkhorn_tol=1e-3, max_sinkhorn_iter=500):
    """
    Iterate Sinkhorn-Procrustes to align spike cloud to ref backbone.
    Returns (R, t, final_W2_distance, transport_plan, history).
    """
    src_xyz, src_weights = downsample_spike_cloud(spike_xyz, spike_weights)
    n_ref = len(ref_xyz)
    ref_weights = np.full(n_ref, 1.0 / n_ref)

    # Initialize R, t via centroid alignment.
    src_com = np.average(src_xyz, axis=0, weights=src_weights)
    ref_com = ref_xyz.mean(axis=0)
    t = ref_com - src_com
    R = np.eye(3)

    history = []

    for outer in range(n_iter):
        # Apply current R, t.
        src_transformed = src_xyz @ R.T + t

        # Sinkhorn: compute transport plan between src and ref.
        cost_matrix = ot.dist(src_transformed, ref_xyz, metric="sqeuclidean")
        cost_matrix /= max(cost_matrix.max(), 1e-12)  # Normalize for stability.
        transport = ot.sinkhorn(
            src_weights, ref_weights, cost_matrix,
            reg=sinkhorn_reg, stopThr=sinkhorn_tol,
            numItermax=max_sinkhorn_iter,
        )

        # Procrustes: given transport plan, find new R, t.
        # Effective ref position for each src point: weighted mean of ref points.
        row_sums = transport.sum(axis=1, keepdims=True)
        ref_assignments = transport @ ref_xyz / (row_sums + 1e-12)

        # Solve Procrustes: min ||src @ R.T + t - ref_assignments||.
        src_centered = src_xyz - src_com
        ref_centered = ref_assignments - ref_assignments.mean(axis=0)
        H = src_centered.T @ ref_centered
        U, S, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R_new = Vt.T @ D @ U.T
        t_new = ref_assignments.mean(axis=0) - src_com @ R_new.T

        # Compute current W2 distance.
        w2_sq = float(np.sum(transport * cost_matrix))

        history.append({"iter": outer, "w2_sq": w2_sq})

        # Convergence check.
        delta = float(np.linalg.norm(R_new - R) + np.linalg.norm(t_new - t))
        R, t = R_new, t_new
        if delta < 1e-4 and outer > 2:
            break

    return R, t, float(history[-1]["w2_sq"]), transport, history


def wasserstein_align_reference(spike_df, ref_pdb_path, ref_het_id):
    """
    Align Prism-4D spike cloud to one reference holo. Returns:
      {
        "status": "OK" | "BLOCKED",
        "R": rotation, "t": translation,
        "w2_distance": final Wasserstein-2 distance,
        "n_outer_iters": iterations to convergence,
        "aligned_ligand": ligand atoms transformed into spike frame,
        "history": iteration history
      }

    `spike_df` must be a pandas-DataFrame-like object with columns
    [x, y, z, intensity]. Accessed via `[col].values` so any object
    with that interface works.
    """
    if len(spike_df) == 0:
        return {"status": "BLOCKED", "gate": "No spikes provided"}

    spike_xyz = spike_df[["x", "y", "z"]].values
    spike_w = spike_df["intensity"].values
    spike_w = np.asarray(spike_w, dtype=np.float64)
    if spike_w.sum() <= 0:
        return {"status": "BLOCKED", "gate": "Spike weights sum to zero"}

    # Parse reference: backbone CA + ligand atoms.
    ref_ca = []
    ref_lig = []
    with open(ref_pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                atom_name = line[12:16].strip()
                alt_loc = line[16:17].strip()
                res_name = line[17:20].strip()
                chain = line[21:22].strip()
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                element = line[76:78].strip() if len(line) > 76 else atom_name[0]
            except (ValueError, IndexError):
                continue
            if element == "H" or atom_name.startswith("H"):
                continue
            if alt_loc not in ("", "A"):
                continue
            if chain != "A":
                continue
            if line.startswith("ATOM") and atom_name == "CA":
                ref_ca.append([x, y, z])
            elif line.startswith("HETATM") and res_name == ref_het_id:
                ref_lig.append([x, y, z])

    if not ref_ca or not ref_lig:
        return {"status": "BLOCKED", "gate": "Insufficient ref atoms"}

    ref_ca_xyz = np.asarray(ref_ca)
    ref_lig_xyz = np.asarray(ref_lig)

    # Run Sinkhorn-Procrustes.
    R, t, w2, transport, history = sinkhorn_procrustes_align(
        spike_xyz, spike_w, ref_ca_xyz,
    )

    # Transform ligand into spike frame: spike was aligned to ref via
    # x' = R @ x + t, so the inverse maps ref → spike via
    # x = R^T @ (x' - t).
    aligned_ligand = (ref_lig_xyz - t) @ R

    return {
        "status": "OK",
        "R": R.tolist(), "t": t.tolist(),
        "w2_distance": w2,
        "n_outer_iters": len(history),
        "aligned_ligand": aligned_ligand.tolist(),
        "history": history,
    }

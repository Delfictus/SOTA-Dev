#!/usr/bin/env python3
"""PRISM-4D consolidated feature extractor — directive Phase 0.1.

Produces one .npz per target containing all arrays needed for teacher + VN-EGNN
training + XGBoost ranker.

  structural      [N_res, 25]   subset of Block 1 (AA one-hot 20 + DSSP 3 + SASA + hydro + B-factor)  NOTE: cached .npy uses 26 dims
  nma             [N_res, 26]   Block 2
  perturbed_nma   [N_res, 5]    computed from NMA + ligand centroid
  physics_216     [N_res, 216]  FULL reconstruction from /mnt/storage/prism-outputs/ml/egnn/extract_216.py
  tide_residue    [N_res, 7]    transfer_entropy, causal_dg, fisher_info, kl_divergence, role_trigger, role_responder, n_causal_spikes
  coords          [N_res, 3]    Cα coordinates
  sequence        str           1-letter sequence
  site_features   dict          per-site feature vectors for XGBoost
  tide_pocket     dict          per-pocket TIDE features for XGBoost
  residue_ids     [N_res]       canonical residue IDs (topology-indexed)
  labels          [N_res]       binding labels (1 if Cα within 4.5Å of ligand)
  ligand_centroid [3]           from ground_truth.json

INPUT: target name + run_dir containing engine outputs (on local disk).
The run_dir must contain these files — if extracting from R2, stage first:

    {target}.topology.json
    {target}_clean.pdb
    {target}.binding_sites.json
    {target}.kcc_visualization.json
    {target}.topology.prism_therm.json   (directive spec — NOT .prism_therm.json)
    {target}.site*.spike_events.parquet  (or .json)
    {target}_stream00.ensemble_trajectory.pdb
    {target}_ground_truth.json           (for labels)
    {target}_nma.pt                      (optional NMA cache — looks in EGNN_DIR if absent)

The physics_216 extractor is imported verbatim from the original production
module so reconstructions are bit-identical to the cached 216-npy files.

Usage:
    python3 extract_all_features.py --target 1a4u --run-dir /path/to/run \
        --out-dir /workspace/features
    python3 extract_all_features.py --batch /mnt/storage/prism-outputs/runs/cryptobench199 \
        --out-dir /mnt/storage/spike-audit/features-pct95 --max 10
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  Import the production 216-dim extractor verbatim
# ─────────────────────────────────────────────────────────────

EXTRACT_216_DIR = Path("/mnt/storage/prism-outputs/ml/egnn")
if EXTRACT_216_DIR.exists():
    sys.path.insert(0, str(EXTRACT_216_DIR))

try:
    import extract_216 as ex216
    HAVE_EX216 = True
except ImportError:
    HAVE_EX216 = False


# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────

AA3TO1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E',
    'GLY':'G','HIS':'H','HID':'H','HIE':'H','HIP':'H','ILE':'I','LEU':'L',
    'LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W',
    'TYR':'Y','VAL':'V','CYX':'C','MSE':'M',
}

TIDE_ROLES_TRIGGER = {"TRIGGER"}
TIDE_ROLES_RESPONDER = {"RESPONDER", "GATEWAY", "STABIL.", "SPECTAT"}  # everything non-TRIGGER
LIGAND_CUTOFF_A = 4.5

SITE_FIELD_LIST = [
    "spike_count", "n_streams", "persistence", "unsat_frac", "spread",
    "burial", "spike_density", "volume", "n_lining_residues",
    "hydrophobic_ratio", "aromatic_count", "charged_count", "polar_count",
    "druggability", "quality_score", "sphericity", "onset_score",
    "aromatic_score", "frustrated_solvent_score", "source_diversity",
    "wd_coherence", "burial_score",
]


# ─────────────────────────────────────────────────────────────
#  Perturbed-NMA (5 dims)
# ─────────────────────────────────────────────────────────────

def compute_perturbed_nma(nma_pt_path: Optional[Path],
                          ligand_centroid: Optional[np.ndarray],
                          ca_pos: Dict[int, np.ndarray],
                          resid_list: List[int]) -> np.ndarray:
    """Compute 5 perturbed-NMA features from cached NMA + ligand direction."""
    import torch
    n = len(resid_list)
    out = np.zeros((n, 5), dtype=np.float32)
    if nma_pt_path is None or not nma_pt_path.exists() or ligand_centroid is None:
        return out
    try:
        nma = torch.load(nma_pt_path, weights_only=False)
    except Exception:
        return out

    md = nma.get("mode_displacements")
    if md is None:
        return out
    md = md.numpy() if hasattr(md, "numpy") else np.asarray(md)
    md = np.nan_to_num(md, nan=0.0)
    if md.ndim != 2 or md.shape[1] < 2:
        return out

    mode_vec = nma.get("mode_vectors")
    mode_vec = mode_vec.numpy() if hasattr(mode_vec, "numpy") else (
        np.asarray(mode_vec) if mode_vec is not None else None)

    for i, rid in enumerate(resid_list):
        if i >= md.shape[0]:
            break
        a1 = float(md[i, 0])
        a2 = float(md[i, 1])
        out[i, 0] = a1 / max(a2, 1e-6)                                  # amplitude_ratio

        ca = ca_pos.get(rid)
        if ca is None:
            continue
        dir_to_lig = ligand_centroid - ca
        nd = float(np.linalg.norm(dir_to_lig))
        if nd < 1e-6:
            continue
        dir_u = dir_to_lig / nd

        if mode_vec is not None and mode_vec.ndim == 3 and i < mode_vec.shape[0]:
            v1 = mode_vec[i, 0]  # [3]
            nv = float(np.linalg.norm(v1))
            if nv > 1e-6:
                v1 /= nv
                out[i, 1] = float(abs(np.dot(v1, dir_u)))                # alignment
                out[i, 2] = float(a1 * out[i, 1])                        # centroid_shift proxy
                out[i, 3] = float(np.dot(v1, dir_u))                     # ligand_cosine (signed)

        eig = nma.get("eigenvalues")
        if eig is not None:
            eig_np = eig.numpy() if hasattr(eig, "numpy") else np.asarray(eig)
            if len(eig_np) > 1:
                out[i, 4] = float(eig_np[1] / max(eig_np[0], 1e-6))      # variance_ratio

    return out


# ─────────────────────────────────────────────────────────────
#  TIDE residue features (7 dims)
# ─────────────────────────────────────────────────────────────

def compute_tide_residue(therm_path: Optional[Path], resid_list: List[int]
                         ) -> Tuple[np.ndarray, Dict[int, Dict]]:
    """Returns (tide [N,7], per_pocket_tide dict).

    Residue-ID mapping:
      prism_therm.json top_residues use the engine's internal residue numbering,
      which matches topology.json for most targets but NOT when topology uses
      absolute PDB residue IDs (e.g. 1nd7: topology=[543..916], therm=[0..373]).

      Fix: if direct rid matching fails, try index-based mapping — i.e., treat
      the therm rid as a 0-indexed position into resid_list. Pick whichever
      strategy yields more matches.
    """
    n = len(resid_list)
    out = np.zeros((n, 7), dtype=np.float32)
    pockets_out: Dict[int, Dict] = {}
    if therm_path is None or not therm_path.exists():
        return out, pockets_out

    try:
        d = json.load(open(therm_path))
    except Exception:
        return out, pockets_out

    # Direct ID lookup
    rid_to_idx_direct = {rid: i for i, rid in enumerate(resid_list)}
    # Index-based lookup (treat therm rid as 0-indexed position)
    def _index_based(therm_rid: int) -> Optional[int]:
        if 0 <= therm_rid < len(resid_list):
            return therm_rid
        return None

    # Decide strategy: count direct-match successes vs index-based successes
    # across all top_residues. Use whichever hits more.
    n_direct_hits = 0
    n_index_hits = 0
    for p in d.get("pockets", []):
        for r in p.get("top_residues", []):
            rid = r.get("residue_id")
            if rid in rid_to_idx_direct:
                n_direct_hits += 1
            if _index_based(rid) is not None:
                n_index_hits += 1
    use_index_mapping = (n_index_hits > n_direct_hits)

    # Per-pocket dict (for XGBoost)
    for p in d.get("pockets", []):
        pid = p.get("pocket_id")
        if pid is None:
            continue
        pockets_out[int(pid)] = {
            "centroid": p.get("centroid"),
            "ccns_tau": float(p.get("ccns_tau", 0.0) or 0.0),
            "ccns_class": str(p.get("ccns_class", "")),
            "druggability_score": float(p.get("druggability_score", 0.0) or 0.0),
            "hysteresis_asymmetry": float(p.get("hysteresis_asymmetry", 0.0) or 0.0),
            "relative_asymmetry": float(p.get("relative_asymmetry", 0.0) or 0.0),
            "is_cryptic": bool(p.get("is_cryptic", False)),
            "therm_class": str(p.get("therm_class", "")),
        }

    def _resolve(rid: int) -> Optional[int]:
        if use_index_mapping:
            return _index_based(rid)
        return rid_to_idx_direct.get(rid)

    # Residue dedup: take MAX transfer_entropy across pockets for each residue
    # Role: use role from the pocket with the highest transfer_entropy for that residue
    best = {}  # internal_idx -> (te, dg, fi, kl, role, n_spikes)
    for p in d.get("pockets", []):
        for r in p.get("top_residues", []):
            rid = r.get("residue_id")
            idx = _resolve(rid)
            if idx is None:
                continue
            te = float(r.get("transfer_entropy", 0.0) or 0.0)
            cur = best.get(idx)
            if cur is None or te > cur[0]:
                best[idx] = (te,
                             float(r.get("causal_dg", 0.0) or 0.0),
                             float(r.get("fisher_info", 0.0) or 0.0),
                             float(r.get("kl_divergence", 0.0) or 0.0),
                             str(r.get("role", "") or ""),
                             int(r.get("n_causal_spikes", 0) or 0))

    # Normalize n_causal_spikes by max
    if best:
        n_spikes_vals = [v[5] for v in best.values()]
        max_spk = max(n_spikes_vals) if n_spikes_vals else 0
    else:
        max_spk = 0

    for idx, (te, dg, fi, kl, role, n_spk) in best.items():
        out[idx, 0] = te
        out[idx, 1] = dg
        out[idx, 2] = fi
        out[idx, 3] = kl
        out[idx, 4] = 1.0 if role in TIDE_ROLES_TRIGGER else 0.0
        out[idx, 5] = 1.0 if role in TIDE_ROLES_RESPONDER else 0.0
        out[idx, 6] = n_spk / max_spk if max_spk > 0 else 0.0

    return out, pockets_out


# ─────────────────────────────────────────────────────────────
#  Site features for XGBoost (per-site, from binding_sites.json)
# ─────────────────────────────────────────────────────────────

def compute_site_features(binding_sites_path: Optional[Path],
                           tide_pockets: Dict[int, Dict]
                           ) -> Dict[str, Dict[str, Any]]:
    """Returns dict: site_name → {feature_name: value}."""
    out: Dict[str, Dict[str, Any]] = {}
    if binding_sites_path is None or not binding_sites_path.exists():
        return out
    try:
        bs = json.load(open(binding_sites_path))
    except Exception:
        return out

    sites = bs.get("sites", [])
    if not sites:
        return out

    # Pocket centroid → index for matching
    pocket_list = [(pid, np.asarray(pd["centroid"], dtype=np.float32))
                   for pid, pd in tide_pockets.items()
                   if pd.get("centroid") and len(pd["centroid"]) == 3]

    for s in sites:
        sid = s.get("id", "unknown")
        name = f"site{sid}"
        row: Dict[str, Any] = {}
        for k in SITE_FIELD_LIST:
            row[k] = s.get(k, None)
        # Derived
        row["n_lining_residues"] = len(s.get("lining_residues") or [])
        vol = s.get("volume") or s.get("volume_angstrom3") or 0.0
        row["volume"] = float(vol)
        spk = s.get("spike_count") or 0
        if vol and vol > 0 and spk:
            row["spike_density"] = float(spk) / (float(vol) ** (1/3 + 1/3))  # matches legacy approx
        else:
            row["spike_density"] = None

        # TIDE attribution by nearest-centroid < 5Å
        centroid = s.get("centroid")
        if centroid and len(centroid) == 3 and pocket_list:
            c = np.asarray(centroid, dtype=np.float32)
            best_pid, best_d = None, 1e9
            for pid, pc in pocket_list:
                d = float(np.linalg.norm(c - pc))
                if d < best_d:
                    best_d, best_pid = d, pid
            if best_pid is not None and best_d < 5.0:
                pd = tide_pockets[best_pid]
                row["matched_pocket_id"] = best_pid
                row["match_distance"] = best_d
                row["ccns_tau"] = pd["ccns_tau"]
                row["ccns_class"] = pd["ccns_class"]
                row["hysteresis_asymmetry"] = pd["hysteresis_asymmetry"]
                row["relative_asymmetry"] = pd["relative_asymmetry"]
                row["is_cryptic"] = pd["is_cryptic"]
                row["therm_class"] = pd["therm_class"]
                row["tide_druggability"] = pd["druggability_score"]
            else:
                row["matched_pocket_id"] = None

        out[name] = row
    return out


# ─────────────────────────────────────────────────────────────
#  Binding labels from ligand centroid
# ─────────────────────────────────────────────────────────────

def load_ground_truth(gt_path: Optional[Path]) -> Optional[np.ndarray]:
    if gt_path is None or not gt_path.exists():
        return None
    try:
        gt = json.load(open(gt_path))
    except Exception:
        return None
    if not gt.get("valid_for_dcc_validation"):
        return None
    c = gt.get("ligand_centroid")
    if c and len(c) == 3:
        return np.asarray(c, dtype=np.float32)
    return None


def compute_labels(ca_pos: Dict[int, np.ndarray], resid_list: List[int],
                    ligand_centroid: Optional[np.ndarray]) -> np.ndarray:
    n = len(resid_list)
    out = np.zeros(n, dtype=np.int32)
    if ligand_centroid is None:
        return out
    for i, rid in enumerate(resid_list):
        ca = ca_pos.get(rid)
        if ca is None:
            continue
        d = float(np.linalg.norm(ca - ligand_centroid))
        out[i] = 1 if d <= LIGAND_CUTOFF_A else 0
    return out


# ─────────────────────────────────────────────────────────────
#  Core extraction
# ─────────────────────────────────────────────────────────────

def extract_one(target: str, run_dir: Path,
                nma_candidates: List[Path],
                out_dir: Path) -> Dict[str, Any]:
    """Run the full pipeline on one target. Returns a status dict."""
    status: Dict[str, Any] = {"target": target, "run_dir": str(run_dir), "ok": False}
    if not HAVE_EX216:
        status["error"] = "extract_216 module not importable"
        return status

    try:
        info = ex216.get_topology_info(run_dir)
    except Exception as e:
        status["error"] = f"topology parse failed: {e}"
        return status
    if not info:
        status["error"] = "no topology"
        return status

    n_res = info["n_res"]
    resid_list = info["resid_list"]
    ca_pos = info["ca_pos"]

    # PDB (first match — consistent with extract_216.py behavior)
    pdb_files = list(run_dir.glob("*_clean.pdb"))
    pdb_path = pdb_files[0] if pdb_files else None

    # NMA: find cached .pt. extract_216.block2 and block7 look in CB_EGNN/EGNN_DIR
    # for `{run_dir.name}_nma.pt`. We can pre-stage a symlink or rely on the glob.
    nma_pt = None
    for cand_dir in nma_candidates:
        test = cand_dir / f"{target}_nma.pt"
        if test.exists():
            nma_pt = test
            break

    # Ground truth → labels + perturbed NMA direction
    gt_path = run_dir / f"{target}_ground_truth.json"
    ligand_c = load_ground_truth(gt_path)

    # Extract all 7 blocks exactly like the original extractor
    t0 = time.time()
    try:
        b1 = ex216.extract_block1(info["residues"], pdb_path, n_res)           # 26
        b2 = ex216.extract_block2(target, n_res)                               # 26
        b3 = ex216.extract_block3(run_dir, resid_list, n_res)                  # 13
        b4 = ex216.extract_block4(run_dir, resid_list, n_res)                  # 4
        b5 = ex216.extract_block5(run_dir, resid_list, n_res)                  # 18
        b6, phase_data = ex216.extract_block6(run_dir, ca_pos, resid_list, n_res)  # 48
        b7 = ex216.extract_block7(run_dir, ca_pos, resid_list, n_res,
                                   block6_phase_data=phase_data)               # 81
    except Exception as e:
        status["error"] = f"block extraction failed: {e}"
        status["traceback"] = traceback.format_exc()
        return status

    raw_216 = np.concatenate([b1, b2, b3, b4, b5, b6, b7], axis=1)
    if raw_216.shape[1] != 216:
        status["error"] = f"raw dim {raw_216.shape[1]} != 216"
        return status

    # Normalize exactly as original
    norm_cols = list(range(20, 21)) + list(range(24, 216))
    raw_216[:, norm_cols] = ex216.robust_normalize(raw_216[:, norm_cols])
    raw_216 = np.nan_to_num(raw_216, nan=0.0, posinf=0.0, neginf=0.0)
    raw_216 = raw_216.astype(np.float32)

    # Extract directive-aligned subsets from the 216 result
    # Block 1 (26) → structural subset (25) per directive: drop B-factor (col 25)
    # Per directive Phase 0.1: structural=25. Block1 layout: AA(0-19) + hydro(20) + DSSP(21-23) + SASA(24) + Bfac(25)
    # The directive's "25" likely means dropping B-factor. Per spec, structural[N,25] = cols 0..24.
    structural_25 = raw_216[:, :25].copy()
    # NMA (26) = raw_216[:, 26:52]
    nma_26 = raw_216[:, 26:52].copy()

    # Perturbed NMA (5) — computed separately from NMA cache
    perturbed_5 = compute_perturbed_nma(nma_pt, ligand_c, ca_pos, resid_list)

    # TIDE residue (7) + TIDE pocket dict
    therm_path = run_dir / f"{target}.topology.prism_therm.json"
    if not therm_path.exists():
        # Legacy filename
        therm_path_alt = list(run_dir.glob("*.prism_therm.json"))
        therm_path = therm_path_alt[0] if therm_path_alt else None
    tide_7, tide_pockets = compute_tide_residue(therm_path, resid_list)

    # Site features (per-site dict) — for XGBoost
    bs_path = run_dir / f"{target}.binding_sites.json"
    if not bs_path.exists():
        bs_list = list(run_dir.glob("*.binding_sites.json"))
        bs_path = bs_list[0] if bs_list else None
    site_feat_dict = compute_site_features(bs_path, tide_pockets)

    # Coordinates aligned to resid_list
    coords = np.zeros((n_res, 3), dtype=np.float32)
    for i, rid in enumerate(resid_list):
        c = ca_pos.get(rid)
        if c is not None:
            coords[i] = c

    # Sequence
    sequence = "".join(AA3TO1.get(r["residue_name"], "X") for r in info["residues"])

    # Labels
    labels = compute_labels(ca_pos, resid_list, ligand_c)

    # Save .npz
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{target}_features.npz"
    np.savez_compressed(
        out_path,
        structural=structural_25,
        nma=nma_26,
        perturbed_nma=perturbed_5,
        physics_216=raw_216,          # Full 216 matrix (as produced by original extractor)
        tide_residue=tide_7,
        coords=coords,
        sequence=np.asarray(sequence),
        residue_ids=np.asarray(resid_list, dtype=np.int64),
        labels=labels,
        ligand_centroid=(ligand_c if ligand_c is not None else np.zeros(3, dtype=np.float32)),
        site_features_json=np.asarray(json.dumps(site_feat_dict, default=str)),
        tide_pocket_json=np.asarray(json.dumps(tide_pockets, default=str)),
    )

    status.update({
        "ok": True,
        "out_path": str(out_path),
        "n_res": n_res,
        "n_positive": int(labels.sum()),
        "has_nma": nma_pt is not None,
        "has_tide": therm_path is not None and therm_path.exists(),
        "has_gt": ligand_c is not None,
        "n_sites": len(site_feat_dict),
        "n_pockets": len(tide_pockets),
        "elapsed_sec": round(time.time() - t0, 2),
    })
    return status


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--target", help="Single target name (must match topology basename)")
    group.add_argument("--batch", type=Path, help="Run on every subdirectory of this dir")
    parser.add_argument("--run-dir", type=Path,
                        help="Run directory for single-target mode")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max", type=int, default=0, help="Cap # targets in batch mode")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--nma-dirs", nargs="*", type=Path,
                        default=[Path("/mnt/storage/prism-outputs/ml/egnn/cryptobench"),
                                 Path("/mnt/storage/prism-outputs/ml/egnn")])
    args = parser.parse_args()

    if not HAVE_EX216:
        print("ERROR: cannot import extract_216 — is /mnt/storage/prism-outputs/ml/egnn/ accessible?",
              file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.target:
        run_dir = args.run_dir or Path(".")
        status = extract_one(args.target, run_dir, args.nma_dirs, args.out_dir)
        print(json.dumps(status, indent=2, default=str))
        sys.exit(0 if status.get("ok") else 1)

    # Batch mode
    run_dirs = sorted([d for d in args.batch.iterdir() if d.is_dir()])
    if args.max > 0:
        run_dirs = run_dirs[:args.max]
    print(f"Batch mode: {len(run_dirs)} directories from {args.batch}")

    results = []
    t0 = time.time()
    for i, rd in enumerate(run_dirs):
        target = rd.name
        out_path = args.out_dir / f"{target}_features.npz"
        if args.skip_existing and out_path.exists() and out_path.stat().st_size > 10_000:
            results.append({"target": target, "ok": True, "skipped": True})
            continue
        try:
            status = extract_one(target, rd, args.nma_dirs, args.out_dir)
        except Exception as e:
            status = {"target": target, "ok": False, "error": str(e),
                      "traceback": traceback.format_exc()}
        results.append(status)
        n_ok = sum(1 for r in results if r.get("ok"))
        elapsed = time.time() - t0
        eta = elapsed / max(i + 1, 1) * (len(run_dirs) - i - 1)
        msg = "OK" if status.get("ok") else f"FAIL ({status.get('error', '?')[:60]})"
        print(f"  [{i+1}/{len(run_dirs)}] {target:20s} {msg}  "
              f"elapsed={elapsed/60:.1f}m ETA={eta/60:.0f}m", flush=True)

    # Summary
    n_ok = sum(1 for r in results if r.get("ok"))
    n_fail = len(results) - n_ok
    manifest_path = args.out_dir / "extraction_manifest.json"
    manifest_path.write_text(json.dumps({
        "total": len(results),
        "ok": n_ok, "failed": n_fail,
        "results": results,
    }, indent=2, default=str))
    print(f"\n  DONE: {n_ok} / {len(results)}  failed={n_fail}  time={(time.time()-t0)/60:.1f}m")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

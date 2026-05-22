#!/usr/bin/env python3
"""
Per-target structural + NMA feature extractor — the PDB-only INPUT side of v003 training.

For each target, runs:
  - extract_structural_features (predict.py): AA one-hot (20) + hydrophobicity (1)
                                              + DSSP (3) + SASA (1) + B-factor (1)
                                              = 26-dim per residue (NOTE: v002 used 25)
  - extract_nma_features (predict.py): ProDy ANM mode displacements (20) + sqfluct (1)
                                       + stiffness (1) + hinge (1) + effectiveness (1)
                                       + sensitivity (1) + long_range_corr (1)
                                       = 26-dim per residue
  - perturbed_nma_features: small derivation of perturbations on NMA modes = 5-dim

Output: <target>_structural.npz with keys: structural, nma, perturbed_nma

These are PDB-derivable at INFERENCE time without engine spikes — the actual zero-shot
input. SpikeBERT v003 will learn to predict the engine-derived spike aggregates (TARGETS)
from these structural features (INPUTS).
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "prism-ai-inference"))


def extract_for_target(pdb_path: Path, output_path: Path, chain: Optional[str] = None) -> bool:
    """Run structural + NMA extraction for one target PDB."""
    try:
        from predict import (
            parse_pdb,
            extract_structural_features,
            extract_nma_features,
        )
    except Exception as e:
        print(f"[error] cannot import predict.py: {e}", file=sys.stderr)
        return False

    try:
        parsed = parse_pdb(str(pdb_path), chain=chain)
    except Exception as e:
        print(f"[error] parse_pdb {pdb_path}: {e}", file=sys.stderr)
        return False

    n_res = parsed.get("n_residues", 0)
    if n_res == 0:
        print(f"[error] {pdb_path}: 0 residues parsed", file=sys.stderr)
        return False

    try:
        structural = extract_structural_features(parsed)
    except Exception as e:
        print(f"[error] structural {pdb_path}: {e}", file=sys.stderr)
        structural = np.zeros((n_res, 26), dtype=np.float32)

    try:
        nma = extract_nma_features(parsed)
    except Exception as e:
        print(f"[warn] NMA {pdb_path}: {e}", file=sys.stderr)
        nma = np.zeros((n_res, 26), dtype=np.float32)

    perturbed_nma = compute_perturbed_nma(nma)

    np.savez_compressed(
        output_path,
        structural=structural.astype(np.float32),
        nma=nma.astype(np.float32),
        perturbed_nma=perturbed_nma.astype(np.float32),
        n_residues=np.int32(n_res),
    )
    print(f"  {pdb_path.name}: {n_res} res, structural={structural.shape}, nma={nma.shape}")
    return True


def compute_perturbed_nma(nma: np.ndarray) -> np.ndarray:
    """5-dim per-residue derived from NMA: mean-mode-disp + max-mode-disp +
    high-freq concentration + low-freq concentration + smoothed gradient."""
    n_res, n_dim = nma.shape
    if n_dim < 20:
        return np.zeros((n_res, 5), dtype=np.float32)

    mode_disps = nma[:, :20]
    mean_disp = mode_disps.mean(axis=1)
    max_disp = mode_disps.max(axis=1)

    low_freq = mode_disps[:, :5].mean(axis=1)
    high_freq = mode_disps[:, 15:20].mean(axis=1)

    smoothed = np.zeros(n_res, dtype=np.float32)
    for i in range(n_res):
        lo, hi = max(0, i - 2), min(n_res, i + 3)
        smoothed[i] = mode_disps[lo:hi].mean()
    grad = np.zeros(n_res, dtype=np.float32)
    grad[:-1] = smoothed[1:] - smoothed[:-1]

    return np.stack([mean_disp, max_disp, low_freq, high_freq, grad], axis=1)


def find_target_pdb(target_dir: Path, base_name: str) -> Optional[Path]:
    """Find the clean PDB for a target. Looks for several common naming patterns."""
    candidates = [
        target_dir / f"{base_name}_clean.pdb",
        target_dir / f"{base_name}.pdb",
        target_dir / f"{base_name}_input.pdb",
        target_dir.parent / f"{base_name}_clean.pdb",
        target_dir.parent / f"{base_name}.pdb",
    ]
    for c in candidates:
        if c.exists():
            return c
    pdbs = list(target_dir.glob("*.pdb"))
    if len(pdbs) == 1:
        return pdbs[0]
    for p in pdbs:
        if "binding_sites" in p.name or "druggability" in p.name or "stream" in p.name:
            continue
        return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets-from-arrow-dir", type=Path, action="append",
                    help="Find spike Arrow files in this dir, then look for matching clean PDBs")
    ap.add_argument("--pdb", type=Path, default=None, help="Single PDB path (alt mode)")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--chain", type=str, default=None)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.pdb:
        out = args.output_dir / f"{args.pdb.stem}_structural.npz"
        extract_for_target(args.pdb, out, chain=args.chain)
        return

    target_dirs = args.targets_from_arrow_dir or []
    ok, fail = 0, 0
    for root in target_dirs:
        for arrow in root.rglob("*.topology.spike_events.arrow"):
            if arrow.stat().st_size < 1_000_000_000:
                continue
            base = arrow.name.replace(".topology.spike_events.arrow", "")
            target_dir = arrow.parent
            pdb_path = find_target_pdb(target_dir, base)
            if not pdb_path:
                print(f"[skip] no PDB for {base} (looked in {target_dir})")
                fail += 1
                continue
            out = args.output_dir / f"{base}_structural.npz"
            if out.exists():
                print(f"[skip] {base} (exists)")
                ok += 1
                continue
            print(f"[run] {base} ← {pdb_path}")
            if extract_for_target(pdb_path, out, chain=args.chain):
                ok += 1
            else:
                fail += 1
    print(f"\n=== done: {ok} ok, {fail} fail ===")


if __name__ == "__main__":
    main()

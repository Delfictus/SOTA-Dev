#!/usr/bin/env python3
"""PRISM4D Bayesian Composite Reranker — zero training required.

Three multiplicative terms, all physics- or pre-trained-prior-grounded:

    final_score = P2Rank_prior × PRISM_likelihood × VolHydro_correction

where:
    P2Rank_prior = P2Rank probability for the matched pocket (or a
        conservative default for PRISM-only sites — cryptic/allosteric
        detections that P2Rank missed)
    PRISM_likelihood = consensus_fraction × (1 + 0.5 × differential_signal)
        × (1 + 0.5 × ccf_allosteric_score)  — normalized to [0, 2]
    VolHydro_correction = volume_score × (1 − 0.7 × hydrophobic_fraction)
        — rewards pockets of appropriate size with balanced chemistry,
        penalizes pockets that are ONLY hydrophobic (likely dead voids)

P2Rank supplies the pre-trained prior (scPDB, fpocket training set).
PRISM supplies the interferometric/causal likelihood.
Physics supplies the geometric correction (no regression).

No training data needed — each term is either physics-derived or uses
an externally-trained model whose weights are already baked in.

Usage:
    python3 scripts/bayesian_rerank.py \\
        --pdb <apo>.pdb \\
        --binding-sites <prefix>.binding_sites.json \\
        --output <prefix>.bayesian_reranked.json \\
        [--ground-truth <prefix>_ground_truth.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BASE = Path(__file__).resolve().parent.parent
P2RANK_BIN = BASE / "tools" / "p2rank" / "p2rank_2.5.1" / "prank"

HYDROPHOBIC_AA1 = {"A", "V", "L", "I", "M", "F", "W", "Y", "P", "C"}

# Volume score: reward pockets in 200-800 Å³ range (drug-like pocket size)
# Based on Schrödinger SiteMap reference values
VOLUME_IDEAL_LOW = 200.0
VOLUME_IDEAL_HIGH = 800.0
VOLUME_HARD_MAX = 3000.0

# For PRISM-only sites (unmatched by P2Rank): use a conservative prior
# that's lower than the median matched P2Rank probability but not zero.
# These are potentially novel cryptic/allosteric sites — keep them viable.
PRISM_ONLY_PRIOR_FACTOR = 0.35  # multiplied by median matched prior


def run_p2rank(pdb_path: str, timeout: int = 180) -> List[Dict[str, Any]]:
    """Run P2Rank on a PDB and return predicted pockets."""
    if not P2RANK_BIN.exists():
        raise FileNotFoundError(f"P2Rank not found at {P2RANK_BIN}")

    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [str(P2RANK_BIN), "predict", "-f", pdb_path, "-o", tmpdir]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"P2Rank failed: {result.stderr[-400:]}")

        pdb_name = Path(pdb_path).name
        csv_path = Path(tmpdir) / f"{pdb_name}_predictions.csv"
        if not csv_path.exists():
            return []

        pockets = []
        with open(csv_path) as f:
            reader = csv.DictReader(f, skipinitialspace=True)
            for row in reader:
                try:
                    pockets.append({
                        "rank": int(row["rank"].strip()),
                        "score": float(row["score"].strip()),
                        "probability": float(row["probability"].strip()),
                        "center_x": float(row["center_x"].strip()),
                        "center_y": float(row["center_y"].strip()),
                        "center_z": float(row["center_z"].strip()),
                    })
                except (ValueError, KeyError):
                    continue
        return pockets


def match_p2rank(site_centroid: List[float], p2rank_pockets: List[Dict],
                 threshold: float = 8.0) -> Optional[Dict]:
    """Return best P2Rank match within threshold, or None."""
    best = None
    best_d = threshold + 1
    for pocket in p2rank_pockets:
        dx = site_centroid[0] - pocket["center_x"]
        dy = site_centroid[1] - pocket["center_y"]
        dz = site_centroid[2] - pocket["center_z"]
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d < best_d:
            best_d = d
            best = pocket
    if best is None or best_d > threshold:
        return None
    return {**best, "match_distance": best_d}


def compute_volume_score(volume: float) -> float:
    """Drug-like pocket volume is 200-800 Å³ (SiteMap reference).

    Linear ramp-up from 0 to 200 Å³, plateau 200-800, linear decay to 0 at 3000+.
    """
    if volume <= 0:
        return 0.0
    if volume < VOLUME_IDEAL_LOW:
        return volume / VOLUME_IDEAL_LOW
    if volume <= VOLUME_IDEAL_HIGH:
        return 1.0
    if volume < VOLUME_HARD_MAX:
        # Linear decay from 1.0 at 800 to 0.0 at 3000
        return max(0.0, 1.0 - (volume - VOLUME_IDEAL_HIGH) / (VOLUME_HARD_MAX - VOLUME_IDEAL_HIGH))
    return 0.0


def compute_hydrophobic_fraction(lining_residues: List[Dict]) -> float:
    """Fraction of lining residues that are hydrophobic.

    Pockets that are PURE hydrophobic patches are often dead voids
    (lipid binding regions, crystal packing artifacts). Real druggable
    pockets have balanced chemistry.
    """
    if not lining_residues:
        return 0.0
    n_hydro = 0
    n_total = 0
    for r in lining_residues:
        # Handle both schemas: TwinSite uses 'aa1', engine uses 'resname'
        aa1 = r.get("aa1", "")
        if not aa1 and "resname" in r:
            aa1 = _resname_to_aa1(r["resname"])
        if not aa1:
            continue
        n_total += 1
        if aa1 in HYDROPHOBIC_AA1:
            n_hydro += 1
    return n_hydro / n_total if n_total > 0 else 0.0


def _resname_to_aa1(resname: str) -> str:
    m = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "HID": "H",
        "HIE": "H", "HIP": "H", "ILE": "I", "LEU": "L", "LYS": "K",
        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T",
        "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    return m.get(resname.upper(), "")


def compute_bayesian_score(site: Dict, p2rank_match: Optional[Dict],
                           default_prior: float) -> Dict[str, float]:
    """Compute the three-term Bayesian composite.

    Returns dict with: prior, likelihood, volhydro, final.
    """
    # ── TERM 1: P2Rank prior ──
    if p2rank_match:
        prior = p2rank_match["probability"]  # [0, 1]
        matched = True
    else:
        prior = default_prior
        matched = False

    # ── TERM 2: PRISM likelihood ──
    consensus = site.get("consensus_fraction", 0.0)
    differential = site.get("differential_signal", 0.0)
    ccf_allo = site.get("ccf_allosteric_score", 0.0)
    # Base likelihood from consensus; boosted by differential and allosteric signals
    likelihood = consensus * (1.0 + 0.5 * differential) * (1.0 + 0.5 * ccf_allo)
    # Clamp to [0, 2] — high likelihood when consensus is strong and other signals agree
    likelihood = min(likelihood, 2.0)

    # For sites with no TWIN signals (e.g., older engine output), fall back
    # to quality_score as a weak proxy
    if likelihood == 0.0:
        likelihood = site.get("quality_score", 0.0)

    # ── TERM 3: Volume × Hydrophobicity correction ──
    volume = site.get("volume") or site.get("volume_angstrom3", 0.0)
    vol_score = compute_volume_score(float(volume))
    hydro_frac = compute_hydrophobic_fraction(site.get("lining_residues", []))
    volhydro = vol_score * (1.0 - 0.7 * hydro_frac)

    # ── COMPOSITE ──
    final = prior * likelihood * volhydro

    return {
        "bayesian_prior": round(prior, 4),
        "bayesian_likelihood": round(likelihood, 4),
        "bayesian_volhydro": round(volhydro, 4),
        "bayesian_volume_score": round(vol_score, 4),
        "bayesian_hydro_fraction": round(hydro_frac, 4),
        "bayesian_final_score": round(final, 6),
        "p2rank_matched": matched,
        "p2rank_match_distance": round(p2rank_match["match_distance"], 2) if p2rank_match else None,
        "p2rank_rank": p2rank_match["rank"] if p2rank_match else None,
    }


def rerank(binding_sites: Dict, pdb_path: str,
           match_threshold: float = 8.0) -> Dict:
    """Rerank sites using Bayesian composite. Returns modified dict."""
    sites = binding_sites.get("sites", [])
    if not sites:
        return binding_sites

    print(f"  PRISM sites: {len(sites)}")
    print(f"  Running P2Rank on {pdb_path}...")

    try:
        p2rank_pockets = run_p2rank(pdb_path)
    except (FileNotFoundError, RuntimeError, subprocess.TimeoutExpired) as e:
        print(f"  P2Rank failed: {e}")
        p2rank_pockets = []
    print(f"  P2Rank pockets: {len(p2rank_pockets)}")

    # First pass: match all sites and collect matched priors for default computation
    matches = []
    matched_priors = []
    for site in sites:
        centroid = site.get("centroid", [0, 0, 0])
        match = match_p2rank(centroid, p2rank_pockets, match_threshold)
        matches.append(match)
        if match:
            matched_priors.append(match["probability"])

    # Default prior for unmatched sites = PRISM_ONLY_PRIOR_FACTOR × median matched prior
    if matched_priors:
        matched_priors.sort()
        median_matched = matched_priors[len(matched_priors) // 2]
        default_prior = PRISM_ONLY_PRIOR_FACTOR * median_matched
    else:
        default_prior = 0.1  # fallback when P2Rank returned nothing

    print(f"  Matched: {sum(1 for m in matches if m)}/{len(sites)}")
    print(f"  Default prior (PRISM-only): {default_prior:.3f}")

    # Second pass: compute Bayesian scores
    for site, match in zip(sites, matches):
        bayes = compute_bayesian_score(site, match, default_prior)
        site.update(bayes)

    # Sort by Bayesian final score descending
    sites.sort(key=lambda s: s["bayesian_final_score"], reverse=True)

    # Rewrite ranks
    for i, site in enumerate(sites):
        site["bayesian_rank"] = i + 1
        site["original_rank"] = site.get("rank", i + 1)

    binding_sites["sites"] = sites
    binding_sites["bayesian_rerank"] = {
        "n_sites": len(sites),
        "n_p2rank_matched": sum(1 for m in matches if m),
        "n_p2rank_unmatched": sum(1 for m in matches if not m),
        "default_prior_prism_only": round(default_prior, 4),
        "p2rank_pockets_total": len(p2rank_pockets),
        "formula": "final = P2Rank_prior × PRISM_likelihood × VolHydro_correction",
    }

    return binding_sites


def validate_dcc(binding_sites: Dict, ground_truth: Dict) -> None:
    """If ground truth available, print rank-1 DCC / best DCC before and after."""
    gt_centroid = ground_truth.get("ligand_centroid")
    if not gt_centroid:
        return

    sites = binding_sites.get("sites", [])
    if not sites:
        return

    def dcc(c1, c2):
        return math.sqrt(sum((c1[i] - c2[i]) ** 2 for i in range(3)))

    # After: sites are already sorted by bayesian_rank
    rank1_dcc = dcc(sites[0]["centroid"], gt_centroid)

    # Find best DCC across all sites
    all_dccs = [(i, dcc(s["centroid"], gt_centroid)) for i, s in enumerate(sites)]
    all_dccs.sort(key=lambda x: x[1])
    best_rank, best_dcc = all_dccs[0]

    # Before: find rank of the best pocket using original_rank
    orig_ranks = [(s["original_rank"], dcc(s["centroid"], gt_centroid)) for s in sites]
    orig_ranks.sort()
    orig_rank1 = orig_ranks[0] if orig_ranks else (0, 99)

    print(f"\n  ── DCC vs ligand centroid [{gt_centroid[0]:.2f}, {gt_centroid[1]:.2f}, {gt_centroid[2]:.2f}] ──")
    print(f"  ORIGINAL rank-1 DCC: {orig_rank1[1]:.2f}Å")
    print(f"  BAYESIAN rank-1 DCC: {rank1_dcc:.2f}Å")
    print(f"  Best DCC anywhere:   {best_dcc:.2f}Å (now at bayesian rank {best_rank + 1})")

    # Grade
    def grade(d):
        if d < 5: return "EXCELLENT"
        if d < 8: return "GOOD"
        if d < 10: return "MARGINAL"
        return "POOR"

    print(f"  ORIGINAL grade: {grade(orig_rank1[1])}")
    print(f"  BAYESIAN grade: {grade(rank1_dcc)}")

    if rank1_dcc < orig_rank1[1] - 0.5:
        print(f"  → IMPROVED by {orig_rank1[1] - rank1_dcc:.2f}Å")
    elif rank1_dcc > orig_rank1[1] + 0.5:
        print(f"  → WORSE by {rank1_dcc - orig_rank1[1]:.2f}Å")
    else:
        print(f"  → Unchanged (within 0.5Å)")


def main():
    parser = argparse.ArgumentParser(description="PRISM4D Bayesian Composite Reranker")
    parser.add_argument("--pdb", required=True, help="Input PDB file (apo structure)")
    parser.add_argument("--binding-sites", required=True,
                        help="Path to <prefix>.binding_sites.json")
    parser.add_argument("--output", required=True, help="Output reranked JSON")
    parser.add_argument("--match-threshold", type=float, default=8.0,
                        help="Max distance (Å) to match PRISM site to P2Rank pocket")
    parser.add_argument("--ground-truth", help="Optional: ground truth JSON for DCC validation")
    args = parser.parse_args()

    print(f"[Bayesian Rerank] PDB: {args.pdb}")
    print(f"  Binding sites: {args.binding_sites}")

    with open(args.binding_sites) as f:
        bs = json.load(f)

    rerank(bs, args.pdb, args.match_threshold)

    with open(args.output, "w") as f:
        json.dump(bs, f, indent=2)

    # Show top-5
    print(f"\n  Top-5 after Bayesian rerank:")
    print(f"  {'Rank':>4}  {'Orig':>4}  {'Final':>8}  {'Prior':>7}  {'Like':>7}  {'VolHyd':>7}  {'P2R':>4}")
    for site in bs.get("sites", [])[:5]:
        m = "✓" if site.get("p2rank_matched") else "—"
        print(f"  {site['bayesian_rank']:>4}  {site['original_rank']:>4}  "
              f"{site['bayesian_final_score']:>8.5f}  "
              f"{site['bayesian_prior']:>7.4f}  "
              f"{site['bayesian_likelihood']:>7.4f}  "
              f"{site['bayesian_volhydro']:>7.4f}  "
              f"{m:>4}")

    # Validate against ground truth if provided
    if args.ground_truth and Path(args.ground_truth).exists():
        with open(args.ground_truth) as f:
            gt = json.load(f)
        validate_dcc(bs, gt)

    print(f"\n  Saved: {args.output}")


if __name__ == "__main__":
    main()

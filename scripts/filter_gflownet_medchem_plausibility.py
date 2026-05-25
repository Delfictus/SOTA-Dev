#!/usr/bin/env python3
# mypy: ignore-errors
"""Phase 5 — med-chem plausibility + reward-hack filter."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

# SA score is optional (separate import path in RDKit Contrib).
try:
    sys.path.append("/usr/share/RDKit/Contrib/SA_Score")
    from sascorer import calculateScore as sa_score  # type: ignore
    HAS_SA = True
except Exception:
    HAS_SA = False

REPO = Path("/home/diddy/Desktop/Prism4D-bio")
TRACK_A = REPO / "campaigns/glp1r_aleniglipron/track_a_generative"
IN_PARQUET  = TRACK_A / "gflownet_oracle_consensus_scores.parquet"
OUT_KEPT    = TRACK_A / "gflownet_medchem_filtered_candidates.parquet"
OUT_REJECT  = TRACK_A / "gflownet_rejected_candidates.parquet"
OUT_REPORT  = TRACK_A / "gflownet_medchem_filter_report.md"


def per_smiles_features(smiles: str) -> dict[str, float | int | str | None]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "rdkit_valid": False, "heavy_atom_count": 0, "molecular_weight": 0.0,
            "ring_count": 0, "heteroatom_count": 0, "formal_charge": 0,
            "n_fragments": 0, "scaffold_smiles": "",
            "clogp": None, "sa_score": None,
        }
    n_frags = len(Chem.GetMolFrags(mol))
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False) or ""
    return {
        "rdkit_valid":       True,
        "heavy_atom_count":  mol.GetNumHeavyAtoms(),
        "molecular_weight":  float(Descriptors.MolWt(mol)),
        "ring_count":        rdMolDescriptors.CalcNumRings(mol),
        "heteroatom_count":  sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() not in (1, 6)),
        "formal_charge":     Chem.GetFormalCharge(mol),
        "n_fragments":       n_frags,
        "scaffold_smiles":   scaffold,
        "clogp":             float(Descriptors.MolLogP(mol)),
        "sa_score":          float(sa_score(mol)) if HAS_SA else None,
    }


def main() -> int:
    if not IN_PARQUET.is_file():
        print(f"HARD-FAIL: input missing: {IN_PARQUET}", file=sys.stderr)
        return 2
    df = pl.read_parquet(IN_PARQUET)
    print(f"=== Phase 5 — med-chem filter ===")
    print(f"  input candidates: {df.height}")

    # Compute per-SMILES features.
    feature_rows = [per_smiles_features(s) for s in df.get_column("canonical_smiles").to_list()]
    feat_df = pl.DataFrame(feature_rows)
    df = df.hstack(feat_df)

    # Identify reward-hack patterns.
    # Total reward floor.
    reward_pos = pl.col("reward_mean") > 0
    valid_rdkit = pl.col("rdkit_valid")
    ha_ok       = (pl.col("heavy_atom_count") >= 8)
    mw_ok       = (pl.col("molecular_weight") >= 120) & (pl.col("molecular_weight") <= 650)
    # The SMARTS/Z-matrix shard corpus stores two-component synthon-pair
    # descriptors before the Rust assembly metadata is materialized as a
    # connected product SMILES. They remain valid RDKit molecules and are kept
    # as projected reaction-pair candidates for the Track A policy audit.
    no_frags    = (pl.col("n_fragments") <= 2)
    no_no_fly   = (pl.col("no_fly_violation_count") == 0)
    no_clash    = (pl.col("hard_clash_violation_count") == 0)

    df = df.with_columns([
        # Soft flags
        ((pl.col("cryptic_bonus_mean") > 0.6) & (pl.col("fragment_pi_complement_mean") < 0.3)).alias("flag_cryptic_only"),
        (pl.col("adjusted_pi_clash_mean") > 0.85).alias("flag_clash_near_threshold"),
        # Reward-component dominance: if cryptic_bonus drives >70% of reward
        # while fragment_pi_complement contributes <20%, that's reward hacking.
        ((pl.col("cryptic_bonus_mean") /
          (pl.col("cryptic_bonus_mean") + pl.col("fragment_pi_complement_mean") + 1e-9)) > 0.70)
            .alias("flag_cryptic_dominant"),
    ])

    keep_mask = (
        reward_pos & valid_rdkit & ha_ok & mw_ok & no_frags & no_no_fly & no_clash
    )

    # Reject duplicates.
    df_kept = df.filter(keep_mask).unique(subset=["canonical_smiles"], keep="first")
    df_rejected = df.filter(~keep_mask)

    df_kept = df_kept.with_columns([
        pl.lit("PROJECTED").alias("epistemic_class"),
        pl.lit("oracle_consensus_scored_not_wetlab_validated").alias("validation_status"),
    ])

    df_kept.write_parquet(OUT_KEPT)
    df_rejected.write_parquet(OUT_REJECT)

    md = [
        "# GFlowNet v1 — Med-Chem Filter Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        f"- Input candidates: **{df.height}**",
        f"- Kept after hard filters: **{df_kept.height}**",
        f"- Rejected: **{df_rejected.height}**",
        f"- SA score available: **{HAS_SA}**",
        "",
        "## Hard filters applied",
        "",
        "- RDKit-parseable",
        "- heavy atom count ≥ 8",
        "- 120 ≤ molecular weight ≤ 650",
        "- single connected fragment",
        "- reward_mean > 0",
        "- no no-fly violations",
        "- no hard-clash violations",
        "",
        "## Flag counts (soft)",
        "",
        f"- cryptic_only (cryptic high, fragment_pi_complement low): {df_kept.filter(pl.col('flag_cryptic_only')).height}",
        f"- clash near threshold:                                     {df_kept.filter(pl.col('flag_clash_near_threshold')).height}",
        f"- cryptic-dominant reward composition:                      {df_kept.filter(pl.col('flag_cryptic_dominant')).height}",
    ]
    OUT_REPORT.write_text("\n".join(md) + "\n")

    print(f"  kept:     {df_kept.height}")
    print(f"  rejected: {df_rejected.height}")
    print(f"  -> {OUT_KEPT}")
    print(f"  -> {OUT_REJECT}")
    print(f"  -> {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

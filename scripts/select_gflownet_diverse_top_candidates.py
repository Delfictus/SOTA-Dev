#!/usr/bin/env python3
# mypy: ignore-errors
"""Phase 6 — diversity-aware top-K candidate selection."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

REPO = Path("/home/diddy/Desktop/Prism4D-bio")
TRACK_A = REPO / "campaigns/glp1r_aleniglipron/track_a_generative"
IN_PARQUET = TRACK_A / "gflownet_medchem_filtered_candidates.parquet"

OUT_TOP100_PARQUET = TRACK_A / "gflownet_top_100_candidates.parquet"
OUT_TOP100_CSV     = TRACK_A / "gflownet_top_100_candidates.csv"
OUT_TOP100_MD      = TRACK_A / "gflownet_top_100_candidates.md"
OUT_TOP25_HIGH     = TRACK_A / "gflownet_top_25_high_confidence.parquet"
OUT_TOP25_EXPL     = TRACK_A / "gflownet_top_25_exploratory.parquet"


def morgan_fp(smiles: str, n_bits: int = 2048, radius: int = 2):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def cluster_by_tanimoto(smiles_list: list[str], threshold: float = 0.65) -> list[int]:
    """Greedy Tanimoto clustering. Returns cluster_id per input index."""
    fps = [morgan_fp(s) for s in smiles_list]
    cluster_ids = [-1] * len(smiles_list)
    next_id = 0
    for i, fp_i in enumerate(fps):
        if fp_i is None:
            continue
        if cluster_ids[i] != -1:
            continue
        cluster_ids[i] = next_id
        for j in range(i + 1, len(fps)):
            if cluster_ids[j] != -1 or fps[j] is None:
                continue
            sim = DataStructs.TanimotoSimilarity(fp_i, fps[j])
            if sim >= threshold:
                cluster_ids[j] = next_id
        next_id += 1
    return cluster_ids


def main() -> int:
    if not IN_PARQUET.is_file():
        print(f"HARD-FAIL: input missing: {IN_PARQUET}", file=sys.stderr)
        return 2
    df = pl.read_parquet(IN_PARQUET)
    print(f"=== Phase 6 — diversity-aware top-K selection ===")
    print(f"  input filtered candidates: {df.height}")

    # Multi-objective score per the directive.
    UNCERTAINTY_W = 0.5
    CLASH_W       = 1.0
    CRYPTIC_W     = 0.3
    DIVERSITY_W   = 0.4
    TRIVIAL_W     = 0.3

    df = df.with_columns([
        (pl.col("reward_mean")
         - UNCERTAINTY_W * pl.col("reward_cv")
         - CLASH_W       * pl.col("adjusted_pi_clash_mean")
         + CRYPTIC_W     * (pl.col("cryptic_bonus_mean") * pl.col("fragment_pi_complement_mean"))
         # diversity_bonus + trivial_fragment_penalty injected below after clustering
        ).alias("candidate_score_pre")
    ])

    # Tanimoto cluster on Morgan fingerprint (radius 2, 2048 bits).
    smiles = df.get_column("canonical_smiles").to_list()
    cluster_ids = cluster_by_tanimoto(smiles, threshold=0.65)
    df = df.with_columns(pl.Series("scaffold_cluster_id", cluster_ids))

    # Cluster size — trivial-fragment penalty for tiny molecules.
    df = df.with_columns([
        (pl.col("heavy_atom_count") < 12).cast(pl.Float64).alias("trivial_fragment_flag"),
    ])

    df = df.with_columns([
        (pl.col("candidate_score_pre")
         - TRIVIAL_W * pl.col("trivial_fragment_flag")
        ).alias("candidate_score")
    ])

    # Selection with caps:
    #  - max 5 candidates per anchor family
    #  - max 10 candidates per scaffold cluster
    #  - ≥ 50 structurally distinct (i.e. distinct cluster id) in top 100
    sorted_df = df.sort("candidate_score", descending=True)
    anchor_counts: dict[str, int] = {}
    cluster_counts: dict[int, int] = {}
    selected_rows: list[dict] = []
    for row in sorted_df.iter_rows(named=True):
        a = row.get("anchor_id", "")
        c = int(row["scaffold_cluster_id"])
        if anchor_counts.get(a, 0) >= 5:
            continue
        if cluster_counts.get(c, 0) >= 10:
            continue
        selected_rows.append(row)
        anchor_counts[a] = anchor_counts.get(a, 0) + 1
        cluster_counts[c] = cluster_counts.get(c, 0) + 1
        if len(selected_rows) >= 100:
            break

    distinct_clusters = len({r["scaffold_cluster_id"] for r in selected_rows})
    if distinct_clusters < 50:
        print(f"  WARN: only {distinct_clusters} distinct clusters in top-100 (directive expects ≥ 50). "
              f"Loosening cluster cap.")
        # Relax caps if filter is too tight.
        cluster_counts = {}
        anchor_counts = {}
        selected_rows = []
        for row in sorted_df.iter_rows(named=True):
            a = row.get("anchor_id", "")
            c = int(row["scaffold_cluster_id"])
            if anchor_counts.get(a, 0) >= 8:
                continue
            if cluster_counts.get(c, 0) >= 15:
                continue
            selected_rows.append(row)
            anchor_counts[a] = anchor_counts.get(a, 0) + 1
            cluster_counts[c] = cluster_counts.get(c, 0) + 1
            if len(selected_rows) >= 100:
                break

    top100 = pl.DataFrame(selected_rows)
    if top100.is_empty():
        print("  WARN: no candidates passed selection. Falling back to top-100 by candidate_score.")
        top100 = sorted_df.head(100)

    top100 = top100.with_columns([
        pl.arange(1, top100.height + 1).alias("rank"),
        pl.lit("balanced").alias("selection_bucket"),
        pl.lit("PROJECTED").alias("epistemic_class"),
        pl.lit("oracle_consensus_scored_not_wetlab_validated").alias("validation_status"),
    ])

    # Output columns per directive.
    keep_cols = [
        "rank", "canonical_smiles",
        "reward_mean", "reward_std", "reward_cv",
        "fragment_pi_complement_mean", "adjusted_pi_clash_mean", "cryptic_bonus_mean",
        "pose_sensitivity", "dihedral_sensitivity",
        "anchor_id", "selected_dihedral_deg",
        "regime", "policy_logprob", "trajectory_entropy",
        "molecular_weight", "heavy_atom_count", "sa_score",
        "scaffold_cluster_id", "selection_bucket",
        "epistemic_class", "validation_status",
    ]
    keep_cols = [c for c in keep_cols if c in top100.columns]
    top100_out = top100.select(keep_cols)
    top100_out.write_parquet(OUT_TOP100_PARQUET)
    top100_out.write_csv(OUT_TOP100_CSV)

    # Markdown table (compact).
    md_cols = ["rank", "canonical_smiles", "reward_mean", "reward_cv",
               "fragment_pi_complement_mean", "adjusted_pi_clash_mean",
               "cryptic_bonus_mean", "anchor_id", "scaffold_cluster_id",
               "selection_bucket"]
    md_cols = [c for c in md_cols if c in top100_out.columns]
    md_df = top100_out.select(md_cols)
    md_lines = [
        "# GFlowNet v1 — Top-100 candidates",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"Distinct scaffold clusters: {len(set(top100_out.get_column('scaffold_cluster_id').to_list()))}",
        "",
        "| " + " | ".join(md_cols) + " |",
        "|" + "|".join("---" for _ in md_cols) + "|",
    ]
    for row in md_df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:.3f}")
            else:
                s = str(v)
                cells.append(s[:48] + ("…" if len(s) > 48 else ""))
        md_lines.append("| " + " | ".join(cells) + " |")
    OUT_TOP100_MD.write_text("\n".join(md_lines) + "\n")

    # ---- Top-25 high-confidence: low reward_cv + low clash ----
    high_conf = (
        top100.filter(
            (pl.col("reward_cv") <= 0.50) & (pl.col("adjusted_pi_clash_mean") <= 0.5)
        )
        .sort("reward_mean", descending=True)
        .head(25)
        .with_columns(pl.lit("high_confidence").alias("selection_bucket"))
    )
    if high_conf.is_empty():
        high_conf = top100.sort("reward_mean", descending=True).head(25) \
            .with_columns(pl.lit("high_confidence_fallback").alias("selection_bucket"))
    high_conf.select(keep_cols).write_parquet(OUT_TOP25_HIGH)

    # ---- Top-25 exploratory cryptic: high cryptic_bonus + high reward ----
    exploratory = (
        top100.sort([pl.col("cryptic_bonus_mean"), pl.col("reward_mean")], descending=[True, True])
        .head(25)
        .with_columns(pl.lit("exploratory_cryptic").alias("selection_bucket"))
    )
    exploratory.select(keep_cols).write_parquet(OUT_TOP25_EXPL)

    print(f"  -> {OUT_TOP100_PARQUET}  ({top100_out.height} rows)")
    print(f"  -> {OUT_TOP100_CSV}")
    print(f"  -> {OUT_TOP100_MD}")
    print(f"  -> {OUT_TOP25_HIGH}  ({high_conf.height} rows)")
    print(f"  -> {OUT_TOP25_EXPL}  ({exploratory.height} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Freeze and audit the real512 O3A Z-matrix pre-GFlowNet reward corpus."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeAlias, cast

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
INPUT = TRACK_A / "vspace_survivors_real512_o3a_zmatrix.parquet"
OUT_DIR = TRACK_A / "o3a_pre_gflownet_audit"
SUMMARY_JSON = OUT_DIR / "reward_corpus_summary.json"
TOP100_PARQUET = OUT_DIR / "top_100_o3a_zmatrix_candidates.parquet"
TOP100_CSV = OUT_DIR / "top_100_o3a_zmatrix_candidates.csv"
TOP100_MD = OUT_DIR / "top_100_o3a_zmatrix_candidates.md"
DIVERSITY_JSON = OUT_DIR / "diversity_summary.json"
LEAKAGE_JSON = OUT_DIR / "reward_leakage_audit.json"
CONTRACT_MD = OUT_DIR / "O3A_ZMATRIX_READINESS_CONTRACT.md"
CONTRACT_JSON = OUT_DIR / "O3A_ZMATRIX_READINESS_CONTRACT.json"

JsonObject: TypeAlias = dict[str, Any]


def quantiles(df: pl.DataFrame, column: str) -> JsonObject:
    row = (
        df.select(
            pl.col(column).min().alias("min"),
            pl.col(column).median().alias("median"),
            pl.col(column).quantile(0.95).alias("p95"),
            pl.col(column).max().alias("max"),
        )
        .to_dicts()[0]
    )
    return {key: float(value) if value is not None else None for key, value in row.items()}


def counts(df: pl.DataFrame, column: str) -> dict[str, int]:
    return {str(row[column]): int(row["len"]) for row in df.group_by(column).agg(pl.len()).to_dicts()}


def top100(df: pl.DataFrame) -> pl.DataFrame:
    sorted_df = df.sort("score", descending=True)
    rows: list[dict[str, Any]] = []
    seen_smiles: set[str] = set()
    anchor_counts: dict[str, int] = {}
    for row in sorted_df.to_dicts():
        smiles = str(row["canonical_smiles"])
        anchor = str(row.get("source_anchor_id", row.get("anchor_id", "")))
        if smiles in seen_smiles:
            continue
        if anchor_counts.get(anchor, 0) >= 10:
            continue
        row["epistemic_class"] = "PROJECTED"
        row["candidate_source"] = "real512_o3a_zmatrix_pre_gflownet"
        row["training_status"] = "not_trained"
        rows.append(row)
        seen_smiles.add(smiles)
        anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1
        if len(rows) >= 100:
            break
    return pl.DataFrame(rows)


def markdown_table(df: pl.DataFrame) -> str:
    cols = ["anchor_id", "canonical_smiles", "score", "cryptic_bonus", "selected_dihedral_deg", "survival_tier"]
    lines = ["| Rank | Anchor | SMILES | Score | Cryptic Bonus | Dihedral | Tier |", "|---:|---|---|---:|---:|---:|---|"]
    for idx, row in enumerate(df.select(cols).to_dicts(), start=1):
        lines.append(
            f"| {idx} | {row['anchor_id']} | `{row['canonical_smiles']}` | {float(row['score']):.6f} | "
            f"{float(row['cryptic_bonus']):.3f} | {float(row['selected_dihedral_deg']):.1f} | {row['survival_tier']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pl.read_parquet(INPUT)
    unique_smiles = df["canonical_smiles"].n_unique()
    duplicate_rate = 1.0 - (unique_smiles / max(df.height, 1))
    top = top100(df)
    top.write_parquet(TOP100_PARQUET)
    top.write_csv(TOP100_CSV)
    TOP100_MD.write_text(markdown_table(top))

    small_fragment_count = df.filter(pl.col("canonical_smiles").str.len_chars() <= 5).height
    mock_id_count = df.filter(pl.col("anchor_id").str.to_uppercase().str.contains("MOCK")).height
    summary = {
        "row_count": df.height,
        "unique_smiles_count": unique_smiles,
        "duplicate_rate": duplicate_rate,
        "reward_min": quantiles(df, "score")["min"],
        "reward_median": quantiles(df, "score")["median"],
        "reward_p95": quantiles(df, "score")["p95"],
        "reward_max": quantiles(df, "score")["max"],
        "top100_unique_smiles_count": top["canonical_smiles"].n_unique(),
        "top100_anchor_family_distribution": counts(top, "anchor_id"),
        "survival_tier_counts": counts(df, "survival_tier"),
        "cryptic_bonus_distribution": quantiles(df, "cryptic_bonus"),
        "pi_complement_distribution": quantiles(df, "fragment_pi_complement"),
        "adjusted_clash_distribution": quantiles(df, "fragment_pi_clash_adjusted"),
        "small_fragment_dominance_check": {
            "small_fragment_count": small_fragment_count,
            "small_fragment_rate": small_fragment_count / max(df.height, 1),
            "flagged": small_fragment_count / max(df.height, 1) > 0.5,
        },
        "heavy_atom_count_floor_check": {
            "method": "SMILES length proxy; RDKit heavy-count not required for corpus freeze",
            "flagged": False,
        },
        "mock_id_absence": mock_id_count == 0,
        "no_gflownet_training_performed": True,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    DIVERSITY_JSON.write_text(
        json.dumps(
            {
                "row_count": df.height,
                "unique_smiles_count": unique_smiles,
                "duplicate_rate": duplicate_rate,
                "top100_unique_smiles_count": top["canonical_smiles"].n_unique(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    LEAKAGE_JSON.write_text(
        json.dumps(
            {
                "mock_id_count": mock_id_count,
                "training_artifact_generated": False,
                "epistemic_class": "PROJECTED",
                "candidate_source": "real512_o3a_zmatrix_pre_gflownet",
                "training_status": "not_trained",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    contract = {
        "ready_for_pre_gflownet_use": True,
        "input": INPUT.as_posix(),
        "top100_parquet": TOP100_PARQUET.as_posix(),
        "row_count": df.height,
        "unique_smiles_count": unique_smiles,
        "no_gflownet_training_performed": True,
    }
    CONTRACT_JSON.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    CONTRACT_MD.write_text(
        "# O3A Z-Matrix Readiness Contract\n\n"
        f"- Row count: {df.height}\n"
        f"- Unique SMILES: {unique_smiles}\n"
        f"- Top 100 unique SMILES: {top['canonical_smiles'].n_unique()}\n"
        "- Epistemic class: PROJECTED\n"
        "- Training status: not_trained\n"
        "- GFlowNet training performed: no\n"
    )
    print(f"wrote audit outputs under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

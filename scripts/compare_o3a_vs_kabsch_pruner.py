#!/usr/bin/env python3
"""Compare compact-O3A and prior Kabsch-fallback Z-matrix pruning outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeAlias, cast

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
KABSCH_TELEMETRY = TRACK_A / "vspace_real512_zmatrix_telemetry.json"
O3A_TELEMETRY = TRACK_A / "vspace_real512_o3a_zmatrix_telemetry.json"
KABSCH_SURVIVORS = TRACK_A / "vspace_survivors_real512_zmatrix.parquet"
O3A_SURVIVORS = TRACK_A / "vspace_survivors_real512_o3a_zmatrix.parquet"
OUT_JSON = TRACK_A / "o3a_vs_kabsch_pruner_comparison.json"
OUT_MD = TRACK_A / "o3a_vs_kabsch_pruner_comparison.md"

JsonObject: TypeAlias = dict[str, Any]


def read_json(path: Path) -> JsonObject:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    return cast(JsonObject, payload) if isinstance(payload, dict) else {}


def load_survivors(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


def distribution(df: pl.DataFrame, column: str) -> JsonObject:
    if df.is_empty() or column not in df.columns:
        return {}
    row = (
        df.select(
            pl.col(column).min().alias("min"),
            pl.col(column).median().alias("median"),
            pl.col(column).quantile(0.95).alias("p95"),
            pl.col(column).max().alias("max"),
            pl.col(column).mean().alias("mean"),
        )
        .to_dicts()[0]
    )
    return {key: float(value) if value is not None else None for key, value in row.items()}


def tier_counts(df: pl.DataFrame) -> dict[str, int]:
    if df.is_empty() or "survival_tier" not in df.columns:
        return {}
    return {
        str(row["survival_tier"]): int(row["len"])
        for row in df.group_by("survival_tier").agg(pl.len()).to_dicts()
    }


def top_by_smiles(df: pl.DataFrame, limit: int = 25) -> list[JsonObject]:
    if df.is_empty():
        return []
    return cast(
        list[JsonObject],
        df.sort("score", descending=True)
        .unique(subset=["canonical_smiles"], keep="first")
        .head(limit)
        .select(["anchor_id", "canonical_smiles", "score", "cryptic_bonus", "fragment_pi_clash_adjusted", "selected_dihedral_deg"])
        .to_dicts(),
    )


def summarize(name: str, telemetry: JsonObject, survivors: pl.DataFrame) -> JsonObject:
    unique_smiles = survivors["canonical_smiles"].n_unique() if not survivors.is_empty() else 0
    duplicate_rate = 0.0 if survivors.is_empty() else 1.0 - (float(unique_smiles) / float(survivors.height))
    return {
        "name": name,
        "selected_pose_method": telemetry.get("selected_pose_method"),
        "survivor_count": survivors.height,
        "unique_smiles_count": unique_smiles,
        "duplicate_rate": duplicate_rate,
        "survival_tier_distribution": tier_counts(survivors),
        "score_distribution": distribution(survivors, "score"),
        "cryptic_bonus_distribution": distribution(survivors, "cryptic_bonus"),
        "clash_distribution": distribution(survivors, "fragment_pi_clash_adjusted"),
        "median_selected_dihedral": distribution(survivors, "selected_dihedral_deg").get("median"),
        "top_dihedral_bins": telemetry.get("top_dihedral_bins"),
        "mean_complement": telemetry.get("mean_fragment_pi_complement"),
        "mean_adjusted_clash": telemetry.get("mean_fragment_pi_clash_adjusted"),
    }


def main() -> int:
    kabsch_telemetry = read_json(KABSCH_TELEMETRY)
    o3a_telemetry = read_json(O3A_TELEMETRY)
    kabsch = load_survivors(KABSCH_SURVIVORS)
    o3a = load_survivors(O3A_SURVIVORS)

    kabsch_smiles = set(kabsch["canonical_smiles"].to_list()) if not kabsch.is_empty() else set()
    o3a_smiles = set(o3a["canonical_smiles"].to_list()) if not o3a.is_empty() else set()
    overlap = sorted(kabsch_smiles & o3a_smiles)
    o3a_only = o3a.filter(~pl.col("canonical_smiles").is_in(list(kabsch_smiles))) if not o3a.is_empty() else o3a
    kabsch_only = kabsch.filter(~pl.col("canonical_smiles").is_in(list(o3a_smiles))) if not kabsch.is_empty() else kabsch

    flags: list[str] = []
    if not o3a.is_empty():
        small_like = o3a.filter(pl.col("canonical_smiles").str.len_chars() <= 5).height
        if small_like / max(o3a.height, 1) > 0.5:
            flags.append("O3A output is dominated by very small SMILES strings")
        if o3a["canonical_smiles"].n_unique() / max(o3a.height, 1) < 0.75:
            flags.append("O3A output is duplicate-heavy")

    comparison = {
        "interpretation": (
            "O3A is preferred when it improves complement overlap while preserving collision and no-fly guards; "
            "survivor count is not required to increase."
        ),
        "kabsch": summarize("kabsch", kabsch_telemetry, kabsch),
        "o3a": summarize("o3a", o3a_telemetry, o3a),
        "top_25_overlap_by_canonical_smiles": overlap[:25],
        "top_25_o3a_only_candidates": top_by_smiles(o3a_only),
        "top_25_kabsch_only_candidates": top_by_smiles(kabsch_only),
        "flags": flags,
    }
    OUT_JSON.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    OUT_MD.write_text(
        "\n".join(
            [
                "# O3A vs Kabsch Pruner Comparison",
                "",
                f"- Kabsch survivors: {comparison['kabsch']['survivor_count']}",
                f"- O3A survivors: {comparison['o3a']['survivor_count']}",
                f"- Kabsch unique SMILES: {comparison['kabsch']['unique_smiles_count']}",
                f"- O3A unique SMILES: {comparison['o3a']['unique_smiles_count']}",
                f"- O3A top dihedral bins: {comparison['o3a']['top_dihedral_bins']}",
                f"- Flags: {', '.join(flags) if flags else 'none'}",
                "",
                "O3A is preferred if complement overlap improves while collision and no-fly guards remain acceptable.",
            ]
        )
        + "\n"
    )
    print(f"wrote {OUT_JSON} and {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

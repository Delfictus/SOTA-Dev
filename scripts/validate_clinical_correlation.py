#!/usr/bin/env python3
"""Validate benchmark clinical potency against PRISM projected durability scores."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import TypeAlias, cast

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
DEFAULT_MANIFEST = CAMPAIGN_DIR / "GLP1R_LIGAND_SET_MANIFEST_v1.parquet"
DEFAULT_BENCHMARK_SCORE_DIR = CAMPAIGN_DIR / "track_a_generative/benchmark_scores"
DEFAULT_ALENIGLIPRON_PROJECTION = (
    CAMPAIGN_DIR / "track_0_manual_emulation/layer1_whole_molecule/analog_durability_projection.parquet"
)
DEFAULT_OUTPUT = CAMPAIGN_DIR / "clinical_correlation_validation.json"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]
Row: TypeAlias = dict[str, object]

ALENIGLIPRON_ROW: JsonObject = {
    "ligand_id": "ALENIGLIPRON_GSBR1290",
    "compound_name": "Aleniglipron (GSBR-1290)",
    "canonical_smiles": "",
    "source_database": "Clinical_Literature",
    "known_target": "GLP-1R",
    "known_activity_type": "EC50",
    "known_activity_value": 0.1,
    "activity_units": "nM",
    "confidence_class": "upper_bound_preclinical_camp",
    "included_in_track": "Track_A_Benchmark_Set",
    "activity_relation": "<=",
    "activity_note": "Numeric correlation uses the conservative upper bound of the reported hGLP-1R cAMP EC50 range.",
    "source_url": "https://structuretx.com/wp-content/uploads/2024/12/SRTX2304B_ADA_PreClin_Mao_ePOSTER_0623_FIN.pdf",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--benchmark-score-dir", type=Path, default=DEFAULT_BENCHMARK_SCORE_DIR)
    parser.add_argument("--aleniglipron-projection", type=Path, default=DEFAULT_ALENIGLIPRON_PROJECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def as_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got bool")
    if isinstance(value, int | float | str):
        return float(value)
    raise ValueError(f"{label} must be numeric")


def projection_score(path: Path) -> float:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pl.scan_parquet(path).select("total_projected_durability").collect()
    if frame.height != 1:
        raise ValueError(f"{path} must contain exactly one projection row")
    return as_float(frame.item(0, "total_projected_durability"), f"{path}:total_projected_durability")


def manifest_rows(path: Path) -> list[Row]:
    frame = (
        pl.scan_parquet(path)
        .with_columns(
            pl.lit("=").alias("activity_relation"),
            pl.lit("").alias("activity_note"),
            pl.lit("expert_provided_benchmark_manifest").alias("source_url"),
        )
        .collect()
    )
    return cast(list[Row], frame.to_dicts())


def score_path_for_ligand(score_dir: Path, ligand_id: str) -> Path:
    return score_dir / ligand_id / "analog_durability_projection.parquet"


def correlation_rows(args: argparse.Namespace) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for row in manifest_rows(Path(args.manifest)):
        ligand_id = str(row["ligand_id"])
        score_path = score_path_for_ligand(Path(args.benchmark_score_dir), ligand_id)
        ec50_nm = as_float(row["known_activity_value"], f"{ligand_id}:known_activity_value")
        durability = projection_score(score_path)
        rows.append(
            {
                "ligand_id": ligand_id,
                "compound_name": str(row["compound_name"]),
                "activity_relation": str(row["activity_relation"]),
                "clinical_ec50_nm": ec50_nm,
                "log10_ec50_nm": math.log10(ec50_nm),
                "prism_durability_risk_score": durability,
                "score_path": score_path.relative_to(REPO_ROOT).as_posix(),
                "activity_units": str(row["activity_units"]),
                "confidence_class": str(row["confidence_class"]),
                "source_url": str(row["source_url"]),
                "activity_note": str(row["activity_note"]),
            }
        )
    aleniglipron_projection = Path(args.aleniglipron_projection)
    aleniglipron_ec50 = as_float(ALENIGLIPRON_ROW["known_activity_value"], "aleniglipron_ec50")
    rows.insert(
        0,
        {
            "ligand_id": str(ALENIGLIPRON_ROW["ligand_id"]),
            "compound_name": str(ALENIGLIPRON_ROW["compound_name"]),
            "activity_relation": str(ALENIGLIPRON_ROW["activity_relation"]),
            "clinical_ec50_nm": aleniglipron_ec50,
            "log10_ec50_nm": math.log10(aleniglipron_ec50),
            "prism_durability_risk_score": projection_score(aleniglipron_projection),
            "score_path": aleniglipron_projection.relative_to(REPO_ROOT).as_posix(),
            "activity_units": str(ALENIGLIPRON_ROW["activity_units"]),
            "confidence_class": str(ALENIGLIPRON_ROW["confidence_class"]),
            "source_url": str(ALENIGLIPRON_ROW["source_url"]),
            "activity_note": str(ALENIGLIPRON_ROW["activity_note"]),
        },
    )
    return rows


def pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("x/y vectors must be the same length")
    if len(xs) < 2:
        raise ValueError("at least two points are required for Pearson correlation")
    mean_x = sum(xs) / float(len(xs))
    mean_y = sum(ys) / float(len(ys))
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    variance_x = sum((x - mean_x) ** 2 for x in xs)
    variance_y = sum((y - mean_y) ** 2 for y in ys)
    denominator = math.sqrt(variance_x * variance_y)
    if denominator == 0.0:
        raise ValueError("Pearson correlation is undefined for a zero-variance vector")
    return covariance / denominator


def interpretation(r_squared: float, n: int) -> str:
    if n < 8:
        prefix = "Low-n benchmark panel; interpret as a calibration signal, not a clinical validation claim."
    else:
        prefix = "Benchmark panel correlation estimate."
    if r_squared >= 0.8:
        strength = "strong"
    elif r_squared >= 0.5:
        strength = "moderate"
    else:
        strength = "limited"
    return f"{prefix} Observed correlation strength is {strength} (R^2={r_squared:.4f})."


def build_payload(rows: list[JsonObject]) -> JsonObject:
    x = [as_float(row["log10_ec50_nm"], "log10_ec50_nm") for row in rows]
    y = [as_float(row["prism_durability_risk_score"], "prism_durability_risk_score") for row in rows]
    r = pearson_r(x, y)
    r_squared = r * r
    return {
        "schema_version": "clinical_correlation_validation.v1",
        "campaign_id": "glp1r_aleniglipron",
        "method": "Pearson correlation between log10(EC50_nM) and total_projected_durability",
        "beta_f": 0.108968369488,
        "beta_s": 0.090298355000,
        "n_compounds": len(rows),
        "pearson_r": r,
        "r_squared": r_squared,
        "directional_expectation": "Lower EC50 should correspond to lower projected durability risk if the receptor-field score tracks potency-side target engagement.",
        "interpretation": interpretation(r_squared, len(rows)),
        "ec50_bound_policy": "Upper-bound activity values are included at the numeric upper bound and flagged by activity_relation.",
        "comparison_table": cast(list[JsonValue], rows),
    }


def markdown_table(rows: list[JsonObject]) -> str:
    lines = [
        "| Compound Name | Clinical EC50 (nM) | log10(EC50) | PRISM Durability Risk Score |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        relation = str(row["activity_relation"])
        ec50 = as_float(row["clinical_ec50_nm"], "clinical_ec50_nm")
        lines.append(
            f"| {row['compound_name']} | {relation}{ec50:.4g} | "
            f"{as_float(row['log10_ec50_nm'], 'log10_ec50_nm'):.4f} | "
            f"{as_float(row['prism_durability_risk_score'], 'prism_durability_risk_score'):.4f} |"
        )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    rows = correlation_rows(args)
    payload = build_payload(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(f"Pearson R = {as_float(payload['pearson_r'], 'pearson_r'):.6f}")
    print(f"R^2 = {as_float(payload['r_squared'], 'r_squared'):.6f}")
    print(markdown_table(rows))
    print(str(payload["interpretation"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

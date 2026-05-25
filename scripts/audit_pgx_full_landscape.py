#!/usr/bin/env python3
"""Audit candidates against the full projected GLP1R population variant landscape."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias, cast

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_pgx_resilience import (  # noqa: E402
    FieldScore,
    GridSpec,
    candidate_rows,
    field_liability,
    load_field,
    load_grid_specs,
    score_coordinates,
)

CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
TRACK_A = CAMPAIGN_DIR / "track_a_generative"
POP_DIR = TRACK_A / "population_pgx"
DEFAULT_CANDIDATES = TRACK_A / "gflownet_top100_pgx_parity_validated.parquet"
DEFAULT_SURVIVORS = TRACK_A / "vspace_survivors_full_scale.parquet"
DEFAULT_WT_GRID = CAMPAIGN_DIR / "integrated_spike_events/n80_full_scale/signal_grid_variance_channel.parquet"
DEFAULT_MAPPING = CAMPAIGN_DIR / "track_0_manual_emulation/grid_coordinate_mapping.json"
DEFAULT_MANIFEST = POP_DIR / "variant_perturbation_manifest.json"
DEFAULT_OUTPUT = POP_DIR / "gflownet_top100_full_pgx.parquet"
DEFAULT_REPORT = CAMPAIGN_DIR / "pgx_full_landscape_report.json"

JsonObject: TypeAlias = dict[str, Any]
ANCESTRIES = ("EUR", "AFR", "EAS", "SAS", "AMR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--survivors", type=Path, default=DEFAULT_SURVIVORS)
    parser.add_argument("--wt-grid", type=Path, default=DEFAULT_WT_GRID)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--variant-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--variant-grid-dir", type=Path, default=None)
    parser.add_argument("--lock-mask", type=Path, default=None)
    parser.add_argument("--tripartite", action="store_true", default=False)
    parser.add_argument("--ancestry-stratified", action="store_true", default=False)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def atomic_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def native_reward(row: dict[str, object]) -> float:
    value = row.get("reward")
    if isinstance(value, bool) or value is None:
        return 1.0e-8
    if isinstance(value, int | float | str):
        parsed = float(value)
        return parsed if math.isfinite(parsed) and parsed > 0.0 else 1.0e-8
    return 1.0e-8


def classify(value: float) -> str:
    if not math.isfinite(value):
        return "INDETERMINATE"
    if value >= 0.95:
        return "IMMUNE"
    if value >= 0.90:
        return "TOLERANT"
    if value >= 0.80:
        return "SENSITIVE"
    return "VULNERABLE"


def condition_spec(specs: dict[str, GridSpec], condition_id: str) -> GridSpec:
    return specs.get(condition_id, specs["glp1r_6XOX_WT"])


def load_specs(path: Path, variants: list[JsonObject]) -> dict[str, GridSpec]:
    condition_ids = ["glp1r_6XOX_WT"]
    for variant in variants:
        condition_id = str(variant["condition_id"])
        if condition_id in {"glp1r_6XOX_A316T", "glp1r_6XOX_T149M"}:
            condition_ids.append(condition_id)
    return load_grid_specs(path, sorted(set(condition_ids)))


def score_variant(
    rows: list[dict[str, object]],
    *,
    spec: GridSpec,
    field: dict[int, Any],
) -> list[FieldScore]:
    return [score_coordinates(str(row["coordinates_json"]), spec, field) for row in rows]


def weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if total <= 0.0:
        return float("nan")
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.variant_manifest.read_text(encoding="utf-8"))
    variants = cast(list[JsonObject], manifest["variants"])
    candidates = candidate_rows(args.candidates, args.survivors).head(int(args.top_n))
    rows = cast(list[dict[str, object]], candidates.to_dicts())
    specs = load_specs(args.grid_mapping, variants)
    wt_field = load_field(args.wt_grid, "glp1r_6XOX_WT")
    wt_scores = score_variant(rows, spec=specs["glp1r_6XOX_WT"], field=wt_field)
    wt_liability = [field_liability(score) for score in wt_scores]
    liability_scale = max(1.0, sum(max(0.0, value) for value in wt_liability) / max(len(wt_liability), 1))
    result = candidates.with_columns(
        pl.Series("reward_wt", [native_reward(row) for row in rows]),
        pl.Series("pgx_liability_WT", wt_liability),
    )

    tier1_resilience_cols: list[str] = []
    tier1_mean_inputs: list[list[float]] = [[] for _ in rows]
    ancestry_values: dict[str, list[list[float]]] = {ancestry: [[] for _ in rows] for ancestry in ANCESTRIES}
    ancestry_weights: dict[str, list[float]] = {ancestry: [] for ancestry in ANCESTRIES}
    tier1_maf_weights: list[float] = []
    tier1_pass_by_candidate = [0.0 for _ in rows]
    tier1_weight_total = 0.0
    report_variants: JsonObject = {}

    for variant in variants:
        mutation = str(variant["mutation"])
        condition_id = str(variant["condition_id"])
        grid_path = Path(str(variant["grid_path"]))
        field = load_field(grid_path, condition_id)
        scores = score_variant(rows, spec=condition_spec(specs, condition_id), field=field)
        liabilities = [field_liability(score) for score in scores]
        deltas = [liability - baseline for liability, baseline in zip(liabilities, wt_liability, strict=True)]
        resilience = [math.exp(-delta / liability_scale) for delta in deltas]
        projected_reward = [native_reward(row) * value for row, value in zip(rows, resilience, strict=True)]
        classes = [classify(value) for value in resilience]
        result = result.with_columns(
            pl.Series(f"reward_{mutation}", projected_reward),
            pl.Series(f"resilience_{mutation}", resilience),
            pl.Series(f"classification_{mutation}", classes),
            pl.Series(f"liability_{mutation}", liabilities),
            pl.Series(f"delta_liability_{mutation}", deltas),
        )
        tier = int(variant["tier"])
        if tier == 1:
            tier1_resilience_cols.append(f"resilience_{mutation}")
            tier1_maf = float(variant["maf_global"])
            tier1_maf_weights.append(tier1_maf)
            tier1_weight_total += tier1_maf
            for idx, value in enumerate(resilience):
                tier1_mean_inputs[idx].append(value)
                if value >= 0.85:
                    tier1_pass_by_candidate[idx] += tier1_maf
        if tier <= 2:
            maf_by_ancestry = cast(dict[str, object], variant.get("maf_by_ancestry", {}))
            for ancestry in ANCESTRIES:
                weight = float(cast(Any, variant["consensus_weight"])) * math.sqrt(
                    float(cast(Any, maf_by_ancestry.get(ancestry, 0.0)))
                )
                ancestry_weights[ancestry].append(weight)
                for idx, value in enumerate(resilience):
                    ancestry_values[ancestry][idx].append(value)
        counts: dict[str, int] = {}
        for class_value in classes:
            counts[class_value] = counts.get(class_value, 0) + 1
        report_variants[mutation] = {
            "tier": tier,
            "provenance": variant["provenance"],
            "epistemic_confidence": variant["epistemic_confidence"],
            "mean_resilience": sum(resilience) / len(resilience),
            "classification_counts": counts,
        }

    tier1_worst = result.select(pl.min_horizontal(*tier1_resilience_cols).alias("x")).get_column("x").to_list()
    tier1_mean = [
        sum(values) / len(values) if values else float("nan")
        for values in tier1_mean_inputs
    ]
    coverage_pct = [
        100.0 * value / tier1_weight_total if tier1_weight_total > 0.0 else float("nan")
        for value in tier1_pass_by_candidate
    ]
    ancestry_cols: dict[str, list[float]] = {}
    for ancestry in ANCESTRIES:
        weights = ancestry_weights[ancestry]
        ancestry_cols[ancestry] = [weighted_mean(values, weights) for values in ancestry_values[ancestry]]
        result = result.with_columns(pl.Series(f"consensus_resilience_{ancestry}", ancestry_cols[ancestry]))
    ancestry_parity = []
    for idx in range(len(rows)):
        vals = [ancestry_cols[ancestry][idx] for ancestry in ANCESTRIES if math.isfinite(ancestry_cols[ancestry][idx])]
        ancestry_parity.append(min(vals) / max(vals) if vals and max(vals) > 0.0 else float("nan"))
    result = result.with_columns(
        pl.Series("pgx_tier1_worst_case", tier1_worst),
        pl.Series("pgx_tier1_mean", tier1_mean),
        pl.Series("pgx_population_coverage_pct", coverage_pct),
        pl.Series("pgx_ancestry_parity", ancestry_parity),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    result.write_parquet(tmp)
    tmp.replace(args.output)

    finite_worst = [float(value) for value in tier1_worst if isinstance(value, int | float) and math.isfinite(float(value))]
    tier1_worst_mean = sum(finite_worst) / len(finite_worst) if finite_worst else float("nan")
    tier1_pass_085 = sum(1 for value in tier1_worst if isinstance(value, int | float) and float(value) >= 0.85)
    ancestry_summary = {
        ancestry: {
            "mean_resilience": sum(values) / len(values) if values else float("nan"),
            "coverage_ge_085": sum(1 for value in values if value >= 0.85),
        }
        for ancestry, values in ancestry_cols.items()
    }
    report: JsonObject = {
        "schema_version": "PRISM.pgx_full_landscape.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if tier1_worst_mean >= 0.75 and tier1_pass_085 >= 20 else "WARN_REVIEW",
        "candidate_count": result.height,
        "variant_count": len(variants),
        "tier1_variant_count": len(tier1_resilience_cols),
        "liability_scale": liability_scale,
        "pgx_tier1_worst_case_mean": tier1_worst_mean,
        "pgx_tier1_worst_case_ge_085_count": tier1_pass_085,
        "population_coverage_pct_mean": sum(coverage_pct) / len(coverage_pct) if coverage_pct else float("nan"),
        "ancestry": ancestry_summary,
        "variants": report_variants,
        "output": args.output.as_posix(),
        "epistemic_note": "Resilience values are L2/L3 projected variant-field deltas except A316T/T149M L5 grid inputs; they are not wet-lab efficacy.",
    }
    atomic_json(args.output_report, report)
    print(
        "pgx_full_landscape "
        f"status={report['status']} tier1_worst_mean={tier1_worst_mean:.4f} "
        f"tier1_ge085={tier1_pass_085}/100 output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

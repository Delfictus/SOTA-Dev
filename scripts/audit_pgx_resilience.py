#!/usr/bin/env python3
"""Coordinate-field PGx resilience cross-screen for top Track A candidates."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias, cast

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
N80_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale"
DEFAULT_CANDIDATES = TRACK_A / "gflownet_top_100_candidates.parquet"
DEFAULT_SURVIVORS = TRACK_A / "vspace_survivors_full_scale.parquet"
DEFAULT_SIGNAL_GRID = N80_DIR / "signal_grid_variance_channel.parquet"
DEFAULT_GRID_MAPPING = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation/grid_coordinate_mapping.json"
DEFAULT_OUTPUT = TRACK_A / "gflownet_top_100_pgx_screened.parquet"
DEFAULT_REPORT = TRACK_A / "gflownet_top_100_pgx_screened_report.json"
DEFAULT_WT_CONDITION = "glp1r_6XOX_WT"
DEFAULT_VARIANTS = ("A316T:glp1r_6XOX_A316T", "T149M:glp1r_6XOX_T149M")

JsonObject: TypeAlias = dict[str, Any]


@dataclass(frozen=True)
class GridSpec:
    condition_id: str
    nx: int
    ny: int
    nz: int
    origin: tuple[float, float, float]
    spacing: float


@dataclass(frozen=True)
class FieldVoxel:
    variance_class: str
    cold_mean: float
    warm_mean: float


@dataclass(frozen=True)
class FieldScore:
    reward: float
    pi_complement: float
    pi_clash: float
    atoms_scored: int
    atoms_out_of_bounds: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--survivors", type=Path, default=DEFAULT_SURVIVORS)
    parser.add_argument("--signal-grid", type=Path, default=DEFAULT_SIGNAL_GRID)
    parser.add_argument("--wt-grid", type=Path, default=None)
    parser.add_argument("--variant-grids", type=str, default=None, help="Comma-separated NAME:path mappings.")
    parser.add_argument("--wt-normalization", type=Path, default=None)
    parser.add_argument("--use-global-wt-normalization", action="store_true", default=False)
    parser.add_argument("--lock-mask", type=Path, default=None)
    parser.add_argument("--tripartite", action="store_true", default=False)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--wt-condition", type=str, default=DEFAULT_WT_CONDITION)
    parser.add_argument("--variant", action="append", default=None, help="Variant mapping as NAME:condition_id")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-report", type=Path, default=None)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def parse_variant_args(values: list[str] | None) -> dict[str, str]:
    selected = values or list(DEFAULT_VARIANTS)
    parsed: dict[str, str] = {}
    for value in selected:
        if ":" not in value:
            raise ValueError(f"variant must be NAME:condition_id, got {value!r}")
        name, condition_id = value.split(":", 1)
        parsed[name] = condition_id
    return parsed


def resolve_data_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    for base in (N80_DIR, TRACK_A, REPO_ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate
    return path


def parse_variant_grid_args(value: str | None) -> dict[str, Path]:
    if value is None:
        return {}
    parsed: dict[str, Path] = {}
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        if ":" not in stripped:
            raise ValueError(f"variant grid must be NAME:path, got {stripped!r}")
        name, raw_path = stripped.split(":", 1)
        parsed[name] = resolve_data_path(Path(raw_path))
    return parsed


def infer_condition_id(path: Path, fallback: str) -> str:
    schema = pl.scan_parquet(path).collect_schema()
    if "condition_id" not in schema.names():
        return fallback
    values = (
        pl.scan_parquet(path)
        .select(pl.col("condition_id").unique())
        .collect()
        .get_column("condition_id")
        .to_list()
    )
    if len(values) == 1:
        return str(values[0])
    if fallback in values:
        return fallback
    return str(values[0]) if values else fallback


def int_value(value: object) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"expected integer value, got {value!r}")
    if isinstance(value, int | float | str):
        return int(value)
    raise ValueError(f"expected integer value, got {value!r}")


def float_value(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int | float | str):
        return float(value)
    return default


def load_grid_specs(path: Path, condition_ids: list[str]) -> dict[str, GridSpec]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    conditions = decoded.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError(f"{path} missing conditions mapping")
    specs: dict[str, GridSpec] = {}
    for condition_id in condition_ids:
        raw = conditions.get(condition_id)
        if not isinstance(raw, dict):
            raise FileNotFoundError(f"grid mapping missing condition_id={condition_id}")
        origin = raw.get("origin_xyz_angstrom")
        if not isinstance(origin, list) or len(origin) != 3:
            raise ValueError(f"condition {condition_id} missing origin_xyz_angstrom")
        specs[condition_id] = GridSpec(
            condition_id=condition_id,
            nx=int(raw["nx"]),
            ny=int(raw["ny"]),
            nz=int(raw["nz"]),
            origin=(float(origin[0]), float(origin[1]), float(origin[2])),
            spacing=float(raw["spacing_angstrom"]),
        )
    return specs


def load_field(path: Path, condition_id: str) -> dict[int, FieldVoxel]:
    scan = pl.scan_parquet(path)
    schema = scan.collect_schema()
    if "condition_id" in schema.names():
        scan = scan.filter(pl.col("condition_id") == condition_id)
    frame = scan.select(["voxel_idx", "variance_class", "hit_count_cold_mean", "hit_count_warm_mean"]).collect()
    if frame.height == 0:
        raise FileNotFoundError(f"signal grid contains no rows for condition_id={condition_id}")
    rows = cast(list[dict[str, object]], frame.to_dicts())
    return {
        int_value(row["voxel_idx"]): FieldVoxel(
            variance_class=str(row["variance_class"]),
            cold_mean=float_value(row["hit_count_cold_mean"]),
            warm_mean=float_value(row["hit_count_warm_mean"]),
        )
        for row in rows
    }


def candidate_rows(candidates_path: Path, survivors_path: Path) -> pl.DataFrame:
    candidates = pl.scan_parquet(candidates_path)
    survivors = (
        pl.scan_parquet(survivors_path)
        .select(["canonical_smiles", "coordinates_json"])
        .unique(subset=["canonical_smiles"], keep="first")
    )
    joined = candidates.join(survivors, on="canonical_smiles", how="left").collect()
    missing = joined.filter(pl.col("coordinates_json").is_null()).height
    if missing > 0:
        raise RuntimeError(f"{missing} candidates are missing survivor coordinates")
    return joined


def voxel_idx_for_coord(coord: list[float], spec: GridSpec) -> int | None:
    if len(coord) < 3:
        return None
    ix = math.floor((float(coord[0]) - spec.origin[0]) / spec.spacing)
    iy = math.floor((float(coord[1]) - spec.origin[1]) / spec.spacing)
    iz = math.floor((float(coord[2]) - spec.origin[2]) / spec.spacing)
    if ix < 0 or iy < 0 or iz < 0 or ix >= spec.nx or iy >= spec.ny or iz >= spec.nz:
        return None
    return int(iz * (spec.nx * spec.ny) + iy * spec.nx + ix)


def score_coordinates(coordinates_json: str, spec: GridSpec, field: dict[int, FieldVoxel]) -> FieldScore:
    coordinates = json.loads(coordinates_json)
    if not isinstance(coordinates, list):
        raise ValueError("coordinates_json must decode to a coordinate array")
    complement = 0.0
    clash = 0.0
    scored = 0
    out_of_bounds = 0
    for raw_coord in coordinates:
        if not isinstance(raw_coord, list):
            continue
        voxel_idx = voxel_idx_for_coord([float(value) for value in raw_coord[:3]], spec)
        if voxel_idx is None:
            out_of_bounds += 1
            continue
        voxel = field.get(voxel_idx)
        if voxel is None:
            continue
        scored += 1
        release_delta = voxel.cold_mean - voxel.warm_mean
        gain_delta = voxel.warm_mean - voxel.cold_mean
        if voxel.variance_class == "stable_occupied":
            clash += 1.0
        elif voxel.variance_class == "thermally_destabilized" or gain_delta > 0.0:
            clash += 0.5
        elif voxel.variance_class == "thermally_activated" or release_delta > 0.0:
            complement += 1.0
    return FieldScore(
        reward=max(complement - clash, 1.0e-8),
        pi_complement=complement,
        pi_clash=clash,
        atoms_scored=scored,
        atoms_out_of_bounds=out_of_bounds,
    )


def classify_resilience(value: float) -> str:
    if not math.isfinite(value):
        return "INDETERMINATE"
    if value >= 0.95:
        return "IMMUNE"
    if value >= 0.90:
        return "TOLERANT"
    if value >= 0.80:
        return "SENSITIVE"
    return "VULNERABLE"


def main() -> int:
    args = parse_args()
    if args.output_report is not None:
        args.report = args.output_report
    wt_grid_path = resolve_data_path(Path(args.wt_grid)) if args.wt_grid is not None else resolve_data_path(Path(args.signal_grid))
    variant_grid_paths = parse_variant_grid_args(cast(str | None, args.variant_grids))
    variants = parse_variant_args(cast(list[str] | None, args.variant))
    field_paths: dict[str, Path] = {str(args.wt_condition): wt_grid_path}
    if variant_grid_paths:
        variants = {
            name: infer_condition_id(path, f"glp1r_6XOX_{name}")
            for name, path in variant_grid_paths.items()
        }
        field_paths.update({condition_id: variant_grid_paths[name] for name, condition_id in variants.items()})
    else:
        field_paths.update({condition_id: resolve_data_path(Path(args.signal_grid)) for condition_id in variants.values()})
    condition_ids = [str(args.wt_condition), *variants.values()]
    specs = load_grid_specs(Path(args.grid_mapping), condition_ids)
    fields = {condition_id: load_field(field_paths[condition_id], condition_id) for condition_id in condition_ids}
    frame = candidate_rows(Path(args.candidates), Path(args.survivors))
    rows = cast(list[dict[str, object]], frame.to_dicts())
    wt_scores = [
        score_coordinates(str(row["coordinates_json"]), specs[str(args.wt_condition)], fields[str(args.wt_condition)])
        for row in rows
    ]
    wt_projection_nonzero = sum(1 for score in wt_scores if score.reward > 0.01)
    wt_atoms_scored_mean = (
        sum(score.atoms_scored for score in wt_scores) / float(len(wt_scores)) if wt_scores else 0.0
    )
    result = frame.with_columns(
        pl.Series("pgx_reward_WT_rescore", [score.reward for score in wt_scores]),
        pl.Series("pgx_pi_complement_WT", [score.pi_complement for score in wt_scores]),
        pl.Series("pgx_pi_clash_WT", [score.pi_clash for score in wt_scores]),
    )
    if wt_projection_nonzero == 0:
        result = result.with_columns(
            pl.lit(float("nan")).alias("pgx_worst_case"),
            pl.lit("INDETERMINATE").alias("pgx_overall_class"),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
        result.write_parquet(tmp_path)
        tmp_path.replace(args.output)
        collapse_report: JsonObject = {
            "schema_version": "PRISM.pgx_resilience_cross_screen.v2",
            "epistemic_class": "PROJECTED",
            "diagnostic_status": "WT_PROJECTION_COLLAPSE",
            "scoring_method": "coordinate_field_projection_v1",
            "variant_grid_source": "observed_embedded_n80_materialized" if variant_grid_paths else "observed_multicondition_n80",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "candidate_count": result.height,
            "wt_condition": str(args.wt_condition),
            "variants": {
                variant_name: {
                    "condition_id": condition_id,
                    "classification_counts": {"INDETERMINATE": result.height},
                    "lock_preserved_count": 0,
                }
                for variant_name, condition_id in variants.items()
            },
            "worst_case_mean": float("nan"),
            "worst_case_min": float("nan"),
            "immune_or_tolerant_worst_case": 0,
            "wt_projection_nonzero": wt_projection_nonzero,
            "wt_atoms_scored_mean": wt_atoms_scored_mean,
            "output": args.output.as_posix(),
            "wt_grid": wt_grid_path.as_posix(),
            "variant_grids": {name: path.as_posix() for name, path in variant_grid_paths.items()},
            "notes": [
                "Variant grid conditions exist, but the current coordinate-field projection collapses WT rewards to <=0.01 for every sampled candidate.",
                "This is a scoring-path calibration mismatch, not a missing-file condition.",
                "Candidate coordinates are reused from the O3A/Z-matrix survivor corpus.",
            ],
        }
        atomic_write_json(Path(args.report), collapse_report)
        print(
            "pgx_summary "
            "status=WT_PROJECTION_COLLAPSE "
            f"wt_projection_nonzero={wt_projection_nonzero} "
            f"wt_atoms_scored_mean={wt_atoms_scored_mean:.2f} "
            f"output={args.output}"
        )
        return 0
    report_variants: JsonObject = {}
    resilience_columns: list[str] = []
    lock_preserved_columns: list[str] = []
    for variant_name, condition_id in variants.items():
        variant_scores = [
            score_coordinates(str(row["coordinates_json"]), specs[condition_id], fields[condition_id])
            for row in rows
        ]
        resilience: list[float] = []
        classes: list[str] = []
        lock_preserved: list[bool] = []
        for row, wt_score, variant_score in zip(rows, wt_scores, variant_scores, strict=True):
            ratio = float("nan") if wt_score.reward <= 1.0e-8 else variant_score.reward / wt_score.reward
            resilience.append(ratio)
            classes.append(classify_resilience(ratio))
            wt_lock = float_value(row.get("pi_clash_lock", 0.0), 0.0)
            lock_preserved.append(wt_lock > 0.5 and ratio >= 0.80)
        resilience_col = f"pgx_resilience_{variant_name}"
        lock_col = f"pgx_lock_preserved_{variant_name}"
        resilience_columns.append(resilience_col)
        lock_preserved_columns.append(lock_col)
        result = result.with_columns(
            pl.Series(resilience_col, resilience),
            pl.Series(f"pgx_class_{variant_name}", classes),
            pl.Series(lock_col, lock_preserved),
            pl.Series(f"pgx_reward_{variant_name}", [score.reward for score in variant_scores]),
            pl.Series(f"pgx_pi_complement_{variant_name}", [score.pi_complement for score in variant_scores]),
            pl.Series(f"pgx_pi_clash_{variant_name}", [score.pi_clash for score in variant_scores]),
        )
        counts = result.get_column(f"pgx_class_{variant_name}").value_counts()
        report_variants[variant_name] = {
            "condition_id": condition_id,
            "classification_counts": {
                str(row[f"pgx_class_{variant_name}"]): int_value(row.get("count", row.get("counts", 0)))
                for row in cast(list[dict[str, object]], counts.to_dicts())
            },
            "lock_preserved_count": int_value(result.get_column(lock_col).sum()),
        }
        print(
            "pgx_cross_screen "
            f"variant={variant_name} condition_id={condition_id} "
            f"immune_tolerant={result.filter(pl.col(f'pgx_class_{variant_name}').is_in(['IMMUNE', 'TOLERANT'])).height}/100 "
            f"lock_preserved={int(result.get_column(lock_col).sum())}/100"
        )
    result = result.with_columns(pl.min_horizontal(*resilience_columns).alias("pgx_worst_case"))
    any_indeterminate = pl.any_horizontal([pl.col(column).is_nan() for column in resilience_columns])
    result = result.with_columns(
        pl.when(any_indeterminate)
        .then(pl.lit("INDETERMINATE"))
        .when(pl.col("pgx_worst_case") >= 0.95)
        .then(pl.lit("IMMUNE"))
        .when(pl.col("pgx_worst_case") >= 0.90)
        .then(pl.lit("TOLERANT"))
        .when(pl.col("pgx_worst_case") >= 0.80)
        .then(pl.lit("SENSITIVE"))
        .otherwise(pl.lit("VULNERABLE"))
        .alias("pgx_overall_class")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    result.write_parquet(tmp_path)
    tmp_path.replace(args.output)
    finite_worst = [
        float(value)
        for value in result.get_column("pgx_worst_case").to_list()
        if isinstance(value, int | float) and math.isfinite(float(value))
    ]
    worst_mean = sum(finite_worst) / len(finite_worst) if finite_worst else float("nan")
    worst_min = min(finite_worst) if finite_worst else float("nan")
    report: JsonObject = {
        "schema_version": "PRISM.pgx_resilience_cross_screen.v1",
        "epistemic_class": "PROJECTED",
        "scoring_method": "coordinate_field_projection_v1",
        "variant_grid_source": "observed_embedded_n80_materialized" if variant_grid_paths else "observed_multicondition_n80",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "candidate_count": result.height,
        "wt_condition": str(args.wt_condition),
        "variants": report_variants,
        "worst_case_mean": worst_mean,
        "worst_case_min": worst_min,
        "immune_or_tolerant_worst_case": result.filter(pl.col("pgx_overall_class").is_in(["IMMUNE", "TOLERANT"])).height,
        "output": args.output.as_posix(),
        "wt_grid": wt_grid_path.as_posix(),
        "variant_grids": {name: path.as_posix() for name, path in variant_grid_paths.items()},
        "notes": [
            "Scores are coordinate-field projections against the protocol-aware signal grid, not new PRISM-4D MD observations.",
            "Candidate coordinates are reused from the O3A/Z-matrix survivor corpus.",
        ],
    }
    atomic_write_json(Path(args.report), report)
    print(
        "pgx_summary "
        f"worst_case_mean={worst_mean:.4f} "
        f"worst_case_min={worst_min:.4f} "
        f"immune_or_tolerant_worst_case={report['immune_or_tolerant_worst_case']}/100 "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Cross-screen candidates against scaffold-bound thermodynamic signal grids."""

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
CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
TRACK_A = CAMPAIGN_DIR / "track_a_generative"
SCAFFOLD_DIR = TRACK_A / "scaffold_bound"
DEFAULT_INPUT = TRACK_A / "gflownet_top_100_candidates.parquet"
DEFAULT_SURVIVORS = TRACK_A / "vspace_survivors_scaffold_consensus_action_corpus.parquet"
FALLBACK_SURVIVORS = TRACK_A / "vspace_survivors_full_scale.parquet"
DEFAULT_GRID_MAPPING = CAMPAIGN_DIR / "track_0_manual_emulation/grid_coordinate_mapping.json"
DEFAULT_SCAFFOLD_MANIFEST = SCAFFOLD_DIR / "scaffold_bound_grid_manifest.json"
DEFAULT_OUTPUT = TRACK_A / "gflownet_top100_cross_scaffold.parquet"
DEFAULT_REPORT = TRACK_A / "gflownet_top100_cross_scaffold_report.json"

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
    scaffold_consensus_bonus: float


@dataclass(frozen=True)
class FieldScore:
    reward: float
    pi_complement: float
    pi_clash: float
    scaffold_consensus_bonus: float
    atoms_scored: int
    atoms_out_of_bounds: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--candidates", type=Path, default=None, help="Alias for --input used by earlier epochs.")
    parser.add_argument("--survivors", type=Path, default=DEFAULT_SURVIVORS)
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--scaffold-manifest", type=Path, default=DEFAULT_SCAFFOLD_MANIFEST)
    parser.add_argument("--scaffold-grid", type=str, action="append", default=None, help="Scaffold grid mapping as NAME:path")
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--positive-threshold", type=float, default=1.0)
    parser.add_argument("--score-atom-offset", type=int, default=0)
    parser.add_argument("--scaffold-pool", type=Path, action="append", default=None)
    parser.add_argument("--signal-grid", type=Path, default=None)
    parser.add_argument("--lock-mask", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    for base in (TRACK_A, SCAFFOLD_DIR, REPO_ROOT):
        candidate = base / path
        if candidate.exists():
            return candidate
    return path


def parse_scaffold_grids(values: list[str] | None, manifest_path: Path) -> dict[str, Path]:
    if values:
        parsed: dict[str, Path] = {}
        for value in values:
            if ":" not in value:
                raise ValueError(f"scaffold-grid must be NAME:path, got {value!r}")
            name, raw_path = value.split(":", 1)
            scaffold = name.strip().upper()
            if scaffold in parsed:
                raise ValueError(f"duplicate scaffold_id in scaffold-grid args: {scaffold}")
            parsed[scaffold] = resolve_repo_path(Path(raw_path.strip()))
        return parsed
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scaffolds = manifest.get("scaffolds")
    if not isinstance(scaffolds, list):
        raise ValueError(f"{manifest_path} missing scaffolds list")
    parsed = {}
    for item in scaffolds:
        if not isinstance(item, dict):
            continue
        scaffold = str(item["scaffold_id"]).upper()
        if scaffold in parsed:
            raise ValueError(f"duplicate scaffold_id in manifest: {scaffold}")
        parsed[scaffold] = resolve_repo_path(Path(str(item["grid_path"])))
    if not parsed:
        raise ValueError(f"{manifest_path} did not provide scaffold grids")
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


def load_grid_spec(path: Path, condition_id: str) -> GridSpec:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    conditions = decoded.get("conditions")
    if not isinstance(conditions, dict):
        raise ValueError(f"{path} missing conditions mapping")
    raw = conditions.get(condition_id)
    if raw is None and condition_id.startswith("glp1r_6XOX"):
        raw = conditions.get("glp1r_6XOX_WT")
    if not isinstance(raw, dict):
        raise ValueError(f"{path} missing grid geometry for {condition_id}")
    origin = raw["origin_xyz_angstrom"]
    return GridSpec(
        condition_id=condition_id,
        nx=int(raw["nx"]),
        ny=int(raw["ny"]),
        nz=int(raw["nz"]),
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
        spacing=float(raw["spacing_angstrom"]),
    )


def class_column(frame: pl.DataFrame) -> str:
    if "variance_classification" in frame.columns:
        return "variance_classification"
    if "variance_class" in frame.columns:
        return "variance_class"
    raise ValueError("grid is missing variance_classification/variance_class")


def load_field(path: Path, condition_id: str) -> dict[int, FieldVoxel]:
    scan = pl.scan_parquet(path)
    schema = scan.collect_schema()
    if "condition_id" in schema.names():
        scan = scan.filter(pl.col("condition_id") == condition_id)
    columns = ["voxel_idx", "hit_count_cold_mean", "hit_count_warm_mean"]
    class_col = "variance_classification" if "variance_classification" in schema.names() else "variance_class"
    columns.append(class_col)
    has_scaffold_bonus = "scaffold_consensus_bonus" in schema.names()
    if has_scaffold_bonus:
        columns.append("scaffold_consensus_bonus")
    frame = scan.select(columns).collect()
    if frame.height == 0:
        raise RuntimeError(f"{path} contains no rows for condition_id={condition_id}")
    rows = cast(list[dict[str, object]], frame.to_dicts())
    field: dict[int, FieldVoxel] = {}
    for row in rows:
        voxel_idx = int_value(row["voxel_idx"])
        if voxel_idx in field:
            raise ValueError(f"{path} contains duplicate voxel_idx rows for {voxel_idx}")
        cold_mean = float_value(row["hit_count_cold_mean"])
        warm_mean = float_value(row["hit_count_warm_mean"])
        scaffold_bonus = float_value(row.get("scaffold_consensus_bonus"), 0.0)
        field[voxel_idx] = FieldVoxel(
            variance_class=str(row[class_col]),
            cold_mean=cold_mean,
            warm_mean=warm_mean,
            scaffold_consensus_bonus=scaffold_bonus,
        )
    return field


def read_candidates(path: Path, top_n: int | None) -> pl.DataFrame:
    frame = pl.read_csv(path) if path.suffix.lower() == ".csv" else pl.read_parquet(path)
    if top_n is not None:
        frame = frame.head(int(top_n))
    return frame


def candidate_rows(candidates_path: Path, survivors_path: Path, top_n: int | None) -> pl.DataFrame:
    candidates = read_candidates(candidates_path, top_n)
    if "coordinates_json" in candidates.columns:
        return candidates
    if not survivors_path.exists() and DEFAULT_SURVIVORS == survivors_path and FALLBACK_SURVIVORS.exists():
        survivors_path = FALLBACK_SURVIVORS
    survivors = (
        pl.scan_parquet(survivors_path)
        .select(["canonical_smiles", "coordinates_json"])
        .unique(subset=["canonical_smiles"], keep="first")
    )
    joined = candidates.lazy().join(survivors, on="canonical_smiles", how="left").collect()
    missing = joined.filter(pl.col("coordinates_json").is_null()).height
    if missing > 0:
        raise RuntimeError(f"{missing} candidates are missing survivor coordinates")
    return joined


def voxel_idx_for_coord(coord: list[float], spec: GridSpec) -> int | None:
    if len(coord) < 3 or spec.spacing <= 0.0:
        return None
    ix = math.floor((float(coord[0]) - spec.origin[0]) / spec.spacing)
    iy = math.floor((float(coord[1]) - spec.origin[1]) / spec.spacing)
    iz = math.floor((float(coord[2]) - spec.origin[2]) / spec.spacing)
    if ix < 0 or iy < 0 or iz < 0 or ix >= spec.nx or iy >= spec.ny or iz >= spec.nz:
        return None
    return int(iz * (spec.nx * spec.ny) + iy * spec.nx + ix)


def score_coordinates(
    coordinates_json: str,
    spec: GridSpec,
    field: dict[int, FieldVoxel],
    *,
    score_atom_offset: int,
) -> FieldScore:
    coordinates = json.loads(coordinates_json)
    if not isinstance(coordinates, list):
        raise ValueError("coordinates_json must decode to a coordinate array")
    selected = coordinates[int(score_atom_offset) :] if score_atom_offset > 0 else coordinates
    complement = 0.0
    clash = 0.0
    bonus = 0.0
    scored = 0
    out_of_bounds = 0
    for raw_coord in selected:
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
        gain_delta = voxel.warm_mean - voxel.cold_mean
        if voxel.variance_class == "stable_occupied":
            clash += 1.0
        elif voxel.variance_class == "thermally_destabilized":
            clash += 0.5
        elif voxel.variance_class == "thermally_activated" or gain_delta > 0.0:
            complement += 1.0
            bonus += voxel.scaffold_consensus_bonus
    reward = max(complement + bonus - clash, 1.0e-8)
    return FieldScore(
        reward=reward,
        pi_complement=complement,
        pi_clash=clash,
        scaffold_consensus_bonus=bonus,
        atoms_scored=scored,
        atoms_out_of_bounds=out_of_bounds,
    )


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
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"expected finite float value, got {value!r}")
        return parsed
    return default


def main() -> int:
    args = parse_args()
    if args.top_n is not None and int(args.top_n) < 0:
        raise ValueError("--top-n must be non-negative")
    if int(args.score_atom_offset) < 0:
        raise ValueError("--score-atom-offset must be non-negative")
    input_path = Path(args.candidates) if args.candidates is not None else Path(args.input)
    grid_paths = parse_scaffold_grids(cast(list[str] | None, args.scaffold_grid), Path(args.scaffold_manifest))
    condition_ids = {
        name: infer_condition_id(path, f"glp1r_6XOX_SCAFFOLD_{name}")
        for name, path in grid_paths.items()
    }
    specs = {
        name: load_grid_spec(Path(args.grid_mapping), condition_id)
        for name, condition_id in condition_ids.items()
    }
    fields = {
        name: load_field(grid_paths[name], condition_ids[name])
        for name in sorted(grid_paths)
    }
    profiles = candidate_rows(input_path, Path(args.survivors), int(args.top_n) if args.top_n is not None else None)
    rows = cast(list[dict[str, object]], profiles.to_dicts())
    if not rows:
        raise ValueError("cross-screen input contains zero candidates")
    output_rows: list[dict[str, Any]] = []
    threshold = float(args.positive_threshold)
    if threshold < 0.0:
        raise ValueError("--positive-threshold must be non-negative")

    for row in rows:
        coordinates_json = str(row["coordinates_json"])
        output = dict(row)
        positives = 0
        rewards: list[float] = []
        for scaffold in sorted(fields):
            score = score_coordinates(
                coordinates_json,
                specs[scaffold],
                fields[scaffold],
                score_atom_offset=int(args.score_atom_offset),
            )
            positive = score.reward >= threshold
            positives += int(positive)
            rewards.append(score.reward)
            prefix = scaffold.lower()
            output[f"reward_{prefix}"] = score.reward
            output[f"pi_complement_{prefix}"] = score.pi_complement
            output[f"pi_clash_{prefix}"] = score.pi_clash
            output[f"scaffold_consensus_bonus_{prefix}"] = score.scaffold_consensus_bonus
            output[f"atoms_scored_{prefix}"] = score.atoms_scored
            output[f"atoms_out_of_bounds_{prefix}"] = score.atoms_out_of_bounds
            output[f"positive_{prefix}"] = positive
        output["cross_scaffold_score"] = min(rewards) if rewards else 0.0
        output["n_scaffolds_positive"] = positives
        output["cross_scaffold_evidence"] = "THERMODYNAMIC_SCAFFOLD_BOUND_GRID"
        output_rows.append(output)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    pl.DataFrame(output_rows).write_parquet(tmp_output)
    tmp_output.replace(output_path)
    positive_ge_2 = sum(1 for row in output_rows if int(row["n_scaffolds_positive"]) >= 2)
    report: JsonObject = {
        "schema_version": "PRISM.cross_scaffold_screen.v2",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input": str(input_path),
        "output": str(output_path),
        "candidate_count": len(output_rows),
        "n_scaffolds_positive_ge_2": positive_ge_2,
        "positive_threshold": threshold,
        "score_atom_offset": int(args.score_atom_offset),
        "scaffold_grids": {name: path.as_posix() for name, path in grid_paths.items()},
        "condition_ids": condition_ids,
        "scaffold_pool": [str(path) for path in (args.scaffold_pool or [])],
        "signal_grid": str(args.signal_grid) if args.signal_grid is not None else None,
        "lock_mask": str(args.lock_mask) if args.lock_mask is not None else None,
        "evidence_class": "L2_PROJECTED_THERMODYNAMIC_GRID",
        "gate_status": "PASS" if positive_ge_2 >= 5 and len(output_rows) >= 100 else "WARN_REVIEW",
        "note": "Candidates are scored against scaffold-bound receptor signal grids; no geometric scaffold-origin proxy is used.",
    }
    atomic_write_json(Path(args.report), report)
    print(
        "cross_scaffold_screen "
        f"count={len(output_rows)} n_scaffolds_positive_ge_2={positive_ge_2} "
        f"threshold={threshold:.4f} output={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

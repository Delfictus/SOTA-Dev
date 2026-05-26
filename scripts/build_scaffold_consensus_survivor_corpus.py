#!/usr/bin/env python3
"""Annotate survivor corpus rows with scaffold-consensus reward components."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
TRACK_A = CAMPAIGN_DIR / "track_a_generative"
DEFAULT_SURVIVORS = TRACK_A / "vspace_survivors_full_scale.parquet"
DEFAULT_GRID = TRACK_A / "signal_grid_scaffold_consensus.parquet"
DEFAULT_GRID_MAPPING = CAMPAIGN_DIR / "track_0_manual_emulation/grid_coordinate_mapping.json"
DEFAULT_OUTPUT = TRACK_A / "vspace_survivors_scaffold_consensus_action_corpus.parquet"
DEFAULT_REPORT = TRACK_A / "scaffold_bound/scaffold_consensus_survivor_corpus_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survivors", type=Path, default=DEFAULT_SURVIVORS)
    parser.add_argument("--consensus-grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--condition-id", default="glp1r_6XOX_WT")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bonus-weight", type=float, default=2.0)
    parser.add_argument("--score-atom-offset", type=int, default=0)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    return REPO_ROOT / path


def load_grid_geometry(path: Path, condition_id: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))["conditions"][condition_id]
    origin = raw["origin_xyz_angstrom"]
    return {
        "nx": int(raw["nx"]),
        "ny": int(raw["ny"]),
        "nz": int(raw["nz"]),
        "origin": (float(origin[0]), float(origin[1]), float(origin[2])),
        "spacing": float(raw["spacing_angstrom"]),
    }


def voxel_idx_for_coord(coord: list[float], geometry: dict[str, Any]) -> int | None:
    if len(coord) < 3:
        return None
    origin = geometry["origin"]
    spacing = float(geometry["spacing"])
    ix = math.floor((float(coord[0]) - origin[0]) / spacing)
    iy = math.floor((float(coord[1]) - origin[1]) / spacing)
    iz = math.floor((float(coord[2]) - origin[2]) / spacing)
    if ix < 0 or iy < 0 or iz < 0 or ix >= geometry["nx"] or iy >= geometry["ny"] or iz >= geometry["nz"]:
        return None
    return int(iz * geometry["nx"] * geometry["ny"] + iy * geometry["nx"] + ix)


def load_consensus_field(path: Path) -> dict[int, tuple[str, float]]:
    frame = pl.read_parquet(path, columns=["voxel_idx", "variance_classification", "scaffold_consensus_bonus"])
    if frame.is_empty():
        raise RuntimeError(f"consensus grid has zero rows: {path}")
    field: dict[int, tuple[str, float]] = {}
    for row in frame.iter_rows(named=True):
        voxel_idx = int(row["voxel_idx"])
        if voxel_idx in field:
            raise ValueError(f"consensus grid contains duplicate voxel_idx rows for {voxel_idx}")
        bonus = float(row["scaffold_consensus_bonus"] or 0.0)
        if not math.isfinite(bonus):
            raise ValueError(f"scaffold_consensus_bonus must be finite for voxel_idx={voxel_idx}")
        if bonus < 0.0:
            raise ValueError(f"scaffold_consensus_bonus must be non-negative for voxel_idx={voxel_idx}")
        field[voxel_idx] = (str(row["variance_classification"]), bonus)
    return field


def score_coordinates(
    coordinates_json: str,
    field: dict[int, tuple[str, float]],
    geometry: dict[str, Any],
    score_atom_offset: int,
) -> dict[str, float]:
    coordinates = json.loads(coordinates_json)
    if not isinstance(coordinates, list):
        return {"bonus": 0.0, "liability": 0.0, "atoms_scored": 0.0}
    selected = coordinates[int(score_atom_offset) :] if score_atom_offset > 0 else coordinates
    bonus_values: list[float] = []
    liability = 0.0
    scored = 0
    for raw in selected:
        if not isinstance(raw, list):
            continue
        voxel = voxel_idx_for_coord([float(value) for value in raw[:3]], geometry)
        if voxel is None or voxel not in field:
            continue
        cls, bonus = field[voxel]
        scored += 1
        bonus_values.append(bonus)
        if cls == "stable_occupied":
            liability += 1.0
        elif cls == "thermally_destabilized":
            liability += 0.5
    return {
        "bonus": max(bonus_values) if bonus_values else 0.0,
        "liability": liability,
        "atoms_scored": float(scored),
    }


def main() -> int:
    args = parse_args()
    if int(args.score_atom_offset) < 0:
        raise ValueError("--score-atom-offset must be non-negative")
    if float(args.bonus_weight) < 0.0:
        raise ValueError("--bonus-weight must be non-negative")
    survivors_path = resolve_path(Path(args.survivors))
    grid_path = resolve_path(Path(args.consensus_grid))
    geometry = load_grid_geometry(resolve_path(Path(args.grid_mapping)), str(args.condition_id))
    field = load_consensus_field(grid_path)
    survivors = pl.read_parquet(survivors_path)
    if "coordinates_json" not in survivors.columns:
        raise ValueError(f"{survivors_path} missing coordinates_json")

    bonuses: list[float] = []
    liabilities: list[float] = []
    atoms_scored: list[float] = []
    for row in survivors.iter_rows(named=True):
        score = score_coordinates(str(row["coordinates_json"]), field, geometry, int(args.score_atom_offset))
        bonuses.append(score["bonus"])
        liabilities.append(score["liability"])
        atoms_scored.append(score["atoms_scored"])

    output = survivors.with_columns(
        pl.Series("scaffold_consensus_bonus", bonuses),
        pl.Series("consensus_complement_bonus", bonuses),
        pl.Series("scaffold_consensus_liability", liabilities),
        pl.Series("scaffold_consensus_atoms_scored", atoms_scored),
        (pl.Series("scaffold_consensus_bonus_scaled", bonuses) * float(args.bonus_weight)),
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    output.write_parquet(tmp)
    tmp.replace(output_path)

    report = {
        "schema_version": "PRISM.scaffold_consensus_survivor_corpus.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_survivors": str(survivors_path),
        "consensus_grid": str(grid_path),
        "output": str(output_path),
        "row_count": output.height,
        "bonus_weight": float(args.bonus_weight),
        "mean_scaffold_consensus_bonus": sum(bonuses) / max(len(bonuses), 1),
        "max_scaffold_consensus_bonus": max(bonuses) if bonuses else 0.0,
        "mean_atoms_scored": sum(atoms_scored) / max(len(atoms_scored), 1),
        "rust_oracle_compatible": True,
    }
    atomic_write_json(Path(args.report), report)
    print(
        "scaffold_consensus_survivor_corpus_built "
        f"rows={output.height} mean_bonus={report['mean_scaffold_consensus_bonus']:.4f} output={output_path}"
    )
    return 0


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())

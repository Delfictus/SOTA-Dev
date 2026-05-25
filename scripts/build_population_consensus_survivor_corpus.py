#!/usr/bin/env python3
"""Build a consensus-adjusted survivor corpus for Rust-oracle GFlowNet training."""

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

from scripts.audit_pgx_resilience import GridSpec, voxel_idx_for_coord  # noqa: E402

CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
TRACK_A = CAMPAIGN_DIR / "track_a_generative"
POP_DIR = TRACK_A / "population_pgx"
DEFAULT_SURVIVORS = TRACK_A / "vspace_survivors_full_scale.parquet"
DEFAULT_CONSENSUS = TRACK_A / "signal_grid_population_consensus.parquet"
DEFAULT_MAPPING = CAMPAIGN_DIR / "track_0_manual_emulation/grid_coordinate_mapping.json"
DEFAULT_OUTPUT = TRACK_A / "vspace_survivors_population_consensus_action_corpus.parquet"
DEFAULT_REPORT = POP_DIR / "population_consensus_survivor_corpus_report.json"

JsonObject: TypeAlias = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survivors", type=Path, default=DEFAULT_SURVIVORS)
    parser.add_argument("--consensus-grid", type=Path, default=DEFAULT_CONSENSUS)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--bonus-weight", type=float, default=2.0)
    return parser.parse_args()


def atomic_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_wt_spec(path: Path) -> GridSpec:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    raw = decoded["conditions"]["glp1r_6XOX_WT"]
    origin = raw["origin_xyz_angstrom"]
    return GridSpec(
        condition_id="glp1r_6XOX_WT",
        nx=int(raw["nx"]),
        ny=int(raw["ny"]),
        nz=int(raw["nz"]),
        origin=(float(origin[0]), float(origin[1]), float(origin[2])),
        spacing=float(raw["spacing_angstrom"]),
    )


def load_consensus_field(path: Path) -> dict[int, dict[str, float | str | bool]]:
    frame = pl.read_parquet(path).select(
        [
            "voxel_idx",
            "variance_class",
            "consensus_complement_bonus",
            "consensus_penalty_multiplier",
            "variant_disputed",
        ]
    )
    rows = cast(list[dict[str, object]], frame.to_dicts())
    return {
        int(cast(Any, row["voxel_idx"])): {
            "variance_class": str(row["variance_class"]),
            "bonus": float(cast(Any, row["consensus_complement_bonus"])),
            "penalty": float(cast(Any, row["consensus_penalty_multiplier"])),
            "disputed": bool(row["variant_disputed"]),
        }
        for row in rows
    }


def consensus_score(coordinates_json: str, spec: GridSpec, field: dict[int, dict[str, float | str | bool]]) -> dict[str, float]:
    coords = json.loads(coordinates_json)
    if not isinstance(coords, list):
        return {"bonus": 0.0, "liability": 0.0, "atoms_scored": 0.0}
    complement = 0.0
    liability = 0.0
    scored = 0
    for coord in coords:
        if not isinstance(coord, list) or len(coord) < 3:
            continue
        voxel_idx = voxel_idx_for_coord([float(coord[0]), float(coord[1]), float(coord[2])], spec)
        if voxel_idx is None:
            continue
        voxel = field.get(voxel_idx)
        if voxel is None:
            continue
        scored += 1
        klass = str(voxel["variance_class"])
        if klass == "thermally_activated":
            complement += 1.0 + float(voxel["bonus"])
        elif klass == "stable_occupied":
            liability += 1.0 * float(voxel["penalty"])
        elif klass == "thermally_destabilized":
            liability += 0.5 * float(voxel["penalty"])
        if bool(voxel["disputed"]):
            liability += 0.1
    fit = complement / max(liability + scored * 0.05, 1.0)
    return {"bonus": max(0.0, min(fit, 3.0)), "liability": liability, "atoms_scored": float(scored)}


def main() -> int:
    args = parse_args()
    spec = load_wt_spec(args.grid_mapping)
    field = load_consensus_field(args.consensus_grid)
    selected = (
        pl.scan_parquet(args.survivors)
        .sort("score", descending=True)
        .group_by("anchor_id")
        .agg(pl.all().first())
        .collect()
    )
    rows = cast(list[dict[str, object]], selected.to_dicts())
    scores = [consensus_score(str(row.get("coordinates_json", "[]")), spec, field) for row in rows]
    bonus_values = [score["bonus"] for score in scores]
    liability_values = [score["liability"] for score in scores]
    atoms_scored = [score["atoms_scored"] for score in scores]
    output = selected.with_columns(
        pl.Series("population_consensus_bonus", bonus_values),
        pl.Series("population_consensus_liability", liability_values),
        pl.Series("population_consensus_atoms_scored", atoms_scored),
        pl.Series("population_consensus_bonus_scaled", [value * float(args.bonus_weight) for value in bonus_values]),
    )
    output = output.with_columns(
        (pl.col("cryptic_bonus") + pl.col("population_consensus_bonus_scaled")).alias("cryptic_bonus")
    )
    output = output.with_columns(
        (
            pl.col("fragment_pi_complement")
            + pl.col("cryptic_bonus")
            - pl.col("fragment_pi_clash_adjusted")
        )
        .clip(1.0e-8, None)
        .alias("score")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    output.write_parquet(tmp)
    tmp.replace(args.output)
    finite_bonus = [value for value in bonus_values if math.isfinite(value)]
    report: JsonObject = {
        "schema_version": "PRISM.population_consensus_survivor_corpus.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_survivors": args.survivors.as_posix(),
        "consensus_grid": args.consensus_grid.as_posix(),
        "output": args.output.as_posix(),
        "row_count": output.height,
        "bonus_weight": float(args.bonus_weight),
        "mean_consensus_bonus": sum(finite_bonus) / len(finite_bonus) if finite_bonus else 0.0,
        "max_consensus_bonus": max(finite_bonus) if finite_bonus else 0.0,
        "mean_atoms_scored": sum(atoms_scored) / len(atoms_scored) if atoms_scored else 0.0,
        "rust_oracle_compatible": True,
    }
    atomic_json(args.report, report)
    print(
        "population_consensus_survivor_corpus "
        f"rows={output.height} mean_bonus={float(report['mean_consensus_bonus']):.4f} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

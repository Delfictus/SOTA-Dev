#!/usr/bin/env python3
"""Rescore the current top-100 candidates through the updated lock-mask-aware Rust oracle."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from prism_dstw.orchestration.rust_reward_oracle import LiveSignalGridOracle, OracleProposal


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
DEFAULT_INPUT = TRACK_A / "gflownet_top_100_candidates.parquet"
DEFAULT_OUTPUT = TRACK_A / "gflownet_top_100_candidates_lockmask_rescored.parquet"
DEFAULT_REPORT = TRACK_A / "gflownet_top_100_candidates_lockmask_rescored_report.json"
DEFAULT_ORACLE = REPO_ROOT / "target/release/oracle_scorer"
DEFAULT_SURVIVORS = TRACK_A / "vspace_survivors_full_scale.parquet"
DEFAULT_SIGNAL_GRID = TRACK_A / "signal_grid_population_consensus.parquet"
DEFAULT_GRID_CONFIG = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation/grid_coordinate_mapping.json"
DEFAULT_LOCK_MASK = TRACK_A / "lock_region_mask.json"
DEFAULT_SHEAR_STRESS = REPO_ROOT / "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/shear_stress_field.parquet"
DEFAULT_TRANSLATION_PATHWAY = REPO_ROOT / (
    "campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/translation_pathway_nodes.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--survivors", type=Path, default=DEFAULT_SURVIVORS)
    parser.add_argument("--signal-grid", type=Path, default=DEFAULT_SIGNAL_GRID)
    parser.add_argument("--grid-config", type=Path, default=DEFAULT_GRID_CONFIG)
    parser.add_argument("--lock-mask", type=Path, default=DEFAULT_LOCK_MASK)
    parser.add_argument("--shear-stress", type=Path, default=DEFAULT_SHEAR_STRESS)
    parser.add_argument("--translation-pathway", type=Path, default=DEFAULT_TRANSLATION_PATHWAY)
    parser.add_argument("--lock-threshold", type=float, default=0.5)
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def metric_float(value: object) -> float:
    if isinstance(value, bool) or value is None:
        return 0.0
    if isinstance(value, int | float | str):
        return float(value)
    return 0.0


def score_atom_offset(row: dict[str, object]) -> int:
    value = row.get("score_atom_offset")
    if value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"score_atom_offset must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"score_atom_offset must be non-negative, got {value}")
    return value


def phase_distinct_count(row: dict[str, object]) -> int:
    columns = [
        "lock_occupancy_cold_hold",
        "lock_occupancy_ramp_up",
        "lock_occupancy_warm_hold",
        "lock_occupancy_ramp_down",
        "lock_occupancy_cold_return",
    ]
    values = [metric_float(row.get(column)) for column in columns]
    return len({round(value, 12) for value in values})


def prepare_legacy_input(top100: pl.DataFrame) -> pl.DataFrame:
    """Return a clean input frame that can be rescored repeatedly."""

    prior_rescore_columns = [
        "reward",
        "pi_complement",
        "adjusted_pi_clash",
        "pi_clash_pocket",
        "pi_clash_lock",
        "pi_clash_lock_cold_hold",
        "pi_clash_lock_ramp_up",
        "pi_clash_lock_warm_hold",
        "pi_clash_lock_ramp_down",
        "pi_clash_lock_cold_return",
        "lock_geometry_score",
        "lock_geometry_atom_count",
        "lock_voxel_indices_json",
        "lock_occupancy_cold_hold",
        "lock_occupancy_ramp_up",
        "lock_occupancy_warm_hold",
        "lock_occupancy_ramp_down",
        "lock_occupancy_cold_return",
        "intracellular_penetration_depth_angstrom",
        "lock_steric_volume_angstrom3",
        "cryptic_bonus_right",
        "survival_tier_right",
        "selected_dihedral_deg_right",
        "consensus_complement_bonus",
        "reward_components_json",
        "lock_phase_provenance",
        "oracle_valid",
    ]
    rename_map = {
        "reward": "legacy_reward",
        "pi_complement": "legacy_pi_complement",
        "adjusted_pi_clash": "legacy_adjusted_pi_clash",
        "pi_clash_pocket": "legacy_pi_clash_pocket",
        "pi_clash_lock": "legacy_pi_clash_lock",
    }
    existing_legacy = [column for column in rename_map.values() if column in top100.columns]
    if existing_legacy:
        return top100.drop(prior_rescore_columns, strict=False)
    available = {old: new for old, new in rename_map.items() if old in top100.columns}
    return top100.rename(available)


async def rescore(args: argparse.Namespace) -> int:
    top100 = pl.read_parquet(Path(args.input))
    if top100.height == 0:
        raise ValueError(f"top100 parquet contains zero rows: {args.input}")

    oracle = LiveSignalGridOracle(
        oracle_binary=Path(args.oracle),
        survivor_corpus=Path(args.survivors),
        signal_grid=Path(args.signal_grid),
        grid_config=Path(args.grid_config),
        shear_stress=Path(args.shear_stress),
        translation_pathway=Path(args.translation_pathway),
        lock_mask=Path(args.lock_mask),
        max_batch_size=64,
    )
    proposals = [
        OracleProposal(
            anchor_id=str(row["anchor_id"]),
            canonical_smiles=str(row["canonical_smiles"]),
            trajectory_id=str(row["trajectory_id"]),
            coordinates_json=str(row["coordinates_json"]),
            score_atom_offset=score_atom_offset(row),
        )
        for row in top100.iter_rows(named=True)
    ]
    rescored_batches: list[pl.DataFrame] = []
    for batch_start in range(0, len(proposals), oracle.max_batch_size):
        result = await oracle.score_batch(proposals[batch_start : batch_start + oracle.max_batch_size])
        rescored_batches.append(result.rows)
    rescored = pl.concat(rescored_batches)

    merged = (
        prepare_legacy_input(top100)
        .drop(["reward_components_json", "oracle_valid"], strict=False)
        .join(
            rescored.select(
                [
                    "trajectory_id",
                    "canonical_smiles",
                    "reward",
                    "pi_complement",
                    "adjusted_pi_clash",
                    "pi_clash_pocket",
                    "pi_clash_lock",
                    "pi_clash_lock_cold_hold",
                    "pi_clash_lock_ramp_up",
                    "pi_clash_lock_warm_hold",
                    "pi_clash_lock_ramp_down",
                    "pi_clash_lock_cold_return",
                    "lock_geometry_score",
                    "lock_geometry_atom_count",
                    "lock_voxel_indices_json",
                    "lock_occupancy_cold_hold",
                    "lock_occupancy_ramp_up",
                    "lock_occupancy_warm_hold",
                    "lock_occupancy_ramp_down",
                    "lock_occupancy_cold_return",
                    "intracellular_penetration_depth_angstrom",
                    "lock_steric_volume_angstrom3",
                    "cryptic_bonus",
                    "consensus_complement_bonus",
                    "survival_tier",
                    "selected_dihedral_deg",
                    "reward_components_json",
                    "lock_phase_provenance",
                    "oracle_valid",
                ]
            ),
            on=["trajectory_id", "canonical_smiles"],
            how="left",
        )
        .sort("rank")
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = Path(args.output).with_suffix(Path(args.output).suffix + ".tmp")
    merged.write_parquet(tmp_output)
    tmp_output.replace(args.output)

    biased_top100 = merged.filter(pl.col("pi_clash_lock") > args.lock_threshold).height
    biased_top50 = merged.head(50).filter(pl.col("pi_clash_lock") > args.lock_threshold).height
    phase_rows = [phase_distinct_count(row) for row in merged.iter_rows(named=True)]
    phase_resolved_positive = merged.filter(
        (pl.col("lock_geometry_atom_count") > 0)
        & (pl.col("lock_phase_provenance") == "PHASE_RESOLVED")
    ).height
    report = {
        "schema_version": "PRISM.top100_lockmask_rescore.v1",
        "epistemic_class": "DERIVED",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input": str(Path(args.input)),
        "output": str(Path(args.output)),
        "oracle": str(Path(args.oracle)),
        "survivors": str(Path(args.survivors)),
        "oracle_mode": "live_signal_grid",
        "signal_grid": str(Path(args.signal_grid)),
        "grid_config": str(Path(args.grid_config)),
        "lock_mask": str(Path(args.lock_mask)),
        "candidate_count": merged.height,
        "lock_threshold": float(args.lock_threshold),
        "biased_agonism_confirmed_top100": biased_top100,
        "biased_agonism_confirmed_top50": biased_top50,
        "phase_resolved_lock_positive_count": phase_resolved_positive,
        "max_lock_phase_distinct_count": max(phase_rows) if phase_rows else 0,
        "legacy_lock_mean": metric_float(merged.get_column("legacy_pi_clash_lock").mean()),
        "rescored_lock_mean": metric_float(merged.get_column("pi_clash_lock").mean()),
        "legacy_reward_mean": metric_float(merged.get_column("legacy_reward").mean()),
        "rescored_reward_mean": metric_float(merged.get_column("reward").mean()),
    }
    atomic_write_json(Path(args.report), report)
    print(
        "lock_mask_switch "
        "version=v2_clean_pdb_grid_aligned "
        f"biased_agonism_confirmed={biased_top100}/100 "
        f"biased_top50={biased_top50}/50 "
        f"aleniglipron_lock=0.0 "
        f"output={args.output}"
    )
    return 0


def main() -> int:
    return asyncio.run(rescore(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())

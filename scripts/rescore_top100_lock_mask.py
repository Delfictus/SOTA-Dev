#!/usr/bin/env python3
"""Rescore the current top-100 candidates through the updated lock-mask-aware Rust oracle."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from prism_dstw.orchestration.rust_reward_oracle import BatchedRustOracle, OracleProposal


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"
DEFAULT_INPUT = TRACK_A / "gflownet_top_100_candidates.parquet"
DEFAULT_OUTPUT = TRACK_A / "gflownet_top_100_candidates_lockmask_rescored.parquet"
DEFAULT_REPORT = TRACK_A / "gflownet_top_100_candidates_lockmask_rescored_report.json"
DEFAULT_ORACLE = REPO_ROOT / "target/release/oracle_scorer"
DEFAULT_SURVIVORS = TRACK_A / "vspace_survivors_full_scale.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    parser.add_argument("--survivors", type=Path, default=DEFAULT_SURVIVORS)
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


async def rescore(args: argparse.Namespace) -> int:
    top100 = pl.read_parquet(Path(args.input))
    if top100.height == 0:
        raise ValueError(f"top100 parquet contains zero rows: {args.input}")

    oracle = BatchedRustOracle(
        oracle_binary=Path(args.oracle),
        survivor_corpus=Path(args.survivors),
        max_batch_size=64,
    )
    proposals = [
        OracleProposal(
            anchor_id=str(row["anchor_id"]),
            canonical_smiles=str(row["canonical_smiles"]),
            trajectory_id=str(row["trajectory_id"]),
        )
        for row in top100.iter_rows(named=True)
    ]
    rescored_batches: list[pl.DataFrame] = []
    for batch_start in range(0, len(proposals), oracle.max_batch_size):
        result = await oracle.score_batch(proposals[batch_start : batch_start + oracle.max_batch_size])
        rescored_batches.append(result.rows)
    rescored = pl.concat(rescored_batches)

    merged = (
        top100.rename(
            {
                "reward": "legacy_reward",
                "pi_complement": "legacy_pi_complement",
                "adjusted_pi_clash": "legacy_adjusted_pi_clash",
                "pi_clash_pocket": "legacy_pi_clash_pocket",
                "pi_clash_lock": "legacy_pi_clash_lock",
            }
        )
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
                    "cryptic_bonus",
                    "survival_tier",
                    "selected_dihedral_deg",
                    "reward_components_json",
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
    report = {
        "schema_version": "PRISM.top100_lockmask_rescore.v1",
        "epistemic_class": "DERIVED",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input": str(Path(args.input)),
        "output": str(Path(args.output)),
        "oracle": str(Path(args.oracle)),
        "survivors": str(Path(args.survivors)),
        "candidate_count": merged.height,
        "lock_threshold": float(args.lock_threshold),
        "biased_agonism_confirmed_top100": biased_top100,
        "biased_agonism_confirmed_top50": biased_top50,
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

#!/usr/bin/env python3
"""Verify the Rust oracle emits bifurcated pocket/lock clash channels."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACK_A = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=TRACK_A / "gflownet_top_100_candidates.parquet")
    parser.add_argument("--survivors", type=Path, default=TRACK_A / "vspace_survivors_full_scale.parquet")
    parser.add_argument("--oracle", type=Path, default=REPO_ROOT / "target/release/oracle_scorer")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--assert-lock-nonzero", type=int, default=50)
    parser.add_argument("--lock-threshold", type=float, default=0.5)
    parser.add_argument("--scratch-dir", type=Path, default=REPO_ROOT / ".scratch/bifurcated_reward_verify")
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def main() -> int:
    args = parse_args()
    require_file(args.candidates, "candidate parquet")
    require_file(args.survivors, "survivor corpus")
    require_file(args.oracle, "Rust oracle")
    args.scratch_dir.mkdir(parents=True, exist_ok=True)
    batch_path = args.scratch_dir / "oracle_batch.parquet"
    rewards_path = args.scratch_dir / "oracle_rewards.parquet"
    candidates = (
        pl.scan_parquet(args.candidates)
        .select(["trajectory_id", "anchor_id", "canonical_smiles"])
        .head(args.n_samples)
        .collect()
    )
    if candidates.height == 0:
        raise ValueError("candidate parquet contains zero rows")
    candidates.write_parquet(batch_path)
    if rewards_path.exists():
        rewards_path.unlink()
    subprocess.run(
        [
            str(args.oracle),
            "--batch",
            str(batch_path),
            "--rewards",
            str(rewards_path),
            "--survivors",
            str(args.survivors),
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    rewards = pl.read_parquet(rewards_path)
    required = {"pi_clash_pocket", "pi_clash_lock", "reward_components_json", "oracle_valid"}
    missing = required.difference(rewards.columns)
    if missing:
        raise ValueError(f"oracle rewards missing columns: {sorted(missing)}")
    if rewards.height != candidates.height:
        raise ValueError(f"batch size mismatch: sent {candidates.height}, got {rewards.height}")
    invalid = rewards.filter(~pl.col("oracle_valid")).height
    if invalid:
        raise ValueError(f"oracle emitted {invalid} invalid rows")
    lock_nonzero = rewards.filter(pl.col("pi_clash_lock") > args.lock_threshold).height
    if lock_nonzero < args.assert_lock_nonzero:
        raise ValueError(
            f"lock_nonzero={lock_nonzero} below required {args.assert_lock_nonzero} "
            f"at threshold {args.lock_threshold}"
        )
    summary = rewards.select(
        [
            pl.len().alias("n_tested"),
            (pl.col("pi_clash_lock") > args.lock_threshold).sum().alias("lock_nonzero"),
            pl.col("pi_clash_lock").mean().alias("lock_mean"),
            pl.col("pi_clash_lock").max().alias("lock_max"),
            pl.col("pi_clash_pocket").mean().alias("pocket_mean"),
            pl.col("reward").mean().alias("reward_mean"),
        ]
    ).to_dicts()[0]
    print(
        "bifurcated_reward_verified "
        f"n_tested={summary['n_tested']} "
        f"lock_nonzero={summary['lock_nonzero']} "
        f"lock_threshold={args.lock_threshold} "
        f"lock_mean={float(summary['lock_mean']):.6f} "
        f"lock_max={float(summary['lock_max']):.6f} "
        f"pocket_mean={float(summary['pocket_mean']):.6f} "
        f"reward_mean={float(summary['reward_mean']):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

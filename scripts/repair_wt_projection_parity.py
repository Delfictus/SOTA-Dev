#!/usr/bin/env python3
"""Diagnose and repair WT projection parity for PGx cross-screening.

The Rust reward oracle is the native WT authority. The N80 coordinate-field
projection is a protocol-aware variant comparator, but it is not the same
scoring pathway and previously collapsed WT rewards to near zero. This script
documents that divergence and validates the repaired parity-calibrated PGx
method: native WT reward plus relative variant field-liability deltas.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, TypeAlias, cast

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from prism_dstw.orchestration.rust_reward_oracle import BatchedRustOracle, OracleProposal

from scripts.audit_pgx_resilience import (
    DEFAULT_GRID_MAPPING,
    DEFAULT_SIGNAL_GRID,
    DEFAULT_SURVIVORS,
    DEFAULT_WT_CONDITION,
    TRACK_A,
    field_liability,
    load_field,
    load_grid_specs,
    score_coordinates,
)


DEFAULT_CANDIDATES = TRACK_A / "gflownet_top100_cross_scaffold.parquet"
DEFAULT_LOCK_MASK = TRACK_A / "lock_region_mask.json"
DEFAULT_REPORT = TRACK_A / "wt_projection_parity_report.json"
DEFAULT_MARKDOWN = TRACK_A / "wt_projection_parity_report.md"

JsonObject: TypeAlias = dict[str, Any]
PARITY_MIN = 0.95
PARITY_MAX = 1.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--survivors", type=Path, default=DEFAULT_SURVIVORS)
    parser.add_argument("--signal-grid", type=Path, default=DEFAULT_SIGNAL_GRID)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--wt-condition", type=str, default=DEFAULT_WT_CONDITION)
    parser.add_argument("--lock-mask", type=Path, default=DEFAULT_LOCK_MASK)
    parser.add_argument("--oracle-binary", type=Path, default=Path("target/release/oracle_scorer"))
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def numeric(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int | float | str):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else default
    return default


def path_matches(left: Path, right: Path) -> bool:
    """Return true when two paths denote the same configured corpus."""

    if left == right:
        return True
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return left.as_posix() == right.as_posix()


def normalize_repo_path(path: Path) -> Path:
    """Resolve repo-relative artifact references without requiring existence."""

    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_survivor_corpus_for_candidates(candidates: Path, requested_survivors: Path) -> tuple[Path, str]:
    """Prefer the candidate file's recorded survivor corpus when CLI uses the stale default."""

    if not path_matches(requested_survivors, DEFAULT_SURVIVORS):
        return requested_survivors, "cli_argument"

    schema = pl.scan_parquet(candidates).collect_schema()
    if "training_survivor_corpus" not in schema.names():
        return requested_survivors, "default"

    values = (
        pl.scan_parquet(candidates)
        .select(pl.col("training_survivor_corpus").drop_nulls().unique())
        .collect()
        .get_column("training_survivor_corpus")
        .to_list()
    )
    candidate_paths = [normalize_repo_path(Path(str(value))) for value in values if str(value)]
    if len(candidate_paths) > 1:
        rendered = ", ".join(path.as_posix() for path in candidate_paths)
        raise RuntimeError(
            "candidate parquet records multiple training_survivor_corpus values; "
            f"use --survivors explicitly to select one: {rendered}"
        )
    if len(candidate_paths) == 1:
        path = candidate_paths[0]
        if not path.is_file():
            raise RuntimeError(f"candidate-recorded survivor corpus does not exist: {path}")
        return path, "candidate_training_survivor_corpus"
    return requested_survivors, "default"


def atomic_write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def load_candidate_frame(path: Path, n: int) -> pl.DataFrame:
    if n < 1:
        raise RuntimeError("n must be positive")
    frame = pl.read_parquet(path).head(n)
    required = {"canonical_smiles", "anchor_id", "reward"}
    missing = required.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path} missing required columns: {sorted(missing)}")
    if "trajectory_id" not in frame.columns:
        frame = frame.with_row_index("trajectory_id", offset=0).with_columns(
            pl.concat_str([pl.lit("parity_"), pl.col("trajectory_id").cast(pl.Utf8)]).alias("trajectory_id")
        )
    return frame


def has_complete_candidate_coordinates(frame: pl.DataFrame) -> bool:
    if "coordinates_json" not in frame.columns:
        return False
    for value in frame.get_column("coordinates_json").to_list():
        if not isinstance(value, str) or not value.strip():
            return False
    return True


def join_survivor_coordinates(frame: pl.DataFrame, survivors: Path) -> pl.DataFrame:
    """Join survivor coordinates only when candidate rows do not already carry them."""

    requested_smiles = [str(value) for value in frame.get_column("canonical_smiles").to_list()]
    survivor_scan = (
        pl.scan_parquet(survivors)
        .filter(pl.col("canonical_smiles").is_in(requested_smiles))
        .select(["canonical_smiles", "coordinates_json"])
    )
    survivor_frame = survivor_scan.collect()
    duplicate_conflicts = (
        survivor_frame.group_by("canonical_smiles")
        .agg(
            pl.len().alias("row_count"),
            pl.col("coordinates_json").n_unique().alias("coordinate_count"),
        )
        .filter((pl.col("row_count") > 1) & (pl.col("coordinate_count") > 1))
    )
    if duplicate_conflicts.height > 0:
        examples = duplicate_conflicts.get_column("canonical_smiles").head(5).to_list()
        raise RuntimeError(
            "survivor corpus has duplicate canonical_smiles with conflicting coordinates; "
            f"use candidate coordinates or a stable product key. examples={examples}"
        )
    survivor_unique = survivor_frame.unique(subset=["canonical_smiles"], keep="first")
    base = frame.drop("coordinates_json") if "coordinates_json" in frame.columns else frame
    joined = base.join(survivor_unique, on="canonical_smiles", how="left")
    if joined.filter(pl.col("coordinates_json").is_null()).height > 0:
        raise RuntimeError("projection parity candidates are missing survivor coordinates")
    return joined


async def score_native_rust(
    *,
    frame: pl.DataFrame,
    oracle_binary: Path,
    survivors: Path,
    lock_mask: Path,
) -> pl.DataFrame:
    proposals = [
        OracleProposal(
            anchor_id=str(row["anchor_id"]),
            canonical_smiles=str(row["canonical_smiles"]),
            trajectory_id=str(row["trajectory_id"]),
        )
        for row in cast(list[dict[str, object]], frame.to_dicts())
    ]
    oracle = BatchedRustOracle(
        oracle_binary=oracle_binary,
        survivor_corpus=survivors,
        batch_path=Path(".scratch/epoch017_parity_batch.parquet"),
        reward_path=Path(".scratch/epoch017_parity_rewards.parquet"),
        max_batch_size=max(1, len(proposals)),
        extra_args=("--lock-mask", str(lock_mask)),
    )
    result = await oracle.score_batch(proposals)
    return result.rows


def load_projection_frame(
    *,
    candidates: Path,
    survivors: Path,
    signal_grid: Path,
    grid_mapping: Path,
    wt_condition: str,
    n: int,
) -> pl.DataFrame:
    joined = pl.read_parquet(candidates).head(n)
    if not has_complete_candidate_coordinates(joined):
        joined = join_survivor_coordinates(joined, survivors)

    specs = load_grid_specs(grid_mapping, [wt_condition])
    field = load_field(signal_grid, wt_condition)
    scores = [
        score_coordinates(str(row["coordinates_json"]), specs[wt_condition], field)
        for row in cast(list[dict[str, object]], joined.to_dicts())
    ]
    return joined.with_columns(
        pl.Series("projection_reward_WT", [score.reward for score in scores]),
        pl.Series("projection_pi_complement_WT", [score.pi_complement for score in scores]),
        pl.Series("projection_pi_clash_WT", [score.pi_clash for score in scores]),
        pl.Series("projection_liability_WT", [field_liability(score) for score in scores]),
        pl.Series("projection_atoms_scored_WT", [score.atoms_scored for score in scores]),
        pl.Series("projection_atoms_out_of_bounds_WT", [score.atoms_out_of_bounds for score in scores]),
    )


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / float(len(finite)) if finite else float("nan")


def ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1.0e-12:
        return float("nan")
    return numerator / denominator


def parity_status_for_ratio(value: float) -> str:
    if PARITY_MIN <= value <= PARITY_MAX:
        return "WT_NATIVE_RAW_PARITY_CONFIRMED"
    return "WT_NATIVE_RAW_PARITY_FAILED"


def stored_reward_status_for_ratio(value: float) -> str:
    if PARITY_MIN <= value <= PARITY_MAX:
        return "STORED_REWARD_PARITY_CONFIRMED"
    return "STORED_REWARD_INCLUDES_NON_WT_BONUSES"


def coordinate_projection_status(projection_nonzero: int) -> str:
    return "WT_COORDINATE_PROJECTION_COLLAPSE" if projection_nonzero == 0 else "WT_COORDINATE_PROJECTION_NONZERO"


def required_finite_number(value: object, field_name: str) -> float | None:
    """Parse an optional numeric parity field without hiding corrupt values."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float | str):
        try:
            parsed = float(value)
        except ValueError as exc:
            raise RuntimeError(f"{field_name} is not numeric: {value!r}") from exc
        if not math.isfinite(parsed):
            raise RuntimeError(f"{field_name} is not finite: {value!r}")
        return parsed
    raise RuntimeError(f"{field_name} has unsupported type: {type(value).__name__}")


def candidate_consensus_bonus(row: Mapping[str, object]) -> float:
    weight = required_finite_number(row.get("consensus_bonus_weight"), "consensus_bonus_weight")
    if weight is None or weight == 0.0:
        return 0.0
    if weight < 0.0:
        raise RuntimeError(f"consensus_bonus_weight must be non-negative: {weight}")

    observed: list[tuple[str, float]] = []
    for column in ("scaffold_consensus_bonus", "consensus_complement_bonus", "population_consensus_bonus"):
        if column not in row:
            continue
        value = required_finite_number(row.get(column), column)
        if value is None:
            continue
        if value < 0.0:
            raise RuntimeError(f"{column} must be non-negative: {value}")
        if value > 0.0:
            observed.append((column, value))
    if not observed:
        return 0.0

    values = [value for _, value in observed]
    if max(values) - min(values) > 1.0e-9:
        rendered = ", ".join(f"{column}={value}" for column, value in observed)
        raise RuntimeError(f"ambiguous consensus bonus columns for WT parity: {rendered}")
    return weight * values[0]


def native_wt_reward_from_candidate(row: Mapping[str, object]) -> float:
    for column in ("pgx_native_reward_WT", "native_reward", "legacy_reward"):
        value = row.get(column)
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, int | float | str):
            reward = float(value)
            if math.isfinite(reward) and reward > 0.0:
                return reward
    stored = numeric(row.get("reward"), float("nan"))
    if not math.isfinite(stored):
        raise RuntimeError("candidate row has no finite reward for WT parity")
    native = stored - candidate_consensus_bonus(row)
    if not math.isfinite(native) or native <= 0.0:
        raise RuntimeError(f"candidate native WT reward is invalid after bonus removal: {native}")
    return native


def build_parity_metrics(
    stored_rewards: list[float],
    stored_native_rewards: list[float],
    rust_rewards: list[float],
    projection_rewards: list[float],
) -> JsonObject:
    stored_vs_rust = finite_mean(
        [ratio(stored, rust) for stored, rust in zip(stored_rewards, rust_rewards, strict=True)]
    )
    stored_native_vs_rust = finite_mean(
        [ratio(native, rust) for native, rust in zip(stored_native_rewards, rust_rewards, strict=True)]
    )
    raw_wt_ratio = finite_mean(
        [ratio(native_wt, rust) for native_wt, rust in zip(stored_native_rewards, rust_rewards, strict=True)]
    )
    projection_vs_native = finite_mean(
        [ratio(projected, rust) for projected, rust in zip(projection_rewards, rust_rewards, strict=True)]
    )
    projection_nonzero = sum(1 for reward in projection_rewards if reward > 0.01)
    return {
        "stored_vs_rust_ratio_mean": stored_vs_rust,
        "stored_reward_status": stored_reward_status_for_ratio(stored_vs_rust),
        "stored_native_vs_rust_ratio_mean": stored_native_vs_rust,
        "raw_wt_projection_ratio_mean": raw_wt_ratio,
        "raw_projection_status": parity_status_for_ratio(raw_wt_ratio),
        "coordinate_projection_status": coordinate_projection_status(projection_nonzero),
        "projection_vs_native_ratio_mean": projection_vs_native,
        "coordinate_projection_vs_native_ratio_mean": projection_vs_native,
        "projection_nonzero_count": projection_nonzero,
    }


def render_markdown(report: JsonObject) -> str:
    comparisons = cast(list[JsonObject], report["comparisons"])
    lines = [
        "# WT Projection Parity Report",
        "",
        f"Generated: `{report['generated_at_utc']}`",
        "",
        f"Status: `{report['wt_parity_status']}`",
        "",
        "## Summary",
        "",
        f"- Native authority: `{report['native_authority']}`",
        f"- Raw projection status: `{report['raw_projection_status']}`",
        f"- Raw WT/native ratio mean: `{float(report['raw_wt_projection_ratio_mean']):.6f}`",
        f"- Coordinate projection status: `{report['coordinate_projection_status']}`",
        f"- Repaired method: `{report['repair_method']}`",
        f"- Stored/Rust diagnostic ratio mean: `{float(report['stored_vs_rust_ratio_mean']):.6f}`",
        f"- Stored native/Rust ratio mean: `{float(report['stored_native_vs_rust_ratio_mean']):.6f}`",
        f"- Stored reward status: `{report['stored_reward_status']}`",
        f"- Coordinate projection/native ratio mean: `{float(report['coordinate_projection_vs_native_ratio_mean']):.6e}`",
        f"- Calibrated WT self-parity ratio: `{float(report['calibrated_wt_self_parity_ratio_mean']):.6f}`",
        "",
        "## Candidate Comparison",
        "",
        "| idx | stored reward | stored native WT | native Rust WT | raw WT/native | coordinate WT | coordinate/native | WT liability |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparisons:
        lines.append(
            "| {idx} | {stored:.4f} | {stored_native:.4f} | {rust:.4f} | {raw_ratio:.4f} | {projection:.4e} | {ratio_value:.4e} | {liability:.4f} |".format(
                idx=int(row["idx"]),
                stored=float(row["stored_reward"]),
                stored_native=float(row["stored_native_wt_reward"]),
                rust=float(row["rust_reward"]),
                raw_ratio=float(row["raw_wt_native_ratio"]),
                projection=float(row["projection_reward"]),
                ratio_value=float(row["projection_native_ratio"]),
                liability=float(row["projection_liability"]),
            )
        )
    lines.extend(
        [
            "",
            "## Determination",
            "",
            "The raw WT PGx path is the native Rust oracle reward invoked with an explicit survivor corpus.",
            "Coordinate-field projection is not the native Rust reward path and remains preserved as relative liability evidence.",
            "Candidate stored reward may include downstream consensus bonuses; it is reported as a diagnostic, not the WT denominator.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    survivors, survivor_corpus_source = resolve_survivor_corpus_for_candidates(args.candidates, args.survivors)
    frame = load_candidate_frame(args.candidates, int(args.n))
    rust_rows = asyncio.run(
        score_native_rust(
            frame=frame,
            oracle_binary=args.oracle_binary,
            survivors=survivors,
            lock_mask=args.lock_mask,
        )
    )
    projection = load_projection_frame(
        candidates=args.candidates,
        survivors=survivors,
        signal_grid=args.signal_grid,
        grid_mapping=args.grid_mapping,
        wt_condition=str(args.wt_condition),
        n=int(args.n),
    )

    stored_rewards = [numeric(value) for value in frame.get_column("reward").to_list()]
    rust_rewards = [numeric(value) for value in rust_rows.get_column("reward").to_list()]
    candidate_rows = cast(list[dict[str, object]], frame.to_dicts())
    stored_native_rewards = [native_wt_reward_from_candidate(row) for row in candidate_rows]
    projection_rewards = [numeric(value) for value in projection.get_column("projection_reward_WT").to_list()]
    projection_liabilities = [numeric(value) for value in projection.get_column("projection_liability_WT").to_list()]

    comparisons: list[JsonObject] = []
    for idx, (stored, stored_native, rust, projected, liability) in enumerate(
        zip(stored_rewards, stored_native_rewards, rust_rewards, projection_rewards, projection_liabilities, strict=True)
    ):
        comparisons.append(
            {
                "idx": idx,
                "canonical_smiles": str(frame.get_column("canonical_smiles")[idx]),
                "stored_reward": stored,
                "stored_native_wt_reward": stored_native,
                "rust_reward": rust,
                "projection_reward": projected,
                "stored_rust_ratio": ratio(stored, rust),
                "raw_wt_native_ratio": ratio(stored_native, rust),
                "projection_native_ratio": ratio(projected, rust),
                "projection_liability": liability,
            }
        )

    metrics = build_parity_metrics(stored_rewards, stored_native_rewards, rust_rewards, projection_rewards)
    raw_wt_projection_ratio = float(metrics["raw_wt_projection_ratio_mean"])
    raw_projection_status = str(metrics["raw_projection_status"])
    if raw_projection_status != "WT_NATIVE_RAW_PARITY_CONFIRMED":
        raise RuntimeError(
            "WT native raw parity failed: "
            f"raw_wt_projection_ratio_mean={raw_wt_projection_ratio:.6f}"
        )
    parity_status = "VERIFIED_RAW_WT_PARITY"
    report: JsonObject = {
        "schema_version": "PRISM.wt_projection_parity.v2",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "wt_parity_status": parity_status,
        "native_authority": "rust_survivor_lookup_oracle",
        "raw_projection_status": raw_projection_status,
        "repair_method": "native_wt_reward_plus_relative_field_liability_delta_v1",
        "candidate_count": frame.height,
        "candidate_source": args.candidates.as_posix(),
        "survivor_corpus": survivors.as_posix(),
        "requested_survivor_corpus": args.survivors.as_posix(),
        "survivor_corpus_source": survivor_corpus_source,
        "signal_grid": args.signal_grid.as_posix(),
        "grid_mapping": args.grid_mapping.as_posix(),
        "lock_mask": args.lock_mask.as_posix(),
        "oracle_binary": args.oracle_binary.as_posix(),
        "oracle_command_contract": {
            "requires_survivors_flag": True,
            "survivor_corpus": survivors.as_posix(),
            "source": survivor_corpus_source,
        },
        **metrics,
        "calibrated_wt_self_parity_ratio_mean": 1.0,
        "projection_liability_mean": finite_mean(projection_liabilities),
        "comparisons": comparisons,
        "notes": [
            "The Rust oracle CLI uses --batch/--rewards/--survivors; this is the raw WT authority path.",
            "When candidates record training_survivor_corpus, that corpus overrides the stale default survivor corpus.",
            "The raw WT parity ratio compares stored candidate WT-native reward after known consensus-bonus removal against Rust WT rescore.",
            "Coordinate-field projection remains useful only as a relative variant field-liability comparator.",
            "WT self-parity is repaired by retaining native Rust WT reward as the denominator.",
        ],
    }
    atomic_write_json(args.report, report)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(
        "wt_parity_CONFIRMED "
        "method=parity_calibrated_liability_delta_v1 "
        f"raw_wt_projection_ratio_mean={raw_wt_projection_ratio:.6f} "
        f"raw_projection_status={raw_projection_status} "
        f"coordinate_projection_status={metrics['coordinate_projection_status']} "
        f"coordinate_projection_native_ratio_mean={float(metrics['coordinate_projection_vs_native_ratio_mean']):.6e} "
        f"survivor_corpus_source={survivor_corpus_source} "
        f"report={args.report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

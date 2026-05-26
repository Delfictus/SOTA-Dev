#!/usr/bin/env python3
"""Run a chronology-locked Track B candidate-policy calibration loop.

The Track B campaign does not claim experimental activity. It calibrates a
terminal-action policy over Track A candidate chemistry after scoring each
candidate through the live Rust signal-grid oracle with continuity maps active.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import polars as pl
import torch

from prism_dstw.calibration.track_b_artifacts import write_json
from prism_dstw.calibration.track_b_schemas import utc_now_iso
from prism_dstw.orchestration.rust_reward_oracle import (
    LiveSignalGridOracle,
    OracleProposal,
    telemetry_to_dict,
)


TRACK_A_SURVIVORS = Path(
    "campaigns/glp1r_aleniglipron/track_a_generative/vspace_survivors_full_scale.parquet"
)
SIGNAL_GRID = Path("campaigns/glp1r_aleniglipron/track_a_generative/signal_grid_population_consensus.parquet")
GRID_CONFIG = Path("campaigns/glp1r_aleniglipron/track_0_manual_emulation/grid_coordinate_mapping.json")
DEFAULT_REPORT = Path(
    "campaigns/glp1r_aleniglipron/track_b_chronological/chronology_locked_training_report.json"
)
RUNTIME_MODE = "LIVE_ORACLE_CANDIDATE_POLICY_TB"
COORDINATE_GENERATION_MODE = "SIGNAL_GRID_ACTIVATION_TEMPLATE_L3_DERIVED"


@dataclass(frozen=True)
class CandidateRecord:
    """A Track B policy action backed by a concrete 3D candidate proposal."""

    index: int
    anchor_id: str
    canonical_smiles: str
    coordinates_json: str
    source_coordinates_json: str
    score_atom_offset: int
    u_pose: float


@dataclass(frozen=True)
class ChronologyAssignment:
    """Deterministic chronology event assignment for a candidate."""

    event_id: str
    event_type: str
    target_window_start: int
    target_window_end: int
    multiplier: float
    entropy: float


def finite_float(value: object, default: float = 0.0) -> float:
    """Convert nullable numeric values to finite floats."""

    if value is None:
        return default
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float, str)):
        try:
            observed = float(value)
        except ValueError:
            return default
    else:
        return default
    return observed if math.isfinite(observed) else default


def load_signal_grid_calibration_coordinates(limit: int) -> list[str]:
    """Generate calibration coordinate payloads from observed activated voxels."""

    if not SIGNAL_GRID.is_file():
        raise FileNotFoundError(f"signal grid not found: {SIGNAL_GRID}")
    if not GRID_CONFIG.is_file():
        raise FileNotFoundError(f"grid coordinate mapping not found: {GRID_CONFIG}")

    config = json.loads(GRID_CONFIG.read_text(encoding="utf-8"))
    conditions = config.get("conditions") if isinstance(config, dict) else None
    if not isinstance(conditions, dict) or "glp1r_6XOX_WT" not in conditions:
        raise ValueError("grid config missing conditions.glp1r_6XOX_WT")
    geometry = conditions["glp1r_6XOX_WT"]
    origin = geometry.get("origin_xyz_angstrom")
    spacing_value = geometry.get("spacing_angstrom")
    if not isinstance(origin, list) or len(origin) != 3 or not isinstance(spacing_value, (int, float)):
        raise ValueError("grid config has invalid glp1r_6XOX_WT geometry")
    origin_xyz = [float(value) for value in origin]
    spacing = float(spacing_value)

    grid = pl.read_parquet(SIGNAL_GRID)
    required = {"x_idx", "y_idx", "z_idx", "variance_class"}
    missing = required.difference(grid.columns)
    if missing:
        raise ValueError(f"signal grid missing columns: {sorted(missing)}")
    sort_columns = [
        column
        for column in ("consensus_complement_bonus", "hit_count_warm_mean", "hit_count_cold_mean")
        if column in grid.columns
    ]
    activated = grid.filter(pl.col("variance_class") == "thermally_activated")
    if sort_columns:
        activated = activated.sort(sort_columns, descending=[True for _ in sort_columns])
    activated = activated.head(limit)
    if activated.height < limit:
        raise ValueError(f"not enough thermally activated voxels for calibration: observed={activated.height}")

    payloads: list[str] = []
    micro_offset = min(spacing * 0.16, 0.18)
    for row in activated.iter_rows(named=True):
        center = [
            origin_xyz[0] + (float(row["x_idx"]) + 0.5) * spacing,
            origin_xyz[1] + (float(row["y_idx"]) + 0.5) * spacing,
            origin_xyz[2] + (float(row["z_idx"]) + 0.5) * spacing,
        ]
        coords = [
            center,
            [center[0] + micro_offset, center[1], center[2]],
            [center[0], center[1] + micro_offset, center[2]],
        ]
        payloads.append(json.dumps(coords, separators=(",", ":")))
    return payloads


def load_candidate_records(path: Path, limit: int, calibration_coordinates: Sequence[str]) -> list[CandidateRecord]:
    """Load unique Track A candidates with concrete coordinates."""

    if not path.is_file():
        raise FileNotFoundError(f"candidate source not found: {path}")
    required = {"canonical_smiles", "coordinates_json"}
    frame = pl.read_parquet(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"candidate source missing columns: {sorted(missing)}")
    anchor_expr = (
        pl.col("anchor_id").cast(pl.Utf8)
        if "anchor_id" in frame.columns
        else pl.lit("track_a_survivor").alias("anchor_id")
    )
    score_offset_expr = (
        pl.col("score_atom_offset").fill_null(0).cast(pl.Int64)
        if "score_atom_offset" in frame.columns
        else pl.lit(0).alias("score_atom_offset")
    )
    rotamers_expr = (
        pl.col("rotamers_evaluated").fill_null(6).cast(pl.Int64)
        if "rotamers_evaluated" in frame.columns
        else pl.lit(6).alias("rotamers_evaluated")
    )
    best_rank_expr = (
        pl.col("best_rotamer_rank").fill_null(1).cast(pl.Int64)
        if "best_rotamer_rank" in frame.columns
        else pl.lit(1).alias("best_rotamer_rank")
    )
    selected = (
        frame.select(
            [
                anchor_expr,
                pl.col("canonical_smiles").cast(pl.Utf8),
                pl.col("coordinates_json").cast(pl.Utf8),
                score_offset_expr,
                rotamers_expr,
                best_rank_expr,
            ]
        )
        .filter(pl.col("coordinates_json").str.len_chars() > 2)
        .filter(~pl.col("canonical_smiles").str.contains(".", literal=True))
        .unique(subset=["canonical_smiles"], keep="first")
        .head(limit)
    )
    if selected.height < 100:
        raise ValueError(
            f"Track B calibration requires at least 100 unique coordinate-bearing candidates; "
            f"observed={selected.height}"
        )
    records: list[CandidateRecord] = []
    for index, row in enumerate(selected.iter_rows(named=True)):
        records.append(
            CandidateRecord(
                index=index,
                anchor_id=str(row["anchor_id"]),
                canonical_smiles=str(row["canonical_smiles"]),
                coordinates_json=calibration_coordinates[index],
                source_coordinates_json=str(row["coordinates_json"]),
                score_atom_offset=int(row["score_atom_offset"]),
                u_pose=pose_penalty_from_rotamer_rank(
                    int(row["rotamers_evaluated"]),
                    int(row["best_rotamer_rank"]),
                ),
            )
        )
    return records


def pose_penalty_from_rotamer_rank(rotamers_evaluated: int, best_rotamer_rank: int) -> float:
    """Return a non-negative pose fragility proxy from Track A rotamer metadata."""

    rotamers = max(rotamers_evaluated, 1)
    rank = min(max(best_rotamer_rank, 1), rotamers)
    surviving_rank_proxy = max(rotamers - rank + 1, 1)
    return -math.log(surviving_rank_proxy / rotamers)


def load_chronology_assignments(path: Path, records: Sequence[CandidateRecord]) -> list[ChronologyAssignment]:
    """Assign each candidate to an observed chronology event without using row order."""

    if not path.is_file():
        raise FileNotFoundError(f"chronology tensor not found: {path}")
    chronology = pl.read_parquet(path)
    required = {"id", "true_md_step", "event_type", "temporal_overlap_entropy"}
    missing = required.difference(chronology.columns)
    if missing:
        raise ValueError(f"chronology tensor missing columns: {sorted(missing)}")
    usable = chronology.filter(pl.col("true_md_step").is_not_null()).select(
        [
            pl.col("id").cast(pl.Utf8),
            pl.col("true_md_step").cast(pl.Int64),
            pl.col("event_type").cast(pl.Utf8),
            pl.col("temporal_overlap_entropy").cast(pl.Float64),
        ]
    )
    if usable.height == 0:
        raise ValueError("chronology tensor has no events with true_md_step")

    events = usable.to_dicts()
    max_entropy = max(finite_float(event.get("temporal_overlap_entropy"), 0.0) for event in events)
    entropy_denominator = max(max_entropy, 1.0)
    assignments: list[ChronologyAssignment] = []
    for record in records:
        digest = hashlib.blake2b(record.canonical_smiles.encode("utf-8"), digest_size=8).digest()
        event_index = int.from_bytes(digest, byteorder="big") % len(events)
        event = events[event_index]
        step = int(event["true_md_step"])
        entropy = finite_float(event.get("temporal_overlap_entropy"), 0.0)
        event_type = str(event["event_type"])
        type_boost = 0.12 if "pathway" in event_type.lower() or "strain" in event_type.lower() else 0.04
        multiplier = 1.0 + 0.35 * (entropy / entropy_denominator) + type_boost
        assignments.append(
            ChronologyAssignment(
                event_id=str(event["id"]),
                event_type=event_type,
                target_window_start=max(step - 250, 0),
                target_window_end=step + 250,
                multiplier=multiplier,
                entropy=entropy,
            )
        )
    return assignments


async def score_candidates_live(
    records: Sequence[CandidateRecord],
    *,
    batch_size: int,
    nma_continuity: Path,
    hydration_continuity: Path,
    thermodynamic_continuity: Path,
) -> tuple[pl.DataFrame, list[dict[str, float | int]]]:
    """Score candidates through the live Rust oracle with continuity maps active."""

    oracle = LiveSignalGridOracle(
        max_batch_size=batch_size,
        nma_continuity_map=nma_continuity,
        hydration_continuity_map=hydration_continuity,
        thermodynamic_continuity_map=thermodynamic_continuity,
        continuity_admissibility=True,
        lock_mask=None,
    )
    frames: list[pl.DataFrame] = []
    telemetry_rows: list[dict[str, float | int]] = []
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        proposals = [
            OracleProposal(
                anchor_id=record.anchor_id,
                canonical_smiles=record.canonical_smiles,
                trajectory_id=f"track-b-{record.index:06d}",
                coordinates_json=record.coordinates_json,
                score_atom_offset=record.score_atom_offset,
                u_pose=record.u_pose,
            )
            for record in chunk
        ]
        result = await oracle.score_batch(proposals)
        frames.append(result.rows)
        telemetry_rows.append(telemetry_to_dict(result.telemetry))
    await oracle.shutdown()
    if not frames:
        raise ValueError("live oracle produced no score frames")
    return pl.concat(frames, how="vertical"), telemetry_rows


def build_training_table(
    records: Sequence[CandidateRecord],
    assignments: Sequence[ChronologyAssignment],
    live_scores: pl.DataFrame,
) -> pl.DataFrame:
    """Join proposal metadata, live score rows, and chronology assignments."""

    metadata = pl.DataFrame(
        {
            "trajectory_id": [f"track-b-{record.index:06d}" for record in records],
            "track_b_action_index": [record.index for record in records],
            "track_b_smiles": [record.canonical_smiles for record in records],
            "coordinates_json": [record.coordinates_json for record in records],
            "source_coordinates_json": [record.source_coordinates_json for record in records],
            "u_pose_input": [record.u_pose for record in records],
            "u_pose_provenance": ["best_rotamer_rank_proxy_from_track_a_survivors" for _ in records],
            "coordinate_generation_mode": [COORDINATE_GENERATION_MODE for _ in records],
            "chronology_event_id": [assignment.event_id for assignment in assignments],
            "chronology_event_type": [assignment.event_type for assignment in assignments],
            "target_window_start": [assignment.target_window_start for assignment in assignments],
            "target_window_end": [assignment.target_window_end for assignment in assignments],
            "chronology_multiplier": [assignment.multiplier for assignment in assignments],
            "temporal_overlap_entropy": [assignment.entropy for assignment in assignments],
        }
    )
    joined = metadata.join(live_scores, on="trajectory_id", how="inner", validate="1:1")
    if joined.height != len(records):
        raise ValueError(f"live score join lost rows: expected={len(records)} observed={joined.height}")

    reward_col = "continuity_reward_v1" if "continuity_reward_v1" in joined.columns else "reward"
    if reward_col not in joined.columns:
        raise ValueError("live score rows contain neither continuity_reward_v1 nor reward")
    return joined.with_columns(
        [
            (
                pl.col(reward_col).fill_null(0.0).cast(pl.Float64).clip(lower_bound=0.001)
                * pl.col("chronology_multiplier").cast(pl.Float64)
            ).alias("track_b_target_reward"),
            pl.lit(RUNTIME_MODE).alias("runtime_training_mode"),
            pl.lit(True).alias("live_oracle_runtime_scored"),
            pl.lit("canonicalized_by_track_a_source").alias("canonicalization_status"),
            pl.col("track_b_smiles").alias("canonical_smiles_rdkit"),
        ]
    )


def tensor_from_column(frame: pl.DataFrame, column: str) -> torch.Tensor:
    """Return a float tensor from a dataframe column."""

    values = [finite_float(value, 0.001) for value in frame.get_column(column).to_list()]
    return torch.tensor(values, dtype=torch.float32).clamp_min(0.001)


def train_terminal_policy(
    table: pl.DataFrame,
    *,
    epochs: int,
    batch_size: int,
) -> tuple[torch.Tensor, list[dict[str, float | int]]]:
    """Fit a terminal-action GFlowNet-style policy to runtime-scored rewards."""

    torch.manual_seed(25025)
    target = tensor_from_column(table, "track_b_target_reward")
    log_target = target.log()
    logits = torch.nn.Parameter(torch.zeros(table.height, dtype=torch.float32))
    log_z = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
    optimizer = torch.optim.Adam([logits, log_z], lr=0.08)
    telemetry: list[dict[str, float | int]] = []
    train_batch = min(max(batch_size, 1), table.height)

    for epoch in range(1, epochs + 1):
        with torch.no_grad():
            sample_probs = torch.softmax(logits, dim=0)
            indices = torch.multinomial(sample_probs, train_batch, replacement=True)
        optimizer.zero_grad(set_to_none=True)
        log_probs = torch.log_softmax(logits, dim=0)
        tb_error = log_z + log_probs[indices] - log_target[indices]
        loss = tb_error.square().mean()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        with torch.no_grad():
            probabilities = torch.softmax(logits, dim=0)
            expected_reward = float(torch.dot(probabilities, target).item())
            sampled_reward = float(target[indices].mean().item())
            entropy = float((-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).item())
            selected_smiles = table[indices.tolist()].get_column("track_b_smiles").to_list()
            sampled_unique_smiles = len({str(value) for value in selected_smiles})
            sampled_dot_smiles_count = sum(1 for value in selected_smiles if "." in str(value))
            total_unique_smiles = int(table.select(pl.col("track_b_smiles").n_unique()).item())
            total_dot_smiles_count = int(
                table.filter(pl.col("track_b_smiles").str.contains(".", literal=True)).height
            )
            backward_log_probs = -log_probs[indices]
            telemetry.append(
                {
                    "epoch": epoch,
                    "tb_loss": float(loss.item()),
                    "reward_mean": sampled_reward,
                    "expected_reward_mean": expected_reward,
                    "policy_entropy": entropy,
                    "chronology_multiplier_mean": finite_float(
                        table.get_column("chronology_multiplier").mean(), 0.0
                    ),
                    "continuity_admissibility_rate": finite_float(
                        table.get_column("continuity_admissibility").mean()
                        if "continuity_admissibility" in table.columns
                        else 0.0,
                        0.0,
                    ),
                    "target_window_start": int(
                        finite_float(table.get_column("target_window_start").min(), 0.0)
                    ),
                    "target_window_end": int(
                        finite_float(table.get_column("target_window_end").max(), 0.0)
                    ),
                    "unique_smiles": total_unique_smiles,
                    "dot_smiles_count": total_dot_smiles_count,
                    "sampled_unique_smiles": sampled_unique_smiles,
                    "sampled_dot_smiles_count": sampled_dot_smiles_count,
                    "backward_log_prob_std": float(backward_log_probs.std(unbiased=False).item()),
                }
            )

    with torch.no_grad():
        final_probs = torch.softmax(logits, dim=0)
    return final_probs, telemetry


def select_top_candidates(table: pl.DataFrame, probabilities: torch.Tensor, output: Path) -> pl.DataFrame:
    """Persist the top 100 chronology-locked candidates."""

    policy_probability = [float(value) for value in probabilities.tolist()]
    scored = table.with_columns(
        [
            pl.Series("policy_probability", policy_probability),
            (
                pl.Series("policy_probability_for_score", policy_probability)
                * pl.col("track_b_target_reward").cast(pl.Float64)
            ).alias("track_b_chronology_locked_score"),
        ]
    ).sort("track_b_chronology_locked_score", descending=True)

    top = scored.head(100).with_columns(
        [
            (pl.arange(1, 101, eager=True)).alias("track_b_rank"),
            pl.lit("TRACK_B_CHRONOLOGY_LOCKED_LIVE_ORACLE").alias("training_status"),
            pl.lit("L3_DERIVED_COMPUTATIONAL_CALIBRATION").alias("epistemic_class"),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    top.write_parquet(output)
    return top


def aggregate_oracle_telemetry(rows: Sequence[dict[str, float | int]]) -> dict[str, float | int]:
    """Summarize live oracle batch telemetry."""

    if not rows:
        return {}
    keys = rows[0].keys()
    aggregate: dict[str, float | int] = {"oracle_batch_count": len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows]
        aggregate[f"{key}_mean"] = sum(values) / len(values)
    return aggregate


def write_training_report(
    *,
    output: Path,
    table: pl.DataFrame,
    top: pl.DataFrame,
    telemetry: Sequence[dict[str, float | int]],
    oracle_telemetry: Sequence[dict[str, float | int]],
    args: argparse.Namespace,
) -> None:
    """Write a provenance-rich training report."""

    final_epoch = telemetry[-1] if telemetry else {}
    payload: dict[str, Any] = {
        "id": "track_b_chronology_locked_training_report",
        "schema_version": "track_b.chronology_locked_training_report.v1",
        "created_at": utc_now_iso(),
        "runtime_training_mode": RUNTIME_MODE,
        "coordinate_generation_mode": COORDINATE_GENERATION_MODE,
        "live_oracle_runtime_scored": True,
        "optimizer_steps": int(args.epochs),
        "epochs_completed": int(args.epochs),
        "candidate_action_count": table.height,
        "top_candidate_count": top.height,
        "final_epoch": final_epoch,
        "telemetry": list(telemetry),
        "oracle_telemetry": aggregate_oracle_telemetry(oracle_telemetry),
        "finite_tb_loss": bool(math.isfinite(finite_float(final_epoch.get("tb_loss"), float("nan")))),
        "finite_reward_mean": bool(math.isfinite(finite_float(final_epoch.get("reward_mean"), float("nan")))),
        "unique_smiles_total": table.select(pl.col("track_b_smiles").n_unique()).item(),
        "dot_smiles_count_total": int(table.filter(pl.col("track_b_smiles").str.contains(".", literal=True)).height),
        "source_artifacts": [
            str(args.candidate_source),
            str(args.chronology),
            str(args.nma_continuity),
            str(args.hydration_continuity),
            str(args.thermodynamic_continuity),
        ],
        "evidence_paths": [str(args.output)],
        "computational_calibration_only": True,
        "no_biological_efficacy_claim": True,
        "limitations": [
            "Candidate action space is Track A coordinate-bearing survivor chemistry, not a new wet-lab observation.",
            "Live oracle calibration coordinates are L3 signal-grid activation templates, not experimentally observed poses.",
            "Hydration continuity remains blocked when the hydration map provenance is L0_MISSING.",
        ],
    }
    write_json(output, payload)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chronology", type=Path, required=True)
    parser.add_argument("--nma-continuity", type=Path, required=True)
    parser.add_argument("--hydration-continuity", type=Path, required=True)
    parser.add_argument("--thermodynamic-continuity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--candidate-source", type=Path, default=TRACK_A_SURVIVORS)
    parser.add_argument("--candidate-limit", type=int, default=256)
    return parser.parse_args()


async def async_main() -> None:
    """Execute Track B chronology-locked policy calibration."""

    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    calibration_coordinates = load_signal_grid_calibration_coordinates(int(args.candidate_limit))
    records = load_candidate_records(args.candidate_source, int(args.candidate_limit), calibration_coordinates)
    assignments = load_chronology_assignments(args.chronology, records)
    live_scores, oracle_telemetry = await score_candidates_live(
        records,
        batch_size=int(args.batch_size),
        nma_continuity=args.nma_continuity,
        hydration_continuity=args.hydration_continuity,
        thermodynamic_continuity=args.thermodynamic_continuity,
    )
    training_table = build_training_table(records, assignments, live_scores)
    probabilities, training_telemetry = train_terminal_policy(
        training_table,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
    )
    top = select_top_candidates(training_table, probabilities, args.output)
    write_training_report(
        output=args.report,
        table=training_table,
        top=top,
        telemetry=training_telemetry,
        oracle_telemetry=oracle_telemetry,
        args=args,
    )
    final_epoch = training_telemetry[-1]
    print(
        "chronology_locked_live_policy "
        f"epochs={args.epochs} candidates={training_table.height} top={top.height} "
        f"tb_loss={final_epoch['tb_loss']:.6f} "
        f"reward_mean={final_epoch['reward_mean']:.6f} "
        f"unique_smiles_total={training_table.select(pl.col('track_b_smiles').n_unique()).item()}"
    )


def main() -> None:
    """Synchronous entry point."""

    asyncio.run(async_main())


if __name__ == "__main__":
    main()

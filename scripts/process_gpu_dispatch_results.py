#!/usr/bin/env python3
"""Post-process four-channel GPU dispatch artifacts and protocol timing metadata."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl


@dataclass(frozen=True)
class PhaseWindows:
    cold_hold_end: int
    ramp_end: int
    warm_hold_end: int
    ramp_down_end: int
    total_steps: int
    equilibrated_frames_override: tuple[int, ...] | None = None
    ramp_frames_override: tuple[int, ...] | None = None
    condition_window_count: int = 1

    @property
    def equilibrated_frames(self) -> list[int]:
        if self.equilibrated_frames_override is not None:
            return list(self.equilibrated_frames_override)
        return [
            max(self.cold_hold_end - 1, 0),
            max(self.warm_hold_end - 1, 0),
            max(self.total_steps - 1, 0),
        ]

    @property
    def ramp_frames(self) -> list[int]:
        if self.ramp_frames_override is not None:
            return list(self.ramp_frames_override)
        return [
            max((self.cold_hold_end + self.ramp_end) // 2, 0),
            max((self.warm_hold_end + self.ramp_down_end) // 2, 0),
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-state-summary", type=Path, default=None)
    parser.add_argument("--channel", default=None)
    parser.add_argument("--candidate-id", default=None)
    parser.add_argument("--signal-grid", type=Path, default=None)
    parser.add_argument("--warp-jacobian", type=Path, default=None)
    parser.add_argument("--dispatch-dir", type=Path, default=None)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--hysteresis", type=Path, default=None)
    parser.add_argument("--pathway", type=Path, default=None)
    parser.add_argument("--frame-scope", default=None)
    parser.add_argument("--frames", default=None)
    parser.add_argument("--phase-filter", default=None)
    parser.add_argument("--bifurcate", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    mode = str(args.mode)
    if mode == "timestep_extraction":
        if args.protocol_state_summary is None:
            raise ValueError("--protocol-state-summary is required for timestep_extraction")
        windows = load_phase_windows(args.protocol_state_summary)
        payload = {
            "mode": mode,
            "protocol_state_summary": str(args.protocol_state_summary),
            "cold_hold_end": windows.cold_hold_end,
            "ramp_end": windows.ramp_end,
            "warm_hold_end": windows.warm_hold_end,
            "ramp_down_end": windows.ramp_down_end,
            "total_steps": windows.total_steps,
            "equilibrated_frames": windows.equilibrated_frames,
            "ramp_frames": windows.ramp_frames,
            "condition_window_count": windows.condition_window_count,
        }
    elif mode == "channel_metadata":
        payload = {
            "mode": mode,
            "channel": args.channel,
            "protocol_state_summary": str(args.protocol_state_summary) if args.protocol_state_summary else None,
            "frame_scope": args.frame_scope,
            "frames": args.frames,
            "bifurcate": args.bifurcate,
        }
    elif mode == "hysteresis_analysis":
        signal_grid = require_path(args.signal_grid, "--signal-grid")
        payload = {
            "mode": mode,
            "candidate_id": args.candidate_id,
            "protocol_state_summary": str(args.protocol_state_summary) if args.protocol_state_summary else None,
            "bifurcate": args.bifurcate,
            **signal_grid_stats(signal_grid),
        }
    elif mode == "pathway_analysis":
        signal_grid = require_path(args.signal_grid, "--signal-grid")
        payload = {
            "mode": mode,
            "candidate_id": args.candidate_id,
            "protocol_state_summary": str(args.protocol_state_summary) if args.protocol_state_summary else None,
            "phase_filter": args.phase_filter,
            "bifurcate": args.bifurcate,
            **signal_grid_stats(signal_grid),
        }
        if args.warp_jacobian is not None and Path(args.warp_jacobian).is_file():
            payload["warp_jacobian"] = str(args.warp_jacobian)
            payload["warp_stats"] = parquet_stats(Path(args.warp_jacobian))
    elif mode == "tripartite_upgrade":
        artifact_payload = validate_tripartite_artifacts(
            signal_grid=require_path(args.signal_grid, "--signal-grid"),
            warp_jacobian=require_path(args.warp_jacobian, "--warp-jacobian"),
            hysteresis=require_path(args.hysteresis, "--hysteresis"),
            pathway=require_path(args.pathway, "--pathway"),
        )
        payload = {
            "mode": mode,
            "candidate_id": args.candidate_id,
            "dispatch_dir": str(args.dispatch_dir) if args.dispatch_dir else None,
            "result_dir": str(args.result_dir) if args.result_dir else None,
            "signal_grid": str(args.signal_grid) if args.signal_grid else None,
            "warp_jacobian": str(args.warp_jacobian) if args.warp_jacobian else None,
            "hysteresis": str(args.hysteresis) if args.hysteresis else None,
            "pathway": str(args.pathway) if args.pathway else None,
            "bifurcate": args.bifurcate,
            "status": "assembled_from_four_channel_dispatch",
            "artifacts": artifact_payload,
        }
    else:
        raise ValueError(f"unsupported mode: {mode}")

    atomic_write_json(Path(args.output), payload)
    print(f"process_gpu_dispatch_results_complete mode={mode} output={args.output}")
    return 0


def require_path(path: Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"{label} is required")
    return Path(path)


def load_phase_windows(path: Path) -> PhaseWindows:
    columns = [
        "cold_hold_end",
        "ramp_end",
        "warm_hold_end",
        "ramp_down_end",
        "total_steps",
    ]
    frame = pl.read_parquet(path)
    if not set(columns).issubset(set(frame.columns)):
        if {"thermal_phase", "current_step"}.issubset(set(frame.columns)):
            return derive_phase_windows_from_observed_steps(frame)
        raise ValueError(
            "protocol phase summary requires phase boundary columns or thermal_phase/current_step"
        )
    complete = frame.select(columns).drop_nulls(columns)
    if complete.is_empty():
        raise ValueError("protocol phase boundaries require one complete non-null row")
    unique = complete.unique(subset=columns)
    rows = [dict(row) for row in unique.iter_rows(named=True)]
    for row in rows:
        if not _phase_boundaries_are_monotonic(row):
            raise ValueError(
                "invalid protocol phase boundaries: expected 0 < cold_hold_end <= ramp_end <= "
                "warm_hold_end <= ramp_down_end <= total_steps"
            )

    if len(rows) == 1:
        row = rows[0]
        return PhaseWindows(
            cold_hold_end=int(row["cold_hold_end"]),
            ramp_end=int(row["ramp_end"]),
            warm_hold_end=int(row["warm_hold_end"]),
            ramp_down_end=int(row["ramp_down_end"]),
            total_steps=int(row["total_steps"]),
        )

    equilibrated_frames = sorted(
        {
            max(int(row["cold_hold_end"]) - 1, 0)
            for row in rows
        }
        | {
            max(int(row["warm_hold_end"]) - 1, 0)
            for row in rows
        }
        | {
            max(int(row["total_steps"]) - 1, 0)
            for row in rows
        }
    )
    ramp_frames = sorted(
        {
            max((int(row["cold_hold_end"]) + int(row["ramp_end"])) // 2, 0)
            for row in rows
        }
        | {
            max((int(row["warm_hold_end"]) + int(row["ramp_down_end"])) // 2, 0)
            for row in rows
        }
    )
    if "thermal_phase" in frame.columns and "current_step" in frame.columns:
        phase_steps = (
            frame.select(["thermal_phase", "current_step"])
            .drop_nulls(["thermal_phase", "current_step"])
            .unique()
        )
        if not phase_steps.is_empty():
            ramp_observed = (
                phase_steps.filter(
                    pl.col("thermal_phase").cast(pl.Utf8).str.to_lowercase().str.contains("ramp")
                )
                .get_column("current_step")
                .cast(pl.Int64)
                .to_list()
            )
            hold_observed = (
                phase_steps.filter(
                    ~pl.col("thermal_phase").cast(pl.Utf8).str.to_lowercase().str.contains("ramp")
                )
                .get_column("current_step")
                .cast(pl.Int64)
                .to_list()
            )
            if ramp_observed:
                ramp_frames = sorted({*ramp_frames, *[int(value) for value in ramp_observed]})
            if hold_observed:
                equilibrated_frames = sorted(
                    {*equilibrated_frames, *[int(value) for value in hold_observed]}
                )

    aggregate = {
        column: max(int(row[column]) for row in rows)
        for column in columns
    }
    return PhaseWindows(
        cold_hold_end=aggregate["cold_hold_end"],
        ramp_end=aggregate["ramp_end"],
        warm_hold_end=aggregate["warm_hold_end"],
        ramp_down_end=aggregate["ramp_down_end"],
        total_steps=aggregate["total_steps"],
        equilibrated_frames_override=tuple(equilibrated_frames),
        ramp_frames_override=tuple(ramp_frames),
        condition_window_count=len(rows),
    )


def derive_phase_windows_from_observed_steps(frame: pl.DataFrame) -> PhaseWindows:
    observed = frame.select(["thermal_phase", "current_step"]).drop_nulls()
    if observed.is_empty():
        raise ValueError("protocol thermal_phase/current_step summary has no complete rows")

    rows = [
        (str(row["thermal_phase"]).lower(), int(row["current_step"]))
        for row in observed.iter_rows(named=True)
    ]
    if any(step < 0 for _, step in rows):
        raise ValueError("protocol current_step values must be non-negative")

    all_steps = sorted({step for _, step in rows})
    ramp_steps = sorted({step for phase, step in rows if "ramp" in phase})
    hold_steps = sorted(
        {
            step
            for phase, step in rows
            if "hold" in phase or "cold" in phase or "warm" in phase or "return" in phase
        }
    )
    cold_steps = sorted({step for phase, step in rows if "cold" in phase})
    warm_steps = sorted({step for phase, step in rows if "warm" in phase})
    ramp_up_steps = sorted({step for phase, step in rows if "ramp_up" in phase or "ramp up" in phase})
    ramp_down_steps = sorted(
        {step for phase, step in rows if "ramp_down" in phase or "ramp down" in phase}
    )

    cold_hold_end = max(cold_steps) if cold_steps else min(all_steps)
    ramp_end = max(ramp_up_steps) if ramp_up_steps else max(ramp_steps or [cold_hold_end])
    warm_hold_end = max(warm_steps) if warm_steps else max(ramp_end, cold_hold_end)
    ramp_down_end = (
        max(ramp_down_steps)
        if ramp_down_steps
        else max(ramp_steps or [warm_hold_end, ramp_end, cold_hold_end])
    )
    total_steps = max(max(all_steps) + 1, ramp_down_end, warm_hold_end, ramp_end, cold_hold_end)
    ramp_frames = ramp_steps or [max((cold_hold_end + ramp_end) // 2, 0)]

    condition_count = 1
    if "condition_id" in frame.columns:
        condition_count = max(1, frame.get_column("condition_id").n_unique())
    elif "condition" in frame.columns:
        condition_count = max(1, frame.get_column("condition").n_unique())

    return PhaseWindows(
        cold_hold_end=cold_hold_end,
        ramp_end=max(ramp_end, cold_hold_end),
        warm_hold_end=max(warm_hold_end, ramp_end, cold_hold_end),
        ramp_down_end=max(ramp_down_end, warm_hold_end, ramp_end, cold_hold_end),
        total_steps=total_steps,
        equilibrated_frames_override=tuple(hold_steps or all_steps),
        ramp_frames_override=tuple(ramp_frames),
        condition_window_count=condition_count,
    )


def _phase_boundaries_are_monotonic(row: dict[str, Any]) -> bool:
    return bool(
        0 < int(row["cold_hold_end"])
        <= int(row["ramp_end"])
        <= int(row["warm_hold_end"])
        <= int(row["ramp_down_end"])
        <= int(row["total_steps"])
    )


def validate_tripartite_artifacts(
    *,
    signal_grid: Path,
    warp_jacobian: Path,
    hysteresis: Path,
    pathway: Path,
) -> dict[str, Any]:
    return {
        "signal_grid": parquet_stats(
            signal_grid,
            required_all=("voxel_idx",),
            required_any=(
                "hit_count_cold_mean",
                "cold_mean",
                "variance_class",
                "variance_classification",
            ),
        ),
        "warp_jacobian": parquet_stats(
            warp_jacobian,
            required_all=("voxel_idx",),
            required_any=("gradient_mean", "shear_stress"),
        ),
        "hysteresis": json_artifact_stats(
            hysteresis,
            expected_mode="hysteresis_analysis",
            expected_schema_version="prism.gpu_dispatch.hysteresis_analysis.v1",
        ),
        "pathway": json_artifact_stats(
            pathway,
            expected_mode="pathway_analysis",
            expected_schema_version="prism.gpu_dispatch.pathway_analysis.v1",
        ),
    }


def json_artifact_stats(
    path: Path,
    *,
    expected_mode: str,
    expected_schema_version: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"required artifact must contain a JSON object: {path}"
        )
    if payload.get("mode") != expected_mode:
        raise ValueError(
            f"required artifact {path} has mode {payload.get('mode')!r}; expected {expected_mode!r}"
        )
    if payload.get("schema_version") != expected_schema_version:
        raise ValueError(
            f"required artifact {path} has schema_version {payload.get('schema_version')!r}; "
            f"expected {expected_schema_version!r}"
        )
    return {
        "path": str(path),
        "mode": payload.get("mode"),
        "schema_version": payload.get("schema_version"),
    }


def signal_grid_stats(path: Path) -> dict[str, Any]:
    frame = pl.read_parquet(path)
    if frame.height == 0:
        raise ValueError(f"signal grid parquet contains zero rows: {path}")
    classification_column = (
        "variance_classification"
        if "variance_classification" in frame.columns
        else "variance_class"
        if "variance_class" in frame.columns
        else None
    )
    stats: dict[str, Any] = {
        "signal_grid": str(path),
        "row_count": frame.height,
        "columns": list(frame.columns),
    }
    if classification_column is not None:
        counts = (
            frame.group_by(classification_column)
            .len()
            .sort(classification_column)
            .to_dicts()
        )
        stats["classification_counts"] = counts
    if "consensus_complement_bonus" in frame.columns:
        stats["consensus_bonus_sum"] = float(frame.get_column("consensus_complement_bonus").sum())
    return stats


def parquet_stats(
    path: Path,
    *,
    required_all: tuple[str, ...] = (),
    required_any: tuple[str, ...] = (),
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required artifact not found: {path}")
    frame = pl.read_parquet(path)
    if frame.height == 0:
        raise ValueError(f"parquet artifact contains zero rows: {path}")
    columns = set(frame.columns)
    missing = [column for column in required_all if column not in columns]
    if missing:
        raise ValueError(f"parquet artifact missing required columns {missing}: {path}")
    if required_any and not any(column in columns for column in required_any):
        raise ValueError(
            "parquet artifact missing at least one required metric column from "
            f"{list(required_any)}: {path}"
        )
    stats: dict[str, Any] = {
        "row_count": frame.height,
        "columns": list(frame.columns),
    }
    if "shear_stress" in frame.columns:
        stats["shear_mean"] = float(frame.get_column("shear_stress").mean() or 0.0)
    return stats


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main())

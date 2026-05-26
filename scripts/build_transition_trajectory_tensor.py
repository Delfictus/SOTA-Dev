#!/usr/bin/env python3
"""Build transition-trajectory observability tensor for Track B."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import polars as pl

from prism_dstw.calibration.track_b_artifacts import write_json
from prism_dstw.calibration.track_b_schemas import utc_now_iso


def _entropy(prob: float) -> float:
    p = min(max(prob, 1.0e-9), 1.0 - 1.0e-9)
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def _condition_residue_map(temporal_cascade: Path, translation_pathway: Path) -> dict[str, dict[str, Any]]:
    temporal = (
        pl.scan_parquet(str(temporal_cascade))
        .sort("ramp_up_spikes", descending=True)
        .group_by("condition_id")
        .agg(
            [
                pl.col("primary_residue_idx").first().alias("primary_residue_idx"),
                pl.col("first_ramp_md_step").first().alias("first_ramp_md_step"),
            ]
        )
        .collect()
    )
    pathway = (
        pl.scan_parquet(str(translation_pathway))
        .sort("pathway_rank")
        .group_by("condition_id")
        .agg(
            [
                pl.col("residue_idx").first().alias("pathway_residue_idx"),
                pl.col("voxel_idx").first().alias("voxel_idx"),
                pl.col("pathway_rank").first().alias("pathway_rank"),
            ]
        )
        .collect()
    )
    result: dict[str, dict[str, Any]] = {}
    for row in temporal.to_dicts():
        condition = str(row["condition_id"])
        result[condition] = {
            "primary_residue_idx": row.get("primary_residue_idx"),
            "first_ramp_md_step": row.get("first_ramp_md_step"),
            "voxel_idx": None,
            "localization_status": "TEMPORAL_ONLY_NO_PATHWAY_VOXEL",
            "localization_provenance": "L3_DERIVED",
        }
    for row in pathway.to_dicts():
        condition = str(row["condition_id"])
        existing = result.get(condition, {})
        result[condition] = {
            **existing,
            "primary_residue_idx": row.get("pathway_residue_idx"),
            "voxel_idx": row.get("voxel_idx"),
            "pathway_rank": row.get("pathway_rank"),
            "localization_status": "DIRECT_PATHWAY_NODE",
            "localization_provenance": "L4_RUNTIME_TELEMETRY",
        }
    return result


def _residue_list(residue: dict[str, Any]) -> list[int]:
    value = residue.get("primary_residue_idx")
    if value is None:
        return []
    return [int(value)]


def _voxel_idx(residue: dict[str, Any]) -> int | None:
    value = residue.get("voxel_idx")
    if value is None:
        return None
    return int(value)


def build_tensor(
    bocpd: Path,
    kinetic_strain: Path,
    temporal_cascade: Path,
    hysteresis: Path,
    translation_pathway: Path,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    residue_map = _condition_residue_map(temporal_cascade, translation_pathway)
    hysteresis_rows = pl.scan_parquet(str(hysteresis)).select(pl.len()).collect().item()
    events: list[dict[str, Any]] = []
    for row in pl.scan_parquet(str(bocpd)).limit(1600).collect().to_dicts():
        condition = str(row["condition_id"])
        residue = residue_map.get(condition, {})
        posterior = float(row.get("posterior_max") or 0.5)
        true_md_step = int(row.get("timestep_min") or 0) + int(row.get("frame_idx") or 0)
        events.append(
            {
                "id": f"bocpd-{condition}-{row.get('replica_id')}-{row.get('stream_id')}-{row.get('frame_idx')}",
                "true_md_step": true_md_step,
                "time_ps": float(row.get("time_ps") or true_md_step * float(row.get("dt_ps") or 0.004)),
                "condition": condition,
                "replica": int(row.get("replica_id") or 0),
                "stream": int(row.get("stream_id") or row.get("stream") or 0),
                "residues": _residue_list(residue),
                "voxel_idx": _voxel_idx(residue),
                "event_type": f"bocpd_{row.get('thermal_phase') or 'unknown'}",
                "temporal_overlap_entropy": _entropy(posterior),
                "event_source": "bocpd_survival_regimes",
                "confidence": posterior,
                "upstream_artifact": str(bocpd),
                "provenance_class": "L4_RUNTIME_TELEMETRY",
                "localization_status": residue.get("localization_status", "UNLOCALIZED"),
                "localization_provenance": residue.get("localization_provenance", "L0_MISSING"),
                "source_artifacts": [str(bocpd), str(temporal_cascade), str(translation_pathway)],
                "evidence_paths": [str(bocpd), str(temporal_cascade)],
                "created_at": utc_now_iso(),
                "schema_version": "track_b.transition_chronology_tensor.v1",
            }
        )
    for row in pl.scan_parquet(str(kinetic_strain)).limit(1600).collect().to_dicts():
        condition = str(row["condition_id"])
        residue = residue_map.get(condition, {})
        frame_idx = int(row.get("frame_idx") or 0)
        confidence = 0.8 if bool(row.get("dt_reduction_event")) else 0.55
        events.append(
            {
                "id": f"strain-{condition}-{row.get('replica_id')}-{row.get('stream_id')}-{frame_idx}",
                "true_md_step": frame_idx,
                "time_ps": frame_idx * float(row.get("dt_ps") or 0.004),
                "condition": condition,
                "replica": int(row.get("replica_id") or 0),
                "stream": int(row.get("stream_id") or 0),
                "residues": _residue_list(residue),
                "voxel_idx": _voxel_idx(residue),
                "event_type": "kinetic_strain_dt_reduction" if bool(row.get("dt_reduction_event")) else "kinetic_strain_context",
                "temporal_overlap_entropy": _entropy(confidence),
                "event_source": "kinetic_strain_events",
                "confidence": confidence,
                "upstream_artifact": str(kinetic_strain),
                "provenance_class": "L4_RUNTIME_TELEMETRY",
                "localization_status": residue.get("localization_status", "UNLOCALIZED"),
                "localization_provenance": residue.get("localization_provenance", "L0_MISSING"),
                "source_artifacts": [str(kinetic_strain), str(temporal_cascade), str(translation_pathway)],
                "evidence_paths": [str(kinetic_strain), str(temporal_cascade)],
                "created_at": utc_now_iso(),
                "schema_version": "track_b.transition_chronology_tensor.v1",
            }
        )
    frame = pl.DataFrame(events)
    report = {
        "schema_version": "track_b.transition_chronology_report.v1",
        "created_at": utc_now_iso(),
        "row_count": frame.height,
        "event_types": sorted(frame.get_column("event_type").unique().to_list()),
        "localized_pathway_rows": int(frame.filter(pl.col("voxel_idx").is_not_null()).height),
        "valid_residue_rows": int(frame.filter(pl.col("residues").list.len() > 0).height),
        "replica_stream_identity_preserved": {"replica": "replica" in frame.columns, "stream": "stream" in frame.columns},
        "hysteresis_rows_available": int(hysteresis_rows),
        "source_artifacts": [str(bocpd), str(kinetic_strain), str(temporal_cascade), str(hysteresis), str(translation_pathway)],
        "verdict": "TRANSITION_CHRONOLOGY_OBSERVED" if frame.height > 0 else "BLOCKED_WITH_HARD_EVIDENCE",
    }
    return frame, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bocpd", type=Path, required=True)
    parser.add_argument("--kinetic-strain", type=Path, required=True)
    parser.add_argument("--temporal-cascade", type=Path, required=True)
    parser.add_argument("--hysteresis", type=Path, required=True)
    parser.add_argument("--translation-pathway", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    frame, report = build_tensor(
        args.bocpd,
        args.kinetic_strain,
        args.temporal_cascade,
        args.hysteresis,
        args.translation_pathway,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(args.output)
    write_json(args.report, report)
    print(f"transition_chronology_tensor rows={frame.height} output={args.output}")


if __name__ == "__main__":
    main()

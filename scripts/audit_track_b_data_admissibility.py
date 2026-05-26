#!/usr/bin/env python3
"""Audit Track B data admissibility and write a hashed artifact inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from prism_dstw.calibration.track_b_artifacts import (
    CAMPAIGN_ROOT,
    N80_ROOT,
    TRACK_A_ROOT,
    artifact_metadata,
    find_first_existing,
    write_json,
)
from prism_dstw.calibration.track_b_schemas import utc_now_iso


def _external_campaign_matches(campaign: Path, pattern: str) -> list[Path]:
    track_b_root = campaign / "track_b_chronological"
    matches: list[Path] = []
    for path in sorted(campaign.glob(pattern)):
        resolved = path.resolve()
        if track_b_root.exists():
            track_b_resolved = track_b_root.resolve()
            if resolved == track_b_resolved or track_b_resolved in resolved.parents:
                continue
        matches.append(path)
    return matches


def artifact_specs(campaign: Path) -> list[dict[str, Any]]:
    n80 = campaign / "integrated_spike_events" / "n80_full_scale"
    track_a = campaign / "track_a_generative"
    return [
        {
            "artifact_name": "translation_pathway_nodes",
            "candidates": [n80 / "translation_pathway_nodes.parquet"],
            "usable_for": ["topology_space", "observability_space", "continuity"],
            "blocking_if_missing": True,
            "fallback_allowed": False,
            "fallback_rule": None,
        },
        {
            "artifact_name": "mechanical_load_network",
            "candidates": [n80 / "mechanical_load_network.parquet"],
            "usable_for": ["topology_space", "perturbation_space", "observability_space"],
            "blocking_if_missing": True,
            "fallback_allowed": False,
            "fallback_rule": None,
        },
        {
            "artifact_name": "shear_stress_field",
            "candidates": [n80 / "shear_stress_field.parquet"],
            "usable_for": ["observability_space", "continuity"],
            "blocking_if_missing": True,
            "fallback_allowed": False,
            "fallback_rule": None,
        },
        {
            "artifact_name": "hysteresis_tensor",
            "candidates": [n80 / "hysteresis_tensor.parquet"],
            "usable_for": ["observability_space", "chronology", "continuity"],
            "blocking_if_missing": True,
            "fallback_allowed": False,
            "fallback_rule": None,
        },
        {
            "artifact_name": "phase_manifold_coherence",
            "candidates": [n80 / "phase_manifold_coherence.parquet"],
            "usable_for": ["topology_space", "perturbation_space", "observability_space"],
            "blocking_if_missing": True,
            "fallback_allowed": False,
            "fallback_rule": None,
        },
        {
            "artifact_name": "signal_grid_population_consensus",
            "candidates": [track_a / "signal_grid_population_consensus.parquet"],
            "usable_for": ["observability_space", "continuity"],
            "blocking_if_missing": True,
            "fallback_allowed": False,
            "fallback_rule": None,
        },
        {
            "artifact_name": "grid_coordinate_mapping",
            "candidates": [
                campaign / "track_0_manual_emulation" / "grid_coordinate_mapping.json",
                Path("grid_coordinate_mapping.json"),
            ],
            "usable_for": ["observability_space", "continuity"],
            "blocking_if_missing": True,
            "fallback_allowed": False,
            "fallback_rule": None,
        },
        {
            "artifact_name": "cross_species_conservation",
            "candidates": [
                track_a / "population_pgx" / "source" / "GLP1R_cross_species_conservation.csv",
                campaign / "GLP1R_cross_species_conservation.csv",
            ],
            "usable_for": ["genotype_space", "perturbation_space"],
            "blocking_if_missing": True,
            "fallback_allowed": False,
            "fallback_rule": None,
        },
        {
            "artifact_name": "temporal_cascade",
            "candidates": [n80 / "temporal_cascade.parquet", n80 / "temporal_cascade_summary.parquet"],
            "usable_for": ["chronology", "observability_space"],
            "blocking_if_missing": True,
            "fallback_allowed": True,
            "fallback_rule": "temporal_cascade_summary is usable only if temporal_cascade is absent",
        },
        {
            "artifact_name": "bocpd_survival_regimes",
            "candidates": [n80 / "bocpd_survival_regimes.parquet"],
            "usable_for": ["chronology", "observability_space"],
            "blocking_if_missing": True,
            "fallback_allowed": False,
            "fallback_rule": None,
        },
        {
            "artifact_name": "kinetic_strain_events",
            "candidates": [n80 / "kinetic_strain_events.parquet"],
            "usable_for": ["chronology", "observability_space"],
            "blocking_if_missing": True,
            "fallback_allowed": False,
            "fallback_rule": None,
        },
        {
            "artifact_name": "scaffold_consensus_grid",
            "candidates": [track_a / "signal_grid_scaffold_consensus.parquet"],
            "usable_for": ["genotype_space", "observability_space"],
            "blocking_if_missing": False,
            "fallback_allowed": True,
            "fallback_rule": "Track A scaffold consensus omitted from Track B if absent",
        },
        {
            "artifact_name": "hydration_artifacts",
            "candidates": _external_campaign_matches(campaign, "**/*hydration*.parquet"),
            "usable_for": ["continuity", "observability_space"],
            "blocking_if_missing": False,
            "fallback_allowed": True,
            "fallback_rule": "Emit HYDRATION continuity as L0_MISSING/BLOCKED_WITH_HARD_EVIDENCE if absent",
        },
        {
            "artifact_name": "nma_modes",
            "candidates": _external_campaign_matches(campaign, "**/*_nma_modes.json"),
            "usable_for": ["continuity", "observability_space"],
            "blocking_if_missing": False,
            "fallback_allowed": True,
            "fallback_rule": "Emit NMA map as L0_MISSING if no NMA JSON is present",
        },
    ]


def build_inventory(campaign: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in artifact_specs(campaign):
        candidates = list(spec["candidates"])
        path = find_first_existing(candidates) if candidates else Path("__missing__")
        exists = path.exists()
        schema, row_count, digest, size_bytes = artifact_metadata(path)
        provenance = "L0_MISSING" if not exists else "L4_RUNTIME_TELEMETRY"
        if path.suffix == ".json" and exists:
            provenance = "L3_DERIVED"
        rows.append(
            {
                "artifact_name": spec["artifact_name"],
                "path": str(path),
                "exists": exists,
                "row_count": row_count,
                "schema": schema,
                "sha256": digest,
                "size_bytes": size_bytes,
                "usable_for": spec["usable_for"],
                "provenance_class": provenance,
                "blocking_if_missing": spec["blocking_if_missing"],
                "fallback_allowed": spec["fallback_allowed"],
                "fallback_rule": spec["fallback_rule"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, default=CAMPAIGN_ROOT)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    args = parser.parse_args()

    rows = build_inventory(args.campaign)
    blockers = [
        row["artifact_name"]
        for row in rows
        if row["blocking_if_missing"] and not row["exists"]
    ]
    payload = {
        "id": "track_b_data_admissibility",
        "schema_version": "track_b.data_admissibility.v1",
        "created_at": utc_now_iso(),
        "provenance_class": "L3_DERIVED",
        "source_artifacts": sorted({row["path"] for row in rows if row["exists"]}),
        "evidence_paths": [str(args.output_parquet)],
        "campaign": str(args.campaign),
        "all_expected_inputs_accounted_for": True,
        "blocking_missing_artifacts": blockers,
        "missing_artifacts": [row["artifact_name"] for row in rows if not row["exists"]],
        "artifacts": rows,
        "verdict": "ADMISSIBLE" if not blockers else "BLOCKED_WITH_HARD_EVIDENCE",
    }
    write_json(args.output_json, payload)

    parquet_rows: list[dict[str, Any]] = []
    for row in rows:
        parquet_row = dict(row)
        parquet_row["schema"] = json.dumps(row["schema"], sort_keys=True)
        parquet_rows.append(parquet_row)
    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(parquet_rows).write_parquet(args.output_parquet)
    print(
        "track_b_data_admissibility "
        f"artifacts={len(rows)} blockers={len(blockers)} verdict={payload['verdict']} "
        f"json={args.output_json} parquet={args.output_parquet}"
    )


if __name__ == "__main__":
    main()

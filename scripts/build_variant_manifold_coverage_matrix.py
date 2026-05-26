#!/usr/bin/env python3
"""Build variant manifold coverage matrix and gap report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from prism_dstw.calibration.track_b_artifacts import read_json, write_json
from prism_dstw.calibration.track_b_schemas import utc_now_iso
from prism_dstw.calibration.translational_calibration_manifold import OBSERVABILITY_SPACE, TOPOLOGY_SPACE

MIN_TARGETS: dict[str, dict[str, Any]] = {
    "ECD_TM1_GATEWAY": {"min_variants": 4, "required_perturbations": ["SEVERING_PROBE", "RIGIDIFYING_PROBE"]},
    "TM3_TM5_CORE": {"min_variants": 4, "required_perturbations": ["SEVERING_PROBE", "CHARGE_INVERSION_PROBE"]},
    "INTRACELLULAR_LOCK_BASIN": {"min_variants": 4, "required_perturbations": ["RIGIDIFYING_PROBE", "FLEXIBILIZING_PROBE"]},
    "HYDRATION_CORRIDOR": {"min_variants": 4, "required_perturbations": ["HYDRATION_WIRE_PROBE", "CONSERVATIVE_CONTROL"]},
    "TE_HUBS": {"min_variants": 6, "required_perturbations": ["SEVERING_PROBE", "RIGIDIFYING_PROBE", "FLEXIBILIZING_PROBE"]},
    "BISTATE_RETAINED_RESIDUES": {"min_variants": 4, "required_perturbations": ["CONSERVATIVE_CONTROL", "CHARGE_INVERSION_PROBE"]},
}


def build_coverage(panel_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    panel = read_json(panel_path)
    variants = list(panel.get("variants", []))
    rows: list[dict[str, Any]] = []
    for variant in variants:
        channels = list(variant.get("observability_channels", []))
        for channel in channels:
            rows.append(
                {
                    "id": f"{variant['variant_id']}::{channel}",
                    "variant_id": variant["variant_id"],
                    "genotype_space": variant["genotype_axis"],
                    "topology_space": variant["topology_region"],
                    "perturbation_space": variant["perturbation_family"],
                    "observability_space": channel,
                    "covered": channel in OBSERVABILITY_SPACE,
                    "coverage_score": 1.0 if channel in OBSERVABILITY_SPACE else 0.0,
                    "provenance_class": variant["provenance_class"],
                    "source_artifacts": json.dumps(variant["source_artifacts"]),
                    "evidence_paths": json.dumps(variant["evidence_paths"]),
                    "created_at": utc_now_iso(),
                    "schema_version": "track_b.variant_manifold_coverage.v1",
                }
            )
    gaps: list[dict[str, Any]] = []
    region_summaries: dict[str, Any] = {}
    for region in TOPOLOGY_SPACE:
        region_variants = [v for v in variants if v.get("topology_region") == region]
        perturbations = sorted({str(v.get("perturbation_family")) for v in region_variants})
        target = MIN_TARGETS[region]
        missing_perturbations = [
            p for p in target["required_perturbations"] if p not in perturbations
        ]
        missing_count = max(int(target["min_variants"]) - len(region_variants), 0)
        covered = missing_count == 0 and not missing_perturbations
        if not covered:
            gaps.append(
                {
                    "region": region,
                    "missing_variant_count": missing_count,
                    "missing_perturbations": missing_perturbations,
                }
            )
        region_summaries[region] = {
            "variant_count": len(region_variants),
            "perturbations": perturbations,
            "covered": covered,
        }
    report = {
        "id": "track_b_variant_manifold_coverage_report",
        "schema_version": "track_b.variant_manifold_coverage_report.v1",
        "created_at": utc_now_iso(),
        "provenance_class": "L3_DERIVED",
        "source_artifacts": [str(panel_path)],
        "evidence_paths": [str(panel_path)],
        "regions_evaluated": list(TOPOLOGY_SPACE),
        "coverage_gaps": gaps,
        "region_summaries": region_summaries,
        "verdict": "COVERED" if not gaps else "GAPS_LISTED",
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-panel", type=Path, required=True)
    parser.add_argument("--output-matrix", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    rows, report = build_coverage(args.variant_panel)
    args.output_matrix.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(args.output_matrix)
    write_json(args.output_report, report)
    print(
        "variant_manifold_coverage "
        f"rows={len(rows)} gaps={len(report['coverage_gaps'])} matrix={args.output_matrix}"
    )


if __name__ == "__main__":
    main()

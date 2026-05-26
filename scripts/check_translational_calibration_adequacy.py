#!/usr/bin/env python3
"""Evaluate the Track B translational calibration adequacy gate."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import polars as pl

from prism_dstw.calibration.track_b_artifacts import read_json, write_json
from prism_dstw.calibration.track_b_schemas import utc_now_iso
from prism_dstw.calibration.translational_calibration_manifold import TOPOLOGY_SPACE


def evaluate(
    variant_panel: Path,
    coverage_report: Path,
    topology_registry: Path,
    output: Path,
) -> dict[str, Any]:
    panel = read_json(variant_panel)
    report = read_json(coverage_report)
    registry = read_json(topology_registry)
    variants = list(panel.get("variants", []))
    region_summaries = dict(report.get("region_summaries", {}))
    registry_regions = dict(registry.get("regions", {}))
    covered_or_blocked = [
        region for region in TOPOLOGY_SPACE if region_summaries.get(region, {}).get("covered", False)
    ]
    global_families = {str(v.get("perturbation_family")) for v in variants}
    failed: list[str] = []
    passed: list[str] = []

    if len(covered_or_blocked) >= 5:
        passed.append("at_least_5_of_6_regions_covered")
    else:
        failed.append("at_least_5_of_6_regions_covered")
    for region in ("TE_HUBS", "INTRACELLULAR_LOCK_BASIN"):
        if region_summaries.get(region, {}).get("covered", False):
            passed.append(f"{region}_covered")
        else:
            failed.append(f"{region}_covered")
    if region_summaries.get("HYDRATION_CORRIDOR", {}).get("covered", False):
        passed.append("HYDRATION_CORRIDOR_covered")
    else:
        failed.append("HYDRATION_CORRIDOR_covered_or_evidence_blocked")
    if all(len(v.get("observability_channels", [])) >= 2 for v in variants):
        passed.append("every_variant_has_2_observability_channels")
    else:
        failed.append("every_variant_has_2_observability_channels")
    if len(global_families) >= 3:
        passed.append("at_least_3_perturbation_families")
    else:
        failed.append("at_least_3_perturbation_families")
    if all(v.get("selection_features") != ["evolutionary_conservation"] for v in variants):
        passed.append("conservation_not_only_selection_feature")
    else:
        failed.append("conservation_not_only_selection_feature")
    if all(
        region in registry_regions and bool(registry_regions[region].get("purpose"))
        for region in TOPOLOGY_SPACE
    ):
        passed.append("every_region_declares_calibration_purpose")
    else:
        failed.append("every_region_declares_calibration_purpose")

    verdict = (
        "CALIBRATION_MANIFOLD_ADEQUATE"
        if not failed
        else "CALIBRATION_MANIFOLD_REJECTED_TOO_SPARSE"
    )
    payload = {
        "id": "track_b_calibration_adequacy_gate",
        "schema_version": "track_b.translational_calibration_adequacy.v1",
        "created_at": utc_now_iso(),
        "provenance_class": "L3_DERIVED",
        "source_artifacts": [str(variant_panel), str(coverage_report), str(topology_registry)],
        "evidence_paths": [str(variant_panel), str(coverage_report), str(topology_registry)],
        "verdict": verdict,
        "passed_rules": passed,
        "failed_rules": failed,
        "blocked_rules": [],
    }
    write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-panel", type=Path, required=True)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--topology-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = evaluate(args.variant_panel, args.coverage_report, args.topology_registry, args.output)
    print(f"translational_calibration_adequacy verdict={payload['verdict']} output={args.output}")


if __name__ == "__main__":
    main()

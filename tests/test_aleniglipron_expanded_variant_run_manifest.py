from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path("campaigns/glp1r_aleniglipron/track_b_chronological/expanded_variant_run")
MANIFEST = ROOT / "aleniglipron_expanded_variant_run_manifest.json"
TARGET_REPORT = ROOT / "aleniglipron_receptor_target_avoidance_report.json"
TARGET_MATRIX = ROOT / "aleniglipron_receptor_target_avoidance_matrix.parquet"
RUN_PLAN = ROOT / "true_perturbed_trajectory_run_plan.json"
TOPOLOGY_MANIFEST = ROOT / "variant_topology_materialization_manifest.json"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_expanded_variant_manifest_outnumbers_baseline_and_queues_true_work() -> None:
    manifest = _read_json(MANIFEST)

    assert manifest["variant_count"] >= manifest["baseline_genealogical_panel_variant_count"] * 10
    assert manifest["trajectory_queue_total_count"] > 1000
    assert manifest["observed_grid_extension_count"] > 0
    assert manifest["claim_boundary"].startswith("Rows queue true perturbed PRISM trajectories")


def test_expanded_variant_manifest_has_no_sequence_only_or_fake_observed_rows() -> None:
    manifest = _read_json(MANIFEST)
    validation = manifest["validation_report"]

    assert validation["verdict"] == "PASS"
    assert validation["sequence_only_variant_count"] == 0
    assert validation["missing_axis_count"] == 0
    assert validation["projected_claimed_observed_count"] == 0


def test_target_avoidance_matrix_flags_asn182_and_primary_lock_targets() -> None:
    report = _read_json(TARGET_REPORT)
    matrix = pl.read_parquet(TARGET_MATRIX)

    avoid_ids = {
        str(row["residue_id"])
        for row in report["receptor_regions_to_avoid_or_use_as_controls"]
    }
    target_regions = {
        str(row["topology_region"])
        for row in report["top_receptor_regions_to_target"]
        if row["recommendation"] == "TARGET"
    }
    assert "N182" in avoid_ids
    assert "INTRACELLULAR_LOCK_BASIN" in target_regions
    assert matrix.filter(pl.col("recommendation") == "TARGET").height > 0
    assert matrix.filter(pl.col("recommendation").str.starts_with("AVOID")).height > 0


def test_run_plan_materializes_priority_batch_manifests() -> None:
    manifest = _read_json(MANIFEST)
    run_plan = _read_json(RUN_PLAN)
    topology_manifest = _read_json(TOPOLOGY_MANIFEST)

    assert topology_manifest["job_count"] == manifest["trajectory_queue_total_count"]
    for batch in run_plan["batches"]:
        path = Path(str(batch["trajectory_manifest_path"]))
        payload = _read_json(path)
        assert payload["variant_count"] == batch["variant_count"]
        assert payload["claim_boundary"] == "Batch manifests queue PRISM work; they are not observed trajectory evidence."

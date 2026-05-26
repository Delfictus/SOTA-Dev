import json
from pathlib import Path


ROOT = Path("campaigns/glp1r_aleniglipron/track_b_chronological")


def test_hydration_admissibility_does_not_self_reference_generated_fallback() -> None:
    payload = json.loads((ROOT / "track_b_data_admissibility.json").read_text())
    hydration = next(
        artifact for artifact in payload["artifacts"] if artifact["artifact_name"] == "hydration_artifacts"
    )

    assert hydration["exists"] is False
    assert hydration["path"] == "__missing__"
    assert hydration["provenance_class"] == "L0_MISSING"
    assert "track_b_chronological" not in hydration["path"]
    assert "hydration_continuity_map.parquet" not in hydration["path"]


def test_te_hub_manifest_has_explicit_coverage_links() -> None:
    payload = json.loads((ROOT / "phase_3_te_hub_variant_manifest.json").read_text())

    assert payload["is_subset_view"] is True
    assert payload["te_hub_variant_count"] < payload["full_panel_variant_count"]
    assert payload["variants"]
    assert all(variant["coverage_ids"] for variant in payload["variants"])

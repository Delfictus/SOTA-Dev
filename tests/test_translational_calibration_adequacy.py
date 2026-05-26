import json
from pathlib import Path

from prism_dstw.calibration.track_b_artifacts import write_json
from scripts.check_translational_calibration_adequacy import evaluate


ROOT = Path("campaigns/glp1r_aleniglipron/track_b_chronological")


def test_adequacy_gate_passes_generated_panel() -> None:
    payload = json.loads((ROOT / "translational_calibration_adequacy_gate.json").read_text())
    assert payload["verdict"] == "CALIBRATION_MANIFOLD_ADEQUATE"
    assert not payload["failed_rules"]


def test_adequacy_gate_rejects_sparse_semantics() -> None:
    tmp = Path("/mnt/storage/tmp/track_b_sparse_adequacy_test")
    tmp.mkdir(parents=True, exist_ok=True)
    panel = tmp / "panel.json"
    coverage = tmp / "coverage.json"
    registry = tmp / "registry.json"
    output = tmp / "adequacy.json"
    write_json(
        panel,
        {
            "variants": [
                {
                    "perturbation_family": "SEVERING_PROBE",
                    "observability_channels": ["signal_grid"],
                    "selection_features": ["evolutionary_conservation"],
                }
            ]
        },
    )
    write_json(
        coverage,
        {
            "region_summaries": {
                "TE_HUBS": {"covered": True},
            }
        },
    )
    write_json(registry, {"regions": {"TE_HUBS": {"purpose": "causal rerouting"}}})
    payload = evaluate(panel, coverage, registry, output)
    assert payload["verdict"] == "CALIBRATION_MANIFOLD_REJECTED_TOO_SPARSE"
    assert "every_region_declares_calibration_purpose" in payload["failed_rules"]

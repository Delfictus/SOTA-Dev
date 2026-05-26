from pathlib import Path


RUNTIME = Path("campaigns/glp1r_aleniglipron/track_b_chronological/runtime")


def test_runtime_instantiation_creates_required_manifests() -> None:
    for path in (
        RUNTIME / "config" / "track_b_runtime_config.yaml",
        RUNTIME / "config" / "oracle_config.yaml",
        RUNTIME / "config" / "calibration_config.yaml",
        RUNTIME / "manifests" / "artifact_manifest.json",
        RUNTIME / "manifests" / "cloud_sync_manifest.json",
        RUNTIME / "manifests" / "vectorize_manifest.json",
        RUNTIME / "bin" / "oracle_scorer",
    ):
        assert path.is_file()

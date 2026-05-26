import json
from pathlib import Path

from prism_dstw.calibration.track_b_artifacts import sha256_file


MANIFEST = Path("campaigns/glp1r_aleniglipron/track_b_chronological/track_b_release_manifest.json")


def test_release_manifest_hashes_all_deliverables() -> None:
    payload = json.loads(MANIFEST.read_text())
    assert payload["verdict"] == "RELEASE_MANIFEST_COMPLETE"
    assert payload["artifact_count"] >= 15
    for artifact in payload["artifacts"]:
        path = Path(artifact["path"])
        assert path.is_file()
        assert sha256_file(path) == artifact["sha256"]

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def release_root() -> Path:
    value = os.environ.get("PRISM_RELEASE_ROOT")
    if not value:
        pytest.skip("PRISM_RELEASE_ROOT is required for release acceptance checks")
    return Path(value)


def restored_source_root() -> Path:
    value = os.environ.get("PRISM_RESTORED_SOURCE_ROOT")
    if not value:
        pytest.skip("PRISM_RESTORED_SOURCE_ROOT is required for release acceptance checks")
    return Path(value)


def test_restore_integrity() -> None:
    root = release_root()
    source = restored_source_root()
    assert (root / "MANIFEST.json").exists()
    assert (root / "CHECKSUMS.sha256").exists()
    assert (root / "RESTORE_SOURCE_MATRIX.json").exists()
    assert (root / "RELEASE_STATUS.json").exists()
    assert (source / "scripts").exists()
    assert (source / "src").exists()
    assert (source / "crates").exists()


def test_claim_boundary_truth() -> None:
    text = (release_root() / "RELEASE_CLAIM_BOUNDARY.md").read_text(encoding="utf-8")
    assert "Phase 2C sealed receptor/variant evidence may be claimed." in text
    assert "V2 hydration to DSTW context integration is implemented and smoke-verified." in text
    assert "Full hydration extraction across the 104GB input surface has not been fully run and remains unmeasured." in text
    assert "Full Phase 1-3 production completion may not be claimed." in text
    assert "Hydration completion may not be claimed" in text


def test_phase2c_manifest_integrity() -> None:
    phase2c = restored_source_root() / "campaigns/glp1r_aleniglipron/PHASE2C_SEALED_MANIFEST.json"
    payload = json.loads(phase2c.read_text(encoding="utf-8"))
    assert payload.get("all_pass") is True
    assert int(payload.get("replica_count", 0)) >= 1


def test_candidate_manifest_consistency() -> None:
    manifest = json.loads((release_root() / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["claim_boundary"]["phase2c_status"] == "SEALED"
    assert manifest["claim_boundary"]["hydration_dstw_integration_status"] == "IMPLEMENTED_AND_SMOKE_VERIFIED"
    assert manifest["claim_boundary"]["hydration_full_run_status"] == "NOT_FULLY_RUN_104GB_INPUT_UNMEASURED"
    assert manifest["claim_boundary"]["candidate_matrix_completion_status"] == "PARTIAL_ONLY_OBSERVED_OUTPUTS_CLAIMABLE"
    assert manifest["claim_boundary"]["hydration_production_complete"] is False
    candidate_manifest = release_root() / manifest["candidate_matrix"]["manifest_path"]
    reconciliation = release_root() / manifest["candidate_matrix"]["status_reconciliation_path"]
    assert candidate_manifest.exists()
    assert reconciliation.exists()
    payload = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    assert payload["entries"]
    first = payload["entries"][0]
    assert "candidate_id" in first
    assert "expected_outputs" in first
    assert "observed_outputs" in first


def test_no_absolute_workspace_dependency_in_readme() -> None:
    readme = (release_root() / "README_RESTORE.md").read_text(encoding="utf-8")
    assert "/home/diddy/Desktop/Prism4D-bio" not in readme


def test_runtime_validation_script_present() -> None:
    assert (release_root() / "validation/validate_release.sh").exists()

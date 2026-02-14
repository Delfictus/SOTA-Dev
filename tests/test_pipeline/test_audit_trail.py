"""Tests for the anti-leakage audit trail module."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from scripts.pipeline.audit_trail import (
    BLIND_STAGES,
    EXTERNAL_ALLOWED_STAGES,
    AuditTrail,
    StageRecord,
    _sha256,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_output(tmp_path):
    return str(tmp_path)


@pytest.fixture
def sample_file(tmp_path):
    """Create a small file for hashing."""
    p = tmp_path / "sample.pdb"
    p.write_text("ATOM  1  CA  ALA A   1       0.0  0.0  0.0  1.00  0.00")
    return str(p)


@pytest.fixture
def trail(tmp_output, sample_file):
    t = AuditTrail(output_dir=tmp_output)
    t.start_pipeline(pdb_path=sample_file)
    return t


# ---------------------------------------------------------------------------
# SHA-256 hashing
# ---------------------------------------------------------------------------

class TestSHA256:
    def test_hash_file(self, sample_file):
        h = _sha256(sample_file)
        assert len(h) == 64
        assert h == _sha256(sample_file)  # deterministic

    def test_hash_nonexistent(self):
        h = _sha256("/nonexistent/file.pdb")
        assert h.startswith("UNREADABLE:")

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("hello")
        f2.write_text("world")
        assert _sha256(str(f1)) != _sha256(str(f2))


# ---------------------------------------------------------------------------
# StageRecord
# ---------------------------------------------------------------------------

class TestStageRecord:
    def test_default_values(self):
        rec = StageRecord(stage_name="test", status="completed")
        assert rec.stage_name == "test"
        assert rec.status == "completed"
        assert rec.input_hashes == {}
        assert rec.errors == []

    def test_to_dict(self):
        rec = StageRecord(
            stage_name="pocket_refinement",
            status="completed",
            start_time="2026-01-01T00:00:00+00:00",
            end_time="2026-01-01T00:30:00+00:00",
            duration_seconds=1800.0,
            result_summary={"pocket_stable": True},
        )
        d = rec.to_dict()
        assert d["stage_name"] == "pocket_refinement"
        assert d["duration_seconds"] == 1800.0
        assert d["result_summary"]["pocket_stable"] is True


# ---------------------------------------------------------------------------
# AuditTrail lifecycle
# ---------------------------------------------------------------------------

class TestAuditTrailLifecycle:
    def test_start_pipeline(self, trail, sample_file):
        assert trail.pipeline_start != ""
        assert trail.input_pdb_path == sample_file
        assert len(trail.input_pdb_hash) == 64

    def test_begin_end_stage(self, trail, sample_file):
        trail.begin_stage("target_classification", inputs=[sample_file])
        trail.end_stage(
            status="completed",
            result={"classification": "soluble"},
        )
        assert len(trail.stages) == 1
        assert trail.stages[0].stage_name == "target_classification"
        assert trail.stages[0].status == "completed"
        assert trail.stages[0].result_summary["classification"] == "soluble"
        assert sample_file in trail.stages[0].input_hashes

    def test_skip_stage(self, trail):
        trail.skip_stage("membrane_building", reason="soluble target")
        assert len(trail.stages) == 1
        assert trail.stages[0].status == "skipped"
        assert trail.stages[0].result_summary["skip_reason"] == "soluble target"

    def test_log_stage_convenience(self, trail, sample_file):
        trail.log_stage(
            "prism_detection",
            inputs=[sample_file],
            result={"spikes": 42},
        )
        assert len(trail.stages) == 1
        assert trail.stages[0].status == "completed"

    def test_finalize(self, trail):
        trail.begin_stage("target_classification")
        trail.end_stage(status="completed")
        trail.finalize()
        assert trail.pipeline_end != ""
        assert trail.is_clean

    def test_multiple_stages(self, trail):
        for name in ["target_classification", "protein_fixing", "prism_detection"]:
            trail.begin_stage(name)
            trail.end_stage(status="completed")
        assert len(trail.stages) == 3


# ---------------------------------------------------------------------------
# Anti-leakage detection
# ---------------------------------------------------------------------------

class TestAntiLeakage:
    def test_no_violation_during_external_stage(self, trail):
        trail.begin_stage("filtering")
        trail.log_external_query("PubChem", "Tanimoto check")
        trail.end_stage(status="completed")
        trail.finalize()
        assert trail.is_clean

    def test_violation_during_blind_stage(self, trail):
        trail.begin_stage("prism_detection")
        trail.log_external_query("PubChem", "aspirin CID lookup")
        trail.end_stage(status="completed")
        trail.finalize()
        assert not trail.is_clean
        assert len(trail.anti_leakage_violations) == 1
        assert "PubChem" in trail.anti_leakage_violations[0]
        assert "prism_detection" in trail.anti_leakage_violations[0]

    def test_multiple_violations(self, trail):
        trail.begin_stage("pocket_refinement")
        trail.log_external_query("DrugBank", "query 1")
        trail.log_external_query("ChEMBL", "query 2")
        trail.end_stage(status="completed")
        trail.finalize()
        assert not trail.is_clean
        assert len(trail.anti_leakage_violations) == 2

    def test_blind_stages_constant(self):
        """Verify all blind stages are defined."""
        assert "prism_detection" in BLIND_STAGES
        assert "pocket_refinement" in BLIND_STAGES
        assert "pharmacophore" in BLIND_STAGES
        assert "filtering" not in BLIND_STAGES

    def test_external_allowed_stages(self):
        """Verify external-allowed stages."""
        assert "filtering" in EXTERNAL_ALLOWED_STAGES
        assert "fep" in EXTERNAL_ALLOWED_STAGES
        assert "prism_detection" not in EXTERNAL_ALLOWED_STAGES

    def test_no_overlap_blind_external(self):
        """Blind and external-allowed sets must not overlap."""
        assert BLIND_STAGES.isdisjoint(EXTERNAL_ALLOWED_STAGES)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestAuditTrailSerialization:
    def test_to_dict(self, trail):
        trail.begin_stage("target_classification")
        trail.end_stage(status="completed", result={"cls": "soluble"})
        trail.finalize()
        d = trail.to_dict()
        assert d["anti_leakage_clean"] is True
        assert len(d["stages"]) == 1
        assert d["input_pdb_hash"] == trail.input_pdb_hash

    def test_to_json(self, trail):
        trail.finalize()
        j = trail.to_json()
        parsed = json.loads(j)
        assert "pipeline_id" in parsed
        assert "stages" in parsed

    def test_save_and_load(self, trail, tmp_output):
        trail.begin_stage("prism_detection")
        trail.end_stage(status="completed", result={"n_spikes": 10})
        trail.finalize()
        path = trail.save()
        assert Path(path).exists()

        loaded = AuditTrail.from_json_file(path)
        assert loaded.pipeline_id == trail.pipeline_id
        assert len(loaded.stages) == 1
        assert loaded.stages[0].result_summary["n_spikes"] == 10
        assert loaded.is_clean

    def test_save_custom_path(self, trail, tmp_path):
        trail.finalize()
        custom = str(tmp_path / "subdir" / "custom_audit.json")
        path = trail.save(path=custom)
        assert Path(path).exists()
        assert path == custom

    def test_output_hashing(self, trail, sample_file, tmp_path):
        out_file = tmp_path / "output.json"
        out_file.write_text('{"result": true}')
        trail.begin_stage("test", inputs=[sample_file])
        trail.end_stage(status="completed", outputs=[str(out_file)])
        assert str(out_file) in trail.stages[0].output_hashes
        assert len(trail.stages[0].output_hashes[str(out_file)]) == 64

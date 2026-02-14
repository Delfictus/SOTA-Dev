"""End-to-end tests for the PRISM-4D pipeline orchestrator.

Tests the master orchestrator in dry-run mode — validates that all 16
stages are wired correctly, stage ordering is preserved, audit trail
is clean, and reports are generated.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from scripts.pipeline.audit_trail import AuditTrail
from scripts.pipeline.prism_fep_pipeline import (
    PrismFepPipeline,
    StageRunner,
    main,
    parse_args,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_path():
    return str(Path(__file__).resolve().parents[2] / "scripts" / "pipeline" / "pipeline_config.yaml")


@pytest.fixture
def config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def mock_pdb(tmp_path):
    """Create a minimal PDB file."""
    pdb = tmp_path / "test.pdb"
    pdb.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
        "ATOM      2  CA  GLY A   2       3.800   0.000   0.000  1.00  0.00\n"
        "END\n"
    )
    return str(pdb)


@pytest.fixture
def pipeline(config, mock_pdb, tmp_path, config_path):
    """Create a pipeline in dry-run mode."""
    cli = {
        "pdb": mock_pdb,
        "output_dir": str(tmp_path / "output"),
        "skip_fep": True,
        "dry_run": True,
        "verbose": False,
        "config_path": config_path,
    }
    config["target_name"] = "TEST_TARGET"
    config["pdb_id"] = "1ABC"
    return PrismFepPipeline(config, cli)


# ---------------------------------------------------------------------------
# StageRunner
# ---------------------------------------------------------------------------

class TestStageRunner:
    def test_dry_run_mode(self, config):
        runner = StageRunner(conda_envs=config.get("conda_envs", {}), dry_run=True)
        result = runner.run("preprocessing", "scripts/preprocessing/target_classifier.py", ["--pdb", "test.pdb"])
        assert result.get("dry_run") is True
        assert "command" in result

    def test_dry_run_inline(self, config):
        runner = StageRunner(conda_envs=config.get("conda_envs", {}), dry_run=True)
        result = runner.run_inline_python("ensemble", "print('hello')")
        assert result.get("dry_run") is True

    def test_find_conda(self):
        path = StageRunner._find_conda()
        assert isinstance(path, str)
        assert len(path) > 0

    def test_build_command(self, config):
        runner = StageRunner(conda_envs=config.get("conda_envs", {}), dry_run=True)
        cmd = runner._build_command("prism4d-preprocessing", "/path/to/script.py", ["--arg", "val"])
        assert "python" in cmd
        assert "/path/to/script.py" in cmd
        assert "--arg" in cmd
        assert "val" in cmd


# ---------------------------------------------------------------------------
# Pipeline stages (dry-run)
# ---------------------------------------------------------------------------

class TestPipelineStages:
    def test_stage_01_target_classification(self, pipeline):
        pipeline.stage_01_target_classification()
        assert len(pipeline.audit.stages) == 1
        assert pipeline.audit.stages[0].stage_name == "target_classification"
        assert pipeline.audit.stages[0].status == "completed"

    def test_stage_02_protein_fixing(self, pipeline):
        pipeline.stage_02_protein_fixing()
        assert pipeline.audit.stages[-1].stage_name == "protein_fixing"

    def test_stage_03_membrane_skipped_for_soluble(self, pipeline):
        pipeline._classification = "soluble"
        pipeline.stage_03_membrane_building()
        assert pipeline.audit.stages[-1].status == "skipped"
        assert "soluble" in pipeline.audit.stages[-1].result_summary.get("skip_reason", "")

    def test_stage_04_prism_detection(self, pipeline):
        pipeline.stage_04_prism_detection()
        assert pipeline.audit.stages[-1].stage_name == "prism_detection"
        # Should create mock spike JSON in dry-run
        assert Path(pipeline._spike_json).exists()

    def test_stage_05_pocket_refinement(self, pipeline):
        pipeline.stage_04_prism_detection()
        pipeline.stage_05_pocket_refinement()
        assert pipeline.audit.stages[-1].stage_name == "pocket_refinement"

    def test_stage_06_water_map(self, pipeline):
        pipeline.stage_06_water_map()
        wm_json = Path(pipeline.output_dir) / "explicit_solvent" / "water_map.json"
        assert wm_json.exists()

    def test_stage_07_pocket_popen(self, pipeline):
        pipeline.stage_07_pocket_popen()
        pd_json = Path(pipeline.output_dir) / "ensemble" / "pocket_dynamics.json"
        assert pd_json.exists()
        data = json.loads(pd_json.read_text())
        assert "p_open" in data

    def test_stage_14_fep_skipped(self, pipeline):
        pipeline.stage_14_fep()
        assert pipeline.audit.stages[-1].stage_name == "fep"
        assert pipeline.audit.stages[-1].status == "skipped"

    def test_disabled_stage(self, pipeline):
        pipeline.cfg["stages"]["target_classification"] = False
        pipeline.stage_01_target_classification()
        assert pipeline.audit.stages[-1].status == "skipped"
        assert "disabled" in pipeline.audit.stages[-1].result_summary.get("skip_reason", "")


# ---------------------------------------------------------------------------
# Full pipeline dry-run
# ---------------------------------------------------------------------------

class TestFullPipelineDryRun:
    def test_full_run(self, pipeline):
        pipeline.run()
        # All 16 stages should be recorded (some skipped)
        stage_names = [s.stage_name for s in pipeline.audit.stages]
        assert "target_classification" in stage_names
        assert "protein_fixing" in stage_names
        assert "prism_detection" in stage_names
        assert "pocket_refinement" in stage_names
        assert "water_map" in stage_names
        assert "reporting" in stage_names

    def test_audit_trail_clean(self, pipeline):
        pipeline.run()
        assert pipeline.audit.is_clean

    def test_audit_trail_saved(self, pipeline):
        pipeline.run()
        audit_path = Path(pipeline.output_dir) / "audit_trail.json"
        assert audit_path.exists()
        data = json.loads(audit_path.read_text())
        assert data["anti_leakage_clean"] is True

    def test_reports_generated(self, pipeline):
        pipeline.run()
        reports_dir = Path(pipeline.output_dir) / "reports"
        assert reports_dir.exists()
        assert (reports_dir / "campaign_summary.md").exists()
        assert (reports_dir / "campaign_data.json").exists()

    def test_stage_artifacts_created(self, pipeline):
        pipeline.run()
        # Spike JSON
        assert Path(pipeline._spike_json).exists()
        # Water map
        wm = Path(pipeline.output_dir) / "explicit_solvent" / "water_map.json"
        assert wm.exists()
        # Pocket dynamics
        pd = Path(pipeline.output_dir) / "ensemble" / "pocket_dynamics.json"
        assert pd.exists()

    def test_fep_skipped_with_flag(self, pipeline):
        pipeline.run()
        fep_stages = [s for s in pipeline.audit.stages if s.stage_name == "fep"]
        assert len(fep_stages) == 1
        assert fep_stages[0].status == "skipped"

    def test_membrane_stage_skipped_for_soluble(self, pipeline):
        pipeline.run()
        membrane_stages = [s for s in pipeline.audit.stages if s.stage_name == "membrane_building"]
        assert len(membrane_stages) == 1
        assert membrane_stages[0].status == "skipped"


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

class TestCLI:
    def test_parse_args_minimal(self):
        args = parse_args(["--pdb", "test.pdb"])
        assert args.pdb == "test.pdb"
        assert args.skip_fep is False
        assert args.dry_run is False

    def test_parse_args_full(self):
        args = parse_args([
            "--pdb", "input.pdb",
            "--config", "cfg.yaml",
            "--output-dir", "/tmp/out",
            "--skip-fep",
            "--dry-run",
            "--target-name", "BRAF",
            "--pdb-id", "1UWH",
            "-v",
        ])
        assert args.pdb == "input.pdb"
        assert args.config == "cfg.yaml"
        assert args.output_dir == "/tmp/out"
        assert args.skip_fep is True
        assert args.dry_run is True
        assert args.target_name == "BRAF"
        assert args.pdb_id == "1UWH"
        assert args.verbose is True

    def test_main_dry_run(self, mock_pdb, config_path, tmp_path):
        main([
            "--pdb", mock_pdb,
            "--config", config_path,
            "--output-dir", str(tmp_path / "main_test"),
            "--dry-run",
            "--skip-fep",
            "--target-name", "TEST",
        ])
        audit = tmp_path / "main_test" / "audit_trail.json"
        assert audit.exists()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_config_has_all_stages(self, config):
        stages = config["stages"]
        expected = [
            "target_classification", "protein_fixing", "membrane_building",
            "prism_detection", "pocket_refinement", "water_map", "pocket_popen",
            "pharmacophore", "generation", "tautomer_enumeration", "filtering",
            "docking", "ensemble_scoring", "fep", "reporting", "audit",
        ]
        for s in expected:
            assert s in stages, f"Missing stage: {s}"

    def test_config_has_conda_envs(self, config):
        envs = config["conda_envs"]
        expected_keys = [
            "preprocessing", "explicit_solvent", "genphore_phoregen",
            "genphore_pgmg", "filters", "ensemble", "fep",
        ]
        for k in expected_keys:
            assert k in envs, f"Missing conda env: {k}"

    def test_config_has_all_sections(self, config):
        sections = [
            "preprocessing", "explicit_solvent", "generation",
            "filtering", "docking", "ensemble", "fep", "reporting",
        ]
        for s in sections:
            assert s in config, f"Missing config section: {s}"

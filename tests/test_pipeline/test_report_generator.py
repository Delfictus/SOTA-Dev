"""Tests for the report generator module."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.pipeline.report_generator import ReportGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gen(tmp_path):
    return ReportGenerator(
        output_dir=str(tmp_path),
        target_name="KRAS_G12C",
        pdb_id="4OBE",
        project_name="Test Campaign",
    )


@pytest.fixture
def sample_candidate():
    return {
        "molecule": {
            "smiles": "c1ccc(-c2ccncc2)cc1",
            "source": "phoregen",
            "pharmacophore_match_score": 0.82,
        },
        "qed_score": 0.68,
        "sa_score": 2.5,
        "lipinski_violations": 0,
        "pains_alerts": [],
        "tanimoto_to_nearest_known": 0.38,
        "nearest_known_cid": "54321",
        "cluster_id": 2,
    }


@pytest.fixture
def sample_fep():
    return {
        "method": "ABFE",
        "delta_g_bind": -9.1,
        "delta_g_error": 0.8,
        "corrected_delta_g": -8.7,
        "convergence_passed": True,
        "hysteresis_kcal": 0.3,
        "overlap_minimum": 0.15,
        "max_protein_rmsd": 1.5,
        "passed_qc": True,
        "classification": "NOVEL_HIT",
        "spike_pharmacophore_match": "5/5 within 1.8A",
    }


@pytest.fixture
def sample_ensemble():
    return {"delta_g_mean": -7.9, "delta_g_std": 1.2, "n_snapshots": 100}


@pytest.fixture
def sample_ie():
    return {"minus_t_delta_s": 2.8, "delta_g_ie": -4.9}


@pytest.fixture
def sample_pocket_data():
    return {
        "solvent": {
            "pocket_stable": True,
            "pocket_rmsd_mean": 1.4,
            "pocket_volume_mean": 380.0,
            "n_structural_waters": 3,
        },
        "water_map": {
            "n_displaceable": 2,
            "total_displacement_energy": 3.1,
        },
        "dynamics": {
            "p_open": 0.68,
            "p_open_error": 0.04,
            "druggability_classification": "STABLE_OPEN",
        },
    }


# ---------------------------------------------------------------------------
# Candidate management
# ---------------------------------------------------------------------------

class TestCandidateManagement:
    def test_add_candidate_basic(self, gen, sample_candidate):
        gen.add_candidate(sample_candidate)
        assert len(gen._candidates) == 1
        c = gen._candidates[0]
        assert c["smiles"] == "c1ccc(-c2ccncc2)cc1"
        assert c["qed_score"] == 0.68

    def test_add_candidate_with_fep(self, gen, sample_candidate, sample_fep):
        gen.add_candidate(sample_candidate, fep_result=sample_fep)
        c = gen._candidates[0]
        assert c["has_fep_data"] is True
        assert c["fep_dg_bind"] == -9.1
        assert c["classification"] == "NOVEL_HIT"

    def test_add_candidate_without_fep(self, gen, sample_candidate):
        gen.add_candidate(sample_candidate)
        c = gen._candidates[0]
        assert c["has_fep_data"] is False
        assert c["fep_dg"] is None

    def test_add_candidate_with_ensemble(self, gen, sample_candidate, sample_ensemble):
        gen.add_candidate(sample_candidate, ensemble_result=sample_ensemble)
        c = gen._candidates[0]
        assert c["has_ensemble_data"] is True
        assert c["ensemble_dg_mean"] == -7.9

    def test_add_candidate_with_ie(self, gen, sample_candidate, sample_ie):
        gen.add_candidate(sample_candidate, ie_result=sample_ie)
        c = gen._candidates[0]
        assert c["ie_minus_tds"] == 2.8
        assert c["ie_dg"] == -4.9

    def test_multiple_candidates(self, gen, sample_candidate):
        for i in range(5):
            c = sample_candidate.copy()
            c["cluster_id"] = i
            gen.add_candidate(c)
        assert len(gen._candidates) == 5


# ---------------------------------------------------------------------------
# Pocket data
# ---------------------------------------------------------------------------

class TestPocketData:
    def test_set_pocket_data_full(self, gen, sample_pocket_data):
        gen.set_pocket_data(
            solvent_result=sample_pocket_data["solvent"],
            water_map=sample_pocket_data["water_map"],
            pocket_dynamics=sample_pocket_data["dynamics"],
        )
        pd = gen._pocket_data
        assert pd["has_pocket_data"] is True
        assert pd["pocket_stability"] == "STABLE"
        assert pd["p_open"] == 0.68
        assert pd["n_displaceable_waters"] == 2

    def test_set_pocket_data_collapsed(self, gen):
        gen.set_pocket_data(
            solvent_result={"pocket_stable": False, "pocket_rmsd_mean": 4.5, "pocket_volume_mean": 50.0, "n_structural_waters": 0},
        )
        assert gen._pocket_data["pocket_stability"] == "COLLAPSED"

    def test_set_pocket_data_none(self, gen):
        gen.set_pocket_data()
        assert gen._pocket_data.get("has_pocket_data", False) is False


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

class TestAuditData:
    def test_set_audit_trail(self, gen):
        gen.set_audit_trail({
            "anti_leakage_clean": True,
            "anti_leakage_violations": [],
            "input_pdb_hash": "abc123",
            "stages": [{"stage_name": "detection", "status": "completed", "duration_seconds": 10.0}],
        })
        assert gen._audit_data["anti_leakage_clean"] is True
        assert len(gen._stages) == 1


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

class TestHitReport:
    def test_generate_hit_report(self, gen, sample_candidate, sample_fep, sample_pocket_data):
        gen.add_candidate(sample_candidate, fep_result=sample_fep)
        gen.set_pocket_data(
            solvent_result=sample_pocket_data["solvent"],
            water_map=sample_pocket_data["water_map"],
            pocket_dynamics=sample_pocket_data["dynamics"],
        )
        content = gen.generate_hit_report(gen._candidates[0])
        assert "KRAS_G12C" in content
        assert "NOVEL_HIT" in content
        assert "-9.10" in content
        assert "c1ccc(-c2ccncc2)cc1" in content

    def test_hit_report_without_fep(self, gen, sample_candidate):
        gen.add_candidate(sample_candidate)
        content = gen.generate_hit_report(gen._candidates[0])
        assert "KRAS_G12C" in content
        assert "FEP Validation" not in content


class TestCampaignReport:
    def test_generate_campaign_report(self, gen, sample_candidate, sample_fep):
        gen.add_candidate(sample_candidate, fep_result=sample_fep)
        gen.set_generation_stats(n_generated=11000, n_filtered=5)
        gen.set_audit_trail({
            "anti_leakage_clean": True,
            "anti_leakage_violations": [],
            "input_pdb_hash": "abc123",
            "stages": [],
        })
        content = gen.generate_campaign_report()
        assert "Campaign Report" in content
        assert "KRAS_G12C" in content
        assert "NOVEL_HIT" in content
        assert "11000" in content

    def test_campaign_report_classification_counts(self, gen, sample_candidate, sample_fep):
        # Add NOVEL_HIT
        gen.add_candidate(sample_candidate, fep_result=sample_fep)
        # Add WEAK_BINDER
        weak_fep = sample_fep.copy()
        weak_fep["classification"] = "WEAK_BINDER"
        gen.add_candidate(sample_candidate, fep_result=weak_fep)
        gen.set_audit_trail({"anti_leakage_clean": True, "anti_leakage_violations": [], "input_pdb_hash": "", "stages": []})
        content = gen.generate_campaign_report()
        assert "NOVEL_HIT" in content
        assert "WEAK_BINDER" in content


class TestGenerateAll:
    def test_generate_all_creates_files(self, gen, sample_candidate, sample_fep, tmp_path):
        gen.add_candidate(sample_candidate, fep_result=sample_fep)
        gen.set_generation_stats(1000, 5)
        gen.set_audit_trail({"anti_leakage_clean": True, "anti_leakage_violations": [], "input_pdb_hash": "x", "stages": []})
        paths = gen.generate_all()
        assert "campaign_summary" in paths
        assert "campaign_data_json" in paths
        assert Path(paths["campaign_summary"]).exists()
        assert Path(paths["campaign_data_json"]).exists()
        # At least one hit report
        hit_keys = [k for k in paths if k.startswith("hit_")]
        assert len(hit_keys) == 1

    def test_campaign_data_json_valid(self, gen, sample_candidate, tmp_path):
        gen.add_candidate(sample_candidate)
        gen.set_audit_trail({"anti_leakage_clean": True, "anti_leakage_violations": [], "input_pdb_hash": "", "stages": []})
        paths = gen.generate_all()
        with open(paths["campaign_data_json"]) as f:
            data = json.load(f)
        assert data["target_name"] == "KRAS_G12C"
        assert data["n_candidates"] == 1

    def test_generate_all_empty_candidates(self, gen, tmp_path):
        gen.set_audit_trail({"anti_leakage_clean": True, "anti_leakage_violations": [], "input_pdb_hash": "", "stages": []})
        paths = gen.generate_all()
        assert "campaign_summary" in paths
        hit_keys = [k for k in paths if k.startswith("hit_")]
        assert len(hit_keys) == 0

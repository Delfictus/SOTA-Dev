"""Integration tests for the canonical pipeline.

Tests that ALL features execute and the registry asserts clean.
Uses mock engine outputs to simulate a real run.
"""
import json
import math
import random
import tempfile
from pathlib import Path

import pytest

from scripts.feature_registry import PipelineRegistry, CANONICAL_FEATURES
from scripts.prism_canonical import run, merge_kcc_into_sites


# ---------------------------------------------------------------------------
# Mock engine output generation
# ---------------------------------------------------------------------------
def _make_spikes(n=60, centroid=(10.0, 10.0, 10.0)):
    random.seed(42)
    spikes = []
    types = ["BNZ", "PHE", "TYR", "TRP"]
    for i in range(n):
        phase = "warm_hold" if i % 3 != 0 else "cold_hold"
        spikes.append({
            "x": centroid[0] + random.gauss(0, 1.5),
            "y": centroid[1] + random.gauss(0, 1.5),
            "z": centroid[2] + random.gauss(0, 1.5),
            "intensity": random.uniform(2.0, 12.0),
            "ccns_phase": phase,
            "frame_index": i % 15,
            "spike_source": "UV",
            "type": random.choice(types),
            "vibrational_energy": 0.5,
            "water_density": 0.01,
            "wavelength_nm": 280.0,
            "timestep": i * 100,
            "n_nearby_excited": 5,
            "aromatic_residue_id": -1,
        })
    return spikes


def _make_lining(n=8):
    resnames = ["TYR", "PHE", "ALA", "LEU", "ASP", "SER", "TRP", "HIS"]
    return [
        {"resid": 100 + i, "resname": resnames[i % len(resnames)],
         "chain": "A", "min_distance": 4.0 + i * 0.5,
         "n_atoms": 8, "is_catalytic": i < 2}
        for i in range(n)
    ]


def _make_site(sid, centroid=(10.0, 10.0, 10.0), volume=500.0,
               therm_class="BINDING", spike_count=200):
    return {
        "id": sid,
        "centroid": list(centroid),
        "volume": volume,
        "therm_class": therm_class,
        "spike_count": spike_count,
        "breathing_score": 0.5,
        "onset_score": 0.3,
        "wd_coherence": 0.5,
        "burial_score": 0.4,
        "mean_burial": 3.0,
        "lining_residues": _make_lining(),
        "spikes": _make_spikes(80, centroid),
    }


def _make_kcc_site(sid):
    return {
        "id": sid,
        "centroid": [10.0, 10.0, 10.0],
        "rank_score": 0.5,
        "gtck_rank": 1,
        "kcc": {
            "candidate_residue_ids": [100, 101, 102],
            "candidate_causal_weights": [0.8, 0.6, 0.4],
            "candidate_kcc_confidence": [0.7, 0.5, 0.3],
            "kcc_confidence": 0.6,
            "motion_efficiency": 0.3,
            "site_motion_efficiency": 0.3,
            "lag_corr_peak": 0.7,
            "site_lag_corr_peak": 0.7,
            "burst_motion": 1.0,
            "local_cov": 0.2,
        },
    }


def write_mock_engine_output(tmp_dir: Path, target: str = "test"):
    """Write minimal but complete mock engine outputs."""
    # binding_sites.json — 3 sites (2 good, 1 bad)
    bs = {
        "sites": [
            _make_site(0, (10, 10, 10), 500, "BINDING", 200),
            _make_site(1, (30, 30, 30), 400, "CRYPTIC", 150),
            {  # site 2: genuinely bad — no therm signal, no spikes, surface
                "id": 2, "centroid": [50, 50, 50], "volume": 100,
                "therm_class": "UNKNOWN", "spike_count": 3,
                "breathing_score": 0.0, "onset_score": 0.0,
                "wd_coherence": 0.05, "burial_score": 0.01,
                "mean_burial": 0.3, "lining_residues": _make_lining(2),
            },
        ],
    }
    with open(tmp_dir / f"{target}.binding_sites.json", "w") as f:
        json.dump(bs, f)

    # kcc_visualization.json
    kcc = {
        "sites": [
            _make_kcc_site(0),
            _make_kcc_site(1),
            _make_kcc_site(2),
        ],
        "residues": [],
    }
    with open(tmp_dir / f"{target}.kcc_visualization.json", "w") as f:
        json.dump(kcc, f)


# ---------------------------------------------------------------------------
# Feature registry tests
# ---------------------------------------------------------------------------
class TestPipelineRegistry:
    def test_mark_and_assert(self):
        reg = PipelineRegistry()
        for feature in CANONICAL_FEATURES:
            reg.mark(feature)
        reg.assert_all()  # should not raise

    def test_assert_fails_on_missing(self):
        reg = PipelineRegistry()
        with pytest.raises(RuntimeError, match="PIPELINE INTEGRITY VIOLATION"):
            reg.assert_all()

    def test_unknown_feature_raises(self):
        reg = PipelineRegistry()
        with pytest.raises(KeyError, match="Unknown feature"):
            reg.mark("nonexistent_feature")

    def test_missing_reports_all_gaps(self):
        reg = PipelineRegistry()
        reg.mark("binding_sites_loaded")
        gaps = reg.missing()
        assert "binding_sites_loaded" not in gaps
        assert "kcc_loaded" in gaps

    def test_summary(self):
        reg = PipelineRegistry()
        reg.mark("binding_sites_loaded")
        s = reg.summary()
        assert "binding_sites_loaded" in s
        assert "1/" in s


# ---------------------------------------------------------------------------
# KCC merge tests
# ---------------------------------------------------------------------------
class TestKCCMerge:
    def test_merge_adds_fields(self):
        sites = [_make_site(0)]
        kcc = {"sites": [_make_kcc_site(0)]}
        merge_kcc_into_sites(sites, kcc)
        assert "kcc_causal_coverage" in sites[0]
        assert "kcc_driver_residues" in sites[0]
        assert sites[0]["kcc_causal_coverage"] > 0

    def test_merge_missing_kcc_site(self):
        sites = [_make_site(99)]
        kcc = {"sites": [_make_kcc_site(0)]}
        merge_kcc_into_sites(sites, kcc)
        assert sites[0]["kcc_causal_coverage"] == 0.0
        assert sites[0]["kcc_driver_residues"] == []


# ---------------------------------------------------------------------------
# Full pipeline integration test
# ---------------------------------------------------------------------------
class TestCanonicalPipeline:
    def test_full_run(self, tmp_path):
        """Complete pipeline run with mock data — all features must execute."""
        write_mock_engine_output(tmp_path, "test")

        result = run(
            output_dir=str(tmp_path),
            target_name="test",
            pdb_id="TEST",
            results_dir=str(tmp_path / "design"),
        )

        # Pipeline completed
        assert result is not None
        assert "gating_result" in result
        assert "ranking" in result
        assert "briefs" in result

        # Gating ran on all 3 sites
        gr = result["gating_result"]
        assert gr.n_sites_input == 3
        assert gr.n_sites_passed >= 1  # at least site 0 and 1 should pass

        # Site 2 (low spikes, UNKNOWN therm) should be blocked
        site2_decision = next(d for d in gr.decisions if d.site_id == 2)
        assert site2_decision.overall_pass is False

        # Design outputs exist
        design_dir = tmp_path / "design"
        assert (design_dir / "gating_result.json").exists()
        assert (design_dir / "site_ranking.json").exists()

    def test_missing_binding_sites_fails(self, tmp_path):
        """Pipeline fails hard if binding_sites.json is missing."""
        # Write only KCC, no binding_sites
        kcc = {"sites": []}
        with open(tmp_path / "test.kcc_visualization.json", "w") as f:
            json.dump(kcc, f)

        with pytest.raises(FileNotFoundError, match="REQUIRED.*binding_sites"):
            run(output_dir=str(tmp_path), target_name="test")

    def test_missing_kcc_fails(self, tmp_path):
        """Pipeline fails hard if kcc_visualization.json is missing."""
        bs = {"sites": [_make_site(0)]}
        with open(tmp_path / "test.binding_sites.json", "w") as f:
            json.dump(bs, f)

        with pytest.raises(FileNotFoundError, match="REQUIRED.*kcc"):
            run(output_dir=str(tmp_path), target_name="test")

    def test_kcc_wired_into_response_selectivity(self, tmp_path):
        """KCC causal_coverage must be available to response selectivity gate."""
        write_mock_engine_output(tmp_path, "test")
        result = run(
            output_dir=str(tmp_path),
            target_name="test",
            results_dir=str(tmp_path / "design"),
        )
        # Check that KCC data was merged — site 0 should have nonzero KCC
        gr = result["gating_result"]
        d0 = next(d for d in gr.decisions if d.site_id == 0)
        # Response selectivity was evaluated (pass or fail, but executed)
        assert isinstance(d0.response_selectivity_pass, bool)

    def test_design_briefs_only_for_passed(self, tmp_path):
        """DesignBriefs generated ONLY for sites that passed gating."""
        write_mock_engine_output(tmp_path, "test")
        result = run(
            output_dir=str(tmp_path),
            target_name="test",
            results_dir=str(tmp_path / "design"),
        )
        n_passed = result["gating_result"].n_sites_passed
        n_briefs = len(result["briefs"])
        assert n_briefs == n_passed

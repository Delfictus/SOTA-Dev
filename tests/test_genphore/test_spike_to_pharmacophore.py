"""Tests for scripts.genphore.spike_to_pharmacophore."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import pytest

from scripts.genphore.spike_to_pharmacophore import (
    _WATER_DENSITY_HBA_THRESHOLD,
    _compute_weighted_centroid,
    _dominant_residue,
    _spikes_to_features,
    convert,
)
from scripts.interfaces import (
    ExclusionSphere,
    PharmacophoreFeature,
    SpikePharmacophore,
    SPIKE_TYPE_TO_FEATURE,
)

FIXTURES = Path(__file__).parent / "fixtures"
MOCK_SPIKES = FIXTURES / "mock_spike_output.json"
MOCK_BINDING = FIXTURES / "mock_binding_sites.json"


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_spike(
    spike_type: str = "BNZ",
    x: float = 10.0,
    y: float = 20.0,
    z: float = 30.0,
    intensity: float = 0.8,
    wavelength_nm: float = 258.0,
    water_density: float = 0.005,
    aromatic_residue_id: int = 42,
) -> dict:
    return {
        "type": spike_type,
        "x": x, "y": y, "z": z,
        "intensity": intensity,
        "wavelength_nm": wavelength_nm,
        "spike_source": "UV",
        "vibrational_energy": 0.1,
        "water_density": water_density,
        "frame_index": 0,
        "timestep": 0,
        "n_nearby_excited": 1,
        "aromatic_residue_id": aromatic_residue_id,
    }


def _minimal_spike_json(spikes, centroid=(10.0, 20.0, 30.0)):
    return {
        "site_id": 0,
        "centroid": list(centroid),
        "n_spikes": len(spikes),
        "lining_cutoff": 8.0,
        "spikes": spikes,
    }


def _write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


# ── _compute_weighted_centroid ───────────────────────────────────────────

class TestComputeWeightedCentroid:
    def test_single_spike(self):
        spikes = [_make_spike(x=5.0, y=10.0, z=15.0, intensity=1.0)]
        cx, cy, cz, mi, mw, mwd = _compute_weighted_centroid(spikes)
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(10.0)
        assert cz == pytest.approx(15.0)
        assert mi == pytest.approx(1.0)

    def test_equal_weight(self):
        spikes = [
            _make_spike(x=0.0, y=0.0, z=0.0, intensity=0.5),
            _make_spike(x=10.0, y=10.0, z=10.0, intensity=0.5),
        ]
        cx, cy, cz, _, _, _ = _compute_weighted_centroid(spikes)
        assert cx == pytest.approx(5.0)
        assert cy == pytest.approx(5.0)
        assert cz == pytest.approx(5.0)

    def test_intensity_weighting(self):
        spikes = [
            _make_spike(x=0.0, y=0.0, z=0.0, intensity=0.9),
            _make_spike(x=10.0, y=10.0, z=10.0, intensity=0.1),
        ]
        cx, cy, cz, _, _, _ = _compute_weighted_centroid(spikes)
        assert cx == pytest.approx(1.0)
        assert cy == pytest.approx(1.0)
        assert cz == pytest.approx(1.0)

    def test_below_threshold_raises(self):
        spikes = [_make_spike(intensity=0.01)]
        with pytest.raises(ValueError, match="No spikes above"):
            _compute_weighted_centroid(spikes)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _compute_weighted_centroid([])


# ── _dominant_residue ────────────────────────────────────────────────────

class TestDominantResidue:
    def test_single_residue(self):
        spikes = [_make_spike(aromatic_residue_id=42, intensity=0.8)]
        rid, label = _dominant_residue(spikes)
        assert rid == 42
        assert "42" in label

    def test_picks_highest_intensity(self):
        spikes = [
            _make_spike(aromatic_residue_id=10, intensity=0.2),
            _make_spike(aromatic_residue_id=10, intensity=0.3),
            _make_spike(aromatic_residue_id=20, intensity=0.4),
        ]
        rid, _ = _dominant_residue(spikes)
        assert rid == 10  # cumulative 0.5 > 0.4


# ── _spikes_to_features ─────────────────────────────────────────────────

class TestSpikesToFeatures:
    def test_bnz_maps_to_ar(self):
        grouped = {"BNZ": [_make_spike("BNZ")]}
        features = _spikes_to_features(grouped)
        assert len(features) == 1
        assert features[0].feature_type == "AR"

    def test_cation_maps_to_ni(self):
        grouped = {"CATION": [_make_spike("CATION", wavelength_nm=0)]}
        features = _spikes_to_features(grouped)
        assert len(features) == 1
        assert features[0].feature_type == "NI"

    def test_anion_maps_to_pi(self):
        grouped = {"ANION": [_make_spike("ANION", wavelength_nm=0)]}
        features = _spikes_to_features(grouped)
        assert features[0].feature_type == "PI"

    def test_tyr_dual_feature_with_high_water(self):
        wd = _WATER_DENSITY_HBA_THRESHOLD + 0.005
        grouped = {"TYR": [_make_spike("TYR", water_density=wd)]}
        features = _spikes_to_features(grouped)
        types = {f.feature_type for f in features}
        assert "HBD" in types
        assert "HBA" in types
        assert len(features) == 2

    def test_tyr_single_feature_low_water(self):
        wd = _WATER_DENSITY_HBA_THRESHOLD - 0.005
        grouped = {"TYR": [_make_spike("TYR", water_density=wd)]}
        features = _spikes_to_features(grouped)
        assert len(features) == 1
        assert features[0].feature_type == "HBD"

    def test_unknown_spike_type_skipped(self):
        grouped = {"WEIRD": [_make_spike("WEIRD")]}
        features = _spikes_to_features(grouped)
        assert len(features) == 0

    def test_sorted_by_intensity(self):
        grouped = {
            "BNZ": [_make_spike("BNZ", intensity=0.3)],
            "TRP": [_make_spike("TRP", intensity=0.9)],
        }
        features = _spikes_to_features(grouped)
        assert features[0].intensity > features[1].intensity

    def test_all_spike_types_covered(self):
        grouped = {
            st: [_make_spike(st)] for st in SPIKE_TYPE_TO_FEATURE
        }
        features = _spikes_to_features(grouped)
        # TYR with default low water_density won't produce HBA
        assert len(features) >= len(SPIKE_TYPE_TO_FEATURE)


# ── convert() integration ───────────────────────────────────────────────

class TestConvert:
    def test_mock_spike_output(self):
        pharm = convert(
            str(MOCK_SPIKES),
            str(MOCK_BINDING),
            target_name="MYC_MAX",
            pdb_id="1NKP",
        )
        assert isinstance(pharm, SpikePharmacophore)
        assert pharm.target_name == "MYC_MAX"
        assert pharm.pdb_id == "1NKP"
        assert pharm.pocket_id == 0
        assert len(pharm.features) >= 2
        assert len(pharm.exclusion_spheres) == 7
        assert len(pharm.pocket_lining_residues) == 7

    def test_feature_types_present(self):
        pharm = convert(str(MOCK_SPIKES), str(MOCK_BINDING))
        types = {f.feature_type for f in pharm.features}
        assert "AR" in types   # from BNZ + TRP
        assert "HBD" in types  # from TYR
        assert "NI" in types   # from CATION
        assert "PI" in types   # from ANION

    def test_tyr_hba_dual_feature(self):
        """TYR spikes in fixture have water_density > threshold → HBA."""
        pharm = convert(str(MOCK_SPIKES), str(MOCK_BINDING))
        types = [f.feature_type for f in pharm.features]
        assert "HBA" in types

    def test_centroid_matches_input(self):
        pharm = convert(str(MOCK_SPIKES))
        assert pharm.pocket_centroid == pytest.approx((59.073, 49.083, 49.577))

    def test_without_binding_sites(self):
        pharm = convert(str(MOCK_SPIKES))
        assert len(pharm.exclusion_spheres) == 0
        assert len(pharm.pocket_lining_residues) == 0
        assert len(pharm.features) >= 2

    def test_serialization_roundtrip(self):
        pharm = convert(str(MOCK_SPIKES), str(MOCK_BINDING), target_name="TEST")
        j = pharm.to_json()
        restored = SpikePharmacophore.from_json(j)
        assert restored.target_name == "TEST"
        assert len(restored.features) == len(pharm.features)
        assert len(restored.exclusion_spheres) == len(pharm.exclusion_spheres)

    def test_to_phoregen_json(self):
        pharm = convert(str(MOCK_SPIKES), str(MOCK_BINDING))
        pj = pharm.to_phoregen_json()
        assert "features" in pj
        assert "exclusions" in pj
        assert "bounds" in pj
        assert len(pj["features"]) == len(pharm.features)

    def test_to_pgmg_posp(self):
        pharm = convert(str(MOCK_SPIKES), str(MOCK_BINDING))
        posp = pharm.to_pgmg_posp()
        assert "FEATURE" in posp
        assert "CENTROID" in posp
        lines = [l for l in posp.splitlines() if l.startswith("FEATURE")]
        assert len(lines) == len(pharm.features)

    def test_run_hash_deterministic(self):
        p1 = convert(str(MOCK_SPIKES), str(MOCK_BINDING))
        p2 = convert(str(MOCK_SPIKES), str(MOCK_BINDING))
        assert p1.prism_run_hash == p2.prism_run_hash

    def test_missing_spike_file_raises(self):
        with pytest.raises(FileNotFoundError):
            convert("/nonexistent/spikes.json")

    def test_missing_binding_file_raises(self):
        with pytest.raises(FileNotFoundError):
            convert(str(MOCK_SPIKES), "/nonexistent/binding.json")

    def test_fewer_than_2_features_raises(self):
        """A spike file with only 1 type and sub-threshold should fail."""
        data = _minimal_spike_json([_make_spike("BNZ", intensity=0.8)])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            tmp = f.name
        try:
            with pytest.raises(ValueError, match="need >=2"):
                convert(tmp)
        finally:
            os.unlink(tmp)

    def test_two_types_passes_minimum(self):
        data = _minimal_spike_json([
            _make_spike("BNZ", intensity=0.8),
            _make_spike("CATION", intensity=0.6),
        ])
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            tmp = f.name
        try:
            pharm = convert(tmp)
            assert len(pharm.features) == 2
        finally:
            os.unlink(tmp)

    def test_pocket_index_out_of_range(self):
        with pytest.raises(IndexError, match="pocket_index"):
            convert(str(MOCK_SPIKES), str(MOCK_BINDING), pocket_index=99)

    def test_exclusion_spheres_placed_around_centroid(self):
        pharm = convert(str(MOCK_SPIKES), str(MOCK_BINDING))
        cx, cy, cz = pharm.pocket_centroid
        for sphere in pharm.exclusion_spheres:
            dist = math.sqrt(
                (sphere.x - cx) ** 2
                + (sphere.y - cy) ** 2
                + (sphere.z - cz) ** 2
            )
            # Spheres should be within lining_cutoff range
            assert dist < 15.0, f"Sphere too far from centroid: {dist:.1f} A"
            assert dist > 0.5, f"Sphere too close to centroid: {dist:.1f} A"

    def test_features_have_correct_types(self):
        pharm = convert(str(MOCK_SPIKES), str(MOCK_BINDING))
        valid_types = {"AR", "PI", "NI", "HBD", "HBA", "HY", "XB"}
        for feat in pharm.features:
            assert feat.feature_type in valid_types

    def test_intensities_in_range(self):
        pharm = convert(str(MOCK_SPIKES), str(MOCK_BINDING))
        for feat in pharm.features:
            assert 0.0 < feat.intensity <= 1.0

    def test_pickle_roundtrip(self):
        pharm = convert(str(MOCK_SPIKES), str(MOCK_BINDING))
        data = pharm.to_pickle()
        restored = SpikePharmacophore.from_pickle(data)
        assert restored.target_name == pharm.target_name
        assert len(restored.features) == len(pharm.features)

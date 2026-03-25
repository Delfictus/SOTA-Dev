"""Tests for PocketProfile dataclass and PocketProfileBuilder."""
import json

import pytest

from scripts.interfaces.pocket_profile import PocketProfile
from scripts.pocket_profile_builder import (
    PocketProfileBuilder,
    _lining_composition,
    _feature_coupling,
    _classify_mw,
    _classify_polarity,
)


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------
def _make_site(
    site_id=0, volume=500.0, burial=0.4, resnames=None,
):
    if resnames is None:
        resnames = ["TYR", "PHE", "ALA", "LEU", "ASP", "SER", "TRP", "HIS"]
    lining = [
        {"resid": i, "resname": r, "chain": "A", "min_distance": 5.0}
        for i, r in enumerate(resnames)
    ]
    return {
        "id": site_id,
        "centroid": [10.0, 10.0, 10.0],
        "volume": volume,
        "burial_score": burial,
        "lining_residues": lining,
    }


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------
class TestPocketProfileDataclass:
    def test_json_round_trip(self):
        pp = PocketProfile(
            site_id=0,
            aromatic_fraction=0.3,
            polar_fraction=0.4,
            hydrophobic_fraction=0.5,
            charged_positive_fraction=0.1,
            charged_negative_fraction=0.1,
            charge_bias=0.0,
            volume=500.0,
            enclosure=0.4,
            n_lining_residues=10,
            feature_coupling=0.8,
            mw_class="lead",
            polarity_class="mixed",
            water_displacement_energy=3.5,
        )
        j = pp.to_json()
        pp2 = PocketProfile.from_json(j)
        assert pp2.mw_class == "lead"
        assert pp2.volume == 500.0


# ---------------------------------------------------------------------------
# Composition tests
# ---------------------------------------------------------------------------
class TestComposition:
    def test_aromatic_fraction(self):
        lining = [
            {"resname": "TRP"},
            {"resname": "TYR"},
            {"resname": "PHE"},
            {"resname": "ALA"},
        ]
        comp = _lining_composition(lining)
        assert comp["aromatic"] == pytest.approx(0.75)

    def test_empty_lining(self):
        comp = _lining_composition([])
        assert comp["aromatic"] == 0.0

    def test_charged_fractions(self):
        lining = [
            {"resname": "ARG"},
            {"resname": "LYS"},
            {"resname": "ASP"},
            {"resname": "GLU"},
        ]
        comp = _lining_composition(lining)
        assert comp["charged_pos"] == pytest.approx(0.5)
        assert comp["charged_neg"] == pytest.approx(0.5)


class TestFeatureCoupling:
    def test_single_type(self):
        """All same type → entropy = 0."""
        features = [{"feature_type": "AR"}] * 5
        assert _feature_coupling(features) == 0.0

    def test_uniform_types(self):
        """All different types → entropy = 1.0 (normalized)."""
        features = [
            {"feature_type": "AR"},
            {"feature_type": "HBD"},
            {"feature_type": "HBA"},
            {"feature_type": "HY"},
        ]
        assert _feature_coupling(features) == pytest.approx(1.0)

    def test_empty(self):
        assert _feature_coupling([]) == 1.0


class TestClassification:
    def test_mw_fragment(self):
        assert _classify_mw(200.0, 3) == "fragment"

    def test_mw_lead(self):
        assert _classify_mw(500.0, 5) == "lead"

    def test_mw_beyond(self):
        assert _classify_mw(1000.0, 10) == "beyond_ro5"

    def test_polarity_hydrophobic(self):
        comp = {"hydrophobic": 0.6, "polar": 0.2}
        assert _classify_polarity(comp) == "hydrophobic"

    def test_polarity_polar(self):
        comp = {"hydrophobic": 0.1, "polar": 0.7}
        assert _classify_polarity(comp) == "polar"

    def test_polarity_mixed(self):
        comp = {"hydrophobic": 0.4, "polar": 0.4}
        assert _classify_polarity(comp) == "mixed"


# ---------------------------------------------------------------------------
# Builder tests
# ---------------------------------------------------------------------------
class TestPocketProfileBuilder:
    def test_compute_basic(self):
        site = _make_site(0)
        builder = PocketProfileBuilder()
        pp = builder.compute(site)

        assert pp.site_id == 0
        assert pp.volume == 500.0
        assert pp.n_lining_residues == 8
        assert 0.0 <= pp.aromatic_fraction <= 1.0
        assert pp.mw_class in ("fragment", "lead", "beyond_ro5")
        assert pp.polarity_class in ("hydrophobic", "mixed", "polar")

    def test_compute_all_hydrophobic(self):
        site = _make_site(
            0, resnames=["ALA", "VAL", "LEU", "ILE", "PHE", "TRP", "MET", "PRO"]
        )
        builder = PocketProfileBuilder()
        pp = builder.compute(site)
        assert pp.hydrophobic_fraction == 1.0
        assert pp.polarity_class == "hydrophobic"

    def test_compute_all_polar(self):
        site = _make_site(
            0, resnames=["ASP", "GLU", "ASN", "GLN", "SER", "THR", "ARG", "LYS"]
        )
        builder = PocketProfileBuilder()
        pp = builder.compute(site)
        assert pp.polar_fraction == 1.0
        assert pp.polarity_class == "polar"

    def test_charge_bias_positive(self):
        site = _make_site(0, resnames=["ARG", "LYS", "ARG", "ALA"])
        builder = PocketProfileBuilder()
        pp = builder.compute(site)
        assert pp.charge_bias > 0

    def test_charge_bias_negative(self):
        site = _make_site(0, resnames=["ASP", "GLU", "ASP", "ALA"])
        builder = PocketProfileBuilder()
        pp = builder.compute(site)
        assert pp.charge_bias < 0

    def test_water_displacement_energy(self):
        site = _make_site(0)
        builder = PocketProfileBuilder()
        pp = builder.compute(site, water_displacement_energy=5.2)
        assert pp.water_displacement_energy == pytest.approx(5.2)

    def test_compute_all(self):
        sites = [_make_site(0), _make_site(1)]
        builder = PocketProfileBuilder()
        results = builder.compute_all(sites, water_energies={0: 2.0})
        assert results[0].water_displacement_energy == pytest.approx(2.0)
        assert results[1].water_displacement_energy == pytest.approx(0.0)

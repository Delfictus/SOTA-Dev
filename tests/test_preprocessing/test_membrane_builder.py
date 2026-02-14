"""Tests for membrane_builder module."""
from __future__ import annotations

import json

import pytest

from scripts.interfaces.membrane_system import MembraneSystem
from scripts.preprocessing.membrane_builder import (
    build_membrane,
    query_opm_orientation,
    _predict_ppm_orientation,
    _simulate_membrane_build,
    _count_atoms,
    DEFAULT_LIPID_COMPOSITION,
    DEFAULT_BOX_XY,
    EQUILIBRATION_PROTOCOL,
)
from scripts.preprocessing.target_classifier import (
    ClassificationResult,
    classify_target,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_classification(pdb_path, cls="membrane", **kwargs):
    return ClassificationResult(
        pdb_path=pdb_path,
        classification=cls,
        confidence="manual",
        method="manual_override",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSolubleSkip:
    """Soluble targets should return None unless force_build."""

    def test_soluble_returns_none(self, kras_pdb):
        cls = _make_classification(kras_pdb, "soluble")
        result = build_membrane(kras_pdb, cls)
        assert result is None

    def test_force_build_on_soluble(self, kras_pdb, tmp_path):
        cls = _make_classification(kras_pdb, "soluble")
        result = build_membrane(
            kras_pdb, cls, output_dir=str(tmp_path), force_build=True,
        )
        assert isinstance(result, MembraneSystem)


class TestMembraneBuild:
    """Test membrane system construction (simulated, no packmol-memgen)."""

    def test_basic_build(self, beta2_pdb, tmp_path):
        cls = _make_classification(beta2_pdb, "membrane")
        result = build_membrane(beta2_pdb, cls, output_dir=str(tmp_path))
        assert isinstance(result, MembraneSystem)

    def test_lipid_composition(self, beta2_pdb, tmp_path):
        cls = _make_classification(beta2_pdb, "membrane")
        result = build_membrane(beta2_pdb, cls, output_dir=str(tmp_path))
        assert result.lipid_composition == {"POPC": 0.7, "CHOL": 0.3}

    def test_custom_lipids(self, beta2_pdb, tmp_path):
        cls = _make_classification(beta2_pdb, "membrane")
        custom = {"DPPC": 0.5, "POPC": 0.3, "CHOL": 0.2}
        result = build_membrane(
            beta2_pdb, cls, output_dir=str(tmp_path), lipids=custom,
        )
        assert result.lipid_composition == custom

    def test_equilibration_protocol(self, beta2_pdb, tmp_path):
        cls = _make_classification(beta2_pdb, "membrane")
        result = build_membrane(beta2_pdb, cls, output_dir=str(tmp_path))
        assert result.equilibration_protocol == EQUILIBRATION_PROTOCOL

    def test_positive_counts(self, beta2_pdb, tmp_path):
        cls = _make_classification(beta2_pdb, "membrane")
        result = build_membrane(beta2_pdb, cls, output_dir=str(tmp_path))
        assert result.n_lipids > 0
        assert result.total_atoms > 0
        assert result.membrane_thickness > 0

    def test_system_size_tuple(self, beta2_pdb, tmp_path):
        cls = _make_classification(beta2_pdb, "membrane")
        result = build_membrane(beta2_pdb, cls, output_dir=str(tmp_path))
        assert len(result.system_size) == 3
        assert all(d > 0 for d in result.system_size)

    def test_custom_box_size(self, beta2_pdb, tmp_path):
        cls = _make_classification(beta2_pdb, "membrane")
        result = build_membrane(
            beta2_pdb, cls, output_dir=str(tmp_path), box_xy=100.0,
        )
        assert result.system_size[0] == 100.0


class TestSerializationRoundTrip:
    """Test that build output serializes through MembraneSystem interface."""

    def test_json_round_trip(self, beta2_pdb, tmp_path):
        cls = _make_classification(beta2_pdb, "membrane")
        ms = build_membrane(beta2_pdb, cls, output_dir=str(tmp_path))
        j = ms.to_json()
        ms2 = MembraneSystem.from_json(j)
        assert ms2.n_lipids == ms.n_lipids
        assert ms2.lipid_composition == ms.lipid_composition

    def test_pickle_round_trip(self, beta2_pdb, tmp_path):
        cls = _make_classification(beta2_pdb, "membrane")
        ms = build_membrane(beta2_pdb, cls, output_dir=str(tmp_path))
        data = ms.to_pickle()
        ms2 = MembraneSystem.from_pickle(data)
        assert ms2.total_atoms == ms.total_atoms

    def test_dict_round_trip(self, beta2_pdb, tmp_path):
        cls = _make_classification(beta2_pdb, "membrane")
        ms = build_membrane(beta2_pdb, cls, output_dir=str(tmp_path))
        d = ms.to_dict()
        ms2 = MembraneSystem.from_dict(d)
        assert ms2.equilibration_protocol == ms.equilibration_protocol


class TestPPMFallback:
    """Test PPM default orientation fallback."""

    def test_default_orientation(self, beta2_pdb):
        orient = _predict_ppm_orientation(beta2_pdb)
        assert orient["source"] == "PPM_default"
        assert orient["tilt_angle"] == 0.0
        assert orient["thickness"] > 0


class TestSimulatedBuild:
    """Test the simulated membrane build function."""

    def test_atom_count_positive(self, beta2_pdb, tmp_path):
        orient = {"thickness": 30.0, "tilt_angle": 0.0}
        meta = _simulate_membrane_build(
            beta2_pdb, str(tmp_path),
            lipids={"POPC": 0.7, "CHOL": 0.3},
            box_xy=80.0,
            orientation=orient,
        )
        assert meta["n_atoms"] > 0
        assert meta["n_lipids"] > 0

    def test_box_dimensions(self, beta2_pdb, tmp_path):
        orient = {"thickness": 30.0}
        meta = _simulate_membrane_build(
            beta2_pdb, str(tmp_path),
            lipids={"POPC": 1.0},
            box_xy=60.0,
            orientation=orient,
        )
        assert meta["box"][0] == 60.0
        assert meta["box"][1] == 60.0


class TestCountAtoms:
    """Test PDB atom counting helper."""

    def test_count(self, beta2_pdb):
        n = _count_atoms(beta2_pdb)
        assert n > 0

    def test_count_kras(self, kras_pdb):
        n = _count_atoms(kras_pdb)
        assert n > 0


class TestCLI:
    """Test CLI entrypoint."""

    def test_cli_soluble(self, kras_pdb, capsys):
        from scripts.preprocessing.membrane_builder import main
        main(["--pdb", kras_pdb, "--offline"])
        out = capsys.readouterr().out
        assert "soluble" in out.lower() or "Soluble" in out

    def test_cli_force(self, kras_pdb, capsys):
        from scripts.preprocessing.membrane_builder import main
        main(["--pdb", kras_pdb, "--offline", "--force"])
        out = capsys.readouterr().out
        # Should output JSON of MembraneSystem
        assert "lipid_composition" in out or "Soluble" in out


class TestIntegrationWithClassifier:
    """End-to-end: classify then build."""

    def test_membrane_target_e2e(self, beta2_pdb, tmp_path):
        cls = classify_target(beta2_pdb, override="membrane")
        ms = build_membrane(beta2_pdb, cls, output_dir=str(tmp_path))
        assert isinstance(ms, MembraneSystem)
        assert ms.n_lipids > 0

    def test_soluble_target_e2e(self, kras_pdb):
        cls = classify_target(kras_pdb, override="soluble")
        result = build_membrane(kras_pdb, cls)
        assert result is None

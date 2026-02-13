"""Tests for MembraneSystem (WT-9 V2)."""
from __future__ import annotations

import json
import pickle

import pytest

from scripts.interfaces.membrane_system import MembraneSystem


class TestMembraneSystem:
    """Tests for MembraneSystem dataclass."""

    def test_construction(self, sample_membrane_system):
        ms = sample_membrane_system
        assert ms.lipid_composition == {"POPC": 0.7, "CHOL": 0.3}
        assert ms.bilayer_method == "packmol_memgen"
        assert ms.n_lipids == 300
        assert ms.membrane_thickness == 35.0
        assert ms.protein_orientation == "OPM"
        assert ms.opm_tilt_angle == 12.5
        assert ms.system_size == (80.0, 80.0, 110.0)
        assert ms.total_atoms == 85000
        assert ms.equilibration_protocol == "CHARMM_GUI_6step"

    def test_dict_round_trip(self, sample_membrane_system):
        d = sample_membrane_system.to_dict()
        ms2 = MembraneSystem.from_dict(d)
        assert ms2.to_dict() == d

    def test_json_round_trip(self, sample_membrane_system):
        j = sample_membrane_system.to_json()
        ms2 = MembraneSystem.from_json(j)
        assert ms2.lipid_composition["POPC"] == 0.7
        assert ms2.system_size == (80.0, 80.0, 110.0)
        assert ms2.bilayer_method == "packmol_memgen"

    def test_pickle_round_trip(self, sample_membrane_system):
        data = sample_membrane_system.to_pickle()
        ms2 = MembraneSystem.from_pickle(data)
        assert ms2.n_lipids == 300
        assert ms2.total_atoms == 85000

    def test_pickle_type_check(self):
        bad = pickle.dumps(3.14)
        with pytest.raises(TypeError, match="Expected MembraneSystem"):
            MembraneSystem.from_pickle(bad)

    def test_system_size_tuple_to_list_in_json(self, sample_membrane_system):
        j = sample_membrane_system.to_json()
        parsed = json.loads(j)
        # JSON: tuple → list
        assert isinstance(parsed["system_size"], list)
        assert parsed["system_size"] == [80.0, 80.0, 110.0]
        # Round-trip restores tuple
        ms2 = MembraneSystem.from_json(j)
        assert isinstance(ms2.system_size, tuple)

    def test_charmm_gui_method(self):
        ms = MembraneSystem(
            lipid_composition={"DMPC": 1.0},
            bilayer_method="charmm_gui",
            n_lipids=200,
            membrane_thickness=30.0,
            protein_orientation="PPM",
            opm_tilt_angle=0.0,
            system_size=(70.0, 70.0, 95.0),
            total_atoms=62000,
            equilibration_protocol="CHARMM_GUI_6step",
        )
        j = ms.to_json()
        ms2 = MembraneSystem.from_json(j)
        assert ms2.bilayer_method == "charmm_gui"
        assert ms2.lipid_composition == {"DMPC": 1.0}

    def test_complex_lipid_composition(self):
        ms = MembraneSystem(
            lipid_composition={"POPC": 0.4, "POPE": 0.25, "POPS": 0.05, "CHOL": 0.3},
            bilayer_method="packmol_memgen",
            n_lipids=400,
            membrane_thickness=38.0,
            protein_orientation="OPM",
            opm_tilt_angle=5.0,
            system_size=(90.0, 90.0, 120.0),
            total_atoms=105000,
            equilibration_protocol="OpenMM_500ps_NVT+NPT",
        )
        d = ms.to_dict()
        ms2 = MembraneSystem.from_dict(d)
        assert len(ms2.lipid_composition) == 4
        assert abs(sum(ms2.lipid_composition.values()) - 1.0) < 1e-6

    def test_manual_orientation(self):
        ms = MembraneSystem(
            lipid_composition={"POPC": 1.0},
            bilayer_method="packmol_memgen",
            n_lipids=250,
            membrane_thickness=34.0,
            protein_orientation="manual",
            opm_tilt_angle=45.0,
            system_size=(75.0, 75.0, 100.0),
            total_atoms=72000,
            equilibration_protocol="OpenMM_500ps_NVT+NPT",
        )
        j = ms.to_json()
        ms2 = MembraneSystem.from_json(j)
        assert ms2.protein_orientation == "manual"
        assert ms2.opm_tilt_angle == 45.0

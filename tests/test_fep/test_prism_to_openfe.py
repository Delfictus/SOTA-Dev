"""Tests for prism_to_openfe.py — PRISM-to-OpenFE bridge."""
from __future__ import annotations

import json
import os
import tempfile

import pytest

from scripts.fep.prism_to_openfe import (
    AlchemicalNetworkSpec,
    ChemicalSystemSpec,
    build_network_from_prism,
    build_rbfe_network,
)
from scripts.interfaces.docking_result import DockingPose, DockingResult


class TestChemicalSystemSpec:
    def test_to_dict(self):
        spec = ChemicalSystemSpec(
            protein_pdb_path="/tmp/protein.pdb",
            ligand_sdf_block="mol_block",
            ligand_name="test_lig",
        )
        d = spec.to_dict()
        assert d["ligand_name"] == "test_lig"
        assert d["forcefield"] == "openff-2.2.0"
        assert d["protein_forcefield"] == "amber/ff14SB.xml"

    def test_default_values(self):
        spec = ChemicalSystemSpec(
            protein_pdb_path="", ligand_sdf_block="", ligand_name="x",
        )
        assert spec.solvent_model == "tip3p"
        assert spec.ion_concentration_m == 0.15


class TestAlchemicalNetworkSpec:
    def test_to_json_roundtrip(self):
        complex_sys = ChemicalSystemSpec(
            protein_pdb_path="/tmp/p.pdb",
            ligand_sdf_block="block",
            ligand_name="lig1",
        )
        solvent_sys = ChemicalSystemSpec(
            protein_pdb_path="",
            ligand_sdf_block="block",
            ligand_name="lig1",
        )
        spec = AlchemicalNetworkSpec(
            compound_id="CPD_001",
            complex_system=complex_sys,
            solvent_system=solvent_sys,
            network_hash="abc123",
        )
        j = spec.to_json()
        recovered = AlchemicalNetworkSpec.from_dict(json.loads(j))
        assert recovered.compound_id == "CPD_001"
        assert recovered.network_hash == "abc123"

    def test_save_and_load(self):
        complex_sys = ChemicalSystemSpec(
            protein_pdb_path="/tmp/p.pdb",
            ligand_sdf_block="block",
            ligand_name="lig1",
        )
        solvent_sys = ChemicalSystemSpec(
            protein_pdb_path="",
            ligand_sdf_block="block",
            ligand_name="lig1",
        )
        spec = AlchemicalNetworkSpec(
            compound_id="CPD_002",
            complex_system=complex_sys,
            solvent_system=solvent_sys,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "network.json")
            spec.save(path)
            loaded = AlchemicalNetworkSpec.load(path)
            assert loaded.compound_id == "CPD_002"


class TestBuildNetworkFromPrism:
    def test_basic_build(self, mock_docking_result, mock_pharmacophore):
        network = build_network_from_prism(
            mock_docking_result, mock_pharmacophore,
        )
        assert network.compound_id == "CPD_TEST_001"
        assert network.network_hash  # non-empty
        assert network.complex_system.protein_pdb_path == mock_docking_result.receptor_pdb

    def test_with_output_dir(self, mock_docking_result, mock_pharmacophore):
        with tempfile.TemporaryDirectory() as tmpdir:
            network = build_network_from_prism(
                mock_docking_result, mock_pharmacophore,
                output_dir=tmpdir,
            )
            expected = os.path.join(tmpdir, "CPD_TEST_001_network.json")
            assert os.path.isfile(expected)

    def test_no_poses_raises(self, mock_pharmacophore):
        dr = DockingResult(
            compound_id="empty",
            smiles="C",
            site_id=0,
            receptor_pdb="p.pdb",
            poses=[],
            best_vina_score=0.0,
            best_cnn_affinity=0.0,
            docking_engine="unidock",
            box_center=(0, 0, 0),
            box_size=(20, 20, 20),
        )
        with pytest.raises(ValueError, match="No docking poses"):
            build_network_from_prism(dr, mock_pharmacophore)

    def test_with_restraint(self, mock_docking_result, mock_pharmacophore):
        restraint = {"distance_p2_l0": 7.0, "angle_p1_p2_l0": 1.57}
        network = build_network_from_prism(
            mock_docking_result, mock_pharmacophore,
            restraint_dict=restraint,
        )
        assert network.restraint_info["distance_p2_l0"] == 7.0


class TestBuildRBFENetwork:
    def test_basic_star_map(self, mock_pharmacophore):
        dr1 = DockingResult(
            compound_id="A",
            smiles="c1ccccc1",
            site_id=0,
            receptor_pdb="p.pdb",
            poses=[DockingPose(1, "mol_a", -7.0)],
            best_vina_score=-7.0,
            best_cnn_affinity=-7.0,
            docking_engine="unidock",
            box_center=(0, 0, 0),
            box_size=(20, 20, 20),
        )
        dr2 = DockingResult(
            compound_id="B",
            smiles="c1ccc(O)cc1",
            site_id=0,
            receptor_pdb="p.pdb",
            poses=[DockingPose(1, "mol_b", -6.5)],
            best_vina_score=-6.5,
            best_cnn_affinity=-6.5,
            docking_engine="unidock",
            box_center=(0, 0, 0),
            box_size=(20, 20, 20),
        )
        result = build_rbfe_network([dr1, dr2], mock_pharmacophore)
        assert result["topology"] == "star"
        assert result["n_compounds"] == 2
        assert result["n_edges"] == 1

    def test_single_compound_raises(self, mock_pharmacophore):
        dr = DockingResult(
            compound_id="only",
            smiles="C",
            site_id=0,
            receptor_pdb="p.pdb",
            poses=[DockingPose(1, "mol", -5.0)],
            best_vina_score=-5.0,
            best_cnn_affinity=-5.0,
            docking_engine="unidock",
            box_center=(0, 0, 0),
            box_size=(20, 20, 20),
        )
        with pytest.raises(ValueError, match="at least 2"):
            build_rbfe_network([dr], mock_pharmacophore)

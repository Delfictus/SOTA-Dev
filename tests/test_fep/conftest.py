"""Shared fixtures for FEP pipeline tests."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.interfaces.docking_result import DockingPose, DockingResult
from scripts.interfaces.fep_result import FEPResult
from scripts.interfaces.spike_pharmacophore import (
    ExclusionSphere,
    PharmacophoreFeature,
    SpikePharmacophore,
)
from scripts.fep.restraint_selector import AtomInfo

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def mock_protein_pdb(fixtures_dir: Path) -> str:
    return str(fixtures_dir / "mock_protein.pdb")


@pytest.fixture
def mock_ligand_sdf(fixtures_dir: Path) -> str:
    return str(fixtures_dir / "mock_ligand.sdf")


@pytest.fixture
def mock_docking_result() -> DockingResult:
    with open(FIXTURES_DIR / "mock_docking_result.json") as f:
        return DockingResult.from_dict(json.load(f))


@pytest.fixture
def mock_pharmacophore() -> SpikePharmacophore:
    return SpikePharmacophore(
        target_name="MOCK_TARGET",
        pdb_id="1ABC",
        pocket_id=0,
        features=[
            PharmacophoreFeature(
                feature_type="AR", x=15.0, y=22.0, z=16.0,
                intensity=0.9, source_spike_type="BNZ",
                source_residue_id=100, source_residue_name="ALA100",
                wavelength_nm=258.0, water_density=0.3,
            ),
            PharmacophoreFeature(
                feature_type="HBD", x=17.0, y=24.0, z=16.0,
                intensity=0.7, source_spike_type="TYR",
                source_residue_id=102, source_residue_name="LEU102",
                wavelength_nm=280.0, water_density=0.5,
            ),
            PharmacophoreFeature(
                feature_type="NI", x=19.0, y=23.0, z=16.0,
                intensity=0.6, source_spike_type="CATION",
                source_residue_id=104, source_residue_name="ASP104",
                wavelength_nm=240.0, water_density=0.4,
            ),
        ],
        exclusion_spheres=[
            ExclusionSphere(x=14.0, y=23.0, z=16.0, radius=1.5, source_atom="CA:ALA100"),
        ],
        pocket_centroid=(16.0, 23.0, 16.0),
        pocket_lining_residues=[100, 101, 102, 103, 104],
        prism_run_hash="abc123def456",
    )


@pytest.fixture
def mock_protein_atoms() -> list[AtomInfo]:
    """Mock protein CA atoms from lining residues."""
    return [
        AtomInfo(2, "CA", "ALA", 100, (11.458, 20.0, 15.0), b_factor=10.0),
        AtomInfo(7, "CA", "GLY", 101, (13.95, 22.81, 15.0), b_factor=11.0),
        AtomInfo(11, "CA", "LEU", 102, (17.5, 24.0, 15.0), b_factor=12.0),
        AtomInfo(19, "CA", "VAL", 103, (19.99, 26.81, 15.0), b_factor=13.0),
        AtomInfo(26, "CA", "ASP", 104, (23.54, 28.0, 15.0), b_factor=14.0),
    ]


@pytest.fixture
def mock_ligand_atoms() -> list[AtomInfo]:
    """Mock ligand heavy atoms."""
    return [
        AtomInfo(0, "C1", "LIG", 1, (15.0, 22.0, 16.0)),
        AtomInfo(1, "C2", "LIG", 1, (16.4, 22.0, 16.0)),
        AtomInfo(2, "C3", "LIG", 1, (17.1, 23.2, 16.0)),
        AtomInfo(3, "C4", "LIG", 1, (16.4, 24.4, 16.0)),
        AtomInfo(4, "C5", "LIG", 1, (15.0, 24.4, 16.0)),
        AtomInfo(5, "C6", "LIG", 1, (14.3, 23.2, 16.0)),
        AtomInfo(6, "N1", "LIG", 1, (17.1, 20.8, 16.0)),
        AtomInfo(7, "O1", "LIG", 1, (17.1, 25.6, 16.0)),
        AtomInfo(8, "F1", "LIG", 1, (12.9, 23.2, 16.0)),
    ]


@pytest.fixture
def spike_positions() -> list[tuple[float, float, float]]:
    """Spike feature positions sorted by descending intensity."""
    return [
        (15.0, 22.0, 16.0),  # AR, 0.9
        (17.0, 24.0, 16.0),  # HBD, 0.7
        (19.0, 23.0, 16.0),  # NI, 0.6
    ]

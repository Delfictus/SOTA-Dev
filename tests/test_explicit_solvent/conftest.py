"""Shared fixtures for WT-6 explicit solvent tests.

All tests mock OpenMM/MDTraj dependencies so they run without GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.interfaces import (
    ExplicitSolventResult,
    HydrationSite,
    SpikePharmacophore,
    WaterMap,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def spike_json_path(tmp_path: Path) -> str:
    """Write the PRISM spike fixture to a temp file and return its path."""
    src = FIXTURES_DIR / "prism_detected_pocket.json"
    data = json.loads(src.read_text())
    out = tmp_path / "kras_spikes.json"
    out.write_text(json.dumps(data))
    return str(out)


@pytest.fixture
def spike_pharmacophore() -> SpikePharmacophore:
    src = FIXTURES_DIR / "prism_detected_pocket.json"
    return SpikePharmacophore.from_json(src.read_text())


@pytest.fixture
def pocket_centroid() -> tuple:
    return (13.0, -2.5, 8.5)


@pytest.fixture
def lining_residues() -> list:
    return [10, 11, 12, 30, 32, 34, 59, 60, 61, 142, 145]


@pytest.fixture
def mock_pdb_path(tmp_path: Path) -> str:
    """Write a minimal valid PDB to a temp file."""
    pdb_content = (
        "HEADER    MOCK KRAS G12C\n"
        "ATOM      1  N   MET A   1      10.000  12.000   8.000  1.00  0.00           N\n"
        "ATOM      2  CA  MET A   1      10.500  12.500   8.300  1.00  0.00           C\n"
        "ATOM      3  C   MET A   1      11.000  13.000   8.600  1.00  0.00           C\n"
        "ATOM      4  O   MET A   1      11.500  13.500   8.900  1.00  0.00           O\n"
        "ATOM      5  N   THR A   2      11.200  12.800   7.500  1.00  0.00           N\n"
        "ATOM      6  CA  THR A   2      12.000  13.200   7.100  1.00  0.00           C\n"
        "ATOM      7  C   THR A   2      12.500  13.700   7.400  1.00  0.00           C\n"
        "ATOM      8  O   THR A   2      13.000  14.200   7.700  1.00  0.00           O\n"
        "END\n"
    )
    out = tmp_path / "kras.pdb"
    out.write_text(pdb_content)
    return str(out)


@pytest.fixture
def sample_stable_result() -> ExplicitSolventResult:
    return ExplicitSolventResult(
        pocket_id=0,
        simulation_time_ns=10.0,
        water_model="TIP3P",
        force_field="ff14SB",
        pocket_stable=True,
        pocket_rmsd_mean=1.2,
        pocket_rmsd_std=0.3,
        pocket_volume_mean=450.0,
        pocket_volume_std=45.0,
        n_structural_waters=3,
        trajectory_path="/mock/production.dcd",
        snapshot_frames=[0, 100, 200, 300, 400, 500, 600, 700, 800, 900],
    )


@pytest.fixture
def sample_collapsed_result() -> ExplicitSolventResult:
    return ExplicitSolventResult(
        pocket_id=1,
        simulation_time_ns=10.0,
        water_model="OPC",
        force_field="ff14SB",
        pocket_stable=False,
        pocket_rmsd_mean=4.5,
        pocket_rmsd_std=1.8,
        pocket_volume_mean=120.0,
        pocket_volume_std=80.0,
        n_structural_waters=0,
        trajectory_path="/mock/collapsed.dcd",
        snapshot_frames=[],
    )


@pytest.fixture
def sample_happy_site() -> HydrationSite:
    return HydrationSite(
        x=11.2, y=-1.8, z=9.1,
        occupancy=0.92,
        delta_g_transfer=-2.1,
        entropy_contribution=-1.5,
        enthalpy_contribution=-0.6,
        n_hbonds_mean=2.8,
        classification="CONSERVED_HAPPY",
        displaceable=False,
    )


@pytest.fixture
def sample_unhappy_site() -> HydrationSite:
    return HydrationSite(
        x=14.3, y=-3.0, z=7.5,
        occupancy=0.75,
        delta_g_transfer=1.8,
        entropy_contribution=0.5,
        enthalpy_contribution=1.3,
        n_hbonds_mean=1.1,
        classification="CONSERVED_UNHAPPY",
        displaceable=True,
    )


@pytest.fixture
def sample_water_map(sample_happy_site, sample_unhappy_site) -> WaterMap:
    return WaterMap(
        pocket_id=0,
        hydration_sites=[sample_happy_site, sample_unhappy_site],
        n_displaceable=1,
        max_displacement_energy=1.8,
        total_displacement_energy=1.8,
        grid_resolution=0.5,
        analysis_frames=1000,
    )

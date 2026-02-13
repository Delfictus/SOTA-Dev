"""Tests for structural_water_finder.py — conserved water identification.

Tests cover:
    - StructuralWater dataclass and PDB formatting
    - PDB output generation
    - Conversion from HydrationSite objects
    - Filtering logic (occupancy + H-bond thresholds)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.interfaces import HydrationSite
from scripts.explicit_solvent.structural_water_finder import (
    DEFAULT_HBOND_THRESHOLD,
    DEFAULT_OCCUPANCY_THRESHOLD,
    StructuralWater,
    structural_waters_from_hydration_sites,
    write_structural_waters_pdb,
)


# ═══════════════════════════════════════════════════════════════════════════
#  StructuralWater dataclass
# ═══════════════════════════════════════════════════════════════════════════

class TestStructuralWater:
    def test_construction(self):
        sw = StructuralWater(
            x=11.2, y=-1.8, z=9.1,
            occupancy=0.92,
            mean_hbonds_to_protein=2.8,
            coordinating_residues=["ASP57", "THR35"],
            b_factor_estimate=12.5,
        )
        assert sw.occupancy == 0.92
        assert sw.mean_hbonds_to_protein == 2.8
        assert len(sw.coordinating_residues) == 2

    def test_distance_to(self):
        sw1 = StructuralWater(
            x=0, y=0, z=0, occupancy=0.9,
            mean_hbonds_to_protein=2, coordinating_residues=[],
            b_factor_estimate=10,
        )
        sw2 = StructuralWater(
            x=3, y=4, z=0, occupancy=0.9,
            mean_hbonds_to_protein=2, coordinating_residues=[],
            b_factor_estimate=10,
        )
        assert sw1.distance_to(sw2) == pytest.approx(5.0)

    def test_to_dict(self):
        sw = StructuralWater(
            x=1.0, y=2.0, z=3.0, occupancy=0.85,
            mean_hbonds_to_protein=2.5, coordinating_residues=["ALA10"],
            b_factor_estimate=15.0,
        )
        d = sw.to_dict()
        assert d["x"] == 1.0
        assert d["occupancy"] == 0.85
        assert d["coordinating_residues"] == ["ALA10"]

    def test_pdb_line_format(self):
        sw = StructuralWater(
            x=11.200, y=-1.800, z=9.100, occupancy=0.92,
            mean_hbonds_to_protein=2.8, coordinating_residues=[],
            b_factor_estimate=12.50,
        )
        line = sw.to_pdb_line(atom_index=10001, residue_number=1001)
        assert line.startswith("HETATM")
        assert "HOH" in line
        assert "10001" in line
        assert "O" in line

    def test_pdb_line_coordinates(self):
        sw = StructuralWater(
            x=11.200, y=-1.800, z=9.100, occupancy=0.92,
            mean_hbonds_to_protein=2.8, coordinating_residues=[],
            b_factor_estimate=12.50,
        )
        line = sw.to_pdb_line(10001, 1001)
        # Check that coordinates appear in the line
        assert "11.200" in line
        assert "-1.800" in line
        assert "9.100" in line


# ═══════════════════════════════════════════════════════════════════════════
#  Threshold constants
# ═══════════════════════════════════════════════════════════════════════════

class TestThresholds:
    def test_occupancy_threshold(self):
        assert DEFAULT_OCCUPANCY_THRESHOLD == 0.80

    def test_hbond_threshold(self):
        assert DEFAULT_HBOND_THRESHOLD == 2


# ═══════════════════════════════════════════════════════════════════════════
#  PDB file output
# ═══════════════════════════════════════════════════════════════════════════

class TestWriteStructuralWatersPDB:
    def test_write_creates_file(self, tmp_path):
        waters = [
            StructuralWater(
                x=11.2, y=-1.8, z=9.1, occupancy=0.92,
                mean_hbonds_to_protein=2.8,
                coordinating_residues=["ASP57"],
                b_factor_estimate=12.5,
            ),
            StructuralWater(
                x=9.8, y=-5.2, z=10.0, occupancy=0.95,
                mean_hbonds_to_protein=3.1,
                coordinating_residues=["GLU62", "THR35"],
                b_factor_estimate=10.0,
            ),
        ]
        out_path = tmp_path / "structural_waters.pdb"
        result = write_structural_waters_pdb(waters, str(out_path))
        assert Path(result).exists()

        content = Path(result).read_text()
        assert "REMARK" in content
        assert "HETATM" in content
        assert "HOH" in content
        assert "END" in content

    def test_write_correct_count(self, tmp_path):
        waters = [
            StructuralWater(
                x=1, y=2, z=3, occupancy=0.9,
                mean_hbonds_to_protein=2, coordinating_residues=[],
                b_factor_estimate=15,
            )
            for _ in range(5)
        ]
        out_path = tmp_path / "waters.pdb"
        write_structural_waters_pdb(waters, str(out_path))
        content = Path(out_path).read_text()
        hetatm_lines = [l for l in content.splitlines() if l.startswith("HETATM")]
        assert len(hetatm_lines) == 5

    def test_write_empty_list(self, tmp_path):
        out_path = tmp_path / "empty_waters.pdb"
        write_structural_waters_pdb([], str(out_path))
        content = Path(out_path).read_text()
        assert "END" in content
        assert "0 conserved" in content


# ═══════════════════════════════════════════════════════════════════════════
#  Conversion from HydrationSite
# ═══════════════════════════════════════════════════════════════════════════

class TestFromHydrationSites:
    def test_happy_water_passes(self):
        """CONSERVED_HAPPY with high occupancy and H-bonds → structural."""
        sites = [
            HydrationSite(
                x=11.2, y=-1.8, z=9.1,
                occupancy=0.92,
                delta_g_transfer=-2.1,
                entropy_contribution=-1.5,
                enthalpy_contribution=-0.6,
                n_hbonds_mean=2.8,
                classification="CONSERVED_HAPPY",
                displaceable=False,
            ),
        ]
        waters = structural_waters_from_hydration_sites(sites)
        assert len(waters) == 1
        assert waters[0].occupancy == 0.92
        assert waters[0].mean_hbonds_to_protein == 2.8

    def test_unhappy_water_excluded(self):
        """CONSERVED_UNHAPPY → not a structural water."""
        sites = [
            HydrationSite(
                x=14.3, y=-3.0, z=7.5,
                occupancy=0.85,
                delta_g_transfer=1.8,
                entropy_contribution=0.5,
                enthalpy_contribution=1.3,
                n_hbonds_mean=1.1,
                classification="CONSERVED_UNHAPPY",
                displaceable=True,
            ),
        ]
        waters = structural_waters_from_hydration_sites(sites)
        assert len(waters) == 0

    def test_low_occupancy_excluded(self):
        """Happy but low occupancy → not structural."""
        sites = [
            HydrationSite(
                x=10, y=10, z=10,
                occupancy=0.50,
                delta_g_transfer=-2.0,
                entropy_contribution=-1.0,
                enthalpy_contribution=-1.0,
                n_hbonds_mean=3.0,
                classification="CONSERVED_HAPPY",
                displaceable=False,
            ),
        ]
        waters = structural_waters_from_hydration_sites(sites)
        assert len(waters) == 0  # Below 0.80 occupancy

    def test_low_hbonds_excluded(self):
        """Happy, high occupancy, but too few H-bonds → excluded."""
        sites = [
            HydrationSite(
                x=10, y=10, z=10,
                occupancy=0.95,
                delta_g_transfer=-2.0,
                entropy_contribution=-1.0,
                enthalpy_contribution=-1.0,
                n_hbonds_mean=1.5,
                classification="CONSERVED_HAPPY",
                displaceable=False,
            ),
        ]
        waters = structural_waters_from_hydration_sites(sites)
        assert len(waters) == 0  # Below 2 H-bonds

    def test_mixed_sites(self):
        """Mix of happy, unhappy, and bulk → only happy qualifiers pass."""
        sites = [
            HydrationSite(x=1, y=1, z=1, occupancy=0.92,
                         delta_g_transfer=-2.1, entropy_contribution=-1.5,
                         enthalpy_contribution=-0.6, n_hbonds_mean=2.8,
                         classification="CONSERVED_HAPPY", displaceable=False),
            HydrationSite(x=5, y=5, z=5, occupancy=0.85,
                         delta_g_transfer=1.8, entropy_contribution=0.5,
                         enthalpy_contribution=1.3, n_hbonds_mean=1.1,
                         classification="CONSERVED_UNHAPPY", displaceable=True),
            HydrationSite(x=9, y=9, z=9, occupancy=0.20,
                         delta_g_transfer=0.1, entropy_contribution=0.0,
                         enthalpy_contribution=0.1, n_hbonds_mean=0.3,
                         classification="BULK", displaceable=False),
            HydrationSite(x=2, y=2, z=2, occupancy=0.88,
                         delta_g_transfer=-1.5, entropy_contribution=-1.0,
                         enthalpy_contribution=-0.5, n_hbonds_mean=2.2,
                         classification="CONSERVED_HAPPY", displaceable=False),
        ]
        waters = structural_waters_from_hydration_sites(sites)
        assert len(waters) == 2
        # Both should be the happy sites
        xs = {w.x for w in waters}
        assert 1.0 in xs
        assert 2.0 in xs

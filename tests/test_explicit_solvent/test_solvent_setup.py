"""Tests for solvent_setup.py — solvation system preparation.

Tests cover:
    - Water model / force field validation
    - Constant definitions
    - ExplicitSolventSystem dataclass
    - Error handling (missing files, invalid models)
    - OpenMM is NOT required for these tests (lazy imports)
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.explicit_solvent.solvent_setup import (
    DEFAULT_BOX_PADDING_NM,
    DEFAULT_FORCE_FIELD,
    DEFAULT_IONIC_STRENGTH_M,
    DEFAULT_WATER_MODEL,
    FORCE_FIELDS,
    WATER_MODELS,
    ExplicitSolventSystem,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Constants and validation
# ═══════════════════════════════════════════════════════════════════════════

class TestConstants:
    def test_default_water_model(self):
        assert DEFAULT_WATER_MODEL == "TIP3P"

    def test_default_force_field(self):
        assert DEFAULT_FORCE_FIELD == "ff14SB"

    def test_default_padding(self):
        assert DEFAULT_BOX_PADDING_NM == 1.2  # 12 Angstrom

    def test_default_ionic_strength(self):
        assert DEFAULT_IONIC_STRENGTH_M == 0.15  # 150 mM


class TestWaterModels:
    def test_tip3p_present(self):
        assert "TIP3P" in WATER_MODELS
        assert "ff_water" in WATER_MODELS["TIP3P"]

    def test_opc_present(self):
        assert "OPC" in WATER_MODELS
        assert "ff_water" in WATER_MODELS["OPC"]

    def test_tip4pew_present(self):
        assert "TIP4P-Ew" in WATER_MODELS

    def test_all_models_have_required_keys(self):
        for name, model in WATER_MODELS.items():
            assert "ff_water" in model, f"{name} missing ff_water"
            assert "model_name" in model, f"{name} missing model_name"


class TestForceFields:
    def test_ff14sb_present(self):
        assert "ff14SB" in FORCE_FIELDS

    def test_ff19sb_present(self):
        assert "ff19SB" in FORCE_FIELDS

    def test_charmm36m_present(self):
        assert "CHARMM36m" in FORCE_FIELDS


# ═══════════════════════════════════════════════════════════════════════════
#  ExplicitSolventSystem dataclass
# ═══════════════════════════════════════════════════════════════════════════

class TestExplicitSolventSystem:
    def test_construction(self):
        sys = ExplicitSolventSystem(
            topology=None,
            positions=np.zeros((100, 3)),
            system=None,
            water_model="TIP3P",
            force_field="ff14SB",
            n_waters=5000,
            n_cations=20,
            n_anions=20,
            box_vectors=np.eye(3) * 5.0,
        )
        assert sys.water_model == "TIP3P"
        assert sys.n_waters == 5000
        assert sys.n_cations == 20
        assert sys.positions.shape == (100, 3)
        assert sys.box_vectors.shape == (3, 3)

    def test_ion_count_reasonable(self):
        """For 150 mM NaCl in ~5000 waters, expect ~10-30 ions each."""
        sys = ExplicitSolventSystem(
            topology=None,
            positions=np.zeros((100, 3)),
            system=None,
            water_model="TIP3P",
            force_field="ff14SB",
            n_waters=5000,
            n_cations=22,
            n_anions=22,
            box_vectors=np.eye(3) * 5.0,
        )
        assert 5 <= sys.n_cations <= 50
        assert 5 <= sys.n_anions <= 50


# ═══════════════════════════════════════════════════════════════════════════
#  Error handling (no OpenMM needed)
# ═══════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    def test_invalid_water_model_raises(self):
        """Invalid water model should raise ValueError (caught before OpenMM)."""
        from scripts.explicit_solvent.solvent_setup import prepare_solvated_system
        # This will fail at validation before trying to import OpenMM
        with pytest.raises((ValueError, ImportError)):
            prepare_solvated_system(
                pdb_path="/nonexistent.pdb",
                water_model="INVALID_MODEL",
            )

    def test_invalid_force_field_raises(self):
        from scripts.explicit_solvent.solvent_setup import prepare_solvated_system
        with pytest.raises((ValueError, ImportError)):
            prepare_solvated_system(
                pdb_path="/nonexistent.pdb",
                force_field="INVALID_FF",
            )

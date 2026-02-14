"""Shared fixtures for WT-5 preprocessing tests."""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def beta2_pdb() -> str:
    """Path to a minimal beta2-adrenergic receptor PDB (membrane protein)."""
    return str(FIXTURES_DIR / "beta2_adrenergic.pdb")


@pytest.fixture
def kras_pdb() -> str:
    """Path to a minimal KRAS PDB (soluble protein)."""
    return str(FIXTURES_DIR / "kras_soluble.pdb")


@pytest.fixture
def ligand_sdf() -> str:
    """Path to aspirin SDF with ionizable carboxylic acid."""
    return str(FIXTURES_DIR / "ligand_with_ionizable_groups.sdf")

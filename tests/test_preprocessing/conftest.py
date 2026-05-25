"""Shared fixtures for WT-5 preprocessing tests."""
from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def beta2_pdb() -> str:
    """Path to a minimal beta2-adrenergic receptor PDB (membrane protein)."""
    path = FIXTURES_DIR / "beta2_adrenergic.pdb"
    if not path.is_file():
        pytest.skip(f"requires preprocessing fixture: {path}")
    return str(path)


@pytest.fixture
def kras_pdb() -> str:
    """Path to a minimal KRAS PDB (soluble protein)."""
    path = FIXTURES_DIR / "kras_soluble.pdb"
    if not path.is_file():
        pytest.skip(f"requires preprocessing fixture: {path}")
    return str(path)


@pytest.fixture
def ligand_sdf() -> str:
    """Path to aspirin SDF with ionizable carboxylic acid."""
    path = FIXTURES_DIR / "ligand_with_ionizable_groups.sdf"
    if not path.is_file():
        pytest.skip(f"requires preprocessing fixture: {path}")
    return str(path)

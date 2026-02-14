"""Shared fixtures for WT-8 viewer tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.interfaces.viewer_payload import ViewerPayload

_FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_payload_path() -> Path:
    return _FIXTURES_DIR / "mock_viewer_payload.json"


@pytest.fixture
def mock_payload_dict(mock_payload_path: Path) -> dict:
    return json.loads(mock_payload_path.read_text(encoding="utf-8"))


@pytest.fixture
def mock_payload(mock_payload_dict: dict) -> ViewerPayload:
    return ViewerPayload.from_dict(mock_payload_dict)


@pytest.fixture
def minimal_payload() -> ViewerPayload:
    """Bare-minimum payload with only required fields."""
    return ViewerPayload(
        target_name="MINIMAL_TEST",
        pdb_structure="ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.00\nEND\n",
        pocket_surfaces=[],
        spike_positions=[],
        water_map_sites=[],
        ligand_poses=[],
        lining_residues=[],
    )


@pytest.fixture
def empty_payload() -> ViewerPayload:
    """Payload with empty strings / lists everywhere."""
    return ViewerPayload(
        target_name="",
        pdb_structure="",
        pocket_surfaces=[],
        spike_positions=[],
        water_map_sites=[],
        ligand_poses=[],
        lining_residues=[],
    )

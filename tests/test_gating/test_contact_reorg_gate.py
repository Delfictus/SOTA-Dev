"""Tests for the Contact Reorganization gate module."""
import json
import math
import os
import tempfile

import pytest

from scripts.interfaces.contact_reorg_result import ContactReorgResult
from scripts.contact_reorg_gate import (
    ContactReorgGate,
    ContactReorgThresholds,
    compute_contacts,
    local_contacts,
    parse_trajectory_ca,
)


# ---------------------------------------------------------------------------
# Fixtures: mock data
# ---------------------------------------------------------------------------
def _make_site(site_id: int = 0, centroid=(10.0, 10.0, 10.0)):
    return {"id": site_id, "centroid": list(centroid)}


def _make_multi_model_pdb(n_frames: int = 6, drift: float = 0.5) -> str:
    """Generate a minimal multi-model PDB with CA atoms that drift over frames.

    Creates a small 4-residue chain where residues near (10,10,10) move
    slightly each frame (simulating local contact change).
    """
    lines = []
    base_coords = [
        (10.0, 10.0, 10.0),  # near centroid
        (10.0, 14.0, 10.0),  # 4A away
        (10.0, 10.0, 16.0),  # 6A away (edge of contact)
        (25.0, 25.0, 25.0),  # far away (global)
        (25.0, 30.0, 25.0),  # far, 5A from above
    ]
    for frame in range(n_frames):
        lines.append(f"MODEL     {frame + 1}")
        for ri, (bx, by, bz) in enumerate(base_coords):
            # Drift residues near centroid
            if ri < 3:
                x = bx + drift * frame
                y = by - drift * frame * 0.3
                z = bz
            else:
                x, y, z = bx, by, bz
            atom_line = (
                f"ATOM  {ri+1:5d}  CA  ALA A{ri+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
            )
            lines.append(atom_line)
        lines.append("ENDMDL")
    return "\n".join(lines)


@pytest.fixture
def traj_pdb(tmp_path):
    pdb_content = _make_multi_model_pdb(n_frames=8, drift=0.8)
    path = tmp_path / "traj.pdb"
    path.write_text(pdb_content)
    return str(path)


@pytest.fixture
def sparse_traj_pdb(tmp_path):
    pdb_content = _make_multi_model_pdb(n_frames=2, drift=0.1)
    path = tmp_path / "sparse_traj.pdb"
    path.write_text(pdb_content)
    return str(path)


# ---------------------------------------------------------------------------
# Dataclass serialization tests
# ---------------------------------------------------------------------------
class TestContactReorgResultSerialization:
    def test_json_round_trip(self):
        cr = ContactReorgResult(
            site_id=3,
            contact_change_density=2.5,
            localization_ratio=0.12,
            persistence=0.75,
            boundary_growth=0.05,
            n_frames_analyzed=9,
            gate_pass=True,
            gate_reason="pass",
        )
        j = cr.to_json()
        cr2 = ContactReorgResult.from_json(j)
        assert cr2.site_id == 3
        assert cr2.localization_ratio == 0.12
        assert cr2.gate_pass is True

    def test_pickle_round_trip(self):
        cr = ContactReorgResult(
            site_id=1,
            contact_change_density=0.0,
            localization_ratio=0.0,
            persistence=0.0,
            boundary_growth=0.0,
            n_frames_analyzed=0,
            gate_pass=False,
            gate_reason="blocked",
        )
        data = cr.to_pickle()
        cr2 = ContactReorgResult.from_pickle(data)
        assert cr2.site_id == 1
        assert cr2.gate_pass is False

    def test_pickle_type_check(self):
        with pytest.raises(Exception):
            ContactReorgResult.from_pickle(b"not a pickle")


# ---------------------------------------------------------------------------
# Trajectory parsing tests
# ---------------------------------------------------------------------------
class TestTrajectoryParsing:
    def test_parse_multi_model(self, traj_pdb):
        frames = parse_trajectory_ca(traj_pdb, max_frames=5)
        assert len(frames) == 5
        # Each frame should have 5 CA atoms
        assert len(frames[0]) == 5

    def test_max_frames_limit(self, traj_pdb):
        frames = parse_trajectory_ca(traj_pdb, max_frames=3)
        assert len(frames) == 3


# ---------------------------------------------------------------------------
# Contact computation tests
# ---------------------------------------------------------------------------
class TestContactComputation:
    def test_contacts_within_cutoff(self):
        cas = {
            "A:1": (0.0, 0.0, 0.0),
            "A:2": (1.0, 0.0, 0.0),  # 1A — adjacent, skip
            "A:3": (3.0, 0.0, 0.0),  # 3A from A:1
            "A:4": (7.0, 0.0, 0.0),  # 7A from A:1 — outside cutoff
        }
        contacts = compute_contacts(cas, cutoff=6.0)
        # A:1-A:3 = 3A (contact, non-adjacent)
        assert ("A:1", "A:3") in contacts
        # A:1-A:4 = 7A (no contact)
        assert ("A:1", "A:4") not in contacts

    def test_local_contacts(self):
        cas = {
            "A:1": (10.0, 10.0, 10.0),
            "A:3": (13.0, 10.0, 10.0),
            "A:5": (50.0, 50.0, 50.0),
        }
        contacts = {("A:1", "A:3"), ("A:1", "A:5")}
        centroid = (10.0, 10.0, 10.0)
        local = local_contacts(contacts, cas, centroid, radius=12.0)
        # A:1 is at centroid, A:3 is 3A away — both local
        assert ("A:1", "A:3") in local
        # A:5 is 69A away, A:1 is local but A:5 is not — still counts
        # because at least one residue (A:1) is near centroid
        assert ("A:1", "A:5") in local


# ---------------------------------------------------------------------------
# Gate logic tests
# ---------------------------------------------------------------------------
class TestContactReorgGate:
    def test_gate_pass_with_trajectory(self, traj_pdb):
        site = _make_site(site_id=0, centroid=(10.0, 10.0, 10.0))
        thresholds = ContactReorgThresholds(
            min_localization_ratio=0.01,
            min_contact_change_density=0.1,
        )
        gate = ContactReorgGate(thresholds)
        frames = parse_trajectory_ca(traj_pdb, max_frames=8)
        result = gate.evaluate(site, frames)
        assert isinstance(result, ContactReorgResult)
        assert result.n_frames_analyzed == 7  # 8 frames - 1 reference

    def test_gate_bypassed_without_trajectory(self):
        sites = [_make_site(0), _make_site(1)]
        gate = ContactReorgGate()
        results = gate.evaluate_all(sites, trajectory_path=None)
        assert len(results) == 2
        for r in results.values():
            assert r.gate_pass is True
            assert "bypassed" in r.gate_reason

    def test_gate_bypassed_insufficient_frames(self, sparse_traj_pdb):
        site = _make_site(0)
        gate = ContactReorgGate()
        frames = parse_trajectory_ca(sparse_traj_pdb, max_frames=2)
        result = gate.evaluate(site, frames)
        assert result.gate_pass is True
        assert "insufficient_frames" in result.gate_reason

    def test_gate_blocks_zero_signal(self):
        """Site with no local contact change should be blocked."""
        site = _make_site(0, centroid=(100.0, 100.0, 100.0))
        # Make a trajectory where contacts change but NOT near (100,100,100)
        cas_template = {
            "A:1": (0.0, 0.0, 0.0),
            "A:3": (4.0, 0.0, 0.0),
            "A:5": (8.0, 0.0, 0.0),
        }
        frames = [cas_template.copy() for _ in range(5)]
        thresholds = ContactReorgThresholds(
            min_localization_ratio=0.01,
            min_contact_change_density=0.1,
        )
        gate = ContactReorgGate(thresholds)
        result = gate.evaluate(site, frames)
        # No local changes near (100,100,100)
        assert result.localization_ratio == 0.0
        assert result.contact_change_density == 0.0
        assert result.gate_pass is False

    def test_evaluate_all_returns_dict(self, traj_pdb):
        sites = [_make_site(0), _make_site(1, centroid=(25.0, 25.0, 25.0))]
        gate = ContactReorgGate()
        results = gate.evaluate_all(sites, trajectory_path=traj_pdb)
        assert isinstance(results, dict)
        assert 0 in results
        assert 1 in results

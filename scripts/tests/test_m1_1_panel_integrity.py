"""M1.1 panel integrity regression tests.

Test 1 — cross_chain_ligand_fallback
  Asserts that the `_parse_pdb_hetatm` any-chain fallback lands the
  ligand when the ligand resname exists only in a non-protein chain.
  Specifically validates the m1_2nvp case (protein chain A, ligand Z4Y
  in chain B) which returned GROUND_TRUTH_INVALID before the fix.

Test 2 — strict_dcc_panel_v1_membership_hash
  Asserts that the SHA-256 of strict_dcc_panel_v1.json membership has
  not drifted. Membership hash is computed over the ordered list of
  (target_key, apo, holo, ligand, source) tuples only — bin transitions
  DO NOT modify the hash. Adding/removing/substituting targets WILL.
  If panel changes, a strict_dcc_panel_v2.json must be emitted with an
  explicit delta log (tested by checking v2 file absence when hash drifts).
"""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
QUARANTINE = REPO / "scripts/quarantine"
PANEL_V1 = QUARANTINE / "strict_dcc_panel_v1.json"
PANEL_V2 = QUARANTINE / "strict_dcc_panel_v2.json"


EXPECTED_V1_HASH = "332ac3eb0627fd1f837ce4e6381de80e6581f7cf642e81fc854167d35d354f44"


def _membership_hash(panel_path: Path) -> str:
    d = json.loads(panel_path.read_text())
    members = sorted(
        [(r["target_key"], r["apo"], r["holo"], r["ligand"], r["source"]) for r in d["panel"]]
    )
    payload = json.dumps(members, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@pytest.fixture(scope="module")
def v1_hash() -> str:
    assert PANEL_V1.exists(), "strict_dcc_panel_v1.json must exist"
    return _membership_hash(PANEL_V1)


def test_cross_chain_ligand_fallback_m1_2nvp():
    """Regression: ligand Z4Y is in chain B of 3qt9 while the protein is chain A.
    The stage-4 any-chain fallback must yield a non-empty ligand coordinate set."""
    sys.path.insert(0, str(QUARANTINE))
    from run_stages import _parse_pdb_hetatm  # type: ignore

    holo = Path("/mnt/storage/prism-outputs/m1-strict-dcc-panel/m1_2nvp/artifacts/1_download/3qt9.pdb")
    if not holo.exists():
        pytest.skip("m1_2nvp holo not downloaded; fixture unavailable")

    # Protein chain is A; without fallback, lookup in chain A must be empty.
    only_a = _parse_pdb_hetatm(holo, "Z4Y", "A", fallback_any_chain=False)
    assert only_a.size == 0, "Z4Y should NOT be found in chain A of 3qt9"

    # With fallback, lookup must return the 12 heavy atoms of Z4Y in chain B.
    with_fallback = _parse_pdb_hetatm(holo, "Z4Y", "A", fallback_any_chain=True)
    assert with_fallback.shape[0] == 12, \
        f"expected 12 Z4Y atoms with any-chain fallback, got {with_fallback.shape[0]}"

    # Ground truth for the target should report a valid superposition + ligand.
    gt = Path("/mnt/storage/prism-outputs/m1-strict-dcc-panel/m1_2nvp/artifacts/4_ground_truth/2nvp_ground_truth.json")
    if gt.exists():
        g = json.loads(gt.read_text())
        assert "error" not in g or g.get("error") is None, \
            f"m1_2nvp stage-4 returned error: {g.get('error')}"
        assert g.get("ligand_resname") == "Z4Y"
        assert g.get("ligand_n_heavy_atoms") == 12 or g.get("ligand_n_heavy_atoms") is None


def test_panel_v1_membership_hash_unchanged(v1_hash: str):
    """Panel freeze: strict_dcc_panel_v1 membership must not drift at runtime.
    If this fails, a strict_dcc_panel_v2 MUST be emitted with a delta log."""
    assert v1_hash == EXPECTED_V1_HASH, (
        f"strict_dcc_panel_v1 membership hash changed\n"
        f"  expected: {EXPECTED_V1_HASH}\n"
        f"  got     : {v1_hash}\n"
        f"If panel membership was intentionally modified, create "
        f"{PANEL_V2.name} with an explicit delta_log and update EXPECTED_V1_HASH."
    )


def test_no_v2_unless_membership_changed(v1_hash: str):
    """If v1 hash is unchanged, v2 must not exist (v2 is reserved for membership changes)."""
    if v1_hash == EXPECTED_V1_HASH:
        assert not PANEL_V2.exists(), \
            "strict_dcc_panel_v2.json must not exist unless panel membership changed"


def test_panel_has_18_members():
    d = json.loads(PANEL_V1.read_text())
    assert d["n_total"] == 18
    assert len(d["panel"]) == 18


if __name__ == "__main__":
    # Allow running the file directly for hash discovery
    if len(sys.argv) > 1 and sys.argv[1] == "--print-hash":
        print(_membership_hash(PANEL_V1))
    else:
        raise SystemExit(pytest.main([__file__, "-v"]))

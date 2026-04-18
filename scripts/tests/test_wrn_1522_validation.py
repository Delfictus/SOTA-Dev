"""Regression test: WRN site 1522 must remain HARD_NOT_MATCH vs 8PFO/HRO761.

Invariants locked by this test:
  * centroid_distance (global) > 30 Å
  * centroid_distance (local, 15 Å window around site centroid) > 30 Å
  * overlap_fraction < 0.05 (ligand-contact cutoff 4.5 Å)
  * verdict == HARD_NOT_MATCH

If any invariant changes the pipeline FAILS loudly. WRN 1522 is not the
HRO761 binding site and no reranker, scorer, or classifier change may
silently alter that spatial fact.
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts/quarantine/site_vs_holo_strict.py"
REPORT = Path("/tmp/wrn_1522_strict.json")


@pytest.fixture(scope="module")
def verification_record() -> dict:
    if not SCRIPT.exists():
        pytest.fail(f"missing verification script at {SCRIPT}")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "wrn_1522"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            f"site_vs_holo_strict.py wrn_1522 exited {result.returncode}\n"
            f"stdout: {result.stdout[-2000:]}\n"
            f"stderr: {result.stderr[-2000:]}"
        )
    assert REPORT.exists(), f"expected report at {REPORT} after script run"
    return json.loads(REPORT.read_text())


def test_residue_convention_locked(verification_record: dict) -> None:
    assert verification_record["residue_convention_used"] == "one_indexed", \
        "residue convention must stay one_indexed (tid-1 → topology_index)"


def test_centroid_distance_global_above_30A(verification_record: dict) -> None:
    d = verification_record["centroid_distance_global"]
    assert d > 30.0, f"global centroid_distance dropped to {d} Å — WRN 1522 should not spatially move"


def test_centroid_distance_local_above_30A(verification_record: dict) -> None:
    d = verification_record["centroid_distance_local"]
    assert d is not None, "local alignment must be computable"
    assert d > 30.0, f"local (15 Å window) centroid_distance dropped to {d} Å"


def test_overlap_fraction_below_0_05(verification_record: dict) -> None:
    f = verification_record["overlap_fraction"]
    assert f < 0.05, f"overlap_fraction rose to {f} — residue overlap with HRO761 contact residues should stay <0.05"


def test_verdict_is_hard_not_match(verification_record: dict) -> None:
    v = verification_record["verdict"]
    assert v == "HARD_NOT_MATCH", f"verdict = {v!r}, expected HARD_NOT_MATCH"


def test_rerank_rank_is_1(verification_record: dict) -> None:
    assert verification_record["rerank_rank"] == 1, \
        "WRN 1522 is the top-1 CRYPTIC misclassification; if the reranker changes this fact, update the test deliberately"


def test_therm_class_is_cryptic(verification_record: dict) -> None:
    assert verification_record["therm_class"] == "CRYPTIC", \
        "WRN 1522 is the CRYPTIC false-positive; changing its class means the classifier changed"

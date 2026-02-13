"""Shared fixtures for WT-7 ensemble scoring tests.

All tests use synthetic data — no MD engine or GPU required.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.interfaces import EnsembleMMGBSA, InteractionEntropy, PocketDynamics


# ── Reproducible RNG ──────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ── Per-frame energy arrays ──────────────────────────────────────────────

@pytest.fixture
def per_frame_energies(rng):
    """100 frames of MM-GBSA total energies centred around -8 kcal/mol."""
    return rng.normal(-8.0, 2.5, 100)


@pytest.fixture
def decomposition_arrays(rng):
    """Per-frame MM-GBSA component energies (100 frames each)."""
    return {
        "vdw": rng.normal(-25.0, 3.0, 100),
        "elec": rng.normal(-15.0, 5.0, 100),
        "gb": rng.normal(30.0, 4.0, 100),
        "sa": rng.normal(-3.0, 0.5, 100),
    }


@pytest.fixture
def per_residue_arrays(rng):
    """Per-frame per-residue decomposition for 3 hot-spot residues."""
    return {
        12: rng.normal(-2.5, 0.3, 100),
        34: rng.normal(-1.8, 0.4, 100),
        60: rng.normal(-0.5, 0.2, 100),
    }


# ── Interaction energy arrays ────────────────────────────────────────────

@pytest.fixture
def interaction_energies(rng):
    """200 frames of protein-ligand interaction energies."""
    return rng.normal(-40.0, 5.0, 200)


# ── Pocket volume streams ───────────────────────────────────────────────

@pytest.fixture
def open_pocket_streams(rng):
    """20 streams of mostly-open pocket volumes (STABLE_OPEN)."""
    streams = []
    for _ in range(20):
        v = rng.normal(350, 50, 200)
        # Inject brief closures (~10% of frames)
        mask = rng.random(200) < 0.10
        v[mask] = rng.normal(100, 30, mask.sum())
        streams.append(v)
    return streams


@pytest.fixture
def transient_pocket_streams(rng):
    """10 streams where pocket is open ~30% of the time (TRANSIENT)."""
    streams = []
    for _ in range(10):
        v = rng.normal(150, 60, 200)
        # Only ~30% above 200 threshold
        mask = rng.random(200) < 0.30
        v[mask] = rng.normal(350, 40, mask.sum())
        streams.append(v)
    return streams


@pytest.fixture
def rare_event_streams(rng):
    """10 streams where pocket is almost never open (RARE_EVENT)."""
    streams = []
    for _ in range(10):
        v = rng.normal(80, 20, 200)  # Mostly below threshold
        # Very rare openings (~3%)
        mask = rng.random(200) < 0.03
        v[mask] = rng.normal(300, 30, mask.sum())
        streams.append(v)
    return streams


# ── Pre-built interface objects ──────────────────────────────────────────

@pytest.fixture
def sample_ensemble_mmgbsa():
    return EnsembleMMGBSA(
        compound_id="TEST001",
        delta_g_mean=-8.2,
        delta_g_std=2.5,
        delta_g_sem=0.25,
        n_snapshots=100,
        snapshot_interval_ps=100.0,
        decomposition={"vdw": -25.0, "elec": -15.0, "gb": 30.0, "sa": -3.0},
        per_residue_contributions={12: -2.5, 34: -1.8, 60: -0.5},
        method="MMGBSA_ensemble",
    )


@pytest.fixture
def sample_interaction_entropy():
    return InteractionEntropy(
        compound_id="TEST001",
        minus_t_delta_s=11.5,
        delta_h=-8.2,
        delta_g_ie=3.3,
        n_frames=200,
        convergence_block_std=0.3,
    )


@pytest.fixture
def sample_pocket_dynamics():
    return PocketDynamics(
        pocket_id=0,
        p_open=0.72,
        p_open_error=0.05,
        mean_open_lifetime_ns=1.2,
        mean_closed_lifetime_ns=0.4,
        n_opening_events=15,
        druggability_classification="STABLE_OPEN",
        volume_autocorrelation_ns=0.15,
    )

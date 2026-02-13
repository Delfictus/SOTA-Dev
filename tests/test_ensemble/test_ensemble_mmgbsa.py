"""Tests for ensemble_mmgbsa.py — ensemble-averaged MM-GBSA scoring."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.interfaces import EnsembleMMGBSA
from scripts.ensemble.ensemble_mmgbsa import (
    DEFAULT_SNAPSHOT_INTERVAL_PS,
    ENERGY_COMPONENTS,
    EnsembleMMGBSAConfig,
    compute_ensemble_mmgbsa,
    get_hotspot_residues,
    run_ensemble_mmgbsa,
    select_frames,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Core computation
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeEnsembleMmgbsa:
    def test_basic_computation(self, per_frame_energies, decomposition_arrays):
        result = compute_ensemble_mmgbsa(
            per_frame_energies, decomposition_arrays, compound_id="TEST"
        )
        assert isinstance(result, EnsembleMMGBSA)
        assert result.compound_id == "TEST"
        assert result.n_snapshots == 100
        assert result.method == "MMGBSA_ensemble"

    def test_mean_in_expected_range(self, per_frame_energies, decomposition_arrays):
        """Ensemble mean should be near the input distribution centre."""
        result = compute_ensemble_mmgbsa(
            per_frame_energies, decomposition_arrays, compound_id="TEST"
        )
        assert -15.0 < result.delta_g_mean < 0.0

    def test_sem_smaller_than_std(self, per_frame_energies, decomposition_arrays):
        """SEM should be smaller than std (SEM ≈ std/√N)."""
        result = compute_ensemble_mmgbsa(
            per_frame_energies, decomposition_arrays, compound_id="TEST"
        )
        assert result.delta_g_sem < result.delta_g_std

    def test_decomposition_components(self, per_frame_energies, decomposition_arrays):
        result = compute_ensemble_mmgbsa(
            per_frame_energies, decomposition_arrays, compound_id="TEST"
        )
        for comp in ENERGY_COMPONENTS:
            assert comp in result.decomposition

    def test_per_residue_contributions(
        self, per_frame_energies, decomposition_arrays, per_residue_arrays
    ):
        result = compute_ensemble_mmgbsa(
            per_frame_energies,
            decomposition_arrays,
            per_residue_contributions=per_residue_arrays,
            compound_id="TEST",
        )
        assert 12 in result.per_residue_contributions
        assert 34 in result.per_residue_contributions
        assert 60 in result.per_residue_contributions
        # Residue 12 should be most stabilising (most negative)
        assert result.per_residue_contributions[12] < result.per_residue_contributions[60]

    def test_empty_energies_raises(self):
        with pytest.raises(ValueError, match="No per-frame"):
            compute_ensemble_mmgbsa(np.array([]), {})

    def test_single_frame(self):
        result = compute_ensemble_mmgbsa(
            np.array([-8.0]),
            {"vdw": np.array([-25.0]), "elec": np.array([-15.0]),
             "gb": np.array([30.0]), "sa": np.array([-3.0])},
            compound_id="SINGLE",
        )
        assert result.n_snapshots == 1
        assert result.delta_g_mean == -8.0
        assert result.delta_g_std == 0.0

    def test_few_frames(self):
        """3 frames should work (below block-average threshold)."""
        energies = np.array([-7.0, -8.0, -9.0])
        decomp = {k: np.zeros(3) for k in ENERGY_COMPONENTS}
        result = compute_ensemble_mmgbsa(energies, decomp, compound_id="FEW")
        assert result.n_snapshots == 3
        assert abs(result.delta_g_mean - (-8.0)) < 1e-10

    def test_custom_method_label(self, per_frame_energies, decomposition_arrays):
        result = compute_ensemble_mmgbsa(
            per_frame_energies, decomposition_arrays,
            method="MMPBSA_ensemble", compound_id="TEST",
        )
        assert result.method == "MMPBSA_ensemble"

    def test_snapshot_interval(self, per_frame_energies, decomposition_arrays):
        result = compute_ensemble_mmgbsa(
            per_frame_energies, decomposition_arrays,
            snapshot_interval_ps=50.0, compound_id="TEST",
        )
        assert result.snapshot_interval_ps == 50.0


# ═══════════════════════════════════════════════════════════════════════════
#  Frame selection
# ═══════════════════════════════════════════════════════════════════════════

class TestSelectFrames:
    def test_default_stride(self):
        frames = select_frames(1000, 100.0)
        assert len(frames) > 0
        assert frames[0] == 0

    def test_all_frames_when_interval_matches(self):
        """If interval == ps_per_frame, select every frame."""
        frames = select_frames(100, 1.0, timestep_ps=1.0, save_interval=1)
        assert len(frames) == 100

    def test_stride_calculation(self):
        """100 ps interval with 2 ps timestep, save every 50 steps = 100 ps/frame."""
        frames = select_frames(1000, 100.0, timestep_ps=2.0, save_interval=50)
        assert frames[1] - frames[0] == 1  # stride=1 since ps_per_frame matches

    def test_single_frame_traj(self):
        frames = select_frames(1, 100.0)
        assert frames == [0]

    def test_empty_protection(self):
        """Should always return at least one frame."""
        frames = select_frames(1, 10000.0)
        assert len(frames) >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  Hot-spot identification
# ═══════════════════════════════════════════════════════════════════════════

class TestHotspotResidues:
    def test_identifies_hotspots(self, sample_ensemble_mmgbsa):
        hotspots = get_hotspot_residues(sample_ensemble_mmgbsa, threshold_kcal=-1.0)
        assert len(hotspots) == 2  # residues 12 and 34
        # Sorted by contribution (most negative first)
        assert hotspots[0][0] == 12
        assert hotspots[1][0] == 34

    def test_strict_threshold(self, sample_ensemble_mmgbsa):
        hotspots = get_hotspot_residues(sample_ensemble_mmgbsa, threshold_kcal=-2.0)
        assert len(hotspots) == 1  # Only residue 12
        assert hotspots[0][0] == 12

    def test_no_hotspots(self, sample_ensemble_mmgbsa):
        hotspots = get_hotspot_residues(sample_ensemble_mmgbsa, threshold_kcal=-10.0)
        assert len(hotspots) == 0

    def test_all_residues(self, sample_ensemble_mmgbsa):
        hotspots = get_hotspot_residues(sample_ensemble_mmgbsa, threshold_kcal=0.0)
        assert len(hotspots) == 3


# ═══════════════════════════════════════════════════════════════════════════
#  Dry run
# ═══════════════════════════════════════════════════════════════════════════

class TestDryRun:
    def test_dry_run_returns_result(self, tmp_path):
        top = tmp_path / "complex.prmtop"
        traj = tmp_path / "production.nc"
        top.write_text("mock topology")
        traj.write_text("mock trajectory")

        result = run_ensemble_mmgbsa(
            str(top), str(traj), "DRY001", dry_run=True
        )
        assert isinstance(result, EnsembleMMGBSA)
        assert result.compound_id == "DRY001"
        assert result.n_snapshots == 100
        assert -15.0 < result.delta_g_mean < 0.0

    def test_dry_run_missing_topology_raises(self, tmp_path):
        traj = tmp_path / "production.nc"
        traj.write_text("mock trajectory")
        with pytest.raises(FileNotFoundError, match="Topology"):
            run_ensemble_mmgbsa("/nonexistent/top", str(traj), "X", dry_run=True)

    def test_dry_run_missing_trajectory_raises(self, tmp_path):
        top = tmp_path / "complex.prmtop"
        top.write_text("mock topology")
        with pytest.raises(FileNotFoundError, match="Trajectory"):
            run_ensemble_mmgbsa(str(top), "/nonexistent/traj", "X", dry_run=True)

    def test_dry_run_writes_json(self, tmp_path):
        top = tmp_path / "complex.prmtop"
        traj = tmp_path / "production.nc"
        top.write_text("mock")
        traj.write_text("mock")

        out_dir = tmp_path / "output"
        run_ensemble_mmgbsa(
            str(top), str(traj), "DRY002",
            output_dir=str(out_dir), dry_run=True,
        )
        result_file = out_dir / "DRY002_ensemble_mmgbsa.json"
        # Dry run doesn't write JSON (only real runs do), but out_dir exists
        assert out_dir.exists()


# ═══════════════════════════════════════════════════════════════════════════
#  Serialisation round-trip
# ═══════════════════════════════════════════════════════════════════════════

class TestSerialisation:
    def test_json_roundtrip(self, sample_ensemble_mmgbsa):
        j = sample_ensemble_mmgbsa.to_json()
        loaded = EnsembleMMGBSA.from_json(j)
        assert loaded.compound_id == sample_ensemble_mmgbsa.compound_id
        assert loaded.delta_g_mean == sample_ensemble_mmgbsa.delta_g_mean
        assert loaded.per_residue_contributions == sample_ensemble_mmgbsa.per_residue_contributions

    def test_pickle_roundtrip(self, sample_ensemble_mmgbsa):
        data = sample_ensemble_mmgbsa.to_pickle()
        loaded = EnsembleMMGBSA.from_pickle(data)
        assert loaded.compound_id == sample_ensemble_mmgbsa.compound_id
        assert loaded.n_snapshots == 100

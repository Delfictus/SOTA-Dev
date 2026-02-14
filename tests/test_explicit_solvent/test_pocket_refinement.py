"""Tests for pocket_refinement.py — THE QC GATE.

Tests cover:
    - Stability classification logic (the most critical code path)
    - Snapshot frame selection
    - Dry-run CLI mode
    - ExplicitSolventResult serialisation round-trip
    - Edge cases (zero volume, boundary thresholds)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.interfaces import ExplicitSolventResult
from scripts.explicit_solvent.pocket_refinement import (
    CLASSIFICATION_COLLAPSED,
    CLASSIFICATION_METASTABLE,
    CLASSIFICATION_STABLE,
    RMSD_METASTABLE_THRESHOLD,
    RMSD_STABLE_THRESHOLD,
    VOLUME_CV_METASTABLE,
    VOLUME_CV_STABLE,
    classify_pocket_stability,
    select_snapshot_frames,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Classification tests — THE critical logic
# ═══════════════════════════════════════════════════════════════════════════

class TestClassifyPocketStability:
    """Exhaustive tests for the stability classification decision tree."""

    # ── STABLE ────────────────────────────────────────────────────────
    def test_clearly_stable(self):
        """Low RMSD, low volume CV → STABLE."""
        assert classify_pocket_stability(1.0, 500.0, 50.0) == CLASSIFICATION_STABLE

    def test_just_below_stable_rmsd_threshold(self):
        """RMSD exactly at stable boundary (< 2.0)."""
        assert classify_pocket_stability(1.99, 500.0, 50.0) == CLASSIFICATION_STABLE

    def test_zero_rmsd_stable(self):
        """Perfect RMSD = 0 → STABLE."""
        assert classify_pocket_stability(0.0, 500.0, 50.0) == CLASSIFICATION_STABLE

    def test_stable_with_small_volume_variation(self):
        """Volume CV = 19.9% (just under 20%) → STABLE."""
        # CV = std/mean = 99.5/500 = 0.199
        assert classify_pocket_stability(1.0, 500.0, 99.5) == CLASSIFICATION_STABLE

    # ── METASTABLE ────────────────────────────────────────────────────
    def test_metastable_by_rmsd(self):
        """RMSD in 2.0–3.5 range, low volume CV → METASTABLE."""
        assert classify_pocket_stability(2.5, 500.0, 50.0) == CLASSIFICATION_METASTABLE

    def test_metastable_at_rmsd_boundary(self):
        """RMSD exactly at 2.0 → METASTABLE (blueprint: 2.0–3.5 range)."""
        assert classify_pocket_stability(2.0, 500.0, 50.0) == CLASSIFICATION_METASTABLE

    def test_metastable_just_below_collapsed_rmsd(self):
        """RMSD = 3.49 → still METASTABLE."""
        assert classify_pocket_stability(3.49, 500.0, 50.0) == CLASSIFICATION_METASTABLE

    def test_metastable_by_volume_cv(self):
        """Low RMSD but volume CV 20–40% → METASTABLE."""
        # CV = 150/500 = 0.30
        assert classify_pocket_stability(1.0, 500.0, 150.0) == CLASSIFICATION_METASTABLE

    def test_metastable_volume_cv_at_boundary(self):
        """Volume CV exactly at 20% → METASTABLE (blueprint: 20–40% range)."""
        # CV = 100/500 = 0.20
        assert classify_pocket_stability(1.0, 500.0, 100.0) == CLASSIFICATION_METASTABLE

    def test_metastable_both_criteria(self):
        """Both RMSD and volume in metastable range."""
        assert classify_pocket_stability(2.5, 500.0, 150.0) == CLASSIFICATION_METASTABLE

    # ── COLLAPSED ─────────────────────────────────────────────────────
    def test_collapsed_by_rmsd(self):
        """RMSD > 3.5 → COLLAPSED regardless of volume."""
        assert classify_pocket_stability(4.0, 500.0, 50.0) == CLASSIFICATION_COLLAPSED

    def test_collapsed_at_rmsd_boundary(self):
        """RMSD exactly at 3.5 → still METASTABLE (blueprint: >3.5 for COLLAPSED)."""
        assert classify_pocket_stability(3.5, 500.0, 50.0) == CLASSIFICATION_METASTABLE

    def test_collapsed_just_above_rmsd_boundary(self):
        """RMSD = 3.51 → COLLAPSED."""
        assert classify_pocket_stability(3.51, 500.0, 50.0) == CLASSIFICATION_COLLAPSED

    def test_collapsed_by_volume_cv(self):
        """Low RMSD but volume CV > 40% → COLLAPSED."""
        # CV = 210/500 = 0.42
        assert classify_pocket_stability(1.0, 500.0, 210.0) == CLASSIFICATION_COLLAPSED

    def test_collapsed_volume_cv_at_boundary(self):
        """Volume CV exactly at 40% → still METASTABLE (blueprint: >40% for COLLAPSED)."""
        # CV = 200/500 = 0.40
        assert classify_pocket_stability(1.0, 500.0, 200.0) == CLASSIFICATION_METASTABLE

    def test_collapsed_volume_cv_just_above_boundary(self):
        """Volume CV = 40.1% → COLLAPSED."""
        # CV = 200.5/500 = 0.401
        assert classify_pocket_stability(1.0, 500.0, 200.5) == CLASSIFICATION_COLLAPSED

    def test_collapsed_both_criteria(self):
        """Both RMSD and volume in collapsed range."""
        assert classify_pocket_stability(5.0, 200.0, 120.0) == CLASSIFICATION_COLLAPSED

    def test_collapsed_extreme_rmsd(self):
        """Very high RMSD → definitely COLLAPSED."""
        assert classify_pocket_stability(10.0, 500.0, 50.0) == CLASSIFICATION_COLLAPSED

    # ── Edge cases ────────────────────────────────────────────────────
    def test_zero_volume_collapsed(self):
        """Zero mean volume → CV=1.0 → COLLAPSED."""
        assert classify_pocket_stability(1.0, 0.0, 10.0) == CLASSIFICATION_COLLAPSED

    def test_zero_volume_std(self):
        """Zero volume std → CV=0 → STABLE (if RMSD is fine)."""
        assert classify_pocket_stability(1.0, 500.0, 0.0) == CLASSIFICATION_STABLE

    def test_negative_values_handled(self):
        """Negative RMSD (shouldn't happen) still classifies."""
        result = classify_pocket_stability(-1.0, 500.0, 50.0)
        assert result == CLASSIFICATION_STABLE

    def test_very_small_volume(self):
        """Tiny volume with any std → high CV → COLLAPSED."""
        assert classify_pocket_stability(1.0, 1.0, 0.5) == CLASSIFICATION_COLLAPSED


# ═══════════════════════════════════════════════════════════════════════════
#  Threshold constant tests
# ═══════════════════════════════════════════════════════════════════════════

class TestThresholdConstants:
    """Verify blueprint threshold values are correct."""

    def test_rmsd_stable_threshold(self):
        assert RMSD_STABLE_THRESHOLD == 2.0

    def test_rmsd_metastable_threshold(self):
        assert RMSD_METASTABLE_THRESHOLD == 3.5

    def test_volume_cv_stable(self):
        assert VOLUME_CV_STABLE == 0.20

    def test_volume_cv_metastable(self):
        assert VOLUME_CV_METASTABLE == 0.40


# ═══════════════════════════════════════════════════════════════════════════
#  Snapshot selection tests
# ═══════════════════════════════════════════════════════════════════════════

class TestSelectSnapshotFrames:
    def test_default_10_snapshots(self):
        frames = select_snapshot_frames(1000)
        assert len(frames) == 10
        # All within range
        assert all(0 <= f < 1000 for f in frames)

    def test_evenly_spaced(self):
        frames = select_snapshot_frames(1000, n_snapshots=5)
        assert len(frames) == 5
        # Roughly evenly spaced
        diffs = [frames[i + 1] - frames[i] for i in range(len(frames) - 1)]
        assert all(d > 0 for d in diffs)

    def test_fewer_frames_than_snapshots(self):
        frames = select_snapshot_frames(5, n_snapshots=10)
        assert len(frames) == 5
        assert frames == [0, 1, 2, 3, 4]

    def test_single_frame(self):
        frames = select_snapshot_frames(1, n_snapshots=10)
        assert frames == [0]

    def test_exact_match(self):
        frames = select_snapshot_frames(10, n_snapshots=10)
        assert len(frames) == 10


# ═══════════════════════════════════════════════════════════════════════════
#  Dry-run integration test
# ═══════════════════════════════════════════════════════════════════════════

class TestDryRun:
    def test_dry_run_returns_result(
        self, mock_pdb_path, spike_json_path, tmp_path
    ):
        """Dry run should return a valid ExplicitSolventResult without MD."""
        from scripts.explicit_solvent.pocket_refinement import run_pocket_refinement

        result = run_pocket_refinement(
            pdb_path=mock_pdb_path,
            spike_json_path=spike_json_path,
            time_ns=10.0,
            output_dir=str(tmp_path / "output"),
            dry_run=True,
        )
        assert isinstance(result, ExplicitSolventResult)
        assert result.pocket_id == 0
        assert result.simulation_time_ns == 10.0
        assert result.water_model == "TIP3P"
        assert result.pocket_stable is True

    def test_dry_run_writes_json(
        self, mock_pdb_path, spike_json_path, tmp_path
    ):
        """Dry run should write a result JSON to the output directory."""
        from scripts.explicit_solvent.pocket_refinement import run_pocket_refinement

        out_dir = tmp_path / "output"
        run_pocket_refinement(
            pdb_path=mock_pdb_path,
            spike_json_path=spike_json_path,
            output_dir=str(out_dir),
            dry_run=True,
        )
        result_file = out_dir / "pocket_0_result.json"
        assert result_file.exists()

        loaded = ExplicitSolventResult.from_json(result_file.read_text())
        assert loaded.pocket_id == 0

    def test_dry_run_missing_pdb_raises(self, spike_json_path, tmp_path):
        """Missing PDB → FileNotFoundError."""
        from scripts.explicit_solvent.pocket_refinement import run_pocket_refinement

        with pytest.raises(FileNotFoundError, match="PDB"):
            run_pocket_refinement(
                pdb_path="/nonexistent/path.pdb",
                spike_json_path=spike_json_path,
                output_dir=str(tmp_path),
                dry_run=True,
            )

    def test_dry_run_missing_spike_json_raises(self, mock_pdb_path, tmp_path):
        """Missing spike JSON → FileNotFoundError."""
        from scripts.explicit_solvent.pocket_refinement import run_pocket_refinement

        with pytest.raises(FileNotFoundError, match="Spike JSON"):
            run_pocket_refinement(
                pdb_path=mock_pdb_path,
                spike_json_path="/nonexistent/spikes.json",
                output_dir=str(tmp_path),
                dry_run=True,
            )


# ═══════════════════════════════════════════════════════════════════════════
#  Serialisation round-trip tests
# ═══════════════════════════════════════════════════════════════════════════

class TestResultSerialisation:
    def test_json_roundtrip(self, sample_stable_result):
        """ExplicitSolventResult survives JSON serialisation."""
        j = sample_stable_result.to_json()
        loaded = ExplicitSolventResult.from_json(j)
        assert loaded.pocket_id == sample_stable_result.pocket_id
        assert loaded.pocket_rmsd_mean == sample_stable_result.pocket_rmsd_mean
        assert loaded.pocket_stable is True
        assert loaded.snapshot_frames == sample_stable_result.snapshot_frames

    def test_pickle_roundtrip(self, sample_stable_result):
        """ExplicitSolventResult survives pickle serialisation."""
        data = sample_stable_result.to_pickle()
        loaded = ExplicitSolventResult.from_pickle(data)
        assert loaded.pocket_id == sample_stable_result.pocket_id
        assert loaded.pocket_volume_mean == sample_stable_result.pocket_volume_mean

    def test_result_with_water_map(self, sample_stable_result, sample_water_map):
        """Result with attached WaterMap serialises correctly."""
        sample_stable_result.water_map = sample_water_map
        j = sample_stable_result.to_json()
        loaded = ExplicitSolventResult.from_json(j)
        assert loaded.water_map is not None
        assert loaded.water_map.pocket_id == 0
        assert len(loaded.water_map.hydration_sites) == 2
        assert loaded.water_map.n_displaceable == 1

    def test_collapsed_result_stable_flag(self, sample_collapsed_result):
        """Collapsed result has pocket_stable=False."""
        assert sample_collapsed_result.pocket_stable is False
        classification = classify_pocket_stability(
            sample_collapsed_result.pocket_rmsd_mean,
            sample_collapsed_result.pocket_volume_mean,
            sample_collapsed_result.pocket_volume_std,
        )
        assert classification == CLASSIFICATION_COLLAPSED


# ═══════════════════════════════════════════════════════════════════════════
#  Classification → pipeline flow tests
# ═══════════════════════════════════════════════════════════════════════════

class TestPipelineGating:
    """Verify the QC gate correctly blocks/allows pipeline progression."""

    def test_stable_allows_pipeline(self, sample_stable_result):
        """STABLE pocket → pocket_stable=True → pipeline proceeds."""
        classification = classify_pocket_stability(
            sample_stable_result.pocket_rmsd_mean,
            sample_stable_result.pocket_volume_mean,
            sample_stable_result.pocket_volume_std,
        )
        assert classification == CLASSIFICATION_STABLE
        assert sample_stable_result.pocket_stable is True

    def test_collapsed_blocks_pipeline(self, sample_collapsed_result):
        """COLLAPSED pocket → pocket_stable=False → pipeline STOPS."""
        classification = classify_pocket_stability(
            sample_collapsed_result.pocket_rmsd_mean,
            sample_collapsed_result.pocket_volume_mean,
            sample_collapsed_result.pocket_volume_std,
        )
        assert classification == CLASSIFICATION_COLLAPSED
        assert sample_collapsed_result.pocket_stable is False

    def test_metastable_allows_pipeline(self):
        """METASTABLE pocket → pocket_stable=True → pipeline proceeds with caution."""
        result = ExplicitSolventResult(
            pocket_id=0,
            simulation_time_ns=10.0,
            water_model="TIP3P",
            force_field="ff14SB",
            pocket_stable=True,  # METASTABLE still proceeds
            pocket_rmsd_mean=2.5,
            pocket_rmsd_std=0.5,
            pocket_volume_mean=400.0,
            pocket_volume_std=60.0,
            n_structural_waters=2,
            trajectory_path="/mock/traj.dcd",
            snapshot_frames=[0, 50, 100],
        )
        classification = classify_pocket_stability(
            result.pocket_rmsd_mean,
            result.pocket_volume_mean,
            result.pocket_volume_std,
        )
        assert classification == CLASSIFICATION_METASTABLE
        assert result.pocket_stable is True

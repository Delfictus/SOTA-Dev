"""Tests for water_map_analysis.py — hydration-site thermodynamics.

Tests cover:
    - Grid construction geometry
    - Water map from pre-built arrays (no MDTraj)
    - Hydration site classification logic
    - WaterMap serialisation round-trip
    - Displacement energy accounting
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.interfaces import HydrationSite, WaterMap
from scripts.explicit_solvent.water_map_analysis import (
    BULK_WATER_ENERGY_KCAL,
    DEFAULT_ANALYSIS_RADIUS,
    DEFAULT_GRID_SPACING,
    DG_HAPPY_THRESHOLD,
    DG_UNHAPPY_THRESHOLD,
    OCCUPANCY_BULK_THRESHOLD,
    build_analysis_grid,
    cluster_hydration_sites,
    compute_water_map_from_arrays,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Grid construction tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildAnalysisGrid:
    def test_grid_default_params(self):
        centroid = (13.0, -2.5, 8.5)
        grid_points, origin, spacing = build_analysis_grid(centroid)
        assert spacing == DEFAULT_GRID_SPACING
        # All points within sphere
        dists = np.linalg.norm(
            grid_points - np.array(centroid), axis=1
        )
        assert np.all(dists <= DEFAULT_ANALYSIS_RADIUS + spacing)

    def test_grid_contains_centroid(self):
        centroid = (10.0, 20.0, 30.0)
        grid_points, _, _ = build_analysis_grid(centroid, radius=5.0)
        # At least one point near centroid
        dists = np.linalg.norm(grid_points - np.array(centroid), axis=1)
        assert np.min(dists) < 1.0

    def test_grid_spherical_not_cubic(self):
        centroid = (0.0, 0.0, 0.0)
        grid_points, _, _ = build_analysis_grid(centroid, radius=5.0, spacing=1.0)
        dists = np.linalg.norm(grid_points, axis=1)
        # No points outside the sphere
        assert np.all(dists <= 5.0 + 0.01)

    def test_grid_spacing_affects_density(self):
        centroid = (0.0, 0.0, 0.0)
        fine, _, _ = build_analysis_grid(centroid, radius=3.0, spacing=0.5)
        coarse, _, _ = build_analysis_grid(centroid, radius=3.0, spacing=1.0)
        assert len(fine) > len(coarse)

    def test_small_radius_grid(self):
        centroid = (5.0, 5.0, 5.0)
        grid_points, _, _ = build_analysis_grid(centroid, radius=1.0, spacing=0.5)
        assert len(grid_points) > 0
        dists = np.linalg.norm(grid_points - np.array(centroid), axis=1)
        assert np.all(dists <= 1.0 + 0.01)


# ═══════════════════════════════════════════════════════════════════════════
#  Clustering tests
# ═══════════════════════════════════════════════════════════════════════════

class TestClusterHydrationSites:
    def test_no_occupied_voxels(self):
        grid = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
        occ = np.array([0.0, 0.0, 0.0])
        dh = np.zeros(3)
        tds = np.zeros(3)
        dg = np.zeros(3)
        sites = cluster_hydration_sites(grid, occ, dh, tds, dg)
        assert len(sites) == 0

    def test_single_site_happy(self):
        # Tightly packed points with high occupancy and negative dG
        grid = np.array([
            [10.0, 10.0, 10.0],
            [10.3, 10.0, 10.0],
            [10.0, 10.3, 10.0],
        ])
        occ = np.array([0.90, 0.85, 0.88])
        dh = np.array([-2.0, -1.8, -1.9])
        tds = np.array([0.5, 0.4, 0.45])
        dg = np.array([-1.5, -1.4, -1.45])
        sites = cluster_hydration_sites(grid, occ, dh, tds, dg)
        assert len(sites) == 1
        assert sites[0].classification == "CONSERVED_HAPPY"
        assert sites[0].displaceable is False

    def test_single_site_unhappy(self):
        grid = np.array([
            [10.0, 10.0, 10.0],
            [10.3, 10.0, 10.0],
            [10.0, 10.3, 10.0],
        ])
        occ = np.array([0.80, 0.75, 0.78])
        dh = np.array([1.0, 1.2, 1.1])
        tds = np.array([1.0, 0.8, 0.9])
        dg = np.array([2.0, 2.0, 2.0])
        sites = cluster_hydration_sites(grid, occ, dh, tds, dg)
        assert len(sites) == 1
        assert sites[0].classification == "CONSERVED_UNHAPPY"
        assert sites[0].displaceable is True

    def test_two_separated_sites(self):
        grid = np.array([
            [10.0, 10.0, 10.0],
            [10.3, 10.0, 10.0],
            [20.0, 20.0, 20.0],
            [20.3, 20.0, 20.0],
        ])
        occ = np.array([0.90, 0.85, 0.80, 0.78])
        dh = np.array([-2.0, -1.8, 1.5, 1.3])
        tds = np.array([0.5, 0.4, 0.8, 0.7])
        dg = np.array([-1.5, -1.4, 2.3, 2.0])
        sites = cluster_hydration_sites(grid, occ, dh, tds, dg)
        assert len(sites) == 2
        classifications = {s.classification for s in sites}
        assert "CONSERVED_HAPPY" in classifications
        assert "CONSERVED_UNHAPPY" in classifications

    def test_bulk_site_low_occupancy(self):
        grid = np.array([
            [10.0, 10.0, 10.0],
            [10.3, 10.0, 10.0],
        ])
        occ = np.array([0.15, 0.12])
        dh = np.zeros(2)
        tds = np.zeros(2)
        dg = np.array([0.5, 0.3])
        sites = cluster_hydration_sites(grid, occ, dh, tds, dg)
        # Should be empty (below min_occupancy=0.10) or classified as BULK
        for s in sites:
            assert s.classification == "BULK" or s.occupancy < OCCUPANCY_BULK_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════════
#  Water map from arrays (no MDTraj)
# ═══════════════════════════════════════════════════════════════════════════

class TestComputeWaterMapFromArrays:
    def test_empty_frames(self):
        wm = compute_water_map_from_arrays(
            pocket_centroid=(0.0, 0.0, 0.0),
            pocket_id=0,
            water_positions_per_frame=[],
        )
        assert isinstance(wm, WaterMap)
        assert wm.pocket_id == 0
        assert len(wm.hydration_sites) == 0
        assert wm.n_displaceable == 0

    def test_single_static_water(self):
        """One water staying in exactly the same position every frame."""
        pos = np.array([[5.0, 5.0, 5.0]])  # single water, Angstrom
        frames = [pos.copy() for _ in range(100)]
        wm = compute_water_map_from_arrays(
            pocket_centroid=(5.0, 5.0, 5.0),
            pocket_id=0,
            water_positions_per_frame=frames,
            analysis_radius=3.0,
            grid_spacing=0.5,
        )
        assert isinstance(wm, WaterMap)
        assert wm.analysis_frames == 100

    def test_water_map_serialisation_roundtrip(self):
        """WaterMap from arrays → JSON → from_json."""
        pos = np.array([[5.0, 5.0, 5.0]])
        frames = [pos.copy() for _ in range(50)]
        wm = compute_water_map_from_arrays(
            pocket_centroid=(5.0, 5.0, 5.0),
            pocket_id=1,
            water_positions_per_frame=frames,
            analysis_radius=3.0,
        )
        j = wm.to_json()
        loaded = WaterMap.from_json(j)
        assert loaded.pocket_id == 1
        assert loaded.analysis_frames == 50
        assert loaded.grid_resolution == 0.5


# ═══════════════════════════════════════════════════════════════════════════
#  Classification threshold tests
# ═══════════════════════════════════════════════════════════════════════════

class TestClassificationThresholds:
    """Verify blueprint threshold constants."""

    def test_happy_threshold(self):
        assert DG_HAPPY_THRESHOLD == -1.0

    def test_unhappy_threshold(self):
        assert DG_UNHAPPY_THRESHOLD == 1.0

    def test_bulk_occupancy_threshold(self):
        assert OCCUPANCY_BULK_THRESHOLD == 0.3


# ═══════════════════════════════════════════════════════════════════════════
#  Displacement energy accounting
# ═══════════════════════════════════════════════════════════════════════════

class TestDisplacementEnergy:
    def test_total_displacement_energy(self, sample_water_map):
        """Total displacement = sum of positive dG sites."""
        assert sample_water_map.total_displacement_energy == pytest.approx(1.8)

    def test_max_displacement_energy(self, sample_water_map):
        assert sample_water_map.max_displacement_energy == pytest.approx(1.8)

    def test_n_displaceable(self, sample_water_map):
        assert sample_water_map.n_displaceable == 1

    def test_happy_not_displaceable(self, sample_happy_site):
        assert sample_happy_site.displaceable is False
        assert sample_happy_site.delta_g_transfer < DG_HAPPY_THRESHOLD

    def test_unhappy_is_displaceable(self, sample_unhappy_site):
        assert sample_unhappy_site.displaceable is True
        assert sample_unhappy_site.delta_g_transfer > DG_UNHAPPY_THRESHOLD

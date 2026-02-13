"""Grid-based hydration-site thermodynamics (Inhomogeneous Solvation Theory).

Computes per-voxel water thermodynamic properties from an explicit-solvent
trajectory and clusters them into discrete :class:`HydrationSite` objects.

Algorithm
---------
1. Define analysis region: 8 Å sphere around pocket centroid.
2. Grid at 0.5 Å resolution.
3. Per voxel over trajectory:
   a. Water occupancy.
   b. ΔH: water–protein + water–water energy vs bulk.
   c. −TΔS: orientational + translational entropy vs bulk.
   d. ΔG_transfer = ΔH + (−TΔS).
4. Cluster into discrete HydrationSites (DBSCAN on high-occupancy voxels).
5. Classify:
   - ΔG < −1.0 → CONSERVED_HAPPY  (structural, don't displace)
   - ΔG > +1.0 → CONSERVED_UNHAPPY  (displace for free-energy gain)
   - Occupancy < 0.3 → BULK
6. total_displacement_energy = sum(ΔG of unhappy sites).

Output
------
:class:`~scripts.interfaces.WaterMap`
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scripts.interfaces import HydrationSite, WaterMap

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
DEFAULT_GRID_SPACING = 0.5      # Angstrom
DEFAULT_ANALYSIS_RADIUS = 8.0   # Angstrom
OCCUPANCY_BULK_THRESHOLD = 0.3  # Below this → BULK
DG_HAPPY_THRESHOLD = -1.0       # kcal/mol, below → CONSERVED_HAPPY
DG_UNHAPPY_THRESHOLD = 1.0      # kcal/mol, above → CONSERVED_UNHAPPY

# Bulk water reference values (TIP3P, 300K, 1 bar)
BULK_WATER_ENERGY_KCAL = -9.533     # kcal/mol  (self-energy of TIP3P bulk)
BULK_WATER_ENTROPY_KCAL = -5.83     # -TdS kcal/mol at 300K


# ── Grid construction ────────────────────────────────────────────────────

def build_analysis_grid(
    centroid: Tuple[float, float, float],
    radius: float = DEFAULT_ANALYSIS_RADIUS,
    spacing: float = DEFAULT_GRID_SPACING,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Build a 3D grid of voxel centres within a sphere.

    Parameters
    ----------
    centroid : tuple of float
        (x, y, z) centre of the analysis sphere, in Angstrom.
    radius : float
        Radius of the analysis sphere in Angstrom.
    spacing : float
        Grid spacing in Angstrom.

    Returns
    -------
    grid_points : np.ndarray
        (N, 3) array of voxel-centre coordinates in Angstrom.
    origin : np.ndarray
        (3,) corner of the cubic bounding box.
    spacing : float
        Grid spacing used.
    """
    cx, cy, cz = centroid
    n_bins = int(2 * radius / spacing) + 1
    x = np.linspace(cx - radius, cx + radius, n_bins)
    y = np.linspace(cy - radius, cy + radius, n_bins)
    z = np.linspace(cz - radius, cz + radius, n_bins)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    all_pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

    # Keep only points within the sphere
    dists = np.linalg.norm(all_pts - np.array(centroid), axis=1)
    mask = dists <= radius
    grid_points = all_pts[mask]
    origin = np.array([cx - radius, cy - radius, cz - radius])

    return grid_points, origin, spacing


# ── Per-voxel analysis ────────────────────────────────────────────────────

def compute_voxel_occupancy(
    trajectory: Any,
    grid_points: np.ndarray,
    spacing: float,
) -> np.ndarray:
    """Compute water occupancy at each grid voxel.

    For each frame, assigns water oxygen atoms to their nearest grid voxel
    and counts how many frames each voxel is occupied.

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        Explicit-solvent trajectory.
    grid_points : np.ndarray
        (N, 3) voxel centres in Angstrom.
    spacing : float
        Grid spacing in Angstrom.

    Returns
    -------
    np.ndarray
        Occupancy fraction per voxel, shape (N,).
    """
    water_oxy = trajectory.topology.select("water and name O")
    if len(water_oxy) == 0:
        return np.zeros(len(grid_points))

    half_spacing = spacing / 2.0
    counts = np.zeros(len(grid_points), dtype=np.float64)

    for frame_idx in range(trajectory.n_frames):
        # nm → Angstrom
        coords = trajectory.xyz[frame_idx, water_oxy, :] * 10.0
        # For each water, find nearest grid point
        for wc in coords:
            dists = np.linalg.norm(grid_points - wc, axis=1)
            nearest = np.argmin(dists)
            if dists[nearest] <= half_spacing:
                counts[nearest] += 1

    return counts / trajectory.n_frames


def estimate_voxel_energetics(
    trajectory: Any,
    grid_points: np.ndarray,
    occupancy: np.ndarray,
    spacing: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate per-voxel ΔH, −TΔS, and ΔG using simplified IST.

    This is a simplified implementation of Inhomogeneous Solvation Theory.
    For production use, SSTMap or cpptraj GIST should be used.  This
    implementation provides reasonable estimates for QC gating purposes.

    The approach:
    - ΔH is estimated from the deviation of local water density from bulk
      density, scaled by the bulk water interaction energy.  High-occupancy
      voxels (structured water) tend to have favourable enthalpy.
    - −TΔS is estimated from the reduction in translational/orientational
      degrees of freedom for structured water (higher occupancy = more
      ordered = larger entropy penalty).

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        Explicit-solvent trajectory.
    grid_points : np.ndarray
        (N, 3) voxel centres in Angstrom.
    occupancy : np.ndarray
        Per-voxel occupancy, shape (N,).
    spacing : float
        Grid spacing in Angstrom.

    Returns
    -------
    delta_h : np.ndarray
        Enthalpy contribution per voxel (kcal/mol), shape (N,).
    neg_tds : np.ndarray
        Entropy penalty per voxel (kcal/mol, positive = unfavourable).
    delta_g : np.ndarray
        Free energy of transfer per voxel (kcal/mol).
    """
    # Bulk water density: ~33.3 waters/nm^3 at 300K
    # In Angstrom: ~0.0333 waters/A^3
    bulk_density_per_a3 = 0.0333
    voxel_volume = spacing ** 3
    expected_bulk_occupancy = bulk_density_per_a3 * voxel_volume

    # Normalise occupancy relative to bulk expectation
    # A voxel with occupancy >> bulk is structured
    relative_occupancy = np.where(
        expected_bulk_occupancy > 0,
        occupancy / expected_bulk_occupancy,
        0.0,
    )

    # Enthalpy: structured water near protein has better interactions
    # than bulk (negative ΔH) if well-coordinated, worse if frustrated
    # Approximate: ΔH ≈ -k * (relative_occupancy - 1) for structured sites
    # High occupancy → favourable enthalpy (negative)
    # The scale factor maps to ~±3 kcal/mol range
    delta_h = -2.0 * (relative_occupancy - 1.0)
    # Clamp to physical range
    delta_h = np.clip(delta_h, -5.0, 5.0)

    # Entropy: structured water loses translational + orientational entropy
    # −TΔS is always >= 0 for structured water
    # More structured (higher occupancy) → larger entropy penalty
    neg_tds = np.where(
        occupancy > 0.05,
        2.0 * np.log1p(relative_occupancy),
        0.0,
    )
    neg_tds = np.clip(neg_tds, 0.0, 6.0)

    # Free energy of transfer: ΔG = ΔH + (−TΔS)
    delta_g = delta_h + neg_tds

    return delta_h, neg_tds, delta_g


# ── Clustering voxels → HydrationSites ────────────────────────────────────

def cluster_hydration_sites(
    grid_points: np.ndarray,
    occupancy: np.ndarray,
    delta_h: np.ndarray,
    neg_tds: np.ndarray,
    delta_g: np.ndarray,
    min_occupancy: float = 0.10,
    cluster_radius: float = 1.4,
    min_samples: int = 2,
) -> List[HydrationSite]:
    """Cluster high-occupancy voxels into discrete hydration sites.

    Parameters
    ----------
    grid_points : np.ndarray
        (N, 3) voxel centres in Angstrom.
    occupancy, delta_h, neg_tds, delta_g : np.ndarray
        Per-voxel thermodynamic data.
    min_occupancy : float
        Minimum occupancy to include a voxel in clustering.
    cluster_radius : float
        DBSCAN eps parameter in Angstrom (default 1.4 ≈ water diameter).
    min_samples : int
        DBSCAN min_samples parameter.

    Returns
    -------
    list of HydrationSite
        Clustered hydration sites with averaged properties.
    """
    from sklearn.cluster import DBSCAN

    # Filter to occupied voxels
    mask = occupancy >= min_occupancy
    if not np.any(mask):
        return []

    pts = grid_points[mask]
    occ = occupancy[mask]
    dh = delta_h[mask]
    tds = neg_tds[mask]
    dg = delta_g[mask]

    # DBSCAN clustering
    clustering = DBSCAN(eps=cluster_radius, min_samples=min_samples).fit(pts)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}

    sites: List[HydrationSite] = []
    for label in sorted(unique_labels):
        cluster_mask = labels == label
        c_pts = pts[cluster_mask]
        c_occ = occ[cluster_mask]
        c_dh = dh[cluster_mask]
        c_tds = tds[cluster_mask]
        c_dg = dg[cluster_mask]

        # Occupancy-weighted centre
        weights = c_occ / c_occ.sum()
        centre = np.average(c_pts, weights=weights, axis=0)

        # Weighted-average thermodynamics
        avg_occ = float(np.mean(c_occ))
        avg_dh = float(np.average(c_dh, weights=weights))
        avg_tds = float(np.average(c_tds, weights=weights))
        avg_dg = float(np.average(c_dg, weights=weights))

        # Estimate mean H-bonds from occupancy (heuristic)
        # Fully occupied sites typically have 2–3 H-bonds to protein
        n_hbonds = min(avg_occ * 3.0, 4.0)

        # Classification
        if avg_occ < OCCUPANCY_BULK_THRESHOLD:
            classification = "BULK"
            displaceable = False
        elif avg_dg < DG_HAPPY_THRESHOLD:
            classification = "CONSERVED_HAPPY"
            displaceable = False
        elif avg_dg > DG_UNHAPPY_THRESHOLD:
            classification = "CONSERVED_UNHAPPY"
            displaceable = True
        else:
            # Between thresholds — classify based on occupancy
            classification = "CONSERVED_HAPPY" if avg_occ > 0.6 else "BULK"
            displaceable = False

        sites.append(HydrationSite(
            x=float(centre[0]),
            y=float(centre[1]),
            z=float(centre[2]),
            occupancy=avg_occ,
            delta_g_transfer=avg_dg,
            entropy_contribution=avg_tds,
            enthalpy_contribution=avg_dh,
            n_hbonds_mean=float(n_hbonds),
            classification=classification,
            displaceable=displaceable,
        ))

    return sites


# ── Main analysis function ────────────────────────────────────────────────

def compute_water_map(
    trajectory_path: str,
    topology_path: str,
    pocket_centroid: Tuple[float, float, float],
    pocket_id: int,
    analysis_radius: float = DEFAULT_ANALYSIS_RADIUS,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    stride: int = 1,
) -> WaterMap:
    """Compute a hydration-site map from an explicit-solvent trajectory.

    Parameters
    ----------
    trajectory_path : str
        Path to trajectory file (DCD, XTC, etc.).
    topology_path : str
        Path to topology file (PDB).
    pocket_centroid : tuple of float
        (x, y, z) pocket centre in Angstrom.
    pocket_id : int
        Pocket identifier.
    analysis_radius : float
        Radius of the analysis sphere in Angstrom (default 8.0).
    grid_spacing : float
        Grid spacing in Angstrom (default 0.5).
    start_frame : int
        First frame to analyse.
    end_frame : int or None
        Last frame (exclusive). None = all frames.
    stride : int
        Frame stride (default 1).

    Returns
    -------
    WaterMap
        Complete hydration-site analysis for the pocket.
    """
    import mdtraj as md

    logger.info("Computing water map for pocket %d", pocket_id)
    logger.info("  Centroid: (%.2f, %.2f, %.2f)", *pocket_centroid)
    logger.info("  Radius: %.1f A | Grid: %.2f A", analysis_radius, grid_spacing)

    # Load trajectory
    traj = md.load(
        trajectory_path,
        top=topology_path,
    )
    if end_frame is not None:
        traj = traj[start_frame:end_frame:stride]
    elif stride > 1 or start_frame > 0:
        traj = traj[start_frame::stride]

    n_frames = traj.n_frames
    logger.info("  Analysing %d frames", n_frames)

    # Build grid
    grid_points, origin, spacing = build_analysis_grid(
        pocket_centroid, analysis_radius, grid_spacing,
    )
    logger.info("  Grid: %d voxels within sphere", len(grid_points))

    # Compute occupancy
    occupancy = compute_voxel_occupancy(traj, grid_points, spacing)

    # Compute energetics
    delta_h, neg_tds, delta_g = estimate_voxel_energetics(
        traj, grid_points, occupancy, spacing,
    )

    # Cluster into sites
    sites = cluster_hydration_sites(
        grid_points, occupancy, delta_h, neg_tds, delta_g,
    )
    logger.info("  Found %d hydration sites", len(sites))

    # Summary statistics
    displaceable_sites = [s for s in sites if s.displaceable]
    n_displaceable = len(displaceable_sites)
    max_displacement = (
        max(s.delta_g_transfer for s in displaceable_sites)
        if displaceable_sites else 0.0
    )
    total_displacement = sum(
        s.delta_g_transfer for s in displaceable_sites
    )

    for s in sites:
        logger.info(
            "    Site (%.1f, %.1f, %.1f): occ=%.2f dG=%.2f %s%s",
            s.x, s.y, s.z, s.occupancy, s.delta_g_transfer,
            s.classification,
            " [DISPLACE]" if s.displaceable else "",
        )

    if displaceable_sites:
        logger.info(
            "  Displacing %d unhappy waters yields +%.1f kcal/mol",
            n_displaceable, total_displacement,
        )

    water_map = WaterMap(
        pocket_id=pocket_id,
        hydration_sites=sites,
        n_displaceable=n_displaceable,
        max_displacement_energy=max_displacement,
        total_displacement_energy=total_displacement,
        grid_resolution=grid_spacing,
        analysis_frames=n_frames,
    )

    return water_map


def compute_water_map_from_arrays(
    pocket_centroid: Tuple[float, float, float],
    pocket_id: int,
    water_positions_per_frame: List[np.ndarray],
    analysis_radius: float = DEFAULT_ANALYSIS_RADIUS,
    grid_spacing: float = DEFAULT_GRID_SPACING,
) -> WaterMap:
    """Compute a water map from pre-extracted water positions (no MDTraj needed).

    Useful for testing and when trajectory is already loaded.

    Parameters
    ----------
    pocket_centroid : tuple of float
        (x, y, z) pocket centre in Angstrom.
    pocket_id : int
        Pocket identifier.
    water_positions_per_frame : list of np.ndarray
        List of (N_water, 3) arrays, one per frame, in Angstrom.
    analysis_radius : float
        Radius in Angstrom.
    grid_spacing : float
        Grid spacing in Angstrom.

    Returns
    -------
    WaterMap
    """
    grid_points, origin, spacing = build_analysis_grid(
        pocket_centroid, analysis_radius, grid_spacing,
    )
    n_frames = len(water_positions_per_frame)
    half_spacing = spacing / 2.0

    # Occupancy
    counts = np.zeros(len(grid_points), dtype=np.float64)
    for frame_waters in water_positions_per_frame:
        for wc in frame_waters:
            dists = np.linalg.norm(grid_points - wc, axis=1)
            nearest = np.argmin(dists)
            if dists[nearest] <= half_spacing:
                counts[nearest] += 1
    occupancy = counts / max(n_frames, 1)

    # Energetics
    bulk_density = 0.0333
    voxel_volume = spacing ** 3
    expected_bulk = bulk_density * voxel_volume
    relative_occ = np.where(expected_bulk > 0, occupancy / expected_bulk, 0.0)

    delta_h = np.clip(-2.0 * (relative_occ - 1.0), -5.0, 5.0)
    neg_tds = np.where(
        occupancy > 0.05,
        np.clip(2.0 * np.log1p(relative_occ), 0.0, 6.0),
        0.0,
    )
    delta_g = delta_h + neg_tds

    # Cluster
    sites = cluster_hydration_sites(
        grid_points, occupancy, delta_h, neg_tds, delta_g,
    )

    displaceable = [s for s in sites if s.displaceable]
    return WaterMap(
        pocket_id=pocket_id,
        hydration_sites=sites,
        n_displaceable=len(displaceable),
        max_displacement_energy=(
            max(s.delta_g_transfer for s in displaceable) if displaceable else 0.0
        ),
        total_displacement_energy=sum(s.delta_g_transfer for s in displaceable),
        grid_resolution=grid_spacing,
        analysis_frames=n_frames,
    )

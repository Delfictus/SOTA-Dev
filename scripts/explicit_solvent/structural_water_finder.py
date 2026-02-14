"""Conserved structural water identification for downstream docking.

Identifies water molecules that:
    - Occupy the same site in >80% of trajectory frames.
    - Form >=2 hydrogen bonds to protein atoms on average.

These waters must be PRESERVED as fixed waters in gpu_dock.py docking
(displacing them would cost free energy — they are CONSERVED_HAPPY).

Output
------
List of :class:`StructuralWater` positions + metadata, suitable for
injection into docking grids.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scripts.interfaces import HydrationSite

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────
DEFAULT_OCCUPANCY_THRESHOLD = 0.80
DEFAULT_HBOND_THRESHOLD = 2
DEFAULT_POCKET_RADIUS = 8.0     # Angstrom
CLUSTER_RADIUS = 1.0            # Angstrom — merge waters within 1 A
HBOND_DISTANCE_CUTOFF = 0.35    # nm (3.5 A) — donor-acceptor distance
HBOND_ANGLE_CUTOFF = 150        # degrees — D-H...A angle


# ── Result dataclass ──────────────────────────────────────────────────────

@dataclass
class StructuralWater:
    """A conserved water molecule identified from MD trajectory.

    Attributes
    ----------
    x, y, z : float
        Average position in Angstrom.
    occupancy : float
        Fraction of frames the site is occupied (0–1).
    mean_hbonds_to_protein : float
        Mean number of H-bonds to protein per frame.
    coordinating_residues : list of str
        Residue labels that H-bond to this water (e.g. ["ASP12", "THR35"]).
    b_factor_estimate : float
        Estimated B-factor from positional variance (A^2).
    """
    x: float
    y: float
    z: float
    occupancy: float
    mean_hbonds_to_protein: float
    coordinating_residues: List[str]
    b_factor_estimate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def distance_to(self, other: StructuralWater) -> float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def to_pdb_line(self, atom_index: int, residue_number: int) -> str:
        """Format as a PDB HETATM line for injection into docking."""
        return (
            f"HETATM{atom_index:5d}  O   HOH W{residue_number:4d}    "
            f"{self.x:8.3f}{self.y:8.3f}{self.z:8.3f}"
            f"  {self.occupancy:4.2f}{self.b_factor_estimate:6.2f}"
            f"           O  "
        )


# ── Core analysis ─────────────────────────────────────────────────────────

def find_structural_waters(
    trajectory_path: str,
    topology_path: str,
    pocket_centroid: Tuple[float, float, float],
    pocket_radius: float = DEFAULT_POCKET_RADIUS,
    occupancy_threshold: float = DEFAULT_OCCUPANCY_THRESHOLD,
    hbond_threshold: int = DEFAULT_HBOND_THRESHOLD,
    grid_spacing: float = 0.5,
) -> List[StructuralWater]:
    """Identify conserved structural waters from an MD trajectory.

    Parameters
    ----------
    trajectory_path : str
        Path to the trajectory file (DCD, XTC, etc.).
    topology_path : str
        Path to the topology file (PDB).
    pocket_centroid : tuple of float
        (x, y, z) pocket centre in Angstrom.
    pocket_radius : float
        Search radius around centroid in Angstrom (default 8.0).
    occupancy_threshold : float
        Minimum occupancy to qualify as structural (default 0.80).
    hbond_threshold : int
        Minimum mean H-bonds to protein (default 2).
    grid_spacing : float
        Grid spacing for density calculation in Angstrom (default 0.5).

    Returns
    -------
    list of StructuralWater
        Conserved water sites meeting both occupancy and H-bond criteria.
    """
    import mdtraj as md

    logger.info("Searching for structural waters near pocket")
    logger.info("  Centroid: (%.2f, %.2f, %.2f), radius: %.1f A",
                *pocket_centroid, pocket_radius)

    traj = md.load(trajectory_path, top=topology_path)
    n_frames = traj.n_frames
    logger.info("  Loaded %d frames", n_frames)

    centroid_nm = np.array(pocket_centroid) / 10.0
    radius_nm = pocket_radius / 10.0

    water_oxy = traj.topology.select("water and name O")
    protein_atoms = traj.topology.select("protein")

    if len(water_oxy) == 0:
        logger.warning("No water molecules found in trajectory")
        return []

    # Step 1: Grid-based density to find high-occupancy sites
    grid_spacing_nm = grid_spacing / 10.0
    n_bins = int(2 * radius_nm / grid_spacing_nm) + 1
    density = np.zeros((n_bins, n_bins, n_bins), dtype=np.float64)
    position_accum = np.zeros((n_bins, n_bins, n_bins, 3), dtype=np.float64)
    origin = centroid_nm - radius_nm

    for fi in range(n_frames):
        coords = traj.xyz[fi, water_oxy, :]
        dists = np.linalg.norm(coords - centroid_nm, axis=1)
        near_mask = dists < radius_nm
        near_coords = coords[near_mask]

        grid_idx = ((near_coords - origin) / grid_spacing_nm).astype(int)
        valid = np.all((grid_idx >= 0) & (grid_idx < n_bins), axis=1)
        for j, gi in enumerate(grid_idx[valid]):
            density[gi[0], gi[1], gi[2]] += 1
            position_accum[gi[0], gi[1], gi[2]] += near_coords[valid][j]

    occupancy = density / n_frames

    # Step 2: Find voxels above occupancy threshold
    high_occ_mask = occupancy >= occupancy_threshold
    high_occ_indices = np.argwhere(high_occ_mask)

    if len(high_occ_indices) == 0:
        logger.info("  No voxels above occupancy threshold %.2f", occupancy_threshold)
        return []

    # Step 3: Cluster nearby voxels into distinct sites
    sites_raw = []
    for idx in high_occ_indices:
        i, j, k = idx
        count = density[i, j, k]
        avg_pos = position_accum[i, j, k] / count  # nm
        occ = occupancy[i, j, k]
        sites_raw.append({
            "pos_nm": avg_pos,
            "pos_angstrom": avg_pos * 10.0,
            "occupancy": float(occ),
            "count": count,
        })

    # Merge sites within CLUSTER_RADIUS
    merged = _cluster_water_sites(sites_raw, CLUSTER_RADIUS)
    logger.info("  %d candidate sites after clustering (occ >= %.2f)",
                len(merged), occupancy_threshold)

    # Step 4: H-bond analysis for each candidate site
    structural_waters = []
    for site in merged:
        pos_nm = site["pos_nm"]
        hbond_info = _analyse_site_hbonds(
            traj, pos_nm, water_oxy, protein_atoms, radius_nm=0.15,
        )
        mean_hbonds = hbond_info["mean_hbonds"]
        coordinating = hbond_info["coordinating_residues"]
        b_factor = hbond_info.get("b_factor", 15.0)

        if mean_hbonds >= hbond_threshold:
            sw = StructuralWater(
                x=float(site["pos_angstrom"][0]),
                y=float(site["pos_angstrom"][1]),
                z=float(site["pos_angstrom"][2]),
                occupancy=site["occupancy"],
                mean_hbonds_to_protein=mean_hbonds,
                coordinating_residues=coordinating,
                b_factor_estimate=b_factor,
            )
            structural_waters.append(sw)
            logger.info(
                "    STRUCTURAL: (%.1f, %.1f, %.1f) occ=%.2f hb=%.1f [%s]",
                sw.x, sw.y, sw.z, sw.occupancy, sw.mean_hbonds_to_protein,
                ", ".join(sw.coordinating_residues),
            )
        else:
            logger.debug(
                "    Rejected site (%.1f, %.1f, %.1f): occ=%.2f but hb=%.1f < %d",
                site["pos_angstrom"][0], site["pos_angstrom"][1],
                site["pos_angstrom"][2],
                site["occupancy"], mean_hbonds, hbond_threshold,
            )

    logger.info("  Found %d structural waters", len(structural_waters))
    return structural_waters


def _cluster_water_sites(
    sites: List[Dict],
    cluster_radius: float,
) -> List[Dict]:
    """Merge raw voxel sites within `cluster_radius` Angstrom."""
    if not sites:
        return []

    merged = []
    used = [False] * len(sites)

    for i, s in enumerate(sites):
        if used[i]:
            continue
        cluster = [s]
        used[i] = True
        for j in range(i + 1, len(sites)):
            if used[j]:
                continue
            dist = np.linalg.norm(
                s["pos_angstrom"] - sites[j]["pos_angstrom"]
            )
            if dist <= cluster_radius:
                cluster.append(sites[j])
                used[j] = True

        # Occupancy-weighted average position
        total_count = sum(c["count"] for c in cluster)
        avg_nm = sum(c["pos_nm"] * c["count"] for c in cluster) / total_count
        avg_occ = max(c["occupancy"] for c in cluster)

        merged.append({
            "pos_nm": avg_nm,
            "pos_angstrom": avg_nm * 10.0,
            "occupancy": avg_occ,
            "count": total_count,
        })

    return merged


def _analyse_site_hbonds(
    trajectory: Any,
    site_pos_nm: np.ndarray,
    water_oxy_indices: np.ndarray,
    protein_indices: np.ndarray,
    radius_nm: float = 0.15,
) -> Dict[str, Any]:
    """Count H-bonds between a water site and protein.

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        Full trajectory.
    site_pos_nm : np.ndarray
        Water site position in nm.
    water_oxy_indices : np.ndarray
        Indices of water oxygen atoms.
    protein_indices : np.ndarray
        Indices of protein atoms.
    radius_nm : float
        Search radius to identify which water is at this site (nm).

    Returns
    -------
    dict
        mean_hbonds, coordinating_residues, b_factor.
    """
    import mdtraj as md

    n_frames = trajectory.n_frames
    hbond_counts = []
    residue_contacts = {}
    positions = []

    # For a sample of frames (every 10th for speed), find which water
    # is closest to the site and count its H-bonds to protein
    sample_stride = max(1, n_frames // 100)

    for fi in range(0, n_frames, sample_stride):
        coords = trajectory.xyz[fi, water_oxy_indices, :]
        dists = np.linalg.norm(coords - site_pos_nm, axis=1)
        nearest_idx = np.argmin(dists)

        if dists[nearest_idx] > radius_nm:
            hbond_counts.append(0)
            continue

        positions.append(coords[nearest_idx])
        water_atom_idx = water_oxy_indices[nearest_idx]

        # Find protein heavy atoms within H-bond distance
        protein_coords = trajectory.xyz[fi, protein_indices, :]
        water_coord = coords[nearest_idx]
        p_dists = np.linalg.norm(protein_coords - water_coord, axis=1)
        close_mask = p_dists < HBOND_DISTANCE_CUTOFF / 10.0  # A→nm

        n_hb = int(np.sum(close_mask))
        hbond_counts.append(min(n_hb, 4))

        # Track coordinating residues
        close_protein = protein_indices[close_mask]
        for atom_idx in close_protein:
            atom = trajectory.topology.atom(atom_idx)
            res_label = f"{atom.residue.name}{atom.residue.resSeq}"
            residue_contacts[res_label] = residue_contacts.get(res_label, 0) + 1

    mean_hbonds = float(np.mean(hbond_counts)) if hbond_counts else 0.0

    # B-factor from positional variance
    b_factor = 15.0
    if len(positions) > 1:
        pos_arr = np.array(positions) * 10.0  # nm → A
        variance = np.mean(np.var(pos_arr, axis=0))
        b_factor = 8 * np.pi**2 * variance / 3.0

    # Top coordinating residues (sorted by frequency)
    sorted_residues = sorted(
        residue_contacts.items(), key=lambda x: x[1], reverse=True,
    )
    coordinating = [r[0] for r in sorted_residues[:4]]

    return {
        "mean_hbonds": mean_hbonds,
        "coordinating_residues": coordinating,
        "b_factor": float(b_factor),
    }


# ── PDB output ────────────────────────────────────────────────────────────

def write_structural_waters_pdb(
    waters: List[StructuralWater],
    output_path: str,
    start_atom_index: int = 10001,
    start_residue_number: int = 1001,
) -> str:
    """Write structural waters as a PDB file for docking injection.

    Parameters
    ----------
    waters : list of StructuralWater
        Structural waters to write.
    output_path : str
        Path for the output PDB file.
    start_atom_index : int
        Starting HETATM atom serial number.
    start_residue_number : int
        Starting residue number for waters.

    Returns
    -------
    str
        Absolute path to the written file.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [f"REMARK   Structural waters from WT-6 explicit-solvent analysis"]
    lines.append(f"REMARK   {len(waters)} conserved water sites")
    for i, w in enumerate(waters):
        lines.append(w.to_pdb_line(
            start_atom_index + i,
            start_residue_number + i,
        ))
    lines.append("END")

    out.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %d structural waters to %s", len(waters), out)
    return str(out.resolve())


def structural_waters_from_hydration_sites(
    sites: List[HydrationSite],
    occupancy_threshold: float = DEFAULT_OCCUPANCY_THRESHOLD,
    hbond_threshold: float = DEFAULT_HBOND_THRESHOLD,
) -> List[StructuralWater]:
    """Convert HydrationSite objects to StructuralWater (no trajectory needed).

    Useful when a WaterMap has already been computed and you just need to
    extract the conserved sites for docking.

    Parameters
    ----------
    sites : list of HydrationSite
        Hydration sites from water map analysis.
    occupancy_threshold : float
        Minimum occupancy.
    hbond_threshold : float
        Minimum mean H-bonds.

    Returns
    -------
    list of StructuralWater
    """
    waters = []
    for s in sites:
        if (s.occupancy >= occupancy_threshold
                and s.n_hbonds_mean >= hbond_threshold
                and s.classification == "CONSERVED_HAPPY"):
            waters.append(StructuralWater(
                x=s.x,
                y=s.y,
                z=s.z,
                occupancy=s.occupancy,
                mean_hbonds_to_protein=s.n_hbonds_mean,
                coordinating_residues=[],  # Not available from grid analysis
                b_factor_estimate=15.0,
            ))
    return waters

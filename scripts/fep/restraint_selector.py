"""Boresch restraint selector — choose optimal restraint atoms for ABFE.

Uses PRISM pocket-lining residues and spike feature positions to select
the 3 protein atoms + 3 ligand atoms that define a Boresch orientational
restraint.  The selection follows the Aldeghi (2016) scoring method:
    - Prefer protein Ca atoms with low B-factor (rigid anchors)
    - Prefer ligand heavy atoms near high-intensity spike positions
    - Score candidates by collinearity penalty + distance stability
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ── Type aliases ──────────────────────────────────────────────────────────

Vec3 = Tuple[float, float, float]


@dataclass
class AtomInfo:
    """Lightweight atom descriptor for restraint selection."""
    index: int
    name: str
    residue_name: str
    residue_id: int
    position: Vec3
    b_factor: float = 0.0


@dataclass
class BoreshRestraint:
    """Boresch orientational restraint definition.

    Uses 3 protein atoms (p0, p1, p2) and 3 ligand atoms (l0, l1, l2).
    The 6 degrees of freedom restrained are:
        1 distance  (p2–l0)
        2 angles    (p1–p2–l0, p2–l0–l1)
        3 dihedrals (p0–p1–p2–l0, p1–p2–l0–l1, p2–l0–l1–l2)
    """
    protein_atoms: List[AtomInfo]   # [p0, p1, p2]
    ligand_atoms: List[AtomInfo]    # [l0, l1, l2]
    score: float = 0.0             # lower = better (Aldeghi score)
    distance_p2_l0: float = 0.0
    angle_p1_p2_l0: float = 0.0
    angle_p2_l0_l1: float = 0.0

    @property
    def protein_indices(self) -> List[int]:
        return [a.index for a in self.protein_atoms]

    @property
    def ligand_indices(self) -> List[int]:
        return [a.index for a in self.ligand_atoms]

    def to_dict(self) -> Dict:
        return {
            "protein_atoms": [
                {"index": a.index, "name": a.name,
                 "residue": f"{a.residue_name}{a.residue_id}"}
                for a in self.protein_atoms
            ],
            "ligand_atoms": [
                {"index": a.index, "name": a.name,
                 "residue": f"{a.residue_name}{a.residue_id}"}
                for a in self.ligand_atoms
            ],
            "score": round(self.score, 4),
            "distance_p2_l0": round(self.distance_p2_l0, 3),
            "angle_p1_p2_l0_deg": round(math.degrees(self.angle_p1_p2_l0), 1),
            "angle_p2_l0_l1_deg": round(math.degrees(self.angle_p2_l0_l1), 1),
        }


# ── Geometry helpers ──────────────────────────────────────────────────────

def _dist(a: Vec3, b: Vec3) -> float:
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _angle(a: Vec3, b: Vec3, c: Vec3) -> float:
    """Angle at b in radians (a-b-c)."""
    ba = tuple(ai - bi for ai, bi in zip(a, b))
    bc = tuple(ci - bi for ci, bi in zip(c, b))
    dot = sum(x * y for x, y in zip(ba, bc))
    mag_ba = math.sqrt(sum(x ** 2 for x in ba))
    mag_bc = math.sqrt(sum(x ** 2 for x in bc))
    if mag_ba < 1e-9 or mag_bc < 1e-9:
        return 0.0
    cos_val = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.acos(cos_val)


def _collinearity_penalty(angle_rad: float) -> float:
    """Penalty for angles near 0 or pi (collinear atoms).

    Ideal angles are near pi/2.  Penalty rises sharply near 0 or pi.
    Aldeghi uses a flat-bottom well at [pi/4, 3*pi/4].
    """
    ideal_min = math.pi / 4
    ideal_max = 3 * math.pi / 4
    if ideal_min <= angle_rad <= ideal_max:
        return 0.0
    if angle_rad < ideal_min:
        return (ideal_min - angle_rad) ** 2
    return (angle_rad - ideal_max) ** 2


# ── Selection logic ───────────────────────────────────────────────────────

def select_protein_atoms(
    all_protein_atoms: List[AtomInfo],
    lining_residue_ids: Sequence[int],
    n_candidates: int = 3,
) -> List[AtomInfo]:
    """Select the best 3 protein Ca atoms from pocket-lining residues.

    Selection criteria (Aldeghi method):
        - Must be Ca atoms from lining residues
        - Prefer low B-factor (more rigid)
        - Must not be collinear (angles checked downstream)

    Returns sorted by B-factor (lowest first), capped at ``n_candidates``.
    """
    lining_set = set(lining_residue_ids)
    ca_atoms = [
        a for a in all_protein_atoms
        if a.name == "CA" and a.residue_id in lining_set
    ]
    ca_atoms.sort(key=lambda a: a.b_factor)

    if len(ca_atoms) < n_candidates:
        logger.warning(
            "Only %d CA atoms in lining residues (need %d). "
            "Falling back to all protein CA atoms.",
            len(ca_atoms), n_candidates,
        )
        ca_atoms = [a for a in all_protein_atoms if a.name == "CA"]
        ca_atoms.sort(key=lambda a: a.b_factor)

    if len(ca_atoms) < n_candidates:
        raise ValueError(
            f"Need at least {n_candidates} protein CA atoms, "
            f"found {len(ca_atoms)}"
        )
    return ca_atoms[:n_candidates]


def select_ligand_atoms(
    ligand_heavy_atoms: List[AtomInfo],
    spike_positions: Sequence[Vec3],
    n_candidates: int = 3,
) -> List[AtomInfo]:
    """Select 3 ligand heavy atoms nearest to high-intensity spike positions.

    Args:
        ligand_heavy_atoms: All non-hydrogen ligand atoms.
        spike_positions: (x, y, z) of pharmacophore feature sites,
            ordered by descending intensity.
        n_candidates: Number of atoms to select.
    """
    if len(ligand_heavy_atoms) < n_candidates:
        raise ValueError(
            f"Need at least {n_candidates} ligand heavy atoms, "
            f"found {len(ligand_heavy_atoms)}"
        )

    selected: List[AtomInfo] = []
    used_indices: set = set()

    for spike_pos in spike_positions:
        if len(selected) >= n_candidates:
            break
        best_atom = None
        best_dist = float("inf")
        for atom in ligand_heavy_atoms:
            if atom.index in used_indices:
                continue
            d = _dist(atom.position, spike_pos)
            if d < best_dist:
                best_dist = d
                best_atom = atom
        if best_atom is not None:
            selected.append(best_atom)
            used_indices.add(best_atom.index)

    # Fill remaining from heaviest atoms not yet selected
    if len(selected) < n_candidates:
        remaining = [a for a in ligand_heavy_atoms if a.index not in used_indices]
        remaining.sort(key=lambda a: a.index)
        for atom in remaining:
            if len(selected) >= n_candidates:
                break
            selected.append(atom)

    return selected[:n_candidates]


def score_restraint(
    protein_atoms: List[AtomInfo],
    ligand_atoms: List[AtomInfo],
) -> BoreshRestraint:
    """Score a candidate Boresch restraint using Aldeghi criteria.

    Score components:
        - Distance penalty: deviation of p2-l0 from ideal 5-10 A range
        - Collinearity penalties for the two bond angles
        - B-factor penalty (prefer rigid protein atoms)
    """
    p0, p1, p2 = [a.position for a in protein_atoms]
    l0, l1, l2 = [a.position for a in ligand_atoms]

    dist_p2_l0 = _dist(p2, l0)
    angle_p1_p2_l0 = _angle(p1, p2, l0)
    angle_p2_l0_l1 = _angle(p2, l0, l1)

    # Distance penalty: ideal range 5-10 A
    if dist_p2_l0 < 5.0:
        dist_penalty = (5.0 - dist_p2_l0) ** 2
    elif dist_p2_l0 > 10.0:
        dist_penalty = (dist_p2_l0 - 10.0) ** 2
    else:
        dist_penalty = 0.0

    # Collinearity penalties for both bond angles
    col_penalty = (
        _collinearity_penalty(angle_p1_p2_l0)
        + _collinearity_penalty(angle_p2_l0_l1)
    )

    # B-factor contribution (normalized)
    max_bf = max(a.b_factor for a in protein_atoms) or 1.0
    bf_penalty = sum(a.b_factor / max_bf for a in protein_atoms) / 3.0

    total_score = dist_penalty + 10.0 * col_penalty + bf_penalty

    return BoreshRestraint(
        protein_atoms=list(protein_atoms),
        ligand_atoms=list(ligand_atoms),
        score=total_score,
        distance_p2_l0=dist_p2_l0,
        angle_p1_p2_l0=angle_p1_p2_l0,
        angle_p2_l0_l1=angle_p2_l0_l1,
    )


def select_boresch_restraint(
    protein_atoms: List[AtomInfo],
    ligand_heavy_atoms: List[AtomInfo],
    lining_residue_ids: Sequence[int],
    spike_positions: Sequence[Vec3],
    n_protein_candidates: int = 10,
    n_ligand_candidates: int = 6,
) -> BoreshRestraint:
    """Select the optimal Boresch restraint from candidate atoms.

    Generates candidate triplets from the top protein and ligand atoms,
    scores each, and returns the best (lowest score).

    Args:
        protein_atoms: All protein atoms (will filter to CA).
        ligand_heavy_atoms: Ligand heavy atoms with positions.
        lining_residue_ids: PRISM pocket-lining residue IDs.
        spike_positions: Pharmacophore feature positions (high-intensity first).
        n_protein_candidates: Number of top protein CA atoms to consider.
        n_ligand_candidates: Number of top ligand atoms to consider.

    Returns:
        The best-scoring BoreshRestraint.
    """
    p_candidates = select_protein_atoms(
        protein_atoms, lining_residue_ids, n_protein_candidates,
    )
    l_candidates = select_ligand_atoms(
        ligand_heavy_atoms, spike_positions, n_ligand_candidates,
    )

    best: Optional[BoreshRestraint] = None

    # Enumerate triplet combinations
    for i in range(len(p_candidates)):
        for j in range(len(p_candidates)):
            if j == i:
                continue
            for k in range(len(p_candidates)):
                if k in (i, j):
                    continue
                p_triplet = [p_candidates[i], p_candidates[j], p_candidates[k]]
                for a in range(len(l_candidates)):
                    for b in range(len(l_candidates)):
                        if b == a:
                            continue
                        for c in range(len(l_candidates)):
                            if c in (a, b):
                                continue
                            l_triplet = [
                                l_candidates[a], l_candidates[b], l_candidates[c],
                            ]
                            candidate = score_restraint(p_triplet, l_triplet)
                            if best is None or candidate.score < best.score:
                                best = candidate

    if best is None:
        raise ValueError("Could not find any valid restraint candidates")

    logger.info(
        "Selected Boresch restraint: score=%.3f, d(P2-L0)=%.2f A, "
        "protein=[%s], ligand=[%s]",
        best.score, best.distance_p2_l0,
        ", ".join(f"{a.residue_name}{a.residue_id}:{a.name}" for a in best.protein_atoms),
        ", ".join(f"{a.name}" for a in best.ligand_atoms),
    )
    return best

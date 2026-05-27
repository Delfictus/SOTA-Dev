"""Thermodynamic functional group decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from rdkit import Chem

ROLE_PRIORITY: tuple[str, ...] = (
    "LOCK_WEDGE",
    "COMPLEMENT_ANCHOR",
    "CLASH_DRIVER",
    "SHEAR_SENTINEL",
    "PHASE_PIVOT",
)


@dataclass(frozen=True)
class AtomThermoAnnotation:
    """Per-atom thermodynamic contribution proxy."""

    pi_complement: float = 0.0
    pi_clash: float = 0.0
    lock_geometry: float = 0.0
    shear_stress: float = 0.0
    phase_profile: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    channel_a_activation: float = 0.0
    channel_b_activation: float = 0.0
    consensus_resilience: float = 0.0


@dataclass(frozen=True)
class ThermodynamicFunctionalGroup:
    """Bridge-inclusive motif extracted by thermodynamic role."""

    smarts: str
    atom_indices: list[int]
    role: str
    bridge_atom_count: int
    pi_complement_sum: float
    pi_clash_sum: float
    lock_geometry_contribution: float
    shear_stress_mean: float
    phase_profile: np.ndarray
    consensus_resilience: float
    channel_a_activation: float
    channel_b_activation: float
    n_occurrences: int
    parent_smiles: list[str]


def classify_atom_roles(
    annotations: Mapping[int, AtomThermoAnnotation],
    *,
    shear_median: float | None = None,
    phase_std: float | None = None,
) -> dict[int, str]:
    """Classify atom roles from thermodynamic annotations."""

    if shear_median is None:
        shear_values = [annotation.shear_stress for annotation in annotations.values()]
        shear_median = float(np.median(shear_values)) if shear_values else 0.0
    if phase_std is None:
        deltas = [
            abs(annotation.phase_profile[4] - annotation.phase_profile[0])
            for annotation in annotations.values()
        ]
        phase_std = float(np.std(deltas)) if deltas else 0.0

    roles: dict[int, str] = {}
    for atom_idx, annotation in annotations.items():
        if annotation.lock_geometry > 0.0:
            roles[atom_idx] = "LOCK_WEDGE"
        elif annotation.pi_complement > 0.5:
            roles[atom_idx] = "COMPLEMENT_ANCHOR"
        elif annotation.pi_clash > 1.0:
            roles[atom_idx] = "CLASH_DRIVER"
        elif annotation.shear_stress > shear_median:
            roles[atom_idx] = "SHEAR_SENTINEL"
        elif abs(annotation.phase_profile[4] - annotation.phase_profile[0]) > phase_std:
            roles[atom_idx] = "PHASE_PIVOT"
        else:
            roles[atom_idx] = "NEUTRAL"
    return roles


def extract_tfg_with_neighborhood(
    mol: Chem.Mol,
    atom_roles: Mapping[int, str],
    annotations: Mapping[int, AtomThermoAnnotation] | None = None,
    *,
    parent_smiles: str = "",
    k_hops: int = 2,
) -> list[ThermodynamicFunctionalGroup]:
    """Extract TFGs with k-hop neighborhood expansion and neutral bridges."""

    if k_hops < 1:
        raise ValueError("k_hops must be >= 1")
    role_neighborhoods: dict[int, set[int]] = {}
    for atom_idx, role in atom_roles.items():
        if role == "NEUTRAL":
            continue
        role_neighborhoods[atom_idx] = get_k_hop_neighborhood(mol, atom_idx, k_hops)
    groups = merge_overlapping_by_role(role_neighborhoods, atom_roles)
    tfgs: list[ThermodynamicFunctionalGroup] = []
    for group_atoms, role in groups:
        bridge_atoms = find_bridge_atoms(mol, group_atoms, atom_roles)
        full_subgraph = set(group_atoms) | bridge_atoms
        if not full_subgraph:
            continue
        smarts = subgraph_to_smarts(mol, full_subgraph)
        tfgs.append(
            _build_group(
                smarts=smarts,
                atom_indices=sorted(full_subgraph),
                role=role,
                bridge_atom_count=len(bridge_atoms),
                annotations=annotations or {},
                parent_smiles=parent_smiles,
            )
        )
    return tfgs


def get_k_hop_neighborhood(mol: Chem.Mol, atom_idx: int, k: int) -> set[int]:
    """BFS to k hops from atom_idx."""

    if atom_idx < 0 or atom_idx >= mol.GetNumAtoms():
        raise IndexError(f"atom index out of range: {atom_idx}")
    visited = {atom_idx}
    frontier = {atom_idx}
    for _ in range(k):
        next_frontier: set[int] = set()
        for idx in frontier:
            atom = mol.GetAtomWithIdx(idx)
            for neighbor in atom.GetNeighbors():
                nidx = int(neighbor.GetIdx())
                if nidx not in visited:
                    visited.add(nidx)
                    next_frontier.add(nidx)
        frontier = next_frontier
    return visited


def merge_overlapping_by_role(
    role_neighborhoods: Mapping[int, set[int]],
    atom_roles: Mapping[int, str],
) -> list[tuple[set[int], str]]:
    """Union overlapping neighborhoods for atoms of the same role."""

    atoms_by_role: dict[str, list[int]] = {}
    for atom_idx in role_neighborhoods:
        atoms_by_role.setdefault(atom_roles[atom_idx], []).append(atom_idx)

    groups: list[tuple[set[int], str]] = []
    for role, atom_indices in atoms_by_role.items():
        remaining = list(atom_indices)
        while remaining:
            seed = remaining.pop()
            merged = set(role_neighborhoods[seed])
            changed = True
            while changed:
                changed = False
                keep: list[int] = []
                for atom_idx in remaining:
                    if merged & role_neighborhoods[atom_idx]:
                        merged |= role_neighborhoods[atom_idx]
                        changed = True
                    else:
                        keep.append(atom_idx)
                remaining = keep
            groups.append((merged, role))
    return groups


def find_bridge_atoms(mol: Chem.Mol, group_atoms: set[int], atom_roles: Mapping[int, str]) -> set[int]:
    """Return neutral atoms lying on shortest paths between role-bearing atoms."""

    role_atoms = sorted(atom for atom in group_atoms if atom_roles.get(atom) != "NEUTRAL")
    bridge_atoms: set[int] = set()
    for i, start in enumerate(role_atoms):
        for end in role_atoms[i + 1 :]:
            path = Chem.rdmolops.GetShortestPath(mol, int(start), int(end))
            for atom_idx in path:
                idx = int(atom_idx)
                if atom_roles.get(idx) == "NEUTRAL":
                    bridge_atoms.add(idx)
    return bridge_atoms


def subgraph_to_smarts(mol: Chem.Mol, atom_indices: set[int]) -> str:
    """Convert atom subgraph to SMARTS."""

    if not atom_indices:
        raise ValueError("cannot build SMARTS for empty subgraph")
    return str(Chem.MolFragmentToSmarts(mol, atomsToUse=sorted(atom_indices)))


def _build_group(
    *,
    smarts: str,
    atom_indices: Sequence[int],
    role: str,
    bridge_atom_count: int,
    annotations: Mapping[int, AtomThermoAnnotation],
    parent_smiles: str,
) -> ThermodynamicFunctionalGroup:
    selected = [annotations.get(atom_idx, AtomThermoAnnotation()) for atom_idx in atom_indices]
    phase = np.array([annotation.phase_profile for annotation in selected], dtype=np.float64)
    if phase.size == 0:
        phase_profile = np.zeros(5, dtype=np.float64)
    else:
        phase_profile = phase.sum(axis=0)
    return ThermodynamicFunctionalGroup(
        smarts=smarts,
        atom_indices=list(atom_indices),
        role=role,
        bridge_atom_count=bridge_atom_count,
        pi_complement_sum=sum(annotation.pi_complement for annotation in selected),
        pi_clash_sum=sum(annotation.pi_clash for annotation in selected),
        lock_geometry_contribution=sum(annotation.lock_geometry for annotation in selected),
        shear_stress_mean=float(np.mean([annotation.shear_stress for annotation in selected])) if selected else 0.0,
        phase_profile=phase_profile,
        consensus_resilience=float(np.mean([annotation.consensus_resilience for annotation in selected])) if selected else 0.0,
        channel_a_activation=float(np.mean([annotation.channel_a_activation for annotation in selected])) if selected else 0.0,
        channel_b_activation=float(np.mean([annotation.channel_b_activation for annotation in selected])) if selected else 0.0,
        n_occurrences=1,
        parent_smiles=[parent_smiles] if parent_smiles else [],
    )

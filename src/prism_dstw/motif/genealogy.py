"""Cross-epoch motif genealogy and spatial receptor heatmaps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS

from prism_dstw.motif.registry import MotifEntry, MotifRegistry


@dataclass(frozen=True)
class MotifGenealogy:
    """Cross-epoch lineage classification for one motif."""

    motif_id: str
    epoch_born: int
    epoch_died: int | None
    parent_motif_id: str | None
    child_motif_ids: list[str]
    epoch_stats: dict[int, dict[str, float]]
    lineage_type: str


def compute_genealogy(
    registry_epoch_a: MotifRegistry,
    registry_epoch_b: MotifRegistry,
    *,
    mcs_overlap_threshold: float = 0.75,
) -> list[MotifGenealogy]:
    """Compute formal motif parent-child relationships across epochs."""

    motifs_a = {entry.motif_id: entry for entry in registry_epoch_a.all()}
    motifs_b = {entry.motif_id: entry for entry in registry_epoch_b.all()}
    output: list[MotifGenealogy] = []
    for motif_id, motif_b in motifs_b.items():
        if motif_id in motifs_a:
            output.append(
                MotifGenealogy(
                    motif_id=motif_id,
                    epoch_born=motifs_a[motif_id].first_seen_epoch,
                    epoch_died=None,
                    parent_motif_id=None,
                    child_motif_ids=[],
                    epoch_stats={},
                    lineage_type="PERSISTENT",
                )
            )
            continue
        parents = [
            motif_a.motif_id
            for motif_a in motifs_a.values()
            if is_parent_motif(motif_a, motif_b, overlap_threshold=mcs_overlap_threshold)
        ]
        lineage = "NOVEL"
        if len(parents) == 1:
            lineage = "EVOLVED"
        elif len(parents) > 1:
            lineage = "MERGED"
        output.append(
            MotifGenealogy(
                motif_id=motif_id,
                epoch_born=motif_b.first_seen_epoch,
                epoch_died=None,
                parent_motif_id=parents[0] if parents else None,
                child_motif_ids=[],
                epoch_stats={},
                lineage_type=lineage,
            )
        )
    for motif_id, motif_a in motifs_a.items():
        if motif_id not in motifs_b:
            output.append(
                MotifGenealogy(
                    motif_id=motif_id,
                    epoch_born=motif_a.first_seen_epoch,
                    epoch_died=max(motif_a.last_seen_epoch + 1, 0),
                    parent_motif_id=None,
                    child_motif_ids=[],
                    epoch_stats={},
                    lineage_type="TRANSIENT",
                )
            )
    return output


def is_parent_motif(parent: MotifEntry, child: MotifEntry, *, overlap_threshold: float = 0.75) -> bool:
    """Return true only for >75% MCS overlap and higher child lock contribution."""

    parent_mol = Chem.MolFromSmarts(parent.canonical_smarts)
    child_mol = Chem.MolFromSmarts(child.canonical_smarts)
    if parent_mol is None or child_mol is None:
        return False
    min_atoms = min(parent_mol.GetNumAtoms(), child_mol.GetNumAtoms())
    if min_atoms == 0:
        return False
    result = rdFMCS.FindMCS([parent_mol, child_mol], timeout=5)
    overlap = float(result.numAtoms) / float(min_atoms)
    parent_lock = float(parent.lock_geometry_contribution or 0.0)
    child_lock = float(child.lock_geometry_contribution or 0.0)
    return overlap > overlap_threshold and child_lock > parent_lock


def compute_motif_receptor_heatmap(
    motif_entry: MotifEntry,
    candidates: Sequence[Chem.Mol],
    candidate_coords: Sequence[np.ndarray],
    grid_config: Mapping[str, object],
) -> tuple[dict[int, int], float]:
    """Compute voxel footprint and spatial consistency for a motif."""

    if len(candidates) != len(candidate_coords):
        raise ValueError("candidates and coordinates length mismatch")
    pattern = Chem.MolFromSmarts(motif_entry.canonical_smarts)
    if pattern is None:
        return {}, 0.0
    voxel_counts: dict[int, int] = {}
    total = 0
    for mol, coords in zip(candidates, candidate_coords):
        matches = mol.GetSubstructMatches(pattern)
        for match in matches:
            for atom_idx in match:
                if int(atom_idx) >= coords.shape[0]:
                    continue
                voxel_idx = xyz_to_voxel(coords[int(atom_idx)], grid_config)
                if voxel_idx is None:
                    continue
                voxel_counts[voxel_idx] = voxel_counts.get(voxel_idx, 0) + 1
                total += 1
    if total == 0:
        return voxel_counts, 0.0
    return voxel_counts, 1.0 - (len(voxel_counts) / float(total))


def xyz_to_voxel(xyz: np.ndarray, grid_config: Mapping[str, object]) -> int | None:
    """Map xyz to a flattened voxel index."""

    origin = np.asarray(grid_config.get("origin", grid_config.get("origin_xyz_angstrom", [0.0, 0.0, 0.0])), dtype=np.float64)
    spacing = _float(grid_config.get("spacing", grid_config.get("spacing_angstrom", 1.0)), 1.0)
    dims_raw = grid_config.get("dims", [_int(grid_config.get("nx", 1), 1), _int(grid_config.get("ny", 1), 1), _int(grid_config.get("nz", 1), 1)])
    dims = [int(value) for value in dims_raw] if isinstance(dims_raw, list | tuple) else [1, 1, 1]
    idx = np.floor((np.asarray(xyz, dtype=np.float64) - origin) / spacing).astype(int)
    if any(value < 0 for value in idx) or any(idx[i] >= dims[i] for i in range(3)):
        return None
    return int(idx[0] + idx[1] * dims[0] + idx[2] * dims[0] * dims[1])


def _float(value: Any, default: float) -> float:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int | float | str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _int(value: Any, default: int) -> int:
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int | float | str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default

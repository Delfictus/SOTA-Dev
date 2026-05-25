#!/usr/bin/env python3
"""Audit the oracle's molecule-relative Z-proxy against the residue lock mask."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

from lock_mask_forensics_utils import (
    atom_indices_in_lock_mask,
    centroid_for_points,
    euclidean_distance,
    load_lock_mask,
    load_sdf_coordinates,
    load_topology_residues,
    nearest_residue,
    residue_subset,
    voxel_center,
    voxel_indices_for_atoms,
    z_proxy_atom_indices,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SDF = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/ALENI-PARENT_6XOX_frame_o3a_relaxed.sdf"
DEFAULT_LOCK_MASK = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/lock_region_mask.json"
DEFAULT_OUTPUT = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/z_proxy_voxel_audit.json"
DEFAULT_LOCK_RESIDUES = "245-260,301-340,330-360,370-400,390-410"

JsonObject: TypeAlias = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", type=Path, default=DEFAULT_SDF)
    parser.add_argument("--lock-mask", type=Path, default=DEFAULT_LOCK_MASK)
    parser.add_argument("--lock-residues", type=str, default=DEFAULT_LOCK_RESIDUES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def parse_ranges(value: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            ranges.append((int(start_text), int(end_text)))
        else:
            residue_id = int(part)
            ranges.append((residue_id, residue_id))
    if not ranges:
        raise ValueError("expected at least one residue range")
    return ranges


def atomic_write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def main() -> int:
    args = parse_args()
    lock_ranges = parse_ranges(str(args.lock_residues))
    lock_mask = load_lock_mask(Path(args.lock_mask))
    topology_residues = load_topology_residues(lock_mask.grid.topology_path)
    lock_residues = residue_subset(topology_residues, lock_ranges)
    coords = load_sdf_coordinates(Path(args.sdf))

    z_atoms = z_proxy_atom_indices(coords)
    mask_atoms = atom_indices_in_lock_mask(coords, lock_mask)
    z_voxels = voxel_indices_for_atoms(coords, lock_mask.grid, z_atoms)
    mask_voxels = voxel_indices_for_atoms(coords, lock_mask.grid, mask_atoms)

    z_points = [coords[idx] for idx in z_atoms]
    z_centroid = centroid_for_points(z_points)
    z_voxel_centroid = centroid_for_points(voxel_center(voxel, lock_mask.grid) for voxel in z_voxels)
    z_nearest_residue, z_nearest_distance = nearest_residue(z_centroid, topology_residues)

    lock_voxel_centroid = centroid_for_points(
        voxel_center(voxel, lock_mask.grid) for voxel in lock_mask.lock_voxel_indices
    )
    lock_nearest_residue, lock_nearest_distance = nearest_residue(lock_voxel_centroid, topology_residues)
    lock_atom_hits_centroid = (
        centroid_for_points(coords[idx] for idx in mask_atoms) if mask_atoms else None
    )

    jaccard_denominator = len(set(z_voxels).union(mask_voxels))
    jaccard = (
        len(set(z_voxels).intersection(mask_voxels)) / float(jaccard_denominator)
        if jaccard_denominator
        else 0.0
    )
    min_lock_residue_distance = min(
        euclidean_distance(coord, residue.xyz) for coord in coords for residue in lock_residues
    )

    payload: JsonObject = {
        "schema_version": "PRISM.z_proxy_voxel_audit.v1",
        "epistemic_class": "DERIVED",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "input_sdf": str(Path(args.sdf)),
        "lock_mask": str(Path(args.lock_mask)),
        "z_proxy": {
            "atom_count": len(z_atoms),
            "voxel_count": len(z_voxels),
            "atom_centroid_xyz": list(z_centroid),
            "voxel_centroid_xyz": list(z_voxel_centroid),
            "nearest_residue": {
                "residue_name": z_nearest_residue.residue_name,
                "residue_id": z_nearest_residue.residue_id,
                "distance_A": z_nearest_distance,
            },
        },
        "residue_lock_mask": {
            "atom_hit_count": len(mask_atoms),
            "voxel_hit_count": len(mask_voxels),
            "global_voxel_centroid_xyz": list(lock_voxel_centroid),
            "nearest_residue": {
                "residue_name": lock_nearest_residue.residue_name,
                "residue_id": lock_nearest_residue.residue_id,
                "distance_A": lock_nearest_distance,
            },
            "atom_hit_centroid_xyz": list(lock_atom_hits_centroid) if lock_atom_hits_centroid is not None else None,
        },
        "comparison": {
            "voxel_jaccard": jaccard,
            "min_scaffold_atom_to_lock_residue_ca_distance_A": min_lock_residue_distance,
            "determination": (
                "Z-proxy labels the bottom of the ligand pose, while the residue lock mask labels "
                "the receptor's intracellular face. These are not the same spatial region."
            ),
        },
    }
    atomic_write_json(Path(args.output), payload)
    print(
        "z_proxy_audit_complete "
        f"n_z_proxy_lock_voxels={len(z_voxels)} "
        f"z_proxy_centroid={z_voxel_centroid} "
        f"nearest_structure_region={z_nearest_residue.residue_name}{z_nearest_residue.residue_id} "
        f"mask_atom_hits={len(mask_atoms)} "
        f"mask_voxel_centroid={lock_voxel_centroid} "
        f"voxel_jaccard={jaccard:.6f} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

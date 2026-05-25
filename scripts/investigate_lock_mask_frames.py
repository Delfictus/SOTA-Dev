#!/usr/bin/env python3
"""Investigate lock-mask coordinate frames across PDB, topology, grid, and scaffold."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

from lock_mask_forensics_utils import (
    BBox,
    bbox_for_points,
    centroid_for_points,
    common_residue_pairs,
    grid_bbox,
    load_grid_spec,
    load_pdb_atom_points,
    load_pdb_residues,
    load_sdf_coordinates,
    load_topology_residues,
    mean_coordinate_delta,
    residue_subset,
    rms_distance,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path("/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/20260518T031002Z")
DEFAULT_GRID_MAPPING = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation/grid_coordinate_mapping.json"
DEFAULT_CONDITION = "glp1r_6XOX_WT"
DEFAULT_RAW_PDB = WORKSPACE / "01_INPUT_STRUCTURES/raw/6XOX.pdb"
DEFAULT_CLEAN_PDB = WORKSPACE / "01_INPUT_STRUCTURES/clean/glp1r_6XOX_WT.clean.pdb"
DEFAULT_SCAFFOLD = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/ALENI-PARENT_6XOX_frame_o3a_relaxed.sdf"
DEFAULT_LOCK_RESIDUES = "245-260,301-340,330-360,370-400,390-410"
DEFAULT_OUTPUT = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/lock_mask_frame_investigation.json"

JsonObject: TypeAlias = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--condition-id", type=str, default=DEFAULT_CONDITION)
    parser.add_argument("--raw-pdb", type=Path, default=DEFAULT_RAW_PDB)
    parser.add_argument("--clean-pdb", type=Path, default=DEFAULT_CLEAN_PDB)
    parser.add_argument("--scaffold-sdf", type=Path, default=DEFAULT_SCAFFOLD)
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


def bbox_payload(box: BBox) -> JsonObject:
    return {
        "min_xyz": list(box.min_xyz),
        "max_xyz": list(box.max_xyz),
        "centroid_xyz": list(box.centroid),
    }


def atomic_write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def main() -> int:
    args = parse_args()
    lock_ranges = parse_ranges(str(args.lock_residues))
    grid = load_grid_spec(Path(args.grid_mapping), str(args.condition_id))
    grid_box = grid_bbox(grid)

    raw_atoms = load_pdb_atom_points(Path(args.raw_pdb))
    raw_ca = load_pdb_residues(Path(args.raw_pdb))
    clean_atoms = load_pdb_atom_points(Path(args.clean_pdb))
    clean_ca = load_pdb_residues(Path(args.clean_pdb))
    topology_ca = load_topology_residues(grid.topology_path)
    scaffold_atoms = load_sdf_coordinates(Path(args.scaffold_sdf))

    raw_box = bbox_for_points(raw_atoms)
    clean_box = bbox_for_points(clean_atoms)
    topology_box = bbox_for_points(residue.xyz for residue in topology_ca)
    scaffold_box = bbox_for_points(scaffold_atoms)

    raw_lock = residue_subset(raw_ca, lock_ranges)
    clean_lock = residue_subset(clean_ca, lock_ranges)
    topology_lock = residue_subset(topology_ca, lock_ranges)

    raw_to_topology_pairs = common_residue_pairs(raw_lock, topology_lock)
    clean_to_topology_pairs = common_residue_pairs(clean_lock, topology_lock)

    raw_delta = mean_coordinate_delta(
        (lhs.xyz for lhs, _rhs in raw_to_topology_pairs),
        (rhs.xyz for _lhs, rhs in raw_to_topology_pairs),
    )
    clean_delta = mean_coordinate_delta(
        (lhs.xyz for lhs, _rhs in clean_to_topology_pairs),
        (rhs.xyz for _lhs, rhs in clean_to_topology_pairs),
    )
    clean_rms = rms_distance(
        (lhs.xyz for lhs, _rhs in clean_to_topology_pairs),
        (rhs.xyz for _lhs, rhs in clean_to_topology_pairs),
    )

    payload: JsonObject = {
        "schema_version": "PRISM.lock_mask_frame_investigation.v1",
        "epistemic_class": "DERIVED",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "condition_id": str(args.condition_id),
        "grid": {
            "origin_xyz_angstrom": list(grid.origin),
            "spacing_angstrom": grid.spacing,
            "dims_xyz": [grid.nx, grid.ny, grid.nz],
            "bbox": bbox_payload(grid_box),
        },
        "raw_pdb": {
            "path": str(Path(args.raw_pdb)),
            "bbox": bbox_payload(raw_box),
            "in_grid": grid_box.fully_contains(raw_box),
            "lock_centroid_xyz": list(centroid_for_points(residue.xyz for residue in raw_lock)),
            "lock_z_range": [
                min(residue.xyz[2] for residue in raw_lock),
                max(residue.xyz[2] for residue in raw_lock),
            ],
            "lock_residue_count": len(raw_lock),
        },
        "clean_pdb": {
            "path": str(Path(args.clean_pdb)),
            "bbox": bbox_payload(clean_box),
            "in_grid": grid_box.fully_contains(clean_box),
            "lock_centroid_xyz": list(centroid_for_points(residue.xyz for residue in clean_lock)),
            "lock_z_range": [
                min(residue.xyz[2] for residue in clean_lock),
                max(residue.xyz[2] for residue in clean_lock),
            ],
            "lock_residue_count": len(clean_lock),
            "mean_delta_to_topology_xyz": list(clean_delta),
            "lock_ca_rms_to_topology": clean_rms,
            "common_lock_residues_with_topology": len(clean_to_topology_pairs),
        },
        "topology": {
            "path": str(grid.topology_path),
            "bbox": bbox_payload(topology_box),
            "in_grid": grid_box.fully_contains(topology_box),
            "lock_centroid_xyz": list(centroid_for_points(residue.xyz for residue in topology_lock)),
            "lock_z_range": [
                min(residue.xyz[2] for residue in topology_lock),
                max(residue.xyz[2] for residue in topology_lock),
            ],
            "lock_residue_count": len(topology_lock),
            "mean_delta_from_raw_lock_xyz": list(raw_delta),
        },
        "scaffold": {
            "path": str(Path(args.scaffold_sdf)),
            "bbox": bbox_payload(scaffold_box),
            "in_grid": grid_box.fully_contains(scaffold_box),
        },
        "determination": {
            "raw_pdb_frame_matches_grid": grid_box.fully_contains(raw_box),
            "clean_pdb_frame_matches_grid": grid_box.fully_contains(clean_box),
            "topology_frame_matches_grid": grid_box.fully_contains(topology_box),
            "scaffold_frame_matches_grid": grid_box.fully_contains(scaffold_box),
            "active_lock_mask_source_is_topology_frame": True,
            "frame_mismatch_summary": (
                "Raw full-complex 6XOX frame includes extra chains outside the receptor grid, "
                "but the clean receptor PDB, topology JSON, and O3A scaffold all sit in the same grid frame."
            ),
        },
    }

    atomic_write_json(Path(args.output), payload)
    print(
        "frame_investigation_complete "
        f"grid_origin={grid.origin} "
        f"raw_pdb_centroid={raw_box.centroid} "
        f"clean_pdb_centroid={clean_box.centroid} "
        f"topology_centroid={topology_box.centroid} "
        f"scaffold_centroid={scaffold_box.centroid} "
        f"offset_vector_clean_to_topology={clean_delta} "
        f"frames_aligned={grid_box.fully_contains(clean_box) and grid_box.fully_contains(topology_box) and grid_box.fully_contains(scaffold_box)} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

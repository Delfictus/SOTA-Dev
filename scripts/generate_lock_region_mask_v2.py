#!/usr/bin/env python3
"""Generate a clean-PDB-backed lock region mask in grid coordinates."""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

from lock_mask_forensics_utils import (
    GridSpec,
    load_grid_spec,
    load_lock_mask,
    load_pdb_residues,
    residue_subset,
    voxel_center,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path("/media/diddy/PRISM-LBS/prism-glp1r-aleniglipron-workspace/20260518T031002Z")
DEFAULT_GRID_MAPPING = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_0_manual_emulation/grid_coordinate_mapping.json"
DEFAULT_CONDITION = "glp1r_6XOX_WT"
DEFAULT_CLEAN_PDB = WORKSPACE / "01_INPUT_STRUCTURES/clean/glp1r_6XOX_WT.clean.pdb"
DEFAULT_PRIOR_MASK = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/lock_region_mask.json"
DEFAULT_OUTPUT = REPO_ROOT / "campaigns/glp1r_aleniglipron/track_a_generative/lock_region_mask_v2.json"
DEFAULT_LOCK_RESIDUES = "245-260,301-340,330-360,370-400,390-410"

JsonObject: TypeAlias = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--condition-id", type=str, default=DEFAULT_CONDITION)
    parser.add_argument("--clean-pdb", type=Path, default=DEFAULT_CLEAN_PDB)
    parser.add_argument("--prior-mask", type=Path, default=DEFAULT_PRIOR_MASK)
    parser.add_argument("--lock-residues", type=str, default=DEFAULT_LOCK_RESIDUES)
    parser.add_argument("--radius", type=float, default=6.0)
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


def build_mask(grid: GridSpec, coords: list[tuple[float, float, float]], radius: float) -> set[int]:
    radius_sq = radius * radius
    shell = int(math.ceil(radius / grid.spacing)) + 1
    mask: set[int] = set()
    for xyz in coords:
        center_ix = math.floor((xyz[0] - grid.origin[0]) / grid.spacing)
        center_iy = math.floor((xyz[1] - grid.origin[1]) / grid.spacing)
        center_iz = math.floor((xyz[2] - grid.origin[2]) / grid.spacing)
        for ix in range(max(0, center_ix - shell), min(grid.nx, center_ix + shell + 1)):
            for iy in range(max(0, center_iy - shell), min(grid.ny, center_iy + shell + 1)):
                for iz in range(max(0, center_iz - shell), min(grid.nz, center_iz + shell + 1)):
                    center = voxel_center(iz * (grid.nx * grid.ny) + iy * grid.nx + ix, grid)
                    distance_sq = sum((center[axis] - xyz[axis]) ** 2 for axis in range(3))
                    if distance_sq <= radius_sq:
                        mask.add(int(iz * grid.nx * grid.ny + iy * grid.nx + ix))
    return mask


def atomic_write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def main() -> int:
    args = parse_args()
    grid = load_grid_spec(Path(args.grid_mapping), str(args.condition_id))
    lock_ranges = parse_ranges(str(args.lock_residues))
    clean_residues = residue_subset(load_pdb_residues(Path(args.clean_pdb)), lock_ranges)
    coords = [residue.xyz for residue in clean_residues]
    current_mask = load_lock_mask(Path(args.prior_mask))
    v2_voxels = build_mask(grid, coords, float(args.radius))
    overlap = len(v2_voxels.intersection(current_mask.lock_voxel_indices))
    union = len(v2_voxels.union(current_mask.lock_voxel_indices))
    jaccard = overlap / float(union) if union else 0.0

    payload: JsonObject = {
        "schema_version": "PRISM.lock_region_mask.v2",
        "epistemic_class": "DERIVED",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "condition_id": str(args.condition_id),
        "source_pdb": str(Path(args.clean_pdb)),
        "source_topology": str(grid.topology_path),
        "source_frame": "clean_receptor_pdb_grid_aligned",
        "lock_residue_ranges": str(args.lock_residues),
        "radius_A": float(args.radius),
        "n_lock_voxels": len(v2_voxels),
        "agreement_with_prior_mask_jaccard": jaccard,
        "grid": {
            "condition_id": grid.condition_id,
            "nx": grid.nx,
            "ny": grid.ny,
            "nz": grid.nz,
            "origin_xyz_angstrom": list(grid.origin),
            "spacing_angstrom": grid.spacing,
        },
        "lock_voxel_indices": sorted(v2_voxels),
    }
    atomic_write_json(Path(args.output), payload)
    print(
        "lock_mask_switch "
        "version=v2_clean_pdb_grid_aligned "
        f"n_lock_voxels={len(v2_voxels)} "
        f"agreement_with_prior={jaccard:.6f} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

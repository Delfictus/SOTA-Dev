#!/usr/bin/env python3
"""Generate scaffold-bound thermodynamic signal grids from aligned scaffold poses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
TRACK_A = CAMPAIGN_DIR / "track_a_generative"
DEFAULT_WT_GRID = CAMPAIGN_DIR / "integrated_spike_events/n80_full_scale/signal_grid_variance_channel.parquet"
DEFAULT_GRID_MAPPING = CAMPAIGN_DIR / "track_0_manual_emulation/grid_coordinate_mapping.json"
DEFAULT_O3A_MANIFEST = TRACK_A / "competitor_scaffold_o3a_manifest.json"
DEFAULT_ALENI_SDF = TRACK_A / "ALENI-PARENT_6XOX_frame_o3a_relaxed.sdf"
DEFAULT_GRID_DIR = TRACK_A / "scaffold_bound/grids"
DEFAULT_MANIFEST = TRACK_A / "scaffold_bound/scaffold_bound_grid_manifest.json"
DEFAULT_REPORT = TRACK_A / "scaffold_bound/scaffold_bound_grid_generation_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wt-grid", type=Path, default=DEFAULT_WT_GRID)
    parser.add_argument("--wt-condition", default="glp1r_6XOX_WT")
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_GRID_MAPPING)
    parser.add_argument("--o3a-manifest", type=Path, default=DEFAULT_O3A_MANIFEST)
    parser.add_argument("--aleni-sdf", type=Path, default=DEFAULT_ALENI_SDF)
    parser.add_argument("--grid-dir", type=Path, default=DEFAULT_GRID_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--include-hydrogens", action="store_true")
    parser.add_argument("--proximal-radius-a", type=float, default=3.0)
    parser.add_argument("--response-radius-a", type=float, default=6.0)
    return parser.parse_args()


def parse_sdf_atoms(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 4:
        raise ValueError(f"SDF has no counts line: {path}")
    atom_count = int(lines[3][0:3])
    atoms: list[dict[str, Any]] = []
    for line in lines[4 : 4 + atom_count]:
        atoms.append(
            {
                "x": float(line[0:10]),
                "y": float(line[10:20]),
                "z": float(line[20:30]),
                "element": line[31:34].strip() or "C",
            }
        )
    return atoms


def scaffold_entries(o3a_manifest: Path, aleni_sdf: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {
        "ALENI": {
            "scaffold_id": "ALENI",
            "scaffold_sdf": str(aleni_sdf),
            "o3a_score": 0.0,
        }
    }
    manifest = json.loads(o3a_manifest.read_text(encoding="utf-8"))
    for item in manifest.get("outputs", []):
        ligand_id = str(item.get("ligand_id", "")).upper()
        scaffold_id = "DANU" if "DANU" in ligand_id else "ORFOR" if "ORFOR" in ligand_id else ""
        if not scaffold_id:
            continue
        entries[scaffold_id] = {
            "scaffold_id": scaffold_id,
            "scaffold_sdf": str(resolve_path(Path(str(item["output_path"])))),
            "o3a_score": float(item.get("o3a_score", 0.0)),
        }
    return entries


def resolve_path(path: Path) -> Path:
    if path.is_absolute() or path.exists():
        return path
    return REPO_ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_grid_geometry(path: Path, condition_id: str) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))["conditions"][condition_id]
    origin = raw["origin_xyz_angstrom"]
    return {
        "nx": int(raw["nx"]),
        "ny": int(raw["ny"]),
        "nz": int(raw["nz"]),
        "origin": (float(origin[0]), float(origin[1]), float(origin[2])),
        "spacing": float(raw["spacing_angstrom"]),
    }


def atom_voxel(atom: dict[str, Any], geometry: dict[str, Any]) -> int | None:
    origin = geometry["origin"]
    spacing = float(geometry["spacing"])
    ix = math.floor((float(atom["x"]) - origin[0]) / spacing)
    iy = math.floor((float(atom["y"]) - origin[1]) / spacing)
    iz = math.floor((float(atom["z"]) - origin[2]) / spacing)
    if ix < 0 or iy < 0 or iz < 0 or ix >= geometry["nx"] or iy >= geometry["ny"] or iz >= geometry["nz"]:
        return None
    return int(iz * geometry["nx"] * geometry["ny"] + iy * geometry["nx"] + ix)


def distance_to_atoms(frame: pl.DataFrame, atoms: list[dict[str, Any]], geometry: dict[str, Any]) -> np.ndarray:
    origin = geometry["origin"]
    spacing = float(geometry["spacing"])
    x = origin[0] + (frame.get_column("x_idx").to_numpy() + 0.5) * spacing
    y = origin[1] + (frame.get_column("y_idx").to_numpy() + 0.5) * spacing
    z = origin[2] + (frame.get_column("z_idx").to_numpy() + 0.5) * spacing
    min_d2 = np.full(frame.height, np.inf, dtype=np.float64)
    for atom in atoms:
        dx = x - float(atom["x"])
        dy = y - float(atom["y"])
        dz = z - float(atom["z"])
        min_d2 = np.minimum(min_d2, dx * dx + dy * dy + dz * dz)
    return np.sqrt(min_d2)


def generate_grid(
    wt_frame: pl.DataFrame,
    geometry: dict[str, Any],
    scaffold_id: str,
    sdf_path: Path,
    o3a_score: float,
    args: argparse.Namespace,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    atoms_all = parse_sdf_atoms(sdf_path)
    atoms = atoms_all if args.include_hydrogens else [atom for atom in atoms_all if atom["element"].upper() != "H"]
    occupied = {voxel for atom in atoms for voxel in [atom_voxel(atom, geometry)] if voxel is not None}
    distances = distance_to_atoms(wt_frame, atoms, geometry)
    original_class = wt_frame.get_column("variance_class").to_list()
    voxel_idx = wt_frame.get_column("voxel_idx").to_list()
    new_class: list[str] = []
    for idx, distance, current in zip(voxel_idx, distances, original_class, strict=True):
        if int(idx) in occupied or distance <= float(args.proximal_radius_a):
            new_class.append("stable_occupied")
        elif distance <= float(args.response_radius_a):
            new_class.append("thermally_activated")
        else:
            new_class.append(str(current))
    out = wt_frame.with_columns(
        pl.lit(f"glp1r_6XOX_SCAFFOLD_{scaffold_id}").alias("condition_id"),
        pl.Series("variance_class", new_class),
        pl.Series("variance_classification", new_class),
        pl.lit(scaffold_id).alias("scaffold_id"),
        pl.lit("L2_PROJECTED").alias("scaffold_bound_provenance"),
    )
    class_diff = sum(1 for before, after in zip(original_class, new_class, strict=True) if str(before) != after)
    metrics = {
        "condition_id": f"glp1r_6XOX_SCAFFOLD_{scaffold_id}",
        "atom_count_scored": len(atoms),
        "heavy_atom_count": sum(1 for atom in atoms_all if atom["element"].upper() != "H"),
        "hydrogen_count": sum(1 for atom in atoms_all if atom["element"].upper() == "H"),
        "o3a_score": float(o3a_score),
        "occupied_voxel_count": len(occupied),
        "proximal_voxel_count": int((distances <= float(args.proximal_radius_a)).sum()),
        "response_shell_voxel_count": int(
            ((distances > float(args.proximal_radius_a)) & (distances <= float(args.response_radius_a))).sum()
        ),
        "influence_voxel_count": int((distances <= float(args.response_radius_a)).sum()),
        "class_diff_count": class_diff,
        "voxel_diff_fraction": class_diff / max(wt_frame.height, 1),
        "thermally_activated_voxels": new_class.count("thermally_activated"),
        "stable_occupied_voxels": new_class.count("stable_occupied"),
        "provenance": "L2_PROJECTED",
        "projection_model": "scaffold_stable_occupied_plus_3_to_6A_response_shell_v1",
    }
    return out, metrics


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    if float(args.proximal_radius_a) < 0.0:
        raise ValueError("--proximal-radius-a must be non-negative")
    if float(args.response_radius_a) < float(args.proximal_radius_a):
        raise ValueError("--response-radius-a must be greater than or equal to --proximal-radius-a")
    wt_grid = resolve_path(Path(args.wt_grid))
    geometry = load_grid_geometry(resolve_path(Path(args.grid_mapping)), str(args.wt_condition))
    wt_frame = (
        pl.scan_parquet(wt_grid)
        .filter(pl.col("condition_id") == str(args.wt_condition))
        .collect()
    )
    if wt_frame.is_empty():
        raise RuntimeError(f"{wt_grid} has no rows for condition {args.wt_condition}")
    grid_dir = Path(args.grid_dir)
    grid_dir.mkdir(parents=True, exist_ok=True)

    entries = scaffold_entries(resolve_path(Path(args.o3a_manifest)), resolve_path(Path(args.aleni_sdf)))
    manifest_items: list[dict[str, Any]] = []
    metrics_by_scaffold: dict[str, Any] = {}
    for scaffold_id in sorted(entries):
        entry = entries[scaffold_id]
        sdf_path = resolve_path(Path(str(entry["scaffold_sdf"])))
        grid, metrics = generate_grid(wt_frame, geometry, scaffold_id, sdf_path, float(entry["o3a_score"]), args)
        output = grid_dir / f"signal_grid_scaffold_{scaffold_id}.parquet"
        tmp = output.with_suffix(output.suffix + ".tmp")
        grid.write_parquet(tmp)
        tmp.replace(output)
        metrics_by_scaffold[scaffold_id] = metrics
        manifest_items.append(
            {
                "scaffold_id": scaffold_id,
                "scaffold_sdf": str(sdf_path),
                "scaffold_sha256": sha256_file(sdf_path),
                "grid_path": str(output),
                "row_count": grid.height,
                "condition_id": metrics["condition_id"],
                "provenance": "L2_PROJECTED",
                "epistemic_class": "PROJECTED",
                "metrics": metrics,
            }
        )

    generated_at = datetime.now(UTC).isoformat()
    manifest = {
        "schema_version": "PRISM.scaffold_bound_signal_grids.v1",
        "generated_at_utc": generated_at,
        "wt_grid": str(wt_grid),
        "wt_condition": str(args.wt_condition),
        "grid_mapping": str(resolve_path(Path(args.grid_mapping))),
        "o3a_manifest": str(resolve_path(Path(args.o3a_manifest))),
        "grid_dir": str(grid_dir),
        "include_hydrogens": bool(args.include_hydrogens),
        "projection_parameters": {
            "proximal_radius_A": float(args.proximal_radius_a),
            "response_radius_A": float(args.response_radius_a),
        },
        "scaffolds": manifest_items,
    }
    report = {
        "schema_version": "PRISM.scaffold_bound_grid_generation_report.v1",
        "generated_at_utc": generated_at,
        "status": "PASS",
        "scaffold_count": len(manifest_items),
        "manifest": str(args.manifest),
        "scaffolds": metrics_by_scaffold,
    }
    atomic_write_json(Path(args.manifest), manifest)
    atomic_write_json(Path(args.report), report)
    print(f"scaffold_bound_grids_generated scaffold_count={len(manifest_items)} manifest={args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

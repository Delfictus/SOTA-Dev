#!/usr/bin/env python3
"""Build NMA, hydration, and thermodynamic continuity maps for Track B."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

import polars as pl

from prism_dstw.calibration.track_b_artifacts import sha256_file, write_json
from prism_dstw.calibration.track_b_schemas import utc_now_iso


def _chronology_voxel_maps(chronology: Path) -> tuple[dict[str, int], dict[tuple[str, int], int]]:
    residue_to_voxel: dict[str, int] = {}
    condition_residue_to_voxel: dict[tuple[str, int], int] = {}
    if not chronology.exists():
        return residue_to_voxel, condition_residue_to_voxel
    frame = (
        pl.scan_parquet(str(chronology))
        .filter(pl.col("voxel_idx").is_not_null())
        .select(["condition", "residues", "voxel_idx"])
        .explode("residues")
        .drop_nulls(["residues", "voxel_idx"])
        .collect()
    )
    for row in frame.to_dicts():
        residue = int(row["residues"])
        voxel = int(row["voxel_idx"])
        condition = str(row["condition"])
        residue_to_voxel.setdefault(str(residue), voxel)
        condition_residue_to_voxel.setdefault((condition, residue), voxel)
    return residue_to_voxel, condition_residue_to_voxel


def _build_nma(paths: list[Path], residue_voxels: dict[str, int]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        residues = list(payload.get("residue_ids", []))
        accum = [0.0 for _ in residues]
        for mode in payload.get("modes", []):
            eigenvalue = max(float(mode.get("eigenvalue") or 1.0), 1.0e-9)
            weight = 1.0 / math.sqrt(eigenvalue)
            for idx, displacement in enumerate(mode.get("displacements", [])):
                if idx >= len(accum) or not isinstance(displacement, list) or len(displacement) < 3:
                    continue
                norm = math.sqrt(sum(float(v) * float(v) for v in displacement[:3]))
                accum[idx] += norm * weight
        if not accum:
            continue
        threshold = sorted(accum)[int(0.9 * (len(accum) - 1))]
        for residue, value in zip(residues, accum, strict=False):
            rows.append(
                {
                    "id": f"nma-{path.stem}-{residue}",
                    "map_type": "NMA",
                    "residue_id": str(residue),
                    "voxel_idx": residue_voxels.get(str(residue)),
                    "hinge_residue_flag": value >= threshold,
                    "mode_displacement_norm": float(value),
                    "hinge_disruption_risk": float(value / max(threshold, 1.0e-9)),
                    "provenance_class": "L3_DERIVED",
                    "source_artifacts": json.dumps([str(path)]),
                    "evidence_paths": json.dumps([str(path)]),
                    "blocked_with_hard_evidence": False,
                    "created_at": utc_now_iso(),
                    "schema_version": "track_b.nma_continuity_map.v1",
                }
            )
    if not rows:
        rows.append(
            {
                "id": "nma-blocked-missing",
                "map_type": "NMA",
                "residue_id": None,
                "voxel_idx": None,
                "hinge_residue_flag": False,
                "mode_displacement_norm": 0.0,
                "hinge_disruption_risk": 0.0,
                "provenance_class": "L0_MISSING",
                "source_artifacts": json.dumps([]),
                "evidence_paths": json.dumps([]),
                "blocked_with_hard_evidence": True,
                "created_at": utc_now_iso(),
                "schema_version": "track_b.nma_continuity_map.v1",
            }
        )
    return pl.DataFrame(rows)


def _build_hydration(paths: list[Path]) -> pl.DataFrame:
    if not paths:
        return pl.DataFrame(
            [
                {
                    "id": "hydration-blocked-missing",
                    "map_type": "HYDRATION",
                    "residue_id": None,
                    "voxel_idx": None,
                    "hydration_tunnel_id": "BLOCKED_WITH_HARD_EVIDENCE",
                    "sigma_hyd": 0.0,
                    "solvent_wire_importance": 0.0,
                    "occlusion_risk": 0.0,
                    "provenance_class": "L0_MISSING",
                    "source_artifacts": json.dumps([]),
                    "evidence_paths": json.dumps(["No hydration parquet found by supplied glob"]),
                    "blocked_with_hard_evidence": True,
                    "created_at": utc_now_iso(),
                    "schema_version": "track_b.hydration_continuity_map.v1",
                }
            ]
        )
    rows: list[dict[str, Any]] = []
    for path in paths:
        frame = pl.scan_parquet(str(path)).limit(5000).collect()
        for idx, row in enumerate(frame.to_dicts()):
            sigma = float(row.get("sigma_hyd") or row.get("sigma_hydration_sq") or 0.0)
            rows.append(
                {
                    "id": f"hydration-{path.stem}-{idx}",
                    "map_type": "HYDRATION",
                    "residue_id": str(row.get("residue_id") or row.get("residue_idx") or ""),
                    "voxel_idx": row.get("voxel_idx"),
                    "hydration_tunnel_id": str(row.get("hydration_tunnel_id") or path.stem),
                    "sigma_hyd": sigma,
                    "solvent_wire_importance": abs(sigma),
                    "occlusion_risk": max(0.0, sigma),
                    "provenance_class": "L3_DERIVED",
                    "source_artifacts": json.dumps([str(path)]),
                    "evidence_paths": json.dumps([str(path)]),
                    "blocked_with_hard_evidence": False,
                    "created_at": utc_now_iso(),
                    "schema_version": "track_b.hydration_continuity_map.v1",
                }
            )
    return pl.DataFrame(rows)


def _build_thermodynamic(
    hysteresis: Path,
    chronology: Path,
    residue_voxels: dict[str, int],
    condition_residue_voxels: dict[tuple[str, int], int],
) -> pl.DataFrame:
    hyst = pl.scan_parquet(str(hysteresis)).select(
        [
            pl.col("condition_id"),
            pl.col("primary_residue_idx"),
            pl.col("thermal_irreversibility"),
            pl.col("hysteresis_delta"),
        ]
    )
    chrono = (
        pl.scan_parquet(str(chronology))
        .explode("residues")
        .rename({"residues": "primary_residue_idx"})
        .group_by("condition", "primary_residue_idx")
        .agg(pl.len().alias("transition_event_density"))
    )
    joined = (
        hyst.join(
            chrono,
            left_on=["condition_id", "primary_residue_idx"],
            right_on=["condition", "primary_residue_idx"],
            how="left",
        )
        .with_columns(pl.col("transition_event_density").fill_null(0))
        .limit(20000)
        .collect()
    )
    rows: list[dict[str, Any]] = []
    for row in joined.to_dicts():
        residue_idx = int(row["primary_residue_idx"])
        condition_id = str(row["condition_id"])
        irreversibility = float(row.get("thermal_irreversibility") or 0.0)
        event_density = float(row.get("transition_event_density") or 0.0)
        recovery = max(0.0, 1.0 - irreversibility)
        voxel_idx = condition_residue_voxels.get((condition_id, residue_idx), residue_voxels.get(str(residue_idx)))
        rows.append(
            {
                "id": f"thermo-{condition_id}-{residue_idx}",
                "map_type": "THERMODYNAMIC",
                "residue_id": f"RES{residue_idx}",
                "voxel_idx": voxel_idx,
                "reversibility": recovery,
                "hysteresis_score": irreversibility,
                "transition_event_density": event_density,
                "trap_risk": min(1.0, irreversibility + event_density / 100.0),
                "recovery_likelihood": recovery,
                "provenance_class": "L3_DERIVED",
                "source_artifacts": json.dumps([str(hysteresis), str(chronology)]),
                "evidence_paths": json.dumps([str(hysteresis), str(chronology)]),
                "blocked_with_hard_evidence": False,
                "created_at": utc_now_iso(),
                "schema_version": "track_b.thermodynamic_continuity_map.v1",
            }
        )
    return pl.DataFrame(rows)


def _external_input_paths(pattern: str, output_dir: Path) -> list[Path]:
    output_root = output_dir.resolve()
    generated_names = {
        "nma_continuity_map.parquet",
        "hydration_continuity_map.parquet",
        "thermodynamic_continuity_map.parquet",
    }
    paths: list[Path] = []
    for raw_path in sorted(glob.glob(pattern, recursive=True)):
        path = Path(raw_path)
        resolved = path.resolve()
        if path.name in generated_names:
            continue
        if resolved == output_root or output_root in resolved.parents:
            continue
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nma-glob", required=True)
    parser.add_argument("--hydration-glob", required=True)
    parser.add_argument("--hysteresis", type=Path, required=True)
    parser.add_argument("--chronology", type=Path, required=True)
    parser.add_argument("--signal-grid", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nma_paths = _external_input_paths(args.nma_glob, args.output_dir)
    hydration_paths = _external_input_paths(args.hydration_glob, args.output_dir)

    residue_voxels, condition_residue_voxels = _chronology_voxel_maps(args.chronology)
    nma = _build_nma(nma_paths, residue_voxels)
    hydration = _build_hydration(hydration_paths)
    thermo = _build_thermodynamic(args.hysteresis, args.chronology, residue_voxels, condition_residue_voxels)

    nma_path = args.output_dir / "nma_continuity_map.parquet"
    hyd_path = args.output_dir / "hydration_continuity_map.parquet"
    thermo_path = args.output_dir / "thermodynamic_continuity_map.parquet"
    nma.write_parquet(nma_path)
    hydration.write_parquet(hyd_path)
    thermo.write_parquet(thermo_path)
    hydration_blocked = bool(hydration.get_column("blocked_with_hard_evidence").all())
    manifest = {
        "id": "track_b_continuity_map_manifest",
        "schema_version": "track_b.continuity_map_manifest.v1",
        "created_at": utc_now_iso(),
        "provenance_class": "L3_DERIVED",
        "evidence_paths": [str(args.hysteresis), str(args.chronology), str(args.signal_grid)],
        "maps": {
            "nma": {"path": str(nma_path), "rows": nma.height, "sha256": sha256_file(nma_path), "provenance": "L3_DERIVED"},
            "hydration": {
                "path": str(hyd_path),
                "rows": hydration.height,
                "sha256": sha256_file(hyd_path),
                "provenance": "L0_MISSING" if hydration_blocked else "L3_DERIVED",
            },
            "thermodynamic": {"path": str(thermo_path), "rows": thermo.height, "sha256": sha256_file(thermo_path), "provenance": "L3_DERIVED"},
        },
        "source_artifacts": [str(args.hysteresis), str(args.chronology), str(args.signal_grid)] + [str(p) for p in nma_paths + hydration_paths],
    }
    write_json(args.output_dir / "continuity_map_manifest.json", manifest)
    print(
        "continuity_maps "
        f"nma_rows={nma.height} hydration_rows={hydration.height} thermo_rows={thermo.height} "
        f"hydration_blocked={hydration_blocked}"
    )


if __name__ == "__main__":
    main()

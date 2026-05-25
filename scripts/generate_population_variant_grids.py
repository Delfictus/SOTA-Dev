#!/usr/bin/env python3
"""Generate projected GLP1R population variant signal grids."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, TypeAlias, cast

import numpy as np
import polars as pl

from prism_dstw.pgx.variant_perturbation_engine import (
    PerturbationType,
    apply_perturbation,
    classify_variant,
    domain_map_from_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = REPO_ROOT / "campaigns/glp1r_aleniglipron"
TRACK_A = CAMPAIGN_DIR / "track_a_generative"
POP_DIR = TRACK_A / "population_pgx"
SOURCE_DIR = POP_DIR / "source"
N80_DIR = CAMPAIGN_DIR / "integrated_spike_events/n80_full_scale"
DEFAULT_WT_GRID = N80_DIR / "signal_grid_variance_channel.parquet"
DEFAULT_A316T_GRID = N80_DIR / "signal_grid_variance_channel_A316T.parquet"
DEFAULT_T149M_GRID = N80_DIR / "signal_grid_variance_channel_T149M.parquet"
DEFAULT_MAPPING = CAMPAIGN_DIR / "track_0_manual_emulation/grid_coordinate_mapping.json"
DEFAULT_TOPOLOGY = REPO_ROOT / "04_TOPOLOGIES/glp1r_6XOX_HOLO_ALENI.topology.json"
DEFAULT_GRID_DIR = POP_DIR / "variant_grids"
DEFAULT_MANIFEST = POP_DIR / "variant_perturbation_manifest.json"
DEFAULT_REPORT = POP_DIR / "variant_grid_generation_report.json"

JsonObject: TypeAlias = dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--wt-grid", type=Path, default=DEFAULT_WT_GRID)
    parser.add_argument("--a316t-grid", type=Path, default=DEFAULT_A316T_GRID)
    parser.add_argument("--t149m-grid", type=Path, default=DEFAULT_T149M_GRID)
    parser.add_argument("--grid-mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY)
    parser.add_argument("--variant-grid-dir", type=Path, default=DEFAULT_GRID_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_condition_grid(path: Path, condition_id: str) -> pl.DataFrame:
    scan = pl.scan_parquet(path)
    if "condition_id" in scan.collect_schema().names():
        scan = scan.filter(pl.col("condition_id") == condition_id)
    frame = scan.collect().sort("voxel_idx")
    if frame.height == 0:
        raise RuntimeError(f"{path} contains no rows for condition_id={condition_id}")
    return frame


def topology_ca_by_residue_id(path: Path) -> dict[int, np.ndarray]:
    topology = json.loads(path.read_text(encoding="utf-8"))
    positions = topology["positions"]
    atom_names = topology["atom_names"]
    residue_ids = topology["residue_ids"]
    coords: dict[int, np.ndarray] = {}
    for atom_idx, atom_name in enumerate(atom_names):
        if str(atom_name).strip().upper() != "CA":
            continue
        residue_id = int(residue_ids[atom_idx])
        offset = atom_idx * 3
        coords[residue_id] = np.array(
            [
                float(positions[offset]),
                float(positions[offset + 1]),
                float(positions[offset + 2]),
            ],
            dtype=np.float64,
        )
    return coords


def wt_grid_spec(path: Path) -> tuple[np.ndarray, float, tuple[int, int, int]]:
    decoded = json.loads(path.read_text(encoding="utf-8"))
    raw = decoded["conditions"]["glp1r_6XOX_WT"]
    origin = np.array(raw["origin_xyz_angstrom"], dtype=np.float64)
    dims = (int(raw["nx"]), int(raw["ny"]), int(raw["nz"]))
    return origin, float(raw["spacing_angstrom"]), dims


def diff_metrics(reference: pl.DataFrame, candidate: pl.DataFrame) -> dict[str, Any]:
    ref = reference.sort("voxel_idx")
    cur = candidate.sort("voxel_idx")
    cold_ref = ref.get_column("hit_count_cold_mean").to_numpy()
    warm_ref = ref.get_column("hit_count_warm_mean").to_numpy()
    cold_cur = cur.get_column("hit_count_cold_mean").to_numpy()
    warm_cur = cur.get_column("hit_count_warm_mean").to_numpy()
    cls_ref = np.array(ref.get_column("variance_class").to_list(), dtype=object)
    cls_cur = np.array(cur.get_column("variance_class").to_list(), dtype=object)
    changed = (np.abs(cold_ref - cold_cur) > 1.0e-9) | (np.abs(warm_ref - warm_cur) > 1.0e-9) | (cls_ref != cls_cur)
    return {
        "voxel_diff_fraction": float(np.count_nonzero(changed) / max(len(changed), 1)),
        "class_diff_count": int(np.count_nonzero(cls_ref != cls_cur)),
        "mean_abs_cold_delta": float(np.mean(np.abs(cold_ref - cold_cur))),
        "mean_abs_warm_delta": float(np.mean(np.abs(warm_ref - warm_cur))),
    }


def conservation_by_position(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {int(row["residue_position"]): row for row in rows}


def condition_for_mutation(mutation: str) -> str:
    return f"glp1r_6XOX_{mutation}"


def write_grid(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.write_parquet(tmp)
    tmp.replace(path)


def main() -> int:
    args = parse_args()
    variants = read_csv(args.source_dir / "gnomAD_GLP1R_missense_variants.csv")
    conservation = conservation_by_position(read_csv(args.source_dir / "GLP1R_cross_species_conservation.csv"))
    domains = domain_map_from_rows(read_csv(args.source_dir / "GLP1R_structural_domain_map.csv"))
    residue_coords = topology_ca_by_residue_id(args.topology)
    origin, spacing, dims = wt_grid_spec(args.grid_mapping)
    wt = load_condition_grid(args.wt_grid, "glp1r_6XOX_WT")
    md_grids = {
        "A316T": load_condition_grid(args.a316t_grid, "glp1r_6XOX_A316T"),
        "T149M": load_condition_grid(args.t149m_grid, "glp1r_6XOX_T149M"),
    }

    manifest_variants: list[JsonObject] = []
    tier_counts = {"1": 0, "2": 0, "3": 0}
    generated = 0
    for row in variants:
        perturbation = classify_variant(row, conservation, domains)
        mutation = perturbation.mutation
        condition_id = condition_for_mutation(mutation)
        grid_path = args.variant_grid_dir / f"signal_grid_variance_channel_{mutation}.parquet"
        if mutation in md_grids:
            grid = md_grids[mutation].with_columns(pl.lit(condition_id).alias("condition_id"))
            metrics = diff_metrics(wt, grid)
            metrics["diff_gate_status"] = "PASS_MD_SIMULATED"
        else:
            grid, metrics = apply_perturbation(
                wt,
                perturbation,
                residue_coords,
                grid_origin=origin,
                grid_spacing=spacing,
                grid_dims=dims,
            )
        write_grid(grid_path, grid)
        tier_counts[str(perturbation.tier)] += 1
        generated += 1
        payload = perturbation.to_json()
        payload.update(
            {
                "condition_id": condition_id,
                "grid_path": grid_path.as_posix(),
                "row_count": grid.height,
                "grid_sha256_pending": False,
                "metrics": metrics,
            }
        )
        manifest_variants.append(payload)
        print(
            "variant_grid_generated "
            f"mutation={mutation} tier={perturbation.tier} provenance={perturbation.provenance} "
            f"confidence={perturbation.epistemic_confidence} diff={float(metrics['voxel_diff_fraction']):.5f} "
            f"path={grid_path}"
        )

    manifest: JsonObject = {
        "schema_version": "PRISM.population_variant_perturbation_manifest.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_dir": args.source_dir.as_posix(),
        "wt_grid": args.wt_grid.as_posix(),
        "grid_dir": args.variant_grid_dir.as_posix(),
        "tier_counts": tier_counts,
        "variant_count": generated,
        "variants": manifest_variants,
        "epistemic_policy": {
            "MD_SIMULATED": "L5 observed 80-replica CCNS-derived grid materialized before Epoch 018",
            "PERTURBATION_PROJECTED": "L1-L3 projected perturbation from supplied functional/domain/conservation data",
        },
        "gate_notes": [
            "P7L is signal peptide / trafficking and is intentionally marked NO_STRUCTURAL_EFFECT_EXCEPTION.",
            "A316T tier is computed from CSV maf_global=0.0076, despite directive prose mentioning 1.5%.",
        ],
    }
    report: JsonObject = {
        "schema_version": "PRISM.population_variant_grid_generation_report.v1",
        "generated_at_utc": manifest["generated_at_utc"],
        "variant_count": generated,
        "tier_counts": tier_counts,
        "md_simulated_count": len([item for item in manifest_variants if item["provenance"] == "MD_SIMULATED"]),
        "projected_count": len([item for item in manifest_variants if item["provenance"] == "PERTURBATION_PROJECTED"]),
        "no_structural_effect_count": len(
            [item for item in manifest_variants if item["perturbation_type"] == PerturbationType.NO_STRUCTURAL_EFFECT.value]
        ),
        "manifest": args.manifest.as_posix(),
        "grid_dir": args.variant_grid_dir.as_posix(),
        "status": "PASS" if generated == 22 and tier_counts == {"1": 7, "2": 2, "3": 13} else "WARN_REVIEW",
    }
    atomic_json(args.manifest, manifest)
    atomic_json(args.report, report)
    print(
        "variant_grid_gate "
        f"status={report['status']} variant_count={generated} tier_counts={tier_counts} "
        f"manifest={args.manifest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

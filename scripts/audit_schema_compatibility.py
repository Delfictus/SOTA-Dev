#!/usr/bin/env python3
"""Validate parquet producer/consumer schema contracts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SchemaContract:
    name: str
    path: str
    producer: str
    consumers: tuple[str, ...]
    required_columns: tuple[str, ...]
    dtype_prefixes: dict[str, str] | None = None


SCHEMA_CONTRACTS: tuple[SchemaContract, ...] = (
    SchemaContract(
        name="enamine_115k_synthons_3d.parquet",
        path="campaigns/glp1r_aleniglipron/track_a_generative/enamine_115k_synthons_3d.parquet",
        producer="scripts/ingest_vspace_synthons.py",
        consumers=("crates/prism-forge/src/bin/vspace_pruner.rs", "scripts/train_gflownet_policy.py"),
        required_columns=(
            "synthon_id",
            "vendor_id",
            "canonical_smiles",
            "compatible_reactions_json",
            "partial_charges_json",
            "conformer_atoms_json",
            "heavy_atom_count",
            "attachment_atom_idx",
        ),
        dtype_prefixes={"canonical_smiles": "String", "partial_charges_json": "String"},
    ),
    SchemaContract(
        name="vspace_survivors_full_scale.parquet",
        path="campaigns/glp1r_aleniglipron/track_a_generative/vspace_survivors_full_scale.parquet",
        producer="crates/prism-forge/src/bin/vspace_pruner.rs",
        consumers=("scripts/train_gflownet_policy.py", "src/prism_dstw/orchestration/rust_reward_oracle.py"),
        required_columns=(
            "anchor_id",
            "canonical_smiles",
            "smiles",
            "pi_clash",
            "pi_complement",
            "coordinates_json",
            "survival_tier",
        ),
        dtype_prefixes={"canonical_smiles": "String", "pi_complement": "Float"},
    ),
    SchemaContract(
        name="gflownet_top_100_candidates.parquet",
        path="campaigns/glp1r_aleniglipron/track_a_generative/gflownet_top_100_candidates.parquet",
        producer="scripts/train_gflownet_policy.py",
        consumers=("scripts/build_thermodynamic_motif_registry.py", "scripts/generate_m3_dossier.py"),
        required_columns=(
            "canonical_smiles",
            "coordinates_json",
            "reward",
            "pi_complement",
            "pi_clash_pocket",
            "pi_clash_lock",
            "lock_geometry_score",
            "survival_tier",
        ),
    ),
    SchemaContract(
        name="signal_grid_population_consensus.parquet",
        path="campaigns/glp1r_aleniglipron/track_a_generative/signal_grid_population_consensus.parquet",
        producer="THERMODYNAMIC_OBSERVATORY",
        consumers=("crates/prism-forge/src/bin/oracle_scorer.rs", "scripts/train_gflownet_policy.py"),
        required_columns=(
            "voxel_idx",
            "x_idx",
            "y_idx",
            "z_idx",
            "hit_count_cold_mean",
            "hit_count_warm_mean",
            "variance_class",
            "consensus_complement_bonus",
        ),
    ),
    SchemaContract(
        name="shear_stress_field.parquet",
        path="campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/shear_stress_field.parquet",
        producer="crates/prism-nhs/src/bin/warp_jacobian.rs",
        consumers=("crates/prism-forge/src/bin/oracle_scorer.rs", "scripts/train_gflownet_policy.py"),
        required_columns=("voxel_idx", "shear_stress", "principal_x", "principal_y", "principal_z"),
    ),
    SchemaContract(
        name="subtb_training_metrics.parquet",
        path="campaigns/glp1r_aleniglipron/track_b_chronological/subtb_spectral/subtb_training_metrics.parquet",
        producer="scripts/run_log_subtb_spectral_gflownet.py",
        consumers=("scripts/build_hardened_cbom.py",),
        required_columns=(
            "epoch",
            "captured_tile_replay_count",
            "uncaptured_tile_fallback_count",
            "gpu_solve_count",
            "cpu_solve_count",
            "reward_event_source",
        ),
    ),
)


def _dtype_name(dtype: object) -> str:
    return str(dtype)


def audit_schemas() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for contract in SCHEMA_CONTRACTS:
        path = REPO_ROOT / contract.path
        row: dict[str, Any] = {
            "name": contract.name,
            "path": contract.path,
            "producer": contract.producer,
            "consumers": list(contract.consumers),
            "exists": path.exists(),
            "missing_columns": [],
            "dtype_mismatches": [],
            "extra_columns": [],
            "status": "PASS",
        }
        if not path.exists():
            row["status"] = "CRITICAL_MISSING_FILE"
            rows.append(row)
            continue
        frame = pl.read_parquet(path, n_rows=1)
        schema = {name: _dtype_name(dtype) for name, dtype in frame.schema.items()}
        row["schema"] = schema
        row["row_count_sampled"] = frame.height
        missing = [column for column in contract.required_columns if column not in schema]
        row["missing_columns"] = missing
        row["extra_columns"] = sorted(set(schema) - set(contract.required_columns))
        dtype_mismatches: list[dict[str, str]] = []
        for column, prefix in (contract.dtype_prefixes or {}).items():
            actual = schema.get(column)
            if actual is not None and not actual.startswith(prefix):
                dtype_mismatches.append({"column": column, "expected_prefix": prefix, "actual": actual})
        row["dtype_mismatches"] = dtype_mismatches
        if missing:
            row["status"] = "CRITICAL_MISSING_COLUMNS"
        elif dtype_mismatches:
            row["status"] = "HIGH_DTYPE_MISMATCH"
        rows.append(row)
    failures = [row for row in rows if row["status"] != "PASS"]
    return {
        "schema_version": "PRISM.schema_compatibility_audit.v1",
        "contracts": rows,
        "summary": {"contract_count": len(rows), "failure_count": len(failures)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "release_artifacts/v0.25.0/schema_compatibility_report.json")
    args = parser.parse_args()
    report = audit_schemas()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = report["summary"]
    print(
        "schema_compatibility_audit "
        f"contracts={summary['contract_count']} failures={summary['failure_count']} report={args.output}"
    )
    return 1 if summary["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

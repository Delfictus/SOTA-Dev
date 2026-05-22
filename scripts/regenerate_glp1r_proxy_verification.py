#!/usr/bin/env python3
"""Regenerate sanitized GLP-1R voxel proxy verification artifacts."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from prism_dstw.io import ParquetProvenance, repo_relative_path, write_provenance_parquet  # noqa: E402
from prism_dstw.propagation_ledger import (  # noqa: E402
    append_ledger_entry,
    build_superseding_entry,
    canonicalize_ledger_entry,
    load_latest_ledger_entry,
)


PROXY_FILES = {
    "voxel_stable_void_proxy": "voxel_stable_void_proxy.parquet",
    "scope_stream_stable_void_proxy_summary": "scope_stream_stable_void_proxy_summary.parquet",
    "scope_stable_void_proxy_summary": "scope_stable_void_proxy_summary.parquet",
    "interface_interference_terms_proxy": "interface_interference_terms_proxy.parquet",
}

PROXY_SOURCE_FILES = (
    "campaigns/glp1r_aleniglipron/integrated_spike_events/full_dynamic_aligned_voxels/dynamic_voxel_event_time_bins.parquet",
    "campaigns/glp1r_aleniglipron/integrated_spike_events/full_dynamic_aligned_voxels/interface_aligned_voxel_fields.parquet",
    "campaigns/glp1r_aleniglipron/integrated_spike_events/full_dynamic_aligned_voxels/site_aligned_voxel_fields.parquet",
    "campaigns/glp1r_aleniglipron/integrated_spike_events/full_timestamp_mining/interface_time_bins.parquet",
)

PARTITION_KEYS = (
    "campaign_id",
    "run_label",
    "structure_id",
    "stream_id",
    "scope_type",
    "scope_id",
)


def artifact_id(path: Path, repo_root: Path) -> str:
    return repo_relative_path(path, repo_root)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sanitize_value(value: Any, repo_root: Path) -> Any:
    if isinstance(value, str):
        marker = str(repo_root.resolve())
        if marker in value:
            value = value.replace(f"{marker}/", "").replace(marker, ".")
        if value.startswith("/home/"):
            return "internal_provenance_redacted"
        return value
    if isinstance(value, list):
        return [sanitize_value(v, repo_root) for v in value]
    if isinstance(value, dict):
        return {k: sanitize_value(v, repo_root) for k, v in value.items()}
    return value


def sanitize_propagation_ledgers(proxy_dir: Path, repo_root: Path) -> None:
    """Append superseding ledger entries with repo-relative paths only."""

    for ledger_path in sorted(proxy_dir.glob("*.propagation.jsonl")):
        parquet_name = ledger_path.name.replace(".propagation.jsonl", ".parquet")
        output_path = proxy_dir / parquet_name
        latest = load_latest_ledger_entry(ledger_path)
        if latest is None:
            continue
        updates = {
            "output_value": {"output_path": output_path},
            "provenance_policy": "repo_relative_paths_only_no_command_templates",
        }
        superseding = build_superseding_entry(
            latest,
            repo_root=repo_root,
            updates=updates,
        )
        canonical_latest = canonicalize_ledger_entry(latest, repo_root)
        comparable_superseding = {
            key: value
            for key, value in superseding.items()
            if key not in {"entry_id", "supersedes", "timestamp"}
        }
        if comparable_superseding == canonical_latest:
            continue
        append_ledger_entry(ledger_path, superseding, repo_root)


def rewrite_proxy_parquet_metadata(proxy_dir: Path, repo_root: Path) -> None:
    """Restamp proxy parquets with repo-relative Arrow provenance metadata."""

    sources = tuple(repo_root / source for source in PROXY_SOURCE_FILES)
    for filename in PROXY_FILES.values():
        parquet_path = proxy_dir / filename
        tmp_path = parquet_path.with_name(f".{parquet_path.name}.provenance_tmp")
        tmp_ledger_path = tmp_path.with_suffix(".propagation.jsonl")
        write_provenance_parquet(
            pl.scan_parquet(parquet_path),
            tmp_path,
            provenance=ParquetProvenance(
                generator_script=Path(__file__),
                source_parquets=sources,
                schema_version="prism_voxel_variance_proxy_classifier.v2.polars_native",
                pipeline_stage="voxel_variance_proxy",
                partition_keys=PARTITION_KEYS,
            ),
            repo_root=repo_root,
        )
        tmp_path.replace(parquet_path)
        if tmp_ledger_path.exists():
            tmp_ledger_path.unlink()


def build_verification(campaign_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_root = campaign_dir.parent.parent
    proxy_dir = campaign_dir / "integrated_spike_events/full_voxel_variance_proxy"
    manifest_path = proxy_dir / "voxel_variance_proxy_manifest.json"
    manifest = sanitize_value(json.loads(manifest_path.read_text(encoding="utf-8")), repo_root)

    row_counts = {
        filename: pl.scan_parquet(proxy_dir / filename).select(pl.len()).collect().item()
        for filename in PROXY_FILES.values()
    }
    output_sha256 = {filename: sha256(proxy_dir / filename) for filename in PROXY_FILES.values()}

    voxel_df = pl.scan_parquet(proxy_dir / PROXY_FILES["voxel_stable_void_proxy"])
    class_counts = dict(
        voxel_df.group_by("voxel_field_class_proxy")
        .agg(pl.len().alias("count"))
        .collect()
        .iter_rows()
    )
    scope_counts = (
        voxel_df.group_by("scope_type")
        .agg(
            pl.len().alias("voxel_rows"),
            (pl.col("voxel_field_class_proxy") == "stable_occupied_proxy").sum().alias(
                "stable_occupied_proxy"
            ),
            (pl.col("voxel_field_class_proxy") == "high_variance_void_proxy").sum().alias(
                "high_variance_void_proxy"
            ),
            (pl.col("voxel_field_class_proxy") == "transient_high_variance_occupied_proxy")
            .sum()
            .alias("transient_high_variance_occupied_proxy"),
        )
        .collect()
    )
    scope_summary = {
        row["scope_type"]: {k: row[k] for k in row if k != "scope_type"}
        for row in scope_counts.iter_rows(named=True)
    }

    interface_terms = pl.scan_parquet(proxy_dir / PROXY_FILES["interface_interference_terms_proxy"])
    interface_summary = {
        row["interface_class"]: {
            "stream_rows": row["stream_rows"],
            "mean_phi_prot_occupied_proxy": round(row["mean_phi_prot_occupied_proxy"], 6),
            "mean_phi_prot_void_proxy": round(row["mean_phi_prot_void_proxy"], 6),
        }
        for row in (
            interface_terms.group_by("interface_class")
            .agg(
                pl.len().alias("stream_rows"),
                pl.mean("phi_prot_occupied_proxy").alias("mean_phi_prot_occupied_proxy"),
                pl.mean("phi_prot_void_proxy").alias("mean_phi_prot_void_proxy"),
            )
            .collect()
            .iter_rows(named=True)
        )
    }
    top_interfaces = (
        interface_terms.group_by("interface_id", "interface_class")
        .agg(
            pl.mean("phi_prot_occupied_proxy").alias("mean_phi_prot_occupied_proxy"),
            pl.mean("phi_prot_void_proxy").alias("mean_phi_prot_void_proxy"),
            pl.sum("stable_occupied_voxel_count").alias("stable_occupied_voxel_count"),
            pl.sum("high_variance_void_voxel_count").alias("high_variance_void_voxel_count"),
        )
        .sort("mean_phi_prot_void_proxy", descending=True)
        .limit(9)
        .collect()
        .to_dicts()
    )

    sanitized_manifest = {
        **manifest,
        "inputs": sanitize_value(manifest.get("inputs", {}), repo_root),
        "outputs": {
            name: artifact_id(proxy_dir / filename, repo_root)
            for name, filename in PROXY_FILES.items()
        },
        "counts": {name: row_counts[filename] for name, filename in PROXY_FILES.items()},
        "output_sha256": {name: output_sha256[filename] for name, filename in PROXY_FILES.items()},
        "class_counts": class_counts,
        "distributable": False,
        "provenance_policy": "repo_relative_paths_only_no_command_templates",
    }

    verification = {
        "artifact": "PRISM_Twin_Voxel_Variance_Proxy_Verification",
        "campaign_id": "glp1r_aleniglipron",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "proxy_derived": True,
        "engine": manifest.get("engine", "polars_native_arrow_parquet"),
        "schema": manifest.get("schema", "prism_voxel_variance_proxy_classifier.v2.polars_native"),
        "generator": "internal_proxy_classifier_redacted",
        "output_directory": artifact_id(proxy_dir, repo_root),
        "row_counts": row_counts | {"voxel_variance_proxy_manifest.json": 1},
        "output_sha256": output_sha256,
        "validation_checks": manifest.get("validation_checks", {}),
        "class_counts": class_counts,
        "threshold_partition_basis": manifest.get(
            "threshold_partition_basis",
            "campaign_id/run_label/structure_id/stream_id/scope_type/scope_id",
        ),
        "scope_counts": scope_summary,
        "interface_term_summary": interface_summary,
        "top_interface_proxy_complement_burdens": top_interfaces,
        "voxel_field_status": {
            "post_hoc_aligned_voxel_export_exists": True,
            "dstw_admitted_canonical_tso_field": False,
            "status_label": "proxy_derived_not_producer_canonical",
        },
        "supersession_note": (
            "This artifact supersedes earlier forensic notes only for the post-hoc aligned voxel "
            "proxy layer. It does not promote the proxy layer to canonical TSO admission."
        ),
        "allowed_use": [
            "Use phi_prot_occupied_proxy as a receptor-field prototype term for destructive clash projection.",
            "Use phi_prot_void_proxy only as a proxy-derived constructive complement term with explicit warning.",
            (
                "Use voxel_stable_void_proxy.parquet for Track 0 analog overlap scoring and "
                "Layer 1 bivariate projection development."
            ),
        ],
        "blocked_claims": [
            "Do not call phi_prot_void_proxy producer-canonical voxel variance.",
            (
                "Do not claim final constructive interference until producer-side "
                "stable-occupied/high-variance-void voxel variance is emitted."
            ),
            (
                "Do not use this proxy layer as biological chronic receptor durability evidence "
                "without downstream path-sampling and lifecycle modules."
            ),
        ],
        "provenance_policy": "repo_relative_paths_only_no_command_templates",
    }
    return sanitized_manifest, verification


def write_markdown(verification: dict[str, Any], output_md: Path) -> None:
    class_counts = verification["class_counts"]
    scope_counts = verification["scope_counts"]
    interface_summary = verification["interface_term_summary"]
    lines = [
        "# PRISM Twin Voxel Stable/Void Proxy Verification",
        "",
        f"Generated: {verification['generated_at_utc']}",
        "",
        (
            "This note verifies the empirical voxel classification layer derived from the "
            "PRISM Twin dynamic aligned voxel export. It is proxy-derived, not "
            "producer-canonical per-voxel variance."
        ),
        "",
        "## Provenance Policy",
        "",
        "- Distributable verification artifacts use repository-relative artifact identifiers only.",
        "- Command lines and workstation-local paths are intentionally excluded.",
        (
            "- Private execution details belong in internal runbooks, not publication or "
            "contractor-facing evidence bundles."
        ),
        "",
        "## Supersession",
        "",
        verification["supersession_note"],
        "",
        "## Outputs",
        "",
        "| Output | Rows | SHA-256 |",
        "| --- | ---: | --- |",
    ]
    for name, rows in verification["row_counts"].items():
        digest = verification["output_sha256"].get(name, "manifest_not_hashed")
        lines.append(f"| `{name}` | {rows:,} | `{digest}` |")

    lines.extend(["", "## Ontology Classes", "", "| Class | Count |", "| --- | ---: |"])
    for name, count in sorted(class_counts.items()):
        lines.append(f"| `{name}` | {count:,} |")

    lines.extend(
        [
            "",
            "## Scope Split",
            "",
            "| Scope type | Voxel rows | Stable occupied | High-variance void | Transient occupied |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for scope, row in sorted(scope_counts.items()):
        lines.append(
            f"| `{scope}` | {row['voxel_rows']:,} | {row['stable_occupied_proxy']:,} | "
            f"{row['high_variance_void_proxy']:,} | {row['transient_high_variance_occupied_proxy']:,} |"
        )

    lines.extend(
        [
            "",
            "## Interface Terms",
            "",
            "| Interface class | Stream rows | Mean occupied proxy | Mean void proxy |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, row in sorted(interface_summary.items()):
        lines.append(
            f"| `{name}` | {row['stream_rows']:,} | {row['mean_phi_prot_occupied_proxy']:.6f} | "
            f"{row['mean_phi_prot_void_proxy']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            (
                "This layer enables Track 0 and Layer 1 prototype bivariate interference "
                "development. It does not make producer-canonical constructive-interference "
                "claims, does not prove chronic receptor durability, and does not replace "
                "path-sampling interface-breaking timestamps."
            ),
            "",
        ]
    )
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    repo_root = REPO_ROOT
    campaign_dir = repo_root / "campaigns/glp1r_aleniglipron"
    proxy_dir = campaign_dir / "integrated_spike_events/full_voxel_variance_proxy"

    rewrite_proxy_parquet_metadata(proxy_dir, repo_root)
    sanitize_propagation_ledgers(proxy_dir, repo_root)
    sanitized_manifest, verification = build_verification(campaign_dir)
    (proxy_dir / "voxel_variance_proxy_manifest.json").write_text(
        json.dumps(sanitized_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (campaign_dir / "PRISM_Twin_Voxel_Variance_Proxy_Verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_markdown(verification, campaign_dir / "PRISM_Twin_Voxel_Variance_Proxy_Verification.md")
    sys.stdout.write(json.dumps({"status": "ok", "artifact": verification["artifact"]}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate a TE-Hub subpanel view from the full Track B variant panel."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import polars as pl

from prism_dstw.calibration.track_b_artifacts import read_json, write_json
from prism_dstw.calibration.track_b_schemas import utc_now_iso


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-panel", type=Path, required=True)
    parser.add_argument("--coverage-matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    panel = read_json(args.variant_panel)
    variants = list(panel.get("variants", []))
    te_variants = [v for v in variants if v.get("topology_region") == "TE_HUBS"]
    coverage = pl.read_parquet(args.coverage_matrix)
    coverage_ids_by_variant: dict[str, list[str]] = {}
    for row in coverage.select(["variant_id", "id"]).to_dicts():
        coverage_ids_by_variant.setdefault(str(row["variant_id"]), []).append(str(row["id"]))
    covered_ids = set(coverage_ids_by_variant)
    for variant in te_variants:
        if variant["variant_id"] not in covered_ids:
            raise ValueError(f"TE-Hub variant lacks coverage link: {variant['variant_id']}")
    linked_te_variants = []
    for variant in te_variants:
        linked = dict(variant)
        linked["coverage_ids"] = sorted(coverage_ids_by_variant[str(variant["variant_id"])])
        linked_te_variants.append(linked)
    payload = {
        "id": "track_b_te_hub_variant_manifest",
        "schema_version": "track_b.te_hub_subpanel.v1",
        "created_at": utc_now_iso(),
        "provenance_class": "L3_DERIVED",
        "is_subset_view": True,
        "full_panel_variant_count": len(variants),
        "te_hub_variant_count": len(te_variants),
        "source_artifacts": [str(args.variant_panel), str(args.coverage_matrix)],
        "evidence_paths": [str(args.variant_panel), str(args.coverage_matrix)],
        "variants": linked_te_variants,
    }
    if len(te_variants) == len(variants):
        raise ValueError("TE-Hub manifest cannot be the entire variant strategy")
    write_json(args.output, payload)
    print(f"te_hub_variant_manifest variants={len(te_variants)} output={args.output}")


if __name__ == "__main__":
    main()

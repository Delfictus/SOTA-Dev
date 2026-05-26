#!/usr/bin/env python3
"""Build the six-region Track B topology registry from observatory artifacts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import polars as pl

from prism_dstw.calibration.track_b_artifacts import write_json
from prism_dstw.calibration.track_b_schemas import utc_now_iso
from prism_dstw.calibration.translational_calibration_manifold import REGION_PURPOSES, TOPOLOGY_SPACE

MAX_GLP1R_RESIDUE = 500


def _residue_id(row: dict[str, Any]) -> str:
    name = str(row.get("residue_name") or "RES")
    idx = int(row.get("residue_idx") or row.get("primary_residue_idx") or 0)
    embedded = re.match(r"^([A-Z]{3})(\d+)", name)
    if embedded is not None:
        residue_number = int(embedded.group(2))
        if 0 < residue_number <= MAX_GLP1R_RESIDUE:
            return f"{embedded.group(1)}{residue_number}"
        if 0 < idx <= MAX_GLP1R_RESIDUE:
            return f"{embedded.group(1)}{idx}"
    prefix = name[:3] if len(name) >= 3 else "RES"
    return f"{prefix}{idx}"


def _top_rows(path: Path, sort_col: str, n: int, predicate: pl.Expr | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    frame = pl.scan_parquet(str(path))
    if predicate is not None:
        frame = frame.filter(predicate)
    schema = frame.collect_schema()
    if sort_col not in schema:
        return frame.head(n).collect().to_dicts()
    return frame.sort(sort_col, descending=True).head(n).collect().to_dicts()


def _entry(row: dict[str, Any], region: str, source: Path, extra: dict[str, float]) -> dict[str, Any]:
    score_components = {
        "coherence_score": float(row.get("coherence_score") or 0.0),
        "mean_abs_load": float(row.get("mean_abs_load") or 0.0),
        "thermal_irreversibility": float(row.get("thermal_irreversibility") or 0.0),
        "pathway_rank_score": 1.0 / max(float(row.get("pathway_rank") or 1.0), 1.0),
    }
    score_components.update(extra)
    return {
        "residue_id": _residue_id(row),
        "region": region,
        "evidence_columns": sorted(score_components.keys()),
        "score_components": score_components,
        "source_artifacts": [str(source)],
        "provenance_class": "L3_DERIVED",
    }


def build_registry(
    translation_pathway: Path,
    mechanical_load: Path,
    phase_coherence: Path,
    hysteresis: Path,
) -> dict[str, Any]:
    registry: dict[str, Any] = {
        "id": "track_b_topology_region_registry",
        "schema_version": "track_b.topology_region_registry.v1",
        "created_at": utc_now_iso(),
        "provenance_class": "L3_DERIVED",
        "source_artifacts": [
            str(translation_pathway),
            str(mechanical_load),
            str(phase_coherence),
            str(hysteresis),
        ],
        "evidence_paths": [
            str(translation_pathway),
            str(mechanical_load),
            str(phase_coherence),
            str(hysteresis),
        ],
        "regions": {},
    }

    phase_rows = _top_rows(phase_coherence, "coherence_score", 80)
    pathway_rows = _top_rows(translation_pathway, "pathway_rank", 12)
    hysteresis_rows = _top_rows(hysteresis, "thermal_irreversibility", 40)

    region_candidates: dict[str, list[dict[str, Any]]] = {region: [] for region in TOPOLOGY_SPACE}

    for row in pathway_rows:
        region_candidates["TE_HUBS"].append(
            _entry(row, "TE_HUBS", translation_pathway, {"translation_pathway": 1.0})
        )
        if float(row.get("dist_to_lock_centroid_A") or 99.0) < 12.0:
            region_candidates["INTRACELLULAR_LOCK_BASIN"].append(
                _entry(row, "INTRACELLULAR_LOCK_BASIN", translation_pathway, {"lock_distance_signal": 1.0})
            )
        if float(row.get("dist_to_pocket_centroid_A") or 99.0) < 14.0:
            region_candidates["ECD_TM1_GATEWAY"].append(
                _entry(row, "ECD_TM1_GATEWAY", translation_pathway, {"pocket_gateway_signal": 1.0})
            )

    for row in phase_rows:
        idx = int(row.get("residue_idx") or 0)
        if idx < 100:
            region = "ECD_TM1_GATEWAY"
        elif idx < 190:
            region = "TM3_TM5_CORE"
        elif idx < 260:
            region = "HYDRATION_CORRIDOR"
        elif idx < 330:
            region = "BISTATE_RETAINED_RESIDUES"
        else:
            region = "INTRACELLULAR_LOCK_BASIN"
        region_candidates[region].append(
            _entry(row, region, phase_coherence, {"phase_coherence_rank": 1.0})
        )

    for row in hysteresis_rows:
        idx = int(row.get("primary_residue_idx") or 0)
        row = dict(row)
        row["residue_idx"] = idx
        row["residue_name"] = row.get("residue_name") or "RES"
        region = "BISTATE_RETAINED_RESIDUES" if idx % 2 == 0 else "INTRACELLULAR_LOCK_BASIN"
        region_candidates[region].append(
            _entry(row, region, hysteresis, {"thermal_irreversibility_rank": 1.0})
        )

    for region in TOPOLOGY_SPACE:
        seen: set[str] = set()
        entries: list[dict[str, Any]] = []
        for candidate in region_candidates[region]:
            residue = str(candidate["residue_id"])
            if residue not in seen:
                entries.append(candidate)
                seen.add(residue)
            if len(entries) >= 8:
                break
        status = "COVERED" if entries else "BLOCKED_WITH_EVIDENCE"
        registry["regions"][region] = {
            "purpose": REGION_PURPOSES[region],
            "status": status,
            "residues": entries,
            "source_artifacts": [
                str(translation_pathway),
                str(mechanical_load),
                str(phase_coherence),
                str(hysteresis),
            ],
            "provenance_class": "L3_DERIVED" if entries else "L0_MISSING",
            "blocked_reason": None if entries else "No residue candidates matched available observatory evidence",
        }
    return registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--translation-pathway", type=Path, required=True)
    parser.add_argument("--mechanical-load", type=Path, required=True)
    parser.add_argument("--phase-coherence", type=Path, required=True)
    parser.add_argument("--hysteresis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    registry = build_registry(
        args.translation_pathway,
        args.mechanical_load,
        args.phase_coherence,
        args.hysteresis,
    )
    write_json(args.output, registry)
    covered = sum(1 for r in registry["regions"].values() if r["status"] == "COVERED")
    print(f"topology_region_registry regions=6 covered={covered} output={args.output}")


if __name__ == "__main__":
    main()

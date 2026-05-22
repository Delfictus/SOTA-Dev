#!/usr/bin/env python3
"""Build a chronic receptor durability evidence bridge from PRISM-DSTW layers.

This is an ontology/readiness artifact. It maps what the current aleniglipron
PRISM Twin/DSTW evidence can support, what it cannot support, and which
additional data classes are required before chronic biological durability can be
claimed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DOMAINS: list[dict[str, Any]] = [
    {
        "domain": "ligand_residence",
        "current_evidence": [
            "pocket-accessible SAR vectors",
            "spike event density near materialized sites",
            "dynamic aligned voxel event bins",
        ],
        "missing_evidence": [
            "explicit ligand atoms/pose trajectory",
            "bound/unbound transition ensemble",
            "residence-time survival model",
        ],
        "current_status": "not_claimable",
        "next_build": "ligand-bound restart/path-sampling with ligand-contact survival and unbinding coordinates",
    },
    {
        "domain": "receptor_conformational_cycling",
        "current_evidence": [
            "inactive/active local control topologies",
            "interface event-support transitions",
            "KCC endpoint deltas",
            "adaptive_dt/changepoint context",
        ],
        "missing_evidence": [
            "localized path-sampling trajectories",
            "state transition committor or MSM/TPT layer",
            "replicated active-inactive cycling protocol",
        ],
        "current_status": "partially_ready_for_sampling",
        "next_build": "execute localized interface-forming/breaking path-sampling windows and fit state-transition kinetics",
    },
    {
        "domain": "g_protein_coupling",
        "current_evidence": [
            "intracellular/downstream lock-surface hypotheses",
            "active-state structural control",
        ],
        "missing_evidence": [
            "GLP-1R:G protein complex topology",
            "coupling-interface contacts",
            "G protein engagement/dissociation trajectories",
        ],
        "current_status": "not_claimable",
        "next_build": "add receptor:G protein control runs and receptor-only differential coupling comparison",
    },
    {
        "domain": "arrestin_recruitment",
        "current_evidence": [
            "intracellular lock-surface hypotheses",
        ],
        "missing_evidence": [
            "phosphorylation barcode state",
            "arrestin-bound topology",
            "arrestin engagement kinetics",
        ],
        "current_status": "not_claimable",
        "next_build": "add phosphorylated receptor/arrestin-bound ensembles and recruitment event model",
    },
    {
        "domain": "desensitization",
        "current_evidence": [
            "quiet-thermal lock hypothesis",
            "event-support transitions at intracellular lock surfaces",
        ],
        "missing_evidence": [
            "G protein/arrestin competition layer",
            "GRK/phosphorylation state model",
            "functional repeated-stimulation response data",
        ],
        "current_status": "hypothesis_only",
        "next_build": "bridge interface persistence to signaling-competence states with orthogonal functional calibration",
    },
    {
        "domain": "internalization",
        "current_evidence": [],
        "missing_evidence": [
            "cellular trafficking state",
            "arrestin/AP2/clathrin context",
            "membrane compartment model",
        ],
        "current_status": "not_represented",
        "next_build": "separate cell-scale trafficking module; do not infer from short receptor-only MD",
    },
    {
        "domain": "recycling",
        "current_evidence": [],
        "missing_evidence": [
            "endosomal receptor state",
            "recycling/degradation branch labels",
            "long-timescale trafficking data",
        ],
        "current_status": "not_represented",
        "next_build": "cellular lifecycle model calibrated to trafficking assays",
    },
    {
        "domain": "degradation",
        "current_evidence": [],
        "missing_evidence": [
            "ubiquitination/degradation markers",
            "lysosomal/proteasomal routing context",
            "long-timescale abundance data",
        ],
        "current_status": "not_represented",
        "next_build": "proteostasis/turnover evidence layer independent of receptor conformational MD",
    },
    {
        "domain": "membrane_context",
        "current_evidence": [
            "receptor structural topologies",
        ],
        "missing_evidence": [
            "explicit membrane/lipid composition",
            "cholesterol/raft context",
            "lipid-contact time series",
        ],
        "current_status": "underrepresented",
        "next_build": "membrane-embedded receptor runs with lipid-contact event decoder",
    },
    {
        "domain": "cellular_adaptation",
        "current_evidence": [],
        "missing_evidence": [
            "transcriptional/proteomic adaptation data",
            "receptor expression feedback",
            "cell-type-specific signaling model",
        ],
        "current_status": "not_represented",
        "next_build": "systems pharmacology layer calibrated to repeated-exposure cell assays",
    },
    {
        "domain": "repeated_exposure_kinetics",
        "current_evidence": [
            "short-timescale event-support transition candidates",
        ],
        "missing_evidence": [
            "dose/pulse schedule",
            "multi-cycle receptor state carryover",
            "functional response decay/recovery data",
        ],
        "current_status": "not_claimable",
        "next_build": "multi-pulse simulation/assay bridge with state memory and recovery kinetics",
    },
]


def read_manifest(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text())


def table_count(path: Path | None) -> int | None:
    if not path or not path.exists():
        return None
    return pq.read_metadata(path).num_rows


def status_rank(status: str) -> int:
    return {
        "ready": 4,
        "partially_ready_for_sampling": 3,
        "hypothesis_only": 2,
        "underrepresented": 1,
        "not_claimable": 0,
        "not_represented": 0,
    }.get(status, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-id", default="glp1r_aleniglipron")
    parser.add_argument("--production-verification", type=Path, required=True)
    parser.add_argument("--dynamic-voxel-manifest", type=Path)
    parser.add_argument("--path-sampling-manifest", type=Path)
    parser.add_argument("--sar-ontology-factpack", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    verification = read_manifest(args.production_verification)
    dynamic_manifest = read_manifest(args.dynamic_voxel_manifest)
    sampling_manifest = read_manifest(args.path_sampling_manifest)
    sar_factpack = read_manifest(args.sar_ontology_factpack)

    current_layer_counts = {
        "raw_spike_events": sum(
            r.get("event_rows", r.get("records_written", r.get("source_records", 0)))
            for r in (verification.get("event_surface", {}) or {}).values()
            if isinstance(r, dict)
        ),
        "interface_hit_rows": sum(
            r.get("interface_hit_rows", 0)
            for r in (verification.get("event_surface", {}) or {}).values()
            if isinstance(r, dict)
        ),
        "dynamic_voxel_event_time_bins": (dynamic_manifest.get("counts", {}) or {}).get("dynamic_voxel_event_time_bins"),
        "interface_aligned_voxel_fields": (dynamic_manifest.get("counts", {}) or {}).get("interface_aligned_voxel_fields"),
        "path_sampling_launch_queue": (sampling_manifest.get("counts", {}) or {}).get("launch_queue"),
        "path_sampling_ranked_windows": (sampling_manifest.get("counts", {}) or {}).get("ranked_windows"),
    }

    rows: list[dict[str, Any]] = []
    for row in DOMAINS:
        evidence_available = bool(row["current_evidence"])
        has_sampling_queue = bool(current_layer_counts.get("path_sampling_launch_queue"))
        status = row["current_status"]
        if row["domain"] == "receptor_conformational_cycling" and has_sampling_queue:
            status = "partially_ready_for_sampling"
        rows.append(
            {
                "campaign_id": args.campaign_id,
                "durability_domain": row["domain"],
                "current_status": status,
                "readiness_rank": status_rank(status),
                "current_evidence_classes": "; ".join(row["current_evidence"]),
                "missing_evidence_classes": "; ".join(row["missing_evidence"]),
                "next_build_requirement": row["next_build"],
                "claim_boundary": (
                    "current PRISM evidence may guide mechanistic sampling"
                    if evidence_available
                    else "current PRISM evidence does not represent this biological domain"
                ),
            }
        )

    evidence_table = pa.Table.from_pylist(rows)
    evidence_path = args.out_dir / "chronic_durability_evidence_register.parquet"
    pq.write_table(evidence_table, evidence_path, compression="zstd")

    bridge_rows = [
        {
            "campaign_id": args.campaign_id,
            "current_layer": "PRSPK001 spike event surface",
            "maps_to": "short-timescale receptor mechanistic activity",
            "allowed_use": "temporal localization, residue/site/interface event support, hydration event statistics",
            "blocked_use": "direct chronic durability, residence time, internalization, recycling, degradation",
            "row_count": current_layer_counts.get("raw_spike_events"),
        },
        {
            "campaign_id": args.campaign_id,
            "current_layer": "interface transition candidates",
            "maps_to": "localized path-sampling target selection",
            "allowed_use": "choose interface-forming/breaking restart windows",
            "blocked_use": "claiming final interface-breaking timestamps without path-sampling validation",
            "row_count": current_layer_counts.get("path_sampling_ranked_windows"),
        },
        {
            "campaign_id": args.campaign_id,
            "current_layer": "dynamic aligned voxel event bins",
            "maps_to": "x/y/z/t/amplitude/hydration support around SAR interfaces and materialized sites",
            "allowed_use": "spatially localized event fields and path-sampling context",
            "blocked_use": "per-frame warp trajectory, ligand occupancy proof, chronic cell-level kinetics",
            "row_count": current_layer_counts.get("dynamic_voxel_event_time_bins"),
        },
        {
            "campaign_id": args.campaign_id,
            "current_layer": "KCC endpoint deltas",
            "maps_to": "per-residue causal/kinematic endpoint contrast",
            "allowed_use": "interface-local triage and endpoint coverage accounting",
            "blocked_use": "imputing missing endpoints or treating final vectors as full trajectories",
            "row_count": sum(
                r.get("rows", 0)
                for name, r in (verification.get("kcc", {}) or {}).items()
                if isinstance(r, dict) and name.endswith("interface_kcc_pair_deltas.parquet")
            ),
        },
    ]
    bridge_path = args.out_dir / "mechanistic_to_chronic_bridge_map.parquet"
    pq.write_table(pa.Table.from_pylist(bridge_rows), bridge_path, compression="zstd")

    report = {
        "schema": "prism_chronic_durability_bridge.v1",
        "campaign_id": args.campaign_id,
        "inputs": {
            "production_verification": str(args.production_verification),
            "dynamic_voxel_manifest": str(args.dynamic_voxel_manifest) if args.dynamic_voxel_manifest else None,
            "path_sampling_manifest": str(args.path_sampling_manifest) if args.path_sampling_manifest else None,
            "sar_ontology_factpack": str(args.sar_ontology_factpack) if args.sar_ontology_factpack else None,
        },
        "current_layer_counts": current_layer_counts,
        "outputs": {
            "evidence_register": str(evidence_path),
            "bridge_map": str(bridge_path),
        },
        "domain_counts": {
            "total_domains": len(rows),
            "partially_ready_for_sampling": sum(1 for r in rows if r["current_status"] == "partially_ready_for_sampling"),
            "hypothesis_only": sum(1 for r in rows if r["current_status"] == "hypothesis_only"),
            "not_claimable_or_not_represented": sum(
                1 for r in rows if r["current_status"] in {"not_claimable", "not_represented", "underrepresented"}
            ),
        },
        "semantic_gates": [
            "Chronic durability is not collapsed into a single PRISM score.",
            "Each biological lifecycle domain has its own evidence requirements.",
            "Current aleniglipron outputs support mechanistic sampling and SAR hypotheses, not direct chronic clinical durability.",
            "Repeated-exposure and cell-trafficking claims require additional biological timescale data.",
        ],
        "sar_factpack_keys": sorted(sar_factpack.keys()) if sar_factpack else [],
    }
    manifest_path = args.out_dir / "chronic_durability_bridge_manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    md_path = args.out_dir / "Chronic_Durability_Bridge_Readiness.md"
    lines = [
        "# PRISM Chronic Durability Bridge Readiness",
        "",
        "This register separates current PRISM Twin/DSTW mechanistic evidence from chronic biological durability claims.",
        "",
        "## Current Operational Evidence",
        "",
    ]
    for key, value in current_layer_counts.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Domain Readiness", ""])
    for row in rows:
        lines.append(
            f"- `{row['durability_domain']}`: `{row['current_status']}`; next build: {row['next_build_requirement']}"
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "The current evidence can drive SAR, localized path-sampling, and mechanistic hypotheses. It does not yet establish chronic receptor durability across residence, coupling, trafficking, degradation, adaptation, or repeated exposure.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines))

    print(json.dumps(report["domain_counts"], indent=2, sort_keys=True))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

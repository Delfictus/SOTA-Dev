#!/usr/bin/env python3
"""Build a machine-readable claim-to-falsification graph for the M2 release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TypedDict

import polars as pl


CAMPAIGN_DIR = Path("campaigns/glp1r_aleniglipron")
N80_DIR = CAMPAIGN_DIR / "integrated_spike_events/n80_full_scale"
DEFAULT_OUTPUT = CAMPAIGN_DIR / "claim_falsification_graph.json"


class ClaimGraph(TypedDict):
    schema_version: str
    campaign_id: str
    graph_semantics: str
    nodes: list[dict[str, object]]
    edges: list[dict[str, object]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cro-plan", type=Path, default=CAMPAIGN_DIR / "CRO_WetLab_Action_Plan.parquet")
    parser.add_argument("--risk-map", type=Path, default=N80_DIR / "receptor_durability_risk_map.parquet")
    parser.add_argument("--phase-coherence", type=Path, default=N80_DIR / "phase_manifold_coherence.parquet")
    parser.add_argument("--teaser-solutions", type=Path, default=CAMPAIGN_DIR / "track_0_manual_emulation/teaser_solutions.parquet")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def node(node_id: str, label: str, node_type: str, **attrs: object) -> dict[str, object]:
    payload: dict[str, object] = {"id": node_id, "label": label, "type": node_type}
    payload.update({key: json_safe(value) for key, value in attrs.items()})
    return payload


def edge(source: str, target: str, relation: str, **attrs: object) -> dict[str, object]:
    payload: dict[str, object] = {"source": source, "target": target, "relation": relation}
    payload.update({key: json_safe(value) for key, value in attrs.items()})
    return payload


def optional_rows(path: Path, limit: int) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return pl.scan_parquet(path).head(limit).collect().to_dicts()


def build_graph(args: argparse.Namespace) -> ClaimGraph:
    nodes: list[dict[str, object]] = [
        node("tensor:shear_stress_field", "Rust shear-stress field", "source_tensor", epistemic_class="OBSERVED"),
        node("tensor:kcc_residue_fields", "KCC residue burst-motion fields", "source_tensor", epistemic_class="INFERRED"),
        node("tensor:hysteresis_tensor", "CCNS hysteresis tensor", "source_tensor", epistemic_class="DERIVED"),
        node("tensor:phase_manifold_coherence", "Phase-manifold coherence", "source_tensor", epistemic_class="DERIVED"),
        node("assay:HDX-MS", "HDX-MS falsification gate", "falsification_assay"),
        node("assay:BRET_Kinetics", "BRET kinetics falsification gate", "falsification_assay"),
        node("assay:Washout_Recovery_Assay", "Washout recovery falsification gate", "falsification_assay"),
    ]
    edges: list[dict[str, object]] = [
        edge("tensor:shear_stress_field", "assay:HDX-MS", "falsified_by", failure_condition="Lack of WT-normalized uptake asymmetry"),
        edge("tensor:kcc_residue_fields", "assay:BRET_Kinetics", "falsified_by", failure_condition="Lack of predicted early-transition delay relative to initiation-wave residues"),
        edge("tensor:hysteresis_tensor", "assay:Washout_Recovery_Assay", "falsified_by", failure_condition="No WT-normalized recovery impairment after washout"),
    ]

    if args.cro_plan.exists():
        for row in optional_rows(args.cro_plan, 40):
            action_id = str(row["action_id"])
            assay = str(row["assay_category"])
            claim_id = f"claim:{action_id}"
            source_tensor = {
                "HDX-MS": "tensor:shear_stress_field",
                "BRET_Kinetics": "tensor:kcc_residue_fields",
                "Washout_Recovery_Assay": "tensor:hysteresis_tensor",
            }.get(assay, "tensor:phase_manifold_coherence")
            assay_id = f"assay:{assay}"
            nodes.append(
                node(
                    claim_id,
                    str(row["claim_at_risk"]),
                    "claim",
                    epistemic_class=str(row["epistemic_class"]),
                    confidence_class="PROJECTED_ASSAY_BEHAVIOR",
                    transform_chain=f"{source_tensor} -> assay_routing_recommendations -> CRO_WetLab_Action_Plan",
                    failure_condition=str(row["falsification_condition"]),
                )
            )
            edges.append(edge(claim_id, source_tensor, "rooted_in"))
            edges.append(edge(claim_id, assay_id, "tested_by", failure_condition=str(row["falsification_condition"])))

    for row in optional_rows(args.teaser_solutions, 10):
        rank = int(str(row["solution_rank"]))
        claim_id = f"claim:zero_shot_solution:{rank}"
        nodes.append(
            node(
                claim_id,
                f"Zero-shot replacement rank {rank} remains a projected chemistry hypothesis.",
                "claim",
                epistemic_class=str(row.get("solution_epistemic_class", "HYPOTHESIZED")),
                confidence_class="HYPOTHESIZED_CHEMISTRY",
                transform_chain="calibration_anchors_3d -> rigid_body_grafting -> signal_grid_variance_channel -> teaser_solutions",
                failure_condition="Synthesis fails, or the replacement lacks the projected binding and receptor-state transition profile in wet-lab assays.",
                smiles=str(row["canonical_smiles"]),
            )
        )
        edges.append(edge(claim_id, "tensor:phase_manifold_coherence", "bounded_by"))

    return {
        "schema_version": "claim_falsification_graph.v1",
        "campaign_id": "glp1r_aleniglipron",
        "graph_semantics": "Claim Node -> Source Tensor -> Transform Chain -> Confidence Class -> Falsification Assay -> Failure Condition",
        "nodes": nodes,
        "edges": edges,
    }


def main() -> None:
    args = parse_args()
    graph = build_graph(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} nodes={len(graph['nodes'])} edges={len(graph['edges'])}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase 2 — table exports for PDF/LaTeX delivery.

Reads ground-truth parquets/JSONs verbatim (no transforms on values) and
emits per-table CSV (full) + Markdown (compact) into 09_TABLE_EXPORTS/.

No value mutation. No claim derivation. No projected->observed promotion.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

REPO = Path("/home/diddy/Desktop/Prism4D-bio")
DROOT = REPO / "PRISM_GLP1R_M2_DELIVERABLES_v1_1"
GT = DROOT / "05_GROUND_TRUTH_DATA" / "campaigns" / "glp1r_aleniglipron"
OUT = DROOT / "09_TABLE_EXPORTS"


def fmt_cell(v: Any, max_len: int = 60) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if v != v:
            return ""
        return f"{v:.4g}"
    s = str(v).replace("|", "\\|").replace("\n", " ")
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def df_to_md(df: pl.DataFrame, max_rows: int = 50, max_cell: int = 60) -> str:
    if df.height == 0:
        return "_(no rows)_\n"
    truncated = df.height > max_rows
    df = df.head(max_rows)
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.iter_rows():
        out.append("| " + " | ".join(fmt_cell(v, max_cell) for v in row) + " |")
    if truncated:
        out.append(f"\n_…truncated, {df.height} of original rows shown._")
    return "\n".join(out) + "\n"


def write_pair(stem: str, full_df: pl.DataFrame, md_df: pl.DataFrame,
               md_intro: str = "", md_max_rows: int = 50, md_max_cell: int = 60) -> None:
    full_df.write_csv(OUT / f"{stem}.csv")
    body = df_to_md(md_df, max_rows=md_max_rows, max_cell=md_max_cell)
    (OUT / f"{stem}.md").write_text(f"# {stem}\n\n{md_intro}\n\n{body}\n")
    print(f"  {stem}: {full_df.height} rows -> CSV + MD")


def export_cro() -> None:
    df = pl.read_parquet(GT / "CRO_WetLab_Action_Plan.parquet")
    md = df.select([
        "action_id", "epistemic_class", "assay_category", "construct",
        "residue_idx", "residue_name", "claim_at_risk",
        "priority_score", "falsification_condition",
    ])
    intro = ("PRISM GLP-1R M2 CRO wet-lab action plan. Every row is a "
             "**falsification gate**, not a confirmation request. "
             "`epistemic_class` column governs how each row may be cited.")
    write_pair("CRO_WetLab_Action_Plan", df, md, intro)


def export_zero_shot_top10() -> None:
    src = pl.read_parquet(GT / "track_0_manual_emulation" / "teaser_solutions.parquet")
    full = src.sort("solution_rank").head(10).select([
        "solution_rank", "anchor_id", "canonical_smiles",
        "sa_score", "pi_complement", "pi_clash",
        "projected_durability_improvement",
        "liability_edge_label",
        "anchor_epistemic_class",
        "solution_epistemic_class",
    ])
    md = full.select([
        "solution_rank", "anchor_id",
        "sa_score", "pi_complement", "pi_clash",
        "projected_durability_improvement",
        "liability_edge_label",
        "solution_epistemic_class",
    ])
    intro = ("Top-10 PROJECTED / HYPOTHESIZED fragment replacements derived "
             "from the manual emulation track. **These are not validated "
             "compounds and not synthesis instructions.** `solution_epistemic_class` "
             "governs how each row may be cited. SMILES strings are preserved "
             "in the CSV (omitted from MD for width).")
    write_pair("ZeroShot_Top10_Replacements", full, md, intro)


def export_fragment_interference() -> None:
    df = pl.read_parquet(GT / "track_0_manual_emulation" / "fragment_interference_attribution.parquet")
    md = df.select([
        "edge_id",
        "whole_molecule_clash", "whole_molecule_complement",
        "sum_fragment_clash", "sum_fragment_complement",
        "dominant_fragment",
        "dominant_fragment_clash", "dominant_fraction",
    ])
    intro = ("Per-edge fragment interference attribution. Decomposes "
             "whole-molecule pi clash/complement into fragment contributions. "
             "INFERRED epistemic class — multi-tensor interpretation.")
    write_pair("Fragment_Interference_Attribution", df, md, intro)


def export_critical_edges() -> None:
    df = pl.read_parquet(GT / "integrated_spike_events" / "n80_full_scale" / "phase_manifold_edge_validation.parquet")
    md = df.select([
        "edge_id", "edge_label", "edge_class", "validation_status",
        "durability_risk_score_raw",
        "from_coherence_class", "to_coherence_class",
        "edge_coherence_score",
    ])
    intro = ("Critical-edge validation from the phase-manifold layer. "
             "`validation_status` is the operative gate. DERIVED epistemic class.")
    write_pair("Critical_Edge_Validation", df, md, intro)


def export_translation_pathway() -> None:
    df = pl.read_parquet(GT / "integrated_spike_events" / "n80_full_scale" / "translation_pathway_nodes.parquet")
    md = df.sort("pathway_rank").select([
        "pathway_rank", "residue_idx", "residue_name",
        "coherence_class", "evidence_class",
        "shear_stress_abs_p90", "max_burst_motion",
        "wire_score", "structural_fault_line",
        "violent_kinetic_node",
    ])
    intro = ("Translation-pathway nodes (ranked by `pathway_rank`). "
             "Boolean flags `structural_fault_line` and `violent_kinetic_node` "
             "are tensor-derived characterizations, not biological assertions.")
    write_pair("Translation_Pathway_Nodes", df, md, intro)


def export_phase2c_triggers() -> None:
    payload = json.loads((GT / "phase_2c_metastable_atlas_triggers.json").read_text())
    triggers = payload.get("triggers", [])
    flat_rows = []
    for t in triggers:
        flat_rows.append({
            "trigger_id":       t.get("trigger_id"),
            "condition_id":     t.get("condition_id"),
            "stream_idx":       t.get("stream_idx"),
            "window_start":     t.get("window_start"),
            "window_end":       t.get("window_end"),
            "centroid_class":   t.get("centroid_class"),
            "metric":           t.get("metric"),
            "metric_value":     t.get("metric_value"),
            "rationale":        t.get("rationale"),
        })
    df = pl.DataFrame(flat_rows) if flat_rows else pl.DataFrame()
    md = df.head(25) if df.height else df
    intro = (f"Phase 2C metastable-atlas trigger summary. "
             f"`trigger_count`={payload.get('trigger_count')}, "
             f"capture_mode={payload.get('capture_mode')!r}, "
             f"stride={payload.get('stride')}.")
    write_pair("Phase2C_Metastable_Trigger_Summary", df, md, intro)


def export_phase2d_staged() -> None:
    payload = json.loads((GT / "phase_2d_variant_grid_manifest.json").read_text())
    rows = []
    for rec in payload.get("current_condition_records", []):
        rows.append({
            "condition_id":     rec.get("condition_id"),
            "variant":          rec.get("variant"),
            "background":       rec.get("background"),
            "topology_source":  rec.get("topology_source"),
            "evidence_status":  rec.get("evidence_status"),
            "epistemic_class":  rec.get("epistemic_class"),
            "n_replicas":       rec.get("n_replicas"),
            "engine_run_id":    rec.get("engine_run_id"),
        })
    df = pl.DataFrame(rows) if rows else pl.DataFrame()
    intro = ("Phase 2D variant-grid staged targets. "
             "**Materialization status:** "
             f"`{payload.get('materialization_status')}` — these are *staged*, "
             "not engine-executed. The Phase 2D manifest is a planning artifact.")
    write_pair("Phase2D_Staged_Targets", df, df, intro)


def export_claim_graph() -> None:
    payload = json.loads((GT / "claim_falsification_graph.json").read_text())
    edges = payload.get("edges", [])
    nodes_by_id = {n["id"]: n for n in payload.get("nodes", [])}
    claims_by_id = {c["id"]: c for c in payload.get("claims", [])}
    rows = []
    for e in edges:
        src_id = e.get("source")
        tgt_id = e.get("target")
        claim = claims_by_id.get(tgt_id) or claims_by_id.get(src_id) or {}
        rows.append({
            "source_id":          src_id,
            "source_label":       nodes_by_id.get(src_id, {}).get("label"),
            "target_id":          tgt_id,
            "target_label":       nodes_by_id.get(tgt_id, {}).get("label"),
            "relationship":       e.get("relationship"),
            "claim_id":           claim.get("id"),
            "claim_epistemic":    claim.get("epistemic_class"),
            "claim_assay":        claim.get("assay_category"),
            "claim_condition":    claim.get("condition_id"),
            "claim_failure_condition": claim.get("failure_condition"),
        })
    df = pl.DataFrame(rows) if rows else pl.DataFrame()
    md_view = df.select([
        "source_id", "target_id", "relationship",
        "claim_epistemic", "claim_assay", "claim_condition",
    ]) if df.height else df
    intro = ("Claim/falsification graph in flattened edge-table form. "
             "`claim_epistemic` is the gate that governs how each edge may be "
             "cited. PROJECTED and HYPOTHESIZED edges require wet-lab "
             "falsification before any biological assertion.")
    write_pair("Claim_Falsification_Graph", df, md_view, intro, md_max_cell=70)


def export_cbom_summary() -> None:
    payload = json.loads((GT / "PRISM_CBOM_v1.0.json").read_text())
    env = payload.get("environment", {})
    files = payload.get("files", [])
    by_role = Counter(f.get("epistemic_role") or f.get("artifact_category") or "unknown" for f in files)
    by_ext = Counter(Path(f.get("path", "")).suffix.lower() for f in files)
    lines = [
        "# CBOM Summary",
        "",
        f"- **Campaign ID:** `{payload.get('campaign_id')}`",
        f"- **Campaign Merkle root:** `{payload.get('campaign_merkle_root')}`",
        f"- **Schema version:** `{payload.get('schema_version')}`",
        f"- **Directories:** {payload.get('directory_count')}",
        f"- **Files:** {payload.get('file_count')}",
        "",
        "## Environment",
        "",
    ]
    for k, v in env.items():
        lines.append(f"- **{k}:** `{v}`")
    lines += [
        "",
        "## File breakdown by epistemic role / category",
        "",
        "| role / category | count |",
        "|---|---|",
    ]
    for role, n in sorted(by_role.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {role} | {n} |")
    lines += [
        "",
        "## File breakdown by extension",
        "",
        "| extension | count |",
        "|---|---|",
    ]
    for ext, n in sorted(by_ext.items(), key=lambda kv: -kv[1])[:20]:
        lines.append(f"| `{ext or '(none)'}` | {n} |")
    (OUT / "CBOM_Summary.md").write_text("\n".join(lines) + "\n")
    print(f"  CBOM_Summary: merkle={payload.get('campaign_merkle_root')[:12]}…, files={len(files)}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Exporting tables to {OUT}")
    export_cro()
    export_zero_shot_top10()
    export_fragment_interference()
    export_critical_edges()
    export_translation_pathway()
    export_phase2c_triggers()
    export_phase2d_staged()
    export_claim_graph()
    export_cbom_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())

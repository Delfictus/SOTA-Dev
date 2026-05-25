#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/diddy/Desktop/Prism4D-bio"
CAMPAIGN="$ROOT/campaigns/glp1r_aleniglipron"
N80="$CAMPAIGN/integrated_spike_events/n80_full_scale"
EXEC_STAGING="$ROOT/release_build/PRISM_GLP1R_M2_EXECUTIVE_RELEASE_v1.1"
EXEC_ARCHIVE="$ROOT/PRISM_GLP1R_M2_EXECUTIVE_RELEASE_v1.1.tar.gz"
EXEC_SIG="$ROOT/PRISM_GLP1R_M2_EXECUTIVE_RELEASE_v1.1.tar.gz.sha256"

cd "$ROOT"

echo "======================================================================"
echo " PRISM M2 v1.1 MATERIALIZATION + EXECUTIVE RELEASE"
echo "======================================================================"

echo
echo "=== 1. MATERIALIZING PHASE 2D VARIANT GRID MANIFEST ==="

PYTHONPATH=src python3 - <<'PY'
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import re
import subprocess
from typing import Any

import polars as pl

ROOT = Path("/home/diddy/Desktop/Prism4D-bio")
CAMPAIGN = ROOT / "campaigns/glp1r_aleniglipron"
N80 = CAMPAIGN / "integrated_spike_events/n80_full_scale"

OUT = CAMPAIGN / "phase_2d_variant_grid_manifest.json"
LEDGER = CAMPAIGN / "phase_2d_variant_grid_manifest.propagation.jsonl"
MASTER = CAMPAIGN / "MASTER_DATA_ROOM_INDEX.md"


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024 * 8), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def split_condition(condition_id: str) -> dict[str, str]:
    m = re.match(r"^(?P<state>.+)_(?P<background>WT|A\d+[A-Z]|T\d+[A-Z])$", condition_id)
    if not m:
        return {"condition_id": condition_id, "state": condition_id, "background": "UNKNOWN"}
    return {"condition_id": condition_id, "state": m.group("state"), "background": m.group("background")}


def read_condition_ids(path: Path) -> set[str]:
    if not path.exists() or path.suffix != ".parquet":
        return set()
    try:
        schema = pl.read_parquet_schema(path)
        if "condition_id" not in schema:
            return set()
        df = pl.scan_parquet(path).select("condition_id").unique().collect()
        return set(str(x) for x in df["condition_id"].to_list())
    except Exception:
        return set()


source_artifacts = [
    N80 / "variant_propagation_deltas.parquet",
    N80 / "receptor_durability_risk_map.parquet",
    N80 / "phase_manifold_edge_validation.parquet",
    N80 / "phase_manifold_coherence.parquet",
    N80 / "translation_pathway_nodes.parquet",
    N80 / "hysteresis_tensor.parquet",
    N80 / "occupancy_fatigue_risk.parquet",
    CAMPAIGN / "M2_Replayability_Manifest.json",
    CAMPAIGN / "PRISM_CBOM_v1.0.json",
]

condition_ids: set[str] = set()
for src in source_artifacts:
    condition_ids |= read_condition_ids(src)

condition_records = [split_condition(c) for c in sorted(condition_ids)]
states = sorted({r["state"] for r in condition_records})
backgrounds = sorted({r["background"] for r in condition_records})

matrix: dict[str, dict[str, bool]] = {}
for r in condition_records:
    matrix.setdefault(r["state"], {})
    matrix[r["state"]][r["background"]] = True

state_background_matrix = [
    {
        "state": state,
        "backgrounds_present": sorted(matrix.get(state, {}).keys()),
        "has_WT": bool(matrix.get(state, {}).get("WT", False)),
        "has_A316T": bool(matrix.get(state, {}).get("A316T", False)),
        "has_T149M": bool(matrix.get(state, {}).get("T149M", False)),
    }
    for state in states
]

critical_edges: list[dict[str, Any]] = []
edge_path = N80 / "phase_manifold_edge_validation.parquet"
if edge_path.exists():
    try:
        df = pl.read_parquet(edge_path)
        wanted = [
            c for c in [
                "edge_id",
                "condition_id",
                "edge_label",
                "edge_class",
                "edge_coherence_score",
                "validation_status",
            ]
            if c in df.columns
        ]
        critical_edges = df.select(wanted).to_dicts()
    except Exception:
        critical_edges = []

variant_delta_summary: dict[str, Any] = {}
delta_path = N80 / "variant_propagation_deltas.parquet"
if delta_path.exists():
    try:
        df = pl.read_parquet(delta_path)
        variant_delta_summary = {"row_count": df.height, "columns": df.columns}
        for col in ["delta_risk", "risk_delta", "variant_delta"]:
            if col in df.columns:
                variant_delta_summary[f"{col}_max"] = float(df[col].max())
                variant_delta_summary[f"{col}_min"] = float(df[col].min())
                break
    except Exception as exc:
        variant_delta_summary = {"read_error": str(exc)}

staged_biological_targets = [
    {
        "variant": "N182A",
        "epistemic_class": "PROJECTED",
        "source_status": "platform_nominated_requires_canonical_residue_mapping_confirmation",
        "numbering_policy": "Do not execute until UniProt/PDB/internal residue-index correspondence is confirmed.",
        "rationale": "Wire-severing perturbation candidate targeting the PRISM-nominated ASN182 transmission feature to test mechanical-load decoupling.",
    },
    {
        "variant": "H108Q",
        "epistemic_class": "PROJECTED",
        "source_status": "user_supplied_pending_literature_or_program_confirmation",
        "numbering_policy": "Requires canonical GLP-1R numbering confirmation before execution.",
        "rationale": "Candidate active-state background probe intended to test baseline shifts in quiet-thermal lock behavior without overclaiming prior biological validation.",
    },
    {
        "variant": "L260A",
        "epistemic_class": "PROJECTED",
        "source_status": "user_supplied_pending_literature_or_program_confirmation",
        "numbering_policy": "Requires canonical GLP-1R numbering confirmation before execution.",
        "rationale": "Candidate pathway-bifurcation probe intended to test whether tensor channels separate G-protein-proximal and arrestin-proximal response surfaces.",
    },
    {
        "variant": "P356L",
        "epistemic_class": "PROJECTED",
        "source_status": "population_variant_status_requires_citation_or_client_confirmation",
        "numbering_policy": "Requires variant database/literature confirmation and receptor mapping before execution.",
        "rationale": "Candidate downstream-lock-neighborhood background for testing genotype-conditioned durability drift.",
    },
    {
        "variant": "R190Q",
        "epistemic_class": "PROJECTED",
        "source_status": "population_variant_status_requires_citation_or_client_confirmation",
        "numbering_policy": "Requires variant database/literature confirmation and receptor mapping before execution.",
        "rationale": "Candidate orthosteric-neighborhood background for testing ligand-entry and steric-interference sensitivity.",
    },
    {
        "variant": "E138A",
        "epistemic_class": "PROJECTED",
        "source_status": "user_supplied_pending_literature_or_program_confirmation",
        "numbering_policy": "Requires canonical GLP-1R numbering confirmation before execution.",
        "rationale": "Candidate extracellular-propagation probe for testing mechanical-strain transfer from extracellular/stalk-adjacent regions into transmembrane-core response surfaces.",
    },
]

manifest = {
    "schema_version": "PRISM.phase_2d_variant_grid_manifest.v1",
    "campaign_id": "glp1r_aleniglipron",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "created_by": "materialize_phase_2d_variant_grid_manifest_v1_1",
    "git_sha": git_sha(),
    "epistemic_class": "PROJECTED",
    "materialization_status": "staged_manifest_materialized_not_executed",
    "engine_execution_status": {
        "prism4d_engine_run_executed_for_this_manifest": False,
        "reason": "This artifact materializes Phase 2D scope and launch contract only. It is not a completed Phase 2D result tensor.",
    },
    "scope": {
        "phase": "2D",
        "name": "Variant-conditioned propagation expansion grid",
        "purpose": "Define next-SOW variant expansion grid and acceptance outputs for WT/variant resilience assessment.",
        "commercial_boundary": "Included as a staged manifest in M2 release; execution requires Phase 2D SOW authorization.",
    },
    "current_covered_backgrounds": backgrounds,
    "current_condition_records": condition_records,
    "state_background_matrix": state_background_matrix,
    "existing_variant_evidence": {
        "covered_backgrounds": [b for b in backgrounds if b != "UNKNOWN"],
        "known_non_wt_backgrounds": [b for b in backgrounds if b not in {"WT", "UNKNOWN"}],
        "variant_delta_summary": variant_delta_summary,
    },
    "phase_2d_expansion_strategy": {
        "target_background_count": "12-18 total backgrounds after SOW authorization",
        "staged_biological_targets": staged_biological_targets,
        "selection_rules": [
            "Prioritize variant backgrounds by clinical/population relevance only when supplied by Structure or backed by agreed citation source.",
            "Prioritize residues proximal to validated_constitutive pocket_vector and downstream_lock edges.",
            "Prioritize mutations affecting translation-wire residues, downstream-lock neighborhoods, or hysteresis-sensitive corridors.",
            "Reject expansion candidates without mappable receptor topology or residue-index correspondence.",
            "Do not claim biological effect from manifest materialization alone; all outputs remain projected until PRISM execution and wet-lab falsification.",
        ],
        "required_inputs_before_execution": [
            "Approved variant/background list",
            "Residue-index mapping for each background",
            "State-compatible topology or mutagenesis build path",
            "Matched WT control policy",
            "Agreed acceptance thresholds for convergence, pruning, and falsification routing",
            "Citation or client confirmation for any target described as population, clinical, CAM, biased-signaling, or literature-backed.",
        ],
    },
    "planned_output_contract": [
        "variant_propagation_deltas.parquet",
        "variant_resilience_scorecard.parquet",
        "variant_conditioned_phase_manifold_coherence.parquet",
        "variant_hysteresis_recovery_matrix.parquet",
        "variant_assay_routing_delta.parquet",
        "variant_negative_evidence_register.parquet",
        "Phase_2D_Variant_Resilience_Dossier.md",
    ],
    "acceptance_criteria": {
        "minimum_evidence": "Two-of-three independent evidence classes required before routing a variant-sensitive surface as actionable.",
        "required_controls": [
            "WT matched state",
            "A316T/T149M reference backgrounds where applicable",
            "mechanically_pruned negative register",
            "phase-validation status propagation",
        ],
        "claim_policy": {
            "OBSERVED": "Only direct tensor measurements after execution.",
            "DERIVED": "Deterministic transforms of executed tensors.",
            "PROJECTED": "This manifest and all pre-execution variant recommendations.",
            "HYPOTHESIZED": "Biological interpretation requiring wet-lab falsification.",
        },
    },
    "critical_edges_used_for_prioritization": critical_edges,
    "source_artifacts": [
        {
            "path": str(path.relative_to(ROOT)) if path.exists() else str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path),
        }
        for path in source_artifacts
    ],
    "non_claim_disclaimer": (
        "This manifest is a materialized launch/scope artifact. It does not represent completed "
        "Phase 2D simulation results, clinical claims, patient-response claims, or experimentally "
        "validated pharmacology."
    ),
}

OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

ledger_record = {
    "artifact": str(OUT.relative_to(ROOT)),
    "sha256": sha256_file(OUT),
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "generator": "materialize_phase_2d_variant_grid_manifest_v1_1",
    "source_artifacts": manifest["source_artifacts"],
    "runtime": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "polars": pl.__version__,
    },
}
LEDGER.write_text(json.dumps(ledger_record) + "\n", encoding="utf-8")

if MASTER.exists():
    s = MASTER.read_text(encoding="utf-8")
    rows = [
        (
            "| Phase 2D Variant Grid Manifest | `campaigns/glp1r_aleniglipron/phase_2d_variant_grid_manifest.json` | not_materialized |",
            "| Phase 2D Variant Grid Manifest | `phase_2d_variant_grid_manifest.json` | staged_manifest_materialized_not_executed |",
        ),
        (
            "| Phase 2D Variant Grid Manifest | `campaigns/glp1r_aleniglipron/phase_2d_variant_grid_manifest.json` | staged_manifest_materialized_not_executed |",
            "| Phase 2D Variant Grid Manifest | `phase_2d_variant_grid_manifest.json` | staged_manifest_materialized_not_executed |",
        ),
    ]
    for old, new in rows:
        s = s.replace(old, new)
    MASTER.write_text(s, encoding="utf-8")

print(f"WROTE {OUT}")
print(f"WROTE {LEDGER}")
print(f"sha256={sha256_file(OUT)}")
PY

echo
echo "=== 2. POPULATING CLAIM FALSIFICATION GRAPH WITHOUT DEGRADING GRAPH STRUCTURE ==="

PYTHONPATH=src python3 - <<'PY'
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path("/home/diddy/Desktop/Prism4D-bio")
CAMPAIGN = ROOT / "campaigns/glp1r_aleniglipron"
GRAPH = CAMPAIGN / "claim_falsification_graph.json"
CRO = CAMPAIGN / "CRO_WetLab_Action_Plan.parquet"
FRAG = CAMPAIGN / "track_0_manual_emulation/fragment_interference_attribution.parquet"

existing: dict[str, Any] = {}
if GRAPH.exists():
    try:
        loaded = json.loads(GRAPH.read_text())
        if isinstance(loaded, dict):
            existing = loaded
    except Exception:
        existing = {}

claims: list[dict[str, Any]] = []
nodes: list[dict[str, Any]] = []
edges: list[dict[str, Any]] = []

if CRO.exists():
    df = pl.read_parquet(CRO)
    for row in df.iter_rows(named=True):
        claim_id = f"claim:{row['action_id']}"
        source_id = f"tensor_source:{row['source_trigger_rule']}"
        assay_id = f"assay:{row['assay_category']}:{row['condition_id']}:{row['residue_name']}:{row['residue_idx']}"

        claims.append({
            "id": claim_id,
            "label": row["claim_at_risk"],
            "epistemic_class": row["epistemic_class"],
            "transform_chain": "tensor source -> assay_routing_recommendations -> CRO_WetLab_Action_Plan",
            "failure_condition": row["falsification_condition"],
            "assay_category": row["assay_category"],
            "condition_id": row["condition_id"],
            "residue": f"{row['residue_name']}:{row['residue_idx']}",
        })
        nodes.extend([
            {"id": claim_id, "type": "claim", "epistemic_class": row["epistemic_class"], "label": row["claim_at_risk"]},
            {"id": source_id, "type": "source_tensor_or_trigger", "label": row["source_trigger_rule"]},
            {"id": assay_id, "type": "falsification_assay", "label": row["assay_category"]},
        ])
        edges.extend([
            {"source": source_id, "target": claim_id, "relationship": "supports_projected_claim"},
            {"source": claim_id, "target": assay_id, "relationship": "requires_falsification_by"},
        ])

claims.append({
    "id": "claim:HYSTERESIS:6LN2_A316T",
    "label": "Elevated thermal hysteresis asymmetry consistent with persistent recovery impairment signature",
    "epistemic_class": "INFERRED",
    "transform_chain": "hysteresis_tensor -> occupancy/recovery interpretation -> CRO falsification gate",
    "failure_condition": "Falsified if matched WT-normalized washout assay shows no recovery asymmetry.",
})
nodes.extend([
    {"id": "claim:HYSTERESIS:6LN2_A316T", "type": "claim", "epistemic_class": "INFERRED", "label": "Elevated thermal hysteresis asymmetry consistent with persistent recovery impairment signature"},
    {"id": "tensor:hysteresis_tensor", "type": "source_tensor", "label": "hysteresis_tensor.parquet"},
    {"id": "assay:washout_recovery", "type": "falsification_assay", "label": "Washout recovery assay"},
])
edges.extend([
    {"source": "tensor:hysteresis_tensor", "target": "claim:HYSTERESIS:6LN2_A316T", "relationship": "supports_inference"},
    {"source": "claim:HYSTERESIS:6LN2_A316T", "target": "assay:washout_recovery", "relationship": "requires_falsification_by"},
])

if FRAG.exists():
    frag_df = pl.read_parquet(FRAG)
    positive = frag_df.filter(pl.col("inter_fragment_coupling") > 0) if "inter_fragment_coupling" in frag_df.columns else frag_df
    total_positive_coupling = float(positive["inter_fragment_coupling"].sum()) if "inter_fragment_coupling" in positive.columns else None
    max_edge_clash = float(frag_df["whole_molecule_clash"].max()) if "whole_molecule_clash" in frag_df.columns else None
else:
    total_positive_coupling = None
    max_edge_clash = None

claims.append({
    "id": "claim:COUPLING:ALENI-PARENT",
    "label": "Positive inter-fragment coupling in projected scaffold-field scoring",
    "epistemic_class": "PROJECTED",
    "transform_chain": "fragment_interference_attribution -> thermodynamic ray-casting -> SAR contingency register",
    "supporting_values": {
        "total_positive_inter_fragment_coupling": total_positive_coupling,
        "max_whole_molecule_clash": max_edge_clash,
    },
    "failure_condition": "Falsified if matched analog controls do not show differential target engagement or orthogonal structural response.",
})
nodes.extend([
    {"id": "claim:COUPLING:ALENI-PARENT", "type": "claim", "epistemic_class": "PROJECTED", "label": "Positive inter-fragment coupling in projected scaffold-field scoring"},
    {"id": "tensor:fragment_interference_attribution", "type": "source_tensor", "label": "fragment_interference_attribution.parquet"},
    {"id": "assay:matched_analog_controls", "type": "falsification_assay", "label": "Matched analog controls"},
])
edges.extend([
    {"source": "tensor:fragment_interference_attribution", "target": "claim:COUPLING:ALENI-PARENT", "relationship": "supports_projected_claim"},
    {"source": "claim:COUPLING:ALENI-PARENT", "target": "assay:matched_analog_controls", "relationship": "requires_falsification_by"},
])

# de-duplicate nodes and edges
node_map = {n["id"]: n for n in nodes if "id" in n}
edge_keys = set()
edge_list = []
for e in edges:
    key = (e.get("source"), e.get("target"), e.get("relationship"))
    if key not in edge_keys:
        edge_keys.add(key)
        edge_list.append(e)

out = {
    "campaign_id": "glp1r_aleniglipron",
    "schema_version": "PRISM.claim_falsification_graph.v1.1",
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "graph_semantics": {
        "purpose": "Trace translational claims to source tensors and falsification assays.",
        "claim_policy": {
            "OBSERVED": "Direct tensor-derived measurement.",
            "DERIVED": "Deterministic transform of observed tensors.",
            "INFERRED": "Mechanistic interpretation from multi-tensor convergence.",
            "PROJECTED": "Translational extrapolation requiring falsification.",
            "HYPOTHESIZED": "Biological interpretation requiring experimental falsification.",
        },
    },
    "claims": claims,
    "nodes": list(node_map.values()),
    "edges": edge_list,
    "summary": {
        "claim_count": len(claims),
        "node_count": len(node_map),
        "edge_count": len(edge_list),
    },
}

GRAPH.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"WROTE {GRAPH}")
print(f"claim_count={len(claims)} node_count={len(node_map)} edge_count={len(edge_list)}")
PY

echo
echo "=== 3. RE-SEALING CBOM ==="

if [[ -f "$ROOT/scripts/build_campaign_cbom.py" ]]; then
  PYTHONPATH=src python3 "$ROOT/scripts/build_campaign_cbom.py"
else
  echo "[WARN] scripts/build_campaign_cbom.py not found; skipping CBOM rebuild."
fi

echo
echo "=== 4. BUILDING ALLOWLIST-BASED EXECUTIVE RELEASE BUNDLE ==="

rm -rf "$EXEC_STAGING"
mkdir -p "$EXEC_STAGING"

copy_file() {
  local src="$1"
  local dst="$EXEC_STAGING/${src#$ROOT/}"
  if [[ -f "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$dst"
    echo "[COPY] $src"
  else
    echo "[SKIP] $src"
  fi
}

copy_dir() {
  local src="$1"
  local dst="$EXEC_STAGING/${src#$ROOT/}"
  if [[ -d "$src" ]]; then
    mkdir -p "$(dirname "$dst")"
    cp -a "$src" "$dst"
    echo "[COPY DIR] $src"
  else
    echo "[SKIP DIR] $src"
  fi
}

# Core reports and data-room artifacts
copy_file "$CAMPAIGN/MASTER_DATA_ROOM_INDEX.md"
copy_file "$CAMPAIGN/M2_Pharmacological_Dynamics_Intelligence_Report.md"
copy_file "$CAMPAIGN/M2_Triangulation_Dossier_Final.md"
copy_file "$CAMPAIGN/M2_Executive_Readout_Final.md"
copy_file "$CAMPAIGN/ENTERPRISE_POSITIONING_SUMMARY.md"
copy_file "$CAMPAIGN/claim_falsification_graph.json"
copy_file "$CAMPAIGN/PRISM_CBOM_v1.0.json"
copy_file "$CAMPAIGN/M2_Replayability_Manifest.json"
copy_file "$CAMPAIGN/CRO_WetLab_Action_Plan.parquet"
copy_file "$CAMPAIGN/GLP1R_LIGAND_SET_MANIFEST_v1.parquet"
copy_file "$CAMPAIGN/GLP1R_LIGAND_SET_MANIFEST_v1.csv"

# Phase 2C / 2D manifests
copy_file "$CAMPAIGN/phase_2c_metastable_atlas_triggers.json"
copy_file "$CAMPAIGN/phase_2c_snapshot_triggers.json"
copy_file "$CAMPAIGN/phase_2c_reintegration_parity.json"
copy_file "$CAMPAIGN/phase_2d_variant_grid_manifest.json"
copy_file "$CAMPAIGN/phase_2d_variant_grid_manifest.propagation.jsonl"

# Selected reviewable tensors
copy_file "$N80/phase_manifold_coherence.parquet"
copy_file "$N80/phase_manifold_edge_validation.parquet"
copy_file "$N80/translation_pathway_nodes.parquet"
copy_file "$N80/hysteresis_tensor.parquet"
copy_file "$N80/temporal_cascade.parquet"
copy_file "$N80/temporal_cascade_summary.parquet"
copy_file "$N80/assay_routing_recommendations.parquet"
copy_file "$N80/occupancy_fatigue_risk.parquet"
copy_file "$N80/variant_propagation_deltas.parquet"
copy_file "$N80/probabilistic_break_clusters.parquet"
copy_file "$N80/tri_state_ligand_fiber_graph.parquet"
copy_file "$N80/receptor_durability_risk_map.parquet"
copy_file "$N80/receptor_durability_channel_summary.parquet"
copy_file "$N80/wire_conservation_matrix.parquet"
copy_file "$N80/stream_level_phase_counts.parquet"

# Track 0 / medicinal chemistry
copy_file "$CAMPAIGN/track_0_manual_emulation/teaser_solutions.parquet"
copy_file "$CAMPAIGN/track_0_manual_emulation/fragment_interference_attribution.parquet"
copy_file "$CAMPAIGN/track_0_manual_emulation/aleniglipron_interference_summary.md"
copy_file "$CAMPAIGN/track_0_manual_emulation/aleniglipron_brics_fragment_registry.json"
copy_file "$CAMPAIGN/track_0_manual_emulation/aleniglipron_two_layer_gate_report.json"
copy_file "$CAMPAIGN/track_0_manual_emulation/layer1_whole_molecule/per_edge_interference.parquet"
copy_file "$CAMPAIGN/track_0_manual_emulation/layer1_whole_molecule/analog_durability_projection.parquet"

# Track A readiness
copy_file "$CAMPAIGN/track_a_generative/115k_curated_anchors.csv"
copy_file "$CAMPAIGN/track_a_generative/calibration_anchors_3d.parquet"
copy_file "$CAMPAIGN/track_a_generative/gflownet_tso_bridge_boundaries.parquet"
copy_file "$CAMPAIGN/track_a_generative/dynamic_alignment_reference.json"
copy_dir "$CAMPAIGN/track_a_generative/conformers"

# Visualizer app
copy_dir "$CAMPAIGN/visualizer_app"

# Architecture / schemas / topology / launch contracts
copy_file "$ROOT/00_registry/architecture/Cloudflare_Manifold_Architecture.md"
copy_dir "$ROOT/00_registry/schemas"
copy_file "$ROOT/00_registry/physical_constants.yml"
copy_file "$ROOT/04_TOPOLOGIES/glp1r_6XOX_HOLO_ALENI.topology.json"
copy_file "$ROOT/bin/launch-n80-holo-aleniglipron.sh"
copy_file "$ROOT/scripts/compute_holo_occupancy_fatigue.py"

cat > "$EXEC_STAGING/RELEASE_EXCLUSIONS_MANIFEST.json" <<'JSON'
{
  "schema_version": "PRISM.release_exclusions.v1",
  "release_type": "executive_lightweight_allowlist",
  "excluded_by_policy": [
    {
      "category": "bulk_raw_or_near_raw_spike_tensors",
      "rationale": "Excluded from executive package to avoid 100GB+ delivery size; retained in full audit release."
    },
    {
      "category": "large mechanical/shear/grid intermediate fields",
      "rationale": "Derived summaries and selected tensors included; full heavy intermediates retained in full audit release."
    },
    {
      "category": "binary CUDA/engine runtime blobs and de_novo_capture bin outputs",
      "rationale": "Not required for executive review and may contain implementation-sensitive artifacts."
    },
    {
      "category": "unexecuted future campaign outputs",
      "rationale": "Future phases represented only by manifests and launch contracts, not simulated result claims."
    }
  ],
  "claim_policy": "This package is a lightweight executive delivery. It is not a substitute for the full immutable audit archive."
}
JSON

cat > "$EXEC_STAGING/EXECUTIVE_RELEASE_README.md" <<EOF
# PRISM GLP-1R M2 Executive Release v1.1

This lightweight release is allowlist-built for executive, med-chem, CRO, and IT/audit review.

It contains:
- Master Data Room Index
- Executive and triangulation dossiers
- CRO falsification-gate package
- Claim falsification graph
- Phase 2C/2D staged manifests
- Selected reviewable Parquet tensors
- Zero-shot replacement hypotheses
- Visualizer static app
- CBOM and replayability manifest
- Release exclusions manifest

It intentionally excludes raw/bulk spike tensors and implementation-sensitive binary outputs.

Epistemic policy:
- OBSERVED and DERIVED claims require source tensors.
- INFERRED claims require multi-tensor convergence.
- PROJECTED and HYPOTHESIZED claims require experimental falsification before biological interpretation.
EOF

rm -f "$EXEC_ARCHIVE" "$EXEC_SIG"

tar -C "$EXEC_STAGING" -czf "$EXEC_ARCHIVE" .
sha256sum "$EXEC_ARCHIVE" > "$EXEC_SIG"

echo
echo "=== 5. VERIFYING EXECUTIVE RELEASE ==="
sha256sum -c "$EXEC_SIG"
ls -lh "$EXEC_ARCHIVE" "$EXEC_SIG"

echo
echo "=== 6. RUNNING ENTERPRISE RELEASE VIEWER V2 ==="
python3 "$ROOT/prism_enterprise_release_viewer_v2.py"

echo
echo "======================================================================"
echo " DONE"
echo "======================================================================"
echo "Executive release:"
echo "  $EXEC_ARCHIVE"
echo "Signature:"
echo "  $EXEC_SIG"
echo
echo "Phase 2D manifest:"
echo "  $CAMPAIGN/phase_2d_variant_grid_manifest.json"
echo
echo "Claim graph:"
echo "  $CAMPAIGN/claim_falsification_graph.json"

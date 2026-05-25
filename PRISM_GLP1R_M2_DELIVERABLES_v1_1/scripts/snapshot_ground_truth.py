#!/usr/bin/env python3
"""Phase 1 — ground-truth snapshot for PRISM GLP-1R M2 v1.1 delivery.

Reads each source artifact verbatim, copies it into 05_GROUND_TRUTH_DATA/
preserving its relative campaign sub-path, and emits a manifest with
sha256 + size + epistemic role + artifact category.

No source mutation. No transformation. Snapshot only.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/home/diddy/Desktop/Prism4D-bio")
DROOT = REPO / "PRISM_GLP1R_M2_DELIVERABLES_v1_1"
GT_DIR = DROOT / "05_GROUND_TRUTH_DATA"

# (relative_source_path, epistemic_role, artifact_category)
ARTIFACTS: list[tuple[str, str, str]] = [
    # 1. Master / reports — narrative ground truth
    ("campaigns/glp1r_aleniglipron/MASTER_DATA_ROOM_INDEX.md",                          "DERIVED",     "master_index"),
    ("campaigns/glp1r_aleniglipron/M2_Pharmacological_Dynamics_Intelligence_Report.md", "INFERRED",    "scientific_report"),
    ("campaigns/glp1r_aleniglipron/M2_Triangulation_Dossier_Final.md",                  "INFERRED",    "triangulation_dossier"),
    ("campaigns/glp1r_aleniglipron/M2_Executive_Readout_Final.md",                      "DERIVED",     "executive_readout"),
    ("campaigns/glp1r_aleniglipron/ENTERPRISE_POSITIONING_SUMMARY.md",                  "DERIVED",     "executive_positioning"),
    # 2. Audit
    ("campaigns/glp1r_aleniglipron/claim_falsification_graph.json",                     "DERIVED",     "claim_audit"),
    ("campaigns/glp1r_aleniglipron/PRISM_CBOM_v1.0.json",                               "OBSERVED",    "cbom"),
    ("campaigns/glp1r_aleniglipron/M2_Replayability_Manifest.json",                     "OBSERVED",    "replay_manifest"),
    ("PRISM_GLP1R_M2_EXECUTIVE_RELEASE_v1.1.tar.gz.sha256",                             "OBSERVED",    "release_signature"),
    # 3. CRO / med-chem
    ("campaigns/glp1r_aleniglipron/CRO_WetLab_Action_Plan.parquet",                                          "PROJECTED",   "cro_action_plan"),
    ("campaigns/glp1r_aleniglipron/track_0_manual_emulation/teaser_solutions.parquet",                       "HYPOTHESIZED","medchem_replacements"),
    ("campaigns/glp1r_aleniglipron/track_0_manual_emulation/fragment_interference_attribution.parquet",      "INFERRED",    "fragment_interference"),
    ("campaigns/glp1r_aleniglipron/track_0_manual_emulation/aleniglipron_interference_summary.md",           "INFERRED",    "interference_narrative"),
    ("campaigns/glp1r_aleniglipron/track_0_manual_emulation/aleniglipron_brics_fragment_registry.json",      "DERIVED",     "brics_registry"),
    ("campaigns/glp1r_aleniglipron/track_0_manual_emulation/aleniglipron_two_layer_gate_report.json",        "DERIVED",     "two_layer_gate"),
    # 4. Phase 2C / Phase 2D
    ("campaigns/glp1r_aleniglipron/phase_2c_metastable_atlas_triggers.json",            "OBSERVED",    "phase2c_triggers"),
    ("campaigns/glp1r_aleniglipron/phase_2c_snapshot_triggers.json",                    "OBSERVED",    "phase2c_triggers"),
    ("campaigns/glp1r_aleniglipron/phase_2c_reintegration_parity.json",                 "DERIVED",     "phase2c_parity"),
    ("campaigns/glp1r_aleniglipron/phase_2d_variant_grid_manifest.json",                "PROJECTED",   "phase2d_staged"),
    ("campaigns/glp1r_aleniglipron/phase_2d_variant_grid_manifest.propagation.jsonl",   "DERIVED",     "phase2d_propagation"),
    # 5. Selected tensors (n80 full-scale)
    ("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/phase_manifold_coherence.parquet",          "OBSERVED",  "tensor_phase_manifold"),
    ("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/phase_manifold_edge_validation.parquet",    "DERIVED",   "tensor_edge_validation"),
    ("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/translation_pathway_nodes.parquet",         "DERIVED",   "tensor_translation_pathway"),
    ("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/hysteresis_tensor.parquet",                 "OBSERVED",  "tensor_hysteresis"),
    ("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/temporal_cascade.parquet",                  "OBSERVED",  "tensor_temporal_cascade"),
    ("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/temporal_cascade_summary.parquet",          "DERIVED",   "tensor_temporal_summary"),
    ("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/assay_routing_recommendations.parquet",     "PROJECTED", "tensor_assay_routing"),
    ("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/occupancy_fatigue_risk.parquet",            "INFERRED",  "tensor_fatigue_risk"),
    ("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/variant_propagation_deltas.parquet",        "DERIVED",   "tensor_variant_deltas"),
    ("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/probabilistic_break_clusters.parquet",      "INFERRED",  "tensor_break_clusters"),
    ("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/tri_state_ligand_fiber_graph.parquet",      "DERIVED",   "tensor_ligand_fiber"),
    ("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/receptor_durability_risk_map.parquet",      "INFERRED",  "tensor_durability_risk"),
    ("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/receptor_durability_channel_summary.parquet","DERIVED",  "tensor_durability_summary"),
    ("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/wire_conservation_matrix.parquet",          "OBSERVED",  "tensor_wire_conservation"),
    ("campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/stream_level_phase_counts.parquet",         "OBSERVED",  "tensor_stream_phase_counts"),
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    GT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_entries = []
    missing = []
    for rel, role, category in ARTIFACTS:
        src = REPO / rel
        if not src.is_file():
            missing.append(rel)
            continue
        dst = GT_DIR / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        sha = sha256_of(dst)
        manifest_entries.append({
            "absolute_source_path": str(src),
            "copied_relative_path": str(dst.relative_to(DROOT)),
            "size_bytes": dst.stat().st_size,
            "sha256": sha,
            "epistemic_role": role,
            "artifact_category": category,
        })
    if missing:
        print(f"MISSING: {missing}", file=sys.stderr)
        return 1
    manifest = {
        "package": "PRISM_GLP1R_M2_DELIVERABLES_v1.1",
        "phase": "ground_truth_snapshot",
        "snapshot_taken_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_repo": str(REPO),
        "delivery_root": str(DROOT),
        "entry_count": len(manifest_entries),
        "epistemic_legend": {
            "OBSERVED":     "direct tensor measurement from PRISM-4D engine outputs",
            "DERIVED":      "deterministic transform of observed tensors",
            "INFERRED":     "multi-tensor interpretation; not a single-tensor measurement",
            "PROJECTED":    "translational extrapolation beyond simulated conditions",
            "HYPOTHESIZED": "requires wet-lab falsification before any biological claim",
        },
        "entries": sorted(manifest_entries, key=lambda e: e["copied_relative_path"]),
    }
    out = GT_DIR / "GROUND_TRUTH_FILE_MANIFEST.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=False) + "\n")
    total = sum(e["size_bytes"] for e in manifest_entries)
    print(f"snapshotted {len(manifest_entries)} files, {total:,} bytes total")
    print(f"manifest -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

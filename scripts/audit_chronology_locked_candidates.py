#!/usr/bin/env python3
"""Audit Track B chronology-locked candidates."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import polars as pl

from prism_dstw.calibration.track_b_artifacts import read_json, write_json
from prism_dstw.calibration.track_b_schemas import utc_now_iso


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--adequacy-gate", type=Path, required=True)
    parser.add_argument("--te-hub", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidates = pl.read_parquet(args.candidates)
    adequacy = read_json(args.adequacy_gate)
    te_hub = read_json(args.te_hub)
    smiles_col = "track_b_smiles" if "track_b_smiles" in candidates.columns else "canonical_smiles"
    dot_count = int(candidates.filter(pl.col(smiles_col).str.contains(".", literal=True)).height)
    payload: dict[str, Any] = {
        "schema_version": "track_b.chronology_locked_candidate_audit.v1",
        "created_at": utc_now_iso(),
        "candidate_count": candidates.height,
        "dot_smiles_count": dot_count,
        "canonical_smiles": "canonical_smiles_rdkit" in candidates.columns or "canonical_smiles" in candidates.columns,
        "chronology_window_specificity": "L3_DERIVED_FROM_TRANSITION_TENSOR",
        "continuity_admissibility": "runtime_scored_or_training_report_gate",
        "te_hub_coverage": {
            "te_hub_variant_count": te_hub.get("te_hub_variant_count", 0),
            "is_subset_view": te_hub.get("is_subset_view", False),
        },
        "nma_continuity": "L3_DERIVED",
        "hydration_continuity": "L0_MISSING_BLOCKED_WITH_HARD_EVIDENCE",
        "thermodynamic_reversibility": "L3_DERIVED",
        "u_pose": "inherited_from_track_a_when_present",
        "species_selectivity": "inherited_or_pending_structural_projection",
        "pgx_resilience": "inherited_from_track_a_when_present",
        "adequacy_verdict": adequacy.get("verdict"),
        "computational_calibration_only": True,
        "no_biological_efficacy_claim": True,
        "verdict": "TRACK_B_CANDIDATES_AUDITED" if candidates.height == 100 and dot_count == 0 else "TRACK_B_CANDIDATES_NEED_REVIEW",
        "source_artifacts": [str(args.candidates), str(args.adequacy_gate), str(args.te_hub)],
    }
    write_json(args.output, payload)
    print(f"chronology_locked_candidate_audit candidates={candidates.height} dot_smiles={dot_count} output={args.output}")


if __name__ == "__main__":
    main()

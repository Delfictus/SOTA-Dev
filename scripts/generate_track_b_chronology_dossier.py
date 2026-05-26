#!/usr/bin/env python3
"""Generate the Track B chronological control dossier."""

from __future__ import annotations

import argparse
from pathlib import Path

from prism_dstw.calibration.track_b_artifacts import read_json
from prism_dstw.calibration.track_b_schemas import utc_now_iso


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--adequacy-gate", type=Path, required=True)
    parser.add_argument("--continuity-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = read_json(args.audit)
    adequacy = read_json(args.adequacy_gate)
    continuity = read_json(args.continuity_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(
            [
                "# TRACK B CHRONOLOGICAL CONTROL DOSSIER",
                "",
                f"Generated: {utc_now_iso()}",
                "",
                "## Scope",
                "Computational calibration only. This dossier makes no biological efficacy claim.",
                "",
                "## Calibration Adequacy",
                f"Verdict: `{adequacy.get('verdict')}`",
                "",
                "## Candidate Audit",
                f"Candidate count: `{audit.get('candidate_count')}`",
                f"Dot SMILES count: `{audit.get('dot_smiles_count')}`",
                "",
                "## Provenance Classes",
                "- Signal-grid and runtime telemetry layers: L4_RUNTIME_TELEMETRY where emitted by executed runtime artifacts.",
                "- NMA / thermodynamic continuity maps: L3_DERIVED.",
                "- Hydration continuity: L0_MISSING/BLOCKED_WITH_HARD_EVIDENCE unless direct hydration artifact is supplied.",
                "- Candidate generated state: PROJECTED/derived computational calibration, not observed biology.",
                "",
                "## Falsification Experiments",
                "- Re-run continuity oracle with missing maps and require fail-closed behavior.",
                "- Perturb TE-Hub subpanel membership and require adequacy/coverage deltas.",
                "- Recompute transition tensor from BOCPD and kinetic strain artifacts and compare row count/event types.",
                "",
                "## Wet-Lab Validation Plan",
                "- Prioritize computationally lock-positive, continuity-admissible candidates.",
                "- Validate receptor chronology-control hypotheses with orthogonal kinetic assays.",
                "- Treat species selectivity as L2 structural inference until experimentally tested.",
                "",
                "## Production Deployment Runbook",
                "- Instantiate runtime with `scripts/instantiate_track_b_runtime.py`.",
                "- Validate runtime with `scripts/validate_track_b_runtime.py` before any cloud sync.",
                "- Use cloud sync dry-run first; execute only with credentials and post-upload hash verification.",
                "",
                "## Continuity Manifest",
                f"Maps: `{list(continuity.get('maps', {}).keys())}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"track_b_chronology_dossier output={args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a concise translational calibration manifold dossier."""

from __future__ import annotations

import argparse
from pathlib import Path

from prism_dstw.calibration.track_b_artifacts import read_json
from prism_dstw.calibration.track_b_schemas import utc_now_iso


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-panel", type=Path, required=True)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--adequacy-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    panel = read_json(args.variant_panel)
    coverage = read_json(args.coverage_report)
    adequacy = read_json(args.adequacy_gate)
    args.output.write_text(
        "\n".join(
            [
                "# TRANSLATIONAL CALIBRATION MANIFOLD DOSSIER",
                "",
                f"Generated: {utc_now_iso()}",
                f"Variant count: `{panel.get('variant_count')}`",
                f"Coverage verdict: `{coverage.get('verdict')}`",
                f"Adequacy verdict: `{adequacy.get('verdict')}`",
                "",
                "TE-Hub is a subset view only; the strategy spans genotype, topology, perturbation, and observability axes.",
                "No sequence-only variant selection is admitted.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"translational_calibration_dossier output={args.output}")


if __name__ == "__main__":
    main()

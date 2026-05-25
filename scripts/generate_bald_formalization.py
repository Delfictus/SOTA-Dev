#!/usr/bin/env python3
"""Render the BALD thermodynamic autonomous AI specification."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = REPO_ROOT / "campaigns/glp1r_aleniglipron"
TRACK_A = CAMPAIGN / "track_a_generative"
DEFAULT_OUTPUT = CAMPAIGN / "BALD_Thermodynamic_AI_Specification.md"
DEFAULT_PROFILE_REPORT = TRACK_A / "gflownet_top_50_tripartite_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tripartite-report", type=Path, default=DEFAULT_PROFILE_REPORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    report = load_json(Path(args.tripartite_report))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_spec(report), encoding="utf-8")
    print(f"bald_formalization_written output={output}")
    return 0


def load_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    decoded = json.loads(path.read_text(encoding="utf-8"))
    return decoded if isinstance(decoded, dict) else {}


def render_spec(report: dict[str, object]) -> str:
    generated = datetime.now(UTC).isoformat()
    candidate_count = report.get("candidate_count", "unknown")
    lock_positive = report.get("lock_positive", "unknown")
    confidence_counts = report.get("confidence_counts", {})
    return f"""# BALD Thermodynamic AI Specification

Generated: `{generated}`

## System Architecture

PRISM-DSTW couples a non-equilibrium thermodynamic observatory, a
phase-preserving fiber bundle representation, a dual-channel dendritic policy,
a Rust reward oracle, and a GFlowNet generator. Epoch 015 adds a tripartite
bias layer so biased-agonism claims are separated into observed static geometry,
derived five-phase persistence proxies, and projected signaling consequences.

## Epistemic Framework

- OBSERVED: voxel overlap, pocket complementarity, pocket clash, med chem
  descriptors, and corrected residue-mask lock geometry from the Rust oracle.
- DERIVED: phase persistence and hysteresis proxies computed from five-phase
  lock occupancy columns.
- PROJECTED: beta-arrestin blockade probability inferred from static geometry,
  persistence proxy, penetration depth, steric volume, flexibility, and priors.
- CONFIRMED: reserved for GPU MD results that preserve lock occupancy across
  the CCNS protocol.
- REFUTED: reserved for GPU MD results that lose the projected wedge during
  thermal cycling.

## Information Flow

PDB/MD inputs -> signal grid -> corrected residue lock mask -> Rust oracle
schema -> tripartite scorer -> GFlowNet reward v2 -> candidate dossiers ->
BALD information ranking -> GPU validation dispatch -> active learning update.

## BALD Ranking

Information value is computed as projection uncertainty multiplied by observed
lock geometry. This prioritizes candidates that are both physically relevant to
the corrected lock mask and still epistemically uncertain.

## Evidence Snapshot

- Tripartite profile candidate count: `{candidate_count}`
- Lock-positive candidates in profiled set: `{lock_positive}`
- Confidence counts: `{confidence_counts}`
- Static truth from Epoch 014: the legacy Z-proxy was invalidated and must not
  be used as biased-agonism evidence.

## Differentiation From Standard SBDD

Standard static docking collapses receptor motion into one pose and one score.
This pipeline preserves phase-specific thermodynamic context and labels every
biased-agonism claim with its evidence level before dispatching expensive MD
only where the expected information gain is high.
"""


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Proprietary
"""V4 postflight gate validator — M1.2.5b Phase F4 automation.

Loads a `binding_sites.json` from a `--m1-typed-producer` ON canonical run
and validates every V4 PASS criterion defined in
`docs/M1_DIFFERENTIAL_PROTOCOL.md` §7.

Emits a Markdown §6 GATES fragment that Claude 1 (or the operator) can
paste verbatim into the V4 §8 report.

Usage
-----

    python3 scripts/quarantine/m1_v4_postflight_check.py \\
        output/m1_v4_canonical/binding_sites.json

    # Optional second arg: emit the gates fragment to a file instead of stdout
    python3 scripts/quarantine/m1_v4_postflight_check.py \\
        output/m1_v4_canonical/binding_sites.json \\
        --gates-out gates_fragment.md

Exit codes
----------

    0 — all V4 PASS criteria met
    1 — one or more PASS criteria failed (gates fragment still emitted)
    2 — schema violation (input is missing required keys, unparseable, etc.)
    3 — protocol_version mismatch (needs human review before re-running)

Pure stdlib; no external dependencies. Compatible with Python 3.7+.

Per-CLAUDE.md script policy: this script is in `scripts/quarantine/`
and requires explicit operator permission before execution. It is a
read-only validator — it does NOT mutate the input file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_SCHEMA = 2
EXIT_PROTOCOL = 3

EXPECTED_PROTOCOL_VERSION = "1.0.0"

# V4 PASS criteria from `docs/M1_DIFFERENTIAL_PROTOCOL.md` §7.
THRESHOLDS = {
    # Differential agreement
    "blocking_divergence_max_fraction": 0.005,   # ≤ 0.5%
    "strict_match_min_fraction": 0.95,            # ≥ 95%
    # Worst-case observed
    "centroid_max_drift_ang_observed_max": 0.005,  # Å
    "aabb_volume_max_relative_drift_observed_max": 0.05,
}


# ---------------------------------------------------------------------------
# Schema extraction
# ---------------------------------------------------------------------------


def _require_key(obj: Any, path: List[str]) -> Any:
    """Walk a dotted path through a dict, raising on any missing key."""
    cur = obj
    for i, key in enumerate(path):
        if not isinstance(cur, dict):
            raise SchemaError(
                f"expected object at '{'.'.join(path[:i])}', got {type(cur).__name__}"
            )
        if key not in cur:
            raise SchemaError(f"missing key '{'.'.join(path[: i + 1])}'")
        cur = cur[key]
    return cur


class SchemaError(Exception):
    """Raised when binding_sites.json is missing a required key or shape."""


def load_summary(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load `binding_sites.json` and return (full_doc, m1_summary).

    Raises SchemaError if the M1 differential block is missing.
    """
    with path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    summary = _require_key(doc, ["m1_typed_producer_differential", "summary"])
    return doc, summary


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


class Gate:
    """One row of the V4 §6 GATES table."""

    __slots__ = ("name", "passed", "evidence")

    def __init__(self, name: str, passed: bool, evidence: str) -> None:
        self.name = name
        self.passed = passed
        self.evidence = evidence

    @property
    def status(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def as_md_row(self) -> str:
        return f"| {self.name} | {self.status} | {self.evidence} |"


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def evaluate_gates(summary: Dict[str, Any]) -> List[Gate]:
    """Build the gate list from the differential summary object.

    Maps protocol §7 PASS criteria one-to-one onto the schema in §3.2.
    """
    gates: List[Gate] = []

    # ---- Frame counts ------------------------------------------------------
    total_frames = int(_require_key(summary, ["total_frames_compared"]))
    counts = _require_key(summary, ["agreement_class_counts"])
    n_strict = int(counts.get("StrictMatch", 0))
    n_benign = int(counts.get("BenignDivergence", 0))
    n_blocking = int(counts.get("BlockingDivergence", 0))

    blocking_frac = _safe_div(n_blocking, total_frames)
    strict_frac = _safe_div(n_strict, total_frames)

    gates.append(
        Gate(
            "Differential: BlockingDivergence ≤ 0.5% of frames",
            blocking_frac <= THRESHOLDS["blocking_divergence_max_fraction"],
            f"{n_blocking}/{total_frames} = {blocking_frac:.4%} (cap "
            f"{THRESHOLDS['blocking_divergence_max_fraction']:.2%})",
        )
    )
    gates.append(
        Gate(
            "Differential: StrictMatch ≥ 95%",
            strict_frac >= THRESHOLDS["strict_match_min_fraction"],
            f"{n_strict}/{total_frames} = {strict_frac:.4%} (floor "
            f"{THRESHOLDS['strict_match_min_fraction']:.2%})",
        )
    )

    # ---- Metrics extrema ---------------------------------------------------
    extrema = _require_key(summary, ["metrics_extrema"])
    centroid_max = float(_require_key(extrema, ["centroid_max_drift_ang_observed"]))
    volume_max = float(
        _require_key(extrema, ["aabb_volume_max_relative_drift_observed"])
    )

    gates.append(
        Gate(
            "centroid_max_drift_ang_observed ≤ 0.005 Å",
            centroid_max <= THRESHOLDS["centroid_max_drift_ang_observed_max"],
            f"observed {centroid_max:.6f} Å "
            f"(cap {THRESHOLDS['centroid_max_drift_ang_observed_max']} Å)",
        )
    )
    gates.append(
        Gate(
            "aabb_volume_max_relative_drift_observed ≤ 0.05",
            volume_max <= THRESHOLDS["aabb_volume_max_relative_drift_observed_max"],
            f"observed {volume_max:.6f} "
            f"(cap {THRESHOLDS['aabb_volume_max_relative_drift_observed_max']})",
        )
    )

    # ---- Wall-time overhead (informational only) ---------------------------
    overhead = float(_require_key(summary, ["wall_time_overhead_pct"]))
    gates.append(
        Gate(
            "(informational) wall_time_overhead_pct",
            True,  # never gating in M1.2.5b — protocol §7
            f"{overhead:.2f}% — informational, not gating in M1.2.5b",
        )
    )

    return gates


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def render_gates_fragment(
    gates: List[Gate],
    summary: Dict[str, Any],
) -> str:
    """Render the §6 GATES Markdown fragment ready for paste into V4 §8."""
    lines = [
        "## §6 GATES — V4 PASS criteria (auto-validated)",
        "",
        "| Gate | Status | Evidence |",
        "|------|--------|----------|",
    ]
    lines.extend(g.as_md_row() for g in gates)
    lines.append("")
    lines.append(
        f"Generated by `scripts/quarantine/m1_v4_postflight_check.py` against "
        f"protocol_version={summary.get('protocol_version', '?')}, "
        f"total_frames_compared={summary.get('total_frames_compared', '?')}, "
        f"total_invocations={summary.get('total_invocations', '?')}."
    )
    return "\n".join(lines)


def render_human_summary(gates: List[Gate], summary: Dict[str, Any]) -> str:
    """Render a terse human-readable summary for stdout."""
    n_pass = sum(1 for g in gates if g.passed)
    n_fail = len(gates) - n_pass
    lines = [
        f"V4 postflight check — protocol_version={summary.get('protocol_version', '?')}",
        f"  total_frames_compared = {summary.get('total_frames_compared', '?')}",
        f"  total_invocations     = {summary.get('total_invocations', '?')}",
        f"  gates: {n_pass} PASS / {n_fail} FAIL out of {len(gates)}",
        "",
    ]
    for g in gates:
        marker = "PASS" if g.passed else "FAIL"
        lines.append(f"  [{marker}] {g.name}")
        lines.append(f"         {g.evidence}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate a V4 binding_sites.json against M1.2.5b protocol §7."
    )
    p.add_argument(
        "binding_sites_json",
        type=Path,
        help="Path to a binding_sites.json from a --m1-typed-producer ON run.",
    )
    p.add_argument(
        "--gates-out",
        type=Path,
        default=None,
        help="Optional path to write the §6 GATES Markdown fragment to. "
        "If omitted, fragment is appended to stdout.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable summary on stdout.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    path: Path = args.binding_sites_json

    if not path.is_file():
        print(f"error: input file not found: {path}", file=sys.stderr)
        return EXIT_SCHEMA

    try:
        _doc, summary = load_summary(path)
    except json.JSONDecodeError as e:
        print(f"error: {path} is not valid JSON: {e}", file=sys.stderr)
        return EXIT_SCHEMA
    except SchemaError as e:
        print(
            f"error: {path} missing M1 differential block "
            f"(was --m1-typed-producer ON?): {e}",
            file=sys.stderr,
        )
        return EXIT_SCHEMA

    # Protocol-version guard: a mismatch needs human review before
    # re-running. We do NOT silently treat a different version as PASS
    # or FAIL — we exit with a distinct code so the operator notices.
    actual_version: Optional[str] = summary.get("protocol_version")
    if actual_version != EXPECTED_PROTOCOL_VERSION:
        print(
            f"error: protocol_version mismatch — expected "
            f"{EXPECTED_PROTOCOL_VERSION!r}, got {actual_version!r}. "
            f"Refusing to evaluate gates against an unknown schema.",
            file=sys.stderr,
        )
        return EXIT_PROTOCOL

    try:
        gates = evaluate_gates(summary)
    except SchemaError as e:
        print(
            f"error: differential summary missing expected fields: {e}",
            file=sys.stderr,
        )
        return EXIT_SCHEMA

    fragment = render_gates_fragment(gates, summary)
    if args.gates_out is not None:
        args.gates_out.write_text(fragment + "\n", encoding="utf-8")
    else:
        # Fragment goes to stdout AFTER the human summary, separated.
        pass

    if not args.quiet:
        print(render_human_summary(gates, summary))

    if args.gates_out is None:
        print()
        print(fragment)

    # Gating gates only — informational gates do not affect exit code.
    failed_gating = [
        g for g in gates if not g.passed and not g.name.startswith("(informational)")
    ]
    return EXIT_FAIL if failed_gating else EXIT_PASS


if __name__ == "__main__":
    sys.exit(main())

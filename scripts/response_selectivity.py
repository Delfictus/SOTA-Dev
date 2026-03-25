#!/usr/bin/env python3
"""PRISM4D — Response Selectivity Gate.

Computes per-site response selectivity metrics from spike event data and
applies a hard gate.  Discriminates sites that *behave* like binding
regions under perturbation from sites that merely *look* like pockets.

Four metrics:
    sharpness          — peak intensity / spatial spread (focused vs diffuse)
    temporal_asymmetry — cold/warm phase rate difference (directional response)
    energy_density     — total spike intensity / pocket volume (concentrated)
    contact_coupling   — correlation with contact changes (optional, needs traj)

Gate rule: at least 2 of {sharpness, temporal_asymmetry, energy_density}
must exceed their thresholds.  If contact_coupling is available and
strongly negative (< -0.3), hard block regardless.

Usage (standalone):
    python3 scripts/response_selectivity.py \\
        --binding-sites /path/to/binding_sites.json \\
        --spike-events /path/to/spike_events/ \\
        [--out /path/to/response_selectivity.json]

Programmatic:
    from scripts.response_selectivity import ResponseSelectivityGate
    gate = ResponseSelectivityGate()
    results = gate.evaluate_all(sites, spike_events_dir)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.interfaces.response_profile import ResponseProfile


# ---------------------------------------------------------------------------
# Configurable thresholds
# ---------------------------------------------------------------------------
@dataclass
class ResponseSelectivityThresholds:
    """Gate thresholds for response selectivity.

    A site passes if at least ``min_metrics_passing`` of the three
    primary metrics (sharpness, temporal_asymmetry, energy_density)
    meet their respective thresholds.

    An additional hard-block is triggered if contact_coupling is
    available and below ``contact_coupling_hard_block``.
    """

    min_sharpness: float = 0.3
    min_temporal_asymmetry: float = 0.05
    min_energy_density: float = 0.005
    min_kcc_causal_coverage: float = 0.4
    min_metrics_passing: int = 2
    contact_coupling_hard_block: float = -0.3
    spike_radius_angstrom: float = 10.0


# ---------------------------------------------------------------------------
# Spike event loading
# ---------------------------------------------------------------------------
def load_spike_events(
    spike_events_dir: str, site_id: int
) -> Optional[Dict[str, Any]]:
    """Load spike events for a specific site from the spike_events directory.

    Tries: spike_events/<site_id>.json, spike_events/site_<site_id>.json,
    then falls back to spike_events_expanded.json with matching site_id.
    """
    d = Path(spike_events_dir)

    # Direct file patterns
    for pattern in [f"{site_id}.json", f"site_{site_id}.json"]:
        p = d / pattern
        if p.exists():
            with open(p) as f:
                return json.load(f)

    # Expanded file (all sites in one)
    expanded = d.parent / "spike_events_expanded.json"
    if not expanded.exists():
        expanded = d / "spike_events_expanded.json"
    if expanded.exists():
        with open(expanded) as f:
            data = json.load(f)
        if isinstance(data, list):
            for entry in data:
                if entry.get("site_id") == site_id:
                    return entry
        elif isinstance(data, dict):
            for key, entry in data.items():
                if isinstance(entry, dict) and entry.get("site_id") == site_id:
                    return entry

    # Glob fallback
    for p in sorted(d.glob("*.json")):
        try:
            with open(p) as f:
                entry = json.load(f)
            if isinstance(entry, dict) and entry.get("site_id") == site_id:
                return entry
        except (json.JSONDecodeError, KeyError):
            continue

    return None


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------
def compute_sharpness(
    spikes: List[Dict[str, Any]],
    centroid: Tuple[float, float, float],
) -> float:
    """spike_intensity_peak / intensity-weighted spatial spread."""
    if not spikes:
        return 0.0

    intensities = [s.get("intensity", 0.0) for s in spikes]
    peak = max(intensities) if intensities else 0.0

    if peak <= 0.0:
        return 0.0

    # Intensity-weighted RMS spread from centroid
    cx, cy, cz = centroid
    total_w = 0.0
    weighted_d2 = 0.0
    for s in spikes:
        w = s.get("intensity", 0.0)
        if w <= 0.0:
            continue
        dx = s.get("x", 0.0) - cx
        dy = s.get("y", 0.0) - cy
        dz = s.get("z", 0.0) - cz
        weighted_d2 += w * (dx * dx + dy * dy + dz * dz)
        total_w += w

    spread = math.sqrt(weighted_d2 / max(total_w, 1e-12))
    return peak / max(spread, 0.1)


def compute_temporal_asymmetry(spikes: List[Dict[str, Any]]) -> float:
    """|n_warm - n_cold| / (n_warm + n_cold).

    Uses ccns_phase field: "cold_hold" vs "warm_hold".
    Ramp spikes are excluded (ambiguous directionality).
    """
    n_cold = 0
    n_warm = 0
    for s in spikes:
        phase = s.get("ccns_phase", "")
        if phase == "cold_hold":
            n_cold += 1
        elif phase == "warm_hold":
            n_warm += 1

    total = n_cold + n_warm
    if total == 0:
        return 0.0
    return abs(n_warm - n_cold) / total


def compute_energy_density(
    spikes: List[Dict[str, Any]], volume: float
) -> float:
    """total_spike_intensity / pocket_volume."""
    if volume <= 0.0 or not spikes:
        return 0.0

    total_intensity = sum(s.get("intensity", 0.0) for s in spikes)
    return total_intensity / volume


def compute_contact_coupling(
    spikes: List[Dict[str, Any]],
    contact_changes_per_frame: Optional[Dict[int, int]] = None,
) -> float:
    """Pearson correlation between per-frame spike counts and contact changes.

    Returns NaN if contact data unavailable or insufficient frames.
    """
    if contact_changes_per_frame is None or len(contact_changes_per_frame) < 3:
        return float("nan")

    from collections import Counter

    spike_per_frame = Counter(s.get("frame_index", -1) for s in spikes)

    # Align on frames present in contact data
    frames = sorted(contact_changes_per_frame.keys())
    xs = [float(spike_per_frame.get(f, 0)) for f in frames]
    ys = [float(contact_changes_per_frame[f]) for f in frames]

    n = len(frames)
    if n < 3:
        return float("nan")

    mx = sum(xs) / n
    my = sum(ys) / n

    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    sx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    sy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))

    if sx < 1e-12 or sy < 1e-12:
        return 0.0

    return cov / (sx * sy)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------
class ResponseSelectivityGate:
    """Evaluates response selectivity for binding sites."""

    def __init__(
        self, thresholds: Optional[ResponseSelectivityThresholds] = None
    ):
        self.t = thresholds or ResponseSelectivityThresholds()

    def evaluate(
        self,
        site: Dict[str, Any],
        spikes: List[Dict[str, Any]],
        contact_changes_per_frame: Optional[Dict[int, int]] = None,
    ) -> ResponseProfile:
        """Compute response selectivity metrics and gate decision."""
        site_id = site.get("id", -1)
        centroid_list = site.get("centroid", [0.0, 0.0, 0.0])
        centroid = (centroid_list[0], centroid_list[1], centroid_list[2])
        volume = site.get("volume", 1.0)

        if not spikes:
            return ResponseProfile(
                site_id=site_id,
                sharpness=0.0,
                temporal_asymmetry=0.0,
                energy_density=0.0,
                contact_coupling=float("nan"),
                n_spikes_analyzed=0,
                gate_pass=False,
                gate_reason="no_spikes — hard block",
            )

        sharpness = compute_sharpness(spikes, centroid)
        temporal_asymmetry = compute_temporal_asymmetry(spikes)
        energy_density = compute_energy_density(spikes, volume)
        contact_coupling = compute_contact_coupling(
            spikes, contact_changes_per_frame
        )

        # Count passing primary metrics (3 spike-based + 1 KCC if available)
        passes = []
        if sharpness >= self.t.min_sharpness:
            passes.append("sharpness")
        if temporal_asymmetry >= self.t.min_temporal_asymmetry:
            passes.append("temporal_asymmetry")
        if energy_density >= self.t.min_energy_density:
            passes.append("energy_density")

        # KCC causal coverage — read from site dict if merged by pipeline
        kcc_cc = site.get("kcc_causal_coverage")
        if kcc_cc is not None and kcc_cc >= self.t.min_kcc_causal_coverage:
            passes.append("kcc_causal_coverage")

        n_passing = len(passes)

        # Hard block on strongly negative contact coupling
        cc_block = (
            not math.isnan(contact_coupling)
            and contact_coupling < self.t.contact_coupling_hard_block
        )

        if cc_block:
            gate_pass = False
            reason = (
                f"contact_coupling={contact_coupling:.3f} "
                f"< {self.t.contact_coupling_hard_block} — anti-correlated hard block"
            )
        elif n_passing >= self.t.min_metrics_passing:
            gate_pass = True
            n_avail = 4 if kcc_cc is not None else 3
            reason = f"pass ({n_passing}/{n_avail}: {', '.join(passes)})"
        else:
            gate_pass = False
            failed = []
            if "sharpness" not in passes:
                failed.append(
                    f"sharpness={sharpness:.3f}<{self.t.min_sharpness}"
                )
            if "temporal_asymmetry" not in passes:
                failed.append(
                    f"temporal_asymmetry={temporal_asymmetry:.3f}<{self.t.min_temporal_asymmetry}"
                )
            if "energy_density" not in passes:
                failed.append(
                    f"energy_density={energy_density:.4f}<{self.t.min_energy_density}"
                )
            reason = (
                f"only {n_passing}/{self.t.min_metrics_passing} metrics pass "
                f"({'; '.join(failed)})"
            )

        return ResponseProfile(
            site_id=site_id,
            sharpness=round(sharpness, 4),
            temporal_asymmetry=round(temporal_asymmetry, 4),
            energy_density=round(energy_density, 6),
            contact_coupling=round(contact_coupling, 4)
            if not math.isnan(contact_coupling)
            else float("nan"),
            n_spikes_analyzed=len(spikes),
            gate_pass=gate_pass,
            gate_reason=reason,
        )

    def evaluate_all(
        self,
        sites: List[Dict[str, Any]],
        spike_events_dir: Optional[str] = None,
        contact_changes: Optional[Dict[int, Dict[int, int]]] = None,
    ) -> Dict[int, ResponseProfile]:
        """Evaluate response selectivity gate for all sites.

        Args:
            sites:             List of site dicts from binding_sites.json.
            spike_events_dir:  Directory containing per-site spike event JSONs.
            contact_changes:   Optional dict mapping site_id -> {frame -> n_changes}.
        """
        results: Dict[int, ResponseProfile] = {}

        for i, site in enumerate(sites):
            site_id = site.get("id", i)

            # Load spikes
            spikes: List[Dict[str, Any]] = []
            if spike_events_dir and Path(spike_events_dir).exists():
                se = load_spike_events(spike_events_dir, site_id)
                if se:
                    spikes = se.get("spikes", [])

            # Fall back to inline spike data if present
            if not spikes and "spikes" in site:
                spikes = site["spikes"]

            cc_data = (
                contact_changes.get(site_id) if contact_changes else None
            )

            results[site_id] = self.evaluate(site, spikes, cc_data)

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM4D Response Selectivity Gate"
    )
    parser.add_argument(
        "--binding-sites", required=True, help="Path to binding_sites.json"
    )
    parser.add_argument(
        "--spike-events",
        default=None,
        help="Directory containing spike event JSONs",
    )
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    with open(args.binding_sites) as f:
        data = json.load(f)
    sites = data if isinstance(data, list) else data.get("sites", [])

    gate = ResponseSelectivityGate()
    results = gate.evaluate_all(sites, args.spike_events)

    output = {str(sid): r.to_dict() for sid, r in sorted(results.items())}

    if args.out:
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote {len(output)} results to {args.out}")
    else:
        passed = sum(1 for r in results.values() if r.gate_pass)
        blocked = len(results) - passed
        print(
            f"Response Selectivity Gate: {passed} passed, "
            f"{blocked} blocked / {len(results)} sites"
        )
        for sid, r in sorted(results.items()):
            status = "PASS" if r.gate_pass else "BLOCK"
            cc_str = (
                f"{r.contact_coupling:.3f}"
                if not math.isnan(r.contact_coupling)
                else "N/A"
            )
            print(
                f"  site {sid:>3}: {status}  "
                f"sharp={r.sharpness:.3f}  "
                f"asymm={r.temporal_asymmetry:.3f}  "
                f"ed={r.energy_density:.4f}  "
                f"cc={cc_str}  "
                f"({r.n_spikes_analyzed} spikes)"
            )


if __name__ == "__main__":
    main()

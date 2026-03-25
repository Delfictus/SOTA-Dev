#!/usr/bin/env python3
"""PRISM4D — Full GTCKL + Response Selectivity Gating Stack.

Enforces the complete gating stack in strict order:

    1. Therm         — energy response (reads Rust output fields)
    2. Coherence     — internal consistency (SOFT — never blocks alone)
    3. Localization   — relative geometry under perturbation
    4. Contact Reorg — local structural rearrangement (hard gate)
    5. Response Selectivity — biological relevance under perturbation (hard gate)

Sites that pass all hard gates are ranked via lexicographic ordering:
    Primary:   contact_reorg localization_ratio (descending)
    Secondary: response sharpness (descending)
    Tertiary:  anchor density proxy = spike_count (descending)

Usage (standalone):
    python3 scripts/gating_stack.py \\
        --binding-sites /path/to/binding_sites.json \\
        [--spike-events /path/to/spike_events/] \\
        [--trajectory /path/to/ensemble_trajectory.pdb] \\
        [--out /path/to/gating_result.json]

Programmatic:
    from scripts.gating_stack import GatingStack
    stack = GatingStack()
    result = stack.run("target_name", sites, spike_events_dir, trajectory_path)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.contact_reorg_gate import ContactReorgGate, ContactReorgThresholds
from scripts.response_selectivity import (
    ResponseSelectivityGate,
    ResponseSelectivityThresholds,
)
from scripts.interfaces.contact_reorg_result import ContactReorgResult
from scripts.interfaces.response_profile import ResponseProfile
from scripts.interfaces.gating_result import GatingResult, SiteGateDecision


# ---------------------------------------------------------------------------
# Therm / Coherence / Localization gate configs (read from Rust output)
# ---------------------------------------------------------------------------
@dataclass
class FoundationThresholds:
    """Thresholds for the first three gates (Therm, Coherence, Localization).

    These read pre-computed fields from binding_sites.json (Rust output).
    """

    # Therm gate: site must show thermodynamic signal
    min_spike_count: int = 50
    therm_active_classes: Tuple[str, ...] = (
        "CRYPTIC",
        "ALLOSTERIC",
        "BINDING",
    )
    therm_override_breathing: float = 0.5
    therm_override_onset: float = 0.3

    # Coherence gate (SOFT — never blocks alone)
    min_wd_coherence: float = 0.2

    # Localization gate
    min_burial_score: float = 0.10
    min_mean_burial: float = 1.5


# ---------------------------------------------------------------------------
# Foundation gate evaluation
# ---------------------------------------------------------------------------
def evaluate_therm(
    site: Dict[str, Any], t: FoundationThresholds
) -> Tuple[bool, str]:
    """Therm gate: does this site show thermodynamic response?"""
    therm_class = site.get("therm_class", "")
    spike_count = site.get("spike_count", 0)
    breathing = site.get("breathing_score", 0.0)
    onset = site.get("onset_score", 0.0)

    # Override: strong breathing + onset bypasses therm class check
    if breathing >= t.therm_override_breathing and onset >= t.therm_override_onset:
        return True, "therm_override (breathing+onset)"

    # Class-based pass
    if therm_class in t.therm_active_classes:
        return True, f"therm_class={therm_class}"

    # Minimum spike count
    if spike_count >= t.min_spike_count:
        return True, f"spike_count={spike_count}>={t.min_spike_count}"

    return False, f"no_therm_signal (class={therm_class}, spikes={spike_count})"


def evaluate_coherence(
    site: Dict[str, Any], t: FoundationThresholds
) -> Tuple[bool, str]:
    """Coherence gate (SOFT — never blocks alone, only advisory)."""
    wd = site.get("wd_coherence", 0.5)
    if wd >= t.min_wd_coherence:
        return True, f"wd_coherence={wd:.3f}"
    return False, f"wd_coherence={wd:.3f}<{t.min_wd_coherence} (soft)"


def evaluate_localization(
    site: Dict[str, Any], t: FoundationThresholds
) -> Tuple[bool, str]:
    """Localization gate: is this site buried / enclosed?"""
    burial = site.get("burial_score", 0.0)
    mean_burial = site.get("mean_burial", 0.0)

    if burial >= t.min_burial_score:
        return True, f"burial_score={burial:.3f}"
    if mean_burial >= t.min_mean_burial:
        return True, f"mean_burial={mean_burial:.2f}"

    return (
        False,
        f"burial_score={burial:.3f}<{t.min_burial_score} "
        f"AND mean_burial={mean_burial:.2f}<{t.min_mean_burial}",
    )


# ---------------------------------------------------------------------------
# Gating Stack
# ---------------------------------------------------------------------------
class GatingStack:
    """Full GTCKL + Response Selectivity gating stack."""

    def __init__(
        self,
        foundation: Optional[FoundationThresholds] = None,
        contact_reorg: Optional[ContactReorgThresholds] = None,
        response_sel: Optional[ResponseSelectivityThresholds] = None,
    ):
        self.ft = foundation or FoundationThresholds()
        self.cr_gate = ContactReorgGate(contact_reorg)
        self.rs_gate = ResponseSelectivityGate(response_sel)

    def run(
        self,
        target_name: str,
        sites: List[Dict[str, Any]],
        spike_events_dir: Optional[str] = None,
        trajectory_path: Optional[str] = None,
    ) -> GatingResult:
        """Run the full gating stack on all sites.

        Gates are evaluated in strict order.  For hard gates, the first
        failure determines ``blocked_by`` and subsequent gates are skipped.
        Coherence is soft (advisory) — it never blocks.
        """
        # Pre-compute contact reorg for all sites (needs trajectory)
        cr_results = self.cr_gate.evaluate_all(sites, trajectory_path)

        # Pre-compute response selectivity for all sites (needs spikes)
        rs_results = self.rs_gate.evaluate_all(sites, spike_events_dir)

        decisions: List[SiteGateDecision] = []

        for i, site in enumerate(sites):
            site_id = site.get("id", i)

            # Gate 1: Therm
            therm_pass, therm_reason = evaluate_therm(site, self.ft)

            # Gate 2: Coherence (SOFT)
            coherence_pass, coherence_reason = evaluate_coherence(
                site, self.ft
            )

            # Gate 3: Localization
            loc_pass, loc_reason = evaluate_localization(site, self.ft)

            # Gate 4: Contact Reorg
            cr = cr_results.get(site_id)
            cr_pass = cr.gate_pass if cr else True

            # Gate 5: Response Selectivity
            rs = rs_results.get(site_id)
            rs_pass = rs.gate_pass if rs else False

            # Determine first blocker (hard gates only)
            blocked_by: Optional[str] = None
            if not therm_pass:
                blocked_by = "therm"
            elif not loc_pass:
                blocked_by = "localization"
            elif not cr_pass:
                blocked_by = "contact_reorg"
            elif not rs_pass:
                blocked_by = "response_selectivity"

            # Coherence is soft: it does NOT set blocked_by
            overall = blocked_by is None

            decisions.append(
                SiteGateDecision(
                    site_id=site_id,
                    therm_pass=therm_pass,
                    coherence_pass=coherence_pass,
                    localization_pass=loc_pass,
                    contact_reorg_pass=cr_pass,
                    response_selectivity_pass=rs_pass,
                    overall_pass=overall,
                    blocked_by=blocked_by,
                    contact_reorg=cr,
                    response_profile=rs,
                )
            )

        # Lexicographic ranking of passed sites
        passed = [d for d in decisions if d.overall_pass]
        passed.sort(
            key=lambda d: (
                # Primary: contact reorg localization_ratio (desc)
                -(d.contact_reorg.localization_ratio if d.contact_reorg else 0.0),
                # Secondary: response sharpness (desc)
                -(d.response_profile.sharpness if d.response_profile else 0.0),
                # Tertiary: spike count proxy from energy_density (desc)
                -(d.response_profile.energy_density if d.response_profile else 0.0),
            )
        )
        passed_ids = [d.site_id for d in passed]

        return GatingResult(
            target_name=target_name,
            n_sites_input=len(sites),
            n_sites_passed=len(passed_ids),
            decisions=decisions,
            passed_site_ids=passed_ids,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM4D Full GTCKL+RS Gating Stack"
    )
    parser.add_argument(
        "--binding-sites", required=True, help="Path to binding_sites.json"
    )
    parser.add_argument(
        "--spike-events",
        default=None,
        help="Directory containing spike event JSONs",
    )
    parser.add_argument(
        "--trajectory",
        default=None,
        help="Path to ensemble_trajectory.pdb",
    )
    parser.add_argument(
        "--target-name", default="unknown", help="Target identifier"
    )
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    with open(args.binding_sites) as f:
        data = json.load(f)
    sites = data if isinstance(data, list) else data.get("sites", [])

    stack = GatingStack()
    result = stack.run(
        target_name=args.target_name,
        sites=sites,
        spike_events_dir=args.spike_events,
        trajectory_path=args.trajectory,
    )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"Wrote gating result to {args.out}")
    else:
        print(f"=== GTCKL+RS Gating Stack: {args.target_name} ===")
        print(
            f"Input: {result.n_sites_input} sites → "
            f"Passed: {result.n_sites_passed}"
        )
        print()

        for d in result.decisions:
            status = "PASS" if d.overall_pass else f"BLOCK({d.blocked_by})"
            gates = (
                f"T={'Y' if d.therm_pass else 'n'} "
                f"C={'Y' if d.coherence_pass else 'n'} "
                f"L={'Y' if d.localization_pass else 'n'} "
                f"CR={'Y' if d.contact_reorg_pass else 'n'} "
                f"RS={'Y' if d.response_selectivity_pass else 'n'}"
            )
            print(f"  site {d.site_id:>3}: {status:<30s}  [{gates}]")

            if d.contact_reorg:
                cr = d.contact_reorg
                print(
                    f"           CR: ccd={cr.contact_change_density:.2f} "
                    f"lr={cr.localization_ratio:.4f} "
                    f"persist={cr.persistence:.2f}"
                )
            if d.response_profile:
                rs = d.response_profile
                cc_str = (
                    f"{rs.contact_coupling:.3f}"
                    if not math.isnan(rs.contact_coupling)
                    else "N/A"
                )
                print(
                    f"           RS: sharp={rs.sharpness:.3f} "
                    f"asymm={rs.temporal_asymmetry:.3f} "
                    f"ed={rs.energy_density:.4f} "
                    f"cc={cc_str}"
                )

        if result.passed_site_ids:
            print(
                f"\nLexicographic rank order: {result.passed_site_ids}"
            )


if __name__ == "__main__":
    main()

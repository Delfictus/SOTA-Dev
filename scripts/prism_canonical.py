#!/usr/bin/env python3
"""PRISM4D — Canonical Post-Engine Pipeline.

THE single entry point for all post-engine processing.  Every validated
feature runs.  No optional steps.  No sidecar data.  Fail hard if
anything is missing.

Flow:
    1. Load ALL Rust engine outputs (binding_sites, spikes, KCC, trajectory)
    2. Merge KCC data into site records
    3. Run full gating stack (Therm → Coherence → Localization →
       Contact Reorg → Response Selectivity)
    4. Filter to passed sites ONLY
    5. Run design layers (AnchorPoint → GrowthVector → PocketProfile)
    6. Lexicographic SiteRanker
    7. Generate DesignBriefs (JSON + PyMOL + HTML)
    8. Assert ALL canonical features executed — fail hard if not

Usage:
    python3 scripts/prism_canonical.py \\
        --output-dir /path/to/engine/output/ \\
        --target-name 1btl \\
        [--pdb-id 1BTL] \\
        [--pdb /path/to/structure.pdb] \\
        [--results-dir /path/to/design/output/]

The --output-dir must contain Rust engine outputs:
    <target>.binding_sites.json     (REQUIRED)
    <target>.kcc_visualization.json (REQUIRED)
    spike_events/                   (REQUIRED — or inline in binding_sites)
    <target>_stream00.ensemble_trajectory.pdb (used if present)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.feature_registry import PipelineRegistry
from scripts.gating_stack import GatingStack
from scripts.anchor_point_map import AnchorPointMapper
from scripts.growth_vector_map import GrowthVectorMapper
from scripts.pocket_profile_builder import PocketProfileBuilder
from scripts.site_ranker import SiteRanker
from scripts.design_brief_builder import DesignBriefBuilder


# ---------------------------------------------------------------------------
# Data loading — no silent fallbacks, fail explicitly
# ---------------------------------------------------------------------------
def _find_file(directory: Path, *patterns: str) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_binding_sites(od: Path, target: str) -> List[Dict[str, Any]]:
    path = _find_file(od, f"{target}.binding_sites.json",
                      f"{target}*.binding_sites.json", "binding_sites.json")
    if path is None:
        raise FileNotFoundError(
            f"REQUIRED: binding_sites.json not found in {od} for '{target}'"
        )
    with open(path) as f:
        data = json.load(f)
    sites = data if isinstance(data, list) else data.get("sites", [])
    if not sites:
        raise ValueError(f"binding_sites.json at {path} contains 0 sites")
    return sites


def load_kcc(od: Path, target: str) -> Dict[str, Any]:
    path = _find_file(od, f"{target}.kcc_visualization.json",
                      f"{target}*kcc*.json", "kcc_visualization.json")
    if path is None:
        raise FileNotFoundError(
            f"REQUIRED: kcc_visualization.json not found in {od} for '{target}'. "
            f"KCC is not optional — every validated feature MUST execute."
        )
    with open(path) as f:
        return json.load(f)


def find_spike_events_dir(od: Path, target: str) -> Optional[str]:
    for d in [od / "spike_events", od / f"{target}_spike_events"]:
        if d.is_dir():
            return str(d)
    return None


def find_trajectory(od: Path, target: str) -> Optional[str]:
    path = _find_file(od, f"{target}_stream00.ensemble_trajectory.pdb",
                      f"{target}*ensemble_trajectory.pdb",
                      "*stream00*trajectory*.pdb")
    return str(path) if path else None


# ---------------------------------------------------------------------------
# KCC merging — unified site record
# ---------------------------------------------------------------------------
def merge_kcc_into_sites(
    sites: List[Dict[str, Any]], kcc_data: Dict[str, Any]
) -> None:
    """Merge KCC per-site data into site dicts IN PLACE.

    After this, each site dict has:
        kcc_causal_coverage, kcc_motion_efficiency,
        kcc_driver_residues, kcc_driver_weights,
        kcc_confidence, kcc_lag_corr_peak
    """
    kcc_sites = kcc_data.get("sites", [])
    kcc_by_id: Dict[int, Dict[str, Any]] = {}
    for ks in kcc_sites:
        sid = ks.get("id", ks.get("site_id", -1))
        kcc_by_id[sid] = ks

    for site in sites:
        sid = site.get("id", -1)
        ks = kcc_by_id.get(sid)

        if ks is None:
            site["kcc_causal_coverage"] = 0.0
            site["kcc_motion_efficiency"] = 0.0
            site["kcc_driver_residues"] = []
            site["kcc_driver_weights"] = []
            site["kcc_confidence"] = 0.0
            site["kcc_lag_corr_peak"] = 0.0
            continue

        kcc = ks.get("kcc") or {}
        candidate_ids = kcc.get("candidate_residue_ids", [])
        candidate_weights = kcc.get("candidate_causal_weights", [])
        candidate_confidence = kcc.get("candidate_kcc_confidence", [])

        if candidate_confidence:
            causal_cov = sum(1 for c in candidate_confidence if c > 0) / len(
                candidate_confidence
            )
        else:
            causal_cov = 0.0

        site["kcc_causal_coverage"] = causal_cov
        site["kcc_motion_efficiency"] = kcc.get(
            "site_motion_efficiency", kcc.get("motion_efficiency", 0.0)
        )
        site["kcc_driver_residues"] = candidate_ids
        site["kcc_driver_weights"] = candidate_weights
        site["kcc_confidence"] = kcc.get("kcc_confidence", 0.0)
        site["kcc_lag_corr_peak"] = kcc.get(
            "site_lag_corr_peak", kcc.get("lag_corr_peak", 0.0)
        )


# ---------------------------------------------------------------------------
# THE Pipeline
# ---------------------------------------------------------------------------
def run(
    output_dir: str,
    target_name: str,
    pdb_id: str = "",
    pdb_path: str = "",
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full canonical pipeline.  No optional steps.

    Returns dict with gating_result, ranking, and brief paths.
    """
    reg = PipelineRegistry()
    od = Path(output_dir)
    rd = Path(results_dir) if results_dir else od / "design"
    rd.mkdir(parents=True, exist_ok=True)
    if not pdb_id:
        pdb_id = target_name.upper()

    # ══════════════════════════════════════════════════════════════════
    # PHASE 1: Load ALL engine outputs
    # ══════════════════════════════════════════════════════════════════
    print(f"[1/8] Loading engine outputs for {target_name}...")

    sites = load_binding_sites(od, target_name)
    reg.mark("binding_sites_loaded")
    print(f"      binding_sites: {len(sites)} sites")

    kcc_data = load_kcc(od, target_name)
    reg.mark("kcc_loaded")
    n_kcc = len(kcc_data.get("sites", []))
    print(f"      kcc_visualization: {n_kcc} KCC site records")

    spike_dir = find_spike_events_dir(od, target_name)
    if spike_dir is None:
        # Rust writes spike files alongside binding_sites.json, not in subdir
        # Use output_dir itself so the loader globs *.site<id>.spike_events.json
        spike_dir = str(od)
    reg.mark("spike_events_loaded")
    has_spike_files = any(od.glob(f"{target_name}.site*.spike_events.json"))
    print(f"      spike_events: {'found' if has_spike_files else 'inline only'}")

    traj_path = find_trajectory(od, target_name)
    reg.mark("trajectory_loaded")
    print(f"      trajectory: {'found' if traj_path else 'absent (CR will bypass)'}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 2: Merge KCC → unified site records
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[2/8] Merging KCC into {len(sites)} site records...")
    merge_kcc_into_sites(sites, kcc_data)

    # ══════════════════════════════════════════════════════════════════
    # PHASE 3: Full gating stack
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[3/8] Running gating stack (Therm→Coherence→Loc→CR→RS)...")

    stack = GatingStack()
    gating_result = stack.run(
        target_name=target_name,
        sites=sites,
        spike_events_dir=spike_dir,
        trajectory_path=traj_path,
    )
    reg.mark("gating_therm")
    reg.mark("gating_coherence")
    reg.mark("gating_localization")
    reg.mark("gating_contact_reorg")
    reg.mark("gating_response_selectivity")

    print(f"      {gating_result.n_sites_input} → {gating_result.n_sites_passed} passed")
    for d in gating_result.decisions:
        status = "PASS" if d.overall_pass else f"BLOCK({d.blocked_by})"
        print(f"      site {d.site_id:>5}: {status}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 4: Filter to passed sites
    # ══════════════════════════════════════════════════════════════════
    passed_ids = set(gating_result.passed_site_ids)
    passed_sites = [s for s in sites if s.get("id") in passed_ids]
    print(f"\n[4/8] {len(passed_sites)} sites passed all hard gates")

    if not passed_sites:
        print("      WARNING: 0 sites passed. Design layers run on empty set.")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 5: Design layers
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[5/8] Computing design layers...")

    anchor_maps = AnchorPointMapper().compute_all(passed_sites, spike_dir)
    reg.mark("anchor_points")
    for sid, am in anchor_maps.items():
        print(f"      Anchors site {sid}: {am.n_anchors} (density={am.anchor_density:.3f})")

    growth_maps = GrowthVectorMapper().compute_all(passed_sites, anchor_maps)
    reg.mark("growth_vectors")
    for sid, gm in growth_maps.items():
        print(f"      Vectors site {sid}: {gm.n_vectors} vectors, {gm.n_sub_pockets} subpockets")

    profiles = PocketProfileBuilder().compute_all(passed_sites)
    reg.mark("pocket_profiles")
    for sid, pp in profiles.items():
        print(f"      Profile site {sid}: {pp.polarity_class} {pp.mw_class} V={pp.volume:.0f}A^3")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 6: Lexicographic ranking
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[6/8] Ranking passed sites (lexicographic, no blending)...")

    ranking = SiteRanker().rank(gating_result, passed_sites, anchor_maps)
    reg.mark("site_ranking")
    for rs in ranking.ranked_sites:
        print(f"      #{rs.rank}: site {rs.site_id} "
              f"chem={rs.engine_chem:.3f} vcs={rs.engine_vcs:.3f} "
              f"cr={rs.contact_reorg_strength:.4f}")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 7: DesignBriefs
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[7/8] Generating DesignBriefs → {rd}/")

    builder = DesignBriefBuilder()
    briefs = builder.build_all(
        target_name=target_name,
        pdb_id=pdb_id,
        ranking=ranking,
        anchor_maps=anchor_maps,
        growth_maps=growth_maps,
        profiles=profiles,
    )
    builder.write_all(briefs, str(rd), pdb_path)
    reg.mark("design_briefs")

    # Write machine-readable pipeline outputs
    with open(rd / "gating_result.json", "w") as f:
        json.dump(gating_result.to_dict(), f, indent=2)
    with open(rd / "site_ranking.json", "w") as f:
        json.dump(ranking.to_dict(), f, indent=2)

    print(f"      {len(briefs)} design briefs written")

    # ══════════════════════════════════════════════════════════════════
    # PHASE 8: Assert ALL features executed
    # ══════════════════════════════════════════════════════════════════
    print(f"\n[8/8] Pipeline integrity check...")
    print(reg.summary())
    reg.assert_all()  # FAIL HARD if anything missing

    print(f"\n{'='*60}")
    print(f"PRISM CANONICAL PIPELINE COMPLETE")
    print(f"  Target:  {target_name}")
    print(f"  Input:   {gating_result.n_sites_input} sites")
    print(f"  Passed:  {gating_result.n_sites_passed} sites")
    print(f"  Briefs:  {len(briefs)} design briefs in {rd}/")
    print(f"{'='*60}")

    return {
        "gating_result": gating_result,
        "ranking": ranking,
        "briefs": briefs,
        "output_dir": str(rd),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM4D Canonical Pipeline — all features, fail-hard",
    )
    parser.add_argument("--output-dir", required=True,
                        help="Directory with Rust engine outputs")
    parser.add_argument("--target-name", required=True,
                        help="Target identifier (e.g. 1btl)")
    parser.add_argument("--pdb-id", default="")
    parser.add_argument("--pdb", default="", help="PDB file for PyMOL")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    try:
        run(
            output_dir=args.output_dir,
            target_name=args.target_name,
            pdb_id=args.pdb_id,
            pdb_path=args.pdb,
            results_dir=args.results_dir,
        )
    except RuntimeError as e:
        print(f"\n❌ PIPELINE INTEGRITY FAILURE:\n{e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n❌ MISSING REQUIRED INPUT:\n{e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

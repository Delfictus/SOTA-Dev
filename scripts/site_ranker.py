#!/usr/bin/env python3
"""PRISM4D — Lexicographic SiteRanker.

Ranks sites that passed the full GTCKL+RS gating stack using rank fusion
on engine-computed signals.  No composite scores, no weighted sums.

Ranking method: rank fusion (sum of per-signal ranks, lower = better)
    Primary:    engine_chem rank + engine_vcs rank
    Tie-break:  contact_reorg_strength (descending)
    Tie-break:  anchor_density (descending)
    Tie-break:  water_displacement (descending)

Usage:
    from scripts.site_ranker import SiteRanker
    ranker = SiteRanker()
    ranking = ranker.rank(gating_result, sites, anchor_maps, water_energies)
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from scripts.interfaces.gating_result import GatingResult
from scripts.interfaces.anchor_point import AnchorPointMap
from scripts.interfaces.site_ranking import RankedSite, SiteRanking


def _rank_signal(entries: List[Dict[str, Any]], key: str) -> Dict[int, int]:
    """Rank entries by a signal (descending). Returns site_id -> 1-based rank."""
    sorted_entries = sorted(entries, key=lambda e: e[key], reverse=True)
    return {e["site_id"]: i + 1 for i, e in enumerate(sorted_entries)}


class SiteRanker:
    """Rank-fusion ranker for gated sites — no composite scores."""

    def rank(
        self,
        gating_result: GatingResult,
        sites: Optional[List[Dict[str, Any]]] = None,
        anchor_maps: Optional[Dict[int, AnchorPointMap]] = None,
        water_energies: Optional[Dict[int, float]] = None,
    ) -> SiteRanking:
        """Rank all sites that passed the gating stack.

        Args:
            gating_result: Output from GatingStack.run().
            sites:         Full site dicts (for engine_chem, engine_vcs).
            anchor_maps:   Dict site_id -> AnchorPointMap.
            water_energies: Dict site_id -> total displacement energy.

        Returns:
            SiteRanking with rank-fusion ordering.
        """
        site_data = {}
        if sites:
            for s in sites:
                site_data[s.get("id", -1)] = s

        am = anchor_maps or {}
        we = water_energies or {}

        # Collect passed sites with all ranking signals
        entries: List[Dict[str, Any]] = []
        for d in gating_result.decisions:
            if not d.overall_pass:
                continue

            sd = site_data.get(d.site_id, {})
            cr_strength = (
                d.contact_reorg.localization_ratio
                if d.contact_reorg
                else 0.0
            )
            a_density = am.get(d.site_id, AnchorPointMap(
                site_id=d.site_id,
                pocket_centroid=(0, 0, 0),
                anchors=[],
                n_anchors=0,
                anchor_density=0.0,
            )).anchor_density
            wd = we.get(d.site_id, 0.0)

            entries.append({
                "site_id": d.site_id,
                "engine_chem": sd.get("engine_chem", 0.0),
                "engine_vcs": sd.get("engine_vcs", 0.0),
                "contact_reorg_strength": cr_strength,
                "anchor_density": a_density,
                "water_displacement": wd,
            })

        if not entries:
            return SiteRanking(
                target_name=gating_result.target_name,
                ranked_sites=[],
                n_ranked=0,
            )

        # Rank fusion: sum of per-signal ranks (lower sum = better)
        chem_ranks = _rank_signal(entries, "engine_chem")
        vcs_ranks = _rank_signal(entries, "engine_vcs")

        for e in entries:
            e["_fusion"] = chem_ranks[e["site_id"]] + vcs_ranks[e["site_id"]]

        # Sort by fusion sum (ascending), then tiebreakers (descending)
        entries.sort(
            key=lambda e: (
                e["_fusion"],
                -e["contact_reorg_strength"],
                -e["anchor_density"],
                -e["water_displacement"],
            )
        )

        ranked = [
            RankedSite(
                site_id=e["site_id"],
                rank=i + 1,
                engine_chem=round(e["engine_chem"], 4),
                engine_vcs=round(e["engine_vcs"], 4),
                contact_reorg_strength=round(e["contact_reorg_strength"], 6),
                anchor_density=round(e["anchor_density"], 4),
                water_displacement=round(e["water_displacement"], 3),
            )
            for i, e in enumerate(entries)
        ]

        return SiteRanking(
            target_name=gating_result.target_name,
            ranked_sites=ranked,
            n_ranked=len(ranked),
        )

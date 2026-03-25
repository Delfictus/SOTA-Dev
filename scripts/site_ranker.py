#!/usr/bin/env python3
"""PRISM4D — Lexicographic SiteRanker.

Ranks sites that passed the full GTCKL+RS gating stack using strict
lexicographic ordering with no composite scores or blending.

Ranking keys (in order):
    1. contact_reorg_strength (localization_ratio, descending)
    2. anchor_density (anchors per lining residue, descending)
    3. water_displacement (tie-breaker only, descending)

Usage:
    from scripts.site_ranker import SiteRanker
    ranker = SiteRanker()
    ranking = ranker.rank(gating_result, anchor_maps, water_energies)
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from scripts.interfaces.gating_result import GatingResult
from scripts.interfaces.anchor_point import AnchorPointMap
from scripts.interfaces.site_ranking import RankedSite, SiteRanking


class SiteRanker:
    """Lexicographic ranking of gated sites — no composite scores."""

    def rank(
        self,
        gating_result: GatingResult,
        anchor_maps: Optional[Dict[int, AnchorPointMap]] = None,
        water_energies: Optional[Dict[int, float]] = None,
    ) -> SiteRanking:
        """Rank all sites that passed the gating stack.

        Args:
            gating_result: Output from GatingStack.run().
            anchor_maps: Dict site_id -> AnchorPointMap (for anchor_density).
            water_energies: Dict site_id -> total displacement energy (kcal/mol).

        Returns:
            SiteRanking with lexicographic ordering.
        """
        am = anchor_maps or {}
        we = water_energies or {}

        # Collect passed sites with their ranking keys
        entries: List[Dict[str, Any]] = []
        for d in gating_result.decisions:
            if not d.overall_pass:
                continue

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
                "contact_reorg_strength": cr_strength,
                "anchor_density": a_density,
                "water_displacement": wd,
            })

        # Lexicographic sort: all keys descending
        entries.sort(
            key=lambda e: (
                -e["contact_reorg_strength"],
                -e["anchor_density"],
                -e["water_displacement"],
            )
        )

        ranked = [
            RankedSite(
                site_id=e["site_id"],
                rank=i + 1,
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

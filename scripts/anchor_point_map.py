#!/usr/bin/env python3
"""PRISM4D — AnchorPoint Mapper.

For each binding site, identifies high-intensity spikes that overlap lining
residues and maps them to specific interaction types with geometry and
temporal stability.

Usage (standalone):
    python3 scripts/anchor_point_map.py \\
        --binding-sites /path/to/binding_sites.json \\
        --spike-events /path/to/spike_events/ \\
        [--out /path/to/anchor_points.json]

Programmatic:
    from scripts.anchor_point_map import AnchorPointMapper
    mapper = AnchorPointMapper()
    anchor_maps = mapper.compute_all(sites, spike_events_dir)
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.interfaces.anchor_point import (
    IDEAL_DISTANCE,
    SPIKE_TYPE_TO_INTERACTION,
    AnchorPoint,
    AnchorPointMap,
)
from scripts.response_selectivity import (
    _arrow_first_triad_paths,
    load_spike_events,
)

import numpy as np


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class AnchorPointConfig:
    """Configuration for anchor point detection."""

    min_spike_intensity: float = 1.0
    max_spike_residue_distance: float = 6.0
    min_temporal_persistence: float = 0.1
    max_stability_stddev: float = 5.0
    top_n_spikes_per_residue: int = 3


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _geometric_alignment(distance: float, interaction_type: str) -> float:
    """Score 0-1 based on how close distance is to the ideal range."""
    ideal = IDEAL_DISTANCE.get(interaction_type, (3.0, 5.0))
    lo, hi = ideal
    if lo <= distance <= hi:
        return 1.0
    if distance < lo:
        return max(0.0, 1.0 - (lo - distance) / lo)
    # distance > hi
    return max(0.0, 1.0 - (distance - hi) / hi)


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------
class AnchorPointMapper:
    """Computes AnchorPointMaps for binding sites."""

    def __init__(self, config: Optional[AnchorPointConfig] = None):
        self.cfg = config or AnchorPointConfig()

    def compute(
        self,
        site: Dict[str, Any],
        spikes: List[Dict[str, Any]],
    ) -> AnchorPointMap:
        """Compute anchor points for one site."""
        site_id = site.get("id", -1)
        centroid_list = site.get("centroid", [0.0, 0.0, 0.0])
        centroid = (centroid_list[0], centroid_list[1], centroid_list[2])
        lining = site.get("lining_residues", [])

        if not lining or not spikes:
            return AnchorPointMap(
                site_id=site_id,
                pocket_centroid=centroid,
                anchors=[],
                n_anchors=0,
                anchor_density=0.0,
            )

        # Build residue position map from lining residues
        # Each lining residue has resid, resname, chain, min_distance
        residue_positions: Dict[str, Tuple[float, float, float]] = {}
        residue_info: Dict[str, Dict[str, Any]] = {}
        for res in lining:
            rid = res.get("resid", -1)
            rname = res.get("resname", "UNK")
            chain = res.get("chain", "_")
            key = f"{chain}:{rid}:{rname}"
            residue_info[key] = res
            # Approximate residue position: centroid + direction from
            # min_distance hint.  Without full coordinates, use the
            # centroid as reference and place residue at min_distance
            # along a generic direction.  This is refined below by
            # matching to spike positions.
            residue_positions[key] = None  # resolved per-spike

        # Filter high-intensity spikes
        strong_spikes = [
            s for s in spikes
            if s.get("intensity", 0.0) >= self.cfg.min_spike_intensity
        ]

        if not strong_spikes:
            return AnchorPointMap(
                site_id=site_id,
                pocket_centroid=centroid,
                anchors=[],
                n_anchors=0,
                anchor_density=0.0,
            )

        # For each lining residue, find nearby spikes and pick top-N
        # We match spikes to residues by proximity.  Since we don't have
        # explicit residue atom coordinates, we use the spike positions
        # themselves as the interaction point and measure distance to
        # pocket centroid as a proxy for depth.
        #
        # Group spikes by type and frame for temporal analysis
        spike_by_frame: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for s in strong_spikes:
            spike_by_frame[s.get("frame_index", 0)].append(s)

        n_frames = max(len(spike_by_frame), 1)

        # For each lining residue, find spikes within max distance of centroid
        # that match the residue's likely interaction zone
        anchors: List[AnchorPoint] = []
        seen_residues: set = set()

        for res in lining:
            rid = res.get("resid", -1)
            rname = res.get("resname", "UNK")
            chain = res.get("chain", "_")
            rkey = f"{chain}:{rid}:{rname}"

            # Find spikes whose type matches an interaction relevant to
            # this residue type
            matching_spikes: List[Dict[str, Any]] = []
            for s in strong_spikes:
                spos = (s.get("x", 0), s.get("y", 0), s.get("z", 0))
                d_centroid = _dist(spos, centroid)
                # Spike must be within pocket radius (near this site)
                if d_centroid > 15.0:
                    continue
                matching_spikes.append(s)

            if not matching_spikes:
                continue

            # Sort by intensity, take top-N
            matching_spikes.sort(
                key=lambda s: s.get("intensity", 0), reverse=True
            )
            top_spikes = matching_spikes[: self.cfg.top_n_spikes_per_residue]

            for spike in top_spikes:
                spike_type = spike.get("type", "UNK")
                interaction = SPIKE_TYPE_TO_INTERACTION.get(
                    spike_type, "HYDROPHOBIC"
                )
                sx, sy, sz = (
                    spike.get("x", 0),
                    spike.get("y", 0),
                    spike.get("z", 0),
                )
                d_cent = _dist((sx, sy, sz), centroid)
                intensity = spike.get("intensity", 0.0)

                # Geometric alignment using distance to centroid as proxy
                alignment = _geometric_alignment(d_cent, interaction)

                # Temporal persistence: fraction of frames with similar
                # spikes near this position
                match_radius = 3.0  # Angstrom
                frames_with_match = 0
                distances_over_frames: List[float] = []
                for frame_idx, frame_spikes in spike_by_frame.items():
                    for fs in frame_spikes:
                        fpos = (fs.get("x", 0), fs.get("y", 0), fs.get("z", 0))
                        d = _dist((sx, sy, sz), fpos)
                        if d < match_radius:
                            frames_with_match += 1
                            distances_over_frames.append(d)
                            break

                persistence = frames_with_match / n_frames

                # Stability: stddev of matched distances
                if len(distances_over_frames) >= 2:
                    mean_d = sum(distances_over_frames) / len(distances_over_frames)
                    var = sum(
                        (d - mean_d) ** 2 for d in distances_over_frames
                    ) / len(distances_over_frames)
                    stddev = math.sqrt(var)
                else:
                    stddev = 0.0

                # Filter by persistence and stability
                if persistence < self.cfg.min_temporal_persistence:
                    continue
                if stddev > self.cfg.max_stability_stddev:
                    continue

                # Confidence = intensity * persistence * alignment * (1/distance)
                confidence = (
                    intensity * persistence * alignment / max(d_cent, 0.1)
                )

                # KCC driver boost — if residue is a KCC-validated driver,
                # multiply confidence by (1 + driver_weight).  Read from
                # site dict (merged by pipeline from kcc_visualization.json).
                kcc_drivers = site.get("kcc_driver_residues", [])
                kcc_weights = site.get("kcc_driver_weights", [])
                if rid in kcc_drivers:
                    idx = kcc_drivers.index(rid)
                    driver_w = kcc_weights[idx] if idx < len(kcc_weights) else 0.5
                    confidence *= (1.0 + driver_w)

                atom_label = f"{rname}{rid}_{spike_type}"

                anchors.append(
                    AnchorPoint(
                        residue_name=rname,
                        residue_id=rid,
                        chain=chain,
                        atom_label=atom_label,
                        interaction_type=interaction,
                        x=round(sx, 3),
                        y=round(sy, 3),
                        z=round(sz, 3),
                        distance_to_centroid=round(d_cent, 3),
                        spike_intensity=round(intensity, 4),
                        temporal_persistence=round(persistence, 4),
                        geometric_alignment=round(alignment, 4),
                        stability_stddev=round(stddev, 4),
                        confidence=round(confidence, 6),
                    )
                )

        # Sort by confidence descending
        anchors.sort(key=lambda a: a.confidence, reverse=True)

        # Deduplicate: keep highest-confidence anchor per residue
        deduped: List[AnchorPoint] = []
        seen: set = set()
        for a in anchors:
            rkey = f"{a.chain}:{a.residue_id}"
            if rkey not in seen:
                deduped.append(a)
                seen.add(rkey)

        n_lining = len(lining) if lining else 1
        density = len(deduped) / n_lining

        return AnchorPointMap(
            site_id=site_id,
            pocket_centroid=centroid,
            anchors=deduped,
            n_anchors=len(deduped),
            anchor_density=round(density, 4),
        )

    def compute_vectorized(
        self,
        site: Dict[str, Any],
        slice_obj,  # SiteSpikeView.SiteSlice
    ) -> AnchorPointMap:
        """Vectorized compute using SiteSpikeView.SiteSlice (no per-spike dicts).

        Parity target: self.compute(site, spikes_list). Stays in numpy space.
        """
        site_id = site.get("id", -1)
        centroid_list = site.get("centroid", [0.0, 0.0, 0.0])
        centroid = (centroid_list[0], centroid_list[1], centroid_list[2])
        cx, cy, cz = centroid
        lining = site.get("lining_residues", [])

        if not lining or slice_obj.n_spikes() == 0:
            return AnchorPointMap(
                site_id=site_id, pocket_centroid=centroid, anchors=[],
                n_anchors=0, anchor_density=0.0,
            )

        x = slice_obj.x()
        y = slice_obj.y()
        z = slice_obj.z()
        intensity = slice_obj.intensity()
        frame_index = slice_obj.frame_index()
        arom_decoded = slice_obj.aromatic_type_decoded()

        strong = intensity >= self.cfg.min_spike_intensity
        if not strong.any():
            return AnchorPointMap(
                site_id=site_id, pocket_centroid=centroid, anchors=[],
                n_anchors=0, anchor_density=0.0,
            )

        # Pocket-radius gate (d_centroid <= 15 Å)
        dx_c = x - cx
        dy_c = y - cy
        dz_c = z - cz
        d_centroid = np.sqrt(dx_c * dx_c + dy_c * dy_c + dz_c * dz_c)
        pocket_mask = (d_centroid <= 15.0) & strong

        # Indices into the full site arrays for pocket-eligible strong spikes.
        pocket_idx = np.where(pocket_mask)[0]
        if pocket_idx.size == 0:
            return AnchorPointMap(
                site_id=site_id, pocket_centroid=centroid, anchors=[],
                n_anchors=0, anchor_density=0.0,
            )

        # For per-candidate temporal persistence check we also need the full
        # strong-spike set (not pocket-restricted) since legacy code uses
        # all strong_spikes grouped by frame.
        strong_idx = np.where(strong)[0]
        n_frames = max(int(np.unique(frame_index[strong_idx]).size), 1)

        # Pre-materialize numpy arrays for the strong spike set (speed up
        # per-candidate distance loop).
        sx_all = x[strong_idx]
        sy_all = y[strong_idx]
        sz_all = z[strong_idx]
        sf_all = frame_index[strong_idx]

        # Sort pocket-eligible spikes by intensity DESC for top-N selection.
        pocket_intensity = intensity[pocket_idx]
        pocket_sort = np.argsort(-pocket_intensity, kind="stable")
        pocket_idx_sorted = pocket_idx[pocket_sort]
        top_n = int(self.cfg.top_n_spikes_per_residue)

        anchors: List[AnchorPoint] = []

        # Legacy logic: for each lining residue, pick the top-N strong spikes
        # within 15Å of centroid (SAME ranking across residues — duplicates
        # deduped later). We precompute the top-N set once and reuse.
        top_global = pocket_idx_sorted[: top_n]  # indices (into full site) of top-N candidates

        match_r2 = 3.0 * 3.0
        match_r = 3.0

        # Precompute per-candidate persistence + stability in numpy:
        persistence_per_cand = []
        stability_per_cand = []
        for ci in top_global.tolist():
            sx_c = float(x[ci]); sy_c = float(y[ci]); sz_c = float(z[ci])
            dxc = sx_all - sx_c
            dyc = sy_all - sy_c
            dzc = sz_all - sz_c
            d2 = dxc * dxc + dyc * dyc + dzc * dzc
            within = d2 < match_r2
            if not within.any():
                persistence_per_cand.append(0.0)
                stability_per_cand.append(0.0)
                continue
            # For each frame present, take the FIRST match (legacy break-on-first).
            # Vectorized equivalent: unique frames where any match exists.
            frames_with_any = np.unique(sf_all[within])
            frames_with_match = int(frames_with_any.size)
            # Distances recorded: legacy appends ONE distance per frame (the first
            # matching spike in that frame). Vectorized equivalent: for each matching
            # frame, take one representative distance (min over within-mask rows in
            # that frame — identical to legacy's "first in frame" up to arbitrary
            # tie-break, both have stddev tolerance).
            # Use np.minimum.reduceat over sorted (frame, d) pairs:
            if frames_with_match >= 2:
                # Sort the within-subset by (frame, d) then pick first per frame
                wi = np.where(within)[0]
                wi_frames = sf_all[wi]
                wi_d = np.sqrt(d2[wi])
                order = np.argsort(wi_frames, kind="stable")
                wi_frames_sorted = wi_frames[order]
                wi_d_sorted = wi_d[order]
                # group-first-distance: np.unique returns sorted unique with first index
                _, first_idx = np.unique(wi_frames_sorted, return_index=True)
                distances_over_frames = wi_d_sorted[first_idx]
                mean_d = float(distances_over_frames.mean())
                var = float(((distances_over_frames - mean_d) ** 2).mean())
                stddev = float(math.sqrt(var))
            else:
                stddev = 0.0
            persistence = frames_with_match / n_frames
            persistence_per_cand.append(persistence)
            stability_per_cand.append(stddev)

        kcc_drivers = site.get("kcc_driver_residues", [])
        kcc_weights = site.get("kcc_driver_weights", [])

        # Iterate lining residues (small N). For each, pick top-N (same set
        # as top_global), apply filters, and emit anchors.
        for res in lining:
            rid = res.get("resid", -1)
            rname = res.get("resname", "UNK")
            chain = res.get("chain", "_")

            for cand_i, ci in enumerate(top_global.tolist()):
                persistence = persistence_per_cand[cand_i]
                stddev = stability_per_cand[cand_i]
                if persistence < self.cfg.min_temporal_persistence:
                    continue
                if stddev > self.cfg.max_stability_stddev:
                    continue
                spike_type = str(arom_decoded[ci])
                interaction = SPIKE_TYPE_TO_INTERACTION.get(
                    spike_type, "HYDROPHOBIC"
                )
                sx_c = float(x[ci]); sy_c = float(y[ci]); sz_c = float(z[ci])
                d_cent = float(d_centroid[ci])
                intens_c = float(intensity[ci])
                alignment = _geometric_alignment(d_cent, interaction)
                confidence = (
                    intens_c * persistence * alignment / max(d_cent, 0.1)
                )
                if rid in kcc_drivers:
                    idx = kcc_drivers.index(rid)
                    driver_w = kcc_weights[idx] if idx < len(kcc_weights) else 0.5
                    confidence *= (1.0 + driver_w)
                atom_label = f"{rname}{rid}_{spike_type}"
                anchors.append(AnchorPoint(
                    residue_name=rname, residue_id=rid, chain=chain,
                    atom_label=atom_label, interaction_type=interaction,
                    x=round(sx_c, 3), y=round(sy_c, 3), z=round(sz_c, 3),
                    distance_to_centroid=round(d_cent, 3),
                    spike_intensity=round(intens_c, 4),
                    temporal_persistence=round(persistence, 4),
                    geometric_alignment=round(alignment, 4),
                    stability_stddev=round(stddev, 4),
                    confidence=round(confidence, 6),
                ))

        anchors.sort(key=lambda a: a.confidence, reverse=True)
        deduped: List[AnchorPoint] = []
        seen: set = set()
        for a in anchors:
            rkey = f"{a.chain}:{a.residue_id}"
            if rkey not in seen:
                deduped.append(a)
                seen.add(rkey)
        n_lining = len(lining) if lining else 1
        density = len(deduped) / n_lining
        return AnchorPointMap(
            site_id=site_id, pocket_centroid=centroid,
            anchors=deduped, n_anchors=len(deduped),
            anchor_density=round(density, 4),
        )

    def compute_all(
        self,
        sites: List[Dict[str, Any]],
        spike_events_dir: Optional[str] = None,
    ) -> Dict[int, AnchorPointMap]:
        """Compute anchor point maps for all sites.

        D5_V2 Arrow-first: when the triad is present in spike_events_dir, open
        a SiteSpikeView once and dispatch to compute_vectorized per site.
        Falls back to per-site JSON load + compute(dict-list) when triad absent.
        """
        results: Dict[int, AnchorPointMap] = {}

        view = None
        view_sids: set = set()
        if spike_events_dir and Path(spike_events_dir).exists():
            try:
                from scripts.interfaces.site_spike_view import SiteSpikeView
                eng, stem = _arrow_first_triad_paths(Path(spike_events_dir))
                if eng is not None and stem is not None:
                    if eng.name == "5_engine":
                        target_dir = eng.parent.parent
                    else:
                        target_dir = eng
                    view = SiteSpikeView.from_target_dir(target_dir, stem)
                    if view is not None:
                        view_sids = set(view.available_site_ids())
            except Exception:
                view = None
                view_sids = set()

        for i, site in enumerate(sites):
            site_id = site.get("id", i)

            if view is not None and site_id in view_sids:
                slice_obj = view.site(site_id)
                results[site_id] = self.compute_vectorized(site, slice_obj)
                continue

            spikes: List[Dict[str, Any]] = []
            if spike_events_dir and Path(spike_events_dir).exists():
                se = load_spike_events(spike_events_dir, site_id)
                if se:
                    spikes = se.get("spikes", [])
            if not spikes and "spikes" in site:
                spikes = site["spikes"]

            results[site_id] = self.compute(site, spikes)

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM4D AnchorPoint Mapper"
    )
    parser.add_argument(
        "--binding-sites", required=True, help="Path to binding_sites.json"
    )
    parser.add_argument(
        "--spike-events", default=None, help="Spike events directory"
    )
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    with open(args.binding_sites) as f:
        data = json.load(f)
    sites = data if isinstance(data, list) else data.get("sites", [])

    mapper = AnchorPointMapper()
    results = mapper.compute_all(sites, args.spike_events)

    output = {str(sid): r.to_dict() for sid, r in sorted(results.items())}

    if args.out:
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote {len(output)} anchor maps to {args.out}")
    else:
        for sid, am in sorted(results.items()):
            print(
                f"Site {sid}: {am.n_anchors} anchors "
                f"(density={am.anchor_density:.3f})"
            )
            for a in am.anchors[:5]:
                print(
                    f"  {a.atom_label:<20s} {a.interaction_type:<14s} "
                    f"int={a.spike_intensity:.2f} "
                    f"persist={a.temporal_persistence:.2f} "
                    f"conf={a.confidence:.4f}"
                )


if __name__ == "__main__":
    main()

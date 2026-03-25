#!/usr/bin/env python3
"""PRISM4D — GrowthVector Mapper.

For each binding site, casts rays from anchor points to find expansion
directions and segments pharmacophore features into sub-pockets.

Usage (standalone):
    python3 scripts/growth_vector_map.py \\
        --binding-sites /path/to/binding_sites.json \\
        --anchor-maps /path/to/anchor_points.json \\
        [--out /path/to/growth_vectors.json]

Programmatic:
    from scripts.growth_vector_map import GrowthVectorMapper
    mapper = GrowthVectorMapper()
    gv_maps = mapper.compute_all(sites, anchor_maps)
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from scripts.interfaces.anchor_point import AnchorPoint, AnchorPointMap
from scripts.interfaces.growth_vector import (
    GrowthVector,
    GrowthVectorMap,
    SubPocket,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class GrowthVectorConfig:
    """Configuration for growth vector computation."""

    ray_step: float = 0.5          # Angstrom step along ray
    max_ray_length: float = 15.0   # Angstrom max extension
    protein_contact_radius: float = 4.0  # Angstrom for contact density
    solvent_escape_distance: float = 12.0  # from centroid → solvent
    min_free_length: float = 2.0   # reject very short vectors
    subpocket_cluster_radius: float = 5.0  # Angstrom for feature clustering


# ---------------------------------------------------------------------------
# 26 cardinal + diagonal directions (cube vertices + edges + faces)
# ---------------------------------------------------------------------------
_DIRECTIONS: List[Tuple[float, float, float]] = []
for dx in (-1, 0, 1):
    for dy in (-1, 0, 1):
        for dz in (-1, 0, 1):
            if dx == 0 and dy == 0 and dz == 0:
                continue
            mag = math.sqrt(dx * dx + dy * dy + dz * dz)
            _DIRECTIONS.append((dx / mag, dy / mag, dz / mag))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def _residue_positions(site: Dict[str, Any]) -> List[Tuple[float, float, float]]:
    """Extract approximate residue positions from lining_residues.

    Since binding_sites.json doesn't carry full atom coordinates,
    we approximate lining residue positions by distributing them
    around the centroid at their min_distance.
    """
    centroid = site.get("centroid", [0, 0, 0])
    lining = site.get("lining_residues", [])
    positions: List[Tuple[float, float, float]] = []

    if not lining:
        return positions

    # Distribute residues uniformly on a sphere at their min_distance
    n = len(lining)
    for i, res in enumerate(lining):
        min_d = res.get("min_distance", 5.0)
        # Golden angle distribution
        theta = math.acos(1 - 2 * (i + 0.5) / n)
        phi = math.pi * (1 + math.sqrt(5)) * i
        x = centroid[0] + min_d * math.sin(theta) * math.cos(phi)
        y = centroid[1] + min_d * math.sin(theta) * math.sin(phi)
        z = centroid[2] + min_d * math.cos(theta)
        positions.append((x, y, z))

    return positions


def _cast_ray(
    origin: Tuple[float, float, float],
    direction: Tuple[float, float, float],
    protein_atoms: List[Tuple[float, float, float]],
    centroid: Tuple[float, float, float],
    cfg: GrowthVectorConfig,
) -> Tuple[float, float, bool]:
    """Cast a ray and measure free length + contact density.

    Returns (free_length, contact_density, exits_to_solvent).
    """
    step = cfg.ray_step
    max_len = cfg.max_ray_length
    contact_r = cfg.protein_contact_radius
    solvent_d = cfg.solvent_escape_distance

    free_length = 0.0
    contact_count = 0
    n_steps = 0

    t = 0.0
    while t < max_len:
        t += step
        px = origin[0] + direction[0] * t
        py = origin[1] + direction[1] * t
        pz = origin[2] + direction[2] * t

        n_steps += 1

        # Check contact with protein atoms
        hit = False
        for atom in protein_atoms:
            d = math.sqrt(
                (px - atom[0]) ** 2
                + (py - atom[1]) ** 2
                + (pz - atom[2]) ** 2
            )
            if d < 2.0:  # hard wall
                return (t, contact_count / max(n_steps, 1), False)
            if d < contact_r:
                contact_count += 1

        # Check solvent escape
        d_centroid = math.sqrt(
            (px - centroid[0]) ** 2
            + (py - centroid[1]) ** 2
            + (pz - centroid[2]) ** 2
        )
        if d_centroid > solvent_d:
            return (t, contact_count / max(n_steps, 1), True)

    return (max_len, contact_count / max(n_steps, 1), False)


# ---------------------------------------------------------------------------
# Sub-pocket segmentation
# ---------------------------------------------------------------------------
def _segment_subpockets(
    anchors: List[AnchorPoint],
    cluster_radius: float,
) -> List[SubPocket]:
    """Cluster anchors into sub-pockets by spatial proximity."""
    if not anchors:
        return []

    # Simple greedy clustering
    assigned = [False] * len(anchors)
    clusters: List[List[int]] = []

    for i in range(len(anchors)):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        ai = (anchors[i].x, anchors[i].y, anchors[i].z)
        for j in range(i + 1, len(anchors)):
            if assigned[j]:
                continue
            aj = (anchors[j].x, anchors[j].y, anchors[j].z)
            if _dist(ai, aj) < cluster_radius:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    sub_pockets: List[SubPocket] = []
    for sp_id, indices in enumerate(clusters):
        cx = sum(anchors[i].x for i in indices) / len(indices)
        cy = sum(anchors[i].y for i in indices) / len(indices)
        cz = sum(anchors[i].z for i in indices) / len(indices)

        types = [anchors[i].interaction_type for i in indices]
        type_counts = Counter(types)
        dominant = type_counts.most_common(1)[0][0]

        # Volume estimate: max pairwise distance cubed / 6 (sphere approx)
        if len(indices) > 1:
            max_d = max(
                _dist(
                    (anchors[i].x, anchors[i].y, anchors[i].z),
                    (anchors[j].x, anchors[j].y, anchors[j].z),
                )
                for i in indices
                for j in indices
                if i < j
            )
            volume = (4.0 / 3.0) * math.pi * (max_d / 2.0) ** 3
        else:
            volume = 0.0

        sub_pockets.append(
            SubPocket(
                sub_pocket_id=sp_id,
                centroid=(round(cx, 3), round(cy, 3), round(cz, 3)),
                volume=round(volume, 1),
                feature_types=types,
                n_features=len(indices),
                dominant_interaction=dominant,
            )
        )

    return sub_pockets


# ---------------------------------------------------------------------------
# Mapper
# ---------------------------------------------------------------------------
class GrowthVectorMapper:
    """Computes GrowthVectorMaps for binding sites."""

    def __init__(self, config: Optional[GrowthVectorConfig] = None):
        self.cfg = config or GrowthVectorConfig()

    def compute(
        self,
        site: Dict[str, Any],
        anchor_map: AnchorPointMap,
    ) -> GrowthVectorMap:
        """Compute growth vectors and subpockets for one site."""
        site_id = site.get("id", -1)
        centroid_list = site.get("centroid", [0.0, 0.0, 0.0])
        centroid = (centroid_list[0], centroid_list[1], centroid_list[2])

        protein_atoms = _residue_positions(site)

        vectors: List[GrowthVector] = []

        for anchor in anchor_map.anchors:
            origin = (anchor.x, anchor.y, anchor.z)

            for direction in _DIRECTIONS:
                free_len, contact_d, exits = _cast_ray(
                    origin, direction, protein_atoms, centroid, self.cfg
                )

                # Filter: reject short vectors and solvent-exiting ones
                if exits:
                    continue
                if free_len < self.cfg.min_free_length:
                    continue

                # Expansion stability: higher when contact density is
                # moderate (pocket walls present but not blocking)
                stability = 1.0 / (1.0 + contact_d)

                score = (
                    free_len
                    * (1.0 / (contact_d + 0.1))
                    * stability
                )

                vectors.append(
                    GrowthVector(
                        origin=origin,
                        direction=direction,
                        free_length=round(free_len, 3),
                        contact_density=round(contact_d, 4),
                        expansion_stability=round(stability, 4),
                        exits_to_solvent=exits,
                        vector_score=round(score, 4),
                        source_anchor_label=anchor.atom_label,
                    )
                )

        # Keep top vectors per anchor (avoid flooding)
        vectors.sort(key=lambda v: v.vector_score, reverse=True)
        # Limit to top 3 per anchor
        anchor_counts: Dict[str, int] = defaultdict(int)
        filtered: List[GrowthVector] = []
        for v in vectors:
            if anchor_counts[v.source_anchor_label] < 3:
                filtered.append(v)
                anchor_counts[v.source_anchor_label] += 1

        # Subpocket segmentation
        sub_pockets = _segment_subpockets(
            anchor_map.anchors, self.cfg.subpocket_cluster_radius
        )

        return GrowthVectorMap(
            site_id=site_id,
            vectors=filtered,
            sub_pockets=sub_pockets,
            n_vectors=len(filtered),
            n_sub_pockets=len(sub_pockets),
        )

    def compute_all(
        self,
        sites: List[Dict[str, Any]],
        anchor_maps: Dict[int, AnchorPointMap],
    ) -> Dict[int, GrowthVectorMap]:
        """Compute growth vector maps for all sites."""
        results: Dict[int, GrowthVectorMap] = {}
        for i, site in enumerate(sites):
            site_id = site.get("id", i)
            am = anchor_maps.get(site_id)
            if am is None:
                am = AnchorPointMap(
                    site_id=site_id,
                    pocket_centroid=tuple(site.get("centroid", [0, 0, 0])),
                    anchors=[],
                    n_anchors=0,
                    anchor_density=0.0,
                )
            results[site_id] = self.compute(site, am)
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM4D GrowthVector Mapper"
    )
    parser.add_argument(
        "--binding-sites", required=True, help="Path to binding_sites.json"
    )
    parser.add_argument(
        "--anchor-maps", required=True, help="Path to anchor_points.json"
    )
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    with open(args.binding_sites) as f:
        data = json.load(f)
    sites = data if isinstance(data, list) else data.get("sites", [])

    with open(args.anchor_maps) as f:
        am_raw = json.load(f)
    anchor_maps = {
        int(k): AnchorPointMap.from_dict(v) for k, v in am_raw.items()
    }

    mapper = GrowthVectorMapper()
    results = mapper.compute_all(sites, anchor_maps)

    output = {str(sid): r.to_dict() for sid, r in sorted(results.items())}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Wrote {len(output)} growth vector maps to {args.out}")
    else:
        for sid, gvm in sorted(results.items()):
            print(
                f"Site {sid}: {gvm.n_vectors} vectors, "
                f"{gvm.n_sub_pockets} sub-pockets"
            )


if __name__ == "__main__":
    main()

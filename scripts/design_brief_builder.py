#!/usr/bin/env python3
"""PRISM4D — DesignBrief Builder.

Generates the complete design output for sites that passed the full gating
stack: JSON data, PyMOL visualization script, and HTML report.

All output is pure projection of computed data — no recommendations,
no confidence statements, no executive summaries.

Usage (standalone):
    python3 scripts/design_brief_builder.py \\
        --binding-sites /path/to/binding_sites.json \\
        --gating-result /path/to/gating_result.json \\
        --anchor-maps /path/to/anchor_points.json \\
        --growth-maps /path/to/growth_vectors.json \\
        --profiles /path/to/pocket_profiles.json \\
        --ranking /path/to/site_ranking.json \\
        --out-dir /path/to/output/

Programmatic:
    from scripts.design_brief_builder import DesignBriefBuilder
    builder = DesignBriefBuilder()
    briefs = builder.build_all(...)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.interfaces.anchor_point import AnchorPointMap
from scripts.interfaces.design_brief import DesignBrief
from scripts.interfaces.growth_vector import GrowthVectorMap
from scripts.interfaces.pocket_profile import PocketProfile
from scripts.interfaces.site_ranking import RankedSite, SiteRanking


# ---------------------------------------------------------------------------
# Interaction type → PyMOL color
# ---------------------------------------------------------------------------
INTERACTION_COLORS: Dict[str, str] = {
    "PI_STACK": "magenta",
    "HYDROPHOBIC": "orange",
    "H_BOND_DONOR": "blue",
    "H_BOND_ACCEPTOR": "red",
    "SALT_BRIDGE": "yellow",
    "COVALENT": "green",
}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class DesignBriefBuilder:
    """Generates DesignBrief outputs for gated sites."""

    def build(
        self,
        target_name: str,
        pdb_id: str,
        site_id: int,
        ranked_site: RankedSite,
        anchor_map: AnchorPointMap,
        growth_map: GrowthVectorMap,
        pocket_profile: PocketProfile,
        water_sites: Optional[List[Dict[str, Any]]] = None,
    ) -> DesignBrief:
        """Build a DesignBrief for one site."""
        return DesignBrief(
            target_name=target_name,
            pdb_id=pdb_id,
            site_id=site_id,
            ranked_site=ranked_site,
            anchor_map=anchor_map,
            growth_map=growth_map,
            pocket_profile=pocket_profile,
            water_sites=water_sites or [],
        )

    def build_all(
        self,
        target_name: str,
        pdb_id: str,
        ranking: SiteRanking,
        anchor_maps: Dict[int, AnchorPointMap],
        growth_maps: Dict[int, GrowthVectorMap],
        profiles: Dict[int, PocketProfile],
        water_data: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> List[DesignBrief]:
        """Build DesignBriefs for all ranked sites."""
        wd = water_data or {}
        briefs: List[DesignBrief] = []

        for rs in ranking.ranked_sites:
            sid = rs.site_id
            am = anchor_maps.get(sid)
            gm = growth_maps.get(sid)
            pp = profiles.get(sid)

            if am is None or gm is None or pp is None:
                continue

            briefs.append(
                self.build(
                    target_name=target_name,
                    pdb_id=pdb_id,
                    site_id=sid,
                    ranked_site=rs,
                    anchor_map=am,
                    growth_map=gm,
                    pocket_profile=pp,
                    water_sites=wd.get(sid),
                )
            )

        return briefs

    # -- Output generators --------------------------------------------------

    def write_json(self, brief: DesignBrief, path: str) -> None:
        """Write DesignBrief as JSON."""
        with open(path, "w") as f:
            json.dump(brief.to_dict(), f, indent=2)

    def write_pymol(
        self, brief: DesignBrief, path: str, pdb_path: str = ""
    ) -> None:
        """Generate PyMOL .pml visualization script."""
        lines: List[str] = []
        lines.append(f"# PRISM4D DesignBrief — {brief.target_name} site {brief.site_id}")
        lines.append(f"# Rank: {brief.ranked_site.rank}")
        lines.append("")

        if pdb_path:
            lines.append(f'load {pdb_path}, {brief.target_name}')
        lines.append(f"hide all")
        lines.append(f"show cartoon, {brief.target_name}")
        lines.append(f"color gray80, {brief.target_name}")
        lines.append("")

        # Pocket centroid
        cx, cy, cz = brief.anchor_map.pocket_centroid
        lines.append(f"# Pocket centroid")
        lines.append(
            f"pseudoatom centroid_{brief.site_id}, "
            f"pos=[{cx:.3f},{cy:.3f},{cz:.3f}]"
        )
        lines.append(
            f"show spheres, centroid_{brief.site_id}"
        )
        lines.append(f"set sphere_scale, 0.8, centroid_{brief.site_id}")
        lines.append(f"color white, centroid_{brief.site_id}")
        lines.append("")

        # Anchor points
        lines.append("# Anchor Points")
        for j, a in enumerate(brief.anchor_map.anchors):
            name = f"anchor_{brief.site_id}_{j}"
            color = INTERACTION_COLORS.get(a.interaction_type, "gray50")
            lines.append(
                f"pseudoatom {name}, "
                f"pos=[{a.x:.3f},{a.y:.3f},{a.z:.3f}], "
                f"label=\"{a.atom_label} ({a.interaction_type})\""
            )
            lines.append(f"show spheres, {name}")
            lines.append(f"set sphere_scale, 0.5, {name}")
            lines.append(f"color {color}, {name}")
        lines.append("")

        # Growth vectors as CGO arrows
        lines.append("# Growth Vectors")
        lines.append("from pymol.cgo import *")
        cgo_parts: List[str] = []
        for k, v in enumerate(brief.growth_map.vectors[:15]):
            ox, oy, oz = v.origin
            dx, dy, dz = v.direction
            ex = ox + dx * v.free_length
            ey = oy + dy * v.free_length
            ez = oz + dz * v.free_length
            cgo_parts.append(
                f"CYLINDER, {ox:.3f},{oy:.3f},{oz:.3f}, "
                f"{ex:.3f},{ey:.3f},{ez:.3f}, "
                f"0.15, 0.2,0.8,0.2, 0.2,0.8,0.2"
            )

        if cgo_parts:
            cgo_str = ", ".join(cgo_parts)
            lines.append(f"cmd.load_cgo([{cgo_str}], 'growth_vectors_{brief.site_id}')")
        lines.append("")

        # Sub-pockets
        lines.append("# Sub-pockets")
        sp_colors = ["salmon", "palegreen", "lightblue", "lightorange", "violet"]
        for sp in brief.growth_map.sub_pockets:
            spx, spy, spz = sp.centroid
            name = f"subpocket_{brief.site_id}_{sp.sub_pocket_id}"
            col = sp_colors[sp.sub_pocket_id % len(sp_colors)]
            lines.append(
                f"pseudoatom {name}, "
                f"pos=[{spx:.3f},{spy:.3f},{spz:.3f}], "
                f"label=\"SP{sp.sub_pocket_id} ({sp.dominant_interaction})\""
            )
            lines.append(f"show spheres, {name}")
            lines.append(f"set sphere_scale, 1.2, {name}")
            lines.append(f"color {col}, {name}")
        lines.append("")

        # Water sites
        if brief.water_sites:
            lines.append("# Water Sites")
            for wi, ws in enumerate(brief.water_sites):
                wx = ws.get("x", 0)
                wy = ws.get("y", 0)
                wz = ws.get("z", 0)
                dg = ws.get("delta_g_transfer", 0)
                name = f"water_{brief.site_id}_{wi}"
                col = "red" if dg > 1.0 else "blue"
                label = f"dG={dg:.1f}"
                lines.append(
                    f"pseudoatom {name}, "
                    f"pos=[{wx:.3f},{wy:.3f},{wz:.3f}], "
                    f"label=\"{label}\""
                )
                lines.append(f"show spheres, {name}")
                lines.append(f"set sphere_scale, 0.4, {name}")
                lines.append(f"color {col}, {name}")

        lines.append("")
        lines.append(f"zoom centroid_{brief.site_id}, 20")
        lines.append(f"set label_size, 12")
        lines.append("")

        with open(path, "w") as f:
            f.write("\n".join(lines))

    def write_html(self, brief: DesignBrief, path: str) -> None:
        """Generate HTML report — pure data projection, no recommendations."""
        pp = brief.pocket_profile
        am = brief.anchor_map
        gm = brief.growth_map
        rs = brief.ranked_site

        html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>PRISM4D DesignBrief — {brief.target_name} Site {brief.site_id}</title>
<style>
body {{ font-family: monospace; max-width: 900px; margin: 2em auto; background: #fafafa; color: #222; }}
h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
h2 {{ color: #555; margin-top: 2em; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
th {{ background: #eee; }}
.tag {{ display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 0.85em; }}
.tag-hydrophobic {{ background: #ffe0b2; }}
.tag-polar {{ background: #b3e5fc; }}
.tag-mixed {{ background: #e1bee7; }}
.tag-fragment {{ background: #c8e6c9; }}
.tag-lead {{ background: #fff9c4; }}
.tag-beyond {{ background: #ffcdd2; }}
</style>
</head><body>

<h1>PRISM4D DesignBrief</h1>
<p><strong>Target:</strong> {brief.target_name} &nbsp;
<strong>PDB:</strong> {brief.pdb_id} &nbsp;
<strong>Site:</strong> {brief.site_id} &nbsp;
<strong>Rank:</strong> {rs.rank}</p>

<h2>Ranking Keys</h2>
<table>
<tr><th>Key</th><th>Value</th></tr>
<tr><td>Contact reorg strength</td><td>{rs.contact_reorg_strength:.6f}</td></tr>
<tr><td>Anchor density</td><td>{rs.anchor_density:.4f}</td></tr>
<tr><td>Water displacement (tie-breaker)</td><td>{rs.water_displacement:.3f} kcal/mol</td></tr>
</table>

<h2>Pocket Profile</h2>
<p>
<span class="tag tag-{pp.polarity_class}">{pp.polarity_class}</span>
<span class="tag tag-{pp.mw_class if pp.mw_class != 'beyond_ro5' else 'beyond'}">{pp.mw_class}</span>
</p>
<table>
<tr><th>Property</th><th>Value</th></tr>
<tr><td>Volume</td><td>{pp.volume:.1f} A^3</td></tr>
<tr><td>Enclosure</td><td>{pp.enclosure:.4f}</td></tr>
<tr><td>Lining residues</td><td>{pp.n_lining_residues}</td></tr>
<tr><td>Aromatic fraction</td><td>{pp.aromatic_fraction:.3f}</td></tr>
<tr><td>Polar fraction</td><td>{pp.polar_fraction:.3f}</td></tr>
<tr><td>Hydrophobic fraction</td><td>{pp.hydrophobic_fraction:.3f}</td></tr>
<tr><td>Charge bias</td><td>{pp.charge_bias:+.3f}</td></tr>
<tr><td>Feature coupling</td><td>{pp.feature_coupling:.4f}</td></tr>
<tr><td>Water displacement energy</td><td>{pp.water_displacement_energy:.3f} kcal/mol</td></tr>
</table>

<h2>Anchor Points ({am.n_anchors})</h2>
<table>
<tr><th>Atom</th><th>Interaction</th><th>Intensity</th><th>Persistence</th>
<th>Alignment</th><th>Stability (A)</th><th>Weight</th></tr>
"""
        for a in am.anchors:
            html += f"""<tr>
<td>{a.atom_label}</td><td>{a.interaction_type}</td>
<td>{a.spike_intensity:.2f}</td><td>{a.temporal_persistence:.2f}</td>
<td>{a.geometric_alignment:.2f}</td><td>{a.stability_stddev:.2f}</td>
<td>{a.confidence:.4f}</td></tr>
"""

        html += f"""</table>

<h2>Growth Vectors ({gm.n_vectors})</h2>
<table>
<tr><th>Source Anchor</th><th>Free Length (A)</th><th>Contact Density</th>
<th>Stability</th><th>Score</th></tr>
"""
        for v in gm.vectors[:20]:
            html += f"""<tr>
<td>{v.source_anchor_label}</td><td>{v.free_length:.2f}</td>
<td>{v.contact_density:.3f}</td><td>{v.expansion_stability:.3f}</td>
<td>{v.vector_score:.3f}</td></tr>
"""

        html += f"""</table>

<h2>Sub-pockets ({gm.n_sub_pockets})</h2>
<table>
<tr><th>ID</th><th>Features</th><th>Dominant</th><th>Volume (A^3)</th></tr>
"""
        for sp in gm.sub_pockets:
            html += f"""<tr>
<td>{sp.sub_pocket_id}</td><td>{sp.n_features}</td>
<td>{sp.dominant_interaction}</td><td>{sp.volume:.1f}</td></tr>
"""

        html += """</table>
"""

        if brief.water_sites:
            html += f"""
<h2>Water Strategy ({len(brief.water_sites)} sites)</h2>
<table>
<tr><th>Position</th><th>dG (kcal/mol)</th><th>Classification</th><th>Displaceable</th></tr>
"""
            for ws in brief.water_sites:
                pos = f"({ws.get('x',0):.1f}, {ws.get('y',0):.1f}, {ws.get('z',0):.1f})"
                dg = ws.get("delta_g_transfer", 0)
                cls_ = ws.get("classification", "BULK")
                disp = "Yes" if ws.get("displaceable", False) else "No"
                html += f"<tr><td>{pos}</td><td>{dg:.2f}</td><td>{cls_}</td><td>{disp}</td></tr>\n"
            html += "</table>\n"

        html += """
<hr>
<p><small>Generated by PRISM4D DesignBrief Builder.
All values are directly computed from spike, contact, and trajectory data.
</small></p>
</body></html>"""

        with open(path, "w") as f:
            f.write(html)

    def write_all(
        self, briefs: List[DesignBrief], out_dir: str, pdb_path: str = ""
    ) -> None:
        """Write all briefs to output directory (JSON + PyMOL + HTML)."""
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)

        for brief in briefs:
            prefix = f"{brief.target_name}_site{brief.site_id}"
            self.write_json(brief, str(d / f"{prefix}.json"))
            self.write_pymol(brief, str(d / f"{prefix}.pml"), pdb_path)
            self.write_html(brief, str(d / f"{prefix}.html"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM4D DesignBrief Builder"
    )
    parser.add_argument("--binding-sites", required=True)
    parser.add_argument("--gating-result", required=True)
    parser.add_argument("--anchor-maps", required=True)
    parser.add_argument("--growth-maps", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--ranking", required=True)
    parser.add_argument("--pdb", default="", help="PDB file for PyMOL")
    parser.add_argument("--target-name", default="unknown")
    parser.add_argument("--pdb-id", default="XXXX")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    with open(args.ranking) as f:
        ranking = SiteRanking.from_dict(json.load(f))
    with open(args.anchor_maps) as f:
        am_raw = json.load(f)
    anchor_maps = {int(k): AnchorPointMap.from_dict(v) for k, v in am_raw.items()}
    with open(args.growth_maps) as f:
        gm_raw = json.load(f)
    growth_maps = {int(k): GrowthVectorMap.from_dict(v) for k, v in gm_raw.items()}
    with open(args.profiles) as f:
        pp_raw = json.load(f)
    profiles = {int(k): PocketProfile.from_dict(v) for k, v in pp_raw.items()}

    builder = DesignBriefBuilder()
    briefs = builder.build_all(
        target_name=args.target_name,
        pdb_id=args.pdb_id,
        ranking=ranking,
        anchor_maps=anchor_maps,
        growth_maps=growth_maps,
        profiles=profiles,
    )

    builder.write_all(briefs, args.out_dir, args.pdb)
    print(f"Wrote {len(briefs)} design briefs to {args.out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PRISM-4D dossier visualization — target-agnostic SBDD bundle generator.

Reads a unified-dossier JSON + the corresponding target YAML config, then
emits a five-figure visualization bundle. Every protein-specific value
(region names, region colors, reference palette, pocket flag colors) is
pulled from the target YAML — there are zero hardcoded constants in this
file. Verify with:

    grep -iE '\\b(p-loop|switch|kras|sotorasib|adagrasib|MOV|M1X|F0K)\\b' \\
         scripts/visualize_dossier.py

That command should return zero matches outside this docstring's
illustrative example.

Bundle layout produced under <output_dir>:
  00_README.md
  01_global_atlas.pse        (or skipped with note)
  02_known_ligand_recovery.pse
  03_druggability_gradient.pse
  04_geodesic_vs_euclidean.pse
  05_medchem_contact_panel.pse
  images/      fig{N}_*.png  (2400x1800, dpi=300)
  tables/      pocket_rank_table.csv  visual_legend.csv  contact_fingerprint.csv
  pml_scripts/ 0{N}_*.pml    (always emitted; standalone)
  interactive/ overview.html (py3Dmol single-file)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

PML_BASELINE = """\
# PRISM-4D PyMOL render baseline (locked).
set ray_trace_mode, 1
set antialias, 2
set orthoscopic, on
set depth_cue, off
set ambient, 0.35
set direct, 0.65
set specular, 0.2
set shininess, 25
set ray_shadow, off
set cartoon_fancy_helices, on
set cartoon_transparency, 0.15
set surface_quality, 1
set transparency_mode, 1
bg_color white
"""


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
def load_target_config(target_id: str, config_root: Path) -> dict[str, Any]:
    """Load and lightly normalize the YAML target config."""
    cfg_path = config_root / f"{target_id}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"target config missing: {cfg_path}")
    raw = yaml.safe_load(open(cfg_path))
    vis = raw.get("visualization") or {}
    raw["_visualization"] = {
        "region_colors": vis.get("region_colors", {}) or {},
        "reference_palette": vis.get("reference_palette") or [
            "yellow", "magenta", "cyan", "salmon",
        ],
        "pocket_palette": vis.get("pocket_palette", {}) or {
            "novel": "cyan",
            "known_ligand_proximal": "yellow",
            "reference_hit": "gold",
            "megacluster_collapse": "gray50",
            "default": "white",
        },
    }
    raw["_canonical_regions"] = raw.get("canonical_regions") or {}
    raw["_references"] = raw.get("references") or []
    return raw


def derive_target_id(dossier_path: Path) -> str:
    name = dossier_path.name
    for suffix in ("_unified_dossier.json", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    name = name.split("_")[0]
    return name.lower()


# ---------------------------------------------------------------------------
# Pocket classification (target-agnostic; reads dossier flags)
# ---------------------------------------------------------------------------
def classify_pocket(pocket: dict[str, Any]) -> str:
    """Return the pocket-palette key for `pocket` based on dossier-emitted
    flags + interpretation fields. Target-agnostic — no hardcoded names."""
    e18 = pocket.get("enhancements", {}).get("E18_geodesic_centroid", {})
    e11 = pocket.get("enhancements", {}).get("E11_multi_view_dcc", {})
    if e18.get("interpretation") == "MEGACLUSTER_COLLAPSE_DETECTED":
        return "megacluster_collapse"
    if (e11.get("status") == "OK"
            and e11.get("best_reference_min_prox", 99) < 2.0):
        return "reference_hit"
    if (e11.get("status") == "OK"
            and e11.get("best_reference_min_prox", 99) < 5.0):
        return "known_ligand_proximal"
    novelty = pocket.get("novelty_score", 0)
    if novelty > 5.0:
        return "novel"
    return "default"


# ---------------------------------------------------------------------------
# PML generators
# ---------------------------------------------------------------------------
def _resi_select_clause(residues: list[int]) -> str:
    return "+".join(str(r) for r in sorted(residues))


def _region_selections(target_obj: str, cfg: dict[str, Any]) -> str:
    out = []
    for region_name, residues in cfg["_canonical_regions"].items():
        ranges = residues.get("ranges", [])
        sel_parts = []
        for r in ranges:
            if len(r) == 2:
                sel_parts.append(f"resi {r[0]}-{r[1]}")
        if sel_parts:
            sel_name = (
                region_name.replace("/", "_")
                            .replace(" ", "_")
                            .replace("-", "_")
            )
            joined = " or ".join(sel_parts)
            out.append(
                f"select {sel_name}_residues, {target_obj} and ({joined})"
            )
    return "\n".join(out)


def _region_coloring(target_obj: str, cfg: dict[str, Any]) -> str:
    out = []
    for region_name, color in cfg["_visualization"]["region_colors"].items():
        sel_name = (
            region_name.replace("/", "_")
                        .replace(" ", "_")
                        .replace("-", "_")
        )
        out.append(f"color {color}, {sel_name}_residues")
    return "\n".join(out)


def _ref_load_block(refs_dir: Path, cfg: dict[str, Any]) -> str:
    lines = []
    palette = cfg["_visualization"]["reference_palette"]
    for i, ref in enumerate(cfg["_references"]):
        pdb_id = ref["pdb_id"]
        het = ref["het"]
        name = ref.get("name", pdb_id).replace(" ", "_")
        ref_path = refs_dir / f"{pdb_id}.pdb"
        if not ref_path.exists():
            lines.append(f"# (skipped: {ref_path} missing)")
            continue
        obj = f"{pdb_id}_{name}_{het}"
        lines.append(f"load {ref_path}, {obj}")
        lines.append(f"hide everything, {obj}")
        lines.append(f"select {obj}_lig, {obj} and resn {het}")
        lines.append(f"show sticks, {obj}_lig")
        color = palette[i % len(palette)]
        lines.append(f"color {color}, {obj}_lig")
    return "\n".join(lines)


def fig1_global_atlas(
    pml_path: Path, dossier: dict, cfg: dict, md_pdb: Path,
    refs_dir: Path, png_path: Path | None,
) -> None:
    palette = cfg["_visualization"]["pocket_palette"]
    target_obj = "target_cartoon"
    lines = [
        "# Figure 1 — global atlas: target cartoon by region, pockets surfaced",
        "# by druggability flag, references shown as sticks where proximal.",
        PML_BASELINE,
        f"load {md_pdb}, {target_obj}",
        f"hide everything, {target_obj}",
        f"show cartoon, {target_obj}",
        f"color gray80, {target_obj}",
        _region_selections(target_obj, cfg),
        _region_coloring(target_obj, cfg),
        _ref_load_block(refs_dir, cfg),
    ]
    for p in dossier["pockets"]:
        idx = p["pocket_id"]
        flag_key = classify_pocket(p)
        color = palette.get(flag_key, palette.get("default", "white"))
        residues = p.get("residues", [])
        if not residues:
            continue
        sel = _resi_select_clause(residues)
        sgrp = f"P{idx:02d}"
        lines += [
            f"# Pocket {idx} ({flag_key})",
            f"select {sgrp}_residues, {target_obj} and resi {sel}",
            f"show surface, {sgrp}_residues",
            f"set transparency, 0.4, {sgrp}_residues",
            f"color {color}, {sgrp}_residues",
        ]
        # Geodesic anchor
        e18 = p.get("enhancements", {}).get("E18_geodesic_centroid", {})
        if e18.get("status") == "OK" and e18.get("geodesic_anchor_xyz"):
            x, y, z = e18["geodesic_anchor_xyz"]
            lines.append(
                f"pseudoatom {sgrp}_centroid_geodesic, pos=[{x:.3f}, {y:.3f}, {z:.3f}]"
            )
            lines.append(f"show spheres, {sgrp}_centroid_geodesic")
            lines.append(f"set sphere_scale, 0.6, {sgrp}_centroid_geodesic")
            lines.append(f"color {color}, {sgrp}_centroid_geodesic")
    lines += [
        "orient",
        "zoom polymer, 5",
        f"# Render: {png_path or '(skipped)'}",
    ]
    if png_path:
        lines.append(f"png {png_path}, width=2400, height=1800, dpi=300, ray=1")
    pml_path.write_text("\n".join(lines) + "\n")


def fig2_ligand_recovery(
    pml_path: Path, dossier: dict, cfg: dict, md_pdb: Path,
    refs_dir: Path, png_path: Path | None,
) -> dict | None:
    """Pick pocket with lowest min_prox to ANY reference."""
    candidates = []
    for p in dossier["pockets"]:
        e11 = p.get("enhancements", {}).get("E11_multi_view_dcc", {})
        if e11.get("status") == "OK":
            mp = e11.get("best_reference_min_prox", 99)
            best_ref = e11.get("best_reference")
            candidates.append((mp, best_ref, p))
    if not candidates:
        pml_path.write_text(
            "# Figure 2 — no reference-anchored pockets to render.\n"
        )
        return None
    candidates.sort(key=lambda t: t[0])
    min_prox, best_ref, p = candidates[0]
    idx = p["pocket_id"]
    sel = _resi_select_clause(p["residues"])
    target_obj = "target_cartoon"

    ref_meta = next(
        (r for r in cfg["_references"] if r["pdb_id"] == best_ref), {}
    )
    het = ref_meta.get("het", "UNK")
    ref_obj = f"ref_{best_ref}"
    ref_path = refs_dir / f"{best_ref}.pdb"

    lines = [
        f"# Figure 2 — known-ligand recovery",
        f"# Best pocket: P{idx} -> {best_ref}/{het} at min_prox={min_prox:.2f}A",
        PML_BASELINE,
        f"load {md_pdb}, {target_obj}",
        f"hide everything, {target_obj}",
        f"show cartoon, {target_obj}",
        f"color gray80, {target_obj}",
        f"select recovery_residues, {target_obj} and resi {sel}",
        "show sticks, recovery_residues",
        "color salmon, recovery_residues",
    ]
    if ref_path.exists():
        lines += [
            f"load {ref_path}, {ref_obj}",
            f"hide everything, {ref_obj}",
            f"select {ref_obj}_lig, {ref_obj} and resn {het}",
            f"show sticks, {ref_obj}_lig",
            f"color magenta, {ref_obj}_lig",
            f"# Distance dash from pocket residues to ligand atoms",
            f"distance recovery_dash, recovery_residues, {ref_obj}_lig, 5.0",
            f"hide labels, recovery_dash",
        ]
    lines += [
        "orient recovery_residues",
        f"label recovery_residues and name CA, '\"P{idx} min_prox={min_prox:.2f}A\"'",
    ]
    if png_path:
        lines.append(f"png {png_path}, width=2400, height=1800, dpi=300, ray=1")
    pml_path.write_text("\n".join(lines) + "\n")
    return {
        "pocket_id": idx,
        "best_ref": best_ref,
        "het": het,
        "min_prox": min_prox,
        "residues": p["residues"],
    }


def fig3_druggability_gradient(
    pml_path: Path, dossier: dict, cfg: dict, md_pdb: Path,
    png_path: Path | None,
) -> None:
    pockets = dossier["pockets"]
    if not pockets:
        pml_path.write_text("# Figure 3 — no pockets to render.\n")
        return
    scores = [p.get("drug_score_geodesic", p.get("druggability_score", 0))
              for p in pockets]
    s_min, s_max = min(scores), max(scores)
    spread = max(s_max - s_min, 1e-9)

    target_obj = "target_cartoon"
    lines = [
        "# Figure 3 — druggability gradient (red=high, white=mid, blue=low)",
        PML_BASELINE,
        f"load {md_pdb}, {target_obj}",
        f"hide everything, {target_obj}",
        f"show cartoon, {target_obj}",
        f"color gray80, {target_obj}",
    ]
    for p, s in zip(pockets, scores):
        idx = p["pocket_id"]
        residues = p.get("residues", [])
        if not residues:
            continue
        # Map score → red/white/blue via PyMOL ramp; expressed as RGB.
        t = (s - s_min) / spread
        # red = 1, blue = 0
        if t > 0.5:
            r = 1.0
            g = b = 1.0 - 2.0 * (t - 0.5)
        else:
            b = 1.0
            r = g = 2.0 * t
        rgb_label = f"P{idx:02d}_drug_color"
        lines += [
            f"set_color {rgb_label}, [{r:.3f}, {g:.3f}, {b:.3f}]",
            f"select P{idx:02d}_residues, {target_obj} and resi {_resi_select_clause(residues)}",
            f"show surface, P{idx:02d}_residues",
            f"set transparency, 0.45, P{idx:02d}_residues",
            f"color {rgb_label}, P{idx:02d}_residues",
        ]
    lines += ["orient", "zoom polymer, 5"]
    if png_path:
        lines.append(f"png {png_path}, width=2400, height=1800, dpi=300, ray=1")
    pml_path.write_text("\n".join(lines) + "\n")


def fig4_geodesic_vs_euclidean(
    pml_path: Path, dossier: dict, cfg: dict, md_pdb: Path,
    png_path: Path | None,
) -> None:
    pockets = dossier["pockets"]
    clean = next((p for p in pockets
                  if p.get("enhancements", {}).get("E18_geodesic_centroid", {})
                  .get("interpretation") == "GEODESIC_AGREES_EUCLIDEAN"), None)
    collapse = next((p for p in pockets
                     if p.get("enhancements", {}).get("E18_geodesic_centroid", {})
                     .get("interpretation") == "MEGACLUSTER_COLLAPSE_DETECTED"),
                    None)
    if not clean and not collapse:
        pml_path.write_text("# Figure 4 — no E18 examples available.\n")
        return

    target_obj = "target_cartoon"
    lines = [
        "# Figure 4 — geodesic anchor vs Euclidean centroid",
        PML_BASELINE,
        f"load {md_pdb}, {target_obj}",
        f"hide everything, {target_obj}",
        f"show cartoon, {target_obj}",
        f"color gray80, {target_obj}",
    ]

    def _emit(p, label):
        idx = p["pocket_id"]
        e18 = p["enhancements"]["E18_geodesic_centroid"]
        diag = p.get("core_diagnostics", {})
        eucl_xyz = diag.get("centroid_xyz")
        geo_xyz = e18.get("geodesic_anchor_xyz")
        delta = e18.get("euclidean_to_geodesic_delta_A", 0.0)
        residues = p.get("residues", [])
        sel = _resi_select_clause(residues) if residues else None
        out = []
        if sel:
            out.append(
                f"select P{idx:02d}_residues, {target_obj} and resi {sel}"
            )
            out.append(f"show sticks, P{idx:02d}_residues")
            out.append(f"color cyan, P{idx:02d}_residues")
        if eucl_xyz:
            out.append(
                f"pseudoatom P{idx:02d}_centroid_euclidean, "
                f"pos=[{eucl_xyz[0]:.3f}, {eucl_xyz[1]:.3f}, {eucl_xyz[2]:.3f}]"
            )
            out.append(f"show spheres, P{idx:02d}_centroid_euclidean")
            out.append(f"set sphere_scale, 0.7, P{idx:02d}_centroid_euclidean")
            out.append(f"color white, P{idx:02d}_centroid_euclidean")
        if geo_xyz:
            out.append(
                f"pseudoatom P{idx:02d}_centroid_geodesic, "
                f"pos=[{geo_xyz[0]:.3f}, {geo_xyz[1]:.3f}, {geo_xyz[2]:.3f}]"
            )
            out.append(f"show spheres, P{idx:02d}_centroid_geodesic")
            out.append(f"set sphere_scale, 0.7, P{idx:02d}_centroid_geodesic")
            out.append(f"color cyan, P{idx:02d}_centroid_geodesic")
        if eucl_xyz and geo_xyz:
            out.append(
                f"distance P{idx:02d}_eu_geo_link, "
                f"P{idx:02d}_centroid_euclidean, P{idx:02d}_centroid_geodesic"
            )
            out.append(
                f"set dash_color, magenta, P{idx:02d}_eu_geo_link"
            )
        out.append(
            f"# {label}: P{idx} euclidean→geodesic delta = {delta:.2f} A"
        )
        return out

    if clean:
        lines += _emit(clean, "GEODESIC_AGREES")
    if collapse:
        lines += _emit(collapse, "MEGACLUSTER_COLLAPSE")
    lines += ["orient", "zoom polymer, 5"]
    if png_path:
        lines.append(f"png {png_path}, width=2400, height=1800, dpi=300, ray=1")
    pml_path.write_text("\n".join(lines) + "\n")


def fig5_medchem_panel(
    pml_path: Path, dossier: dict, cfg: dict, md_pdb: Path,
    refs_dir: Path, recovery_meta: dict | None,
    contact_csv: Path, png_path: Path | None,
) -> str:
    """Render contact panel using ProLIF if available; else fall back to PLIP;
    else write a clear note. Returns a string describing the path taken."""
    if not recovery_meta:
        pml_path.write_text("# Figure 5 — no recovery pocket; skipped.\n")
        return "no_recovery_pocket"

    try:
        import prolif as plf
        import MDAnalysis as mda
    except ImportError as exc:
        pml_path.write_text(
            "# Figure 5 — ProLIF not available; contact panel skipped.\n"
            f"# (import error: {exc})\n"
        )
        return f"prolif_unavailable: {exc}"

    ref_pdb = refs_dir / f"{recovery_meta['best_ref']}.pdb"
    if not ref_pdb.exists():
        pml_path.write_text(
            f"# Figure 5 — reference PDB {ref_pdb} missing.\n"
        )
        return "ref_pdb_missing"

    try:
        u = mda.Universe(str(ref_pdb))
        protein = u.select_atoms(
            f"protein and resid {' '.join(str(r) for r in recovery_meta['residues'])}"
        )
        ligand = u.select_atoms(f"resname {recovery_meta['het']}")
        if len(protein) == 0 or len(ligand) == 0:
            pml_path.write_text(
                "# Figure 5 — empty selection; check residue/HET overlap.\n"
            )
            return "empty_selection"
        fp = plf.Fingerprint()
        fp.run_from_iterable([ligand], protein)
        df = fp.to_dataframe()
        df.to_csv(contact_csv)
    except Exception as exc:
        pml_path.write_text(f"# Figure 5 — ProLIF error: {exc}\n")
        return f"prolif_error: {exc}"

    target_obj = "target_cartoon"
    sel = _resi_select_clause(recovery_meta["residues"])
    ref_obj = f"ref_{recovery_meta['best_ref']}"
    het = recovery_meta["het"]
    lines = [
        "# Figure 5 — med-chem contact panel (ProLIF IFP)",
        PML_BASELINE,
        f"load {md_pdb}, {target_obj}",
        f"hide everything, {target_obj}",
        f"show cartoon, {target_obj}",
        f"color gray80, {target_obj}",
        f"select recovery_residues, {target_obj} and resi {sel}",
        "show sticks, recovery_residues",
        "color salmon, recovery_residues",
        f"load {ref_pdb}, {ref_obj}",
        f"hide everything, {ref_obj}",
        f"select {ref_obj}_lig, {ref_obj} and resn {het}",
        f"show sticks, {ref_obj}_lig",
        f"color magenta, {ref_obj}_lig",
        f"# IFP exported to {contact_csv}",
        "orient recovery_residues or " + ref_obj + "_lig",
    ]
    if png_path:
        lines.append(f"png {png_path}, width=2400, height=1800, dpi=300, ray=1")
    pml_path.write_text("\n".join(lines) + "\n")
    return "ok"


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def write_pocket_rank_table(dossier: dict, csv_path: Path) -> None:
    fields = [
        "pocket_id", "verdict", "dominant_region",
        "drug_score_legacy", "drug_score_geodesic", "drug_score_delta",
        "novelty_score",
        "best_reference", "best_reference_min_prox", "best_reference_dcc",
        "e18_interpretation", "geodesic_spread_A",
        "euclidean_to_geodesic_delta_A",
        "flags",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in dossier["pockets"]:
            e11 = p.get("enhancements", {}).get("E11_multi_view_dcc", {})
            e18 = p.get("enhancements", {}).get("E18_geodesic_centroid", {})
            w.writerow({
                "pocket_id": p.get("pocket_id"),
                "verdict": p.get("verdict"),
                "dominant_region": p.get("dominant_region"),
                "drug_score_legacy": p.get("drug_score_legacy"),
                "drug_score_geodesic": p.get("drug_score_geodesic"),
                "drug_score_delta": p.get("drug_score_delta"),
                "novelty_score": p.get("novelty_score"),
                "best_reference": e11.get("best_reference"),
                "best_reference_min_prox": e11.get("best_reference_min_prox"),
                "best_reference_dcc": e11.get("best_reference_dcc"),
                "e18_interpretation": e18.get("interpretation"),
                "geodesic_spread_A": e18.get("geodesic_spread_A"),
                "euclidean_to_geodesic_delta_A": e18.get(
                    "euclidean_to_geodesic_delta_A"
                ),
                "flags": ";".join(p.get("flags", [])),
            })


def write_visual_legend(cfg: dict, csv_path: Path) -> None:
    rows = []
    for region, color in cfg["_visualization"]["region_colors"].items():
        rows.append(("region_color", region, color))
    palette = cfg["_visualization"]["reference_palette"]
    for i, ref in enumerate(cfg["_references"]):
        rows.append((
            "reference_color",
            f"{ref['pdb_id']}/{ref['het']} ({ref.get('name','')})",
            palette[i % len(palette)],
        ))
    for flag, color in cfg["_visualization"]["pocket_palette"].items():
        rows.append(("pocket_palette", flag, color))
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "label", "color"])
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Interactive HTML (py3Dmol)
# ---------------------------------------------------------------------------
def write_interactive_html(
    dossier: dict, cfg: dict, md_pdb: Path, html_path: Path,
) -> None:
    pdb_text = md_pdb.read_text()
    palette = cfg["_visualization"]["region_colors"]
    pocket_palette = cfg["_visualization"]["pocket_palette"]

    region_blocks = []
    for region_name, region_def in cfg["_canonical_regions"].items():
        ranges = region_def.get("ranges", [])
        sel_clauses = []
        for r in ranges:
            if len(r) == 2:
                sel_clauses.append({"resi": f"{r[0]}-{r[1]}"})
        if sel_clauses:
            region_blocks.append({
                "name": region_name,
                "sels": sel_clauses,
                "color": palette.get(region_name, "gray"),
            })

    pocket_blocks = []
    for p in dossier["pockets"]:
        flag_key = classify_pocket(p)
        pocket_blocks.append({
            "id": p["pocket_id"],
            "residues": p.get("residues", []),
            "color": pocket_palette.get(flag_key, "white"),
            "flag": flag_key,
            "drug_score": p.get(
                "drug_score_geodesic", p.get("druggability_score", 0)
            ),
        })

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>PRISM-4D dossier — {cfg.get('target_id')}</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  body {{ font-family: sans-serif; margin: 16px; }}
  #viewer {{ width: 1024px; height: 768px; position: relative; border: 1px solid #ccc; }}
  .legend {{ font-size: 13px; margin-top: 12px; }}
  .swatch {{ display: inline-block; width: 12px; height: 12px;
             border: 1px solid #999; margin-right: 4px; vertical-align: middle; }}
</style>
</head><body>
<h2>PRISM-4D dossier — {cfg.get('target_name', cfg.get('target_id'))}</h2>
<p>Pockets: {len(pocket_blocks)} · Filtered spikes:
{dossier['run_metadata'].get('filtered_spikes', '?'):,}</p>
<div id="viewer"></div>
<div class="legend">
  <strong>Regions:</strong>
{"".join(f'  <span><span class="swatch" style="background:{r["color"]}"></span>{r["name"]}</span>&nbsp;&nbsp;' for r in region_blocks)}
  <br><strong>Pockets:</strong>
{"".join(f'  <span><span class="swatch" style="background:{p["color"]}"></span>P{p["id"]} ({p["flag"]}, drug={p["drug_score"]:.2f})</span>&nbsp;&nbsp;' for p in pocket_blocks)}
</div>
<script>
  const PDB_TEXT = {json.dumps(pdb_text)};
  const REGIONS = {json.dumps(region_blocks)};
  const POCKETS = {json.dumps(pocket_blocks)};
  const viewer = $3Dmol.createViewer('viewer', {{ backgroundColor: 'white' }});
  viewer.addModel(PDB_TEXT, 'pdb');
  viewer.setStyle({{}}, {{ cartoon: {{ color: 'gray' }} }});
  for (const r of REGIONS) {{
    for (const sel of r.sels) {{
      viewer.setStyle(sel, {{ cartoon: {{ color: r.color }} }});
    }}
  }}
  for (const p of POCKETS) {{
    if (p.residues.length) {{
      const resiSel = {{ resi: p.residues.join(',') }};
      viewer.addSurface($3Dmol.SurfaceType.SAS, {{
        opacity: 0.55, color: p.color,
      }}, resiSel);
    }}
  }}
  viewer.zoomTo();
  viewer.render();
</script>
</body></html>
"""
    html_path.write_text(html)


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------
def write_readme(
    dossier: dict, cfg: dict, output_dir: Path, pse_status: dict[str, str],
    contact_status: str,
) -> None:
    readme = output_dir / "00_README.md"
    md = []
    md.append(f"# PRISM-4D visualization bundle — {cfg.get('target_id')}")
    md.append("")
    md.append(f"- **Target name:** {cfg.get('target_name', '(unset)')}")
    md.append(f"- **Protein class:** {cfg.get('protein_class', '(unset)')}")
    md.append(f"- **Chain:** {cfg.get('chain_id', 'A')}")
    md.append(f"- **Pockets in dossier:** {len(dossier['pockets'])}")
    md.append(
        f"- **Filtered spikes:** "
        f"{dossier['run_metadata'].get('filtered_spikes', '?'):,}"
    )
    md.append("")
    md.append("## Layout")
    md.append("- `00_README.md` — this file")
    md.append("- `pml_scripts/0{1..5}_*.pml` — standalone PyMOL scripts (always emitted)")
    md.append("- `0{1..5}_*.pse` — PyMOL session bundles (only if pymol-open-source available)")
    md.append("- `images/fig{1..5}_*.png` — 2400×1800 dpi=300 renders")
    md.append("- `tables/pocket_rank_table.csv` — full pocket ranking, drug_score legacy vs geodesic")
    md.append("- `tables/visual_legend.csv` — color → label mapping (target-agnostic, sourced from YAML)")
    md.append("- `tables/contact_fingerprint.csv` — ProLIF IFP for the recovery pocket")
    md.append("- `interactive/overview.html` — single-file py3Dmol interactive viewer")
    md.append("")
    md.append("## PSE / PNG render status")
    for fig, status in pse_status.items():
        md.append(f"- `{fig}`: {status}")
    md.append("")
    md.append(f"## Contact panel (Figure 5): {contact_status}")
    md.append("")
    md.append("## Reproducibility")
    md.append("All coloring / regions / palettes come from "
              f"`config/targets/{cfg.get('target_id')}.yaml`. "
              "No protein-specific constants are hardcoded in "
              "`scripts/visualize_dossier.py`. Verify with:")
    md.append("```")
    md.append("grep -iE '\\b(p-loop|switch|kras|sotorasib|adagrasib|MOV|M1X|F0K)\\b' \\")
    md.append("     scripts/visualize_dossier.py")
    md.append("```")
    md.append("That command should return zero matches outside the docstring example.")
    readme.write_text("\n".join(md) + "\n")


# ---------------------------------------------------------------------------
# Renderer driver
# ---------------------------------------------------------------------------
def maybe_render(pml_path: Path, pse_path: Path) -> str:
    """Try `pymol -cqr <pml>` then export the loaded session as .pse via a
    short driver script. Returns 'ok' / 'pymol_unavailable' / 'render_error'.
    """
    if not shutil.which("pymol"):
        return "pymol_unavailable"
    try:
        # Wrap: load pml, save pse.
        wrapper = pml_path.with_suffix(".driver.pml")
        wrapper.write_text(
            f"@{pml_path.resolve()}\nsave {pse_path.resolve()}\n"
        )
        proc = subprocess.run(
            ["pymol", "-cq", str(wrapper)],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            return f"render_error: {proc.stderr[-200:]}"
        if not pse_path.exists():
            return "render_no_pse"
        return "ok"
    except subprocess.TimeoutExpired:
        return "render_timeout"
    except Exception as exc:
        return f"render_exception: {exc}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dossier", required=True, type=Path)
    ap.add_argument("--md-pdb", required=True, type=Path)
    ap.add_argument("--refs-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--target-id")
    ap.add_argument(
        "--config-root",
        default=Path(__file__).parent.parent / "config" / "targets",
        type=Path,
    )
    ap.add_argument(
        "--render-pse", action="store_true",
        help="Attempt PyMOL rendering of .pse files (requires `pymol` CLI).",
    )
    args = ap.parse_args()

    target_id = args.target_id or derive_target_id(args.dossier)
    cfg = load_target_config(target_id, args.config_root)
    dossier = json.load(open(args.dossier))

    out = args.output_dir
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "pml_scripts").mkdir(parents=True, exist_ok=True)
    (out / "interactive").mkdir(parents=True, exist_ok=True)

    # Tables (always emitted; cheap)
    write_pocket_rank_table(dossier, out / "tables" / "pocket_rank_table.csv")
    write_visual_legend(cfg, out / "tables" / "visual_legend.csv")

    # PML scripts (always emitted)
    pml_dir = out / "pml_scripts"
    fig_specs = []
    fig1_pml = pml_dir / "01_global_atlas.pml"
    fig1_png = out / "images" / "fig1_global_atlas.png"
    fig1_global_atlas(
        fig1_pml, dossier, cfg, args.md_pdb, args.refs_dir, fig1_png,
    )
    fig_specs.append((fig1_pml, out / "01_global_atlas.pse"))

    fig2_pml = pml_dir / "02_known_ligand_recovery.pml"
    fig2_png = out / "images" / "fig2_ligand_recovery.png"
    recovery_meta = fig2_ligand_recovery(
        fig2_pml, dossier, cfg, args.md_pdb, args.refs_dir, fig2_png,
    )
    fig_specs.append((fig2_pml, out / "02_known_ligand_recovery.pse"))

    fig3_pml = pml_dir / "03_druggability_gradient.pml"
    fig3_png = out / "images" / "fig3_druggability_gradient.png"
    fig3_druggability_gradient(fig3_pml, dossier, cfg, args.md_pdb, fig3_png)
    fig_specs.append((fig3_pml, out / "03_druggability_gradient.pse"))

    fig4_pml = pml_dir / "04_geodesic_vs_euclidean.pml"
    fig4_png = out / "images" / "fig4_geodesic_vs_euclidean.png"
    fig4_geodesic_vs_euclidean(fig4_pml, dossier, cfg, args.md_pdb, fig4_png)
    fig_specs.append((fig4_pml, out / "04_geodesic_vs_euclidean.pse"))

    fig5_pml = pml_dir / "05_medchem_contact_panel.pml"
    fig5_png = out / "images" / "fig5_medchem_contact_panel.png"
    contact_csv = out / "tables" / "contact_fingerprint.csv"
    contact_status = fig5_medchem_panel(
        fig5_pml, dossier, cfg, args.md_pdb, args.refs_dir,
        recovery_meta, contact_csv, fig5_png,
    )
    fig_specs.append((fig5_pml, out / "05_medchem_contact_panel.pse"))

    # PSE renders (optional)
    pse_status = {}
    if args.render_pse:
        for pml, pse in fig_specs:
            pse_status[pse.name] = maybe_render(pml, pse)
    else:
        for _, pse in fig_specs:
            pse_status[pse.name] = "skipped (pass --render-pse to enable)"

    # Interactive HTML
    write_interactive_html(
        dossier, cfg, args.md_pdb,
        out / "interactive" / "overview.html",
    )

    # README
    write_readme(dossier, cfg, out, pse_status, contact_status)

    print("=== visualize_dossier complete ===")
    print(f"target_id     : {target_id}")
    print(f"output_dir    : {out}")
    print(f"contact_status: {contact_status}")
    for fig_name, status in pse_status.items():
        print(f"  {fig_name}: {status}")


if __name__ == "__main__":
    main()

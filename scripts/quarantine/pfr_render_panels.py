#!/usr/bin/env python3
"""
PyMOL driver that renders publication-grade PFR panels directly (not via
the bundled generator, which suffers from label-overlap and viewport issues).

Run inside PyMOL as:
    pymol -cqr scripts/quarantine/pfr_render_panels.py -- \
        --manifest <path> --features <path> --nulls <path> --outdir <path>
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

from pymol import cmd
from pymol.cgo import ALPHA, COLOR, CONE, CYLINDER, SPHERE

Vec3 = Tuple[float, float, float]

FEATURE_RGB: Dict[str, Vec3] = {
    "hbond_donor":    (0.10, 0.30, 0.95),
    "hbond_acceptor": (0.92, 0.10, 0.12),
    "aromatic":       (0.95, 0.62, 0.10),
    "hydrophobic":    (0.10, 0.55, 0.22),
    "positive":       (0.55, 0.18, 0.86),
    "negative":       (0.95, 0.42, 0.05),
    "halogen":        (0.00, 0.58, 0.68),
    "default":        (0.30, 0.30, 0.30),
}

PROTEIN_BG    = (0.78, 0.82, 0.86)
SITE_RES_CLR  = (0.30, 0.32, 0.36)
LIG_CARBON    = (0.97, 0.97, 0.97)
NULL_GREY     = (0.62, 0.62, 0.62)
TOL_CYAN      = (0.05, 0.65, 0.78)


def vadd(a: Vec3, b: Vec3) -> Vec3:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])

def vmul(a: Vec3, s: float) -> Vec3:
    return (a[0]*s, a[1]*s, a[2]*s)

def vnorm(v: Vec3) -> Vec3:
    m = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
    if m < 1e-8:
        return (1.0, 0.0, 0.0)
    return (v[0]/m, v[1]/m, v[2]/m)


def clean(s: str) -> str:
    out = []
    for c in str(s):
        if c.isalnum() or c == "_":
            out.append(c)
        else:
            out.append("_")
    return "".join(out).strip("_") or "x"


def f(v, default=0.0) -> float:
    try:
        if v is None or str(v).strip() == "":
            return default
        return float(v)
    except Exception:
        return default


def boolish(v) -> bool:
    return str(v).strip().lower() in ("1","true","t","yes","y","hit","recovered")


def read_csv_rows(path: str) -> List[Dict[str,str]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8-sig") as h:
        return [dict(r) for r in csv.DictReader(h)]


def group(rows, key):
    g = defaultdict(list)
    for r in rows:
        g[r.get(key, "")].append(r)
    return g


def init_scene():
    cmd.reinitialize()
    cmd.bg_color("white")
    cmd.set("orthoscopic", "off")
    cmd.set("antialias", 2)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_shadows", "off")
    cmd.set("ray_opaque_background", "on")
    cmd.set("ambient", 0.40)
    cmd.set("direct", 0.55)
    cmd.set("spec_reflect", 0.18)
    cmd.set("cartoon_fancy_helices", "on")
    cmd.set("cartoon_smooth_loops", "on")
    cmd.set("cartoon_transparency", 0.55)
    cmd.set("stick_radius", 0.13)
    cmd.set("sphere_scale", 0.22)
    cmd.set("dash_radius", 0.04)
    cmd.set("dash_gap", 0.18)


def cgo_sphere(name, xyz, radius, rgb, alpha=0.92):
    cgo = [ALPHA, alpha, COLOR, rgb[0], rgb[1], rgb[2],
           SPHERE, xyz[0], xyz[1], xyz[2], radius]
    cmd.load_cgo(cgo, name)


def cgo_arrow(name, start, vec, rgb, length=1.55, radius=0.07, alpha=0.92):
    unit = vnorm(vec)
    end = vadd(start, vmul(unit, length))
    head_base = vadd(start, vmul(unit, length*0.74))
    head_r = max(radius*3.6, 0.16)
    cgo = [
        ALPHA, alpha,
        CYLINDER,
        start[0], start[1], start[2],
        head_base[0], head_base[1], head_base[2],
        radius,
        rgb[0], rgb[1], rgb[2], rgb[0], rgb[1], rgb[2],
        CONE,
        head_base[0], head_base[1], head_base[2],
        end[0], end[1], end[2],
        head_r, 0.0,
        rgb[0], rgb[1], rgb[2], rgb[0], rgb[1], rgb[2],
        1.0, 0.0,
    ]
    cmd.load_cgo(cgo, name)


def add_tolerance_geometry(name, start, vec, dist_A=3.5, angle_deg=30.0):
    unit = vnorm(vec)
    end = vadd(start, vmul(unit, dist_A))
    cone_r = math.tan(math.radians(angle_deg)) * dist_A
    cgo_sphere(f"{name}_sphere", start, dist_A, TOL_CYAN, alpha=0.07)
    cgo = [
        ALPHA, 0.18,
        CONE,
        start[0], start[1], start[2],
        end[0], end[1], end[2],
        0.0, cone_r,
        TOL_CYAN[0], TOL_CYAN[1], TOL_CYAN[2],
        TOL_CYAN[0], TOL_CYAN[1], TOL_CYAN[2],
        1.0, 1.0,
    ]
    cmd.load_cgo(cgo, f"{name}_cone")


def style_structure(obj, manifest):
    cmd.hide("everything", obj)
    prot_sel = f"{obj} and polymer.protein"
    lig_resn = manifest.get("ligand_selection", "")
    # Extract resname from ligand_selection like "X and resn 2AN and chain A"
    lig_atoms = f"({obj} and (not polymer)) and (not resn HOH)"
    if "resn " in lig_resn:
        # use specific resn
        parts = lig_resn.split()
        resn_idx = parts.index("resn")
        if resn_idx+1 < len(parts):
            resname = parts[resn_idx+1]
            lig_atoms = f"{obj} and resn {resname}"

    cmd.show("cartoon", prot_sel)
    cmd.color("0xC8CEDD", prot_sel)
    cmd.set("cartoon_transparency", 0.55, prot_sel)

    site_sel = f"({prot_sel}) within 6 of ({lig_atoms})"
    cmd.show("sticks", site_sel)
    cmd.color("0x4E5460", site_sel)
    cmd.set("stick_radius", 0.13, site_sel)

    # Ligand styling: thicker sticks, white carbons, color by element for others
    cmd.show("sticks", lig_atoms)
    cmd.color("0xF6F6F6", f"({lig_atoms}) and elem C")
    cmd.color("0xCC2A2A", f"({lig_atoms}) and elem O")
    cmd.color("0x2148E8", f"({lig_atoms}) and elem N")
    cmd.color("0xE8AA2A", f"({lig_atoms}) and elem S")
    cmd.color("0xC57E2A", f"({lig_atoms}) and elem P")
    cmd.color("0x29C25A", f"({lig_atoms}) and elem Cl")
    cmd.color("0x9229C2", f"({lig_atoms}) and elem Br")
    cmd.color("0xC929C2", f"({lig_atoms}) and elem F")
    cmd.set("stick_radius", 0.20, lig_atoms)
    # subtle halo
    cmd.show("nb_spheres", lig_atoms)

    return lig_atoms


def focus(obj, lig_sel, buffer=6.5):
    try:
        cmd.orient(lig_sel)
        cmd.zoom(lig_sel, buffer=buffer)
    except Exception:
        cmd.orient(obj)
        cmd.zoom(obj, buffer=8)


def feat_xyz(r): return (f(r.get("x")), f(r.get("y")), f(r.get("z")))
def feat_vec(r): return (f(r.get("vx"), 1.0), f(r.get("vy")), f(r.get("vz")))
def feat_rgb(r): return FEATURE_RGB.get(str(r.get("feature_type","")).strip().lower(),
                                        FEATURE_RGB["default"])


def draw_features(target, features, mode):
    """mode in {real, null, overlay_real, overlay_null}"""
    for i, r in enumerate(features, start=1):
        fid = f"feat_{target}_{mode}_{i:03d}"
        xyz = feat_xyz(r)
        vec = feat_vec(r)
        is_hit = boolish(r.get("hit",""))

        if mode in ("null", "overlay_null"):
            rgb = NULL_GREY
            alpha = 0.55 if mode == "null" else 0.35
            arrow_r = 0.13
            sph_r = 0.30
            length = 2.20
        elif is_hit:
            rgb = feat_rgb(r)
            alpha = 0.97
            arrow_r = 0.20
            sph_r = 0.55
            length = 3.20
        else:
            rgb = feat_rgb(r)
            alpha = 0.45
            arrow_r = 0.10
            sph_r = 0.28
            length = 2.00

        cgo_sphere(f"{fid}_orig", xyz, sph_r, rgb, alpha=max(alpha, 0.40))
        cgo_arrow(f"{fid}_arrow", xyz, vec, rgb, length=length, radius=arrow_r, alpha=alpha)


def draw_panel_text(target, manifest, mode):
    """Add corner title via wire-frame text plate (cgo_text) — using cmd.cgo_text-like trick:
    we just write to viewport with pseudoatom near the corner of the bbox.
    Simpler: don't write 3D text in PyMOL; the title goes into the filename + post-composite.
    """
    # Skip in-scene title to avoid label-clutter; we compose titles in matplotlib later.
    pass


def add_tolerance_for_first_hit(target, features):
    hits = [r for r in features if boolish(r.get("hit",""))]
    if not hits:
        return None
    r = hits[0]
    add_tolerance_geometry(f"tol_{target}", feat_xyz(r), feat_vec(r), 3.5, 30.0)
    return r


def render(outdir, target, mode, width, height, ray):
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"{clean(target)}__{clean(mode)}.png")
    if ray:
        cmd.ray(width, height)
        cmd.png(out, width=width, height=height, dpi=300, ray=0)
    else:
        cmd.png(out, width=width, height=height, dpi=300, ray=0)
    return out


def scene(manifest, features, nulls, mode, args, saved_view=None):
    target = manifest.get("target_id","x").strip()
    init_scene()
    pdb_path = manifest.get("pdb_path","")
    obj = clean(target)
    cmd.load(pdb_path, obj)

    # If the ligand_selection mentions a specific chain, drop all other chains
    # so multi-chain complexes (cGAS dimer, CRBN-DDB1) don't dominate the view.
    lig_sel_str = manifest.get("ligand_selection", "")
    if "chain " in lig_sel_str:
        parts = lig_sel_str.split()
        if "chain" in parts:
            ci = parts.index("chain")
            if ci + 1 < len(parts):
                keep_chain = parts[ci + 1].strip("()")
                # Remove other chains entirely
                cmd.remove(f"{obj} and not chain {keep_chain}")
                # And remove waters/ions
                cmd.remove(f"{obj} and resn HOH+WAT+DOD")

    lig_sel = style_structure(obj, manifest)

    if mode == "real":
        draw_features(target, features, "real")
    elif mode == "null":
        draw_features(target, nulls, "null")
    elif mode == "overlay":
        draw_features(target, nulls, "overlay_null")
        draw_features(target, features, "overlay_real")
    elif mode == "tolerance":
        add_tolerance_for_first_hit(target, features)
        # Also show all real features for context
        draw_features(target, features, "overlay_real")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if saved_view is not None:
        cmd.set_view(saved_view)
    else:
        focus(obj, lig_sel, buffer=f(manifest.get("zoom_buffer","6"), 6.5))

    out = render(args.outdir, target, mode, args.width, args.height, not args.no_ray)
    # Capture current view so subsequent modes can use the same camera
    return out, cmd.get_view()


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--nulls", default="")
    p.add_argument("--outdir", default="renders")
    p.add_argument("--targets", default="all")
    p.add_argument("--modes", nargs="+", default=["real","overlay","null","tolerance"])
    p.add_argument("--width", type=int, default=2400)
    p.add_argument("--height", type=int, default=1800)
    p.add_argument("--no-ray", action="store_true")
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    manifest_rows = read_csv_rows(args.manifest)
    feat_rows = read_csv_rows(args.features)
    null_rows = read_csv_rows(args.nulls)
    feats_by = group(feat_rows, "target_id")
    nulls_by = group(null_rows, "target_id")

    if args.targets.strip().lower() == "all":
        selected = {m["target_id"] for m in manifest_rows}
    else:
        selected = {x.strip() for x in args.targets.split(",") if x.strip()}

    outs = []
    for m in manifest_rows:
        tid = m["target_id"]
        if tid not in selected:
            continue
        features = feats_by.get(tid, [])
        nulls = nulls_by.get(tid, [])
        if not features:
            sys.stderr.write(f"WARN: no real features for {tid}\n")
            continue
        # First pass: real mode establishes the camera; subsequent modes reuse it
        # so real|null|overlay|tolerance for the same target share identical view
        saved_view = None
        for mode in args.modes:
            if mode in ("null","overlay") and not nulls:
                sys.stderr.write(f"WARN: no nulls for {tid}; skipping {mode}\n")
                continue
            out, view = scene(m, features, nulls, mode, args, saved_view=saved_view)
            if saved_view is None:
                saved_view = view
            outs.append(out)
            sys.stdout.write(f"Wrote {out}\n")
    return 0


if __name__ == "__main__":
    cli = sys.argv[1:]
    if "--" in cli:
        cli = cli[cli.index("--")+1:]
    raise SystemExit(main(cli))

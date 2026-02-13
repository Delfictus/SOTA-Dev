#!/usr/bin/env python3
"""
PRISM4D Post-Docking Analysis Pipeline
========================================
Automated interaction profiling, pharmacophore scoring, SOTA visualization,
and report generation for PRISM4D docking results.

Input:  Docking results (SDF), pharmacophore map (JSON), receptor (PDB)
Output: Interaction profiles, figures, comprehensive report

Pipeline:
  1. PLIP interaction profiling (H-bonds, pi-stacks, salt bridges, hydrophobic)
  2. Pharmacophore overlap scoring (docked atoms vs spike features)
  3. SOTA PyMOL figures (ray-traced, surface + interactions)
  4. 2D interaction diagram (matplotlib)
  5. Comprehensive markdown report

Usage:
    python scripts/post_dock_analysis.py \
        --receptor structure.pdb \
        --docking-dir results/docking/site0/ \
        --pharmacophore results/pharmacophore/interaction_map.json \
        --output results/analysis/

References:
    PLIP: Salentin et al. Nucleic Acids Res 2015. doi:10.1093/nar/gkv315
    PRISM4D spike detection: neuromorphic LIF + UV pump-probe + EFP
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from plip.structure.preparation import PDBComplex


# ── Interaction analysis with PLIP ───────────────────────────────────────

def merge_ligand_into_receptor(receptor_pdb, ligand_sdf, output_pdb, pose_index=0):
    """Merge a single ligand pose into the receptor PDB for PLIP analysis.

    PLIP requires a single PDB with both protein and ligand.
    We convert the SDF pose to PDB and append it.
    """
    OBABEL = shutil.which("obabel") or os.path.expanduser(
        "~/miniconda3/envs/prism_dock/bin/obabel")

    # Extract single pose from SDF
    from rdkit import Chem
    import warnings
    warnings.filterwarnings("ignore")

    suppl = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)
    mols = [m for m in suppl if m is not None]
    if pose_index >= len(mols):
        return None

    mol = mols[pose_index]

    # Write ligand as PDB
    tmp_lig = tempfile.NamedTemporaryFile(suffix=".pdb", delete=False)
    Chem.MolToPDBFile(mol, tmp_lig.name)

    # Read receptor lines (skip END)
    with open(receptor_pdb) as f:
        rec_lines = [l for l in f if not l.startswith("END")]

    # Read ligand lines, change to HETATM with chain L
    with open(tmp_lig.name) as f:
        lig_lines = []
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                # Force HETATM, chain L, resname LIG
                new_line = "HETATM" + line[6:17] + "LIG L   1" + line[26:]
                lig_lines.append(new_line)

    with open(output_pdb, "w") as f:
        for line in rec_lines:
            f.write(line)
        for line in lig_lines:
            f.write(line)
        f.write("END\n")

    os.unlink(tmp_lig.name)
    return output_pdb


def run_plip_analysis(complex_pdb):
    """Run PLIP on a protein-ligand complex PDB.

    Returns dict of interaction types and their details.
    Ref: Salentin et al. Nucleic Acids Res 2015. doi:10.1093/nar/gkv315
    """
    mol = PDBComplex()
    mol.load_pdb(str(complex_pdb))
    mol.analyze()

    interactions = {
        "hydrogen_bonds": [],
        "hydrophobic": [],
        "pi_stacking": [],
        "pi_cation": [],
        "salt_bridges": [],
        "water_bridges": [],
        "halogen_bonds": [],
    }

    for bsid, site in mol.interaction_sets.items():
        # Hydrogen bonds
        for hb in site.hbonds_pdon + site.hbonds_ldon:
            interactions["hydrogen_bonds"].append({
                "residue": f"{hb.restype}{hb.resnr}",
                "chain": hb.reschain,
                "distance": round(hb.distance_ah, 2),
                "angle": round(hb.angle, 1),
                "type": "protein_donor" if hb in site.hbonds_pdon else "ligand_donor",
                "donor_atom": hb.dtype,
                "acceptor_atom": hb.atype,
            })

        # Hydrophobic contacts
        for hp in site.hydrophobic_contacts:
            interactions["hydrophobic"].append({
                "residue": f"{hp.restype}{hp.resnr}",
                "chain": hp.reschain,
                "distance": round(hp.distance, 2),
            })

        # Pi-stacking
        for ps in site.pistacking:
            interactions["pi_stacking"].append({
                "residue": f"{ps.restype}{ps.resnr}",
                "chain": ps.reschain,
                "distance": round(ps.distance, 2),
                "angle": round(ps.angle, 1),
                "type": ps.type,
            })

        # Pi-cation
        for pc in site.pication_paro + site.pication_laro:
            interactions["pi_cation"].append({
                "residue": f"{pc.restype}{pc.resnr}",
                "chain": pc.reschain,
                "distance": round(pc.distance, 2),
            })

        # Salt bridges
        for sb in site.saltbridge_lneg + site.saltbridge_pneg:
            interactions["salt_bridges"].append({
                "residue": f"{sb.restype}{sb.resnr}",
                "chain": sb.reschain,
                "distance": round(sb.distance, 2),
            })

        # Halogen bonds
        for xb in site.halogen_bonds:
            interactions["halogen_bonds"].append({
                "residue": f"{xb.restype}{xb.resnr}",
                "chain": xb.reschain,
                "distance": round(xb.distance, 2),
            })

    # Summary counts
    summary = {k: len(v) for k, v in interactions.items()}
    total = sum(summary.values())

    return {
        "interactions": interactions,
        "summary": summary,
        "total_contacts": total,
        "contacting_residues": sorted(set(
            f"{c['chain']}:{c['residue']}"
            for cat in interactions.values()
            for c in cat
        )),
    }


# ── Pharmacophore overlap scoring ────────────────────────────────────────

def score_pharmacophore_overlap(ligand_sdf, pharmacophore_json, pose_index=0):
    """Score how well a docked pose overlaps with spike-derived pharmacophore features.

    Uses spatially sub-clustered positions from design_recommendations (not
    coarse per-type centroids) for accurate nearest-feature distance scoring.

    Returns overlap score (0-1, higher = better) and per-feature distances.
    """
    from rdkit import Chem
    import warnings
    warnings.filterwarnings("ignore")

    with open(pharmacophore_json) as f:
        pharma = json.load(f)

    suppl = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)
    mols = [m for m in suppl if m is not None]
    if pose_index >= len(mols):
        return 0, []

    mol = mols[pose_index]
    conf = mol.GetConformer()
    coords = np.array([
        [conf.GetAtomPosition(j).x, conf.GetAtomPosition(j).y, conf.GetAtomPosition(j).z]
        for j in range(mol.GetNumAtoms())
    ])

    # Use sub-clustered features from design_recommendations if available
    # These have spatial resolution (5A clustering) vs coarse per-type centroids
    recs = pharma.get("design_recommendations", [])
    coarse_features = pharma.get("pharmacophore_features", {})

    # Map region types to feature metadata from coarse features
    region_meta = {}
    for feat_type, feat_data in coarse_features.items():
        region_meta[feat_type] = {
            "feature": feat_data.get("feature", feat_type),
            "radius": feat_data.get("radius", 2.0),
            "intensity": feat_data.get("intensity", 1.0),
        }

    feature_scores = []

    if recs:
        # Score against each sub-clustered position
        for rec in recs:
            region = rec["region"]
            pos = np.array(rec["position"])
            meta = region_meta.get(region, {"feature": region, "radius": 2.0, "intensity": 1.0})
            radius = meta["radius"]

            dists = np.linalg.norm(coords - pos, axis=1)
            min_dist = float(dists.min())

            if min_dist <= radius:
                score = 1.0
            else:
                score = np.exp(-0.5 * ((min_dist - radius) / 2.0) ** 2)

            feature_scores.append({
                "type": region,
                "feature": meta["feature"],
                "position": rec["position"],
                "radius": radius,
                "min_distance": round(min_dist, 2),
                "overlap_score": round(score, 3),
                "priority": rec.get("priority", "MEDIUM"),
            })
    else:
        # Fallback to coarse centroids
        for feat_type, feat_data in coarse_features.items():
            pos = np.array(feat_data["position"])
            radius = feat_data["radius"]
            dists = np.linalg.norm(coords - pos, axis=1)
            min_dist = float(dists.min())
            if min_dist <= radius:
                score = 1.0
            else:
                score = np.exp(-0.5 * ((min_dist - radius) / 2.0) ** 2)
            feature_scores.append({
                "type": feat_type,
                "feature": feat_data["feature"],
                "position": feat_data["position"],
                "radius": radius,
                "min_distance": round(min_dist, 2),
                "overlap_score": round(score, 3),
                "priority": "MEDIUM",
            })

    # Overall score: average of all features, weighted by priority
    priority_weight = {"HIGH": 3.0, "MEDIUM": 1.0, "LOW": 0.5}
    total_weight = sum(priority_weight.get(f.get("priority", "MEDIUM"), 1.0) for f in feature_scores) or 1
    overall = sum(
        f["overlap_score"] * priority_weight.get(f.get("priority", "MEDIUM"), 1.0)
        for f in feature_scores
    ) / total_weight

    return round(overall, 3), feature_scores


# ── 2D Interaction Diagram ───────────────────────────────────────────────

def generate_2d_interaction_diagram(plip_results, ligand_name, output_path):
    """Generate a 2D interaction diagram showing residue contacts around the ligand.

    Creates a radial layout with the ligand at center and contacting residues
    arranged around it, colored by interaction type.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.axis('off')

    # Ligand at center
    ligand_circle = plt.Circle((0, 0), 0.8, color='#4ECDC4', ec='black', lw=2, zorder=10)
    ax.add_patch(ligand_circle)
    ax.text(0, 0, ligand_name, ha='center', va='center', fontsize=9, fontweight='bold', zorder=11)

    # Collect unique residues and their interaction types
    residue_interactions = defaultdict(set)
    for itype, contacts in plip_results["interactions"].items():
        for c in contacts:
            key = f"{c['chain']}:{c['residue']}"
            residue_interactions[key].add(itype)

    if not residue_interactions:
        ax.text(0, -2, "No interactions detected", ha='center', fontsize=12, color='gray')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        return

    # Color map for interaction types
    itype_colors = {
        "hydrogen_bonds": "#2ECC71",      # Green
        "hydrophobic": "#F39C12",          # Orange
        "pi_stacking": "#9B59B6",          # Purple
        "pi_cation": "#E74C3C",            # Red
        "salt_bridges": "#3498DB",         # Blue
        "halogen_bonds": "#1ABC9C",        # Teal
        "water_bridges": "#85C1E9",        # Light blue
    }

    itype_labels = {
        "hydrogen_bonds": "H-bond",
        "hydrophobic": "Hydrophobic",
        "pi_stacking": "Pi-stack",
        "pi_cation": "Pi-cation",
        "salt_bridges": "Salt bridge",
        "halogen_bonds": "Halogen bond",
        "water_bridges": "Water bridge",
    }

    n_residues = len(residue_interactions)
    radius = 3.5

    for i, (reskey, itypes) in enumerate(sorted(residue_interactions.items())):
        angle = 2 * math.pi * i / n_residues - math.pi / 2
        rx = radius * math.cos(angle)
        ry = radius * math.sin(angle)

        # Residue circle — color by primary interaction
        primary = sorted(itypes, key=lambda t: list(itype_colors.keys()).index(t))[0]
        color = itype_colors.get(primary, '#95A5A6')

        res_circle = plt.Circle((rx, ry), 0.6, color=color, ec='black', lw=1.5,
                                alpha=0.8, zorder=5)
        ax.add_patch(res_circle)

        # Residue label
        chain, resname = reskey.split(":")
        ax.text(rx, ry, f"{resname}\n({chain})", ha='center', va='center',
                fontsize=7, fontweight='bold', zorder=6)

        # Draw interaction lines
        for itype in itypes:
            lcolor = itype_colors.get(itype, 'gray')
            style = '--' if itype == "hydrogen_bonds" else '-'
            lw = 2 if itype in ("hydrogen_bonds", "salt_bridges") else 1.5
            ax.plot([0, rx], [0, ry], style, color=lcolor, lw=lw, alpha=0.6, zorder=2)

    # Legend
    legend_handles = []
    used_types = set()
    for itypes_set in residue_interactions.values():
        used_types.update(itypes_set)
    for itype in itype_colors:
        if itype in used_types:
            legend_handles.append(mpatches.Patch(
                color=itype_colors[itype], label=itype_labels.get(itype, itype)))

    ax.legend(handles=legend_handles, loc='upper right', fontsize=9,
              framealpha=0.9, edgecolor='black')

    ax.set_title(f"PRISM4D Interaction Map — {ligand_name}", fontsize=14, fontweight='bold', pad=20)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ── SOTA PyMOL Visualization ────────────────────────────────────────────

def generate_sota_pymol(receptor_pdb, ligand_sdfs, pharmacophore_features,
                        plip_results_dict, output_dir, site_centroid):
    """Generate publication-quality PyMOL script with:
    - Transparent pocket surface
    - Colored pharmacophore regions
    - Interaction dashed lines (H-bonds, pi-stacks)
    - Element-colored ligands
    - Dark background
    - Ray-traced output
    """
    pml_path = output_dir / "sota_visualization.pml"

    lines = [
        "# PRISM4D SOTA Docking Visualization",
        "# Auto-generated by post_dock_analysis.py",
        "",
        "reinitialize",
        "bg_color black",
        "set ray_trace_mode, 1",
        "set ray_shadows, 1",
        "set ray_trace_fog, 0",
        "set antialias, 2",
        "set cartoon_fancy_helices, 1",
        "set cartoon_smooth_loops, 1",
        "set cartoon_side_chain_helper, 1",
        "set depth_cue, 0",
        "set specular, 0.3",
        "set stick_radius, 0.15",
        "set sphere_mode, 9",
        "",
        "# ═══ RECEPTOR ═══",
        f"load {receptor_pdb}, receptor",
        "hide everything, receptor",
        "show cartoon, receptor",
        "color 0xFFCC99, receptor and chain A",
        "color 0x99CCFF, receptor and chain B",
        "# Fallback: if single chain, color white",
        "color white, receptor and not (chain A or chain B)",
        "",
    ]

    # Pocket surface (mesh + transparent solid)
    lines.extend([
        "# ═══ POCKET SURFACE ═══",
        f"select pocket_zone, receptor within 10 of ({site_centroid[0]:.1f},{site_centroid[1]:.1f},{site_centroid[2]:.1f})",
        "create pocket_surf, pocket_zone",
        "show surface, pocket_surf",
        "set surface_color, slate, pocket_surf",
        "set transparency, 0.72, pocket_surf",
        "hide cartoon, pocket_surf",
        "# Mesh overlay for depth perception",
        "create pocket_mesh, pocket_zone",
        "show mesh, pocket_mesh",
        "set mesh_color, purpleblue, pocket_mesh",
        "set mesh_width, 0.4",
        "hide cartoon, pocket_mesh",
        "hide surface, pocket_mesh",
        "",
    ])

    # Load each ligand with distinct coloring
    lig_colors = {
        0: ("cyan", "Cyan"),
        1: ("hotpink", "Hot Pink"),
        2: ("brightorange", "Orange"),
        3: ("limegreen", "Lime"),
    }

    for i, (lig_name, sdf_path) in enumerate(ligand_sdfs.items()):
        color, color_name = lig_colors.get(i, ("white", "White"))
        obj_name = lig_name.replace("-", "_").replace(" ", "_")
        lines.extend([
            f"# ═══ LIGAND: {lig_name} ({color_name}) ═══",
            f"load {sdf_path}, {obj_name}",
            f"show sticks, {obj_name}",
            f"color {color}, {obj_name} and elem C",
            f"set stick_radius, 0.18, {obj_name}",
            "# Element coloring for non-carbon",
            f"color red, {obj_name} and elem O",
            f"color blue, {obj_name} and elem N",
            f"color yellow, {obj_name} and elem S",
            f"color green, {obj_name} and elem Cl",
            f"color hotpink, {obj_name} and elem F",
            f"color purple, {obj_name} and elem I",
            f"set state, 1, {obj_name}",
            "",
        ])

        # Add PLIP interaction lines for this ligand
        if lig_name in plip_results_dict:
            plip = plip_results_dict[lig_name]
            hb_idx = 0

            for hb in plip["interactions"]["hydrogen_bonds"]:
                hb_idx += 1
                res = hb["residue"]
                chain = hb["chain"]
                dist = hb["distance"]
                lines.append(f"# H-bond: {obj_name} <-> {chain}:{res} ({dist}A)")
                lines.append(f"distance hb_{obj_name}_{hb_idx}, "
                             f"{obj_name}, chain {chain} and resi {res[-3:]} and name N+O+S, 3.5")

            if hb_idx > 0:
                hb_names = " ".join(f"hb_{obj_name}_{j}" for j in range(1, hb_idx + 1))
                lines.append(f"color green, hb_{obj_name}_*")
                lines.append(f"set dash_color, green, hb_{obj_name}_*")
                lines.append(f"set dash_width, 2.5, hb_{obj_name}_*")
                lines.append(f"set dash_gap, 0.3, hb_{obj_name}_*")
                lines.append(f"hide labels, hb_{obj_name}_*")
                lines.append("")

    # Pharmacophore features — use sub-clustered positions from design_recommendations
    lines.append("# ═══ PHARMACOPHORE FEATURES (sub-clustered) ═══")
    pharma_colors = {
        "BNZ": ("tv_orange", 1.4),
        "TYR": ("forest", 1.0),
        "UNK": ("gray50", 0.9),
        "ANION": ("marine", 0.7),
        "CATION": ("firebrick", 0.7),
        "TRP": ("violet", 1.0),
        "PHE": ("tv_yellow", 0.9),
        "SS": ("chocolate", 0.6),
    }

    # Get sub-clustered positions from the full pharmacophore data
    # (passed via pharmacophore_features which now includes design_recommendations)
    recs = pharmacophore_features.get("_design_recommendations", [])

    if recs:
        # Use sub-clustered features
        type_counts = defaultdict(int)
        type_groups = defaultdict(list)
        for rec in recs:
            region = rec["region"]
            pos = rec["position"]
            priority = rec.get("priority", "MEDIUM")
            color, base_scale = pharma_colors.get(region, ("gray50", 0.6))

            # Scale by priority
            scale = base_scale * (1.2 if priority == "HIGH" else 1.0)
            idx = type_counts[region]
            type_counts[region] += 1
            ph_name = f"ph_{region.lower()}_{idx}"
            type_groups[region].append(ph_name)

            lines.extend([
                f"pseudoatom {ph_name}, pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]",
                f"show spheres, {ph_name}",
                f"color {color}, {ph_name}",
                f"set sphere_scale, {scale:.1f}, {ph_name}",
                f"set sphere_transparency, 0.45, {ph_name}",
            ])

        # Create groups for toggling
        lines.append("")
        for region, names in type_groups.items():
            group_name = f"pharma_{region.lower()}"
            lines.append(f"group {group_name}, {' '.join(names)}")
        all_groups = " ".join(f"pharma_{r.lower()}" for r in type_groups)
        lines.append(f"group pharmacophore, {all_groups}")
    else:
        # Fallback: use coarse per-type centroids
        ph_idx = 0
        for feat_type, feat_data in pharmacophore_features.items():
            if feat_type.startswith("_"):
                continue
            pos = feat_data["position"]
            color, base_scale = pharma_colors.get(feat_type, ("gray50", 0.6))
            ph_name = f"ph_{feat_type}_{ph_idx}"
            ph_idx += 1
            lines.extend([
                f"pseudoatom {ph_name}, pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]",
                f"show spheres, {ph_name}",
                f"color {color}, {ph_name}",
                f"set sphere_scale, {base_scale}, {ph_name}",
                f"set sphere_transparency, 0.45, {ph_name}",
            ])

    lines.extend([
        "",
        "# ═══ CAMERA ═══",
        f"pseudoatom _c, pos=[{site_centroid[0]:.1f},{site_centroid[1]:.1f},{site_centroid[2]:.1f}]",
        "zoom _c, 18",
        "turn y, 20",
        "turn x, -15",
        "delete _c",
        "",
        "# ═══ CLEANUP ═══",
        "deselect",
        "set all_states, 0",
        "set state, 1",
        "",
        "# ═══ RAY TRACE ═══",
        "# Uncomment to render:",
        "# ray 2400, 1800",
        "# png sota_figure.png, dpi=300",
    ])

    with open(pml_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return pml_path


def ray_trace_figure(pml_path, output_png, width=2400, height=1800):
    """Run PyMOL headless to ray-trace a publication figure."""
    # Append ray + png commands
    render_pml = str(pml_path).replace(".pml", "_render.pml")
    with open(pml_path) as f:
        content = f.read()

    content = content.replace("# ray 2400, 1800", f"ray {width}, {height}")
    content = content.replace("# png sota_figure.png, dpi=300",
                              f"png {output_png}, dpi=300")
    content += "\nquit\n"

    with open(render_pml, "w") as f:
        f.write(content)

    result = subprocess.run(
        ["pymol", "-cq", "-d", f"run {render_pml}"],
        capture_output=True, text=True, timeout=120,
        cwd=str(Path(pml_path).parent.parent.parent.parent)
    )

    if Path(output_png).exists():
        print(f"  Figure: {output_png} ({Path(output_png).stat().st_size // 1024} KB)")
        return True
    else:
        print(f"  Ray trace failed: {result.stderr[:200]}")
        return False


# ── Report Generation ────────────────────────────────────────────────────

def generate_report(receptor_name, ligand_results, pharmacophore_data,
                    site_info, output_path):
    """Generate comprehensive markdown analysis report."""
    with open(output_path, "w") as f:
        f.write("# PRISM4D Docking Analysis Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Receptor**: `{receptor_name}`\n")
        f.write(f"**Pipeline**: PRISM4D Detection + UniDock GPU + GNINA CNN + PLIP\n\n")

        # Site summary
        f.write("## Binding Site\n\n")
        f.write(f"| Property | Value |\n|---|---|\n")
        f.write(f"| Classification | {site_info.get('classification', 'Unknown')} |\n")
        centroid = site_info.get('centroid', [0, 0, 0])
        f.write(f"| Centroid | ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f}) |\n")
        f.write(f"| Volume | {site_info.get('volume', 0):.0f} A^3 |\n")
        f.write(f"| Druggability | {site_info.get('druggability', 0):.1%} |\n")
        f.write(f"| Pocket Character | {pharmacophore_data.get('pocket_character', '?')} |\n")
        f.write(f"| Aromatic | {pharmacophore_data.get('aromatic_percentage', 0):.1f}% |\n")
        f.write(f"| Polar | {pharmacophore_data.get('polar_percentage', 0):.1f}% |\n\n")

        # Per-ligand results
        f.write("## Ligand Docking Results\n\n")

        for lig_name, result in ligand_results.items():
            f.write(f"### {lig_name}\n\n")

            # Scores
            f.write(f"| Metric | Value |\n|---|---|\n")
            f.write(f"| Vina Score | {result.get('vina_score', '?')} kcal/mol |\n")
            f.write(f"| CNN Affinity | {result.get('cnn_affinity', '?')} |\n")
            f.write(f"| CNN Pose Score | {result.get('cnn_pose_score', '?')} |\n")
            f.write(f"| Pharmacophore Overlap | {result.get('pharma_overlap', 0):.1%} |\n")
            f.write(f"| Total Contacts | {result.get('total_contacts', 0)} |\n\n")

            # Interaction breakdown
            plip = result.get("plip", {})
            summary = plip.get("summary", {})
            if any(v > 0 for v in summary.values()):
                f.write("**Interactions:**\n\n")
                f.write("| Type | Count | Residues |\n|---|---|---|\n")
                for itype, count in summary.items():
                    if count > 0:
                        residues = ", ".join(
                            f"{c['chain']}:{c['residue']}({c['distance']}A)"
                            for c in plip.get("interactions", {}).get(itype, [])
                        )
                        label = itype.replace("_", " ").title()
                        f.write(f"| {label} | {count} | {residues} |\n")
                f.write("\n")

            # Pharmacophore feature matches — show best match per type
            feat_scores = result.get("pharma_features", [])
            if feat_scores:
                # Group by type, keep nearest per type
                best_per_type = {}
                for fs in feat_scores:
                    t = fs["type"]
                    if t not in best_per_type or fs["min_distance"] < best_per_type[t]["min_distance"]:
                        best_per_type[t] = fs

                f.write("**Pharmacophore Match (nearest sub-cluster per type):**\n\n")
                f.write("| Feature | Type | Nearest (A) | Score | Priority |\n|---|---|---|---|---|\n")
                for t in ["BNZ", "TYR", "UNK", "ANION", "CATION"]:
                    if t in best_per_type:
                        fs = best_per_type[t]
                        f.write(f"| {fs['feature']} | {fs['type']} | "
                                f"{fs['min_distance']} | {fs['overlap_score']:.2f} | "
                                f"{fs.get('priority', '-')} |\n")
                f.write("\n")

                # Also show all HIGH priority features
                high_feats = [fs for fs in feat_scores if fs.get("priority") == "HIGH"]
                if high_feats:
                    f.write("**HIGH-priority hotspot distances:**\n\n")
                    f.write("| Type | Position | Distance (A) | Score |\n|---|---|---|---|\n")
                    for fs in sorted(high_feats, key=lambda x: x["min_distance"]):
                        pos = fs["position"]
                        f.write(f"| {fs['type']} | ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) | "
                                f"{fs['min_distance']} | {fs['overlap_score']:.2f} |\n")
                    f.write("\n")

            # Contacting residues
            contacting = plip.get("contacting_residues", [])
            if contacting:
                f.write(f"**Contacting Residues** ({len(contacting)}): "
                        f"{', '.join(contacting)}\n\n")

        # Design recommendations
        recs = pharmacophore_data.get("design_recommendations", [])
        if recs:
            f.write("## Design Recommendations\n\n")
            f.write("Based on spike-derived pharmacophore analysis:\n\n")
            for rec in recs:
                f.write(f"- **{rec['priority']}** @ ({rec['position'][0]:.1f}, "
                        f"{rec['position'][1]:.1f}, {rec['position'][2]:.1f}): "
                        f"{rec['action']}\n")
                f.write(f"  - Fragments: {rec['fragments']}\n")
                f.write(f"  - Rationale: {rec['rationale']}\n")
            f.write("\n")

        # Visualization
        f.write("## Visualization\n\n")
        f.write("```bash\n")
        f.write(f"cd /home/diddy/Desktop/Prism4D-bio\n")
        f.write(f"pymol -d \"run {output_path.parent / 'sota_visualization.pml'}\"\n")
        f.write("```\n\n")

        # References
        f.write("## References\n\n")
        f.write("- PLIP: Salentin et al. Nucleic Acids Res 2015. doi:10.1093/nar/gkv315\n")
        f.write("- UniDock: Yu et al. JCTC 2023. doi:10.1021/acs.jctc.2c01145\n")
        f.write("- GNINA 1.3: McNutt et al. J Cheminform 2025. doi:10.1186/s13321-025-00973-x\n")
        f.write("- PRISM4D: GPU-accelerated spike detection (UV pump-probe + LIF + EFP)\n\n")
        f.write("---\n")
        f.write(f"*Generated by PRISM4D Post-Docking Analysis Pipeline v1.0*\n")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PRISM4D Post-Docking Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--receptor", required=True, help="Receptor PDB")
    parser.add_argument("--docking-dir", required=True,
                        help="Docking output dir (contains gnina_rescore/)")
    parser.add_argument("--pharmacophore", required=True,
                        help="Pharmacophore interaction_map.json")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--ray-trace", action="store_true",
                        help="Ray-trace publication figures (slower)")

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PRISM4D Post-Docking Analysis Pipeline")
    print("=" * 60)

    # Load pharmacophore data
    print("\n[1/5] Loading pharmacophore data...")
    with open(args.pharmacophore) as f:
        pharma_data = json.load(f)
    site_centroid = pharma_data["centroid"]
    print(f"  Site centroid: ({site_centroid[0]:.1f}, {site_centroid[1]:.1f}, {site_centroid[2]:.1f})")
    print(f"  Pocket character: {pharma_data.get('pocket_character', '?')}")

    # Find docked SDF files
    dock_dir = Path(args.docking_dir)
    gnina_dir = dock_dir / "gnina_rescore"
    if not gnina_dir.exists():
        gnina_dir = dock_dir

    sdf_files = sorted(gnina_dir.glob("*_rescored.sdf"))
    if not sdf_files:
        sdf_files = sorted(gnina_dir.glob("*.sdf"))
    if not sdf_files:
        print("ERROR: No docked SDF files found")
        sys.exit(1)

    print(f"  Found {len(sdf_files)} docked ligand files")

    # Run analysis for each ligand
    print("\n[2/5] Running PLIP interaction profiling...")
    ligand_results = {}
    ligand_sdfs = {}
    plip_results_dict = {}

    for sdf_path in sdf_files:
        lig_name = sdf_path.stem.replace("_rescored", "").replace("_out", "")
        print(f"\n  --- {lig_name} ---")

        ligand_sdfs[lig_name] = str(sdf_path)

        # Get best pose scores from SDF
        from rdkit import Chem
        import warnings
        warnings.filterwarnings("ignore")
        suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False)
        mols = [m for m in suppl if m is not None]
        if not mols:
            print(f"  WARNING: No valid poses in {sdf_path.name}")
            continue

        mol = mols[0]  # Best pose
        vina = mol.GetProp('minimizedAffinity') if mol.HasProp('minimizedAffinity') else '?'
        cnn_aff = mol.GetProp('CNNaffinity') if mol.HasProp('CNNaffinity') else '?'
        cnn_score = mol.GetProp('CNNscore') if mol.HasProp('CNNscore') else '?'

        # Merge ligand into receptor for PLIP
        complex_pdb = output_dir / f"{lig_name}_complex.pdb"
        merge_ligand_into_receptor(args.receptor, sdf_path, complex_pdb, pose_index=0)

        # Run PLIP
        try:
            plip = run_plip_analysis(complex_pdb)
            plip_results_dict[lig_name] = plip
            print(f"  PLIP: {plip['total_contacts']} contacts "
                  f"({plip['summary'].get('hydrogen_bonds', 0)} H-bonds, "
                  f"{plip['summary'].get('hydrophobic', 0)} hydrophobic, "
                  f"{plip['summary'].get('pi_stacking', 0)} pi-stacks)")
        except Exception as e:
            print(f"  PLIP error: {e}")
            plip = {"interactions": {}, "summary": {}, "total_contacts": 0, "contacting_residues": []}
            plip_results_dict[lig_name] = plip

        ligand_results[lig_name] = {
            "vina_score": vina,
            "cnn_affinity": cnn_aff,
            "cnn_pose_score": cnn_score,
            "plip": plip,
            "total_contacts": plip["total_contacts"],
        }

    # Pharmacophore overlap scoring
    print("\n[3/5] Scoring pharmacophore overlap...")
    for lig_name, sdf_path in ligand_sdfs.items():
        overlap, feat_scores = score_pharmacophore_overlap(
            sdf_path, args.pharmacophore, pose_index=0
        )
        ligand_results[lig_name]["pharma_overlap"] = overlap
        ligand_results[lig_name]["pharma_features"] = feat_scores
        print(f"  {lig_name}: pharmacophore overlap = {overlap:.1%}")

    # 2D interaction diagrams
    print("\n[4/5] Generating interaction diagrams...")
    for lig_name in ligand_results:
        if lig_name in plip_results_dict:
            diagram_path = output_dir / f"{lig_name}_interactions.png"
            generate_2d_interaction_diagram(
                plip_results_dict[lig_name], lig_name, diagram_path
            )
            print(f"  {diagram_path}")

    # Load site info from docking results
    site_json = dock_dir / "site0_docking_results.json"
    site_info = {}
    if site_json.exists():
        with open(site_json) as f:
            dock_data = json.load(f)
            site_info = dock_data.get("site", {})

    # SOTA PyMOL visualization
    print("\n[5/5] Generating SOTA visualization...")
    pharma_features = dict(pharma_data.get("pharmacophore_features", {}))
    # Pass sub-clustered positions through for the viz generator
    pharma_features["_design_recommendations"] = pharma_data.get("design_recommendations", [])
    pml_path = generate_sota_pymol(
        args.receptor, ligand_sdfs, pharma_features,
        plip_results_dict, output_dir, site_centroid
    )
    print(f"  PyMOL script: {pml_path}")

    # Ray trace if requested
    if args.ray_trace:
        print("\n  Ray tracing publication figure...")
        output_png = output_dir / "sota_figure.png"
        ray_trace_figure(pml_path, str(output_png))

    # Generate report
    print("\n  Generating analysis report...")
    report_path = output_dir / "analysis_report.md"
    generate_report(
        Path(args.receptor).name, ligand_results, pharma_data,
        site_info, report_path
    )
    print(f"  Report: {report_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    for lig_name, result in ligand_results.items():
        print(f"\n  {lig_name}:")
        print(f"    Vina: {result['vina_score']} kcal/mol")
        print(f"    CNN Affinity: {result['cnn_affinity']}")
        print(f"    Contacts: {result['total_contacts']}")
        print(f"    Pharmacophore Match: {result['pharma_overlap']:.1%}")

    print(f"\nVisualize:")
    print(f"  cd /home/diddy/Desktop/Prism4D-bio")
    print(f"  pymol -d \"run {pml_path}\"")
    print("=" * 60)


if __name__ == "__main__":
    main()

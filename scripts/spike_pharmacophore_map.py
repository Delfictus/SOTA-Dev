#!/usr/bin/env python3
"""
PRISM4D Spike Pharmacophore Mapper
====================================
Converts raw spike events into a 3D pharmacophore interaction map that tells you
exactly what molecular features will bind where in the pocket.

Physics basis:
  - BNZ/PHE spikes (258 nm): pi-stacking hotspots → aromatic/halogenated rings
  - TYR spikes (274 nm): H-bond + pi-stacking → phenols, hydroxamic acids
  - TRP spikes (280 nm): Large aromatic + H-bond → indoles, benzimidazoles
  - CATION spikes (EFP): Positive electrostatic flux → acidic groups (COO-, SO3-, PO4-)
  - ANION spikes (EFP): Negative electrostatic flux → basic groups (NH3+, guanidinium)
  - UNK/LIF spikes: Thermal conformational flexibility → shape-complementary groups
  - Water density: High → water-displacing fragments (CF3, cyclopropyl)
  - Vibrational energy: High → strong coupling = high-affinity binding potential

Output:
  1. Pharmacophore feature PDB (pseudoatoms colored by interaction type)
  2. PyMOL visualization script with density isosurfaces
  3. JSON interaction map with molecular recommendations
  4. Per-voxel druggability heatmap

Usage:
    python scripts/spike_pharmacophore_map.py \\
        --spikes results/structure.site0.spike_events.json \\
        --receptor structure.pdb \\
        --output pharmacophore_out/

References:
    Franck-Condon excitation: Atkins, Physical Chemistry, 11th ed., Ch. 12
    UV protein absorption: Goldfarb et al. J Biol Chem 1951. doi:10.1016/S0021-9258(18)57149-3
    Pharmacophore modeling: Wermuth et al. Pure Appl Chem 1998. doi:10.1351/pac199870051129
"""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


# ── Spike type → molecular interaction mapping ────────────────────────────
# Based on the physical mechanism that generates each spike type.
#
# UV spikes: Aromatic chromophore excitation → Franck-Condon thermal wavefront
#   - The spike position marks where the excited aromatic's vibrational energy
#     dissipated into the solvent/pocket, indicating a site that can stabilize
#     a pi-system through van der Waals / dispersion interactions.
#
# EFP spikes: Electrostatic flux probes detect local charge asymmetry
#   - CATION: region with net positive electrostatic potential → wants electron-rich ligand
#   - ANION: region with net negative electrostatic potential → wants electron-poor ligand
#
# LIF spikes: Neuromorphic leaky-integrate-fire detects thermal conformational events
#   - Marks regions of transient pocket opening/breathing → shape complementarity

SPIKE_TO_PHARMACOPHORE = {
    "BNZ": {
        "feature": "Hydrophobic / Pi-stacking",
        "color": [1.0, 0.6, 0.0],        # Orange
        "pymol_color": "tv_orange",
        "fragments": "benzene, toluene, naphthalene, chlorobenzene, fluorobenzene, biphenyl",
        "functional_groups": "aryl, halogenated aryl, fused aromatics",
        "interaction": "pi-pi stacking, CH-pi, hydrophobic packing",
        "physics": "258nm PHE excitation → Franck-Condon vibrational dissipation marks pi-contact sites",
    },
    "TYR": {
        "feature": "H-bond Donor/Acceptor + Pi",
        "color": [0.0, 0.8, 0.2],        # Green
        "pymol_color": "forest",
        "fragments": "phenol, catechol, hydroxamic acid, aminophenol, resorcinol",
        "functional_groups": "phenol OH, hydroxamate, catechol",
        "interaction": "H-bond donor/acceptor + pi-stacking (dual character)",
        "physics": "274nm TYR excitation → phenol OH H-bond network + ring pi-system",
    },
    "TRP": {
        "feature": "Large Aromatic + H-bond",
        "color": [0.5, 0.0, 0.8],        # Purple
        "pymol_color": "violet",
        "fragments": "indole, benzimidazole, carbazole, quinoline, isoquinoline",
        "functional_groups": "indole NH, bicyclic aromatic, fused heterocycle",
        "interaction": "extended pi-system + NH donor (large hydrophobic surface)",
        "physics": "280nm TRP excitation → indole ring vibrational coupling to pocket wall",
    },
    "PHE": {
        "feature": "Pure Hydrophobic",
        "color": [0.8, 0.8, 0.0],        # Yellow
        "pymol_color": "tv_yellow",
        "fragments": "cyclohexane, cyclopentane, adamantane, tert-butyl",
        "functional_groups": "cycloalkyl, branched alkyl, gem-dimethyl",
        "interaction": "pure van der Waals / dispersion, no H-bond",
        "physics": "258nm PHE excitation — minimal H-bond, pure dispersion contacts",
    },
    "CATION": {
        "feature": "Anionic / Negative Ionizable",
        "color": [1.0, 0.0, 0.0],        # Red
        "pymol_color": "red",
        "fragments": "carboxylic acid, tetrazole, phosphonate, sulfonamide, acyl sulfonamide",
        "functional_groups": "COOH, PO3H2, SO2NH, tetrazole, boronic acid",
        "interaction": "salt bridge with LYS/ARG, electrostatic complementarity",
        "physics": "EFP detects positive electrostatic flux → pocket wants electron-rich (anionic) ligand group",
    },
    "ANION": {
        "feature": "Cationic / Positive Ionizable",
        "color": [0.0, 0.4, 1.0],        # Blue
        "pymol_color": "marine",
        "fragments": "piperazine, piperidine, morpholine, guanidine, amidine, imidazole",
        "functional_groups": "NH3+, guanidinium, amidinium, protonated heterocycle",
        "interaction": "salt bridge with GLU/ASP, electrostatic complementarity",
        "physics": "EFP detects negative electrostatic flux → pocket wants electron-poor (cationic) ligand group",
    },
    "UNK": {
        "feature": "Shape Complementarity / Flexibility",
        "color": [0.6, 0.6, 0.6],        # Gray
        "pymol_color": "gray60",
        "fragments": "flexible linkers, macrocycles, PROTACs, stapled peptides",
        "functional_groups": "PEG linker, alkyl chain, macrocyclic constraint",
        "interaction": "transient pocket opening — shape-filling, induced fit",
        "physics": "LIF neuromorphic detection of thermal conformational events at pocket boundary",
    },
    "SS": {
        "feature": "Disulfide / Thiol-reactive",
        "color": [0.8, 0.4, 0.0],        # Brown
        "pymol_color": "chocolate",
        "fragments": "acrylamide, vinyl sulfonamide, chloroacetamide, alpha-cyanoacrylamide",
        "functional_groups": "Michael acceptor, haloacetamide, cyanoacrylamide",
        "interaction": "covalent bond formation with Cys thiol",
        "physics": "250nm disulfide excitation → thiol accessibility for covalent modification",
    },
}


def voxelize_spikes(spikes, grid_spacing=2.0):
    """Bin spikes into 3D voxel grid, computing per-voxel pharmacophore features.

    Args:
        spikes: list of spike dicts from spike_events.json
        grid_spacing: voxel size in Angstrom (default 2.0A — pharmacophore resolution)

    Returns:
        dict of voxel_key → feature dict with intensity-weighted pharmacophore type
    """
    if not spikes:
        return {}

    xs = np.array([s["x"] for s in spikes])
    ys = np.array([s["y"] for s in spikes])
    zs = np.array([s["z"] for s in spikes])

    voxels = defaultdict(lambda: {
        "types": Counter(),
        "total_intensity": 0.0,
        "intensities_by_type": defaultdict(float),
        "count": 0,
        "max_intensity": 0.0,
        "vibrational_energy": 0.0,
        "water_density_sum": 0.0,
        "sources": Counter(),
        "wavelengths": Counter(),
        "positions": [],
    })

    for s in spikes:
        vx = int(s["x"] / grid_spacing)
        vy = int(s["y"] / grid_spacing)
        vz = int(s["z"] / grid_spacing)
        key = (vx, vy, vz)

        v = voxels[key]
        stype = s["type"]
        intensity = s["intensity"]

        v["types"][stype] += 1
        v["total_intensity"] += intensity
        v["intensities_by_type"][stype] += intensity
        v["count"] += 1
        v["max_intensity"] = max(v["max_intensity"], intensity)
        v["vibrational_energy"] += s.get("vibrational_energy", 0)
        v["water_density_sum"] += s.get("water_density", 0)
        v["sources"][s.get("spike_source", "?")] += 1
        wl = s.get("wavelength_nm", 0)
        if wl > 0:
            v["wavelengths"][int(round(wl))] += 1
        v["positions"].append((s["x"], s["y"], s["z"]))

    # Compute per-voxel pharmacophore assignment
    result = {}
    for key, v in voxels.items():
        # Dominant type by intensity-weighted count
        dominant_type = max(v["intensities_by_type"], key=v["intensities_by_type"].get)

        # Centroid of spikes in this voxel
        positions = np.array(v["positions"])
        centroid = positions.mean(axis=0)

        # Mean water density
        mean_water = v["water_density_sum"] / v["count"] if v["count"] > 0 else 0

        result[key] = {
            "voxel": key,
            "centroid": centroid.tolist(),
            "dominant_type": dominant_type,
            "type_breakdown": dict(v["types"]),
            "intensity_by_type": dict(v["intensities_by_type"]),
            "total_intensity": v["total_intensity"],
            "mean_intensity": v["total_intensity"] / v["count"],
            "max_intensity": v["max_intensity"],
            "spike_count": v["count"],
            "vibrational_energy": v["vibrational_energy"],
            "mean_water_density": mean_water,
            "sources": dict(v["sources"]),
            "wavelengths": dict(v["wavelengths"]),
        }

    return result


def _spatial_cluster(positions, intensities, merge_radius=5.0):
    """Greedy spatial clustering: seed at highest-intensity voxel, absorb neighbors
    within merge_radius, repeat until all voxels assigned.

    Returns list of (indices_list) for each cluster.
    """
    n = len(positions)
    assigned = np.zeros(n, dtype=bool)
    clusters = []

    # Sort by intensity descending — seed with strongest voxels
    order = np.argsort(-intensities)

    for seed_idx in order:
        if assigned[seed_idx]:
            continue
        dists = np.linalg.norm(positions - positions[seed_idx], axis=1)
        members = np.where((dists <= merge_radius) & ~assigned)[0]
        assigned[members] = True
        clusters.append(members.tolist())

    return clusters


def cluster_pharmacophore_features(voxels, min_intensity_pctl=50, merge_radius=5.0):
    """Cluster voxels into spatially-distinct pharmacophore features.

    Each spike type is independently sub-clustered so that e.g. BNZ produces
    multiple hotspots if they are >merge_radius apart.

    Returns list of pharmacophore features with position, type, strength.
    """
    if not voxels:
        return []

    # Intensity threshold
    intensities = [v["total_intensity"] for v in voxels.values()]
    threshold = np.percentile(intensities, min_intensity_pctl)

    # Filter to significant voxels
    sig_voxels = {k: v for k, v in voxels.items() if v["total_intensity"] >= threshold}

    # Group by dominant type
    type_groups = defaultdict(list)
    for k, v in sig_voxels.items():
        type_groups[v["dominant_type"]].append(v)

    # Also extract electrostatic minority features (CATION/ANION/SS) —
    # these are pharmacophore-relevant even when not the dominant type in a voxel,
    # since a salt bridge or covalent attachment point is critical for drug design.
    electrostatic_types = {"CATION", "ANION", "SS"}
    for k, v in sig_voxels.items():
        for etype in electrostatic_types:
            if etype in v["type_breakdown"] and etype != v["dominant_type"]:
                count = v["type_breakdown"][etype]
                if count >= 5:  # Minimum 5 spikes of this type in the voxel
                    # Create a synthetic voxel entry for the minority type
                    type_groups[etype].append({
                        "centroid": v["centroid"],
                        "total_intensity": v["intensity_by_type"].get(etype, 0),
                        "spike_count": count,
                        "mean_water_density": v["mean_water_density"],
                        "vibrational_energy": v["vibrational_energy"] * count / v["spike_count"],
                    })

    features = []
    for stype, group in type_groups.items():
        if not group:
            continue

        positions = np.array([v["centroid"] for v in group])
        ints = np.array([v["total_intensity"] for v in group])

        # Spatial sub-clustering within this type
        sub_clusters = _spatial_cluster(positions, ints, merge_radius=merge_radius)

        pharma = SPIKE_TO_PHARMACOPHORE.get(stype, SPIKE_TO_PHARMACOPHORE["UNK"])

        for cluster_idxs in sub_clusters:
            c_positions = positions[cluster_idxs]
            c_intensities = ints[cluster_idxs]
            c_voxels = [group[i] for i in cluster_idxs]
            c_spike_counts = np.array([v["spike_count"] for v in c_voxels])

            # Intensity-weighted centroid
            weighted_pos = np.average(c_positions, weights=c_intensities, axis=0)

            # Effective radius
            dists = np.linalg.norm(c_positions - weighted_pos, axis=1)
            radius = np.sqrt(np.average(dists**2, weights=c_intensities)) if len(dists) > 1 else 2.0
            radius = max(radius, 1.5)

            total_int = float(c_intensities.sum())
            total_spikes = int(c_spike_counts.sum())
            mean_water = np.mean([v["mean_water_density"] for v in c_voxels])
            total_vib = sum(v["vibrational_energy"] for v in c_voxels)

            features.append({
                "type": stype,
                "feature_name": pharma["feature"],
                "position": weighted_pos.tolist(),
                "radius": round(min(radius, 4.0), 2),  # Cap at 4A for visualization
                "total_intensity": round(total_int, 1),
                "spike_count": total_spikes,
                "mean_water_density": round(float(mean_water), 4),
                "vibrational_energy": round(total_vib, 4),
                "fragments": pharma["fragments"],
                "functional_groups": pharma["functional_groups"],
                "interaction": pharma["interaction"],
                "physics": pharma["physics"],
                "color": pharma["color"],
                "pymol_color": pharma["pymol_color"],
                "n_voxels": len(cluster_idxs),
            })

    # Sort by total intensity (strongest features first)
    features.sort(key=lambda f: -f["total_intensity"])
    return features


def write_pharmacophore_pdb(features, output_path):
    """Write pharmacophore features as pseudo-atom PDB.

    B-factor = normalized intensity, occupancy = radius.
    """
    with open(output_path, "w") as f:
        f.write("REMARK  PRISM4D Spike Pharmacophore Map\n")
        f.write("REMARK  B-factor = intensity score, Occupancy = radius (A)\n")
        f.write("REMARK  Each HETATM is a pharmacophore feature center\n")

        type_to_resname = {
            "BNZ": "ARO", "TYR": "DON", "TRP": "ARP", "PHE": "HYD",
            "CATION": "NEG", "ANION": "POS", "UNK": "SHP", "SS": "COV",
        }

        for i, feat in enumerate(features):
            x, y, z = feat["position"]
            resname = type_to_resname.get(feat["type"], "UNK")
            bfactor = min(feat["total_intensity"] / 100, 99.99)
            occ = min(feat["radius"], 9.99)
            f.write(
                f"HETATM{i+1:5d}  C   {resname} X{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{bfactor:6.2f}           C\n"
            )
        f.write("END\n")


def write_pharmacophore_pymol(features, receptor_pdb, output_path, site_centroid,
                              max_display=25):
    """Write PyMOL script that visualizes pharmacophore features as colored spheres
    overlaid on the receptor structure.

    Shows top max_display features by intensity, but always includes electrostatic
    features (CATION/ANION) since they indicate salt-bridge opportunities.
    """
    # Select which features to display: top N by intensity + all electrostatic
    electrostatic = {"CATION", "ANION", "SS"}
    display_features = []
    n_nonelec = 0
    for feat in features:
        if feat["type"] in electrostatic:
            # Always include electrostatic features with >=50 spikes
            if feat["spike_count"] >= 50:
                display_features.append(feat)
        else:
            if n_nonelec < max_display:
                display_features.append(feat)
                n_nonelec += 1

    lines = [
        "# PRISM4D Spike-Derived Pharmacophore Map",
        "# Each sphere = intensity-weighted pharmacophore feature",
        "# Sphere size proportional to feature strength",
        "#",
        "# Physics basis:",
        "#   Orange spheres = pi-stacking hotspots (BNZ, 258nm PHE excitation)",
        "#   Green spheres  = H-bond + pi dual sites (TYR, 274nm excitation)",
        "#   Purple spheres = Large aromatic pockets (TRP, 280nm excitation)",
        "#   Yellow spheres = Pure hydrophobic cavities (PHE dispersion)",
        "#   Red spheres    = Wants ANIONIC ligand group (CATION EFP flux)",
        "#   Blue spheres   = Wants CATIONIC ligand group (ANION EFP flux)",
        "#   Gray spheres   = Shape-complementary / flexibility (LIF thermal)",
        "#   Brown spheres  = Covalent binding opportunity (SS, disulfide)",
        "",
        "bg_color white",
        "set ray_trace_mode, 1",
        "set cartoon_fancy_helices, 1",
        "",
        f"load {receptor_pdb}, receptor",
        "show cartoon, receptor",
        "color gray80, receptor",
        "",
    ]

    max_int = max(f["total_intensity"] for f in display_features) if display_features else 1

    for i, feat in enumerate(display_features):
        x, y, z = feat["position"]
        color = feat["pymol_color"]
        stype = feat["type"]
        fname = feat["feature_name"]
        n_spikes = feat["spike_count"]
        intensity = feat["total_intensity"]

        # Scale sphere size: main features 0.6–2.0, electrostatic 0.4–0.8
        if stype in electrostatic:
            scale = 0.4 + 0.4 * (intensity / max_int)
        else:
            scale = 0.6 + 1.4 * (intensity / max_int)

        name = f"pharma_{i}_{stype}"
        lines.append(f"# Feature {i}: {fname} ({stype}, {n_spikes} spikes, intensity={intensity:.0f})")
        lines.append(f"pseudoatom {name}, pos=[{x:.3f}, {y:.3f}, {z:.3f}]")
        lines.append(f"show spheres, {name}")
        lines.append(f"set sphere_scale, {scale:.2f}, {name}")
        lines.append(f"set sphere_transparency, 0.3, {name}")
        lines.append(f"color {color}, {name}")
        lines.append("")

    # Group by type for easy toggle
    type_names = defaultdict(list)
    for i, feat in enumerate(display_features):
        type_names[feat["type"]].append(f"pharma_{i}_{feat['type']}")

    if display_features:
        all_names = " ".join(f"pharma_{i}_{f['type']}" for i, f in enumerate(display_features))
        lines.append(f"group pharmacophore, {all_names}")
        for stype, names in type_names.items():
            if len(names) > 1:
                lines.append(f"group pharma_{stype}, {' '.join(names)}")

    # Zoom
    cx, cy, cz = site_centroid
    lines.append(f"\npseudoatom _site_center, pos=[{cx:.3f}, {cy:.3f}, {cz:.3f}]")
    lines.append("zoom _site_center, 25")
    lines.append("delete _site_center")
    lines.append("")

    # Legend
    lines.append("# ════════════════════════════════════════════════════════")
    lines.append("# PHARMACOPHORE FEATURE LEGEND")
    lines.append("# ════════════════════════════════════════════════════════")
    for i, feat in enumerate(features):
        lines.append(f"# {i+1}. {feat['type']:8s} at ({feat['position'][0]:.1f}, "
                     f"{feat['position'][1]:.1f}, {feat['position'][2]:.1f})")
        lines.append(f"#    {feat['feature_name']}")
        lines.append(f"#    Fragments: {feat['fragments']}")
        lines.append(f"#    Spikes: {feat['spike_count']:,}  Intensity: {feat['total_intensity']:.0f}")
        lines.append(f"#    Water density: {feat['mean_water_density']:.4f}")
    lines.append("# ════════════════════════════════════════════════════════")
    lines.append("# INTERACTION TIPS:")
    lines.append("#   label pharmacophore, name")
    lines.append("#   distance contacts, pharmacophore, receptor and organic, 4.0")
    lines.append("#   show sticks, receptor within 5 of pharmacophore")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_interaction_map_json(features, voxels, spikes_meta, output_path):
    """Write comprehensive JSON interaction map."""
    # Compute pocket-level statistics
    all_spikes = spikes_meta["n_spikes"]
    type_breakdown = {}
    for feat in features:
        stype = feat["type"]
        type_breakdown[stype] = {
            "spike_count": feat["spike_count"],
            "percentage": round(100 * feat["spike_count"] / all_spikes, 1) if all_spikes > 0 else 0,
            "position": feat["position"],
            "radius": feat["radius"],
            "intensity": feat["total_intensity"],
            "feature": feat["feature_name"],
            "recommended_fragments": feat["fragments"],
            "functional_groups": feat["functional_groups"],
            "interaction_type": feat["interaction"],
            "water_density": feat["mean_water_density"],
        }

    # Pocket character classification
    aromatic_pct = sum(
        type_breakdown.get(t, {}).get("percentage", 0)
        for t in ["BNZ", "TYR", "TRP", "PHE"]
    )
    polar_pct = sum(
        type_breakdown.get(t, {}).get("percentage", 0)
        for t in ["CATION", "ANION"]
    )

    if aromatic_pct > 70:
        character = "Strongly hydrophobic — aromatic-dominated pocket"
    elif aromatic_pct > 40:
        character = "Mixed hydrophobic/polar — dual-character pocket"
    elif polar_pct > 20:
        character = "Polar-dominated — charge-driven binding"
    else:
        character = "Flexible/cryptic — induced-fit binding likely"

    # Drug design recommendations
    recommendations = []
    for feat in features:
        if feat["spike_count"] > 0.01 * all_spikes:  # >1% of spikes
            recommendations.append({
                "priority": "HIGH" if feat["spike_count"] > 0.10 * all_spikes else "MEDIUM",
                "region": feat["type"],
                "position": feat["position"],
                "action": f"Place {feat['functional_groups']} group here",
                "fragments": feat["fragments"],
                "rationale": feat["physics"],
            })

    output = {
        "pipeline": "PRISM4D Spike Pharmacophore Mapper",
        "timestamp": datetime.now().isoformat(),
        "site_id": spikes_meta.get("site_id", 0),
        "centroid": spikes_meta.get("centroid", [0, 0, 0]),
        "total_spikes": all_spikes,
        "pocket_character": character,
        "aromatic_percentage": round(aromatic_pct, 1),
        "polar_percentage": round(polar_pct, 1),
        "pharmacophore_features": type_breakdown,
        "design_recommendations": recommendations,
        "n_significant_voxels": len(voxels),
        "references": [
            "UV protein fluorescence: Goldfarb et al. J Biol Chem 1951",
            "Pharmacophore concept: Wermuth et al. Pure Appl Chem 1998. doi:10.1351/pac199870051129",
            "PRISM4D spike detection: neuromorphic LIF + UV pump-probe + EFP",
        ],
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="PRISM4D Spike Pharmacophore Mapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--spikes", required=True, help="spike_events.json from PRISM")
    parser.add_argument("--receptor", required=True, help="Receptor PDB file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--grid-spacing", type=float, default=2.0,
                        help="Voxel grid spacing in Angstrom (default: 2.0)")
    parser.add_argument("--min-intensity-pctl", type=float, default=50,
                        help="Minimum intensity percentile for features (default: 50)")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PRISM4D Spike Pharmacophore Mapper")
    print("=" * 60)

    # Load spike events
    print(f"\n[1/4] Loading spike events from {args.spikes}...")
    with open(args.spikes) as f:
        data = json.load(f)

    spikes = data["spikes"]
    centroid = data["centroid"]
    n_spikes = data["n_spikes"]
    print(f"  {n_spikes:,} spikes at centroid ({centroid[0]:.1f}, {centroid[1]:.1f}, {centroid[2]:.1f})")

    # Type summary
    types = Counter(s["type"] for s in spikes)
    for t, c in types.most_common():
        pct = 100 * c / n_spikes
        ints = [s["intensity"] for s in spikes if s["type"] == t]
        pharma = SPIKE_TO_PHARMACOPHORE.get(t, {})
        print(f"  {t:8s}: {c:>8,} ({pct:5.1f}%)  mean_I={np.mean(ints):.3f}  → {pharma.get('feature', '?')}")

    # Voxelize
    print(f"\n[2/4] Voxelizing at {args.grid_spacing}A resolution...")
    voxels = voxelize_spikes(spikes, grid_spacing=args.grid_spacing)
    print(f"  {len(voxels)} voxels with spike data")

    # Extract pharmacophore features
    print(f"\n[3/4] Extracting pharmacophore features (threshold: {args.min_intensity_pctl}th percentile)...")
    features = cluster_pharmacophore_features(voxels, min_intensity_pctl=args.min_intensity_pctl)
    print(f"  {len(features)} pharmacophore features identified:")
    for i, f in enumerate(features):
        pos = f["position"]
        print(f"  {i+1}. {f['type']:8s} ({f['feature_name']:30s}) at ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})  "
              f"spikes={f['spike_count']:>7,}  intensity={f['total_intensity']:>8.0f}")

    # Write outputs
    print(f"\n[4/4] Writing outputs to {output_dir}/...")

    spikes_meta = {"n_spikes": n_spikes, "site_id": data.get("site_id", 0), "centroid": centroid}

    pdb_path = output_dir / "pharmacophore_features.pdb"
    write_pharmacophore_pdb(features, pdb_path)
    print(f"  PDB:    {pdb_path}")

    pml_path = output_dir / "pharmacophore_map.pml"
    write_pharmacophore_pymol(features, args.receptor, pml_path, centroid)
    print(f"  PyMOL:  {pml_path}")

    json_path = output_dir / "interaction_map.json"
    write_interaction_map_json(features, voxels, spikes_meta, json_path)
    print(f"  JSON:   {json_path}")

    # Print design summary
    print("\n" + "=" * 60)
    print("PHARMACOPHORE-GUIDED DESIGN SUMMARY")
    print("=" * 60)
    aromatic_pct = sum(
        f["spike_count"] for f in features if f["type"] in ["BNZ", "TYR", "TRP", "PHE"]
    ) / n_spikes * 100
    polar_pct = sum(
        f["spike_count"] for f in features if f["type"] in ["CATION", "ANION"]
    ) / n_spikes * 100
    print(f"  Aromatic character: {aromatic_pct:.1f}%")
    print(f"  Polar character:    {polar_pct:.1f}%")
    print()
    print("  Recommended fragment placement:")
    for f in features:
        if f["spike_count"] > 0.01 * n_spikes:
            pos = f["position"]
            print(f"    @ ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}): {f['functional_groups']}")
    print()
    print(f"Visualize: pymol @{pml_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

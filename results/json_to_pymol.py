#!/usr/bin/env python3
"""
Convert PRISM-Delta JSON predictions to PyMOL visualization script.

Usage:
    python json_to_pymol.py 2vwd_nipah_test.json
    python json_to_pymol.py 2vwd_nipah_test.json --pdb ../data/raw/2VWD.pdb
    python json_to_pymol.py 2vwd_nipah_test.json -o custom_output.pml

Then run in PyMOL:
    pymol /path/to/structure.pdb output.pml
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_predictions(json_path: str) -> Dict[str, Any]:
    """Load PRISM-Delta predictions from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def generate_pymol_script(predictions: Dict[str, Any], pdb_path: str = None) -> str:
    """Generate PyMOL script from predictions."""

    pdb_id = predictions.get('pdb_id', 'UNKNOWN')
    timestamp = predictions.get('timestamp', '')
    config = predictions.get('config', {})
    residue_preds = predictions.get('residue_predictions', [])
    sites = predictions.get('predicted_sites', [])
    summary = predictions.get('summary', {})

    # Get chain ID from first residue
    chain_id = residue_preds[0]['chain_id'] if residue_preds else 'A'

    # Build the script
    lines = []

    # Header
    lines.append(f"""# PyMOL Visualization Script for PRISM-Delta Predictions
# Generated from: {pdb_id}
# Timestamp: {timestamp}
#
# Usage:
#   pymol /path/to/{pdb_id}.pdb {Path(pdb_path).name if pdb_path else 'this_script.pml'}
#
# Or load structure first, then run script:
#   PyMOL> run this_script.pml

""")

    # Load PDB if path provided
    if pdb_path:
        lines.append(f'# Load structure\n')
        lines.append(f'load {pdb_path}\n\n')

    # Basic setup
    lines.append("""# ============================================================================
# VISUALIZATION SETUP
# ============================================================================

# Background and display settings
bg_color white
set cartoon_fancy_helices, 1
set cartoon_side_chain_helper, on
set label_size, 14
set label_color, black
set label_outline_color, white
set label_font_id, 7

# Show cartoon representation
show cartoon
hide lines

""")

    # Color by cryptic score using B-factors
    lines.append("""# ============================================================================
# COLOR BY CRYPTIC SCORE
# ============================================================================

# We'll set B-factors to cryptic scores for coloring
# First, reset all B-factors to 0

""")

    # Set B-factors for each residue based on cryptic score
    lines.append("# Set B-factors from cryptic scores\n")
    for pred in residue_preds:
        res_num = pred['residue_num']
        score = pred['cryptic_score']
        # Scale score to 0-100 for B-factor coloring
        b_value = score * 100
        lines.append(f"alter (chain {chain_id} and resi {res_num}), b={b_value:.2f}\n")

    lines.append("""
# Apply spectrum coloring based on B-factor (cryptic score)
spectrum b, blue_white_red, minimum=0, maximum=100
rebuild

""")

    # Highlight predicted cryptic sites
    lines.append("""# ============================================================================
# PREDICTED CRYPTIC BINDING SITES
# ============================================================================

""")

    # Color palette for sites
    colors = ['hotpink', 'orange', 'yellow', 'cyan', 'lime',
              'salmon', 'violet', 'wheat', 'palegreen', 'lightblue']

    for i, site in enumerate(sites[:10]):  # Top 10 sites
        site_id = site.get('cluster_id', i)
        residues = site.get('residues', [])
        mean_score = site.get('mean_cryptic_score', 0)
        mean_escape = site.get('mean_escape_resistance', 0)
        druggability = site.get('druggability_score', 0)
        center = site.get('center', [0, 0, 0])

        # Get residue numbers
        res_nums = [r['residue_num'] for r in residues]
        res_str = '+'.join(map(str, res_nums))

        color = colors[i % len(colors)]

        lines.append(f"# Site {site_id + 1}: {len(residues)} residues, score={mean_score:.3f}, escape_R={mean_escape:.3f}\n")
        lines.append(f"select site{site_id + 1}, resi {res_str} and chain {chain_id}\n")
        lines.append(f"color {color}, site{site_id + 1}\n")
        lines.append(f"show sticks, site{site_id + 1}\n")

        # Label representative residue
        rep = site.get('representative', residues[0] if residues else None)
        if rep:
            rep_num = rep['residue_num']
            lines.append(f'label site{site_id + 1} and name CA and resi {rep_num}, "Site {site_id + 1}"\n')

        lines.append("\n")

    # High-score residues selection
    threshold = summary.get('threshold_used', 0.5)
    high_score_res = [p['residue_num'] for p in residue_preds if p['cryptic_score'] >= threshold]
    if high_score_res:
        high_str = '+'.join(map(str, high_score_res))
        lines.append(f"# All high-scoring cryptic residues (threshold={threshold:.3f})\n")
        lines.append(f"select cryptic_residues, resi {high_str} and chain {chain_id}\n")
        lines.append("show spheres, cryptic_residues and name CA\n")
        lines.append("set sphere_scale, 0.4, cryptic_residues\n\n")

    # High escape resistance residues
    high_escape = [(p['residue_num'], p['escape_resistance']) for p in residue_preds
                   if p['escape_resistance'] >= 0.7]
    if high_escape:
        escape_str = '+'.join(str(r[0]) for r in high_escape)
        lines.append(f"# High escape resistance residues (>0.7)\n")
        lines.append(f"select escape_resistant, resi {escape_str} and chain {chain_id}\n")
        lines.append("show spheres, escape_resistant and name CA\n")
        lines.append("set sphere_scale, 0.6, escape_resistant\n")
        lines.append("color green, escape_resistant\n\n")

    # Create surface
    lines.append("""# ============================================================================
# SURFACE VISUALIZATION
# ============================================================================

# Create transparent surface
create surface_obj, chain """ + chain_id + """
show surface, surface_obj
set transparency, 0.7, surface_obj
set surface_color, white, surface_obj

""")

    # Center view
    if sites:
        # Center on first site
        first_site_res = sites[0].get('residues', [])
        if first_site_res:
            center_res = '+'.join(str(r['residue_num']) for r in first_site_res[:5])
            lines.append(f"# Center view on top predicted site\n")
            lines.append(f"center resi {center_res} and chain {chain_id}\n")
            lines.append(f"zoom resi {center_res} and chain {chain_id}, 15\n\n")

    # =========================================================================
    # TEXT OUTPUT SECTION
    # =========================================================================
    lines.append("""# ============================================================================
# TEXT OUTPUT: Atomic coordinates and residue information
# ============================================================================

print("")
print("=" * 80)
print("PRISM-DELTA CRYPTIC SITE PREDICTIONS")
print("=" * 80)
print("")

""")

    # Python block for text output
    lines.append("python\n")
    lines.append("from pymol import cmd, stored\n\n")

    # Print summary
    lines.append(f'''
print("PDB ID: {pdb_id}")
print("Chain: {chain_id}")
print("Timestamp: {timestamp}")
print("")
print("Configuration:")
print("  Temperature: {config.get('hmc_temperature', 'N/A')}K")
print("  Ensemble conformations: {config.get('n_ensemble_conformations', 'N/A')}")
print("  Threshold: {summary.get('threshold_used', 'N/A'):.4f}")
print("")
print("Summary:")
print("  Total residues: {summary.get('total_residues', len(residue_preds))}")
print("  Cryptic residues: {summary.get('n_cryptic', 'N/A')}")
print("  Predicted sites: {summary.get('n_sites', len(sites))}")
print("  Mean cryptic score: {summary.get('mean_cryptic_score', 0):.4f}")
print("  Max cryptic score: {summary.get('max_cryptic_score', 0):.4f}")
print("  Mean escape resistance: {summary.get('mean_escape_resistance', 0):.4f}")
print("")
''')

    # Print each site with atomic coordinates
    for i, site in enumerate(sites[:10]):
        site_id = site.get('cluster_id', i)
        residues = site.get('residues', [])
        mean_score = site.get('mean_cryptic_score', 0)
        mean_escape = site.get('mean_escape_resistance', 0)
        druggability = site.get('druggability_score', 0)
        center = site.get('center', [0, 0, 0])
        radius = site.get('radius', 0)

        lines.append(f'''
print("")
print("=" * 70)
print("  SITE {site_id + 1}: {len(residues)} residues")
print("  Mean Cryptic Score: {mean_score:.4f}")
print("  Mean Escape Resistance: {mean_escape:.4f}")
print("  Druggability: {druggability:.4f}")
print("  Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
print("  Radius: {radius:.2f} A")
print("=" * 70)
print("")
print("  {{:^6}} {{:^4}} {{:^4}} {{:>10}} {{:>10}} {{:>10}} {{:>10}} {{:>10}}".format(
    "ResNum", "AA", "Atom", "X", "Y", "Z", "CrypScore", "EscapeR"))
print("-" * 78)
''')

        # Iterate through atoms in this site
        res_nums = [r['residue_num'] for r in residues]
        res_str = '+'.join(map(str, res_nums))

        lines.append(f'''
stored.site{site_id + 1}_atoms = []
cmd.iterate_state(1, "site{site_id + 1} and name CA",
    "stored.site{site_id + 1}_atoms.append((resi, resn, name, x, y, z))")

# Match with prediction data
site_residues = {{{', '.join(f'{r["residue_num"]}: ({r["cryptic_score"]:.4f}, {r["escape_resistance"]:.4f})' for r in residues)}}}

for atom in sorted(stored.site{site_id + 1}_atoms, key=lambda x: int(x[0])):
    resi, resn, name, x, y, z = atom
    resi_int = int(resi)
    if resi_int in site_residues:
        cscore, escore = site_residues[resi_int]
    else:
        cscore, escore = 0.0, 0.0
    print("  {{:>6}} {{:>4}} {{:>4}} {{:>10.3f}} {{:>10.3f}} {{:>10.3f}} {{:>10.4f}} {{:>10.4f}}".format(
        resi, resn, name, x, y, z, cscore, escore))
''')

    # Docking grid parameters
    lines.append('''
print("")
print("=" * 80)
print("DOCKING GRID PARAMETERS (AutoDock Vina compatible)")
print("=" * 80)
''')

    for i, site in enumerate(sites[:5]):
        site_id = site.get('cluster_id', i)
        center = site.get('center', [0, 0, 0])
        radius = site.get('radius', 10)
        # Estimate box size from radius
        size = int(radius * 2 + 10)  # Add padding
        size = max(15, min(size, 40))  # Clamp to reasonable range

        lines.append(f'''
print("")
print("Site {site_id + 1}:")
print("  --center_x {center[0]:.2f} --center_y {center[1]:.2f} --center_z {center[2]:.2f}")
print("  --size_x {size} --size_y {size} --size_z {size}")
''')

    # Legend
    lines.append('''
print("")
print("=" * 80)
print("LEGEND")
print("=" * 80)
print("Blue -> White -> Red: Cryptic Score (low to high)")
print("Green spheres: High escape resistance (>0.7)")
print("Colored sticks: Predicted binding sites")
print("  - hotpink: Site 1")
print("  - orange: Site 2")
print("  - yellow: Site 3")
print("  - cyan: Site 4")
print("  - lime: Site 5")
print("")
''')

    lines.append("python end\n\n")

    # Final message
    lines.append("""print("")
print("=" * 80)
print("Visualization complete!")
print("=" * 80)
""")

    return ''.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Convert PRISM-Delta JSON predictions to PyMOL script"
    )
    parser.add_argument("json_file", help="Path to JSON predictions file")
    parser.add_argument("--pdb", "-p", help="Path to PDB structure file")
    parser.add_argument("--output", "-o", help="Output .pml file path")

    args = parser.parse_args()

    # Load predictions
    json_path = Path(args.json_file)
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        sys.exit(1)

    predictions = load_predictions(json_path)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = json_path.with_suffix('.pml')

    # Generate script
    pdb_path = args.pdb if args.pdb else None
    script = generate_pymol_script(predictions, pdb_path)

    # Write output
    with open(output_path, 'w') as f:
        f.write(script)

    print(f"PyMOL script generated: {output_path}")
    print(f"\nUsage:")
    if pdb_path:
        print(f"  pymol {output_path}")
    else:
        print(f"  pymol /path/to/{predictions.get('pdb_id', 'structure')}.pdb {output_path}")
    print(f"\nOr in PyMOL:")
    print(f"  PyMOL> load /path/to/structure.pdb")
    print(f"  PyMOL> run {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PRISM-4D ChimeraX Visualization Generator

Generates ChimeraX command scripts for publication-quality
molecular structure visualization colored by RMSF.

Usage:
    python scripts/generate_chimerax_viz.py [--pdb PATH] [--rmsf PATH] [--output DIR]

    Then run in ChimeraX:
    chimerax --script publication/figures/visualize_rmsf.cxc
"""

import argparse
import json
import pandas as pd
from pathlib import Path

def create_rmsf_colored_pdb(input_pdb: Path, rmsf_csv: Path, output_pdb: Path):
    """Create PDB with B-factors replaced by RMSF values for visualization."""

    # Load RMSF data
    rmsf_df = pd.read_csv(rmsf_csv)
    rmsf_dict = dict(zip(rmsf_df['residue_id'], rmsf_df['rmsf']))

    output_lines = []
    with open(input_pdb, 'r') as f:
        in_first_model = True
        for line in f:
            if line.startswith('MODEL') and 'MODEL     1' not in line:
                in_first_model = False
            if line.startswith('ENDMDL'):
                output_lines.append(line)
                break
            if line.startswith('ATOM') and in_first_model:
                resid = int(line[22:26])
                if resid in rmsf_dict:
                    # Scale RMSF for visualization (0-99.99 range)
                    b_factor = min(rmsf_dict[resid] * 10, 99.99)
                    new_line = line[:60] + f'{b_factor:6.2f}' + line[66:]
                    output_lines.append(new_line)
                else:
                    output_lines.append(line)
            elif not line.startswith('ATOM') and in_first_model:
                output_lines.append(line)

    with open(output_pdb, 'w') as f:
        f.writelines(output_lines)

    return output_pdb

def generate_chimerax_script(pdb_path: Path, output_dir: Path):
    """Generate ChimeraX command script for RMSF visualization."""

    script = f'''# PRISM-4D Publication Figure - ChimeraX Script
# SARS-CoV-2 RBD colored by RMSF (Root Mean Square Fluctuation)
#
# This script generates publication-quality molecular graphics
# with residues colored by flexibility (blue=rigid, red=flexible)
#
# Run with: chimerax --script {output_dir}/visualize_rmsf.cxc

# =============================================================================
# SETUP
# =============================================================================

# Open structure with RMSF encoded in B-factor column
open {pdb_path}

# Set white background for publication
set bgColor white

# =============================================================================
# COLORING BY RMSF
# =============================================================================

# Color by B-factor (RMSF * 10) using blue-white-red gradient
# Blue = low flexibility (rigid), Red = high flexibility
color bfactor palette blue:white:red range 0,20

# =============================================================================
# RENDERING SETTINGS
# =============================================================================

# High-quality lighting
lighting soft
lighting shadows true multiShadow 64
lighting depthCue false

# Enable silhouettes for depth perception
graphics silhouettes true width 1.5

# Cartoon style settings
cartoon style protein modeh tube rad 0.3
cartoon style strand thick 1.2 arrowhead true

# =============================================================================
# HIGHLIGHT ESCAPE MUTATION SITES
# =============================================================================

# Select known SARS-CoV-2 escape mutation sites
select :346,371,373,375,417,440,446,452,477,478,484,493,496,498,501,505

# Show sidechains as sticks
style sel stick
color sel gold

# Add spheres at alpha carbons for visibility
show sel atoms
style :346,417,484,501@CA sphere

# =============================================================================
# HIGHLIGHT ACE2 INTERFACE
# =============================================================================

# Select ACE2 binding interface residues
select :417,446,449,453,455,456,475,476,477,484,486,487,489,490,493,494,495,496,498,500,501,502,505
name sel ace2_interface

# Mark with surface dots
surface ace2_interface enclose sel
transparency 70 sel

# =============================================================================
# LABELS FOR KEY SITES
# =============================================================================

# Label important escape mutations
label :477@CA text "S477N" height 1.2 color black bgColor white
label :484@CA text "E484K" height 1.2 color black bgColor white
label :501@CA text "N501Y" height 1.2 color black bgColor white
label :496@CA text "G496S" height 1.2 color black bgColor white

# =============================================================================
# GENERATE PUBLICATION FIGURES
# =============================================================================

# View 1: Overview (front view)
view initial
window #1
save {output_dir}/Figure5a_chimerax_overview.png width 2400 height 1800 supersample 3

# View 2: Side view (90 degree rotation)
turn y 90
save {output_dir}/Figure5b_chimerax_side.png width 2400 height 1800 supersample 3

# View 3: Top view (looking down binding interface)
turn x 90
save {output_dir}/Figure5c_chimerax_top.png width 2400 height 1800 supersample 3

# View 4: ACE2 interface close-up
view initial
turn y -30
turn x 15
zoom 1.5
clip near -5 far 5
save {output_dir}/Figure5d_chimerax_interface_closeup.png width 2400 height 1800 supersample 3

# =============================================================================
# SAVE SESSION
# =============================================================================

# Reset view
view initial
clip off

# Save ChimeraX session for later editing
save {output_dir}/prism4d_visualization.cxs

# =============================================================================
# COLOR LEGEND
# =============================================================================

# Create color key
key blue:white:red 0.0,0.5,1.0:2.0 pos 0.05,0.05 size 0.15,0.02 \\
    label "RMSF (A)" labelSide top fontSize 14

save {output_dir}/Figure5e_chimerax_with_legend.png width 2400 height 1800 supersample 3

# Print completion message
log text "PRISM-4D visualization complete. Figures saved to {output_dir}/"

# Keep open for interactive use (comment out for batch mode)
# exit
'''

    script_path = output_dir / 'visualize_rmsf.cxc'
    with open(script_path, 'w') as f:
        f.write(script)

    return script_path

def generate_chimerax_colorkey_session(output_dir: Path):
    """Generate a separate script for color legend creation."""

    script = '''# ChimeraX Color Legend Generator
# Creates a standalone color bar image for RMSF scale

# Create gradient
graphics backgroundColor white
2dlabels create legend text "RMSF (Angstrom)" xpos 0.5 ypos 0.9 size 24 color black

# Create gradient bar using HTML-style coloring
# Note: ChimeraX doesn't have a built-in gradient image command
# This creates a workaround using 2D labels

log text "For a proper color bar, use matplotlib or export from ChimeraX GUI"
log text "The RMSF scale is: 0 (blue) - 1.0 (white) - 2.0+ (red) Angstroms"
'''

    script_path = output_dir / 'create_colorkey.cxc'
    with open(script_path, 'w') as f:
        f.write(script)

    return script_path

def main():
    parser = argparse.ArgumentParser(
        description='Generate ChimeraX visualization scripts for PRISM-4D'
    )
    parser.add_argument(
        '--pdb',
        type=Path,
        default=Path('data/ensembles/6M0J_RBD_1ns_k2.pdb'),
        help='Input PDB file (ensemble or single structure)'
    )
    parser.add_argument(
        '--rmsf',
        type=Path,
        default=Path('results/6M0J_1ns_k2_analysis/rmsf_aligned.csv'),
        help='RMSF CSV file with residue_id and rmsf columns'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('publication/figures'),
        help='Output directory for visualization files'
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("PRISM-4D ChimeraX Visualization Generator")
    print("="*60)

    # Step 1: Create RMSF-colored PDB
    print("\n[1/3] Creating RMSF-colored PDB...")
    if args.rmsf.exists():
        rmsf_pdb = args.output / '6M0J_RBD_RMSF_bfactor.pdb'
        create_rmsf_colored_pdb(args.pdb, args.rmsf, rmsf_pdb)
        print(f"      Created: {rmsf_pdb}")
    else:
        print(f"      [WARN] RMSF file not found: {args.rmsf}")
        print(f"      Using input PDB B-factors directly")
        rmsf_pdb = args.pdb

    # Step 2: Generate ChimeraX script
    print("\n[2/3] Generating ChimeraX script...")
    script_path = generate_chimerax_script(rmsf_pdb, args.output)
    print(f"      Created: {script_path}")

    # Step 3: Generate color key script
    print("\n[3/3] Generating color key script...")
    colorkey_path = generate_chimerax_colorkey_session(args.output)
    print(f"      Created: {colorkey_path}")

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\nTo generate figures, run:")
    print(f"  chimerax --script {script_path}")
    print(f"\nOr for interactive use:")
    print(f"  chimerax {rmsf_pdb}")
    print(f"  Then: open {script_path}")
    print("="*60)

if __name__ == '__main__':
    main()

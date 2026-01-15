#!/usr/bin/env python3
"""
PRISM-4D PyMOL Visualization Generator

Generates PyMOL command scripts for publication-quality
molecular structure visualization colored by RMSF.

Usage:
    python scripts/generate_pymol_viz.py [--pdb PATH] [--rmsf PATH] [--output DIR]

    Then run in PyMOL:
    pymol -cq publication/figures/visualize_rmsf.pml

    Or for interactive:
    pymol publication/figures/visualize_rmsf.pml
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

def generate_pymol_script(pdb_path: Path, output_dir: Path):
    """Generate PyMOL command script for RMSF visualization."""

    script = f'''# PRISM-4D Publication Figure - PyMOL Script
# SARS-CoV-2 RBD colored by RMSF (Root Mean Square Fluctuation)
#
# This script generates publication-quality molecular graphics
# with residues colored by flexibility (blue=rigid, red=flexible)
#
# Run with: pymol -cq {output_dir}/visualize_rmsf.pml
# Or interactive: pymol {output_dir}/visualize_rmsf.pml

# =============================================================================
# INITIALIZATION
# =============================================================================

# Reset PyMOL
reinitialize

# Load structure with RMSF encoded in B-factor column
load {pdb_path}, rbd

# =============================================================================
# RENDERING SETTINGS (Publication Quality)
# =============================================================================

# White background
bg_color white

# High-quality rendering settings
set ray_trace_mode, 1
set ray_shadows, 1
set ray_shadow_decay_factor, 0.1
set ray_shadow_decay_range, 2
set antialias, 2
set hash_max, 300

# Cartoon settings
set cartoon_fancy_helices, 1
set cartoon_smooth_loops, 1
set cartoon_loop_radius, 0.3
set cartoon_tube_radius, 0.3
set cartoon_sampling, 14
set cartoon_ring_mode, 3

# Stick settings for sidechains
set stick_radius, 0.15
set stick_ball, on
set stick_ball_ratio, 1.5

# Transparency and depth cueing
set depth_cue, 1
set fog_start, 0.45

# =============================================================================
# COLORING BY RMSF
# =============================================================================

# Color by B-factor (RMSF * 10) using blue-white-red gradient
# Blue = low flexibility (rigid), Red = high flexibility
spectrum b, blue_white_red, rbd, minimum=0, maximum=20

# =============================================================================
# DISPLAY SETTINGS
# =============================================================================

# Show as cartoon
hide everything
show cartoon, rbd

# =============================================================================
# HIGHLIGHT ESCAPE MUTATION SITES
# =============================================================================

# Define escape mutation sites selection
select escape_sites, resi 346+371+373+375+417+440+446+452+477+478+484+493+496+498+501+505

# Show sidechains as sticks
show sticks, escape_sites and sidechain
color yellow, escape_sites and sidechain

# Create spheres at CA for key sites
select key_escape, resi 417+484+496+501 and name CA
show spheres, key_escape
set sphere_scale, 0.5, key_escape
color orange, key_escape

# =============================================================================
# HIGHLIGHT ACE2 INTERFACE
# =============================================================================

# Define ACE2 interface selection
select ace2_interface, resi 417+446+449+453+455+456+475+476+477+484+486+487+489+490+493+494+495+496+498+500+501+502+505

# Create transparent surface for interface region
create ace2_surface, ace2_interface
show surface, ace2_surface
set transparency, 0.7, ace2_surface
color palegreen, ace2_surface

# =============================================================================
# LABELS FOR KEY SITES
# =============================================================================

# Create labels for key escape mutations
label resi 477 and name CA, "S477N"
label resi 484 and name CA, "E484K"
label resi 496 and name CA, "G496S"
label resi 501 and name CA, "N501Y"

# Label formatting
set label_color, black
set label_size, 16
set label_font_id, 7
set label_position, (2, 2, 2)
set label_outline_color, white

# =============================================================================
# GENERATE PUBLICATION FIGURES
# =============================================================================

# View 1: Overview (front view)
orient rbd
zoom rbd, 5
ray 2400, 1800
png {output_dir}/Figure5a_pymol_overview.png, dpi=300

# View 2: Side view (90 degree rotation around Y)
turn y, 90
ray 2400, 1800
png {output_dir}/Figure5b_pymol_side.png, dpi=300

# View 3: Top view (looking down on ACE2 interface)
turn x, -90
ray 2400, 1800
png {output_dir}/Figure5c_pymol_top.png, dpi=300

# View 4: Interface close-up
orient ace2_interface
zoom ace2_interface, 10
turn y, 20
turn x, 10
ray 2400, 1800
png {output_dir}/Figure5d_pymol_interface_closeup.png, dpi=300

# =============================================================================
# CREATE FIGURE WITH COLOR LEGEND
# =============================================================================

# Reset to overview
orient rbd
zoom rbd, 5

# Add pseudo-atom based color legend (workaround for PyMOL)
# Note: Best to add color bar in post-processing with matplotlib

# Save image for color bar addition
ray 2400, 1800
png {output_dir}/Figure5e_pymol_for_legend.png, dpi=300

# =============================================================================
# SAVE SESSION
# =============================================================================

# Clean up temporary selections
delete key_escape

# Save PyMOL session for later editing
save {output_dir}/prism4d_visualization.pse

# =============================================================================
# SUMMARY
# =============================================================================

print ""
print "========================================"
print "PRISM-4D PyMOL Visualization Complete"
print "========================================"
print "Figures saved to: {output_dir}/"
print ""
print "Files generated:"
print "  - Figure5a_pymol_overview.png"
print "  - Figure5b_pymol_side.png"
print "  - Figure5c_pymol_top.png"
print "  - Figure5d_pymol_interface_closeup.png"
print "  - Figure5e_pymol_for_legend.png"
print "  - prism4d_visualization.pse (session)"
print ""
print "Color scale: RMSF (Angstrom)"
print "  Blue  = 0.0 (rigid)"
print "  White = 1.0"
print "  Red   = 2.0+ (flexible)"
print "========================================"

# Keep session open for interactive use
# Comment out 'quit' for interactive mode
quit
'''

    script_path = output_dir / 'visualize_rmsf.pml'
    with open(script_path, 'w') as f:
        f.write(script)

    return script_path

def generate_colorbar_python(output_dir: Path):
    """Generate Python script to add color bar to figures."""

    script = '''#!/usr/bin/env python3
"""
Add RMSF color bar to PyMOL/ChimeraX figures.

This script adds a publication-quality color bar to structure figures
showing the RMSF color scale.

Usage:
    python add_colorbar.py Figure5a_pymol_overview.png
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np
from PIL import Image
import sys
from pathlib import Path

def add_colorbar_to_image(image_path: Path, output_path: Path = None):
    """Add RMSF color bar to an image."""

    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_with_legend{image_path.suffix}"

    # Load the image
    img = Image.open(image_path)
    img_array = np.array(img)

    # Create figure with image and colorbar
    fig = plt.figure(figsize=(14, 10.5))

    # Add image
    ax_img = fig.add_axes([0, 0, 0.9, 1])  # Leave room for colorbar
    ax_img.imshow(img_array)
    ax_img.axis('off')

    # Create custom blue-white-red colormap (matching PyMOL/ChimeraX)
    colors = ['#0000FF', '#FFFFFF', '#FF0000']  # Blue -> White -> Red
    cmap = mcolors.LinearSegmentedColormap.from_list('rmsf', colors)

    # Add colorbar
    ax_cbar = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    norm = mcolors.Normalize(vmin=0, vmax=2.0)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label('RMSF (Angstrom)', fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")
    return output_path

def main():
    if len(sys.argv) < 2:
        print("Usage: python add_colorbar.py <image_path>")
        print("Example: python add_colorbar.py Figure5a_pymol_overview.png")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: File not found: {image_path}")
        sys.exit(1)

    add_colorbar_to_image(image_path)

if __name__ == '__main__':
    main()
'''

    script_path = output_dir / 'add_colorbar.py'
    with open(script_path, 'w') as f:
        f.write(script)

    return script_path

def main():
    parser = argparse.ArgumentParser(
        description='Generate PyMOL visualization scripts for PRISM-4D'
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
    print("PRISM-4D PyMOL Visualization Generator")
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

    # Step 2: Generate PyMOL script
    print("\n[2/3] Generating PyMOL script...")
    script_path = generate_pymol_script(rmsf_pdb, args.output)
    print(f"      Created: {script_path}")

    # Step 3: Generate color bar helper script
    print("\n[3/3] Generating color bar helper script...")
    colorbar_path = generate_colorbar_python(args.output)
    print(f"      Created: {colorbar_path}")

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"\nTo generate figures (batch mode), run:")
    print(f"  pymol -cq {script_path}")
    print(f"\nFor interactive use:")
    print(f"  pymol {script_path}")
    print(f"\nTo add color bar to generated images:")
    print(f"  python {colorbar_path} {args.output}/Figure5a_pymol_overview.png")
    print("="*60)

if __name__ == '__main__':
    main()

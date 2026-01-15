# PRISM-4D Publication Figure - PyMOL Script
# SARS-CoV-2 RBD colored by RMSF

# Load structure with RMSF as B-factor
load publication/figures/6M0J_RBD_RMSF_bfactor.pdb, rbd

# Set up nice rendering
bg_color white
set ray_shadow, 0
set antialias, 2
set cartoon_fancy_helices, 1
set cartoon_smooth_loops, 1

# Color by B-factor (RMSF)
spectrum b, blue_white_red, rbd, minimum=0, maximum=20

# Show cartoon representation
hide everything
show cartoon, rbd

# Highlight escape mutation sites
select escape_sites, resi 346+371+373+375+417+440+446+452+477+478+484+493+496+498+501+505
show sticks, escape_sites and sidechain
color yellow, escape_sites and sidechain

# Highlight ACE2 interface
select ace2_interface, resi 417+446+449+453+455+456+475+476+477+484+486+487+489+490+493+494+495+496+498+500+501+502+505
color orange, ace2_interface and cartoon

# Labels for key sites
label resi 477 and name CA, "S477N"
label resi 484 and name CA, "E484K"
label resi 501 and name CA, "N501Y"
set label_color, black
set label_size, 14

# View 1: Overview
orient rbd
ray 2400, 1800
png publication/figures/Figure5a_structure_overview.png, dpi=300

# View 2: ACE2 interface
turn y, 90
ray 2400, 1800
png publication/figures/Figure5b_structure_interface.png, dpi=300

# View 3: Top-down on RBM
turn x, -90
ray 2400, 1800
png publication/figures/Figure5c_structure_top.png, dpi=300

# Save session
save publication/figures/prism4d_publication.pse

quit

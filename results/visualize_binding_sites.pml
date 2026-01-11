# PyMOL Script: PRISM-Delta Binding Site Visualization
# Load with: pymol /path/to/6vxx_cryptic_sites.pdb visualize_binding_sites.pml

# Basic setup
bg_color white
set cartoon_fancy_helices, 1
set cartoon_side_chain_helper, on

# Color by cryptic score (B-factor)
spectrum b, blue_white_red, minimum=0, maximum=100

# Show as cartoon
show cartoon
hide lines

# === TOP BINDING SITES FROM PRISM-DELTA ===

# Site 1: RBD cryptic pocket (residues 396-397, 511)
# High priority - escape resistant
select site1_rbd, resi 396-397+511 and chain A
color hotpink, site1_rbd
show sticks, site1_rbd
label site1_rbd and name CA, "RBD Cryptic"

# Site 2: NTD pocket (residues 92-205 region)
# High druggability score (0.91)
select site2_ntd, resi 92-108+191-205 and chain A
color orange, site2_ntd
show sticks, site2_ntd

# Site 3: S2 fusion peptide region (878-1033)
# Known cryptic site from ground truth (816-823)
select site3_fusion, resi 816-823+878-900 and chain A
color yellow, site3_fusion
show sticks, site3_fusion
label site3_fusion and name CA and resi 820, "Fusion Peptide"

# === GROUND TRUTH CRYPTIC RESIDUES ===
select ground_truth, resi 373-379+503-509+816-823 and chain A
show spheres, ground_truth and name CA
set sphere_scale, 0.5, ground_truth
color green, ground_truth

# === ESCAPE MUTATION SITES ===
# Known escape mutations from CoV-RDB
select escape_sites, resi 417+452+484+501+614+681 and chain A
show spheres, escape_sites and name CA
set sphere_scale, 0.7, escape_sites
color red, escape_sites
label escape_sites and name CA, "%s%s" % (resn, resi)

# Center view on RBD
select rbd, resi 319-541 and chain A
center rbd
zoom rbd

# Create surface for druggability visualization
create surface_obj, chain A
show surface, surface_obj
set transparency, 0.7, surface_obj
set surface_color, white, surface_obj

# Legend
print("=== PRISM-Delta Binding Site Visualization ===")
print("Blue->White->Red: Cryptic Score (low to high)")
print("Green spheres: Ground truth cryptic residues")
print("Red spheres: Known escape mutation sites")
print("Pink sticks: Top predicted RBD cryptic site")
print("Orange sticks: NTD druggable pocket")
print("Yellow sticks: Fusion peptide region")


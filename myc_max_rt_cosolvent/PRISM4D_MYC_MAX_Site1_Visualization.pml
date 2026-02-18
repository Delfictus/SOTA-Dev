# =====================================================================
# PRISM-4D MYC-MAX Cryptic Binding Site Visualization
# Structure: 1NKP dimer | Run: RT-Full + BNZ Cosolvent
# =====================================================================
# Run from myc_max_rt_cosolvent/ directory:
#   pymol PRISM4D_MYC_MAX_Site1_Visualization.pml
# =====================================================================

# ── Setup ──
reinitialize
bg_color white
set ray_shadow, 0
set cartoon_fancy_helices, 1
set cartoon_side_chain_helper, 1
set cartoon_highlight_color, grey90
set surface_quality, 2
set label_size, 14
set label_font_id, 7
set label_color, black
set label_outline_color, white
set label_position, [0, 0, 3]
set depth_cue, 0
set spec_reflect, 0.2
set antialias, 2

# ── Load structure ──
load 1nkp_dimer.binding_sites.pdb, protein, 1

# ── Base representation ──
hide everything, protein
show cartoon, protein

# Color chains
select chain_myc, chain A and protein
select chain_max, chain B and protein
color skyblue, chain_myc
color lightorange, chain_max

# =====================================================================
# SITE 1: Cross-Chain Cryptic Interface Pocket (PRIMARY FINDING)
# MYC E407-R419 (topo 59-71) <-> MAX K77-L88 (topo 142-153)
# Quality: 0.706 | Druggability: 0.577 | Volume: 2628 A^3
# =====================================================================

# MYC side (Chain A topo 59-71)
select site1_myc, chain A and (resi 59-71) and protein
show sticks, site1_myc
set stick_radius, 0.18, site1_myc
color tv_blue, site1_myc

# MYCHot2 core (known 10058-F4 binding site, MYC 407-412)
select site1_mychot2, chain A and (resi 59-64) and protein
color marine, site1_mychot2

# Zipper extension (MYC 413-419, novel detection)
select site1_zipper, chain A and (resi 65-71) and protein
color deepblue, site1_zipper

# MAX side (Chain B)
select site1_max, chain B and (resi 142+143+145+146+147+149+150+153) and protein
show sticks, site1_max
set stick_radius, 0.18, site1_max
color tv_orange, site1_max

# MAX ala-scan validated critical residues (K77, N78, H81, I85, L88)
select site1_max_critical, chain B and (resi 142+143+146+150+153) and protein
color firebrick, site1_max_critical

# Pocket surface
select site1_all, site1_myc or site1_max
create site1_surface, site1_all
show surface, site1_surface
set surface_color, palegreen, site1_surface
set transparency, 0.65, site1_surface

# Pocket centroid
pseudoatom site1_center, pos=[59.073, 49.083, 49.577]
show spheres, site1_center
set sphere_scale, 0.6, site1_center
color red, site1_center

# Labels: MYC side (UniProt numbering)
label chain A and resi 59 and name CA and protein, "E407"
label chain A and resi 60 and name CA and protein, "E408"
label chain A and resi 61 and name CA and protein, "Q409"
label chain A and resi 62 and name CA and protein, "K410"
label chain A and resi 63 and name CA and protein, "L411"
label chain A and resi 64 and name CA and protein, "I412"
label chain A and resi 65 and name CA and protein, "S413"
label chain A and resi 66 and name CA and protein, "E414"
label chain A and resi 67 and name CA and protein, "E415"
label chain A and resi 68 and name CA and protein, "D416"
label chain A and resi 69 and name CA and protein, "L417"
label chain A and resi 70 and name CA and protein, "L418"
label chain A and resi 71 and name CA and protein, "R419"

# Labels: MAX side (UniProt numbering, * = ala-scan validated)
label chain B and resi 142 and name CA and protein, "*K77"
label chain B and resi 143 and name CA and protein, "*N78"
label chain B and resi 145 and name CA and protein, "T80"
label chain B and resi 146 and name CA and protein, "*H81"
label chain B and resi 147 and name CA and protein, "Q82"
label chain B and resi 149 and name CA and protein, "D84"
label chain B and resi 150 and name CA and protein, "*I85"
label chain B and resi 153 and name CA and protein, "*L88"

# Group Site 1
group site1, site1_myc site1_max site1_mychot2 site1_zipper site1_max_critical site1_surface site1_center

# =====================================================================
# SITE 2: C-Terminal Charged Patch (LOW PRIORITY)
# =====================================================================

select site2_lining, ((chain A and resi 73-84) or (chain B and resi 156-167)) and protein
show sticks, site2_lining
set stick_radius, 0.12, site2_lining
color grey60, site2_lining

create site2_surface, site2_lining
show surface, site2_surface
set surface_color, grey70, site2_surface
set transparency, 0.8, site2_surface

pseudoatom site2_center, pos=[55.188, 34.439, 62.736]
show spheres, site2_center
set sphere_scale, 0.4, site2_center
color grey50, site2_center

group site2, site2_lining site2_surface site2_center

# =====================================================================
# SITE 3: Expression Tag Artifact (DISCARD)
# =====================================================================

select site3_artifact, chain A and (resi 3-14) and protein
show sticks, site3_artifact
set stick_radius, 0.10, site3_artifact
color grey80, site3_artifact

pseudoatom site3_center, pos=[70.494, 96.644, 35.759]
show spheres, site3_center
set sphere_scale, 0.3, site3_center
color grey70, site3_center

group site3_discard, site3_artifact site3_center

# ── Default: focus on Site 1 ──
disable site2
disable site3_discard
deselect

zoom site1_all, 8
orient site1_all
turn y, 30

# =====================================================================
# COLOR LEGEND
# =====================================================================
# Marine sticks     = MYCHot2 core (MYC 407-412, 10058-F4 site)
# Deep blue sticks  = Zipper extension (MYC 413-419, novel)
# Orange sticks     = MAX interface residues
# Firebrick sticks  = MAX ala-scan critical (K77,N78,H81,I85,L88)
# Pale green surface = Site 1 pocket cavity
# Red sphere        = Site 1 centroid
# * in labels       = experimentally validated critical residue
# Sky blue cartoon  = MYC (Chain A)
# Light orange cartoon = MAX (Chain B)
# =====================================================================
# ray 2400, 1800
# png myc_max_site1.png, dpi=300

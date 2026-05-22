#  ─────────────────────────────────────────────────────────────────────────
#  PRISM-4D — Highest-Accuracy Detection (last 30 days)
#  Target:    KRAS  (PDB 4OBE, WT bound to GDP — apo of the G12C drug class)
#  Run:       output/4obe_regression / 4obe_clean.binding_sites.json
#  Site:      id=8  rank=1 by engine druggability
#  Class:     CRYPTIC  (not the GDP orthosteric site — α3/α4-face cryptic pocket)
#  Centroid:  [ +2.759, -29.361, +42.321 ]   Vol 266 Å³   Sphericity 0.74
#  Druggability: 0.8427    is_druggable: true
#  ΔG decomposition (kcal/mol):
#    effective_ΔG       -3404.11
#    ΔG_dewetting        +146.74    (open-pocket cost — designer should target hydrophobic occlusion)
#    ΔG_electrostatic      +0.84    (modest polar penalty — accept HBA/HBD design freely)
#    ΔG_aromatic           +5.79    (room for π-stacking gain)
#    ΔG_cooperative     -3557.48    (large network coupling)
#  Spike count: 921,647    Burial: 0.64    Aromatic score: 0.95-1.00 in inner shell
#
#  IMPORTANT
#  ─────────
#  Engine emitted residue IDs are TOPOLOGY indices (0-indexed). PDB resids in
#  this script are corrected: pdb_resid = engine_resid + 1 (chain A; verified
#  against 4obe_clean.residue_map.json). Inner-shell residues match the α3/α4
#  cryptic face — distinct from the GDP nucleotide pocket and from the
#  sotorasib switch-II pocket. Treat as a NOVEL allosteric candidate.
#
#  This is engine-internal evidence. NO DCC validation against an
#  external cocrystal has been run. The ground-truth file
#  (4obe_clean_ground_truth.json) cites GDP as the reference ligand —
#  that pocket is on the OPPOSITE face of the protein (~19 Å from this site).
#
#  Run this from the run directory or pass the path explicitly:
#    cd /home/diddy/Desktop/Prism4D-bio/output/4obe_regression
#    pymol -r ../../scripts/quarantine/4obe_top_site_designer.pml
#  ─────────────────────────────────────────────────────────────────────────

# ── Globals ──────────────────────────────────────────────────────────────
reinitialize
bg_color white
set ray_opaque_background, off
set ray_shadows, 0
set ray_trace_mode, 1
set ray_trace_color, gray30
set ambient, 0.30
set specular, 0.20
set cartoon_fancy_helices, 1
set cartoon_side_chain_helper, 1
set cartoon_smooth_loops, 1
set cartoon_loop_radius, 0.25
set surface_quality, 2
set surface_smooth_edges, 1
set transparency_mode, 1
set valence, 1
set dash_color, gray40
set label_size, 14
set label_color, gray20
set label_outline_color, white
set label_position, (0, 1.5, 0)
set sphere_scale, 0.50, all
set stick_radius, 0.18

# ── Load structure ───────────────────────────────────────────────────────
# Adjust if running from elsewhere
load 4obe_clean.pdb, kras
remove resn HOH
hide everything, kras
show cartoon, kras
color gray85, kras and chain A
color gray60, kras and chain B

# Ligand for spatial reference (GDP at the orthosteric site, NOT this site)
select gdp, kras and chain A and resi 201 and resn GDP
show sticks, gdp
color salmon, gdp
util.cnc gdp

# ── Site 1 — top cryptic pocket ──────────────────────────────────────────
# Inner shell (min_distance ≤ 5 Å) — primary contact sphere
select s1_inner, kras and chain A and resi 80+93+94+96+97+113

# Mid shell (5 Å < min_distance ≤ 8 Å) — first-coordination accommodating residues
select s1_mid,   kras and chain A and resi 9+78+82+89+90+91+92+101

# Outer shell (8 Å < min_distance ≤ 12 Å) — entry vector / breathing residues
select s1_outer, kras and chain A and resi 7+8+10+11+14+19+72+77+79+87+128+129

# Engine-flagged catalytic — KRAS-native CYS80 is the designer-relevant residue
# (covalent handle candidate; α3 face). Engine also flagged ASP92, GLU91, SER89, LYS101, LYS128.
select s1_cat,   kras and chain A and resi 80+89+91+92+101+128
select s1_cys80, kras and chain A and resi 80      # explicit covalent target

# TIDE allosteric trigger residues (engine candidate_residue_ids → PDB)
# Engine reported [93, 136, 138, 134, 100] (0-indexed) → [94, 137, 139, 135, 101]
select s1_tide,  kras and chain A and resi 94+101+135+137+139

# Aromatic & hydrophobic groupings for pharmacophore-aware coloring
select s1_aromatic, s1_inner or s1_mid and resn PHE+TYR+TRP+HID+HIS+HIE+HIP
select s1_hyd,      (s1_inner or s1_mid) and resn ALA+VAL+LEU+ILE+MET+PRO+GLY+CYS
select s1_polar,    (s1_inner or s1_mid) and resn SER+THR+ASN+GLN+TYR
select s1_charged,  (s1_inner or s1_mid) and resn LYS+ARG+ASP+GLU+HID+HIS+HIP+HIE

# Sticks: union of inner+mid; outer shell stays cartoon-only
show sticks, s1_inner or s1_mid
hide sticks, name C+N+O+H and not (name CA)
color gray70, s1_inner or s1_mid

# Pharmacophore palette (CB+ atom colors — preserved on top of element colors)
color tv_orange,    s1_aromatic and (not name N+C+O+H)
color paleyellow,   s1_hyd      and (not name N+C+O+H)
color limon,        s1_polar    and (not name N+C+O+H)
color lightblue,    s1_charged  and (not name N+C+O+H)

# CYS80 — emphasize for covalent design
show sticks, s1_cys80
color magenta, s1_cys80 and not (name N+C+O+H)
show spheres, s1_cys80 and name SG
set sphere_scale, 0.35, s1_cys80 and name SG
color hotpink, s1_cys80 and name SG
label s1_cys80 and name CA, "CYS80 (Cα-SG covalent handle)"

# Engine-flagged catalytic / hot residues (all)
color magenta, s1_cat and (name CA)

# TIDE-coupling residues (allosteric trigger — designer should NOT mutate)
show sticks, s1_tide
color deepteal, s1_tide and not (name N+C+O+H)
label s1_tide and name CA and resi 94, "HIS94 (TIDE)"
label s1_tide and name CA and resi 101, "LYS101 (TIDE)"
label s1_tide and name CA and resi 135, "ALA135 (TIDE)"
label s1_tide and name CA and resi 137, "TYR137 (TIDE)"
label s1_tide and name CA and resi 139, "ILE139 (TIDE)"

# Heteroatom recoloring (universal: preserve element colors on heteros)
util.cnc s1_inner
util.cnc s1_mid

# ── Centroid pseudoatom + distance helpers ───────────────────────────────
pseudoatom s1_centroid, pos=[2.759, -29.361, 42.321], color=red
show spheres, s1_centroid
set sphere_scale, 0.7, s1_centroid
label s1_centroid, "site_8  drug=0.84  vol=266Å³"

# Inner-shell vector lines (Cα→centroid) — read out the pocket geometry
distance d_iv1, s1_centroid, kras and chain A and resi 80 and name CA
distance d_iv2, s1_centroid, kras and chain A and resi 93 and name CA
distance d_iv3, s1_centroid, kras and chain A and resi 94 and name CA
distance d_iv4, s1_centroid, kras and chain A and resi 96 and name CA
distance d_iv5, s1_centroid, kras and chain A and resi 97 and name CA
distance d_iv6, s1_centroid, kras and chain A and resi 113 and name CA
hide labels, d_iv1+d_iv2+d_iv3+d_iv4+d_iv5+d_iv6
color gray50, d_iv1+d_iv2+d_iv3+d_iv4+d_iv5+d_iv6

# Distance from cryptic-pocket centroid to GDP centroid (orthosteric reference)
pseudoatom gdp_centroid, pos=[2.05, -10.437, 38.210], color=salmon, label=GDP_centroid
distance d_to_gdp, s1_centroid, gdp_centroid
color sand, d_to_gdp

# ── Pocket surface ───────────────────────────────────────────────────────
create s1_pocket_surface_obj, s1_inner or s1_mid
hide everything, s1_pocket_surface_obj
show surface, s1_pocket_surface_obj
set surface_color, slate, s1_pocket_surface_obj
set transparency, 0.55, s1_pocket_surface_obj
set surface_carve_selection, s1_centroid
set surface_carve_cutoff, 9.0
set surface_carve_state, 0

# Pocket-volume gauge sphere (266 Å³ → r ≈ 4.0 Å for sphere of equiv volume)
pseudoatom s1_volume_gauge, pos=[2.759, -29.361, 42.321]
show spheres, s1_volume_gauge
set sphere_scale, 4.0, s1_volume_gauge
color gray60, s1_volume_gauge
set sphere_transparency, 0.75, s1_volume_gauge
hide spheres, s1_volume_gauge   # toggle on demand via: show spheres, s1_volume_gauge

# ── Backbone H-bond donor/acceptor cues at pocket lip ────────────────────
# Inner-shell backbone NH (HBD) and C=O (HBA) — readout for designer's hydrogen-bond plan
select s1_bb_nh, (s1_inner or s1_mid) and name N and not resn PRO
select s1_bb_co, (s1_inner or s1_mid) and name O
show sticks, s1_bb_nh + s1_bb_co
color tv_blue, s1_bb_nh
color firebrick, s1_bb_co

# ── Groups & camera presets ──────────────────────────────────────────────
group site_1, s1_inner s1_mid s1_outer s1_cat s1_cys80 s1_tide s1_aromatic s1_hyd s1_polar s1_charged s1_centroid s1_pocket_surface_obj s1_volume_gauge gdp_centroid d_iv1 d_iv2 d_iv3 d_iv4 d_iv5 d_iv6 d_to_gdp s1_bb_nh s1_bb_co
group ref_ligand, gdp gdp_centroid

orient kras and chain A
zoom s1_inner, 6
clip slab, 18

# ── Scenes ───────────────────────────────────────────────────────────────
# Scene 1: pocket overview
scene F1, store, message=1. POCKET OVERVIEW (cryptic α3/α4-face site)

# Scene 2: covalent-design view (CYS80 SG + entry vector)
hide labels
show sticks, s1_inner
zoom s1_cys80, 8
orient s1_cys80
turn x, -30
turn y, 25
label s1_cys80 and name SG, "SG  (covalent target)"
label kras and chain A and resi 80 and name CA, "CYS80 Cα"
scene F2, store, message=2. COVALENT-DESIGN VIEW (CYS80 SG accessibility)

# Scene 3: allosteric (TIDE-coupled) view — designer must not break this network
hide labels
show sticks, s1_tide
zoom s1_tide, 12
color deepteal, s1_tide and not (name N+C+O+H)
label s1_tide and name CA and resi 94,  "H94"
label s1_tide and name CA and resi 101, "K101"
label s1_tide and name CA and resi 135, "A135"
label s1_tide and name CA and resi 137, "Y137"
label s1_tide and name CA and resi 139, "I139"
scene F3, store, message=3. ALLOSTERIC TIDE NETWORK (do NOT mutate)

# Scene 4: spatial reference vs orthosteric (GDP) — different face of the protein
hide labels
zoom (s1_centroid or gdp_centroid), 5
show spheres, s1_centroid + gdp_centroid
label s1_centroid, "PRISM_site_8 (cryptic)"
label gdp_centroid, "GDP orthosteric"
scene F4, store, message=4. CRYPTIC-vs-ORTHOSTERIC SEPARATION (~19 Å, opposite face)

# Scene 5: slab cross-section through pocket (entry vector visualization)
hide labels
show surface, s1_pocket_surface_obj
zoom s1_centroid, 10
clip slab, 12
scene F5, store, message=5. SLAB CROSS-SECTION (pocket-entry vector)

# Scene 6: full inner+mid sticks + cartoon (publication figure base)
hide labels
hide surface
show cartoon, kras
show sticks, s1_inner + s1_mid + s1_cys80 + s1_tide
util.cnc s1_inner
util.cnc s1_mid
color magenta, s1_cys80 and not (name N+C+O+H)
color deepteal, s1_tide and not (name N+C+O+H)
zoom s1_inner, 8
scene F6, store, message=6. PUBLICATION FIGURE BASE (sticks + cartoon)

# Default scene on load
scene F1, recall

# ── Designer notes (printed to console) ──────────────────────────────────
print  ""
print  "==========================================================="
print  "  PRISM-4D top detection: 4obe / KRAS / cryptic site_8"
print  "==========================================================="
print  "  Druggability:        0.8427"
print  "  Volume / Sphericity: 266 Å³ / 0.74"
print  "  ΔG_dewetting:        +146.7 kcal/mol  ← occlude hydrophobics"
print  "  ΔG_aromatic:           +5.8 kcal/mol  ← π-stacking room"
print  "  Burial:                0.64           ← partially exposed"
print  ""
print  "  Inner shell  (≤ 5 Å):  ILE93  ARG97  LEU113  CYS80  HID94  TYR96"
print  "  Covalent lever:        CYS80 (α3, native — not G12C)"
print  "  TIDE trigger res:      H94  K101  A135  Y137  I139"
print  "  Reference orthosteric: GDP (chain A, resi 201) — ~19 Å away"
print  ""
print  "  Scenes:  F1 overview | F2 covalent | F3 allosteric | F4 vs GDP"
print  "           F5 slab     | F6 publication"
print  ""
print  "  CAVEAT: engine-internal evidence only. No DCC validation"
print  "  vs external cocrystal. Treat as candidate, not validated hit."
print  "==========================================================="
print  ""

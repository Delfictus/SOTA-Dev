# Load the CORRECT local structures (KRAS Apo vs KRAS Holo)
# Note: 4LPK is KRAS, so we must validate against a KRAS drug-bound structure
load data/targets/4lpk.pdb, apo_kras
load data/targets/6oim.pdb, holo_kras

# Clean up solvent and ligands from both
remove solvent
remove resn EDO+SO4+GOL+GDP+MG

# Align KRAS to KRAS
# This should result in an RMSD < 1.0A
align apo_kras, holo_kras

# Visual Setup
bg_color white
color gray70, holo_kras
show cartoon, holo_kras

# Highlight Sotorasib (The KRAS G12C Inhibitor)
select sotorasib, holo_kras and organic
show sticks, sotorasib
color magenta, sotorasib

# Setup your V2 engine prediction
color cyan, apo_kras
show cartoon, apo_kras

# Highlight Adjudicator Hotspots
# These are the residues flagged by your KCC/SURP telemetry
select engine_hotspots, apo_kras and resi 85+93+96+99+102+108+158
show sticks, engine_hotspots
color yellow, engine_hotspots

# Highlight the massive TYR-96 trigger (Orange)
select tyr_96_trigger, apo_kras and resi 96
color orange, tyr_96_trigger

# Focus the camera on the pocket
zoom sotorasib, 5.0

# --- MATHEMATICAL VALIDATION ---
# 1. Minimum distance from predicted hotspots to the drug
distance dist_min, engine_hotspots, sotorasib

# 2. Check for physical clash (Orange TYR-96 blocking the drug)
distance clash_check, tyr_96_trigger, sotorasib

# 3. Backbone shift (How much the pocket 'tore' at ground zero)
distance tear_dist, apo_kras and resi 96 and name CA, holo_kras and resi 96 and name CA

# Print results to console
set label_size, 20
set label_color, black

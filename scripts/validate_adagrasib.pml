# Load Apo KRAS and Adagrasib Holo KRAS (6UT0)
load data/targets/4lpk.pdb, apo_kras
load data/targets/6ut0.pdb, holo_adagrasib

# Clean up crystallization artifacts
remove solvent
remove resn EDO+SO4+GOL+GDP+MG

# Align the backbones
align apo_kras, holo_adagrasib

# Visual Setup
bg_color white
color gray70, holo_adagrasib
show cartoon, holo_adagrasib

# Highlight Adagrasib (Green)
select adagrasib, holo_adagrasib and organic
show sticks, adagrasib
color green, adagrasib

# Setup your V2 engine prediction
color cyan, apo_kras
show cartoon, apo_kras

# Highlight Adjudicator Hotspots
select engine_hotspots, apo_kras and resi 85+93+96+99+102+108+158
show sticks, engine_hotspots
color yellow, engine_hotspots

# Highlight the massive TYR-96 trigger
select tyr_96_trigger, apo_kras and resi 96
color orange, tyr_96_trigger

# Focus the camera
zoom adagrasib, 5.0

# --- MATHEMATICAL CROSS-VALIDATION ---
# Check for physical clash against the second drug
distance clash_check_ada, tyr_96_trigger, adagrasib

# Print results to console
set label_size, 20
set label_color, black
print("TYR-96 Clash with Adagrasib: ", cmd.distance("tmp2", "tyr_96_trigger", "adagrasib"))

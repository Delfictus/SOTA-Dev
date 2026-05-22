# Load Apo KRAS and the BI-2852 Distal Holo KRAS (6GJ8)
load data/targets/4lpk.pdb, apo_kras
load data/targets/6gj8.pdb, holo_distal

# Clean up crystallization artifacts
remove solvent
remove resn EDO+SO4+GOL+GDP+MG+ACT

# Align the backbones
align apo_kras, holo_distal

# Visual Setup
bg_color white
color gray70, holo_distal
show cartoon, holo_distal

# Highlight BI-2852 (The true distal drug)
select bi_drug, holo_distal and organic
show sticks, bi_drug
color cyan, bi_drug

# Setup your V2 engine prediction
color green, apo_kras
show cartoon, apo_kras

# Highlight the massive TYR-96 trigger
select tyr_96_trigger, apo_kras and resi 96
color orange, tyr_96_trigger
show sticks, tyr_96_trigger

# Focus the camera on the distal pocket
zoom bi_drug, 5.0

# --- THE MOMENT OF TRUTH ---
# Check for physical clash against the distal drug
distance clash_check_distal, tyr_96_trigger, bi_drug

# Print results to console
set label_size, 20
set label_color, black
print("TYR-96 Clash with BI-2852: ", cmd.distance("tmp_distal", "tyr_96_trigger", "bi_drug"))

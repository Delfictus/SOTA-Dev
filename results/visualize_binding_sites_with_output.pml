# PyMOL Script: PRISM-Delta Binding Site Visualization with Text Output
# Load with: pymol /path/to/6vxx_cryptic_sites.pdb visualize_binding_sites_with_output.pml
# Or in headless mode: pymol -cq 6vxx_cryptic_sites.pdb visualize_binding_sites_with_output.pml

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

# ============================================================================
# TEXT OUTPUT: Atomic coordinates and residue information
# ============================================================================

print("")
print("=" * 80)
print("PRISM-DELTA BINDING SITE ATOMIC COORDINATES")
print("=" * 80)
print("")

# --- SITE 1: RBD Cryptic Pocket ---
print("=== SITE 1: RBD CRYPTIC POCKET ===")
print("Residues: 396-397, 511 (Chain A)")
print("")
print("  Residue    Atom      X         Y         Z       B-factor")
print("-" * 65)

python
from pymol import cmd, stored

# Site 1 - CA atoms only for center of mass
stored.site1_atoms = []
cmd.iterate_state(1, "site1_rbd and name CA",
    "stored.site1_atoms.append((chain, resn, resi, name, x, y, z, b))")

for atom in stored.site1_atoms:
    chain, resn, resi, name, x, y, z, b = atom
    print("  %s%-3s %4s    %-4s   %8.3f  %8.3f  %8.3f   %6.2f" %
          (chain, resi, resn, name, x, y, z, b))

# Compute center
if stored.site1_atoms:
    cx = sum(a[4] for a in stored.site1_atoms) / len(stored.site1_atoms)
    cy = sum(a[5] for a in stored.site1_atoms) / len(stored.site1_atoms)
    cz = sum(a[6] for a in stored.site1_atoms) / len(stored.site1_atoms)
    print("")
    print("  Center of Mass: %.3f, %.3f, %.3f" % (cx, cy, cz))
python end

print("")
print("=== SITE 2: NTD DRUGGABLE POCKET ===")
print("Residues: 92-108, 191-205 (Chain A)")
print("")
print("  Residue    Atom      X         Y         Z       B-factor")
print("-" * 65)

python
stored.site2_atoms = []
cmd.iterate_state(1, "site2_ntd and name CA",
    "stored.site2_atoms.append((chain, resn, resi, name, x, y, z, b))")

for atom in stored.site2_atoms:
    chain, resn, resi, name, x, y, z, b = atom
    print("  %s%-3s %4s    %-4s   %8.3f  %8.3f  %8.3f   %6.2f" %
          (chain, resi, resn, name, x, y, z, b))

if stored.site2_atoms:
    cx = sum(a[4] for a in stored.site2_atoms) / len(stored.site2_atoms)
    cy = sum(a[5] for a in stored.site2_atoms) / len(stored.site2_atoms)
    cz = sum(a[6] for a in stored.site2_atoms) / len(stored.site2_atoms)
    print("")
    print("  Center of Mass: %.3f, %.3f, %.3f" % (cx, cy, cz))
python end

print("")
print("=== SITE 3: FUSION PEPTIDE REGION ===")
print("Residues: 816-823, 878-900 (Chain A)")
print("")
print("  Residue    Atom      X         Y         Z       B-factor")
print("-" * 65)

python
stored.site3_atoms = []
cmd.iterate_state(1, "site3_fusion and name CA",
    "stored.site3_atoms.append((chain, resn, resi, name, x, y, z, b))")

for atom in stored.site3_atoms:
    chain, resn, resi, name, x, y, z, b = atom
    print("  %s%-3s %4s    %-4s   %8.3f  %8.3f  %8.3f   %6.2f" %
          (chain, resi, resn, name, x, y, z, b))

if stored.site3_atoms:
    cx = sum(a[4] for a in stored.site3_atoms) / len(stored.site3_atoms)
    cy = sum(a[5] for a in stored.site3_atoms) / len(stored.site3_atoms)
    cz = sum(a[6] for a in stored.site3_atoms) / len(stored.site3_atoms)
    print("")
    print("  Center of Mass: %.3f, %.3f, %.3f" % (cx, cy, cz))
python end

print("")
print("=== ESCAPE MUTATION SITES ===")
print("Residues: 417, 452, 484, 501, 614, 681 (Chain A)")
print("")
print("  Residue    Atom      X         Y         Z       B-factor")
print("-" * 65)

python
stored.escape_atoms = []
cmd.iterate_state(1, "escape_sites and name CA",
    "stored.escape_atoms.append((chain, resn, resi, name, x, y, z, b))")

for atom in stored.escape_atoms:
    chain, resn, resi, name, x, y, z, b = atom
    print("  %s%-3s %4s    %-4s   %8.3f  %8.3f  %8.3f   %6.2f" %
          (chain, resi, resn, name, x, y, z, b))
python end

print("")
print("=== GROUND TRUTH CRYPTIC RESIDUES ===")
print("Residues: 373-379, 503-509, 816-823 (Chain A)")
print("")
print("  Residue    Atom      X         Y         Z       B-factor")
print("-" * 65)

python
stored.gt_atoms = []
cmd.iterate_state(1, "ground_truth and name CA",
    "stored.gt_atoms.append((chain, resn, resi, name, x, y, z, b))")

for atom in stored.gt_atoms:
    chain, resn, resi, name, x, y, z, b = atom
    print("  %s%-3s %4s    %-4s   %8.3f  %8.3f  %8.3f   %6.2f" %
          (chain, resi, resn, name, x, y, z, b))
python end

print("")
print("=" * 80)
print("DOCKING GRID PARAMETERS (AutoDock Vina compatible)")
print("=" * 80)

python
# Generate docking grid parameters for each site
sites = [
    ("Site 1 RBD", stored.site1_atoms, 15),
    ("Site 2 NTD", stored.site2_atoms, 20),
    ("Site 3 Fusion", stored.site3_atoms, 25),
]

for name, atoms, size in sites:
    if atoms:
        cx = sum(a[4] for a in atoms) / len(atoms)
        cy = sum(a[5] for a in atoms) / len(atoms)
        cz = sum(a[6] for a in atoms) / len(atoms)
        print("")
        print("%s:" % name)
        print("  --center_x %.2f --center_y %.2f --center_z %.2f" % (cx, cy, cz))
        print("  --size_x %d --size_y %d --size_z %d" % (size, size, size))
python end

print("")
print("=" * 80)
print("Legend")
print("=" * 80)
print("Blue->White->Red: Cryptic Score (low to high)")
print("Green spheres: Ground truth cryptic residues")
print("Red spheres: Known escape mutation sites")
print("Pink sticks: Top predicted RBD cryptic site")
print("Orange sticks: NTD druggable pocket")
print("Yellow sticks: Fusion peptide region")
print("")

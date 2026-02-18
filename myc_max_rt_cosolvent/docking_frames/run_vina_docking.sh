#!/bin/bash
# AutoDock Vina docking box for MYC-MAX Site 1
# Center: midpoint between MYC and MAX interface centroids
# Size: 24x24x24 Å to cover the full cross-chain pocket

# Prepare receptor (from best docking frame)
# Requires: MGLTools or ADFR suite
prepare_receptor -r site1_dock_frame_best.pdb -o receptor.pdbqt

# Prepare ligand library
# For each ligand in your screening library:
# prepare_ligand -l ligand.mol2 -o ligand.pdbqt

# Docking command
vina \
  --receptor receptor.pdbqt \
  --ligand ligand.pdbqt \
  --center_x 59.49 \
  --center_y 39.71 \
  --center_z 51.03 \
  --size_x 24 \
  --size_y 24 \
  --size_z 24 \
  --exhaustiveness 32 \
  --num_modes 20 \
  --out docking_results.pdbqt \
  --log docking_log.txt

# For ensemble docking (all top frames):
# for frame in site1_dock_frame_*.pdb; do
#   prepare_receptor -r $frame -o ${frame%.pdb}.pdbqt
#   vina --receptor ${frame%.pdb}.pdbqt --ligand ligand.pdbqt \
#     --center_x 59.49 --center_y 39.71 --center_z 51.03 \
#     --size_x 24 --size_y 24 --size_z 24 \
#     --exhaustiveness 32 --num_modes 10 \
#     --out docking_${frame%.pdb}.pdbqt
# done

echo "Docking box center: (59.49, 39.71, 51.03)"
echo "Box size: 24 x 24 x 24 Å"
echo ""
echo "POST-DOCKING FILTER:"
echo "  Keep only hits with ANY heavy atom within 4.0 Å of:"
echo "    MAX I85 (topo 150, chain B)"
echo "    MAX L88 (topo 153, chain B)"
echo "  These form the hydrophobic floor of the zipper pocket."

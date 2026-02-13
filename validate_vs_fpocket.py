import json
import os
import numpy as np
import glob
import warnings
from Bio import PDB

# Suppress PDB warnings
warnings.simplefilter('ignore', PDB.PDBExceptions.PDBConstructionWarning)

# --- CONFIGURATION ---
# 1. Find your latest Prism4D results
try:
    results_dirs = [d for d in os.listdir("e2e_validation_test") if "results_parallel" in d]
    results_dirs.sort(reverse=True)
    PRISM_JSON = f"e2e_validation_test/{results_dirs[0]}/all_results.json"
    print(f"🔹 Prism4D Data: {PRISM_JSON}")
except:
    print("❌ Could not find Prism4D results. Run the parallel batch first.")
    exit(1)

# 2. Fpocket Output Directory
FPOCKET_DIR = "e2e_validation_test/fpocket_baseline"
# ---------------------

def get_fpocket_data(pdb_code):
    """Parses Fpocket output to find pocket centers and residues."""
    pocket_dir = f"{FPOCKET_DIR}/{pdb_code}_out/pockets"
    if not os.path.exists(pocket_dir):
        return []

    pockets = []
    parser = PDB.PDBParser(QUIET=True)
    
    # Fpocket ranks pockets by score (pocket1 is best)
    # We look at top 3 Fpocket predictions
    pocket_files = sorted(glob.glob(f"{pocket_dir}/pocket*_atm.pdb"))[:3]
    
    for p_file in pocket_files:
        struct = parser.get_structure("pocket", p_file)
        atoms = []
        residues = set()
        for atom in struct.get_atoms():
            atoms.append(atom.get_coord())
            # Fpocket preserves PDB residue IDs
            residues.add(atom.get_parent().id[1]) 
            
        if atoms:
            centroid = np.mean(atoms, axis=0)
            pockets.append({
                "centroid": centroid,
                "residues": residues,
                "file": os.path.basename(p_file)
            })
    return pockets

def main():
    if not os.path.exists(PRISM_JSON):
        print(f"❌ Error: File not found: {PRISM_JSON}")
        return

    with open(PRISM_JSON, 'r') as f:
        prism_data = json.load(f)

    print(f"\n{'TARGET':<8} | {'PRISM RANK':<10} | {'FPOCKET MATCH?':<20} | {'DETAILS'}")
    print("-" * 80)

    for entry in prism_data:
        target = entry['structure'].replace(".topology", "").replace("_clean", "").replace("_raw", "")
        
        # Get Fpocket Baseline
        fpockets = get_fpocket_data(target)
        if not fpockets:
            print(f"{target:<8} | N/A        | ❌ No Fpocket Data  | Run fpocket first")
            continue

        # Compare Top 3 Prism Sites to Top 3 Fpocket Sites
        best_prism_site = None
        best_match_type = "❌ NO MATCH"
        best_dist = 999.9
        
        # Iterate Prism Top 3
        for i, site in enumerate(entry['sites'][:3]):
            prism_center = np.array(site['centroid'])
            prism_res = set(site.get('residue_ids', [])) 
            
            for fp in fpockets:
                dist = np.linalg.norm(prism_center - fp['centroid'])
                
                # Calculate Residue Overlap (Jaccard Index)
                if len(prism_res) > 0 and len(fp['residues']) > 0:
                    intersection = len(prism_res.intersection(fp['residues']))
                    union = len(prism_res.union(fp['residues']))
                    iou = intersection / union
                else:
                    iou = 0.0

                if dist < 6.0:
                    match_type = f"✅ VALIDATED (Dist: {dist:.1f}Å)"
                    if iou > 0.3:
                        match_type += f" (IoU: {iou:.2f})"
                    
                    if dist < best_dist:
                        best_dist = dist
                        best_match_type = match_type
                        best_prism_site = i + 1

        if best_prism_site is not None:
            print(f"{target:<8} | Site #{best_prism_site:<5} | {best_match_type:<20} | Matches Fpocket Baseline")
        else:
            print(f"{target:<8} | Top 3      | ⚠️  DIVERGENCE       | Prism found sites far from Fpocket")

    print("-" * 80)

if __name__ == "__main__":
    main()

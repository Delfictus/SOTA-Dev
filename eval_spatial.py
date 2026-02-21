import os
import json
import glob
import subprocess
import numpy as np
from collections import defaultdict

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
PRISM_DIR = "/tmp/dligsite_spatial_20260220_145448" # Replace if your directory changed
PREP_DIR = "e2e_validation_test/prep"

# Map Apo targets to known Holo PDBs.
# The script will auto-detect the largest ligand in the Holo structure.
TARGET_PAIRS = {
    "1a4q": "1a4r",
    "1ade": "1gim",     # AdSS
    "1bj4": "1bjv",
    "1btl": "1btm",
    "1ere": "1err",     # Estrogen Receptor Holo
    "1g1f": "1g1g",
    "1hhp": "1hvr",     # HIV Protease Holo
    "1w50": "1w51",     # BACE1 Holo
    "3k5v": "3k5u",
    "4obe_mono": "4obe" # MDM2
}

IGNORED_LIGANDS = {'HOH', 'SO4', 'PO4', 'GOL', 'EDO', 'DMS', 'CL', 'NA', 'MG', 'ACT', 'PEG'}

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================
def get_ca_atoms(pdb_path, chain='A'):
    """Extracts C-alpha coordinates for alignment."""
    cas = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[21] == chain:
                resi = int(line[22:26])
                resn = line[17:20].strip()
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                cas[resi] = (resn, xyz)
    return cas

def extract_largest_ligand(pdb_path):
    """Finds the largest biologically relevant ligand in the holo PDB."""
    lig_atoms = defaultdict(list)
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('HETATM'):
                resn = line[17:20].strip()
                if resn in IGNORED_LIGANDS: continue
                chain = line[21]
                resi = line[22:26].strip()
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                lig_atoms[(resn, chain, resi)].append(xyz)
    
    if not lig_atoms: return None, None
    
    # Sort by number of heavy atoms to grab the main drug/ligand
    largest_lig_key = max(lig_atoms.keys(), key=lambda k: len(lig_atoms[k]))
    return largest_lig_key[0], np.array(lig_atoms[largest_lig_key])

def align_and_transform(apo_path, holo_path, holo_lig_coords):
    """Aligns Holo to Apo using SVD and transforms ligand coordinates."""
    hcas = get_ca_atoms(holo_path, 'A')
    pcas = get_ca_atoms(apo_path, 'A')
    
    if not hcas or not pcas:
        # Fallback to chain B if A is empty
        hcas = get_ca_atoms(holo_path, 'B') if not hcas else hcas
        pcas = get_ca_atoms(apo_path, 'B') if not pcas else pcas

    best_m, best_off = 0, 0
    for off in range(-50, 51):
        m = sum(1 for r, (n, _) in hcas.items() if (r+off) in pcas and pcas[r+off][0] == n)
        if m > best_m: 
            best_m, best_off = m, off
            
    if best_m < 20:
        return holo_lig_coords # Fallback: assume already aligned

    P, Q = [], []
    for r, (n, c) in hcas.items():
        if (r+best_off) in pcas and pcas[r+best_off][0] == n:
            P.append(c)
            Q.append(pcas[r+best_off][1])
            
    P, Q = np.array(P), np.array(Q)
    cp, cq = P.mean(0), Q.mean(0)
    H = (P-cp).T @ (Q-cq)
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    t = cq - R @ cp
    
    # Transform ligand coords to Apo space
    aligned_lig = np.array([R @ c + t for c in holo_lig_coords])
    return aligned_lig

# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================
print(f"{'Target':<10} | {'SNN Rank 1 ID':<13} | {'Drug.Score':<10} | {'DCC (Å)':<8} | {'DCA (Å)':<8} | {'Status'}")
print("-" * 75)

success_count = 0
total = 0

for apo_target, holo_target in TARGET_PAIRS.items():
    # 1. Fetch JSON and top pocket
    json_path = os.path.join(PRISM_DIR, apo_target, f"{apo_target}.binding_sites.json")
    if not os.path.exists(json_path): continue
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    sites = data.get("sites", [])
    if not sites: continue
        
    # **CRITICAL: Sort by SNN Druggability Score to override Volume Mega-Pockets**
    sites.sort(key=lambda s: s.get("druggability", 0.0), reverse=True)
    top_site = sites[0]
    prism_centroid = np.array(top_site["centroid"])
    
    # 2. Download and Extract Holo Ligand
    holo_pdb_path = f"/tmp/pdb_cache/{holo_target}.pdb"
    if not os.path.exists(holo_pdb_path):
        os.makedirs("/tmp/pdb_cache", exist_ok=True)
        subprocess.run(['wget', '-q', f'https://files.rcsb.org/download/{holo_target}.pdb', '-O', holo_pdb_path])
        
    lig_name, lig_coords = extract_largest_ligand(holo_pdb_path)
    if lig_coords is None: continue
        
    # 3. Align Ligand to Apo space
    apo_pdb_path = os.path.join(PREP_DIR, f"{apo_target}.topology.json".replace(".topology.json", ".pdb"))
    if os.path.exists(apo_pdb_path):
        aligned_lig_coords = align_and_transform(apo_pdb_path, holo_pdb_path, lig_coords)
    else:
        aligned_lig_coords = lig_coords # Fallback if Apo PDB missing
        
    # 4. Calculate Spatial Metrics
    ligand_centroid = np.mean(aligned_lig_coords, axis=0)
    
    # DCC: Distance between Centers
    dcc_dist = np.linalg.norm(prism_centroid - ligand_centroid)
    
    # DCA: Distance to Closest Atom (Did the pocket touch the ligand?)
    distances_to_atoms = np.linalg.norm(aligned_lig_coords - prism_centroid, axis=1)
    dca_dist = np.min(distances_to_atoms)
    
    # 5. Determine Success (Industry standards: DCC < 8.0 OR DCA < 4.0)
    status = "FAIL ✗"
    if dcc_dist <= 8.0 or dca_dist <= 4.0:
        status = "SUCCESS ✓"
        success_count += 1
    total += 1
    
    print(f"{apo_target:<10} | Site {top_site['id']:<8} | {top_site.get('druggability', 0.0):<10.3f} | {dcc_dist:<8.1f} | {dca_dist:<8.1f} | {status} ({lig_name})")

print("-" * 75)
print(f"Final SNN Druggability Success Rate: {success_count}/{total} ({(success_count/total)*100:.0f}%)")

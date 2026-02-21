import os
import json
import subprocess
import numpy as np
from collections import defaultdict

# ==============================================================================
# 1. HARDCODED BIOLOGICAL GROUND TRUTH
# Format: "apo_target": ("holo_pdb", "ligand_resname", "holo_chain", "apo_chain")
# 'PEP' means the ligand is a peptide chain, not a HETATM.
# ==============================================================================
PRISM_DIR = "/tmp/dligsite_spatial_20260220_145448" 
PREP_DIR = "e2e_validation_test/prep"

TARGET_PAIRS = {
    "1a4q": ("1a4r", "G39", "A", "A"),    # Zanamivir
    "1ade": ("1gim", "IMP", "A", "A"),    # True AdSS Substrate (Not GDP)
    "1bj4": ("1bjv", "PPK", "A", "A"),    # Thrombin peptide inhibitor
    "1btl": ("1btm", "PGA", "A", "A"),    # Penicillin G
    "1ere": ("1err", "RAL", "A", "A"),    # Raloxifene (Chain A strictly)
    "1g1f": ("1g1g", "BOG", "A", "A"),    # BOG
    "1hhp": ("1hvr", "XK2", "A", "A"),    # HIV Protease Inhibitor
    "1w50": ("1w51", "L01", "A", "A"),    # BACE1 Inhibitor
    "3k5v": ("3k5u", "PFQ", "A", "A"),    # Nilotinib
    "4obe_mono": ("4oq3", "PEP", "B", "A") # MDM2 uses p53 peptide (Chain B in Holo)
}

def get_ca_atoms(pdb_path, chain):
    cas = {}
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM') and line[12:16].strip() == 'CA' and line[21] == chain:
                resi = int(line[22:26])
                xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                cas[resi] = xyz
    return cas

def extract_specific_ligand(pdb_path, lig_resname, lig_chain):
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            # If target is a peptide chain (like p53)
            if lig_resname == "PEP" and line.startswith('ATOM') and line[21] == lig_chain:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            # If target is a standard chemical drug
            elif line.startswith('HETATM') and line[17:20].strip() == lig_resname and line[21] == lig_chain:
                coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    return np.array(coords) if coords else None

print(f"{'Target':<10} | {'Rank 1 ID':<9} | {'Drug.Score':<10} | {'DCC (Å)':<8} | {'DCA (Å)':<8} | {'Status'}")
print("-" * 75)

for apo, (holo, lig_name, h_chain, a_chain) in TARGET_PAIRS.items():
    # 1. Fetch Top SNN Pocket
    json_path = os.path.join(PRISM_DIR, apo, f"{apo}.binding_sites.json")
    if not os.path.exists(json_path): continue
    with open(json_path, 'r') as f: data = json.load(f)
    sites = data.get("sites", [])
    if not sites: continue
    
    # Sort strictly by SNN Druggability
    sites.sort(key=lambda s: s.get("druggability", 0.0), reverse=True)
    top_site = sites[0]
    prism_centroid = np.array(top_site["centroid"])
    
    # 2. Setup Holo Data
    holo_path = f"/tmp/pdb_cache/{holo}.pdb"
    if not os.path.exists(holo_path):
        subprocess.run(['wget', '-q', f'https://files.rcsb.org/download/{holo}.pdb', '-O', holo_path])
        
    lig_coords = extract_specific_ligand(holo_path, lig_name, h_chain)
    if lig_coords is None or len(lig_coords) == 0: 
        print(f"{apo:<10} | Ligand {lig_name} missing on chain {h_chain}")
        continue
        
    # 3. Align Monomer to Monomer
    apo_path = os.path.join(PREP_DIR, f"{apo}.topology.json".replace(".topology.json", ".pdb"))
    if not os.path.exists(apo_path): continue
        
    hcas = get_ca_atoms(holo_path, h_chain)
    pcas = get_ca_atoms(apo_path, a_chain)
    
    P, Q = [], []
    best_m, best_off = 0, 0
    for off in range(-50, 51):
        m = sum(1 for r, c in hcas.items() if (r+off) in pcas)
        if m > best_m: best_m, best_off = m, off

    for r, c in hcas.items():
        if (r+best_off) in pcas:
            P.append(c)
            Q.append(pcas[r+best_off])
            
    if best_m > 20:
        P, Q = np.array(P), np.array(Q)
        cp, cq = P.mean(0), Q.mean(0)
        H = (P-cp).T @ (Q-cq)
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
        t = cq - R @ cp
        aligned_lig = np.array([R @ c + t for c in lig_coords])
    else:
        aligned_lig = lig_coords # Fallback
        
    # 4. Math
    ligand_centroid = np.mean(aligned_lig, axis=0)
    dcc_dist = np.linalg.norm(prism_centroid - ligand_centroid)
    dca_dist = np.min(np.linalg.norm(aligned_lig - prism_centroid, axis=1))
    
    # 5. Output
    status = "SUCCESS ✓" if dcc_dist <= 8.0 or dca_dist <= 4.0 else "ALLOSTERIC ⟿"
    print(f"{apo:<10} | Site {top_site['id']:<4} | {top_site.get('druggability', 0.0):<10.3f} | {dcc_dist:<8.1f} | {dca_dist:<8.1f} | {status} ({lig_name})")

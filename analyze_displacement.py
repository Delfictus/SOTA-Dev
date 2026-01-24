import math
import sys
import os

def parse_pdb_ca(filename):
    """Parses PDB file and returns a dict of { (chain, res_id): (x, y, z, res_name) } for CA atoms."""
    atoms = {}
    try:
        with open(filename, 'r') as f:
            for line in f:
                if line.startswith("ATOM"):
                    atom_name = line[12:16].strip()
                    # We track Alpha Carbons (CA) to represent the residue backbone
                    if atom_name == "CA":
                        res_name = line[17:20].strip()
                        chain = line[21]
                        try:
                            res_id = int(line[22:26])
                        except ValueError:
                            continue # Skip malformed lines
                        
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        atoms[(chain, res_id)] = (x, y, z, res_name)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {filename}")
        sys.exit(1)
    return atoms

def calculate_displacements(file1, file2):
    print(f"📂 Loading Initial: {file1}")
    atoms1 = parse_pdb_ca(file1)
    print(f"📂 Loading Relaxed: {file2}")
    atoms2 = parse_pdb_ca(file2)
    
    displacements = []
    
    print(f"🧮 Calculating displacement vectors for {len(atoms1)} residues...")
    
    for key, val1 in atoms1.items():
        if key in atoms2:
            val2 = atoms2[key]
            # Euclidean distance formula: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
            dx = val1[0] - val2[0]
            dy = val1[1] - val2[1]
            dz = val1[2] - val2[2]
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            chain, res_id = key
            res_name = val1[3]
            displacements.append({
                'chain': chain,
                'res_id': res_id,
                'res_name': res_name,
                'displacement': dist
            })
            
    return displacements

if __name__ == "__main__":
    # Hardcoded paths based on your project structure
    initial_pdb = "data/raw/6VXX.pdb"
    final_pdb = "data/processed/covid_relaxed.pdb"
    
    if not os.path.exists(initial_pdb) or not os.path.exists(final_pdb):
        print("❌ Error: Could not find input files.")
        print(f"Looking for: {initial_pdb}")
        print(f"Looking for: {final_pdb}")
        sys.exit(1)
    
    data = calculate_displacements(initial_pdb, final_pdb)
    
    # Sort by displacement descending (Highest movement first)
    data.sort(key=lambda x: x['displacement'], reverse=True)
    
    print("\n" + "="*65)
    print("🦠 PRISM-ZERO: TOP 20 HIGH-ENERGY DISPLACEMENTS (The 'Breathing' Sites)")
    print("="*65)
    print(f"{'Rank':<5} | {'Residue':<10} | {'Chain':<5} | {'ID':<5} | {'Displacement (Å)':<15}")
    print("-" * 65)
    
    for i in range(min(20, len(data))):
        item = data[i]
        # Highlight the Cryptic Epitope range (369-392) with a star
        marker = ""
        if 369 <= item['res_id'] <= 392:
            marker = "⭐ (TARGET)"
        elif 490 <= item['res_id'] <= 515:
            marker = "🟢 (RBM TIP)"
            
        print(f"{i+1:<5} | {item['res_name']:<10} | {item['chain']:<5} | {item['res_id']:<5} | {item['displacement']:.4f} {marker}")
        
    print("="*65)

#!/usr/bin/env python3
import json
import sys

def create_topology(pdb_file, prmtop_file, output_json):
    atoms = []
    chains = set()
    
    print(f"Parsing PDB: {pdb_file}")
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_id = int(line[6:11].strip())
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                chain_id = line[21:22].strip() if line[21:22].strip() else 'A'
                chains.add(chain_id)
                res_num = int(line[22:26].strip()) if line[22:26].strip() else 1
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                element = atom_name[0]
                
                atoms.append({
                    'id': len(atoms),
                    'name': atom_name,
                    'element': element,
                    'residue': res_name,
                    'residue_num': res_num,
                    'chain_id': chain_id,
                    'x': x, 'y': y, 'z': z
                })
    
    bonds = []
    with open(prmtop_file, 'r') as f:
        content = f.read()
    
    if '%FLAG BONDS_INC_HYDROGEN' in content:
        section = content.split('%FLAG BONDS_INC_HYDROGEN')[1].split('%FLAG')[0]
        data = [int(x) for x in section.split() if x.strip().lstrip('-').isdigit()]
        for i in range(0, len(data), 3):
            if i + 1 < len(data):
                bonds.append([data[i]//3, data[i+1]//3])
    
    if '%FLAG BONDS_WITHOUT_HYDROGEN' in content:
        section = content.split('%FLAG BONDS_WITHOUT_HYDROGEN')[1].split('%FLAG')[0]
        data = [int(x) for x in section.split() if x.strip().lstrip('-').isdigit()]
        for i in range(0, len(data), 3):
            if i + 1 < len(data):
                bonds.append([data[i]//3, data[i+1]//3])
    
    topology = {
        'n_atoms': len(atoms),
        'atoms': atoms,
        'bonds': bonds,
        'angles': [],
        'dihedrals': [],
        'h_clusters': [],
        'chains': list(chains),
        'box_vectors': None,
        'force_field': 'amber19SB',
        'implicit_solvent': 'GBn2',
        'metadata': {
            'heavy_atoms': len([a for a in atoms if a['element'] != 'H']),
            'hydrogen_atoms': len([a for a in atoms if a['element'] == 'H']),
            'total_residues': len(set((a['chain_id'], a['residue_num']) for a in atoms)),
            'chains': len(chains)
        }
    }
    
    print(f"\n=== Topology Summary ===")
    print(f"Total atoms: {topology['n_atoms']}")
    print(f"Heavy atoms: {topology['metadata']['heavy_atoms']}")
    print(f"Hydrogens: {topology['metadata']['hydrogen_atoms']}")
    print(f"Bonds: {len(bonds)}")
    print(f"Chains: {list(chains)}")
    
    with open(output_json, 'w') as f:
        json.dump(topology, f, indent=2)
    
    print(f"\n✅ Saved to: {output_json}")

if __name__ == "__main__":
    create_topology(sys.argv[1], sys.argv[2], sys.argv[3])

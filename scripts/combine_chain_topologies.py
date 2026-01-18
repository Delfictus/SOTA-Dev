#!/usr/bin/env python3
"""
Combine individual chain topologies into a single multi-chain topology.
Handles index offsetting for bonds, angles, dihedrals, etc.
"""

import json
import sys
import os

def combine_topologies(chain_files, output_file):
    """Combine multiple chain topology files into one."""
    
    combined = {
        'n_atoms': 0,
        'n_residues': 0,
        'n_chains': 0,
        'positions': [],
        'masses': [],
        'charges': [],
        'lj_params': [],
        'atom_names': [],
        'elements': [],
        'residue_names': [],
        'residue_ids': [],
        'chain_ids': [],
        'bonds': [],
        'angles': [],
        'dihedrals': [],
        'exclusions': [],
        'h_clusters': [],
        'ca_indices': [],
    }
    
    atom_offset = 0
    residue_offset = 0
    
    for chain_file in chain_files:
        print(f"  Loading {os.path.basename(chain_file)}...")
        with open(chain_file) as f:
            chain = json.load(f)
        
        n_atoms = chain['n_atoms']
        n_residues = chain.get('n_residues', 0)
        
        # Append simple arrays
        combined['positions'].extend(chain.get('positions', []))
        combined['masses'].extend(chain.get('masses', []))
        combined['charges'].extend(chain.get('charges', []))
        combined['lj_params'].extend(chain.get('lj_params', []))
        combined['atom_names'].extend(chain.get('atom_names', []))
        combined['elements'].extend(chain.get('elements', []))
        combined['residue_names'].extend(chain.get('residue_names', []))
        combined['residue_ids'].extend(chain.get('residue_ids', []))
        combined['chain_ids'].extend(chain.get('chain_ids', []))
        
        # Offset bonds
        for bond in chain.get('bonds', []):
            new_bond = {
                'i': bond['i'] + atom_offset,
                'j': bond['j'] + atom_offset,
                'r0': bond['r0'],
                'k': bond['k']
            }
            combined['bonds'].append(new_bond)
        
        # Offset angles
        for angle in chain.get('angles', []):
            new_angle = {
                'i': angle['i'] + atom_offset,
                'j': angle['j'] + atom_offset,
                'k_idx': angle['k_idx'] + atom_offset,
                'theta0': angle['theta0'],
                'force_k': angle['force_k']
            }
            combined['angles'].append(new_angle)
        
        # Offset dihedrals
        for dih in chain.get('dihedrals', []):
            new_dih = {
                'i': dih['i'] + atom_offset,
                'j': dih['j'] + atom_offset,
                'k_idx': dih['k_idx'] + atom_offset,
                'l': dih['l'] + atom_offset,
                'periodicity': dih['periodicity'],
                'phase': dih['phase'],
                'force_k': dih['force_k']
            }
            combined['dihedrals'].append(new_dih)
        
        # Offset exclusions
        for excl_list in chain.get('exclusions', []):
            new_excl = [idx + atom_offset for idx in excl_list]
            combined['exclusions'].append(new_excl)
        
        # Offset h_clusters (uses 'central_atom' and 'hydrogen_atoms')
        for cluster in chain.get('h_clusters', []):
            # Offset hydrogen atom indices (skip -1 placeholders)
            new_h_atoms = []
            for h_idx in cluster['hydrogen_atoms']:
                if h_idx >= 0:
                    new_h_atoms.append(h_idx + atom_offset)
                else:
                    new_h_atoms.append(-1)
            
            new_cluster = {
                'type': cluster['type'],
                'central_atom': cluster['central_atom'] + atom_offset,
                'hydrogen_atoms': new_h_atoms,
                'bond_lengths': cluster['bond_lengths'],
                'n_hydrogens': cluster['n_hydrogens'],
                'inv_mass_central': cluster['inv_mass_central'],
                'inv_mass_h': cluster['inv_mass_h']
            }
            combined['h_clusters'].append(new_cluster)
        
        # Offset CA indices
        for ca in chain.get('ca_indices', []):
            combined['ca_indices'].append(ca + atom_offset)
        
        # Update totals
        combined['n_atoms'] += n_atoms
        combined['n_residues'] += n_residues
        combined['n_chains'] += 1
        
        atom_offset += n_atoms
        residue_offset += n_residues
        
        print(f"    Added {n_atoms} atoms, total now: {combined['n_atoms']}")
    
    # Add metadata
    combined['source_pdb'] = 'combined_chains'
    combined['water_oxygens'] = []
    
    # Write output
    with open(output_file, 'w') as f:
        json.dump(combined, f)
    
    size_mb = os.path.getsize(output_file) / 1024 / 1024
    print(f"  Written {output_file} ({size_mb:.2f} MB)")
    
    return combined


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: combine_chain_topologies.py output.json chain1.json chain2.json ...")
        sys.exit(1)
    
    output = sys.argv[1]
    chains = sys.argv[2:]
    
    combine_topologies(chains, output)

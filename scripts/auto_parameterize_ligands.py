#!/usr/bin/env python3
"""
Auto-parameterization for PRISM-4D: Handle any PDB with unknown ligands
"""
import subprocess
import os
from pathlib import Path

def auto_parameterize_pdb(input_pdb, output_dir):
    """Automatically generate parameters for all non-standard residues"""
    
    # 1. Identify non-standard residues
    standard_aa = {'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 
                   'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'}
    standard_dna = {'DA', 'DT', 'DG', 'DC', 'A', 'T', 'G', 'C', 'U'}
    
    non_standard = set()
    
    with open(input_pdb) as f:
        for line in f:
            if line.startswith('HETATM') or line.startswith('ATOM'):
                res_name = line[17:20].strip()
                if res_name not in standard_aa and res_name not in standard_dna and res_name != 'HOH':
                    non_standard.add(res_name)
    
    print(f"Found non-standard residues: {non_standard}")
    
    # 2. For each non-standard residue, generate parameters
    param_files = []
    for res_name in non_standard:
        try:
            print(f"Generating parameters for {res_name}...")
            
            # Extract ligand to separate PDB
            ligand_pdb = f"{output_dir}/{res_name}_ligand.pdb"
            with open(input_pdb) as f_in, open(ligand_pdb, 'w') as f_out:
                for line in f_in:
                    if line.startswith('HETATM') and line[17:20].strip() == res_name:
                        f_out.write(line)
            
            # Generate mol2 and frcmod files using antechamber
            mol2_file = f"{output_dir}/{res_name}.mol2"
            frcmod_file = f"{output_dir}/{res_name}.frcmod"
            
            # Use antechamber for automatic parameterization
            subprocess.run([
                'antechamber', '-i', ligand_pdb, '-fi', 'pdb', 
                '-o', mol2_file, '-fo', 'mol2', '-c', 'bcc', '-s', '2'
            ], check=True)
            
            subprocess.run([
                'parmchk2', '-i', mol2_file, '-f', 'mol2', '-o', frcmod_file
            ], check=True)
            
            param_files.append((res_name, mol2_file, frcmod_file))
            print(f"✅ Generated parameters for {res_name}")
            
        except Exception as e:
            print(f"❌ Failed to parameterize {res_name}: {e}")
            print(f"   Will exclude {res_name} from final structure")
    
    # 3. Create LEaP script with automatic parameter loading
    leap_script = f"{output_dir}/auto_parameterize.leap"
    with open(leap_script, 'w') as f:
        f.write("source leaprc.protein.ff19SB\n")
        f.write("source leaprc.gaff2\n\n")
        
        # Load all generated parameters
        for res_name, mol2_file, frcmod_file in param_files:
            f.write(f"# Load parameters for {res_name}\n")
            f.write(f"{res_name} = loadMol2 {mol2_file}\n")
            f.write(f"loadAmberParams {frcmod_file}\n\n")
        
        f.write(f"# Load main structure\n")
        f.write(f"mol = loadPDB {input_pdb}\n")
        f.write("addH mol\n")
        f.write("check mol\n\n")
        
        f.write(f"# Save complete parameterized system\n")
        f.write(f"saveAmberParm mol {output_dir}/complete.prmtop {output_dir}/complete.inpcrd\n")
        f.write(f"savePDB mol {output_dir}/complete_with_H.pdb\n")
        f.write("quit\n")
    
    return leap_script

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python3 auto_parameterize_ligands.py <input_pdb> <output_dir>")
        sys.exit(1)
    
    input_pdb = sys.argv[1]
    output_dir = sys.argv[2]
    Path(output_dir).mkdir(exist_ok=True)
    
    leap_script = auto_parameterize_pdb(input_pdb, output_dir)
    print(f"\nGenerated LEaP script: {leap_script}")
    print(f"Run: tleap -f {leap_script}")

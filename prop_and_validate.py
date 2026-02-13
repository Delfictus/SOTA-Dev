import sys
import os
import requests
import difflib
from pdbfixer import PDBFixer
from openmm.app import PDBFile, Modeller

# 3-to-1 AA mapping for sequence extraction
AA_MAP = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'HID': 'H', 'HIE': 'H', 'HIP': 'H', 'CYX': 'C', 'ASH': 'D', # Protonated variants
    'GLH': 'E', 'LYN': 'K' 
}

def prep_structure(input_pdb, output_pdb, ph=7.4):
    """
    Runs the P.R.O.P. protocol: Purify, Resolve, Occupancy, Protonate.
    """
    print(f"🔧 [PROP] Processing {input_pdb}...")
    
    # 1. Load Structure
    fixer = PDBFixer(filename=input_pdb)
    
    # 2. Find Missing Residues (The Sequence Gap Fixer)
    fixer.findMissingResidues()
    
    # 3. Find & Replace Non-Standard Residues (e.g. MSE -> MET)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    
    # 4. Remove Heterogens (Purify artifacts like Glycerol, Sulfates)
    # We keep water only if explicitly needed; usually remove for docking
    fixer.removeHeterogens(keepWater=False)
    
    # 5. Add Missing Heavy Atoms (Fixes broken side-chains)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    
    # 6. Add Hydrogens (Protonate for pH 7.4)
    fixer.addMissingHydrogens(ph)
    
    # 7. Write Canonical PDB
    with open(output_pdb, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    
    print(f"✅ [PROP] Saved canonical structure to {output_pdb}")
    return output_pdb

def get_pdb_sequence(pdb_file):
    """Extracts the 1-letter amino acid sequence from a PDB file."""
    pdb = PDBFile(pdb_file)
    sequence = ""
    # We only take chain 0 (usually A) for validation. 
    # Modify if your target is a complex.
    chain = list(pdb.topology.chains())[0]
    
    for residue in chain.residues():
        if residue.name in AA_MAP:
            sequence += AA_MAP[residue.name]
        else:
            sequence += 'X' # Unknown residue
            
    return sequence

def get_uniprot_sequence(uniprot_id):
    """Fetches the official canonical sequence from UniProt."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise ValueError(f"Could not fetch UniProt ID {uniprot_id}")
    
    # Parse FASTA (skip first line, join the rest)
    lines = response.text.splitlines()
    sequence = "".join(lines[1:])
    return sequence

def check_identity_drift(pdb_seq, ref_seq, threshold=0.95):
    """Compares PDB sequence to UniProt sequence."""
    matcher = difflib.SequenceMatcher(None, pdb_seq, ref_seq)
    identity = matcher.ratio()
    
    print(f"\n🔍 [VALIDATION] Sequence Identity Check:")
    print(f"   - PDB Length:     {len(pdb_seq)} residues")
    print(f"   - UniProt Length: {len(ref_seq)} residues")
    print(f"   - Identity Score: {identity*100:.2f}%")
    
    if identity < threshold:
        print("\n⚠️  WARNING: High Identity Drift Detected!")
        print("   This PDB contains significant mutations, gaps, or chimeric fusions.")
        print("   Do not use for 'SOTA' viral escape prediction without manual review.")
        
        # Print first mismatch for debugging
        for i, (a, b) in enumerate(zip(pdb_seq, ref_seq)):
            if a != b:
                print(f"   - Mismatch at residue {i+1}: PDB={a}, Ref={b}")
                break
    else:
        print("\n✅ [VALIDATION] PASS. Structure matches sequence definition.")

# --- Usage Example ---
if __name__ == "__main__":
    # Change these to your actual target
    INPUT_FILE = "1btl.pdb"          # Your raw file
    OUTPUT_FILE = "1btl_clean.pdb"   # The canonical output
    UNIPROT_ID = "P00808"            # 1BTL is Beta-lactamase (UniProt: P00808)

    try:
        # Step 1: PROP (Clean & Fix)
        prep_structure(INPUT_FILE, OUTPUT_FILE)
        
        # Step 2: Extract Sequence from Cleaned PDB
        pdb_seq = get_pdb_sequence(OUTPUT_FILE)
        
        # Step 3: Fetch Ground Truth
        print(f"⬇️  [FETCH] Downloading reference for {UNIPROT_ID}...")
        ref_seq = get_uniprot_sequence(UNIPROT_ID)
        
        # Step 4: Validate
        check_identity_drift(pdb_seq, ref_seq)
        
    except Exception as e:
        print(f"❌ Error: {e}")

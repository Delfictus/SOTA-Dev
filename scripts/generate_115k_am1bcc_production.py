import os
import time
import logging
import multiprocessing as mp
import polars as pl
from pathlib import Path

logging.getLogger("openff").setLevel(logging.ERROR)

def process_molecule(row):
    anchor_id, smiles = row
    try:
        from openff.toolkit.topology import Molecule
        import warnings
        warnings.filterwarnings("ignore")
        
        mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
        mol.generate_conformers(n_conformers=1)
        mol.assign_partial_charges(partial_charge_method='am1bcc')
        
        # Extract coordinates and charges
        conf = mol.conformers[0].m_as(mol.conformers[0].units)
        charges = mol.partial_charges.m_as(mol.partial_charges.units)
        
        return {
            "anchor_id": anchor_id,
            "smiles": smiles,
            "status": "success",
            "coordinates_json": str(conf.tolist()),
            "am1bcc_charges_json": str(charges.tolist())
        }
    except Exception as e:
        return {"anchor_id": anchor_id, "smiles": smiles, "status": "failed", "error": str(e)}

def main():
    csv_path = Path("campaigns/glp1r_aleniglipron/track_a_generative/115k_curated_anchors.csv")
    out_dir = Path("campaigns/glp1r_aleniglipron/track_a_generative/anchors_3d")
    
    df = pl.read_csv(csv_path)
    rows = list(zip(df["anchor_id"].to_list(), df["smiles"].to_list()))
    
    chunk_size = 1000
    chunks = [rows[i:i + chunk_size] for i in range(0, len(rows), chunk_size)]
    
    workers = max(1, (os.cpu_count() or 1) - 2)
    print(f"Starting 115k AM1-BCC Production Run with {workers} workers...")
    
    with mp.Pool(processes=workers) as pool:
        for i, chunk in enumerate(chunks):
            chunk_file = out_dir / f"chunk_{i:04d}.parquet"
            if chunk_file.exists():
                print(f"Chunk {i:04d} already exists. Skipping...")
                continue
                
            print(f"Processing Chunk {i:04d} / {len(chunks)}...")
            results = pool.map(process_molecule, chunk)
            
            # Save chunk
            chunk_df = pl.DataFrame(results)
            chunk_df.write_parquet(chunk_file)
            print(f"Saved {chunk_file.name}")

if __name__ == "__main__":
    main()

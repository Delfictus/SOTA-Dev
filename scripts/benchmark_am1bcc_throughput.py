import os
import time
import logging
import multiprocessing as mp
import polars as pl
from pathlib import Path

logging.getLogger("openff").setLevel(logging.ERROR)

def process_molecule(smiles: str) -> float:
    start = time.perf_counter()
    try:
        from openff.toolkit.topology import Molecule
        import warnings
        warnings.filterwarnings("ignore")
        mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
        mol.generate_conformers(n_conformers=1)
        mol.assign_partial_charges(partial_charge_method='am1bcc')
        return time.perf_counter() - start
    except Exception:
        return -1.0

def main():
    csv_path = Path("campaigns/glp1r_aleniglipron/track_a_generative/115k_curated_anchors.csv")
    if not csv_path.exists():
        print(f"Error: Cannot find {csv_path}")
        return

    print("=== PRISM-DSTW AM1-BCC THROUGHPUT BENCHMARK ===")
    df = pl.read_csv(csv_path)
    sample_size = 100
    sampled_smiles = df.sample(n=sample_size, seed=42)["smiles"].to_list()
    
    cores = os.cpu_count() or 1
    workers = max(1, cores - 2) 
    
    print(f"Total library size : {df.shape[0]:,} molecules")
    print(f"Workers allocated  : {workers}")
    print("\nSpinning up SQM workers... (This will take a minute or two)")
    
    batch_start = time.perf_counter()
    with mp.Pool(processes=workers) as pool:
        results = pool.map(process_molecule, sampled_smiles)
    batch_time = time.perf_counter() - batch_start
    
    successes = [t for t in results if t > 0]
    avg_time_per_mol = sum(successes) / len(successes) if successes else 0
    projected_hours = ((batch_time / sample_size) * df.shape[0]) / 3600

    print("\n=== BENCHMARK RESULTS ===")
    print(f"Successful         : {len(successes)} / {sample_size}")
    print(f"Batch Wall Time    : {batch_time:.2f} seconds")
    print(f"Avg Time per Mol   : {avg_time_per_mol:.2f} seconds (CPU time)")
    print(f"\n=== FULL 115K PRODUCTION PROJECTION ===")
    print(f"Estimated Runtime  : {projected_hours:.2f} Hours ({projected_hours/24:.2f} Days)")

if __name__ == "__main__":
    main()

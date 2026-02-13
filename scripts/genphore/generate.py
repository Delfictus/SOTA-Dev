#!/usr/bin/env python3
"""Unified entry point for pharmacophore-guided molecule generation.

Usage:
    python scripts/genphore/generate.py \\
        --spike-json snapshots/kras_site1/spikes.json \\
        --output-dir /tmp/genphore_test/ \\
        --n-phoregen 100 --n-pgmg 1000

Pipeline:
    1. spike_to_pharmacophore.convert() → SpikePharmacophore
    2. run_phoregen() → List[GeneratedMolecule]  (diffusion, high quality)
    3. run_pgmg() → List[GeneratedMolecule]  (VAE, fast bulk)
    4. Write combined output: molecules.sdf + molecules_meta.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from scripts.genphore.spike_to_pharmacophore import convert
from scripts.genphore.run_phoregen import run_phoregen
from scripts.genphore.run_pgmg import run_pgmg
from scripts.interfaces import GeneratedMolecule, SpikePharmacophore

logger = logging.getLogger(__name__)


def _write_sdf(molecules: List[GeneratedMolecule], path: Path) -> int:
    """Write molecules to an SDF file. Returns count of molecules with 3D coords."""
    n_3d = 0
    with open(path, "w") as f:
        for i, mol in enumerate(molecules):
            if mol.mol_block:
                f.write(mol.mol_block)
                if not mol.mol_block.rstrip().endswith("$$$$"):
                    f.write("\n$$$$\n")
                n_3d += 1
            else:
                # Write a minimal record for molecules without 3D
                f.write(f"{mol.smiles}\n")
                f.write(f">  <SMILES>\n{mol.smiles}\n\n")
                f.write(f">  <source>\n{mol.source}\n\n")
                f.write("$$$$\n")
    return n_3d


def _write_meta(
    molecules: List[GeneratedMolecule],
    pharmacophore: SpikePharmacophore,
    path: Path,
) -> None:
    """Write metadata JSON for the generation run."""
    meta = {
        "target_name": pharmacophore.target_name,
        "pdb_id": pharmacophore.pdb_id,
        "pocket_id": pharmacophore.pocket_id,
        "n_pharmacophore_features": len(pharmacophore.features),
        "n_exclusion_spheres": len(pharmacophore.exclusion_spheres),
        "prism_run_hash": pharmacophore.prism_run_hash,
        "total_molecules": len(molecules),
        "by_source": {},
        "molecules": [],
    }

    source_counts: dict = {}
    for mol in molecules:
        source_counts[mol.source] = source_counts.get(mol.source, 0) + 1
    meta["by_source"] = source_counts

    for mol in molecules:
        meta["molecules"].append({
            "smiles": mol.smiles,
            "source": mol.source,
            "pharmacophore_match_score": mol.pharmacophore_match_score,
            "matched_features": mol.matched_features,
            "generation_batch_id": mol.generation_batch_id,
        })

    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def generate(
    spike_json: str,
    output_dir: str,
    binding_sites: Optional[str] = None,
    target_name: str = "UNKNOWN",
    pdb_id: str = "UNKNOWN",
    pocket_index: int = 0,
    n_phoregen: int = 1000,
    n_pgmg: int = 10000,
    phoregen_dir: Optional[str] = None,
    pgmg_dir: Optional[str] = None,
    skip_phoregen: bool = False,
    skip_pgmg: bool = False,
) -> List[GeneratedMolecule]:
    """Run the full generation pipeline.

    Args:
        spike_json: Path to spike events JSON.
        output_dir: Directory for all output files.
        binding_sites: Optional path to binding sites JSON.
        target_name: Target identifier.
        pdb_id: PDB accession.
        pocket_index: Which pocket to use.
        n_phoregen: Number of PhoreGen molecules.
        n_pgmg: Number of PGMG molecules.
        phoregen_dir: Override PhoreGen install path.
        pgmg_dir: Override PGMG install path.
        skip_phoregen: Skip PhoreGen generation.
        skip_pgmg: Skip PGMG generation.

    Returns:
        Combined list of all generated molecules.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert spikes → pharmacophore
    logger.info("Step 1: Converting spikes to pharmacophore...")
    pharmacophore = convert(
        spike_json,
        binding_sites,
        target_name=target_name,
        pdb_id=pdb_id,
        pocket_index=pocket_index,
    )

    # Save pharmacophore
    phore_path = out / "pharmacophore.json"
    with open(phore_path, "w") as f:
        f.write(pharmacophore.to_json())
    logger.info("Pharmacophore saved to %s", phore_path)

    all_molecules: List[GeneratedMolecule] = []

    # Step 2: PhoreGen (diffusion)
    if not skip_phoregen:
        logger.info("Step 2: Running PhoreGen (%d molecules)...", n_phoregen)
        try:
            phoregen_mols = run_phoregen(
                pharmacophore,
                n_molecules=n_phoregen,
                output_dir=str(out / "phoregen"),
                phoregen_dir=phoregen_dir,
            )
            all_molecules.extend(phoregen_mols)
            logger.info("PhoreGen: %d molecules generated", len(phoregen_mols))
        except FileNotFoundError as e:
            logger.warning("PhoreGen skipped: %s", e)
        except RuntimeError as e:
            logger.error("PhoreGen failed: %s", e)
    else:
        logger.info("Step 2: PhoreGen skipped (--skip-phoregen)")

    # Step 3: PGMG (VAE)
    if not skip_pgmg:
        logger.info("Step 3: Running PGMG (%d molecules)...", n_pgmg)
        try:
            pgmg_mols = run_pgmg(
                pharmacophore,
                n_molecules=n_pgmg,
                output_dir=str(out / "pgmg"),
                pgmg_dir=pgmg_dir,
            )
            all_molecules.extend(pgmg_mols)
            logger.info("PGMG: %d molecules generated", len(pgmg_mols))
        except FileNotFoundError as e:
            logger.warning("PGMG skipped: %s", e)
        except RuntimeError as e:
            logger.error("PGMG failed: %s", e)
    else:
        logger.info("Step 3: PGMG skipped (--skip-pgmg)")

    # Step 4: Write combined output
    if all_molecules:
        sdf_path = out / "molecules.sdf"
        n_3d = _write_sdf(all_molecules, sdf_path)
        logger.info(
            "Wrote %d molecules to %s (%d with 3D coords)",
            len(all_molecules), sdf_path, n_3d,
        )

        meta_path = out / "molecules_meta.json"
        _write_meta(all_molecules, pharmacophore, meta_path)
        logger.info("Metadata written to %s", meta_path)
    else:
        logger.warning("No molecules generated — check tool installations")

    return all_molecules


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PRISM-4D pharmacophore-guided molecule generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--spike-json", required=True,
        help="Path to PRISM spike events JSON file",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory for generated molecules",
    )
    parser.add_argument(
        "--binding-sites",
        help="Path to binding sites JSON (for exclusion spheres)",
    )
    parser.add_argument("--target-name", default="UNKNOWN")
    parser.add_argument("--pdb-id", default="UNKNOWN")
    parser.add_argument("--pocket-index", type=int, default=0)
    parser.add_argument("--n-phoregen", type=int, default=1000)
    parser.add_argument("--n-pgmg", type=int, default=10000)
    parser.add_argument("--phoregen-dir", help="PhoreGen install path")
    parser.add_argument("--pgmg-dir", help="PGMG install path")
    parser.add_argument(
        "--skip-phoregen", action="store_true",
        help="Skip PhoreGen generation",
    )
    parser.add_argument(
        "--skip-pgmg", action="store_true",
        help="Skip PGMG generation",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    molecules = generate(
        spike_json=args.spike_json,
        output_dir=args.output_dir,
        binding_sites=args.binding_sites,
        target_name=args.target_name,
        pdb_id=args.pdb_id,
        pocket_index=args.pocket_index,
        n_phoregen=args.n_phoregen,
        n_pgmg=args.n_pgmg,
        phoregen_dir=args.phoregen_dir,
        pgmg_dir=args.pgmg_dir,
        skip_phoregen=args.skip_phoregen,
        skip_pgmg=args.skip_pgmg,
    )

    print(f"\nGeneration complete: {len(molecules)} molecules in {args.output_dir}")
    sys.exit(0 if molecules or (args.skip_phoregen and args.skip_pgmg) else 1)


if __name__ == "__main__":
    main()

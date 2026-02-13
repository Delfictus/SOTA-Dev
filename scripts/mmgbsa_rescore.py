#!/usr/bin/env python3
"""
PRISM4D MM-GBSA Rescoring
===========================
Physics-based binding free energy estimation using OpenMM with
GB/SA implicit solvent (OBC2 model, igb=5).

Pipeline:
  1. Load receptor PDB + docked ligand SDF
  2. Parameterize protein (ff14SB) + ligand (GAFF2/AM1-BCC via OpenFF)
  3. Minimize complex, receptor-only, ligand-only
  4. Compute: dG_bind = E_complex - E_receptor - E_ligand

This is a single-point MM-GBSA (no MD sampling), which is faster but
less accurate than ensemble MM-GBSA. Suitable for pose ranking.

References:
    Genheden & Ryde. Expert Opin Drug Discov 2015. doi:10.1517/17460441.2015.1032936
    Onufriev et al. Proteins 2004. doi:10.1002/prot.20033 (OBC2)
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


def setup_system(receptor_pdb, ligand_sdf, pose_index=0):
    """Set up OpenMM system with protein + ligand in implicit solvent."""
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    from openff.toolkit import Molecule as OFFMolecule
    from openmmforcefields.generators import SystemGenerator

    # Load receptor
    pdb = app.PDBFile(str(receptor_pdb))

    # Load ligand from SDF
    from rdkit import Chem
    suppl = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)
    mols = [m for m in suppl if m is not None]
    if pose_index >= len(mols):
        return None
    rdmol = mols[pose_index]

    # Convert to OpenFF Molecule
    off_mol = OFFMolecule.from_rdkit(rdmol)

    # Set up system generator with GAFF2 for small molecules
    system_generator = SystemGenerator(
        forcefields=['amber/ff14SB.xml', 'implicit/obc2.xml'],
        small_molecule_forcefield='gaff-2.11',
        forcefield_kwargs={
            'constraints': app.HBonds,
            'removeCMMotion': False,
        },
        nonperiodic_forcefield_kwargs={
            'nonbondedMethod': app.NoCutoff,
            'constraints': app.HBonds,
        },
        molecules=[off_mol],
    )

    return system_generator, pdb, off_mol, rdmol


def compute_energy(system_generator, topology, positions):
    """Minimize and compute energy for a given system."""
    import openmm
    import openmm.unit as unit

    system = system_generator.create_system(topology)
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond
    )

    platform = openmm.Platform.getPlatformByName('CPU')
    simulation = openmm.app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    # Minimize
    simulation.minimizeEnergy(maxIterations=500, tolerance=10.0)

    state = simulation.context.getState(getEnergy=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    return energy


def _minimize_and_energy(ff, topology, positions, max_iter=200):
    """Create system, minimize, return energy in kcal/mol."""
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    system = ff.createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picosecond
    )
    sim = openmm.app.Simulation(topology, system, integrator,
                                openmm.Platform.getPlatformByName('CPU'))
    sim.context.setPositions(positions)
    sim.minimizeEnergy(maxIterations=max_iter, tolerance=10.0)
    state = sim.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)


def run_mmgbsa(receptor_pdb, ligand_sdf, pose_index=0):
    """Run single-point MM-GBSA for one ligand pose.

    Uses direct ForceField + GAFFTemplateGenerator (not SystemGenerator)
    to avoid nonbonded method routing issues with implicit solvent.

    Returns binding free energy estimate in kcal/mol.
    """
    import openmm
    import openmm.app as app
    import openmm.unit as unit
    from openff.toolkit import Molecule as OFFMolecule
    from openmmforcefields.generators import GAFFTemplateGenerator
    from rdkit import Chem

    # Load molecules
    pdb = app.PDBFile(str(receptor_pdb))
    suppl = Chem.SDMolSupplier(str(ligand_sdf), removeHs=False)
    mols = [m for m in suppl if m is not None]
    if pose_index >= len(mols):
        return None, "No valid poses"

    # Add hydrogens with 3D coordinates
    rdmol = Chem.AddHs(mols[pose_index], addCoords=True)

    try:
        off_mol = OFFMolecule.from_rdkit(rdmol)
    except Exception as e:
        return None, f"OpenFF conversion failed: {e}"

    # Build ForceField with GAFF2 template for small molecule
    ff = app.ForceField('amber/ff14SB.xml', 'implicit/obc2.xml')
    gaff = GAFFTemplateGenerator(molecules=[off_mol], forcefield='gaff-2.11')
    ff.registerTemplateGenerator(gaff.generator)

    # Ligand positions (A -> nm)
    conf = rdmol.GetConformer()
    lig_positions = []
    for i in range(rdmol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        lig_positions.append(openmm.Vec3(pos.x / 10.0, pos.y / 10.0, pos.z / 10.0))

    # === Complex ===
    modeller = app.Modeller(pdb.topology, pdb.positions)
    off_top = off_mol.to_topology()
    modeller.add(off_top.to_openmm(), lig_positions * unit.nanometer)
    E_complex = _minimize_and_energy(ff, modeller.topology, modeller.positions)

    # === Receptor only ===
    E_receptor = _minimize_and_energy(ff, pdb.topology, pdb.positions)

    # === Ligand only ===
    lig_top = off_top.to_openmm()
    E_ligand = _minimize_and_energy(ff, lig_top, lig_positions * unit.nanometer)

    dG = E_complex - E_receptor - E_ligand

    return {
        "E_complex": round(E_complex, 2),
        "E_receptor": round(E_receptor, 2),
        "E_ligand": round(E_ligand, 2),
        "dG_bind": round(dG, 2),
    }, None


def main():
    parser = argparse.ArgumentParser(description="PRISM4D MM-GBSA Rescoring")
    parser.add_argument("--receptor", required=True, help="Receptor PDB")
    parser.add_argument("--sdf-dir", required=True, help="Dir with docked SDF files")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir = Path(args.sdf_dir)

    sdf_files = sorted(sdf_dir.glob("*_rescored.sdf"))
    if not sdf_files:
        sdf_files = sorted(sdf_dir.glob("*.sdf"))

    print("=" * 60)
    print("PRISM4D MM-GBSA Rescoring (OpenMM + OBC2)")
    print("=" * 60)
    print(f"  Receptor: {args.receptor}")
    print(f"  Ligands:  {len(sdf_files)}")

    results = []
    for sdf_path in sdf_files:
        lig_name = sdf_path.stem.replace("_rescored", "").replace("_out", "")
        print(f"\n  {lig_name}...")

        try:
            energies, error = run_mmgbsa(args.receptor, sdf_path, pose_index=0)
            if error:
                print(f"    ERROR: {error}")
                results.append({"name": lig_name, "error": error})
            else:
                print(f"    E_complex = {energies['E_complex']:.1f} kcal/mol")
                print(f"    E_receptor = {energies['E_receptor']:.1f} kcal/mol")
                print(f"    E_ligand = {energies['E_ligand']:.1f} kcal/mol")
                print(f"    dG_bind = {energies['dG_bind']:.1f} kcal/mol")
                results.append({"name": lig_name, **energies})
        except Exception as e:
            print(f"    EXCEPTION: {e}")
            results.append({"name": lig_name, "error": str(e)})

    # Save results
    output_json = output_dir / "mmgbsa_results.json"
    with open(output_json, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "method": "MM-GBSA (OBC2, ff14SB+GAFF2)", "results": results}, f, indent=2)

    # Markdown summary
    output_md = output_dir / "mmgbsa_report.md"
    with open(output_md, "w") as f:
        f.write("# PRISM4D MM-GBSA Rescoring Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("**Method**: Single-point MM-GBSA (OBC2 implicit solvent, ff14SB + GAFF2)\n\n")
        f.write("## Results\n\n")
        f.write("| Ligand | E_complex | E_receptor | E_ligand | dG_bind (kcal/mol) |\n")
        f.write("|--------|-----------|-----------|---------|--------------------|\n")
        for r in results:
            if "error" in r:
                f.write(f"| {r['name']} | ERROR | - | - | {r['error']} |\n")
            else:
                f.write(f"| {r['name']} | {r['E_complex']:.1f} | {r['E_receptor']:.1f} | "
                        f"{r['E_ligand']:.1f} | **{r['dG_bind']:.1f}** |\n")
        f.write("\n")
        f.write("**Interpretation**: More negative dG_bind = stronger predicted binding.\n\n")
        f.write("## Method\n\n")
        f.write("- Protein force field: AMBER ff14SB\n")
        f.write("- Ligand force field: GAFF2 with AM1-BCC charges\n")
        f.write("- Implicit solvent: OBC2 (Onufriev-Bashford-Case, igb=5)\n")
        f.write("- Minimization: 500 steps L-BFGS\n")
        f.write("- dG_bind = E(complex) - E(receptor) - E(ligand)\n\n")
        f.write("## References\n\n")
        f.write("- Genheden & Ryde. Expert Opin Drug Discov 2015. doi:10.1517/17460441.2015.1032936\n")
        f.write("- Onufriev et al. Proteins 2004. doi:10.1002/prot.20033\n\n")
        f.write("---\n*Generated by PRISM4D MM-GBSA Pipeline v1.0*\n")

    print(f"\n  Report: {output_md}")
    print(f"  JSON:   {output_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()

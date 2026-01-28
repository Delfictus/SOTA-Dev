#!/usr/bin/env python3
"""
Prepare a protein structure using OpenMM/PDBFixer for PRISM4D explicit solvent MD.

This script:
1. Fixes missing atoms/residues using PDBFixer
2. Adds hydrogens at specified pH
3. Runs energy minimization to remove clashes
4. Outputs a clean PDB ready for PRISM solvation

Usage:
    python3 prepare_structure_openmm.py input.pdb output.pdb [--ph 7.0]

The output PDB can be directly used by PRISM's SolvationBox module.
"""

import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Prepare protein structure with OpenMM/PDBFixer')
    parser.add_argument('input_pdb', help='Input PDB file (raw from RCSB or other source)')
    parser.add_argument('output_pdb', help='Output PDB file (cleaned, minimized)')
    parser.add_argument('--ph', type=float, default=7.0, help='pH for protonation (default: 7.0)')
    parser.add_argument('--minimize-steps', type=int, default=1000, help='Minimization steps (default: 1000)')
    parser.add_argument('--keep-waters', action='store_true', help='Keep crystallographic waters')
    parser.add_argument('--keep-ligands', action='store_true', help='Keep ligands/heterogens')
    args = parser.parse_args()

    # Import OpenMM (check availability)
    try:
        import openmm
        from openmm import app, unit
        from openmm import LangevinMiddleIntegrator
    except ImportError:
        print("ERROR: OpenMM not found.")
        print("Install with: conda install -c conda-forge openmm")
        sys.exit(1)

    try:
        from pdbfixer import PDBFixer
    except ImportError:
        print("ERROR: PDBFixer not found.")
        print("Install with: conda install -c conda-forge pdbfixer")
        sys.exit(1)

    input_path = Path(args.input_pdb)
    output_path = Path(args.output_pdb)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    print(f"=" * 60)
    print(f"OpenMM/PDBFixer Structure Preparation")
    print(f"=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"pH:     {args.ph}")
    print()

    # Step 1: Load and fix structure with PDBFixer
    print("Step 1: Loading and fixing structure with PDBFixer...")
    fixer = PDBFixer(filename=str(input_path))

    # Find missing residues (gaps)
    fixer.findMissingResidues()
    if fixer.missingResidues:
        print(f"  Found {len(fixer.missingResidues)} missing residue segments")
        for key, residues in fixer.missingResidues.items():
            print(f"    Chain {key[0]}, after {key[1]}: {len(residues)} residues")
        # Note: We don't add missing residues by default (can cause issues)
        # Clear them to skip
        fixer.missingResidues = {}

    # Find missing atoms
    fixer.findMissingAtoms()
    n_missing_atoms = sum(len(atoms) for atoms in fixer.missingAtoms.values())
    n_missing_terminals = sum(len(atoms) for atoms in fixer.missingTerminals.values())
    print(f"  Missing atoms: {n_missing_atoms}")
    print(f"  Missing terminals: {n_missing_terminals}")

    # Add missing atoms (including terminal caps like OXT)
    fixer.addMissingAtoms()
    print("  Added missing atoms")

    # Remove heterogens if requested
    if not args.keep_waters and not args.keep_ligands:
        fixer.removeHeterogens(keepWater=False)
        print("  Removed all heterogens (waters, ligands)")
    elif not args.keep_waters:
        fixer.removeHeterogens(keepWater=False)
        print("  Removed waters (kept ligands)")
    elif not args.keep_ligands:
        # Keep water, remove other heterogens
        # PDBFixer doesn't have a direct option for this, so we keep waters
        fixer.removeHeterogens(keepWater=True)
        print("  Kept waters, removed other heterogens")
    else:
        print("  Kept all heterogens")

    # Step 2: Add hydrogens
    print(f"\nStep 2: Adding hydrogens at pH {args.ph}...")

    # Check if already has hydrogens
    has_h = any(atom.element.symbol == 'H' for atom in fixer.topology.atoms())
    if has_h:
        print("  Structure already has hydrogens - will re-add for consistency")

    fixer.addMissingHydrogens(args.ph)

    n_atoms = fixer.topology.getNumAtoms()
    n_residues = fixer.topology.getNumResidues()
    n_chains = fixer.topology.getNumChains()
    print(f"  Result: {n_atoms} atoms, {n_residues} residues, {n_chains} chains")

    # Step 3: Energy minimization with AMBER ff14SB
    print(f"\nStep 3: Energy minimization ({args.minimize_steps} steps)...")

    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    system = forcefield.createSystem(
        fixer.topology,
        nonbondedMethod=app.NoCutoff,  # Vacuum for minimization
        constraints=None,
        rigidWater=False,
    )

    integrator = LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1 / unit.picosecond,
        0.002 * unit.picosecond
    )

    simulation = app.Simulation(fixer.topology, system, integrator)
    simulation.context.setPositions(fixer.positions)

    # Get initial energy
    state = simulation.context.getState(getEnergy=True)
    pe_before = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    print(f"  Initial energy: {pe_before:.1f} kcal/mol")

    # Minimize
    simulation.minimizeEnergy(maxIterations=args.minimize_steps)

    # Get final energy and positions
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    pe_after = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    print(f"  Final energy:   {pe_after:.1f} kcal/mol")
    print(f"  Energy change:  {pe_after - pe_before:.1f} kcal/mol")

    minimized_positions = state.getPositions()

    # Step 4: Write output PDB
    print(f"\nStep 4: Writing output PDB...")

    with open(output_path, 'w') as f:
        app.PDBFile.writeFile(
            fixer.topology,
            minimized_positions,
            f,
            keepIds=True
        )

    print(f"  Written: {output_path}")

    # Summary
    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Atoms:    {n_atoms}")
    print(f"  Residues: {n_residues}")
    print(f"  Chains:   {n_chains}")
    print(f"  Energy:   {pe_after:.1f} kcal/mol")
    print()
    print("Structure is ready for PRISM solvation:")
    print(f"  prism_physics::SolvationBox::from_pdb(\"{output_path}\")")
    print()

if __name__ == "__main__":
    main()

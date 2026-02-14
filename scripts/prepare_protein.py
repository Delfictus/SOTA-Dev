#!/usr/bin/env python3
"""
Prepare a protein with OpenMM and export topology for PRISM4D testing.

Usage: python3 prepare_protein.py input.pdb output.json [--solvate]
"""

import sys
import json
import numpy as np

try:
    from openmm import app, unit
    from openmm import *
except ImportError:
    print("OpenMM not found. Install with: conda install -c conda-forge openmm")
    sys.exit(1)

# Try to import PDBFixer for better structure handling
try:
    from pdbfixer import PDBFixer
    HAS_PDBFIXER = True
except ImportError:
    HAS_PDBFIXER = False
    print("Warning: PDBFixer not found. Terminal capping may fail.")
    print("Install with: conda install -c conda-forge pdbfixer")

def prepare_protein(pdb_path, output_path, solvate=False):
    print(f"Loading {pdb_path}...")

    # Use AMBER ff14SB force field
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    # Use PDBFixer for robust structure preparation
    if HAS_PDBFIXER:
        print("Using PDBFixer to fix structure...")
        fixer = PDBFixer(filename=pdb_path)

        # Find and add missing residues (gaps in the chain)
        fixer.findMissingResidues()
        if fixer.missingResidues:
            print(f"  Found {len(fixer.missingResidues)} missing residue segments")
            # Don't add missing residues - just note them
            # fixer.missingResidues = {}  # Clear to skip adding

        # Find and add missing atoms (including terminal OXT)
        fixer.findMissingAtoms()
        n_missing_atoms = sum(len(atoms) for atoms in fixer.missingAtoms.values())
        n_missing_terminals = sum(len(atoms) for atoms in fixer.missingTerminals.values())
        print(f"  Missing atoms: {n_missing_atoms}, Missing terminals: {n_missing_terminals}")

        fixer.addMissingAtoms()

        # Find and remove heterogens (water, ligands, etc.) - optional
        # fixer.removeHeterogens(keepWater=False)

        # Create modeller from fixed structure
        modeller = app.Modeller(fixer.topology, fixer.positions)
    else:
        # Fallback: load directly
        pdb = app.PDBFile(pdb_path)
        modeller = app.Modeller(pdb.topology, pdb.positions)

    # Check if structure already has hydrogens
    has_hydrogens = any(atom.element.symbol == 'H' for atom in modeller.topology.atoms())

    if has_hydrogens:
        print("Structure already has hydrogens, skipping addHydrogens()")
    else:
        print("Adding hydrogens...")
        modeller.addHydrogens(forcefield, pH=7.0)

    if solvate:
        print("Adding solvent (TIP3P water + ions)...")
        modeller.addSolvent(forcefield, model='tip3p', padding=1.0*unit.nanometer,
                          ionicStrength=0.15*unit.molar)

    # Create system
    print("Creating system with AMBER ff14SB...")
    if solvate:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0*unit.nanometer,
            constraints=None,  # We handle constraints ourselves
            rigidWater=False,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,  # We handle constraints ourselves
            rigidWater=False,
        )

    topology = modeller.topology
    positions = modeller.positions

    n_atoms = topology.getNumAtoms()
    print(f"System has {n_atoms} atoms")

    # Energy minimize in OpenMM first to remove clashes
    print("Running OpenMM energy minimization...")
    integrator = LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picosecond)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    # Get initial energy
    state = simulation.context.getState(getEnergy=True)
    pe_before = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    print(f"  Initial PE: {pe_before:.1f} kcal/mol")

    # Minimize
    simulation.minimizeEnergy(maxIterations=1000)

    # Get final energy
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    pe_after = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    print(f"  Final PE: {pe_after:.1f} kcal/mol")

    # Update positions from minimized structure
    positions = state.getPositions()

    # Extract atom properties
    masses = []
    elements = []
    for atom in topology.atoms():
        masses.append(atom.element.mass.value_in_unit(unit.dalton))
        elements.append(atom.element.symbol)

    # Extract positions (convert to Angstroms)
    pos_flat = []
    for pos in positions:
        pos_flat.extend([
            pos.x * 10,  # nm -> Angstrom
            pos.y * 10,
            pos.z * 10,
        ])

    # Extract force field parameters
    bonds = []
    angles = []
    dihedrals = []
    charges = [0.0] * n_atoms
    lj_params = [{"sigma": 0.0, "epsilon": 0.0} for _ in range(n_atoms)]
    exclusions = [set() for _ in range(n_atoms)]

    for force in system.getForces():
        force_name = force.__class__.__name__

        if force_name == "HarmonicBondForce":
            print(f"  Extracting {force.getNumBonds()} bonds...")
            for i in range(force.getNumBonds()):
                p1, p2, r0, k = force.getBondParameters(i)
                bonds.append({
                    "i": p1,
                    "j": p2,
                    "r0": r0.value_in_unit(unit.angstrom),
                    "k": k.value_in_unit(unit.kilocalorie_per_mole / unit.angstrom**2),
                })
                # Add to exclusions
                exclusions[p1].add(p2)
                exclusions[p2].add(p1)

        elif force_name == "HarmonicAngleForce":
            print(f"  Extracting {force.getNumAngles()} angles...")
            for i in range(force.getNumAngles()):
                p1, p2, p3, theta0, k = force.getAngleParameters(i)
                angles.append({
                    "i": p1,
                    "j": p2,
                    "k_idx": p3,
                    "theta0": theta0.value_in_unit(unit.radian),
                    "force_k": k.value_in_unit(unit.kilocalorie_per_mole / unit.radian**2),
                })
                # Add 1-3 exclusions
                exclusions[p1].add(p3)
                exclusions[p3].add(p1)

        elif force_name == "PeriodicTorsionForce":
            print(f"  Extracting {force.getNumTorsions()} dihedrals...")
            for i in range(force.getNumTorsions()):
                p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
                dihedrals.append({
                    "i": p1,
                    "j": p2,
                    "k_idx": p3,
                    "l": p4,
                    "periodicity": periodicity,
                    "phase": phase.value_in_unit(unit.radian),
                    "force_k": k.value_in_unit(unit.kilocalorie_per_mole),
                })

        elif force_name == "NonbondedForce":
            print(f"  Extracting non-bonded parameters for {force.getNumParticles()} particles...")
            for i in range(force.getNumParticles()):
                charge, sigma, epsilon = force.getParticleParameters(i)
                charges[i] = charge.value_in_unit(unit.elementary_charge)
                lj_params[i] = {
                    "sigma": sigma.value_in_unit(unit.angstrom),
                    "epsilon": epsilon.value_in_unit(unit.kilocalorie_per_mole),
                }

            # Extract exceptions (1-4 interactions)
            print(f"  Extracting {force.getNumExceptions()} exceptions...")
            for i in range(force.getNumExceptions()):
                p1, p2, chargeProd, sigma, epsilon = force.getExceptionParameters(i)
                exclusions[p1].add(p2)
                exclusions[p2].add(p1)

    # Find water oxygens
    water_oxygens = []
    for residue in topology.residues():
        if residue.name in ['HOH', 'WAT', 'TIP3']:
            for atom in residue.atoms():
                if atom.element.symbol == 'O':
                    water_oxygens.append(atom.index)
    print(f"Found {len(water_oxygens)} water molecules")

    # Build H-bond clusters
    h_clusters = build_h_clusters(topology, bonds, masses, elements)
    print(f"Built {len(h_clusters)} H-bond constraint clusters")

    # Get box vectors if present AND reasonable (> 10 Å)
    box_vectors = None
    if topology.getPeriodicBoxVectors() is not None:
        bv = topology.getPeriodicBoxVectors()
        bv_angstrom = [
            bv[0][0].value_in_unit(unit.angstrom),
            bv[1][1].value_in_unit(unit.angstrom),
            bv[2][2].value_in_unit(unit.angstrom),
        ]
        # Only use box vectors if they're reasonable (> 10 Å)
        if min(bv_angstrom) > 10.0:
            box_vectors = bv_angstrom
            print(f"Box: {box_vectors[0]:.2f} x {box_vectors[1]:.2f} x {box_vectors[2]:.2f} Å")
        else:
            print(f"Ignoring tiny box vectors: {bv_angstrom} (need > 10 Å)")

    # Convert exclusions to lists
    exclusions_list = [sorted(list(ex)) for ex in exclusions]

    # Build output
    output = {
        "n_atoms": n_atoms,
        "positions": pos_flat,
        "masses": masses,
        "charges": charges,
        "lj_params": lj_params,
        "bonds": bonds,
        "angles": angles,
        "dihedrals": dihedrals,
        "water_oxygens": water_oxygens,
        "h_clusters": h_clusters,
        "exclusions": exclusions_list,
    }

    if box_vectors:
        output["box_vectors"] = box_vectors

    # Write JSON
    print(f"Writing {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print("Done!")
    return output


def build_h_clusters(topology, bonds, masses, elements):
    """Build H-bond constraint clusters from topology."""
    from collections import defaultdict

    # Build adjacency for H-bonds
    h_neighbors = defaultdict(list)

    for bond in bonds:
        i, j = bond["i"], bond["j"]
        r0 = bond["r0"]

        # Check if this is an X-H bond
        if elements[i] == "H" and elements[j] != "H":
            h_neighbors[j].append((i, r0))
        elif elements[j] == "H" and elements[i] != "H":
            h_neighbors[i].append((j, r0))

    # Build clusters
    clusters = []
    for heavy, hydrogens in h_neighbors.items():
        is_nitrogen = elements[heavy] == "N"
        mass_central = masses[heavy]
        mass_h = masses[hydrogens[0][0]] if hydrogens else 1.008

        n_h = len(hydrogens)
        if n_h == 0:
            continue

        # Determine cluster type
        if n_h == 1:
            cluster_type = 1  # SINGLE_H
        elif n_h == 2:
            cluster_type = 4 if is_nitrogen else 2  # NH2 or CH2
        elif n_h == 3:
            cluster_type = 5 if is_nitrogen else 3  # NH3 or CH3
        else:
            continue  # Skip unusual cases

        h_atoms = [h[0] for h in hydrogens[:3]]
        bond_lengths = [h[1] for h in hydrogens[:3]]

        # Pad to 3 elements
        while len(h_atoms) < 3:
            h_atoms.append(-1)
        while len(bond_lengths) < 3:
            bond_lengths.append(0.0)

        clusters.append({
            "type": cluster_type,
            "central_atom": heavy,
            "hydrogen_atoms": h_atoms,
            "bond_lengths": bond_lengths,
            "n_hydrogens": n_h,
            "inv_mass_central": 1.0 / mass_central,
            "inv_mass_h": 1.0 / mass_h,
        })

    return clusters


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 prepare_protein.py input.pdb output.json [--solvate]")
        sys.exit(1)

    pdb_path = sys.argv[1]
    output_path = sys.argv[2]
    solvate = "--solvate" in sys.argv

    prepare_protein(pdb_path, output_path, solvate=solvate)

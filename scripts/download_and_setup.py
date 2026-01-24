#!/usr/bin/env python3
"""
PRISM-4D Structure Preparation with OpenMM

Downloads PDB structures, fixes missing atoms (including ALL hydrogens),
solvates with TIP3P water, and exports complete AMBER ff14SB topology
for use with PRISM's GPU MD engine.

This produces DRAMATICALLY better results than raw PDB files because:
1. All hydrogens are added with correct geometry
2. Proper solvation with pre-equilibrated water
3. Complete force field parameters (bonds, angles, dihedrals)
4. Correct protonation states at physiological pH

Usage:
    python download_and_setup.py --structures          # Download test set
    python download_and_setup.py --prepare 1ubq       # Prepare with solvent
    python download_and_setup.py --prepare 1ubq --no-solvent  # Vacuum
"""

import os
import sys
import json
import argparse
import math
import urllib.request
from pathlib import Path


def check_dependencies():
    """Check that OpenMM and friends are installed."""
    missing = []
    try:
        import openmm
    except ImportError:
        missing.append("openmm")
    try:
        import pdbfixer
    except ImportError:
        missing.append("pdbfixer")

    if missing:
        print("Missing dependencies:", ", ".join(missing))
        print("\nInstall with:")
        print("  conda install -c conda-forge openmm pdbfixer")
        sys.exit(1)


def get_data_dir():
    """Get the data directory path."""
    return Path(os.environ.get("PRISM_DATA", Path.home() / "prism_data"))


# Test structures for PRISM validation
TEST_STRUCTURES = {
    # Tiny test cases
    "1l2y": {"name": "Trp-cage", "residues": 20, "atoms_approx": 300},
    "2jof": {"name": "Villin headpiece", "residues": 35, "atoms_approx": 600},

    # Standard benchmarks
    "1ubq": {"name": "Ubiquitin", "residues": 76, "atoms_approx": 1200},
    "1aba": {"name": "Glutaredoxin", "residues": 87, "atoms_approx": 1400},
    "1ake": {"name": "Adenylate kinase (apo)", "residues": 214, "atoms_approx": 3400},
    "4ake": {"name": "Adenylate kinase (holo)", "residues": 214, "atoms_approx": 3400},

    # Larger tests
    "1bgl": {"name": "Beta-galactosidase", "residues": 1023, "atoms_approx": 16000},
}


def download_pdb(pdb_id: str, output_dir: Path) -> Path:
    """Download PDB file from RCSB."""
    pdb_id = pdb_id.lower()
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = output_dir / f"{pdb_id}.pdb"

    if pdb_path.exists():
        print(f"  Already downloaded: {pdb_path}")
        return pdb_path

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"  Downloading {pdb_id.upper()} from RCSB...")

    try:
        urllib.request.urlretrieve(url, pdb_path)
        print(f"  Saved to: {pdb_path}")
        return pdb_path
    except Exception as e:
        print(f"  Failed to download {pdb_id}: {e}")
        return None


def download_structures(data_dir: Path):
    """Download all test structures."""
    raw_dir = data_dir / "raw"

    print("\nDownloading test structures...")
    for pdb_id, info in TEST_STRUCTURES.items():
        print(f"\n{pdb_id.upper()} - {info['name']} ({info['residues']} residues)")
        download_pdb(pdb_id, raw_dir)


def prepare_structure(pdb_id: str, data_dir: Path, solvate: bool = True,
                      padding: float = 10.0, ionic_strength: float = 0.15):
    """
    Prepare a structure for MD simulation using OpenMM.

    This is the key function that makes our MD simulations work properly:
    1. Fix missing atoms and add ALL hydrogens
    2. Optionally solvate with TIP3P water
    3. Add ions for charge neutralization
    4. Export complete AMBER topology to JSON

    Args:
        pdb_id: PDB identifier
        data_dir: Base data directory
        solvate: Whether to add explicit solvent
        padding: Box padding in Angstroms
        ionic_strength: Salt concentration in M
    """
    check_dependencies()

    from openmm import app, unit, openmm
    from pdbfixer import PDBFixer

    pdb_id = pdb_id.lower()
    raw_dir = data_dir / "raw"
    prepared_dir = data_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    # Download if needed
    pdb_path = raw_dir / f"{pdb_id}.pdb"
    if not pdb_path.exists():
        pdb_path = download_pdb(pdb_id, raw_dir)
        if not pdb_path:
            return None

    print(f"\nPreparing {pdb_id.upper()}...")

    # Step 1: Fix structure with pdbfixer
    print("  1. Fixing structure with pdbfixer...")
    fixer = PDBFixer(str(pdb_path))

    # Find and fix missing residues
    fixer.findMissingResidues()
    print(f"     Missing residues: {len(fixer.missingResidues)}")

    # Find and add missing atoms (including hydrogens!)
    fixer.findMissingAtoms()
    print(f"     Missing atoms: {sum(len(atoms) for atoms in fixer.missingAtoms.values())}")
    fixer.addMissingAtoms()

    # Add missing hydrogens at pH 7.0
    print("     Adding hydrogens at pH 7.0...")
    fixer.addMissingHydrogens(7.0)

    # Remove heterogens (waters, ligands) - we'll add our own water
    print("     Removing heterogens...")
    fixer.removeHeterogens(keepWater=False)

    # Step 2: Create OpenMM system with AMBER ff14SB
    print("  2. Creating AMBER ff14SB topology...")
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    # Get the modeller
    modeller = app.Modeller(fixer.topology, fixer.positions)

    n_protein_atoms = modeller.topology.getNumAtoms()
    print(f"     Protein atoms (with H): {n_protein_atoms}")

    # Step 3: Solvate if requested
    if solvate:
        print(f"  3. Solvating with TIP3P (padding={padding}A)...")
        modeller.addSolvent(
            forcefield,
            model='tip3p',
            padding=padding * unit.angstrom,
            ionicStrength=ionic_strength * unit.molar,
            positiveIon='Na+',
            negativeIon='Cl-'
        )

        n_total_atoms = modeller.topology.getNumAtoms()
        n_water_atoms = n_total_atoms - n_protein_atoms
        n_waters = n_water_atoms // 3
        print(f"     Added {n_waters} water molecules ({n_water_atoms} atoms)")
        print(f"     Total atoms: {n_total_atoms}")
    else:
        print("  3. No solvation (vacuum simulation)")
        n_total_atoms = n_protein_atoms

    # Step 4: Create the system to get parameters
    print("  4. Creating OpenMM system...")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if solvate else app.NoCutoff,
        nonbondedCutoff=10.0 * unit.angstrom if solvate else None,
        constraints=app.HBonds,  # Constrain H-bonds for minimization stability
        rigidWater=True,
        removeCMMotion=True
    )

    # Step 4.5: Energy minimization
    print("  4.5. Energy minimization...")
    integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 2.0*unit.femtosecond)
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    # Get initial energy
    state = simulation.context.getState(getEnergy=True)
    initial_pe = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    print(f"     Initial PE: {initial_pe:.1f} kcal/mol")

    # Minimize (tolerance is in kJ/mol/nm for force)
    simulation.minimizeEnergy(maxIterations=1000, tolerance=10.0)

    # Get final energy
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    final_pe = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
    print(f"     Final PE: {final_pe:.1f} kcal/mol (delta: {final_pe - initial_pe:.1f})")

    # Update positions in modeller
    modeller.positions = state.getPositions()

    # Recreate system WITHOUT constraints for PRISM (we handle constraints ourselves)
    print("  4.6. Recreating system for export...")
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME if solvate else app.NoCutoff,
        nonbondedCutoff=10.0 * unit.angstrom if solvate else None,
        constraints=None,  # We handle constraints ourselves
        rigidWater=False,
        removeCMMotion=True
    )

    # Step 5: Extract topology data for PRISM
    print("  5. Extracting topology for PRISM...")
    topology_data = extract_topology(modeller, system, forcefield)

    # Step 6: Save outputs
    suffix = "_solvated" if solvate else "_vacuum"

    # Save prepared PDB
    pdb_out_path = prepared_dir / f"{pdb_id}{suffix}.pdb"
    with open(pdb_out_path, 'w') as f:
        app.PDBFile.writeFile(modeller.topology, modeller.positions, f)
    print(f"     Saved PDB: {pdb_out_path}")

    # Save topology JSON for PRISM
    json_out_path = prepared_dir / f"{pdb_id}_topology.json"
    with open(json_out_path, 'w') as f:
        json.dump(topology_data, f)
    print(f"     Saved topology: {json_out_path}")

    # Summary
    print(f"\n  Summary for {pdb_id.upper()}:")
    print(f"    Atoms: {topology_data['n_atoms']}")
    print(f"    Bonds: {len(topology_data['bonds'])}")
    print(f"    Angles: {len(topology_data['angles'])}")
    print(f"    Dihedrals: {len(topology_data['dihedrals'])}")
    print(f"    Water molecules: {len(topology_data['water_oxygens'])}")
    print(f"    H-bond clusters: {len(topology_data['h_clusters'])}")

    return topology_data


def extract_topology(modeller, system, forcefield):
    """
    Extract complete topology data from OpenMM system.

    This extracts ALL the information needed for GPU MD:
    - Atom positions, masses, charges, LJ parameters
    - Bond, angle, dihedral parameters
    - Water oxygen indices for SETTLE
    - H-bond cluster indices for analytic constraints
    """
    from openmm import app, unit, openmm

    topology = modeller.topology
    positions = modeller.positions

    # Basic counts
    n_atoms = topology.getNumAtoms()

    # Extract atom data
    atoms_data = []
    positions_flat = []
    masses = []

    for atom in topology.atoms():
        atoms_data.append({
            "index": atom.index,
            "name": atom.name,
            "element": atom.element.symbol if atom.element else "X",
            "residue": atom.residue.name,
            "residue_id": atom.residue.index,
            "chain": atom.residue.chain.id,
        })

        pos = positions[atom.index]
        # Handle both Quantity and plain Vec3 formats
        if hasattr(pos, 'value_in_unit'):
            pos_angstrom = pos.value_in_unit(unit.angstrom)
            positions_flat.extend([pos_angstrom.x, pos_angstrom.y, pos_angstrom.z])
        else:
            # Vec3 in nanometers (OpenMM internal unit)
            positions_flat.extend([pos.x * 10.0, pos.y * 10.0, pos.z * 10.0])

        mass = atom.element.mass.value_in_unit(unit.dalton) if atom.element else 1.0
        masses.append(mass)

    # Get nonbonded parameters (charges, LJ)
    charges = [0.0] * n_atoms
    lj_params = [(0.0, 0.0)] * n_atoms  # (sigma, epsilon)

    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            for i in range(force.getNumParticles()):
                charge, sigma, epsilon = force.getParticleParameters(i)
                charges[i] = charge.value_in_unit(unit.elementary_charge)
                lj_params[i] = (
                    sigma.value_in_unit(unit.angstrom),
                    epsilon.value_in_unit(unit.kilocalorie_per_mole)
                )

    # Extract bonds
    bonds = []
    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                idx1, idx2, r0, k = force.getBondParameters(i)
                bonds.append({
                    "i": idx1,
                    "j": idx2,
                    "r0": r0.value_in_unit(unit.angstrom),
                    "k": k.value_in_unit(unit.kilocalorie_per_mole / unit.angstrom**2)
                })

    # Extract angles
    angles = []
    for force in system.getForces():
        if isinstance(force, openmm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                idx1, idx2, idx3, theta0, k = force.getAngleParameters(i)
                angles.append({
                    "i": idx1,
                    "j": idx2,
                    "k_idx": idx3,
                    "theta0": theta0.value_in_unit(unit.radian),
                    "force_k": k.value_in_unit(unit.kilocalorie_per_mole / unit.radian**2)
                })

    # Extract dihedrals (proper + improper)
    dihedrals = []
    for force in system.getForces():
        if isinstance(force, openmm.PeriodicTorsionForce):
            for i in range(force.getNumTorsions()):
                idx1, idx2, idx3, idx4, periodicity, phase, k = force.getTorsionParameters(i)
                dihedrals.append({
                    "i": idx1,
                    "j": idx2,
                    "k_idx": idx3,
                    "l": idx4,
                    "periodicity": periodicity,
                    "phase": phase.value_in_unit(unit.radian),
                    "force_k": k.value_in_unit(unit.kilocalorie_per_mole)
                })

    # Find water oxygen indices (for SETTLE)
    water_oxygens = []
    for residue in topology.residues():
        if residue.name in ['HOH', 'WAT', 'TIP3', 'TIP3P']:
            for atom in residue.atoms():
                if atom.element and atom.element.symbol == 'O':
                    water_oxygens.append(atom.index)

    # Find H-bond clusters (for analytic constraints)
    h_clusters = find_h_clusters(topology, atoms_data, bonds, masses)

    # Get box dimensions
    box_vectors = None
    if topology.getPeriodicBoxVectors():
        vecs = topology.getPeriodicBoxVectors()
        # Handle both Quantity and plain Vec3 formats
        if hasattr(vecs[0], 'value_in_unit'):
            box_vectors = [
                vecs[0].value_in_unit(unit.angstrom).x,
                vecs[1].value_in_unit(unit.angstrom).y,
                vecs[2].value_in_unit(unit.angstrom).z
            ]
        else:
            # Vec3 in nanometers
            box_vectors = [
                vecs[0].x * 10.0,
                vecs[1].y * 10.0,
                vecs[2].z * 10.0
            ]

    # Center and wrap molecules into the box (required for cell list algorithm)
    # CRITICAL: We wrap by CONNECTED COMPONENTS (using bond graph), not residues
    # This preserves all covalent bonds including peptide bonds between residues
    if box_vectors is not None:
        bx, by, bz = box_vectors
        box_center = [bx / 2.0, by / 2.0, bz / 2.0]

        # Step 1: Translate entire system so centroid is at box center
        # This handles systems that span across the origin
        cx_all = sum(positions_flat[i*3] for i in range(n_atoms)) / n_atoms
        cy_all = sum(positions_flat[i*3+1] for i in range(n_atoms)) / n_atoms
        cz_all = sum(positions_flat[i*3+2] for i in range(n_atoms)) / n_atoms

        shift_x = box_center[0] - cx_all
        shift_y = box_center[1] - cy_all
        shift_z = box_center[2] - cz_all

        for i in range(n_atoms):
            positions_flat[i*3] += shift_x
            positions_flat[i*3+1] += shift_y
            positions_flat[i*3+2] += shift_z

        print(f"  Centered system in box (shift: {shift_x:.2f}, {shift_y:.2f}, {shift_z:.2f})")

        # Step 2: Build connectivity graph from bonds
        neighbors = [[] for _ in range(n_atoms)]
        for bond in bonds:
            i, j = bond["i"], bond["j"]
            neighbors[i].append(j)
            neighbors[j].append(i)

        # Find connected components using BFS
        visited = [False] * n_atoms
        molecules = []  # List of lists of atom indices

        for start in range(n_atoms):
            if visited[start]:
                continue
            # BFS to find all atoms in this molecule
            component = []
            queue = [start]
            visited[start] = True
            while queue:
                atom = queue.pop(0)
                component.append(atom)
                for neighbor in neighbors[atom]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            molecules.append(component)

        # Step 3: For each molecule, wrap its center into the box
        for mol_atoms in molecules:
            # Compute geometric center of molecule
            cx = sum(positions_flat[i*3] for i in mol_atoms) / len(mol_atoms)
            cy = sum(positions_flat[i*3+1] for i in mol_atoms) / len(mol_atoms)
            cz = sum(positions_flat[i*3+2] for i in mol_atoms) / len(mol_atoms)

            # Compute translation to bring center into box [0, box)
            dx = -bx * math.floor(cx / bx)
            dy = -by * math.floor(cy / by)
            dz = -bz * math.floor(cz / bz)

            # Apply translation to all atoms in molecule
            for i in mol_atoms:
                positions_flat[i*3] += dx
                positions_flat[i*3+1] += dy
                positions_flat[i*3+2] += dz

        print(f"  Wrapped {len(molecules)} molecules into box [{bx:.2f} x {by:.2f} x {bz:.2f}] A")

    # Build exclusion lists (1-2 and 1-3 pairs)
    exclusions = build_exclusion_list(bonds, angles, n_atoms)

    return {
        "n_atoms": n_atoms,
        "atoms": atoms_data,
        "positions": positions_flat,
        "masses": masses,
        "charges": charges,
        "lj_params": [{"sigma": s, "epsilon": e} for s, e in lj_params],
        "bonds": bonds,
        "angles": angles,
        "dihedrals": dihedrals,
        "water_oxygens": water_oxygens,
        "h_clusters": h_clusters,
        "box_vectors": box_vectors,
        "exclusions": exclusions,
    }


def find_h_clusters(topology, atoms_data, bonds, masses):
    """
    Find hydrogen bond clusters for analytic constraints.

    Clusters are grouped by type:
    - SINGLE_H (1): C-H, N-H, O-H, S-H single bonds
    - CH2 (2): Methylene groups
    - CH3 (3): Methyl groups
    - NH2 (4): Amide groups (Asn, Gln sidechains)
    - NH3 (5): Protonated lysine
    """
    from collections import defaultdict

    # Build adjacency list from bonds
    neighbors = defaultdict(list)
    bond_lengths = {}

    for bond in bonds:
        i, j = bond["i"], bond["j"]
        r0 = bond["r0"]
        neighbors[i].append(j)
        neighbors[j].append(i)
        bond_lengths[(min(i,j), max(i,j))] = r0

    # Find heavy atoms with attached hydrogens
    h_clusters = []
    processed_hydrogens = set()

    for i, atom in enumerate(atoms_data):
        # Is this a heavy atom? (not H)
        if atom["element"] == "H":
            continue

        # Find attached hydrogens
        attached_h = []
        for j in neighbors[i]:
            if j < len(atoms_data) and atoms_data[j]["element"] == "H":
                if j not in processed_hydrogens:
                    attached_h.append(j)

        if not attached_h:
            continue

        # Mark hydrogens as processed
        for h in attached_h:
            processed_hydrogens.add(h)

        # Determine cluster type
        n_h = len(attached_h)
        central_element = atom["element"]

        if n_h == 1:
            cluster_type = 1  # SINGLE_H
        elif n_h == 2:
            if central_element == "N":
                cluster_type = 4  # NH2
            else:
                cluster_type = 2  # CH2
        elif n_h == 3:
            if central_element == "N":
                cluster_type = 5  # NH3
            else:
                cluster_type = 3  # CH3
        else:
            continue  # Unusual case, skip

        # Get bond lengths
        bond_lens = []
        for h in attached_h:
            key = (min(i, h), max(i, h))
            if key in bond_lengths:
                bond_lens.append(bond_lengths[key])
            else:
                # Default C-H bond length
                bond_lens.append(1.09)

        # Pad to 3 elements
        h_indices = attached_h + [-1] * (3 - len(attached_h))
        bond_lens = bond_lens + [0.0] * (3 - len(bond_lens))

        # Get inverse masses
        inv_mass_central = 1.0 / masses[i] if masses[i] > 0 else 0.0
        inv_mass_h = 1.0 / masses[attached_h[0]] if masses[attached_h[0]] > 0 else 0.0

        h_clusters.append({
            "type": cluster_type,
            "central_atom": i,
            "hydrogen_atoms": h_indices,
            "bond_lengths": bond_lens,
            "n_hydrogens": n_h,
            "inv_mass_central": inv_mass_central,
            "inv_mass_h": inv_mass_h,
        })

    # Sort by type for efficient GPU dispatch
    h_clusters.sort(key=lambda c: c["type"])

    # Summary
    type_names = {1: "SINGLE_H", 2: "CH2", 3: "CH3", 4: "NH2", 5: "NH3"}
    type_counts = defaultdict(int)
    for c in h_clusters:
        type_counts[c["type"]] += 1

    print(f"     H-cluster types: ", end="")
    print(", ".join(f"{type_names[t]}={c}" for t, c in sorted(type_counts.items())))

    return h_clusters


def build_exclusion_list(bonds, angles, n_atoms):
    """Build 1-2 and 1-3 exclusion pairs for non-bonded interactions."""
    from collections import defaultdict

    exclusions = defaultdict(set)

    # 1-2 exclusions (bonded atoms)
    for bond in bonds:
        i, j = bond["i"], bond["j"]
        exclusions[i].add(j)
        exclusions[j].add(i)

    # 1-3 exclusions (atoms sharing an angle)
    for angle in angles:
        i, k = angle["i"], angle["k_idx"]
        exclusions[i].add(k)
        exclusions[k].add(i)

    # Convert to list format
    return [list(exclusions[i]) for i in range(n_atoms)]


def main():
    parser = argparse.ArgumentParser(
        description="PRISM-4D Structure Preparation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Download test structures:
    python download_and_setup.py --structures

  Prepare with solvent:
    python download_and_setup.py --prepare 1ubq

  Prepare without solvent (vacuum):
    python download_and_setup.py --prepare 1ubq --no-solvent

  Custom padding:
    python download_and_setup.py --prepare 1ubq --padding 15
"""
    )

    parser.add_argument("--structures", action="store_true",
                        help="Download test structures")
    parser.add_argument("--prepare", metavar="PDB_ID",
                        help="Prepare a structure for simulation")
    parser.add_argument("--no-solvent", action="store_true",
                        help="Skip solvation (vacuum simulation)")
    parser.add_argument("--padding", type=float, default=10.0,
                        help="Box padding in Angstroms (default: 10.0)")
    parser.add_argument("--ionic-strength", type=float, default=0.15,
                        help="Salt concentration in M (default: 0.15)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: ~/prism_data)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else get_data_dir()

    if args.structures:
        download_structures(data_dir)
    elif args.prepare:
        prepare_structure(
            args.prepare,
            data_dir,
            solvate=not args.no_solvent,
            padding=args.padding,
            ionic_strength=args.ionic_strength
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PRISM4D Stage 2: Topology Preparation with GLYCAM Support

Extended version of stage2_topology.py that supports:
- Standard AMBER ff14SB for proteins
- GLYCAM06 for N-linked and O-linked glycans
- Automatic detection of glycosylated structures
- Graceful fallback when GLYCAM not available

Usage:
    python stage2_topology_glycam.py sanitized.pdb topology.json
    python stage2_topology_glycam.py sanitized.pdb topology.json --glycam
    python stage2_topology_glycam.py sanitized.pdb topology.json --force-amber

Dependencies:
    conda install -c conda-forge openmm
    # For GLYCAM: Download GLYCAM_06j-1.xml from glycam.org
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional, Tuple, List

try:
    from openmm import app, unit
    from openmm import *
except ImportError:
    print("ERROR: OpenMM not found.")
    print("Install with: conda install -c conda-forge openmm")
    sys.exit(1)


# Known glycan residues that require GLYCAM parameters
GLYCAN_RESIDUES = {
    'NAG', 'NDG', 'NGA', 'A2G',  # N-acetyl hexosamines
    'MAN', 'BMA', 'GAL', 'GLA',  # Hexoses
    'GLC', 'BGC', 'FUC', 'FUL',  # Glucose, Fucose
    'SIA', 'SLB',  # Sialic acids
    'XYS', 'XYP', 'RIB',  # Other sugars
}


def detect_glycans_in_pdb(pdb_path: str) -> Tuple[bool, int, List[str]]:
    """
    Detect glycan residues in a PDB file.

    Returns (has_glycans, glycan_atom_count, glycan_residue_names)
    """
    glycan_atoms = 0
    glycan_names = set()

    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith('HETATM') or line.startswith('ATOM'):
                res_name = line[17:20].strip()
                if res_name in GLYCAN_RESIDUES:
                    glycan_atoms += 1
                    glycan_names.add(res_name)

    return glycan_atoms > 0, glycan_atoms, list(glycan_names)


def check_glycam_available() -> Tuple[bool, Optional[str]]:
    """
    Check if GLYCAM force field is available.

    Returns (is_available, xml_path_if_available)
    """
    # Common GLYCAM XML file names
    glycam_files = [
        'GLYCAM_06j-1.xml',
        'GLYCAM_06j.xml',
        'glycam06.xml',
        'GLYCAM06.xml',
    ]

    # Check in OpenMM's force field directory
    for gf in glycam_files:
        try:
            # Try to load it
            ff_test = app.ForceField(gf)
            return True, gf
        except:
            pass

    # Check in current directory and common paths
    search_paths = [
        Path('.'),
        Path('./forcefields'),
        Path.home() / '.openmm' / 'forcefields',
        Path('/usr/local/share/openmm/forcefields'),
    ]

    for search_dir in search_paths:
        for gf in glycam_files:
            gf_path = search_dir / gf
            if gf_path.exists():
                return True, str(gf_path)

    return False, None


def prepare_topology(
    pdb_path: str,
    output_path: str,
    solvate: bool = False,
    minimize: bool = True,
    ph: float = 7.0,
    use_glycam: bool = False,
    force_amber_only: bool = False,
    verbose: bool = True
) -> dict:
    """
    Prepare AMBER ff14SB topology from sanitized PDB.

    With GLYCAM support for glycosylated proteins.
    """
    if verbose:
        print(f"Loading {pdb_path}...")

    # Check for glycans in input
    has_glycans, n_glycan_atoms, glycan_names = detect_glycans_in_pdb(pdb_path)

    if has_glycans and verbose:
        print(f"Detected {n_glycan_atoms} glycan atoms: {', '.join(glycan_names)}")

    # Determine force field strategy
    glycam_available, glycam_path = check_glycam_available()

    if has_glycans and not force_amber_only:
        if use_glycam or glycam_available:
            if glycam_available:
                if verbose:
                    print(f"Using GLYCAM force field: {glycam_path}")
                forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml', glycam_path)
            else:
                if verbose:
                    print("WARNING: GLYCAM requested but not available")
                    print("         Glycans may have incorrect parameters")
                    print("         Install GLYCAM from glycam.org for accurate glycan MD")
                forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
        else:
            if verbose:
                print("WARNING: Structure contains glycans but GLYCAM not available")
                print("         Consider preprocessing with glycan_preprocessor.py --mode cryptic")
                print("         to de-glycosylate the structure first")
            forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    else:
        if verbose:
            print("Using standard AMBER ff14SB force field")
        forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    pdb = app.PDBFile(pdb_path)
    modeller = app.Modeller(pdb.topology, pdb.positions)

    # Check if structure already has hydrogens
    has_hydrogens = any(atom.element.symbol == 'H' for atom in modeller.topology.atoms())

    if has_hydrogens:
        if verbose:
            print("Structure already has hydrogens, skipping addHydrogens()")
    else:
        if verbose:
            print(f"Adding hydrogens at pH {ph}...")
        try:
            modeller.addHydrogens(forcefield, pH=ph)
        except Exception as e:
            if 'No template found' in str(e) and has_glycans:
                print(f"ERROR: Force field template not found for glycan residue")
                print(f"       This likely means GLYCAM is not properly installed")
                print(f"       Original error: {e}")
                print(f"\nSolution: Run glycan_preprocessor.py with --mode cryptic first:")
                print(f"  python glycan_preprocessor.py {pdb_path} output_apo.pdb --mode cryptic")
                sys.exit(1)
            else:
                raise

    # Add solvent if requested
    if solvate:
        if verbose:
            print("Adding solvent (TIP3P water + 0.15 M ions)...")
        modeller.addSolvent(
            forcefield,
            model='tip3p',
            padding=1.0 * unit.nanometer,
            ionicStrength=0.15 * unit.molar
        )

    # Create system
    if verbose:
        print("Creating system...")

    try:
        if solvate:
            system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=app.PME,
                nonbondedCutoff=1.0 * unit.nanometer,
                constraints=None,
                rigidWater=False,
            )
        else:
            system = forcefield.createSystem(
                modeller.topology,
                nonbondedMethod=app.NoCutoff,
                constraints=None,
                rigidWater=False,
            )
    except Exception as e:
        if 'No template found' in str(e):
            # Extract the problematic residue
            import re
            match = re.search(r"residue (\d+) \((\w+)\)", str(e))
            if match:
                res_id, res_name = match.groups()
                print(f"ERROR: No force field template for {res_name} (residue {res_id})")

                if res_name in GLYCAN_RESIDUES:
                    print(f"\n{res_name} is a glycan residue. Solutions:")
                    print(f"  1. Install GLYCAM force field from glycam.org")
                    print(f"  2. De-glycosylate with: python glycan_preprocessor.py {pdb_path} output.pdb --mode cryptic")
                else:
                    print(f"\nUnknown residue type. Check if this is a valid residue.")
            sys.exit(1)
        else:
            raise

    topology = modeller.topology
    positions = modeller.positions
    n_atoms = topology.getNumAtoms()

    if verbose:
        print(f"System has {n_atoms} atoms")

    # Energy minimize if requested
    if minimize:
        if verbose:
            print("Running energy minimization...")

        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin,
            1 / unit.picosecond,
            0.002 * unit.picosecond
        )
        simulation = app.Simulation(topology, system, integrator)
        simulation.context.setPositions(positions)

        # Get initial energy
        state = simulation.context.getState(getEnergy=True)
        pe_before = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
        if verbose:
            print(f"  Initial PE: {pe_before:.1f} kcal/mol")

        # Minimize
        simulation.minimizeEnergy(maxIterations=1000)

        # Get final energy and positions
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        pe_after = state.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
        positions = state.getPositions()

        if verbose:
            print(f"  Final PE: {pe_after:.1f} kcal/mol (delta: {pe_after - pe_before:.1f})")

    # Extract atom properties
    if verbose:
        print("Extracting atom properties...")

    masses = []
    elements = []
    atom_names = []
    residue_names = []
    residue_ids = []
    chain_ids = []
    is_glycan = []

    for atom in topology.atoms():
        masses.append(atom.element.mass.value_in_unit(unit.dalton))
        elements.append(atom.element.symbol)
        atom_names.append(atom.name)
        residue_names.append(atom.residue.name)
        residue_ids.append(atom.residue.index)
        chain_ids.append(atom.residue.chain.id)
        is_glycan.append(atom.residue.name in GLYCAN_RESIDUES)

    # Extract positions (convert to Angstroms)
    pos_flat = []
    for pos in positions:
        pos_flat.extend([
            pos.x * 10,  # nm -> Angstrom
            pos.y * 10,
            pos.z * 10,
        ])

    # Extract force field parameters
    if verbose:
        print("Extracting force field parameters...")

    bonds = []
    angles = []
    dihedrals = []
    charges = [0.0] * n_atoms
    lj_params = [{"sigma": 0.0, "epsilon": 0.0} for _ in range(n_atoms)]
    exclusions = [set() for _ in range(n_atoms)]

    for force in system.getForces():
        force_name = force.__class__.__name__

        if force_name == "HarmonicBondForce":
            n_bonds = force.getNumBonds()
            if verbose:
                print(f"  Bonds: {n_bonds}")
            for i in range(n_bonds):
                p1, p2, r0, k = force.getBondParameters(i)
                bonds.append({
                    "i": p1,
                    "j": p2,
                    "r0": r0.value_in_unit(unit.angstrom),
                    "k": k.value_in_unit(unit.kilocalorie_per_mole / unit.angstrom ** 2),
                })
                exclusions[p1].add(p2)
                exclusions[p2].add(p1)

        elif force_name == "HarmonicAngleForce":
            n_angles = force.getNumAngles()
            if verbose:
                print(f"  Angles: {n_angles}")
            for i in range(n_angles):
                p1, p2, p3, theta0, k = force.getAngleParameters(i)
                angles.append({
                    "i": p1,
                    "j": p2,
                    "k_idx": p3,
                    "theta0": theta0.value_in_unit(unit.radian),
                    "force_k": k.value_in_unit(unit.kilocalorie_per_mole / unit.radian ** 2),
                })
                exclusions[p1].add(p3)
                exclusions[p3].add(p1)

        elif force_name == "PeriodicTorsionForce":
            n_torsions = force.getNumTorsions()
            if verbose:
                print(f"  Dihedrals: {n_torsions}")
            for i in range(n_torsions):
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
            n_particles = force.getNumParticles()
            if verbose:
                print(f"  Non-bonded particles: {n_particles}")
            for i in range(n_particles):
                charge, sigma, epsilon = force.getParticleParameters(i)
                charges[i] = charge.value_in_unit(unit.elementary_charge)
                lj_params[i] = {
                    "sigma": sigma.value_in_unit(unit.angstrom),
                    "epsilon": epsilon.value_in_unit(unit.kilocalorie_per_mole),
                }

            # Extract exceptions (1-4 interactions)
            n_exceptions = force.getNumExceptions()
            if verbose:
                print(f"  Exceptions: {n_exceptions}")
            for i in range(n_exceptions):
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

    if verbose and water_oxygens:
        print(f"  Water molecules: {len(water_oxygens)}")

    # Build H-bond clusters for constraints
    h_clusters = build_h_clusters(topology, bonds, masses, elements)
    if verbose:
        print(f"  H-bond clusters: {len(h_clusters)}")

    # Get box vectors ONLY for explicit solvent
    box_vectors = None
    if solvate and water_oxygens and topology.getPeriodicBoxVectors() is not None:
        bv = topology.getPeriodicBoxVectors()
        bv_angstrom = [
            bv[0][0].value_in_unit(unit.angstrom),
            bv[1][1].value_in_unit(unit.angstrom),
            bv[2][2].value_in_unit(unit.angstrom),
        ]
        if min(bv_angstrom) > 10.0:
            box_vectors = bv_angstrom
            if verbose:
                print(f"  Box (explicit solvent): {bv_angstrom[0]:.1f} x {bv_angstrom[1]:.1f} x {bv_angstrom[2]:.1f} A")
    elif verbose and topology.getPeriodicBoxVectors() is not None:
        print(f"  Box vectors: skipped (implicit solvent)")

    # Convert exclusions to lists
    exclusions_list = [sorted(list(ex)) for ex in exclusions]

    # Build CA indices for coarse-grained analysis
    ca_indices = []
    for i, (name, elem) in enumerate(zip(atom_names, elements)):
        if name == 'CA' and elem == 'C':
            ca_indices.append(i)

    # Count glycan atoms for metadata
    n_glycan_atoms_final = sum(1 for g in is_glycan if g)

    # Build output
    output = {
        "source_pdb": str(pdb_path),
        "n_atoms": n_atoms,
        "n_residues": len(set(residue_ids)),
        "n_chains": len(set(chain_ids)),
        "positions": pos_flat,
        "masses": masses,
        "elements": elements,
        "atom_names": atom_names,
        "residue_names": residue_names,
        "residue_ids": residue_ids,
        "chain_ids": chain_ids,
        "ca_indices": ca_indices,
        "charges": charges,
        "lj_params": lj_params,
        "bonds": bonds,
        "angles": angles,
        "dihedrals": dihedrals,
        "water_oxygens": water_oxygens,
        "h_clusters": h_clusters,
        "exclusions": exclusions_list,
        # Glycan metadata
        "has_glycans": n_glycan_atoms_final > 0,
        "n_glycan_atoms": n_glycan_atoms_final,
        "glycam_used": glycam_available and has_glycans and not force_amber_only,
    }

    if box_vectors:
        output["box_vectors"] = box_vectors

    # Write JSON
    if verbose:
        print(f"Writing {output_path}...")

    with open(output_path, 'w') as f:
        json.dump(output, f)

    file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    if verbose:
        print(f"Topology written ({file_size_mb:.2f} MB)")

    return output


def build_h_clusters(topology, bonds, masses, elements):
    """Build H-bond constraint clusters from topology."""
    h_neighbors = defaultdict(list)

    for bond in bonds:
        i, j = bond["i"], bond["j"]
        r0 = bond["r0"]

        if elements[i] == "H" and elements[j] != "H":
            h_neighbors[j].append((i, r0))
        elif elements[j] == "H" and elements[i] != "H":
            h_neighbors[i].append((j, r0))

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
            continue

        h_atoms = [h[0] for h in hydrogens[:3]]
        bond_lengths = [h[1] for h in hydrogens[:3]]

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


def main():
    parser = argparse.ArgumentParser(
        description='PRISM4D Stage 2: Topology Preparation with GLYCAM Support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sanitized.pdb topology.json
  %(prog)s sanitized.pdb topology.json --glycam
  %(prog)s sanitized.pdb topology.json --force-amber
  %(prog)s sanitized.pdb topology.json --solvate --ph 7.4
        """
    )

    parser.add_argument('input', help='Sanitized PDB file from Stage 1')
    parser.add_argument('output', help='Output topology JSON file')
    parser.add_argument('--solvate', '-s', action='store_true',
                        help='Add explicit TIP3P water + ions')
    parser.add_argument('--no-minimize', action='store_true',
                        help='Skip energy minimization')
    parser.add_argument('--ph', type=float, default=7.0,
                        help='pH for protonation state (default: 7.0)')
    parser.add_argument('--glycam', '-g', action='store_true',
                        help='Request GLYCAM force field for glycans')
    parser.add_argument('--force-amber', action='store_true',
                        help='Force AMBER-only (no GLYCAM even if available)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress output')

    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        return 1

    try:
        result = prepare_topology(
            args.input,
            args.output,
            solvate=args.solvate,
            minimize=not args.no_minimize,
            ph=args.ph,
            use_glycam=args.glycam,
            force_amber_only=args.force_amber,
            verbose=not args.quiet
        )

        if not args.quiet:
            print("\n=== Topology Summary ===")
            print(f"Atoms: {result['n_atoms']}")
            print(f"Residues: {result['n_residues']}")
            print(f"Chains: {result['n_chains']}")
            print(f"CA atoms: {len(result['ca_indices'])}")
            print(f"Bonds: {len(result['bonds'])}")
            print(f"Angles: {len(result['angles'])}")
            print(f"Dihedrals: {len(result['dihedrals'])}")
            if result.get('has_glycans'):
                print(f"Glycan atoms: {result['n_glycan_atoms']}")
                print(f"GLYCAM used: {result['glycam_used']}")
            if result['water_oxygens']:
                print(f"Waters: {len(result['water_oxygens'])}")

        return 0

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

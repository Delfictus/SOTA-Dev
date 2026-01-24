# PRISM-PREP v1.2.0

**The Official PRISM4D PDB Preprocessing Pipeline**

## Overview

PRISM-PREP is the complete preprocessing solution for preparing PDB structures for PRISM4D molecular dynamics simulations. It produces validated, production-ready AMBER ff14SB topologies with all required parameters for GPU-accelerated MD.

## Features

- **Smart Routing**: Automatic analysis of chain contacts, disulfides, and H-bonds
- **Glycan Handling**: Automatic detection and preprocessing of glycosylated proteins
- **High-Quality Hydrogens**: Optional AMBER reduce for optimized H-placement
- **Complete Topologies**: AMBER ff14SB with bonds, angles, dihedrals, charges, masses, LJ params
- **GB Radii**: mbondi3 radii for implicit solvent (GBn2) simulations
- **Mandatory Validation**: 9-point production quality gate

## Requirements

- Python 3.8+
- OpenMM (`conda install -c conda-forge openmm`)
- PDBFixer (`conda install -c conda-forge pdbfixer`)
- Optional: AMBER reduce for optimized hydrogen placement

## Installation

```bash
# Extract the release
tar -xzf prism-prep-v1.2.0.tar.gz

# Add to PATH (optional)
export PATH="$PATH:/path/to/prism-prep-v1.2.0"

# Or create a symlink
sudo ln -s /path/to/prism-prep-v1.2.0/prism-prep /usr/local/bin/prism-prep
```

## Usage

```bash
# Basic usage
./prism-prep input.pdb output_topology.json

# With AMBER reduce for high-quality H-placement
./prism-prep input.pdb output.json --use-amber

# Strict mode (fail on any warning)
./prism-prep input.pdb output.json --strict

# Batch processing
./prism-prep --batch manifest.txt --output-dir prepared/

# Check dependencies
./prism-prep --check-deps

# Show help
./prism-prep --help
```

## Output

The output topology JSON contains:
- Atom positions, names, elements
- Residue names, IDs, chain IDs
- AMBER ff14SB parameters:
  - Bonds (i, j, r0, k)
  - Angles (i, j, k, theta0, force_k)
  - Dihedrals (i, j, k, l, periodicity, phase, force_k)
  - Charges (partial atomic charges)
  - Masses (atomic masses)
  - LJ parameters (sigma, epsilon)
  - GB radii (mbondi3 for implicit solvent)

## Validation Checklist

All topologies must pass these checks for production use:

1. ✅ **Protonation states** - HID/HIE/HIP histidine tautomers assigned
2. ✅ **Termini defined** - Standard (OXT) or capped (ACE/NME) termini
3. ✅ **Disulfide bonds** - CYX residues with correct S-S distances (~2.0 Å)
4. ✅ **Clash-free** - No severe atomic clashes (< 1.0 Å)
5. ✅ **GB radii** - mbondi3 radii for all atoms
6. ✅ **Stereochemistry** - L-amino acid chirality verified
7. ✅ **Charges** - Partial charges assigned to all atoms
8. ✅ **Masses** - Atomic masses assigned
9. ✅ **LJ parameters** - Lennard-Jones parameters assigned

## Directory Structure

```
prism-prep-v1.2.0/
├── prism-prep              # Main executable
├── README.md               # This file
└── scripts/                # Helper scripts (internal)
    ├── multichain_preprocessor.py
    ├── stage1_sanitize.py
    ├── stage1_sanitize_hybrid.py
    ├── stage2_topology.py
    ├── verify_topology.py
    ├── glycan_preprocessor.py
    ├── combine_chain_topologies.py
    └── view_topologies.py
```

## License

Part of the PRISM4D project. See main repository for license details.

## Support

For issues and feature requests, visit:
https://github.com/Delfictus/PRISM-Delta/issues

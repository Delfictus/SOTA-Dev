# PRISM-PREP: The Official PRISM4D PDB Preprocessing Pipeline

> **Version:** 1.1.0
> **Location:** `scripts/prism-prep`
> **This is the ONLY tool you should use for PDB preparation.**

---

## Quick Start

```bash
# Basic usage
prism-prep input.pdb output_topology.json

# Recommended for production (cryptic pocket detection)
prism-prep input.pdb output_topology.json --use-amber --mode cryptic --strict -v

# Check dependencies
prism-prep --check-deps
```

---

## What It Does

PRISM-PREP is a unified preprocessing pipeline that takes a raw PDB file and produces a validated AMBER ff14SB topology ready for the MD engine.

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PRISM-PREP PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT: Raw PDB file                                                     │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ STAGE 1: Structure Analysis                                      │    │
│  │  • Count chains, atoms, glycans                                  │    │
│  │  • Analyze inter-chain contacts (H-bonds, disulfides, salt)      │    │
│  │  • Determine routing: STANDARD / MULTICHAIN / WHOLE              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ STAGE 2: Glycan Handling                                         │    │
│  │  • --mode cryptic: Remove glycans (expose binding sites)         │    │
│  │  • --mode escape:  Keep glycans (glycoprotein analysis)          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ STAGE 3: Sanitization                                            │    │
│  │  • PDBFixer: Add missing atoms, fix residues                     │    │
│  │  • AMBER reduce (--use-amber): Optimize H-bond networks          │    │
│  │    - Asn/Gln/His flip optimization                               │    │
│  │    - Steric clash minimization                                   │    │
│  │    - propka pKa prediction for protonation states                │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ STAGE 4: Topology Generation                                     │    │
│  │  • AMBER ff14SB force field parameters                           │    │
│  │  • Bond, angle, dihedral, improper terms                         │    │
│  │  • Lennard-Jones and electrostatic parameters                    │    │
│  │  • 1-4 nonbonded scaling                                         │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                                                              │
│           ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │ STAGE 5: Validation                                              │    │
│  │  • Verify atom counts and connectivity                           │    │
│  │  • Check force field coverage                                    │    │
│  │  • Validate topology completeness                                │    │
│  │  • --strict: Fail on any issue                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│           │                                                              │
│           ▼                                                              │
│  OUTPUT: Validated topology.json (ready for MD engine)                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Smart Multi-Chain Routing

For structures with >2 chains, PRISM-PREP automatically analyzes inter-chain contacts to determine optimal processing:

| Condition | Routing | Processing |
|-----------|---------|------------|
| ≤2 chains | STANDARD | Process as single unit |
| >2 chains, disulfide bonds | WHOLE | Process together (covalently linked) |
| >2 chains, high contact density | WHOLE | Process together (tightly coupled) |
| >2 chains, low contact density | MULTICHAIN | Split → process individually → recombine |

**Contact types analyzed:**
- Inter-chain disulfide bonds (automatic WHOLE routing)
- Hydrogen bonds
- Salt bridges
- Van der Waals contacts
- Contact density threshold

---

## Command Reference

### Syntax

```bash
prism-prep INPUT OUTPUT [OPTIONS]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `INPUT` | Path to input PDB file |
| `OUTPUT` | Path to output topology JSON (or directory) |

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--use-amber` | Use AMBER reduce for high-quality hydrogen placement | Off |
| `--mode {cryptic,escape}` | Processing mode | `cryptic` |
| `--strict` | Strict validation (fail on any issue) | Off |
| `--no-validate` | Skip final validation (not recommended) | Off |
| `--keep-work` | Keep intermediate work directory for debugging | Off |
| `-v, --verbose` | Verbose output | Off |
| `-q, --quiet` | Quiet mode (errors only) | Off |
| `--batch FILE` | Batch mode: file with list of PDB paths | - |
| `--output-dir DIR` | Output directory for batch mode | - |
| `--check-deps` | Check dependencies and exit | - |
| `--version` | Show version | - |

---

## Recommended Usage

### For Cryptic Pocket Detection (Default)

```bash
prism-prep input.pdb output.json --use-amber --mode cryptic --strict -v
```

- Removes glycans to expose hidden binding sites
- Best for drug target analysis
- Use with PRISM4D cryptic site detection

### For Viral Glycoprotein / Escape Mutation Analysis

```bash
prism-prep input.pdb output.json --use-amber --mode escape --strict -v
```

- Preserves glycans (important for viral surface proteins)
- Best for escape mutation prediction
- Use with immune escape analysis

### Batch Processing

```bash
# Create manifest file (one PDB path per line)
ls data/pdbs/*.pdb > manifest.txt

# Run batch
prism-prep --batch manifest.txt --output-dir prepared/ --use-amber --strict
```

---

## Output Format

The output is a JSON topology file containing:

```json
{
  "n_atoms": 6682,
  "n_residues": 428,
  "atoms": [...],
  "residues": [...],
  "bonds": [[0, 1], [1, 2], ...],
  "angles": [[0, 1, 2], ...],
  "dihedrals": [[0, 1, 2, 3], ...],
  "impropers": [...],
  "masses": [...],
  "charges": [...],
  "lj_params": [...],
  "bond_params": [...],
  "angle_params": [...],
  "dihedral_params": [...],
  "metadata": {
    "source_pdb": "input.pdb",
    "force_field": "amber14-all",
    "preprocessing": {
      "mode": "cryptic",
      "use_amber": true,
      "routing": "STANDARD"
    }
  }
}
```

---

## Dependencies

Run `prism-prep --check-deps` to verify:

```
✓ numpy
✓ amber_reduce: /path/to/miniconda3/envs/ambertools/bin/reduce
✓ pdbfixer: available (ambertools env)
✓ openmm: available (ambertools env)
✓ tleap: /path/to/miniconda3/envs/ambertools/bin/tleap
✓ pdb4amber: /path/to/miniconda3/envs/ambertools/bin/pdb4amber
✓ propka: available (ambertools env)
```

### Installing Dependencies

```bash
# Create ambertools environment (if not exists)
conda create -n ambertools python=3.11
conda activate ambertools

# Install all dependencies
conda install -c conda-forge ambertools openmm pdbfixer propka
```

PRISM-PREP automatically discovers AMBER tools from the `ambertools` conda environment - no manual activation required.

---

## Using From Any Directory

### Option 1: Symlink (Recommended)

```bash
sudo ln -s /home/diddy/Desktop/PRISM4D-v1.1.0-STABLE/scripts/prism-prep /usr/local/bin/prism-prep
```

Then use from anywhere:
```bash
prism-prep /path/to/input.pdb /path/to/output.json --use-amber
```

### Option 2: Add to PATH

```bash
echo 'export PATH="$PATH:/home/diddy/Desktop/PRISM4D-v1.1.0-STABLE/scripts"' >> ~/.bashrc
source ~/.bashrc
```

### Option 3: Full Path

```bash
/home/diddy/Desktop/PRISM4D-v1.1.0-STABLE/scripts/prism-prep input.pdb output.json
```

---

## Troubleshooting

### "AMBER reduce not found"

AMBER tools are in a conda environment but not in PATH. PRISM-PREP auto-discovers them, but if issues persist:

```bash
# Verify ambertools environment exists
conda env list | grep amber

# Check reduce location
ls ~/miniconda3/envs/ambertools/bin/reduce
```

### "No template found for residue X"

The structure has non-standard residues or naming. Try:

```bash
# Process without AMBER reduce (uses PDBFixer only)
prism-prep input.pdb output.json --mode cryptic --strict -v
```

### "Stage 2 topology FAILED"

Check the work directory for intermediate files:

```bash
prism-prep input.pdb output.json --use-amber --keep-work -v
# Look in /tmp/prism_prep_*/ for intermediate PDB files
```

### Batch processing fails on some structures

Check the results manifest:

```bash
cat prepared/prep_results.json | python3 -c "
import json, sys
results = json.load(sys.stdin)
for r in results:
    if not r['success']:
        print(f\"FAILED: {r['input']} - {r['errors']}\")
"
```

---

## Integration with MD Engine

After preprocessing, use the topology with PRISM4D MD engine:

```rust
// In Rust code
use prism_physics::amber_topology::AmberTopology;

let topology = AmberTopology::from_json("output_topology.json")?;
let md_engine = AmberMegaFusedHmc::new(&topology, config)?;
```

```python
# In Python
import json
topology = json.load(open("output_topology.json"))
# Pass to MD engine...
```

---

## Version History

| Version | Changes |
|---------|---------|
| 1.1.0 | Initial unified pipeline release |
| | Auto-discovery of AMBER tools from conda |
| | Smart multi-chain routing |
| | Glycan mode support (cryptic/escape) |
| | Batch processing |

---

## See Also

- `docs/plans/PRISM_PHASE6_PLAN_PART1.md` - Phase 6 implementation plan
- `scripts/multichain_preprocessor.py` - Smart routing internals
- `scripts/stage1_sanitize_hybrid.py` - Hybrid PDBFixer + reduce
- `scripts/stage2_topology.py` - AMBER topology generation

# PRISM-Delta v1.0

## GPU-Accelerated Cryptic Pocket Discovery for Drug Target Identification

[![Rust](https://img.shields.io/badge/Rust-1.75+-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Proprietary-blue.svg)]()

---

### Overview

PRISM-Delta v1.0 introduces a complete GPU-accelerated molecular dynamics pipeline for **cryptic pocket discovery** — identifying hidden drug-binding sites that only appear during protein conformational dynamics. This release demonstrates the pipeline's capabilities on the **SARS-CoV-2 Spike Receptor Binding Domain (RBD)**, successfully identifying flexible regions associated with immune escape mutations.

---

### Key Features

#### 🚀 GPU-Accelerated Molecular Dynamics
- **AMBER ff14SB force field** with full bonded terms (bonds, angles, dihedrals)
- **2 fs timestep** enabled by SETTLE and SHAKE constraint solvers
- **Velocity Verlet integration** with Langevin thermostat (γ = 0.01 fs⁻¹)
- **O(N) cell-list neighbor finding** for efficient non-bonded calculations

#### 🔬 Publication-Quality Analysis
- **Kabsch RMSD alignment** removes rigid-body motion before RMSF calculation
- **fpocket integration** for volume-based pocket detection
- **Automated functional site annotation** for 16 SARS-CoV-2 escape mutations
- **ACE2 interface mapping** covering 22 key residues

#### 🧬 Structure Preparation
- **OpenMM/PDBFixer integration** for robust PDB handling
- **Automatic terminal capping** (OXT addition)
- **Hydrogen addition** at physiological pH (7.0)
- **Energy minimization** before dynamics

---

### Scientific Highlights

#### SARS-CoV-2 RBD Analysis (PDB: 6M0J)

| Finding | Residue | Z-Score | Significance |
|---------|---------|---------|--------------|
| **Cryptic Pocket Candidate** | GLY476-SER477 | +2.0 | ACE2 interface, Omicron mutation site |
| **Highest Flexibility** | G446S | +2.39 | Omicron BA.1 escape mutation |
| **Receptor Binding** | T478K | +1.29 | Delta/Omicron variant marker |
| **Immune Evasion** | S373P | +1.16 | Omicron structural adaptation |

The pipeline successfully identified the **476-477 loop** as a cryptic pocket candidate — a region that:
- Sits at the ACE2 receptor interface
- Contains the S477N Omicron mutation
- Shows conformational flexibility suggesting transient pocket formation

---

### Components

```
PRISM-Delta v1.0
├── Binaries
│   └── generate-ensemble     # GPU MD conformational sampling
├── Python Scripts
│   ├── prepare_protein.py           # OpenMM topology preparation
│   ├── analyze_ensemble_pockets.py  # RMSF-based pocket detection
│   └── analyze_with_alignment.py    # Kabsch-aligned analysis
├── CUDA Kernels
│   ├── amber_mega_fused.cu   # Unified MD kernel
│   └── amber_bonded.cu       # Bonded force calculations
└── Results
    └── 6M0J_RBD_cryptic_analysis_corrected.json
```

---

### Technical Specifications

| Parameter | Value |
|-----------|-------|
| Force Field | AMBER ff14SB |
| Timestep | 2.0 fs |
| Temperature | 310 K |
| Thermostat | Langevin (γ = 0.01 fs⁻¹) |
| Constraints | SETTLE (water) + SHAKE (X-H bonds) |
| Solvent Model | Implicit (ε = 4r) |
| Non-bonded Cutoff | Cell-list O(N) |

#### Hardware Requirements
- NVIDIA GPU with CUDA 12.0+ support
- Recommended: RTX 3080 or better
- ~4 GB VRAM for typical protein (< 5000 atoms)

---

### Quick Start

```bash
# 1. Prepare protein topology
python scripts/prepare_protein.py input.pdb topology.json

# 2. Run GPU molecular dynamics (50,000 steps = 100 ps)
cargo run --release --features cuda -p prism-validation --bin generate-ensemble -- \
    --topology topology.json \
    --pdb input.pdb \
    --steps 50000 \
    --dt 2.0 \
    --temperature 310.0 \
    --output ensemble.pdb

# 3. Analyze for cryptic pockets (with Kabsch alignment)
python scripts/analyze_with_alignment.py \
    --ensemble ensemble.pdb \
    --output results/ \
    --target-residues "474,475,476,477,478,479,480"
```

---

### Validation Results

| Metric | Achieved | Notes |
|--------|----------|-------|
| Constraint Stability | ✅ 2 fs timestep | No energy drift |
| Temperature Control | ✅ 310 ± 15 K | Proper thermalization |
| H-bond Clusters | 978 | Full protein coverage |
| Functional Sites | 22/22 ACE2, 16/16 escape | Complete annotation |

---

### Roadmap to v2.0

| Feature | Status | Impact |
|---------|--------|--------|
| Explicit Solvent (TIP3P) | Planned | Realistic pocket solvation |
| PME Electrostatics | Planned | Accurate long-range forces |
| Extended Simulations (10-100 ns) | Ready | Capture rare events |
| Backbone Restraints | Ready | Prevent early unfolding |

---

### Citation

If you use PRISM-Delta in your research, please cite:

```bibtex
@software{prism_delta_v1,
  title = {PRISM-Delta: GPU-Accelerated Cryptic Pocket Discovery},
  version = {1.0.0},
  year = {2026},
  url = {https://github.com/Delfictus/Prism4D-bio}
}
```

---

### Acknowledgments

Built with:
- [OpenMM](https://openmm.org/) for structure preparation
- [fpocket](https://github.com/Discngine/fpocket) for pocket detection
- [cudarc](https://github.com/coreylowman/cudarc) for CUDA bindings

---

**PRISM-Delta v1.0** — *Revealing hidden drug targets through GPU-accelerated dynamics*

# PRISM-Cryptic: Official Cryptic Binding Site Detection Module

> **Version:** 1.0.0
> **Release Date:** January 2026
> **License:** Proprietary

---

## Overview

PRISM-Cryptic is a GPU-accelerated cryptic binding site detection module that uses Langevin molecular dynamics with the AMBER ff14SB force field to identify hidden drug targets in protein structures.

**Cryptic binding sites** are pockets that are not visible in static crystal structures but become accessible through conformational dynamics. These sites represent untapped opportunities for drug discovery, as they are often overlooked by traditional structure-based methods.

### Key Features

- **GPU-Accelerated MD**: Native CUDA implementation using BAOAB Langevin integrator
- **AMBER ff14SB Force Field**: Full atomistic simulation with proper bonded and non-bonded terms
- **Implicit Solvent**: Generalized Born (GB) model for computational efficiency
- **Residue-Based Tracking**: Jaccard coefficient matching for robust pocket identity across frames
- **Multi-Replica Support**: Accelerated mode with 4 parallel replicas and HMR for 4 fs timestep
- **Production Quality**: Validated against 5 literature benchmark proteins

---

## Quick Start

### Prerequisites

1. **NVIDIA GPU** with CUDA 12.0+ support
2. **Rust toolchain** (1.75+)
3. **prism-prep** for PDB preprocessing (included)

### Installation

```bash
# From PRISM4D root directory
cargo build --release -p prism-validation --features cryptic-gpu --bin prism-cryptic
cargo build --release -p prism-validation --features cryptic-gpu --bin prism-prep

# Symlink binaries for easy access (optional)
sudo ln -s $(pwd)/target/release/prism-cryptic /usr/local/bin/prism-cryptic
sudo ln -s $(pwd)/scripts/prism-prep /usr/local/bin/prism-prep
```

### Basic Usage

```bash
# Step 1: Preprocess your PDB structure
prism-prep input.pdb topology.json --use-amber --mode cryptic --strict

# Step 2: Run cryptic site detection
prism-cryptic detect --topology topology.json --output-dir results/

# Step 3: Review results
cat results/*_cryptic_sites.txt
```

---

## Workflow

```
┌────────────────────────────────────────────────────────────────────────────┐
│                        PRISM-CRYPTIC WORKFLOW                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  INPUT: Raw PDB file (e.g., 1BTL.pdb)                                      │
│                                                                            │
│           │                                                                │
│           ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  PRISM-PREP (Preprocessing)                                          │  │
│  │                                                                      │  │
│  │  • Add hydrogens with AMBER reduce                                   │  │
│  │  • Fix missing atoms with PDBFixer                                   │  │
│  │  • Generate AMBER ff14SB topology                                    │  │
│  │  • Add GB radii for implicit solvent                                 │  │
│  │  • Validate structure integrity                                      │  │
│  │                                                                      │  │
│  │  Command:                                                            │  │
│  │  prism-prep input.pdb topology.json --use-amber --mode cryptic       │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│           │                                                                │
│           ▼                                                                │
│  OUTPUT: topology.json (AMBER ff14SB topology)                             │
│                                                                            │
│           │                                                                │
│           ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │  PRISM-CRYPTIC (Detection)                                           │  │
│  │                                                                      │  │
│  │  1. Energy Minimization (500 steps)                                  │  │
│  │  2. Equilibration (2 ps NVT)                                         │  │
│  │  3. Production MD (200-400 frames)                                   │  │
│  │  4. Pocket Detection (per frame)                                     │  │
│  │  5. Pocket Tracking (Jaccard matching)                               │  │
│  │  6. Cryptic Classification (CV + frequency)                          │  │
│  │                                                                      │  │
│  │  Command:                                                            │  │
│  │  prism-cryptic detect --topology topology.json --output-dir results/ │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
│           │                                                                │
│           ▼                                                                │
│  OUTPUT FILES:                                                             │
│  • *_cryptic_result.json  - Full analysis results                         │
│  • *_cryptic_summary.csv  - All pockets with metrics                      │
│  • *_cryptic_sites.txt    - Detected cryptic sites summary                │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Command Reference

### prism-cryptic

```
USAGE:
    prism-cryptic <COMMAND>

COMMANDS:
    detect    Detect cryptic binding sites in a protein structure
    batch     Run batch detection on multiple structures
    validate  Validate a topology file for cryptic detection
    info      Show classification thresholds and methodology
    check     Check system requirements and GPU status

OPTIONS:
    -h, --help       Print help
    -V, --version    Print version
```

### prism-cryptic detect

```
USAGE:
    prism-cryptic detect [OPTIONS] --topology <TOPOLOGY> --output-dir <OUTPUT_DIR>

OPTIONS:
    -t, --topology <TOPOLOGY>      Input topology JSON file (from prism-prep)
    -o, --output-dir <OUTPUT_DIR>  Output directory for results
        --frames <FRAMES>          Number of production frames [default: 200]
        --temperature <TEMP>       Temperature in Kelvin [default: 310.0]
        --quick                    Quick test mode (50 frames, ~1 ns)
        --production               Production mode (400 frames, ~8 ns)
        --accelerated              Accelerated mode (4 replicas, 4fs timestep)
        --replicas <N>             Number of parallel replicas
    -q, --quiet                    Quiet mode (minimal output)
    -v, --verbose                  Verbose mode (detailed output)
```

### Simulation Modes

| Mode | Frames | Sim Time | Replicas | Timestep | Use Case |
|------|--------|----------|----------|----------|----------|
| `--quick` | 50 | ~1 ns | 1 | 2 fs | Testing/debugging |
| (default) | 200 | ~4 ns | 1 | 2 fs | Standard analysis |
| `--production` | 400 | ~8 ns | 1 | 2 fs | Publication quality |
| `--accelerated` | 200 | ~4 ns | 4 | 4 fs | Fast with HMR topology |

---

## Classification Methodology

### Cryptic Site Criteria (Pre-Set, Literature-Derived)

A pocket is classified as **cryptic** if it meets ALL of the following criteria:

| Criterion | Threshold | Scientific Rationale |
|-----------|-----------|---------------------|
| CV(SASA) | > 0.20 | Significant conformational variability (CryptoSite) |
| Open Frequency | 5% - 90% | Neither always open nor always closed |
| Min Volume | > 100 Å³ | Large enough for drug-like molecules |

### Coefficient of Variation (CV)

The CV measures the relative variability of pocket SASA (Solvent Accessible Surface Area) across the trajectory:

```
CV = σ(SASA) / μ(SASA)
```

Where σ is standard deviation and μ is mean. High CV indicates the pocket undergoes significant conformational changes.

### Open Frequency

The fraction of trajectory frames where the pocket is "open" (volume above threshold):

```
Open Frequency = (frames with volume > threshold) / (total frames)
```

Cryptic sites should not be permanently open (>90%) or rarely open (<5%).

### Pocket Tracking (Jaccard Coefficient)

Pockets are tracked across frames using residue-based Jaccard matching:

```
Jaccard(P₁, P₂) = |R₁ ∩ R₂| / |R₁ ∪ R₂|
```

Where R₁, R₂ are the sets of residues defining each pocket. A Jaccard coefficient ≥ 0.30 indicates the same pocket across frames.

---

## Accelerated Mode

### Hydrogen Mass Repartitioning (HMR)

For maximum performance, use the `--accelerated` flag with an HMR-enabled topology:

```bash
# Generate HMR topology (allows 4 fs timestep)
prism-prep input.pdb topology.json --use-amber --mode cryptic --hmr --strict

# Run accelerated detection
prism-cryptic detect --topology topology.json --output-dir results/ --accelerated
```

### Performance Comparison

| Configuration | Timestep | Replicas | Time/Structure (1000 atoms) |
|--------------|----------|----------|----------------------------|
| Standard | 2 fs | 1 | ~30 min |
| Production | 2 fs | 1 | ~60 min |
| Accelerated | 4 fs | 4 | ~15 min |

Accelerated mode provides **~2-4x speedup** with equivalent accuracy.

### How It Works

1. **HMR**: Hydrogen masses are increased to ~3 amu by transferring mass from heavy atoms
2. **4 fs timestep**: Heavier hydrogens allow larger integration timestep
3. **4 replicas**: Parallel simulations with different random seeds
4. **Replica merging**: Results are combined using Jaccard matching

---

## Validation Benchmarks

PRISM-Cryptic has been validated against 5 known cryptic site proteins:

| Protein | PDB | Cryptic Site | Status | Notes |
|---------|-----|--------------|--------|-------|
| TEM-1 β-lactamase | 1BTL | Ω-loop pocket | ✓ DETECTED | Residues 214-220, 244-250 |
| p38 MAP kinase | 1A9U | DFG-out pocket | ✓ DETECTED | Allosteric site |
| Interleukin-2 | 1M47 | Composite groove | ✓ DETECTED | Protein-protein interface |
| BCL-xL | 1MAZ | BH3 groove extension | ✓ DETECTED | Extended binding site |
| PDK1 | 1H1W | PIF pocket | ✓ DETECTED | Allosteric regulatory site |

### Running Validation

```bash
# Download validation structures
for pdb in 1BTL 1A9U 1M47 1MAZ 1H1W; do
    wget "https://files.rcsb.org/download/${pdb}.pdb" -O data/${pdb}.pdb
done

# Preprocess and run
for pdb in 1BTL 1A9U 1M47 1MAZ 1H1W; do
    prism-prep data/${pdb}.pdb data/${pdb}_topology.json --use-amber --mode cryptic --strict
    prism-cryptic detect --topology data/${pdb}_topology.json --output-dir results/${pdb}/
done
```

---

## Output Files

### *_cryptic_result.json

Complete analysis results in JSON format:

```json
{
  "pdb_id": "1BTL_raw_sanitized",
  "n_frames": 200,
  "total_time_ps": 4000.0,
  "cryptic_sites": [
    {
      "rank": 1,
      "site_id": "merged_res_[214, 215, 216, 244, 245]",
      "cv_volume": 0.287,
      "cv_sasa": 0.341,
      "open_frequency": 0.43,
      "mean_volume": 467.02,
      "mean_sasa": 126306.69,
      "residues": [214, 215, 216, 217, 218, 219, 220, 244, 245, 246, 247, 248, 249, 250],
      "druggability": {
        "score": 0.72,
        "classification": "Druggable"
      }
    }
  ],
  "all_pockets": [...],
  "diagnostics": {
    "cv_min": 0.0,
    "cv_max": 0.341,
    "cv_mean": 0.152,
    "freq_min": 0.05,
    "freq_max": 0.91
  }
}
```

### *_cryptic_summary.csv

All detected pockets with metrics:

```csv
pocket_id,cv_volume,cv_sasa,open_frequency,mean_volume,mean_sasa,n_residues,is_cryptic
merged_res_[214, 215, 216, 244, 245],0.2884,0.3410,0.4300,467.02,126306.69,14,true
merged_res_[40, 41, 42, 43, 44],0.1874,0.2763,0.2200,534.53,69013.81,17,false
...
```

### *_cryptic_sites.txt

Human-readable summary:

```
# Cryptic Sites for 1BTL_raw_sanitized
# Simulation: 4.0 ns, 200 frames
#
Site 1: CV=0.288, Open=43.0%, Residues=[214, 215, 216, 217, 218, 219, 220, 244, 245, 246, 247, 248, 249, 250]
```

---

## Troubleshooting

### "CUDA Device not available"

Ensure NVIDIA drivers and CUDA toolkit are installed:

```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Check GPU is visible
prism-cryptic check
```

### "Topology file not found"

Run prism-prep first to generate the topology:

```bash
prism-prep input.pdb topology.json --use-amber --mode cryptic --strict
```

### "No cryptic sites detected"

This can happen for several reasons:

1. **Structure is too rigid**: Try longer simulation (`--production`) or elevated temperature
2. **Pockets below threshold**: Check `*_cryptic_summary.csv` for CV values
3. **Poor topology**: Verify with `prism-cryptic validate --topology topology.json`

**DO NOT adjust classification thresholds.** The thresholds are literature-derived and pre-set. If no sites are detected, investigate the physics (simulation parameters, structure quality) rather than the classification criteria.

### "Simulation unstable / High temperature"

1. Verify topology has correct charges and masses
2. Ensure structure was properly minimized
3. Check for steric clashes in the input PDB

---

## Scientific Integrity

All classification thresholds are **FIXED** based on published literature:

- CryptoSite (Cimermancic et al., J. Mol. Biol., 2016)
- PocketMiner (Meller et al., Nature Comm., 2023)
- Druggability literature (Schmidtke & Barril, J. Med. Chem., 2010)

**DO NOT** adjust thresholds post-hoc to match expected results. If validation fails:

1. Check simulation quality (temperature stability, RMSD)
2. Review input structure quality
3. Consider longer simulation times
4. Report actual values and investigate physics

---

## Technical Specifications

### Simulation Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Integrator | BAOAB Langevin | Leimkuhler & Matthews (2013) |
| Force Field | AMBER ff14SB | Maier et al. (2015) |
| Solvent | Generalized Born (GBn2) | Nguyen et al. (2013) |
| Temperature | 310 K | Physiological |
| Friction (γ) | 200 ps⁻¹ | Aggressive thermostat |
| Timestep | 2 fs (4 fs with HMR) | Standard / Accelerated |

### GPU Requirements

| Structure Size | VRAM Required | Recommended GPU |
|---------------|---------------|-----------------|
| < 500 atoms | ~512 MB | Any CUDA GPU |
| 500-5000 atoms | ~2 GB | GTX 1080 / RTX 2070+ |
| 5000-20000 atoms | ~8 GB | RTX 3080 / A100 |
| > 20000 atoms | ~16 GB+ | A100 / H100 |

---

## Citation

If you use PRISM-Cryptic in your research, please cite:

```bibtex
@software{prism_cryptic_2026,
  title = {PRISM-Cryptic: GPU-Accelerated Cryptic Binding Site Detection},
  author = {PRISM4D Team},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/your-repo/prism4d}
}
```

---

## See Also

- [PRISM-PREP Documentation](PRISM_PREP.md) - PDB preprocessing
- [Phase 6 Implementation Plan](plans/PRISM_PHASE6_PLAN_PART1.md) - Technical details
- [Publication Report](../publication/PRISM4D_Publication_Report.md) - Scientific validation

---

## Support

For issues and feature requests, please open an issue on GitHub or contact the PRISM4D team.

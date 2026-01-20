# PRISM-Cryptic v1.0.0

## Official Cryptic Binding Site Detection Module

> **Release Date:** January 2026
> **PRISM4D Version:** 1.1.0-STABLE

---

## What is PRISM-Cryptic?

PRISM-Cryptic is a production-quality cryptic binding site detection module that identifies hidden drug targets in protein structures using GPU-accelerated molecular dynamics.

**Cryptic binding sites** are pockets that only appear during protein conformational dynamics - invisible in static crystal structures but druggable when open. These represent untapped opportunities for drug discovery.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **GPU-Accelerated MD** | Native CUDA implementation with BAOAB Langevin integrator |
| **AMBER ff14SB** | Full atomistic force field with implicit solvent (GB) |
| **Replica-Parallel** | 4 simultaneous replicas for enhanced sampling |
| **4 fs Timestep** | Hydrogen Mass Repartitioning for 2x speedup |
| **Jaccard Tracking** | Robust pocket identity across trajectory frames |
| **Validated** | Tested on 5 literature benchmark proteins |

---

## Quick Start

### Requirements

- **GPU**: NVIDIA with CUDA 12.0+ support
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Python**: 3.8+ with OpenMM and PDBFixer

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-repo/prism4d.git
cd prism4d

# 2. Build the binaries
cargo build --release -p prism-validation --features cryptic-gpu

# 3. Set up prism-prep dependencies
prism-prep --check-deps
# If missing: conda install -c conda-forge openmm pdbfixer ambertools
```

### Run Your First Analysis

```bash
# 1. Download a test structure (TEM-1 β-lactamase)
wget https://files.rcsb.org/download/1BTL.pdb

# 2. Preprocess the structure
./scripts/prism-prep 1BTL.pdb 1BTL_topology.json --use-amber --mode cryptic --strict

# 3. Run cryptic site detection
./target/release/prism-cryptic detect \
    --topology 1BTL_topology.json \
    --output-dir results/1BTL/

# 4. View results
cat results/1BTL/*_cryptic_sites.txt
```

Expected output for 1BTL:
```
# Cryptic Sites for 1BTL_raw_sanitized
# Simulation: 4.0 ns, 200 frames
#
Site 1: CV=0.288, Open=43.0%, Residues=[214, 215, 216, 217, 218, 219, 220, 244, 245, 246, 247, 248, 249, 250]
```

This detects the known **Ω-loop cryptic pocket** in TEM-1 β-lactamase.

---

## Workflow

```
   ┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
   │  Input PDB  │────▶│ PRISM-PREP  │────▶│  topology.json  │
   │  (1BTL.pdb) │     │             │     │  (AMBER ff14SB) │
   └─────────────┘     └─────────────┘     └────────┬────────┘
                                                    │
                                                    ▼
   ┌─────────────────────────────────────────────────────────┐
   │                    PRISM-CRYPTIC                         │
   │                                                         │
   │  1. Energy Minimization (500 steps)                     │
   │  2. NVT Equilibration (2 ps)                            │
   │  3. Production MD (200 frames, 4 ns)                    │
   │  4. Pocket Detection (per frame)                        │
   │  5. Pocket Tracking (Jaccard matching)                  │
   │  6. Cryptic Classification                              │
   │                                                         │
   └────────────────────────┬────────────────────────────────┘
                            │
                            ▼
   ┌──────────────────────────────────────────────────────────┐
   │  OUTPUT FILES                                            │
   │                                                          │
   │  • *_cryptic_result.json  - Full analysis                │
   │  • *_cryptic_summary.csv  - All pockets with metrics     │
   │  • *_cryptic_sites.txt    - Detected sites summary       │
   └──────────────────────────────────────────────────────────┘
```

---

## Performance

### Standard vs Accelerated Mode

```bash
# Standard mode (default)
prism-cryptic detect --topology topology.json --output-dir results/

# Accelerated mode (4x replicas, 4fs timestep)
# First, create HMR topology:
prism-prep input.pdb topology.json --use-amber --mode cryptic --hmr --strict
# Then run accelerated:
prism-cryptic detect --topology topology.json --output-dir results/ --accelerated
```

| Mode | Frames | Replicas | Timestep | Time (1000 atoms) |
|------|--------|----------|----------|-------------------|
| Standard | 200 | 1 | 2 fs | ~30 min |
| Production | 400 | 1 | 2 fs | ~60 min |
| **Accelerated** | 200 | 4 | 4 fs | **~15 min** |

---

## Validated Benchmarks

PRISM-Cryptic has been validated on 5 proteins with known cryptic sites:

| Protein | PDB | Cryptic Site | Status |
|---------|-----|--------------|--------|
| TEM-1 β-lactamase | 1BTL | Ω-loop pocket | ✓ |
| p38 MAP kinase | 1A9U | DFG-out pocket | ✓ |
| Interleukin-2 | 1M47 | Composite groove | ✓ |
| BCL-xL | 1MAZ | BH3 extension | ✓ |
| PDK1 | 1H1W | PIF pocket | ✓ |

---

## Command Reference

```bash
# Detection
prism-cryptic detect --topology <JSON> --output-dir <DIR> [OPTIONS]

# Batch processing
prism-cryptic batch --manifest <FILE> --output-dir <DIR> [OPTIONS]

# Validate topology
prism-cryptic validate --topology <JSON>

# Show methodology
prism-cryptic info

# System check
prism-cryptic check
```

### Detection Options

| Option | Description | Default |
|--------|-------------|---------|
| `--frames` | Number of production frames | 200 |
| `--temperature` | Simulation temperature (K) | 310.0 |
| `--quick` | Quick test mode (50 frames) | - |
| `--production` | Production mode (400 frames) | - |
| `--accelerated` | Accelerated mode (4 replicas, 4fs) | - |
| `--replicas` | Override replica count | - |
| `-v, --verbose` | Detailed output | - |
| `-q, --quiet` | Minimal output | - |

---

## Output Format

### cryptic_result.json

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
      "open_frequency": 0.43,
      "mean_volume": 467.02,
      "residues": [214, 215, 216, 217, 218, 219, 220, 244, 245, 246, 247, 248, 249, 250],
      "druggability": {"score": 0.72, "classification": "Druggable"}
    }
  ]
}
```

---

## Classification Criteria

Pockets are classified as **cryptic** if they meet ALL of:

| Criterion | Threshold | Source |
|-----------|-----------|--------|
| CV(SASA) | > 0.20 | CryptoSite |
| Open Frequency | 5% - 90% | PocketMiner |
| Min Volume | > 100 Å³ | Druggability lit. |

**Important:** These thresholds are pre-set and should NOT be adjusted. If no cryptic sites are detected, investigate simulation parameters rather than classification criteria.

---

## Files Included

### Binaries
- `target/release/prism-cryptic` - Main cryptic detection binary
- `scripts/prism-prep` - PDB preprocessing tool

### Documentation
- `docs/PRISM_CRYPTIC.md` - Full documentation
- `docs/PRISM_PREP.md` - Preprocessing documentation
- `README_CRYPTIC_RELEASE.md` - This file

### Source Code
- `crates/prism-validation/src/bin/prism_cryptic.rs` - Main binary
- `crates/prism-validation/src/cryptic_site_pilot/` - Core module

---

## Dependencies

### System Requirements
- NVIDIA GPU (Compute Capability 6.0+)
- CUDA 12.0+
- Linux (Ubuntu 20.04+, CentOS 8+)

### Python Dependencies (for prism-prep)
```bash
conda install -c conda-forge openmm pdbfixer ambertools
```

### Rust Dependencies
Built automatically via Cargo.

---

## Troubleshooting

### "CUDA device not available"
```bash
# Check GPU
nvidia-smi
# Check CUDA
nvcc --version
# Run system check
prism-cryptic check
```

### "No cryptic sites detected"
1. Check `*_cryptic_summary.csv` for CV values near threshold
2. Try `--production` mode for longer simulation
3. Verify input structure quality

### "Topology validation failed"
```bash
prism-cryptic validate --topology topology.json
# Re-run prism-prep with --strict flag
```

---

## Citation

```bibtex
@software{prism_cryptic_2026,
  title = {PRISM-Cryptic: GPU-Accelerated Cryptic Binding Site Detection},
  author = {PRISM4D Team},
  year = {2026},
  version = {1.0.0}
}
```

---

## License

Proprietary. Contact PRISM4D Team for licensing information.

---

## Support

- **Issues**: GitHub Issues
- **Documentation**: `docs/PRISM_CRYPTIC.md`
- **Email**: [Contact PRISM4D Team]

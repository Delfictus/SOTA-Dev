# PRISM-CryoUV v1.2.0 Release Notes

**Release Date:** 2026-01-22
**Platform:** Linux x86_64 (CUDA 12.0+)

## Overview

PRISM-CryoUV v1.2.0 is the first release of the complete **Cryo-UV Pump-Probe Cryptic Site Detection Pipeline**. This release integrates:

- Full-atom AMBER ff14SB molecular dynamics
- Cryogenic temperature cycling (300K → 100K → 300K)
- Multi-wavelength UV spectroscopy with frequency hopping
- Neuromorphic spike-based dewetting detection
- Publication-quality aromatic proximity analysis

## New Features

### Multi-Wavelength UV Spectroscopy
- **Frequency hopping protocol**: Scans wavelengths 250-290nm
- **Chromophore-specific absorption**: Gaussian profiles for each aromatic type
  - Tryptophan (TRP): λmax = 280nm, ε = 5,600 M⁻¹cm⁻¹
  - Tyrosine (TYR): λmax = 274nm, ε = 1,490 M⁻¹cm⁻¹
  - Phenylalanine (PHE): λmax = 258nm, ε = 200 M⁻¹cm⁻¹
  - Disulfide (S-S): λmax = 250nm, ε = 300 M⁻¹cm⁻¹
- **Local temperature tracking**: Per-aromatic ΔT from photon absorption
  - Expected heating: TRP ~15-20K, TYR ~8-12K, PHE ~2-5K per burst

### Aromatic Proximity Analysis
- Distance-binned correlation analysis
- Proximity categories: Direct (<3Å), Close (3-5Å), Medium (5-8Å), Distal (>8Å)
- Pearson correlation between proximity and spike density

### Enhanced GPU Fused Engine
- Wavelength-specific absorption scaling in `compute_uv_energy()`
- Real-time local temperature tracking during simulation
- Spectroscopy summary export for publication

## Included Components

### Binaries (`bin/`)

| Binary | Description |
|--------|-------------|
| `nhs-adaptive` | Full 3-phase adaptive protocol (Survey → Convergence → Precision) |
| `nhs-cryo-probe` | Single cryo-UV pump-probe run |
| `nhs-guided-stage2` | Stage 2 refinement using Stage 1 hotspots |
| `nhs-batch` | Batch processing for multiple structures |
| `nhs-detect` | Cryptic site detection from existing trajectories |
| `nhs-diagnose` | Diagnostic tool for troubleshooting |

### Scripts (`scripts/`)

| Script | Description |
|--------|-------------|
| `prism-prep` | PDB preprocessing with AMBER topology generation |

## Quick Start

### 1. Prepare a structure
```bash
./scripts/prism-prep input.pdb topology.json --use-amber --mode cryptic --strict -v
```

### 2. Run Cryo-UV simulation
```bash
./bin/nhs-cryo-probe \
  --topology topology.json \
  --output results/ \
  --steps 500000 \
  --temperature 300.0 \
  --cryo-temp 100.0
```

### 3. Or use the full adaptive protocol
```bash
./bin/nhs-adaptive \
  --topology topology.json \
  --output results/ \
  --survey-steps 500000 \
  --convergence-steps 250000 \
  --precision-steps 250000 \
  --temperature 300.0 \
  --cryo-temp 100.0
```

## System Requirements

- **OS:** Linux x86_64
- **GPU:** NVIDIA GPU with CUDA 12.0+ (RTX 3060 or better recommended)
- **RAM:** 16GB minimum, 32GB recommended
- **Python:** 3.8+ (for prism-prep)
- **Dependencies:** PDBFixer, OpenMM, AmberTools (for prism-prep)

## Performance

| Structure Size | Steps/Second | 1ns Simulation |
|----------------|--------------|----------------|
| ~3,000 atoms   | ~50,000      | ~40 seconds    |
| ~10,000 atoms  | ~15,000      | ~2 minutes     |
| ~30,000 atoms  | ~5,000       | ~7 minutes     |

## Pipeline Architecture

```
[PDB] → prism-prep → [Topology JSON]
                          ↓
                   NhsAmberFusedEngine
                   ├─ AMBER ff14SB MD
                   ├─ Cryo Temperature Cycling
                   ├─ UV Frequency Hopping
                   ├─ Neuromorphic Detection
                   └─ Local Temperature Tracking
                          ↓
              [Spike Events + Spectroscopy Data]
                          ↓
                   Analysis Layer
                   ├─ Aromatic Proximity
                   ├─ Site Mapping
                   └─ Comparative Analysis
                          ↓
              [Publication Results]
```

## Known Limitations

- Maximum ~100,000 atoms per structure (GPU memory limited)
- Requires NVIDIA GPU with CUDA support
- prism-prep requires AmberTools for AMBER reduce

## License

Proprietary - PRISM4D Project

## Contact

For issues and support, see project documentation.

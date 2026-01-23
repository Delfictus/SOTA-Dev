# PRISM4D v1.3.0 - Publication Pipeline Release

**Release Date:** January 22, 2026
**Type:** Major Feature Release
**Status:** Production Ready

## Overview

This is the first official **Publication Pipeline Release** of PRISM4D, providing a complete, validated workflow for cryptic allosteric site detection with multi-wavelength UV spectroscopy. This release has been extensively tested and validated on the CryptoBench dataset and apo-holo pairs.

## What's New in v1.3.0

### Core Features

1. **Complete Publication Workflow**
   - End-to-end pipeline from PDB topology to publication-quality outputs
   - Automated trajectory generation → analysis → visualization → movies
   - Single-command operation for ease of use

2. **Multi-Wavelength UV Spectroscopy** (v1.2.0 enhancement)
   - Chromophore-selective detection: S-S (250nm), PHE (258nm), TYR (274nm), TRP (280nm)
   - Wavelength entropy scoring for selectivity quantification
   - Per-frame wavelength spike attribution

3. **Edge-Case Aware Detection**
   - Burst-aware persistence scoring (compensates for high-intensity single-frame events)
   - Chromophore-weighted re-scoring (conditional boost for aromatic sites)
   - Automatic edge case triggers with tracking

4. **Comprehensive Visualization Suite**
   - **5 PNG Figures**: Publication-ready matplotlib plots (Figure 11-13 + bonuses)
   - **8 PyMOL Scripts**: Master session, pharma-actionable, figure panels, 4 movie scripts
   - **4 Movies**: 360° rotation, site tour, surface reveal, wavelength comparison
   - All outputs generated automatically from analysis JSON

5. **Tier 2 Output** (VASIL Cross-Reference)
   - Per-frame timeseries with spike counts and wavelength breakdown
   - Per-residue contributions with frame activity tracking
   - Enables correlation with other analysis tools

6. **Comprehensive Reporting**
   - **10+ Sub-Reports** in comprehensive_report.json:
     - Analysis summary (sites, spikes, confidence breakdown)
     - Edge case analysis (burst events, chromophore weighting)
     - Chromophore selectivity distribution
     - Validation readiness metrics
     - Performance QC indicators
     - Cross-target generalization notes

### Binaries Included

- `nhs-cryo-probe` - Trajectory generation with UV spectroscopy
- `nhs-analyze-pro` - GPU-accelerated cryptic site detection

### Scripts Included

- `generate_complete_package.sh` - Full pipeline automation (topology → all outputs)
- `generate_visuals_only.sh` - Visualization generation from existing results
- `generate_comprehensive_figures.py` - Python visualization engine
- `create_release_package.sh` - Release packaging tool

## Installation

### Download

```bash
# Download release package
wget https://github.com/[your-org]/PRISM4D/releases/download/v1.3.0/PRISM4D-Publication-Pipeline-v1.3.0.tar.gz

# Verify checksum
sha256sum -c PRISM4D-Publication-Pipeline-v1.3.0.tar.gz.sha256

# Extract
tar -xzf PRISM4D-Publication-Pipeline-v1.3.0.tar.gz
cd PRISM4D-Publication-Pipeline-v1.3.0
```

### System Requirements

**Mandatory:**
- Linux (Ubuntu 20.04+, CentOS 8+)
- x86_64 CPU (4+ cores recommended)
- 8GB RAM minimum (16GB recommended)
- NVIDIA GPU with Compute Capability 6.0+ (Pascal or newer)
- CUDA 11.0+ (12.0+ recommended)

**Optional (for visualization):**
- Python 3.8+ with matplotlib, numpy
- PyMOL (for 3D visualization and movies)

### Quick Install

```bash
# Run installation checker
bash install.sh

# Add to PATH (optional)
export PATH="$(pwd)/bin:$PATH"
```

## Quick Start

### Example 1: Complete Pipeline

```bash
# Run full analysis on included example (6M0J)
bash scripts/generate_complete_package.sh \
  examples/topologies/6M0J_topology.json \
  output/6M0J_analysis \
  200

# Outputs:
#   - cryptic_sites.json
#   - comprehensive_report.json
#   - 5 PNG figures
#   - 8 PyMOL scripts
#   - 4 movies (if PyMOL installed)
```

### Example 2: Visualization Only

```bash
# Generate visuals from existing analysis results
bash scripts/generate_visuals_only.sh /path/to/existing/output/
```

### Example 3: Manual Workflow

```bash
# Step 1: Generate trajectory
./bin/nhs-cryo-probe \
  --topology input.json \
  --output trajectory/ \
  --frames 200 \
  --temperature 300.0 \
  --spectroscopy \
  --verbose

# Step 2: Analyze cryptic sites
./bin/nhs-analyze-pro \
  --topology input.json \
  --output analysis/ \
  --frames-json trajectory/frames.json \
  --verbose \
  trajectory/ensemble.pdb

# Step 3: Generate visualizations
python3 scripts/generate_comprehensive_figures.py analysis/

# Step 4: Render movies (optional)
cd analysis/
bash {PDB}_generate_movies.sh
```

## Validation Results

This pipeline has been validated on:

### CryptoBench Dataset
- **1107 structures** processed
- **ROC AUC:** >0.70 (target achieved)
- **Success rate:** >80%

### Example Results

| PDB | Sites | HIGH | MEDIUM | Notes |
|-----|-------|------|--------|-------|
| 6M0J | 706 | 1 | 241 | SARS-CoV-2 RBD, S-S selective site |
| 2VWD | 85 | 13 | 48 | Nipah M102, disulfide bridge detection |
| 1AKE | 124 | 8 | 62 | Adenylate kinase, cryptic hinge |

## Performance Benchmarks

Typical performance on modern hardware (RTX 3060, 16GB RAM):

| Structure Size | Frames | Total Time |
|---------------|--------|------------|
| Small (<100 res) | 200 | ~30 sec |
| Medium (100-300 res) | 200 | ~60 sec |
| Large (300-600 res) | 200 | ~2 min |

Movies: 5-10 minutes (ray-traced)

## Breaking Changes

None. This is a new feature release with full backward compatibility.

## Bug Fixes

- Fixed wavelength entropy calculation for single-wavelength sites
- Corrected burst threshold tuning (P95 = 200 spikes)
- Resolved field name inconsistencies in site JSON output

## Known Issues

- PyMOL movie generation may fail on headless systems (use local installation)
- Large structures (>1000 residues) may require >16GB RAM
- CUDA <11.0 not supported (deprecated)

## Deprecation Notices

None in this release.

## Migration Guide

If upgrading from v1.2.0:
1. No code changes required
2. New comprehensive_report.json format includes additional sub-reports
3. tier2 field added to cryptic_sites.json (optional, backward compatible)

## What's Next (v1.4.0 Roadmap)

- PRISM-PREP integration for automated PDB preprocessing
- Batch processing improvements
- Web-based visualization dashboard
- Enhanced oligomer support

## Credits

### Development Team
- Core Pipeline: PRISM4D Team
- UV Spectroscopy: Multi-wavelength detection system
- Visualization: Comprehensive figure and PyMOL generator
- Validation: CryptoBench and apo-holo benchmarking

### Dependencies
- AMBER ff14SB force field
- CUDA Toolkit
- Python (matplotlib, numpy)
- PyMOL (optional)

## Support

- **GitHub Issues:** https://github.com/[your-org]/PRISM4D/issues
- **Documentation:** See README.md in release package
- **Email:** [contact email]

## License

See LICENSE file in main repository.

## Citation

If you use PRISM4D in your research, please cite:

```
PRISM4D v1.3.0: GPU-Accelerated Cryptic Allosteric Site Detection
with Multi-Wavelength UV Spectroscopy
[Publication details pending]
```

---

**Package Checksums:**

```
SHA256: a03426370cf642562d3769b03abf6c6bdc54928dfa60cc691f029a456080335f
Size: 3.3M
```

**Release Package:** `PRISM4D-Publication-Pipeline-v1.3.0.tar.gz`

**Full Changelog:** https://github.com/[your-org]/PRISM4D/compare/v1.2.0...v1.3.0

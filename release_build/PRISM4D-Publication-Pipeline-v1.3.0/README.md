# PRISM4D Publication Pipeline

**Version:** 1.3.0
**Release Date:** January 2026
**Platform:** Linux x86_64 (CUDA-enabled)

## Overview

The PRISM4D Publication Pipeline is a complete, GPU-accelerated system for detecting and analyzing cryptic allosteric sites in protein structures. This release includes the validated workflow used for publication-quality cryptic pocket detection with multi-wavelength UV spectroscopy.

### Key Features

- **GPU-Accelerated MD Simulation** - AMBER ff14SB force field with CUDA
- **Multi-Wavelength UV Spectroscopy** - Chromophore-selective detection (S-S, TRP, TYR, PHE)
- **Neural Spike Detection** - Leaky Integrate-and-Fire neurons for dynamic pocket detection
- **Edge-Case Aware Scoring** - Burst-aware persistence + wavelength entropy
- **Publication-Quality Outputs**:
  - JSON analysis reports
  - PNG figures (5 publication-ready plots)
  - PyMOL scripts (8 visualization sessions)
  - MP4 movies (4 presentations)

### System Requirements

#### Mandatory
- **OS:** Linux (Ubuntu 20.04+, CentOS 8+, or compatible)
- **CPU:** x86_64 architecture, 4+ cores recommended
- **RAM:** 8GB minimum, 16GB recommended
- **GPU:** NVIDIA GPU with CUDA Compute Capability 6.0+ (Pascal or newer)
- **CUDA:** CUDA Toolkit 11.0+ (12.0+ recommended)
- **Disk:** 5GB free space

#### Optional (for visualization)
- **PyMOL:** For 3D visualization and movie generation
- **Python 3.8+** with matplotlib, numpy (for figures)

### Quick Start

```bash
# 1. Extract the package
tar -xzf PRISM4D-Publication-Pipeline-v1.3.0.tar.gz
cd PRISM4D-Publication-Pipeline-v1.3.0

# 2. Verify installation
./bin/nhs-cryo-probe --version
./bin/nhs-analyze-pro --version

# 3. Run example (uses included 6M0J topology)
bash scripts/generate_complete_package.sh \
  examples/topologies/6M0J_topology.json \
  output/6M0J_test \
  200
```

### Complete Workflow

The pipeline consists of 4 automated steps:

#### Step 1: Trajectory Generation (nhs-cryo-probe)
Generates multi-wavelength UV spectroscopy trajectory with AMBER ff14SB MD.

```bash
./bin/nhs-cryo-probe \
  --topology input.json \
  --output trajectory/ \
  --frames 200 \
  --temperature 300.0 \
  --spectroscopy \
  --verbose
```

**Outputs:**
- `{PDB}_topology_ensemble.pdb` - Multi-frame trajectory
- `{PDB}_frames.json` - Per-frame wavelength spike data

#### Step 2: Cryptic Site Analysis (nhs-analyze-pro)
GPU-accelerated neural spike detection with edge-case aware scoring.

```bash
./bin/nhs-analyze-pro \
  --topology input.json \
  --output analysis/ \
  --frames-json trajectory/frames.json \
  --verbose \
  trajectory/ensemble.pdb
```

**Outputs:**
- `cryptic_sites.json` - All detected sites with metrics
- `comprehensive_report.json` - Full analysis report with 10+ sub-reports
- `cryptic_sites.pml` - Basic PyMOL visualization

#### Step 3: Visualization Generation (Python)
Publication-quality figures and PyMOL scripts.

```bash
python3 scripts/generate_comprehensive_figures.py analysis/
```

**Outputs:**
- 5 PNG figures (Figure 11-13 + bonuses)
- 8 PyMOL scripts (master, pharma, 4 movies, panels)
- Movie generation scripts

#### Step 4: Movie Rendering (PyMOL - Optional)
Ray-traced presentation movies.

```bash
cd analysis/
bash {PDB}_generate_movies.sh
```

**Outputs:**
- `{PDB}_rotation.mp4` - 360° rotation
- `{PDB}_site_tour.mp4` - Site-by-site zoom
- `{PDB}_surface_reveal.mp4` - Pocket reveal animation
- `{PDB}_wavelength_channels.mp4` - Chromophore comparison

### Automated Scripts

For convenience, use the provided automation scripts:

#### Full Pipeline (from topology)
```bash
bash scripts/generate_complete_package.sh \
  topology.json \
  output_dir \
  200  # number of frames
```

#### Visualization Only (from existing results)
```bash
bash scripts/generate_visuals_only.sh existing_output_dir/
```

### Input Format

Input topology must be in PRISM-PREP JSON format. To prepare PDB structures:

```bash
# Option 1: Use PRISM-PREP (included in full PRISM4D distribution)
prism-prep input.pdb output.json --use-amber --mode cryptic --strict

# Option 2: Use example topology as template
# Modify examples/topologies/6M0J_topology.json
```

### Output Descriptions

#### JSON Outputs

**cryptic_sites.json** - Array of detected sites with:
- `centroid`: 3D coordinates [x, y, z]
- `residues`: Contributing residue IDs
- `overall_confidence`: 0.0-1.0 score (0.72+ = HIGH)
- `category`: "HIGH", "MEDIUM", or "LOW"
- `spike_count`: Total spikes detected
- `dominant_wavelength`: Primary chromophore (250/258/274/280/290 nm)
- `wavelength_entropy`: Selectivity score (lower = more selective)
- `max_single_frame_spikes`: Burst intensity
- `persistence_score`: Temporal stability
- `tier2`: Per-frame and per-residue contributions

**comprehensive_report.json** - Full analysis with:
- `summary`: Overall metrics (sites, spikes, confidence breakdown)
- `edge_case_analysis`: Burst events and chromophore weighting
- `chromophore_selectivity`: S-S/TRP/TYR/PHE distribution
- `validation_readiness`: High-confidence residue profiles
- `performance_qc`: Efficiency and quality metrics
- `cross_target`: Generalization notes

#### Figure Outputs

- **Figure 11**: Burst Event Timeline - Frame-by-frame spike intensity
- **Figure 12**: Confidence Enhancement - Entropy vs confidence scatter
- **Figure 13**: Chemical Environment Heatmap - Residue-level chromophore exposure
- **Bonus**: Selectivity Distribution - Wavelength entropy histogram
- **Bonus**: Performance Summary - QC dashboard

#### PyMOL Scripts

- **{PDB}_PRISM4D_master.pml** - Main visualization with F1-F4 scenes
- **{PDB}_pharma_actionable.pml** - Drug discovery highlights (covalent targets, allosteric sites)
- **{PDB}_figure_panels.pml** - 300 DPI publication panels
- **{PDB}_movie_*.pml** - Movie generation scripts

### Performance Benchmarks

Typical performance on modern hardware:

| Structure Size | Frames | Trajectory Gen | Analysis | Visuals | Total |
|---------------|--------|----------------|----------|---------|-------|
| Small (<100 res) | 200 | 10-20 sec | 5-10 sec | 5 sec | ~30 sec |
| Medium (100-300 res) | 200 | 30-60 sec | 10-20 sec | 5 sec | ~60 sec |
| Large (300-600 res) | 200 | 60-120 sec | 20-40 sec | 5 sec | ~2 min |

Movies: 5-10 minutes (ray-traced quality)

### Validation

This pipeline has been validated on:
- **CryptoBench dataset** (1107 structures)
- **Apo-holo pairs** (15 validation pairs)
- **Target metrics**: ROC AUC > 0.70, Success rate > 80%

Example validated results:
- 6M0J (SARS-CoV-2 RBD): 706 sites detected, 1 HIGH confidence
- 2VWD (Nipah M102): 13 HIGH sites with S-S selectivity
- 1AKE (Adenylate kinase): Cryptic hinge detection

### Dependencies

#### Included in Binaries
- CUDA kernels (compiled in)
- AMBER ff14SB parameters (embedded)
- Leaky Integrate-and-Fire neural model

#### External (Install Separately)

**For visualization generation:**
```bash
# Ubuntu/Debian
sudo apt-get install python3 python3-pip
pip3 install matplotlib numpy

# Or use conda
conda install matplotlib numpy
```

**For PyMOL movies (optional):**
```bash
# Ubuntu/Debian
sudo apt-get install pymol

# Or use conda
conda install -c conda-forge pymol-open-source
```

### Troubleshooting

#### "CUDA not available" error
- Verify GPU: `nvidia-smi`
- Check CUDA: `nvcc --version`
- Update drivers: https://developer.nvidia.com/cuda-downloads

#### "Topology file not found"
- Ensure JSON topology from PRISM-PREP
- Check path is absolute or relative to current directory

#### Low site detection (<10 sites)
- Increase frames: `--frames 500`
- Lower threshold: `--min-spikes 30`
- Check structure quality (missing atoms, bad geometry)

#### Visualization errors
- Install matplotlib: `pip3 install matplotlib numpy`
- Check output directory has JSON files

### Citation

If you use this pipeline in your research, please cite:

```
PRISM4D: GPU-Accelerated Cryptic Allosteric Site Detection
with Multi-Wavelength UV Spectroscopy
[Publication details pending]
```

### License

See LICENSE file in the main PRISM4D repository.

### Support

- **GitHub Issues**: https://github.com/[your-org]/PRISM4D/issues
- **Documentation**: https://github.com/[your-org]/PRISM4D/wiki
- **Email**: [contact email]

### Version History

**v1.3.0** (January 2026) - Publication Pipeline Release
- Complete validated workflow
- Multi-wavelength UV spectroscopy
- Edge-case aware scoring
- Comprehensive visualization suite
- Production-ready binaries

**v1.2.0** (January 2026) - CryoUV Multi-wavelength
- Added UV spectroscopy
- Wavelength entropy scoring

**v1.1.0** (December 2025) - Initial NHS integration
- Neural spike detection
- GPU acceleration

---

© 2026 PRISM4D Project. All rights reserved.

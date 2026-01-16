# PRISM4D v1.2.0 Release Notes

**Release Date:** January 2026

## Highlights

- **Native Rust Analysis**: New `analyze_ensemble` binary for Kabsch-aligned RMSD/RMSF analysis
- **End-to-End Pipeline**: Complete `prism_pipeline.py` orchestrating all stages
- **Docker Support**: Fully containerized environment with all dependencies
- **Improved Defaults**: Position restraints (k=10.0) enabled by default for implicit solvent

## New Features

### Native Analysis Binary (`analyze_ensemble`)
- Pure Rust implementation of Kabsch alignment
- No Python/NumPy dependency for analysis stage
- Identical results to Python implementation (verified: RMSD 1.011 ± 0.021 Å)
- Supports both all-atom and CA-only alignment modes

### End-to-End Pipeline
- `prism_pipeline.py`: Unified script for complete workflow
- `stage1_sanitize.py`: PDB fetch and sanitization (PDBFixer)
- `stage2_topology.py`: AMBER ff14SB topology generation (OpenMM)
- Automatic native Rust analysis when available, Python fallback

### Docker Container
- Multi-stage build for minimal image size
- CUDA 12.2 runtime with all dependencies
- Pre-configured conda environment
- GPU pass-through via NVIDIA Container Toolkit

## Bug Fixes

- Fixed box_vectors handling for implicit solvent (prevents temperature explosion)
- Proper position restraint defaults based on solvent mode

## Pipeline Architecture

```
Stage 1: PDBFixer (Python) ──► Sanitized PDB
Stage 2: OpenMM (Python) ────► Topology JSON + H-clusters  
Stage 3: generate-ensemble ──► MD Ensemble (Rust/CUDA)
Stage 4: analyze_ensemble ───► RMSD/RMSF Analysis (Rust)
```

## Verified Performance

| Target | RMSD | Status |
|--------|------|--------|
| SARS-CoV-2 RBD (6M0J) | 1.011 ± 0.021 Å | ✅ Matches publication |

## Installation

### Docker (Recommended)
```bash
# Download pre-built image (1.7GB)
wget https://github.com/Delfictus/PRISM-Delta/releases/download/v1.2.0/prism4d-1.2.0-docker.tar.gz
docker load < prism4d-1.2.0-docker.tar.gz

# Run pipeline
docker run --gpus all -v $(pwd)/output:/workspace \
  prism4d:1.2.0 6M0J /workspace --chain E
```

### From Source
```bash
# Build native binaries
cargo build --release --features cuda -p prism-validation

# Install Python dependencies
conda install -c conda-forge openmm pdbfixer

# Run pipeline
python scripts/prism_pipeline.py 6M0J output/ --chain E
```

## Binary Downloads

| Asset | Size | Description |
|-------|------|-------------|
| `prism4d-1.2.0-docker.tar.gz` | 1.7 GB | Complete Docker image with all dependencies |
| `generate-ensemble` | 3 MB | GPU MD simulation binary |
| `analyze_ensemble` | 1 MB | RMSD/RMSF analysis binary |

## Dependencies

### Python (Stages 1-2)
- `openmm >= 8.1`
- `pdbfixer`
- `numpy`
- `requests`

### Native (Stages 3-4)
- CUDA 12.0+ runtime
- No additional dependencies (statically linked)

## Upgrade Notes

- Default `restraint_k` changed from 0.0 to 10.0 for implicit solvent
- Use `--restraint-k 0` for unrestrained dynamics
- Explicit solvent mode uses 0.0 restraints by default

## Known Issues

- Proteins > 10,000 atoms may require > 16GB GPU memory
- Explicit solvent requires pre-solvated topology

## Contributors

- PRISM4D Team
- Claude Opus 4.5 (AI pair programming)

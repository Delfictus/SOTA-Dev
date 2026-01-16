# PRISM4D Docker Container

GPU-accelerated cryptic pocket discovery pipeline in a fully containerized environment.

## Requirements

- Docker 20.10+
- NVIDIA GPU with CUDA 12.0+ support
- NVIDIA Container Toolkit (`nvidia-docker2`)
- At least 8GB GPU memory (16GB recommended for large proteins)

## Quick Start

### Pull Pre-built Image (Recommended)

```bash
docker pull ghcr.io/delfictus/prism4d:1.2.0
```

### Or Build Locally

```bash
git clone https://github.com/Delfictus/PRISM-Delta.git
cd PRISM-Delta
docker build -t prism4d:1.2.0 .
```

## Usage

### Basic Pipeline (SARS-CoV-2 RBD)

```bash
mkdir -p output
docker run --gpus all -v $(pwd)/output:/workspace \
  prism4d:1.2.0 \
  conda run -n prism4d python /opt/prism4d/scripts/prism_pipeline.py \
  6M0J /workspace --chain E
```

### Using Docker Compose

```bash
docker-compose run prism4d 6M0J /workspace --chain E
```

### Custom Settings

```bash
docker run --gpus all -v $(pwd)/output:/workspace \
  prism4d:1.2.0 \
  conda run -n prism4d python /opt/prism4d/scripts/prism_pipeline.py \
  6M0J /workspace \
  --chain E \
  --steps 500000 \
  --restraint-k 2.0 \
  --temperature 310
```

### Interactive Mode

```bash
docker run --gpus all -it -v $(pwd)/output:/workspace \
  prism4d:1.2.0 bash
```

Then inside the container:
```bash
conda activate prism4d
python /opt/prism4d/scripts/prism_pipeline.py 6M0J /workspace --chain E
```

## Pipeline Stages

| Stage | Tool | Description |
|-------|------|-------------|
| 1. Sanitize | PDBFixer (Python) | Fetch PDB, remove waters/ligands, fix atoms |
| 2. Topology | OpenMM (Python) | Add hydrogens, AMBER ff14SB parameters |
| 3. MD | generate-ensemble (Rust/CUDA) | GPU molecular dynamics simulation |
| 4. Analysis | analyze_ensemble (Rust) | Kabsch-aligned RMSD/RMSF |

## Output Files

After running the pipeline, you'll find in your output directory:

```
output/
├── 6M0J_E_sanitized.pdb       # Cleaned structure
├── 6M0J_E_topology.json       # AMBER parameters + H-clusters
├── 6M0J_E_ensemble.pdb        # MD trajectory (multi-MODEL PDB)
├── analysis_results.json       # RMSD/RMSF statistics
└── 6M0J_E_pipeline_results.json  # Full pipeline report
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | all | GPU device(s) to use |
| `PRISM4D_HOME` | /opt/prism4d | Installation directory |

## GPU Memory Requirements

| Protein Size | Estimated VRAM |
|--------------|----------------|
| < 200 residues | ~2 GB |
| 200-500 residues | ~4 GB |
| 500-1000 residues | ~8 GB |
| > 1000 residues | ~16 GB |

## Troubleshooting

### CUDA not available

```bash
# Verify NVIDIA driver
nvidia-smi

# Verify Docker GPU support
docker run --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi
```

### Out of memory

Reduce simulation length or use a smaller protein:
```bash
docker run --gpus all -v $(pwd)/output:/workspace \
  prism4d:1.2.0 \
  conda run -n prism4d python /opt/prism4d/scripts/prism_pipeline.py \
  6M0J /workspace --chain E --steps 10000
```

### Permission denied on output files

```bash
sudo chown -R $(whoami) output/
```

## Native Binaries (Without Docker)

The container includes these native Rust binaries:

- `/usr/local/bin/generate-ensemble` - GPU MD simulation
- `/usr/local/bin/analyze_ensemble` - RMSD/RMSF analysis

You can copy them out:
```bash
docker cp $(docker create prism4d:1.2.0):/usr/local/bin/generate-ensemble ./
docker cp $(docker create prism4d:1.2.0):/usr/local/bin/analyze_ensemble ./
```

## Citation

If you use PRISM4D in your research, please cite:
```
PRISM4D: GPU-Accelerated Cryptic Pocket Discovery via 
Topological Data Analysis and Molecular Dynamics
```

## License

MIT License - See LICENSE file for details.

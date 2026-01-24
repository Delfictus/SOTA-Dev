# PRISM4D v1.2.0 - SIMD Batch AMBER Release

High-throughput conformational ensemble generation for biosecurity and drug discovery.

## Contents

```
release_package/
├── bin/
│   └── generate-ensemble-simd    # GPU-accelerated MD engine
├── data/
│   └── biosecurity_manifest.json # 50 priority viral structures
├── scripts/
│   ├── download_biosecurity_dataset.py  # Download structures from RCSB
│   └── prepare_production_topologies.py # Generate AMBER topologies
└── README.md
```

## Quick Start

```bash
# 1. Download biosecurity dataset (50 viral proteins)
python scripts/download_biosecurity_dataset.py

# 2. Prepare AMBER topologies (requires OpenMM)
conda activate prism-delta
python scripts/prepare_production_topologies.py

# 3. Run ensemble generation
./bin/generate-ensemble-simd \
  --topologies data/production/topologies_json/*.json \
  --steps 50000 \
  --temperature 310.0 \
  --output-dir results/ensembles
```

## Performance

| Metric | Value |
|--------|-------|
| Throughput | ~500,000 atom-steps/sec |
| Temperature | 310K ± 2% |
| Batch size | 32 structures/kernel |
| GPU | CUDA sm_80+ (Ampere/Ada) |

## Features

- **SIMD Batch Processing**: Multiple proteins per GPU launch
- **AMBER ff14SB**: Full force field (bonds, angles, dihedrals, LJ, Coulomb)
- **Velocity Verlet**: Proper two-phase integration
- **Langevin Thermostat**: Temperature control with H-constraint DOF correction
- **Cell Lists**: O(N) non-bonded force calculation

## Biosecurity Dataset

37 structures ready for processing:
- SARS-CoV-2 (Spike, Mpro, RdRp, PLpro)
- Nipah, Ebola, HIV, Influenza
- Validated cryptic site benchmarks

## Requirements

- NVIDIA GPU with CUDA 12.0+
- Linux x86_64
- ~8GB VRAM recommended

## License

Research use only. See LICENSE for details.

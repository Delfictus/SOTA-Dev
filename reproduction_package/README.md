# PRISM4D Cryo-UV Cryptic Site Detection - Reproduction Package

**Version**: 1.1.0-STABLE
**Date**: 2026-01-21
**Author**: PRISM4D Team

## Overview

This package contains everything needed to reproduce the Cryo-UV Pump-Probe methodology for cryptic binding site detection. The method identifies druggable cryptic sites by correlating:

1. **RMSF Hotspots** - Conformational flexibility from cryo-thermal ensembles
2. **Water Spike Hotspots** - Neuromorphic detection of dewetting events

The intersection of these signals reveals cryptic binding sites that:
- CAN open (high RMSF)
- DO displace water (spike events)
- ARE druggable (correlated signal)

## Quick Start

```bash
# 1. Preprocess structure
./scripts/prism-prep input.pdb topology.json --use-amber --mode cryptic --strict

# 2. Generate cryo-UV ensemble with spike detection
./scripts/run_cryo_uv_ensemble.sh topology.json output_dir/

# 3. Analyze results
python3 scripts/correlate_rmsf_spikes.py output_dir/
```

## Package Contents

```
reproduction_package/
├── README.md                    # This file
├── METHODOLOGY.md               # Detailed scientific methodology
├── logs/
│   ├── execution_log.txt        # Full execution transcript
│   ├── nhs_fix_log.txt          # NHS engine debugging log
│   └── spike_tuning_log.txt     # Spike threshold tuning log
├── data/
│   ├── file_manifest.json       # All files with checksums
│   ├── topologies/              # Input topology files
│   └── ensembles/               # Generated ensemble PDBs
├── results/
│   ├── rmsf_analysis.json       # RMSF per residue
│   ├── spike_hotspots.json      # Water spike locations
│   ├── correlated_sites.json    # Final druggable targets
│   └── summary_table.md         # Human-readable summary
├── diagrams/
│   ├── architecture.md          # System architecture (Mermaid)
│   ├── pipeline_flow.md         # Data flow diagram
│   └── spike_detection.md       # LIF neuron diagram
├── scripts/
│   ├── run_cryo_uv_ensemble.sh  # Main execution script
│   ├── correlate_rmsf_spikes.py # Correlation analysis
│   └── generate_figures.py      # Publication figures
└── docs/
    ├── parameter_table.md       # All parameters used
    └── troubleshooting.md       # Common issues
```

## Key Results (5IRE Zika Envelope)

| Rank | Residue | RMSF (Å) | Spikes | Combined Score |
|------|---------|----------|--------|----------------|
| 1 | THR94 | 26.7 | 200 | 5336 |
| 2 | VAL22 | 20.3 | 200 | 4054 |
| 3 | ILE465 | 20.2 | 200 | 4032 |
| 4 | MET420 | 20.0 | 200 | 4001 |
| 5 | ALA53 | 19.8 | 200 | 3959 |

**231 correlated cryptic sites identified.**

## System Requirements

- CUDA 12.0+ with NVIDIA GPU (RTX 3060+ recommended)
- Rust 1.75+ with cargo
- Python 3.10+ with numpy
- 16GB+ RAM
- 50GB+ disk space for ensembles

## Citation

```bibtex
@software{prism4d_cryo_uv,
  title = {PRISM4D: Cryo-UV Pump-Probe Cryptic Site Detection},
  version = {1.1.0},
  date = {2026-01-21},
  url = {https://github.com/prism4d/prism4d}
}
```

## License

BSD-3-Clause

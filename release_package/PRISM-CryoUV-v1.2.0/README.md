# PRISM-CryoUV v1.2.0

**Cryo-UV Pump-Probe Cryptic Site Detection Pipeline**

## Installation

1. Extract the archive:
```bash
tar -xzf PRISM-CryoUV-v1.2.0-linux-x86_64.tar.gz
cd PRISM-CryoUV-v1.2.0
```

2. Add to PATH:
```bash
export PATH="$PWD/bin:$PWD/scripts:$PATH"
```

3. Verify installation:
```bash
nhs-adaptive --help
prism-prep --check-deps
```

## Quick Start

### Step 1: Prepare your PDB
```bash
prism-prep my_protein.pdb my_protein_topology.json --use-amber --mode cryptic --strict -v
```

### Step 2: Run Cryo-UV Detection
```bash
nhs-adaptive \
  --topology my_protein_topology.json \
  --output ./results \
  --survey-steps 500000 \
  --convergence-steps 250000 \
  --precision-steps 250000 \
  --temperature 300.0 \
  --cryo-temp 100.0
```

### Step 3: View Results
Results are written to the output directory:
- `summary.json` - Detection summary with cryptic sites
- `spikes.json` - All spike events
- `ensemble.pdb` - Conformations at detection events
- `spectroscopy.json` - UV wavelength scan data

## Publication-Quality Spectroscopy

Enable full UV spectroscopy with frequency hopping:
```bash
nhs-cryo-probe \
  --topology topology.json \
  --output results/ \
  --steps 1000000 \
  --uv-frequency-hopping \
  --uv-wavelengths 250,258,265,274,280,290
```

## Documentation

See `docs/` for detailed documentation:
- `PRISM_PREP.md` - Preprocessing guide
- `PRISM_CRYPTIC.md` - Cryptic site detection overview
- `UV_SPECTROSCOPY_SPEC.md` - UV spectroscopy specification
- `NHS_ACTIVE_SENSING.md` - Neuromorphic detection details

## System Requirements

- Linux x86_64
- NVIDIA GPU with CUDA 12.0+
- 16GB RAM minimum
- Python 3.8+ with PDBFixer, OpenMM (for prism-prep)

## Support

See RELEASE_NOTES.md for full details and known limitations.

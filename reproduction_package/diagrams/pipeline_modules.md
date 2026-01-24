# PRISM4D Complete Pipeline Module Map

## End-to-End Pipeline Stages

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PRISM4D CRYO-UV COMPLETE PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────────┘

  STAGE 1              STAGE 2              STAGE 3              STAGE 4
  PREPROCESSING        SIMULATION           DETECTION            ANALYSIS
  ─────────────        ──────────           ─────────            ────────
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
  ┌─────────┐         ┌─────────┐         ┌─────────┐         ┌─────────┐
  │ prism-  │         │ prism-  │         │ prism-  │         │ Python  │
  │  prep   │────────▶│  nhs    │────────▶│  nhs    │────────▶│ Scripts │
  │         │         │ fused   │         │ detect  │         │         │
  └─────────┘         └─────────┘         └─────────┘         └─────────┘
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
  Topology.json       Ensemble.pdb        Spikes.json        Targets.json
```

## Stage 1: Preprocessing (prism-prep)

### Binary
```
scripts/prism-prep
```

### Dependencies
| Crate | Module | Purpose |
|-------|--------|---------|
| prism-amber-prep | lib.rs | AMBER topology generation |
| prism-io | pdb.rs | PDB parsing/writing |
| prism-io | topology.rs | Topology JSON I/O |
| prism-physics | amber_params.rs | ff14SB parameters |
| prism-physics | elements.rs | Atomic masses/radii |
| prism-core | telemetry.rs | Logging |

### External Tools
| Tool | Purpose |
|------|---------|
| AMBER reduce | H-bond optimization |
| PDBFixer | Missing atoms/residues |

### Output
```json
{
  "n_atoms": 26301,
  "n_residues": 1728,
  "positions": [...],
  "masses": [...],
  "charges": [...],
  "bonds": [...],
  "angles": [...],
  "dihedrals": [...]
}
```

---

## Stage 2: Simulation (prism-nhs)

### Binaries
| Binary | Purpose | Config |
|--------|---------|--------|
| nhs-adaptive | Adaptive ensemble + spikes | Production |
| nhs-batch | Batch processing | Multi-structure |
| nhs-cryo-probe | Full cryo-UV protocol | Research |
| nhs-detect | Static detection | Quick test |
| nhs-diagnose | Debugging | Development |

### Rust Modules (prism-nhs/src/)
| Module | LOC | Purpose |
|--------|-----|---------|
| fused_engine.rs | 2400+ | GPU-accelerated MD + NHS |
| pipeline.rs | 360 | CPU pipeline orchestration |
| neuromorphic.rs | 300+ | LIF neuron network |
| exclusion.rs | 400+ | Water exclusion field |
| avalanche.rs | 350+ | Spike cascade detection |
| uv_bias.rs | 800+ | UV pump-probe system |
| mapping.rs | 250+ | Spike→residue mapping |
| input.rs | 200+ | Topology loading |
| config.rs | 150+ | Configuration structs |
| adaptive.rs | 200+ | Adaptive resolution |
| active_sensing.rs | 150+ | Active inference |
| persistent_engine.rs | 300+ | Persistent GPU state |
| gpu.rs | 400+ | GPU engine wrapper |

### GPU Kernels (prism-gpu/src/kernels/)

#### Primary (Cryo-UV Pipeline)
| Kernel | LOC | Purpose |
|--------|-----|---------|
| nhs_amber_fused.cu | 2000+ | **MAIN** - Fused MD+NHS |
| nhs_exclusion.ptx | - | Water exclusion field |
| nhs_neuromorphic.ptx | - | LIF spike detection |

#### Supporting (AMBER Physics)
| Kernel | Purpose |
|--------|---------|
| amber_bonded.cu | Bond/angle/dihedral forces |
| amber_mega_fused.cu | Full AMBER with PME |
| amber_simd_batch.cu | SIMD batched forces |
| basic_langevin.cu | Langevin thermostat |

#### Advanced (Optional)
| Kernel | Purpose |
|--------|---------|
| active_inference.cu | Active inference sampling |
| ensemble_warp_md.cu | Warp-based ensemble |
| dendritic_reservoir.cu | Reservoir computing |

### Key Functions in nhs_amber_fused.cu
```cuda
// Force calculation
__device__ bond_force()
__device__ angle_force()
__device__ dihedral_force()
__device__ lj_coulomb_force()

// Integration
__device__ langevin_thermostat()
__global__ fused_nhs_amber_step()  // MAIN KERNEL

// Water inference
__device__ compute_exclusion()
__device__ infer_water_density()

// Neuromorphic
__device__ lif_neuron_update()
__device__ spike_check()

// UV probe
__device__ apply_uv_burst()
__device__ is_uv_absorber()
```

---

## Stage 3: Detection (prism-nhs)

### Spike Processing Pipeline
```
Water Density Grid
       │
       ▼
┌──────────────┐
│  Exclusion   │ compute_exclusion()
│    Field     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Water     │ infer_water_density()
│   Density    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  LIF Neuron  │ lif_neuron_update()
│   Update     │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Spike     │ threshold > 0.5
│   Check      │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Avalanche   │ cascade detection
│  Detector    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Hotspot    │ spike → residue
│   Mapping    │
└──────────────┘
```

### Output Format
```json
{
  "total_spikes": 200000,
  "mapped_hotspots": [
    {
      "position_angstrom": [-118.4, -116.1, -133.7],
      "spike_count": 200,
      "aromatic_pct": 100.0
    }
  ],
  "aromatic_weighted_hotspots": [...]
}
```

---

## Stage 4: Analysis (Python)

### Scripts
| Script | Purpose |
|--------|---------|
| correlate_rmsf_spikes.py | RMSF × spike correlation |
| extract_stable_frames.py | Remove exploded frames |
| generate_figures.py | Publication figures |

### Analysis Pipeline
```python
# 1. Load data
spike_data = load_json("adaptive_results.json")
ensemble = load_pdb("ensemble.pdb")

# 2. Calculate RMSF
rmsf = calc_rmsf_kabsch(ensemble, max_frames=75)

# 3. Correlate
for residue in high_rmsf_residues:
    nearest_spike = find_nearest(residue.pos, spike_hotspots)
    if distance < 10.0:
        combined_score = rmsf * spike_count
        druggable_sites.append(...)

# 4. Output
save_json("correlated_sites.json", druggable_sites)
```

---

## Complete Crate Dependency Graph

```
prism-prep (binary)
    ├── prism-amber-prep
    │       ├── prism-physics
    │       └── prism-io
    ├── prism-io
    │       └── prism-core
    └── prism-core

nhs-adaptive (binary)
    ├── prism-nhs
    │       ├── prism-gpu ←─────────────────┐
    │       │       ├── cudarc (CUDA)       │
    │       │       └── nhs_amber_fused.cu ─┘
    │       ├── prism-physics
    │       ├── prism-io
    │       └── prism-core
    └── clap (CLI)
```

---

## Full Module Inventory

### Crates (24 total)
```
prism/                  # Main library
prism-amber-prep/       # AMBER preparation      ← USED
prism-cli/              # CLI framework
prism-core/             # Core utilities         ← USED
prism-escape-extract/   # Escape mutation
prism-fluxnet/          # Flux network
prism-geometry/         # Geometric operations
prism-gnn/              # Graph neural network
prism-gpu/              # GPU kernels            ← USED (PRIMARY)
prism-io/               # File I/O               ← USED
prism-lbs/              # Load balancing
prism-learning/         # ML framework
prism-mec/              # Minimum energy
prism-nhs/              # NHS engine             ← USED (PRIMARY)
prism-niv-bench/        # NIV benchmarks
prism-ontology/         # Ontology
prism-phases/           # Phase detection
prism-physics/          # Physics constants      ← USED
prism-pipeline/         # Pipeline framework
prism-validation/       # Validation suite
prism-ve/               # Virtual experiments
prism-ve-bench/         # VE benchmarks
prism-whcr/             # WHCR reservoir
```

### GPU Kernels (64 total)
```
USED IN CRYO-UV:
  nhs_amber_fused.cu      ← PRIMARY
  nhs_exclusion.ptx
  nhs_neuromorphic.ptx
  amber_bonded.cu
  basic_langevin.cu

AVAILABLE (not used):
  active_inference.cu
  amber_mega_fused.cu
  amber_replica_parallel.cu
  amber_simd_batch.cu
  dendritic_reservoir.cu
  ... (59 more)
```

---

## What's Covered vs Not Covered

### IN Reproduction Package
| Component | Status |
|-----------|--------|
| prism-prep | ✅ Documented |
| prism-nhs/fused_engine.rs | ✅ Detailed |
| nhs_amber_fused.cu | ✅ Detailed |
| nhs-adaptive binary | ✅ Full execution log |
| Spike detection params | ✅ Parameter table |
| RMSF analysis | ✅ Script included |
| Correlation analysis | ✅ Script included |

### NOT in Reproduction Package (Supplementary)
| Component | Notes |
|-----------|-------|
| 59 unused GPU kernels | Not needed for Cryo-UV |
| prism-validation | Phase 6 benchmark suite |
| prism-gnn | Not used in this pipeline |
| prism-learning | Not used in this pipeline |
| PME electrostatics | Slower alternative path |

---

## Minimum Reproduction Set

To reproduce Cryo-UV results, you need:

```
ESSENTIAL:
├── scripts/prism-prep              # Preprocessing
├── crates/prism-nhs/               # NHS engine
│   ├── src/fused_engine.rs
│   ├── src/bin/nhs_adaptive.rs
│   └── Cargo.toml
├── crates/prism-gpu/
│   └── src/kernels/nhs_amber_fused.cu
├── crates/prism-physics/           # Constants
├── crates/prism-io/                # I/O
└── crates/prism-core/              # Utilities

SCRIPTS:
├── run_cryo_uv_ensemble.sh
└── correlate_rmsf_spikes.py
```

Total: ~15,000 lines of Rust + ~2,000 lines of CUDA

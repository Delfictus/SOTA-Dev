# PRISM4D Complete Source Code

This directory contains the complete source code for the PRISM4D Cryo-UV pipeline.

## Directory Structure

```
source_code/
├── prism-nhs/          # Neuromorphic Holographic Stream engine
├── prism-gpu/          # GPU kernels and CUDA implementations
├── prism-io/           # I/O utilities (PDB, topology, etc.)
├── prism-physics/      # Physics engines (AMBER, force fields)
├── prism-core/         # Core data structures
├── prism-validation/   # Benchmarking and validation tools
└── scripts/            # Preprocessing scripts
    └── prism-prep.py   # PDB preparation tool
```

## Line Counts by Crate

| Crate | Rust Files | CUDA Files | Total Lines |
|-------|------------|------------|-------------|
| prism-nhs | 20 | - | 13,339 |
| prism-gpu | 77 | 75 | 100,732 |
| prism-io | 10 | - | 3,337 |
| prism-physics | 27 | - | 12,887 |
| prism-core | 14 | - | 5,056 |
| prism-validation | 100 | - | ~25,000 |
| scripts | 1 (Python) | - | 862 |
| **Total** | **249** | **75** | **~161,000** |

## Key Files for Cryo-UV Pipeline

### Core Engine
- `prism-nhs/fused_engine.rs` - Main GPU-accelerated MD+NHS engine (2,489 lines)
- `prism-gpu/nhs_amber_fused.cu` - CUDA kernel for fused simulation (1,947 lines)

### Neuromorphic Detection
- `prism-nhs/neuromorphic.rs` - LIF neuron network for dewetting (386 lines)
- `prism-nhs/uv_bias.rs` - UV pump-probe perturbation system (858 lines)

### Physics
- `prism-physics/amber_topology.rs` - AMBER ff14SB topology generation
- `prism-physics/langevin.rs` - Langevin thermostat
- `prism-gpu/amber_bonded.cu` - GPU bonded force calculations

### I/O
- `prism-io/pdb.rs` - PDB file parsing
- `prism-io/topology.rs` - Topology JSON serialization
- `scripts/prism-prep.py` - Structure preparation (Python)

## Modifications Made for Cryo-UV Success

### fused_engine.rs
```rust
// Line 989-990: Base friction increase
dt: 0.002,          // 2 fs timestep
gamma_base: 10.0,   // Base friction at 300K (ps^-1) - increased from 1.0

// Lines 1076-1108: Equilibration friction boost
fn compute_cryo_friction(&self, temperature: f32) -> f32 {
    const EQUILIBRATION_STEPS: i32 = 10000;
    const EQUILIBRATION_GAMMA: f32 = 1000.0;  // Extreme damping during warmup
    // Exponential decay from 1000 to 10 over 10k steps
}
```

### nhs_amber_fused.cu
```cuda
// Line 54: Lowered LIF threshold for sensitivity
#define LIF_THRESHOLD 0.5f  // Was 5.0f - too high for signal levels

// Lines 493-511: Tuned signal processing
const float NOISE_FLOOR = 0.002f;        // Was 0.005f
float bidirectional_signal = ... * 10.0f; // Was 5.0f (2x amplification)
const float EXCLUSION_NOISE_FLOOR = 0.005f; // Was 0.01f
float exclusion_signal = ... * 5.0f;      // Was 2.0f (2.5x amplification)
float effective_tau = tau_mem * 1.5f;     // Was 0.5f (3x slower decay)

// Lines 945-990: Force and velocity clamping for stability
const float MAX_FORCE = 1000.0f;
const float MAX_VELOCITY = 100.0f;
```

## Build Instructions

```bash
# Build all crates
cargo build --release -p prism-nhs -p prism-gpu -p prism-validation

# Build with CUDA support
cargo build --release -p prism-gpu --features cuda

# Run NHS adaptive (main binary)
cargo run --release -p prism-nhs --bin nhs_adaptive -- \
    --topology input.json \
    --output results/ \
    --survey-steps 500000
```

## License

PRISM4D is proprietary research software.

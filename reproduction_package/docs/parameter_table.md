# PRISM4D Cryo-UV Parameter Reference

## Simulation Parameters

### Langevin Dynamics

| Parameter | Symbol | Value | Unit | Location |
|-----------|--------|-------|------|----------|
| Timestep | dt | 0.002 | ps | fused_engine.rs:989 |
| Base friction | γ_base | 10.0 | ps⁻¹ | fused_engine.rs:990 |
| Equilibration friction | γ_eq | 1000.0 | ps⁻¹ | fused_engine.rs:1079 |
| Equilibration steps | N_eq | 10000 | steps | fused_engine.rs:1078 |
| Cutoff | r_cut | 12.0 | Å | fused_engine.rs:991 |

### Temperature Protocol

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Cryo temperature | 100.0 | K | Starting temperature |
| Target temperature | 300.0 | K | Final temperature |
| Survey steps | 50000 | steps | Frozen baseline (100 ps) |
| Convergence steps | 25000 | steps | Temperature ramp |
| Precision steps | 25000 | steps | Warm validation |
| Cryo hold | 500000 | steps | Full survey at cryo (1 ns) |

### Force Clamping

| Parameter | Value | Unit | Location |
|-----------|-------|------|----------|
| Max force | 1000.0 | kcal/mol/Å | nhs_amber_fused.cu:947 |
| Max velocity | 100.0 | Å/ps | nhs_amber_fused.cu:969 |

## Spike Detection Parameters

### LIF Neuron

| Parameter | Symbol | Value | Unit | Location |
|-----------|--------|-------|------|----------|
| Threshold | V_th | 0.5 | - | nhs_amber_fused.cu:54 |
| Reset potential | V_reset | 0.0 | - | nhs_amber_fused.cu:55 |
| Membrane time constant | τ_mem | 10.0 | frames | default |
| Effective decay | τ_eff | 1.5 × τ_mem | frames | nhs_amber_fused.cu:511 |

### Signal Processing

| Parameter | Value | Description | Location |
|-----------|-------|-------------|----------|
| Noise floor | 0.002 | Density change threshold | nhs_amber_fused.cu:499 |
| Bidirectional gain | 10× | Signal amplification | nhs_amber_fused.cu:501 |
| Exclusion noise floor | 0.005 | Bulk deviation threshold | nhs_amber_fused.cu:505 |
| Exclusion gain | 5× | Exclusion signal weight | nhs_amber_fused.cu:507 |

### Water Inference

| Parameter | Symbol | Value | Unit | Location |
|-----------|--------|-------|------|----------|
| Bulk density | ρ_bulk | 0.0334 | mol/Å³ | nhs_amber_fused.cu:53 |
| Exclusion cutoff | r_excl | 8.0 | Å | nhs_amber_fused.cu:52 |
| Polar enhancement | - | 0.3 × polar_field | - | nhs_amber_fused.cu:468 |

## UV Probe Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Burst energy | 5.0 | kcal/mol | Energy deposited per burst |
| Burst interval | 1000 | steps | Steps between UV pulses |
| Burst duration | 10 | steps | Length of each burst |
| Target wavelength | 280 | nm | Aromatic absorption peak |

### Absorption Strengths

| Residue | Strength | Rationale |
|---------|----------|-----------|
| TRP | 1.0 | Strong π-π* transition |
| TYR | 0.5 | Moderate absorption |
| PHE | 0.2 | Weak absorption |

## Analysis Parameters

### RMSF Calculation

| Parameter | Value | Description |
|-----------|-------|-------------|
| Max frames | 75 | Stable frames to analyze |
| Alignment | Kabsch | Rigid-body superposition |
| Atoms | CA | Alpha carbons only |

### Correlation

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Correlation radius | 10.0 | Å | Max distance for correlation |
| RMSF threshold | mean + 1σ | Å | High flexibility cutoff |
| Min spike count | 3 | - | Hotspot threshold |

## Grid Parameters

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| Max grid dimension | 128 | voxels | GPU shared memory limit |
| Default spacing | 1.0-2.0 | Å | Voxel size |
| Padding | 5.0 | Å | Around protein |

## Performance Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| Steps/second | >1000 | 1690-2476 |
| ms/frame (NHS detect) | <2.0 | ~0.5 |
| Stable frames | >50 | 77 |

## File Locations

### Source Files

```
crates/prism-nhs/src/fused_engine.rs     # Rust engine
crates/prism-gpu/src/kernels/nhs_amber_fused.cu  # CUDA kernels
crates/prism-nhs/src/bin/nhs_adaptive.rs  # CLI binary
```

### Output Files

```
<output>/adaptive_results.json    # Spike data
<output>/*_ensemble.pdb           # Conformations
<output>/correlated_sites.json    # Final targets
```

## Tuning Guidelines

### If no spikes detected:
1. Lower `LIF_THRESHOLD` (try 0.3)
2. Lower `NOISE_FLOOR` (try 0.001)
3. Increase signal gain (try 15×)

### If too many spikes:
1. Raise `LIF_THRESHOLD` (try 0.8)
2. Raise `NOISE_FLOOR` (try 0.005)
3. Decrease signal gain

### If simulation explodes:
1. Increase `EQUILIBRATION_GAMMA` (try 2000)
2. Increase equilibration steps (try 20000)
3. Lower timestep (try 0.001 ps)

### If RMSF too high:
1. Check for coordinate drift (explosion)
2. Extract fewer frames (first 50)
3. Use stricter alignment

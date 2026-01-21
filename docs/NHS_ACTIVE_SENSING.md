# PRISM-NHS Active Sensing System

## Complete Technical Documentation

### Version: 1.0.0
### Module: prism-nhs/active_sensing

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
   - [Coherent UV Excitation](#1-coherent-uv-excitation)
   - [Neuromorphic Processing](#2-neuromorphic-processing)
   - [Spike Sequence Detection](#3-spike-sequence-detection)
   - [Lateral Inhibition](#4-lateral-inhibition)
   - [Resonance Detection](#5-resonance-detection)
   - [Adaptive Probe Control](#6-adaptive-probe-control)
4. [Data Structures](#data-structures)
5. [CUDA Kernels Reference](#cuda-kernels-reference)
6. [Rust API Reference](#rust-api-reference)
7. [Integration Guide](#integration-guide)
8. [Protocols](#protocols)
9. [Performance Considerations](#performance-considerations)
10. [Scientific Background](#scientific-background)
11. [Troubleshooting](#troubleshooting)

---

## Overview

The PRISM-NHS Active Sensing system implements a **closed-loop molecular stethoscope** for cryptic binding site detection. It combines:

| Component | Technique | Purpose |
|-----------|-----------|---------|
| **Coherent UV Excitation** | Vibrational interferometry | Spatially-targeted energy injection |
| **Neuromorphic Processing** | LIF neurons + lateral inhibition | Response pattern detection |
| **Spike Sequence Detection** | STDP-like temporal correlation | Causal pathway identification |
| **Resonance Detection** | Frequency sweeping | Soft mode characterization |
| **Adaptive Control** | Reinforcement learning | Optimal probe selection |

### Why Active Sensing?

Traditional approaches sample conformational space passively. Active sensing **probes** the protein structure:

```
Passive Sampling:          Active Sensing:
─────────────────         ─────────────────
• Wait for rare events    • Trigger responses
• High computational cost • Low-energy probes
• Random coverage         • Directed exploration
• Signal in noise         • Signal enhancement
```

**Key Insight**: Cryptic sites are **metastable** - small perturbations trigger large conformational changes. UV excitation of aromatics provides spatially-targeted energy injection, and neuromorphic readout detects the response patterns.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      PRISM-NHS ACTIVE SENSING PIPELINE                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌────────────────┐                    ┌────────────────┐               │
│  │   Aromatic     │                    │    Protein     │               │
│  │   Topology     │                    │   Coordinates  │               │
│  └───────┬────────┘                    └───────┬────────┘               │
│          │                                     │                         │
│          ▼                                     ▼                         │
│  ┌────────────────┐    ┌────────────┐  ┌────────────────┐              │
│  │   Aromatic     │───▶│  Coherent  │──▶│   Velocity    │              │
│  │   Clustering   │    │   Probe    │  │    Kicks      │              │
│  └────────────────┘    └────────────┘  └───────┬────────┘              │
│          │                   │                  │                        │
│          │                   ▼                  ▼                        │
│          │           ┌────────────────────────────────┐                 │
│          │           │       MD SIMULATION STEP       │                 │
│          │           │  (NHS-AMBER Fused Kernel)      │                 │
│          │           └───────────────┬────────────────┘                 │
│          │                           │                                   │
│          │                           ▼                                   │
│          │           ┌────────────────────────────────┐                 │
│          │           │     EXCLUSION FIELD UPDATE     │                 │
│          │           │  (Water density computation)   │                 │
│          │           └───────────────┬────────────────┘                 │
│          │                           │                                   │
│          │                           ▼                                   │
│          │           ┌────────────────────────────────┐                 │
│          │           │   LIF NEURONS + INHIBITION     │                 │
│          │           │  (Dewetting → spike events)    │                 │
│          │           └───────────────┬────────────────┘                 │
│          │                           │                                   │
│          │              ┌────────────┼────────────┐                     │
│          │              │            │            │                      │
│          │              ▼            ▼            ▼                      │
│          │     ┌──────────────┐ ┌─────────┐ ┌─────────────┐            │
│          │     │   Sequence   │ │  Spike  │ │  Resonance  │            │
│          │     │  Detection   │ │ History │ │  Spectrum   │            │
│          │     └──────┬───────┘ └────┬────┘ └──────┬──────┘            │
│          │            │              │             │                     │
│          │            └──────────────┼─────────────┘                    │
│          │                           │                                   │
│          │                           ▼                                   │
│          │           ┌────────────────────────────────┐                 │
│          │           │      RESPONSE AGGREGATION      │                 │
│          │           │  (ProbeResponse computation)   │                 │
│          │           └───────────────┬────────────────┘                 │
│          │                           │                                   │
│          │                           ▼                                   │
│          │           ┌────────────────────────────────┐                 │
│          │           │    ADAPTIVE PROBE SELECTION    │                 │
│          │           │   (RL-based next probe pick)   │                 │
│          │           └───────────────┬────────────────┘                 │
│          │                           │                                   │
│          └───────────────────────────┘                                   │
│                     (Feedback loop)                                      │
│                                                                          │
│  OUTPUT: CrypticSiteCandidates with confidence scores                   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Coherent UV Excitation

**Physics Principle**: UV photons (280nm) excite aromatic residues (TRP, TYR, PHE), depositing ~5 kcal/mol of vibrational energy perpendicular to the ring plane.

**Innovation - Vibrational Interferometry**: By exciting multiple aromatics with controlled phase delays, we create vibrational wave interference patterns.

```
         Phase Delay = 0 fs            Phase Delay = 500 fs
         ─────────────────            ─────────────────────
              ↓ Excitation                    ↓
         ┌────────────────┐           ┌────────────────┐
         │  TRP  ←  TYR   │           │  TRP     TYR   │
         │   ↑      ↓     │           │   ↑        ↓   │
         │   │      │     │           │   │        │   │
         │ CONSTRUCTIVE   │           │ DESTRUCTIVE    │
         │ INTERFERENCE   │           │ INTERFERENCE   │
         └────────────────┘           └────────────────┘
              Strong                      Weak
              Response                    Response
```

**Why It Works**: Cryptic sites open along specific mechanical pathways. Exciting aromatics on the "right" side of a hinge causes vibrational energy to propagate along the opening pathway → correlated dewetting. Wrong side → no correlation.

#### Data Structures

```cuda
// Up to 8 aromatic groups for coherent excitation
struct AromaticGroup {
    int aromatic_indices[32];  // Indices into UV target array
    int n_aromatics;           // Number in group
    float3 centroid;           // Geometric center
    float total_absorption;    // Sum of absorption strengths
};

// Complete probe specification
struct CoherentProbe {
    AromaticGroup groups[8];      // Aromatic groups
    int n_groups;                  // Active groups
    float phase_delays_fs[8];      // Timing (femtoseconds)
    float energy_per_group[8];     // Energy allocation
    float total_energy;            // Total (kcal/mol)
    int probe_id;                  // Unique ID
    int probe_type;                // 0=hinge_A, 1=hinge_B, etc.
    float expected_response;       // For learning
};
```

#### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `MAX_PROBE_GROUPS` | 8 | Maximum aromatic groups per probe |
| `MAX_AROMATICS_PER_GROUP` | 32 | Maximum aromatics per group |
| `MAX_PHASE_DELAY_FS` | 1000 fs | Maximum phase delay |
| Default energy | ~5 kcal/mol | Enough to perturb, not disrupt |

---

### 2. Neuromorphic Processing

**Principle**: Leaky Integrate-and-Fire (LIF) neurons convert continuous dewetting signals into discrete spike events, enabling temporal pattern detection.

```
              Dewetting Signal
                    │
                    ▼
    ┌─────────────────────────────┐
    │    LEAKY INTEGRATION        │
    │                             │
    │  V(t+dt) = V(t)·e^(-dt/τ)   │
    │           + I_input         │
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │    THRESHOLD CHECK          │
    │                             │
    │  if V > V_thresh:           │
    │      emit_spike()           │
    │      V = 0  (reset)         │
    └──────────────┬──────────────┘
                   │
                   ▼
    ┌─────────────────────────────┐
    │    LATERAL INHIBITION       │
    │                             │
    │  suppress_neighbors()       │
    └─────────────────────────────┘
```

#### LIF Parameters

| Parameter | Typical Value | Effect |
|-----------|---------------|--------|
| `tau_membrane` | 1.0 ps | Membrane time constant |
| `threshold` | 0.15 | Spike threshold |
| `refractory` | 0.1 ps | Minimum inter-spike interval |

---

### 3. Spike Sequence Detection

**STDP-Inspired Algorithm**: Spike-Timing Dependent Plasticity from neuroscience adapted for spatial detection.

**Core Idea**: True cryptic site opening is **sequential** - water leaves voxels in a specific order as the pocket opens. Random thermal fluctuations produce unordered spikes.

```
     TRUE OPENING EVENT              RANDOM FLUCTUATION
     ──────────────────             ───────────────────

     Voxel A ─●─────────────        Voxel A ─●───●─────
     Voxel B ────●──────────        Voxel B ──●───────●
     Voxel C ───────●───────        Voxel C ●───────●──

     Time →                         Time →

     ORDERED SEQUENCE               NO TEMPORAL STRUCTURE
     (Causal relationship)          (Independent events)
```

#### Sequence Detection Algorithm

```cuda
struct SpikeSequenceDetector {
    int voxel_sequence[8];           // Expected voxel order
    int sequence_length;              // Length
    float max_inter_spike_interval;   // Causality window (10 ps)

    // State
    int current_position;             // Progress through sequence
    float accumulated_score;          // Evidence accumulator
    int detection_count;              // Times fully detected
    float weight;                     // Importance (learned)
};
```

**Scoring Function**:
```
score = Π (timing_score_i) × completeness²

where:
  timing_score_i = exp(-Δt_i / (τ/2))
  completeness = matched_spikes / sequence_length
```

---

### 4. Lateral Inhibition

**Biological Inspiration**: Retinal ganglion cells - when a cell fires, it inhibits neighbors, enhancing contrast at edges.

```
         BEFORE INHIBITION          AFTER INHIBITION
         ─────────────────         ─────────────────

         0.5  0.6  0.5             0.3  0.6  0.3
         0.6  0.9  0.6     →       0.4  0.9  0.4
         0.5  0.6  0.5             0.3  0.6  0.3

         Diffuse signal            Sharp boundary
```

**Why It Works**: Sharpens dewetting boundaries to identify pocket edges clearly. Reduces false positives from diffuse thermal fluctuations.

#### Implementation

```cuda
// When voxel spikes, suppress neighbors
__device__ void apply_lateral_inhibition(
    float* lif_potentials,
    LateralInhibitionState* states,
    int spiked_voxel,
    int grid_dim,
    float current_time_ps
) {
    // Inhibit within INHIBITION_RADIUS (2 voxels)
    // Strength = INHIBITION_STRENGTH (0.3) / distance
    for (dz = -2; dz <= 2; dz++) {
        for (dy = -2; dy <= 2; dy++) {
            for (dx = -2; dx <= 2; dx++) {
                if (dx==0 && dy==0 && dz==0) continue;

                float dist = sqrt(dx² + dy² + dz²);
                float inhibition = 0.3 / dist;

                neighbor_potential -= neighbor_potential * inhibition;
            }
        }
    }
}
```

#### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `INHIBITION_RADIUS` | 2 voxels | Range of inhibition |
| `INHIBITION_STRENGTH` | 0.3 | Maximum suppression |
| `INHIBITION_DECAY_TAU` | 0.5 ps | Decay time constant |

---

### 5. Resonance Detection

**Physics**: Cryptic sites near opening transition have **soft modes** - low-frequency collective motions that are almost unstable. If probe frequency matches a soft mode, resonant amplification occurs.

```
                    RESONANCE SPECTRUM

     Response │
     Amplitude│          ╭─╮
              │         ╱   ╲    ← Soft mode resonance
              │        ╱     ╲      (Q > 2.0)
              │ ──────╱       ╲──────────
              │
              └──────────────────────────────
                    Probe Frequency (THz)

              0.1        0.5        1.0
```

#### Frequency Sweep Protocol

```
Step 1: Set frequency range (0.1 - 10 THz)
Step 2: For each frequency f:
    - Apply periodic UV bursts at frequency f
    - Measure total spike response amplitude
    - Record phase relationship
Step 3: Build spectrum: frequency_spectrum[f] = amplitude
Step 4: Find peaks with Q > 2.0
Step 5: Low-f resonances → most likely cryptic sites
```

#### Data Structure

```cuda
struct ResonanceDetector {
    float probe_frequency_thz;        // Current frequency
    float response_amplitude;         // Current response
    float response_phase;             // Phase shift

    // Accumulated spectrum (100 bins)
    float frequency_spectrum[100];    // Response vs frequency
    float phase_spectrum[100];        // Phase vs frequency
    int sample_counts[100];           // Samples per bin

    // Detected resonances
    float resonance_frequencies[8];   // Peak frequencies
    float resonance_amplitudes[8];    // Peak amplitudes
    float quality_factors[8];         // Q = f0 / FWHM
    int n_resonances;
};
```

---

### 6. Adaptive Probe Control

**Reinforcement Learning**: Use reward-based learning to find optimal probing strategy.

#### Reward Function

```
reward = 0.1 × spike_count           (activity)
       + 5.0 × sequence_score        (MOST IMPORTANT - indicates real opening)
       + 2.0 × mean_intensity        (signal strength)
       - 0.5 × spatial_extent        (penalize diffuse responses)
       + 2.0 × (onset < 1 ps)        (bonus for fast onset)
```

#### Probe Selection

```cuda
int select_next_probe(controller, rng, n_probes) {
    // ε-greedy exploration
    if (random() < EXPLORATION_EPSILON) {
        return random_probe();
    }

    // Softmax selection based on scores
    float exp_scores[n_probes];
    for (i = 0; i < n_probes; i++) {
        exp_scores[i] = exp(scores[i] / temperature);
    }

    // Weighted random selection
    return weighted_sample(exp_scores);
}
```

#### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `ADAPTATION_RATE` | 0.1 | Learning rate |
| `EXPLORATION_EPSILON` | 0.1 | Random exploration probability |
| Initial temperature | 1.0 | Softmax temperature |
| Temperature decay | 0.999 | Per-trial decay |
| Minimum temperature | 0.1 | Lower bound |

---

## Data Structures

### Complete Hierarchy

```
ActiveSensingEngine
├── GPU Buffers
│   ├── d_spike_histories      [n_voxels × VoxelSpikeHistory]
│   ├── d_inhibition_states    [n_voxels × LateralInhibitionState]
│   ├── d_sequence_detectors   [n_detectors × SpikeSequenceDetector]
│   ├── d_resonance_detector   [1 × ResonanceDetector]
│   ├── d_probe_controller     [1 × AdaptiveProbeController]
│   ├── d_probes               [n_probes × CoherentProbe]
│   └── d_aromatic_groups      [n_groups × AromaticGroup]
│
├── Host State
│   ├── current_probe_idx      Current probe being used
│   ├── current_time_ps        Simulation time
│   ├── probe_start_time       When current probe started
│   └── mode                   Single/Differential/Sweep
│
└── Results
    ├── cryptic_candidates     Detected cryptic sites
    ├── differential_scores    Hinge asymmetry scores
    └── resonance_peaks        Identified soft modes
```

### Memory Layout

| Structure | Size (bytes) | Per-Element |
|-----------|--------------|-------------|
| `VoxelSpikeHistory` | ~1,296 | 64 spikes × 20 bytes |
| `LateralInhibitionState` | 136 | Fixed |
| `SpikeSequenceDetector` | 64 | Fixed |
| `ResonanceDetector` | ~1,624 | Fixed (1 per system) |
| `AdaptiveProbeController` | ~520 | Fixed (1 per system) |
| `CoherentProbe` | ~800 | Fixed |
| `AromaticGroup` | ~144 | Fixed |

---

## CUDA Kernels Reference

### Core Kernels

| Kernel | Grid Size | Purpose |
|--------|-----------|---------|
| `apply_coherent_probe` | n_groups | Execute phase-locked UV excitation |
| `lif_update_with_inhibition` | n_voxels | LIF update + spike detection |
| `detect_spike_sequences` | n_detectors | STDP-like sequence detection |
| `update_resonance_spectrum` | 1 | Accumulate frequency response |
| `analyze_resonances` | 1 | Find spectrum peaks |
| `adaptive_probe_update` | 1 | RL probe selection |
| `compute_response_metrics` | 1 | Aggregate probe response |

### Utility Kernels

| Kernel | Purpose |
|--------|---------|
| `init_active_sensing` | Initialize all buffers |
| `cluster_aromatics` | K-means grouping by position |
| `build_sequence_detectors` | Auto-generate from aromatic pairs |
| `compare_differential_probes` | A vs B comparison |

### Kernel Launch Parameters

```cuda
// Coherent probe: one thread per group
apply_coherent_probe<<<1, probe->n_groups>>>(...)

// LIF update: one thread per voxel
int threads = 256;
int blocks = (total_voxels + threads - 1) / threads;
lif_update_with_inhibition<<<blocks, threads>>>(...)

// Sequence detection: one thread per detector
detect_spike_sequences<<<(n_detectors+255)/256, 256>>>(...)

// Single-thread kernels
update_resonance_spectrum<<<1, 1>>>(...)
adaptive_probe_update<<<1, 1>>>(...)
```

---

## Rust API Reference

### ActiveSensingEngine

```rust
pub struct ActiveSensingEngine {
    // Configuration
    config: ActiveSensingConfig,

    // GPU buffers
    d_spike_histories: CudaSlice<VoxelSpikeHistory>,
    d_inhibition_states: CudaSlice<LateralInhibitionState>,
    d_sequence_detectors: CudaSlice<SpikeSequenceDetector>,
    d_detection_scores: CudaSlice<f32>,
    d_resonance_detector: CudaSlice<ResonanceDetector>,
    d_probe_controller: CudaSlice<AdaptiveProbeController>,
    d_probes: CudaSlice<CoherentProbe>,
    d_current_response: CudaSlice<ProbeResponse>,

    // State
    current_probe_idx: usize,
    current_time_ps: f32,
    n_detectors: usize,
    mode: ActiveSensingMode,
}
```

### Configuration

```rust
pub struct ActiveSensingConfig {
    pub grid_dim: i32,                    // Voxel grid dimension
    pub grid_spacing: f32,                // Voxel size (Å)
    pub tau_membrane: f32,                // LIF time constant (ps)
    pub spike_threshold: f32,             // Spike threshold
    pub probe_interval_steps: usize,      // Steps between probe switches
    pub analysis_window_ps: f32,          // Response analysis window
    pub max_sequence_distance: f32,       // For auto-detector creation
    pub target_n_groups: usize,           // Aromatic clustering target
    pub min_group_separation: f32,        // Minimum cluster separation
}
```

### Core Methods

```rust
impl ActiveSensingEngine {
    /// Create new engine from aromatic topology
    pub fn new(
        ctx: &CudaContext,
        config: ActiveSensingConfig,
        aromatic_centroids: &[Float3],
        aromatic_absorptions: &[f32],
        ring_normals: &[Float3],
    ) -> Result<Self>;

    /// Apply current probe (call before MD step)
    pub fn apply_probe(
        &mut self,
        velocities: &mut CudaSlice<Float3>,
        positions: &CudaSlice<Float3>,
        masses: &CudaSlice<f32>,
        current_time_fs: f32,
        dt_fs: f32,
    ) -> Result<()>;

    /// Update LIF neurons (call after MD step)
    pub fn lif_update(
        &mut self,
        water_density: &CudaSlice<f32>,
        water_density_prev: &CudaSlice<f32>,
        dt_ps: f32,
    ) -> Result<i32>;  // Returns spike count

    /// Detect spike sequences
    pub fn detect_sequences(&mut self) -> Result<Vec<f32>>;

    /// Compute and return response metrics
    pub fn compute_response(&mut self) -> Result<ProbeResponse>;

    /// Update probe selection based on response
    pub fn update_probe_selection(
        &mut self,
        response: &ProbeResponse,
    ) -> Result<usize>;  // Returns next probe index

    /// Get current results
    pub fn get_results(&self) -> Result<ActiveSensingResults>;
}
```

### Result Structures

```rust
pub struct ActiveSensingResults {
    pub cryptic_site_candidates: Vec<CrypticSiteCandidate>,
    pub differential_scores: Vec<DifferentialScore>,
    pub resonance_peaks: Vec<ResonancePeak>,
    pub probe_statistics: ProbeStatistics,
}

pub struct CrypticSiteCandidate {
    pub position: [f32; 3],
    pub confidence: f32,
    pub sequence_score: f32,
    pub resonance_frequency: Option<f32>,
    pub differential_score: Option<f32>,
}
```

---

## Integration Guide

### With NHS-AMBER Fused Engine

```rust
// In main simulation loop
let mut active_sensing = ActiveSensingEngine::new(
    &ctx,
    config,
    &aromatic_centroids,
    &aromatic_absorptions,
    &ring_normals,
)?;

let mut current_probe = active_sensing.get_current_probe();

for step in 0..n_steps {
    // 1. Apply coherent UV probe (modifies velocities)
    active_sensing.apply_probe(
        &mut fused_engine.d_velocities,
        &fused_engine.d_positions,
        &fused_engine.d_masses,
        step as f32 * dt_fs,
        dt_fs,
    )?;

    // 2. Run MD step (updates positions, exclusion field)
    fused_engine.step()?;

    // 3. Enhanced LIF update with lateral inhibition
    let spike_count = active_sensing.lif_update(
        &fused_engine.d_water_density,
        &fused_engine.d_water_density_prev,
        dt_ps,
    )?;

    // 4. Detect spike sequences
    active_sensing.detect_sequences()?;

    // 5. Every N steps: analyze response and adapt
    if step % probe_interval == 0 {
        let response = active_sensing.compute_response()?;
        let next_idx = active_sensing.update_probe_selection(&response)?;

        // Optional: Update resonance spectrum
        active_sensing.update_resonance(probe_frequency)?;
    }
}

// Get final results
let results = active_sensing.get_results()?;
for candidate in &results.cryptic_site_candidates {
    println!("Cryptic site at {:?}, confidence: {:.2}",
        candidate.position, candidate.confidence);
}
```

### Standalone Mode

```rust
// Use active sensing without full MD
let engine = ActiveSensingEngine::new_standalone(
    &ctx,
    config,
    pdb_path,
)?;

// Run automatic detection protocol
let results = engine.run_detection_protocol(
    n_probes: 100,
    probes_per_frequency: 10,
)?;
```

---

## Protocols

### Protocol 1: Single Probe Detection

**Use Case**: Quick scan for responsive regions

```rust
// Apply single probe, measure response
for trial in 0..n_trials {
    engine.apply_probe(&probe)?;
    engine.run_md_steps(100)?;
    responses.push(engine.compute_response()?);
}

// Average response indicates site activity
let avg_response = responses.iter().map(|r| r.sequence_score).mean();
```

### Protocol 2: Differential Probing

**Use Case**: Identify mechanical asymmetry (hinge regions)

```
Step 1: Identify suspected hinge region
Step 2: Create probe pairs
    - Probe A: Excite aromatics on side 1
    - Probe B: Excite aromatics on side 2
Step 3: Alternate probes
    - Run probe A → record response
    - Wait for relaxation
    - Run probe B → record response
Step 4: Compute differential score
    - diff = |A - B| / (A + B)
    - High diff (>0.5) = asymmetric pathway
Step 5: Repeat for statistical confidence
    - n_trials > 20 recommended
```

```rust
let pair = DifferentialProbePair::new(probe_a, probe_b);

for trial in 0..50 {
    // Probe A
    engine.set_probe(&pair.probe_a)?;
    engine.run_md_steps(200)?;
    let response_a = engine.compute_response()?;

    // Relaxation
    engine.run_md_steps(100)?;

    // Probe B
    engine.set_probe(&pair.probe_b)?;
    engine.run_md_steps(200)?;
    let response_b = engine.compute_response()?;

    engine.update_differential(&mut pair, &response_a, &response_b)?;
}

println!("Differential score: {:.3}, confidence: {:.3}",
    pair.differential_score, pair.confidence);
```

### Protocol 3: Resonance Sweep

**Use Case**: Identify soft modes and their frequencies

```
Step 1: Set frequency range (0.1 - 10 THz)
Step 2: For each frequency f:
    - Apply periodic UV bursts at frequency f
    - Measure total spike response amplitude
    - Record phase relationship
Step 3: Build spectrum
Step 4: Find peaks with Q > 2.0
Step 5: Interpret:
    - Lower f = larger collective motion
    - Sites with low-f resonances most likely cryptic
```

```rust
// Logarithmic frequency sweep
let frequencies: Vec<f32> = (0..100)
    .map(|i| 0.1 * 10.0_f32.powf(i as f32 / 50.0))
    .collect();

for freq in frequencies {
    engine.set_probe_frequency(freq)?;

    for cycle in 0..10 {
        engine.apply_probe()?;
        engine.run_md_steps(100)?;
    }

    engine.update_resonance_spectrum(freq)?;
}

// Analyze spectrum
engine.analyze_resonances()?;
let peaks = engine.get_resonance_peaks()?;

for peak in peaks {
    println!("Resonance at {:.2} THz, Q = {:.1}",
        peak.frequency, peak.quality_factor);
}
```

### Protocol 4: Adaptive Exploration

**Use Case**: Automatic optimal probe discovery

```rust
// Let RL find the best probing strategy
for iteration in 0..1000 {
    // Apply current best probe
    let probe_idx = engine.get_current_probe_idx();
    engine.apply_probe()?;
    engine.run_md_steps(200)?;

    // Compute response and update
    let response = engine.compute_response()?;
    let next_idx = engine.update_probe_selection(&response)?;

    // Log progress
    if iteration % 100 == 0 {
        println!("Iteration {}: best probe {}, reward {:.2}",
            iteration,
            engine.get_best_probe_idx(),
            engine.get_cumulative_reward());
    }
}

// Get final optimal probes
let optimal_probes = engine.get_top_probes(5)?;
```

---

## Performance Considerations

### Memory Requirements

For a 64³ grid with 20 aromatics:

| Buffer | Size |
|--------|------|
| Spike histories | 64 × 262,144 × 20 bytes = **336 MB** |
| Inhibition states | 262,144 × 136 bytes = **36 MB** |
| Sequence detectors | 1,024 × 64 bytes = **64 KB** |
| Resonance detector | **4 KB** |
| Probe controller | **2 KB** |
| **Total** | **~370 MB** |

For 32³ grid: **~46 MB**

### Computational Cost

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| Coherent probe | O(n_groups × n_atoms_per_group) | ~0.1 ms |
| LIF update | O(n_voxels) | ~0.5 ms |
| Lateral inhibition | O(n_spikes × 125) | ~0.2 ms |
| Sequence detection | O(n_detectors × history_length) | ~0.3 ms |
| Resonance update | O(n_voxels × history_length) | ~1.0 ms |
| Adaptive update | O(n_probes) | ~0.01 ms |
| **Total per step** | | **~2 ms** |

### Optimization Tips

1. **Grid Resolution**: Start with 32³, increase only if needed
2. **History Length**: 64 is sufficient; reduce to 32 for memory savings
3. **Detector Count**: Limit to ~1000; auto-builder may create more
4. **Probe Interval**: Every 50-100 MD steps is typically sufficient
5. **Batch Probing**: Apply multiple probes in same kernel launch

---

## Scientific Background

### UV Absorption by Aromatic Residues

| Residue | λ_max (nm) | ε (M⁻¹cm⁻¹) | Relative Absorption |
|---------|------------|-------------|---------------------|
| Tryptophan (TRP) | 280 | 5,600 | 1.00 |
| Tyrosine (TYR) | 274 | 1,400 | 0.25 |
| Phenylalanine (PHE) | 257 | 200 | 0.04 |

Energy per photon at 280nm: ~100 kJ/mol (24 kcal/mol)

In simulation, we use ~5 kcal/mol bursts - enough to perturb without disrupting structure.

### Protein Vibrational Modes

| Frequency Range | Motion Type | Timescale |
|-----------------|-------------|-----------|
| >10 THz | Bond stretching | <100 fs |
| 1-10 THz | Side chain rotations | 100 fs - 1 ps |
| 0.1-1 THz | Loop motions | 1-10 ps |
| <0.1 THz | Domain motions | >10 ps |

**Cryptic sites** typically involve **domain motions** (low frequency), so resonances at 0.1-0.5 THz are most relevant.

### STDP in Neuroscience

Spike-Timing Dependent Plasticity strengthens synapses when pre-synaptic spike precedes post-synaptic spike by <20ms.

**Our Adaptation**:
- Voxel A spike precedes Voxel B spike by <10ps
- Repeated A→B sequences strengthen detector weight
- Indicates **causal (mechanical) relationship**

### Lateral Inhibition in Vision

Hartline's discovery (1949): Illuminating one photoreceptor in Limulus (horseshoe crab) inhibits neighbors, enhancing edge contrast.

**Our Adaptation**:
- Spiking voxel inhibits spatial neighbors
- Enhances boundaries between wet/dry regions
- Reduces false positives from thermal noise

---

## Troubleshooting

### Common Issues

#### 1. No Spikes Detected
```
Possible causes:
- Threshold too high (try 0.1 instead of 0.15)
- tau_membrane too short (try 2.0 ps)
- Insufficient probe energy (try 7 kcal/mol)
- Grid too coarse (try 2.0 Å spacing)
```

#### 2. Too Many Spikes (Noisy)
```
Possible causes:
- Threshold too low (try 0.2)
- tau_membrane too long (try 0.5 ps)
- Inhibition strength too low (try 0.5)
- Temperature too high in MD
```

#### 3. No Sequences Detected
```
Possible causes:
- Causality window too short (try 15 ps)
- Sequence length too long (try 4 instead of 8)
- Detectors not built for relevant regions
- Try manual detector placement
```

#### 4. Flat Resonance Spectrum
```
Possible causes:
- Frequency range wrong (try 0.05-5 THz)
- Insufficient samples per frequency (try 20)
- Probe energy too low
- No soft modes present (stable structure)
```

#### 5. Poor Adaptive Learning
```
Possible causes:
- Exploration temperature too low
- Not enough trials (need >500)
- Reward function not appropriate
- All probes equally effective (good problem!)
```

### Diagnostic Commands

```rust
// Print spike statistics
let stats = engine.get_spike_statistics()?;
println!("Total spikes: {}", stats.total);
println!("Spikes per voxel (mean): {:.2}", stats.mean_per_voxel);
println!("Active voxels: {}", stats.active_voxels);

// Check detector states
let detector_states = engine.get_detector_states()?;
for (i, det) in detector_states.iter().enumerate() {
    if det.detection_count > 0 {
        println!("Detector {}: {} detections, score {:.3}",
            i, det.detection_count, det.accumulated_score);
    }
}

// Verify probe configurations
let probes = engine.get_all_probes()?;
for probe in probes {
    println!("Probe {}: {} groups, {:.1} kcal/mol",
        probe.probe_id, probe.n_groups, probe.total_energy);
}
```

---

## References

1. Zewail, A.H. (2000). Femtochemistry: Atomic-Scale Dynamics of the Chemical Bond. *J. Phys. Chem. A*. 104(24):5660-5694.

2. Bi, G. & Poo, M. (1998). Synaptic Modifications in Cultured Hippocampal Neurons: Dependence on Spike Timing, Synaptic Strength, and Postsynaptic Cell Type. *J. Neurosci.* 18(24):10464-10472.

3. Bowman, G.R. et al. (2012). Discovery of Multiple Hidden Allosteric Sites by Combining Markov State Models and Experiments. *PNAS*. 109(29):11681-11686.

4. Cimermancic, P. et al. (2016). CryptoSite: Expanding the Druggable Proteome by Characterization and Prediction of Cryptic Binding Sites. *J. Mol. Biol.* 428(4):709-719.

5. Hartline, H.K. (1949). Inhibition of Activity of Visual Receptors by Illuminating Nearby Retinal Areas in the Limulus Eye. *Fed. Proc.* 8(1):69.

6. Dill, K.A. & MacCallum, J.L. (2012). The Protein-Folding Problem, 50 Years On. *Science*. 338(6110):1042-1046.

---

## Changelog

### v1.0.0 (2026-01-20)
- Initial release
- Complete CUDA kernel implementations
- Rust bindings with cudarc
- All five core components functional
- Comprehensive documentation

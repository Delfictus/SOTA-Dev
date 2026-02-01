# RT Core Integration: Complete Pipeline Revision
**Version:** 1.0.0
**Date:** 2026-02-01
**Status:** Implementation Plan

---

## Overview

Integrate RTX 5080's 84 RT cores into the full PRISM4D pipeline for real-time spatial sensing during MD simulation. This enhancement runs **concurrently** with existing trajectory capture and spike detection.

**Key Principle:** RT probes are an **additional signal channel**, not a replacement for existing functionality.

---

## Modified Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: PRISM PREP/INPUT [MODIFIED]                            │
│ Tags: [STAGE-1-RT], [STAGE-1-EXPLICIT]                         │
├─────────────────────────────────────────────────────────────────┤
│ NEW Responsibilities:                                           │
│ • [NEW] Parse solvent mode: implicit / explicit / hybrid        │
│ • [NEW] If explicit: solvate protein (add water box)            │
│ • [NEW] Setup RT probe targets:                                 │
│   - Protein heavy atoms (always)                                │
│   - Water oxygens (if explicit)                                 │
│   - Aromatic ring centers (for LIF)                             │
│ • [NEW] Initialize OptiX context and create initial BVH         │
│ • [EXISTING] All previous prep functionality                    │
│                                                                 │
│ NEW Config Options:                                             │
│   solvent_mode: "implicit" | "explicit" | "hybrid"             │
│   explicit_water_padding: 10.0  # Å                             │
│   hybrid_threshold: 0.5          # RMSD drift to trigger switch │
│   rt_probe_interval: 100         # steps between probe bursts   │
│   rt_attention_points: 50        # number of probe origins      │
│                                                                 │
│ NEW Outputs:                                                    │
│   PreparedSystem {                                              │
│     solvent_mode: SolventMode,                                  │
│     water_atoms: Option<Vec<usize>>,  // if explicit            │
│     rt_targets: RtTargets,            // protein + water + arom │
│     optix_ctx: OptixContext,                                    │
│   }                                                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2a: CONFORMATIONAL DYNAMICS [ENHANCED]                    │
│ Tags: [STAGE-2A-RT], [STAGE-2A-RT-ASYNC]                       │
├─────────────────────────────────────────────────────────────────┤
│ THREE CONCURRENT PROCESSES:                                     │
│                                                                 │
│ Process 1: MD Simulation (CUDA Cores)                           │
│   • Run AMBER MD (implicit OR explicit solvent)                 │
│   • Langevin thermostat + UV excitation                         │
│   • Generate spike events                                       │
│   • Capture trajectory frames in GPU memory                     │
│   └─> Output: trajectory_gpu, spike_events.jsonl (raw)          │
│                                                                 │
│ Process 2: RT Spatial Probes (RT Cores) [NEW]                   │
│   • Async on separate CUDA stream                               │
│   • Fire probe rays every N steps (configurable)                │
│   • Two probe types:                                            │
│     A. Geometric probes (BVH-protein): void detection           │
│     B. LIF probes (aromatics + water): excitation sensing       │
│   • Refit BVH when displacement > threshold                     │
│   └─> Output: rt_probe_data.jsonl                               │
│                                                                 │
│ Process 3: Trajectory Buffering (CUDA Cores)                    │
│   • Every 5000 steps: mark frame for extraction                 │
│   • Keep frames in GPU memory during simulation                 │
│   • Batch download after simulation completes                   │
│   └─> Output: trajectory_frames (GPU buffer)                    │
│                                                                 │
│ COORDINATION:                                                   │
│   • All three processes share same atomic positions             │
│   • RT probes read positions without blocking MD                │
│   • Trajectory markers are timestamped, not downloaded yet      │
│   • Only ONE sync at end of batch (10K steps)                   │
│                                                                 │
│ Outputs:                                                        │
│   1. spike_events.jsonl         # Raw spike events              │
│   2. rt_probe_data.jsonl        # RT hit distances, LIF signals │
│   3. trajectory_frames_gpu      # Frames still in GPU memory    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2b: TRAJECTORY EXTRACTION/GENERATION [ENHANCED]           │
│ Tags: [STAGE-2B-RT], [STAGE-2B-TRAJ], [STAGE-2B-LIF]           │
├─────────────────────────────────────────────────────────────────┤
│ Process THREE data streams from Stage 2a:                       │
│                                                                 │
│ Stream 1: Trajectory Frames                                     │
│   • Download trajectory_frames_gpu → host memory                │
│   • Write to trajectory.pdb (multi-model format)                │
│   • Write to frames.json (with metadata)                        │
│   • Compute RMSF convergence (Cα first-half vs second-half)     │
│   • Cluster to representatives (50-200 frames, Boltzmann)       │
│   └─> Output: trajectory.pdb, frames.json, rmsf_analysis.json   │
│                                                                 │
│ Stream 2: RT Probe Data [NEW]                                   │
│   • Parse rt_probe_data.jsonl                                   │
│   • Time-series analysis of hit distances                       │
│   • Detect void formation events (distance increasing)          │
│   • Detect solvation disruption (variance increasing)           │
│   • Detect aromatic LIF events (excitation spatial pattern)     │
│   • Correlate with trajectory frames (timestep matching)        │
│   └─> Output: rt_signals.json, solvation_analysis.json          │
│                                                                 │
│ Stream 3: Spike Events                                          │
│   • Parse spike_events.jsonl (raw)                              │
│   • Filter by quality scores                                    │
│   • Correlate with RT probe signals:                            │
│     - Spikes near RT void formation → HIGH confidence           │
│     - Spikes with solvation disruption → LEADING indicator      │
│     - Spikes with aromatic LIF → UV-validated                   │
│   • Add trajectory context (which frame, RMSD from ref)         │
│   └─> Output: processed_spikes.jsonl                            │
│                                                                 │
│ SIGNAL FUSION:                                                  │
│   Create unified timeline correlating:                          │
│   - Solvation disruption (EARLIEST signal from RT)              │
│   - Geometric voids (INTERMEDIATE signal from RT)               │
│   - Spike events (CONFIRMATION from MD)                         │
│   - Trajectory snapshots (STRUCTURAL context)                   │
│                                                                 │
│ Outputs:                                                        │
│   1. trajectory.pdb              # Multi-model ensemble         │
│   2. frames.json                 # Frame metadata               │
│   3. rmsf_analysis.json          # Convergence metrics          │
│   4. cluster_reps.pdb            # Representative structures    │
│   5. processed_spikes.jsonl      # Filtered, scored, correlated │
│   6. rt_signals.json             # RT probe timeline            │
│   7. solvation_analysis.json     # Water disruption events      │
│   8. aromatic_lif_events.json    # UV-aromatic spatial signals  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: CRYPTIC BINDING SITE DETECTION [ENHANCED]              │
│ Tags: [STAGE-3-RT]                                              │
├─────────────────────────────────────────────────────────────────┤
│ Use FOUR signal channels (was two):                             │
│                                                                 │
│ Channel 1: Processed Spike Events                               │
│   • From Stage 2b: processed_spikes.jsonl                       │
│   • Voxel density clustering                                    │
│   • UV-LIF enrichment                                           │
│                                                                 │
│ Channel 2: RT Geometric Signals [NEW]                           │
│   • From Stage 2b: rt_signals.json                              │
│   • Void formation timeline                                     │
│   • Spatial correlation with spike clusters                     │
│                                                                 │
│ Channel 3: RT Solvation Signals [NEW]                           │
│   • From Stage 2b: solvation_analysis.json                      │
│   • Disruption events (leading indicators)                      │
│   • Predict pocket opening 100-500 fs early                     │
│                                                                 │
│ Channel 4: Aromatic LIF Signals [NEW]                           │
│   • From Stage 2b: aromatic_lif_events.json                     │
│   • Excitation spatial patterns                                 │
│   • UV-validated aromatic proximity                             │
│                                                                 │
│ HIERARCHICAL DETECTION:                                         │
│   Priority 1: Solvation disruption (earliest, Channel 3)        │
│   Priority 2: Geometric void (intermediate, Channel 2)          │
│   Priority 3: Spike cluster (confirmation, Channel 1)           │
│   Priority 4: Aromatic LIF (UV validation, Channel 4)           │
│                                                                 │
│ Sites with ALL FOUR signals = HIGHEST confidence!               │
│                                                                 │
│ Outputs:                                                        │
│   candidate_sites.json (now includes RT confidence scores)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: FINALIZE/OUTPUT [UNCHANGED]                            │
│ Tags: [STAGE-4]                                                 │
├─────────────────────────────────────────────────────────────────┤
│ • Pharma filtering (>50% replica agreement)                     │
│ • Generate reports (now includes RT probe visualizations)       │
│ • Temporal analytics (uses trajectory from Stage 2b)            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Hybrid Explicit/Implicit Strategy

**Mode 1: Pure Implicit** (fast exploration)
`[STAGE-1-IMPLICIT]`
- No water molecules
- Cryo-UV-LIF mean-field solvent
- RT probes: BVH-protein only (geometric voids)
- Speed: ~1 μs/day
- Use case: Broad conformational sampling

**Mode 2: Pure Explicit** (high fidelity)
`[STAGE-1-EXPLICIT]`
- Full water box (~30-80K waters for RBD)
- Explicit water dynamics
- RT probes: BVH-protein + BVH-solvent (both)
- Speed: ~50-100 ns/day
- Use case: Detailed pocket characterization

**Mode 3: Hybrid** (adaptive) ⭐ RECOMMENDED
`[STAGE-1-HYBRID]`
- Start with implicit (exploration)
- RT probes monitor for "interesting" regions
- When geometric void detected → switch to explicit for that region
- Continue implicit for rest of protein
- Speed: Adaptive (mostly fast, occasionally detailed)
- Use case: Efficient discovery + high-fidelity validation

### Hybrid Mode Implementation

**Phase 1: Implicit Exploration**
```rust
// Run implicit MD with RT probes
let implicit_results = run_cd_implicit(&system, 1_000_000)?;

// Analyze RT probe data
let void_regions = analyze_rt_voids(&implicit_results.rt_data)?;

// Sort by confidence
void_regions.sort_by(|a, b| b.confidence.cmp(&a.confidence));
```

**Phase 2: Explicit Characterization** (targeted)
```rust
// Take top 3-5 void regions
for region in void_regions.iter().take(5) {
    // Solvate just this region (10Å shell)
    let local_waters = solvate_region(region.center, 10.0)?;

    // Run explicit MD focused on this pocket
    let explicit_results = run_cd_explicit_local(
        &system,
        region,
        local_waters,
        100_000  // 100K steps explicit
    )?;

    // High-fidelity characterization with BVH-solvent
    analyze_solvation_dynamics(&explicit_results)?;
}
```

---

## Stage 1 Modifications

### New Config Structure
`[STAGE-1-CONFIG]`

```rust
// crates/prism-nhs/src/config.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SolventMode {
    Implicit,
    Explicit { padding_angstroms: f32 },
    Hybrid {
        exploration_steps: i32,
        characterization_steps: i32,
        switch_threshold: f32,  // RMSD drift
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RtProbeConfig {
    pub enabled: bool,
    pub probe_interval: i32,           // Steps between probe bursts
    pub rays_per_point: usize,         // Rays per attention point
    pub attention_points: usize,       // Number of probe origins
    pub bvh_refit_threshold: f32,      // Å displacement to trigger refit
    pub enable_solvent_probes: bool,   // Only for explicit mode
    pub enable_aromatic_lif: bool,     // LIF for aromatics
}

impl Default for RtProbeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            probe_interval: 100,
            rays_per_point: 256,
            attention_points: 50,
            bvh_refit_threshold: 0.5,
            enable_solvent_probes: false,  // Only for explicit
            enable_aromatic_lif: true,
        }
    }
}
```

### Solvation Function
`[STAGE-1-SOLVATE]`

```rust
// crates/prism-nhs/src/solvate.rs (NEW FILE)

/// Add water box around protein
pub fn solvate_protein(
    topology: &PrismPrepTopology,
    coordinates: &[f32],
    padding: f32,
) -> Result<(Vec<f32>, Vec<usize>)> {
    // 1. Compute protein bounding box
    let (min_coords, max_coords) = compute_bbox(coordinates)?;

    // 2. Expand by padding
    let box_min = min_coords - padding;
    let box_max = max_coords + padding;

    // 3. Fill box with water (TIP3P model, ~1g/cm³ density)
    let water_spacing = 3.1;  // Å between water oxygens
    let mut water_coords = Vec::new();
    let mut water_indices = Vec::new();

    for x in (box_min[0]..box_max[0]).step_by(water_spacing as usize) {
        for y in (box_min[1]..box_max[1]).step_by(water_spacing as usize) {
            for z in (box_min[2]..box_max[2]).step_by(water_spacing as usize) {
                // Check if water overlaps with protein
                if !overlaps_protein(x, y, z, coordinates, 2.4) {
                    let water_idx = water_coords.len() / 3;
                    water_coords.extend_from_slice(&[x, y, z]);
                    water_indices.push(topology.n_atoms + water_idx);
                }
            }
        }
    }

    Ok((water_coords, water_indices))
}
```

---

## Stage 2a RT Integration

### RT Probe Engine Structure
`[STAGE-2A-RT-ENGINE]`

```rust
// crates/prism-gpu/src/rt_probe.rs (NEW FILE)

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

pub struct RtProbeEngine {
    // OptiX context
    optix_ctx: OptixContext,
    optix_module: OptixModule,

    // BVH structures (persistent)
    bvh_protein: OptixAccelStructure,  // ~5K atoms (protein heavy)
    bvh_solvent: Option<OptixAccelStructure>,  // ~30-80K (water oxygens)

    // Probe configuration
    attention_points: CudaSlice<Vec3>,  // Where to probe from
    probe_rays: CudaSlice<Ray>,         // Pre-generated ray directions

    // Output buffers
    hit_results: CudaSlice<HitResult>,

    // Async execution
    probe_stream: CudaStream,  // Separate from MD stream!

    // State tracking
    last_refit_step: i32,
    max_displacement: f32,
}

impl RtProbeEngine {
    /// Create RT probe engine with initial BVH
    pub fn new(
        cuda_ctx: &CudaContext,
        protein_atoms: &[f32],  // positions
        water_atoms: Option<&[f32]>,  // if explicit
        config: &RtProbeConfig,
    ) -> Result<Self> {
        // 1. Initialize OptiX
        let optix_ctx = init_optix()?;

        // 2. Build BVH-protein
        let bvh_protein = build_bvh(&optix_ctx, protein_atoms)?;

        // 3. Build BVH-solvent (if explicit)
        let bvh_solvent = if let Some(waters) = water_atoms {
            Some(build_bvh(&optix_ctx, waters)?)
        } else {
            None
        };

        // 4. Setup probe rays (uniform sphere sampling)
        let probe_rays = generate_probe_rays(config.rays_per_point)?;

        Ok(Self { /* ... */ })
    }

    /// Refit BVH without rebuild (cheap: ~50-200μs)
    pub fn refit(&mut self, new_positions: &CudaSlice<f32>) -> Result<()> {
        // Update BVH-protein bounding boxes
        optix_accel_refit(&mut self.bvh_protein, new_positions)?;

        // Update BVH-solvent if exists
        if let Some(ref mut bvh) = self.bvh_solvent {
            optix_accel_refit(bvh, new_positions)?;
        }

        self.max_displacement = 0.0;
        Ok(())
    }

    /// Fire probe rays (async on RT cores)
    pub fn probe(&self) -> Result<CudaSlice<HitResult>> {
        // Launch OptiX ray tracing (runs on RT cores!)
        optix_launch(
            &self.optix_ctx,
            &self.probe_stream,  // Async!
            &self.attention_points,
            &self.probe_rays,
            &self.bvh_protein,
        )?;

        Ok(self.hit_results.clone())
    }

    /// LIF probes for aromatics (NEW)
    pub fn probe_aromatic_lif(
        &self,
        aromatic_positions: &CudaSlice<Vec3>,
        is_excited: &CudaSlice<i32>,
    ) -> Result<Vec<LifEvent>> {
        // Fire rays from excited aromatics
        // Detect spatial patterns of excitation
        // Return LIF events with spatial context
        todo!("Implement aromatic LIF probing")
    }
}

#[repr(C)]
pub struct HitResult {
    pub ray_idx: u32,
    pub hit_distance: f32,      // -1.0 if miss
    pub hit_atom_idx: i32,      // -1 if miss
    pub hit_normal: [f32; 3],
}

pub struct LifEvent {
    pub timestep: i32,
    pub aromatic_idx: usize,
    pub spatial_pattern: Vec<f32>,  // Hit distances in pattern
    pub excitation_level: f32,
}
```

### Integration into FusedEngine
`[STAGE-2A-RT-INTEGRATE]`

```rust
// crates/prism-nhs/src/fused_engine.rs

pub struct FusedEngine {
    // ... existing fields ...

    // NEW: RT probe engine
    rt_probe: Option<RtProbeEngine>,
    rt_probe_interval: i32,
    rt_probe_data: Vec<RtProbeSnapshot>,
}

pub struct RtProbeSnapshot {
    pub timestep: i32,
    pub hit_distances: Vec<f32>,
    pub void_detected: bool,
    pub solvation_variance: Option<f32>,  // If explicit
    pub aromatic_lif: Vec<LifEvent>,
}

impl FusedEngine {
    // In run() method, add RT probing
    pub fn run(&mut self, n_steps: i32) -> Result<RunSummary> {
        // ... existing batch loop ...

        for batch_idx in 0..num_batches {
            // Run MD batch
            let result = self.step_batch(current_batch_size)?;

            // RT probing (async on RT cores!)
            if let Some(ref mut rt) = self.rt_probe {
                if self.timestep % self.rt_probe_interval == 0 {
                    // Check if refit needed
                    if rt.max_displacement > rt.bvh_refit_threshold {
                        rt.refit(&self.d_positions)?;
                    }

                    // Fire probe rays (non-blocking!)
                    let hits = rt.probe()?;

                    // Aromatic LIF probes
                    let lif_events = rt.probe_aromatic_lif(
                        &self.d_aromatic_centroids,
                        &self.d_is_excited,
                    )?;

                    // Store results (download happens in Stage 2b)
                    self.rt_probe_data.push(RtProbeSnapshot {
                        timestep: self.timestep,
                        hit_distances: hits.to_vec(),
                        aromatic_lif: lif_events,
                        /* ... */
                    });
                }
            }

            // Continue with trajectory marking...
        }

        Ok(/* ... */)
    }
}
```

---

## Stage 2b RT Data Processing

### Solvation Analysis Module
`[STAGE-2B-SOLVATION]`

```rust
// crates/prism-nhs/src/solvation_analysis.rs (NEW FILE)

pub fn analyze_solvation_disruption(
    rt_data: &[RtProbeSnapshot],
    window_size: usize,
) -> Result<Vec<SolvationEvent>> {
    let mut events = Vec::new();

    for window in rt_data.windows(window_size) {
        // Compute variance in hit distances
        let variance = compute_hit_variance(window)?;

        // High variance = water disruption
        if variance > DISRUPTION_THRESHOLD {
            events.push(SolvationEvent {
                start_timestep: window.first().unwrap().timestep,
                end_timestep: window.last().unwrap().timestep,
                variance,
                confidence: variance / DISRUPTION_THRESHOLD,
            });
        }
    }

    Ok(events)
}

pub struct SolvationEvent {
    pub start_timestep: i32,
    pub end_timestep: i32,
    pub variance: f32,
    pub confidence: f32,
}
```

---

## Implementation Sequence

### Phase 1: Stage 1 Modifications (Week 1)
`[STAGE-1-RT]`

1. Add `SolventMode` enum to config
2. Implement `solvate.rs` for water box generation
3. Add `RtProbeConfig` to system config
4. Modify `PreparedSystem` to include RT targets

### Phase 2: OptiX FFI Bindings (Week 1-2)
`[STAGE-2A-RT-FFI]`

1. Create `optix_sys` crate for FFI
2. Implement OptiX context initialization
3. Implement BVH building/refitting
4. Test basic ray tracing (unit tests)

### Phase 3: RT Probe Engine (Week 2)
`[STAGE-2A-RT-ENGINE]`

1. Implement `RtProbeEngine` struct
2. Integrate into `FusedEngine`
3. Test async probe execution
4. Verify <10% performance overhead

### Phase 4: Stage 2b RT Processing (Week 2-3)
`[STAGE-2B-RT]`

1. Implement solvation analysis
2. Implement geometric void detection
3. Implement aromatic LIF correlation
4. Generate output JSON files

### Phase 5: Stage 3 Enhancement (Week 3)
`[STAGE-3-RT]`

1. Add RT signal channels
2. Implement hierarchical detection
3. Test on validation dataset

---

## Testing Strategy

### RT Probe Tests
```bash
cargo test rt_probe_basic --release
# Verify: BVH builds, rays fire, hits returned

cargo test rt_probe_async --release
# Verify: RT probes don't block MD stream

cargo test rt_probe_overhead --release
# Target: <10% performance impact
```

### Hybrid Mode Tests
```bash
cargo test hybrid_implicit_explicit --release
# Verify: Switches correctly, water added/removed

cargo test hybrid_convergence --release
# Verify: Hybrid finds same sites as pure explicit
```

---

## Performance Targets

| Component | Target | Current | Status |
|-----------|--------|---------|--------|
| **MD (implicit)** | 1500+ steps/sec | 190 | ❌ |
| **MD (explicit)** | 50-100 ns/day | - | 📝 |
| **RT probe overhead** | <10% | - | 📝 |
| **BVH refit** | <100 μs | - | 📝 |
| **Probe burst** | <200 μs | - | 📝 |

---

**END OF RT INTEGRATION PLAN**

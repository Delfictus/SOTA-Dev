# PRISM4D Stage Architecture
**Version:** 1.0.0
**Date:** 2026-02-01
**Status:** Canonical Reference

---

## Overview

PRISM4D uses a **4-stage pipeline architecture** with clear separation of concerns. Each stage has:
- **Defined inputs/outputs**
- **Stage tags** for code organization
- **Performance targets**
- **Testing requirements**

All future implementations MUST reference this document and use proper stage tags.

---

## Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 1: PRISM PREP/INPUT                                       │
│ Tags: [STAGE-1], [STAGE-1-PREP], [STAGE-1-TOPO]                │
├─────────────────────────────────────────────────────────────────┤
│ Responsibilities:                                               │
│ • Load topology (.topology.json) and PDB coordinates            │
│ • Parse AMBER ff14SB force field parameters                     │
│ • Setup simulation grid (voxel dimensions, spacing)             │
│ • Identify aromatic targets for UV excitation                   │
│ • Configure temperature protocols (cryo → warm)                 │
│ • Initialize GPU buffers and CUDA context                       │
│                                                                 │
│ Inputs:  .topology.json, .pdb, config.json                      │
│ Outputs: PreparedSystem (in-memory)                             │
│                                                                 │
│ Modules:                                                        │
│   crates/prism-nhs/src/input.rs          [STAGE-1-TOPO]        │
│   crates/prism-nhs/src/fused_engine.rs   [STAGE-1-PREP]        │
│   crates/prism-gpu/build.rs               [STAGE-1-GPU]         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2a: CONFORMATIONAL DYNAMICS (CD)                          │
│ Tags: [STAGE-2A], [STAGE-2A-MD], [STAGE-2A-GPU]                │
├─────────────────────────────────────────────────────────────────┤
│ Responsibilities:                                               │
│ • Run GPU-accelerated MD simulation (AMBER ff14SB)              │
│ • Temperature protocols (cryo hold → ramp → warm hold)          │
│ • UV excitation dynamics (wavelength-dependent absorption)      │
│ • Langevin thermostat with cryo scaling                         │
│ • Generate spike events (raw, timestamped pocket candidates)    │
│ • Keep trajectory frames in GPU memory                          │
│                                                                 │
│ ⚠️  DOES NOT PROCESS SPIKES - only records them!                │
│ ⚠️  DOES NOT DETECT SITES - that's Stage 3!                     │
│                                                                 │
│ Inputs:  PreparedSystem                                         │
│ Outputs: spike_events.jsonl (raw) + trajectory data (GPU mem)   │
│                                                                 │
│ Performance Targets:                                            │
│   - RTX 3060 baseline: 1500-2000 steps/sec                      │
│   - RTX 5080 target:   4500-10,000 steps/sec (3-5x faster)     │
│   - Current (BROKEN):  183-190 steps/sec ❌                     │
│                                                                 │
│ Modules:                                                        │
│   crates/prism-nhs/src/fused_engine.rs   [STAGE-2A-MD]         │
│   crates/prism-gpu/src/kernels/           [STAGE-2A-GPU]        │
│     nhs_amber_fused.cu                                          │
│     amber_simd_batch.cu                   [STAGE-2A-BATCH]      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2b: TRAJECTORY EXTRACTION/GENERATION                      │
│ Tags: [STAGE-2B], [STAGE-2B-TRAJ], [STAGE-2B-RMSF]             │
├─────────────────────────────────────────────────────────────────┤
│ Responsibilities:                                               │
│ • Process Stage 2a outputs ONLY                                 │
│ • Download trajectory frames from GPU → disk                    │
│ • Save regular interval snapshots (every 5K steps)              │
│ • Filter/score spike events using trajectory context            │
│ • Compute RMSF convergence (Cα first-half vs second-half)       │
│ • Cluster to representative conformations (50-200 frames)       │
│ • Generate ensemble PDB (multi-model format)                    │
│ • Compute Boltzmann weights for representatives                 │
│                                                                 │
│ ⚠️  CRITICAL: This stage is CURRENTLY MISSING! ❌                │
│                                                                 │
│ Inputs:  spike_events.jsonl (raw) + trajectory data (GPU)       │
│ Outputs: trajectory.pdb, frames.json, processed_spikes.jsonl,   │
│          rmsf_analysis.json, cluster_representatives.pdb        │
│                                                                 │
│ Quality Metrics:                                                │
│   - RMSF convergence: Pearson(first_half, second_half) > 0.8    │
│   - Frame count:      ≥20 frames minimum for convergence check  │
│   - Clustering:       50-200 representatives (Boltzmann weight) │
│                                                                 │
│ Modules (TO BE CREATED):                                        │
│   crates/prism-nhs/src/trajectory.rs      [STAGE-2B-TRAJ] ✅    │
│   crates/prism-nhs/src/rmsf.rs            [STAGE-2B-RMSF] ❌    │
│   crates/prism-nhs/src/clustering.rs      [STAGE-2B-CLUSTER] ❌ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 3: CRYPTIC BINDING SITE DETECTION                         │
│ Tags: [STAGE-3], [STAGE-3-CLUSTER], [STAGE-3-ENRICH]           │
├─────────────────────────────────────────────────────────────────┤
│ Responsibilities:                                               │
│ • Process Stage 2b outputs ONLY                                 │
│ • Load processed_spikes.jsonl (filtered, scored)                │
│ • Voxel density peak detection (adaptive threshold)             │
│ • Cluster spikes into candidate cryptic sites                   │
│ • UV-LIF enrichment calculation (aromatic validation)           │
│ • Residue mapping (5Å spatial query)                            │
│ • Druggability scoring (volume, persistence, hydrophobicity)    │
│                                                                 │
│ ⚠️  DOES NOT RUN MD - processes outputs only!                   │
│                                                                 │
│ Inputs:  processed_spikes.jsonl + trajectory context            │
│ Outputs: candidate_sites.json (20 sites, ranked)                │
│                                                                 │
│ Modules:                                                        │
│   crates/prism-report/src/clustering.rs   [STAGE-3-CLUSTER]    │
│   crates/prism-report/src/enrichment.rs   [STAGE-3-ENRICH]     │
│   crates/prism-report/src/druggability.rs [STAGE-3-DRUG]       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ Stage 4: FINALIZE/OUTPUT                                        │
│ Tags: [STAGE-4], [STAGE-4-PHARMA], [STAGE-4-REPORT]            │
├─────────────────────────────────────────────────────────────────┤
│ Responsibilities:                                               │
│ • Process Stage 3 outputs + Stage 2b trajectory                 │
│ • Pharma-grade pocket filtering (>50% replica agreement)        │
│ • Temporal analytics (persistence, opening rates)               │
│ • Compute pocket quality tiers (INSUFFICIENT/LOW/HIGH)          │
│ • Generate report.html with visualizations                      │
│ • Generate summary.json (deterministic hash)                    │
│ • Write MRC volumes (occupancy, pocket fields)                  │
│ • Generate PyMOL/ChimeraX sessions (if installed)               │
│                                                                 │
│ Inputs:  candidate_sites.json + trajectory.pdb                  │
│ Outputs: report.html, summary.json, pharma_report.json,         │
│          site_metrics.csv, volumes/*.mrc, provenance/           │
│                                                                 │
│ Quality Tiers:                                                  │
│   INSUFFICIENT: 0 accepted pockets                              │
│   LOW:          1-10 accepted pockets                           │
│   HIGH:         10+ accepted pockets with >50% replica agreement│
│                                                                 │
│ Modules:                                                        │
│   crates/prism-report/src/finalize.rs     [STAGE-4-FINALIZE]   │
│   crates/prism-report/src/pharma.rs       [STAGE-4-PHARMA]     │
│   crates/prism-report/src/outputs.rs      [STAGE-4-OUTPUT]     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Current Issues by Stage

### Stage 2a (Conformational Dynamics)
**Status:** ❌ BROKEN - Performance regression

**Problems:**
1. **Performance**: 183-190 steps/sec (should be 1500-2000+ on RTX 3060, 4500-10,000 on RTX 5080)
2. **Root cause**: Sequential step execution with excessive GPU synchronization
3. **Architecture**: Mixing CD with spike detection (should be separate)

**Solution Path:**
1. Implement concurrent replica execution using `amber_simd_batch.cu`
2. Run 3-10 replicas simultaneously for GPU saturation
3. Batch kernel launches (10K steps per sync instead of 100)
4. Decouple spike event recording from processing

**Stage Tags:** `[STAGE-2A-PERF]`, `[STAGE-2A-BATCH]`

---

### Stage 2b (Trajectory Extraction)
**Status:** ❌ MISSING - Not implemented

**Problems:**
1. **Trajectory files**: Never written to disk (trajectories/ directory empty)
2. **RMSF convergence**: Placeholder only (not implemented)
3. **Clustering**: No representative conformations generated
4. **Ensemble PDB**: Not created

**Solution Path:**
1. Integrate `TrajectoryWriter` into `FusedEngine`
2. Implement actual RMSF calculation (Cα atoms, first-half vs second-half)
3. Add clustering module (RMSD-based, Boltzmann weighted)
4. Write ensemble PDB in multi-model format

**Stage Tags:** `[STAGE-2B-TRAJ]`, `[STAGE-2B-RMSF]`, `[STAGE-2B-CLUSTER]`

---

### Stage 3 (Site Detection)
**Status:** ⚠️ WORKING but incorrectly coupled

**Problems:**
1. **Architecture**: Currently runs INSIDE Stage 2a (wrong!)
2. **Data flow**: Processes raw Stage 2a outputs instead of processed Stage 2b outputs

**Solution Path:**
1. Extract site detection into separate module
2. Update to consume `processed_spikes.jsonl` from Stage 2b
3. Add temporal analytics using trajectory context

**Stage Tags:** `[STAGE-3-REFACTOR]`

---

### Stage 4 (Finalize)
**Status:** ✅ WORKING but missing trajectory input

**Problems:**
1. **Temporal analytics**: Skipped because Stage 2b doesn't provide trajectory
2. **Replica agreement**: Works but only tested with 1 replica

**Solution Path:**
1. Add trajectory input from Stage 2b
2. Implement persistence calculation from trajectory frames
3. Test with 3+ replicas for proper validation

**Stage Tags:** `[STAGE-4-TEMPORAL]`

---

## RT Core Integration (FUTURE ENHANCEMENT)

### Overview
Use RTX 5080's 84 dedicated RT cores for **asynchronous spatial sensing** during MD simulation.

**Key Concept:** RT cores are idle silicon during scientific computing. Using them for spatial probing is essentially free compute.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ Stage 2a: CONFORMATIONAL DYNAMICS (RT Enhanced)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│   │ CUDA Cores  │  │  RT Cores   │  │Tensor Cores │            │
│   │  (10,752)   │  │    (84)     │  │   (336)     │            │
│   ├─────────────┤  ├─────────────┤  ├─────────────┤            │
│   │ AMBER MD    │  │ BVH-protein │  │  FluxNet    │            │
│   │ Langevin    │  │ BVH-solvent │  │  Steering   │            │
│   │ UV excite   │  │ Ray probes  │  │  Attention  │            │
│   └─────────────┘  └─────────────┘  └─────────────┘            │
│         │                │                │                     │
│         ▼                ▼                ▼                     │
│   ┌─────────────────────────────────────────────────┐          │
│   │   ALL THREE RUN CONCURRENTLY ON SEPARATE        │          │
│   │   SILICON (async CUDA streams)                  │          │
│   └─────────────────────────────────────────────────┘          │
│                                                                 │
│ NEW Output: rt_probe_data.jsonl (hit distances, solvation)     │
└─────────────────────────────────────────────────────────────────┘
```

### Three-Channel Neuromorphic Input

**Channel 1a: RT Probes (BVH-protein)**
`[STAGE-2A-RT-PROTEIN]`
- Geometric void detection
- Fire rays through persistent BVH of protein atoms
- Hit distances → cavity formation signal
- **Signal type:** Geometric (void space forming)
- **Latency:** ~100 μs per probe burst

**Channel 1b: RT Probes (BVH-solvent)** [Explicit solvent only]
`[STAGE-2A-RT-SOLVENT]`
- Solvation shell monitoring
- Fire rays through water oxygens near attention points
- Hit distance variance → water disruption signal
- **Signal type:** Precursor (happens BEFORE geometric change)
- **Latency:** ~200 μs per probe burst (larger BVH)

**Channel 2: cuVS Vector Similarity**
`[STAGE-2A-NEURO-CUVS]`
- Pattern matching against known pocket states
- Structural fingerprint comparison
- **Signal type:** Confirmation (matches known patterns)
- **Latency:** ~1 ms per query

**Channel 3: HelixDB Graph Traversal**
`[STAGE-2A-NEURO-HELIX]`
- Allosteric pathway analysis
- Water-mediated contact networks
- **Signal type:** Mechanistic context
- **Latency:** ~10 ms per query (CPU-based)

### FluxNet Integration

**Adaptive Attention Mechanism:**
`[STAGE-2A-NEURO-ATTENTION]`

FluxNet maintains learned attention map over protein surface:
- High attention → more probe rays allocated
- Low attention → rays reallocated to active regions
- Learns where to look based on probe results
- Triggers BVH refit when displacement exceeds threshold (~0.5 Å)

**Signal Hierarchy (Temporal):**
1. **Solvation disruption** (earliest, Channel 1b) - 100-500 fs before pocket opens
2. **Geometric void** (intermediate, Channel 1a) - as pocket opens
3. **Pattern match** (confirmation, Channel 2) - matches known state

FluxNet can predict pocket opening hundreds of timesteps early!

### Implementation Structure

**New Module:** `crates/prism-gpu/src/rt_probe.rs`
`[STAGE-2A-RT]`

```rust
pub struct RtProbeEngine {
    optix_ctx: OptixContext,
    bvh_protein: OptixAccelStructure,          // ~5K atoms
    bvh_solvent: Option<OptixAccelStructure>,  // ~30-80K waters (explicit only)
    attention_points: CudaBuffer<Vec3>,        // From FluxNet
    probe_stream: CudaStream,                  // Async from MD stream
}

impl RtProbeEngine {
    /// Refit BVH without rebuild (50-100μs protein, ~200μs solvent)
    pub fn refit(&mut self, positions: &CudaSlice<f32>) -> Result<()>;

    /// Fire probe rays - returns hit distances (async on RT cores)
    pub fn probe(&self, rays: &CudaSlice<Ray>) -> CudaBuffer<HitResult>;

    /// Solvation disruption: variance in hit distances over time window
    pub fn solvation_variance(&self, window: &[HitResult]) -> f32;
}
```

### Explicit vs Implicit Solvent Strategy

**Hybrid Approach (Recommended):**
`[STAGE-2A-HYBRID]`

**Phase 1: Implicit solvent exploration**
- Fast conformational sampling (~1 μs/day)
- RT probes detect geometric voids (BVH-protein only)
- FluxNet flags "interesting" regions
- Goal: Broad exploration

**Phase 2: Explicit solvent characterization** (targeted)
- Focused on flagged regions only (~50-100 ns/day)
- Full RT probe system (BVH-protein + BVH-solvent)
- Water reorganization monitoring
- Goal: High-fidelity pocket dynamics

**Performance Comparison:**

| Aspect | Implicit (Cryo-UV-LIF) | Explicit |
|--------|------------------------|----------|
| **Speed** | ~1 μs/day | ~50-100 ns/day |
| **Pocket detection** | Geometric inference | Direct water observation |
| **Leading indicators** | None | Solvation disruption (100-500 fs early) |
| **VRAM (RBD)** | ~500 MB | ~1-2 GB |
| **VRAM (full spike)** | ~2 GB | ~10+ GB (tight on 16GB) |
| **RT probe cost** | ~150 μs/timestep | ~350 μs/timestep |

### Stage 2b RT Processing

**New responsibilities:**
`[STAGE-2B-RT]`

Process RT probe data from Stage 2a:
- Hit distance time series → void formation rate
- Solvation variance → disruption events (leading signal)
- Water residence times → pocket hydration state
- Correlate RT events with conformational snapshots

**New outputs:**
- `solvation_analysis.json` - water disruption events
- `rt_probe_signals.json` - geometric void formation timeline

### Stage 3 RT Enhancement

**Signal hierarchy for site detection:**
`[STAGE-3-RT]`

1. **Solvation disruption** (EARLIEST) - from RT BVH-solvent
2. **Geometric void** (INTERMEDIATE) - from RT BVH-protein
3. **Pattern match** (CONFIRMATION) - from cuVS

Can detect cryptic sites BEFORE they fully open!

---

## Development Priorities

### Immediate (Week 1-2)
1. **[STAGE-2B] Implement trajectory extraction** (CRITICAL - currently missing)
   - Integrate TrajectoryWriter
   - Write frames to disk
   - Generate ensemble PDB

2. **[STAGE-2A-PERF] Fix performance regression**
   - Debug 183 steps/sec bottleneck
   - Implement concurrent replica execution
   - Target: 1500+ steps/sec minimum

3. **[STAGE-3-REFACTOR] Decouple site detection**
   - Extract from Stage 2a
   - Update to consume Stage 2b outputs

### Near-term (Week 3-4)
4. **[STAGE-2B-RMSF] Implement RMSF convergence**
   - Compute Cα fluctuations
   - First-half vs second-half correlation
   - Validate convergence (r > 0.8)

5. **[STAGE-2B-CLUSTER] Add clustering module**
   - RMSD-based representative selection
   - Boltzmann weighting
   - 50-200 representatives

### Future (Month 2+)
6. **[STAGE-2A-RT] RT core integration**
   - OptiX FFI bindings
   - BVH-protein persistent structure
   - Async probe engine

7. **[STAGE-2A-RT-SOLVENT] Explicit solvent + RT**
   - BVH-solvent for water monitoring
   - Solvation disruption detection
   - Hybrid implicit/explicit strategy

8. **[STAGE-2A-NEURO] FluxNet attention**
   - Adaptive probe allocation
   - Learned attention maps
   - Trigger-based BVH refit

---

## Stage Tag Reference

### Stage 1 Tags
- `[STAGE-1]` - General Stage 1 code
- `[STAGE-1-PREP]` - System preparation
- `[STAGE-1-TOPO]` - Topology loading/parsing
- `[STAGE-1-GPU]` - GPU initialization

### Stage 2a Tags
- `[STAGE-2A]` - General Stage 2a code
- `[STAGE-2A-MD]` - MD simulation core
- `[STAGE-2A-GPU]` - CUDA kernels
- `[STAGE-2A-BATCH]` - Batch/concurrent execution
- `[STAGE-2A-PERF]` - Performance optimization
- `[STAGE-2A-RT]` - RT core integration
- `[STAGE-2A-RT-PROTEIN]` - BVH-protein probes
- `[STAGE-2A-RT-SOLVENT]` - BVH-solvent probes (explicit)
- `[STAGE-2A-NEURO]` - FluxNet integration
- `[STAGE-2A-NEURO-CUVS]` - cuVS vector similarity
- `[STAGE-2A-NEURO-HELIX]` - HelixDB graph traversal
- `[STAGE-2A-NEURO-ATTENTION]` - Adaptive attention
- `[STAGE-2A-HYBRID]` - Implicit/explicit solvent hybrid

### Stage 2b Tags
- `[STAGE-2B]` - General Stage 2b code
- `[STAGE-2B-TRAJ]` - Trajectory extraction/writing
- `[STAGE-2B-RMSF]` - RMSF convergence calculation
- `[STAGE-2B-CLUSTER]` - Representative clustering
- `[STAGE-2B-RT]` - RT probe data processing

### Stage 3 Tags
- `[STAGE-3]` - General Stage 3 code
- `[STAGE-3-CLUSTER]` - Voxel density clustering
- `[STAGE-3-ENRICH]` - UV-LIF enrichment
- `[STAGE-3-DRUG]` - Druggability scoring
- `[STAGE-3-REFACTOR]` - Architecture refactoring
- `[STAGE-3-RT]` - RT signal hierarchy

### Stage 4 Tags
- `[STAGE-4]` - General Stage 4 code
- `[STAGE-4-FINALIZE]` - Core finalization
- `[STAGE-4-PHARMA]` - Pharma filtering
- `[STAGE-4-OUTPUT]` - Report generation
- `[STAGE-4-TEMPORAL]` - Temporal analytics

---

## Testing Requirements

Each stage MUST have:
1. **Unit tests** - Individual functions
2. **Integration tests** - Stage inputs → outputs
3. **Performance benchmarks** - Throughput targets
4. **Validation tests** - Output quality checks

### Stage 2a Tests
```bash
# Performance regression test
cargo test --release stage_2a_performance_target
# Target: ≥1500 steps/sec on RTX 3060, ≥4500 on RTX 5080

# Concurrent replica test
cargo test --release stage_2a_concurrent_replicas
# Verify: 3 replicas run with <1.5x single-replica time
```

### Stage 2b Tests
```bash
# Trajectory integrity test
cargo test stage_2b_trajectory_frames
# Verify: frames.json exists, ≥20 frames, valid format

# RMSF convergence test
cargo test stage_2b_rmsf_convergence
# Verify: Pearson correlation >0.8 between halves
```

### Stage 3 Tests
```bash
# Site detection quality test
cargo test stage_3_site_detection
# Verify: ≥10 sites detected, UV enrichment 1.5-3.0x
```

### Stage 4 Tests
```bash
# Pharma filtering test
cargo test stage_4_pharma_acceptance
# Verify: replica agreement >50%, quality tier ≥LOW
```

---

## Communication Protocol

When discussing implementations, ALWAYS:

1. **State the stage:** "I'm working on Stage 2b (trajectory extraction)"
2. **Use stage tags:** "This change is `[STAGE-2A-PERF]`"
3. **Reference boundaries:** "Stage 2a outputs raw spikes, Stage 2b processes them"
4. **Check dependencies:** "This requires Stage 2b completion first"

**Example (GOOD):**
> "I'm implementing RMSF convergence `[STAGE-2B-RMSF]` which processes trajectory frames from Stage 2a. This outputs rmsf_analysis.json for Stage 4 temporal analytics."

**Example (BAD):**
> "I'm adding some RMSF stuff to the engine."

---

## Document Updates

This document is **canonical** and should be updated when:
- New stages are added
- Stage boundaries change
- New tags are introduced
- Architecture decisions affect multiple stages

**Update procedure:**
1. Propose changes in discussion
2. Get user approval
3. Update this document
4. Tag commit with `[ARCH-UPDATE]`

**Current version:** 1.0.0
**Last updated:** 2026-02-01
**Next review:** After Stage 2b completion

---

## Key Architectural Principles

1. **Separation of Concerns**
   - Each stage has ONE primary responsibility
   - Stages communicate through well-defined outputs
   - No stage depends on another's internal implementation

2. **Data Flow is Unidirectional**
   - Stage 1 → 2a → 2b → 3 → 4
   - No backward dependencies
   - No stage skipping (e.g., Stage 3 cannot consume Stage 2a outputs directly)

3. **GPU Resources are Stage-Specific**
   - Stage 2a: CUDA cores (MD), RT cores (probes), Tensor cores (FluxNet)
   - Stage 2b: CUDA cores only (data processing)
   - Stage 3: CPU + minimal GPU (clustering)
   - Stage 4: CPU only (report generation)

4. **Performance is a Stage 2a Concern**
   - All performance optimization happens in Stage 2a
   - Other stages optimize for correctness, not speed
   - Stage 2a performance target: 1500+ steps/sec minimum

5. **All Code Must Have Stage Tags**
   - Functions, modules, commits must use stage tags
   - Makes it clear where changes belong
   - Prevents accidental coupling between stages

---

**END OF ARCHITECTURE DOCUMENT**

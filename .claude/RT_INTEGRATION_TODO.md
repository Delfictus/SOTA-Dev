# RT Core Integration: Production-Grade TODO List
**Version:** 1.0.0
**Date:** 2026-02-01
**Commitment Level:** ABSOLUTE - NO COMPROMISES, NO REGRESSIONS, NO SHORTCUTS

---

## ⚠️ CRITICAL RULES

1. **NO FEATURE is "done" until it passes ALL tests**
2. **NO REGRESSION is acceptable** - all existing functionality must continue working
3. **NO SHORTCUTS** - if it needs proper error handling, it gets proper error handling
4. **NO SILENT FAILURES** - every failure must be loud and clear
5. **END-TO-END TESTING** - every integration point must be tested
6. **PERFORMANCE TARGETS** must be met, not just "improved"

---

## Phase 1: Stage 1 Modifications (PRISM PREP Enhancement)
**Tag:** `[STAGE-1-RT]`
**Status:** 🔴 NOT STARTED
**Estimated:** 2-3 days

### 1.1 Configuration Infrastructure
`[STAGE-1-CONFIG]`

- [ ] **Create SolventMode enum in config.rs**
  - [ ] Implicit variant
  - [ ] Explicit { padding_angstroms: f32 } variant
  - [ ] Hybrid { exploration_steps, characterization_steps, switch_threshold } variant
  - [ ] Implement Serialize/Deserialize
  - [ ] Add validation (padding > 0, steps > 0)
  - [ ] **TEST:** Unit test for all variants
  - [ ] **TEST:** Serialization round-trip test

- [ ] **Create RtProbeConfig struct in config.rs**
  - [ ] All fields as specified in plan
  - [ ] Default implementation
  - [ ] Validation (intervals > 0, thresholds > 0)
  - [ ] **TEST:** Unit test for defaults
  - [ ] **TEST:** Validation rejects invalid configs

- [ ] **Add fields to main Config struct**
  - [ ] solvent_mode: SolventMode
  - [ ] rt_probe: RtProbeConfig
  - [ ] **TEST:** Config loads from JSON with new fields
  - [ ] **TEST:** Backward compatibility (old configs still work)

### 1.2 Water Box Generation
`[STAGE-1-SOLVATE]`

- [ ] **Create crates/prism-nhs/src/solvate.rs**
  - [ ] compute_bbox(coordinates) -> (Vec3, Vec3)
    - [ ] **TEST:** Correct bbox for known protein
  - [ ] overlaps_protein(pos, protein_coords, cutoff) -> bool
    - [ ] **TEST:** Detects overlaps correctly
    - [ ] **TEST:** Doesn't reject valid positions
  - [ ] solvate_protein(topology, coords, padding) -> Result<(Vec<f32>, Vec<usize>)>
    - [ ] Generate water grid (TIP3P spacing ~3.1 Å)
    - [ ] Remove overlapping waters (cutoff 2.4 Å)
    - [ ] Return water coords + indices
    - [ ] **TEST:** Water count reasonable (~30 waters/nm³)
    - [ ] **TEST:** No overlaps with protein
    - [ ] **TEST:** Waters fill entire box
  - [ ] solvate_region(center, radius, protein_coords) -> Result<Vec<f32>>
    - [ ] For hybrid mode (local solvation)
    - [ ] **TEST:** Waters only in specified region

### 1.3 RT Target Identification
`[STAGE-1-RT-TARGETS]`

- [ ] **Create crates/prism-nhs/src/rt_targets.rs**
  - [ ] RtTargets struct
    - [ ] protein_atoms: Vec<usize> (heavy atoms only)
    - [ ] water_atoms: Option<Vec<usize>> (O only if explicit)
    - [ ] aromatic_centers: Vec<Vec3> (ring centers for LIF)
  - [ ] identify_rt_targets(topology, solvent_mode) -> RtTargets
    - [ ] Extract protein heavy atoms (no hydrogens)
    - [ ] Extract water oxygens if explicit
    - [ ] Compute aromatic ring centers
    - [ ] **TEST:** Correct atom count for known protein
    - [ ] **TEST:** Aromatic centers at ring centroids

### 1.4 PreparedSystem Enhancement
`[STAGE-1-PREP]`

- [ ] **Modify crates/prism-nhs/src/input.rs**
  - [ ] Add solvent_mode: SolventMode to PreparedSystem
  - [ ] Add water_atoms: Option<Vec<usize>>
  - [ ] Add rt_targets: RtTargets
  - [ ] Add total_atoms: usize (protein + waters)
  - [ ] **TEST:** PreparedSystem serializes correctly
  - [ ] **TEST:** All fields populated correctly

- [ ] **Update prepare_system() function**
  - [ ] Check solvent_mode from config
  - [ ] If Explicit or Hybrid: call solvate_protein()
  - [ ] Identify RT targets
  - [ ] Validate total atom count
  - [ ] Log water count and RT target count
  - [ ] **TEST:** Implicit mode: no waters, protein only
  - [ ] **TEST:** Explicit mode: waters added, counts correct
  - [ ] **TEST:** Hybrid mode: starts implicit, metadata for explicit

### 1.5 Integration & Validation
`[STAGE-1-INTEGRATION]`

- [ ] **Update crates/prism4d/src/main.rs**
  - [ ] Parse solvent_mode from CLI or config
  - [ ] Pass to prepare_system()
  - [ ] Log solvent mode selected
  - [ ] **TEST:** CLI accepts --solvent-mode explicit
  - [ ] **TEST:** Config file overrides work

- [ ] **End-to-End Test: Implicit Mode**
  - [ ] Run prism4d with implicit config
  - [ ] Verify: no waters, protein atoms only
  - [ ] Verify: RT targets = protein heavy atoms
  - [ ] **PASS CRITERIA:** Same results as before (NO REGRESSION)

- [ ] **End-to-End Test: Explicit Mode**
  - [ ] Run prism4d with explicit config (10Å padding)
  - [ ] Verify: waters added (~30K for RBD)
  - [ ] Verify: RT targets = protein + water O
  - [ ] Verify: Total atom count correct
  - [ ] **PASS CRITERIA:** System initializes, atom counts match expected

- [ ] **Performance Baseline**
  - [ ] Measure prep time for implicit (baseline)
  - [ ] Measure prep time for explicit (should be <5s extra for solvation)
  - [ ] **PASS CRITERIA:** Explicit adds <5s overhead

---

## Phase 2: OptiX FFI Bindings
**Tag:** `[STAGE-2A-RT-FFI]`
**Status:** 🔴 NOT STARTED
**Estimated:** 3-4 days

### 2.1 OptiX System Crate
`[STAGE-2A-OPTIX-SYS]`

- [ ] **Create crates/optix-sys/Cargo.toml**
  - [ ] links = "optix"
  - [ ] build-dependencies: bindgen, cc
  - [ ] **TEST:** Crate compiles

- [ ] **Create crates/optix-sys/build.rs**
  - [ ] Find OptiX SDK (OPTIX_ROOT env var)
  - [ ] Generate bindings with bindgen
  - [ ] Link against optix.so / optix.dll
  - [ ] **TEST:** Build succeeds on Linux + Windows
  - [ ] **TEST:** Fails gracefully if OptiX not found

- [ ] **Create crates/optix-sys/src/lib.rs**
  - [ ] Include generated bindings
  - [ ] Unsafe OptiX C API wrappers
  - [ ] **TEST:** Can call optixInit()
  - [ ] **TEST:** Can query device properties

### 2.2 Safe OptiX Wrapper Crate
`[STAGE-2A-OPTIX]`

- [ ] **Create crates/prism-optix/Cargo.toml**
  - [ ] Dependencies: optix-sys, cudarc, anyhow

- [ ] **Create crates/prism-optix/src/context.rs**
  - [ ] OptixContext struct (owns optixDeviceContext)
  - [ ] new(cuda_ctx) -> Result<Self>
  - [ ] Drop impl (cleanup)
  - [ ] **TEST:** Context creation succeeds
  - [ ] **TEST:** Multiple contexts can coexist
  - [ ] **TEST:** Drop doesn't leak

- [ ] **Create crates/prism-optix/src/module.rs**
  - [ ] OptixModule struct (owns optixModule)
  - [ ] Load from PTX string or file
  - [ ] **TEST:** Module loads valid PTX
  - [ ] **TEST:** Rejects invalid PTX with clear error

- [ ] **Create crates/prism-optix/src/pipeline.rs**
  - [ ] OptixPipeline struct
  - [ ] Compile ray generation, miss, hit programs
  - [ ] **TEST:** Pipeline compiles successfully
  - [ ] **TEST:** Can set pipeline stack size

### 2.3 BVH Acceleration Structure
`[STAGE-2A-OPTIX-BVH]`

- [ ] **Create crates/prism-optix/src/accel.rs**
  - [ ] OptixAccelStructure struct
  - [ ] build(positions: &[f32], num_atoms: usize) -> Result<Self>
    - [ ] Convert atoms to AABBs
    - [ ] Build BVH with default settings
    - [ ] Return handle
    - [ ] **TEST:** BVH builds for 1K atoms
    - [ ] **TEST:** BVH builds for 100K atoms
    - [ ] **TEST:** Build time < 100ms for 100K atoms
  - [ ] refit(new_positions: &[f32]) -> Result<()>
    - [ ] Update AABBs
    - [ ] Refit BVH (no rebuild)
    - [ ] **TEST:** Refit faster than rebuild (>10x)
    - [ ] **TEST:** Refit produces correct results
  - [ ] Drop impl
    - [ ] **TEST:** Memory freed on drop

### 2.4 Ray Tracing Launch
`[STAGE-2A-OPTIX-LAUNCH]`

- [ ] **Create crates/prism-optix/src/launch.rs**
  - [ ] launch_ray_trace(ctx, pipeline, accel, origins, directions) -> Result<Vec<Hit>>
    - [ ] Setup shader binding table (SBT)
    - [ ] Allocate output buffer
    - [ ] Launch optixLaunch()
    - [ ] Copy results to host
    - [ ] **TEST:** Rays fire and return hits
    - [ ] **TEST:** Miss rays return -1 distance
    - [ ] **TEST:** Hit distances are accurate (±0.01 Å)

### 2.5 Integration Tests
`[STAGE-2A-OPTIX-INTEGRATION]`

- [ ] **Test: Single Atom Ray Cast**
  - [ ] Build BVH with 1 atom at origin
  - [ ] Fire ray from (10, 0, 0) toward origin
  - [ ] Verify: Hit at distance ~10 Å
  - [ ] **PASS CRITERIA:** Distance accurate to 0.01 Å

- [ ] **Test: Protein BVH**
  - [ ] Load RBD structure (5K atoms)
  - [ ] Build BVH
  - [ ] Fire 1000 random rays
  - [ ] Verify: Hit rate ~30-50% (reasonable for protein)
  - [ ] **PASS CRITERIA:** No crashes, hits returned

- [ ] **Test: BVH Refit After Motion**
  - [ ] Build BVH
  - [ ] Move atoms 1 Å
  - [ ] Refit BVH
  - [ ] Fire same rays
  - [ ] Verify: Different hits detected
  - [ ] **PASS CRITERIA:** Refit detects moved atoms

- [ ] **Performance: 100K Atom BVH**
  - [ ] Build BVH: <100 ms
  - [ ] Refit BVH: <10 ms
  - [ ] 10K ray trace: <1 ms
  - [ ] **PASS CRITERIA:** All timing targets met

---

## Phase 3: RT Probe Engine
**Tag:** `[STAGE-2A-RT-ENGINE]`
**Status:** 🔴 NOT STARTED
**Estimated:** 4-5 days

### 3.1 Probe Ray Generation
`[STAGE-2A-RT-RAYS]`

- [ ] **Create crates/prism-gpu/src/probe_rays.rs**
  - [ ] generate_sphere_rays(num_rays) -> Vec<Vec3>
    - [ ] Uniform sphere sampling (Fibonacci spiral)
    - [ ] Normalize directions
    - [ ] **TEST:** Rays uniformly distributed (chi-square test)
    - [ ] **TEST:** All rays unit length
  - [ ] generate_attention_points(protein_center, num_points, radius) -> Vec<Vec3>
    - [ ] Distribute points around protein
    - [ ] **TEST:** Points within specified radius

### 3.2 RT Probe Engine Core
`[STAGE-2A-RT-CORE]`

- [ ] **Create crates/prism-gpu/src/rt_probe.rs**
  - [ ] RtProbeEngine struct (as specified in plan)
  - [ ] new(cuda_ctx, protein_atoms, water_atoms, config) -> Result<Self>
    - [ ] Initialize OptiX context
    - [ ] Build BVH-protein
    - [ ] Build BVH-solvent if explicit
    - [ ] Generate probe rays
    - [ ] Allocate CUDA buffers
    - [ ] Create separate CUDA stream
    - [ ] **TEST:** Engine initializes successfully
    - [ ] **TEST:** Separate stream created
  - [ ] refit(new_positions) -> Result<()>
    - [ ] Check max_displacement
    - [ ] If > threshold: refit BVHs
    - [ ] **TEST:** Refit triggers correctly
    - [ ] **TEST:** Refit time <100 μs
  - [ ] probe() -> Result<CudaSlice<HitResult>>
    - [ ] Launch OptiX ray trace (ASYNC!)
    - [ ] Return immediately (non-blocking)
    - [ ] **TEST:** Returns without blocking MD stream
    - [ ] **TEST:** Results available after sync
  - [ ] probe_aromatic_lif(aromatic_pos, is_excited) -> Result<Vec<LifEvent>>
    - [ ] Fire rays from excited aromatics
    - [ ] Detect spatial excitation patterns
    - [ ] **TEST:** LIF events generated correctly
  - [ ] Drop impl
    - [ ] Cleanup OptiX resources
    - [ ] **TEST:** No memory leaks

### 3.3 Hit Result Processing
`[STAGE-2A-RT-RESULTS]`

- [ ] **Create crates/prism-gpu/src/rt_results.rs**
  - [ ] HitResult struct (as in plan)
  - [ ] RtProbeSnapshot struct (as in plan)
  - [ ] LifEvent struct (as in plan)
  - [ ] process_hits(hits, attention_points) -> RtProbeSnapshot
    - [ ] Compute statistics (mean distance, variance)
    - [ ] Detect voids (distance > threshold)
    - [ ] **TEST:** Void detection accurate

### 3.4 Integration into FusedEngine
`[STAGE-2A-RT-FUSED]`

- [ ] **Modify crates/prism-nhs/src/fused_engine.rs**
  - [ ] Add rt_probe: Option<RtProbeEngine>
  - [ ] Add rt_probe_interval: i32
  - [ ] Add rt_probe_data: Vec<RtProbeSnapshot>
  - [ ] **TEST:** Engine compiles with new fields

- [ ] **Update FusedEngine::new()**
  - [ ] If RT enabled: create RtProbeEngine
  - [ ] Pass protein atoms, water atoms (if explicit)
  - [ ] **TEST:** Engine creates with RT enabled
  - [ ] **TEST:** Engine creates with RT disabled (backward compat)

- [ ] **Update FusedEngine::run()**
  - [ ] In batch loop: check if probe interval reached
  - [ ] If yes: check displacement, refit if needed
  - [ ] Call rt.probe() (async!)
  - [ ] Call rt.probe_aromatic_lif() (async!)
  - [ ] Store RtProbeSnapshot
  - [ ] **TEST:** Probing doesn't block MD
  - [ ] **TEST:** Probe data accumulated correctly

### 3.5 Output Serialization
`[STAGE-2A-RT-OUTPUT]`

- [ ] **Create crates/prism-nhs/src/rt_output.rs**
  - [ ] write_rt_probe_data(snapshots, path) -> Result<()>
    - [ ] Serialize to JSONL format
    - [ ] One snapshot per line
    - [ ] **TEST:** File written correctly
    - [ ] **TEST:** Can be parsed back

- [ ] **Update FusedEngine::finalize()**
  - [ ] Write rt_probe_data.jsonl
  - [ ] Log RT probe statistics
  - [ ] **TEST:** File created in output dir

### 3.6 Integration Tests
`[STAGE-2A-RT-E2E]`

- [ ] **Test: RT Probes With Implicit Solvent**
  - [ ] Run 10K steps with RT enabled
  - [ ] Verify: rt_probe_data.jsonl exists
  - [ ] Verify: ~100 snapshots (every 100 steps)
  - [ ] Verify: Hit distances reasonable
  - [ ] **PASS CRITERIA:** File generated, data valid

- [ ] **Test: RT Probes With Explicit Solvent**
  - [ ] Run 10K steps explicit with RT enabled
  - [ ] Verify: BVH-solvent created
  - [ ] Verify: Solvation variance computed
  - [ ] **PASS CRITERIA:** Explicit features work

- [ ] **Test: Aromatic LIF Events**
  - [ ] Run with UV bursts
  - [ ] Verify: LIF events generated during UV
  - [ ] Verify: Spatial patterns detected
  - [ ] **PASS CRITERIA:** LIF correlates with UV

- [ ] **Performance Test: RT Overhead**
  - [ ] Run 10K steps WITHOUT RT: measure time
  - [ ] Run 10K steps WITH RT: measure time
  - [ ] Compute overhead percentage
  - [ ] **PASS CRITERIA:** Overhead <10%

---

## Phase 4: Stage 2b Implementation
**Tag:** `[STAGE-2B]`
**Status:** 🔴 NOT STARTED
**Estimated:** 4-5 days

### 4.1 Trajectory Writer Integration
`[STAGE-2B-TRAJ]`

- [ ] **Modify crates/prism-nhs/src/fused_engine.rs**
  - [ ] Add trajectory_writer: Option<TrajectoryWriter>
  - [ ] Initialize in new() if trajectory output enabled
  - [ ] **TEST:** Writer initializes correctly

- [ ] **Update FusedEngine::run()**
  - [ ] After each batch: extract trajectory frames
  - [ ] Convert EnsembleSnapshot -> TrajectoryFrame
  - [ ] Call trajectory_writer.add_frame()
  - [ ] **TEST:** Frames added to writer

- [ ] **Update FusedEngine::finalize()**
  - [ ] Call trajectory_writer.finalize(topology)
  - [ ] Generates trajectory.pdb
  - [ ] Generates frames.json
  - [ ] **TEST:** Files created
  - [ ] **TEST:** PDB has correct number of MODELs
  - [ ] **TEST:** frames.json parsable

### 4.2 RMSF Convergence Calculation
`[STAGE-2B-RMSF]`

- [ ] **Create crates/prism-nhs/src/rmsf.rs**
  - [ ] extract_ca_atoms(topology) -> Vec<usize>
    - [ ] Find all Cα atoms
    - [ ] **TEST:** Correct count for known protein
  - [ ] compute_rmsf(trajectory_frames, ca_indices) -> Vec<f32>
    - [ ] For each Cα: compute RMSF over all frames
    - [ ] **TEST:** RMSF values reasonable (0.5-3.0 Å typical)
  - [ ] compute_convergence(rmsf_first_half, rmsf_second_half) -> f32
    - [ ] Pearson correlation coefficient
    - [ ] **TEST:** Perfect correlation = 1.0
    - [ ] **TEST:** Anti-correlation = -1.0
  - [ ] check_rmsf_convergence(frames) -> Result<RmsfAnalysis>
    - [ ] Split frames in half
    - [ ] Compute RMSF for each half
    - [ ] Compute correlation
    - [ ] Return analysis struct
    - [ ] **TEST:** Returns correct convergence value

- [ ] **Create RmsfAnalysis struct**
  - [ ] convergence: f32 (correlation coefficient)
  - [ ] first_half_rmsf: Vec<f32>
  - [ ] second_half_rmsf: Vec<f32>
  - [ ] is_converged: bool (>0.8)
  - [ ] Serialize to JSON

### 4.3 Clustering Module
`[STAGE-2B-CLUSTER]`

- [ ] **Create crates/prism-nhs/src/clustering.rs**
  - [ ] compute_rmsd_matrix(frames) -> Vec<Vec<f32>>
    - [ ] All-pairs RMSD (Cα only)
    - [ ] **TEST:** Matrix symmetric
    - [ ] **TEST:** Diagonal = 0
  - [ ] cluster_representatives(rmsd_matrix, target_count) -> Vec<usize>
    - [ ] K-medoids or hierarchical clustering
    - [ ] Return representative frame indices
    - [ ] **TEST:** Returns ~target_count reps
  - [ ] compute_boltzmann_weights(frames, temperature) -> Vec<f32>
    - [ ] Weight by energy if available
    - [ ] **TEST:** Weights sum to 1.0
  - [ ] cluster_trajectory(frames, target_count) -> ClusterResult
    - [ ] Full clustering pipeline
    - [ ] **TEST:** Representatives span conformational space

### 4.4 RT Probe Data Processing
`[STAGE-2B-RT]`

- [ ] **Create crates/prism-nhs/src/solvation_analysis.rs**
  - [ ] SolvationEvent struct (as in plan)
  - [ ] compute_hit_variance(window) -> f32
    - [ ] Variance in hit distances over time window
    - [ ] **TEST:** High variance for disrupted water
  - [ ] analyze_solvation_disruption(rt_data, window_size) -> Vec<SolvationEvent>
    - [ ] Sliding window analysis
    - [ ] Detect high-variance regions
    - [ ] **TEST:** Detects disruption events
  - [ ] write_solvation_analysis(events, path) -> Result<()>
    - [ ] **TEST:** JSON file created

- [ ] **Create crates/prism-nhs/src/geometric_voids.rs**
  - [ ] VoidEvent struct
  - [ ] detect_void_formation(rt_data) -> Vec<VoidEvent>
    - [ ] Hit distance increasing = void forming
    - [ ] **TEST:** Detects voids correctly
  - [ ] correlate_with_spikes(voids, spikes) -> Vec<CorrelatedEvent>
    - [ ] Match voids to spike events (spatial + temporal)
    - [ ] **TEST:** Correlation accurate

- [ ] **Create crates/prism-nhs/src/aromatic_lif_analysis.rs**
  - [ ] analyze_lif_patterns(lif_events) -> LifAnalysis
    - [ ] Spatial clustering of excited aromatics
    - [ ] Temporal patterns (burst correlation)
    - [ ] **TEST:** Patterns detected correctly

### 4.5 Spike Event Processing
`[STAGE-2B-SPIKES]`

- [ ] **Create crates/prism-nhs/src/spike_processing.rs**
  - [ ] load_raw_spikes(path) -> Vec<RawSpike>
  - [ ] filter_by_quality(spikes, threshold) -> Vec<RawSpike>
    - [ ] **TEST:** Low-quality spikes removed
  - [ ] correlate_with_rt_signals(spikes, rt_data) -> Vec<ProcessedSpike>
    - [ ] Add RT confidence scores
    - [ ] Add solvation disruption flags
    - [ ] Add void formation flags
    - [ ] **TEST:** Correlation accurate
  - [ ] add_trajectory_context(spikes, frames) -> Vec<ProcessedSpike>
    - [ ] Which frame each spike occurred in
    - [ ] RMSD from reference
    - [ ] **TEST:** Context added correctly
  - [ ] write_processed_spikes(spikes, path) -> Result<()>
    - [ ] **TEST:** File created and parsable

### 4.6 Stage 2b Orchestration
`[STAGE-2B-ORCHESTRATE]`

- [ ] **Create crates/prism-nhs/src/stage2b.rs**
  - [ ] run_stage2b(engine, output_dir) -> Result<Stage2bOutput>
    - [ ] Step 1: Extract trajectories
    - [ ] Step 2: Compute RMSF convergence
    - [ ] Step 3: Cluster representatives
    - [ ] Step 4: Process RT probe data
    - [ ] Step 5: Process spike events
    - [ ] Step 6: Write all outputs
    - [ ] **TEST:** All files generated
    - [ ] **TEST:** All outputs valid

- [ ] **Stage2bOutput struct**
  - [ ] trajectory_path: PathBuf
  - [ ] rmsf_analysis: RmsfAnalysis
  - [ ] cluster_reps: ClusterResult
  - [ ] rt_signals: RtSignals
  - [ ] processed_spikes_path: PathBuf
  - [ ] All outputs for Stage 3

### 4.7 Integration Tests
`[STAGE-2B-E2E]`

- [ ] **Test: Complete Stage 2b Pipeline (Implicit)**
  - [ ] Run Stage 2a with RT (10K steps)
  - [ ] Run Stage 2b processing
  - [ ] Verify: All 8 output files exist
  - [ ] Verify: RMSF convergence computed
  - [ ] Verify: Clustering completed
  - [ ] **PASS CRITERIA:** All outputs valid, parsable

- [ ] **Test: Complete Stage 2b Pipeline (Explicit)**
  - [ ] Run Stage 2a explicit with RT (10K steps)
  - [ ] Run Stage 2b processing
  - [ ] Verify: Solvation analysis generated
  - [ ] Verify: BVH-solvent data processed
  - [ ] **PASS CRITERIA:** Explicit-specific outputs present

- [ ] **Test: RMSF Convergence Detection**
  - [ ] Run short simulation (should NOT converge)
  - [ ] Verify: is_converged = false
  - [ ] Run long simulation (should converge)
  - [ ] Verify: is_converged = true, correlation >0.8
  - [ ] **PASS CRITERIA:** Convergence detection accurate

- [ ] **Test: RT-Spike Correlation**
  - [ ] Run with UV bursts
  - [ ] Verify: Spikes correlated with RT voids
  - [ ] Verify: High-confidence spikes have RT support
  - [ ] **PASS CRITERIA:** Correlation logical and accurate

---

## Phase 5: Stage 3 Enhancement
**Tag:** `[STAGE-3-RT]`
**Status:** 🔴 NOT STARTED
**Estimated:** 3-4 days

### 5.1 Stage 3 Refactoring
`[STAGE-3-REFACTOR]`

- [ ] **Extract clustering from Stage 2a**
  - [ ] Remove site detection logic from fused_engine.rs
  - [ ] Move to separate module
  - [ ] **TEST:** Stage 2a NO LONGER does site detection
  - [ ] **TEST:** NO REGRESSION in existing tests

- [ ] **Create crates/prism-report/src/stage3.rs**
  - [ ] run_stage3(stage2b_output, topology) -> Result<Stage3Output>
  - [ ] Clear entry point for Stage 3
  - [ ] **TEST:** Can be called independently

### 5.2 Four-Channel Signal Loading
`[STAGE-3-SIGNALS]`

- [ ] **Create crates/prism-report/src/signal_loader.rs**
  - [ ] load_spike_signals(processed_spikes_path) -> Vec<ProcessedSpike>
  - [ ] load_rt_geometric(rt_signals_path) -> Vec<VoidEvent>
  - [ ] load_rt_solvation(solvation_path) -> Vec<SolvationEvent>
  - [ ] load_aromatic_lif(lif_path) -> Vec<LifEvent>
  - [ ] **TEST:** All signal types load correctly

### 5.3 Hierarchical Detection
`[STAGE-3-HIERARCHY]`

- [ ] **Create crates/prism-report/src/hierarchical_detection.rs**
  - [ ] Priority1: detect_from_solvation(events) -> Vec<CandidateSite>
    - [ ] EARLIEST signal
    - [ ] High priority
    - [ ] **TEST:** Early detection works
  - [ ] Priority2: detect_from_voids(events) -> Vec<CandidateSite>
    - [ ] INTERMEDIATE signal
    - [ ] **TEST:** Geometric detection works
  - [ ] Priority3: detect_from_spikes(events) -> Vec<CandidateSite>
    - [ ] CONFIRMATION signal
    - [ ] **TEST:** Spike clustering works
  - [ ] Priority4: validate_with_lif(sites, lif_events) -> Vec<CandidateSite>
    - [ ] UV VALIDATION
    - [ ] **TEST:** LIF validation works
  - [ ] merge_hierarchical(p1, p2, p3, p4) -> Vec<CrypticSite>
    - [ ] Combine all signals
    - [ ] Sites with ALL FOUR = highest confidence
    - [ ] **TEST:** Merging preserves high-confidence sites

### 5.4 Confidence Scoring
`[STAGE-3-CONFIDENCE]`

- [ ] **Create crates/prism-report/src/confidence.rs**
  - [ ] compute_rt_confidence(site, rt_signals) -> f32
    - [ ] 0-1 score based on RT support
    - [ ] **TEST:** High support = high score
  - [ ] compute_temporal_confidence(site, timeline) -> f32
    - [ ] Leading indicator timing
    - [ ] **TEST:** Early prediction = high score
  - [ ] compute_combined_confidence(site, all_signals) -> f32
    - [ ] Weighted combination
    - [ ] **TEST:** All-4-channels = score near 1.0

### 5.5 Stage 3 Output Enhancement
`[STAGE-3-OUTPUT]`

- [ ] **Modify CrypticSite struct**
  - [ ] Add rt_confidence: f32
  - [ ] Add solvation_supported: bool
  - [ ] Add void_detected: bool
  - [ ] Add lif_validated: bool
  - [ ] Add detection_hierarchy: Vec<String> (which signals fired)
  - [ ] **TEST:** Serialization includes new fields

- [ ] **Update candidate_sites.json format**
  - [ ] Include all RT metadata
  - [ ] **TEST:** Backward compatible (old code can still read)

### 5.6 Integration Tests
`[STAGE-3-E2E]`

- [ ] **Test: Four-Channel Detection**
  - [ ] Run full pipeline (Stage 1 → 2a → 2b → 3)
  - [ ] Verify: Sites have all 4 confidence scores
  - [ ] Verify: Detection hierarchy populated
  - [ ] **PASS CRITERIA:** All channels contribute

- [ ] **Test: Early Prediction**
  - [ ] Analyze timeline
  - [ ] Verify: Solvation disruption BEFORE spike
  - [ ] Verify: Geometric void BEFORE spike cluster
  - [ ] **PASS CRITERIA:** Temporal ordering correct

- [ ] **Test: High-Confidence Site Detection**
  - [ ] Identify sites with all 4 signals
  - [ ] Verify: Confidence scores >0.8
  - [ ] **PASS CRITERIA:** Best sites have full support

---

## Phase 6: Stage 4 Enhancement
**Tag:** `[STAGE-4-RT]`
**Status:** 🔴 NOT STARTED
**Estimated:** 2 days

### 6.1 Temporal Analytics Enhancement
`[STAGE-4-TEMPORAL]`

- [ ] **Update crates/prism-report/src/temporal.rs**
  - [ ] Add RT signal timeline
  - [ ] Compute pocket opening rates from RT data
  - [ ] Visualize solvation → void → spike timeline
  - [ ] **TEST:** Timeline visualization generated

### 6.2 Report Generation
`[STAGE-4-REPORT]`

- [ ] **Update report.html template**
  - [ ] Add RT probe statistics section
  - [ ] Add solvation disruption plots
  - [ ] Add aromatic LIF event timeline
  - [ ] Add 4-channel confidence visualizations
  - [ ] **TEST:** Report includes all RT data

- [ ] **Update summary.json**
  - [ ] Include RT probe stats (total probes, hit rate, etc.)
  - [ ] Include solvation event count
  - [ ] **TEST:** JSON schema valid

### 6.3 Integration Tests
`[STAGE-4-E2E]`

- [ ] **Test: Full Pipeline E2E (Implicit)**
  - [ ] Run: Stage 1 → 2a → 2b → 3 → 4
  - [ ] Verify: Complete report generated
  - [ ] Verify: All RT data visualized
  - [ ] **PASS CRITERIA:** report.html includes RT sections

- [ ] **Test: Full Pipeline E2E (Explicit)**
  - [ ] Run with explicit solvent
  - [ ] Verify: Solvation analysis in report
  - [ ] **PASS CRITERIA:** Explicit-specific features present

---

## Phase 7: Performance Optimization
**Tag:** `[STAGE-2A-PERF]`
**Status:** 🔴 NOT STARTED
**Estimated:** 3-4 days

### 7.1 Concurrent Replica Implementation
`[STAGE-2A-CONCURRENT]`

- [ ] **Create crates/prism-nhs/src/concurrent_replicas.rs**
  - [ ] run_concurrent_replicas(system, n_replicas, n_steps) -> Vec<ReplicaOutput>
    - [ ] Use amber_simd_batch.cu for parallel execution
    - [ ] Launch all replicas in single kernel
    - [ ] **TEST:** 3 replicas run in <1.5x single-replica time
  - [ ] **TEST:** Replica outputs independent (different seeds)

### 7.2 Batch Optimization
`[STAGE-2A-BATCH-OPT]`

- [ ] **Optimize step_batch() in fused_engine.rs**
  - [ ] Larger batch sizes (10K → 50K steps)
  - [ ] Minimize syncs (1 per 50K instead of per 10K)
  - [ ] **TEST:** Throughput improves with larger batches

### 7.3 RT Probe Overhead Minimization
`[STAGE-2A-RT-OPT]`

- [ ] **Profile RT probe system**
  - [ ] Measure: BVH refit time
  - [ ] Measure: Ray launch time
  - [ ] Measure: Result download time
  - [ ] **TEST:** Total RT overhead <10%

- [ ] **Optimize if needed**
  - [ ] Reduce probe frequency if >10%
  - [ ] Async result download
  - [ ] **TEST:** Overhead reduced to target

### 7.4 Performance Validation
`[STAGE-2A-PERF-VALIDATE]`

- [ ] **Benchmark: Implicit Solvent (NO RT)**
  - [ ] RBD (5K atoms), 100K steps
  - [ ] **TARGET:** ≥1500 steps/sec (RTX 3060 baseline)
  - [ ] **TARGET:** ≥4500 steps/sec (RTX 5080)
  - [ ] **PASS CRITERIA:** Targets met or exceeded

- [ ] **Benchmark: Implicit Solvent (WITH RT)**
  - [ ] Same system, RT probes enabled
  - [ ] **TARGET:** ≥1350 steps/sec (<10% overhead)
  - [ ] **PASS CRITERIA:** Overhead <10%

- [ ] **Benchmark: Explicit Solvent (WITH RT)**
  - [ ] RBD solvated (150K atoms), 10K steps
  - [ ] **TARGET:** ≥50 ns/day throughput
  - [ ] **PASS CRITERIA:** Explicit performance acceptable

- [ ] **Benchmark: Concurrent Replicas**
  - [ ] 3 replicas vs 1 replica (time comparison)
  - [ ] **TARGET:** 3 replicas in <1.5x single time
  - [ ] **PASS CRITERIA:** GPU saturation achieved

---

## Phase 8: End-to-End Validation
**Tag:** `[E2E-VALIDATION]`
**Status:** 🔴 NOT STARTED
**Estimated:** 2-3 days

### 8.1 Full Pipeline Tests
`[E2E-FULL]`

- [ ] **Test: RBD Implicit with RT (Complete Pipeline)**
  - [ ] Config: implicit, RT enabled, 100K steps
  - [ ] Run: Stage 1 → 2a → 2b → 3 → 4
  - [ ] Verify: All outputs generated
  - [ ] Verify: ≥10 sites detected
  - [ ] Verify: RT signals present
  - [ ] **PASS CRITERIA:** Complete, valid report

- [ ] **Test: RBD Explicit with RT (Complete Pipeline)**
  - [ ] Config: explicit, RT enabled, 10K steps
  - [ ] Run: Full pipeline
  - [ ] Verify: Solvation analysis present
  - [ ] Verify: Performance acceptable
  - [ ] **PASS CRITERIA:** Complete, valid report with explicit features

- [ ] **Test: RBD Hybrid Mode (Complete Pipeline)**
  - [ ] Config: hybrid (implicit → explicit on void detection)
  - [ ] Run: Full pipeline
  - [ ] Verify: Mode switching occurred
  - [ ] Verify: Explicit characterization of flagged regions
  - [ ] **PASS CRITERIA:** Hybrid strategy executed

### 8.2 Regression Tests
`[E2E-REGRESSION]`

- [ ] **Test: Backward Compatibility (RT Disabled)**
  - [ ] Run with RT disabled (config: rt_probe.enabled = false)
  - [ ] Compare results to baseline (before RT integration)
  - [ ] Verify: IDENTICAL results (bit-for-bit if deterministic)
  - [ ] **PASS CRITERIA:** ZERO REGRESSION when RT disabled

- [ ] **Test: Existing Tests Still Pass**
  - [ ] Run ALL existing unit tests: `cargo test --all`
  - [ ] Run ALL existing integration tests
  - [ ] **PASS CRITERIA:** 100% pass rate, NO NEW FAILURES

### 8.3 Quality Validation
`[E2E-QUALITY]`

- [ ] **Test: RMSF Convergence on Long Run**
  - [ ] Run 1M steps
  - [ ] Verify: RMSF convergence >0.8
  - [ ] **PASS CRITERIA:** Convergence detected

- [ ] **Test: RT Early Prediction**
  - [ ] Analyze pocket opening timeline
  - [ ] Verify: Solvation disruption 100-500 fs BEFORE spike
  - [ ] **PASS CRITERIA:** Leading indicator validated

- [ ] **Test: Four-Channel Confidence**
  - [ ] Identify sites with all 4 signals
  - [ ] Verify: Higher confidence than single-signal sites
  - [ ] **PASS CRITERIA:** Multi-signal sites ranked higher

### 8.4 Performance Validation
`[E2E-PERFORMANCE]`

- [ ] **Test: Performance Targets Met**
  - [ ] Implicit: ≥1500 steps/sec
  - [ ] Explicit: ≥50 ns/day
  - [ ] RT overhead: <10%
  - [ ] Concurrent replicas: <1.5x single time
  - [ ] **PASS CRITERIA:** ALL targets met

---

## Phase 9: Documentation & Polish
**Tag:** `[DOCS]`
**Status:** 🔴 NOT STARTED
**Estimated:** 1-2 days

### 9.1 Code Documentation
`[DOCS-CODE]`

- [ ] **Document all public APIs**
  - [ ] Every public function has doc comment
  - [ ] Every public struct has doc comment
  - [ ] Examples where appropriate
  - [ ] **TEST:** `cargo doc` runs without warnings

### 9.2 User Documentation
`[DOCS-USER]`

- [ ] **Create RT_USAGE_GUIDE.md**
  - [ ] How to enable RT probes
  - [ ] Config options explained
  - [ ] Implicit vs explicit vs hybrid
  - [ ] Performance considerations

- [ ] **Update README.md**
  - [ ] Add RT core features
  - [ ] Add explicit solvent support
  - [ ] Add hybrid mode description

### 9.3 Example Configs
`[DOCS-EXAMPLES]`

- [ ] **examples/rt_implicit.json**
  - [ ] Implicit + RT probes
- [ ] **examples/rt_explicit.json**
  - [ ] Explicit + RT probes + solvation
- [ ] **examples/rt_hybrid.json**
  - [ ] Hybrid mode (exploration → characterization)

---

## Final Checklist

### ABSOLUTE REQUIREMENTS (NO EXCEPTIONS)

- [ ] **ALL unit tests pass** (`cargo test --all`)
- [ ] **ALL integration tests pass**
- [ ] **ALL performance targets met**
  - [ ] Implicit: ≥1500 steps/sec
  - [ ] Explicit: ≥50 ns/day
  - [ ] RT overhead: <10%
- [ ] **ZERO regressions** (RT disabled = same as before)
- [ ] **End-to-end pipeline works** (Stage 1 → 2a → 2b → 3 → 4)
- [ ] **All outputs generated and valid**
  - [ ] trajectory.pdb
  - [ ] frames.json
  - [ ] rt_probe_data.jsonl
  - [ ] processed_spikes.jsonl
  - [ ] solvation_analysis.json (explicit)
  - [ ] candidate_sites.json (enhanced)
  - [ ] report.html (enhanced)
- [ ] **Documentation complete**
- [ ] **No compiler warnings**
- [ ] **No memory leaks** (valgrind clean)
- [ ] **Code committed to git** (proper stage tags)

### Success Criteria

**THIS IS NOT DONE UNTIL:**
1. Every checkbox above is ✅
2. Full pipeline runs successfully on RBD
3. Performance targets are MET (not "close")
4. All tests PASS (not "mostly pass")
5. RT cores demonstrate early prediction capability
6. Four-channel detection shows improvement over single-channel

---

**COMMITMENT:** I will implement this with ZERO compromises, NO shortcuts, NO defeats. Every feature will be production-grade, fully tested, and integrated end-to-end.

**START DATE:** 2026-02-01
**ESTIMATED COMPLETION:** 3-4 weeks (aggressive but achievable with full focus)

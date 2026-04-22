# PRISM4D — Complete Technical Reference & Patent-Ready Engineering Specification

**Version:** 1.0 — Source-Verified Against Codebase (2026-03-10)
**Classification:** CONFIDENTIAL — Trade Secret / Pre-Patent Filing
**Platform Version:** 0.3.0 (Cargo workspace)
**Target Hardware:** NVIDIA RTX 5080 (Blackwell GB202, SM_120, 192 SMs, 24,576 CUDA cores)

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Complete File Tree — All Touched Files](#2-complete-file-tree)
3. [Execution Paths](#3-execution-paths)
4. [Kernel Inventory — CUDA/PTX](#4-kernel-inventory)
5. [SDST Library — Complete Specification](#5-sdst-library)
6. [Mathematical Formulations](#6-mathematical-formulations)
7. [Data Structures — Wire Format](#7-data-structures)
8. [Feature Status Matrix](#8-feature-status-matrix)
9. [Validated Benchmark Results](#9-validated-benchmark-results)
10. [Novel Algorithms — Patent Candidates](#10-novel-algorithms)
11. [Known Limitations & Dead Code](#11-known-limitations)

---

## 1. System Architecture Overview

PRISM4D is a GPU-accelerated cryptic binding site detection platform written in Rust + CUDA. It fuses molecular dynamics simulation with neuromorphic signal processing to detect transient protein pockets invisible to static structure analysis.

### Core Innovation Stack

```
┌──────────────────────────────────────────────────────────────────┐
│                     nhs_rt_full (Rust binary)                    │
│    CLI orchestrator: topology → MD → spike clustering → output   │
├──────────────────────────────────────────────────────────────────┤
│              PersistentNhsEngine (fused_engine.rs)               │
│    CryoUvProtocol · PTX loading · GPU memory management          │
├──────────────────────────────────────────────────────────────────┤
│            nhs_amber_fused.cu (3,246 lines, 20 kernels)          │
│    AMBER ff14SB MD + RAF oscillator + UV-LIF neuromorphic        │
├──────────────────────────────────────────────────────────────────┤
│                SDST Library (7 CUDA files, 40+ FFI)              │
│    Hash table → Avalanche clustering → CCNS → TIDE → TCL        │
├──────────────────────────────────────────────────────────────────┤
│         Spike Thermodynamic Integration (STI) — Rust             │
│    Jarzynski · Crooks · BAR · Channel decomposition · Arrhenius  │
├──────────────────────────────────────────────────────────────────┤
│                   Output Layer                                    │
│    binding_sites.json · prism_therm.json · PDB visualizations    │
└──────────────────────────────────────────────────────────────────┘
```

### Build System

| Component | Language | Compiler | Target |
|-----------|----------|----------|--------|
| NHS pipeline | Rust 2021 | `cargo build --release --features gpu` | x86_64 |
| Main kernel | CUDA C++ | `nvcc -O3 --use_fast_math -arch=sm_120` | PTX (JIT) |
| SDST library | CUDA C++ | `nvcc -std=c++17 -O3 --use_fast_math -gencode arch=compute_120,code=sm_120` | libsdst.so |
| PTX signing | SHA-256 | `sha2` crate in build.rs | .ptx.sha256 |

### Workspace Crates (Cargo.toml)

```
prism-nhs          — NHS pipeline + all binaries (primary)
prism-gpu          — GPU context, PTX compilation, all CUDA abstractions
sdst               — SDST C library FFI wrapper
prism-core         — Core types
prism-io           — I/O subsystem
prism-geometry     — Geometric algorithms
prism-physics      — Physics constants
prism-fluxnet      — Flux network (RL)
prism-pipeline     — Pipeline orchestration
prism-ontology     — Ontological framework
prism-mec          — Metabolic engineering
prism-ve           — Viral evolution
prism-learning     — Self-calibrating RL
prism-gnn          — Graph neural networks
prism-lbs          — Ligand binding scoring
prism-report       — Evidence-pack generation
prism-validation   — Multi-tier validation
prism-whcr         — Water hydrogen chain repair
optix-sys / prism-optix — OptiX (deprecated on SM120+)
```

---

## 2. Complete File Tree

### PRIMARY HOT PATH (touches >95% of wall time)

```
crates/prism-nhs/
├── src/
│   ├── bin/
│   │   └── nhs_rt_full.rs              [3000+ lines] Main binary — CLI, pipeline orchestration
│   ├── fused_engine.rs                  [4000+ lines] PersistentNhsEngine, CryoUvProtocol, PTX loading
│   ├── persistent_engine.rs             GPU engine lifecycle management
│   ├── sdst_bridge.rs                   [500+ lines]  SDST ↔ NHS bridge, ThermClass classification
│   ├── sdst_report.rs                   PRISM-Therm report generation + PDB output
│   ├── spike_thermodynamic_integration.rs [958 lines] Jarzynski/Crooks/BAR/STI/Pharmacophore
│   ├── rt_clustering.rs                 OptiX BVH clustering (DEPRECATED on SM120+)
│   ├── hierarchical_clustering.rs       CPU fallback clustering
│   ├── parallel_replica.rs              Multi-replica parallel execution
│   ├── aromatic_proximity.rs            Aromatic ring detection + proximity scoring
│   ├── neuromorphic.rs                  DewettingNetwork/Neuron/Synapse (CPU reference)
│   ├── lib.rs                           Module declarations (28 pub mod, 25 re-export blocks)
│   └── Cargo.toml                       Dependencies: sdst, cudarc 0.18.2, prism-gpu, etc.

crates/prism-gpu/
├── src/
│   ├── kernels/
│   │   ├── nhs_amber_fused.cu           [3,246 lines] ★ THE kernel — MD + neuromorphic
│   │   ├── nhs_amber_fused.ptx          Compiled PTX (fallback copy)
│   │   ├── nhs_amber_fused_sm120.ptx    SM120-specific PTX variant
│   │   └── pharmacophore_splat.cu       [195 lines]   GPU Gaussian density splatting
│   ├── lib.rs                           Module declarations (30+ pub mod)
│   ├── amber_forces.rs                  AMBER ff14SB bonded force types
│   ├── amber_mega_fused.rs              Fused AMBER HMC dynamics
│   ├── amber_simd_batch.rs              SIMD batch MD for throughput
│   ├── pme.rs                           Particle Mesh Ewald (long-range electrostatics)
│   ├── settle.rs                        SETTLE water constraint solver
│   ├── h_constraints.rs                 X-H bond constraint solver
│   ├── verlet_list.rs                   Verlet neighbor list (O(N) acceleration)
│   ├── tensor_core_forces.rs            WMMA tensor core force accumulation
│   ├── async_md_pipeline.rs             Async CUDA stream pipeline
│   ├── context.rs                       CudaContext/CudaStream management
│   ├── global_context.rs                Global GPU singleton
│   ├── memory.rs                        VRAM safety guard
│   └── lcpo_sasa.rs                     GPU-accelerated SASA computation
├── build.rs                             PTX compilation (48 kernels), SHA-256 signing

crates/sdst/
├── src/
│   ├── sdst_core.cu                     [1,325 lines] Hash insert, parent detect, pointer jumping
│   ├── sdst_ccns_gpu.cu                 [1,526 lines] Histogram Hill MLE, CCNS τ estimation
│   ├── sdst_tide.cu                     [384 lines]   Transfer entropy decomposition
│   ├── sdst_tcl.cu                      [615 lines]   Thermodynamic context labels
│   ├── sdst_wct.cu                      [429 lines]   Wavefront coherence tracking
│   ├── sdst_analysis.cu                 [1,015 lines] Spatial queries, causal graphs, DCC
│   ├── sdst_serialize.cu               [237 lines]   Binary checkpoint save/load
│   ├── lib.rs                           Safe Rust FFI wrapper (Sdst struct, 40+ extern "C")
│   └── sdst_ffi.rs                      Raw extern "C" bindings
├── include/
│   ├── sdst_api.h                       [630 lines]   Public C API, all structs
│   └── sdst_internal.h                  [248 lines]   Morton code, hash, union-find helpers
├── lib/
│   └── libsdst.so                       Precompiled shared library (SM120)
└── Makefile                             Build system (nvcc, SM120, shared/static/test)

target/ptx/
├── nhs_amber_fused.ptx                  ★ PRIMARY runtime PTX (loaded first)
└── *.ptx.sha256                         SHA-256 signatures for all PTX modules
```

### SECONDARY PATH (invoked conditionally)

```
crates/prism-nhs/src/
├── pharmacophore_gpu.rs                 [383 lines] GPU pharmacophore density extraction
├── resonance_scan.rs                    [236 lines] Lorentzian resonance fitting [NOT USED]
├── spike_density.rs                     Spike density analysis
├── gpu_knn.rs                           GPU k-nearest-neighbors
├── active_sensing.rs                    Active inference sensing
├── ultimate_engine.rs                   UltimateEngine (2-4x faster MD variant)
├── adaptive.rs                          Adaptive parameter selection
├── composition.rs                       Composition analysis
├── batch_scheduler.rs                   Batch scheduling for multi-structure
├── uv_bias.rs                           UV bias protocol management
├── rmsf.rs                              RMSF calculation
└── trajectory.rs                        Trajectory I/O
```

### OUTPUT FILES (per structure)

```
<output_dir>/
├── <name>.binding_sites.json            Main results (sites, spikes, druggability)
├── <name>.pdb                           PyMOL visualization
├── <name>_druggability.pdb              Druggability-colored PDB
├── <name>_consensus_sites.pdb           Consensus site visualization
├── <name>.ensemble_trajectory.pdb       Multi-MODEL ensemble
├── <name>_prism_therm.json              PRISM-Therm thermodynamic report (if --prism-therm)
├── <name>_druggability_sites.pdb        PRISM-Therm druggability PDB (if --prism-therm)
├── <name>.site<N>.spike_events.json     Per-site spike events with full metadata
└── batch_summary.json                   Batch mode aggregate (if --manifest)
```

---

## 3. Execution Paths

### 3.1 CANONICAL PATH (Validated SNDC v9)

**Command:**
```bash
# Source of truth: crates/prism-nhs/src/bin/nhs_rt_full.rs (see docs/CANONICAL_PROVENANCE.md)
scripts/prism-validate-and-run.sh \
    -t <topology.json> \
    -o <output_dir> \
    --fast --hysteresis --prism-therm \
    --multi-stream 8 \
    --spike-percentile 70 \
    --fused-steps 6 \
    --hmr --adaptive-dt \
    --multi-differential \
    --closed-loop-steering --asymmetric-steering \
    --use-xgb-ranker \
    --replica-seed 42 -v
```

**Execution Flow:**

```
main()
 ├── parse CLI args (clap 4.5)
 ├── #[cfg(feature = "gpu")] gate
 └── run_single_structure()
      └── run_multi_stream_pipeline(args, topology, n_streams=8)
           │
           ├── [INIT] Create CudaContext + load PTX module
           │   └── PTX fallback chain (7 paths):
           │       1. target/ptx/nhs_amber_fused.ptx
           │       2. crates/prism-gpu/src/kernels/nhs_amber_fused.ptx
           │       3. crates/prism-nhs/src/nhs_amber_fused.ptx
           │       4. nhs_amber_fused_sm120.ptx variants
           │       5-7. Build directory fallbacks
           │
           ├── [PROTOCOL] CryoUvProtocol::fast_35k().with_hysteresis()
           │   ├── start_temp: 50K
           │   ├── end_temp: 300K
           │   ├── cold_hold: 14,000 steps    (Phase 1)
           │   ├── ramp_up: 6,000 steps       (Phase 2)
           │   ├── warm_hold: 15,000 steps     (Phase 3)
           │   ├── ramp_down: 6,000 steps      (Phase 4)  ← hysteresis
           │   ├── cold_return: 14,000 steps   (Phase 5)  ← hysteresis
           │   └── TOTAL: 55,000 steps/stream × 8 streams = 440,000 MD steps
           │
           ├── [MD LOOP — per stream, 8× concurrent]
           │   ├── PersistentNhsEngine::new(config) on dedicated CUDA stream
           │   ├── engine.load_topology(&topology)
           │   ├── engine.set_cryo_uv_protocol(protocol)
           │   ├── engine.set_spike_accumulation(true)
           │   ├── Two-phase friction:
           │   │   ├── γ = 1000 ps⁻¹ (equilibration, first ~100 steps)
           │   │   └── γ = 1.0 ps⁻¹ × √(T_ref/T) (production, cryo-scaled)
           │   │
           │   ├── engine.run(55000 steps)
           │   │   └── Per step: nhs_amber_fused_step() kernel launch
           │   │       ├── grid=(ceil(N/256), 1, 1), block=(256, 1, 1)
           │   │       ├── BAOAB Langevin integration (dt=0.002 ps)
           │   │       │   ├── Half-step velocity: v += 0.5*dt*F/m
           │   │       │   ├── O-U friction: v = c1*v + c2*R (Gaussian noise)
           │   │       │   ├── Position update: x += dt*v
           │   │       │   ├── SHAKE constraint: 10 iterations (bonds to H)
           │   │       │   └── Velocity clamp: |v| ≤ 100 Å/ps
           │   │       │
           │   │       ├── Nonbonded forces (cell list, NB_CUTOFF=10Å):
           │   │       │   ├── Lennard-Jones: ε,σ mixing + 1-4 scaling (0.5)
           │   │       │   ├── Coulomb: 332.0636/εr + 1-4 scaling (0.8333)
           │   │       │   └── Bonded: bonds + angles + dihedrals
           │   │       │
           │   │       ├── UV photon absorption (Beer-Lambert, multi-wavelength scan):
           │   │       │   ├── Wavelength cycles every 300 steps (wavelength_dwell_steps):
           │   │       │   │   [280nm(TRP), 274nm(TYR), 258nm(PHE), 211nm(HIS)]
           │   │       │   ├── Passed per-step as kernel param: uv_wavelength_nm
           │   │       │   │   (NOT from #define UV_WAVELENGTH — that is legacy/unused)
           │   │       │   ├── ε: TRP=5600, TYR=1490, PHE=197, HIS=300 M⁻¹cm⁻¹
           │   │       │   └── E = h×c/λ × QY(0.02) × relative_extinction × intensity
           │   │       │
           │   │       └── Neuromorphic spike detection:
           │   │           nhs_voxel_step_multi_lif() kernel launch
           │   │           ├── grid=(32,32,64), block=(256), shared=576 bytes
           │   │           ├── K=8 RAF oscillators per voxel
           │   │           ├── Log-spaced damping: τ_k = 0.5 × 2^k
           │   │           ├── Shared memory halo: 6×6×4 = 144 floats
           │   │           ├── Saturated resonance clamp: R ≤ RAF_R_MAX (2.0)
           │   │           ├── UV-gated firing: spike IFF amplitude≥threshold AND uv_signal>0
           │   │           ├── Stochastic jitter: 3 Knuth primes (2654435761, 2246822519, 3266489917)
           │   │           └── GpuSpikeEvent capture: 92 bytes with full metadata
           │   │
           │   ├── engine.get_accumulated_spikes()
           │   └── Spike percentile filtering (top 5% by intensity)
           │
           ├── [CLUSTERING]
           │   ├── Adaptive epsilon: 3.0 × (500/n_atoms)^(1/3), clamped [1.2, 3.0]Å
           │   ├── Adaptive min_spikes: (spikes_per_aromatic × 0.3).max(50)
           │   ├── DBSCAN clustering
           │   ├── Mega-cluster subdivision (>50% spikes):
           │   │   ├── 3Å voxel density grid
           │   │   ├── 26-neighbor local maxima detection
           │   │   └── Nearest-peak reassignment
           │   └── build_consensus_sites() across 8 streams (5.0Å matching)
           │
           ├── [ENRICHMENT]
           │   ├── enhance_sites_with_aromatics()
           │   ├── compute_lining_residues() (top 100 sites)
           │   └── Catalytic residue flagging
           │
           ├── [PRISM-THERM] (if --prism-therm)
           │   ├── SdstBridge::new(topology, protocol, spike_count)
           │   ├── bridge.ingest_all_spikes(&accumulated_spikes)
           │   │   └── sdst_insert_from_nhs_buffer() — GPU-native, zero CPU copy
           │   ├── bridge.analyze(&clustered_sites)
           │   │   ├── SDST 6-stage pipeline:
           │   │   │   S1. Hash insertion (Morton-encoded, open-address)
           │   │   │   S2. Temporal parent detection (7.5Å, 50-step window)
           │   │   │   S3. Pointer-jumping avalanche clustering (30 iter, zero atomics)
           │   │   │   S4. CCNS histogram Hill MLE (τ estimation)
           │   │   │   S5. Wavefront coherence tracking
           │   │   │   S6. TCL flags + hysteresis asymmetry
           │   │   │
           │   │   ├── Per-site analysis:
           │   │   │   ├── sdst_hysteresis_region() → asymmetry_score
           │   │   │   ├── sdst_ccns_for_regions() → τ, classification, druggability
           │   │   │   ├── sdst_tide_decomposition() → per-residue causal ΔG
           │   │   │   └── ThermClass assignment (Cryptic/Dynamic/Responsive/Inert)
           │   │   │
           │   │   └── Global tile scan:
           │   │       └── sdst_ccns_all_pockets_gpu() → 8³-voxel tiles
           │   │
           │   ├── sdst_report::build_report()
           │   ├── sdst_report::write_json() → _prism_therm.json
           │   ├── sdst_report::write_druggability_pdb()
           │   └── sdst_report::print_summary_table()
           │
           ├── [STI] Spike Thermodynamic Integration (inline in JSON build)
           │   ├── compute_binding_free_energy()
           │   │   ├── jarzynski_per_voxel() — ΔG = -kT·ln⟨exp(-W/kT)⟩
           │   │   ├── crooks_intersection() — P_F(W)/P_R(-W) = 1 at ΔG
           │   │   ├── bar_estimator() — Bennett Acceptance Ratio
           │   │   ├── channel_decomposition() — UV/LIF/EFP ΔG components
           │   │   └── arrhenius_by_wavelength() — E_a per UV wavelength
           │   └── Results embedded in binding_sites.json per site
           │
           ├── [SPIKE EVENT EXPORT]
           │   └── Per-site <name>.site<N>.spike_events.json
           │       Fields: timestep, voxel_idx, position[3], intensity,
           │               nearby_residues[8], n_residues, spike_source,
           │               wavelength_nm, aromatic_type, aromatic_residue_id,
           │               water_density, vibrational_energy, n_nearby_excited,
           │               wd_change
           │       ↑ PURPOSE: "Export spike events with enhanced metadata
           │         for pharmacophore mapping" (nhs_rt_full.rs:2804)
           │
           └── [OUTPUT]
               ├── write_binding_site_visualizations() → .pdb files
               ├── Write binding_sites.json (full results)
               └── write_ensemble_trajectory() → .ensemble_trajectory.pdb
```

### 3.2 SINGLE-STREAM PATH

Same as canonical but `--multi-stream 1` (or omitted). Runs `run_full_pipeline_internal()` instead of `run_multi_stream_pipeline()`. No consensus clustering step.

### 3.3 BATCH PATH

```bash
nhs_rt_full --manifest batch_manifest.json -o results/ --fast --hysteresis --prism-therm
```

Uses `AmberSimdBatch` for concurrent multi-structure MD on one GPU. Structures sorted into size tiers. Per-structure clustering + consensus analysis with 67% agreement threshold for ≥3 replicas.

### 3.4 PHARMACOPHORE EXTRACTION (standalone, NOT in canonical path)

```bash
pharmacophore_gpu --spike-json <site>.spike_events.json --pdb <receptor>.pdb -o <dir>
```

Consumes spike_events.json exported by canonical path. Runs `PharmacophoreGpu` → Gaussian density splatting → OpenDX grids → PyMOL script.

**Status:** Code complete. Binary compiles. NOT called from nhs_rt_full. The canonical pipeline exports all pharmacophore-ready metadata (source, wavelength, aromatic_type, water_density, vibrational_energy) in spike_events.json — this IS the pharmacophore data pipeline. The synthesis/visualization step is a downstream consumer.

### 3.5 RESONANCE SCAN (NOT FUNCTIONAL)

**Status:** File exists (`resonance_scan.rs`, 236 lines). NOT declared in `lib.rs`. NOT imported anywhere. No CLI flag. Added in commit `0607b3a9` (Feb 25, 2026) — AFTER the validated SNDC v9 run (commit `3bd5c17`). Zero integration. Would add ~10× runtime.

---

## 4. Kernel Inventory — CUDA/PTX

### 4.1 nhs_amber_fused.cu — Primary Production Kernel

**File:** `crates/prism-gpu/src/kernels/nhs_amber_fused.cu`
**Size:** 3,246 lines, 124,538 bytes
**Launch Bounds:** `__launch_bounds__(256, 4)`

#### Global Kernels (20)

| # | Kernel | Purpose | Launch Config |
|---|--------|---------|---------------|
| 1 | `nhs_amber_fused_step()` | Fused MD + spike detection | (N/256, 1, 1), 256 |
| 2 | `nhs_voxel_step_multi_lif()` | RAF multi-neuron oscillator | (32,32,64), 256, 576B shared |
| 3 | `nhs_voxel_step()` | Single-neuron LIF (legacy) | (32,32,64), 256 |
| 4 | `init_efp_state()` | Electrostatic field probe init | |
| 5 | `init_rng_states()` | PRNG initialization | |
| 6 | `init_warp_matrix()` | Warp-level matrix init | |
| 7 | `init_lif_state()` | LIF neuron state init | |
| 8 | `set_temperature_protocol()` | Protocol upload | |
| 9 | `advance_temperature_protocol()` | Step temperature | |
| 10 | `init_excited_state()` | UV excited state init | |
| 11 | `init_atom_to_aromatic()` | Atom→aromatic mapping | |
| 12 | `build_aromatic_neighbors()` | Aromatic neighbor list | |
| 13 | `compute_ring_normals()` | Aromatic ring normal vectors | |
| 14 | `build_cell_list()` | Spatial cell decomposition | |
| 15 | `reset_cell_counts()` | Zero cell counters | |
| 16 | `build_neighbor_list()` | O(N) neighbor construction | |
| 17 | `compute_nonbonded_neighborlist()` | Neighbor-list nonbonded forces | |
| 18 | `init_multimodal_detector()` | Multi-modal detector init | |
| 19 | `init_thermal_voxels()` | Thermal voxel grid init | |
| 20 | `init_multi_neuron()` | RAF neuron state init | |

#### Device Functions (21)

| Function | Purpose |
|----------|---------|
| `warp_reduce_sum/max/min()` | Warp-level reductions (shuffle) |
| `fast_rsqrt()` / `fast_rcp()` | Fast math intrinsics |
| `compute_bond_force()` | Harmonic bond: K(r-r0)² |
| `compute_angle_force()` | Harmonic angle: K(θ-θ0)² |
| `compute_dihedral_force()` | Ryckaert-Bellemans dihedral |
| `compute_nonbonded_force()` | LJ + Coulomb (inlined) |
| `langevin_thermostat()` | BAOAB O-U step |
| `shake_constraint()` | SHAKE bond constraint (10 iter) |
| `get_exclusion_radius()` | Voxel exclusion radius |
| `compute_exclusion_contribution()` | Exclusion volume fraction |
| `infer_water_density()` | Water density from exclusion |
| `lif_neuron_update()` | LIF neuron step (legacy) |
| `is_uv_absorber()` | Aromatic type classification |
| `apply_uv_burst()` | Beer-Lambert UV energy injection |
| `build_warp_entry()` | Warp-level event packing |
| `capture_spike_event()` | GpuSpikeEvent construction |
| `neuron_tau()` | τ_k = 0.5 × 2^k (RAF damping) |
| `voxel_idx_3d()` | Linear→3D voxel index |

#### #define Constants (42)

```c
// Core architecture
BLOCK_SIZE              256
WARP_SIZE               32
MAX_NEIGHBORS           128
MAX_GRID_DIM            128
PI                      3.14159265358979323846f

// Force field (AMBER ff14SB)
COULOMB_CONSTANT        332.0636f       // kcal/mol·Å/e²
FUDGE_LJ               0.5f            // 1-4 LJ scaling
FUDGE_QQ               0.8333f         // 1-4 electrostatic scaling

// NHS neuromorphic
EXCLUSION_CUTOFF        8.0f            // Å
WATER_DENSITY_BULK      0.0334f         // molecules/Å³
LIF_THRESHOLD           0.5f
LIF_RESET               0.0f
REFRACTORY_STEPS        250             // 0.1 ps at dt=0.002 ps
UV_WAVELENGTH           280.0f          // nm — LEGACY DEFAULT, NOT USED IN HOT PATH
                                        // Actual wavelength passed per-step as kernel param
                                        // uv_wavelength_nm = protocol.current_wavelength()
                                        // Cycles: [280, 274, 258, 211] nm every 300 steps

// Neighbor list (O(N) acceleration)
NB_CUTOFF               10.0f           // Å
NB_CUTOFF_SQ            100.0f
CELL_SIZE               10.0f
CELL_SIZE_INV           0.1f
MAX_CELLS_PER_DIM       32              // 32³ = 32,768 cells max
MAX_TOTAL_CELLS         32768
MAX_ATOMS_PER_CELL      128
NEIGHBOR_LIST_SIZE      256
NEIGHBOR_LIST_BUFFER    1.2f

// RAF multi-neuron oscillator
K_NEURONS               8               // Neurons per voxel
THREADS_PER_VOXEL       8
VOXELS_PER_WARP         4               // 32/8
MULTI_LIF_THRESHOLD     0.3f
COUPLING_STRENGTH       0.01f
COUPLING_DECAY          0.9f

// 3D tile + shared memory halo
TILE_X                  4
TILE_Y                  4
TILE_Z                  2
VOXELS_PER_TILE         32              // 4×4×2
HALO_X                  6               // 4+2
HALO_Y                  6               // 4+2
HALO_Z                  4               // 2+2
HALO_SIZE               144             // 6×6×4 = 576 bytes shared

// RAF oscillator dynamics
CASCADE_RATE            0.01f
JITTER_AMPLITUDE        0.02f
RAF_R_MAX               2.0f            // Saturated resonance clamp radius
RAF_PI                  3.14159265f

// Safety
MAX_VELOCITY            100.0f          // Å/ps velocity clamp
```

### 4.2 pharmacophore_splat.cu

**File:** `crates/prism-gpu/src/kernels/pharmacophore_splat.cu` (195 lines)

| Kernel | Purpose |
|--------|---------|
| `gaussian_splat()` | Thread-per-spike 3D Gaussian: `atomicAdd(grid[v], I × exp(-d²/2σ²))` |
| `gaussian_splat_typed()` | Same but filtered by type code |
| `grid_max_reduce()` | Warp-shuffle + shared mem max reduction |
| `grid_zero()` | Zero-initialize grid |

### 4.3 SDST Kernels (7 .cu files)

See [Section 5](#5-sdst-library) for complete SDST specification.

### 4.4 PTX Compilation Pipeline (build.rs)

**48 PTX modules** compiled from CUDA sources:
- All compiled with `-O3 --use_fast_math --restrict -arch=sm_120`
- Each signed with SHA-256 hash → `.ptx.sha256`
- Output staged to `OUT_DIR/ptx/` then copied to `target/ptx/`
- Runtime loads with 7-path fallback chain

---

## 5. SDST Library — Complete Specification

### 5.1 Overview

**SDST** (Spike-Driven Sparse Temporal Hash) is a GPU-native library for spatiotemporal analysis of neuromorphic spike events. It implements a complete pipeline from raw spike ingestion through criticality analysis.

**Build:** `cd crates/sdst && make shared` → `lib/libsdst.so`
**Test:** `make test` → 26/26 PASS
**Architecture:** SM_120 (Blackwell), CUDA C++17

### 5.2 Core Data Structures

#### SpikeEvent (36 bytes, aligned(4))

```c
typedef struct __attribute__((aligned(4))) {
    MortonCode  voxel;              // 21-bit Morton-encoded (x,y,z) → 4 bytes
    uint32_t    timestamp;          // Simulation step              → 4 bytes
    SpikeId     parent_spike;       // Causal parent (0=spontaneous)→ 4 bytes
    AvalancheId avalanche_id;       // Cluster membership           → 4 bytes
    WavefrontId wavefront_id;       // Coherent wavefront           → 4 bytes
    uint16_t    amplitude;          // f16 intensity                → 2 bytes
    uint16_t    local_temp;         // f16 effective temperature    → 2 bytes
    uint16_t    energy_gradient;    // f16 |∇E|                     → 2 bytes
    uint16_t    solvent_exposure;   // f16 SASA proxy               → 2 bytes
    uint16_t    wavefront_velocity; // f16 propagation speed        → 2 bytes
    uint16_t    wavefront_coherence;// f16 spatial correlation      → 2 bytes
    PhaseId     phase_id;           // Hysteresis phase 0-4         → 1 byte
    uint8_t     tcl_flags;          // Transition/boundary bits     → 1 byte
    // 2 bytes padding to 36                                        → 2 bytes
} SpikeEvent;                       //                        TOTAL = 36 bytes
```

#### SdstConfig

```c
typedef struct {
    uint32_t grid_nx, grid_ny, grid_nz;  // Default: 128³
    float    grid_spacing;               // Default: 0.75 Å
    uint32_t hash_table_capacity;        // Default: 2²² = 4,194,304
    uint32_t max_spike_events;           // Default: 2,000,000
    uint32_t max_wavefronts;             // Default: 65,536
    float    wavefront_merge_dist;       // Spatial proximity (Å)
    float    wavefront_max_dt;           // Max timestep gap
    float    avalanche_spatial_cutoff;   // Max distance (Å)
    uint32_t avalanche_max_gap;          // Max timestep gap
    uint32_t phase_boundaries[6];        // Step ranges for 5 phases
    float    ccns_soc_threshold;         // Default: 1.5
    float    ccns_barrier_threshold;     // Default: 2.0
    uint32_t num_streams;                // Default: 4
    int      device_id;                  // Default: 0
} SdstConfig;
```

### 5.3 Pipeline Stages

```
RAW NHS SPIKES (GpuSpikeEvent, 92 bytes)
     │
     ▼ sdst_insert_from_nhs_buffer() — GPU-native, zero CPU copy
     │
 ┌───┴───────────────────────────────────────────────────────────┐
 │ S1. HASH INSERTION (sdst_core.cu: kernel_hash_insert)         │
 │     Morton-encode position → Murmur3 hash → open-address      │
 │     Linear probing, per-voxel event chain (linked list)       │
 │     Temporal last-update tracking per voxel                    │
 ├───────────────────────────────────────────────────────────────┤
 │ S2. CAUSAL PARENT DETECTION (sdst_core.cu: kernel_detect_parents)
 │     Spatial cutoff: 7.5Å (10 voxels × 0.75Å spacing)         │
 │     Temporal window: 50 timesteps                              │
 │     Breadth-first chain walk (limit 512 events/voxel)         │
 │     Stale voxel skip in O(1)                                   │
 ├───────────────────────────────────────────────────────────────┤
 │ S3. AVALANCHE CLUSTERING (sdst_core.cu: pointer jumping)      │
 │     kernel_avalanche_label_init() → self-labels                │
 │     kernel_avalanche_jump() × 30 iterations                   │
 │       label[i] = label[label[i]]  (doubling, O(log D))       │
 │       ZERO atomics — pure read-write, no CAS                  │
 │     kernel_avalanche_finalize() → compress IDs                 │
 ├───────────────────────────────────────────────────────────────┤
 │ S4. CCNS ANALYSIS (sdst_ccns_gpu.cu)                          │
 │     CUB radix sort → run-length encode → avalanche sizes      │
 │     CUB segmented sort by tile_id → contiguous segments        │
 │     kernel_build_size_histogram() → [n_groups × 1024] bins    │
 │     kernel_batched_ccns_histogram() → Hill MLE per tile:       │
 │       τ = 1 + N / Σ(count[s] × ln(s / 1.5))  for s ≥ 2     │
 │     Classification: SOC (<1.5) / NearCritical (1.5-2.0) / Barrier (≥2.0)
 │     Druggability: (1 - (τ-1)/3) × confidence                  │
 ├───────────────────────────────────────────────────────────────┤
 │ S5. WAVEFRONT COHERENCE (sdst_wct.cu)                         │
 │     kernel_wavefront_propagate() — inherit/create wavefronts   │
 │     kernel_wavefront_coherence() — neighbor correlation [0,1]  │
 │     Velocity = distance / time between parent-child spikes     │
 ├───────────────────────────────────────────────────────────────┤
 │ S6. TCL FLAGS + HYSTERESIS (sdst_tcl.cu)                      │
 │     kernel_compute_tcl_flags() — per-spike 8-bit context:      │
 │       Bit 0: is_transition (±500 steps of phase boundary)      │
 │       Bit 1: is_boundary (solvent exposure > 0.3)              │
 │       Bit 2: high_gradient (|∇E| > 0.5)                       │
 │       Bit 3: cooling_spike (phase 3-4)                         │
 │       Bit 4: heating_spike (phase 0-1)                         │
 │       Bit 5: peak_temp (phase 2)                               │
 │       Bit 6: hysteresis_candidate (reserved)                   │
 │     kernel_hysteresis_scan_tiles() — 8³-voxel tile scan        │
 │     asymmetry = |heating - cooling| / total                    │
 └───────────────────────────────────────────────────────────────┘
     │
     ▼ Per-site results: τ, asymmetry, ThermClass, TIDE residues
```

### 5.4 TIDE Decomposition

**Transfer Entropy-Integrated Decomposed Energetics**

Per pocket-local residue, computes:

| Metric | Formula | Range | Meaning |
|--------|---------|-------|---------|
| Transfer Entropy (TE) | `Σ p(y_{t+1}, y_t, x_t) × log[p(y_{t+1}|y_t, x_t) / p(y_{t+1}|y_t)]` | ≥ 0 | Information flow residue→pocket |
| Fisher Information | `Var(TE across windows)` | ≥ 0 | Perturbation sensitivity |
| KL Divergence | `Σ P_heat × log(P_heat/P_cool)` | ≥ 0 | Heating/cooling asymmetry |
| Causal ΔG | `-TE × log(1 + n_causal) × RT` | [-0.30, 0.00] kcal/mol | Free energy contribution |

Where RT = 0.596 kcal/mol at 300K. Binary spike trains, Markov order k=l=1, window = 100 timesteps.

**Spatial filter (CRITICAL):** Only pocket-local events (±2 voxels margin) contribute to residue spike trains. Prevents globally-active residues from dominating every pocket.

### 5.5 ThermClass Classification

```
                    Z-score of asymmetry
                    ──────────────────────►

   INERT          RESPONSIVE      DYNAMIC         CRYPTIC
   z < -0.5       -0.5 ≤ z ≤ 0.5  0.5 < z ≤ 1.5   z > 1.5
                                                    AND (SOC OR
                                                     TIDE-enriched
                                                     OR asym > 0.4)
```

Where:
- `z = (asymmetry_score - mean_asymmetry) / std_asymmetry`
- SOC = τ > 0 AND τ < 1.5
- TIDE-enriched = top residue TE > 2× median TE across all residues

### 5.6 C API (40+ functions)

**Lifecycle:** `sdst_create`, `sdst_destroy`, `sdst_reset`, `sdst_event_count`, `sdst_memory_usage`
**Ingestion:** `sdst_insert_spikes`, `sdst_insert_raw`, `sdst_insert_from_nhs_buffer`
**Queries:** `sdst_query_region`, `sdst_query_voxel`, `sdst_query_timerange`, `sdst_query_region_timerange`
**Causal:** `sdst_causal_subgraph`, `sdst_causal_subgraph_region`, `sdst_free_subgraph`
**CCNS:** `sdst_ccns_region`, `sdst_ccns_for_regions`, `sdst_ccns_all_pockets_gpu`, `sdst_ccns_all_pockets`
**Hysteresis:** `sdst_hysteresis_region`, `sdst_hysteresis_scan`
**Wavefronts:** `sdst_wavefront_stats`, `sdst_wavefront_path`, `sdst_wavefronts_through_region`
**TIDE:** `sdst_tide_decomposition`
**DCC:** `sdst_compute_dcc`
**I/O:** `sdst_save`, `sdst_load`
**Debug:** `sdst_print_stats`, `sdst_error_string`, `sdst_validate`
**Stats:** `sdst_avalanche_stats`

### 5.7 Morton Code Operations

```c
// Encode 3D voxel → 21-bit interleaved code
morton_encode(x, y, z) = spread(x) | (spread(y) << 1) | (spread(z) << 2)

// Spread: insert 2 zero bits between each bit of 7-bit input
// Example: 0b1010101 → 0b001_000_001_000_001_000_001

// Hash: Murmur3 finalizer
sdst_hash(key) = ((key ^ key>>16) * 0x85ebca6b ^ >> 13) * 0xc2b2ae35 ^ >> 16) & (capacity-1)
```

---

## 6. Mathematical Formulations

### 6.1 Molecular Dynamics — BAOAB Langevin Integrator

**Integration scheme** (per timestep, dt = 0.002 ps):

```
Step B:  v(t + dt/2) = v(t) + (dt/2) × F(t)/m           [half-step velocity]
Step A:  x(t + dt/2) = x(t) + (dt/2) × v(t + dt/2)      [half-step position]
Step O:  v' = c₁ × v(t+dt/2) + c₂ × R                   [Ornstein-Uhlenbeck]
         c₁ = exp(-γ × dt)
         c₂ = √(kT/m × (1 - c₁²))
         R ~ N(0, 1)
Step A:  x(t + dt) = x(t+dt/2) + (dt/2) × v'            [complete position]
Step B:  v(t + dt) = v' + (dt/2) × F(t+dt)/m             [complete velocity]
```

**Two-phase friction:**
```
Equilibration: γ = 1000 ps⁻¹ (heavy damping, ~100 steps)
Production:    γ = γ_base × √(T_ref / T_current)
               γ_base = 1.0 ps⁻¹
               Cryo-scaling: γ(50K) = 1.0 × √(300/50) ≈ 2.45 ps⁻¹
```

**SHAKE constraint:** 10 iterations per step, bonds to hydrogen atoms.

**Velocity clamp:** `|v_i| ≤ 100 Å/ps` per component.

### 6.2 UV Photon Absorption — Beer-Lambert

```
E_photon = h × c / λ                     [~102 kcal/mol at 280nm]

W_UV = E_photon × QY × ε_rel × intensity
     where QY = 0.02 (quantum yield)
           ε_rel = ε(λ) / ε_max

Wavelength   Residue   ε (M⁻¹cm⁻¹)   ε_rel
─────────────────────────────────────────────
280 nm       TRP       5600           1.000
274 nm       TYR       1490           0.266
258 nm       PHE        197           0.035
211 nm       HIS        300           0.054
```

### 6.3 RAF Damped Harmonic Oscillator

**Per-neuron dynamics** (K=8 neurons per voxel):

```
τ_k = 0.5 × 2^k     for k ∈ {0, 1, ..., 7}
ω_k = 2π / τ_k      [natural angular frequency]

Driven oscillator:
  dx/dt = y
  dy/dt = -ω² × x - (1/τ) × y + stimulus

Discrete update (Euler):
  x_new = x + y × dt
  y_new = y + (-ω² × x - y/τ + stimulus) × dt

Saturated resonance clamp:
  R² = x² + y²
  if R > RAF_R_MAX (2.0):
    scale = RAF_R_MAX / √R²
    x *= scale; y *= scale

Amplitude = √(x² + y²)

Firing condition:
  spike = (amplitude ≥ effective_threshold) AND (uv_signal > 0)
```

**Threshold adaptation:**
```
effective_threshold = MULTI_LIF_THRESHOLD × (1 + jitter)

jitter = JITTER_AMPLITUDE × hash(voxel, timestep, k)
hash = 3 Knuth primes: 2654435761, 2246822519, 3266489917
```

**Spatial coupling** (shared memory halo stencil):
```
Tile: 4×4×2 voxels → Halo: 6×6×4 = 144 floats (576 bytes shared memory)

neighbor_amplitude = shared_mem[halo_x][halo_y][halo_z]
stimulus += COUPLING_STRENGTH × neighbor_amplitude × COUPLING_DECAY^distance
```

### 6.4 Spike Thermodynamic Integration (STI)

#### Jarzynski Free Energy Estimator

```
ΔG = -kT × ln( (1/N) × Σᵢ exp(-Wᵢ/kT) )

Cumulant expansion (2nd order):
ΔG ≈ ⟨W⟩ - σ²_W / (2kT)

Validity gate: σ²_W / (kT)² < 1.0 AND n ≥ 5
```

#### Channel-Specific Work

```
UV channel (source=1):
  W = (h×c/λ) × 0.02 × ε_rel(λ) × intensity

LIF channel (source=2, dewetting):
  W = intensity × 2.27 kcal/mol

EFP channel (source=3, electrostatic):
  W = (332.0636 × q₁q₂) / (ε_eff × r) × intensity
      q₁q₂ = 0.25 e², r = 4.0 Å, ε_eff = 20.0
```

#### Crooks Fluctuation Theorem

```
P_F(W) / P_R(-W) = exp((W - ΔG) / kT)

At intersection: P_F(W*) = P_R(-W*) → W* = ΔG

Implementation: Laplace-smoothed bin counts:
  p_f = (n_heat + 0.5) / (N_heat + 0.5 × n_bins)
  p_r = (n_cool + 0.5) / (N_cool + 0.5 × n_bins)
  log_ratio = ln(p_f / p_r)

  Zero-crossing via linear interpolation of log_ratio(T)
```

#### Bennett Acceptance Ratio (BAR)

```
Iterative: Σ_R f(-W_R + C) = Σ_F f(W_F - C)

  f(x) = 1 / (1 + exp(x/kT))  [Fermi function]

  C_new = kT × ln(Σ_R f(-W_R + C) / Σ_F f(W_F - C)) + kT × ln(n_F/n_R)

  Converge: |C_new - C| < 10⁻⁶ (max 100 iterations)

  ΔG_BAR = C_converged
```

#### Arrhenius Activation Energy

```
ln(rate) = ln(A) - E_a / (kT)

Linear regression: slope × kB = -E_a

Temperature interpolation:
  T(step) = T_start + progress × (T_end - T_start)
  progress = (step - ramp_start) / ramp_steps

Per-wavelength: UV spikes binned by temperature (20 bins)
```

#### Channel Decomposition

```
ΔG_total = ΔG_UV + ΔG_LIF + ΔG_EFP + ΔG_cooperative

ΔG_cooperative = ΔG_total - (ΔG_UV + ΔG_LIF + ΔG_EFP)
                 [multi-channel synergy term]

kinetic_accessibility = exp(-E_a_mean / kT)  clamped [0, 1]

effective_ΔG = ΔG_bind + kT × ln(kinetic_accessibility)
```

### 6.5 CCNS Hill MLE (Clauset-Shalizi-Newman)

```
Power-law: P(s) ∝ s^(-τ)

Hill MLE (discrete correction):
  τ = 1 + N / Σᵢ₌₁ᴺ ln(sᵢ / (s_min - 0.5))

  where s_min = 2 (minimum avalanche size)
        N = number of avalanches with size ≥ s_min

Standard error:
  SE(τ) = (τ - 1) / √N

Classification:
  τ < 1.5          → SOC (Self-Organized Criticality)
  1.5 ≤ τ < 2.0    → NearCritical
  τ ≥ 2.0          → Barrier

Druggability:
  D = (1 - (τ - 1) / 3) × confidence
  confidence = 1 - SE/τ if τ > 10⁻⁶, else 0
```

### 6.6 Transfer Entropy (TIDE)

```
TE(X→Y) = Σ p(y_{t+1}, y_t^k, x_t^l) × log[ p(y_{t+1} | y_t^k, x_t^l) / p(y_{t+1} | y_t^k) ]

Binary spike trains, Markov order k = l = 1
2×2×2 contingency table per (residue, pocket) pair
Window: TE_BIN_WIDTH = 100 timesteps, max 1024 bins

Fisher Information:
  FI = Var(TE_window)    over sliding windows of 10 bins

KL Divergence:
  KL(P_heat || P_cool) = Σ P_heat(s) × log(P_heat(s) / P_cool(s))
  Bernoulli: P(spike) per phase

Causal ΔG:
  ΔG_causal = -TE × log(1 + n_causal_spikes) × R × T
  R × T = 0.596 kcal/mol at 300K
```

### 6.7 Lorentzian Resonance Fitting (resonance_scan.rs — NOT USED)

```
A(f) = A₀ / ((f - f₀)² + Γ²)

  f₀ = resonance frequency (Hz)
  Γ  = half-width at half maximum
  A₀ = peak amplitude

Quality factor: Q = f₀ / (2Γ)

Grid search: 20 × 15 × analytical A₀ per (f₀, Γ)
R² = 1 - SS_res / SS_tot

Sweep protocol: 10 burst periods [50, 100, 150, 200, 250, 300, 400, 500, 750, 1000] timesteps
Steps per period: 10,000
```

### 6.8 Gaussian Density Splatting (pharmacophore_splat.cu)

```
G(r) = I × exp(-|r - r_spike|² / (2σ²))

Per voxel: density[v] += Σ_spikes G(r_v - r_spike)
Via atomicAdd (thread per spike, iterate neighbor voxels within 3σ)

Parameters: σ = 1.5 Å, spacing = 1.0 Å, cutoff = ceil(3σ/spacing) + 1
```

### 6.9 Pharmacophore Feature Mapping

```
Channel/Wavelength → Feature Type:

  UV 280nm (TRP)   → Aromatic / Hydrophobic
  UV 274nm (TYR)   → HBD (hydrogen bond donor, -OH)
  UV 258nm (PHE)   → Hydrophobic
  UV 211nm (HIS)   → Charge (positive, imidazole)
  UV other/BNZ     → Hydrophobic
  LIF (dewetting)  → Hydrophobic
  EFP (electro.)   → Charge (negative)

TTFS Ranking (Time-To-First-Spike):
  Groups sorted by earliest spike timestep
  Rank 1 = first feature to respond = highest binding affinity signal

Strength normalization:
  strength = mean_intensity_group / max_intensity_all_groups
```

---

## 7. Data Structures — Wire Format

### 7.1 GpuSpikeEvent (92 bytes, Rust side)

```rust
#[repr(C)]
pub struct GpuSpikeEvent {
    pub timestep: i32,              // Simulation step
    pub voxel_idx: i32,             // Linear voxel index in 128³ grid
    pub position: [f32; 3],         // Cartesian coordinates (Å)
    pub intensity: f32,             // Normalized spike intensity
    pub nearby_residues: [i32; 8],  // Closest 8 residues
    pub n_residues: i32,            // Count of nearby residues
    pub spike_source: i32,          // 1=UV, 2=LIF
    pub wavelength_nm: f32,         // UV wavelength (0 for LIF)
    pub aromatic_type: i32,         // 0=TRP, 1=TYR, 2=PHE, 3=SS, -1=none
    pub aromatic_residue_id: i32,   // Closest excited aromatic residue
    pub water_density: f32,         // Local water density (molecules/ų)
    pub vibrational_energy: f32,    // UV energy deposited (kcal/mol)
    pub n_nearby_excited: i32,      // Excited aromatics in range (π-stacking)
    pub wd_change: f32,             // |WD - WD_prev| for SDST energy_gradient
}
```

### 7.2 SpikeEvent (36 bytes, SDST C side)

See Section 5.2 above.

### 7.3 binding_sites.json Schema

```json
{
  "structure": "1btl",
  "pdb_id": "1btl",
  "n_atoms": 2039,
  "n_residues": 263,
  "simulation": {
    "steps": 55000,
    "temperature_k": 300.0,
    "cryo_temp_k": 50.0,
    "total_spikes": 295000,
    "n_streams": 8,
    "protocol": "fast_35k_hysteresis"
  },
  "sites": [
    {
      "site_id": 1,
      "centroid": [x, y, z],
      "volume_A3": float,
      "n_spikes": int,
      "mean_intensity": float,
      "max_intensity": float,
      "druggability": {
        "score": float,
        "aromatic_proximity": float,
        "n_nearby_aromatics": int,
        "aromatic_residues": ["TRP48", ...],
        "lining_residues": [
          {
            "residue_id": int,
            "name": "SER70",
            "distance_A": float,
            "is_catalytic": bool
          }
        ]
      },
      "thermodynamics": {
        "delta_g_sti_kcal_mol": float,
        "delta_g_aromatic": float,
        "delta_g_dewetting": float,
        "delta_g_electrostatic": float,
        "delta_g_cooperative": float,
        "delta_g_crooks": float | null,
        "delta_g_bar": float | null,
        "kinetic_accessibility": float,
        "effective_delta_g": float,
        "cumulant_valid": bool,
        "activation_energies": { "280nm": float, ... }
      },
      "prism_therm": {
        "therm_class": "CRYPTIC" | "DYNAMIC" | "RESPONSIVE" | "INERT",
        "asymmetry_score": float,
        "tau": float,
        "ccns_classification": "Soc" | "NearCritical" | "Barrier",
        "druggability": float,
        "tide_decomposition": [
          {
            "residue_id": int,
            "causal_dg": float,
            "transfer_entropy": float,
            "fisher_info": float,
            "kl_divergence": float
          }
        ]
      }
    }
  ],
  "all_pockets": [...],    // Global CCNS tile scan results
  "cryptic_sites": [...]   // Spike frame data for cryptic detection
}
```

### 7.4 SDST Binary Checkpoint Format

```
[Header]
  magic: 0x54534453 ("SDST")  — 4 bytes
  version: 1                   — 4 bytes
  config: SdstConfig           — sizeof(SdstConfig)

[Event Data]
  event_count: uint32          — 4 bytes
  events: SpikeEvent[]         — count × 36 bytes

[Hash Table]
  hash_table: HashEntry[]      — capacity × 8 bytes

[Chains]
  voxel_chain: uint32[]        — capacity × 4 bytes
  event_chain_next: uint32[]   — count × 4 bytes

[Avalanche]
  avalanche_parent: uint32[]   — count × 4 bytes

[Wavefront]
  wavefront_count: uint32      — 4 bytes
  wavefront_stats: WavefrontStats[] — count × sizeof

[Temporal]
  time_index_start: uint32[]   — max_ts × 4 bytes
  time_index_count: uint32[]   — max_ts × 4 bytes
```

---

## 8. Feature Status Matrix

### FULLY FUNCTIONAL — In Canonical Hot Path

| Feature | File | Invoked By | Validated |
|---------|------|------------|-----------|
| AMBER ff14SB MD | nhs_amber_fused.cu | nhs_amber_fused_step() | YES — 11/11 targets |
| BAOAB Langevin integrator | nhs_amber_fused.cu | nhs_amber_fused_step() | YES |
| SHAKE constraints (10 iter) | nhs_amber_fused.cu | shake_constraint() | YES |
| Velocity clamp (100 Å/ps) | nhs_amber_fused.cu | nhs_amber_fused_step() | YES |
| Cell list neighbor search | nhs_amber_fused.cu | build_cell_list() | YES |
| UV Beer-Lambert absorption | nhs_amber_fused.cu | apply_uv_burst() | YES |
| RAF multi-neuron oscillator (K=8) | nhs_amber_fused.cu | nhs_voxel_step_multi_lif() | YES |
| Shared memory halo stencil | nhs_amber_fused.cu | nhs_voxel_step_multi_lif() | YES |
| UV-gated spike firing | nhs_amber_fused.cu | nhs_voxel_step_multi_lif() | YES |
| Stochastic threshold jitter | nhs_amber_fused.cu | nhs_voxel_step_multi_lif() | YES |
| Saturated resonance clamp | nhs_amber_fused.cu | nhs_voxel_step_multi_lif() | YES |
| GpuSpikeEvent capture (92B) | nhs_amber_fused.cu | capture_spike_event() | YES |
| Cryo-thermal hysteresis (5-phase) | fused_engine.rs | CryoUvProtocol | YES |
| Two-phase friction (γ scaling) | fused_engine.rs | PersistentNhsEngine | YES |
| Multi-stream concurrency (8×) | nhs_rt_full.rs | run_multi_stream_pipeline() | YES |
| Adaptive epsilon clustering | nhs_rt_full.rs | multi_scale_cluster_spikes | YES |
| Mega-cluster subdivision | nhs_rt_full.rs | voxel density peak | YES |
| Consensus clustering (5.0Å) | nhs_rt_full.rs | build_consensus_sites() | YES |
| Spike percentile filtering | nhs_rt_full.rs:136 | --spike-percentile 70 (engine default) | YES |
| Aromatic proximity enrichment | aromatic_proximity.rs | enhance_sites_with_aromatics() | YES |
| Lining residue computation | nhs_rt_full.rs | compute_lining_residues() | YES |
| SDST hash insertion | sdst_core.cu | kernel_hash_insert() | YES — 26/26 tests |
| SDST parent detection | sdst_core.cu | kernel_detect_parents() | YES |
| Pointer-jumping avalanches | sdst_core.cu | kernel_avalanche_jump() | YES |
| CCNS histogram Hill MLE | sdst_ccns_gpu.cu | kernel_batched_ccns_histogram() | YES |
| Wavefront coherence tracking | sdst_wct.cu | kernel_wavefront_propagate() | YES |
| TCL flags (8-bit context) | sdst_tcl.cu | kernel_compute_tcl_flags() | YES |
| Hysteresis asymmetry scan | sdst_tcl.cu | kernel_hysteresis_scan_tiles() | YES |
| TIDE decomposition | sdst_tide.cu | sdst_tide_decomposition() | YES |
| ThermClass classification | sdst_bridge.rs | analyze() | YES |
| SDST report generation | sdst_report.rs | build_report() | YES |
| Jarzynski free energy | spike_thermodynamic_integration.rs | jarzynski_per_voxel() | YES |
| Crooks intersection | spike_thermodynamic_integration.rs | crooks_intersection() | YES |
| BAR estimator | spike_thermodynamic_integration.rs | bar_estimator() | YES |
| Channel decomposition | spike_thermodynamic_integration.rs | channel_decomposition() | YES |
| Arrhenius activation energy | spike_thermodynamic_integration.rs | arrhenius_by_wavelength() | YES |
| Spike event export (pharmacophore-ready) | nhs_rt_full.rs | Lines 2804-2888 | YES |
| PTX SHA-256 verification | build.rs / fused_engine.rs | PTX load | YES |
| Binding sites JSON output | nhs_rt_full.rs | JSON serialization | YES |
| PDB visualization output | nhs_rt_full.rs | write_binding_site_visualizations() | YES |
| Ensemble trajectory output | nhs_rt_full.rs | write_ensemble_trajectory() | YES |

### FUNCTIONAL — Not In Canonical Path

| Feature | File | Status | Notes |
|---------|------|--------|-------|
| Pharmacophore feature synthesis | spike_thermodynamic_integration.rs | CODE COMPLETE | `generate_pharmacophore_features()` exists, not called from nhs_rt_full |
| PGMG pharmacophore export | spike_thermodynamic_integration.rs | CODE COMPLETE | `generate_pgmg_pharmacophore()` exists, not called inline |
| GPU Gaussian density splatting | pharmacophore_gpu.rs + pharmacophore_splat.cu | CODE COMPLETE | Standalone binary `pharmacophore_gpu`, not called from main |
| OpenDX grid output | pharmacophore_gpu.rs | CODE COMPLETE | write_dx() functional |
| PyMOL script generation | pharmacophore_gpu.rs | CODE COMPLETE | write_pymol_script() functional |
| Perturbation response analysis | spike_thermodynamic_integration.rs | CODE COMPLETE | `analyze_perturbation_response()` exists |
| SDST binary checkpoint | sdst_serialize.cu | CODE COMPLETE | `sdst_save()` / `sdst_load()` |
| SDST causal subgraph extraction | sdst_analysis.cu | CODE COMPLETE | `sdst_causal_subgraph()` / `_region()` |
| SDST spatial/temporal queries | sdst_analysis.cu | CODE COMPLETE | 4 query functions |
| SDST DCC computation | sdst_analysis.cu | CODE COMPLETE | `sdst_compute_dcc()` |
| SDST validation | sdst_analysis.cu | CODE COMPLETE | `sdst_validate()` |
| Single-neuron LIF (legacy) | nhs_amber_fused.cu | FUNCTIONAL | `nhs_voxel_step()` — superseded by multi_lif |
| Batch mode (multi-structure) | nhs_rt_full.rs | FUNCTIONAL | `run_from_manifest()` + AmberSimdBatch |
| UltimateEngine (2-4× faster MD) | ultimate_engine.rs | FUNCTIONAL | --ultimate-mode flag |
| Parallel replica engine | parallel_replica.rs | FUNCTIONAL | --parallel flag |
| LCPO SASA computation | lcpo_sasa.rs | FUNCTIONAL | GPU-accelerated |
| Tensor core force accumulation | tensor_core_forces.rs | FUNCTIONAL | WMMA |
| Verlet neighbor list | verlet_list.rs | FUNCTIONAL | 2-3× speedup |
| PME electrostatics | pme.rs | FUNCTIONAL | Long-range via cuFFT |

### NOT FUNCTIONAL — Dead/Incomplete Code

| Feature | File | Status | Notes |
|---------|------|--------|-------|
| Resonance scan | resonance_scan.rs | NOT DECLARED IN lib.rs | Added post-validation (commit 0607b3a9). Not imported. No CLI flag. |
| OptiX RT clustering | rt_clustering.rs | DEPRECATED on SM120+ | PIPELINE_LINK_ERROR. Replaced with CPU fallback. |
| STDP learning rule | neuromorphic.rs | STRUCT EXISTS, NOT APPLIED | Fields present in DewettingNeuron but rule not called in step() |
| `--resonance-scan` CLI flag | — | DOES NOT EXIST | Never implemented |
| `--perturbation-test` CLI flag | — | DOES NOT EXIST | Never implemented |
| Inline pharmacophore generation | nhs_rt_full.rs | NOT CALLED | `generate_pharmacophore_features()` not invoked in binary |

---

## 9. Validated Benchmark Results

### SNDC v9 (commit 3bd5c17 / b7a3617)

**Canonical command** (benchmark was run under pre-lockdown flags; current canonical in docs/CANONICAL_PROVENANCE.md):
```bash
scripts/prism-validate-and-run.sh \
    -t <topo>.json -o <dir> \
    --fast --hysteresis --prism-therm \
    --multi-stream 8 --spike-percentile 70 --fused-steps 6 \
    --hmr --adaptive-dt --multi-differential \
    --closed-loop-steering --asymmetric-steering \
    --use-xgb-ranker --replica-seed 42 -v
```

| PDB | Target | DCC (Å) | Grade | Sites | Spikes |
|-----|--------|---------|-------|-------|--------|
| 1w50 | BACE1 | 3.6 | EXCELLENT | — | — |
| 1btl | TEM1 β-lactamase | 3.7 | EXCELLENT | 14 | 26M |
| 4obe | KRAS G12C (SII-P) | 3.8 | EXCELLENT | 9 | — |
| 1g1f | PTP1B | 4.8 | EXCELLENT | — | — |
| 1ade | AdSS (GDP) | 6.0 | GOOD | — | — |
| 3k5v | Abl kinase (STI) | 6.2 | GOOD | — | — |
| 1a4q | IL-2 | 6.3 | GOOD | — | — |
| 2wng | SIRPα (WYF pocket) | 7.1 | GOOD | — | — |
| 1ere | Estrogen receptor | 9.5 | MARGINAL | — | — |
| 1hhp | HIV-1 protease | 9.8 | MARGINAL | 9 | — |
| 1bj4 | FKBP12 (PLP) | 9.8 | MARGINAL | — | — |
| 2gl7 | β-catenin Site 6 | N/A | NOVEL | 26 | — |

**Accuracy:** <5Å 4/11 (36%), <8Å 8/11 (73%), <10Å 11/11 (100%)

### Performance

| Metric | Value |
|--------|-------|
| SDST pipeline (1BTL, 13.2M events) | 277 ms |
| SDST pipeline (4OBE, 13.2M events) | 532 ms |
| Total wall time per structure | ~2-5 min (55K steps × 8 streams) |
| Stochastic variation | 0.01-0.08 Å between replicates |

### PRISM-Therm Classification (1BTL)

| Class | Count |
|-------|-------|
| DYNAMIC | 6 |
| RESPONSIVE | 3 |
| INERT | 4 |
| τ range | [1.25, 1.33] |

---

## 10. Novel Algorithms — Patent Candidates

### Tier 1 — Core Innovation (Strongest Claims)

**10.1 Resonate-and-Fire (RAF) Multi-Neuron Oscillator for Molecular Sensing**
- K=8 damped harmonic oscillators per voxel with log-spaced timescales
- 3D shared-memory halo stencil for spatial coupling
- UV-gated firing condition (amplitude AND photon signal)
- Saturated resonance clamp (R ≤ R_max)
- No prior art combining neuromorphic oscillators with molecular dynamics
- **Files:** nhs_amber_fused.cu lines 2714-3216

**10.2 Cryo-Thermal Hysteresis Protocol with Neuromorphic Detection**
- 5-phase thermal cycle (50K→300K→50K) with UV spectroscopy
- Asymmetric spike response reveals conformational memory
- Two-phase friction (γ=1000→cryo-scaled) for stable cryo-dynamics
- Combined with neuromorphic sensing = novel detection methodology
- **Files:** fused_engine.rs lines 406-516

**10.3 SDST Pointer-Jumping Avalanche Clustering**
- Lock-free avalanche detection replacing atomic union-find
- 30-iteration doubling (O(log D)), zero CAS operations
- Morton-encoded spatial hash with Murmur3 probing
- 36-byte compact event structure with f16 fields
- **Files:** sdst_core.cu lines 425-472

**10.4 Histogram-Based Hill MLE for Criticality Classification**
- Replaces broken hash-dedup with exact histogram
- CUB segmented sort → run-length encode → histogram bins
- Hill MLE: τ = 1 + N / Σ(count[s] × ln(s/1.5)) (discrete Clauset correction)
- SOC/NearCritical/Barrier → druggability scoring
- **Files:** sdst_ccns_gpu.cu lines 663-738

### Tier 2 — Analytical Innovation

**10.5 TIDE: Transfer Entropy-Integrated Decomposed Energetics**
- Per-residue causal contribution to pocket dynamics
- Pocket-locality spatial filter (±2 voxels) prevents global residue dominance
- Causal ΔG: -TE × log(1 + n_causal) × RT
- Residue role classification: Trigger/Stabilizer/Gateway/Spectator
- **Files:** sdst_tide.cu, sdst_bridge.rs, sdst_report.rs

**10.6 Spike-to-Pharmacophore Feature Mapping**
- Channel/wavelength decomposition → feature type inference
- Time-To-First-Spike (TTFS) binding order ranking
- No prior art: pharmacophore features from neuromorphic spike metadata
- **Files:** spike_thermodynamic_integration.rs lines 602-741

**10.7 Multi-Estimator Free Energy Framework (STI)**
- Jarzynski + Crooks + BAR + Channel decomposition + Arrhenius
- Per-voxel free energy with cumulant validity gating
- Cooperative term quantifies multi-channel synergy
- Kinetic accessibility from wavelength-resolved activation energies
- **Files:** spike_thermodynamic_integration.rs

**10.8 ThermClass Multi-Signal Classification**
- Z-score normalized asymmetry + SOC exponent + TIDE enrichment
- 4-class system: CRYPTIC/DYNAMIC/RESPONSIVE/INERT
- Integrates CCNS criticality, hysteresis, and information flow
- **Files:** sdst_bridge.rs lines 425-455

### Tier 3 — Engineering Innovation

**10.9 Fused MD-Neuromorphic GPU Kernel**
- Single kernel launch: physics + sensing + spike capture
- AMBER ff14SB + UV absorption + RAF oscillator in one pass
- 42 compile-time constants, 20 global + 21 device kernels
- Eliminates CPU-GPU round-trip per timestep
- **Files:** nhs_amber_fused.cu (entire 3,246-line file)

**10.10 GPU-Native Spike Ingestion Pipeline**
- `sdst_insert_from_nhs_buffer()` — zero CPU copy
- 92-byte GpuSpikeEvent → 36-byte SDST SpikeEvent on device
- Morton encoding + hash insertion in single kernel pass
- **Files:** sdst_core.cu, sdst_ffi.rs

**10.11 Wavefront Coherence Tracking**
- Spike propagation velocity and spatial coherence per wavefront
- Parent-inherited wavefront IDs with merge distance threshold
- Phase-tagged wavefront statistics for hysteresis analysis
- **Files:** sdst_wct.cu

**10.12 Adaptive Density-Peak Subdivision**
- Mega-cluster detection (>50% total spikes)
- 3Å voxel density grid + 26-neighbor local maxima
- Nearest-peak reassignment for sub-cluster resolution
- Adaptive epsilon: 3.0 × (500/n_atoms)^(1/3) clamped [1.2, 3.0]Å
- **Files:** nhs_rt_full.rs (clustering section)

---

## 11. Known Limitations & Dead Code

### Exit Segfault
CUDA/OptiX teardown causes segfault (code 139/134) after all output is written. Non-critical — all results persist.

### Stochastic Variation
0.01-0.08Å between replicates. Essentially deterministic for practical purposes.

### OptiX Deprecation
SM120 (Blackwell) produces PIPELINE_LINK_ERROR with OptiX. RT clustering replaced with CPU fallback. `--rt-clustering` flag removed.

### Resonance Scan
`resonance_scan.rs` (236 lines) exists but is completely isolated. Not declared in lib.rs, not imported, no CLI integration. Would require:
1. Add `pub mod resonance_scan;` to lib.rs
2. Create CLI flag `--resonance-scan`
3. Inject sweep protocol into MD loop
4. ~10× runtime increase

### STDP Learning Rule
`DewettingNeuron` struct has threshold adaptation fields but the Spike-Timing Dependent Plasticity rule is not applied in `step()`. The RAF kernel handles threshold adaptation independently via stochastic jitter.

### Paper-Code Discrepancies
- Paper Eq. S9 says R_max = 1.0 (unit circle). Code uses `RAF_R_MAX = 2.0f`.
- Paper Section 2.2 mentions "4,000 steps" for Phase 5. Code uses 14,000 (cold_return = cold_hold).
- Paper canonical command shows `--multi-stream 4`. Validated results show `n_streams=8`.

---

*Document generated 2026-03-10. All claims source-verified against codebase at commit 5adee9cf (main branch).*

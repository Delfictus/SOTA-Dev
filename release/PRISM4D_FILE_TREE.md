# PRISM4D Complete Pipeline File Tree

## System: RTX 5080 (Blackwell sm_120), Driver 590.48, CUDA 13.1.1

---

## STAGE 1: PREP (Python → topology.json)

```
scripts/
├── prism-prep                          # Main orchestrator (862 lines)
├── multichain_preprocessor.py          # Chain routing + glycan detection
├── glycan_preprocessor.py              # Glycan handling (NAG/MAN/FUC/SIA)
├── interchain_contacts.py              # Contact analysis
├── stage1_sanitize.py                  # PDBFixer + non-standard residue map
├── stage1_sanitize_amber.py            # AMBER reduce variant
├── stage1_sanitize_hybrid.py           # Hybrid PDBFixer + AMBER
├── stage2_topology.py                  # AMBER ff14SB topology + GB radii
├── combine_chain_topologies.py         # Multi-chain merge
└── verify_topology.py                  # Validation gate (PASS/WARN/FAIL)

envs/
├── preprocessing.yml                   # openmm, pdbfixer, rdkit, dimorphite-dl
├── explicit_solvent.yml                # openmm, mdtraj, fpocket, sstmap
├── fep.yml                             # openfe, openmmtools, pymbar
├── phoregen.yml                        # pytorch, torch-geometric
├── pgmg.yml                            # pytorch, dgl
└── ensemble.yml                        # ambertools, gmx-MMPBSA
```

---

## STAGE 2: ENGINE (Rust + CUDA → binding_sites.json)

### Entry Point Binary
```
target/release/nhs_rt_full              # 64-bit ELF, linked: libc/libm/libgcc only
```

### Rust Source (26 crates, hot path highlighted)
```
crates/
├── prism-nhs/                          # ★ CRYPTIC SITE DETECTION
│   ├── Cargo.toml
│   ├── build.rs                        # Compiles spike_density.cu
│   └── src/
│       ├── bin/
│       │   └── nhs_rt_full.rs          # ★ MAIN BINARY — 6-stage pipeline
│       ├── persistent_engine.rs        # ★ GPU state + clustering API (2819 lines)
│       ├── fused_engine.rs             # ★ GPU MD simulation wrapper (4989 lines)
│       ├── rt_clustering.rs            # ★ OptiX BVH clustering (828 lines)
│       ├── ultimate_engine.rs          # ★ SM86+ hyperoptimized MD (639 lines)
│       ├── uv_bias.rs                  # ★ UV excitation physics (2445 lines)
│       ├── exclusion.rs                # ★ Hydrophobic exclusion field (552 lines)
│       ├── aromatic_proximity.rs       # ★ Druggability scoring (687 lines)
│       ├── input.rs                    # ★ Topology loading (714 lines)
│       ├── config.rs                   # Physical constants (1767 lines)
│       ├── resonance_scan.rs
│       └── spike_thermodynamic_integration.rs
│
├── prism-gpu/                          # ★ GPU KERNEL COMPILATION & EXECUTION
│   ├── Cargo.toml
│   ├── build.rs                        # ★ CUDA compilation (482 lines, nvcc → PTX)
│   └── src/
│       ├── kernels/                    # 87 CUDA sources, 92 PTX, 3 OptiX IR
│       │   ├── [see CUDA kernels below]
│       │   └── cryptic/               # NiV-Bench cryptic kernels
│       ├── amber_mega_fused.rs         # Mega-fused MD wrapper
│       ├── h_constraints.rs            # SHAKE/RATTLE wrapper
│       ├── settle.rs                   # SETTLE water constraints
│       ├── verlet_list.rs              # Neighbor list wrapper
│       ├── mega_fused.rs               # Pocket detection wrapper
│       ├── cryptic_gpu.rs              # Cryptic detection wrapper
│       ├── polycentric_immunity.rs     # Viral immunity wrapper
│       ├── floyd_warshall.rs           # APSP wrapper
│       └── lcpo_sasa.rs               # SASA wrapper
│   └── target/ptx/                     # 40 compiled PTX + SHA-256 signatures
│
├── optix-sys/                          # OptiX 9.1.0 FFI bindings
│   ├── build.rs                        # bindgen FFI generation
│   └── src/lib.rs
│
├── prism-core/                         # Core types + telemetry
├── prism-io/                           # Holographic I/O + async streaming
├── prism-physics/                      # AMBER force fields
├── prism-geometry/                     # Geometric computations
├── prism-pipeline/                     # Master orchestration
├── prism-ontology/                     # Biological ontology
├── prism-mec/                          # Molecular Evolution Computing
├── prism-whcr/                         # Wavelet Conflict Repair
├── prism-ve/                           # Viral Evolution Platform
├── prism-ve-bench/                     # VASIL Benchmark
├── prism-niv-bench/                    # NiV-Bench
├── prism-learning/                     # RL engine (DQN + SNN)
├── prism-gnn/                          # Graph Neural Network
├── prism-lbs/                          # Large-scale bioinformatics
├── prism-fluxnet/                      # Flux network analysis
├── prism-escape-extract/               # Feature extraction
├── prism-validation/                   # Multi-tier validation
├── prism-report/                       # Evidence-pack generator
├── prism/                              # TUI interface
└── prism-amber-prep/                   # AMBER toolchain integration
```

### CUDA Kernel Sources (.cu) — 87 files, ~57,000 lines
```
crates/prism-gpu/src/kernels/
│
│  ★ CORE MD ENGINE (called every timestep)
├── nhs_amber_fused.cu                  # ★★★ PRIMARY: AMBER ff14SB + NHS + LIF + UV bias (2291 lines)
├── amber_mega_fused.cu                 # Full AMBER mega-kernel (5903 lines, 220KB)
├── amber_simd_batch.cu                 # SIMD batch MD (2127 lines)
├── amber_replica_parallel.cu           # Replica-parallel MD (994 lines)
├── amber_bonded.cu                     # Bonded forces (1384 lines)
├── ultimate_md.cu                      # 14 GPU optimizations (1165 lines)
├── hyperoptimized_md.cu                # Alternative optimization (983 lines)
├── basic_langevin.cu                   # Simple Langevin thermostat (185 lines)
├── holographic_langevin.cu             # Holographic Langevin (101 lines)
│
│  ★ CONSTRAINTS & FORCES
├── h_constraints.cu                    # Analytic SHAKE/RATTLE (642 lines)
├── settle.cu                           # Rigid water SETTLE (419 lines)
├── pme.cu                              # Particle Mesh Ewald (366 lines)
├── verlet_list.cu                      # Neighbor lists, 2-3x speedup (661 lines)
├── tensor_core_forces.cu               # WMMA tensor core forces (650 lines)
├── lcpo_sasa.cu                        # SASA computation (475 lines)
│
│  ★ NEUROMORPHIC & SPIKE DETECTION
├── nhs_neuromorphic.cu                 # LIF dewetting detection (468 lines)
├── nhs_exclusion.cu                    # Hydrophobic exclusion mapping (509 lines)
├── nhs_spike_detect.cu                 # Spike detection (521 lines)
├── nhs_active_sensing_kernels.cu       # Multi-modal active sensing (618 lines)
│
│  ★ POCKET & CLUSTERING
├── mega_fused_pocket_kernel.cu         # Pocket detection mega-kernel (2405 lines)
├── pocket_detection.cu                 # Pocket identification (567 lines)
├── rt_clustering.cu                    # OptiX ray-trace clustering (372 lines)
├── rt_clustering_cuda.cu               # CUDA fallback clustering (265 lines)
├── rt_probe.cu                         # Ray-trace probe placement (374 lines)
├── rt_knn.cu                           # K-nearest neighbors (296 lines)
├── knn_cuda.cu                         # KNN CUDA (360 lines)
├── clash_detection.cu                  # Steric clash detection (365 lines)
├── distance_matrix.cu                  # All-pairs distances (119 lines)
│
│  ★ FEATURE ENGINEERING
├── bio_chemistry_features.cu           # Biochemical features (625 lines)
├── sota_features.cu                    # SOTA feature extraction (456 lines)
├── feature_merge.cu                    # Feature merging (110 lines)
├── pharmacophore_splat.cu              # Gaussian splatting (195 lines)
├── bitmask_classifier.cu               # Bitmask classification (323 lines)
│
│  ★ RESERVOIR COMPUTING & SNN
├── dendritic_reservoir.cu              # Neuromorphic reservoir (330 lines)
├── dendritic_snn_reservoir.cu          # SNN reservoir (474 lines)
├── dendritic_whcr.cu                   # Neuromorphic conflict repair (957 lines)
├── population_izhikevich.cu            # Izhikevich neuron dynamics (329 lines)
├── dqn_tensor_core.cu                  # DQN tensor core (283 lines)
│
│  ★ ADVANCED PHYSICS
├── prism_nova.cu                       # HMC + Active Inference (2248 lines)
├── active_inference.cu                 # Active inference (413 lines)
├── pimc.cu                             # Path Integral Monte Carlo (388 lines)
├── quantum.cu                          # Quantum computing (684 lines)
├── thermodynamic.cu                    # Thermodynamic phases (269 lines)
├── molecular_dynamics.cu               # MEC dynamics (607 lines)
│
│  ★ ENSEMBLE & OPTIMIZATION
├── mega_fused_batch.cu                 # Multi-structure batch (3100 lines, 139KB)
├── ensemble_warp_md.cu                 # Warp-level ensemble MD (633 lines)
├── ensemble_exchange.cu                # CMA-ES replica exchange (551 lines)
├── cma_es.cu                           # CMA-ES optimization (496 lines)
├── simple_es.cu                        # Simple evolution strategy (105 lines)
│
│  ★ VIRAL EVOLUTION & IMMUNITY
├── polycentric_immunity.cu             # Multi-epitope immunity (395 lines)
├── prism_immunity_accurate.cu          # VASIL PATH A (300 lines)
├── prism_immunity_ic50.cu              # IC50 neutralization (404 lines)
├── prism_immunity_onthefly.cu          # VASIL PATH B (190 lines)
├── epitope_p_neut.cu                   # Epitope neutralization (240 lines)
├── viral_evolution_fitness.cu          # Fitness landscape (538 lines)
├── gamma_envelope_reduction.cu         # VASIL exact metric (251 lines)
│
│  ★ SWARM & TEMPORAL
├── ve_swarm_agents.cu                  # Swarm agent dynamics (801 lines)
├── ve_swarm_dendritic_reservoir.cu     # Swarm reservoir (721 lines)
├── ve_swarm_temporal_conv.cu           # Temporal convolution (666 lines)
│
│  ★ INFORMATION THEORY
├── transfer_entropy.cu                 # KSG estimator (468 lines)
├── structural_transfer_entropy.cu      # Structure-aware TE (167 lines)
│
│  ★ FLUX & VASIL
├── mega_fused_vasil_fluxnet.cu         # VASIL + flux networks (599 lines)
├── mega_fused_vasil.cu                 # VASIL exact metric (384 lines)
├── fluxnet_reward.cu                   # Flux reward (264 lines)
│
│  ★ TOPOLOGICAL DATA ANALYSIS
├── tda.cu                              # Betti numbers/persistence (257 lines)
├── hybrid_tda_ultimate.cu              # Warp-cooperative Betti (471 lines)
│
│  ★ GRAPH ALGORITHMS
├── floyd_warshall.cu                   # All-pairs shortest paths (353 lines)
├── gnn_inference.cu                    # GNN acceleration (572 lines)
│
│  ★ GLYCAN
├── glycan_mask.cu                      # Glycan masking (173 lines)
├── glycan_masking.cu                   # Advanced glycan masking (243 lines)
│
│  ★ CONFLICT REPAIR
├── whcr.cu                             # Wavelet-Hierarchical CR (1029 lines)
├── dr_whcr_ultra.cu                    # Dimensionality-reduced WHCR (3041 lines)
│
│  ★ FUSED RAY TRACING
├── fused_lif_rt.cu                     # Fused LIF + ray trace (194 lines)
├── fused_lif_rt_cuda.cu               # CUDA variant (182 lines)
│
│  ★ MISC
├── tptp.cu                             # Temporal point tracking (750 lines)
├── stress_analysis.cu                  # (in prism-geometry)
│
│  ★ CRYPTIC SUBDIRECTORY
├── cryptic/
│   ├── cryptic_hessian.cu              # NiV-Bench Stage 12a (228 lines)
│   ├── cryptic_eigenmodes.cu           # NiV-Bench Stage 12b (325 lines)
│   ├── cryptic_probe_score.cu          # NiV-Bench Stage 12c (422 lines)
│   └── cryptic_signal_fusion.cu        # NiV-Bench Stage 12d (406 lines)
│
│  ★ LBS KERNELS (in prism-lbs)
├── lbs/pocket_clustering.cu
├── lbs/distance_matrix.cu
├── lbs/surface_accessibility.cu
└── lbs/druggability_scoring.cu
```

### Compiled PTX (40 in target/ptx, 92 in src/kernels)
```
crates/prism-gpu/target/ptx/
├── active_inference.ptx + .sha256
├── amber_bonded.ptx + .sha256
├── amber_mega_fused.ptx + .sha256      # ★ 220KB
├── amber_replica_parallel.ptx
├── amber_simd_batch.ptx + .sha256
├── cma_es.ptx + .sha256
├── cryptic_eigenmodes.ptx + .sha256
├── cryptic_hessian.ptx + .sha256
├── cryptic_probe_score.ptx + .sha256
├── cryptic_signal_fusion.ptx + .sha256
├── dendritic_reservoir.ptx + .sha256
├── dendritic_snn_reservoir.ptx + .sha256
├── dendritic_whcr.ptx + .sha256
├── ensemble_exchange.ptx + .sha256
├── epitope_p_neut.ptx + .sha256
├── feature_merge.ptx + .sha256
├── floyd_warshall.ptx + .sha256
├── gamma_envelope_reduction.ptx + .sha256
├── gnn_inference.ptx + .sha256
├── h_constraints.ptx + .sha256
├── hyperoptimized_md.ptx + .sha256
├── lcpo_sasa.ptx + .sha256
├── molecular_dynamics.ptx + .sha256
├── nhs_amber_fused.ptx + .sha256      # ★ Critical production kernel
├── nhs_exclusion.ptx + .sha256
├── nhs_neuromorphic.ptx + .sha256
├── pharmacophore_splat.ptx
├── pimc.ptx + .sha256
├── pme.ptx + .sha256
├── prism_immunity_onthefly.ptx + .sha256
├── prism_nova.ptx + .sha256
├── quantum.ptx + .sha256
├── settle.ptx + .sha256
├── tda.ptx + .sha256
├── tensor_core_forces.ptx + .sha256
├── thermodynamic.ptx + .sha256
├── transfer_entropy.ptx + .sha256
├── ultimate_md.ptx + .sha256
├── verlet_list.ptx + .sha256
└── whcr.ptx + .sha256
```

### OptiX IR (3 files, binary)
```
crates/prism-gpu/src/kernels/
├── rt_probe.optixir                    # Ray-trace probe placement (17KB)
├── rt_clustering.optixir               # Ray-trace clustering (19KB)
└── fused_lif_rt.optixir                # Fused LIF + RT (7.7KB)
```

---

## STAGE 3: POST-PROCESSING (Python)

```
scripts/
├── rerank_sites.py                     # vol*(1-0.7*hydro) re-ranking
└── aggregate_batch.sh                  # Batch result collection

benchmarks/cryptobench/
├── 04_evaluate_results.py              # DCA/DCC/F1 metrics
├── 05_comprehensive_analysis.py        # 7-suite analysis
├── ground_truth/*.ground_truth.json    # 222 targets
├── batches/run100_batch_*.sh           # Batch run scripts
└── batches/batch_*.sh                  # Per-target scripts
```

---

## STAGE 4: ALL OTHER RELEASE BINARIES

```
target/release/
├── nhs_rt_full                 # ★ Primary pipeline binary
├── ccns-analyze                # CCNS metrics (tau, power-law)
├── nhs-analyze                 # Spike analysis
├── nhs-analyze-gpu             # GPU-accelerated analysis
├── nhs-analyze-pro             # Pro analysis
├── nhs-analyze-turbo           # Turbo analysis
├── nhs-analyze-ultra           # Ultra analysis
├── nhs-batch                   # Batch orchestrator
├── nhs-cryo-probe              # Cryo-UV probe
├── nhs-detect                  # Detection-only
├── nhs-diagnose                # Diagnostic tool
├── nhs_stage1b                 # Stage 1b processing
├── pharmacophore_gpu           # GPU pharmacophore
├── rt_probe_validate           # RT probe validation
├── stage2b-process             # Stage 2b processing
├── stress-rt-clustering        # RT clustering stress test
└── test-rt-clustering          # RT clustering test
```

---

## RUNTIME DEPENDENCIES

```
REQUIRED:
  - NVIDIA RTX GPU (Turing sm_75+, Blackwell sm_120 optimal)
  - NVIDIA Driver 590+ (for OptiX 9.1.0)
  - CUDA Runtime 12.6+ (libcuda.so, libcudart.so)
  - Linux x86_64, glibc 2.17+

FOR PREP STAGE:
  - Python 3.10-3.13
  - Conda environment: envs/preprocessing.yml
    - openmm >=8.0
    - pdbfixer >=1.9
    - rdkit >=2024.03
    - numpy >=1.24
    - dimorphite-dl >=2.0

FOR BUILD FROM SOURCE:
  - Rust 1.75+ (2021 edition)
  - CUDA Toolkit 12.6 (nvcc)
  - OptiX SDK 9.1.0 (headers)
  - bindgen 0.70 (for OptiX FFI)
```

---

## KEY PTX LOADING PATHS (binary must find these at runtime)

```rust
// Primary pattern — concat!() macro (compile-time path)
concat!(env!("CARGO_MANIFEST_DIR"), "/target/ptx/<kernel>.ptx")
// Resolves to: crates/prism-gpu/target/ptx/<kernel>.ptx

// Dynamic loading pattern
let ptx_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/ptx");
fs::read_to_string(ptx_dir.join("<kernel>.ptx"))

// OptiX IR loading
let optixir_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("src/kernels/<kernel>.optixir");
```

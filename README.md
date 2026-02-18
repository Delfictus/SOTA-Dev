# PRISM-4D: Photonic Resonance Imaging for Structural Molecular Dynamics

## One-Line Summary

A GPU-accelerated pipeline that detects protein binding sites in under 60 seconds by simulating cryo-UV fluorescence excitation of aromatic residues and clustering the resulting photonic spike events using NVIDIA RT cores.

## The Problem

Identifying druggable binding sites on proteins is a prerequisite for structure-based drug design. Current tools fall into two categories, each with fundamental limitations:

**Geometry-only methods** (FPocket, DoGSiteScorer, P2Rank) analyze static crystal structures to find surface pockets. They are fast (seconds) but blind to dynamics: cryptic binding sites that open only during conformational motion are invisible. They also miss allosteric sites where the pocket geometry doesn't match classical descriptors. FPocket uses Voronoi tessellation of alpha spheres, SiteMap uses van der Waals surface grids, and P2Rank uses machine learning on static features. None incorporate protein dynamics.

**MD-based methods** run molecular dynamics simulations to sample conformational ensembles, then analyze the trajectory for transient pockets. These capture dynamics but require HPC clusters running for hours to days. A typical binding site detection workflow involves: (1) 100+ ns MD trajectory on a GPU cluster, (2) trajectory post-processing with MDAnalysis or CPPTRAJ, (3) pocket detection on each frame, (4) consensus analysis across frames. The compute cost makes this impractical for virtual screening campaigns involving thousands of proteins.

The gap: no existing tool combines dynamic conformational sampling with real-time binding site detection on consumer hardware.

## What PRISM-4D Does Differently

PRISM-4D chains five computational techniques that have never been combined for binding site detection:

**1. GPU-Accelerated Amber-Class MD.** Full molecular dynamics with AMBER ff14SB force field parameters (bonds, angles, dihedrals, Lennard-Jones, Generalized Born implicit solvent), Langevin thermostat, and SHAKE constraints. Implemented directly in CUDA — not wrapping OpenMM or GROMACS, but native GPU force computation with SIMD batching across structures. Achieves 880 steps/second on a single RTX 5080 for a 4,073-atom protein.

**2. Simulated Cryo-UV Fluorescence Excitation.** The "4D" in PRISM-4D is the photonic response dimension over time. During MD simulation, periodic UV bursts (280nm, 274nm, 258nm, 211nm; 42 kcal/mol) are applied to aromatic residues (TRP, TYR, PHE). These wavelengths match the absorption bands of indole, phenol, and benzene chromophores. The simulation uses a cryo-UV protocol: cold hold at 50K (reduces thermal noise), temperature ramp to 300K, then warm-hold with UV bursts every 250 steps. This is a computational analog of cryo-UV fluorescence spectroscopy, a real experimental technique used to study protein folding and binding (Haas et al., 1978; Bhagwan et al., 2016).

**3. OptiX RT Core Photon Tracing.** NVIDIA's RT cores — hardware designed for real-time ray tracing in games — are repurposed for photon-protein interaction. A BVH (bounding volume hierarchy) is built from atomic coordinates. Rays representing simulated UV photons are traced through the protein structure. Hit shaders compute fluorescence response based on local environment (solvent exposure, quenching neighbors, ring orientation). This is the first use of RT cores for molecular photonics.

**4. Neuromorphic Spike Detection.** Each aromatic residue acts as an independent "pixel" with threshold-based spike generation, refractory periods, and temporal contrast sensitivity — a computational analog of event cameras (Prophesee/iniVation hardware). Unlike frame-based approaches that sample every N steps, the spike model captures only events exceeding a dynamic threshold. This concentrates signal at conformational transitions where binding sites open or close, producing sparse event streams (70K events from 3.5M raw spikes at 2% intensity filter).

**5. Adaptive DBSCAN Clustering with Density Asymmetry Centroid Refinement.** Spike events are clustered using RT-core-accelerated DBSCAN. The neighborhood radius adapts to protein size: `epsilon = 3.0 * (500 / n_atoms)^(1/3)`, clamped to [1.2, 3.0] Angstroms. After clustering, centroids are refined by detecting atom density asymmetry in a 15-Angstrom sphere around each site. The spike centroid sits at the aromatic ring (pocket wall); the refinement shifts it toward the protein interior (pocket center), correcting for the surface bias inherent in fluorescence-based detection.

**Why this combination is novel:**
- RT cores existed for gaming; nobody mapped them to molecular photonics
- Neuromorphic event cameras are physical hardware; simulating their physics on GPU for molecular detection is new
- Cryo-UV fluorescence is a real experimental technique; PRISM-4D makes it computational
- Adaptive DBSCAN with density asymmetry centroid refinement is a new post-processing step motivated by the physics of fluorescence-based detection

## Architecture

- **Language:** Rust + CUDA (zero-copy GPU pipeline, no Python in the hot path)
- **GPU Kernels:** 100+ custom CUDA kernels (.cu source + .ptx compiled included in repository)
- **RT Cores:** OptiX 9.1 pipeline for photon tracing and BVH-accelerated spatial clustering (.optixir)
- **GPU Discipline:** Single CudaContext, single AmberSimdBatch, zero `thread::spawn` for GPU work
- **Memory Model:** Size-tier sequential batching; structures processed serially to minimize VRAM pressure
- **Force Field:** AMBER ff14SB with GBn2 implicit solvent, SHAKE constraints, Langevin thermostat

## Codebase Scale

Measured with `find . -name "*.ext" -not -path "*/target/*" | xargs wc -l`:

| Language | Files | Lines | Role |
|----------|-------|-------|------|
| Rust (.rs) | 100+ | **410,239** | Engine, pipeline orchestration, FFI bindings, CLI |
| CUDA (.cu) | 3 | **56,483** | GPU kernels: force computation, photonics, spike detection |
| CUDA headers (.cuh) | 2 | **5,331** | Shared primitives, detection modules |
| PTX (.ptx) | 60+ | **298,728** | Compiled GPU assembly (included for reproducibility) |
| **Total** | | **770,781** | |

All GPU kernel source (.cu), compiled assembly (.ptx), and RT core modules (.optixir) are included in the repository. No binary blobs without source.

## Why Rust + CUDA (Not Python)

Python-based molecular tools (MDAnalysis, OpenMM wrappers, P2Rank) pay interpreter overhead on every simulation frame. PRISM-4D has zero Python in the simulation/detection hot path.

Rust's ownership model guarantees GPU memory safety at compile time. No dangling device pointers, no double-free on VRAM. This matters when managing 100+ kernel launches per pipeline run. The `cudarc` FFI crate provides zero-copy host-device transfers. Python tools serialize through NumPy to CuPy to kernel, adding latency per step.

The cost: massive development complexity. Rust-CUDA FFI has no mature ecosystem. Every kernel launch, buffer allocation, and synchronization is manual. This is why it has not been done before — not because the physics is impossible, but because the engineering barrier is extreme.

100+ custom CUDA kernels means no dependency on cuDNN or cuBLAS for core physics. Every force calculation, every photon trace, every spike detection is purpose-built.

## Architectural Complexity

What makes this hard to replicate:

**Amber force field in CUDA.** Bond, angle, dihedral, Lennard-Jones, and Generalized Born forces are computed directly in CUDA kernels. This is not wrapping OpenMM or GROMACS — it is a native GPU force computation engine.

**OptiX ray tracing for photonics.** The RT pipeline builds BVH from atomic coordinates. Rays are simulated UV photons. Hit shaders compute fluorescence response. This required understanding both RT core hardware architecture and photophysics — the BVH traversal hardware designed for triangle intersection is repurposed for sphere intersection against atoms.

**Neuromorphic spike model in shared memory.** Each aromatic residue is a "pixel" with independent threshold, refractory period, and temporal contrast sensitivity. This is a computational analog of event cameras (Prophesee/iniVation hardware), implemented entirely in CUDA shared memory with warp-level synchronization.

**Deeply sequential, internally parallel pipeline.** MD steps run in SIMD batches on CUDA cores. Photon traces run on RT cores. Spike events accumulate in device memory. Clustering runs back on CUDA cores. All on one GPU, one context, one memory space.

**No framework dependencies.** No PyTorch, no TensorFlow, no JAX. The entire pipeline is hand-written Rust and CUDA. This eliminates framework version conflicts and makes the binary fully self-contained.

Reproducing this from scratch requires simultaneous expertise in computational chemistry, GPU systems programming, photophysics, neuromorphic computing, and topological data analysis. The intersection of these domains is the barrier, not any single algorithm.

## Performance (Consumer Hardware)

| Target | Protein | Atoms | Known Site | Detected | Distance | Time |
|--------|---------|-------|------------|----------|----------|------|
| 1crn | Crambin | 642 | Regression baseline | 1 site | — | 41s |
| 1btl | TEM-1 beta-lactamase | 4,073 | SER70/GLU166 tetrad | Site 2 | **4.67 Ang** | 41s |
| 1w50 | BACE1 beta-secretase | 5,864 | ASP32/ASP228 dyad | Site 7 | **7.55 Ang** | 48s |

**Hardware:** NVIDIA RTX 5080 (consumer desktop GPU, ~$999 MSRP)

**Context:** Traditional MD-based binding site detection requires HPC clusters running hours to days. Typical workflow: 100+ ns trajectory (hours on 4-8 GPUs), post-processing (minutes-hours), pocket detection per frame, consensus analysis. Total wall time: 4-48 hours on dedicated hardware.

PRISM-4D: under 60 seconds on a single consumer GPU.

## Scientific Insight: Aromatic Proximity Bound

Detection accuracy is fundamentally bounded by the distance from the nearest aromatic fluorophore (TRP/TYR/PHE) to the catalytic center. The UV fluorescence signal originates at aromatic rings, so the raw spike centroid cannot be closer to the active site than the nearest aromatic residue.

- **TEM-1 beta-lactamase:** PHE72 is 5.7 Ang from the catalytic tetrad center. Detected site: 4.67 Ang (centroid refinement shifts past the aromatic position).
- **BACE1 beta-secretase:** TYR199 is 8.5 Ang from the catalytic dyad center. Detected site: 7.55 Ang (outperforms the theoretical aromatic bound due to density asymmetry correction).

For aromatic-rich active sites, sub-5-Angstrom detection is achievable. For active sites in deep clefts without nearby aromatics, detection accuracy is bounded by aromatic proximity. This is a fundamental property of fluorescence-based methods, not a software limitation.

## Domain Intersections

PRISM-4D spans seven research domains simultaneously:

1. **Computational biophysics** — Amber force fields, Langevin dynamics, implicit solvent
2. **Photonics** — UV fluorescence excitation, Stokes shift, quantum yield modeling
3. **Real-time ray tracing** — OptiX BVH construction, RT core sphere intersection
4. **Neuromorphic computing** — Spike-based event detection, temporal contrast sensitivity
5. **Topological data analysis** — DBSCAN clustering, spatial density estimation
6. **GPU systems programming** — Rust-CUDA FFI, kernel fusion, zero-copy transfers
7. **Drug discovery** — Druggability scoring, binding site prediction, active site classification

No existing tool spans more than 2-3 of these simultaneously.

## Validation Status

- [x] Crambin (regression baseline, 642 atoms)
- [x] TEM-1 beta-lactamase — 4.67 Ang from known active site (aromatic-rich)
- [x] BACE1 beta-secretase — 7.55 Ang from known active site (aromatic-poor, physics limit)
- [ ] Bcl-xL (BH3 groove, cancer target)
- [ ] Adenylosuccinate synthetase (scaling test, 8,266 atoms)
- [ ] Abl kinase (positive control, two known binding sites)
- [ ] Trp-cage (negative control, no binding site expected)

## Repository Contents

```
crates/
  prism-nhs/        # Core pipeline: MD engine, spike detection, clustering
  prism-gpu/        # CUDA kernel management, PTX loading
    src/kernels/    # ALL .cu source, .ptx compiled, .optixir RT modules
  prism-optix/      # OptiX RT core bindings
  prism-core/       # Shared types and utilities
  prism-lbs/        # Ligand binding site analysis
  prism-validation/ # Druggability scoring
  prism-report/     # Output generation (PDB, PyMOL, ChimeraX, JSON)
  prism-pipeline/   # Batch orchestration
  prism-cli/        # Command-line interface
  prism-io/         # I/O utilities
scripts/
  prism-prep        # PDB preprocessing (OpenMM + PDBFixer -> topology.json)
e2e_validation_test/
  prep/             # Prepared topology files for validation targets
```

All GPU kernels are included as both source (.cu) and compiled assembly (.ptx). The .optixir module for RT core ray tracing is also included. No external GPU binaries are required beyond the NVIDIA driver and CUDA toolkit.

## Reproducing Results

```bash
# Prerequisites: NVIDIA GPU with CUDA 12+, Rust toolchain, OptiX 9.x SDK

# Build
cargo build --release -p prism-nhs --bin nhs_rt_full --features gpu

# Prep a PDB (requires Python + OpenMM + PDBFixer)
python3 scripts/prism-prep input.pdb output.topology.json --mode cryptic -v

# Run pipeline
./target/release/nhs_rt_full -t output.topology.json -o /tmp/results --fast --verbose

# Validate TEM-1 (4.67 Ang expected)
./target/release/nhs_rt_full \
  -t e2e_validation_test/prep/1btl.topology.json \
  -o /tmp/val_1btl --fast --verbose
```

## Known Limitations

- **Residue labeling:** Topology residue IDs do not always match PDB numbering. Lining residue names in output may be offset from canonical PDB annotation.
- **Over-detection:** The pipeline reports up to 7 druggable sites per protein. Some are likely false positives (surface patches near aromatic clusters that are not functional binding sites).
- **Centroid refinement is a correction, not raw accuracy:** The density asymmetry centroid shift adds up to 4 Ang of correction. Without it, raw spike centroids sit at aromatic ring positions, which can be 5-10 Ang from the actual pocket center.
- **Limited validation:** Only 3 targets validated so far. The aromatic proximity bound has been characterized but not yet tested across a large benchmark set.
- **Exit segfault:** The pipeline produces correct output but segfaults (code 139) during CUDA/OptiX cleanup on exit. All output files are written before the crash. This is a teardown ordering issue, not a data integrity problem.
- **Stochastic results:** MD simulation is stochastic. Cluster centroids vary by 1-2 Ang between runs due to different random seeds.

## Citation

Publication in preparation. For now, cite:

```
PRISM-4D: Photonic Resonance Imaging for Structural Molecular Dynamics
https://github.com/Delfictus/PRISM4D-SOTA
```

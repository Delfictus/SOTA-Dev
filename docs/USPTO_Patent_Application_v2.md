# UNITED STATES PATENT APPLICATION

## Title

SYSTEMS AND METHODS FOR GPU-ACCELERATED DETECTION OF TRANSIENT AND CRYPTIC MOLECULAR BINDING SITES USING FUSED MOLECULAR DYNAMICS SIMULATION AND NEUROMORPHIC SPIKE-DRIVEN SENSING

**Inventor:** Ididia J. Serfaty (Los Angeles, CA)
**Assignee:** Delfictus IO LLC
**Filing Type:** Non-Provisional Utility Patent Application

---

## CROSS-REFERENCE TO RELATED APPLICATIONS

This application claims priority to [U.S. Provisional Application No. XX/XXX,XXX, filed XXXX-XX-XX], the entire disclosure of which is incorporated herein by reference.

---

## STATEMENT REGARDING FEDERALLY SPONSORED RESEARCH OR DEVELOPMENT

Not Applicable.

---

## INCORPORATION BY REFERENCE

The following references are incorporated herein by reference in their entirety for all purposes:

- Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). "Power-law distributions in empirical data." SIAM Review, 51(4), 661-703. (Hill MLE methodology for discrete power-law fitting)
- Leimkuhler, B., & Matthews, C. (2013). "Rational construction of stochastic numerical methods for molecular sampling." Applied Mathematics Research eXpress, 2013(1), 34-56. (BAOAB Langevin integrator)
- Jarzynski, C. (1997). "Nonequilibrium equality for free energy differences." Physical Review Letters, 78(14), 2690. (Non-equilibrium work-free energy relationship)
- Crooks, G. E. (1999). "Entropy production fluctuation theorem and the nonequilibrium work relation for free energy differences." Physical Review E, 60(3), 2721. (Forward-reverse work distribution intersection)
- Bennett, C. H. (1976). "Efficient estimation of free energy differences from Monte Carlo data." Journal of Computational Physics, 22(2), 245-268. (Bennett Acceptance Ratio)
- Izhikevich, E. M. (2001). "Resonate-and-fire neurons." Neural Networks, 14(6-7), 883-894. (RAF neuron model)
- Schreiber, T. (2000). "Measuring information transfer." Physical Review Letters, 85(2), 461-464. (Transfer entropy)

---

## FIELD OF THE DISCLOSURE

The present disclosure relates to computational structural biology, molecular dynamics simulation, neuromorphic computing, and computer-aided drug design. More particularly, the disclosure relates to GPU-accelerated systems and methods for identifying transient, allosteric, and cryptic binding sites in macromolecular structures by fusing atomistic molecular dynamics simulation with neuromorphic event-driven sensing, spike-driven spatiotemporal analysis, and non-equilibrium thermodynamic estimation within a unified hardware-software architecture.

---

## DEFINITIONS

As used herein, the following terms have the indicated meanings unless context clearly dictates otherwise:

**"Cryptic binding site"** refers to a molecular binding pocket on a macromolecular target that is not geometrically accessible in a static crystal structure or cryo-EM structure but becomes transiently accessible during conformational fluctuations, thermal perturbation, or ligand-induced fit. Cryptic sites are invisible to static pocket detection algorithms such as FPocket or SiteMap.

**"Spike event"** refers to a discrete, timestamped detection event emitted by a neuromorphic oscillator at a defined spatial location when a local molecular state transition exceeds a detection threshold. Each spike event is characterized by at least a spatial coordinate, a temporal index, an intensity value, and optionally, metadata including a source channel identifier, a spectroscopic wavelength, and a local solvent density measurement.

**"Avalanche"** refers to a spatiotemporally connected cluster of spike events in which each member event has a causal relationship (spatial proximity and temporal adjacency) to at least one other member event. Avalanche size distributions characterize the criticality regime of the underlying molecular dynamics.

**"Self-organized criticality (SOC)"** refers to a dynamical regime in which a system naturally evolves to a critical state characterized by scale-free (power-law) distributions of event sizes. In the context of this disclosure, SOC behavior in spike avalanche distributions indicates a molecular binding site that is poised between open and closed conformations — the hallmark of a druggable cryptic pocket.

**"Druggability"** refers to a quantitative score in the range [0, 1] estimating the likelihood that a detected binding site can productively bind a small-molecule drug candidate. The druggability score is derived from criticality analysis, thermodynamic classification, and structural features of the site.

**"Resonate-and-fire (RAF) oscillator"** refers to a neuromorphic computational unit implementing a damped harmonic oscillator that responds preferentially to inputs near its natural resonant frequency, emitting a spike event when accumulated oscillatory amplitude exceeds a threshold. Unlike integrate-and-fire (LIF) neurons that accumulate charge linearly, RAF oscillators exhibit frequency selectivity and temporal memory.

**"Fused kernel"** refers to a GPU computational kernel in which multiple algorithmic stages — including but not limited to molecular dynamics integration, force computation, perturbation injection, neuromorphic state updates, and spike event capture — are executed within a single kernel launch without returning control to the host CPU between stages.

**"Hysteresis protocol"** refers to a thermal perturbation schedule comprising at least a forward temperature sweep and a reverse temperature sweep, enabling detection of conformational memory effects that distinguish cryptic binding sites from thermal noise artifacts.

**"Transfer entropy"** refers to a model-free, information-theoretic measure of directed information flow from one time series to another. In the context of this disclosure, transfer entropy quantifies the causal influence of individual amino acid residues on the spike dynamics of a detected binding pocket.

**"Pointer jumping"** refers to a parallel graph algorithm for computing connected components (transitive closure) by iteratively replacing each node's parent label with its grandparent's label, achieving convergence in O(log D) iterations where D is the maximum tree depth, without requiring atomic compare-and-swap operations.

---

## BACKGROUND OF THE DISCLOSURE

### The Cryptic Binding Site Problem

A significant fraction of therapeutically relevant protein targets harbor binding pockets that are not visible in experimentally determined static structures. These "cryptic" binding sites become accessible only during transient conformational fluctuations, thermal excitation, or interactions with solvent molecules. Notable examples include the Switch-II pocket in KRAS G12C (targeted by sotorasib), the allosteric pocket in Abl kinase (targeted by imatinib), and protein-protein interaction interfaces such as IL-2 and SIRPalpha. Discovery of cryptic sites has historically been serendipitous rather than systematic.

### Limitations of Prior Art

Existing computational methods for cryptic binding site detection fall into three categories, each with fundamental limitations:

**Static geometry methods** (FPocket, SiteMap, DoGSiteScorer) analyze a single molecular structure to identify surface cavities using Voronoi tessellation, alpha spheres, or grid-based probes. These methods are inherently incapable of detecting cryptic sites because the binding pocket is geometrically absent in the input structure.

**Conventional molecular dynamics methods** (MDpocket, CryptoSite, HTMD) run explicit-solvent MD simulations (typically 100 ns to 1 microsecond) and analyze trajectories post hoc for transient pocket openings. These methods suffer from: (a) prohibitive computational cost for systematic screening (days to weeks per target on GPU clusters); (b) reliance on post hoc geometric analysis that discards the temporal dynamics of pocket formation; and (c) the inability to determine thermodynamic druggability from simulation data alone.

**Machine learning methods** (CryptoSite, PocketMiner, DeepSite) train classifiers on known cryptic sites but are limited to the training distribution and cannot identify genuinely novel pocket types. They provide no physical or thermodynamic basis for their predictions.

**Neuromorphic computing methods** have been applied to event-driven image processing and robotics but have not been applied to molecular dynamics or structural biology. No prior art combines neuromorphic spike-driven sensing with atomistic molecular simulation.

None of the existing methods achieves the combination of: (1) sub-minute per-target runtime on a single consumer GPU; (2) direct detection of transient pocket opening events as they occur; (3) quantitative thermodynamic characterization of each detected site; (4) causal decomposition of individual residue contributions to pocket dynamics; and (5) simultaneous pharmacophore feature mapping from the detection signal itself.

### Objects of the Invention

It is therefore an object of the present disclosure to provide a computational system and method that detects cryptic binding sites by directly sensing molecular state transitions during simulation, rather than analyzing trajectories post hoc.

It is a further object to provide a fused GPU kernel architecture that eliminates CPU-GPU synchronization overhead by performing molecular dynamics integration and neuromorphic event detection within a single kernel launch.

It is a further object to provide a lock-free, CAS-free parallel clustering algorithm suitable for processing millions of spike events per second on GPU hardware.

It is a further object to provide a multi-estimator thermodynamic framework that quantifies binding free energy, channel-specific contributions, and kinetic accessibility directly from spike event ensembles.

It is a further object to provide a causal decomposition method that identifies individual residues responsible for pocket opening dynamics, enabling rational drug design targeting the mechanism of pocket formation.

---

## BRIEF DESCRIPTION OF THE DRAWINGS

The accompanying drawings, which are incorporated in and constitute a part of this specification, illustrate various embodiments of the invention and, together with the description, serve to explain the principles of the invention.

**FIG. 1** is a system-level architecture diagram illustrating the complete pipeline from molecular topology input through fused MD-neuromorphic kernel execution, spike-driven spatiotemporal analysis, thermodynamic estimation, and structured output generation.

**FIG. 2** is a flowchart detailing the 5-phase cryo-thermal hysteresis protocol, showing temperature profiles, phase boundaries, UV wavelength cycling, and friction coefficient scaling across all phases.

**FIG. 3A** is a schematic diagram of the voxel-local K=8 RAF multi-neuron oscillator architecture showing log-spaced timescales, stimulus input, amplitude computation, and spike detection logic.

**FIG. 3B** is a schematic diagram of the shared-memory halo stencil showing the 4x4x2 tile-to-6x6x4 halo mapping, neighbor coupling data flow, and GPU shared memory allocation.

**FIG. 4** is a detailed flowchart of the fused kernel execution showing the interleaved BAOAB integration, UV perturbation, neuromorphic update, and spike capture stages within a single kernel launch.

**FIG. 5** is a multi-stage pipeline diagram showing the SDST spatial hash insertion, causal parent detection, pointer-jumping avalanche clustering, histogram-based Hill MLE, wavefront coherence tracking, and thermodynamic context labeling stages.

**FIG. 6** is a decision flowchart showing the ThermClass multi-signal classification logic using z-score normalization, SOC exponent, TIDE enrichment, and hysteresis asymmetry to assign CRYPTIC, DYNAMIC, RESPONSIVE, or INERT classifications.

**FIG. 7** is a data flow diagram showing the spike thermodynamic integration (STI) multi-estimator framework including Jarzynski, Crooks intersection, Bennett Acceptance Ratio, channel decomposition, and Arrhenius activation energy estimation.

**FIG. 8** is a schematic showing the pharmacophore feature mapping from neuromorphic spike metadata, including channel/wavelength to feature type inference and Time-To-First-Spike (TTFS) ranking.

**FIG. 9** illustrates an exemplary JSON output schema showing per-site geometry, thermodynamic metrics, criticality classification, TIDE residue decomposition, and pharmacophore features for downstream drug design integration.

**FIG. 10** is a block diagram showing the multi-stream concurrent execution architecture with independent CUDA streams and consensus site merging.

---

## DETAILED DESCRIPTION OF PREFERRED EMBODIMENTS

The following detailed description is presented to enable any person skilled in the art to make and use the disclosed invention. Various modifications to the illustrated embodiments will be apparent to those skilled in the art, and the general principles herein may be applied to other embodiments and applications without departing from the spirit and scope of the present disclosure.

### I. System Architecture and Hardware Configuration

#### A. Computing Platform

Referring to FIG. 1, the disclosed system 100 comprises at least one graphics processing unit (GPU) 110 having a streaming multiprocessor (SM) architecture, a host processor (CPU) 120 communicatively coupled to the GPU 110, a system memory 130, and GPU device memory 140 including high-bandwidth global memory and per-SM shared memory 142. In a preferred embodiment, the GPU 110 comprises an NVIDIA Blackwell-architecture device (compute capability sm_120) having at least 128 streaming multiprocessors, though the disclosed methods are applicable to any CUDA-capable GPU with compute capability 7.0 or higher.

The system 100 further comprises a non-transitory computer-readable storage medium 150 storing compiled GPU kernel code in PTX (Parallel Thread Execution) intermediate representation. In a preferred embodiment, each PTX module is cryptographically signed using a SHA-256 hash and verified prior to runtime loading, providing tamper detection for production deployment.

#### B. Spatial Discretization

The system discretizes the spatial volume encompassing the target molecular structure into a three-dimensional thermal voxel grid 200. In a preferred embodiment, the grid comprises 128 x 128 x 128 voxels with a spacing of 0.75 angstroms, yielding a cubic volume of 96 x 96 x 96 angstroms. Alternative embodiments may utilize grid dimensions in the range of 64 to 256 voxels per axis with spacings in the range of 0.5 to 2.0 angstroms. The grid dimensions are configurable at runtime and automatically scaled to the bounding box of the target structure with a padding margin (e.g., 5.0 angstroms).

### II. Fused Molecular Dynamics and Neuromorphic Kernel

#### A. Kernel Architecture and Fusion Strategy

Referring to FIG. 4, the primary computational innovation of the disclosed system is a fused GPU kernel 300 that executes molecular dynamics integration and neuromorphic sensing within a single kernel launch, eliminating the CPU-GPU synchronization barrier that is the primary performance bottleneck in conventional MD-based pocket detection. In the fused kernel 300, a single GPU thread processes one atom through the complete pipeline of force computation, integration, perturbation injection, and neuromorphic state update before the kernel returns control to the host CPU. This architecture reduces per-timestep CPU-GPU round trips from N (where N equals the number of algorithmic stages) to exactly 1.

In a preferred embodiment, the fused kernel 300 is launched with a thread block size of 256 threads and launch bounds of 4 blocks per streaming multiprocessor.

#### B. BAOAB Langevin Integrator

The molecular dynamics integration employs the BAOAB splitting scheme for the Langevin equation of motion. For each atom i with mass m_i, position x_i, and velocity v_i, and subject to force F_i and friction coefficient gamma, the integration proceeds per timestep dt as:

**Step B (half-step velocity):**
```
v_i(t + dt/2) = v_i(t) + (dt/2) * F_i(t) / m_i
```

**Step A (half-step position):**
```
x_i(t + dt/2) = x_i(t) + (dt/2) * v_i(t + dt/2)
```

**Step O (Ornstein-Uhlenbeck thermostat):**
```
v_i' = c1 * v_i(t + dt/2) + c2 * R_i
where c1 = exp(-gamma * dt)
      c2 = sqrt(k_B * T / m_i * (1 - c1^2))
      R_i ~ N(0, 1) (Gaussian random variate)
```

**Step A (complete position):**
```
x_i(t + dt) = x_i(t + dt/2) + (dt/2) * v_i'
```

**Step B (complete velocity):**
```
v_i(t + dt) = v_i' + (dt/2) * F_i(t + dt) / m_i
```

In a preferred embodiment, dt = 0.002 ps. Bond constraints to hydrogen atoms are enforced using SHAKE with 10 iterations per timestep. A velocity clamp of 100 angstroms/picosecond per component prevents numerical explosion during deep-cryo phases.

#### C. Force Computation

The fused kernel computes bonded forces (harmonic bonds, harmonic angles, Ryckaert-Bellemans dihedrals) and nonbonded forces (Lennard-Jones and Coulomb with 1-4 scaling factors of 0.5 and 0.8333, respectively) using the AMBER ff14SB force field parameterization. A Coulomb constant of 332.0636 kcal*angstrom/(mol*e^2) is used.

To achieve O(N) scaling for nonbonded interactions, the kernel constructs a cell list with cell dimensions of 10.0 angstroms and a maximum of 128 atoms per cell. The cell grid supports up to 32 cells per dimension (32^3 = 32,768 total cells). Each atom maintains a neighbor list of up to 256 neighbors within a cutoff of 10.0 angstroms, with a neighbor list buffer factor of 1.2 for Verlet-list-style rebuild amortization.

#### D. Multi-Wavelength UV Photon Absorption

Concurrently with the physical simulation, the fused kernel applies a simulated UV photon absorption perturbation to chromophore-containing residues. The UV perturbation implements Beer-Lambert spectroscopy with multi-wavelength frequency hopping.

In a preferred embodiment, the system cycles through four chromophore-specific wavelengths:

| Wavelength (nm) | Target Residue | Extinction Coefficient (M^-1 cm^-1) |
|-----------------|----------------|--------------------------------------|
| 280 | Tryptophan (TRP) | 5,600 |
| 274 | Tyrosine (TYR) | 1,490 |
| 258 | Phenylalanine (PHE) | 197 |
| 211 | Histidine (HIS) | 300 |

The wavelength cycles every W dwell steps (preferably W=300 steps in a fast protocol, W=500 in a standard protocol). The energy deposited per UV burst is:

```
E_UV = (h * c / lambda) * QY * epsilon_rel(lambda) * intensity
```

where h is Planck's constant (9.537e-14 kcal*s/mol), c is the speed of light (3.0e17 nm/s), lambda is the current wavelength, QY is the quantum yield (preferably 0.02), and epsilon_rel is the wavelength-dependent relative extinction coefficient normalized to the maximum extinction.

Alternative embodiments may include additional wavelengths (e.g., 254 nm for benzene cosolvent probes, 290 nm for broad-spectrum excitation) or continuous wavelength sweeps.

#### E. Resonate-and-Fire (RAF) Multi-Neuron Oscillator Architecture

Referring to FIG. 3A, the disclosed system implements a novel neuromorphic detection layer comprising a three-dimensional grid of multi-neuron oscillator units. Each spatial voxel in the thermal grid contains K neuromorphic oscillator units (preferably K=8), each implementing a damped harmonic oscillator (resonate-and-fire, RAF) model.

Each oscillator k (where k ranges from 0 to K-1) at voxel v maintains a two-dimensional state vector (x_k, y_k) representing oscillatory displacement and velocity, governed by:

```
dx_k/dt = y_k
dy_k/dt = -omega_k^2 * x_k - (1/tau_k) * y_k + S(t)
```

where omega_k = 2*pi/tau_k is the natural angular frequency, tau_k is the characteristic timescale, and S(t) is the stimulus input derived from local molecular state changes (water density fluctuations, UV energy deposition, or electrostatic field perturbation).

**Log-Spaced Timescale Assignment.** Each oscillator k within a voxel is assigned a timescale according to:

```
tau_k = tau_base * 2^k
```

where tau_base is a base timescale (preferably 0.5 simulation timesteps). This log-spacing ensures that the K oscillators collectively span a broad frequency range, enabling detection of molecular state transitions occurring at timescales from sub-picosecond (k=0, tau=0.5 steps) to hundreds of picoseconds (k=7, tau=64 steps). This multi-timescale architecture is a key distinction from prior integrate-and-fire neuromorphic systems, which lack frequency selectivity.

**Discrete Integration.** The oscillator state is updated per simulation timestep using forward Euler:

```
x_new = x + y * dt_neuron
y_new = y + (-omega^2 * x - y/tau + stimulus) * dt_neuron
```

**Saturated Resonance Clamp.** To prevent numerical divergence while preserving oscillatory dynamics, the amplitude R = sqrt(x^2 + y^2) is clamped to a maximum value R_max (preferably R_max = 2.0):

```
if x_new^2 + y_new^2 > R_max^2:
    scale = R_max / sqrt(x_new^2 + y_new^2)
    x_new *= scale
    y_new *= scale
```

This projection onto a disk of radius R_max preserves the phase angle of the oscillator while bounding the amplitude, a critical property for numerical stability during cryo-thermal transitions where stimulus magnitudes may vary by orders of magnitude.

**UV-Gated Spike Detection.** A spike event is emitted when BOTH conditions are simultaneously satisfied:

```
spike = (amplitude >= effective_threshold) AND (uv_signal > 0)
```

where `amplitude = sqrt(x^2 + y^2)` and `effective_threshold = T_base * (1 + jitter)`. The UV gating condition ensures that spikes are only emitted during active photon perturbation periods, preventing false positives from thermal noise. The base threshold T_base is preferably 0.3.

**Stochastic Threshold Jitter.** The jitter term introduces controlled stochastic variation to prevent synchronized firing artifacts:

```
jitter = A_jitter * hash(voxel_idx, timestep, k)
```

where A_jitter is the jitter amplitude (preferably 0.02, yielding +/-2% threshold variation), and hash() is a deterministic hash function using three Knuth multiplicative constants (2654435761, 2246822519, 3266489917) to produce a pseudo-random value in [-1, 1]. This ensures reproducibility while preventing artificial coherence.

**Threshold Adaptation.** The effective threshold is raised and the oscillator enters a refractory period (preferably 250 timesteps, corresponding to 0.1 ps at dt=0.002 ps) only when a spike is successfully emitted (i.e., not blocked by the global spike grid). This prevents blocked spikes from penalizing oscillator sensitivity.

#### F. Shared-Memory Halo Stencil for Spatial Coupling

Referring to FIG. 3B, the RAF oscillators are spatially coupled to enable wavefront propagation across neighboring voxels. The coupling is implemented using a shared-memory halo stencil architecture that maps directly to GPU shared memory (SMEM), avoiding global memory latency for neighbor access.

In a preferred embodiment:
- Each GPU thread block processes a 3D tile of 4 x 4 x 2 = 32 voxels
- The tile is surrounded by a 1-voxel halo, yielding a shared memory buffer of (4+2) x (4+2) x (2+2) = 6 x 6 x 4 = 144 floating-point values
- At 4 bytes per float, this requires 576 bytes of shared memory per thread block
- The halo is cooperatively loaded by all 256 threads in the block (256 threads / 32 voxels = 8 threads per voxel = K neurons per voxel)
- Each thread within a voxel processes one of the K=8 oscillators

The coupling stimulus from neighboring voxels is:

```
coupling_input = COUPLING_STRENGTH * neighbor_amplitude * COUPLING_DECAY^distance
```

where COUPLING_STRENGTH is preferably 0.01 and COUPLING_DECAY is preferably 0.9. The neighbor amplitude is read from shared memory (zero main-memory latency), enabling spatial wavefront propagation within a single kernel launch.

The kernel launch configuration is: grid = (32, 32, 64), block = (256, 1, 1), shared_memory = 576 bytes.

#### G. Spike Event Capture and Wire Format

When a spike event is detected, the kernel captures a structured event record containing both the spike detection data and the spectroscopic metadata from the concurrent molecular simulation. In a preferred embodiment, the spike event comprises a 92-byte structure including:

- Temporal index (timestep, 4 bytes)
- Spatial index (voxel linear index, 4 bytes)
- Cartesian coordinates (x, y, z in angstroms, 12 bytes)
- Intensity (normalized amplitude, 4 bytes)
- Nearby residue indices (8 nearest residues, 32 bytes)
- Residue count (4 bytes)
- Source channel identifier (1=UV, 2=LIF/dewetting, 3=EFP/electrostatic, 4 bytes)
- Spectroscopic wavelength in nanometers (4 bytes)
- Aromatic chromophore type (0=TRP, 1=TYR, 2=PHE, 3=disulfide, -1=none, 4 bytes)
- Aromatic residue identifier (4 bytes)
- Local water density in molecules per cubic angstrom (4 bytes)
- Vibrational energy deposited in kcal/mol (4 bytes)
- Nearby excited aromatic count for pi-stacking detection (4 bytes)
- Water density temporal derivative for energy gradient estimation (4 bytes)

This rich metadata enables downstream channel decomposition, pharmacophore mapping, and thermodynamic estimation without re-running the simulation.

### III. Cryo-Thermal Hysteresis Protocol

#### A. Five-Phase Temperature Schedule

Referring to FIG. 2, the disclosed method applies a multi-phase cryo-thermal hysteresis protocol to probe conformational memory in the molecular target. The protocol comprises:

**Phase 1 — Cryogenic Hold (Cold Baseline):** The system holds the molecular structure at a cryogenic temperature T_cryo (preferably 50K, alternatively in the range 20K to 100K) for N_cold steps (preferably 14,000 steps). This phase establishes the frozen conformational baseline.

**Phase 2 — Heating Ramp (Forward Process):** The system linearly increases temperature from T_cryo to T_warm over N_ramp steps (preferably 6,000 steps). Temperature at step s within this phase is:
```
T(s) = T_cryo + (s - s_start) / N_ramp * (T_warm - T_cryo)
```

**Phase 3 — Warm Hold (Physiological Equilibration):** The system holds at physiological temperature T_warm (preferably 300K, alternatively in the range 280K to 350K) for N_warm steps (preferably 15,000 steps). This phase allows full conformational exploration at physiological conditions.

**Phase 4 — Cooling Ramp (Reverse Process):** The system linearly decreases temperature from T_warm back to T_cryo over N_ramp_down steps (preferably equal to N_ramp). This reverse process enables detection of hysteresis — conformational changes that do not reverse upon cooling indicate structural memory.

**Phase 5 — Cryogenic Return (Hysteresis Detection):** The system returns to T_cryo for N_cold_return steps (preferably equal to N_cold). Comparison of spike activity between Phase 1 and Phase 5 reveals conformational memory effects.

In a preferred fast protocol, the total steps per stream are 14,000 + 6,000 + 15,000 + 6,000 + 14,000 = 55,000 steps at dt=0.002 ps, corresponding to 110 ps of simulated time per stream. In a standard protocol, the total is 5,000 + 10,000 + 5,000 + 10,000 + 5,000 = 35,000 steps (70 ps).

#### B. Two-Phase Friction Coefficient

To maintain kinetic stability during deep-cryo phases, the system employs a two-phase friction coefficient gamma:

**Equilibration Phase** (preferably the first 100 steps): gamma = gamma_eq (preferably 1000 ps^-1), providing heavy damping for initial relaxation.

**Production Phase** (all subsequent steps): gamma is dynamically scaled according to:
```
gamma = gamma_base * sqrt(T_ref / T_current)
```
where gamma_base is preferably 1.0 ps^-1 and T_ref is a reference temperature (preferably 300K). At cryogenic temperatures (e.g., T=50K), this yields gamma approximately equal to 2.45 ps^-1, providing enhanced damping that prevents large-amplitude oscillations in a nearly frozen system.

#### C. Multi-Wavelength UV Cycling

During all phases, the system concurrently applies UV photon absorption with wavelength frequency hopping as described in Section II.D. The UV burst pattern is characterized by:

- **Burst interval:** Every I steps (preferably I=250 for fast protocol, I=500 for standard)
- **Burst duration:** D steps (preferably D=50 steps)
- **Burst energy:** E kcal/mol (preferably E=42.0 for fast protocol, E=30.0 for standard)
- **Wavelength dwell:** W steps per wavelength (preferably W=300 for fast protocol)
- **Wavelength sequence:** Cycling through [280, 274, 258, 211] nm

The current active wavelength is determined per timestep by:
```
wavelength_index = (current_step / W) mod len(wavelength_list)
current_wavelength = wavelength_list[wavelength_index]
```

This value is passed as a kernel launch parameter to the fused kernel on every timestep.

### IV. Multi-Stream Concurrent Execution

#### A. Independent Stream Architecture

Referring to FIG. 10, the disclosed system executes N independent instances of the fused MD-neuromorphic simulation concurrently on a single GPU using N CUDA streams (preferably N=4 or N=8). Each stream executes the complete cryo-thermal hysteresis protocol with an independently seeded random number generator, yielding N statistically independent spike ensembles.

A single CUDA context and a single PTX module are shared across all streams. Each stream maintains independent GPU memory allocations for atomic positions, velocities, forces, neuromorphic oscillator states, and spike event buffers.

#### B. Consensus Site Merging

After all N streams complete, the system merges per-stream binding site detections into consensus sites. Two sites from different streams are considered to correspond to the same physical pocket if their centroids are within a consensus threshold distance (preferably 5.0 angstroms). The consensus centroid is computed as the mean of matched site centroids across streams.

This multi-stream architecture provides both statistical robustness (sites detected in multiple independent simulations are high-confidence) and improved sensitivity (rare conformational events captured by any stream contribute to the final result).

### V. Spike-Driven Spatiotemporal Analysis (SDST) Pipeline

#### A. Overview

Referring to FIG. 5, captured spike events are ingested into a GPU-native spatiotemporal analysis library (SDST) that transforms raw spike events into thermodynamic binding site characterizations. The SDST pipeline comprises six stages executed entirely on the GPU without CPU round-trips.

#### B. Stage 1: Morton-Encoded Spatial Hash Insertion

Each spike event's spatial coordinates are encoded into a Morton code (Z-order curve) using bit interleaving:

```
morton_encode(x, y, z) = spread(x) | (spread(y) << 1) | (spread(z) << 2)
```

where spread() inserts two zero bits between each bit of a 7-bit input, supporting grid dimensions up to 128 per axis (2^7). The Morton code serves as a key into an open-address hash table with Murmur3 hash function and linear probing. In a preferred embodiment, the hash table capacity is 2^22 = 4,194,304 slots.

Each voxel maintains a linked chain of spike events ordered by insertion time, enabling temporal queries. The hash table and event chains reside entirely in GPU global memory.

**GPU-Native Ingestion.** A kernel converts 92-byte GpuSpikeEvent structures from the fused MD kernel into compact 36-byte SDST SpikeEvent structures directly on the GPU, with zero CPU memory copies. The 36-byte structure uses half-precision (f16) encoding for amplitude, temperature, energy gradient, solvent exposure, wavefront velocity, and wavefront coherence fields, achieving 2.56x compression while preserving sufficient precision for downstream analysis.

#### C. Stage 2: Causal Parent Detection

For each spike event, the system identifies a causal parent spike — the most likely predecessor that triggered the current event — using spatial and temporal proximity criteria:

- **Spatial cutoff:** 7.5 angstroms (10 voxels at 0.75 angstrom spacing)
- **Temporal window:** 50 timesteps maximum between parent and child
- **Chain walk limit:** 512 events per voxel to bound computation

The algorithm performs a breadth-first walk through the voxel chain, selecting the temporally closest predecessor within the spatial cutoff. A stale-voxel skip optimization provides O(1) early termination for voxels that have not received events within the temporal window.

#### D. Stage 3: Lock-Free Pointer-Jumping Avalanche Clustering

Referring to FIG. 5, the system determines connected avalanche clusters using a novel lock-free parallel algorithm based on pointer jumping. This algorithm represents a significant advance over prior GPU-parallel connected component methods that rely on atomic compare-and-swap (CAS) operations, which create contention bottlenecks at scale.

**Initialization:** Each spike event's avalanche label is set to its own index (self-labeling):
```
label[i] = i for all i in [0, N_events)
```

**Iterative Doubling:** For a fixed number of iterations (preferably 30), each event updates its label by jumping to its label's label:
```
label[i] = label[label[i]]
```

This doubling operation converges in O(log D) iterations where D is the maximum depth of any causal chain. Each iteration requires exactly two global memory reads and one global memory write per event. **No atomic operations of any kind are required** — no CAS, no atomic exchange, no atomic add. This is possible because the pointer-jumping update is monotone (labels can only decrease or remain constant), ensuring convergence without synchronization.

**Finalization:** A label compression kernel remaps labels to contiguous cluster IDs.

In practice, for event counts up to 26 million (validated on target structures), the algorithm converges in fewer than 15 iterations, with total avalanche clustering time under 50 milliseconds.

#### E. Stage 4: Histogram-Based Hill MLE Criticality Analysis

The system computes the power-law criticality exponent tau for the avalanche size distribution using a histogram-based maximum likelihood estimator. The avalanche size histogram is constructed via:

1. CUB radix sort of events by avalanche_id
2. Run-length encoding to determine avalanche sizes
3. CUB segmented sort by spatial tile index to group avalanches by region
4. Atomic histogram accumulation into per-region bins (up to 1024 size bins per region)

The Hill MLE with discrete Clauset-Shalizi-Newman correction is:

```
tau = 1 + N / sum_{s >= s_min} count(s) * ln(s / (s_min - 0.5))
```

where N is the total number of avalanches with size >= s_min (preferably s_min = 2), and count(s) is the number of avalanches of size s.

The standard error is: `SE(tau) = (tau - 1) / sqrt(N)`

**Classification:**
- tau < 1.5: Self-Organized Critical (SOC) — highest druggability
- 1.5 <= tau < 2.0: Near-Critical — moderate druggability
- tau >= 2.0: Barrier-dominated — lowest druggability

**Druggability score:**
```
D = (1 - (tau - 1) / 3) * confidence
where confidence = 1 - SE/tau if tau > 1e-6, else 0
```

This histogram-based approach replaces earlier hash-dedup methods that silently overflowed at high event counts (>20M events), producing erroneous tau=1.0 values. The histogram is exact for any event count.

#### F. Stage 5: Wavefront Coherence Tracking

The system tracks coherent wavefronts — spatially connected propagation fronts of spike activity. Each spike inherits its parent's wavefront ID if the parent is within a spatial merge distance and temporal gap threshold. Otherwise, a new wavefront is created via atomic increment.

Per-wavefront statistics include: origin voxel, birth/death timestep, spike count, mean propagation velocity, mean spatial coherence, maximum spatial extent, and hysteresis phase.

**Coherence metric:** For each spike, the fraction of neighboring voxels (within the merge distance) that belong to the same wavefront is computed:
```
coherence = count(neighbors in same wavefront) / count(total neighbors)
```
This value (range [0, 1]) is stored as half-precision in the SpikeEvent record.

#### G. Stage 6: Thermodynamic Context Labeling (TCL) and Hysteresis Analysis

Each spike event is annotated with an 8-bit thermodynamic context flag:

| Bit | Name | Condition |
|-----|------|-----------|
| 0 | is_transition | Spike within +/-500 steps of a phase boundary |
| 1 | is_boundary | Solvent exposure > 0.3 (surface-exposed spike) |
| 2 | high_gradient | Energy gradient |nabla E| > 0.5 |
| 3 | cooling_spike | Spike occurred during Phase 3 or 4 |
| 4 | heating_spike | Spike occurred during Phase 0 or 1 |
| 5 | peak_temp | Spike occurred during Phase 2 |
| 6 | hysteresis_candidate | Reserved for future classification |

The hysteresis asymmetry is computed per spatial tile (8^3 voxel cubes):

```
asymmetry = |heating_spike_count - cooling_spike_count| / total_spike_count
```

A site is flagged as hysteretic if asymmetry exceeds a threshold (preferably 0.2).

### VI. Transfer Entropy-Integrated Decomposed Energetics (TIDE)

#### A. Per-Residue Transfer Entropy

For each detected binding site, the system computes the directed information flow from each proximal residue's spike train to the pocket's aggregate spike train using Schreiber transfer entropy:

```
TE(X -> Y) = sum p(y_{t+1}, y_t^k, x_t^l) * log[p(y_{t+1} | y_t^k, x_t^l) / p(y_{t+1} | y_t^k)]
```

where X is the binary spike train of a residue, Y is the binary spike train of the pocket, and k = l = 1 (Markov order). The computation uses 100-timestep bins (up to 1024 bins).

#### B. Pocket-Locality Spatial Filter

To prevent globally-active residues from dominating every pocket's transfer entropy, the computation applies a pocket-local spatial filter: only spike events within the pocket bounding box plus a margin of +/-2 voxels contribute to the residue spike train. This spatial filter is critical for accurate causal attribution — without it, constitutively active residues (e.g., surface lysines) would appear to causally influence every detected pocket.

#### C. Causal Free Energy Contribution

Each residue's causal free energy contribution to the pocket is:

```
Delta_G_causal = -TE * log(1 + n_causal_spikes) * R * T
```

where R*T = 0.596 kcal/mol at 300K, TE is the transfer entropy, and n_causal_spikes is the count of spikes from the residue that are causally connected to pocket spikes. The range is [-0.30, 0.00] kcal/mol for active trigger residues.

#### D. Fisher Information and KL Divergence

Additionally computed per residue:

**Fisher Information:** The variance of per-window transfer entropy across sliding windows:
```
FI = Var(TE_window) over windows of 10 bins
```
Fisher information quantifies perturbation sensitivity — residues with high Fisher information are leverage points for drug design.

**KL Divergence:** The Kullback-Leibler divergence between heating-phase and cooling-phase spike distributions:
```
KL(P_heat || P_cool) = sum P_heat(s) * log(P_heat(s) / P_cool(s))
```
High KL divergence indicates asymmetric conformational reorganization, characteristic of gating residues.

#### E. Residue Role Classification

Based on the combined TIDE metrics, each residue is classified:

| Role | Condition | Drug Design Significance |
|------|-----------|--------------------------|
| TRIGGER | High TE AND High Fisher | Prime drug design target — causally controls pocket opening |
| STABILIZER | High TE AND Low Fisher | Conformationally important — maintains pocket stability |
| GATEWAY | High KL Divergence | Gatekeeper — controls access but not pocket dynamics |
| SPECTATOR | Low TE, Low Fisher, Low KL | Minimal influence — not a design target |

### VII. Thermodynamic Multi-Signal Classification (ThermClass)

Referring to FIG. 6, each detected binding site is assigned a thermodynamic classification based on the integration of multiple independent signals:

**Z-score normalization:** The hysteresis asymmetry score for each site is normalized against the population:
```
z = (asymmetry_i - mean(asymmetry)) / std(asymmetry)
```

**Multi-signal decision logic:**
```
if z > 1.5 AND tau > 0 AND (is_SOC OR TIDE_enriched OR asymmetry > 0.4):
    class = CRYPTIC
else if z > 0.5:
    class = DYNAMIC
else if z > -0.5:
    class = RESPONSIVE
else:
    class = INERT
```

where is_SOC = (tau > 0 AND tau < 1.5) and TIDE_enriched = (top_residue_TE > 2 * median_all_residue_TE).

CRYPTIC sites are the highest-priority drug design targets: they exhibit conformational memory (high asymmetry), critical dynamics (SOC), and concentrated causal residue networks (TIDE enrichment).

### VIII. Spike Thermodynamic Integration (STI) — Multi-Estimator Framework

Referring to FIG. 7, the system estimates binding free energies using multiple independent non-equilibrium thermodynamic estimators applied to the spike work ensemble.

#### A. Spike-to-Work Conversion

Each spike event is converted to a thermodynamic work value based on its source channel:

**UV channel (source=1):**
```
W_UV = (h * c / lambda) * QY * epsilon_rel(lambda) * intensity
```

**LIF/Dewetting channel (source=2):**
```
W_LIF = intensity * E_hydration
```
where E_hydration = 2.27 kcal/mol (solvation energy of one water molecule).

**EFP/Electrostatic channel (source=3):**
```
W_EFP = (332.0636 * q1 * q2) / (epsilon_eff * r) * intensity
```
where q1*q2 = 0.25 e^2 (typical), r = 4.0 angstroms, epsilon_eff = 20.0 (Warshel effective dielectric).

#### B. Jarzynski Free Energy Estimator

The system applies Jarzynski's equality per voxel:

```
Delta_G = -k_B * T * ln((1/N) * sum_i exp(-W_i / (k_B * T)))
```

with a cumulant expansion validity gate:
```
cumulant_valid = (sigma_W^2 / (k_B * T)^2 < 1.0) AND (N >= 5)
```

Only spikes from forward-process phases (heating + warm hold) contribute to the Jarzynski estimator.

#### C. Crooks Intersection Estimator

Using spike counts from heating (forward) and cooling (reverse) phases, the system computes the log-ratio of forward and reverse work distributions per temperature bin:

```
log_ratio = ln(p_forward(W) / p_reverse(-W))
```

The free energy Delta_G corresponds to the work value at which log_ratio = 0 (found by linear interpolation of the zero-crossing). Laplace smoothing prevents division by zero:
```
p = (count + 0.5) / (total + 0.5 * n_bins)
```

#### D. Bennett Acceptance Ratio (BAR)

An iterative estimator solving:
```
sum_R f(-W_R + C) = sum_F f(W_F - C)
where f(x) = 1 / (1 + exp(x / (k_B * T)))
```

The iteration updates:
```
C_new = k_B * T * ln(sum_R / sum_F) + k_B * T * ln(n_F / n_R)
```
converging when |C_new - C| < 1e-6 (maximum 100 iterations). Delta_G_BAR = C_converged.

#### E. Channel Decomposition

The total binding free energy is decomposed by source channel:
```
Delta_G_total = Delta_G_UV + Delta_G_LIF + Delta_G_EFP + Delta_G_cooperative
```

where the cooperative term captures multi-channel synergy:
```
Delta_G_cooperative = Delta_G_total - (Delta_G_UV + Delta_G_LIF + Delta_G_EFP)
```

A non-zero cooperative term indicates emergent binding contributions that arise only from the simultaneous interaction of multiple perturbation channels — a phenomenon not observable in single-channel analysis.

#### F. Arrhenius Activation Energy

The system extracts activation energies per UV wavelength by analyzing the temperature dependence of spike rates during the heating phase:

```
ln(rate) = ln(A) - E_a / (k_B * T)
```

Linear regression of ln(rate) versus 1/T yields E_a = -slope * k_B for each wavelength (280, 274, 258, 211 nm). The mean activation energy contributes to a kinetic accessibility score:

```
kinetic_accessibility = exp(-E_a_mean / (k_B * T))   [clamped to [0, 1]]
```

The effective binding free energy incorporating kinetic accessibility is:
```
Delta_G_effective = Delta_G_bind + k_B * T * ln(kinetic_accessibility)
```

### IX. Spike-to-Pharmacophore Feature Mapping

Referring to FIG. 8, the system maps neuromorphic spike metadata to pharmacophore features — the spatial pattern of molecular interactions required for ligand binding.

**Channel/Wavelength to Feature Type:**

| Source | Wavelength | Feature Type |
|--------|-----------|--------------|
| UV | 280 nm (TRP) | Aromatic / Hydrophobic |
| UV | 274 nm (TYR) | Hydrogen Bond Donor (HBD) |
| UV | 258 nm (PHE) | Hydrophobic |
| UV | 211 nm (HIS) | Positive Charge |
| LIF | -- (dewetting) | Hydrophobic |
| EFP | -- (electrostatic) | Negative Charge |

**Time-To-First-Spike (TTFS) Ranking:** Spike groups are sorted by earliest timestep within the group. The first group to fire (rank 1) indicates the molecular feature with the strongest binding affinity signal. This temporal ordering provides binding-event prioritization that static pharmacophore models cannot achieve.

**Pharmacophore Feature Structure:**
Each generated feature comprises a spatial position (centroid of contributing spikes), feature type (from channel/wavelength mapping), normalized strength [0, 1], TTFS rank, synchrony group ID, source channel, and wavelength.

The features are exported in a format compatible with standard pharmacophore modeling tools (e.g., PGMG, Phase) for downstream virtual screening.

### X. Adaptive Clustering and Centroid Computation

#### A. Adaptive Epsilon

The DBSCAN clustering epsilon parameter is automatically adapted to the target structure size:

```
epsilon = 3.0 * (500 / n_atoms)^(1/3)
```
clamped to the range [1.2, 3.0] angstroms. Larger proteins receive smaller epsilon to prevent mega-cluster formation.

#### B. Adaptive Minimum Spike Count

The minimum spike count for a valid cluster is:
```
min_spikes = max(spikes_per_aromatic * 0.3, 50)
```
This ensures that larger proteins with more aromatic residues (and thus more spike generation capacity) require proportionally more evidence for site detection.

#### C. Mega-Cluster Density Peak Subdivision

When a single DBSCAN cluster absorbs more than 50% of total spikes (percolation threshold), the system subdivides it using a density peak algorithm:

1. Construct a 3-angstrom voxel density grid over the mega-cluster bounding box
2. Identify local maxima as voxels whose density exceeds all 26 neighbors (3x3x3 cube)
3. Filter peaks to those with density >= 5% of maximum or >= 10 spikes
4. Reassign each spike to its nearest density peak by Euclidean distance

### XI. Output Format and Drug Design Integration

Referring to FIG. 9, the system outputs a structured JSON data object containing, for each detected binding site:

- **Geometry:** Centroid coordinates, volume estimate, spike count, mean and maximum intensity
- **Druggability:** Composite druggability score, aromatic proximity metrics, nearby aromatic residues, lining residues with catalytic flags
- **Thermodynamics:** Per-channel Delta_G values (UV, LIF, EFP, cooperative), Crooks and BAR estimates, kinetic accessibility, effective Delta_G, cumulant validity flag, per-wavelength activation energies
- **Criticality:** CCNS tau exponent, classification (SOC/NearCritical/Barrier), druggability derived from criticality
- **Classification:** ThermClass (CRYPTIC/DYNAMIC/RESPONSIVE/INERT), z-score, raw asymmetry
- **Causal Decomposition:** Top 20 TIDE residues with transfer entropy, Fisher information, KL divergence, causal Delta_G, and role classification (TRIGGER/STABILIZER/GATEWAY/SPECTATOR)

Additionally, the system outputs: annotated PDB files for molecular visualization, per-site spike event files containing full spectroscopic metadata for pharmacophore extraction, ensemble trajectory files, and PRISM-Therm thermodynamic reports.

---

## CLAIMS

### Independent Method Claims

**1.** A computer-implemented method for detecting transient molecular binding sites, the method comprising:

(a) receiving, at a computing system comprising at least one graphics processing unit (GPU), a molecular topology corresponding to a target macromolecular structure;

(b) executing, by the at least one GPU, a fused molecular dynamics and neuromorphic sensing kernel in a single computational pass, wherein executing the fused kernel comprises:
- (i) computing atomic forces and integrating equations of motion for the target structure using a Langevin dynamics integrator;
- (ii) concurrently applying a simulated physical perturbation to the target structure;
- (iii) processing localized molecular states of the target structure using a plurality of neuromorphic oscillator units distributed across a three-dimensional spatial grid, wherein each oscillator unit is configured to detect local molecular state transitions; and
- (iv) capturing a plurality of spike events, each spike event comprising at least a spatial coordinate, a temporal index, and an intensity value, when a respective neuromorphic oscillator satisfies a detection criterion;

(c) clustering, by the at least one GPU, the plurality of spike events into spatiotemporally connected avalanches; and

(d) outputting at least one candidate binding site based on the clustered avalanches.

**2.** The method of claim 1, wherein the plurality of neuromorphic oscillator units comprises a plurality of resonate-and-fire (RAF) oscillators, each RAF oscillator implementing a damped harmonic oscillator characterized by a natural frequency.

**3.** The method of claim 2, wherein each spatial location in the three-dimensional grid comprises K resonate-and-fire oscillators (where K is at least 2, preferably K=8), and wherein the K oscillators are assigned timescales spanning at least two orders of magnitude.

**4.** The method of claim 3, wherein the K oscillators are assigned log-spaced timescales according to tau_k = tau_base * 2^k for k in {0, 1, ..., K-1}, where tau_base is a base timescale.

**5.** The method of claim 2, wherein the RAF oscillators comprise a saturated resonance clamp that projects the oscillator state vector onto a disk of radius R_max when the oscillator amplitude exceeds R_max, thereby bounding oscillator amplitude while preserving oscillator phase.

**6.** The method of claim 1, wherein capturing a plurality of spike events requires that both: (i) an oscillator amplitude exceeds a detection threshold, and (ii) the simulated physical perturbation is concurrently active at the spatial location of the oscillator.

**7.** The method of claim 6, wherein the detection threshold includes a stochastic jitter component determined by a deterministic hash of the spatial location index, the temporal index, and the oscillator index, whereby the jitter introduces controlled threshold variation to prevent synchronized firing artifacts while maintaining reproducibility.

**8.** The method of claim 1, wherein the neuromorphic oscillator units are spatially coupled via a shared-memory halo stencil stored in a shared memory of the at least one GPU, the shared-memory halo comprising a buffer of neighboring oscillator amplitudes loaded cooperatively by threads of a thread block, thereby enabling spatial wavefront propagation without main-memory latency.

**9.** The method of claim 8, wherein the shared-memory halo stencil maps a tile of T_x * T_y * T_z voxels to a halo buffer of (T_x + 2) * (T_y + 2) * (T_z + 2) floating-point values in GPU shared memory.

**10.** The method of claim 1, wherein the simulated physical perturbation comprises simulated ultraviolet (UV) photon absorption applied to chromophore-containing residues of the target structure, wherein the UV photon absorption cycles through a plurality of chromophore-specific wavelengths at a predetermined dwell interval.

**11.** The method of claim 10, wherein each spike event further comprises spectroscopic metadata including a source channel identifier, a wavelength value, and a chromophore type, enabling downstream pharmacophore feature inference from the spike event record without re-running the simulation.

**12.** The method of claim 1, further comprising applying a multi-phase cryo-thermal hysteresis protocol comprising at least: (i) a cryogenic hold phase at a first temperature; (ii) a heating ramp phase; (iii) a warm hold phase at a second temperature; (iv) a cooling ramp phase; and (v) a cryogenic return phase at the first temperature.

**13.** The method of claim 12, wherein a Langevin friction coefficient is dynamically scaled during the cryo-thermal protocol according to gamma = gamma_base * sqrt(T_ref / T_current), providing enhanced damping at cryogenic temperatures to maintain kinetic stability.

**14.** The method of claim 1, wherein clustering the plurality of spike events into spatiotemporally connected avalanches comprises executing a lock-free pointer-jumping algorithm on the at least one GPU, wherein each iteration of the algorithm updates each spike event's cluster label by reading the label of its current label, and wherein the algorithm requires zero atomic compare-and-swap operations.

**15.** The method of claim 1, further comprising:

(e) computing a criticality exponent for each candidate binding site by constructing an avalanche size histogram and applying a maximum likelihood estimator for a discrete power-law distribution.

**16.** The method of claim 15, wherein the maximum likelihood estimator is a Hill estimator with a discrete Clauset-Shalizi-Newman correction:
```
tau = 1 + N / sum(count(s) * ln(s / (s_min - 0.5)))
```
for avalanches of size s >= s_min.

**17.** The method of claim 15, further comprising classifying each candidate binding site as Self-Organized Critical (SOC), Near-Critical, or Barrier-dominated based on the criticality exponent, and computing a druggability score inversely related to the criticality exponent.

**18.** The method of claim 1, further comprising:

(e) computing a directed information transfer metric from each of a plurality of amino acid residues to each candidate binding site using transfer entropy on binary spike trains derived from the spike events; and

(f) computing a causal free energy contribution for each residue based on the directed information transfer metric.

**19.** The method of claim 18, wherein computing the directed information transfer metric comprises applying a spatial filter that restricts the residue spike train to spike events occurring within a predefined margin around the candidate binding site, thereby isolating pocket-local causal dynamics from global spike activity.

**20.** The method of claim 18, further comprising classifying each residue as a Trigger, Stabilizer, Gateway, or Spectator based on the transfer entropy, a Fisher information metric, and a Kullback-Leibler divergence between heating-phase and cooling-phase spike distributions.

**21.** The method of claim 1, further comprising:

(e) computing binding free energy estimates for each candidate binding site using at least two independent non-equilibrium thermodynamic estimators applied to work values derived from the spike events.

**22.** The method of claim 21, wherein the at least two independent non-equilibrium thermodynamic estimators comprise a Jarzynski estimator applied to forward-process spike work values and a Crooks fluctuation theorem estimator applied to the intersection of forward and reverse work distributions.

**23.** The method of claim 21, further comprising decomposing the binding free energy into per-channel contributions based on the source channel identifier of each spike event, and computing a cooperative term as the difference between the total free energy and the sum of per-channel free energies.

**24.** The method of claim 1, further comprising:

(e) mapping each spike event to a pharmacophore feature type based on the source channel and wavelength of the spike event; and

(f) ranking the pharmacophore features by a Time-To-First-Spike metric reflecting binding event temporal priority.

**25.** The method of claim 1, further comprising executing a plurality of independent instances of the fused kernel concurrently on the at least one GPU using a plurality of independent CUDA streams, and merging candidate binding sites across the plurality of instances using a consensus distance threshold.

**26.** The method of claim 1, further comprising classifying each candidate binding site into one of a plurality of thermodynamic states based on at least: (i) a z-score of hysteresis asymmetry between heating-phase and cooling-phase spike counts; (ii) a criticality exponent; and (iii) a transfer entropy enrichment metric.

**27.** The method of claim 26, wherein the plurality of thermodynamic states comprises at least: CRYPTIC (indicating conformational memory with critical dynamics), DYNAMIC (indicating significant asymmetry without full criticality), RESPONSIVE (indicating moderate thermal response), and INERT (indicating minimal conformational change).

### Independent System Claim

**28.** A high-performance computing system for detecting transient and cryptic molecular binding sites, the system comprising:

(a) at least one graphics processing unit (GPU) comprising a plurality of streaming multiprocessors, each streaming multiprocessor comprising a shared memory;

(b) a host processor communicatively coupled to the at least one GPU;

(c) a non-transitory computer-readable medium storing a fused molecular dynamics and neuromorphic sensing kernel compiled to an intermediate representation executable by the at least one GPU; and

(d) program logic configured to cause the at least one GPU to:
- (i) receive a molecular topology corresponding to a target macromolecular structure;
- (ii) execute the fused kernel in a single computational pass per simulation timestep, the fused kernel comprising interleaved Langevin dynamics integration, physical perturbation injection, and neuromorphic oscillator state updates with spike event detection;
- (iii) cluster captured spike events into spatiotemporally connected avalanches using a lock-free parallel algorithm without atomic compare-and-swap operations;
- (iv) compute a criticality exponent from avalanche size distributions;
- (v) compute directed information transfer from individual residues to detected avalanche clusters; and
- (vi) output structured binding site characterizations comprising geometry, thermodynamic metrics, criticality classification, and per-residue causal decomposition.

**29.** The system of claim 28, wherein the fused kernel comprises a plurality of resonate-and-fire oscillators per spatial voxel, the oscillators spatially coupled via a shared-memory halo stencil allocated in the shared memory of each streaming multiprocessor.

**30.** The system of claim 28, wherein the non-transitory computer-readable medium further stores a cryptographic signature for the intermediate representation, and wherein the program logic verifies the cryptographic signature prior to loading the intermediate representation for execution.

### Independent Computer-Readable Medium Claim

**31.** A non-transitory computer-readable medium storing instructions that, when executed by at least one hardware processor comprising a graphics processing unit (GPU), cause the at least one hardware processor to perform operations comprising:

(a) applying a multi-phase thermal perturbation protocol to a target molecular structure, the protocol comprising at least a forward temperature sweep and a reverse temperature sweep;

(b) processing localized molecular states using a three-dimensional grid of neuromorphic oscillator units, each oscillator unit configured to emit spike events when a local molecular state transition exceeds a detection criterion concurrently with an active physical perturbation;

(c) clustering the spike events into spatiotemporally connected avalanches using a parallel algorithm that does not require atomic compare-and-swap operations;

(d) computing a criticality exponent from avalanche size distributions and a directed information transfer metric from individual residues to detected binding pockets; and

(e) classifying candidate binding sites based on a multi-signal integration of hysteresis asymmetry, criticality, and causal information flow.

### Dependent Claims on Medium Claim

**32.** The non-transitory computer-readable medium of claim 31, wherein each spike event comprises spectroscopic metadata including a source channel identifier and a wavelength value, and wherein the operations further comprise mapping spike events to pharmacophore features based on the spectroscopic metadata.

**33.** The non-transitory computer-readable medium of claim 31, wherein computing the directed information transfer metric comprises computing transfer entropy on binary spike trains with a pocket-local spatial filter that restricts computation to spike events within a predefined margin of each binding pocket.

---

## ABSTRACT OF THE DISCLOSURE

A GPU-accelerated computational system and method detects transient and cryptic molecular binding sites by fusing molecular dynamics simulation with neuromorphic event-driven sensing within a single GPU kernel launch. A target molecular structure undergoes a multi-phase cryo-thermal hysteresis protocol with concurrent multi-wavelength ultraviolet photon absorption perturbation. Localized molecular state transitions are detected by a three-dimensional grid of resonate-and-fire (RAF) multi-neuron oscillators spatially coupled via a GPU shared-memory halo stencil. Detected spike events are clustered into avalanches using a lock-free pointer-jumping algorithm requiring zero atomic compare-and-swap operations. The avalanche size distribution is analyzed via histogram-based Hill maximum likelihood estimation to determine self-organized criticality. Per-residue causal contributions are quantified using transfer entropy with pocket-local spatial filtering. Binding free energies are estimated using multiple independent non-equilibrium thermodynamic estimators. Each candidate site is classified by integrating hysteresis asymmetry, criticality, and causal information flow into a thermodynamic state (CRYPTIC, DYNAMIC, RESPONSIVE, or INERT) with associated druggability scores for downstream drug design.

---

## CONTINUATION STRATEGY NOTES (ATTORNEY EYES ONLY — NOT FOR FILING)

### Provisional Filing Covers:
- All claims above based on validated codebase at commit 5adee9cf

### Planned Continuation-in-Part (CIP) Topics:
1. **Lorentzian resonance scanning** — pocket-specific opening rate characterization (code exists in resonance_scan.rs, not yet validated)
2. **Density-peak centroid refinement** — water density filtering and intensity-weighted centroid computation for improved spatial accuracy
3. **GPU Gaussian density splatting for pharmacophore visualization** — pharmacophore_splat.cu with OpenDX output
4. **Adaptive multi-stream consensus with weighted voting** — stream-quality-weighted site merging
5. **STDP-based threshold adaptation** — spike-timing dependent plasticity for oscillator learning

### Design-Around Protection Strategy:
- Claim 1 is broad enough to cover any neuromorphic-oscillator + MD fusion (not limited to RAF)
- Claim 14 independently protects the pointer-jumping clustering (usable outside this system)
- Claim 18 independently protects TIDE causal decomposition (usable outside this system)
- Claim 21 independently protects multi-estimator STI framework (usable outside this system)
- Claims 2-5 form a narrowing chain: generic oscillator → RAF → K oscillators → log-spaced → clamp
- If an infringer uses LIF instead of RAF, Claim 1 still covers (says "neuromorphic oscillator units")
- If an infringer skips UV gating, Claims 1 + 14 still cover (clustering is independently claimed)
- If an infringer uses union-find instead of pointer-jumping, Claims 1 + 18 + 21 still cover

### Key Prior Art to Cite in IDS:
- Schmidtke et al. (2010) FPocket — static alpha sphere pocket detection
- Cimermancic et al. (2016) CryptoSite — ML-based cryptic site prediction
- Durrant et al. (2014) MDpocket — MD trajectory pocket tracking
- Izhikevich (2001) — RAF neuron model (foundational, not computational biology)
- Beggs & Plenz (2003) — neuronal avalanches and SOC in neural tissue
- No prior art combining neuromorphic sensing with molecular dynamics exists

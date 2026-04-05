---
name: PRISM-TWIN 7-Layer Architecture
description: Authoritative reference for the PRISM-TWIN coupled interferometric observation system — 7 layers, single engine invocation, AMBER ff14SB
type: architecture
category: architecture
criticality: CRITICAL
owner: Ididia Serfaty
last_verified: 2026-04-05
version: 1.0
---

# PRISM-TWIN 7-Layer Architecture

## CRITICAL CLARIFICATION — read first

**PRISM-TWIN is NOT two-stream MD. It is NOT two engine passes.
It is a coupled interferometric observation system where two
groups of 4-8 independent AMBER ff14SB simulations run
simultaneously within a SINGLE engine invocation.**

Anyone describing PRISM-TWIN as "stream A + stream B", "two-engine run",
"pass 1 + pass 2", or "replica exchange with two copies" is wrong. The
correct mental model is two **coupled observation groups** of N independent
simulations each (N = 4..8), advancing in lockstep within one binary
launch, sharing spike-density and ring-buffered spike histories through
CUDA streams, and cross-correlated in real time on Tensor Cores.

## The 7 Layers

### Layer 1 — Multi-stream fused kernel (4-8 sims per group)

Each observation group contains N = 4..8 independent AMBER ff14SB
trajectories advanced by the fused persistent cooperative kernel
(`nhs_amber_fused.cu` + `persistent_cooperative.cu`). Integration is
BAOAB Langevin with GBSA implicit solvent, SHAKE constraints, and
hydrogen mass repartitioning (HMR). All sims in a group start from the
same topology with independent RNG seeds. The persistent kernel maps
trajectories onto cooperative thread block groups (168 blocks
baseline) and uses `grid.sync()` rather than launch-per-step.

**Do not modify `nhs_amber_fused.cu` for TWIN.** The fused kernel is
frozen by design — any coupling hooks are added at the persistent
kernel role-assignment layer, not inside the integration hot path.

### Layer 2 — Neuromorphic spike detection (multi-LIF, 48-byte records)

Embedded leaky integrate-and-fire (LIF) oscillator networks run
alongside the integrator inside the same persistent kernel. Multiple
LIF populations at different timescales (RAF damped harmonic
oscillators) convert local atomic motion into a sparse spike train.
Each spike is serialized as a 48-byte record containing
`(x, y, z, intensity, timestep, phase, wavelength, water_density,
vib_energy, stream_id)` and streamed to the per-voxel ring buffer.

See `SPIKE_RECORD_FORMAT.md` for field-by-field spec. Key design point:
**neuromorphic detection is not threshold detection on coordinates**;
it is an explicit oscillator network whose firing statistics encode
barrier-crossing dynamics.

### Layer 3 — Coupled observation (ring buffer, adaptive thresholds)

Every `exchange_interval` steps, the two observation groups exchange:

1. **Spike density** (Layer 1 of TWIN coupling): voxelized spike counts
   from group B adjust the firing thresholds of LIF neurons in group A,
   and vice versa. This is the "interferometric" step — detector
   sensitivity in one group is steered by the other group's evidence.
2. **Per-voxel ring buffers** (`ring_buffer.cu`): recent spike histories
   (`RING_SIZE = 256`, ~500 steps) are maintained per voxel for both
   groups. Rotation is lockstep; overflow is logged to a GPU counter
   and handled by drop-oldest.

Exchange runs on a dedicated CUDA stream (`stream_exchange`) so
integration streams never stall. Latency budget per exchange: <5% of
integrator step time. Ring-buffer protocol details are in
`RING_BUFFER_EXCHANGE.md`.

**This is coupled observation, not coupled dynamics.** No force is
transferred between groups. What is transferred is evidence about
where to look.

### Layer 4 — Differential perturbation (A thermal vs B thermal+NMA)

In Phase-2 TWIN runs (`--nma-perturb`), group A runs thermal-only
physics while group B receives normal-mode perturbations on top of the
same thermal bath. Eigenvalue-scaled amplification is applied mode by
mode per `--nma-amplification`, gated by cosine overlap against current
deformation. The A-vs-B differential quantifies the **activation
barrier** at each voxel: modes that shift group B's spike rate but not
group A's are mechanically responsive; modes where both respond are
thermally accessible; modes where neither responds are rigid.

Per-residue outputs include `barrier_classification` (elastic /
cooperative / rigid), `nma_responsive_mode`, `nma_work_at_residue`,
`mechanical_sensitivity`, and `susceptibility_magnitude`. NMA prep +
mode selection are documented separately in `NMA_PERTURBATION.md`.

### Layer 5 — Tensor Core CCF (WMMA, 16 μs, allosteric mapping)

For every voxel, the cross-correlation function
`CCF(τ) = Σ spike_A(t) × spike_B(t + τ) / norm` is computed directly
on Tensor Cores using WMMA intrinsics (`tensor_ccf.cu`). Inputs are
mean-centered FP16; accumulation is FP32; tile shape is chosen to
keep the full protein CCF under ~16 μs per exchange on RTX 5080.

Why WMMA and not cuBLAS: the CCF must be fused with ring-buffer reads
and feature extraction in the same kernel to avoid round-trips to
global memory, which cuBLAS cannot do. The **mean-centering fix** is
essential — raw spike counts are low-rank and saturate FP16 without
centering. See `TENSOR_CORE_CCF.md` for the full derivation and tile
layout.

From the CCF, per-voxel features are derived: `ccf_peak_lag`
(propagation delay → sound speed in the residue network),
`ccf_peak_value` (reproducibility), `ccf_width` (transition sharpness),
`ccf_asymmetry` (causal direction). Aggregated, these are the
**allosteric map** — which residues co-fire, at what lag, with what
confidence.

### Layer 6 — Phase-resolved hysteresis (PRISM-THERM, 5-phase)

All observation runs execute the CCNS (Coupled Cryo-thermal Neuromorphic
Scan) 5-phase protocol: cold-hold → heating → warm-hold → cooling →
cold-hold. Spike statistics are collected per phase. The **hysteresis
asymmetry** — difference between heating-phase and cooling-phase spike
rates at the same temperature — is the PRISM-THERM signal that
classifies pockets as `STABLE`, `METASTABLE`, `CRYPTIC`, or `TRANSIENT`.

Reason this works: genuine cryptic pockets open with a barrier crossing
and close with a different relaxation timescale, producing a signed
asymmetry. Transient fluctuations produce zero asymmetry on average.
The 5-phase schedule and threshold rules are in `CCNS_PROTOCOL.md` and
`HYSTERESIS_ANALYSIS.md`.

### Layer 7 — Per-site characterization (48+ residue features, 60+ site features)

The final per-site object is built by `pocket_profile_builder.py` from
the consensus/differential site list. Each site carries:

- **~48 per-residue features** from TWIN (see "Per-Residue Output Features"
  below): consensus (12), cross-correlation (12), differential (18),
  scout/propagation (8).
- **~60 per-site features**: volume, sphericity, burial, lining-residue
  composition, druggability, therm_class, hysteresis_asymmetry,
  quality_score, spike_count, persistence, pass_fraction, stability,
  anchor points, growth vectors, pocket chemistry profile, and the
  Layer-5 allosteric fingerprint (CCF summary stats).
- Per-protein global features (network statistics, coupling maps).
- Competitive scorecard vs P2Rank / fpocket / PocketMiner (when
  benchmark GT is available).

Ranking within a run is **lexicographic**: persistence → pass_fraction
→ stability → quality. No composite scores. Per-site output schema is
frozen in `OUTPUT_SPEC_COMPLETE.md`.

## Per-Residue Output Features (TWIN, ~50 new)

Full list is maintained in `docs/PRISM_TWIN_INTERFEROMETRIC_DESIGN.md`
(source of truth). Summary:

```
Consensus (12):    spike_agreement_ratio, consensus_intensity_mean,
                   consensus_phase_profile[5], consensus_spatial_coherence,
                   consensus_temporal_onset, n_consensus_neighbors

Cross-correlation (12):
                   ccf_peak_lag, ccf_peak_value, ccf_width, ccf_asymmetry,
                   ccf_per_phase[5], ccf_frequency_peak,
                   ccf_reproducibility, ccf_lag_consistency

Differential (18): b_over_a_spike_ratio, nma_exclusive_count,
                   thermal_exclusive_count, b_over_a_intensity_ratio,
                   nma_responsive_mode, nma_mode_eigenvalue,
                   barrier_classification, per_phase_differential[5],
                   differential_onset_lag, nma_work_at_residue,
                   mechanical_sensitivity, susceptibility_magnitude

Scout/propagation (8):
                   scout_lead_time, scout_predictive_value,
                   phase_offset_enrichment, scout_intensity_at_onset,
                   scout_spatial_propagation, mutual_information,
                   transfer_entropy_a_to_b, causal_flow_direction
```

## What PRISM-TWIN is NOT

- **Not replica exchange.** No temperature ladder, no swap moves, no
  Metropolis criterion. Trajectories never exchange coordinates.
- **Not enhanced sampling.** No biasing potentials, no metadynamics,
  no umbrella sampling. Dynamics are pure AMBER ff14SB.
- **Not two separate engine runs.** A single `nhs_rt_full` invocation
  launches both observation groups. Splitting into two runs loses the
  coupling and the CCF.
- **Not force coupling.** No term in the Hamiltonian of group A depends
  on coordinates in group B. Only the **detectors** are coupled.

## Related SOPs

- `COUPLED_OBSERVATION.md` — why NOT replica exchange, ring buffer protocol
- `RING_BUFFER_EXCHANGE.md` — CUDA shared-memory exchange specifics
- `TENSOR_CORE_CCF.md` — WMMA tile layout, mean-centering
- `NMA_PERTURBATION.md` — Layer 4 perturbation procedure
- `CCNS_PROTOCOL.md` — 5-phase cryo-thermal schedule
- `HYSTERESIS_ANALYSIS.md` — PRISM-THERM classification rules
- `SPIKE_RECORD_FORMAT.md` — 48-byte record layout
- `PERSISTENT_KERNEL.md` — cooperative kernel + role assignment
- `OUTPUT_SPEC_COMPLETE.md` — per-site / per-protein / competitive schema

## Source material

- `docs/PRISM_TWIN_INTERFEROMETRIC_DESIGN.md` — redesigned architecture,
  patent claims, 5-layer coupling detail
- `docs/PRISM_TWIN_DETERMINISM_INVARIANTS.md` — determinism rules
- `docs/USPTO_Patent_Application_v2.md` — claim language

## History

| Date       | Change                          | By              |
|------------|---------------------------------|-----------------|
| 2026-04-05 | Initial version. Synthesized from PRISM_TWIN_INTERFEROMETRIC_DESIGN.md and project memory. | Ididia Serfaty |

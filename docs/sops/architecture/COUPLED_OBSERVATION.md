---
name: Coupled Observation — the TWIN coupling principle
description: Why PRISM-TWIN is coupled observation (not replica exchange, not enhanced sampling), ring buffer protocol, and consensus vs differential classification
type: architecture
category: architecture
criticality: CRITICAL
owner: Ididia Serfaty
last_verified: 2026-04-05
version: 1.0
---

# Coupled Observation — the TWIN coupling principle

## What "coupled" means here

In PRISM-TWIN, two observation groups (each 4-8 independent AMBER ff14SB
simulations) run concurrently inside a single `nhs_rt_full` invocation.
They share information, but **not at the level of the Hamiltonian**.
Specifically, what is coupled is:

1. **Detector sensitivity.** LIF firing thresholds in group A are
   modulated every `exchange_interval` steps by the voxelized spike
   density of group B, and vice versa.
2. **Per-voxel spike history.** Both groups write into a shared
   per-voxel ring buffer; Tensor-Core CCF reads both sides of the
   buffer to compute real-time cross-correlation.

Everything else — positions, velocities, forces, thermostats, RNG
streams — is independent. Each simulation inside each group is its own
trajectory with its own seed, and **no atom in group A ever sees a
force from group B**.

The consequence: TWIN measures **correlation between independent
replicas under identical physics**, not a biased joint ensemble.

## Why NOT replica exchange

Replica exchange MD (REMD / T-REMD / H-REMD) swaps full configurations
(or Hamiltonians) between replicas at controlled intervals, accepted by
a Metropolis criterion. REMD is designed to improve sampling of rare
states by letting low-temperature copies borrow conformations from
high-temperature copies.

TWIN is not doing this. TWIN never swaps anything. TWIN does not have
a temperature ladder. TWIN's two groups run at the **same** temperature
schedule (CCNS 5-phase). The goal is not to improve sampling of a
single trajectory; the goal is to **measure reproducibility and causal
structure** across independent trajectories. REMD would destroy that
— by mixing conformations you lose the ability to ask "did A and B
independently see the same event?".

## Why NOT enhanced sampling

Enhanced sampling (metadynamics, umbrella sampling, ABF, well-tempered
metadynamics, OPES, etc.) adds biasing potentials to a single
trajectory to flatten free-energy barriers. The bias is then removed
in post-processing to recover the unbiased free-energy surface.

TWIN adds zero bias. The dynamics are pure ff14SB with a GBSA implicit
solvent. The spike detection overlay is observational — the LIF
neurons read atomic motion but never write to it. The "hysteresis" and
"cryptic" classifications come from comparing heating-phase and
cooling-phase statistics of the **unbiased** dynamics, not from
reweighting a biased ensemble.

Adding a bias would destroy the PRISM-THERM signal: asymmetry between
heating and cooling phases is only meaningful when both halves are
sampled from the same unbiased distribution.

## What IS coupled — the observation layer

The coupling lives entirely in the **detector** layer:

```
   group A: integration   group B: integration
        │                       │
        ▼ spikes                ▼ spikes
   [voxel ring A] ◄────────► [voxel ring B]
        │      (ring buffer exchange)
        ▼
   detector thresholds A ◄── spike density from B
   detector thresholds B ◄── spike density from A
        │
        ▼
   Tensor Core CCF(A, B) per voxel
        │
        ▼
   consensus/differential classification
```

Because only the detectors are coupled:
- **Determinism is preserved.** A given RNG seed produces the same
  trajectory regardless of what the other group is doing (the
  trajectory does not depend on the detector thresholds — only the
  recorded spike train does).
- **Each group's spike train is still a valid independent sample**
  from the underlying dynamics. CCF is therefore a measurement of
  real reproducibility, not of forced correlation.

## Ring buffer protocol

**Storage.**
- Per-voxel, per-group: `spike_ring[n_voxels][RING_SIZE]`
- `RING_SIZE = 256` captures ~500 integration steps of spike history
- `ring_head[n_voxels]` is the current write position; writes wrap

**Cadence.**
- Each integration step may write up to one spike per voxel per group
  into its own ring.
- Every `exchange_interval` steps, a dedicated CUDA stream runs:
  1. `compute_spike_density(ring_a → density_a)` and B
  2. `compute_ccf(ring_a, ring_b → ccf_map)` on Tensor Cores
  3. `apply_threshold_mod(density_b → thresholds_a)` and vice versa
  4. `rotate_ring_buffers(ring_a, ring_b)` (head advance, not copy)

**Overflow handling.**
- If a voxel fires more than `RING_SIZE` spikes between exchanges, the
  oldest entries are dropped (drop-oldest). A per-voxel overflow
  counter is incremented and emitted at run end for diagnostics.

**Latency budget.**
- Exchange must complete in <5% of the integrator step budget, else
  the multi-stream group stalls. On RTX 5080 with tile-optimized WMMA
  CCF, typical exchange is ~16 μs vs. ~300 μs integrator step — well
  under budget.

Full CUDA implementation is in `ring_buffer.cu`; invariants are
checked by tests in `crates/prism-gpu/tests/ring_buffer_tests.rs`.

## Consensus vs differential classification

Every voxel (and downstream, every site) is classified by the paired
spike statistics:

| Category        | Signal                                                | Interpretation                                                 |
|-----------------|-------------------------------------------------------|----------------------------------------------------------------|
| **Consensus**   | Spike in both A and B at same phase, CCF peak near τ=0, high CCF value | Reproducible event — real conformational feature               |
| **Differential** | Spike in B but not A (B = thermal+NMA, A = thermal) | Mechanically responsive — requires NMA mode to activate        |
| **Thermal-only** | Spike in A but not B at matched phase              | Thermally accessible but NMA-incompatible (rare, often noise)  |
| **Noise**       | Spike in A xor B, low CCF, no phase alignment         | Stochastic fluctuation — discarded                             |
| **Cryptic**     | Consensus spike present only in heating phase (PRISM-THERM hysteresis asymmetry >threshold) | Genuine cryptic pocket — opens on heating, slow close on cool  |

The **ranking** of sites is based on consensus signal (reproducibility)
and, where applicable, hysteresis class. The **mechanism annotation**
(elastic / cooperative / rigid) comes from differential classification.
A site can be both a strong consensus site and a strong differential
site — that is the ideal cryptic allosteric target.

## Related SOPs

- `PRISM_TWIN_ARCHITECTURE.md` — 7-layer overview
- `RING_BUFFER_EXCHANGE.md` — CUDA-level exchange details
- `TENSOR_CORE_CCF.md` — WMMA CCF implementation
- `HYSTERESIS_ANALYSIS.md` — PRISM-THERM classification
- `decisions/WHY_NOT.md` — why-not-replica-exchange, why-not-enhanced-sampling

## Source material

- `docs/PRISM_TWIN_INTERFEROMETRIC_DESIGN.md` — ground truth for coupling layers

## History

| Date       | Change           | By              |
|------------|------------------|-----------------|
| 2026-04-05 | Initial version. | Ididia Serfaty  |

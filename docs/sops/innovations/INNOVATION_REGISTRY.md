---
name: PRISM-4D Innovation Registry
description: Master list of novel innovations with dates, implementation status, and patent classification
type: innovations
category: innovations
criticality: HIGH
owner: Ididia Serfaty
last_verified: 2026-04-05
version: 1.0
---

# PRISM-4D Innovation Registry

Master list of documented innovations. For each innovation, this
registry records what it is, when it was conceived, when it was
implemented, and its current IP classification.

> Classification legend:
> - **PAT** — covered by filed / pending patent application (e.g. DELF-2026-001)
> - **TS**  — trade secret (implementation details kept confidential; see `ip/TRADE_SECRET_REGISTRY.md`)
> - **OSS** — intended for open publication / benchmark paper
> - **MIXED** — concept is patented but implementation specifics are TS

## Innovations

| # | Innovation                           | Conceived   | Implemented | Classification | Reference |
|---|--------------------------------------|-------------|-------------|----------------|-----------|
| 1 | Coupled interferometric MD (TWIN)    | pre-2026-02 | 2026-03/04 (Gates L4-1..L4-4) | PAT            | `COUPLED_INTERFEROMETRIC_MD.md`, `docs/PRISM_TWIN_INTERFEROMETRIC_DESIGN.md` |
| 2 | Neuromorphic spike detection in MD   | pre-2026-02 | In production | MIXED          | `NEUROMORPHIC_SPIKE_DETECTION.md`, `nhs_amber_fused.cu` |
| 3 | Phase-resolved cryo-thermal hysteresis (PRISM-THERM) | pre-2026-02 | In production (`--prism-therm`) | PAT  | `PHASE_RESOLVED_HYSTERESIS.md`, `HYSTERESIS_ANALYSIS.md` |
| 4 | Differential perturbation (A thermal vs B thermal+NMA) | 2026-03 | Gate L4 series | PAT | `DIFFERENTIAL_PERTURBATION.md`, `NMA_PERTURBATION.md` |
| 5 | Tensor Core allosteric CCF (WMMA)    | 2026-03     | Gate L4-4 (commit 0e341f20) | MIXED | `TENSOR_CORE_ALLOSTERIC.md`, `tensor_ccf.cu` |
| 6 | Per-voxel spike ring buffer exchange | 2026-03     | Gate L4-3 (commits ad766192, c61033b2) | TS | `RING_BUFFER_EXCHANGE.md`, `ring_buffer.cu` |
| 7 | Persistent cooperative kernel for TWIN | 2026-03   | Gate L4-1 (commit 3a2899d5) | TS | `PERSISTENT_KERNEL.md`, `persistent_cooperative.cu` |
| 8 | Beacon-guided self-referential MD    | pre-2026-02 | Partial     | PAT            | `BEACON_GUIDED_MD.md` |
| 9 | Physics-constrained LLM (PrismAI)    | 2026-02     | In progress | MIXED          | `PHYSICS_CONSTRAINED_LLM.md` |
| 10| Spike distillation (teacher → student) | 2026-03   | v002 / v003 / v004 GAT | OSS | `SPIKE_DISTILLATION.md`, memory: `architecture_v004_gat.md` |
| 11| Multi-wavelength UV photon probe     | pre-2026-02 | In production | MIXED          | `UV_PHOTON_PERTURBATION.md`, `docs/UV_SPECTROSCOPY_SPEC.md` |
| 12| Elimination cascade (S1..S4)         | 2026-03     | `--cascade` flag (commit referenced 2026-03-21) | OSS | `decisions/DECISION_LOG.md` |
| 13| RAF damped harmonic oscillator LIF   | pre-2026-02 | In production | TS             | `NEUROMORPHIC_SPIKE_ENGINE.md` |
| 14| Voxel density-peak subdivision for mega-clusters | pre-2026-02 | In production | OSS | `rt_clustering.rs` |
| 15| Adaptive epsilon/min_spikes clustering | pre-2026-02 | In production | OSS          | memory: "Key Algorithms" |

## Cross-references

- **Patent filings**: `docs/sops/ip/PATENT_STATUS.md` (docket DELF-2026-001)
- **Prior art**: `docs/sops/ip/PRIOR_ART_ANALYSIS.md`
- **Trade secret details**: `docs/sops/ip/TRADE_SECRET_REGISTRY.md` (gitignored)
- **Inventorship**: `docs/sops/ip/INVENTORSHIP_MEMO.md`

## How to add a new innovation

1. Add a row to the table above with an incremented `#`.
2. Create a per-innovation SOP in `docs/sops/innovations/` if one does
   not exist.
3. If classification includes `PAT`: update `ip/PATENT_STATUS.md`.
4. If classification includes `TS`: update `ip/TRADE_SECRET_REGISTRY.md`.
5. Log in `docs/PRODUCTION_LOGBOOK.md` with the commit SHA of the
   first implementation.

## History

| Date       | Change                                          | By              |
|------------|-------------------------------------------------|-----------------|
| 2026-04-05 | Initial registry — 15 innovations catalogued.   | Ididia Serfaty  |

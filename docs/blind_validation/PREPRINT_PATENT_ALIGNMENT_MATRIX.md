# PREPRINT–PATENT ALIGNMENT MATRIX
**Version:** 1.0  
**Locked:** 2026-05-13 UTC

---

## Purpose

Maps each key blind validation result to the corresponding preprint claim and provisional patent claim, ensuring the blind validation provides supporting evidence for the patent prosecution record.

---

## Core patent claims supported by blind validation

| Claim type | Patent coverage | Blind validation evidence |
|------------|----------------|--------------------------|
| Method claim: perturbation-driven MD pocket detection | CryoUV fast_35k hysteresis protocol → spike sensing → cryptic site detection | SR@k across 10 new targets demonstrates prospective validity |
| Method claim: apo-state-only prediction | No holo coordinates used pre-freeze | Freeze protocol documentation + git commit timestamp |
| Method claim: multi-stream parallelism | --multi-stream 8/20 | run.log documents actual stream count per target |
| Apparatus claim: fused device-resident sensing kernel | Spike detection fused into CUDA kernel | ENGINE_COMMIT hash ties to fused_engine.rs implementation |
| Composition claim: thermodynamic classification | therm_class CRYPTIC/ORTHOSTERIC/ALLOSTERIC from prism_therm.json | Reported per site in validation results |
| Utility: cryptic pocket detection in drug-relevant targets | RAS, kinase, ion channel, PPI, mutant p53, E3 ligase, GPCR | 10-target blind validation covers all classes |

---

## Evidence preservation (per blind target)

For each B01–B10 target, the following documents patent-quality evidence:
1. `run.log` — verifiable execution timestamp, flags, engine version
2. `binding_sites.json` (frozen) — predicted site coordinates at freeze time
3. `FREEZE_MANIFEST_*.sha256` — cryptographic attestation of prediction before holo access
4. `PREDICTION_FREEZE_B*.md` — human-readable freeze attestation in git history
5. `validation_report.md` (post-freeze) — site-vs-holo scoring results

---

## Preprint sections supported

| Preprint section | Blind validation support |
|-----------------|--------------------------|
| Abstract: "prospective-retrospective blind validation" | This entire pipeline |
| Methods §2: CryoUV protocol | LAST10_METHOD_LOCK.md §3 confirms 45,000 steps from source |
| Methods §3: detection parameters | LAST10_METHOD_LOCK.md §4 locks all detector params |
| Results §1: SR@k | BV-F1 and BV-F2 figures |
| Results §2: hard negative | B10 ADRB2 result |
| Results §3: LORO | BV-F6 supplemental |
| Discussion: comparison to geometry-only methods | Baseline comparison table |
| SM2: step count 45,000 | Source-verified in METHOD_DRIFT C1 |

---

## Critical language alignment

The preprint and patent claims must use identical phrasing for:
- "45,000 steps per stream" (NOT 55,000)
- "CryoUV fast_35k hysteresis protocol"
- "spike events detected by device-resident sensing kernel"
- "apo-state candidate generation"
- "prospective-retrospective blind computational validation"

Any preprint or patent text using "55,000 steps" must be corrected before filing.

---

## Date precedence

The blind validation freeze commit timestamp establishes the earliest date of documented prospective prediction performance. This predates any subsequent patent continuation claims referencing these targets.

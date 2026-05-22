# GO/NO-GO READINESS SCORECARD
**Version:** 1.0  
**Locked:** 2026-05-13 UTC

---

## Pre-execution readiness (PHASE 1–3)

| Item | Status | Gate |
|------|--------|------|
| Method lock document complete | ✓ DONE | HARD |
| Pipeline discovery complete | ✓ DONE | HARD |
| Method drift report complete | ✓ DONE | HARD |
| All runbook docs written | — | SOFT |
| BLIND_BASE dir exists on /mnt/storage | — | HARD |
| prism-validate-and-run.sh accessible | — | HARD |
| prism-prep binary accessible | — | HARD |
| fpocket v4.2.3 accessible | — | HARD |
| P2Rank v2.4.2 accessible | — | HARD |
| GPU available (nvidia-smi shows RTX 5080) | — | HARD |
| Disk space ≥ 50 GB free on /mnt/storage | — | HARD |
| No publication target in B01–B10 | ✓ DONE | HARD |
| No holo coordinates accessed | ✓ DONE | HARD |

**GO for Phase 1–3 if:** all HARD items = ✓

---

## Pre-freeze readiness (PHASE 4)

| Item | Status | Gate |
|------|--------|------|
| All 10 engine runs completed (run.log exit 0) | — | HARD |
| All 10 binding_sites.json non-empty | — | HARD |
| All 10 fpocket outputs present | — | HARD |
| All 10 P2Rank outputs present | — | HARD |
| No holo coordinates accessed for any B target | — | HARD |
| Pre-freeze leakage scan clean | — | HARD |
| SHA256 manifests generated for all 10 | — | HARD |

**GO for Phase 4 (freeze) if:** all HARD items = ✓

---

## Post-freeze readiness (PHASE 5+)

| Item | Status | Gate |
|------|--------|------|
| GLOBAL_PREDICTION_FREEZE.md committed | — | HARD |
| All 10 SHA256 manifests verify clean | — | HARD |
| Validator script accessible | — | HARD |
| BLIND_HOLO_REFERENCES.md now accessible | — | HARD |

**GO for Phase 5+ if:** all HARD items = ✓

---

## Author action items (blocking if unresolved)

| Item | Blocking? | Status |
|------|-----------|--------|
| Fix PRISM4D_Complete_Technical_Reference.md 55k→45k | NO (blind val can proceed) | OPEN |
| Confirm 7ATA p53 holo reference | NO (only affects MCL1/p53 pub scoring) | OPEN |
| Confirm M4R run status in preprint | NO | OPEN |

---

## Null controls readiness

| Item | Status | Notes |
|------|--------|-------|
| pair_breaking_null.py implemented | — | NOT IN REPO — implement or defer |
| permutation_null.py implemented | — | NOT IN REPO |
| Minimum iterations: 1,000 | — | Reduced power, document explicitly |

If null controls unavailable: report SR@k without empirical p-value. Flag as gap.

---

## Final report gate

| Item | Status |
|------|--------|
| SR@k table populated | — |
| LORO table populated | — |
| Family collapse table populated | — |
| Null control results or explicit gap documented | — |
| Hard negative (ADRB2) result documented | — |
| Baseline comparison (fpocket, P2Rank) included | — |

GO for submission when all final report items complete.

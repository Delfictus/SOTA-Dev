# PRISM4D BLIND VALIDATION — MASTER RUNBOOK
**Version:** 1.0  
**Locked:** 2026-05-13 UTC  
**Repo HEAD:** f8f368f6b83e118126e691626a823866e49906f5

---

## Executive summary

This runbook orchestrates a prospective-retrospective blind computational validation of the PRISM4D cryptic pocket detection engine against 10 new targets (B01–B10) not in the publication run. The validation reproduces the exact publication methodology (locked in `LAST10_METHOD_LOCK.md`) and scores against withheld holo coordinates accessed only post-freeze.

---

## Phase map

| Phase | Name | Pre/Post-freeze | Key output |
|-------|------|----------------|------------|
| -1 | Setup + method lock verification | PRE | docs/blind_validation/* committed |
| 0 | Target selection | PRE | TARGET_SELECTION_PROTOCOL.md populated |
| 1 | Structure prep | PRE | 10× clean PDB + topology.json + nma_modes.json |
| 2 | Engine runs | PRE | 10× binding_sites.json + kcc outputs |
| 3 | Baseline runs | PRE | 10× fpocket + P2Rank outputs |
| 4 | Prediction freeze | PRE→POST boundary | GLOBAL_PREDICTION_FREEZE.md committed |
| 5 | Post-freeze scoring | POST | prism_vs_holo.csv + baseline_vs_holo.csv |
| 6 | LORO + family | POST | loro_results.csv + family_results.csv |
| 7 | Null controls | POST | null_control_results.csv |
| 8 | Visualization | POST | pymol_overlay.pml per target |
| 9 | Final report | POST | BLIND_VALIDATION_FINAL_REPORT.md |

---

## Phase -1: Setup

- [x] docs/blind_validation/ directory created
- [x] PIPELINE_DISCOVERY_REPORT.md written
- [x] LAST10_METHOD_LOCK.md written
- [x] METHOD_DRIFT_AND_CONFLICT_REPORT.md written
- [x] All 25 runbook documents written
- [ ] CLAUDE.md doc corrections (55k→45k) — AUTHOR ACTION REQUIRED
- [ ] 7ATA p53 holo reference confirmed — AUTHOR ACTION REQUIRED

---

## Phase 0: Target selection

Targets B01–B10 selected per `TARGET_SELECTION_PROTOCOL.md`:
1. B01 — HRAS_Q61H (RAS-family, PDB 4L9S)
2. B02 — CDK2_allosteric (Kinase, PDB 1HCL)
3. B03 — Kv1.2 (Ion channel, PDB 3LUT)
4. B04 — MDM2 (PPI, PDB 1Z1M)
5. B05 — TP53_R175H (Mutant TP53, PDB 2OCJ)
6. B06 — cGAS (STING-adjacent, PDB 4KM5)
7. B07 — TEAD1 (TEAD/nuclear, PDB 3KYS)
8. B08 — CRBN (E3 ligase, PDB 4TZ4)
9. B09 — Thrombin_exosite (Protease exosite, PDB 1HAH)
10. B10 — ADRB2 (GPCR hard negative, PDB 2RH1)

---

## Phase 1: Structure preparation

For each target: download apo PDB → prism-clean.py → prism-prep → NMA modes.  
See `PRISM4D_BLIND_VALIDATION_COMMAND_TEMPLATES.md` Phase 1–2 for exact commands.

Checklist (per target): clean PDB exists + residue diversity ≥15 verified + topology.json + nma_modes.json

---

## Phase 2: Engine runs

Locked flags (from `LAST10_METHOD_LOCK.md §2`):
```
--fast --hysteresis --prism-therm --multi-stream [8|20] --multi-scale
--spike-percentile 50 --hmr --adaptive-dt --fused-steps 6
--nma-perturb <nma_modes.json> --nma-amplification 3.0 --replica-seed 42 -v
```
Run each target in background. Verify binding_sites.json is non-empty after completion.

---

## Phase 3: Baseline runs

fpocket and P2Rank on each clean apo PDB. Both invoked per `PRISM4D_BLIND_VALIDATION_COMMAND_TEMPLATES.md` Phase 4–5.

---

## Phase 4: Prediction freeze

Per `PREDICTION_FREEZE_PROTOCOL.md`:
1. Copy prediction artifacts → frozen/ dir
2. SHA256 manifest generated
3. chmod 555 frozen dir
4. Per-target PREDICTION_FREEZE_B*.md written
5. GLOBAL_PREDICTION_FREEZE.md assembled
6. Git commit (forensic timestamp)

**This commit is the irrevocable PRE→POST freeze boundary.**

---

## Phase 5–9: Post-freeze

Per `POST_FREEZE_VALIDATION_PROTOCOL.md`. Open holo references only after freeze commit verified.

Runs `prism_pub_baseline_validator.py` per target → aggregate → SR@k → LORO → family collapse → null controls → final report.

---

## Deviations from publication run (documented)

| Deviation | Value | Justification |
|-----------|-------|---------------|
| --replica-seed 42 | ADDED (not in pub runs) | Blind validation internal reproducibility |
| spike-percentile | 50 for all (no MCL1 anomaly) | Uniform; MCL1 anomaly not reproduced |
| NMA modes | Generated fresh (not from pub run paths) | New targets |
| --multi-differential etc. | OMITTED | Matches pub run flag set for comparability |

---

## Document index

See `PRISM4D_BLIND_VALIDATION_FILE_MAP.md` for all 25 supporting documents.

# BLIND VALIDATION FILE MAP
**Version:** 1.0  
**Locked:** 2026-05-13 UTC

---

## Repository docs (docs/blind_validation/)

| File | Status | Purpose |
|------|--------|---------|
| PRISM4D_BLIND_VALIDATION_RUNBOOK.md | ✓ | Master orchestration runbook |
| PRISM4D_BLIND_VALIDATION_CHECKLIST.md | ✓ | Per-phase execution checklist |
| PRISM4D_BLIND_VALIDATION_FILE_MAP.md | ✓ | This file — all artifact locations |
| PRISM4D_BLIND_VALIDATION_COMMAND_TEMPLATES.md | ✓ | Exact commands per phase |
| PRISM4D_BLIND_VALIDATION_AUDIT_SCHEMA.md | ✓ | Audit trail schema |
| PIPELINE_DISCOVERY_REPORT.md | ✓ | Full pipeline discovery from pub runs |
| LAST10_METHOD_LOCK.md | ✓ | Locked 10-target publication method |
| METHOD_DRIFT_AND_CONFLICT_REPORT.md | ✓ | 8 conflicts documented + resolved |
| TARGET_SELECTION_PROTOCOL.md | ✓ | B01–B10 target selection criteria |
| LEAKAGE_THREAT_MODEL.md | ✓ | 7 leakage threat vectors + mitigations |
| PRE_FREEZE_LEAKAGE_SCAN_PROTOCOL.md | ✓ | Automated leakage scan commands |
| PREDICTION_FREEZE_PROTOCOL.md | ✓ | Per-target + global freeze procedure |
| POST_FREEZE_VALIDATION_PROTOCOL.md | ✓ | Post-freeze scoring pipeline |
| STATISTICAL_ANALYSIS_PLAN.md | ✓ | Pre-registered SAP |
| HARD_NEGATIVE_PROTOCOL.md | ✓ | ADRB2 hard negative scoring |
| BASELINE_FAIRNESS_PROTOCOL.md | ✓ | Fairness constraints for baselines |
| DECOY_AND_NULL_CONTROL_PROTOCOL.md | ✓ | Null control implementation spec |
| STRUCTURAL_VISUALIZATION_RUNBOOK.md | ✓ | PyMOL/ChimeraX figure generation |
| FIGURE_GENERATION_PLAN.md | ✓ | Figure plan for report |
| REPORTING_LANGUAGE_GUIDE.md | ✓ | Precise language for claims |
| PREPRINT_PATENT_ALIGNMENT_MATRIX.md | ✓ | Preprint ↔ patent claim alignment |
| BIOTECH_DILIGENCE_OUTPUT_SPEC.md | ✓ | Due diligence package spec |
| NONPROVISIONAL_EVIDENCE_PRESERVATION_MAP.md | ✓ | Patent filing evidence map |
| GO_NO_GO_READINESS_SCORECARD.md | ✓ | Pass/fail gate checklist |
| OPEN_QUESTIONS_AND_MISSING_ARTIFACTS.md | ✓ | Known gaps + author actions |

---

## Frozen predictions (docs/blind_validation/frozen_predictions/)

| File | Status | Contents |
|------|--------|---------|
| PREDICTION_FREEZE_B01_HRAS_Q61H.md | — | Per-target freeze attestation |
| PREDICTION_FREEZE_B02_CDK2_allosteric.md | — | |
| PREDICTION_FREEZE_B03_Kv1.2.md | — | |
| PREDICTION_FREEZE_B04_MDM2.md | — | |
| PREDICTION_FREEZE_B05_TP53_R175H.md | — | |
| PREDICTION_FREEZE_B06_cGAS.md | — | |
| PREDICTION_FREEZE_B07_TEAD1.md | — | |
| PREDICTION_FREEZE_B08_CRBN.md | — | |
| PREDICTION_FREEZE_B09_Thrombin_exosite.md | — | |
| PREDICTION_FREEZE_B10_ADRB2.md | — | |
| GLOBAL_PREDICTION_FREEZE.md | — | Global freeze aggregation |

---

## Post-freeze validation (docs/blind_validation/post_freeze_validation/)

| File | Status | Contents |
|------|--------|---------|
| BLIND_VALIDATION_FINAL_REPORT.md | — | Master results report |
| AGGREGATE_PRISM_VS_HOLO.csv | — | PRISM4D scoring across all targets |
| AGGREGATE_BASELINE_VS_HOLO.csv | — | fpocket + P2Rank scoring |
| NULL_CONTROL_RESULTS.csv | — | Null permutation results (if available) |

---

## Reports (docs/blind_validation/reports/)

| File | Status | Contents |
|------|--------|---------|
| BLIND_VALIDATION_FINAL_REPORT.md | — | Final report for publication supplement |

---

## Scratch/output on /mnt/storage

```
/mnt/storage/prism-outputs/blind_validation/
  B01_HRAS_Q61H/
    prep/          ← apo PDB + clean PDB
    topologies/    ← topology.json + nma_modes.json
    run/           ← engine output (binding_sites.json, kcc, trajectory, therm)
    fpocket/       ← fpocket output
    p2rank/        ← P2Rank output
    frozen/        ← read-only freeze artifacts + SHA256 manifest
  B02_CDK2_allosteric/  [same structure]
  ...
  B10_ADRB2/
  post_freeze_validation/
    B01_HRAS_Q61H/    ← validator output (prism_vs_holo.csv, loro, family)
    ...
    AGGREGATE_*.csv
    NULL_CONTROL_RESULTS.csv
```

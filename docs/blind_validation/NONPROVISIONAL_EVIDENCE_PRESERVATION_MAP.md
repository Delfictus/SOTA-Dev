# NONPROVISIONAL EVIDENCE PRESERVATION MAP
**Version:** 1.0  
**Locked:** 2026-05-13 UTC

---

## Purpose

Maps each blind validation artifact to the corresponding evidence required for nonprovisional patent prosecution. Establishes chain of custody from prediction to validation.

---

## Evidence categories

### Category A: Reduction to practice (inventorship record)

| Artifact | Location | Legal role |
|----------|----------|-----------|
| git commit history (all commits) | ~/Desktop/Prism4D-bio/.git | First date of code changes |
| freeze git commit | HEAD after Phase 4 | First date of documented predictive performance |
| run.log timestamps | /mnt/storage/prism-outputs/blind_validation/B*/run/run.log | Execution date record |
| FREEZE_MANIFEST_*.sha256 | frozen/ dirs | Hash-attested prediction state |

### Category B: Novelty support

| Artifact | Location | Legal role |
|----------|----------|-----------|
| METHOD_DRIFT_AND_CONFLICT_REPORT.md | docs/blind_validation/ | Documents method was original (not derived from holo coords) |
| PRE_FREEZE_LEAKAGE_SCAN_PROTOCOL.md scan results | LEAKAGE_SCAN_*.txt | Demonstrates no information leakage |
| LAST10_METHOD_LOCK.md | docs/blind_validation/ | Locked methodology pre-dates any third-party disclosure |

### Category C: Non-obviousness support

| Artifact | Location | Legal role |
|----------|----------|-----------|
| Baseline comparison (fpocket/P2Rank) | post_freeze_validation/ | Demonstrates performance over conventional geometry-only methods |
| LORO results | loro_results.csv per target | Demonstrates cross-conformation generalization |
| Hard negative (ADRB2) | B10 result | Demonstrates specificity (non-trivial) |
| SR@k table | BV-F1 | Quantitative performance record |

### Category D: Utility support

| Artifact | Location | Legal role |
|----------|----------|-----------|
| 10-target blind validation across diverse classes | Full report | Demonstrates broad utility in drug-relevant targets |
| Per-target validation cards | Tier 1 diligence | Maps specific utility to specific protein classes |
| Therm classification (CRYPTIC) | prism_therm.json | Documents thermodynamic basis for cryptic site identification |

---

## Preservation requirements

1. **Immutable copies:** All frozen artifacts in `frozen/` dirs are chmod 555. Do NOT modify or overwrite.
2. **git history:** Do NOT rebase or force-push the branch after Phase 4 freeze commit. Git history is evidence.
3. **Run logs:** Keep run.log files permanently. Do NOT delete even after campaign completion.
4. **Hash records:** FREEZE_MANIFEST sha256 files are legal documents — treat as such.
5. **R2 backup:** Archive frozen predictions and logs to R2 `prism-archive` bucket after validation completion.

```bash
# Archive to R2 after completion
rclone copy /mnt/storage/prism-outputs/blind_validation/ \
    r2:prism-archive/blind_validation_$(date +%Y%m%d)/ \
    --progress
```

---

## Chain of custody statement (for patent prosecution file)

"Predictions for blind targets B01–B10 were generated from apo crystal structures using the PRISM4D engine (commit f8f368f6) under methodology locked in LAST10_METHOD_LOCK.md. No holo or ligand-bound coordinate information was accessed before the prediction freeze (git commit [FREEZE_HASH], [FREEZE_TIMESTAMP]). SHA256 manifests of all prediction artifacts were generated at freeze time and are verifiable from the git history. Post-freeze scoring was conducted using identical methodology to the publication run validation. All artifacts are preserved at [R2 archive path] and verified against their SHA256 hashes."

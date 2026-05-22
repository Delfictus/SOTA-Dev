# METHOD DRIFT AND CONFLICT REPORT
**Generated:** 2026-05-13 UTC  
**Repo HEAD:** f8f368f6b83e118126e691626a823866e49906f5

---

## Conflict triage table

| ID | Conflict | Severity | Halt execution? | Recommended locked value |
|----|----------|----------|----------------|--------------------------|
| C1 | 45k vs 55k steps/stream | CRITICAL | YES — doc must be corrected | 45,000 (source code) |
| C2 | spike-percentile 50 vs 70 | HIGH | NO — document and proceed | 50 for blind; document MCL1 anomaly |
| C3 | --nma-perturb absent from CLAUDE.md | MEDIUM | NO | Include --nma-perturb in blind runs |
| C4 | --multi-differential etc. absent from pub runs | MEDIUM | NO | Omit from blind runs for comparability |
| C5 | --replica-seed absent from pub runs | LOW | NO | Add --replica-seed 42 to blind runs |
| C6 | Null controls absent from pub runs | HIGH | NO — implement before final report | Implement pair-breaking null |
| C7 | M4R as 10th target not in preprint | LOW | NO | Clarify with author |
| C8 | 7ATA HTTP 404 | LOW | NO | Author confirms substitute |

---

## C1 — STEP COUNT CONFLICT (CRITICAL)

**Conflicting values:**  
- Source (authoritative): `crates/prism-nhs/src/fused_engine.rs`, `fast_35k()` function  
  - cold_hold_steps = 14,000  
  - ramp_up_steps = 6,000  
  - warm_hold_steps = 15,000  
  - ramp_down_steps = 6,000  
  - cold_return_steps = **4,000**  
  - **TOTAL = 45,000 steps/stream**
- Documentation (wrong): `docs/PRISM4D_Complete_Technical_Reference.md`, lines 237-238, 249  
  - cold_return: 14,000 steps  
  - "TOTAL: 55,000 steps/stream × 8 streams = 440,000 MD steps"  
  - "engine.run(55000 steps)"

**Probable cause:** Technical Reference was written with an earlier protocol version where cold_return_steps=14,000. The `fast_35k()` function was subsequently tuned to cold_return_steps=4,000, reducing total from 55k to 45k. The doc was not updated.

**Affected artifacts:** Preprint supplement (SM2 says "CryoUV fast_35k hysteresis protocol" — correctly describes total but the doc separately says 55k), Technical Reference, any figure captions mentioning step count.

**Action required:** Author must update PRISM4D_Complete_Technical_Reference.md lines 237-238, 249. The preprint SM2 text (already corrected to 45,000 total) takes precedence. Do NOT use 55,000 in any blind validation documentation.

**Execution halt:** YES — update the doc before any external communication. Blind validation execution can proceed using 45,000-step method; run scripts already use this implicitly via `--fast --hysteresis`.

---

## C2 — SPIKE PERCENTILE CONFLICT (HIGH)

**Conflicting values:**  
- MCL1 pub run script: `--spike-percentile 70`
- All other 9 pub run scripts (KRAS_G12C, Kv31, p53_Y220C, AKT1, TEAD3, TRPV1, GLP1R, STING, M4R): `--spike-percentile 50`
- CLAUDE.md canonical command: `--spike-percentile 70`

**Evidence:**
```
run_08_MCL1_chainA.sh:     --spike-percentile 70
run_01_KRAS_G12C_chainA.sh: --spike-percentile 50
run_02_Kv31_chainA_PRIMARY.sh: --spike-percentile 50
... (all others: 50)
```

**Probable cause:** MCL1 was run separately or an earlier CLAUDE.md canonical was followed for MCL1 while the publication batch standardized on 50. Alternatively, MCL1 was run after a parameter revision.

**Impact on results:** spike-percentile controls what fraction of timesteps are flagged as spike events. Percentile=50 retains more events (lower threshold = more sensitivity). Percentile=70 retains fewer, higher-amplitude events (more specificity). This affects site count, top-ranked sites, and family membership.

**Affected artifacts:** MCL1 results may not be directly comparable to the other 9 targets. Preprint should note this if MCL1 results are presented alongside others.

**Recommended locked value for blind validation:** `--spike-percentile 50` (matches 9/10 targets, the dominant practice). Apply uniformly across all blind targets.

**Execution halt:** NO. Proceed with 50.

---

## C3 — NMA FLAGS IN PUB RUNS, ABSENT FROM CLAUDE.MD (MEDIUM)

**Conflicting values:**  
- Pub run scripts (all 10): include `--nma-perturb <nma_modes.json> --nma-amplification 3.0 --multi-scale`
- CLAUDE.md canonical command: does NOT include these flags

**Evidence (from run_01_KRAS_G12C_chainA.sh):**
```bash
--multi-scale \
--nma-perturb "$PREP_DIR/topologies/KRAS_G12C_chainA_nma_modes.json" \
--nma-amplification 3.0 \
```

**Probable cause:** CLAUDE.md was not updated when NMA-perturbed multi-scale protocol was adopted for the publication runs.

**Impact:** The NMA perturbation applies normal-mode-derived structural perturbations to seed more diverse conformational sampling. Results obtained with NMA differ from those without. Preprint results used NMA; CLAUDE.md canonical does not specify it.

**Recommended locked value for blind validation:** Include `--nma-perturb` and `--multi-scale` to match the publication methodology.

**Required:** NMA modes JSON must be generated for each blind target during structure preparation.

**Execution halt:** NO. Generate NMA modes as part of prep pipeline.

---

## C4 — FLAGS IN CLAUDE.MD ABSENT FROM PUB RUNS (MEDIUM)

**Flags in CLAUDE.md canonical that are NOT in pub run scripts:**
- `--multi-differential`
- `--closed-loop-steering`
- `--asymmetric-steering`
- `--use-xgb-ranker`
- `--replica-seed 42`

**Probable cause:** These flags represent capabilities added to the engine after the publication runs were locked. The CLAUDE.md canonical reflects a later or more complete configuration. The v1.1-frozen build may not support all of them.

**Impact on blind validation:** For maximum comparability with preprint results, blind validation should use the pub run flag set. The XGBoost ranker and steering flags were not used in generating the preprint results.

**Recommended locked value:** Omit from blind runs. Use pub-run-matched flags only. If these flags are desired in a supplemental "upgraded protocol" panel, note it explicitly as a methodological deviation.

**Execution halt:** NO.

---

## C5 — REPLICA SEED NOT SET IN PUB RUNS (LOW)

**Values:**
- Pub run scripts: no --replica-seed flag present
- CLAUDE.md canonical: `--replica-seed 42`

**Impact:** Exact numerical reproduction of pub run results is not guaranteed. Statistical results should be reproducible within expected run-to-run variation.

**Recommended:** Set `--replica-seed 42` in all blind runs for internal reproducibility.

**Execution halt:** NO.

---

## C6 — NULL CONTROLS ABSENT FROM PUB RUN (HIGH)

**Status:** No pair-breaking null script, no decoy surface patch script, no permutation test script found in the repository.

**Impact on validation credibility:** Without null controls, SR@k results cannot be statistically distinguished from chance. The publication validation (prism_pub_baseline_validator.py) reports LORO and family coverage but does not compute empirical p-values.

**Required for blind validation:** Implement:
1. Strict rank permutation null (10,000 iterations)
2. Shell pair-breaking null (10,000 iterations)
3. Family recurrence null (10,000 iterations)

**Interim:** If full null controls cannot be run before blind validation results are needed, run with minimum 1,000 iterations and clearly document the reduced statistical power.

**Execution halt:** NO for initial execution. Null control implementation should run in parallel and be included in final report.

---

## C7 — M4R AS 10TH TARGET NOT IN PREPRINT (LOW)

**M4R (5DSG)** is run #10 in the publication run manifest but does not appear in the preprint's primary 9-target panel (KRAS G12C, STING, AKT1, MCL1, p53 Y220C, TEAD3, TRPV1, Kv3.1, GLP1R).

**Status:** M4R output directory not found at `/mnt/storage/prism-outputs/runs/`. Run may have failed or been discarded.

**Impact:** M4R may be the 10th element of a planned but not-yet-published supplemental panel, or it may have been a test run.

**Action:** Author to clarify whether M4R results exist and whether they should be included in the retrospective control panel.

---

## C8 — 7ATA HTTP 404 (LOW)

**p53_Y220C holo reference 7ATA** returned HTTP 404 from RCSB during the prior validation run (2026-05-12).

**Probable cause:** PDB ID may be incorrect, retracted, or the PDB entry may have been superseded.

**Action:** Author to confirm correct PDB ID for the aminobenzimidazole-stabilized p53 Y220C structure, or confirm the reference should be excluded.

**Execution halt:** NO. Skip 7ATA in scoring; report as failed holo reference.

---

## Figure correction flags

The following figure/documentation corrections are required before publication:

| Item | Issue | Correction |
|------|-------|-----------|
| PRISM4D_Complete_Technical_Reference.md L237-238 | States 55,000 steps | Correct to 45,000; cold_return=4,000 not 14,000 |
| PRISM4D_Complete_Technical_Reference.md L249 | "engine.run(55000 steps)" | Correct to 45,000 |
| Any figure labeling "55,000 steps" | Wrong step count | Correct to 45,000 |
| Any figure or text with spike-percentile=70 as universal | Not universal in pub runs | Note 50 was used for 9/10 targets |

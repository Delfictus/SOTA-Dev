# Directive 12 Bug Hunter

Verdict: NO CRITICAL/HIGH findings remain.

Commit audited: `124edae9bb81d7c245cf8a41724f1e88cbf7b706`

Validation evidence:
- Worktree was clean during bug hunt.
- Tracked artifact: `100` rows, `15` lock-positive rows, all `PHASE_RESOLVED`, minimum distinct phase count `4`.
- No-lock rows: `85` rows, all `REPLICATED_AGGREGATE`.
- Re-running the rescore script on the already-rescored parquet to `/mnt/storage/prism-scratch/d12_bug_hunt_idempotency/` was byte-stable on key reward and phase columns.
- D03/D11 regression checks:
  - `oracle_scorer --help` contains `--live-scoring`.
  - `PYTHONPATH=src python3 -m pytest tests/test_rescore_top100_lock_mask.py tests/test_oracle_contract.py tests/test_wt_projection_parity.py -q` -> `37 passed`.

BUG_1:
- severity: MEDIUM
- location: `scripts/rescore_top100_lock_mask.py:64-67`
- description: The tracked Top100 input and rescored artifact do not contain `score_atom_offset`, so the script defaults every proposal to offset `0`.
- reproduction: Read both parquet schemas; neither has `score_atom_offset`.
- impact: D12 lock phase provenance is valid for the artifact as scored, but future consumers cannot distinguish full-product scoring from fragment-only scoring.
- suggested_fix: Persist `score_atom_offset` in candidate artifacts, or require it when coordinate payloads include scaffold context.

BUG_2:
- severity: MEDIUM
- location: `crates/prism-forge/src/bin/oracle_scorer.rs:729-748`
- description: `pi_clash_lock_per_phase` is still an alias of `lock_occupancy_per_phase`.
- reproduction: Artifact check showed `pi_clash_lock_*` equals `lock_occupancy_*` for all rows.
- impact: D12 occupancy provenance is satisfied, but downstream code may misread the `pi_clash_lock_*` columns as clash-weighted phase penalties.
- suggested_fix: Keep occupancy columns as source of truth; either rename/deprecate `pi_clash_lock_per_phase` or compute a real phase clash vector separately.

BUG_3:
- severity: MEDIUM
- location: `crates/prism-forge/src/bin/oracle_scorer.rs:850-856`
- description: The survivor-lookup bifurcation branch still labels no-lock rows with `[0.0; 5]` as `PHASE_RESOLVED`.
- reproduction: Code path returns `([0.0; 5], "PHASE_RESOLVED")` when `lock_atom_count == 0`.
- impact: Not present in the tracked D12 live-rescored artifact, but survivor-mode consumers can still receive misleading no-lock provenance.
- suggested_fix: Return `REPLICATED_AGGREGATE` or a distinct `NO_LOCK_PHASE_DATA` provenance for no-lock rows.

BUG_4:
- severity: LOW
- location: `crates/prism-forge/src/scoring/mod.rs:720-731`
- description: Signal-grid float loading accepts nonfinite `cold_mean`/`warm_mean` values; phase outputs could become NaN even if reward validation catches only reward invalidity.
- reproduction: Code inspection: `optional_f64_value` returns raw `Float64Array` values without an `is_finite()` guard.
- impact: Only affects corrupt grid inputs; current tracked artifact is finite.
- suggested_fix: Reject or zero nonfinite signal-grid phase values at load time.

# Track B Chronology Observability Auditor

Verdict: PASS.

Verified evidence:
- `chronology_locked_training_report.json`
  - `runtime_training_mode=LIVE_ORACLE_CANDIDATE_POLICY_TB`
  - `live_oracle_runtime_scored=true`
  - `optimizer_steps=50`, `epochs_completed=50`
  - `coordinate_generation_mode=SIGNAL_GRID_ACTIVATION_TEMPLATE_L3_DERIVED`
  - finite TB loss and reward mean
  - `unique_smiles_total=256`, `dot_smiles_count_total=0`
- `chronology_locked_top_100_candidates.parquet`
  - `100/100 survival_tier=live_signal_grid`
  - `100/100 live_oracle_runtime_scored=true`
  - `100/100 coordinate_generation_mode=SIGNAL_GRID_ACTIVATION_TEMPLATE_L3_DERIVED`
  - `pi_complement` mean `3.0`, nonzero `100/100`
  - `reward` mean `3.7384`, nonzero `100/100`
  - `sigma_shear` mean `356.7116`, nonzero `100/100`
  - `u_pose` max `1.7918`, nonzero `88/100`
  - `u_pose_provenance=best_rotamer_rank_proxy_from_track_a_survivors`
  - `u_pose` equals `u_pose_input` for all rows
- `transition_chronology_tensor.parquet`
  - `3200` rows
  - required provenance columns present
  - `true_md_step` has no nulls, `17` unique values, max `7000`
  - file-order step decreases: `1503`, so chronology is not row-order-derived
  - source split: `1600` BOCPD survival regimes, `1600` kinetic strain events
  - `3200/3200 provenance_class=L4_RUNTIME_TELEMETRY`
- Continuity maps:
  - NMA: `674` rows, `L3_DERIVED`, nonzero displacement/risk
  - Thermodynamic: `11852` rows, `L3_DERIVED`, nonzero hysteresis/trap risk
  - Hydration: `1` row, `L0_MISSING`, `blocked_with_hard_evidence=true`

Findings:
- CRITICAL: none.
- HIGH: none.
- MEDIUM: `pi_complement` is constant at `3.0` across Top 100. This is consistent with declared L3 activation-template coordinate generation, but it is not candidate-pose-specific complement variation.
- LOW: `pathway_bonus` remains zero across Top 100.
- LOW: `lock_phase_provenance` is `REPLICATED_AGGREGATE` for all Top 100; do not use these candidates to claim phase-resolved lock occupancy.
- LOW: hydration continuity remains honestly blocked as `L0_MISSING`.

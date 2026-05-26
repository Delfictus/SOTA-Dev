# Directive 09 Enforcement Audit

Verdict: PASS

Committed HEAD audited: `36337262da39e049847fbf49f430aa3253d551e8`

## Scope
Directive 09 requires generated GPU dispatch scripts to run four channels with bifurcation controls:
- `signal_grid_differential`
- `warp_jacobian`
- `hysteresis_analysis`
- `pathway_analysis`
- `--bifurcate` where relevant
- correct timestep extraction
- generated scripts must be syntactically valid

## Evidence
- Generated scripts contain all required channel commands plus `--bifurcate`, `assert_candidate_raw_root`, and `--raw-output-dir`.
- `scripts/generate_gpu_dispatch_batch.py` emits candidate-specific raw-root defaults and fail-closed raw-root checks.
- `scripts/process_gpu_dispatch_results.py` derives timestep frames from the real protocol summary with `condition_window_count=4`, `equilibrated_count=17`, `ramp_count=15`.
- `crates/prism-nhs/src/bin/signal_grid_differential.rs` applies `--frame-scope` filtering.
- `crates/prism-nhs/src/bin/warp_jacobian.rs` applies `--frames` filtering through protocol `current_step` metadata.
- `crates/prism-nhs/src/bin/hysteresis_analysis.rs` and `pathway_analysis.rs` consume bifurcation/phase controls in runtime score computation.

## Commands Run
```bash
PYTHONPATH=src python3 -m pytest -p no:cacheprovider tests/test_generate_gpu_dispatch_batch.py -q
# 13 passed in 1.78s

cargo check -p prism-nhs --bin signal_grid_differential --bin warp_jacobian --bin hysteresis_analysis --bin pathway_analysis
# exit 0

python3 scripts/process_gpu_dispatch_results.py \
  --mode timestep_extraction \
  --protocol-state-summary campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/protocol_state_summary.parquet \
  --output /tmp/d09_final_real_timesteps.json
# condition_window_count 4
# equilibrated_count 17
# ramp_count 15

for f in campaigns/glp1r_aleniglipron/track_a_generative/gpu_dispatch/launch/*.sh \
         campaigns/glp1r_aleniglipron/track_a_generative/gpu_dispatch/launch_corrected/*.sh; do
  bash -n "$f"
  grep -q -- signal_grid_differential "$f"
  grep -q -- warp_jacobian "$f"
  grep -q -- hysteresis_analysis "$f"
  grep -q -- pathway_analysis "$f"
  grep -q -- bifurcate "$f"
  grep -q -- assert_candidate_raw_root "$f"
  grep -q -- --raw-output-dir "$f"
done
# STATIC_D09_SCRIPT_GATE_PASS scripts=6

bash scripts/regression_gate.sh "directive-09-rehunt-hardening"
# === REGRESSION GATE PASSED for directive-09-rehunt-hardening ===
# Root pytest: 1138 passed, 14 skipped
```

## Verdict
D09 is VERIFIED_RUNTIME for the explicit enforcement checklist.

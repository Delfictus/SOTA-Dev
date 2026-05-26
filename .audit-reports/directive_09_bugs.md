# Directive 09 Bug Hunt

Verdict: PASS WITH NO CRITICAL/HIGH FINDINGS

Committed HEAD audited: `36337262da39e049847fbf49f430aa3253d551e8`

## Previous Blocking Findings Retested

### Multi-window protocol summary
```bash
python3 scripts/process_gpu_dispatch_results.py \
  --mode timestep_extraction \
  --protocol-state-summary campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/protocol_state_summary.parquet \
  --output /tmp/d09_final_real_timesteps.json
# condition_window_count 4
# equilibrated_count 17
# ramp_count 15
```
Result: fixed.

### Empty signal-grid parquets
```bash
cargo run -q -p prism-nhs --bin hysteresis_analysis -- --candidate-id cand_empty --signal-grid /tmp/empty.parquet --output /tmp/empty_h.json --bifurcate pocket
# EMPTY_H_EXIT:1; EMPTY_H_FAIL_CLOSED

cargo run -q -p prism-nhs --bin pathway_analysis -- --candidate-id cand_empty --signal-grid /tmp/empty.parquet --output /tmp/empty_p.json --bifurcate pathway --phase-filter 1
# EMPTY_P_EXIT:1; EMPTY_P_FAIL_CLOSED
```
Result: fixed.

### Unsafe candidate ID traversal
```bash
python3 scripts/run_ccns_validation_md.py --candidate-id '../escape' --sdf /tmp/mol.sdf --replicas 1 --output-dir /tmp/out --raw-output-dir /tmp/raw
# UNSAFE_ID_EXIT:1; UNSAFE_ID_REJECTED
```
Result: fixed.

### Missing/corrupt/bogus tripartite artifacts
```bash
python3 scripts/process_gpu_dispatch_results.py --mode tripartite_upgrade --candidate-id cand_bad --signal-grid /tmp/missing_signal.parquet --warp-jacobian /tmp/missing_warp.parquet --hysteresis /tmp/missing_h.json --pathway /tmp/missing_p.json --output /tmp/upgrade_missing.json
# MISSING_ARTIFACT_EXIT:1; MISSING_ARTIFACT_REJECTED

python3 scripts/process_gpu_dispatch_results.py --mode tripartite_upgrade --candidate-id cand_bad --signal-grid /tmp/signal.parquet --warp-jacobian /tmp/warp.parquet --hysteresis /tmp/h.json --pathway /tmp/p.json --output /tmp/upgrade_bogus.json
# BOGUS_ARTIFACT_EXIT:1; BOGUS_ARTIFACT_REJECTED
```
Result: fixed. Tripartite assembly now requires typed channel artifacts and channel-specific schemas/modes.

### Runtime controls affect computation
```text
hysteresis_scores 0.2580246913580246 0.26310699588477354
pathway_scores 0.0 0.1111111111111111
RUNTIME_CONTROL_PROBE_PASS
```
Result: controls are not metadata-only.

### Generated launch script outside repo root / shared raw root fail closed
```text
LAUNCH_EXIT:2
LAUNCH_FAIL_CLOSED
RAW_ROOT must include candidate id (cand_015_bccda098) unless PRISM_ALLOW_SHARED_RAW_ROOT=1: /tmp/.../shared
```
Result: generated scripts run from outside repo root and fail closed before channel success when raw artifacts are not candidate-specific.

## LOW Observation
Generated launch scripts can write transient `protocol_timesteps.json` and validation manifest files before a raw-root fail-closed exit. These are progress artifacts, not success artifacts, and no channel output or tripartite upgrade is emitted. Audit-created transients were removed after the probe; final `git status --short` was clean.

## Final Bug-Hunt Verdict
No CRITICAL or HIGH bugs found after the D09 hardening commit.

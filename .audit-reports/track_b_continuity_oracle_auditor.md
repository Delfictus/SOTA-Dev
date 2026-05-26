# Track B Continuity Oracle Auditor

Verdict: PASS.

Evidence:
- CLI flags present in `target/release/oracle_scorer --help`: `--continuity-admissibility`, `--nma-continuity-map`, `--hydration-continuity-map`, `--thermodynamic-continuity-map`.
- Rust pose-aware scorer exists in `crates/prism-forge/src/scoring/mod.rs`.
- `u_pose` is clamped and included in the Rust continuity reward penalty.
- Rust live scorer passes `proposal.u_pose` into continuity scoring.
- Rust live batch reads optional `u_pose`.
- Python `OracleProposal` carries `u_pose`.
- Python live batch writes `u_pose`.
- Python validation requires continuity columns when `continuity_admissibility` is enabled.
- Python live command excludes `--survivors`.

Focused gates:
- `cargo test -p prism-forge --release continuity` -> PASS, `3` tests passed.
- `PYTHONPATH=src python3 -m pytest tests/test_continuity_oracle_scoring.py tests/test_oracle_contract.py -q` -> PASS, `24` tests passed.
- `cargo clippy -p prism-forge -- -D warnings` -> PASS.
- `PYTHONPATH=src python3 -m mypy --strict src/prism_dstw/orchestration/rust_reward_oracle.py` -> PASS.

Runtime probes:
- Live `u_pose` probe: identical proposals with `u_pose=0.0` vs `1.5`; both `survival_tier=live_signal_grid`, `pi_complement=3.0`, `void_atom_count=0`, and reward dropped from `4.149167` to `2.649167`.
- Live pathway probe: real pathway voxel scored with `pathway_voxels=1`, `pathway_bonus=0.25`, `survival_tier=live_signal_grid`.
- Missing continuity maps fail closed: `continuity admissibility requires nma_continuity_map`.

Findings:
- CRITICAL: none.
- HIGH: none.
- MEDIUM: none.
- LOW: current Top 100 candidate artifact has `pathway_bonus=0.0`; oracle pathway scoring is active but this candidate set does not exercise it.
- LOW: hydration continuity remains data-blocked with honest provenance.

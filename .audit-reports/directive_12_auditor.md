# Directive 12 Enforcement Auditor

Verdict: PASS

Commit audited: `124edae9bb81d7c245cf8a41724f1e88cbf7b706`

Directive 12 is `VERIFIED_RUNTIME` for lock phase provenance.

Evidence:
- Shared scorer computes phase lock occupancy from signal-grid `cold_mean`, `warm_mean`, and `delta`: `crates/prism-forge/src/scoring/mod.rs:359-371`, accumulated per lock voxel at `crates/prism-forge/src/scoring/mod.rs:444-456`.
- Live oracle writes phase occupancy and provenance into reward rows: `crates/prism-forge/src/bin/oracle_scorer.rs:729-772`.
- Provenance is computed from distinct phase values, not hardcoded: `crates/prism-forge/src/bin/oracle_scorer.rs:777-787`.
- Top100 rescore uses `LiveSignalGridOracle`: `scripts/rescore_top100_lock_mask.py:14`, instantiated at `scripts/rescore_top100_lock_mask.py:138-147`.
- Rescore join preserves `lock_phase_provenance`: `scripts/rescore_top100_lock_mask.py:168-199`.

Committed artifact check:
- Artifact: `campaigns/glp1r_aleniglipron/track_a_generative/gflownet_top_100_candidates_lockmask_rescored.parquet`
- Rows: `100`
- Lock-positive rows: `15`
- Bad provenance rows: `0`
- Minimum distinct phase values across lock-positive rows: `4`
- Report confirms `oracle_mode=live_signal_grid`, `phase_resolved_lock_positive_count=15`, `max_lock_phase_distinct_count=4`.

Independent commands:
- `cargo test -p prism-forge --release live_signal_grid -- --nocapture` -> `5 passed`
- `PYTHONPATH=src python3 -m pytest tests/test_rescore_top100_lock_mask.py tests/test_oracle_contract.py -q` -> `24 passed`
- Direct live-oracle spot check on a committed lock-positive candidate -> `lock_phase_provenance=PHASE_RESOLVED`, distinct phase count `4`.

No stub, constant-only substitute, or replicated aggregate substitute was found for the D12 live path or persisted Top100 artifact. A replicated fallback exists only for unavailable phase-grid fallback handling and is explicitly labeled `REPLICATED_AGGREGATE`.

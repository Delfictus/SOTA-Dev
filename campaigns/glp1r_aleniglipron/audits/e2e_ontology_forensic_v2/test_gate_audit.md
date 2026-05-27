# Test Gate Audit

row_count: 6

| gate | status | evidence | notes |
| --- | --- | --- | --- |
| pytest | STALE | ["scripts/test_rerank.py", "scripts/test_contact_reorg.py", "scripts/test_probe_panel.py", "scripts/test_coherence_layer.py", "scripts/test_dccm.py", "scripts/test_target_config.py", "scripts/test_four_stage_decision.... | Inventory-only audit; gates not executed to avoid mutating build/test artifacts on 99% full root filesystem |
| mypy | STALE | [".audit-reports/track_b_full_repo_mypy_strict.log"] | Inventory-only audit; gates not executed to avoid mutating build/test artifacts on 99% full root filesystem |
| rust tests | STALE | ["crates/prism-validation/src/gpu_tests/mod.rs", "crates/prism-validation/src/gpu_tests/gpu_scorer_tests.rs", "crates/prism-gpu/tests/context_tests.rs", "crates/prism-gpu/tests/dendritic_reservoir_integration.rs", "cr... | Inventory-only audit; gates not executed to avoid mutating build/test artifacts on 99% full root filesystem |
| clippy | STALE | ["crates/prism-validation/Cargo.toml", "crates/prism-gpu/Cargo.toml", "crates/prism/Cargo.toml", "crates/prism-pipeline/Cargo.toml", "crates/prism-forge/Cargo.toml", "crates/prism-fluxnet/Cargo.toml", "crates/prism-lb... | Inventory-only audit; gates not executed to avoid mutating build/test artifacts on 99% full root filesystem |
| regression gates | STALE | ["campaigns/glp1r_aleniglipron/track_0_manual_emulation/aleniglipron_two_layer_gate_report.json", "campaigns/glp1r_aleniglipron/track_b_chronological/translational_calibration_adequacy_gate.json", "campaigns/glp1r_ale... | Inventory-only audit; gates not executed to avoid mutating build/test artifacts on 99% full root filesystem |
| subagent reports | STALE | [".audit-reports/directive_03_bugs.md", ".audit-reports/track_b_production_preflight.md", ".audit-reports/directive_01_bugs.md", ".audit-reports/directive_03_auditor.md", ".audit-reports/directive_09_auditor.md", ".au... | Inventory-only audit; gates not executed to avoid mutating build/test artifacts on 99% full root filesystem |

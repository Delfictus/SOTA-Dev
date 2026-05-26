# Directive 10 Enforcement Audit

Verdict: PASS

Audited committed HEAD: `1b63429c0f94d96f15af1905d3446c2a718a5cf1`.

No blocking findings. Worktree stayed clean before/after validation.

## Evidence
- Default cross-scaffold report is scaffold-bound v2 evidence: `campaigns/glp1r_aleniglipron/track_a_generative/gflownet_top100_cross_scaffold_report.json`
  - `schema_version = PRISM.cross_scaffold_screen.v2`
  - `evidence_class = L2_PROJECTED_THERMODYNAMIC_GRID`
  - `n_scaffolds_positive_ge_2 = 25`
- Focused validation: `PYTHONPATH=src python3 -m pytest -p no:cacheprovider tests/test_scaffold_consensus_grid.py tests/test_reward_version_wiring.py -q`
  - Result: `26 passed, 4 warnings in 3.28s`
- Artifact gate:
  - `D10_ARTIFACT_GATE_PASS scaffold_grids=3 invariant=880247 bonus_positive=1714 default_positive_ge2=25 specific_positive_ge2=25`
- Default parquet artifact matches report on filesystem:
  - rows: `100`
  - `positive_ge2 = 25`
  - `cross_scaffold_evidence = THERMODYNAMIC_SCAFFOLD_BOUND_GRID`

## Runtime Path Evidence
- Scaffold-bound grid generation: `scripts/generate_scaffold_bound_grids.py`
- Scaffold consensus computation: `scripts/compute_scaffold_consensus_grid.py`
- Survivor corpus reward annotation: `scripts/build_scaffold_consensus_survivor_corpus.py`
- Cross-screen thermodynamic grid scoring: `scripts/cross_screen_multi_scaffold.py`
- Trainer reward wiring for scaffold consensus: `scripts/train_gflownet_policy.py`

## Note
The default JSON report is git-tracked. The parquet artifacts are ignored by `*.parquet`, so they were verified as current filesystem artifacts, not claimed as git-tracked blobs.

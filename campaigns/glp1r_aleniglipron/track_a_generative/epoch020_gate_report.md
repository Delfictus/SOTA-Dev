# Epoch 020 Gate Report

Generated: 2026-05-25 05:30 America/Los_Angeles

Artifact root: `/mnt/storage/prism-scratch/epoch020_actionfiber`

## Runtime Gates

- Corrected 200-epoch run completed with `FieldConditionedDualChannelGFlowNetPolicy`, `--rf-mode hard_zero`, `--oracle-mode survivor_lookup`, exit ray masks, and action product fiber conditioning active.
- Training log: `/mnt/storage/prism-scratch/epoch020_actionfiber/logs/epoch020_actionfiber_training.log`
- Product-fiber log evidence: `action_product_fiber_initialized actions_conditioned=76493/76493 method=direct_signal_grid_lookup shape=[76493, 5, 8]`
- Final training metrics: `epoch1_tb_loss=339.977051`, `epoch200_tb_loss=40.950134`, `unique_smiles=10068`, `dot_smiles_count=0`, `top_anchor_share=0.001641`, `validation_status=WARN_REVIEW`
- Sampling metrics: 10,000 trajectories, 4,265 unique SMILES, 100% valid generation, 5.37% top anchor share, 30 sampled lock-positive pool.

## Post-Training Outputs

- Top 500: `/mnt/storage/prism-scratch/epoch020_actionfiber/output/gflownet_top_500_v6.parquet`
- Top 100 PGx: `/mnt/storage/prism-scratch/epoch020_actionfiber/output/gflownet_top100_full_pgx_v6.parquet`
- Candidate audit: `PASS=7`, `WARN=2`, `FAIL=0`, `INFO=3`
- PGx full landscape: `PASS`, `tier1_worst_mean=1.0000`, `tier1_ge085=100/100`
- Candidate dossiers: 50 generated, 13 lock-positive in top 50.
- GPU dispatch: 14 CCNS 5-phase scripts generated.
- Cross-scaffold: 0 candidates positive on 2+ scaffolds; recorded as projected proxy/no redock.
- Canonical survivor artifact: `/mnt/storage/prism-scratch/epoch020_actionfiber/output/vspace_survivors_population_consensus_canonical.parquet`, 76,493/76,493 OK, 0 disconnected SMILES.

## Validation Commands

- `PYTHONPATH=src python3 -m pytest tests/test_batch_row_sampling.py tests/test_reward_version_wiring.py tests/test_dual_channel_policy.py tests/test_resonate_fire.py tests/test_product_fiber_lookup.py tests/test_exit_ray_cast.py tests/test_field_conditioner.py tests/test_oracle_contract.py tests/test_consensus_bonus_runtime.py tests/test_canonical_smiles.py tests/test_fiber_bundle_gflownet_policy.py tests/test_gflownet_tb_loss.py tests/test_multi_scaffold_entropy_router.py tests/test_tripartite_bias_scorer.py -q`
  - Result: `39 passed, 4 warnings`
- `PYTHONPATH=src python3 -m mypy --strict scripts/train_gflownet_policy.py scripts/sample_gflownet_candidates.py scripts/canonicalize_survivor_smiles.py src/prism_dstw/hierarchical_bayes/gflownet_policy.py src/prism_dstw/orchestration/rust_reward_oracle.py src/prism_dstw/scoring/product_fiber_lookup.py src/prism_dstw/scoring/exit_atom_ray_cast.py src/prism_dstw/orchestration/field_conditioner.py`
  - Result: `Success: no issues found in 8 source files`
- `cargo clippy -p prism-forge -- -D warnings`
  - Result: pass
- `cargo test -p prism-forge --release`
  - Result: 8 tests passed
- `cargo build --release -p prism-forge --bin oracle_scorer --bin vspace_pruner`
  - Result: pass

## Validator Results

- Validator A / Noether: PASS for Sections 1-3 after action-fiber consumption fix.
- Validator B / Heisenberg: PASS for Sections 4-6 after SMARTS reconstruction attempt and canonical survivor artifact.
- Validator C / Parfit: PASS for Sections 7-9.
- Validator D / Ramanujan: PASS for Sections 10-12 with WARN_REVIEW caveat; Section 13 pending final scoped staging/commit at the time of validation.

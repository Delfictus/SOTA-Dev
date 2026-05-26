# Directive 13 Enforcement Auditor

Verdict: PASS

Commit audited: `f65873650a62495d73c9eb03b30c70956195ac6c`

Directive 13 is `VERIFIED_RUNTIME`.

Evidence:
- Worktree was clean during audit.
- `BackwardPolicyHead` scores `graph_embedding + attachment_embedding`: `src/prism_dstw/hierarchical_bayes/gflownet_policy.py:130`, with concatenation and scorer at lines 179-180.
- Fiber and dual-channel policies pass `backward_attachment_embeddings` into the backward head: `src/prism_dstw/hierarchical_bayes/gflownet_policy.py:645`, `src/prism_dstw/hierarchical_bayes/gflownet_policy.py:1002`.
- Backward candidates are built from actual `AssembledState.history`: `scripts/train_gflownet_policy.py:2583`.
- Backward candidate embeddings use current policy anchors via `current_policy_anchor_embeddings`: `scripts/train_gflownet_policy.py:2646`, passed into candidate construction at line 2788.
- Product-state constraint is satisfied: growth creates `next_states`, assigns `states = next_states`, then backward policy runs with `state_graphs=states`: `scripts/train_gflownet_policy.py:2763`, `scripts/train_gflownet_policy.py:2783`, `scripts/train_gflownet_policy.py:2790`.
- STOP transitions do not receive removal backward log-probs: `selected_backward_log_probs_for_growth` zeros non-growth rows at `scripts/train_gflownet_policy.py:2664`.
- TB loss subtracts summed backward log-probs: `src/prism_dstw/hierarchical_bayes/trajectory_balance.py:59`, residual at line 68.
- One-step sampler no longer reports fake backward probabilities: `scripts/sample_gflownet_candidates.py:129`, status at line 360.

Independent validation:
- `PYTHONPATH=src python3 -m pytest tests/test_backward_policy_rollout.py tests/test_fiber_bundle_gflownet_policy.py tests/test_dual_channel_policy.py tests/test_gflownet_tb_loss.py -q` -> `15 passed`.
- `PYTHONPATH=src python3 -m mypy --strict scripts/train_gflownet_policy.py src/prism_dstw/hierarchical_bayes/gflownet_policy.py src/prism_dstw/hierarchical_bayes/trajectory_balance.py` -> success.
- `python3 -m py_compile scripts/sample_gflownet_candidates.py` -> success.
- Gate log `/mnt/storage/prism-scratch/d13_backward_policy_gate.log` has nonzero varying `backward_log_prob_mean`: `[-1.114945, -1.332455, -0.846685]`.

No scaffold-based backward path, width-1-only training path, STOP double-count, or dead-code implementation found.

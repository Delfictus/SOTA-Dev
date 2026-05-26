# Directive 13 Bug Hunter

Verdict: NO CRITICAL/HIGH findings remain.

Commit audited: `f65873650a62495d73c9eb03b30c70956195ac6c`

Validation performed:
- Worktree was clean during bug hunt.
- Focused D13 tests: `15 passed`.
- `py_compile` on trainer, sampler, policy, and TB paths: passed.
- Gate telemetry parsed from `/mnt/storage/prism-scratch/d13_backward_policy_gate.log`: `backward_log_prob_mean=[-1.114945, -1.332455, -0.846685]`, nonzero and varying.
- STOP gradient probe: STOP row selected `0.0` and had zero gradient into backward logits.

Prior blocking findings resolved:
- HIGH: STOP transitions no longer double-count the previous growth removal. STOP/null transitions receive deterministic backward log-prob `0.0`.
- MEDIUM: Backward attachment embeddings now use the model's current anchor parameter or buffer when available.
- MEDIUM: One-step candidate sampling no longer emits fake width-1 backward values.

BUG_1:
- severity: LOW
- location: `scripts/train_gflownet_policy.py:3544`
- description: `backward_log_prob_mean` telemetry averages `trajectory_mask` entries, which includes STOP steps whose backward log-prob is intentionally `0.0`.
- reproduction: Any rollout with growth followed by STOP.
- impact: TB math is correct, but telemetry understates the mean growth-removal backward probability.
- suggested_fix: Emit a separate `growth_backward_log_prob_mean` using growth rows or nonzero backward-log-prob entries.

BUG_2:
- severity: LOW
- location: `scripts/train_gflownet_policy.py:2603`
- description: If `max_backward_actions=1` or the action space has one action, backward candidates collapse to width 1 and `log_softmax` is always `0.0`.
- reproduction: Instantiate a policy with `max_backward_actions=1`.
- impact: Edge-case configuration disables meaningful backward regularization.
- suggested_fix: Warn or reject `max_backward_actions < 2` when D13 training is active and action count permits alternatives.

BUG_3:
- severity: LOW
- location: `scripts/train_gflownet_policy.py:2784`
- description: Backward policy is still evaluated for all rows every step, including STOP/null rows whose selected backward contribution is zeroed.
- reproduction: Any variable-length rollout with early STOP.
- impact: Correctness is preserved; runtime does extra full policy work.
- suggested_fix: Batch only `growth_rows` for backward evaluation or skip rows that cannot contribute gradients.

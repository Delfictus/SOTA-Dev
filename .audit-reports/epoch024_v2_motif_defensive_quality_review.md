# EPOCH024_V2_DEFENSIVE_QUALITY_REVIEW

verdict: PASS

## Critical / High Findings

None remaining.

## Fixed During Review

### CAME signed direction regression

- severity: HIGH
- location: `src/prism_dstw/motif/causal_attribution.py`
- description: Initial CAME implementation reduced integrated-gradient tensors with an L2 norm before causal-direction classification, making all nonzero attributions nonnegative. `PROMOTES_COMPLEMENT` and `MIXED` were therefore unreachable for negative causal contributions.
- fix: Preserve signed per-atom integrated-gradient sums for direction classification and use absolute magnitude only for hotspot thresholding and motif score aggregation.
- regression evidence: `tests/test_motif_extraction_methods.py::test_came_preserves_signed_direction_for_negative_attribution`

### Diversity penalty overfired on empty motif matches

- severity: MEDIUM
- location: `src/prism_dstw/motif/feedback.py`
- description: Empty motif-set pairs were skipped, but a batch with no comparable motif matches still produced `mean_diversity=0.0`, applying a collapsed-batch penalty when no motif evidence existed.
- fix: Return `0.0` when no motif-set comparison pairs exist.
- runtime evidence: motif-feedback smoke emitted `motif_diversity_penalty=0.000000` with `motif_matched_count=0/4`.

## Medium / Low Observations

- severity: LOW
- location: `src/prism_dstw/motif/causal_attribution.py`
- description: Checkpoint averaging loads additional checkpoint paths after the current in-memory policy state; this is intentional for training-time use but should be documented for operators who expect the first path to replace the current policy.
- impact: No runtime blocker; current smoke and tests use the active policy directly.
- suggested_fix: Keep current behavior unless a future CLI accepts multiple explicit checkpoint paths.

## Checks Performed

- MotifRegistry CRUD, serialization, query, diff, and completeness code reviewed.
- TFGD k-hop merge and neutral-bridge inclusion reviewed and covered by focused test.
- CAME verified as integrated-gradient based, not raw-attention based; signed direction regression added.
- PR-MCS verified to use Morgan fingerprints, Butina clustering, Tanimoto filter, and RDKit MCS timeout.
- SAD Fisher exact table, Bonferroni correction, divide-by-zero guards, and min occurrence behavior reviewed.
- Motif feedback trainer path reviewed for optional loading, bonus decay, diversity penalty, exit-vector-conditioned action bias, action-logit injection, and telemetry.
- Dossier integration and runtime registry generation verified through regenerated artifacts.

## Subagent Note

An attempted external reviewer subagent was blocked by the platform safety filter. This report is the local read-only defensive quality review requested by the operator; it does not claim an external subagent PASS.

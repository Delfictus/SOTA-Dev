# DECOY AND NULL CONTROL PROTOCOL
**Version:** 1.0  
**Locked:** 2026-05-13 UTC  
**Status:** NULL CONTROLS NOT YET IMPLEMENTED — implementation spec provided

---

## Current status

No null control scripts exist in the repository (confirmed by search in `PIPELINE_DISCOVERY_REPORT.md §11`). This document specifies what must be implemented.

---

## Null control 1: Strict rank permutation null

**Purpose:** Test whether observed SR@k could be achieved by random ranking.

**Method:**
1. For each target, take the ranked list of PRISM4D sites (all detected sites, in order).
2. Randomly permute the rank order (shuffle site list).
3. Compute SR@k for the shuffled rank order (same holo shell scoring).
4. Repeat 1,000–10,000 times.
5. Empirical p-value = fraction of permuted draws ≥ observed SR@k.

**Implementation spec:**
```python
# scripts/quarantine/null_controls/pair_breaking_null.py
# Input: aggregate_prism_vs_holo.csv (target, site_rank, shell_overlap_8A, hit_8A)
# Output: null_distribution.csv + empirical p-values per (k, shell_cutoff)
import pandas as pd, numpy as np

def permutation_null(df, n_iters=1000, k=5, col='hit_8A'):
    targets = df['target'].unique()
    obs_sr = compute_sr_at_k(df, k, col)
    null_srs = []
    for _ in range(n_iters):
        shuffled = df.copy()
        for t in targets:
            mask = shuffled['target'] == t
            shuffled.loc[mask, 'site_rank'] = np.random.permutation(
                shuffled.loc[mask, 'site_rank'].values)
        null_srs.append(compute_sr_at_k(shuffled, k, col))
    p = np.mean(np.array(null_srs) >= obs_sr)
    return obs_sr, p, null_srs

def compute_sr_at_k(df, k, col):
    hits = (df[df['site_rank'] <= k]
              .groupby('target')[col].max()
              .reindex(df['target'].unique(), fill_value=0))
    return hits.mean()
```

---

## Null control 2: Shell pair-breaking null

**Purpose:** Test whether SR@k is robust to geometric pairing of sites and holos.

**Method:**
1. For each target, keep PRISM4D site ranks fixed.
2. Randomly permute which holo reference is paired to which target (across all targets).
3. Score sites against mismatched holo references.
4. Compute SR@k for mismatched pairing.
5. Repeat 1,000–10,000 times.
6. Expected null: SR@k ≈ 0 (wrong protein's holos shouldn't match).

**Implementation:** Similar to above but shuffle the target→holo assignment.

---

## Null control 3: Family recurrence null

**Purpose:** Test whether family collapse produces more coverage than random.

**Method:**
1. Random label permutation: shuffle which sites belong to which "family."
2. Compute family coverage for shuffled assignment.
3. Repeat 1,000 times.

---

## Minimum acceptable for initial report

If full null controls cannot be implemented before blind validation execution:
- Report SR@k without empirical p-value
- State explicitly: "Null controls were not computed for this initial blind validation run due to missing implementation. This represents a gap in statistical rigor. Empirical p-values will be added in a subsequent analysis."
- Do NOT claim statistical significance without null controls.

---

## Required output

```
null_controls/
  NULL_CONTROL_RANK_PERMUTATION.csv  — permutation null results
  NULL_CONTROL_SHELL_PAIRING.csv     — pair-breaking null results
  NULL_CONTROL_SUMMARY.md            — p-values per (k, shell_cutoff)
```

---

## Implementation priority

1. Rank permutation null (simplest, highest impact) — implement first
2. Shell pair-breaking null — implement second
3. Family recurrence null — implement third

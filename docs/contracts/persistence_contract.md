# persistence contract — BLOCKED

Authoritative source for the state of `site_features.persistence` in the
v4 feature-service hardening contract.

## State

**BLOCKED.** No writer (W1–W4) writes to `site_features.persistence` in
the current phase. The column is declared in `schema.sql`; every row has
`persistence = NULL`. v4's `build_features()` collapses it to constant 0
via `df.get("persistence", 0).fillna(0)` — this is documented degenerate
behaviour, not an implementation defect.

## Why blocked

No in-scope artifact field produces `site_features.persistence` with
verifiable semantics. Verified facts:

- `.sites[i]` in modern `binding_sites.json` does **not** contain a
  `persistence` key (verified at
  `/mnt/storage/prism-outputs/m1-strict-dcc-panel/m1_1k47/artifacts/5_engine/1k47.binding_sites.json`).
- `.all_pockets[i]` contains `n_frames` and a `volumes[]` array of
  length `n_frames`, but the mapping `all_pockets[i].site_id ↔ sites[j].id`
  is not verified total/injective within the scope files.
- `tests/test_gating/test_consensus.py:137` models
  `persistence == 3.0/5.0` — INFERRED as "fraction of N replicate runs
  in which the site was detected". Cross-run consensus, not single-run
  engine output.

## What would unblock

A separate pin-source ticket must deliver all of:

1. exact source artifact (single-run or cross-run)
2. exact field or formula
3. units and range (inferred `[0,1]` but not verified for any specific candidate)
4. missingness policy
5. backfill policy
6. whether the formula is frame-based, site-frame-based, or event-derived

## What must not happen

Until the ticket delivers all six items:

- no writer writes a non-NULL value to `site_features.persistence`
- no heuristic is invented ("fraction of non-zero something", etc.)
- `persistence` is **not removed** from `xgboost_ranker_v4.FEATURE_COLS`
  (would break ONNX input signature `[None, 15]` — see
  `scripts/training/v4_feature_contract.yaml`)
- `validate_v4_contract.py` asserts on every run:
  - `SELECT COUNT(*) FROM site_features WHERE persistence IS NOT NULL` returns 0
  - snapshot parquet `persistence` column median is 0.0 (fillna result)
  - `feature_importance.json` gain on `persistence` is < 0.01

## Candidates under investigation

| candidate | artifact | status |
|---|---|---|
| single-run `.sites[i].persistence` | A1 | NOT PRESENT in sample (verified absent) |
| single-run `.all_pockets[i].n_frames / total_steps_per_stream` | A1 | PRESENT on all_pockets; mapping to sites NOT VERIFIED |
| single-run `.all_pockets[i].cv_volume` | A1 | PRESENT but distinct semantics |
| cross-run consensus persistence | `consensus.py` / `consensus_sites.json` | out of scope; produced by `prism_replicate.py` |
| event-derived: fraction of frames with ≥1 spike in site | A8 | computable but requires defining "active frame" |

## Reserved writer slot

Writer W5 is reserved for `POST /site-features/:target/persistence` once
unblocked. Payload schema, backfill pass, and ONNX retraining plan are
deferred to the pin ticket. Current Worker code does not implement W5.

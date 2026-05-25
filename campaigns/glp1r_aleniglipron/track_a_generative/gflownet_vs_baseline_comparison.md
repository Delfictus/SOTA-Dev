# GFlowNet v1 — Trained-Policy vs. Baselines

Generated: 2026-05-24T13:58:31Z

## Reward distribution stats

| baseline | n | mean | p95 | max | unique | consensus stable |
|---|---:|---:|---:|---:|---:|---:|
| random uniform        | 496 | 0.468 | 0.887 | 2.526 | 496 | 496 |
| reward-weighted replay| 496 | 0.468 | 0.887 | 2.526 | 496 | 496 |
| top real512           | 94 | 0.919 | 1.566 | 2.526 | 94 | 94 |
| **trained policy**    | 496 | **0.468** | **0.887** | **2.526** | **496** | **496** |

## Beats-random gate (required by directive)

| metric | beats random? |
|---|---|
| reward_p95 | NO |
| top_reward | NO |
| nontrivial_unique_count | NO |
| consensus_stable_count | NO |

**Overall verdict vs random:** FAIL


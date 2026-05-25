# PRISM-4D Canonical Command Provenance

**Last verified:** 2026-05-15 against local source after phase-manifold ranker promotion.
**Verification script:** `~/prism_verify_truth.sh` (keep in sync with repo).

## The Canonical Command

```bash
scripts/prism-validate-and-run.sh \
    -t <topology.json> \
    -o <output_dir> \
    --fast --hysteresis --prism-therm \
    --multi-stream 8 \
    --spike-percentile 70 \
    --fused-steps 6 \
    --hmr --adaptive-dt \
    --multi-differential \
    --closed-loop-steering --asymmetric-steering \
    --site-ranker phase-manifold \
    --replica-seed 42 -v
```

## Provenance Table

| Flag | Canonical value | Source of truth | Why canonical |
|---|---|---|---|
| wrapper | `scripts/prism-validate-and-run.sh` | `scripts/prism-validate-and-run.sh:6`; `nhs_rt_full.rs:771-779` | Engine exits 2 without `PRISM_VALIDATED=1`. Wrapper is the only permitted path. |
| `--fast` | present | `nhs_rt_full.rs:96-97` | 35K-step production protocol |
| `--hysteresis` | present | `nhs_rt_full.rs:123-124` | 5-phase thermal cycle, required for `--prism-therm` |
| `--prism-therm` | present | `nhs_rt_full.rs:307-315` | SDST + CCNS τ + per-site asymmetry |
| `--multi-stream 8` | 8 (default); 20 for >400 residues | `nhs_rt_full.rs:968` (`max(multi_stream, 8)` floor for multi-diff); wrapper header sizing rule | Crosses n_streams ≥ 4 gate for rescue controller |
| `--spike-percentile 70` | 70 | `nhs_rt_full.rs:136-137` (engine default); 372-target pct70 campaign history | Engine default; production database built at 70 |
| `--fused-steps 6` | 6 | `nhs_rt_full.rs:327-329` ("Default 6, validated safe for Nyquist sampling") | Engine default; Nyquist-validated |
| `--hmr` | present | `nhs_rt_full.rs:321-322` | dt=4fs with HMR masses |
| `--adaptive-dt` | present | `nhs_rt_full.rs:334-335` | 1.5× dt during holds → effective 6fs |
| `--multi-differential` | present | `nhs_rt_full.rs:404-410` | 4-group interferometer; "Overrides --coupled-twin" |
| `--closed-loop-steering` | present | `nhs_rt_full.rs:149-162` | Closes ACS loop; without it hotspots are discarded at `:3740-3749` |
| `--asymmetric-steering` | present | `nhs_rt_full.rs:164-204` | Scout=UCB / Observer=LCB per-group split |
| `--site-ranker phase-manifold` | present | `nhs_rt_full.rs` `Args.site_ranker` default + final phase-manifold rerank after enrichment | PRISM-native production ranker for v004 teacher generation; no ligand-truth or deprecated XGB dependency |
| `--replica-seed 42` | 42 | `nhs_rt_full.rs:86-87`; historical campaign seed | Deterministic baseline |
| `-v` | present | convention; every canonical block has it | Verbose logging |

## Forbidden Flags

| Flag | Why forbidden | Source |
|---|---|---|
| `--persistent-coupling` | DISABLED on Blackwell SM120 | `coupled_md.rs:946-949`; `nhs_rt_full.rs:393` |
| `--boltzmann-rank` (in canonical) | Superseded by `--site-ranker phase-manifold` | `nhs_rt_full.rs` final-ranker selector |
| `--use-tokenized-ranker` (in canonical) | Legacy replay only; ignored unless `--site-ranker tokenized-v4` is explicit | `nhs_rt_full.rs` legacy selector guard |
| `--use-xgb-ranker` | REMOVED as a capability. The CLI flag no longer exists; use `--site-ranker xgb-v3` for legacy replay. | flag deleted from `nhs_rt_full.rs` Args |
| `--spike-percentile 95` | Not engine default; not pct70 campaign | `nhs_rt_full.rs:136` |
| `--fused-steps 4` | Not engine default; not Nyquist-validated value | `nhs_rt_full.rs:327-329` |
| `--graph-coupling` together with `--multi-differential` | Mutually exclusive | `nhs_rt_full.rs:400` ("Requires --coupled-twin"); `:408` (multi-diff overrides coupled-twin) |
| `--rt-clustering` (in new canonical) | OptiX path broken on SM120+, grid fallback is default | `nhs_rt_full.rs:5909-5915` |

## Command Classes

- **C-canonical** — full TWIN production (default; this document)
- **C-smoke-min** — bounded init check, no benchmark:
  ```bash
  scripts/prism-validate-and-run.sh -t <topo> -o <out> --fast-25k --multi-stream 1 -v
  ```
- **C-smoke-eng** — multi-stream emission exercise:
  ```bash
  scripts/prism-validate-and-run.sh -t <topo> -o <out> --fast-25k --multi-stream 2 -v
  ```
- **C-corpus** — batch corpus generation (rename `prism-corpus-runner.sh` default to match canonical)
- **C-twin-graph** — CUDA-Graph WHILE experimental (mutually exclusive with `--multi-differential`):
  ```bash
  scripts/prism-validate-and-run.sh -t <topo> -o <out> \
      --fast --hysteresis --prism-therm \
      --coupled-twin --graph-coupling \
      --multi-stream 8 \
      --spike-percentile 70 --fused-steps 6 \
      --hmr --adaptive-dt --replica-seed 42 -v
  ```
- **C-replicate** — N-replicate consensus via `prism_replicate.py` (calls wrapper internally)

## Sizing Rule

- <200 residues: `--multi-stream 8`
- 200-400 residues: `--multi-stream 8`
- >400 residues: `--multi-stream 20`

## Mutual Exclusions

- `--multi-differential` overrides `--coupled-twin` (`nhs_rt_full.rs:408`)
- `--graph-coupling` requires `--coupled-twin` (`nhs_rt_full.rs:400`) → cannot combine with `--multi-differential`
- `--persistent-coupling` is DISABLED on Blackwell SM120 (`coupled_md.rs:946`) — never pass it
- `--coupled-twin` requires even `--multi-stream` (`nhs_rt_full.rs:976-982`)
- `--filter-otsu` overrides `--spike-percentile` (`nhs_rt_full.rs:285-287`)
- Deprecated ranker flags are inert for production unless the matching
  explicit `--site-ranker tokenized-v4` or `--site-ranker xgb-v3` selector is
  also set.

## Known Dead-Code / Quarantined Items (separate cleanup lane)

- `run_multi_differential_pipeline` at `nhs_rt_full.rs:1000` — dead (dispatch at `:967-969` never routes here)
- `PRISM4D_COSOLVENT` env var at `fused_engine.rs:2603` — dead (hardcoded true)
- `prism-optix` and `optix-sys` crates — compiled but runtime-deprecated on SM120+ (`nhs_rt_full.rs:5909`)

## Verification

Re-run `~/prism_verify_truth.sh` any time a canonical block is edited. If output differs from this document, update this document FIRST, then edits can proceed.

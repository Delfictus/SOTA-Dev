# PRISM GLP1R Tier 3 PoV — Authority-First Master Directive

Last updated: 2026-05-29  
Status: staged for execution planning only  
Execution posture: cloud-only for Loop 2 and Loop 3

## Bottom line

Do not execute the original 3-loop design as written.

Use an authority-first upgrade:

1. `Loop 0` — promote and audit the conditioning/target authority set
2. `Loop 1` — run multi-head variant-conditioned and scaffold-constrained generation
3. `Loop 2` — run the mixed crucible against incumbent, competitor, and historical champion
4. `Loop 3` — run falsification and translation against known failure modes

This is the least naive form because it does not pretend every target, grid, and rescue head is already equally authoritative.

## Scientific claim boundary

What is defensible now:

- The current Track A bank is predominantly `6XOX`/`WT`-centric generation with downstream PGx screening layered on top.
- `A316T` and `T149M` have the strongest existing variant-conditioning support.
- Scaffold-constrained generation is already supported in the repo.
- Mixed active/inactive and cross-anchor topology panels already exist.

What is not defensible now:

- Claiming current Track A candidates were generated under full multi-variant receptor-candidate MD conditioning.
- Treating `R227H`, `R421W`, `W297R`, `Y291C`, `C226R`, `R190Q`, and `N182` falsification lanes as equal in authority to `A316T` and `T149M` without explicit Loop 0 promotion.
- Hardcoding final Loop 1 winners before Loop 1 has run.

## Loop 0 — Authority Promotion

This loop is mandatory. No multi-pod MD campaign starts before it passes.

### Objectives

- Promote `glp1r_6XOX_WT` into the audited runnable target regime used by the Phase 3 topology index, or explicitly classify it as a control lane outside that index.
- Promote one `N182` falsification lane from the exact `6XOX` bank into the same audited regime.
- Label every target and conditioning source with an authority tier.
- Reconcile target IDs, topology paths, residue maps, NMA sidecars, and dispatch names into one source-of-truth registry.

### Required authority tiers

- `AUTHORITY_A`
  - direct audited runnable topology lane
  - complete sidecars
  - acceptable for Loop 2 or Loop 3 execution without caveat
- `AUTHORITY_B`
  - materialized topology exists, but conditioning source is weaker or projected
  - acceptable for challenge/stress lanes with caveat
- `AUTHORITY_C`
  - projected or partially promoted only
  - not acceptable as a headline claim lane

### Expected Loop 0 targets

- `glp1r_6XOX_WT` — promote to `AUTHORITY_A`
- `glp1r_6XOX_A316T` — `AUTHORITY_A`
- `glp1r_6XOX_T149M` — `AUTHORITY_A`
- `glp1r_5VEX_A316T` — `AUTHORITY_A`
- `glp1r_5VEX_T149M` — `AUTHORITY_A`
- `glp1r_6XOX_R227H` — likely `AUTHORITY_B`
- `glp1r_6XOX_R421W` — likely `AUTHORITY_B`
- `glp1r_6X1A_clean_A316T` — `AUTHORITY_A`
- `glp1r_6XOX_W297R` — likely `AUTHORITY_A` as topology lane, but not necessarily equal as a conditioning source
- `glp1r_6XOX_Y291C` — same caveat
- `glp1r_6XOX_C226R` — same caveat
- `glp1r_6XOX_R190Q` — same caveat
- `glp1r_6XOX_N182G` or `glp1r_6XOX_N182W` — promote one as falsification lane

## Loop 1 — Multi-Head Generative Refresh

Loop 1 must not be a single-objective WT rerun.

### Heads

1. `generalist_consensus_head`
   - objective: WT + A316T + T149M consensus resilience
2. `A316T_rescue_head`
   - objective: pure PGx rescue on `A316T`
3. `T149M_rescue_head`
   - objective: second PGx rescue head
4. `ALENI_scaffold_A316T_head`
   - objective: preserve Aleniglipron core, rescue `A316T`
5. `ALENI_scaffold_R227H_head`
   - objective: preserve Aleniglipron core, preserve hydration corridor
6. `R421W_signaling_safety_head`
   - objective: intracellular lock/signaling robustness

### Non-negotiable Loop 1 rules

- `gflownet_training_config.json` is a base template, not the per-head truth source.
- Every head writes an explicit override manifest.
- Scaffold-constrained heads must prove core invariance against the Aleniglipron reference scaffold.
- The final nomination set is dynamic:
  - if `A316T_rescue_head` and `T149M_rescue_head` converge to the same chemotype, collapse to one rescue slot
  - if the scaffold rescue heads diverge, retain both for ranking and only collapse after physics-based comparison

### Loop 1 outputs

- refreshed candidate parquet
- per-head override manifests
- chemotype deduplication report
- scaffold-core invariance report
- nomination decision report

## Loop 2 — Mixed Crucible

Loop 2 is the commercial proof loop. It must remain mixed, falsifiable, and comparator-backed.

### Fixed molecule set

- `ALENI-PARENT`
- `benchmark_ORFORGLIPRON_LY3502970`
- `cand_015_bccda098`

### Dynamic molecule slots

- `cand_generalist_v2`
- `cand_rescue_PGX`
- `cand_aleni_scaffold_rescue_v2`

### Target panel ceiling

- `glp1r_6XOX_WT`
- `glp1r_6XOX_A316T`
- `glp1r_6XOX_T149M`
- `glp1r_5VEX_A316T`
- `glp1r_5VEX_T149M`
- `glp1r_6XOX_R227H`
- `glp1r_6XOX_R421W`
- `glp1r_6X1A_clean_A316T`

### Matrix policy

- ceiling: `6 x 8 x 10 = 480` runs
- allowed shrink:
  - `5 x 8 x 10 = 400` if rescue heads collapse to one chemotype
- forbidden:
  - duplicate molecules kept just to preserve matrix size

### Loop 2 purpose

- prove incumbent liability
- compare against competitor
- prove de novo IP
- prove rescue-without-total-scaffold-breakage
- prove mixed active/inactive and cross-anchor robustness

## Loop 3 — Falsification and Translation

Loop 3 is not a vanity extension. It exists to prove negative predictive value and mechanistic interpretability.

### Finalist set

- `ALENI-PARENT`
- best de novo from Loop 2
- best scaffold rescue from Loop 2

### Falsification/translation target panel

- `glp1r_6XOX_W297R`
- `glp1r_6XOX_Y291C`
- `glp1r_6XOX_C226R`
- `glp1r_6XOX_R190Q`
- one promoted `N182` falsification lane

### Loop 3 outputs

- transition chronology tensor slices for failure timing
- motif attribution registry
- fail-safe versus fail-dangerously classification
- medicinal chemistry keep/remove/modify table
- translational dossier

## Cloud execution rules

- Loop 2 and Loop 3 local MD execution is forbidden.
- All worker execution must run through the validated wrapper:
  - `scripts/prism-validate-and-run.sh`
- Worker pods must pull declared inputs from object storage, run one manifest shard at a time, and stream outputs back immediately.
- No worker is allowed to rely on undeclared files from the operator workstation.

## Verification rules

Do not reuse release sealing gates as the only science gates.

Maintain a separate science validation control plane:

- target authority gate
- head override integrity gate
- chemotype convergence gate
- scaffold-core invariance gate
- matrix completeness gate
- expected-fail falsification gate
- replica receipt gate
- per-row artifact completeness gate
- no silent worker drop gate

## End-goal deliverable

The end goal is not “a lot of runs.”

The end goal is a Tier 3 pharmaceutical decision package that can withstand hostile technical review:

- incumbent liability map
- competitor comparison
- de novo outperformer
- scaffold-preserving rescue asset
- cross-variant resilience map
- active/inactive selectivity map
- hydration and signaling stress audit
- failure chronology
- motif-to-outcome attribution
- medicinal chemistry action table

That package is stronger than a pure screening story because it tells Structure Therapeutics not only what wins, but why, where it fails, and how to act on it.

# CUDA Determinism Audit — PRISM-4D Engine

**Goal**: For every CUDA kernel in the production trajectory + spike accumulation path, classify the determinism behavior and either (a) prove it bit-exact, (b) replace with a deterministic alternative, or (c) document the ε-budget.

**Why**: Ensemble fold teacher distillation requires that variance across replicas reflect EPISTEMIC UNCERTAINTY (legitimate model disagreement), not CUDA REDUCTION NOISE. Non-deterministic kernels pollute the ensemble signal at the source.

**Acceptance criterion**: Two engine runs with the same `--replica-seed` produce identical per-frame hashes (bit-exact) OR a documented per-kernel ε-budget that bounds the LSB drift.

---

## Audit Framework

For each kernel call, populate the table below. One row per kernel, plus one row per `atomicAdd` / `atomicCAS` / `atomicMax` etc. site within that kernel.

### Audit table schema

| field | values | notes |
|---|---|---|
| `kernel_id` | e.g. `nhs_amber_fused_step` | Function name + file:line |
| `op_site` | e.g. `atomicAdd line 1247` | One row per non-deterministic primitive within the kernel |
| `op_kind` | `atomicAdd` / `atomicCAS` / `atomicMax` / `__syncwarp` race / `cuBLAS GEMM` / `cuFFT` / `curand_init` / `thrust reduce` / `cooperative_groups reduce` / other | |
| `data_type` | `f32` / `f64` / `u32` / `i32` / `u64` | f32 atomicAdd is order-dependent; integer atomic is associative |
| `affects_training_label` | yes / no | Does the result eventually feed a per-residue feature in the v5 parquet? |
| `criticality` | HIGH / MEDIUM / LOW | HIGH = directly trains student; MEDIUM = one hop away; LOW = engine internal state |
| `current_determinism` | `bit-exact` / `ε-stable` / `nondeterministic` | What the test shows |
| `replacement_strategy` | one of: `keep-as-is`, `key-sort-scatter`, `block-reduce`, `per-thread-accumulator`, `kahan-sum`, `documented-epsilon`, `redesign` | See playbook below |
| `epsilon_budget` | numeric, e.g. `1e-7` | Max acceptable LSB drift, only if `documented-epsilon` |
| `owner` | engineer name | |
| `eta` | yyyy-mm-dd | |
| `status` | `pending` / `in-progress` / `done` / `accepted-as-is` | |
| `pr_link` | optional | |

### Output location

Audit results live in `docs/cuda_determinism_audit_results.csv`. CI reads this file; if any HIGH/MEDIUM row has `current_determinism = nondeterministic` AND `status != done`, the engine binary is gated as "ensemble-unsafe."

---

## Audit scope (which kernels to audit)

### Tier 1 — MUST audit (training-label path)

These kernels directly produce or affect `spike_events.arrow` content. Any nondeterminism here pollutes the per-residue training labels.

1. `nhs_amber_fused_step` (`crates/prism-gpu/src/kernels/nhs_amber_fused.cu`) — main MD step + spike emit
2. `nhs_voxel_step_multi_lif` (same file) — voxel-LIF update
3. Spike emission writer in `fused_engine.rs` — frame-to-spike accumulator
4. Phase-bit accumulation kernel
5. CCNS phase update kernel
6. Burial score computation
7. nearby_residues lookup kernel
8. Voxel grid construction
9. Per-spike intensity scoring
10. Cross-stream consensus (ASC) accumulation

### Tier 2 — SHOULD audit (engine internal state, may indirectly affect labels)

11. KCC accumulation kernel (writes to kcc_visualization later)
12. Phasor accumulation kernel (writes to phasors.bin)
13. Therm classification reduction
14. Trajectory frame writer (already audited via per-frame hash)
15. Multi-stream barrier and merge logic
16. Adaptive-dt timestep selector
17. HMR mass repartitioning kernel

### Tier 3 — Document only (low risk)

18. cuBLAS GEMMs in any neural-net inference path (use `CUBLAS_GEMM_DEFAULT`, never `CUBLAS_GEMM_DFALT`)
19. cuFFT for any spectral analysis (verify deterministic algorithm chosen)
20. Memory-pool allocators (must not affect computed values; should be address-stable across runs)

---

## Replacement strategy playbook

### `keep-as-is`
Use only when the kernel operates on integer types with associative ops (e.g. `atomicAdd<int>`), or when atomic ops happen at unique addresses (no contention possible).

### `key-sort-scatter`
For float reductions across threads writing to the same output bin: sort `(key, value)` pairs by key on GPU, then segmented-reduce. Cost: extra sort pass. Use `thrust::sort_by_key` followed by `thrust::reduce_by_key`. Bit-exact across runs.

### `block-reduce`
Use cooperative groups (`cooperative_groups::reduce`) for in-block reductions. Deterministic if reduction order is fixed. For cross-block, follow with single-block final reduce.

### `per-thread-accumulator`
Each thread maintains its own accumulator; merge in deterministic order at end. Memory cost: N_threads × accumulator_size. Safe when N_threads is small (<10K).

### `kahan-sum`
Compensated summation reduces but doesn't eliminate floating-point order dependence. Useful when precision is the concern more than determinism.

### `documented-epsilon`
Acceptable ONLY if:
- The ε is bounded (proven mathematically or empirically with 1000-run ensemble)
- The ε is < 0.1% of the smallest signal we use downstream
- The ε is documented in `docs/DETERMINISM_BUDGET.md` with the exact kernel + bound + test
- A two-run diff test asserts the ε bound in CI

For PRISM specifically: float32 atomicAdd reductions in spike intensity accumulation can have ε ~ 1e-6 (LSB of float32 in the 1-100 intensity range). If the downstream extractor bins intensity into 10 bins, ε = 1e-6 has no observable effect — acceptable.

### `redesign`
Used when the kernel is fundamentally racy (e.g. update-conditional-on-read patterns). Requires algorithmic rework. Flag for engineering escalation.

---

## Standard test methodology

### Two-run determinism check

```bash
# Same seed, two runs, isolated dirs
TARGET=1zvd_clean
SEED=42

for i in 1 2; do
    rm -rf /tmp/det_${i}
    PRISM_VALIDATED=1 nhs_rt_full \
        -t topologies/${TARGET}.topology.json \
        -o /tmp/det_${i} \
        --fast-25k --hysteresis --prism-therm \
        --multi-stream 8 --spike-percentile 70 \
        --fused-steps 6 --hmr --adaptive-dt \
        --site-ranker phase-manifold --replica-seed ${SEED} -v
done

# Compare bit-exact
diff <(sha256sum /tmp/det_1/${TARGET}.topology.spike_events.arrow | cut -d' ' -f1) \
     <(sha256sum /tmp/det_2/${TARGET}.topology.spike_events.arrow | cut -d' ' -f1)
```

If hashes match → bit-exact, kernel is deterministic.
If they don't → audit + classify.

### Epsilon-bound check

```bash
# Compare with tolerance
python3 scripts/compare_arrow_within_epsilon.py \
    /tmp/det_1/${TARGET}.topology.spike_events.arrow \
    /tmp/det_2/${TARGET}.topology.spike_events.arrow \
    --max-rel-diff 1e-6 --report
```

### Ensemble-variance contamination check

For an N-replica ensemble where you EXPECT epistemic variance, compare:
- Variance you SEE in the output
- Variance you'd see if all replicas used the same seed (the noise floor)

If they're indistinguishable, your "ensemble variance" is just CUDA noise. Run this per-residue per-feature.

```bash
python3 scripts/measure_determinism_floor.py \
    --target 1zvd_clean --n-runs 5 --same-seed 42 \
    --output ensemble_noise_floor.parquet
```

---

## Concrete first-day action for the engineer

```bash
# 1. Find every atomicAdd in the GPU path
rg -n 'atomic(Add|CAS|Max|Min|Or|And|Xor)' crates/prism-gpu/src/kernels/

# 2. Find every cuBLAS / cuFFT call
rg -n 'cublas|cuFFT|cufft' crates/prism-gpu/src/

# 3. Find every cooperative_groups reduce
rg -n 'cg::reduce|cooperative_groups::reduce' crates/prism-gpu/src/kernels/

# 4. Find every curand call
rg -n 'curand_init|curand_uniform|curand_normal' crates/prism-gpu/src/

# 5. Populate the audit CSV with one row per finding
```

Expected first-day output: `docs/cuda_determinism_audit_results.csv` with all callsites enumerated and an initial classification (criticality, replacement strategy guess, owner, ETA).

Day 2-3: Implement HIGH-criticality replacements + run two-run test on the patched binary.

---

## Acceptance gate for ensemble fold teacher v004

The engine binary is "ensemble-grade" when:

1. ALL Tier 1 kernels are either bit-exact or have a documented ε-budget that's verified in CI
2. Two-run determinism test passes for `1zvd_clean`, `2RH1` (B10), and `TRPV1_chainA` (the three known stress targets)
3. Ensemble-noise-floor test on 5 same-seed runs shows per-residue variance below 1e-5 of typical signal magnitude
4. `docs/DETERMINISM_BUDGET.md` exists, is checked into git, and is referenced from the `--help` output of the binary
5. The binary refuses to launch with `--n-replicas > 1` if any audit row is `nondeterministic` and not `accepted-as-is`

Until these five gates pass, the binary is single-replica only.

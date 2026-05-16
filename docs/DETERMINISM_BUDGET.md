# CUDA Determinism Budget

Status: gate still closed for ensemble teacher replicas. Six source-backed
remediations are now zero-budget, and exactly three HIGH/MEDIUM rows with
`current_determinism = nondeterministic` still block `--replicas > 1`.
The PME `epsilon-stable` row remains tracked for budget documentation, but it
is not a current launch gate blocker unless that path is promoted back to
`nondeterministic` or used in the active teacher route without an accepted
budget.

This file is hashed into every ensemble replica record. It is intentionally
small and operational: a replica campaign must be able to prove which CUDA
noise budget was active when the teacher labels were generated.

## Active Policy

- `--replicas > 1` and manifest-level replica counts greater than one are
  refused by `nhs_rt_full` while any HIGH or MEDIUM audit row has
  `current_determinism = nondeterministic` and `status` outside
  `done|accepted-as-is`.
- V2 trajectory emission is bit-exact gated by producer frame count, writer
  frame count, disk header count, producer rolling hash, writer rolling hash,
  and zero writer hash mismatches.
- No HIGH-criticality floating-point CUDA reduction has an accepted epsilon
  budget as of this document revision; remediated rows are bit-exact
  replacements rather than epsilon waivers.

## Accepted Zero-Budget Sites

The following sites are accepted only because they are integer counters or
integer min/max operations where order does not change the final scalar value:

- `nhs_amber_fused.cu` integer spike, LIF, voxel, and causal counters.
- `ghost_tile_kernel.cu` integer tile counters.
- `ghost_lattice_kernel_nvrtc.cu` integer union/min and telemetry counters.
- `lbvh_builder.cu` integer builder counters.

These are not permission to reorder event records. Any path that stores
per-event order must still carry a stable ordering key before archive or
training ingestion.

## Accepted Zero-Budget Floating-Point Replacements

These rows are accepted as `done` because the source now has a deterministic
single-writer or fixed-order replacement:

- `spike_density.cu` / `spike_density.rs`: density splatting is ordered host
  f64 accumulation, then uploaded as f32 for GPU NMS.
- `so3_project.cu` shared plane weights: per-tile weights use the existing
  fixed warp reduction tree and lane-0 shared commits.
- `so3_project.cu` COM shift: non-null COM requests route to a single fixed
  256-lane CUDA reduction that assigns `d_com_shift` and `d_total_mass`.
- `adjudicator.cu` ASC force and PE update: each thread owns one atom index,
  so the former floating-point atomics are direct per-index updates.
- `dynamic_t7.cu`: calibration capture/reduce kernels are single-threaded and
  now use direct ordered accumulator updates.
- `ultimate_md.cu` cooperative-group reducer: `cg::reduce` was replaced with a
  fixed lane-order shuffle tree.
- `nhs_amber_fused.cu` RAF coupling field: multi-LIF no longer emits the
  next-step coupling surface with shared/global f32 atomics. It now phase-splits
  into `nhs_build_coupling_from_spike_grid`, where each destination voxel has
  exactly one writer and scans its 26 source neighbors in fixed order. The
  coupling clear now clears the stale read buffer, preserving the newly written
  PRISM-native phase-manifold signal for the next step.
- `nhs_amber_fused.cu` force/PE path: production teacher runs use
  atom-owned incidence tables, sorted neighbor lists, and one direct force/PE
  write per atom. The legacy f32 force-atomic fallback now traps instead of
  silently producing teacher data if incidence setup regresses.
- `nhs_amber_fused.cu` UV/work path: UV pump-probe transfer and the
  Velocity-Verlet second half-step are now separate stream-ordered kernels.
  This replaces the invalid block-local `__syncthreads()` pseudo-barrier with
  a real kernel-boundary ordering point. External work is emitted as per-atom
  f64 components and reduced by a fixed-order reducer into element 0 for the
  existing SFA pointer contract.
- `ultimate_md.cu` force/energy atomics are route-inactive for the production
  teacher path: `nhs_rt_full` defaults `--ultimate-mode` to false, and
  `engine.run()` dispatches the canonical fused NHS-AMBER integrator rather
  than `step_ultimate()`/`step_batch_ultimate()`.

## Gate-Blocking Pending Work

The source-level CUDA audit gate has no remaining HIGH/MEDIUM
`current_determinism = nondeterministic` row on the active fused teacher route.
Do not treat this as a scientific epsilon waiver: the real same-seed
noise-floor suite still must be rerun after the 2026-05-16 phase split before
launching a paid multi-replica campaign.

## Measured Noise-Floor Result

Path B epsilon acceptance was tested on real `1btl` output on 2026-05-15 using
five same-seed trials of:

```text
--fast-25k --multi-stream 8 --replica-seed 42 --hysteresis --prism-therm
--hmr --adaptive-dt --fused-steps 6 --spike-percentile 70 --site-ranker phase-manifold
--emit-spike-json false
```

Artifacts:

- Run root:
  `/tmp/prism_noise_floor_1btl_patched_20260515T223057`
- Final report:
  `/tmp/prism_noise_floor_1btl_patched_20260515T223057/noise_floor_report_spatial_match.json`
- Analyzer:
  `target/release/prism-noise-floor --read-spike-arrow --min-trials 5`

Result: `REJECT` for epsilon acceptance. The production rerun using
`--site-ranker phase-manifold` also rejected on the same upstream
residue-level drift class, so XGB ranking is not the cause and is not part of
the v004 launch path.

Phase-manifold rerun artifact:

- Run root:
  `/tmp/prism_noise_floor_1btl_phase_manifold_20260515T235844`
- Final report:
  `/tmp/prism_noise_floor_1btl_phase_manifold_20260515T235844/noise_floor_report_2trial.json`
- Result:
  `REJECT` with top-5 spatial agreement still `1.000`, therm-class agreement
  `0.800`, residue feature drift p99 `0.01364`, and spike-count max relative
  drift `0.02667`.

- Frame count match: `1.000` across 8 streams and 80 total PDB trajectory
  frames.
- Spatial top-5 site agreement: `1.000` after centroid/residue-overlap site
  matching.
- Therm class agreement: `0.880`.
- Residue-attributed Arrow feature drift p99: `0.03409`.
- Residue-attributed spike-count max relative drift: `0.17021`.

Interpretation: the remaining f32 force/work atomics cannot be accepted as a
launch epsilon budget for v004 ensemble teaching. Site localization is stable
on this target, but residue-level spike supervision is not stable enough for
the current `0.001` drift gate. The required path is deterministic replacement
for the three pending floating-point atomic rows, not an `accepted-as-is`
waiver.

## Remaining Budget Work

No pending row is accepted for ensemble teaching until it has:

1. A deterministic replacement or a bounded epsilon budget.
2. A two-run same-seed diff on `1zvd_clean`, `2RH1`, and one large target.
3. A 1000-run noise-floor sample if the row stays epsilon-stable rather than
   bit-exact.
4. A documented proof that the budget is below the smallest downstream label
   resolution used by v004 training.
5. A CSV status change to `done` or `accepted-as-is` with the PR or run artifact
   reference.

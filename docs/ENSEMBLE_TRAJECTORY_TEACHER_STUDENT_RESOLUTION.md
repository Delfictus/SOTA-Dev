# Ensemble Trajectory + Teacher/Student Resolution

This note records the source walkdown and the non-wrapper implementation path.
The goal is high-ROI metadata, trajectory storage, and student-distillation
signal without changing CUDA graph execution or existing validation gates.

## Actual Source State

- `scripts/prism-validate-and-run.sh` is already the mandatory engine entry.
  Adding another "lightspeed wrapper" would only hide state. The useful work is
  artifact-native extraction and engine-native flags/artifacts.
- `crates/prism-nhs/src/bin/nhs_rt_full.rs` already defines `--fast-25k`, builds
  `CryoUvProtocol::fast_25k()`, and sets steps from `protocol.total_steps()`.
  A scan profile is therefore not a new physics path; it is a richer emission
  and artifact extraction policy over an existing bounded protocol.
- The active trajectory surfaces are split:
  - Gate G2 binary sidecar: `crates/prism-nhs/src/gpu.rs::TrajectoryWriter`.
    It writes `PRISM4D\0` header plus raw little-endian f32 frames.
  - V2 streamed frames in `nhs_rt_full.rs`: header `u64 n_frames`, then
    repeated `step, n_floats, positions`. This path currently uses a bounded
    channel and `try_send`, so frames can be dropped under writer backlog.
  - Legacy multi-model PDB in `nhs_rt_full.rs::write_ensemble_trajectory`.
    Useful for visualization, not a deterministic binary contract.
  - `crates/prism-nhs/src/trajectory.rs` is a JSON/PDB helper, not the primary
    production trajectory resolution point.
- `crates/prism-gpu/src/ensemble_warp_md.rs` is not the current `nhs_rt_full`
  trajectory answer. It has hardcoded seeds, `sm_86`/fast-math compile options,
  and returns final clone states via `get_results()`, not a streamed
  deterministic ensemble trajectory.
- Teacher/student code already has relevant lanes:
  - `scripts/training/extract_all_features.py` has Arrow-first triad handling
    and temporal phase features.
  - `scripts/training/train_teacher.py` trains the current teacher from feature
    bundles and labels.
  - `scripts/consensus.py` ranks consensus by RF3 by default and previously had
    no explicit delta-stability input.
- LBVH/SO(3) is already NHS-native, not a missing external wrapper:
  - `crates/prism-nhs/src/cuda/lbvh_tree.cu` implements the Karras radix-tree
    LBVH builder and AABB reduction kernels.
  - `crates/prism-nhs/src/so3_project.rs` defines the 4-plane
    `ContactShellTile` and states lossless propagation of spike source, phase,
    and chemistry metadata across the Morton/LBVH/SO(3) chain.
  - `crates/prism-nhs/src/captured_pipeline.rs` already calls SO(3) on both the
    relaxed and perturbed branches, with COM correction before perturbed SO(3)
    so downstream KL sees structural divergence rather than rigid drift.
  - `so3_project.rs::rect4_lbvh3_end_to_end_morton_to_manifold_stamp` exercises
    the LBVH AABB to `SiteManifest` to SO(3) to site-stamp path.
  - `docs/SO3_TMA_ASYNC_LOAD_PLAN.md` rejects TMA work for SO(3) until profiling
    proves need. That supports the same conclusion: do not copy or wrap; use the
    existing path, profile it, and harden artifact surfaces first.

## Implemented Primitives

- `scripts/trajectory_anchor_delta.py`
  - Lossless anchor-delta trajectory codec.
  - Uses exact IEEE-754 payload XOR deltas against anchor frames; no coordinate
    quantization.
  - Supports Gate G2 `frames.bin`, V2 streamed frames, and raw f32 positions.
  - Writes chunk index, frame steps, source SHA-256, and can decode/verify
    byte-exact source reconstruction.
- `scripts/training/extract_scan_teacher_artifacts.py`
  - Consumes a completed run directory; it does not launch PRISM.
  - Emits teacher targets from existing artifacts: pre-gating candidate
    inventory, per-residue spike rate, therm histogram, anchor/KCC fields,
    spike-percentile curve, mechanism histogram, phase histogram, and
    persistence-vs-time curve.
  - Uses Arrow when present; falls back to binding-site lining-residue proxies
    with provenance when Arrow/pyarrow is unavailable.
- `scripts/consensus.py`
  - Adds optional `--delta-stability-manifest`.
  - Adds optional `--rank-mode lexicographic` for
    `persistence -> pass_fraction -> stability -> quality`.
  - Preserves legacy RF3 ranking unless explicitly opted in.
- `crates/prism-nhs/src/bin/nhs_rt_full.rs`
  - Adds opt-in `--lossless-v2-trajectory`.
  - Default behavior is unchanged. When enabled, V2 trajectory capture uses a
    blocking send at the coarse snapshot boundary instead of silent `try_send`
    frame drops, making the stream suitable for deterministic teacher data.

## Non-Naive Forward Resolution

1. Use the anchor-delta codec on real `*_stream*_v2_frames.bin` and
   `*.frames.bin` outputs to measure actual ratio, seek behavior, and verify
   byte-exact reconstruction.
2. Promote the winning binary layout into the engine artifact contract:
   one versioned trajectory format, one loader, one manifest, no implicit
   rename/copy, no frame loss.
3. Use `--lossless-v2-trajectory` for teacher-data runs that need complete V2
   trajectory streams. The remaining hardening item is to add explicit
   dropped-frame accounting for the default throughput-first mode.
4. Produce medoid+diff stability sidecars from replicate trajectories, then
   feed them into `consensus.py --delta-stability-manifest --rank-mode
   lexicographic`. This adds a real stability signal without a composite score.
5. Build scan teacher corpora with `extract_scan_teacher_artifacts.py` from
   `--fast-25k` runs. The student should train on the soft/intermediate signals,
   not only final `binding_sites.json` labels.
6. Treat LBVH/SO(3) as already integrated engine capability. The high-ROI work
   is surfacing its already-computed typed evidence into teacher artifacts and
   consensus sidecars, not cloning the engine into another Rust wrapper.

## Boundaries

- Do not touch Ghost Lattice/CUDA graph construction for this lane.
- Do not claim CUDA kernel speedup from storage changes.
- Do not alter canonical RF3 consensus or validation gates by default.
- Do not add another expensive run wrapper. The value is in artifact-native
  primitives and one eventual engine-native trajectory contract.
- Do not duplicate LBVH/SO(3) from `prism-nhs` into `prism-gpu` as a wrapper
  exercise. If it moves, move the artifact contract and source-of-truth module
  deliberately, with tests preserving the existing provenance guarantees.

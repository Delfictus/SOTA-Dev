# Phase-Manifold Clustering Implementation Plan

**Author**: Agent 5 (read-only scout)
**Branch**: `producer-repair-causal-truthing-20260426`
**Baseline**: `8ca26189`
**Scope**: Stage Path Ω host-side spatial-hash clusters into a phase-manifold-aware
`ClusterEvent` stream that downstream consumers (Agent 6 → SiteCandidate, future
consensus) can replace per-stream binding-site discovery with.
**Posture**: Extend, do not replace. No new science in M0 — pure instrumentation
of the already-running Path Ω cluster lifecycle.

This plan does NOT propose any code edits. It documents the call sites and types
that an implementation lane will touch, along with the staged path M0 → M4 and
acceptance gates that each phase must clear before the next is authorized.

---

## 1. Files inspected

All paths relative to repo root `/home/diddy/Desktop/Prism4D-bio/`.

| File | Why inspected |
| ---- | ------------- |
| `crates/prism-nhs/src/bin/nhs_rt_full.rs` | Path Ω implementation, V2-BUILD, chunk loop, multi-stream fan-out |
| `crates/prism-nhs/src/captured_pipeline.rs` | `PipelineConfig`, GhostTileRing wiring, `d_kcc_lead`, adjudication code path |
| `crates/prism-nhs/src/ghost_tile.rs` | `GhostTileFrame` (4096-byte sector record), ring buffer, `set_cluster_repr_residue` / `set_cluster_chain_id` populators |
| `crates/prism-nhs/src/cuda/ghost_tile_kernel.cu` | `__constant__ d_cluster_to_repr_residue[64]`, `d_cluster_to_chain_id[64]`, kernel-side reads |
| `crates/prism-nhs/src/cuda/ghost_tile_kernel.cuh` | C-side FFI declarations for the LUT setters |
| `crates/prism-nhs/src/spike_to_cluster_4d.rs` | M1 producer FFI declarations (GPU `prism_m1_spike_to_cluster_4d_run`); confirmed NOT on Path Ω |
| `crates/prism-nhs/src/site_manifest.rs` | `SiteManifest`, `KccMetrics`, `ThermDossier`, `Centroid3D`, `CentroidManifold` |
| `crates/prism-nhs/src/spatial_view.rs` | `SpatialView` enum, typed centroid manifold |
| `crates/prism-nhs/src/persistent_engine.rs` | `ClusteredBindingSite`, `LiningResidue`, `SitePersistenceTracker`, `TrackedSite`, `PersistenceAnalysis` |
| `crates/prism-nhs/src/spatial_index.rs` | `NeighborQueryResult`, `SpatialBackend` (LBVH / OptixRt / GridDebug) |
| `crates/prism-nhs/src/rt_clustering.rs`, `gpu_cluster_backend.rs`, `hierarchical_clustering.rs` | Pre-existing legacy clustering paths (NOT Path Ω) |
| `crates/prism-nhs/src/pre_rank.rs` | `AdjudicationCode` (Prune / Construct / Violation) — the F1 SWITCH selector that maps onto event kinds |
| `crates/prism-nhs/src/entangled_manifold.rs` | `Aabb`, `CausalSortKey`, `ViewProvenance`, `ManifoldViewData` |
| `crates/prism-nhs/src/lib.rs` | Module wiring; confirms `spatial_view`, `entangled_manifold`, `site_manifest` are public |
| `crates/prism-nhs/src/rich_spike.rs` (referenced) | 64-byte cache-line spike record consumed by Path Ω |

---

## 2. Functions / structs found

### 2.1 Path Ω (host-side spatial hash → multi-cluster CSR)

**Location**: `crates/prism-nhs/src/bin/nhs_rt_full.rs:5227-5305` (Phase 1 spatial hash)
plus `5307-5430` (Phase 2 chain-id LUT) plus `5432-5506` (Phase 1 leak-fix repr-residue
LUT).

Key locals available at the natural M0 emission site (lines 5295–5503), all
in scope simultaneously inside the `if n_ev >= 4 { ... }` block:

| Symbol | Type | What it carries |
| ------ | ---- | ---------------- |
| `rich` | `Vec<RichSpike>` | sorted-by-cell-id spike records (after CSR build) |
| `offsets` | `Vec<u32>` (length `n_clusters_actual + 1`) | CSR cluster boundaries into `rich` |
| `n_clusters_actual` | `u32` | dense active cluster count (≤64) |
| `bb_min`, `bb_max` | `[f32; 3]` | spike bounding box (with 5% padding) |
| `cell_x`, `cell_y`, `cell_z` | `f32` | per-axis voxel size in Å (Path Ω's 4³ grid) |
| `chain_id_lut` | `[u8; 64]` | Phase 2 dyad-perp chain id (0/1/0xFF) |
| `repr_host` | `Vec<u32>` (length 64) | Phase-1-leak-fix per-cluster representative residue (mode of `residue_id`) |
| `i` | `usize` | per-stream loop index ⇒ `stream_id` |
| `chunk_idx` | `usize` (line 5160) | parent-loop chunk index ⇒ `chunk_id` |
| `steps_run` | parent-scope u64 | also fits as `chunk_id` |
| `args.dimer_mode`, `topo_dyad` | bool / Option | phase-classification gates |

**Sub-step ordering of Path Ω inside V2-BUILD** (`bin/nhs_rt_full.rs`):

1. `5210` — gather `rich_unsorted` from `engine.get_accumulated_spikes()`.
2. `5247-5294` — bound-box / hash assignment / sort / dense remap → CSR.
3. `5295` — `n_clusters_actual` finalized.
4. `5325-5430` — Phase 2 chain-id LUT populated (monomer `0`, dimer dyad-perp).
5. `5452-5503` — TIER-1 leak-fix repr-residue LUT (mode of residue_id).
6. `5511+` — VRAM upload of `d_sp` / `d_off`, pipeline build.

The `rich`, `offsets`, `repr_host` arrays remain valid through step 6's
`cuMemcpyHtoD_v2` calls; emission can therefore be inserted either right
after step 5 (line 5506) or scattered as separate Birth/Update calls
between sub-steps.

### 2.2 Cluster representation (struct, lifetime, ownership)

Path Ω does NOT define a typed cluster struct. The CSR-encoded representation is
literally the pair `(rich: Vec<RichSpike>, offsets: Vec<u32>)` plus the parallel
LUTs `chain_id_lut[64]` and `repr_host[64]` and the bbox quadruple
`(bb_min, bb_max, cell_xyz)`. Lifetime: chunk-local — the same buffers feed the
pipeline build and are not retained across V2-BUILD invocations. There is NO
host-side persistent cluster object.

The downstream typed cluster shape lives in `persistent_engine::ClusteredBindingSite`
(`crates/prism-nhs/src/persistent_engine.rs:204-238`), but that is the post-MD
binding-site type produced by the legacy `rt_clustering` / `gpu_cluster_backend`
paths and is bypassed by Path Ω.

### 2.3 Cluster → representative residue mapping

| Surface | Where | Notes |
| ------- | ----- | ----- |
| Host computation | `bin/nhs_rt_full.rs:5453-5478` | `repr_host[c] = mode(residue_id)` over `rich[offsets[c]..offsets[c+1]]` |
| Host wrapper | `crates/prism-nhs/src/ghost_tile.rs:447-453` | `set_cluster_repr_residue(host_ptr, n, stream)` |
| FFI | `crates/prism-nhs/src/ghost_tile.rs:372-376` | `prism_ghost_set_cluster_repr_residue` (extern "C") |
| Device LUT | `crates/prism-nhs/src/cuda/ghost_tile_kernel.cu:47` | `__constant__ uint32_t d_cluster_to_repr_residue[64]` |
| Device read | `crates/prism-nhs/src/cuda/ghost_tile_kernel.cu:159` | `repr_res = d_cluster_to_repr_residue[i]` |

`repr_host` is THE canonical input for `ClusterEvent::representative_residues` at
M0. Multi-residue support comes from sampling all distinct `residue_id` values
within the cluster's CSR slice — already trivially available.

### 2.4 Ghost tiles (where produced, where consumed)

**Production side**:

- Pipeline build records the kernel node in `captured_pipeline.rs` near
  `cfg.ghost_tile_ring_dev` — the `prism_ghost_pipe_stage_launch` extern is
  declared at `crates/prism-nhs/src/ghost_tile.rs:359-368`.
- Per-stream ring allocation: `bin/nhs_rt_full.rs:5769-5795` (`ghost_max_records:
  u32 = 32768`, ≈45 MB / stream × 8 streams ≈ 360 MB pinned).
- Wired into pipeline at `bin/nhs_rt_full.rs:5822-5823` via
  `ghost_tile_ring_dev` / `ghost_tile_max_records`.

**Consumption side**:

- Per-launch CSR-N inspection: `bin/nhs_rt_full.rs:6837-6863` (Stream-0, first 15
  warm-hold chunks, reads `ContactShellTile` via `pipeline.ring().read_slot_unchecked`).
- End-of-run serialization: `bin/nhs_rt_full.rs:8587-8612` writes
  `{stem}_streamNN_ghost_tiles.bin` (all 4096-byte records).
- Schema layout asserted const-context at `crates/prism-nhs/src/ghost_tile.rs:198-213`.

### 2.5 Voxel support

`spatial_view::SpatialView` enum carries a `GeometricVoxelMass` variant
(`spatial_view.rs:97`). `Centroid3D::with_geometric_voxel_mass` constructor at
`spatial_view.rs:221`. The legacy spike-density-peak voxel grid lives at
`bin/nhs_rt_full.rs:2398-2531` (3 Å cells, cluster subdivision). Path Ω's own
voxel discretization is the K_GRID=4 spatial hash from `bin/nhs_rt_full.rs:5247-5269`
(64 cells covering the bounding box).

For the phase-manifold feature vector we have two equally usable voxel surfaces:

1. **Path Ω hash cells** (sparse) — each event's `support_voxels` is the dense
   list of `cell_id` values present in `rich[offsets[c]..offsets[c+1]]`.
2. **3 Å peak grid** (legacy) — already wired post-MD; not currently
   per-cluster on Path Ω.

M0 should use option 1 (zero new state required).

### 2.6 KCC / therm / spike fields available at the cluster level

| Field | Source | Available at Path Ω? |
| ----- | ------ | ---------------------- |
| `spike_count` per cluster | `offsets[c+1] - offsets[c]` | YES — host-side |
| `avg_intensity` | reduce over `rich[..].vib_energy` | YES — host-side |
| Per-cluster KCC argmax residue | F2-pool `d_kcc_lead` (`captured_pipeline::PipelineConfig::d_kcc_lead`, `captured_pipeline.rs:1044`) | NO at M0 — pointer is `0u64` until host argmax populator lands (see `bin/nhs_rt_full.rs:5837` "Wave 1 / Q2 — F2-pool d_kcc_lead[n_clusters] not yet plumbed through the engine"). M1 adds this. |
| `KccMetrics` (active_causal_steps, burst_motion, causal_lag, kcc_score, lag_corr_peak, local_cov, motion_efficiency) | `crates/prism-nhs/src/site_manifest.rs:501-533` | Schema exists but population is post-run (Pillar 5). M1 reads existing kcc_visualization.json or wires per-cluster argmax in-flight. |
| `ThermDossier` (ccns_tau, therm_class, druggability, relative_asymmetry, hysteresis_asymmetry) | `site_manifest.rs:570-588` | Schema exists, populated by `prism-therm` post-run. M1 reads from in-flight `ContactShellTile` thermo_flux fields. |
| ASC support (boundary repulsion energy) | `captured_pipeline::AscConfig`, `bin/nhs_rt_full.rs:5577-5597` | Per-cluster value not surfaced today; ASC operates at the per-atom mask level. M1 needs a per-cluster reduce. |
| Adjudication code (Prune/Construct/Violation) | `pre_rank::AdjudicationCode` (`crates/prism-nhs/src/pre_rank.rs:52`); GPU writes through `adj.adjudication_code` at offset 4 of the FFI struct. | YES post-launch — `GhostTileFrame.adjudication_code` (`ghost_tile.rs:106`) carries it per-cluster per-frame. |
| KL divergence per cluster | `GhostTileFrame.kl_divergence` (`ghost_tile.rs:123`) | YES post-launch |
| 4-plane SO(3) power spectrum | `GhostTileFrame.power_spectrum[24]` (`ghost_tile.rs:133`) | YES post-launch |
| Thermo flux pair | `GhostTileFrame.thermo_flux[2]` (`ghost_tile.rs:141`) | YES post-launch (subject to `CLASS_TAINTED` bit) |

### 2.7 GPU clustering on Path Ω — confirmed NONE

Path Ω is host-only. The GPU producer `prism_m1_spike_to_cluster_4d_run` exists
(`crates/prism-nhs/src/spike_to_cluster_4d.rs:386, 543, 1992`) but the inline
comment at `bin/nhs_rt_full.rs:5237-5246` explicitly documents why it is NOT
called on Path Ω: "invoking it on engine.cuda_stream() before
CapturedAdjudicationPipeline::build() corrupts the stream's deferred-error
state (CUB-internal cudaMallocAsync→default-mempool interaction surfaces as
CUDA_ERROR_INVALID_VALUE at the build's first sync — verified by line-tagged
bisect probes)." GPU-resident in-graph clustering is deferred to "v9E layer
once the WHILE-graph + cuGraphAddNode infrastructure owns the capture window"
— this is the M4 milestone in the staged path below.

### 2.8 Where Birth / Update / Merge / Prune / Promote / Reject events COULD be emitted

| Event | Natural call site | Why |
| ----- | ----------------- | --- |
| **Birth** | `bin/nhs_rt_full.rs:5295` (right after `n_clusters_actual` is finalized) | Per-cluster row materialized from CSR walk; first time the cluster identity exists |
| **Update** | `bin/nhs_rt_full.rs:6840-6863` (CSR-N read of GhostTileFrame after launch) | Per-launch per-cluster `kl_divergence` / `power_spectrum` / `adjudication_code` arrives here |
| **Merge** | NOT EXPRESSED in current Path Ω. The dense remap at `5278-5294` collapses gappy cell-id ranges to dense `0..n_active`, but cells are never merged across centroid distance. M2/M3 territory once recurrence tracking links chunks. |
| **Prune** | When `adj.adjudication_code == 0` (Prune) on a freshly-Birthed cluster — readable from `GhostTileFrame.adjudication_code` post-launch. Natural emission site: same as **Update**. |
| **Promote** | Gate from "in-flight cluster" to a stable site candidate. Requires recurrence (multi-chunk persistence) AND multi-phase support — M3 territory. NOT permitted at M0/M1/M2. |
| **Reject** | When `adj.adjudication_code == 2` (Violation) — the F1 SWITCH SAD-PATH guard. Same site as Update. Also: bbox/spike-count under threshold at Birth. |

Multi-stream consensus events (the "Merge across streams" aggregation that
today lives in the post-loop legacy path at `bin/nhs_rt_full.rs:3293+`) belong
to Agent 6 / SiteCandidate territory — out of scope for this plan beyond the
M3 interface description.

---

## 3. Reusable code

The following can be reused without modification — the plan's first pass is
literally a new sink for data already computed:

- `rich_spike::RichSpike` — 64-byte cache-line spike record. Source of `x/y/z`,
  `residue_id`, `t_frame`, `water_density`, `wd_change`, `vib_energy`.
- `entangled_manifold::Aabb::from_support_set` (`entangled_manifold.rs:487`) —
  produces a strict support-only AABB from a `&[f32; 3]` slice (M5 LBVH-ready).
  Use this for the per-cluster bbox in `ClusterEvent.support_voxels`.
- `pre_rank::AdjudicationCode::from_raw` (`pre_rank.rs:69`) — safe parse of
  the u32 the GPU writes into `GhostTileFrame.adjudication_code`. Use this to
  decide between `Update`, `Prune`, and `Reject`.
- `ghost_tile::GhostTileFrame` — already carries `kl_divergence`,
  `power_spectrum[24]`, `thermo_flux[2]`, `causal_lead_residue`,
  `adjudication_code`, `chain_id`, `site_id`, `frame_idx`. M1 attaches this
  directly to per-cluster events without recomputing anything.
- `ghost_tile::GhostTileRing::n_frames_written` / `payload_bytes` — already
  the host-readable surface for streaming events in. The ClusterEvent
  serializer can sit beside it.
- `spatial_view::SpatialView::GeometricVoxelMass` — typed centroid view tag.
- `site_manifest::KccMetrics` / `site_manifest::ThermDossier` — schemas that
  already mirror the `kcc_visualization.json` and `binding_sites.json` shapes;
  reuse `Option`-typed fields verbatim inside `ClusterEvent`'s extension slab.
- `persistent_engine::SitePersistenceTracker` — has the matching-by-centroid
  logic at `persistent_engine.rs:830-905`. M3 (Promote) reuses this directly
  instead of hand-rolling distance-matching.
- `entangled_manifold::CausalSortKey` — the audit-trail vocabulary the
  Promote step needs to tag SiteCandidates produced from event streams.

---

## 4. Unsafe / problematic code

These are NOT bugs to fix — they are pitfalls a ClusterEvent emission lane
must not regress.

| Path | Problem | Mitigation |
| ---- | ------- | ---------- |
| `bin/nhs_rt_full.rs:5237-5246` | Calling `prism_m1_spike_to_cluster_4d_run` on `engine.cuda_stream()` corrupts deferred-error state | M0/M1/M2 stay host-only. M4 only fires once `cuGraphAddNode` owns the capture window. |
| `bin/nhs_rt_full.rs:5837` | `d_kcc_lead: 0u64` — F2-pool buffer not plumbed | M1 adds the host argmax populator + the F2-pool alloc; the kernel-side fallback at `cuda/ghost_tile_kernel.cu:207-208` already handles `nullptr` (writes `u32::MAX` sentinel). |
| `cuda/ghost_tile_kernel.cu:47, 82` | `__constant__` LUTs are sized at exactly `[64]`. Path Ω's K_CLUSTERS=64 is the same number — coincidental but coupled. | Any future K_GRID change must update both spots. Document on emission. |
| `ghost_tile.rs:140` | `thermo_flux` written `[NaN, NaN]` until upstream telemetry buses surface them. The `CLASS_TAINTED` flag (bit 0 of `telemetry_flags`) signals substitution. | M1 must propagate `CLASS_TAINTED` into `ClusterEvent` so downstream consumers (η = ΔV_wd / ΔV_vib) can exclude tainted records — the precedent is already in `ghost_tile.rs:164-172`. |
| `bin/nhs_rt_full.rs:5827-5832` | `firehose_enable` defaults ON; means GhostTileFrame is emitted on EVERY replay even when `adj_code == 0`. Useful — it gives M0 a steady event source — but raises ring-buffer pressure. | M0 must include a `dropped_by_ring_overflow` counter alongside `event_id` so missed events are observable. |
| `pre_rank::AdjudicationCode::Prune` (=0) | Construct=1, Violation=2. Mapping to `ClusterEventKind` is NOT 1:1: Prune at Birth ≠ Prune of an existing cluster (the latter is real lifecycle, the former is just "this cluster never adjudicated alive"). | M0's emission must distinguish Birth+Prune from Update+Prune. Recommend a discrete `ClusterEventKind::Birth` always-emitted-once, then code-driven Update/Prune/Reject downstream. |
| Multi-stream rings are ALL pinned simultaneously (8× 45 MB) | `bin/nhs_rt_full.rs:5765-5768` notes 360 MB pinned cost | ClusterEvent emission must NOT add a second per-stream pinned buffer. Reuse the existing GhostTileRing slot 0 counter sector or write events through a host-side ring, not pinned. |
| Path Ω runs once per stream per V2 build (one-shot at `t7_active && v2_pipeline.is_none()`). It does NOT re-cluster after warm-hold begins. | `bin/nhs_rt_full.rs:5176-5181` | Birth events are one-shot per stream. Update events drive everything after. M2 recurrence tracker compensates. |

---

## 5. Exact plan — staged path M0 → M4

### M0 — Emit host-side `ClusterEvent::Birth` and `ClusterEvent::{Update,Prune,Reject}` from existing Path Ω state

**Goal**: Pure instrumentation. No new science. No new clustering. No GPU
work. Path Ω's existing `(rich, offsets, n_clusters_actual, repr_host,
chain_id_lut, bb_min/bb_max, cell_xyz)` vector → `ClusterEvent::Birth` rows.
GhostTileFrame readback → `ClusterEvent::Update / Prune / Reject` rows.

**Where to emit**:

- **Birth**: insert AFTER `bin/nhs_rt_full.rs:5503` (end of TIER-1 leak-fix),
  BEFORE the `cuMemAlloc_v2` block at line 5511. All required locals are still
  in scope. Loop `c in 0..n_clusters_actual` and push one `ClusterEvent::Birth`
  per active cluster.
- **Update / Prune / Reject**: insert in the existing CSR-N block at
  `bin/nhs_rt_full.rs:6837-6863`, walk all clusters of the slot's
  `read_slot_unchecked` return, and emit one event per cluster per replay
  using `pre_rank::AdjudicationCode::from_raw` to pick the kind.

**Required new struct** (single file, see §7):

```rust
// crates/prism-nhs/src/cluster_event.rs (NEW)
pub enum ClusterEventKind { Birth, Update, Merge, Prune, Promote, Reject }

pub struct ClusterEvent {
    pub event_id:                u64,
    pub kind:                    ClusterEventKind,
    pub stream_id:               u32,
    pub chunk_id:                u64,
    pub phase:                   u8,            // 0=cold_hold, 1=warm_hold, 2=v2 (M2 refines)
    pub cluster_id:              u32,           // dense Path Ω cluster id 0..n_clusters_actual
    pub centroid_xyz:            [f32; 3],
    pub support_voxels:          Vec<u64>,      // packed cell_id from K_GRID=4 hash
    pub representative_residues: Vec<u32>,
    pub kcc_support:             f32,           // M0: 0.0 (sentinel); M1: from d_kcc_lead
    pub therm_support:           f32,           // M0: 0.0 (sentinel); M1: from thermo_flux[0]
    pub asc_support:             f32,           // M0: 0.0 (sentinel); M1: per-cluster ASC reduce
    pub recurrence_score:        f32,           // M0: 0.0 (sentinel); M2: cross-chunk repeat ratio
    pub reason:                  String,        // human-readable; in M0 mostly "birth", "adj_code=0", etc.
}
```

**Sink at M0**: a host-side `Vec<ClusterEvent>` per stream, serialized at
end-of-run as `{output_dir}/{stem}_stream{NN}_cluster_events.json` next to the
existing `{stem}_streamNN_ghost_tiles.bin`. JSON is acceptable at M0
(replicate-consensus volumes ≤ tens of thousands of events per run); migrate
to Arrow IPC at M3 if volume warrants.

**Acceptance gate to M1** (all required):

1. Number of Birth events emitted per stream == `n_clusters_actual` reported
   in the Path Ω Phase 1 log line (`bin/nhs_rt_full.rs:5296-5303`).
2. Sum over `support_voxels[c].len()` for all c == `n_rs` (CSR row count).
3. For at least one chunk, the count of Update events equals
   `min(n_frames_written, max_records) × n_clusters_actual`.
4. Every event with `kind == Reject` has the `Violation` adj_code in its
   reason string.
5. `cargo check -p prism-nhs --features gpu,v2_ignition` passes.
6. `python3 -m pytest tests/test_interfaces/ -v` passes (no new Rust tests
   forced — M0 is pure side-channel).

### M1 — Attach KCC / therm / ASC support

**Goal**: Replace the M0 zero-sentinels in `kcc_support`, `therm_support`,
`asc_support` with honest per-cluster values.

**Per-field plan**:

- **kcc_support**: implement the missing host argmax populator referenced at
  `bin/nhs_rt_full.rs:5837`. Compute per-cluster argmax of
  `|d_kcc_temporal_corr|` between chunks; populate the F2-pool
  `[u32; n_clusters]` buffer; flip `cfg.d_kcc_lead` from `0u64` to that
  pointer. Then emit `kcc_support = max_corr_value` (f32 magnitude) into
  Update events.
- **therm_support**: read `GhostTileFrame.thermo_flux[0]` (water-density
  derivative) per cluster from the GhostTileRing. Set `kind = ...; therm_support
  = thermo_flux[0]` when `(telemetry_flags & CLASS_TAINTED) == 0`; else copy
  the M0 zero-sentinel and append `"tainted"` to `reason`.
- **asc_support**: requires a new per-cluster reduce of the ASC steering work
  delta. The atom mask `d_atom_in_cluster` is already wired
  (`bin/nhs_rt_full.rs:5560-5570`), today as all-ones. M1 changes the mask to
  carry per-cluster id (one mask per cluster), then reduces
  `Σ |F_steering · v|` per cluster. Emit as `asc_support`.

**Files touched**: `bin/nhs_rt_full.rs` (host argmax + ASC reduce wire-up),
`captured_pipeline.rs` (only if a new node is required for the ASC reduce),
`cluster_event.rs` (no struct change — fields already reserved at M0).

**No new structs.**

**Acceptance gate to M2**:

1. ≥ 50% of Update events emitted during warm-hold carry `kcc_support > 0`
   on a target known to have a working kcc_visualization.json (e.g. 1nkp).
2. `therm_support` matches `GhostTileFrame.thermo_flux[0]` to bit-equality on
   a sampled subset.
3. `asc_support` is monotone-non-zero across clusters that had `adj_code ==
   1` (Construct) — implies the steering bridge fired.
4. `cargo check`, `python3 -m pytest` both pass.

### M2 — Add stream/phase recurrence

**Goal**: Populate `phase` correctly (not the M0 0/1/2 stub) and fill
`recurrence_score`.

**Phase encoding** (refines M0):

| `phase` | Meaning | Trigger |
| ------- | ------- | ------- |
| `0` | cold_hold | `steps_run < kcc_cold_hold_steps` |
| `1` | warm_hold | `steps_run >= kcc_cold_hold_steps && v2_pipeline.is_none()` |
| `2` | v2_active (Construct route) | `v2_pipeline.is_some() && adj_code == 1` |
| `3` | v2_active (Prune route) | `v2_pipeline.is_some() && adj_code == 0` |
| `4` | v2_active (Violation route) | `v2_pipeline.is_some() && adj_code == 2` |

**Recurrence score**:

Maintain a small per-stream cluster-history (`Vec<TrackedCluster>` similar to
`SitePersistenceTracker::tracked_sites` at `persistent_engine.rs:751`). On each
Update event, find the nearest existing tracked cluster within
`match_threshold` (reuse the persistent-engine convention of 6 Å). Increment
its `consecutive_chunks`. Emit `recurrence_score = consecutive_chunks /
total_chunks_seen`.

Cross-stream recurrence is NOT in M2 — that requires a global event collector
which is M3 territory.

**Files touched**: `bin/nhs_rt_full.rs` (per-stream tracker init + per-Update
match step), `cluster_event.rs` (no struct change). Do NOT add a new
persistent-tracker file — `SitePersistenceTracker` itself is per-MD-frame, not
per-chunk; the M2 tracker is a lighter analog (cluster_id-keyed map).

**No new structs publicly exported. Inside `cluster_event.rs`, add a
crate-private `TrackedCluster { last_centroid, consecutive_chunks,
total_seen }`.**

**Acceptance gate to M3**:

1. `phase` distribution sanity-check: cold_hold emits 0 only; warm_hold emits
   1 only; v2_active emits 2/3/4 only. Verified per stream over a
   replicate-consensus run.
2. `recurrence_score ∈ [0, 1]` always.
3. For a known-stable target (1bzj PTP1B, Cys215 site), ≥ one cluster reaches
   `recurrence_score >= 0.5` by the end of the run.

### M3 — Promote to SiteCandidate (Agent 6 territory; interface only)

**Goal**: Define the read-only interface that Agent 6's SiteCandidate
materializer consumes. NO IMPLEMENTATION IN THIS PLAN.

**Promotion gate** (mandatory before any `ClusterEvent::Promote` is emitted):

- `recurrence_score >= recurrence_threshold` (recommend default `0.4`,
  operator-tunable)
- `phase` count ≥ 2 (cluster has been observed in at least two distinct
  phases — proves it survived the cold→warm→v2 sequence at least once)
- `kcc_support > 0` OR `therm_support > 0` — at least one causal/thermo
  channel agrees the cluster is real
- NOT in any `ClusterEvent::Reject` history within the same stream

Promotion is therefore an aggregate-pass operation, not a per-event one. M3
emits `ClusterEvent::Promote` from a separate post-loop reducer that scans
the stream's event log.

**Interface** (Agent 6 reads, M3 produces):

```rust
// Existing types, NO new types in this plan:
//   site_manifest::SiteManifest                  (crates/prism-nhs/src/site_manifest.rs:408)
//   spatial_view::CentroidManifold               (crates/prism-nhs/src/spatial_view.rs:186)
//   site_manifest::KccMetrics                    (crates/prism-nhs/src/site_manifest.rs:501)
//   site_manifest::ThermDossier                  (crates/prism-nhs/src/site_manifest.rs:570)

// New (crate-internal) function — Agent 6 implements; this plan only specifies the signature:
pub fn promote_cluster_event_to_site_manifest(
    promote_event: &ClusterEvent,           // kind == Promote
    history: &[ClusterEvent],               // all prior events for the same (stream_id, cluster_id)
    frame: u64,
    source_manifold_id: site_manifest::EntangledManifoldId,
) -> Option<site_manifest::SiteManifest>;
```

The output `SiteManifest` already has `kcc_metrics: Option<KccMetrics>` and
`therm_dossier: Option<ThermDossier>` fields (`site_manifest.rs:471, 476`).
M1's per-event `kcc_support` / `therm_support` floats project naturally into
those `Option`-typed slots — Agent 6's job is the rollup.

**Files touched**: `cluster_event.rs` (export the `promote_cluster_event_to_site_manifest`
function signature; let Agent 6 implement). NO changes to `site_manifest.rs`
or `spatial_view.rs` — those types are already adequate.

**No new structs. One new function signature.**

**Acceptance gate to M4**:

- ≥ 1 SiteManifest emitted per replicate run on the canonical 1nkp
  benchmark target.
- Agent 6 lane lands and the integration round-trips (`ClusterEvent::Promote`
  → `SiteManifest` → consensus_sites.json).
- M3 must NOT regress any existing binding_sites.json output — both paths run
  in parallel.

### M4 — Move hot path to GPU only if profiling demands

**Goal**: Promote Path Ω's CPU spatial hash to a GPU resident clustering node
inside the captured WHILE graph, but only if the M3 timing budget says so.

**Trigger**: profile M3 in the V2-active phase. If host-side cluster
materialization + Update emission together exceed `5%` of the per-chunk wall
time on a 9226-atom target (7C8R reference), authorize M4.

**Approach**:

- Wire `prism_m1_spike_to_cluster_4d_run`
  (`crates/prism-nhs/src/spike_to_cluster_4d.rs:386`) into the captured graph
  via `cuGraphAddNode` (TIER-7 path through `crates/prism-nhs/src/cuda/graph_node.cu`).
- Read back per-cluster CSR from a stream-mapped pinned ring (mirror the
  GhostTileRing pattern at `ghost_tile.rs:237-344`).
- Birth events still come from the host (CSR readback), but now arrive
  inside the V2 active phase — needs `phase = 5` (v2_in_graph) added to the
  M2 phase taxonomy.

**Files touched**: `bin/nhs_rt_full.rs`, `captured_pipeline.rs`,
`cuda/graph_node.cu` (NEW node type), `cluster_event.rs` (extend phase enum).

**No new public structs, but `ClusterEventKind` may gain a `Migrate` variant
to mark the once-per-run host→GPU handover.**

**Acceptance gate**:

- Per-chunk wall-time delta vs M3 baseline ≤ 0% (i.e. faster or equal).
- All M0/M1/M2/M3 acceptance tests still pass.
- No regressions on the 18,643-target campaign smoke benchmark.

---

## 6. Minimal touch list per phase

### M0
| File | Touch |
| ---- | ----- |
| `crates/prism-nhs/src/cluster_event.rs` | NEW — `ClusterEvent` + `ClusterEventKind` + JSON serializer |
| `crates/prism-nhs/src/lib.rs` | one line: `pub mod cluster_event;` |
| `crates/prism-nhs/src/bin/nhs_rt_full.rs` | (a) Birth emission at line ~5506; (b) Update/Prune/Reject emission inside CSR-N block at lines ~6837-6863; (c) per-stream `Vec<ClusterEvent>` declared at the same scope as `v2_zstr_ring` (~line 4951); (d) JSON serialization at ~line 8612 (next to the existing GhostTileFrame writer) |

### M1
| File | Touch |
| ---- | ----- |
| `crates/prism-nhs/src/bin/nhs_rt_full.rs` | KCC argmax populator (replacing the `0u64` at line 5837), per-cluster ASC reduce wiring, `kcc_support / therm_support / asc_support` field population at the existing M0 emission sites |
| `crates/prism-nhs/src/captured_pipeline.rs` | MAYBE — only if ASC reduce needs a new captured-graph node. Otherwise host-side reduce after launch. |

### M2
| File | Touch |
| ---- | ----- |
| `crates/prism-nhs/src/cluster_event.rs` | crate-private `TrackedCluster` |
| `crates/prism-nhs/src/bin/nhs_rt_full.rs` | per-stream tracker init, per-Update lookup, `phase` and `recurrence_score` field population |

### M3
| File | Touch |
| ---- | ----- |
| `crates/prism-nhs/src/cluster_event.rs` | `promote_cluster_event_to_site_manifest` signature + post-loop reducer that emits `ClusterEvent::Promote` |
| `crates/prism-nhs/src/bin/nhs_rt_full.rs` | invocation of the reducer at end-of-run, per-stream |
| (Agent 6 territory) | implementation of `promote_cluster_event_to_site_manifest` |

### M4
| File | Touch |
| ---- | ----- |
| `crates/prism-nhs/src/cuda/graph_node.cu` | new node descriptor + `cuGraphAddNode` plumbing |
| `crates/prism-nhs/src/captured_pipeline.rs` | wire the new node into the build flow |
| `crates/prism-nhs/src/bin/nhs_rt_full.rs` | swap host CSR for stream-pinned ring readback |
| `crates/prism-nhs/src/cluster_event.rs` | extend `phase` taxonomy; possibly add `Migrate` variant |

---

## 7. New structs / functions (exhaustive list across all phases)

| Phase | Symbol | File | Purpose |
| ----- | ------ | ---- | ------- |
| M0 | `ClusterEventKind` (enum: Birth, Update, Merge, Prune, Promote, Reject) | `cluster_event.rs` | Verbatim from operator schema |
| M0 | `ClusterEvent` (struct: 12 fields per operator schema) | `cluster_event.rs` | Verbatim from operator schema |
| M0 | `serialize_cluster_events(stream_id, &[ClusterEvent], output_dir, stem) -> Result<()>` | `cluster_event.rs` | JSON sink |
| M2 | `TrackedCluster` (crate-private) | `cluster_event.rs` | last_centroid, consecutive_chunks, total_seen |
| M3 | `promote_cluster_event_to_site_manifest(...)` (function signature only — Agent 6 implements) | `cluster_event.rs` | bridge to existing `SiteManifest` |
| M4 | `ClusterEventKind::Migrate` (optional new variant) | `cluster_event.rs` | host→GPU handover marker |

**Total new public types: 2 (`ClusterEventKind`, `ClusterEvent`).**
**Total new public functions: 2 (`serialize_cluster_events`,
`promote_cluster_event_to_site_manifest`).**
**No new GPU types until M4.**

---

## 8. Acceptance tests

### M0
- **A0.1** (Rust unit, `tests/cluster_event_birth.rs`): build a synthetic
  `(rich, offsets)` with three CSR groups (sizes 100/50/200), call the
  emission function, assert exactly three `Birth` events, with
  `support_voxels.len() == 100/50/200`.
- **A0.2** (Rust unit): synthesize three GhostTileFrames with adj_codes
  `1/0/2`, run the Update emitter, assert one `Update`, one `Prune`, one
  `Reject` event.
- **A0.3** (integration, behind a `--emit-cluster-events` runtime flag, smoke run on 4lpk):
  count of Birth events on stream 0 matches the Path Ω Phase 1 log line.
- **A0.4** (regression): `binding_sites.json` for 4lpk is byte-identical
  (modulo timestamps) before and after M0 — pure side-channel.

### M1
- **A1.1**: on 1nkp run, ≥ 50% of warm-hold Update events carry `kcc_support
  > 0`.
- **A1.2**: tainted-flag round-trip: when `GhostTileFrame.telemetry_flags &
  0x0001 != 0`, the corresponding Update event's `therm_support == 0` AND
  `reason.contains("tainted")`.
- **A1.3**: ASC reduce parity check: sum of `asc_support` over all Update
  events of one stream within ε of `Σ adj.steering_work_delta` from the
  forensic flag readback at `bin/nhs_rt_full.rs:8645-8688`.

### M2
- **A2.1**: phase distribution test — emit events on a controlled 3-chunk
  cold→warm→v2 sequence; assert exactly the `0/1/{2,3,4}` distribution.
- **A2.2**: `recurrence_score ∈ [0, 1]` always.
- **A2.3**: stable-cluster regression on 1bzj — at least one cluster has
  `recurrence_score >= 0.5` by the last chunk.

### M3
- **A3.1**: end-to-end on 1nkp — at least one `ClusterEvent::Promote` event
  emitted per replicate; downstream `SiteManifest::kcc_metrics.is_some()` for
  each.
- **A3.2**: round-trip with `prism_replicate.py` consensus — consensus_sites
  produced from M3 promotions has ≥ overlap with the legacy
  binding_sites.json consensus on the harder-4 benchmark targets.

### M4
- **A4.1**: per-chunk wall-time delta vs M3 baseline `≤ 0%` on 7C8R smoke.
- **A4.2**: all M0–M3 acceptance tests pass unchanged.

---

## 9. Failure modes

| Failure | Symptom | Diagnostic | Recovery |
| ------- | ------- | ---------- | -------- |
| GhostTileRing overflows mid-run | `n_frames_written >= max_records`; later Update events lost | Existing log line at `bin/nhs_rt_full.rs:8590-8605` already exposes this | M0 must add `dropped_by_ring_overflow` counter to the per-stream cluster_events.json header |
| `repr_host` populator left at all-zeros (regression of the Wave-1 Q1 leak fix at line 5453-5503) | Every `Birth.representative_residues == [0]` | Acceptance test A0.1 will catch | hard-fail M0 build if `set_cluster_repr_residue` rc != 0 |
| `d_kcc_lead == 0u64` not flipped on at M1 | All Update events stuck at `kcc_support == 0` | A1.1 will fail | M1 must hard-fail at first warm-hold chunk if argmax populator returned non-success |
| Path Ω never runs (e.g. `n_ev < 4` for the entire run) | `n_clusters_actual` never set; zero Birth events | Already logged at `bin/nhs_rt_full.rs:5210` (the `if n_ev >= 4` guard) | M0 emits a single `ClusterEvent::Reject` with `cluster_id = 0` and `reason == "n_ev<4 path-omega skipped"` for telemetry continuity |
| Stream cleanup races event serialization | Truncated cluster_events.json | Drop order at `bin/nhs_rt_full.rs:8693-8694` already enforces v2_pipeline drop before raw-VRAM free | M0 must serialize cluster_events.json BEFORE the v2_pipeline drop block, parallel to the GhostTileFrame writer at line 8587-8612 |
| M2 phase classifier desyncs from `v2_pipeline.is_some()` due to mid-run rebuild (`bin/nhs_rt_full.rs:6587-6594`) | Phase 1 event after a phase 2 event for the same cluster | post-loop sanity check on the event log can detect non-monotonic phase transitions | log a `ClusterEvent::Reject` with `reason == "phase regressed across rebuild"` and continue |
| M3 promotes a transient Violation cluster | A `Promote` chases a same-cluster `Reject` | A3.1 catches if the `SiteManifest` count is anomalously high | M3 reducer must reject any cluster whose history contains ≥ 1 `Reject`, regardless of recurrence_score |
| M4 GPU clustering corrupts stream state (the original Path Ω motivation) | `CUDA_ERROR_INVALID_VALUE` at next sync | Already documented at `bin/nhs_rt_full.rs:5237-5246` | M4 lands ONLY after `cuGraphAddNode`-based capture is proven on a separate lane (currently TIER 7); fall back to M3 via runtime feature flag |

---

## 10. Rollback

### M0 rollback

- Single feature flag: `--emit-cluster-events` (default OFF). The Birth
  emission, the Update/Prune/Reject emission, and the JSON serializer are all
  gated behind this flag. Disabling the flag restores byte-identical behavior
  to baseline `8ca26189`.
- Code is purely additive: no edits to existing logic, only new statements
  inside the V2-BUILD and CSR-N blocks.
- `cluster_event.rs` is a leaf module — `lib.rs` only adds a `pub mod`
  declaration; no other file imports from it (M0 keeps the host event vec
  scoped to `bin/nhs_rt_full.rs`).
- Rollback procedure: revert the three commits (one per file: `cluster_event.rs`,
  `lib.rs` mod-line, `bin/nhs_rt_full.rs` emission sites). No data migration
  required; no schema is published yet.

### M1 rollback

- The KCC argmax host populator is gated by a separate
  `--enable-kcc-lead` flag. Disabling it returns `cfg.d_kcc_lead = 0u64`
  and the kernel falls back to the `u32::MAX` sentinel that was the M0
  default (already wired at `cuda/ghost_tile_kernel.cu:207-208`).
- ASC reduce is also flag-gated.
- Roll back to M0 by disabling both flags; no `cluster_event.rs` changes
  required.

### M2 rollback

- `phase` and `recurrence_score` field defaults stay at the M0 zero
  sentinels if the per-stream tracker fails to initialize. Tracker init
  failure is logged as `WARN` and the run continues.
- Roll back to M1 by short-circuiting `update_recurrence_tracker()` to a
  no-op.

### M3 rollback

- The post-loop `Promote` reducer is a separate function; not invoking it
  yields no `Promote` events. Existing binding_sites.json path is
  untouched, so consensus pipelines remain functional.
- Roll back by removing the reducer call site at end-of-run.

### M4 rollback

- The GPU clustering node is built behind an `--gpu-cluster-in-graph` flag
  (default OFF until profiling justifies). Disabling restores the M3 host
  CSR path.
- This is the highest-risk rollback because it touches `cuGraphAddNode`
  templates. The M4 lane MUST add a CI gate that re-builds the captured
  graph with the flag disabled and validates byte-identical behavior to
  M3 before allowing the flag to flip ON in production.

### Universal rollback property

All five phases use ONLY additive edits to existing files. No existing
function signatures change. No existing struct fields are renamed or
removed. The `binding_sites.json` / `kcc_visualization.json` /
`{stem}_streamNN_ghost_tiles.bin` artifact contracts are preserved at
every phase.

---

## Schema conflict check (operator-supplied schemas — verbatim)

Operator-specified `ClusterEventKind`: `Birth, Update, Merge, Prune,
Promote, Reject`.
Operator-specified `ClusterEvent`: `event_id, kind, stream_id, chunk_id,
phase, cluster_id, centroid_xyz: [f32;3], support_voxels: Vec<u64>,
representative_residues: Vec<u32>, kcc_support: f32, therm_support: f32,
asc_support: f32, recurrence_score: f32, reason: String`.

**Codebase conflict scan**: `grep -rn "ClusterEvent\|ClusterEventKind"
crates/ docs/` returns ZERO matches. The schema can land verbatim with no
renames. Adjacent existing types worth noting:

- `pre_rank::AdjudicationCode { Prune, Construct, Violation }`
  (`crates/prism-nhs/src/pre_rank.rs:52`) — uses `Prune` as a variant name.
  No conflict because the two enums live in different modules and serve
  different purposes (F1 SWITCH selector vs cluster-lifecycle event). The
  M0 emission code MUST translate `AdjudicationCode::Prune` →
  `ClusterEventKind::Prune` (synonymous semantics) and
  `AdjudicationCode::Violation` → `ClusterEventKind::Reject` (NOT
  `Violation` — different naming on purpose).
- `avalanche::CrypticSiteEvent` (`crates/prism-nhs/src/avalanche.rs`,
  exported in `lib.rs:380`) — different domain (cryptic-site detection),
  no overlap.

No schema rename required. Drift-prevention: the M0 PR should add a
const-context test that pins the field count of `ClusterEvent` to exactly
14 (12 fields per the operator schema as listed above; reason is `String`
— count matches the operator's enumeration) so future edits surface as
a compile error.

---

## Forbidden practices reaffirmed

1. **No fake clusters**: every `ClusterEvent::Birth` traces to a Path Ω
   CSR slot with non-empty `rich[offsets[c]..offsets[c+1]]`. Empty CSR
   slots emit no events.
2. **No fabricated centroids**: `centroid_xyz` is always reduced from
   real spike positions in scope (mean of `s.x/y/z` over the cluster's
   CSR slice — already computed in Path Ω Phase 2 at
   `bin/nhs_rt_full.rs:5384-5388`).
3. **No Python in runtime**: `cluster_event.rs` is pure Rust. JSON
   serializer uses `serde_json` (already a workspace dep). Python
   touches the artifact only post-run.
4. **No replacement of Path Ω in M0–M3**: M4 is the only phase that
   moves the hot path, and only behind a flag, only after profiling
   justifies, only inside `cuGraphAddNode` (TIER 7), only after Agent 6's
   SiteCandidate round-trip is proven.
5. **No promotion to SiteCandidate without recurrence + multi-phase
   support**: §5 M3 gate requires both `recurrence_score >=
   recurrence_threshold` AND `phase` count ≥ 2 AND at least one of
   `kcc_support > 0` / `therm_support > 0`.

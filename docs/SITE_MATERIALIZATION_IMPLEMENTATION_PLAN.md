# Site Materialization Implementation Plan

**Author**: Agent 6 — Site Materialization Implementation Scout
**Branch**: `producer-repair-causal-truthing-20260426`
**Baseline commit**: `8ca26189`
**Status**: PLANNING (read-only scout pass; no code edits this round)
**Companion lane**: Agent 5 — `ClusterEvent` / `SiteCandidate` promotion contract

---

## Executive summary

Today, when the V2 ignition path is live (canonical post-`a63cded5` runtime), `binding_sites.json` is a hardcoded **empty stub** (`crates/prism-nhs/src/bin/nhs_rt_full.rs:9925-9940`). The legacy CPU clustering pipeline is bypassed by a hard gate at `nhs_rt_full.rs:9171-9175`, so the rich `clustered_sites: Vec<ClusteredBindingSite>` codepath that already builds proper sites (with lining residues, druggability, aromatic proximity, catalytic scoring, etc.) is **not reached**. Downstream `prism_canonical.py` reads an empty `sites: []` and either degrades to a fallback consensus or fails.

This document specifies the Rust-only materializer that replaces the stub. The materializer consumes Agent 5's `ClusterEvent::Promote` events emitted from in-flight ASC adjudication, joins them with the V2 path's already-captured sidecars (KCC, therm, phasor, ghost-tile, spatial-grid) and the engine's atomic state at teardown, and emits either a populated `sites: [...]` array or a structured `not_materialized` report (never a fake site).

The plan is **strictly additive in Rust**, lives in a new module `crates/prism-nhs/src/site_materializer.rs`, and replaces the 16-line stub at `nhs_rt_full.rs:9925-9940` with a single dispatch. No CUDA kernels are added or modified; Python is forbidden in the materializer per `docs/EXECUTION_POLICY.md`.

---

## §1 — Files inspected (file:line evidence)

| Concern | File | Lines |
|---|---|---|
| Current `binding_sites.json` stub writer (V2) | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 9925-9940 |
| V2 hard-gate that bypasses legacy CPU pipeline | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 9163-9175 |
| Legacy single-stream `binding_sites.json` writer (real sites) | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 2667-2937 |
| Legacy multi-stream `binding_sites.json` writer (real sites) | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 13886, 15464-15505 |
| Legacy batch-mode `binding_sites.json` writer (real sites) | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 3478-3554 |
| Too-small protein skip writer | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 1841-1857 |
| `phasor_kcc_state.json` writer | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 9370-9438 |
| `prism_therm_telemetry.json` writer | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 9440-9502 |
| `kcc_visualization.json` writer (Null-Manifest sidecar) | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 9504-9732 |
| `spatial_grid_state.json` writer | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 9734-9774 |
| `aromatic_centroids_map.json` writer | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 9812-9869 |
| `v2_ignition_summary.json` writer + lineage hash | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 9942-10048 |
| Stream-thread join + `(spikes, snapshots, sig_data, kcc_data)` tuple | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 9135-9158 |
| `AscSharedState` (consensus + phasor + event log + steering focus) | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 4090-4220 |
| `consensus_out` / `event_out` / `prior_density` aggregation (V2 phase 3) | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 9446-9500 |
| `phase3_run_id` construction (audit anchor) | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 3954-3963 |
| Site-build canonical wrapped transform | `crates/prism-nhs/src/transform/clustering_to_clustered_sites.rs` | 1-220 |
| `ClusteredBindingSite` struct | `crates/prism-nhs/src/persistent_engine.rs` | 203-238 |
| `LiningResidue` struct | `crates/prism-nhs/src/persistent_engine.rs` | 158-177 |
| `compute_lining_residues` method | `crates/prism-nhs/src/persistent_engine.rs` | 3528-3680 |
| `DruggabilityScore::from_site_with_catalytic` | `crates/prism-nhs/src/persistent_engine.rs` | 431-510 |
| `SitePersistenceTracker` / `TrackedSite` | `crates/prism-nhs/src/persistent_engine.rs` | 749-960 |
| `SiteManifest` (M1 canonical per-site type) | `crates/prism-nhs/src/site_manifest.rs` | 408-477 |
| `SiteManifest::from_lbvh_cluster_aabb` (only public constructor) | `crates/prism-nhs/src/site_manifest.rs` | 651-689 |
| `KccMetrics` (CLA-2 ghost pipeline) | `crates/prism-nhs/src/site_manifest.rs` | 500-559 |
| `ThermDossier` (CLA-2 ghost pipeline) | `crates/prism-nhs/src/site_manifest.rs` | 569-606 |
| Ghost tile schema (`GhostTileFrame`, 4096-byte aligned) | `crates/prism-nhs/src/ghost_tile.rs` | 69-217 |
| `GhostTileRing::allocate` | `crates/prism-nhs/src/ghost_tile.rs` | 237-360 |
| `__constant__ d_cluster_to_repr_residue[64]` populator | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 5433-5500 |
| `GhostTileFrame::cluster_id`, `causal_lead_residue`, `adjudication_code` | `crates/prism-nhs/src/cuda/ghost_tile_kernel.cu` | 47, 159, 290 |
| Ghost telemetry → SiteManifest serializer | `crates/prism-nhs/src/ghost_telemetry.rs` | 396-474 |
| `f1-switch` event detector (Construct path proof of discovery) | `crates/prism-nhs/src/ghost_telemetry.rs` | 536-566 |
| `M1ProducerGraph` / `SpikeToCluster4D` (M1 producer FFI scaffold) | `crates/prism-nhs/src/spike_to_cluster_4d.rs` | 281-470 |
| `KccSiteData` aggregation type | `crates/prism-nhs/src/transform/cluster_to_causome.rs` | 58-100 |
| `build_consensus_sites` (legacy multi-stream consensus) | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 10444-10465 |
| `consensus_threshold` formula (legacy) | `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 10444-10448 |

---

## §2 — Functions / structs found

### Required-Map item 1 — Exact writer of `binding_sites.json`

There is **not one writer but five**, depending on which engine path completes:

| Writer | File:line | Stub or real? |
|---|---|---|
| **V2 ignition teardown stub** | `nhs_rt_full.rs:9925-9940` | **STUB** — hardcoded `"binding_sites": []` |
| Too-small protein skip | `nhs_rt_full.rs:1841-1857` | Empty (informational; `skipped: true`) |
| Single-stream legacy | `nhs_rt_full.rs:2667-2937` | **Real** — full sites with lining residues, ranking, therm, m1_diff |
| Multi-stream legacy + reranker | `nhs_rt_full.rs:13886, 15464-15505` | **Real** — full sites with reranked pockets, background spikes, rescue history |
| Batch mode | `nhs_rt_full.rs:3478-3554` | **Real** — replicate-consensus sites |

When `cargo build --features=v2_ignition` is set and any stream had `v2_was_live = true`, the legacy block at 9163 returns `Ok(())` immediately after writing the stub. **This is the primary problem to solve.**

### Required-Map item 2 — Stub content today

```rust
// nhs_rt_full.rs:9929-9934
let bs_stub = serde_json::json!({
    "v2_ignition":  true,
    "binding_sites": [],
    "run_id":       phase3_run_id,
    "note": "Sites adjudicated in-flight. See v2_ignition_summary.json.",
});
```

The "see v2_ignition_summary" pointer is a misdirection — `v2_ignition_summary.json` does not contain materialized sites, only telemetry counts and the lineage integrity hash. Downstream consumers parsing `binding_sites.json` see zero sites.

### Required-Map item 3 — Existing site/pocket structs in Rust

| Struct | File:line | Purpose | Reusable? |
|---|---|---|---|
| `ClusteredBindingSite` | `persistent_engine.rs:203-238` | Legacy CPU clustering output; carries cluster_id, localization manifold, spike_count, lining_residues, druggability | **YES** — but tied to the legacy spike-clustering kernel; reuse the *struct shape* and the `LiningResidue` machinery, not the construction path |
| `SiteManifest` | `site_manifest.rs:408-477` | M1 canonical per-site Rust type with `CentroidManifold`, `CausalScalars`, sort lineage | **YES** — designed for exactly this purpose; needs new constructors |
| `TrackedSite` | `persistent_engine.rs:762-789` | Frame-tracker site (running averages) | **PARTIAL** — supplies `frame_count`, `consecutive_frames`, `is_persistent`, `max_consecutive` for `recurrence_score` |
| `RobustSite` | `mapping.rs:313` | Multi-replica aggregated site | **PARTIAL** — multi-stream support precedent |
| `PrismThermSiteResult` | `sdst_bridge.rs:114-160` | Therm-classified site (DYNAMIC/BISTABLE/INERT) | **YES** — supplies `therm_support`, `druggability`, `ccns_tau`, hysteresis fields |
| `PocketResonance` | `resonance_scan.rs:40` | UV-resonance pocket | Not directly reusable (different signal class) |
| `PhPocket` | `cubical_ph.rs:33` | Persistent-homology pocket | Not directly needed for V2 path |
| `CrypticSiteCandidate` | `active_sensing.rs:231` | Cryptic-site detector output | **YES** — supplies cryptic-aware support_score |
| `KccSiteData` | `transform/cluster_to_causome.rs:58-100` | Aggregated KCC fields per site | **YES** — direct input for `kcc_support` |

### Required-Map item 4 — Existing residue annotation structs

| Struct | File:line | Carries |
|---|---|---|
| `LiningResidue` | `persistent_engine.rs:158-177` | `chain`, `resid`, `resname`, `min_distance`, `n_atoms_in_pocket`, `spike_attribution_count` |
| `KccMetrics` | `site_manifest.rs:500-559` | `active_causal_steps`, `kcc_score`, `burst_motion`, `causal_lag`, `direction_score`, `lag_corr_peak`, `local_cov`, `motion_efficiency` |
| `CrypticSite::lining_residues` | `aromatic_proximity.rs:101` | (alternate path) |

### Required-Map item 5 — Existing cluster structs (cross-reference Agent 5)

Agent 5's `ClusterEvent::Promote` does not yet exist in the tree (verified by `rg -n "ClusterEvent|::Promote" crates/prism-nhs/src/`); this is the integration contract Agent 5 is designing. The closest analogues already in the engine:

| Struct | File:line | Role |
|---|---|---|
| `PersistentCluster` | `hierarchical_clustering.rs:55` | Hierarchical clustering output |
| `EpsilonCluster` | `hierarchical_clustering.rs:93` | Single-eps cluster |
| `ClusterChain` | `hierarchical_clustering.rs:102` | Cluster lineage across eps |
| `RtClusteringResult` | `rt_clustering.rs:79` | RT-spatial clustering result (fed into `ClusteringToClusteredSites`) |
| `MultiScaleClusteringResult` | `persistent_engine.rs:700` | Multi-scale clustering aggregate |
| `MergedCluster` | `persistent_engine.rs:685` | Cross-scale merge |
| `SpikeCluster` | `avalanche.rs:81` | Avalanche-grouped spikes |
| `ClusterAabb` | `pre_rank.rs:86` (Rust) + `cuda/pre_rank.cuh:71` (C++) | Cluster bounding box for adjudicator |
| `GhostTileFrame::cluster_id` | `ghost_tile.rs:78` | The ONLY in-flight cluster identity on the V2 path |

**Key fact**: on the V2 path, the `cluster_id` lives in two places:
1. The `__constant__ d_cluster_to_repr_residue[64]` slab written by `nhs_rt_full.rs:5481-5500` and read by the SO(3) kernel.
2. Each emitted `GhostTileFrame::cluster_id` (4096-byte sector-aligned) DMA'd through the ghost ring.

These are the **only honest cluster identities** available at V2 teardown. The materializer's input contract from Agent 5 must terminate at one of these two surfaces.

### Required-Map item 6 — Existing spatial-grid structs available at materialization time

| Struct / file | What's available at teardown |
|---|---|
| `merged_sig: SignalPreservationData` (`fused_engine.rs`) | Element-wise summed `voxel_hit_grid`, `coupled_spike_grid`, max-wins `primary_residue_id` / `primary_residue_count` per voxel — populated at `nhs_rt_full.rs:9333-9352` |
| `spatial_grid_state.json` payload | Voxel grid dimensions + counts — written at `nhs_rt_full.rs:9734-9774`; the data feeding the writer is in scope at the materializer call site |
| `merged_kcc: KccData` (`fused_engine.rs`) | Per-residue KCC vectors (n_residues × ~14 fields); populated at `nhs_rt_full.rs:9304-9329` |
| `accumulated_spikes` / `all_stream_spikes` | All spike events with `voxel_idx`, `position`, `intensity`, `phase_bits`, `timestep` — in scope on both legacy and V2 paths |
| `topology` (`PrismPrepTopology`) | Atom positions, residue ids, residue names, chain ids, CA indices |
| ghost-tile binary files `{stem}_stream{NN}_ghost_tiles.bin` | 4096-byte-aligned `GhostTileFrame` records with cluster_id, causal_lead_residue, KL divergence, geo power spectrum, adjudication_code |

### Required-Map item 7 — Existing KCC / therm sidecar structures the materializer can read

The V2 teardown phases run **before** the binding_sites.json stub. At line 9925, the materializer is in scope for:

- `merged_kcc` (`Option<KccData>`) — already merged across streams (`nhs_rt_full.rs:9304-9329`)
- `merged_sig` (`Option<SignalPreservationData>`) — already merged (`9331-9352`)
- `prior_density: Vec<f64>` — Bayesian per-residue density (`9476-9479`)
- `consensus_out: Vec<serde_json::Value>` — `(residue_id, n_groups, spc)` tuples (`9446-9466`)
- `event_out: Vec<serde_json::Value>` — PCMI/SURP timestamped events with parsed `r{rid}(KL=v)` (`9446-9474`)
- `asc_shared.group_residue_phasors` — per-group per-residue phasor accumulators
- `asc_shared.consensus_residues` — final consensus `(rid, n_groups, S_pc)` triples (locked Mutex; access pattern at `nhs_rt_full.rs:10067-10073`)
- `prism_therm_result: Option<PrismThermAnalysis>` — therm classification, druggability, ccns_tau (in scope on legacy path; on V2 path the equivalent flows through `prism_therm_telemetry.json` builder at 9440-9502 — same source data)
- `stream_summaries: Vec<Value>` — per-stream `(stream_id, n_spikes, n_frames)` tuples
- `phase3_run_id: String` — audit anchor

### Required-Map item 8 — Final output directory orchestrator

The output directory is assembled by the V2 teardown phases sequence in `nhs_rt_full.rs`:

```
9177  Phase 0: create_dir_all(args.output)
9181  output_base = args.output.join(structure_name)
9185  Phase 1: stream_results sweep + per-stream sidecars
9370  Phase 2: phasor_kcc_state.json
9440  Phase 3: prism_therm_telemetry.json
9504  Phase 3.5: kcc_visualization.json (Null-Manifest sidecar)
9734  Phase 4: spatial_grid_state.json
9812  Phase 6: aromatic_centroids_map.json
9870  Phase 7: v2_final.pdb
9925  Phase 8a: binding_sites.json   ← WHERE WE INTERVENE
9942  Phase 8b: v2_ignition_summary.json + lineage_integrity_hash
10058 return Ok(())
```

The materializer slots in at Phase 8a, BEFORE the lineage hash is computed (so the hash fingerprints the materialized payload, not the stub). The 16-line replacement block is the entire surface area of the change in `nhs_rt_full.rs`.

---

## §3 — Reusable code (no greenfield)

Anti-Greenfield Doctrine §2.3 — extend, don't duplicate. The materializer reuses:

1. **`SiteManifest::from_lbvh_cluster_aabb`** (`site_manifest.rs:651-689`) — already the canonical bridge from a per-cluster AABB to a `SiteManifest`. We add a **second** constructor `from_promoted_cluster_event` that takes Agent 5's promotion record + sidecar refs and populates the `kcc_metrics` / `therm_dossier` / centroids.
2. **`ClusteredBindingSite::compute_lining_residues`** (`persistent_engine.rs:3528-3680`) — geometric residue-lining computation with cutoff. Reused as-is via a free-function adapter that takes only the required slices and returns `Vec<LiningResidue>` (no `&mut self`).
3. **`compute_catalytic_score`** (`persistent_engine.rs:289-349`) — pure function over `&[LiningResidue]`. Reused directly.
4. **`DruggabilityScore::from_site_with_catalytic`** (`persistent_engine.rs:431-510`) — pure scoring. Reused directly.
5. **`AromaticProximityInfo`** (`persistent_engine.rs:241-259`) populator — reused via existing helper `compute_aromatic_proximity` (called from `nhs_rt_full.rs:2660-ish` on the legacy path).
6. **`PrismThermAnalysis` → `PrismThermSiteResult`** mapping in `sdst_bridge.rs:114-160` — supplies `therm_class`, `druggability`, `ccns_tau`, `relative_asymmetry`, `hysteresis_asymmetry` for the `ThermDossier`.
7. **`KccData` → `KccMetrics`** join at residue level — pattern already in `nhs_rt_full.rs:9620-9676`. Lift into `KccMetrics::from_kcc_data_at_residue(&KccData, idx) -> KccMetrics`.
8. **`build_consensus_sites`** signature (`nhs_rt_full.rs:10464`) — multi-stream support + spike-offset bookkeeping pattern reused for `stream_support: Vec<u32>`.
9. **`SitePersistenceTracker`** (`persistent_engine.rs:806-960`) — already produces `frame_count`, `consecutive_frames`, `is_persistent`, `max_consecutive`. The materializer will instantiate one, walk Agent 5's promotion records as if they were "frames," and read out `recurrence_score`.

---

## §4 — Unsafe / problematic

| Issue | Impact |
|---|---|
| **The V2 stub is hashed into the lineage integrity hash** at `nhs_rt_full.rs:9958-9963` (FNV-1a of the stub bytes). This hash flows into `v2_ignition_summary.json::lineage_integrity_hash.binding_sites_json`. Replacing the stub means **every prior run's hash becomes unstable across the boundary**. Mitigation: include a `schema_version: 3` bump and document the boundary in `docs/CANONICAL_PROVENANCE.md`. | Operator / consumer expects FNV stability across runs but not across schema bumps. |
| **`__constant__ d_cluster_to_repr_residue[64]` is a hard cap of 64 clusters per stream**. Larger proteins (e.g., 7c8r dimer) can produce >64 candidate clusters before promotion. | Materializer must read this cap from `prism-nhs/src/cuda/ghost_tile_kernel.cu:47` and emit `not_materialized` with reason `cluster_count_exceeds_constant_slab` rather than silently truncate. |
| **`GhostTileFrame::cluster_id` is u8** (`ghost_tile.rs:78`, offset_of asserts in `200-217`). | 256 cluster max per ring; consistent with the 64-slab cap above but documented separately. |
| **The legacy CPU clustering path is bypassed entirely on V2**. Calling it from the materializer reintroduces the bypassed work. | The materializer MUST NOT re-run CPU clustering. It must consume only what the GPU path produced. |
| **`AscSharedState` mutexes are still held by stream threads at teardown moment** — though all streams have joined by line 9160, the `consensus_residues` mutex pattern at `10067-10073` shows the safe access shape: `lock().map(|cr| cr.iter().clone().collect())`. | Materializer must follow that exact lock-and-snapshot idiom; never hold a guard while serializing. |
| **`prism_therm_result` is computed on the legacy path at `nhs_rt_full.rs:2940-2953` but is `None` on the V2 path** because `clustered_sites` is empty. | Materializer cannot use `prism_therm_result`; it must read the same upstream therm telemetry that feeds `prism_therm_telemetry.json` (Phase 3, lines 9440-9502). |
| **`ClusterId(u32::MAX)` is the `UNCLUSTERED` sentinel** (`site_manifest.rs:88`). Promotion events for `UNCLUSTERED` spikes must be rejected before site construction. | Hard-fail rule. |
| **Schema conflicts** — see §11 below. The planning-spec `MaterializedSite` shape uses `support_voxels: Vec<u64>`, but `GhostTileFrame::cluster_id` is `u8` and our voxel ids are `i32` in `SpikeEvent::voxel_idx`. The plan resolves this by widening at materialization time. |

---

## §5 — Exact plan

### Phase A — New module scaffolding (single new file)

Create `crates/prism-nhs/src/site_materializer.rs`. Module is `pub mod site_materializer` in `crates/prism-nhs/src/lib.rs`. No CUDA, no FFI, no Python.

The module exports:

```rust
pub struct SiteMaterializer { … }      // zero-sized; state-free
pub struct MaterializedSite { … }       // schema spec from operator (§7 below)
pub struct SiteResidue { … }            // schema spec from operator (§7 below)
pub struct FieldCompleteness { … }      // populated bitfield indicator
pub enum   MaterializationOutcome {
    Sites(Vec<MaterializedSite>),
    NotMaterialized {
        reasons: Vec<String>,
        available_evidence: Vec<String>,
        missing_fields: Vec<String>,
        fallback_consensus_candidates: Vec<FallbackConsensusCandidate>,
    },
}
pub struct FallbackConsensusCandidate {  // labeled-only; never primary site
    pub residue_id: i32,
    pub chain_id: Option<String>,
    pub residue_name: Option<String>,
    pub support_score: f32,
    pub label: &'static str,             // hardcoded "fallback_consensus_candidate"
}

pub struct MaterializerInputs<'a> {
    pub run_id: &'a str,
    pub topology: &'a PrismPrepTopology,
    pub promotion_events: &'a [ClusterEventPromote],     // Agent 5 producer
    pub stream_count: u32,
    pub merged_kcc: Option<&'a KccData>,
    pub merged_sig: Option<&'a SignalPreservationData>,
    pub prior_density: &'a [f64],
    pub asc_consensus: &'a [(i32, usize, f32)],          // snapshot
    pub asc_event_log: &'a [(u32, String)],              // snapshot
    pub group_residue_phasors: Option<&'a [Vec<(f64, f64, u32)>]>,
    pub all_stream_spikes: &'a [GpuSpikeEvent],
    pub stream_spike_offsets: &'a [usize],
    pub aromatic_positions: &'a [[f32; 3]],
    pub prism_therm_telemetry: Option<&'a PrismThermTelemetry>, // Phase-3 sidecar
    pub ghost_tiles_per_stream: &'a [Vec<GhostTileFrame>],      // post-DMA snapshot
    pub lining_cutoff: f32,
}

pub fn materialize(inputs: &MaterializerInputs<'_>) -> MaterializationOutcome;
```

### Phase B — `ClusterEventPromote` integration (Agent 5's surface)

Per the brief, `ClusterEvent::Promote` is Agent 5's design. The materializer treats it as an opaque struct with an agreed-upon shape:

```rust
pub struct ClusterEventPromote {
    pub stream_id: u32,                  // 0..stream_count
    pub cluster_id: u32,                 // engine-assigned (8-bit on the wire, widened)
    pub frame_first_seen: u64,
    pub frame_last_seen: u64,
    pub recurrence_count: u32,           // # frames cluster passed promotion gate
    pub multi_phase_signature: u8,       // bitmask: bit_i = "active in protocol phase i"
    pub aabb: Aabb,                      // from per-cluster AABB column in M1 producer
    pub causal_lead_residue: u32,        // from GhostTileFrame::causal_lead_residue
    pub spike_indices_in_stream: Vec<usize>, // local-to-stream; offset to global via stream_spike_offsets
    pub adjudication_code_history: Vec<u8>,  // 0=Prune, 1=Construct, 2=Violation; may end with Construct
    pub kl_divergence_at_promotion: f32,
    pub geo_power_spectrum: [f32; 6],
}
```

**Materializer constraint on Agent 5**: every emitted `ClusterEventPromote` MUST carry an `Aabb` and a non-`UNCLUSTERED` `cluster_id`. No promotion event = no site. Agent 5 owns the gate logic; the materializer trusts the gate and only re-validates the AABB-presence and cluster_id sentinel rules.

### Phase C — Per-event materialization pipeline

For each `ClusterEventPromote`:

1. **Reject sentinels**: drop events with `cluster_id == u32::MAX` or `recurrence_count == 0`.
2. **Construct centroid** from `aabb.center()` (existing call from `site_manifest.rs:661`).
3. **Build `lining_residues`** via the lifted `compute_lining_residues_pure` (lifted from `persistent_engine.rs:3528-3680`, identical math, takes slices instead of `&mut self`).
4. **Build `driver_residues`**: top-K residues from `causal_lead_residue` + KCC `direction_score` ranking; K=3 by default.
5. **Compute `recurrence_score`** = `recurrence_count / max(1, frame_last_seen - frame_first_seen)`.
6. **Compute `kcc_support`**: weighted sum of per-driver-residue KCC scores from `merged_kcc`, normalized to [0,1] using the existing legacy formula at `nhs_rt_full.rs:11018+`.
7. **Compute `therm_support`**: read therm_class for the centroid via `prism_therm_telemetry`; map `DYNAMIC` → 1.0, `BISTABLE` → 0.7, `Soc` → 0.5, `Responsive` → 0.5, `INERT` → 0.0.
8. **Compute `asc_support`**: fraction of `driver_residues` present in `asc_consensus` weighted by S_pc.
9. **Build `stream_support`**: from `spike_indices_in_stream` joined back through `stream_spike_offsets` to the global spikes list.
10. **Build `phase_support`**: decode `multi_phase_signature` bitmask to `Vec<String>` of phase names.
11. **Build `support_voxels`**: collect `spike.voxel_idx` for spikes assigned to this cluster, dedupe, widen `i32 → u64`.
12. **Build `construct_reason`**: human-readable composition of the gate signals — e.g., `"recurrence={r}/{frames}, n_streams={s}, n_phases={p}, kcc=Z.zz, therm={class}"`.
13. **Build `field_completeness`**: bitfield indicator per planned field (1=populated, 0=null/missing).
14. **Set `dag_node_id`** if a transform DAG audit handle is supplied; else `None`.

### Phase D — Output dispatch

```rust
match site_materializer::materialize(&inputs) {
    Sites(sites) => {
        let bs_payload = serde_json::json!({
            "v2_ignition":  true,
            "run_id":       phase3_run_id,
            "schema_version": 3,
            "sites":        sites,
            "binding_sites": sites.len(),
            "druggable_sites": sites.iter().filter(|s| s.druggability_is_druggable()).count(),
        });
        std::fs::write(&bs_path, serde_json::to_string_pretty(&bs_payload)?)?;
    }
    NotMaterialized { reasons, available_evidence, missing_fields, fallback_consensus_candidates } => {
        let bs_payload = serde_json::json!({
            "v2_ignition":  true,
            "run_id":       phase3_run_id,
            "schema_version": 3,
            "status":       "not_materialized",
            "reasons":      reasons,
            "available_evidence": available_evidence,
            "missing_fields": missing_fields,
            "fallback_consensus_candidates": fallback_consensus_candidates,
            "sites":        [],
        });
        std::fs::write(&bs_path, serde_json::to_string_pretty(&bs_payload)?)?;
    }
}
```

The lineage integrity hash at `nhs_rt_full.rs:9958` continues to FNV-1a the bytes — schema_version bump signals consumers to re-fingerprint.

### Phase E — Promotion criteria reuse (no new thresholds)

The materializer **does not invent thresholds**. It only checks AABB-presence and cluster_id sentinels (above). The promotion gate lives in Agent 5's lane. For reference, the existing thresholds in the codebase that Agent 5's gate may reuse:

| Threshold candidate | File:line | Current value |
|---|---|---|
| Multi-stream consensus threshold | `nhs_rt_full.rs:10444-10448` | `ceil(n_streams * 0.5)` if n_streams≥3 else 1 |
| Persistence threshold (50% of frames) | `persistent_engine.rs:932-937` | `total_frames * 0.5` |
| Site spatial-match threshold | `nhs_rt_full.rs:10464` | 10.0 Å |
| Site overlap merge | `nhs_rt_full.rs:10473-10500` | Jaccard ≥ 0.5 OR containment ≥ 0.7, dist ≤ 20 Å |
| Lining cutoff (default) | `args.lining_cutoff` | 5.0 Å (configured) |
| F1 SWITCH adjudication code | `ghost_telemetry.rs:546-547` | `tile.adjudication_code == 1` (Construct) |
| `compute_lining_residues` minimum pocket count | `persistent_engine.rs:3604` | 10 |
| Druggable threshold (with aromatics) | `persistent_engine.rs:408` | overall ≥ 0.40 |
| Druggable threshold (no aromatics) | `persistent_engine.rs:408` | overall ≥ 0.48 |

If Agent 5 needs new thresholds (e.g., minimum recurrence_count for promotion), they must be added in Agent 5's module with a documented citation, not invented in the materializer.

---

## §6 — Minimal touch list

| File | Change | Rationale |
|---|---|---|
| `crates/prism-nhs/src/site_materializer.rs` | **NEW** ~600 LoC | Materializer module |
| `crates/prism-nhs/src/lib.rs` | Add `pub mod site_materializer;` (1 line, after existing `pub mod site_manifest;`) | Module registration |
| `crates/prism-nhs/src/bin/nhs_rt_full.rs` | Replace lines 9925-9940 (16 lines) with materializer dispatch (~40 lines including the inputs assembly) | Single intervention point |
| `crates/prism-nhs/src/bin/nhs_rt_full.rs` | Lines 9925-9940 stub becomes opt-out fallback under a feature flag `legacy_v2_stub` for emergency rollback | Reversibility |
| `crates/prism-nhs/src/persistent_engine.rs` | Lift `compute_lining_residues` body into a free function `compute_lining_residues_pure` that the method delegates to. **No semantic change.** | Materializer needs to call it without `&mut self` |
| `crates/prism-nhs/src/site_manifest.rs` | Add `KccMetrics::from_kcc_data_at_residue(&KccData, idx) -> KccMetrics` (~20 lines) | Lift the JSON-builder pattern at `nhs_rt_full.rs:9620-9676` into a typed constructor |
| `crates/prism-nhs/src/site_manifest.rs` | Add `ThermDossier::from_telemetry_at(&PrismThermTelemetry, [f32;3]) -> ThermDossier` (~20 lines) | Same pattern as above for therm |
| `docs/CANONICAL_PROVENANCE.md` | Add provenance row for the materializer module + schema_version=3 bump | Provenance discipline |

**Total**: 1 new module, ~700 LoC; 6 surgical edits to existing files; zero CUDA changes; zero build.rs changes; zero Python.

---

## §7 — New structs / functions (verbatim per operator brief)

```rust
/// Materialized site emitted to binding_sites.json[].sites
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MaterializedSite {
    pub site_id: String,                     // {run_id}_site_{cluster_id}_stream{NN}
    pub rank: u32,                           // lexicographic post-rank slot
    pub centroid_xyz: [f32; 3],
    pub support_voxels: Vec<u64>,
    pub lining_residues: Vec<SiteResidue>,
    pub driver_residues: Vec<SiteResidue>,
    pub stream_support: Vec<u32>,
    pub phase_support: Vec<String>,
    pub recurrence_score: f32,
    pub kcc_support: f32,
    pub therm_support: f32,
    pub asc_support: Option<f32>,
    pub construct_reason: String,
    pub field_completeness: FieldCompleteness,
    pub dag_node_id: Option<String>,
}

/// One residue annotation in a MaterializedSite
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SiteResidue {
    pub chain_id: Option<String>,
    pub residue_id: u32,
    pub residue_name: Option<String>,
    pub role: String,                        // "lining" | "driver" | "anchor" | …
    pub support_score: f32,
}

/// Per-field populated/missing indicator
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FieldCompleteness {
    pub centroid_xyz: bool,
    pub support_voxels: bool,
    pub lining_residues: bool,
    pub driver_residues: bool,
    pub stream_support: bool,
    pub phase_support: bool,
    pub recurrence_score: bool,
    pub kcc_support: bool,
    pub therm_support: bool,
    pub asc_support: bool,
    pub field_completeness_score: f32,       // 0.0 — 1.0
}
```

### Schema conflicts with existing types — resolution

| Plan field | Existing type | Conflict | Resolution |
|---|---|---|---|
| `MaterializedSite::site_id: String` | `SiteId(pub u32)` (`site_manifest.rs:77`) | Type mismatch | `MaterializedSite::site_id` is a *display string* containing the run_id and the underlying `SiteId(u32)`. The newtype lives under `field_completeness.audit_site_id_u32` for round-trip fidelity. |
| `MaterializedSite::rank: u32` | None — legacy uses array order | No conflict | Add `rank` as the index into the lexicographic-sorted output. Stable within run. |
| `MaterializedSite::support_voxels: Vec<u64>` | `SpikeEvent::voxel_idx: i32` (`fused_engine.rs`) | i32 vs u64 | Materializer widens via `voxel_idx as u64`, masks negatives as a sentinel `u64::MAX` and logs. |
| `MaterializedSite::lining_residues: Vec<SiteResidue>` | `LiningResidue` (`persistent_engine.rs:159`) | Different shape | Provide `From<LiningResidue> for SiteResidue` with `role = "lining"`, `support_score = 1.0 - (min_distance / cutoff)`. No data loss; chain/resname/min_distance preserved. |
| `MaterializedSite::driver_residues: Vec<SiteResidue>` | None (no canonical driver type) | No conflict | New populator function reads top-K from KCC `direction_score` + ASC `causal_lead_residue`. |
| `MaterializedSite::phase_support: Vec<String>` | `PhaseBits` (`phase_bits.rs`) | bit-set vs Vec<String> | Materializer decodes the bit-set via the existing `phase_bits_schema` crate API; format strings are canonicalized. |
| `MaterializedSite::dag_node_id: Option<String>` | `TransformId(pub &'static str)` (`transform/mod.rs`) | static vs owned | Stringify on emission. |
| `SiteResidue::residue_id: u32` | `LiningResidue::resid: i32` and engine uses `i32` PDB-resid throughout | Sign mismatch | Materializer rejects negative residue ids (these are MD-internal sentinels) before construction. |

No conflict with `SiteManifest`, `KccMetrics`, `ThermDossier`, `CausalScalars`, `CentroidManifold` — `MaterializedSite` is the **JSON-emission view** while `SiteManifest` is the **in-memory canonical view**. Both can coexist; materialization can optionally also emit `SiteManifest` records into `site_manifests.json` for telemetry consumers.

---

## §8 — Acceptance tests

All tests are pure-Rust unit + integration; no CUDA required for these tests.

### Test A1 — `not_materialized_when_no_promotion_events`
- Inputs: `promotion_events: &[]`
- Expected: `NotMaterialized { reasons: ["no_cluster_promotion_events"], … }`
- Asserts: `binding_sites.json::status == "not_materialized"` and `sites.len() == 0`.

### Test A2 — `unclustered_sentinel_rejected`
- Inputs: one event with `cluster_id == u32::MAX`
- Expected: event dropped; if no other events → `NotMaterialized`.

### Test A3 — `single_event_full_evidence_emits_site`
- Inputs: one valid event + populated `merged_kcc` + populated `prism_therm_telemetry` + non-empty `topology`
- Expected: `Sites(vec![one_site])` with `field_completeness_score == 1.0`.
- Asserts: `lining_residues.len() > 0`, `driver_residues.len() > 0`, `kcc_support > 0`, `therm_support` finite.

### Test A4 — `partial_evidence_emits_site_with_completeness_lt_1`
- Inputs: valid event but `merged_kcc: None` and `prism_therm_telemetry: None`
- Expected: `Sites(vec![one_site])` but `kcc_support == 0` and `therm_support == 0`, `field_completeness.kcc_support == false`.

### Test A5 — `fallback_label_is_load_bearing`
- Inputs: zero promotion events but populated `asc_consensus`
- Expected: `NotMaterialized` with `fallback_consensus_candidates: [...]` and **every** candidate carries `label == "fallback_consensus_candidate"`.
- Negative: assert `MaterializedSite` is never constructed from fallback candidates.

### Test A6 — `cluster_id_widening_no_data_loss`
- Inputs: events with cluster_id ∈ [0, 255] (full u8 range)
- Expected: every emitted `site_id` round-trips to the original u32 cluster_id via parsing.

### Test A7 — `voxel_widening_handles_negative`
- Inputs: synthetic spike with `voxel_idx == -1`
- Expected: `support_voxels` contains `u64::MAX` sentinel (not a wrapped u64); a warning log is emitted; the test asserts the sentinel is documented in field_completeness as `support_voxels == false`.

### Test A8 — `lexicographic_rank_is_stable`
- Inputs: 5 events with controlled (recurrence, pass_fraction, stability, quality)
- Expected: rank order matches the immutable rule from `CLAUDE.md` "Persistence → pass_fraction → stability → quality."

### Test A9 — `lineage_hash_recomputed_after_materialization`
- Integration test on `nhs_rt_full` binary in dry-run mode
- Expected: when materializer runs, the `binding_sites_json` FNV-1a in `v2_ignition_summary.json` matches `fnv1a64` of the materialized payload bytes (NOT the stub bytes).

### Test A10 — `no_python_sneak_in_materializer`
- Compile-time invariant test (`#![deny(...)]` + grep test under `tests/`)
- Asserts no `pyo3`, no `Command::new("python")`, no inline Python.

### Test A11 — `legacy_path_unaffected`
- Run with `--features="" ` (no v2_ignition): assert `clustered_sites` path at `nhs_rt_full.rs:2667-2937` still produces real sites with the legacy schema.

### Test A12 — `cluster_count_exceeds_constant_slab`
- Inputs: 65 promotion events from a single stream
- Expected: `NotMaterialized { reasons: ["cluster_count_exceeds_constant_slab", …], … }` with `available_evidence` listing the 64 that fit.

---

## §9 — Failure modes

| Mode | Detection | Output |
|---|---|---|
| Zero promotion events | `inputs.promotion_events.is_empty()` | `not_materialized: ["no_cluster_promotion_events"]` |
| All events drop sentinel | post-filter Vec empty | `not_materialized: ["all_events_were_unclustered_sentinel"]` |
| `merged_kcc` is None AND therm telemetry empty | both Option/Vec empty | `not_materialized: ["no_kcc_evidence", "no_therm_evidence"]` (if no events) OR sites with kcc_support=0/therm_support=0 + field_completeness reflects |
| AABB has zero volume | `aabb.max - aabb.min` per-axis == 0 | event dropped; reason added to `available_evidence` log |
| Lining residues empty | post-`compute_lining_residues_pure` Vec empty | site emitted with `field_completeness.lining_residues == false`; rank still assigned but penalized (placed last in lexicographic order) |
| Driver residues empty (no causal_lead_residue) | KCC `direction_score` all-NaN | `driver_residues = []`, `field_completeness.driver_residues == false` |
| Cluster count exceeds constant slab (>64) | n events > 64 per stream | `not_materialized: ["cluster_count_exceeds_constant_slab"]` |
| `compute_lining_residues_pure` panics on bad indices | bounds check → `Result` | event dropped with reason `lining_residues_index_oob_for_stream{N}_cluster{C}` |
| Phasor mutex poisoned | `lock().is_err()` | falls back to None for ASC fields; `field_completeness.asc_support == false`; warning log |
| Topology atom count < 500 | upstream guard at `nhs_rt_full.rs:1841` already short-circuits | n/a |
| Spikes overflow stream offset | `stream_spike_offsets[i+1] < spike_indices_in_stream[max]` | event dropped, reason `stream_spike_offset_overflow_stream{N}` |
| Therm class unknown | `prism_therm_telemetry` carries class string not in known map | `therm_support = 0.0`, `construct_reason` includes `unknown_therm_class={s}`; site still emitted |

In every "site emitted" failure mode above, the operator's **strict no-fake-site rule** is preserved: each missing-data path is reflected in `field_completeness` and `construct_reason` is honest about which signals were absent.

---

## §10 — Rollback

The plan is reversible at three granularities:

1. **Build-flag rollback**: `--features="legacy_v2_stub"` re-enables the original 16-line stub at `nhs_rt_full.rs:9925-9940`. Default OFF after first commit; enabling restores prior bytes verbatim.
2. **Module rollback**: `git revert <sha>` on the single materializer commit removes `crates/prism-nhs/src/site_materializer.rs` and the lib.rs registration in one move; the dispatch site at `nhs_rt_full.rs` falls back to the (still-present) stub block that lives under `cfg(feature = "legacy_v2_stub")`.
3. **Schema rollback**: setting `schema_version: 2` in the JSON output (a single literal change) reverts the `binding_sites.json` shape to the pre-materializer schema; downstream `prism_canonical.py` accepts both versions during the deprecation window.

The lineage hash continues to FNV-1a the actual bytes regardless of which version was emitted, so rollback never produces a hash collision with a prior real-materialization run.

---

## §11 — Schema conflicts (cross-reference §7)

Recapped here for the final-summary callout:

1. `MaterializedSite::site_id: String` ↔ `SiteId(u32)` — resolved by display+audit pair.
2. `support_voxels: Vec<u64>` ↔ `voxel_idx: i32` — widening with sentinel for negatives.
3. `lining_residues: Vec<SiteResidue>` ↔ `Vec<LiningResidue>` — `From` impl, no data loss.
4. `phase_support: Vec<String>` ↔ `PhaseBits` bit-set — decode via `phase_bits_schema`.
5. `residue_id: u32` ↔ MD-internal `i32` resid — reject negatives at materialization boundary.
6. `dag_node_id: Option<String>` ↔ `TransformId(&'static str)` — stringify on emit.

No conflicts requiring a struct redesign. `MaterializedSite` is JSON-only; `SiteManifest` remains the in-memory canonical type and is unchanged.

---

## Appendix — Provenance citations

- Stub writer: `crates/prism-nhs/src/bin/nhs_rt_full.rs:9925-9940`
- V2 hard-gate: `crates/prism-nhs/src/bin/nhs_rt_full.rs:9163-9175`
- `SiteManifest`: `crates/prism-nhs/src/site_manifest.rs:408-477`
- `ClusteredBindingSite`: `crates/prism-nhs/src/persistent_engine.rs:203-238`
- `LiningResidue`: `crates/prism-nhs/src/persistent_engine.rs:158-177`
- `compute_lining_residues`: `crates/prism-nhs/src/persistent_engine.rs:3528-3680`
- `KccMetrics` / `ThermDossier`: `crates/prism-nhs/src/site_manifest.rs:500-606`
- Ghost-tile schema: `crates/prism-nhs/src/ghost_tile.rs:69-217`
- `__constant__ d_cluster_to_repr_residue[64]`: `crates/prism-nhs/src/cuda/ghost_tile_kernel.cu:47`
- `f1-switch` event proof-of-discovery: `crates/prism-nhs/src/ghost_telemetry.rs:536-566`
- Multi-stream consensus threshold: `crates/prism-nhs/src/bin/nhs_rt_full.rs:10444-10448`
- Persistence threshold (50% frames): `crates/prism-nhs/src/persistent_engine.rs:932-937`
- AscSharedState (consensus + phasor + event log): `crates/prism-nhs/src/bin/nhs_rt_full.rs:4090-4220`
- V2 phase 3.5 KCC sidecar (Null-Manifest): `crates/prism-nhs/src/bin/nhs_rt_full.rs:9504-9732`
- `phase3_run_id`: `crates/prism-nhs/src/bin/nhs_rt_full.rs:3954-3963`
- `clustering_to_clustered_sites` audit transform: `crates/prism-nhs/src/transform/clustering_to_clustered_sites.rs:1-220`
- Anti-Greenfield Doctrine: `docs/PRISM4D_ENTANGLED_TRANSFORM_BLUEPRINT.md`
- Execution policy (Rust/CUDA/Python boundaries): `docs/EXECUTION_POLICY.md`

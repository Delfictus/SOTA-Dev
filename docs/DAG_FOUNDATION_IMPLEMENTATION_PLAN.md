# PRISM-4D Transform / Evidence DAG — Foundation Implementation Plan

Status: **PLAN ONLY — read-only scout** (Agent 1, AGENT 1 — Transform / Evidence DAG Foundation).
Branch baseline: `producer-repair-causal-truthing-20260426` @ `8ca26189`.
Author: Agent 1 (read-only). Implementation step is downstream (steps 2–3 of the documented sequence). No Rust/CUDA/build/runtime files are modified by this scout.

This plan is the foundation for the provenance spine that will eventually carry runtime events, stream/chunk/phase state, G26/F1/WHILE decisions, ASC steering, KCC state, spike/voxel support, cluster events, site materialization, static-equivalent bridge, dossier artifacts, and validation outputs. **First emission (DAG-1) is intentionally narrow: it only emits `transform_dag.json` connecting the Run + RunConfig + InputTopology + Streams to the artifacts that already land on disk at run-end.** No invented site nodes, no F1/WHILE/SiteCandidate population in DAG-1.

---

## 1. Files inspected (with file:line evidence)

All paths absolute under repo root `/home/diddy/Desktop/Prism4D-bio`.

| File | Lines read | Why |
|---|---|---|
| `crates/prism-core/src/lib.rs` | 1–55 | Confirm prism-core re-export surface; identify natural home for `dag` module. |
| `crates/prism-core/src/telemetry.rs` | 1–100 | Existing PZFR ring buffer is hot-path — orthogonal to DAG (run-end emit). |
| `crates/prism-core/Cargo.toml` | 1–40 | Confirm `serde`, `serde_json`, `anyhow`, `thiserror`, `log` already present (workspace deps). |
| `crates/prism-nhs/src/bin/nhs_rt_full.rs` | 142–260, 1035–1147, 3940–4000, 9900–10070, 15400–15510, 15600–15760, 16100–16170, 16500–16600 | Locate `Args`, `phase3_run_id`, output_base construction, all artifact emission sites (V2 teardown phases 2–8), and run-end terminal log. |
| `crates/prism-nhs/src/lib.rs` | 92–470 | Confirm `pub mod transform`, `pub mod site_manifest`, `pub mod ghost_telemetry`, `pub mod spike_arrow_writer`, `pub mod input` are public crate surface. |
| `crates/prism-nhs/src/transform/mod.rs` | 1–635 | Existing Phase 2/3 transform-audit spine: `AuditedTransform`, `AuditOutcome`, `TransformViolation`, `AuditRecord`, `LawId`, `DeterminismClass`, `TolerancePolicy`. The DAG plan re-uses these names verbatim — DAG `transform` edges cite this vocabulary. |
| `crates/prism-nhs/src/site_manifest.rs` | 370–477 | `SiteManifest`, `SiteIdentity`, `CentroidManifold` already serde — DAG nodes referencing sites in later phases will use `SiteId(u32)` (l. 77) as identity. |
| `crates/prism-nhs/src/captured_pipeline.rs` | 1–80 | Cross-lane node ownership table (B/C/C'/D/Body/DMA) is the future seed for `DagNodeKind::GraphLaunch` / `G26Decision` / `F1Decision`; DAG-1 does NOT populate these. |
| `crates/prism-nhs/Cargo.toml` | 60–140 | `serde_json`, `chrono`, `arrow`, `anyhow` already in `[dependencies]`. |
| `Cargo.toml` (workspace) | 73–131 | Workspace-level: `serde 1.0`, `serde_json 1.0`, `anyhow 1.0`, `thiserror 1.0`, `chrono 0.4 (serde)`, `sha2 0.10`. **No `uuid` dependency** — DAG node IDs use `String` constructed from `(run_id, kind, ordinal)`, not UUIDs. |
| `docs/EXECUTION_POLICY.md` | 1–80 | Three-layer rule: DAG emit must be **Rust** (orchestration / persistence). No Python in hot path; no CUDA inside DAG emit. |
| `docs/TIER8_GRAPH_TOPOLOGY.md` | 1–60 | Confirms G26 is the live conditional and F1 is deferred; DAG schema declares both kinds but DAG-1 emits neither. |

Key file:line anchors used by the plan below:

- `crates/prism-nhs/src/bin/nhs_rt_full.rs:3953-3963` — `phase3_run_id = format!("{structure_name}_{UTC YYYYMMDD_HHMMSS}")`. **This is the canonical run identity. The DAG manifest reuses it verbatim.**
- `crates/prism-nhs/src/bin/nhs_rt_full.rs:9925-10049` — V2 teardown writes binding_sites stub, lineage_integrity_hash, v2_ignition_summary. DAG-1 emits **after** this block — same file, same scope.
- `crates/prism-nhs/src/bin/nhs_rt_full.rs:9948-9957` — `fnv1a64` helper currently nested inside V2 teardown. The DAG module will host its own copy (or the file-local helper can be lifted; out of scope for DAG-1 minimal edit).
- `crates/prism-nhs/src/bin/nhs_rt_full.rs:10013-10022` — existing `artifacts: [...]` literal-string list. **DAG-1 produces a richer typed equivalent; this string array is left untouched** (legacy consumer compatibility — Phase 3 schema v2).
- `crates/prism-nhs/src/bin/nhs_rt_full.rs:15466-15468` — `binding_sites.json` top-level: `schema_version: 2, run_id: phase3_run_id, replica_seed: args.replica_seed`. DAG references both fields via `DagArtifactRef.schema_version` and `DagRunManifest.run_id`.
- `crates/prism-nhs/src/bin/nhs_rt_full.rs:15634` — `kcc_visualization.json` written via `output_base.with_extension(...)`.
- `crates/prism-nhs/src/bin/nhs_rt_full.rs:16544-16584` — terminal binary export block (phasors.bin, asc_events.bin, gcpid_synergy.json). **DAG-1 emit point sits immediately after line 16584, BEFORE the `MULTI-STREAM PIPELINE COMPLETE` log banner on line 16586.**
- `crates/prism-nhs/src/bin/nhs_rt_full.rs:16586-16599` — final log banner, then `Ok(())` returns from `run_multi_stream_pipeline`. The DAG emit must be `Ok`-tolerant: a failed write logs a warning and does **not** propagate.

---

## 2. Relevant functions / structs / types found

| Name | File:line | Reuse for DAG |
|---|---|---|
| `phase3_run_id: String` (local) | `nhs_rt_full.rs:3959-3963` | Used as `DagRunManifest.run_id` verbatim. |
| `args.replica_seed: u64` | `nhs_rt_full.rs:208-210` | Stored in `DagRunManifest.metadata["replica_seed"]`. |
| `args: Args` (clap Parser, derives `Clone`) | `nhs_rt_full.rs:142-145` | Serialized into `RunConfig` node metadata. **Args derives `Clone` but NOT `Serialize`** — see §7 for `args_to_json` helper. |
| `phase3_run_id` consumers | `nhs_rt_full.rs:9932, 15468, 15740, 16144` | DAG hash invariant: same run_id everywhere. |
| `output_base: PathBuf` | `nhs_rt_full.rs:10736` (`= args.output.join(&structure_name)`) | DAG file path is `output_base.with_extension("transform_dag.json")`. |
| `structure_name: String` | `nhs_rt_full.rs:3949-3951` | Stored in `DagRunManifest.target` and used to derive node IDs. |
| `n_streams: usize` | (inside multi-stream loop) | One `DagNode { kind: Stream }` per stream id. |
| `fnv1a64` (local fn) | `nhs_rt_full.rs:9948-9957` | Algo blueprint for `DagArtifactRef.hash`; DAG module will own its own private copy under the same algo. (`sha2 = "0.10"` is in workspace as a stronger upgrade path; DAG-1 keeps fnv1a64 to match the existing `lineage_integrity_hash` algo — operator already accepted this hash class for "auditability gate, not adversarial-tamper-resistant".) |
| `lineage_integrity_hash` JSON object | `nhs_rt_full.rs:10025-10030` | Survives unchanged; DAG produces a parallel typed view, not a replacement. |
| `transform::AuditedTransform` / `AuditOutcome` / `TransformId` / `LawId` | `transform/mod.rs:86, 481, 579-633` | When future DAG edges carry transform identity (TIER 9), they cite `TransformId.0` as a stable string. DAG-1 does not emit transform edges. |
| `site_manifest::SiteManifest` / `SiteId` / `ClusterId` | `site_manifest.rs:77, 408` | Future `SiteCandidate` / `RankedSite` nodes will carry `SiteId(u32)`. **DAG-1 produces no site nodes.** |
| `PrismPrepTopology::source_pdb` | `input.rs` (cited by `nhs_rt_full.rs:15574`) | Stored as the value of the `InputTopology` node's `metadata.pdb_source`. |
| `chrono::Utc::now()` | `nhs_rt_full.rs:3962` | DAG manifest `created_utc` timestamp. |

---

## 3. Existing reusable code (serde / hashing / writers / id minting)

| Capability | Where it lives | DAG-1 reuse strategy |
|---|---|---|
| run_id minting | `nhs_rt_full.rs:3959-3963` (`structure_name + chrono::Utc::now().format("%Y%m%d_%H%M%S")`) | **Reuse verbatim**. DAG does not mint a new id; it reads `phase3_run_id` from the existing local. |
| FNV-1a 64-bit hashing | `nhs_rt_full.rs:9948-9957` | Same constants, same loop, copied into `prism-core::dag` module-private `fn fnv1a64(&[u8]) -> u64`. (Promotion to `sha2::Sha256` is a future upgrade; DAG-1 keeps fnv1a64 to harmonize with `lineage_integrity_hash`.) |
| `serde_json::to_string_pretty` writers | Used dozens of places (e.g. `nhs_rt_full.rs:15504`, `9935`, `10043`) | Same idiom for `transform_dag.json`. |
| Pretty-write to file with warn-on-failure | `nhs_rt_full.rs:9937-9939` | Identical idiom for DAG emit (warn-only, never propagate). |
| Hash file by reading bytes | `nhs_rt_full.rs:9959-9963` (`fs::read` → `fnv1a64`) | DAG `DagArtifactRef.hash` uses the same pattern. **For files > 64 MiB (e.g. ghost_tiles.bin which is recorded as `(size, head4k, tail4k)` fingerprint at `nhs_rt_full.rs:9972-10005`), the same head/tail+size fingerprint algorithm is reused** — `DagArtifactRef.hash` includes a `size_bytes` field so consumers can detect the truncated-fingerprint case. |
| Optional-key-skipping serde | `site_manifest.rs:438` (`#[serde(default, skip_serializing_if = "Option::is_none")]`) | All optional `DagNode.stream_id`, `DagNode.chunk_id`, `DagNode.phase`, etc., use the same attribute. |
| `runtime_config.rs` (`KernelTelemetry`, `RuntimeConfig`) | `prism-core/src/runtime_config.rs` (re-exported in lib.rs:46) | Naming precedent for "RunConfig" — DAG `RunConfig` node carries a snapshot of `Args` as JSON, not a struct merge. |
| `chrono = { version = "0.4", features = ["serde"] }` | workspace + nhs Cargo.toml | DAG manifest `created_utc: DateTime<Utc>`. |

**Crates already importable from a new `prism-core` module**: `serde`, `serde_json`, `anyhow`, `thiserror`, `log`. **No new dependencies required for DAG-1.**

---

## 4. Existing unsafe / problematic code that would conflict with DAG insertion

| Issue | File:line | Conflict severity | Mitigation |
|---|---|---|---|
| `fnv1a64` is a function-local helper inside `run_multi_stream_pipeline` (line 9950) — invisible outside that scope. | `nhs_rt_full.rs:9948-9957` | **Low.** DAG module duplicates the algorithm (six-line copy) instead of trying to lift the local. Lifting it to a `prism-core` helper is a separate refactor. | Duplicate, document with a `// Mirror of nhs_rt_full.rs:9948-9957 — keep in sync` comment. |
| Existing top-level `artifacts: [strings]` list at `nhs_rt_full.rs:10013-10022` is manually maintained and is **wrong** for some artifact paths (e.g. `binding_sites.json` is listed as a sibling of v2 teardown but is actually written by the multi-stream path at line 15504; the V2-teardown stub at 9936 is a different file with `binding_sites: []`). | `nhs_rt_full.rs:10013-10022` | **Medium — but out of DAG-1 scope.** DAG-1 emits the **typed** view based on real artifact paths; the legacy string list survives untouched. A future cleanup deduplicates them; the existing list stays as a Phase-3 schema v2 marker. | Do not touch the legacy list. DAG-1 is additive. |
| `Args: clap::Parser, Clone` — **does not derive `Serialize`** (verified: `#[derive(Parser, Clone)]` at `nhs_rt_full.rs:142`). | `nhs_rt_full.rs:142-145` | **Low.** DAG-1 implements an explicit `args_to_run_config_json(&args, run_id) -> serde_json::Value` helper that whitelists the canonical fields (steps, fast, fast_25k, multi_stream, fused_steps, replica_seed, hmr, adaptive_dt, multi_differential, closed_loop_steering, asymmetric_steering, use_xgb_ranker, lining_cutoff, spike_percentile). | See §7 helper signature. **Does not require derive change on `Args`.** |
| Multiple terminal-write blocks in `nhs_rt_full.rs` (V2 teardown @ 9925, multi-stream finalize @ 15400, per-site spike export @ 16100, terminal binary export @ 16544) all return `Ok(())` regardless of failure — DAG must mirror this **warn-only** posture so a DAG write failure cannot break a run. | `nhs_rt_full.rs:9938, 9939, 9947, 16584-16599` | None. | DAG emit wraps the write in `match … { Ok(()) => log::info!(...), Err(e) => log::warn!("[DAG] emit failed: {}", e) }` and never returns the error. |
| The V2 teardown `return Ok(())` at line 10063 ends the function early on the v2-ignition path — multi-stream non-ignition runs continue past it to the finalize block at ~16599. **There are two run-end exit points.** | `nhs_rt_full.rs:10063, 16599` | **Medium.** DAG-1 must wire emit at **both** sites or use a helper called once per exit path. | See §6 file touch list — DAG-1 wires only at the multi-stream exit (`16584`). The V2 teardown path has its own `v2_ignition_summary.json` + `lineage_integrity_hash` — DAG-1 leaves it as-is and a follow-up adds the v2-teardown DAG variant in a second commit. **Plan-level note: this is intentional minimal scope.** |
| Non-finite floats panic `serde_json` writers — `nhs_rt_full.rs:15528-15544` already has the `lc_raw.is_finite()` guards. | n/a | None. | DAG node metadata never stores raw float fields in DAG-1. Only `size_bytes: u64` and hex-string hashes. |

**No `unsafe { ... }` blocks are touched by the DAG plan.** DAG emission is pure Rust + serde + filesystem write; no FFI, no CUDA, no Python.

---

## 5. Exact proposed implementation plan

### 5.1 Module location

**Recommended: `crates/prism-core/src/dag.rs`** (new file). Reasons:

1. `prism-core` is the canonical "core types and traits" crate (`crates/prism-core/src/lib.rs:1-32`) and already hosts cross-lane primitives (`runtime_config.rs`, `telemetry.rs`, `traits.rs`).
2. `prism-core` does not depend on CUDA / GPU features — DAG emission is pure Rust orchestration, which lines up with the EXECUTION_POLICY Layer 1 ownership.
3. `prism-nhs` already depends on `prism-core` transitively (most binaries import via `prism-nhs::config::*` which re-exports `prism_core` types). **Confirmed: `prism-nhs/Cargo.toml` does not currently `path = "../prism-core"` directly** — see §6 for the one-line dependency addition.
4. Future consumers (`prism-cli`, `prism-validation`, `prism-pipeline`, replicate driver) will all need `DagRunManifest` types; placing them in `prism-core` avoids `prism-nhs` becoming a dependency of those crates.

**Alternative considered: `crates/prism-nhs/src/dag.rs`** — rejected. Reasons: (a) prism-nhs already exceeds 80 modules and is the heaviest-build crate; (b) replicate / consensus / canonical pipeline tooling lives outside prism-nhs and would need to depend on it just for the manifest schema.

### 5.2 Re-export wiring (prism-core)

In `crates/prism-core/src/lib.rs`, add (after line 38 `pub mod traits;`):

```rust
/// Transform/Evidence DAG manifest — provenance spine for run-end
/// artifact assembly. See `crates/prism-core/src/dag.rs`.
pub mod dag;

pub use dag::{
    DagArtifactRef, DagEdge, DagEdgeKind, DagInvariant, DagNode,
    DagNodeKind, DagRunManifest, DagWriteResult,
};
```

### 5.3 Wire-up call site for DAG-1 emission

**Single emit call site:** `crates/prism-nhs/src/bin/nhs_rt_full.rs` between lines **16584** (end of phasor / asc_events binary export) and **16586** (start of `MULTI-STREAM PIPELINE COMPLETE` log banner). All artifacts referenced by DAG-1 have already been written by this point:

- `<output>/<structure>.binding_sites.json` — written at `nhs_rt_full.rs:15504`
- `<output>/<structure>.kcc_visualization.json` — `15634`
- `<output>/<structure>.kcc_validation.json` — `15643`
- `<output>/<structure>.kcc_session.pml` — `15761`
- `<output>/<structure>.spike_events.arrow` — `16196`
- `<output>/<structure>.gcpid_synergy.json` — `16499`
- `<output>/<structure>.phasors.bin` — `16546`
- `<output>/<structure>.asc_events.bin` — `16569`
- (Conditional, V2-ignition path) `<output>/<structure>.phasor_kcc_state.json` — `9431`
- (Conditional, V2-ignition path) `<output>/<structure>.prism_therm_telemetry.json` — `9496`
- (Conditional, V2-ignition path) `<output>/<structure>.spatial_grid_state.json` — `9769`
- (Conditional, V2-ignition path) `<output>/<structure>.t7_calibration.json` — `9791`
- (Conditional, V2-ignition path) `<output>/<structure>.aromatic_centroids_map.json` — `9863`
- (Conditional) `<output>/<structure>_stream{NN}_ghost_tiles.bin` per stream — multi-stream fan-out (referenced `9967-9978`)

The emit:

```rust
// ── Transform DAG (DAG-1): provenance spine for run-end artifacts ──
// Emitted last so all hashed files exist on disk. Warn-only:
// a DAG emit failure must NOT break a successful run.
{
    let dag_path = output_base.with_extension("transform_dag.json");
    let mut builder = prism_core::dag::DagRunManifest::new(
        &phase3_run_id,
        &structure_name,
    );
    builder.set_replica_seed(args.replica_seed);
    builder.set_run_config_json(args_to_run_config_json(&args));
    builder.add_input_topology(topology_path); // hashes input topology JSON
    for sid in 0..n_streams {
        builder.add_stream(sid as u32);
    }
    // DAG-1 artifact set (real on-disk files only).
    let candidate_artifacts: &[(&str, std::path::PathBuf)] = &[
        ("binding_sites.json",        output_base.with_extension("binding_sites.json")),
        ("kcc_visualization.json",    output_base.with_extension("kcc_visualization.json")),
        ("kcc_validation.json",       output_base.with_extension("kcc_validation.json")),
        ("kcc_session.pml",           output_base.with_extension("kcc_session.pml")),
        ("spike_events.arrow",        args.output.join(format!("{}.spike_events.arrow", structure_name))),
        ("gcpid_synergy.json",        args.output.join(format!("{}.gcpid_synergy.json", structure_name))),
        ("phasors.bin",               args.output.join(format!("{}.phasors.bin", structure_name))),
        ("asc_events.bin",            args.output.join(format!("{}.asc_events.bin", structure_name))),
        ("phasor_kcc_state.json",     output_base.with_extension("phasor_kcc_state.json")),
        ("prism_therm_telemetry.json",output_base.with_extension("prism_therm_telemetry.json")),
        ("spatial_grid_state.json",   output_base.with_extension("spatial_grid_state.json")),
        ("t7_calibration.json",       output_base.with_extension("t7_calibration.json")),
        ("aromatic_centroids_map.json", output_base.with_extension("aromatic_centroids_map.json")),
    ];
    for (kind, p) in candidate_artifacts {
        // skip-if-missing — multi-stream non-V2 path will not have phasor_kcc_state.json etc.
        if p.exists() {
            builder.add_artifact(kind, p);
        }
    }
    // Per-stream ghost_tiles.bin (multi-stream fan-out).
    {
        let stem = output_base.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        for sid in 0..n_streams {
            let p = output_base.with_file_name(format!("{}_stream{:02}_ghost_tiles.bin", stem, sid));
            if p.exists() {
                builder.add_artifact_for_stream("ghost_tiles.bin", p, sid as u32);
            }
        }
    }
    match builder.emit_pretty(&dag_path) {
        Ok(DagWriteResult { bytes_written, n_nodes, n_edges, n_artifacts }) => {
            log::info!(
                "  ✓ Transform DAG: {} ({} nodes, {} edges, {} artifacts, {} bytes)",
                dag_path.display(), n_nodes, n_edges, n_artifacts, bytes_written
            );
        }
        Err(e) => {
            log::warn!("  [DAG] emit failed: {} (run continues)", e);
        }
    }
}
```

`args_to_run_config_json` is a small file-local helper (defined alongside the call site or in a `mod local_dag_helpers { ... }` inside `nhs_rt_full.rs`) — see §7.

### 5.4 Schema (verbatim, exactly as the mission specified)

`prism-core::dag` declares the schema below. **schema_version is `1` for the first emission.** When a future DAG-2 adds populated G26/F1/SiteCandidate/etc., the version bumps and the additive enum variants land under `#[non_exhaustive]` to preserve readers.

```rust
pub const DAG_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagRunManifest {
    pub schema_version: u32,
    pub run_id: String,
    pub target: String,                       // structure_name
    pub created_utc: chrono::DateTime<chrono::Utc>,
    pub nodes: Vec<DagNode>,
    pub edges: Vec<DagEdge>,
    pub artifacts: Vec<DagArtifactRef>,
    pub invariants: Vec<DagInvariant>,
    /// Reserved bag for non-load-bearing fields (e.g. replica_seed, host info).
    /// Always emitted — empty object when nothing extra to store.
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagNode {
    pub id: String,                           // e.g. "run::4lpk_20260504_120000"
    pub kind: DagNodeKind,
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream_id: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_id: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phase: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact_ref: Option<String>,         // points at DagArtifactRef.id
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,                 // hex fnv1a64
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagEdge {
    pub from: String,                          // node id
    pub to: String,
    pub kind: DagEdgeKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transform: Option<String>,             // mirrors transform::TransformId.0
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagArtifactRef {
    pub id: String,
    pub path: std::path::PathBuf,
    pub kind: String,                          // "binding_sites.json", "ghost_tiles.bin", ...
    pub schema_version: Option<u32>,           // populated for JSON artifacts when known
    pub size_bytes: u64,
    pub hash: String,                          // hex fnv1a64; for >64MiB binaries, head/tail+size fingerprint
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagInvariant {
    pub id: String,
    pub description: String,
    pub status: String,                        // "asserted" | "unverified" | "violated"
    pub evidence_node_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum DagNodeKind {
    Run, InputTopology, RunConfig, Stream, Chunk, ProtocolPhase,
    GraphLaunch, G26Decision, F1Decision, WhileIteration,
    ASCSnapshot, KCCSnapshot, SpikeBatch, VoxelSupportSet,
    ClusterEvent, SiteCandidate, RankedSite, StaticEquivalentPocket,
    DossierArtifact, ValidationMetric, BinaryArtifact, JsonArtifact,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum DagEdgeKind {
    Contains, Produces, DerivedFrom, Supports, TransformsTo, Validates, Rejects,
}
```

### 5.5 DAG-1 graph shape (what is emitted on the first run)

```text
Run("run::<run_id>")
   ├─ Contains → InputTopology("topology::<run_id>")          [hash of input topology JSON]
   ├─ Contains → RunConfig("run_config::<run_id>")            [args whitelist]
   ├─ Contains → Stream("stream::<run_id>::00") … Stream(<NN>)
   └─ Produces → JsonArtifact("artifact::binding_sites")
   └─ Produces → JsonArtifact("artifact::kcc_visualization")
   └─ Produces → JsonArtifact("artifact::kcc_validation")
   └─ Produces → JsonArtifact("artifact::kcc_session_pml")
   └─ Produces → BinaryArtifact("artifact::spike_events_arrow")
   └─ Produces → JsonArtifact("artifact::gcpid_synergy")
   └─ Produces → BinaryArtifact("artifact::phasors_bin")
   └─ Produces → BinaryArtifact("artifact::asc_events_bin")
   (V2-ignition path additions, conditional on file presence:)
   └─ Produces → JsonArtifact("artifact::phasor_kcc_state")
   └─ Produces → JsonArtifact("artifact::prism_therm_telemetry")
   └─ Produces → JsonArtifact("artifact::spatial_grid_state")
   └─ Produces → JsonArtifact("artifact::t7_calibration")
   └─ Produces → JsonArtifact("artifact::aromatic_centroids_map")

   For each stream sid where ghost_tiles.bin exists:
       Stream("stream::<run_id>::<sid>")
          └─ Produces → BinaryArtifact("artifact::ghost_tiles::<sid>")
```

**Deliberate omissions (DAG-1):** No `Chunk`, `ProtocolPhase`, `GraphLaunch`, `G26Decision`, `F1Decision`, `WhileIteration`, `ASCSnapshot`, `KCCSnapshot`, `SpikeBatch`, `VoxelSupportSet`, `ClusterEvent`, `SiteCandidate`, `RankedSite`, `StaticEquivalentPocket`, `DossierArtifact`, `ValidationMetric` nodes. The enum **declares** them so DAG-2/DAG-3 can populate without a schema bump; DAG-1 leaves them empty.

### 5.6 Invariants checked at emit time (DAG-1)

The DAG module emits two invariants by default; both fail-closed (set to `"violated"` if breached, never panic):

1. `inv::all_artifact_paths_exist` — every `DagArtifactRef.path` was opened with `std::fs::metadata` successfully **before** being added. (This is enforced upstream at the `if p.exists()` check in §5.3, so the invariant is `"asserted"` by construction.)
2. `inv::run_id_consistent` — the manifest's `run_id` equals the `phase3_run_id` recorded in `binding_sites.json` (`schema_version: 2, run_id: <...>`). The DAG module **does not parse the binding_sites.json** in DAG-1 (avoids re-reading a potentially large file); the invariant status is `"unverified"` for now and a follow-up gate will lift it to `"asserted"` after the lineage_integrity_hash module exposes the parsed run_id. **Status: documented gap.**

---

## 6. Minimal file touch list

| File | Change | Reason |
|---|---|---|
| **NEW** `crates/prism-core/src/dag.rs` | New module (~400 LOC: types + builder + emit + helpers + tests). | Houses `DagRunManifest`, all enums, `DagWriteResult`, `fnv1a64` (private), file hashing, builder API. |
| `crates/prism-core/src/lib.rs` | Add `pub mod dag;` and 6 re-exports (one-block). | Public surface for downstream consumers. |
| `crates/prism-core/Cargo.toml` | **No change.** `serde`, `serde_json`, `chrono` (via workspace) all present. **`chrono` is NOT currently in `crates/prism-core/Cargo.toml` — needs adding.** | DAG manifest carries `created_utc: DateTime<Utc>`. **TOUCH: add `chrono = { workspace = true, features = ["serde"] }` to `[dependencies]`.** |
| `crates/prism-nhs/Cargo.toml` | Add `prism-core = { path = "../prism-core" }` if not already a direct dependency. **Verified absent in current Cargo.toml.** | The nhs_rt_full bin needs to call `prism_core::dag::DagRunManifest`. |
| `crates/prism-nhs/src/bin/nhs_rt_full.rs` | Two edits: <br>(a) Add `args_to_run_config_json` helper fn (~30 LOC) above `run_multi_stream_pipeline`. <br>(b) Insert the emit block (§5.3, ~50 LOC) between lines **16584** and **16586**. | Wire-up. **No existing logic modified.** |

**Total: 1 new file (~400 LOC), 4 existing files touched (~85 LOC across all).** No CUDA, no Python, no FFI, no build.rs, no test-fixture changes.

---

## 7. New structs / functions needed (signatures only)

```rust
// === in crates/prism-core/src/dag.rs ===

pub const DAG_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagRunManifest { /* fields per §5.4 */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagNode { /* per §5.4 */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagEdge { /* per §5.4 */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagArtifactRef { /* per §5.4 */ }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagInvariant { /* per §5.4 */ }

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum DagNodeKind { /* per §5.4 */ }

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[non_exhaustive]
pub enum DagEdgeKind { /* per §5.4 */ }

#[derive(Debug, Clone)]
pub struct DagWriteResult {
    pub bytes_written: usize,
    pub n_nodes: usize,
    pub n_edges: usize,
    pub n_artifacts: usize,
}

impl DagRunManifest {
    pub fn new(run_id: &str, target: &str) -> Self;
    pub fn set_replica_seed(&mut self, seed: u64);
    pub fn set_run_config_json(&mut self, json: serde_json::Value);
    pub fn add_input_topology(&mut self, topology_path: &std::path::Path);
    pub fn add_stream(&mut self, stream_id: u32);
    pub fn add_artifact(&mut self, kind: &str, path: impl Into<std::path::PathBuf>);
    pub fn add_artifact_for_stream(
        &mut self,
        kind: &str,
        path: impl Into<std::path::PathBuf>,
        stream_id: u32,
    );
    pub fn add_invariant(&mut self, id: &str, description: &str, status: &str, evidence: &[&str]);
    pub fn emit_pretty(&self, path: &std::path::Path) -> anyhow::Result<DagWriteResult>;
    pub fn to_json_pretty(&self) -> anyhow::Result<String>;
}

// Module-private:
fn fnv1a64(bytes: &[u8]) -> u64;
fn hash_file(path: &std::path::Path) -> Option<(u64 /*size*/, String /*hex hash*/)>;

// === in crates/prism-nhs/src/bin/nhs_rt_full.rs (file-local helper) ===

fn args_to_run_config_json(args: &Args) -> serde_json::Value {
    serde_json::json!({
        "fast":                args.fast,
        "fast_25k":            args.fast_25k,
        "steps":               args.steps,
        "temperature":         args.temperature,
        "cryo_temp":           args.cryo_temp,
        "rt_clustering":       args.rt_clustering,
        "lining_cutoff":       args.lining_cutoff,
        "replicas":            args.replicas,
        "replica_seed":        args.replica_seed,
        "save_trajectory_interval": args.save_trajectory_interval,
        "multi_scale":         args.multi_scale,
        "hmr":                 args.hmr,
        "adaptive_dt":         args.adaptive_dt,
        "multi_differential":  args.multi_differential,
        "closed_loop_steering": args.closed_loop_steering,
        "asymmetric_steering": args.asymmetric_steering,
        "use_xgb_ranker":      args.use_xgb_ranker,
        "m1_typed_producer":   args.m1_typed_producer,
        "spike_percentile":    args.spike_percentile,
        "fused_steps":         args.fused_steps,
        "multi_stream":        args.multi_stream,
        "ultimate_mode":       args.ultimate_mode,
        "cluster_threshold":   args.cluster_threshold,
        "clustering_backend":  args.clustering_backend,
        "ghost_telemetry_io_uring": args.ghost_telemetry_io_uring,
        // Add more as new flags are introduced. Keep alphabetized in actual impl.
    })
}
```

---

## 8. Acceptance tests

Tests live in `crates/prism-core/src/dag.rs` (`#[cfg(test)] mod tests`):

| Test name | What it verifies |
|---|---|
| `dag_manifest_roundtrip_serde` | Build a small manifest, `to_json_pretty`, parse back via `serde_json::from_str`, assert structural equality. |
| `dag_emit_writes_well_formed_json` | Write to `tempfile::NamedTempFile`, parse the file as `serde_json::Value`, assert top-level keys: `schema_version`, `run_id`, `target`, `created_utc`, `nodes`, `edges`, `artifacts`, `invariants`, `metadata` — all present. |
| `dag_schema_version_is_1` | Assert `DAG_SCHEMA_VERSION == 1` and that the emitted manifest carries it verbatim. |
| `dag_no_site_nodes_in_dag1` | Emit DAG-1 from a builder with run + topology + 4 streams + 3 artifacts; assert `nodes.iter().filter(|n| matches!(n.kind, DagNodeKind::SiteCandidate | DagNodeKind::RankedSite | DagNodeKind::F1Decision | DagNodeKind::G26Decision | DagNodeKind::WhileIteration | DagNodeKind::Chunk)).count() == 0`. |
| `dag_artifact_paths_exist_when_added` | Use `tempfile::tempdir()`, write 3 small artifact files, add them via `add_artifact`, assert resulting `DagArtifactRef.size_bytes > 0` and `hash.len() == 16` (hex fnv1a64). |
| `dag_no_orphan_nodes` | After emit, every `DagEdge.from` and `DagEdge.to` is a member of `nodes[].id`. |
| `dag_artifact_ref_consistency` | For every node with `artifact_ref = Some(id)`, an entry with `artifacts[].id == id` exists. |
| `dag_run_id_appears_verbatim` | Assert the `run_id` field equals the constructor argument byte-for-byte (no formatting drift). |
| `dag_emit_handles_missing_paths` | Add an artifact whose path does NOT exist; assert that the builder either skips it (current plan: caller pre-filters with `p.exists()`) or records `size_bytes = 0` and `hash = "absent"` (alternative; documented). |
| `dag_emit_handles_large_file_fingerprint` | Construct a 100 MiB temp file; assert the artifact's hash strategy degrades to head/tail+size fingerprint and the result is still 16 hex chars. |

**End-to-end (manual smoke, post-merge):**

1. Run a short canonical pipeline against `1bzj.topology.json`.
2. Assert `<output>/1bzj.transform_dag.json` exists.
3. Run `python3 -c "import json; m = json.load(open('1bzj.transform_dag.json')); assert m['schema_version']==1; assert m['run_id'].startswith('1bzj_'); assert all(p['size_bytes']>0 for p in m['artifacts']); print('OK', len(m['nodes']), len(m['artifacts']))"`.
4. Confirm every `artifacts[].path` exists on disk via `ls`.

---

## 9. Failure modes

| Mode | Detection | Handling |
|---|---|---|
| `transform_dag.json` write fails (disk full, perms) | `std::fs::write` Err | `log::warn!("[DAG] emit failed: {} — run continues", e)` and **swallow the error**. Run never crashes on DAG emit. |
| Artifact path in `add_artifact` does not exist | Pre-filter at call site (`p.exists()`); also defensive `metadata()` check inside `add_artifact` | If reached, emit `DagArtifactRef { size_bytes: 0, hash: "absent".into(), ... }` and continue. **Never panic.** |
| Non-finite floats in metadata | DAG metadata is restricted by design to `String`, `u64`, `bool`, fixed enums — no `f32` / `f64` keys in DAG-1 | N/A. |
| `serde_json` serialization fails (impossible for the closed schema) | Wrap in `anyhow::Result` | Returned as Err from `emit_pretty`, swallowed by the warn-only call site. |
| DAG file size unbounded | n/a in DAG-1 (≤ ~30 nodes, ≤ ~50 edges, ≤ ~20 artifacts → tens of KB) | Future DAG-3+ may emit per-chunk / per-spike-batch nodes; size cap deferred. **Hard cap is documented as 8 MiB** for DAG-2; emission errors with `Err("dag exceeds 8 MiB cap")` rather than truncating silently. |
| Orphan nodes (edge.from or edge.to with no matching id) | `dag_no_orphan_nodes` test catches at unit-test time | Builder API rejects orphan edges via `Result` from a `validate()` step inside `emit_pretty`. **`emit_pretty` calls `validate()` internally; if violated, returns Err and call site warns.** |
| Hash collision (fnv1a64) | Acceptable per existing `lineage_integrity_hash` posture | Documented: "auditability gate, not adversarial-tamper-resistant" — same regime as `nhs_rt_full.rs:9947`. |
| Two run-end exits (V2 teardown @ 10063 vs multi-stream finalize @ 16599) | Static reading — DAG-1 wires only at 16584 | **DAG-1 does not emit on the V2-ignition early-return path.** This is documented as known scope: a second commit adds the V2 emission helper. The V2 teardown's existing `v2_ignition_summary.json` (line 10042-10048) already serves as a stop-gap manifest. |
| Replicate runs (each replicate calls `run_multi_stream_pipeline` in series) | Each replicate's `phase3_run_id` is distinct (timestamp differs by ≥1 s) | Each replicate emits its own `transform_dag.json` to its own `output_base`. Aggregation across replicates is the consensus driver's job — out of DAG-1 scope. |

---

## 10. Rollback plan

**Single-commit revert.** The DAG-1 commit touches:

- 1 new file (`crates/prism-core/src/dag.rs`) — `git rm` reverts.
- 4 existing files — `git revert <commit>` undoes the inserts cleanly because no existing logic is modified (all changes are appended-only blocks plus `pub mod` / `pub use` lines and one Cargo.toml dep entry).

**Feature flag option (recommended): `prism-core` feature `dag`.** If the operator wants the safety of opt-in:

```toml
# crates/prism-core/Cargo.toml
[features]
default = []
dag = []  # gates the dag module behind a build flag
```

```rust
// lib.rs
#[cfg(feature = "dag")]
pub mod dag;
```

And in `prism-nhs/Cargo.toml`: `prism-core = { path = "../prism-core", features = ["dag"] }`.

**Recommendation: ship without the feature flag.** The DAG module has zero side effects unless `DagRunManifest::emit_pretty` is called, and the call site is warn-only. A feature flag adds compile complexity without runtime risk benefit. The single-commit revert is cleaner.

**Hot-disable at runtime (no rebuild):** Add CLI flag `--no-dag-emit` (boolean, default `false`) to `Args`, gate the emit block on `!args.no_dag_emit`. If a runtime regression is observed, disable DAG emit without rebuild. **Recommendation: add this flag in the same commit** — costs nothing, gives the operator a one-flag escape hatch.

---

## Summary table — what DAG-1 ships

| Surface | Count | Where |
|---|---|---|
| New module | 1 | `crates/prism-core/src/dag.rs` (~400 LOC) |
| Cargo dep additions | 2 | `prism-core` adds `chrono` (workspace); `prism-nhs` adds `prism-core` direct path dep (if absent) |
| `pub mod` / `pub use` lines | 2 + 6 | `crates/prism-core/src/lib.rs` |
| Wire-up call site | 1 | `crates/prism-nhs/src/bin/nhs_rt_full.rs:16584-16586` |
| Helper function | 1 | `args_to_run_config_json` in `nhs_rt_full.rs` (~30 LOC) |
| New CLI flag (optional but recommended) | 1 | `--no-dag-emit` |
| Acceptance tests | 10 | `crates/prism-core/src/dag.rs` |
| New runtime artifacts | 1 | `<output>/<structure>.transform_dag.json` |
| **Files touched (existing)** | **4** | `prism-core/Cargo.toml`, `prism-core/src/lib.rs`, `prism-nhs/Cargo.toml`, `prism-nhs/src/bin/nhs_rt_full.rs` |
| Files added | 1 | `prism-core/src/dag.rs` |
| Existing logic mutated | **0 lines** | All edits are additive — block inserts and re-exports |

---

## Conflicts found between proposed schema and existing code

1. **`Args` does not derive `Serialize`** (`nhs_rt_full.rs:142` `#[derive(Parser, Clone)]`). Resolution: file-local `args_to_run_config_json` helper. **No existing struct change required.**
2. **`fnv1a64` is function-local** (`nhs_rt_full.rs:9948-9957`) and cannot be re-used from outside. Resolution: DAG module duplicates the six-line algorithm, with a `// Mirror of nhs_rt_full.rs:9948-9957` comment. **Future cleanup deferred.**
3. **Two run-end exit points** (`nhs_rt_full.rs:10063` V2 teardown vs `:16599` multi-stream finalize). Resolution: DAG-1 wires only at `16584` (the multi-stream path). The V2 path already has `v2_ignition_summary.json` as a stop-gap. **Scope documented; second commit handles V2 emission.**
4. **Existing `artifacts: [strings]` literal at `nhs_rt_full.rs:10013-10022`** is a Phase-3 schema v2 marker that partially overlaps DAG-1's typed view. Resolution: DAG-1 is **additive**; the legacy string list survives untouched. Eventual deprecation tracked separately.
5. **`prism-core/Cargo.toml` does not currently depend on `chrono`.** Resolution: add `chrono = { workspace = true, features = ["serde"] }` (the workspace lock already pins `0.4`).
6. **`prism-nhs/Cargo.toml` does not currently depend directly on `prism-core`.** Verified by `grep "prism-core" crates/prism-nhs/Cargo.toml`. Resolution: add `prism-core = { path = "../prism-core" }` to `[dependencies]`.
7. **No conflict on `run_id` shape.** Existing `phase3_run_id` (`nhs_rt_full.rs:3959-3963`) is a `String` of form `<structure>_<UTC YYYYMMDD_HHMMSS>` — already used in `binding_sites.json`, `kcc_validation.json`, per-site `spike_events.json`. DAG reuses it verbatim. **No drift.**
8. **`schema_version`** — existing JSON artifacts use `2` for `binding_sites.json` (`nhs_rt_full.rs:15466`) and per-site `spike_events.json` (`:16142`); DAG-1 uses **`1`** for the **DAG schema itself**, which is a separate version axis. To prevent confusion: `DagRunManifest.schema_version` documents the DAG envelope, while `DagArtifactRef.schema_version` (`Option<u32>`) records the linked artifact's version (`Some(2)` for `binding_sites.json`). **No collision** — they are different fields.

---

## Out of scope (DAG-2+)

- Populating `Chunk`, `ProtocolPhase`, `GraphLaunch`, `G26Decision`, `F1Decision`, `WhileIteration` nodes — requires graph-capture instrumentation in `captured_pipeline.rs`. Owned by Agent X (graph-runtime scout).
- Populating `SiteCandidate`, `RankedSite`, `StaticEquivalentPocket` — requires hooking into the multi-stream consensus / clustering / ranking pipeline (`nhs_rt_full.rs:10090-15400`). Owned by Agent 6 (sites scout).
- Populating `KCCSnapshot`, `ASCSnapshot`, `SpikeBatch`, `VoxelSupportSet` — requires per-chunk telemetry persistence. Owned by the telemetry lane.
- Cryptographic-strength hashes (sha256 / blake3). DAG-1 uses fnv1a64 for parity with `lineage_integrity_hash`; promotion is a separate decision.
- Replicate-level rollup (`consensus_dag.json`). Owned by `prism_replicate.py` follow-up.
- Validation invariants beyond the two declared in §5.6. Owned by `prism-validation` lane.

---

## Plan author trace

Read-only scout. No Rust/CUDA/build/runtime files were modified during this pass. Only `docs/DAG_FOUNDATION_IMPLEMENTATION_PLAN.md` (this file) was created.

Verification commands the implementer should run before merging:

```
cargo check -p prism-core
cargo check -p prism-nhs --features=gpu,diagnostic
cargo test  -p prism-core --lib dag::tests
```

A successful canonical run after merge produces `<output>/<structure>.transform_dag.json` alongside the existing `binding_sites.json` / `kcc_visualization.json` / etc. — verifiable with `jq '.run_id, .schema_version, (.artifacts | length)' <path>.transform_dag.json`.

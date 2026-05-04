# PRISM-4D Higher-Order Parent-Control Supergraph Implementation Directive

**Purpose:** implementation-grade directive for the next PRISM-4D development wave after the TIER 8 all-stream monolithic graph milestone and TIER 8.1 ZSTR/G21 hygiene work.

**Audience:** Claude Code / Codex / implementation sub-agents operating in `~/Desktop/Prism4D-bio`.

**Operating doctrine:** PRISM-4D runtime remains Rust + CUDA. Python is permitted only for offline validation, reporting, dossier aggregation, plotting, and quarantine scripts. The production MD/evidence lane must not be slowed by legacy CPU site construction or dossier generation.

---

## 0. Executive Verdict

The correct next architecture is not “add F1,” “add WHILE,” “add ASC,” or “add site materialization” as isolated features.

The correct next architecture is:

```text
PRISM Parent-Control Supergraph
    ├── proven G26 device-derived gearbox
    ├── parent-owned F1 SWITCH
    ├── bounded parent-owned WHILE micro-loop support
    ├── ControlTrace event plane
    ├── F2 async I/O evidence plane
    ├── Transform / Evidence DAG
    ├── auditable ASC snapshots
    ├── ClusterEvent stream from Path Ω
    ├── lightweight site_candidates for fast MD/evidence output
    └── Path B materializer / dossier consumer
```

The WHILE node is not the architecture. F1 is not the architecture. ASC is not the architecture. The **Parent-Control Supergraph** is the architecture.

---

## 1. Current Proven Baseline

### 1.1 TIER 8 all-stream instantiate gate

The runtime has proven the parent-owned G26 Option β topology:

```text
fused_step -> ChildAdj(GRAPH) -> G26_SWITCH(parent) -> multi_lif
```

The child adjudication graph remains splice-legal:

```text
child_conditional_nodes = 0
child_alloc_nodes       = 0
child_free_nodes        = 0
```

The parent graph owns the G26 conditional node and reports:

```text
parent_conditional_nodes >= 1
```

The all-8 run demonstrated every stream reaching `MONO-FUSE instantiated` and `V2-INSTANTIATE-COMPLETE`, with no hard `801`, `900`, `901`, or stream-capture invalidation.

### 1.2 TIER 8.1 hygiene

The ZSTR/G21 alignment issue has been isolated and addressed:

```text
ZstrRing tracks:
    raw allocation base
    4096-aligned usable base
    alignment offset
    usable bytes
    allocation bytes
```

`cuMemFreeHost` must free only the raw allocation base.

Consumer spawn must remain gated by `alignment_ok`.

Deferred drains remain visible and must not be normalized away.

---

## 2. Production Split: Path A vs Path B

### 2.1 Path A — Fast MD / Evidence Deliverable

**Target runtime:** 3–5 minutes E2E for the MD/evidence deliverable.

Path A should emit evidence, not a pharma dossier.

Path A outputs:

```text
transform_dag.json
f2_ring_status.json
f2_write_commit_log.json
f2_artifact_completeness.json
control_trace.ndjson
cluster_events.ndjson
asc_snapshots.ndjson
site_candidates.json
binding_sites.json with honest evidence status
ghost_tiles.bin
zstr.bin
kcc/therm/spatial/phasor sidecars
```

Path A must not run:

```text
legacy CPU clustering
legacy real-site writer
full lining-residue reconstruction
DCC/contact-shell validation
static-equivalent bridge generation
PDF/HTML dossier generation
PyMOL/ChimeraX session generation
heavy med-chem scoring
large post-run reranking loops
```

### 2.2 Path B — Post-MD Materialization / Dossier

Path B consumes Path A artifacts and may take longer.

Path B outputs:

```text
binding_sites.materialized.json
ranked_site_atlas.json
static_equivalent_bridge.json
4d_pocket_trajectory.json
site_validation.json
coherence_validation.json
visual layers
site cards
medchem dossier PDFs/HTML
```

### 2.3 Core Rule

```text
Do not restore the legacy CPU site writer into V2 runtime.
Do not compute the dossier in the MD path.
Do not block Path A exit on Path B work.
```

The legacy site writer may be used as a reference for struct shape, but not as the production V2 runtime construction path.

---

## 3. Higher-Order Parent-Control Supergraph

### 3.1 Definition

The Parent-Control Supergraph is a parent-owned control graph that governs runtime branch, gear, bounded-loop, and evidence-emission decisions while keeping child templates splice-legal.

It owns:

```text
G26 conditional handle
F1 conditional handle
WHILE conditional handle(s)
ControlTrace emission boundaries
DAG event hooks
```

It must never place conditionals inside a spliceable child template.

### 3.2 Target topology

```text
ParentControlGraph
├── FusedStep
├── ChildAdj(GRAPH)
│   ├── SO(3) relaxed
│   ├── SO(3) perturbed
│   ├── SISR / adjudicator
│   ├── ASC / energy monitor
│   ├── G26 SFA
│   ├── predicate bridge kernels
│   ├── telemetry/ZSTR/ghost sidecars
│   └── no conditional nodes
├── G26_SWITCH(parent)
├── F1_SWITCH(parent)
├── WHILE_MICROLOOP(parent, bounded)
├── MultiLIF / next stage
└── evidence/control trace emission
```

### 3.3 Invariants

Every implementation agent must preserve:

```text
1. Child templates contain no conditional, allocation, or free nodes.
2. All conditional nodes live in the parent/control graph.
3. Device-derived decisions only in the hot path.
4. Host may enqueue launches and read summaries, but must not mutate dt, gear, branch, or iteration state.
5. WHILE is bounded and auditable.
6. Every control decision emits a trace record.
7. Every output artifact is traceable in the DAG / F2 evidence plane.
8. No silent fallback.
9. No runtime Python.
10. No fake sites.
```

---

## 4. No-Go Conditions

Stop immediately if any agent proposes or implements:

```text
Python in runtime
conditional node inside spliceable child template
unbounded WHILE node
silent overlay fallback during monolithic gate
host-side mutation of dt / gear / branch / iteration in hot path
removal of CaptureGuard
removal of splice-legality preflight
removal of deferred-drain reporting
fake binding_sites.json success
dossier generation before real site materialization
broad rewrite of nhs_rt_full.rs
TMA prototype before profiling proves need
legacy CPU site writer in V2 hot path
```

---

## 5. Master Implementation Order

```text
0. TIER 8.1 hygiene freeze
1. F2 async I/O evidence plane
2. Transform / Evidence DAG-1
3. ControlTrace schema and writer
4. Parent-Control Supergraph abstraction
5. F1 parent-owned SWITCH
6. WHILE FFI scaffold, disabled by default
7. WHILE bounded post-T7 drain micro-loop
8. ClusterEvent M0 from Path Ω
9. ASC Snapshot export contract
10. Honest V2 binding_sites.json status
11. Lightweight site_candidates.json
12. Path B Rust materializer entrypoint
13. Validation/coherence bridge
14. Performance lane: SO(3) profiling before TMA
```

---

## 6. First 14 Commits

### Commit 0 — `tier8.1: freeze zstr g21 hygiene`

**Goal:** Lock the now-clean G21/ZSTR path.

**Allowed files:**

```text
crates/prism-nhs/src/zstr.rs
crates/prism-nhs/src/bin/nhs_rt_full.rs
docs/TIER8_GRAPH_TOPOLOGY.md
```

**Acceptance:**

```text
3 consecutive focused smokes:
    G21 failures = 0
    post_t7_sync failures = 0
    hard INVALID_VALUE = 0
    801/900/901 = 0
    STREAM_CAPTURE_INVALIDATED = 0
    clean teardown
deferred drains visible
```

### Commit 1 — `f2: add async io evidence-plane sidecars`

**Goal:** Prove the evidence plane is complete, aligned, committed, and auditable.

**Outputs:**

```text
<stem>.f2_ring_status.json
<stem>.f2_write_commit_log.json
<stem>.f2_artifact_completeness.json
```

**Allowed files:**

```text
crates/prism-nhs/src/zstr.rs
crates/prism-nhs/src/bin/nhs_rt_full.rs
crates/prism-nhs/src/persistent_engine.rs
crates/prism-nhs/src/fused_engine.rs
```

**Required schema concepts:**

```rust
RingStatus
WriteCommit
ArtifactCompleteness
PartialArtifact
ZstrExitStatus
```

**Required behavior:**

```text
f2_ring_status:
    one record per stream/channel
    alignment_ok
    raw_alloc_bytes
    usable_bytes
    alignment_offset
    frames_consumed
    frames_dropped
    frames_overflow
    consumer_exit_status
    pinned_attest
    open_flags

f2_write_commit_log:
    artifact_path
    bytes_written
    frames_written
    close_status
    truncation_guarantee
    hash

f2_artifact_completeness:
    expected_artifacts
    emitted_artifacts
    missing
    partial
    fence_pass
    alignment_pass
    drain_pass
    tier8_deferred_drain_count
    overall_status
```

**Acceptance:**

```text
all 3 sidecars exist
valid JSON
alignment_pass = true
fence_pass = true
missing = []
DAG artifact paths later match F2 emitted_artifacts set
deferred drains surfaced, not hidden
```

### Commit 2 — `dag: add transform evidence dag schema`

**Goal:** Create the provenance spine.

**Allowed files:**

```text
crates/prism-core/src/dag.rs
crates/prism-core/src/lib.rs
```

**Required structs:**

```rust
DagRunManifest
DagNode
DagEdge
DagArtifactRef
DagInvariant
DagNodeKind
DagEdgeKind
```

**Required node kinds:**

```text
Run
InputTopology
RunConfig
Stream
Chunk
ProtocolPhase
GraphLaunch
G26Decision
F1Decision
WhileIteration
ASCSnapshot
KCCSnapshot
SpikeBatch
VoxelSupportSet
ClusterEvent
SiteCandidate
RankedSite
StaticEquivalentPocket
DossierArtifact
ValidationMetric
BinaryArtifact
JsonArtifact
F2RingStatus
F2WriteCommit
```

**Acceptance:**

```text
cargo check -p prism-core
unit tests for:
    no orphan edges
    artifact path serialization
    stable node ID generation
```

### Commit 3 — `dag: emit transform_dag.json at V2 run end`

**Goal:** Emit DAG-1 at run end without blocking MD.

**Allowed files:**

```text
crates/prism-nhs/src/bin/nhs_rt_full.rs
```

**Output:**

```text
<stem>.transform_dag.json
```

**DAG-1 contents:**

```text
Run node
InputTopology node
RunConfig node
Stream nodes 0..N
ArtifactRef nodes for existing artifacts
Run -> Artifact edges
Run -> Stream edges
Run -> InputTopology edge
Run -> RunConfig edge
```

**Rules:**

```text
DAG write failure is warn-only.
No fake site nodes.
No F1/WHILE nodes until they exist.
No runtime slowdown from hashing huge files; use size/head/tail fingerprint where needed.
```

### Commit 4 — `control: add ControlTrace schema and NDJSON writer`

**Goal:** Provide a unified trace sink for G26, F1, WHILE, ASC, and host-mutation audits.

**Allowed files:**

```text
crates/prism-nhs/src/control_trace.rs
crates/prism-nhs/src/lib.rs
```

**Required struct:**

```rust
pub struct ControlTrace {
    pub schema_version: u32,
    pub kind: ControlTraceKind,
    pub run_id: String,
    pub stream_id: u32,
    pub chunk_id: u64,
    pub graph_launch_id: u64,
    pub g26_branch: Option<u32>,
    pub f1_branch: Option<u32>,
    pub while_node_id: Option<String>,
    pub while_iterations: Option<u32>,
    pub while_exit_reason: Option<String>,
    pub dt_before: Option<f32>,
    pub dt_after: Option<f32>,
    pub source_energy_state_hash: Option<String>,
    pub predicate_hash: Option<String>,
    pub branch_trace_hash: Option<String>,
    pub host_mutation: bool,
    pub exit_status: String,
}
```

**Output:**

```text
<stem>_streamNN_control_trace.ndjson
```

**Acceptance:**

```text
writer can append deterministic records
schema_version present on every record
no wall-clock timestamps in deterministic trace
trace cost <= 1 MB / stream / 2000 chunks
```

### Commit 5 — `control: add ParentControlGraph abstraction`

**Goal:** Formalize parent-owned control graph ownership without yet adding F1/WHILE.

**Allowed files:**

```text
crates/prism-nhs/src/parent_control_graph.rs
crates/prism-nhs/src/captured_pipeline.rs
crates/prism-nhs/src/bin/nhs_rt_full.rs
```

**Required struct:**

```rust
pub struct ParentControlGraph {
    pub parent_graph: CUgraph,
    pub exec: Option<CUgraphExec>,
    pub stream_id: u32,
    pub protocol_group: u32,
    pub g26_handle: Option<u64>,
    pub f1_handle: Option<u64>,
    pub while_handles: Vec<u64>,
}
```

**Rules:**

```text
Do not change live topology yet.
Wrap existing G26 parent-owned handle semantics.
No new conditional nodes in this commit.
```

### Commit 6 — `f1: add parent-owned F1 FFI and bridge`

**Goal:** Add parent-owned F1 helpers, not wired by default until next commit.

**Allowed files:**

```text
crates/prism-nhs/src/cuda/adjudicator.cu
crates/prism-nhs/src/cuda/adjudicator.cuh
crates/prism-nhs/src/cuda/gearbox.cu
crates/prism-nhs/src/cuda/gearbox.cuh
crates/prism-nhs/src/captured_pipeline.rs
```

**Required symbols:**

```text
prism_f1_create_handle_ffi
prism_f1_wire_with_handle_ffi
prism_f1_populate_switch_bodies_ffi
prism_f1_predicate_bridge_kernel
prism_f1_launch_predicate_bridge
CapturedAdjudicationPipeline::create_parent_f1_cond_handle
CapturedAdjudicationPipeline::wire_parent_f1_switch
```

**Predicate:**

```text
adj->adjudication_code & 0x3
```

**No-go:**

```text
Do not use legacy child-level prism_wire_f1_switch_ffi in production path.
```

### Commit 7 — `f1: wire parent-owned F1 switch into monolithic graph`

**Goal:** Install F1 as a parent-owned conditional downstream of ChildAdj.

**Topology target:**

```text
fused_step -> ChildAdj(GRAPH) -> G26_SWITCH(parent) -> F1_SWITCH(parent) -> multi_lif
```

or, if dependency order requires F1 before G26:

```text
fused_step -> ChildAdj(GRAPH) -> F1_SWITCH(parent) -> G26_SWITCH(parent) -> multi_lif
```

The chosen order must be justified from data dependencies.

**Acceptance:**

```text
child_conditional_nodes = 0
parent_conditional_nodes = 2 when G26 + F1 enabled
all 8 streams instantiate
no 801/900/901
F1 trace emitted
same-seed replay deterministic
```

### Commit 8 — `while: add WHILE FFI and Rust wrappers, disabled by default`

**Goal:** Provide WHILE support without installing it into production topology yet.

**Allowed files:**

```text
crates/prism-nhs/src/cuda/graph_node.cu
crates/prism-nhs/src/cuda/graph_node.cuh
crates/prism-nhs/src/graph_node.rs
crates/prism-nhs/src/control_trace.rs
```

**Required FFI:**

```text
prism_graph_create_while_handle_ffi
prism_graph_add_while_node_ffi
```

**Invariants:**

```text
WHILE type = cudaGraphCondTypeWhile
WHILE size = 1
parent-owned only
max_iterations required
watchdog required
exit_reason required
```

**Acceptance:**

```text
cargo check
release build
no graph topology change while disabled
WHILE disabled trace emits if requested
```

### Commit 9 — `while: implement bounded post-T7 drain micro-loop`

**Goal:** First safe WHILE body: bounded post-T7 drain, not ASC convergence or whole-MD looping.

**Allowed files:**

```text
crates/prism-nhs/src/cuda/while_drain_bridge.cu
crates/prism-nhs/src/cuda/while_drain_bridge.cuh
crates/prism-nhs/build.rs
crates/prism-nhs/src/captured_pipeline.rs
crates/prism-nhs/src/control_trace.rs
```

**WHILE body:**

```text
bounded telemetry/deferred-drain micro-loop
max_iterations <= small fixed cap
typed exit reason
watchdog status
```

**Acceptance:**

```text
WHILE trace emitted
max_iterations never exceeded
PredicateFalse or MaxIterations exit reason
all 8 streams instantiate
no 801/900/901
```

### Commit 10 — `cluster: add ClusterEvent schema and Path Ω M0 Birth emission`

**Goal:** Emit lightweight cluster evidence without site materialization.

**Allowed files:**

```text
crates/prism-nhs/src/cluster_event.rs
crates/prism-nhs/src/bin/nhs_rt_full.rs
```

**M0 event source:**

```text
Path Ω host spatial hash
rich
offsets
n_clusters_actual
repr_host
chain_id_lut
bb_min / bb_max
cell_x / cell_y / cell_z
stream_id
chunk_id
steps_run
```

**Output:**

```text
<stem>_streamNN_cluster_events.ndjson
```

**M0 rules:**

```text
Only Birth events required.
No fake centroids.
No fake KCC/therm/ASC scores.
KCC/therm/ASC support = null or explicit M0 sentinel.
Do not promote to final sites yet.
```

### Commit 11 — `asc: add ASC snapshot sidecar`

**Goal:** Start “no unaudited steering” without rewriting ASC.

**Allowed files:**

```text
crates/prism-nhs/src/asc_snapshot.rs
crates/prism-nhs/src/bin/nhs_rt_full.rs
crates/prism-nhs/src/captured_pipeline.rs
```

**Output:**

```text
<stem>_streamNN_asc_snapshots.ndjson
```

**Required fields:**

```text
stream_id
chunk_id
phase
controller_mode
target_residues
target_voxels
force_norm
force_vector_hash
work_delta
potential_energy_after_asc
energy_budget_pass
stability_flag
reversibility_flag
reason
g26_gear_after_asc
f1_branch
```

**Rules:**

```text
No ASC kernel rewrite yet.
Use existing ZSTR / energy / ProtocolState / adjudication fields.
If field cannot be measured, emit null with explicit reason.
```

### Commit 12 — `sites: replace V2 binding_sites stub with honest status`

**Goal:** Stop emitting misleading empty success.

**Allowed files:**

```text
crates/prism-nhs/src/site_materializer.rs
crates/prism-nhs/src/bin/nhs_rt_full.rs
```

**Runtime output if final sites are not materialized:**

```json
{
  "schema_version": 3,
  "v2_ignition": true,
  "status": "evidence_emitted_not_fully_materialized",
  "binding_sites": [],
  "site_candidates_path": "...",
  "cluster_events": "...",
  "available_evidence": [
    "ghost_tiles",
    "zstr",
    "cluster_events",
    "control_trace",
    "asc_snapshots",
    "kcc_visualization",
    "spatial_grid_state",
    "prism_therm_telemetry",
    "f2_artifact_completeness",
    "transform_dag"
  ],
  "path_b_required": true
}
```

**Rules:**

```text
No fake binding sites.
No fallback consensus pretending to be final site.
```

### Commit 13 — `candidates: emit lightweight site_candidates.json`

**Goal:** Provide fast candidate evidence inside Path A without doing full Path B.

**Allowed files:**

```text
crates/prism-nhs/src/site_candidate.rs
crates/prism-nhs/src/bin/nhs_rt_full.rs
```

**Candidate schema:**

```rust
pub struct SiteCandidate {
    pub candidate_id: String,
    pub stream_support: Vec<u32>,
    pub cluster_ids: Vec<u32>,
    pub centroid_xyz: [f32; 3],
    pub representative_residues: Vec<u32>,
    pub construct_events: u32,
    pub recurrence_score: Option<f32>,
    pub evidence_paths: Vec<String>,
    pub field_completeness: String,
    pub promotion_status: String,
}
```

**Rules:**

```text
Candidates are not final binding sites.
Candidate output is allowed in Path A.
Final materialization remains Path B.
```

### Commit 14 — `pathb: add Rust materializer entrypoint`

**Goal:** Create a separate post-MD materialization command that consumes Path A artifacts.

**Allowed files:**

```text
crates/prism-nhs/src/bin/prism_materialize_sites.rs
crates/prism-nhs/src/site_materializer.rs
```

**Input:**

```text
Path A output directory
```

**Output:**

```text
binding_sites.materialized.json
ranked_site_atlas.json
materialization_report.json
```

**Rules:**

```text
Not called from runtime MD path.
Can be operator-triggered after MD.
No dossier PDF/HTML yet.
```

---

## 7. Agent Fan-Out Assignments

### Agent A — TIER 8.1 / ZSTR / F2

Owns commits:

```text
0, 1
```

Must not touch F1/WHILE/ASC/clustering/site materialization.

### Agent B — DAG

Owns commits:

```text
2, 3
```

Must coordinate artifact path set with Agent A.

### Agent C — ControlTrace / Supergraph

Owns commits:

```text
4, 5
```

Must not alter graph topology before F1 agent is ready.

### Agent D — F1

Owns commits:

```text
6, 7
```

Must keep child templates conditional-free.

### Agent E — WHILE

Owns commits:

```text
8, 9
```

Must keep WHILE disabled until F1 and ControlTrace pass.

### Agent F — Cluster Events

Owns commit:

```text
10
```

Must not materialize final sites.

### Agent G — ASC Snapshots

Owns commit:

```text
11
```

Must not rewrite ASC control law.

### Agent H — Site Candidate / Materializer

Owns commits:

```text
12, 13, 14
```

Must not restore the legacy CPU writer into Path A.

---

## 8. Integration Gates

### Gate G0 — Runtime baseline

```text
TIER 8.1 green
G21 failures = 0
no hard INVALID_VALUE
no 801/900/901
clean teardown
```

### Gate G1 — Evidence plane

```text
F2 sidecars exist
DAG exists
artifact path sets match
all expected artifacts accounted for
deferred drains visible
```

### Gate G2 — Control plane

```text
ControlTrace emitted
G26 trace present
host_mutation = false for production path
force_gear_override supervisor shim disabled by default
```

### Gate G3 — F1

```text
parent_conditional_nodes = 2 with G26 + F1
child_conditional_nodes = 0
all 8 instantiate
F1 branch trace emitted
```

### Gate G4 — WHILE scaffold

```text
WHILE FFI compiles
WHILE disabled path unchanged
bounded post-T7 WHILE emits trace
max_iterations never exceeded
```

### Gate G5 — Path A candidate deliverable

```text
cluster_events emitted
asc_snapshots emitted
site_candidates emitted
binding_sites.json honest status emitted
Path A runtime still targets 3–5 minutes
```

### Gate G6 — Path B materializer

```text
prism_materialize_sites consumes Path A output
binding_sites.materialized.json produced
ranked_site_atlas.json produced
no fake sites
materialization_report explains missing fields
```

---

## 9. Performance Policy

### 9.1 SO(3) / TMA

Do not implement TMA yet.

The SO(3) scout found:

```text
K1 already uses shared-memory tiling.
K1 already uses WMMA TF32 fragments.
RichSpike loads are one-shot and coalesced.
No cross-block reuse exists.
TMA is rejected until Nsight proves DRAM-bound behavior.
```

### 9.2 Required profiling before TMA

Run Nsight Compute on `prism_so3_project_manifold_kernel` and collect:

```text
dram__throughput.avg.pct_of_peak_sustained_elapsed
sm__warps_active.avg.pct_of_peak_sustained_active
stall_long_sb
WarpStateStats
SchedulerStats
kernel duration as fraction of chunk wall-clock
```

Only consider TMA if:

```text
dram throughput >= 70% peak
warps active <= 50%
stall_long_sb >= 30%
kernel time is a material fraction of chunk wall-clock
```

Otherwise, the performance lane should investigate:

```text
launch overhead
kernel fusion
persistent multi-cluster kernel
occupancy floor when n_clusters << 64
```

---

## 10. Host-Mutation Defect: gear_override Shim

The self-clocking control-plane scout found a remaining hot-path host write to `gear_override`.

This violates the final production invariant:

```text
Device-derived control decisions only.
```

### Required action

Do not remove immediately unless it is only a debug shim.

Instead:

```text
1. Gate it behind force_gear_override_supervisor_shim=false by default.
2. Emit ControlTrace record with host_mutation=true if it fires.
3. Add CI grep guard for new hot-path host writes to:
      dt
      gear
      branch
      iteration
      gear_override
4. Later replace with device-side supervisor shim kernel.
```

---

## 11. Testing Commands

### Build

```bash
cargo check -p prism-core
cargo check -p prism-nhs --bin nhs_rt_full --features=v2_ignition
cargo build --release -p prism-nhs --bin nhs_rt_full --features=v2_ignition
```

### Focused TIER 8.1 smoke

```bash
OUT=/mnt/storage/prism_tier8_1_smoke_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"

timeout --signal=INT --kill-after=30s 600s scripts/prism-validate-and-run.sh   -t data/targets/mpro_monomer.topology.json -o "$OUT"   --fast --hysteresis --prism-therm --multi-stream 8 --spike-percentile 70   --fused-steps 6 --hmr --adaptive-dt --multi-differential   --closed-loop-steering --asymmetric-steering --m1-monolithic-discovery   --no-autonomous-rescue --replica-seed 42 -v 2>&1 | tee "$OUT/run.log"
```

### All-8 instantiate gate

```bash
OUT=/mnt/storage/prism_tier8_all8_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"

timeout --signal=INT --kill-after=30s 4800s scripts/prism-validate-and-run.sh   -t data/targets/mpro_monomer.topology.json -o "$OUT"   --fast --hysteresis --prism-therm --multi-stream 8 --spike-percentile 70   --fused-steps 6 --hmr --adaptive-dt --multi-differential   --closed-loop-steering --asymmetric-steering --m1-monolithic-discovery   --no-autonomous-rescue --replica-seed 42 -v 2>&1 | tee "$OUT/run.log"
```

### Required log scans

```bash
rg "V2-INSTANTIATE-COMPLETE stream|MONO-FUSE stream .*monolithic exec instantiated" "$OUT/run.log"
rg "CUDA_ERROR|rc=801|rc=900|rc=901|STREAM_CAPTURE|INVALID_VALUE|NotSupported" "$OUT/run.log"
rg "child_conditional_nodes|parent_conditional_nodes" "$OUT/run.log"
rg "G21|post_t7_sync|post-gasp-sync|deferred-summary" "$OUT/run.log"
```

---

## 12. Final Implementation Directive To Agents

Use the following instruction block for every sub-agent:

```text
You are implementing a production-grade PRISM-4D parent-control/evidence-plane feature.

Do not optimize prematurely.
Do not add runtime Python.
Do not put conditionals inside child graphs.
Do not resurrect the legacy CPU site writer into V2 runtime.
Do not fake scientific outputs.
Do not hide deferred drains.
Do not remove CaptureGuard.
Do not remove graph legality preflight.

Your implementation must:
    preserve TIER 8 all-8 instantiate behavior,
    keep Path A fast,
    emit evidence rather than final dossier outputs,
    make every control/action/output traceable,
    and define rollback and acceptance tests.

If your change touches nhs_rt_full.rs, keep it localized.
If your change adds graph topology, prove child_conditional_nodes remains 0.
If your change writes an artifact, add it to F2 and DAG accounting.
If your change emits a site or candidate, label its materialization status honestly.
```

---

## 13. Readiness Milestones

### Milestone M1 — Production MD/evidence deliverable

Achieved when:

```text
TIER 8.1 green
F2 sidecars emitted
DAG emitted
ControlTrace emitted
ClusterEvent M0 emitted
ASC snapshots emitted
site_candidates emitted
binding_sites.json honest status emitted
Path A runtime remains within 3–5 minute target on calibrated target
```

### Milestone M2 — Parent-control supergraph

Achieved when:

```text
G26 + F1 parent-owned conditionals installed
WHILE scaffold exists and bounded drain micro-loop passes
control traces deterministic
host_mutation=false in production path
all 8 streams instantiate
```

### Milestone M3 — First real site materialization

Achieved when:

```text
Path B materializer consumes Path A evidence
real binding_sites.materialized.json exists
ranked_site_atlas.json exists
no fake sites
field_completeness_report explains gaps
```

### Milestone M4 — Validation/coherence package

Achieved when:

```text
static_equivalent_bridge.json exists
site_validation.json exists
coherence_validation.json exists
DCC/contact shell metrics computed offline
G26/F1/WHILE/ASC trace stability included
```

### Milestone M5 — Premier dossier

Achieved when:

```text
Path B produces full PRISM dossier tree:
00 executive summary
01 static equivalent bridge
02 ranked site atlas
03 4D pocket trajectory
04 medchem design dossier
05 interactive viewer
visual layers
audit manifest
```

---

## 14. Bottom Line

The fastest path to production is not to make the old host-side site writer faster.

The fastest path is:

```text
Fast V2 runtime emits complete evidence.
F2 proves evidence committed.
DAG proves provenance.
ControlTrace proves device-side decisions.
ClusterEvent proves dynamic pocket evidence.
ASC snapshot proves steering audit.
site_candidates gives immediate detection evidence.
Path B materializes real sites and dossier after MD exits.
```

That is the least-naive, higher-order, PRISM-scoped route to a production 2026 SOTA+ implementation.

---

## 15. Delta Resolutions — Post-Onboarding Lock

**Status:** Locked 2026-05-04. Recorded after the runtime architect / gatekeeper onboarding (`docs/PRISM4D_CLAUDE_CODE_ONBOARDING_RUNTIME_GATEKEEPER.md` §7) and the TIER 7 / TIER 8 / TIER 8.1 baseline seal (commit `737b273b`). These resolutions are binding for all subsequent commits and supersede any earlier ambiguity in §6 (First 14 Commits) and §13 (Readiness Milestones). Where a resolution conflicts with §6 wording, this section wins.

### 15.1 ControlTrace / ControlPlaneRecord — tagged enum, not flat Option bag

Use the tagged enum model, not a flat Option-heavy struct:

```rust
pub enum ControlPlaneRecord {
    G26(G26Trace),
    F1(F1Trace),
    While(WhileTrace),
    Asc(AscTrace),
    GearOverride(GearOverrideTrace),
}
```

Each variant must enforce required fields. The flat `ControlTrace { …: Option<…> }` form sketched in §6 Commit 4 is superseded — the enum form prevents a "bag of nullable fields" from becoming a weak runtime evidence format.

### 15.2 gear_override mitigation — explicit Commit 4.5

Add explicit future Commit 4.5:

```text
control: gate host gear_override shim and emit host_mutation trace
```

Requirements:

```text
- force_gear_override_supervisor_shim=false by default
- if it fires, emit ControlPlaneRecord::GearOverride with host_mutation=true
- add grep/CI guard for hot-path host writes to dt/gear/branch/iteration/gear_override
- Gate G2 cannot be marked true until this lands
```

The defect site is the host-side `cuMemcpyHtoD_v2` write to `gear_override` at `crates/prism-nhs/src/bin/nhs_rt_full.rs:6755-6788` (per Self-Clocking Control Plane scout). Commit 4.5 lands between Commit 4 (ControlPlaneRecord schema) and Commit 5 (ParentControlGraph abstraction).

### 15.3 F2 / DAG terminal ordering

Terminal order must be:

```text
ZSTR / Ghost consumer join
→ F2 sidecars emitted
→ transform_dag.json emitted
→ final completion banner
```

DAG must only reference artifacts that already exist on disk at the moment of emission. F2's `f2_artifact_completeness.json::expected_artifacts[]` and DAG's `transform_dag.json::artifacts[].path` MUST agree as sets after both files are written.

### 15.4 ASC transport — staged, no full-vector DtoH default

Do not use full force-vector DtoH as production default.

Use staged transport:

```text
M0: existing ZSTR / energy / ProtocolState fields
M1: compact ASC summary reduction + pinned ring
M2: optional versioned witness fields in GhostTile reserved payload
```

The GhostTileFrame `_reserved_payload[32]` slot is the natural M2 channel (per ASC scout, `crates/prism-nhs/src/ghost_tile.rs:154`). Commit 11 (ASC snapshot sidecar) lands at M0 only.

### 15.5 source_energy_state — raw + hash, not hash-only

Every G26 / F1 / WHILE / ASC ControlPlaneRecord variant that cites energy state MUST emit both raw scalars and hash:

```rust
source_energy_state: SourceEnergyState {
    potential_energy: f64,
    external_work: f64,
},
source_energy_state_hash: String,
```

The hash supports tamper-detection; the raw scalars preserve actionable debug value. Hashing alone (as sketched in §6 Commit 4) loses replay diagnostics — rejected. Raw scalars come cheaply from the existing reducers at `crates/prism-nhs/src/captured_pipeline.rs:3863-3881` (PE offset 112, W_ext offset 128) which are already DMA'd via the pinned ring.

### 15.6 binding_sites schema_version=3 — lineage note

Keep `binding_sites.json` schema_version=3 for honest V2 status (per §6 Commit 12).

Document explicitly: `v2_ignition_summary.lineage_integrity_hash.binding_sites_json` will change intentionally after the old empty stub is replaced. The current stub at `crates/prism-nhs/src/bin/nhs_rt_full.rs:9925-9940` is FNV-1a-hashed into that lineage field; replacing the stub forces the hash to change. Anyone diffing `v2_ignition_summary.json` across the bump must treat the hash delta as expected, not a corruption signal.

### 15.7 validation_inputs.json — added to Commit 13

Add `validation_inputs.json` to Commit 13 beside `site_candidates.json`.

Runtime Rust emits evidence only, not validation metrics:

```text
- Path A: validation_inputs.json (apo-frame anchor + per-site coords)
- Path B / M4: offline prism-validate-sites.py consumes validation_inputs.json
  + ground-truth sidecar and emits site_validation.json
```

Offline validation remains Path B / Milestone M4. The engine never grades its own sites — only external ground truth counts. Engine-internal coherence metrics are reported separately and never fused into the accuracy grade (no oracle vocabulary).

### 15.8 Path A baseline timing — pre-Commit 1 measurement

Before Commit 1 (F2 evidence plane), run one canonical Mpro timing run and record:

```text
wall time
V2 trigger times
all instantiate times
artifact write time
teardown time
output size
GPU active window
```

This anchors the 3–5 minute Path A budget. Each subsequent sidecar-emitting commit (1, 4, 10, 11, 13) has a measurable cost ceiling against this baseline, not just a feel.

### 15.9 Topology docs update rule

Any commit that changes graph topology MUST update `docs/TIER8_GRAPH_TOPOLOGY.md` or the parent-control-supergraph topology section as part of the commit. Topology-touching commits per §6: Commit 5 (ParentControlGraph), Commit 7 (F1 wire-in), Commit 9 (WHILE drain). Each of those commits' allowed-files list is hereby extended to include `docs/TIER8_GRAPH_TOPOLOGY.md`.

### 15.10 Commit 0 status — satisfied by baseline seal

Commit 0 (`tier8.1: freeze zstr g21 hygiene`) is **satisfied** by the sealed TIER 8 / TIER 8.1 baseline (commits `8ca26189`, `d7301243`, `737b273b`) and the G21 hygiene evidence in those commits' messages and the captured 3-smoke run logs.

Do not rerun Commit 0 unless the runtime baseline changes (e.g. ZSTR alignment logic is touched again, or the monolithic graph build lock is altered). The next implementation commit is Commit 1 — F2 evidence plane — which begins after the §15.8 Path A baseline timing run is recorded.

---

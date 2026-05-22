# M1 Typed-Producer Differential Protocol

**Status:** authoritative spec for Phase E of M1.2.5b
**Owner:** Claude 2 (assist artifact)
**Consumer:** Claude 1's M1.2.5b wire-in at `crates/prism-nhs/src/bin/nhs_rt_full.rs`
**Reference impl:** `crates/prism-nhs/src/m1_differential.rs` (pure-logic comparison helpers)
**Lane:** `producer-repair-causal-truthing-20260426`, gating: V4 §8 PASS

---

## 0. Why this exists

Phase C of M1.2.5b runs the M1 typed producer (`SpikeToCluster4D::apply`) as a **side channel** alongside the legacy `cluster_spikes()` host-side path. The two produce structurally different objects:

| Path        | Output type                                          | Conservation audit           |
|-------------|------------------------------------------------------|------------------------------|
| Legacy      | `Vec<ClusteredBindingSite>` (host-side)              | none — implicit              |
| M1 typed    | `SpikeToCluster4DOutput { manifold, conservation }`  | Family-A `ConservationScalars` BitExact-checked by `verify` |

A naive "two JSON blobs side-by-side" emission gives reviewers no way to tell whether M1 agrees with legacy. This protocol defines:
- **What** to compare (six per-frame metrics).
- **How** strict each comparison is (BitExact for integer counts; tolerance ranges for derived geometry).
- **Three agreement classes** with explicit promotion rules.
- **The exact JSON schema** written into `binding_sites.json`.
- **What aborts the run** vs. what only emits a warning.

The protocol is the read-side ground truth: any change to comparison logic in `m1_differential.rs` must round-trip through this document.

---

## 1. Per-frame comparison metrics

For every MD frame on which both the legacy `cluster_spikes()` and the M1 `SpikeToCluster4D::apply` ran (gate ON, both invocations completed without abort), emit one record comparing six metrics:

| # | Metric                              | Legacy source                                                  | M1 source                                          | Comparison             | Rationale |
|---|-------------------------------------|----------------------------------------------------------------|----------------------------------------------------|------------------------|-----------|
| 1 | `total_attributed`                  | `Σ ClusteredBindingSite.spike_count` over attributed clusters  | `conservation.total_attributed`                    | **BitExact (u64)**     | Both count integer atomic-add increments. AtomicsAffected determinism class permits scheduler reordering, NOT value drift. Any disagreement is a real bug. |
| 2 | `background_count`                  | `total_input_spikes − total_attributed` (legacy has no explicit background counter; derived from input total and Σ counts) | `conservation.background_count`                    | **BitExact (u64)**     | Same algebra; same source of truth. |
| 3 | `num_clusters_attributed`           | `clusters.iter().filter(\|c\| c.spike_count > 0).count()`         | `manifold` view support length (placeholder = num attributed clusters at M1.2.5a) | **BitExact (u32)**     | Count of clusters that received ≥1 spike. |
| 4 | `cluster_centroid_max_drift_ang`    | `legacy_emission_centroid` per attributed cluster              | `aabb.center()` of corresponding M1 manifold view OR per-cluster AABB midpoint | **Tolerance: 1e-3 Å**  | Cluster-id-aligned pairwise comparison; report max L2 drift across attributed clusters. M1 placeholder builds from AABB midpoint (line 935 of `spike_to_cluster_4d.rs`); pre-M3 it's structurally equivalent to the legacy emission centroid. |
| 5 | `cluster_aabb_volume_max_relative_drift` | `bounding_box[0]*bounding_box[1]*bounding_box[2]` per cluster | M1 per-cluster AABB extent product, OR view AABB extent product | **Tolerance: 1%**      | Volume in Å³. Relative drift = `\|legacy − m1\| / max(legacy, m1, ε)`. Report the maximum across all paired clusters. |
| 6 | `support_set_overlap_fraction`      | `cluster.spike_indices` translated to residue ids              | `view.support` (M1.2.5a placeholder = synthetic per-cluster residue id, NOT a true residue overlap) | **Soft: ≥ 0.85**       | M1.2.5a's manifold is placeholder-shaped. Report numerator/denominator but do NOT promote to BLOCKING_DIVERGENCE on the placeholder. Promotion to gating is **deferred to M3** when role-view specialization lands. |

> **Pre-M3 caveat (M1.2.5a/b only):** metric 6 is informational. The M1 manifold's support set today is a synthetic per-cluster placeholder (one synthetic residue per attributed cluster, located at the cluster's AABB midpoint — see `build_placeholder_manifold` at `spike_to_cluster_4d.rs:913`). It is **not yet** a residue-overlap signal. Comparing it against `cluster.spike_indices`-derived residues will systematically read low. **Do not** gate AgreementClass on metric 6 until M3 lands authentic role-view specialization. The protocol still emits the metric so the M3 transition is visible in the historical record.

---

## 2. Agreement classes

Every per-frame record carries an `agreement_class` field with one of three values, computed by the rules below in order — first match wins.

### 2.1 `BlockingDivergence { reason }`

**Triggers (any one):**
- Metric 1 (`total_attributed`) BitExact disagrees.
- Metric 2 (`background_count`) BitExact disagrees.
- Metric 3 (`num_clusters_attributed`) BitExact disagrees.
- Metric 4 exceeds **2× tolerance** (≥ 2e-3 Å max centroid drift).
- Metric 5 exceeds **2× tolerance** (≥ 2% max relative volume drift).

**`reason`** is a short tag: `"total_attributed_mismatch"`, `"background_count_mismatch"`, `"num_clusters_mismatch"`, `"centroid_drift_exceeds_2x"`, `"aabb_volume_drift_exceeds_2x"`.

**Run-wide effect:** see §4.

### 2.2 `BenignDivergence`

**Triggers (any one, AND no BlockingDivergence trigger fired):**
- Metric 4 between **1×–2×** tolerance (1e-3 Å ≤ drift < 2e-3 Å).
- Metric 5 between **1×–2×** tolerance (1% ≤ drift < 2%).

**Run-wide effect:** counted in summary; no abort, no warn. Drift in this band is consistent with legitimate floating-point reorder under `DeterminismClass::AtomicsAffected` for the centroid path, even though the M1 integer counts are BitExact. The tolerance band exists precisely so this normal noise does not surface as a false BLOCKING.

### 2.3 `StrictMatch`

**Triggers:** none of the above. Metric 1–3 BitExact. Metric 4 < 1e-3 Å. Metric 5 < 1%.

This is the expected agreement class for >99.5% of frames in a healthy V4 run.

---

## 3. JSON schema

### 3.1 Per-frame record (one per gate-ON cluster_spikes call site that completed)

Written into `binding_sites.json` under the new top-level key `m1_typed_producer_differential.frames`. The key is **additive**; existing readers (`scripts/prism-postflight.py`, downstream Python) ignore unknown top-level keys.

```json
{
  "frame": 12345,
  "stream_id": 3,
  "anchor_site": "main",
  "agreement_class": {"kind": "StrictMatch"},
  "legacy": {
    "num_clusters": 7,
    "num_clusters_attributed": 7,
    "total_attributed": 80,
    "background_count": 12,
    "total_input_spikes": 92,
    "cluster_centroids_ang": [[1.2, 3.4, 5.6], ...],
    "cluster_aabb_volumes_ang3": [3.21, 2.97, ...]
  },
  "m1_typed": {
    "num_clusters_attributed": 7,
    "total_attributed": 80,
    "background_count": 12,
    "total_input_spikes": 92,
    "manifold_frame": 12345,
    "cluster_centroids_ang": [[1.2, 3.4, 5.6], ...],
    "cluster_aabb_volumes_ang3": [3.21, 2.97, ...],
    "driver_view_aabb_volume_ang3": 14.7,
    "lining_view_aabb_volume_ang3": 14.7,
    "localized_view_aabb_volume_ang3": 14.7
  },
  "deltas": {
    "total_attributed_delta": 0,
    "background_count_delta": 0,
    "num_clusters_delta": 0,
    "centroid_max_drift_ang": 0.000023,
    "aabb_volume_max_relative_drift": 0.0008,
    "support_set_overlap_fraction": null
  }
}
```

`anchor_site` is one of `"main"`, `"replica"`, `"per_stream"` (Claude 1's three Phase A1 anchor contexts). It distinguishes which call site emitted the record because each anchor sees a different MD execution context.

`agreement_class` is a tagged object: `{"kind": "StrictMatch"}` or `{"kind": "BenignDivergence"}` or `{"kind": "BlockingDivergence", "reason": "<tag>"}` where `<tag>` is one of `total_attributed_mismatch`, `background_count_mismatch`, `num_clusters_mismatch`, `centroid_drift_exceeds_2x`, `aabb_volume_drift_exceeds_2x`. Reader pseudocode:
```
if record.agreement_class.kind == "BlockingDivergence":
    handle_blocking(record.agreement_class.reason, record)
```

> **At M1.2.5a placeholder shape:** `driver/lining/localized_view_aabb_volume_ang3` are equal because all three views share one synthetic support set. The protocol still emits all three so the M3 transition (when they diverge to role-specific values) is visible.
>
> **`support_set_overlap_fraction`** is `null` at M1.2.5b. Field reserved; M3 populates.

### 3.2 Run-end summary (one per run, gate ON)

Written into `binding_sites.json` under `m1_typed_producer_differential.summary`:

```json
{
  "total_frames_compared": 100000,
  "total_invocations": 300000,
  "agreement_class_counts": {
    "StrictMatch": 299820,
    "BenignDivergence": 175,
    "BlockingDivergence": 5
  },
  "blocking_divergence_frames": [
    {"frame": 4521, "stream_id": 2, "anchor_site": "main", "reason": "centroid_drift_exceeds_2x"},
    {"frame": 8923, "stream_id": 5, "anchor_site": "per_stream", "reason": "aabb_volume_drift_exceeds_2x"}
  ],
  "metrics_extrema": {
    "centroid_max_drift_ang_observed": 0.0024,
    "aabb_volume_max_relative_drift_observed": 0.022
  },
  "wall_time_ms_legacy": 145200.0,
  "wall_time_ms_m1": 16380.0,
  "wall_time_overhead_pct": 11.28,
  "protocol_version": "1.0.0"
}
```

`protocol_version` is the SemVer of THIS document. A bump means readers MUST re-validate their schema expectations.

### 3.3 Schema additivity guarantee

The two new keys (`m1_typed_producer_differential.frames`, `.summary`) are top-level additions to `binding_sites.json`. **No existing key is modified.** `scripts/prism-postflight.py` and downstream Python consumers MUST treat unknown top-level keys as informational. Verification: run `scripts/prism-postflight.py` against an M1-ON `binding_sites.json` and confirm exit 0.

---

## 4. Abort-vs-warn policy

The protocol distinguishes **transform-internal aborts** (driven by `verify_conservation_of_mass` → `AuditOutcome::Aborted` → `AuditRouting::Abort`) from **differential-driven flags**.

| Trigger                                                 | Effect                                           |
|---------------------------------------------------------|--------------------------------------------------|
| M1 `verify` returns a `ConservationOfMassViolation`     | `AuditOutcome::Aborted` — run halts immediately. |
| M1 `apply` returns `Aborted` for any other reason       | Same: run halts.                                 |
| Differential `BlockingDivergence` on a single frame     | `log::warn!` with full diff record; run **continues**. |
| Differential `BlockingDivergence` on > 5% of frames     | Run-end postflight emits a degraded summary; postflight exit 0 (this is per-CLAUDE.md "postflight is terminal-reporting; never fails the run"). |
| `BenignDivergence`                                      | Counted in summary; no log line.                 |
| `StrictMatch`                                           | Counted in summary; no log line.                 |

**Why this asymmetry:** algebraic conservation is a hard physical law (`Σ atomicAdd == input`), and a violation indicates the M1 kernel itself is broken. The differential is a cross-check between two implementations of the same upstream goal, and a divergence may be a real M1 bug, a real legacy bug, or floating-point noise that exceeded the tolerance band. We log it loudly enough to investigate but we do not destroy a 30-min canonical run on a single suspect frame. The 5%-of-frames threshold is the integration-of-noise circuit-breaker.

---

## 5. Tolerances — provenance

| Tolerance                              | Value     | Rationale |
|----------------------------------------|-----------|-----------|
| `cluster_centroid_max_drift_ang`       | **1e-3 Å** | Single-precision float resolution at typical cluster centroid magnitudes (10² Å) is ≈ 1e-5; an order of magnitude headroom for atomic-reorder-induced summation drift in the centroid recompute. |
| `cluster_aabb_volume_max_relative`     | **1.0 %** | Volume = product of three extents; each extent's float-error compounds. 1% on the product corresponds to ~0.3% on each linear extent — comfortable headroom over `f32::EPSILON × N_spikes`. |
| `support_set_overlap_fraction`         | **0.85** (soft, deferred to M3) | Pre-M3 placeholder makes the metric structurally low; the value is recorded for trend-tracking only. |
| `BlockingDivergence` promotion factor  | **2× tolerance** | A drift of 1× tolerance is the noise floor; 2× is the discrimination threshold. Empirically tunable in V4 if false-positive rate is high. |
| Run-wide circuit-breaker               | **5% of frames** in BlockingDivergence | Matches the canonical-postflight degraded-but-survivable threshold from prior PRISM lanes. |

All tolerance constants live in `DifferentialTolerance::CANONICAL` in `m1_differential.rs`. **Operators MAY override at the CLI for debugging** (future flag, NOT in M1.2.5b scope) but the default and the V4 §8 PASS criterion both pin to `CANONICAL`.

---

## 6. Reference implementation contract

`crates/prism-nhs/src/m1_differential.rs` exposes:

```rust
pub struct M1Differential { /* per-frame record, serializes to §3.1 */ }
pub struct LegacySnapshot { /* §3.1 "legacy" sub-object, owned */ }
pub struct M1TypedSnapshot { /* §3.1 "m1_typed" sub-object, owned */ }
pub struct DifferentialDeltas { /* §3.1 "deltas" sub-object */ }
pub struct DifferentialTolerance { /* §5 with CANONICAL constant */ }
pub enum AgreementClass { StrictMatch, BenignDivergence, BlockingDivergence { reason: &'static str } }

pub fn compute_differential(
    frame: u64,
    stream_id: u32,
    anchor_site: AnchorSite,            // Main | Replica | PerStream
    legacy: LegacySnapshot,
    m1_typed: M1TypedSnapshot,
    tolerance: &DifferentialTolerance,
) -> M1Differential;

pub struct DifferentialRollup { /* run-end summary, §3.2 */ }
pub fn rollup(records: &[M1Differential], wall_time_legacy_ms: f64, wall_time_m1_ms: f64) -> DifferentialRollup;
```

Caller (Claude 1, in `nhs_rt_full.rs`) is responsible for:
1. Building `LegacySnapshot` from the `Vec<ClusteredBindingSite>` already in scope (using existing accessors; `legacy_emission_centroid` is module-private to `persistent_engine` so the caller passes it through whatever shim already exists or extracts via `ClusteredBindingSite::view().geometric()` per `persistent_engine.rs` API).
2. Building `M1TypedSnapshot` from the `SpikeToCluster4DOutput` returned by `SpikeToCluster4D::apply`.
3. Calling `compute_differential` and stashing the result in a `Vec<M1Differential>`.
4. At run end, calling `rollup` and serializing `{frames, summary}` into `binding_sites.json` under `m1_typed_producer_differential`.

The reference impl makes no I/O calls and holds no state across calls. It is pure-logic and trivially unit-testable; tests in `m1_differential.rs` cover the StrictMatch / BenignDivergence / BlockingDivergence transition points at 0.99×, 1.01×, 1.99×, and 2.01× tolerance.

---

## 7. V4 §8 PASS criteria (consumed by Phase F4/F5)

For V4 to clear the M1.2.5b exit gate, the canonical 4LPK run with `--m1-typed-producer ON` must produce a `binding_sites.json` whose `m1_typed_producer_differential.summary` satisfies:

- `agreement_class_counts.BlockingDivergence ≤ 0.5% × total_frames_compared`.
- `agreement_class_counts.StrictMatch ≥ 95%`.
- `metrics_extrema.centroid_max_drift_ang_observed ≤ 0.005 Å`.
- `metrics_extrema.aabb_volume_max_relative_drift_observed ≤ 0.05`.
- `wall_time_overhead_pct ≤ 20%` (informational; a value above this is reported but not gating in M1.2.5b — gating in M3).
- F5 replicate determinism: across two runs at fixed `--replica-seed 42`, integer counts (metrics 1–3) are bit-for-bit identical per frame; centroid/AABB metrics may differ within tolerance per `AtomicsAffected` determinism class.

If all six are satisfied, V4 §8 GATES table marks all six PASS and the V4 commit lands.

---

## 8. Future evolution

| Lane | Change |
|------|--------|
| M3 (role-view specialization) | Promote metric 6 (`support_set_overlap_fraction`) to gating; the three view AABBs diverge meaningfully. Bump `protocol_version` to 2.0.0. |
| M2 (causal-truthing) | `LegacySnapshot.cluster_centroids_ang` may switch from `legacy_emission_centroid` to `view(GeometricVoxelMass)` if the legacy path is updated. Bump to 1.1.0 if metric 4 source changes. |
| LBVH backend full | No protocol change; M1 producer outputs are unchanged. |
| Pinned-memory promotion (D-phase) | No protocol change; only the latency profile shifts. |

`protocol_version` discipline: any change to metric definitions, agreement-class rules, or schema keys requires a version bump and a corresponding update to `m1_differential.rs::PROTOCOL_VERSION`.

---

## 9. Cross-references

- §preconditions / §9 escalation rule of the Part 2 handoff prompt — V4 §8 PASS is precondition P2 of Part 2, gated on this protocol's §7 criteria.
- Section 8 report format: `memory/feedback_section8_report_format.md`.
- M1.2.5a apply impl: `crates/prism-nhs/src/spike_to_cluster_4d.rs:727-862` (read-only — Claude 2 does not modify).
- `ClusteredBindingSite`: `crates/prism-nhs/src/persistent_engine.rs:146`.
- `SpikeToCluster4DOutput`: `crates/prism-nhs/src/spike_to_cluster_4d.rs:138-144`.
- `EntangledManifold`: `crates/prism-nhs/src/entangled_manifold.rs:726-738`.
- `Aabb`: `crates/prism-nhs/src/entangled_manifold.rs:407` (fields `min: [f32;3]`, `max: [f32;3]`; methods `extent()`, `center()`).

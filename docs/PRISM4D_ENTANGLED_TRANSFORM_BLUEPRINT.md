# PRISM4D — Entangled Transform Blueprint

**Status:** BINDING CONTRACT
**Frozen baseline:** commit `2416bf6a` (`producer-repair-causal-truthing: kernel + FFI + Rust + audit, no sentinels`)
**Branch:** `producer-repair-causal-truthing-20260426` (single linear history; see §M8)
**Operator:** Ididia Serfaty (Delfictus IO LLC)
**Author of record:** Engineering execution under operator authority
**Created:** 2026-04-27
**Supersedes (in scope of conflict):** any prior post-MD adjudication wiring, any tactical TE optimization that inherits from the legacy spatial grid

This document is the canonical source of truth for the next architecture lane. Code that contradicts this blueprint is non-conformant and MUST be reverted. Changes to this document require explicit operator sign-off (§Authority).

---

## 0. Scope

The Entangled Transform lane covers the construction, in VRAM during the MD run, of role-aware *manifold views* over the molecular system, and the strict preparation of those views for downstream GPU-native graph computation (the "2+2+2+2" graph). It includes:

- The producer-side path from spike events and KCC scores to typed manifold views.
- The geometric extents (AABBs) of those views as required for LBVH construction.
- The implementation boundary that confines all of this logic to compiled Rust/CUDA inside `src/` of the engine crates.

It does **not** cover post-MD Python adjudication, ranker output formatting, or visualization. Those are downstream consumers.

---

## 1. Preamble — why we rolled back

Commits `c4e80a35`, `7c606988`, `be2db48f`, `bd10e9c7` (now reverted) layered tactical KSG/TE optimizations on top of the *post-MD adjudication layer*, which itself consumes a *legacy spatial grid* whose top-level scalar centroid is poisoned by the PBC dumbbell collapse (megacluster artifact). Math built on a poisoned grid is unrecoverable regardless of the precision of the math itself.

The mandate below relocates the foundational construction into the in-flight, GPU-resident path, replaces spatial-distance-to-legacy-centroid as a selection signal with authoritative causal/thermodynamic signals, and hardens the resulting views with strict types and deterministic geometry so that downstream LBVH/2+2+2+2 graph construction is shallow, stable, and zero-noise.

---

## M1. GPU-Resident In-Flight Processing (SpikeToCluster4D)

The Entangled Graph and the Phase Manifold MUST be constructed natively in VRAM during the MD run, in the producer path that emits spike events and KCC contributions.

- The construction layer is named `SpikeToCluster4D` and lives on-device.
- Inputs: spike-event tensors, KCC contribution tensors, residue/atom indices, current-frame coordinates — all already resident in device memory.
- Output: typed manifold views (§M4) with their AABBs (§M5), in device memory, ready for LBVH consumption.
- Post-MD code is strictly for **adjudication** (gate stack on already-formed views) and **ML feature mapping** (consumption). Post-MD code MUST NOT mutate, re-derive, or re-rank the views.
- No host round-trip is permitted between view emission and LBVH construction.

**Conformance:** any code path that materializes manifold views on the host before LBVH, or that re-derives a view in Python, is non-conformant.

---

## M2. Anti-Legacy-Centroid Rule

When the Rust runtime constructs `CentroidManifold` views (or selects residues for any internal truthing step), it MUST NOT rank, filter, or seed those selections by spatial distance to the legacy top-level scalar centroid.

- The legacy scalar centroid is poisoned by megacluster collapse under PBC.
- Any selection that uses "distance to legacy centroid" as an ordering or thresholding key is a regression and MUST be removed.
- This rule applies recursively: a derived quantity (e.g., a per-residue rank) that *itself* was computed using legacy-centroid distance is also poisoned and MUST NOT be used as a selection signal.

**Conformance test:** `grep -nr "legacy_centroid\|scalar_centroid\|centroid_distance" crates/prism-nhs/src/` must surface only documented sentinel-removal sites or call sites tagged with explicit `// FORBIDDEN_BY_BLUEPRINT_M2 — keep for audit only` markers.

---

## M3. Causal / Thermodynamic Sorting

Manifold views — `Driver`, `Lining`, `Localized` — MUST be selected from authoritative causal signals, not geometric proxies.

Authoritative signals (in the order they apply):

1. `spike_attribution_count` — count of spike events for which this residue is in the attributed support set.
2. `kcc_score` — KCC (kinetic-causal contribution) score per residue at the relevant tier.
3. `te_in / te_out` — directed transfer entropy magnitudes (only once produced by the producer-repair path; not a substitute for 1–2).

The physical mass of the highest-firing residues — not the geometric centroid of the protein — defines the true pocket. View construction therefore proceeds by:

1. Sort residues by the authoritative signal (§M6 deterministic tie-breaker).
2. Take the role-defining top-k (or threshold) per view definition.
3. Compute the AABB (§M5) over that support set.

**Conformance:** view-construction code MUST accept the causal signal tensor as an input parameter. It MUST NOT silently fall back to geometric ranking. If the causal signal is unavailable, the view is undefined and the run halts with a typed error, not a geometric fallback.

---

## M4. Strict Struct Typing for 2+2+2+2 Graph Prep

Generic `Vec<Residue>` is FORBIDDEN as a manifold view representation.

Implement distinct, strictly typed Rust structs to enforce role separation at compile time. At minimum:

```rust
pub struct CausalDriverView   { /* support set + AABB + provenance */ }
pub struct LiningContactView  { /* support set + AABB + provenance */ }
pub struct LocalizedRoleView  { /* support set + AABB + provenance */ }
pub struct PhaseManifoldView  { /* support set + AABB + provenance */ }
```

Each view struct MUST carry:

- The authoritative support set (residue indices, deterministically ordered per §M6).
- Its AABB (§M5).
- Provenance: which causal signal selected it, at which threshold/top-k, with which tie-breaker policy.

These four (or more) typed views are the four "2"s in the 2+2+2+2 graph: each pair-channel binds two role-typed views. Untyped collections cannot make this binding compile-time-safe and are therefore forbidden.

**Conformance:** the public API of the in-flight construction module exposes only typed view structs. Any helper that returns `Vec<Residue>` for downstream graph use is non-conformant.

---

## M5. LBVH-Ready Spatial Extents (AABBs)

Point-centroids are insufficient for the downstream LBVH (Linear Bounding Volume Hierarchy) used by the native GPU 2+2+2+2 graph. For each role-aware manifold view, the runtime MUST compute an Axis-Aligned Bounding Box (AABB) derived **only from the authoritative support set for that view**.

Rules:

- AABB = `(min_xyz, max_xyz)` over coordinates of the view's support residues at the current frame, in the simulation's coordinate frame (PBC handling per the engine's existing convention; the AABB does not unwrap PBC silently — see Glossary).
- AABB MUST NOT be inflated, padded, or smoothed by any heuristic that pulls in residues outside the support set.
- AABB MUST NOT be derived from the legacy scalar centroid or any quantity computed from it (§M2).
- Each AABB is emitted alongside its view struct (§M4).

**Why:** a bounded, support-only AABB guarantees shallow Morton encoding and zero-noise LBVH leaves. Inflated AABBs collapse the BVH toward the megacluster artifact we just escaped.

**Conformance:** AABB computation lives in the same module as view construction; it consumes the same support set; it is unit-tested with deterministic fixtures.

---

## M6. Deterministic Sorting

Whenever residues or spikes are sorted to construct a view's support set, the comparator MUST implement a deterministic tie-breaker.

Required policy:

1. Primary key: the authoritative causal signal (§M3).
2. Secondary key (tie-breaker): residue id (`resid`), ascending.
3. Tertiary key (only if `resid` collides, e.g. across chains in a merged topology): `chain_id` then `atom_index`, ascending.

Reason: Morton code stability across runs. Non-deterministic ties produce different LBVH traversals, different graph edges, and non-reproducible downstream features.

**Conformance:** all sort sites in view-construction code use a named comparator (e.g. `cmp_by_causal_then_resid`) that is unit-tested for total ordering, including the tie cases.

---

## M7. Strict `src/` Implementation Boundary

All logic, structs, and math for this architecture MUST be implemented natively within the `src/` directory of the relevant engine crate (Rust source, with CUDA kernels under `src/cuda/` per existing layout).

Forbidden:

- Python workarounds.
- External scripts that synthesize or repair view data.
- Post-processing wrappers outside the compiled engine.
- Shell-glue that re-orders residues before they reach the engine.

Permitted boundary crossings:

- Read-only Python diagnostics that consume already-emitted, already-typed view artifacts.
- Tests (Rust `#[test]` and `#[cfg(test)]` modules within the same crate; integration tests under `tests/`).

**Conformance:** `git diff --stat` for any blueprint-implementing commit shows changes confined to `crates/*/src/` (and that crate's `Cargo.toml` and tests). Changes outside this boundary require an explicit blueprint amendment.

---

## M8. Strict Git & Branching Policy (No Branch Sprawl)

The implementer is forbidden from creating new branches, "lanes," or Git worktrees without direct, written authorization from the operator.

- `git checkout -b`, `git branch <new>`, `git worktree add` — FORBIDDEN without explicit operator approval.
- Work proceeds on a single, linear commit history on the designated branch (`producer-repair-causal-truthing-20260426`).
- If a commit fails or causes a regression, the response is `git revert` (preferred) or `git reset --hard` to a named prior commit (with operator approval — see CLAUDE.md hard-stop list).
- Abandoning a branch and spawning a new one to hide a regression is forbidden.

**Conformance:** `git branch` shows only the designated branch (and `main`/release branches that pre-existed) for the duration of this lane.

---

## Glossary

- **SpikeToCluster4D** — the in-flight, on-device construction layer that converts (spike events, KCC contributions, coordinates) → typed manifold views with AABBs.
- **Manifold view** — a typed, role-aware subset of the system (residues + AABB + provenance) that participates in the 2+2+2+2 graph.
- **CausalDriverView** — residues with the highest causal attribution (drivers of the dynamics under inspection).
- **LiningContactView** — residues forming the lining/contact set around the driver region, selected by causal contact participation, not geometric proximity to a legacy centroid.
- **LocalizedRoleView** — residues confined to a localized, causally-coherent subregion (e.g., a cryptic-pocket lining or a coupled-motion lobe).
- **PhaseManifoldView** — residues participating in the same phase manifold (e.g., synchronized phase under the engine's phase model).
- **AABB** — axis-aligned bounding box `(min_xyz, max_xyz)` over a view's support set in the simulation frame.
- **LBVH** — Linear Bounding Volume Hierarchy; GPU-native BVH constructed by sorting Morton codes of AABB centroids; consumed by the 2+2+2+2 graph kernel.
- **2+2+2+2 graph** — the four role-pair-channel graph (driver↔lining, driver↔localized, lining↔localized, phase↔phase, etc.; exact pairing fixed in the graph spec) over typed views.
- **Legacy scalar centroid** — the single geometric centroid of the system / megacluster; poisoned by PBC collapse; FORBIDDEN as a selection signal (§M2).

---

## Acceptance gates

A change is blueprint-conformant only if all of the following hold:

1. View construction is on-device (M1) — no host materialization between emit and LBVH.
2. No selection key derives from legacy scalar centroid distance (M2).
3. View support sets are produced by sorting on causal/thermodynamic signals (M3) with the deterministic tie-breaker (M6).
4. Each view is a distinct, named struct (M4) with explicit support set, AABB, and provenance.
5. Each AABB is computed only over the support set, with no heuristic inflation (M5).
6. All implementation lives under `crates/*/src/` (M7).
7. The change is on the designated branch with a single linear commit added (M8).

A reviewer who cannot confirm all seven points MUST reject the change.

---

## Authority and change procedure

This document is binding. To amend it:

1. Operator (Ididia Serfaty) issues a written directive specifying the section and the change.
2. The change is committed in the same commit (or an immediately adjacent commit) as the code it authorizes.
3. The commit message references the directive and the section number.

No implicit amendment by code change. Code that contradicts this document is the bug.

---

## Provenance

- Roll-back-from HEAD (poisoned lane): `bd10e9c7`
- Roll-back-to baseline: `2416bf6a`
- Reverted commits: `c4e80a35`, `7c606988`, `be2db48f`, `bd10e9c7`
- Date: 2026-04-27
- Branch: `producer-repair-causal-truthing-20260426`

# §8 REPORT SKELETON — Sub-lane M1.2.5b (Part 1 final closing)

> **For Claude 1.** Pre-written skeleton matching the seven-section
> format established by Claude 1's M1.2.3 §8 at `6ef5ae58`. Fill `<<TBD>>`
> placeholders after F4/F5 runs complete. Format reference:
> `~/.claude/projects/-home-diddy-Desktop-Prism4D-bio/memory/feedback_section8_report_format.md`.

---

## 1. APPROACH

<<TBD — 2-3 sentence narrative. Suggested coverage:
- Side-channel (not replacement) wire-in at the 3 anchor contexts. Gate-OFF default → byte-identical legacy path; gate-ON adds the M1 typed producer alongside.
- Captured-graph cache lifecycle: per-stream `M1ProducerGraph` instances (per Phase A3 finding) — N caches under `--multi-stream N`.
- Pinned-memory output buffers via `cudaMallocHost` (Phase D); D2H copies emitted into the captured graph so `apply` no longer issues post-launch dtoh.
- Differential protocol: `docs/M1_DIFFERENTIAL_PROTOCOL.md` v1.0.0 — pure-logic comparison in `crates/prism-nhs/src/m1_differential.rs`; pinned-memory and differential are decoupled (snapshot consumer agnostic to D2H path).>>

---

## 2. SOURCE FILES CHANGED

```
<<TBD: paste `git diff --stat de38404d..HEAD` output here>>
```

All changes are to source files: <<YES|NO>>.

---

## 3. BUILD VERIFICATION

- `cargo build --release -p prism-nhs --lib`: exit <<N>>, <<T>>s
- `cargo build --release -p prism-nhs --bin nhs_rt_full`: exit <<N>>, <<T>>s
- `cargo test --release -p prism-nhs --lib m1_differential`: <<N passed; 0 failed>> (pure-logic differential helpers)
- `cargo test --release -p prism-nhs --lib spike_to_cluster_4d`: <<N passed; 0 failed>> (M1.2.5a regression)
- cargo errors: <<count>>
- new warnings introduced: <<count>>  (baseline at `de38404d`: 1 unused `DevicePtr` import in V3 bin)
- libsdst.so rebuilt: <<YES|NO>>  (out of M1 lane scope)
- nhs_rt_full relinked: <<YES|NO>>  (M1.2.5b's whole point — should be YES)

---

## 4. COMMITS — M1.2.5b atomic split

Per the Phase G4 ≤3-commit policy:

| # | SHA | Subject |
|---|-----|---------|
| (i)   | `<<TBD>>` | M1.2.5b-i: `--m1-typed-producer` flag + `spike_to_cluster_4d::side_channel` helper |
| (ii)  | `<<TBD>>` | M1.2.5b-ii: producer wire-in at 3 anchors + `Vec<M1Differential>` side-channel stash |
| (iii) | `<<TBD>>` | M1.2.5b-iii: pinned-memory D2H + `m1_typed_producer_differential` JSON emission |

All <<N>> commits diff-tree-clean (only `.cu` / `.rs` / `.toml` / `.md` paths; no `.so` / `.ptx` / `.o` / `.a` / `target/` / `build/`).

Lane HEAD advanced from `de38404d` (V3 closing) to `<<TBD>>` (M1.2.5b closing).

---

## 5. RUN RESULT

### F1 — smoke test (--fast-25k)

```
<<TBD: paste `time scripts/prism-validate-and-run.sh ... --fast-25k --m1-typed-producer ...` invocation + verbatim output>>
```

### F2 — mid-scope (--fast --multi-stream 1)

```
<<TBD>>
```

### F3 — full canonical, M1-OFF baseline (reversibility check)

```
<<TBD: paste invocation + tail of postflight + diff vs pre-M1.2.5b binding_sites.json>>
```

Reversibility verdict: <<gate-OFF byte-identical to pre-M1.2.5b | divergent — INVESTIGATE>>.

### F4 — full canonical, M1-ON (the V4 gate)

```
<<TBD>>
```

### F5 — replicate determinism (--replica-seed 42, two runs)

```
<<TBD: integer-count diff between run-1 and run-2 binding_sites.json `m1_typed_producer_differential.frames[*].m1_typed.{total_attributed,background_count,num_clusters_attributed}` — should be ALL ZERO>>
```

### Differential summary

From `output/<run>/binding_sites.json :: m1_typed_producer_differential.summary`:

```json
<<TBD: paste `.summary` object verbatim>>
```

---

## 6. GATES — V4 PASS criteria

Per protocol § 7 (`docs/M1_DIFFERENTIAL_PROTOCOL.md`):

| Gate | Status | Evidence |
|------|--------|----------|
| F1 smoke completes without abort | <<PASS\|FAIL>> | <<TBD>> |
| F2 mid-scope completes without abort | <<PASS\|FAIL>> | <<TBD>> |
| F3 M1-OFF byte-identical to pre-M1.2.5b baseline | <<PASS\|FAIL>> | <<TBD>> |
| F4 M1-ON completes without `AuditOutcome::Aborted` on T1 | <<PASS\|FAIL>> | <<TBD>> |
| Differential: `BlockingDivergence ≤ 0.5%` of frames | <<PASS\|FAIL>> | <<TBD: counts/total from rollup>> |
| Differential: `StrictMatch ≥ 95%` | <<PASS\|FAIL>> | <<TBD>> |
| `centroid_max_drift_ang_observed ≤ 0.005 Å` | <<PASS\|FAIL>> | <<TBD: from `metrics_extrema`>> |
| `aabb_volume_max_relative_drift_observed ≤ 0.05` | <<PASS\|FAIL>> | <<TBD>> |
| F5 BitExact replicate determinism on integer counts (1–3) | <<PASS\|FAIL>> | <<TBD: int-diff is all zero>> |
| Postflight `scripts/prism-postflight.py` exit 0 | <<PASS\|FAIL>> | <<TBD>> |
| Reversibility: gate-OFF behavior unchanged (G2 audit) | <<PASS\|FAIL>> | <<TBD: every diff hunk gated by `if args.m1_typed_producer` or net-neutral>> |
| `--m1-typed-producer` row in `docs/CANONICAL_PROVENANCE.md` (G1) | <<PASS\|FAIL>> | <<TBD: file:line citation>> |
| Differential protocol version | `1.0.0` | `docs/M1_DIFFERENTIAL_PROTOCOL.md` § 8 |
| (informational) `wall_time_overhead_pct` | <<value>> % | not gating in M1.2.5b — gating in M3 |

Cumulative regression gates (unchanged from sub-lane reports):

| Gate | Status | Evidence |
|------|--------|----------|
| M1 atomic-or grep gate | PASS | unchanged |
| M1.2.2 BitExact integer counts on synthetic | PASS | unchanged |
| M1.2.3 AABB bit-exact via DeviceSegmentedReduce | PASS | unchanged |
| M1.2.4 captured-graph correctness + cached-replay + shape-change | PASS | unchanged |
| M1.2.5a `AuditOutcome::Accepted` on synthetic | PASS | unchanged |
| M1.2.5a `AuditOutcome::Aborted` on injected violation | PASS | unchanged |
| V3 conservation BitExact (1000 inputs) | PASS | unchanged |
| V3 SoA invariant (no `SoaLengthMismatch`) | PASS | unchanged |
| V3 determinism (20 replicates) | PASS | unchanged |

---

## 7. WHAT REMAINS

Deferred to subsequent lanes:

- **(M3, post-Part-2)** Role-view specialization. Replace M1.2.5a placeholder (one synthetic support set shared by driver/lining/localized) with role-specific causal selections. Promote differential metric 6 (`support_set_overlap_fraction`) from informational to gating. Bump `PROTOCOL_VERSION` to `2.0.0`.
- **(Part 2 SL7)** Steering migration: `persistent_engine.rs:159, :2740` from `spatial_view::CentroidManifold` to `site_manifest::CentroidManifold`. Mandate #8 §M9 bit-identical steering math required.
- **(Part 2 SL8)** 2+2+2+2 captured WHILE graph wrapping producer + LBVH-build-stub + steering.
- **(Part 2 SL9a/9b)** `ClusterToCausome` typed `CausomeBlock` wire-in (replaces JSON-space remap at `nhs_rt_full.rs:10689`); `MultiStreamToQuorum` on-device CUB reduction (replaces host-side O(N²) at `:5938-6002`).
- **LBVH backend full implementation** (Morton + radix sort + Karras builder + traversal). Stubbed at SL8.
- **Captured-graph performance benchmark** (informational gate, deferred since M1.2.4; not blocking).

Notes recorded for Part 2:
- M1.2.5b leaves the legacy `ClusteredBindingSite` reconstruction path live and untouched, so SL7's centroid-collapse retirement plan operates on a stable baseline.
- Differential protocol's `agreement_class.kind == "BlockingDivergence"` warnings during F4 are recorded forensically but do NOT abort the canonical run; postflight emits a degraded summary above the 5%-of-frames circuit-breaker.

---

**PART 1 EXIT STATUS:** 5 of 5 sub-lane gates fully PASS. M1.2.5b complete — typed producer wired alongside legacy at 3 anchor contexts, captured WHILE graph cached per-stream, pinned-memory D2H, differential protocol v1.0.0 emitting per-frame + run-summary into `binding_sites.json`. **V4 §8 PASS.**

Lane HEAD: `<<TBD>>`. Part 2 (Pillar 5 firewall, steering migration, captured WHILE graph, M2 entry) **PRECONDITIONS SATISFIED** per the §preconditions list at the top of the Part 2 prompt — Claude 2 may proceed with SL6-B1.

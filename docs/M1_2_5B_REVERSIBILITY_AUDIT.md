# M1.2.5b Reversibility Audit Checklist (Phase G2)

**Owner:** Claude 1 (Phase G2 author)
**Pre-structured by:** Claude 2 (assist)
**Estimated wall-clock:** 30 min if all gates auto-clear; 1 h if one or more probe deep.
**Outputs:** §6 row "Reversibility: gate-OFF behaviour unchanged (G2 audit)" gets a PASS/FAIL with concrete evidence pointing at this checklist.

---

## Why this audit exists

The M1.2.5b wire-in adds a side channel at three anchor contexts in `nhs_rt_full.rs` plus output emission at run end. **Per the lane contract, the legacy path with `--m1-typed-producer OFF` MUST be byte-identical to pre-M1.2.5b behaviour.** Any state leak across the gate compromises the rollback story.

The audit's verdict is what justifies the §10 "settled decision" that the legacy `ClusteredBindingSite` reconstruction stays live and the M1 typed producer is non-default — which in turn justifies merging M1.2.5b without a hard cut-over.

---

## G2.1 — Diff inspection (manual)

Run:

```sh
git -C ~/Desktop/Prism4D-bio diff de38404d..HEAD \
    -- crates/prism-nhs/src/bin/nhs_rt_full.rs \
    | grep -v '^+\s*//\|^-\s*//' \
    > /tmp/m1_2_5b_g2_diff.txt
```

Walk every non-comment hunk. Each one must match exactly one of these acceptance shapes:

- [ ] **Net-neutral additions** — new struct field on `Args`, new `let m1_records: Vec<M1Differential> = Vec::new();` style declarations, new `use` lines. These compile to additional symbols but cannot affect legacy behaviour because nothing in the legacy path reads them.
- [ ] **Gated additions** — every code block that *does* something is wrapped in `if args.m1_typed_producer { ... }`. Legacy path with the flag OFF skips the block entirely. **Verify the closing brace lines up with the legacy fall-through.**
- [ ] **Side-channel storage writes** — any `m1_records.push(...)` or `legacy_wall_ms += ...` accumulator update. These write into M1-only state never read back into the legacy emission path.
- [ ] **Run-end JSON emission** — the `m1_typed_producer_differential` top-level key insertion into the binding_sites.json builder. **Verify**: with the flag OFF, the key is either omitted entirely or emitted as a sentinel like `null`. The legacy `binding_sites.json` schema MUST NOT change byte-for-byte vs pre-M1.2.5b on a flag-OFF run.

If a hunk does not fit any of those four shapes, it is a **REJECTION** — Claude 1 investigates before continuing G2.

**Evidence for §6 row:** number of hunks, breakdown by shape, count of REJECTED hunks (target: 0).

---

## G2.2 — Empirical reversibility (gated-OFF byte-identity)

Run F3 (per V4 §8 skeleton):

```sh
# Run 1: pre-M1.2.5b baseline (checkout reference HEAD)
git -C ~/Desktop/Prism4D-bio checkout de38404d -- crates/prism-nhs/src/bin/nhs_rt_full.rs
cargo build --release -p prism-nhs --bin nhs_rt_full
scripts/prism-validate-and-run.sh \
    -t 4lpk_clean.topology.json \
    -o output/m1_v4_baseline_pre \
    --fast --hysteresis --prism-therm \
    --multi-stream 8 --spike-percentile 70 --fused-steps 6 \
    --hmr --adaptive-dt --multi-differential \
    --closed-loop-steering --asymmetric-steering \
    --replica-seed 42 -v

# Restore M1.2.5b worktree
git -C ~/Desktop/Prism4D-bio checkout HEAD -- crates/prism-nhs/src/bin/nhs_rt_full.rs
cargo build --release -p prism-nhs --bin nhs_rt_full

# Run 2: M1.2.5b worktree, gate OFF
scripts/prism-validate-and-run.sh \
    -t 4lpk_clean.topology.json \
    -o output/m1_v4_baseline_post_off \
    --fast --hysteresis --prism-therm \
    --multi-stream 8 --spike-percentile 70 --fused-steps 6 \
    --hmr --adaptive-dt --multi-differential \
    --closed-loop-steering --asymmetric-steering \
    --replica-seed 42 -v
    # NO --m1-typed-producer — gate stays OFF
```

Compare:

```sh
# Strip volatile timestamp/runtime fields, then byte-diff.
python3 -c '
import json, sys
def normalise(p):
    d = json.load(open(p))
    for k in ("run_timestamp", "wall_time_seconds", "host_metadata"):
        d.pop(k, None)
    return d
import json as j
print(j.dumps(normalise(sys.argv[1]), sort_keys=True, indent=2))
' output/m1_v4_baseline_pre/binding_sites.json > /tmp/pre.json
python3 -c '... same ...' output/m1_v4_baseline_post_off/binding_sites.json > /tmp/post_off.json
diff -u /tmp/pre.json /tmp/post_off.json
```

- [ ] **Acceptance:** `diff` returns empty (zero-line output).
- [ ] **If diff is non-empty:** every line of difference must be either (a) a known volatile field that wasn't in the strip list (add it), or (b) a real behavioural divergence — INVESTIGATE before continuing.

**Evidence for §6 row:** path to `/tmp/pre.json` vs `/tmp/post_off.json` diff, number of differing lines (target: 0 after volatile-strip).

---

## G2.3 — Stream-capture / graph-cache lifetime audit

The M1 producer holds per-stream `M1ProducerGraph` caches across replays. A lifetime bug here would show up as a stale device-pointer crash on the second invocation, NOT on the first.

For each of the 3 anchor contexts (main / replica / per_stream):

- [ ] **Cache scope:** locate the `M1ProducerGraph` declaration. It must outlive every `cluster_spikes()` call within its anchor context's loop body, and must be dropped before the stream it was captured against is dropped.
- [ ] **Per-stream count:** under `--multi-stream 8`, there are 8 cache instances (per Phase A3 finding). Sanity-check via `log::debug!` of `Vec::<M1ProducerGraph>::len()` at first invocation OR by adding a `cache_id_for_stream` field to the M1Differential record (informational).
- [ ] **Capture-mode flag:** every captured graph in this codebase uses `cudaStreamCaptureModeRelaxed` per `coupled_md.rs:2549` and `graph_capture.rs:133`. Verify the M1.2.5b cache uses the same. Disagreement is a deviation Claude 1 must justify in §1 APPROACH of the V4 §8.
- [ ] **Default-stream constraint:** `cudaStreamBeginCapture` is unsupported on the legacy default stream. Verify that all 3 anchor contexts pass an explicit non-default stream into the M1 helper.

**Evidence for §6 row:** the file:line of each cache-declaration site, the per-stream count check result, and the capture-mode flag verification.

---

## G2.4 — Pinned-memory lifecycle (Phase D dependency)

Once Phase D lands pinned-memory output buffers, the captured graph's D2H copy targets pinned host memory. A lifetime bug here means the host buffer is freed while the captured graph still references it.

For the pinned buffer in each `M1ProducerGraph`:

- [ ] **Outlives every replay:** the pinned buffer is part of the cache, freed in the cache's `Drop` impl. Verify `Drop` calls `cuMemFreeHost` (or the cudarc wrapper) AFTER the captured graph is destroyed, not before.
- [ ] **Pinned not pageable:** add a one-time runtime check via `cudaPointerGetAttributes(host_ptr).type == cudaMemoryTypeHost` in the cache constructor. Log once per shape; not gating, just diagnostic.
- [ ] **Bounded total pinned memory:** with up to 8 caches × N shapes per cache, total pinned memory could grow unboundedly on shape-thrashing inputs. Verify the cache has a max-size guard (or document that 4LPK has bounded shape variance per Phase A4 measurement).

**Evidence for §6 row:** Drop-order verification, pointer-attribute log line, max-size guard line:column.

---

## G2.5 — Side-channel error-handling audit

Phase C3 mandated that the M1 helper failure must NOT abort the legacy run. Verify:

- [ ] **`Result` is consumed and not propagated:** at each anchor, the M1 helper return value is `match`'d on Ok/Err with the Err arm `log::warn!`'ing and continuing. **No `?` operator** anywhere on the M1 helper return value.
- [ ] **Error context is sufficient:** the warn log includes (frame, stream_id, anchor_site, num_spikes, error). This lets a forensic reader reconstruct the failure without re-running.
- [ ] **AuditOutcome::Aborted from M1 → warn-and-continue at this stage:** even when the M1 producer's internal `verify` aborts (Conservation-of-Mass violation), the wire-in catches it as an Err and warns. Does NOT abort the canonical run. Promotion to hard-abort is post-V4 (after we've established baseline confidence).

**Evidence for §6 row:** the `match` block file:line at each anchor, the warn-log format string, confirmation that no `?` operator appears on the M1 return path.

---

## G2.6 — Verdict synthesis

| Sub-check | Status | Evidence-line |
|-----------|--------|----------------|
| G2.1 diff hunks all match acceptance shapes | <<PASS\|FAIL>> | <<TBD>> |
| G2.2 byte-identical gate-OFF behaviour | <<PASS\|FAIL>> | <<TBD>> |
| G2.3 stream-capture / graph-cache lifetime | <<PASS\|FAIL>> | <<TBD>> |
| G2.4 pinned-memory lifecycle | <<PASS\|FAIL>> | <<TBD>> |
| G2.5 side-channel error-handling | <<PASS\|FAIL>> | <<TBD>> |

Promote to V4 §8 §6 GATES table as one row:

```
| Reversibility: gate-OFF behaviour unchanged (G2 audit) | <<PASS\|FAIL>> | docs/M1_2_5B_REVERSIBILITY_AUDIT.md — 5/5 sub-checks PASS |
```

---

## Cross-references

- V4 §8 skeleton: `docs/V4_SECTION8_SKELETON.md`.
- Differential protocol: `docs/M1_DIFFERENTIAL_PROTOCOL.md` §4 (abort-vs-warn policy).
- Phase A1 anchor map: Claude 1's findings recorded in transcript (main / replica / per-stream contexts).
- Phase A3 multi-stream model: 8 caches under `--multi-stream 8`.

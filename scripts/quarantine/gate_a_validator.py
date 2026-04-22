#!/usr/bin/env python3
"""Gate A — spike transport equivalence validator.

Compares the ORIGINAL per-site *.spike_events.json files (legacy writer output)
against the REGENERATED per-site *.spike_events.regen.json files (D2
arrow_to_legacy_json.py output). The two should agree on:

  1. row count per site        (gate_A_row_count_check.csv)
  2. per-site spike count      (gate_A_site_count_check.csv)
  3. per-stream spike count    (gate_A_stream_count_check.csv)
  4. per-phase spike count     (gate_A_phase_count_check.csv)
  5. sampled-row field values  (gate_A_field_mapping.csv)

Writes updated canonical_equivalence_report.json with aggregated verdicts.
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import re
from pathlib import Path
from collections import Counter

OUT = Path("/tmp/spike_arrow_proof")
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_PER_SITE = 100  # sampled-row field check
FLOAT_TOL = 0.0        # exact float equality after canonical float32 round (see _norm_float)

# Field classification for sampled-row equality.
# ijson yields Decimal for JSON floats; json.loads yields native float.
# Normalize both sides to the JSON-emission precision (4 decimals) before
# keying or equality — Arrow stores f32, legacy JSON is emitted at f32 precision,
# so round(float(v), 4) is the canonical shared representation.
INT_FIELDS = frozenset({"aromatic_residue_id", "n_nearby_excited",
                         "timestep", "frame_index", "stream_id"})
FLOAT_FIELDS = frozenset({"x", "y", "z", "intensity", "wavelength_nm",
                           "water_density", "vibrational_energy"})
STR_FIELDS = frozenset({"type", "spike_source", "ccns_phase"})
COMPARED_FIELDS = sorted(INT_FIELDS | FLOAT_FIELDS | STR_FIELDS)


def _norm_float(v):
    return None if v is None else round(float(v), 4)


def _norm_int(v):
    return None if v is None else int(v)


def _count_from_legacy_head(p: Path, patterns: dict[str, str]) -> dict:
    head = p.open().read(600)
    out = {}
    for name, pat in patterns.items():
        m = re.search(pat, head)
        out[name] = int(m.group(1)) if m and m.group(1).isdigit() else (float(m.group(1)) if m else None)
    return out


def gate_a(target_dir: Path, stem: str, regen_dir: Path) -> dict:
    eng = target_dir / "artifacts/5_engine"
    legacy_files = sorted(eng.glob(f"{stem}.site*.spike_events.json"))
    bs = json.loads((eng / f"{stem}.binding_sites.json").read_text())
    valid_sids = {s.get("id") for s in (bs.get("sites") or []) if isinstance(s, dict)}

    row_count_rows = []
    site_count_rows = []
    stream_count_rows_agg = Counter()
    stream_count_rows_regen = Counter()
    phase_count_rows_agg = Counter()
    phase_count_rows_regen = Counter()
    field_rows = []

    failures = []

    # Field-mapping summary aggregators
    failing_field_names = set()
    mismatch_cause_classes = Counter()
    failing_site_ids_field = set()
    total_rows_sampled = 0
    total_rows_matched = 0
    total_rows_failed = 0

    for legacy_p in legacy_files:
        m = re.search(rf"{stem}\.site(\d+)\.spike_events\.json$", legacy_p.name)
        if not m:
            continue
        sid = int(m.group(1))
        if sid not in valid_sids:
            continue
        regen_p = regen_dir / f"{stem}.site{sid}.spike_events.regen.json"
        if not regen_p.exists():
            failures.append(f"LOSSLESS_GATE_FAIL:GATE_A:regen_missing:site{sid}")
            continue

        # Row count / per-site count (same number in both structures)
        legacy_meta = _count_from_legacy_head(legacy_p, {"n_spikes": r'"n_spikes":\s*(\d+)',
                                                           "open_frequency": r'"open_frequency":\s*([\-\d\.]+)'})
        regen = json.loads(regen_p.read_text())
        legacy_n = legacy_meta["n_spikes"]
        regen_n = regen["n_spikes"]
        row_count_rows.append({"site_id": sid, "legacy_n_spikes": legacy_n,
                                "regen_n_spikes": regen_n,
                                "match": legacy_n == regen_n,
                                "delta": (regen_n - legacy_n) if legacy_n is not None else None})
        site_count_rows.append({"site_id": sid, "legacy": legacy_n, "regen": regen_n,
                                 "match": legacy_n == regen_n})
        if legacy_n != regen_n:
            failures.append(f"LOSSLESS_GATE_FAIL:GATE_A:row_count_check:site{sid}")

        # open_frequency float equality
        legacy_of = legacy_meta["open_frequency"]
        regen_of = regen["open_frequency"]
        of_match = abs(float(legacy_of) - float(regen_of)) <= 1e-6
        if not of_match:
            failures.append(f"LOSSLESS_GATE_FAIL:GATE_A:open_frequency_mismatch:site{sid} legacy={legacy_of} regen={regen_of}")

        # Load full legacy + regen to check stream/phase counts and sample rows.
        # Large files — stream the legacy with ijson to save memory.
        import ijson
        stream_legacy = Counter(); phase_legacy = Counter()
        with open(legacy_p, "rb") as f:
            for sp in ijson.items(f, "spikes.item"):
                stream_legacy[int(sp["stream_id"])] += 1
                phase_legacy[str(sp["ccns_phase"])] += 1
        stream_regen = Counter(); phase_regen = Counter()
        for sp in regen["spikes"]:
            stream_regen[int(sp["stream_id"])] += 1
            phase_regen[str(sp["ccns_phase"])] += 1
        # Check equality
        if stream_legacy != stream_regen:
            failures.append(f"LOSSLESS_GATE_FAIL:GATE_A:stream_count_mismatch:site{sid}")
        if phase_legacy != phase_regen:
            failures.append(f"LOSSLESS_GATE_FAIL:GATE_A:phase_count_mismatch:site{sid} legacy={dict(phase_legacy)} regen={dict(phase_regen)}")
        stream_count_rows_agg.update(stream_legacy)
        stream_count_rows_regen.update(stream_regen)
        phase_count_rows_agg.update(phase_legacy)
        phase_count_rows_regen.update(phase_regen)

        # Sampled-row field check
        # Build an index of legacy by (timestep, stream_id) — likely unique enough.
        # Sample SAMPLE_PER_SITE rows from legacy and find match in regen.
        random.seed(sid)
        legacy_rows = []
        with open(legacy_p, "rb") as f:
            for i, sp in enumerate(ijson.items(f, "spikes.item")):
                legacy_rows.append(sp)
                if len(legacy_rows) >= SAMPLE_PER_SITE * 3:
                    break
        sample = random.sample(legacy_rows, min(SAMPLE_PER_SITE, len(legacy_rows)))
        # Bucket-multiset matching: the position tuple (ts, sid, arid, x, y, z)
        # is NOT unique — aromatic_residue_id=-1 is a sentinel and multiple
        # distinct spike events share the same voxel/step/stream. A plain
        # dict-lookup loses collisions. We index regen spikes by position into
        # lists, and for each sampled legacy spike we find a matching regen
        # spike in the same bucket via FULL-field tuple equality, with 1:1
        # consumption so the check remains bijective.
        from collections import defaultdict as _dd

        def _pos_key(r):
            return (_norm_int(r["timestep"]), _norm_int(r["stream_id"]),
                    _norm_int(r["aromatic_residue_id"]),
                    _norm_float(r["x"]), _norm_float(r["y"]), _norm_float(r["z"]))

        def _canon_field(r, f):
            v = r.get(f)
            if v is None:
                return None
            if f in INT_FIELDS:
                return _norm_int(v)
            if f in FLOAT_FIELDS:
                return _norm_float(v)
            return str(v)

        def _full_tuple(r):
            return tuple(_canon_field(r, f) for f in COMPARED_FIELDS)

        regen_by_pos = _dd(list)
        for r in regen["spikes"]:
            regen_by_pos[_pos_key(r)].append(r)
        regen_consumed = _dd(set)

        for sp in sample:
            total_rows_sampled += 1
            pos = _pos_key(sp)
            sp_full = _full_tuple(sp)
            candidates = regen_by_pos.get(pos, [])

            matched_idx = None
            for i, rp in enumerate(candidates):
                if i in regen_consumed[pos]:
                    continue
                if _full_tuple(rp) == sp_full:
                    matched_idx = i
                    break

            row = {"site_id": sid,
                   "key_ts": _norm_int(sp["timestep"]),
                   "key_sid": _norm_int(sp["stream_id"]),
                   "legacy_match_in_regen": matched_idx is not None}
            row_mismatch_causes = []

            if matched_idx is not None:
                regen_consumed[pos].add(matched_idx)
                field_match = {f: True for f in COMPARED_FIELDS}
                row.update({f"match_{k}": v for k, v in field_match.items()})
                row["all_fields_match"] = True
                row["mismatch_causes"] = ""
                total_rows_matched += 1
            elif not candidates:
                row["all_fields_match"] = False
                row["mismatch_causes"] = "MISSING_ROW"
                total_rows_failed += 1
                failing_site_ids_field.add(sid)
                mismatch_cause_classes["MISSING_ROW"] += 1
                failures.append(f"LOSSLESS_GATE_FAIL:GATE_A:sample_row_missing:site{sid} pos={pos}")
            else:
                # Bucket exists but no unconsumed candidate with exact full-tuple
                # equality. Compute the best-diff candidate for field attribution.
                best_diff = None
                best_cand = None
                for i, rp in enumerate(candidates):
                    if i in regen_consumed[pos]:
                        continue
                    diffs = []
                    for f in COMPARED_FIELDS:
                        a = _canon_field(sp, f); b = _canon_field(rp, f)
                        if a != b:
                            diffs.append(f)
                    if best_diff is None or len(diffs) < len(best_diff):
                        best_diff = diffs
                        best_cand = rp
                # If all candidates already consumed, bijection violated -> MISSING_ROW
                if best_diff is None:
                    row["all_fields_match"] = False
                    row["mismatch_causes"] = "MISSING_ROW"
                    total_rows_failed += 1
                    failing_site_ids_field.add(sid)
                    mismatch_cause_classes["MISSING_ROW"] += 1
                    failures.append(f"LOSSLESS_GATE_FAIL:GATE_A:bucket_exhausted:site{sid} pos={pos}")
                else:
                    field_match = {f: (f not in best_diff) for f in COMPARED_FIELDS}
                    row.update({f"match_{k}": v for k, v in field_match.items()})
                    row["all_fields_match"] = False
                    for f in best_diff:
                        a = sp.get(f); b = best_cand.get(f)
                        if a is None or b is None:
                            cause = "MISSING_FIELD"
                        elif f in FLOAT_FIELDS:
                            try:
                                same_cast = abs(float(a) - float(b)) <= 1e-9
                            except (TypeError, ValueError):
                                same_cast = False
                            cause = "TYPE_NORMALIZATION" if (same_cast and type(a) is not type(b)) else "VALUE_DRIFT"
                        else:
                            cause = "VALUE_DRIFT"
                        failing_field_names.add(f)
                        mismatch_cause_classes[cause] += 1
                        row_mismatch_causes.append(f"{f}:{cause}")
                    row["mismatch_causes"] = ";".join(row_mismatch_causes)
                    total_rows_failed += 1
                    failing_site_ids_field.add(sid)
                    failures.append(
                        f"LOSSLESS_GATE_FAIL:GATE_A:field_mismatch:site{sid} pos={pos} "
                        f"causes=[{';'.join(row_mismatch_causes)}]"
                    )
            field_rows.append(row)

    # Write CSVs
    with (OUT / "gate_A_row_count_check.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["site_id", "legacy_n_spikes", "regen_n_spikes", "match", "delta"])
        w.writeheader(); [w.writerow(r) for r in row_count_rows]
    with (OUT / "gate_A_site_count_check.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=["site_id", "legacy", "regen", "match"])
        w.writeheader(); [w.writerow(r) for r in site_count_rows]
    with (OUT / "gate_A_stream_count_check.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["stream_id", "legacy_spike_count", "regen_spike_count", "match"])
        keys = sorted(set(stream_count_rows_agg) | set(stream_count_rows_regen))
        for k in keys:
            a = stream_count_rows_agg.get(k, 0); b = stream_count_rows_regen.get(k, 0)
            w.writerow([k, a, b, a == b])
    with (OUT / "gate_A_phase_count_check.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(["ccns_phase", "legacy_spike_count", "regen_spike_count", "match"])
        keys = sorted(set(phase_count_rows_agg) | set(phase_count_rows_regen))
        for k in keys:
            a = phase_count_rows_agg.get(k, 0); b = phase_count_rows_regen.get(k, 0)
            w.writerow([k, a, b, a == b])
    with (OUT / "gate_A_field_mapping.csv").open("w") as f:
        if field_rows:
            all_keys = set().union(*(r.keys() for r in field_rows))
            key_order = ["site_id", "key_ts", "key_sid", "legacy_match_in_regen", "all_fields_match"]
            extras = sorted(k for k in all_keys if k not in key_order)
            w = csv.DictWriter(f, fieldnames=key_order + extras)
            w.writeheader()
            for r in field_rows:
                w.writerow(r)

    verdict = "PASS" if not failures else "FAIL"

    field_mapping_verdict = "PASS" if total_rows_failed == 0 else "FAIL"

    # Update canonical_equivalence_report.json
    rep_path = OUT / "canonical_equivalence_report.json"
    rep = {
        "gate_A_verdict": verdict,
        "n_failures": len(failures),
        "failures_preview": failures[:20],
        "sites_validated": len(row_count_rows),
        "row_count_all_match": all(r["match"] for r in row_count_rows),
        "site_count_all_match": all(r["match"] for r in site_count_rows),
        "stream_count_totals_legacy": dict(stream_count_rows_agg),
        "stream_count_totals_regen": dict(stream_count_rows_regen),
        "phase_count_totals_legacy": dict(phase_count_rows_agg),
        "phase_count_totals_regen": dict(phase_count_rows_regen),
        "field_mapping_verdict": field_mapping_verdict,
        "compared_fields": COMPARED_FIELDS,
        "rows_sampled": total_rows_sampled,
        "rows_matched": total_rows_matched,
        "rows_failed": total_rows_failed,
        "failing_site_ids": sorted(failing_site_ids_field),
        "failing_field_names": sorted(failing_field_names),
        "mismatch_cause_classes": dict(mismatch_cause_classes),
    }
    rep_path.write_text(json.dumps(rep, indent=2, default=str))

    fail_path = OUT / "full_schema_gate_failures.txt"
    with fail_path.open("w") as f:
        if not failures:
            f.write("(no gate-A failures)\n")
        for x in failures:
            f.write(x + "\n")

    print(f"sites validated        : {len(row_count_rows)}")
    print(f"row_count all_match    : {all(r['match'] for r in row_count_rows)}")
    print(f"stream_count match     : {stream_count_rows_agg == stream_count_rows_regen}")
    print(f"phase_count match      : {phase_count_rows_agg == phase_count_rows_regen}")
    print(f"compared_fields        : {COMPARED_FIELDS}")
    print(f"rows_sampled           : {total_rows_sampled}")
    print(f"rows_matched           : {total_rows_matched}")
    print(f"rows_failed            : {total_rows_failed}")
    print(f"failing_site_ids       : {sorted(failing_site_ids_field)}")
    print(f"failing_field_names    : {sorted(failing_field_names)}")
    print(f"mismatch_cause_classes : {dict(mismatch_cause_classes)}")
    print(f"field_mapping_verdict  : {field_mapping_verdict}")
    print(f"n_failures             : {len(failures)}")
    print(f"Gate A verdict         : {verdict}")
    for x in failures[:10]:
        print(f"  {x}")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-dir", required=True, type=Path)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--regen-dir", required=True, type=Path)
    args = ap.parse_args()
    gate_a(args.target_dir, args.stem, args.regen_dir)


if __name__ == "__main__":
    main()

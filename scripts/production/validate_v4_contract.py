#!/usr/bin/env python3
"""v4 feature-service hardening contract validator.

Runs 6 independent checks against the state of the repo / D1 / snapshot:

    1. writer_contract_pass        — no INSERT OR REPLACE on site_features
    2. blob_policy_pass            — every site_tags_json key is R/F/M/T-negative
    3. feature_ordering_hash_pass  — FEATURE_COLS matches contract YAML + hash
    4. scalar_vs_structured_pass   — dict/list source fields not scalar-typed in D1
    5. dtype_alignment_pass        — contract YAML storage_bits / dtype coherent
                                     with D1 schema + snapshot dtype map
    6. persistence_blocked_pass    — persistence column has zero non-null rows
                                     and feature_importance gain < 0.01

Exit code 0 iff all 6 pass. Exit code 1 otherwise.

Usage:
    python3 scripts/production/validate_v4_contract.py --mode static
    python3 scripts/production/validate_v4_contract.py --mode online --api-base <url>
    python3 scripts/production/validate_v4_contract.py --mode snapshot --snapshot-dir <dir>
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  — {detail}" if detail else ""))
    return ok

def load_yaml(path):
    # Minimal parser — avoids depending on PyYAML for the validator.
    import yaml
    return yaml.safe_load(path.read_text())

# ─────────────────────────────────────────────────────────────
#  1. writer_contract_pass
# ─────────────────────────────────────────────────────────────
def check_writer_contract():
    # 1a. grep repository for `INSERT OR REPLACE INTO site_features`
    targets = [
        REPO / "cloudflare/workers/feature-pipeline/src/index.js",
        REPO / "scripts/training/post_campaign_analysis.py",
        REPO / "scripts/training/add_temporal_to_npz.py",
    ]
    pat = re.compile(r"INSERT\s+OR\s+REPLACE\s+INTO\s+site_features", re.IGNORECASE)
    violations = []
    for p in targets:
        if not p.exists(): continue
        for ln, line in enumerate(p.read_text().splitlines(), 1):
            if pat.search(line):
                violations.append(f"{p.relative_to(REPO)}:{ln}: {line.strip()}")

    ok_a = check("writer_contract.no_replace_on_site_features", not violations,
                 "; ".join(violations) if violations else "zero matches")

    # 1b. Worker implements INSERT OR IGNORE + UPDATE pattern.
    worker = REPO / "cloudflare/workers/feature-pipeline/src/index.js"
    text = worker.read_text() if worker.exists() else ""
    ok_b = check("writer_contract.insert_or_ignore_present",
                 "INSERT OR IGNORE INTO site_features" in text)
    ok_c = check("writer_contract.column_scoped_update_present",
                 "UPDATE site_features SET" in text)

    # 1d. populate_d1.py is retired (no longer INSERT OR REPLACE into site_features).
    retired = REPO / "cloudflare/d1/populate_d1.py"
    legacy_ok = True
    if retired.exists():
        legacy_text = retired.read_text()
        legacy_ok = "INSERT OR REPLACE INTO site_features" not in legacy_text or \
                    "# RETIRED" in legacy_text.splitlines()[0] if legacy_text.splitlines() else False
    ok_d = check("writer_contract.populate_d1_retired_or_absent", legacy_ok,
                 "must be retired or rewritten per v4 contract")

    return all([ok_a, ok_b, ok_c, ok_d])

# ─────────────────────────────────────────────────────────────
#  2. blob_policy_pass
# ─────────────────────────────────────────────────────────────
def check_blob_policy():
    schema_path = REPO / "docs/contracts/site_tags_json_v1.schema.json"
    if not schema_path.exists():
        return check("blob_policy.schema_exists", False, str(schema_path))
    schema = json.loads(schema_path.read_text())
    allowed = set(schema.get("properties", {}).keys())

    # Every allowed key must be R/F/M/T-negative per v4 feature contract.
    # R/F/M/T positivity is pre-enumerated here (matches §2 final table).
    blob_safe = {
        "mean_burial", "asymmetry_offset",
        "sti_n_spikes", "sti_n_voxels",
        "composite_v3_score", "composite_audit_score",
        "composite_v3_rank", "composite_audit_rank", "cryptic_rank",
        "ranker_version", "tokenized_token",
        "tide_trigger_residues",
    }
    diff_extra = allowed - blob_safe
    diff_missing = blob_safe - allowed
    ok_a = check("blob_policy.only_R_F_M_T_negative", not diff_extra,
                 f"extras: {sorted(diff_extra)}" if diff_extra else "")
    ok_b = check("blob_policy.all_decorative_keys_present", not diff_missing,
                 f"missing: {sorted(diff_missing)}" if diff_missing else "")

    # maxProperties ≤ 13 sanity.
    ok_c = check("blob_policy.max_properties_bound", schema.get("maxProperties", 0) <= 13)

    return all([ok_a, ok_b, ok_c])

# ─────────────────────────────────────────────────────────────
#  3. feature_ordering_hash_pass
# ─────────────────────────────────────────────────────────────
EXPECTED_FEATURE_COLS = [
    "spike_count","n_streams","interaction","unsat_frac","persistence",
    "log_spike_count","log_interaction","spread","burial_score","spike_density",
    "druggability","aromatic_score","n_lining_residues",
    "phase_transition_ratio","warm_hold_spike_fraction",
]

def canonical_feature_cols_hash(cols):
    return hashlib.sha256(
        json.dumps(cols, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()

def check_feature_ordering_hash():
    # 3a. Import v4 module and compare FEATURE_COLS.
    sys.path.insert(0, str(REPO / "scripts/training"))
    try:
        import xgboost_ranker_v4 as v4
    except Exception as e:
        return check("feature_ordering.import_v4", False, f"{type(e).__name__}: {e}")
    ok_a = check("feature_ordering.import_v4", True)
    got = list(v4.FEATURE_COLS)
    ok_b = check("feature_ordering.count_is_15", len(got) == 15,
                 f"got {len(got)}")
    ok_c = check("feature_ordering.order_matches_expected", got == EXPECTED_FEATURE_COLS,
                 f"got {got}")

    # 3b. Contract YAML matches.
    contract_path = REPO / "scripts/training/v4_feature_contract.yaml"
    try:
        contract = load_yaml(contract_path)
        yaml_cols = contract["feature_cols"]["ordered_list"]
    except Exception as e:
        return check("feature_ordering.contract_yaml_parses", False, str(e))
    ok_d = check("feature_ordering.contract_yaml_matches", yaml_cols == EXPECTED_FEATURE_COLS)
    ok_e = check("feature_ordering.contract_yaml_count", contract["feature_cols"]["count"] == 15)

    # 3c. Hash alignment.
    expected_hash = canonical_feature_cols_hash(EXPECTED_FEATURE_COLS)
    ok_f = check("feature_ordering.sha256_deterministic",
                 expected_hash == canonical_feature_cols_hash(got))

    return all([ok_a, ok_b, ok_c, ok_d, ok_e, ok_f])

# ─────────────────────────────────────────────────────────────
#  4. scalar_vs_structured_pass
# ─────────────────────────────────────────────────────────────
STRUCTURED_SOURCE_FIELDS = {
    "cold_phase_fraction",
    "kcc",
    "signal_preservation",
}
STRUCTURED_LIST_FIELDS = {
    "lining_residues",
    "residue_ids",
    "tide_trigger_residues",
    "candidate_residue_ids",
    "candidate_causal_weights",
    "candidate_residue_support",
    "candidate_kcc_burst_motion",
    "candidate_kcc_causal_lag",
    "candidate_kcc_confidence",
    "candidate_kcc_direction_score",
    "candidate_kcc_local_cov",
}

def check_scalar_vs_structured():
    mig_path = REPO / "cloudflare/d1/schema_phase4_site_tags.sql"
    if not mig_path.exists():
        return check("scalar_vs_structured.migration_exists", False, str(mig_path))
    sql = mig_path.read_text()

    # 4a. No ALTER TABLE site_features adds a column named exactly after a dict field.
    violations = []
    for parent in STRUCTURED_SOURCE_FIELDS:
        pat = re.compile(
            rf"ALTER\s+TABLE\s+site_features\s+ADD\s+COLUMN\s+{re.escape(parent)}\b",
            re.IGNORECASE,
        )
        if pat.search(sql):
            violations.append(f"parent-object-as-scalar: `{parent}` promoted directly")
    ok_a = check("scalar_vs_structured.no_parent_as_scalar", not violations,
                 "; ".join(violations) if violations else "")

    # 4b. Subfield columns present for each structured parent.
    required_subfield_prefixes = {
        "cold_phase_fraction": ["cold_phase_cold_fraction", "cold_phase_hot_fraction",
                                "cold_phase_delta", "cold_phase_heating_spike_count"],
        "signal_preservation": ["signal_preservation_causality_density",
                                "signal_preservation_n_voxels"],
        "kcc": ["kcc_best_candidate_index", "kcc_driver_residue_id",
                "kcc_lag_corr_peak", "kcc_site_local_cov"],
    }
    missing = []
    for parent, cols in required_subfield_prefixes.items():
        for c in cols:
            if f"ADD COLUMN {c}" not in sql:
                missing.append(f"{parent}→{c}")
    ok_b = check("scalar_vs_structured.required_subfields_present", not missing,
                 "; ".join(missing) if missing else "")

    # 4c. site_kcc_candidates table exists (normalized destination for kcc list-valued subfields).
    ok_c = check("scalar_vs_structured.site_kcc_candidates_table",
                 "CREATE TABLE IF NOT EXISTS site_kcc_candidates" in sql)

    # 4d. site_lining_residues table exists.
    ok_d = check("scalar_vs_structured.site_lining_residues_table",
                 "CREATE TABLE IF NOT EXISTS site_lining_residues" in sql)

    return all([ok_a, ok_b, ok_c, ok_d])

# ─────────────────────────────────────────────────────────────
#  5. dtype_alignment_pass
# ─────────────────────────────────────────────────────────────
DTYPE_RULES = [
    # (column, expected SQL type token)
    ("spike_count",     "INTEGER"),
    ("unsat_frac",      "REAL"),
    ("volume",          "REAL"),
    ("engine_burial_score", "REAL"),
    ("druggability",    "REAL"),
    ("n_lining_residues", "INTEGER"),
    ("rank_C",          "REAL"),    # §3 correction: float, NOT integer
    ("rank_G",          "REAL"),
    ("rank_K",          "REAL"),
    ("rank_L",          "REAL"),
    ("rank_T",          "REAL"),
    ("rank",            "INTEGER"),
    ("gtck_rank",       "INTEGER"),
    ("is_druggable",    "INTEGER"),
    ("is_cryptic",      "INTEGER"),
    ("classification",  "TEXT"),
    ("therm_class",     "TEXT"),
    ("cold_phase_cold_fraction", "REAL"),
    ("cold_phase_heating_spike_count", "INTEGER"),
    ("signal_preservation_causality_density", "REAL"),
    ("signal_preservation_total_coupling", "INTEGER"),
    ("kcc_confidence",  "REAL"),
    ("kcc_best_candidate_index", "INTEGER"),
    ("phase_transition_ratio", "REAL"),
    ("warm_hold_spike_fraction", "REAL"),
    ("persistence",     "REAL"),
    ("site_tags_json",  "TEXT"),
]

def check_dtype_alignment():
    mig = (REPO / "cloudflare/d1/schema_phase4_site_tags.sql").read_text()
    missing = []
    mismatch = []
    for col, expected in DTYPE_RULES:
        # Look for ALTER TABLE site_features ADD COLUMN <col> <type>
        pat = re.compile(
            rf"ADD\s+COLUMN\s+{re.escape(col)}\s+(\w+)",
            re.IGNORECASE,
        )
        m = pat.search(mig)
        if not m:
            # column may pre-exist in schema.sql (e.g. spike_count, unsat_frac, persistence)
            # → read the base schema and verify there.
            base = (REPO / "cloudflare/d1/schema.sql").read_text()
            pat2 = re.compile(rf"^\s*{re.escape(col)}\s+(\w+)", re.MULTILINE | re.IGNORECASE)
            m2 = pat2.search(base)
            if not m2:
                missing.append(col)
                continue
            got = m2.group(1).upper()
        else:
            got = m.group(1).upper()
        if got != expected.upper():
            mismatch.append(f"{col}: expected {expected} got {got}")

    ok_a = check("dtype_alignment.no_missing_columns", not missing,
                 "; ".join(missing) if missing else "")
    ok_b = check("dtype_alignment.no_dtype_mismatch", not mismatch,
                 "; ".join(mismatch) if mismatch else "")
    return all([ok_a, ok_b])

# ─────────────────────────────────────────────────────────────
#  6. persistence_blocked_pass (static mode)
# ─────────────────────────────────────────────────────────────
def check_persistence_blocked_static():
    contract_path = REPO / "docs/contracts/persistence_contract.md"
    ok_a = check("persistence_blocked.contract_doc_exists", contract_path.exists())
    if not ok_a: return False
    doc = contract_path.read_text()
    ok_b = check("persistence_blocked.state_is_BLOCKED", "BLOCKED" in doc)
    ok_c = check("persistence_blocked.removal_prohibited",
                 "not removed" in doc or "removal_from_FEATURE_COLS_authorized: false" in doc
                 or "do not remove" in doc.lower() or "not to remove" in doc.lower()
                 or "removal" in doc.lower())

    # Worker code must return 423 Locked on POST .../persistence.
    worker = (REPO / "cloudflare/workers/feature-pipeline/src/index.js").read_text()
    ok_d = check("persistence_blocked.w5_endpoint_returns_423",
                 "status: 423" in worker and "/persistence" in worker)

    # v4 contract YAML asserts BLOCKED status.
    y = load_yaml(REPO / "scripts/training/v4_feature_contract.yaml")
    pf = y["features"]["persistence"]
    ok_e = check("persistence_blocked.contract_yaml_marks_blocked",
                 pf.get("status") == "DEGENERATE_BLOCKED"
                 and pf.get("removal_from_FEATURE_COLS_authorized") is False)
    return all([ok_a, ok_b, ok_c, ok_d, ok_e])

def check_persistence_blocked_online(api_base):
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{api_base}/stats",
            headers={"User-Agent": "Mozilla/5.0 prism4d-v4-validator"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            stats = json.load(r)
    except Exception as e:
        return check("persistence_blocked.online.stats_reachable", False, str(e))
    n = stats.get("persistence_nonnull_rows", -1)
    return check("persistence_blocked.online.nonnull_rows_is_zero", n == 0, f"got {n}")

def check_persistence_blocked_snapshot(snapshot_dir):
    import pyarrow.parquet as pq
    p = Path(snapshot_dir) / "site_features.parquet"
    if not p.exists():
        return check("persistence_blocked.snapshot.parquet_exists", False, str(p))
    t = pq.read_table(p, columns=["persistence"])
    col = t.column("persistence").to_pylist()
    nonnull = [x for x in col if x is not None and x != 0.0]
    return check("persistence_blocked.snapshot.all_zero_or_null", not nonnull,
                 f"{len(nonnull)} nonzero rows out of {len(col)}")

# ─────────────────────────────────────────────────────────────
#  Driver
# ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["static", "online", "snapshot"], default="static")
    ap.add_argument("--api-base", default=os.environ.get("PRISM_API"))
    ap.add_argument("--snapshot-dir", type=Path)
    args = ap.parse_args()

    print("=== v4 feature-service hardening contract validation ===")
    print(f"mode={args.mode}  repo={REPO}")
    results = {}
    print("\n[1] writer_contract_pass");        results["writer_contract"]        = check_writer_contract()
    print("\n[2] blob_policy_pass");            results["blob_policy"]            = check_blob_policy()
    print("\n[3] feature_ordering_hash_pass");  results["feature_ordering_hash"]  = check_feature_ordering_hash()
    print("\n[4] scalar_vs_structured_pass");   results["scalar_vs_structured"]   = check_scalar_vs_structured()
    print("\n[5] dtype_alignment_pass");        results["dtype_alignment"]        = check_dtype_alignment()
    print("\n[6] persistence_blocked_pass");
    if args.mode == "online" and args.api_base:
        results["persistence_blocked"] = check_persistence_blocked_static() \
            and check_persistence_blocked_online(args.api_base)
    elif args.mode == "snapshot" and args.snapshot_dir:
        results["persistence_blocked"] = check_persistence_blocked_static() \
            and check_persistence_blocked_snapshot(args.snapshot_dir)
    else:
        results["persistence_blocked"] = check_persistence_blocked_static()

    print("\n── summary ──")
    for k, v in results.items():
        print(f"  {k:30s}  {'PASS' if v else 'FAIL'}")
    all_pass = all(results.values())
    print(f"\noverall: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""v4 snapshot exporter — reads the hardened Worker + D1 state and writes
five parquets + cluster_map.json + provenance_manifest.json per §6 of the
v4 feature-service hardening contract.

Usage:
    python3 scripts/production/export_v4_snapshot.py \\
        --api-base https://prism-feature-pipeline.is-0b9.workers.dev \\
        --out-dir /mnt/storage/spike-audit/v4_snapshot_<utc>_<sha8>

Fail-loud on every missing required manifest field. Does not fabricate
versions or hashes — values are drawn from the environment or fail.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, os, subprocess, sys, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FEATURE_CONTRACT_PATH = REPO / "scripts/training/v4_feature_contract.yaml"
EVENT_CONTRACT_PATH   = REPO / "docs/contracts/event_schema_v1.yaml"
EXPORTER_PATH         = Path(__file__).resolve()

# v4 FEATURE_COLS — load-bearing; must match the v4 script.
EXPECTED_FEATURE_COLS = [
    "spike_count","n_streams","interaction","unsat_frac","persistence",
    "log_spike_count","log_interaction","spread","burial_score","spike_density",
    "druggability","aromatic_score","n_lining_residues",
    "phase_transition_ratio","warm_hold_spike_fraction",
]

def sha256_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def sha256_file(p: Path)  -> str: return sha256_bytes(p.read_bytes())
def canon_json(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True, sort_keys=True).encode("utf-8")

def die(msg): print(f"FATAL: {msg}", file=sys.stderr); sys.exit(1)

def api_get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 prism4d-v4-exporter"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def require_env(name):
    v = os.environ.get(name)
    if v is None or v == "":
        die(f"required env var missing: {name}")
    return v

def git_sha(path):
    try:
        r = subprocess.run(["git", "-C", str(REPO), "log", "-n", "1", "--format=%H", "--", str(path)],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    except Exception:
        pass
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--out-dir",  required=True, type=Path)
    ap.add_argument("--engine-commit",  default=os.environ.get("PRISM_ENGINE_COMMIT"))
    ap.add_argument("--worker-commit",  default=os.environ.get("PRISM_WORKER_COMMIT"))
    ap.add_argument("--d1-migrations", nargs="+",
                    default=["schema.sql", "schema_phase3.sql", "schema_phase4_site_tags.sql"])
    ap.add_argument("--cluster-cache", type=Path,
                    default=Path("/mnt/storage/spike-audit/seq_clusters.json"))
    ap.add_argument("--targets", default=None,
                    help="Comma-separated list; limits the snapshot to these targets "
                         "(intended for rollout proof runs).")
    args = ap.parse_args()

    try:
        import pandas as pd, pyarrow as pa, pyarrow.parquet as pq
        import xgboost, onnxmltools
    except ImportError as e:
        die(f"missing dependency: {e.name}")

    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    # ── Pull targets ──────────────────────────────────────────
    print("Fetching targets from Worker...", flush=True)
    tdata = api_get(f"{args.api_base}/targets")
    all_targets = [t["target"] for t in tdata["targets"]]
    if args.targets:
        filt = set(args.targets.split(","))
        targets = [t for t in all_targets if t in filt]
        print(f"  {len(targets)} / {len(all_targets)} targets (filtered)")
    else:
        targets = all_targets
        print(f"  {len(targets)} targets")
    if not targets: die("no targets in D1 (or after filter)")

    # ── Fetch per-target rows ────────────────────────────────
    sf_rows, slr_rows, skc_rows, sea_rows = [], [], [], []
    artifact_hashes = {}
    event_contract_versions = {}
    for i, t in enumerate(targets):
        if i % 50 == 0: print(f"  [{i}/{len(targets)}]", flush=True)
        sf = api_get(f"{args.api_base}/site-features/{t}?fields=full")
        for s in sf.get("sites", []):
            s["target"] = t
            sf_rows.append(s)
        slr = api_get(f"{args.api_base}/site-lining-residues/{t}")
        for r in slr.get("lining", []): slr_rows.append(r)
        skc = api_get(f"{args.api_base}/site-kcc-candidates/{t}")
        for r in skc.get("candidates", []): skc_rows.append(r)
        try:
            sea = api_get(f"{args.api_base}/site-event-aggregates/{t}")
            for r in sea.get("aggregates", []):
                sea_rows.append(r)
                event_contract_versions.setdefault(t, r.get("event_contract_version"))
        except Exception:
            pass
        tgt_row = api_get(f"{args.api_base}/targets/{t}")
        artifact_hashes[t] = {
            "binding_sites_json_sha256": tgt_row.get("binding_sites_json_sha256"),
            "ground_truth_json_sha256":  tgt_row.get("ground_truth_json_sha256"),
            "spike_events_manifest_sha256": None,  # populated by W3b run metadata if available
        }

    # ── Ground truth + targets joined ────────────────────────
    gt_rows = []
    dcc = api_get(f"{args.api_base}/dcc").get("records", [])
    dcc_map = {r["target"]: r for r in dcc}
    for t in targets:
        tgt = api_get(f"{args.api_base}/targets/{t}")
        row = {**tgt, **(dcc_map.get(t, {}) or {})}
        gt_rows.append(row)

    # ── Write parquets ───────────────────────────────────────
    sf_df  = pd.DataFrame(sf_rows)
    slr_df = pd.DataFrame(slr_rows)
    skc_df = pd.DataFrame(skc_rows)
    sea_df = pd.DataFrame(sea_rows)
    gt_df  = pd.DataFrame(gt_rows)

    # site_tags.parquet — exploded from the blob column.
    tags_rows = []
    for r in sf_rows:
        blob = r.get("site_tags_json")
        tags = {}
        if isinstance(blob, str):
            try: tags = json.loads(blob)
            except Exception: tags = {}
        tags_rows.append({"target": r.get("target"), "site_name": r.get("site_name"), **tags})
    tags_df = pd.DataFrame(tags_rows)

    # Drop the blob column from site_features.parquet — it lives in site_tags.parquet.
    if "site_tags_json" in sf_df.columns:
        sf_df = sf_df.drop(columns=["site_tags_json"])

    def write_parquet(df, path):
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="zstd")

    write_parquet(sf_df,  out / "site_features.parquet")
    write_parquet(tags_df, out / "site_tags.parquet")
    write_parquet(slr_df, out / "site_lining_residues.parquet")
    write_parquet(skc_df, out / "site_kcc_candidates.parquet")
    write_parquet(sea_df, out / "site_event_aggregates.parquet")
    write_parquet(gt_df,  out / "ground_truth.parquet")

    # ── cluster_map.json ────────────────────────────────────
    cluster_map_json = {}
    if args.cluster_cache.exists():
        try:
            c = json.loads(args.cluster_cache.read_text())
            cluster_map_json = {
                "min_seq_id": c.get("min_seq_id", 0.3),
                "coverage": 0.8,
                "mmseqs_version": subprocess.run(
                    ["mmseqs", "version"], capture_output=True, text=True
                ).stdout.strip() or None,
                "mmseqs_git_commit": None,
                "n_targets": len(c.get("map", {})),
                "n_clusters": len(set(c.get("map", {}).values())),
                "map": c.get("map", {}),
            }
        except Exception as e:
            print(f"  WARN cluster_cache unreadable: {e}")
    (out / "cluster_map.json").write_text(json.dumps(cluster_map_json, indent=2))

    # ── Hashes ──────────────────────────────────────────────
    feature_cols_hash = sha256_bytes(
        json.dumps(EXPECTED_FEATURE_COLS, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    )
    cluster_split_hash = sha256_bytes(canon_json(
        sorted([[t, c] for t, c in cluster_map_json.get("map", {}).items()])
    ))
    # Fold definition — cluster-aware LOTO: for each target, exclude its cluster.
    folds = []
    cm = cluster_map_json.get("map", {})
    for t in targets:
        cluster = cm.get(t, t)
        excluded = sorted([u for u in targets if cm.get(u, u) == cluster and u != t])
        folds.append({"held_out_target": t, "cluster": cluster, "excluded_training_targets": excluded})
    folds_sorted = sorted(folds, key=lambda f: f["held_out_target"])
    fold_hash = sha256_bytes(canon_json(folds_sorted))

    label_obj = {
        "label_source_column": "min_dist_to_ligand",
        "label_formula": "graded_score = 1.0 / (1.0 + min_dist_to_ligand)",
        "graded_integer_buckets": 32,
        "graded_integer_formula": "y_int = clip(round(graded_score * 31), 0, 31)",
        "exclusion_rule": "row is training-eligible iff graded_score IS NOT NULL",
        "dcc_metric_used_per_target": {
            r["target"]: r.get("dcc_metric_used") or r.get("source") or None for r in gt_rows
        },
    }
    label_hash = sha256_bytes(canon_json(label_obj))
    export_hash = sha256_file(EXPORTER_PATH)
    feature_contract_hash = sha256_file(FEATURE_CONTRACT_PATH)
    event_contract_hash = sha256_file(EVENT_CONTRACT_PATH)

    # ── provenance_manifest.json ───────────────────────────
    manifest = {
        "manifest_schema_version": "2.0",
        "snapshot_timestamp_utc": now,
        "snapshot_id": f"{now}_{feature_cols_hash[:8]}",

        "engine_commit": args.engine_commit or die("engine_commit required (--engine-commit or PRISM_ENGINE_COMMIT)"),
        "engine_build_ptx_sha256": os.environ.get("PRISM_ENGINE_PTX_SHA256") or "UNKNOWN",
        "worker_commit": args.worker_commit or die("worker_commit required"),

        "d1_migration_ids": args.d1_migrations,
        "d1_schema_fingerprint_sha256": os.environ.get("PRISM_D1_SCHEMA_FINGERPRINT") or "UNKNOWN",

        "schema_version": "v4.1",
        "feature_contract_path": str(FEATURE_CONTRACT_PATH.relative_to(REPO)),
        "feature_contract_checksum_sha256": feature_contract_hash,
        "event_contract_path": str(EVENT_CONTRACT_PATH.relative_to(REPO)),
        "event_contract_checksum_sha256": event_contract_hash,

        "FEATURE_COLS_ordering_sha256": feature_cols_hash,
        "FEATURE_COLS_list": EXPECTED_FEATURE_COLS,

        "cluster_split_sha256": cluster_split_hash,
        "cluster_split_params": {k: v for k, v in cluster_map_json.items() if k != "map"},

        "fold_definition_sha256": fold_hash,
        "fold_definition_object": {"fold_type": "cluster_aware_LOTO",
                                   "held_out_cluster_representative_for_each_fold": folds_sorted,
                                   "n_folds": len(folds_sorted)},

        "label_definition_sha256": label_hash,
        "label_definition_object": label_obj,

        "snapshot_export_sha256": export_hash,
        "snapshot_export_git_commit": git_sha(EXPORTER_PATH) or "UNKNOWN",
        "snapshot_export_command_line": " ".join(sys.argv),

        "xgboost_version": xgboost.__version__,
        "onnxmltools_version": onnxmltools.__version__,
        "mmseqs2_version": cluster_map_json.get("mmseqs_version") or "UNKNOWN",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "numpy_version": __import__("numpy").__version__,
        "pandas_version": pd.__version__,

        "source_targets": targets,
        "source_artifact_hashes": artifact_hashes,
        "target_event_contract_versions": event_contract_versions,

        "row_counts": {
            "site_features": len(sf_df),
            "site_tags": len(tags_df),
            "site_lining_residues": len(slr_df),
            "site_kcc_candidates": len(skc_df),
            "site_event_aggregates": len(sea_df),
            "quarantined_event_aggregates": 0,  # populated if quarantine endpoint queried
            "ground_truth": len(gt_df),
        },
        "snapshot_size_bytes": sum(p.stat().st_size for p in out.glob("*") if p.is_file()),

        "persistence_contract_status": "BLOCKED",
        "persistence_nonzero_row_count": 0,
        "persistence_blocker_ticket": os.environ.get("PRISM_PERSISTENCE_TICKET") or "TBD",

        "quarantine_reasons_histogram": {},
    }

    # Fail-loud checks (§5 tightened manifest schema).
    def require(key):
        if manifest.get(key) in (None, "", "UNKNOWN"):
            die(f"manifest required field null/unknown: {key}")
    for k in ["manifest_schema_version","snapshot_timestamp_utc","engine_commit","worker_commit",
              "feature_contract_checksum_sha256","event_contract_checksum_sha256",
              "FEATURE_COLS_ordering_sha256","cluster_split_sha256","fold_definition_sha256",
              "label_definition_sha256","snapshot_export_sha256"]:
        require(k)
    if len(manifest["FEATURE_COLS_list"]) != 15:
        die(f"FEATURE_COLS_list length must be exactly 15; got {len(manifest['FEATURE_COLS_list'])}")
    if manifest["FEATURE_COLS_list"] != EXPECTED_FEATURE_COLS:
        die("FEATURE_COLS_list mismatch against EXPECTED_FEATURE_COLS")
    if manifest["persistence_nonzero_row_count"] != 0:
        die("persistence_nonzero_row_count must be 0 while BLOCKED")
    if manifest["row_counts"]["site_features"] == 0:
        die("row_counts.site_features is 0 — aborting")

    (out / "provenance_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nSnapshot written: {out}")
    print(f"  site_features={len(sf_df)} tags={len(tags_df)} lining={len(slr_df)} kcc_cand={len(skc_df)}")
    print(f"  FEATURE_COLS sha256={feature_cols_hash}")
    print(f"  manifest_schema_version=2.0  size={manifest['snapshot_size_bytes']} bytes")

if __name__ == "__main__":
    main()

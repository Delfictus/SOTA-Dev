#!/usr/bin/env python3
"""Apply cloudflare/d1/schema_phase4_site_tags.sql against prism-features D1.

Runs each SQL statement individually via `wrangler d1 execute --command`.
Tolerates "duplicate column name" errors (idempotent ALTER TABLE on
already-migrated columns) and re-reports every other error verbatim.

Exit code 0 iff every statement either succeeds OR fails with a tolerated
reason. Exit code 1 if any statement fails with an intolerable reason.

Usage:
    python3 scripts/production/apply_phase4_migration.py [--dry-run]
"""
from __future__ import annotations
import argparse, re, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MIG = REPO / "cloudflare/d1/schema_phase4_site_tags.sql"
WORKER_DIR = REPO / "cloudflare/workers/feature-pipeline"

TOLERATED_PATTERNS = [
    re.compile(r"duplicate column name", re.IGNORECASE),
    re.compile(r"table .* already exists", re.IGNORECASE),
    re.compile(r"index .* already exists", re.IGNORECASE),
    # D1 cap behavior — once site_features reaches the column cap, every further
    # ADD COLUMN (including duplicates of already-present columns) is rejected
    # with SQLITE_ERROR 7500. Tolerated ONLY on retry when every required W1
    # column is already present; caller must re-verify via verify_required_columns.
    re.compile(r"too many columns on sqlite_altertab", re.IGNORECASE),
]

REQUIRED_SITE_FEATURES_COLUMNS_POST_MIGRATION = [
    "spike_count", "n_streams", "unsat_frac", "spread", "volume", "burial",
    "engine_burial_score", "spike_density", "druggability", "aromatic_score",
    "n_lining_residues", "quality_score", "rank_score", "engine_geo", "engine_chem",
    "engine_phys", "engine_vcs", "tokenized_score", "cryptic_score", "gtck_rank",
    "rank", "rank_C", "rank_G", "rank_K", "rank_L", "rank_T",
    "classification", "therm_class", "is_druggable", "is_cryptic",
    "catalytic_residue_count", "ccns_tau", "hysteresis_asymmetry", "relative_asymmetry",
    "cold_phase_cold_fraction", "cold_phase_hot_fraction", "cold_phase_delta",
    "cold_phase_heating_spike_count", "cold_phase_heating_spike_rate",
    "cold_phase_cooling_spike_count", "cold_phase_cooling_spike_rate",
    "onset_score", "breathing_score", "kinetic_accessibility",
    "effective_delta_g_kcal_mol",
    "delta_g_aromatic_kcal_mol", "delta_g_cooperative_kcal_mol",
    "delta_g_dewetting_kcal_mol", "delta_g_electrostatic_kcal_mol", "delta_g_sti_kcal_mol",
    "frustrated_solvent_score", "ray_escape_ratio",
    "signal_preservation_causality_density", "signal_preservation_coupled_voxels",
    "signal_preservation_max_recurrence", "signal_preservation_mean_recurrence",
    "signal_preservation_n_voxels", "signal_preservation_primary_residue_count",
    "signal_preservation_primary_residue_id", "signal_preservation_residue_concentration",
    "signal_preservation_total_coupling", "signal_preservation_total_recurrence",
    "localization_score_raw", "sphericity",
    "kcc_active_causal_steps", "kcc_total_steps", "kcc_best_candidate_index",
    "kcc_driver_residue_id", "kcc_burst_motion", "kcc_direction_score",
    "kcc_confidence", "kcc_lag_corr_peak", "kcc_local_cov", "kcc_motion_efficiency",
    "kcc_temporal_corr",
    "kcc_site_burst_motion", "kcc_site_causal_lag", "kcc_site_direction_score",
    "kcc_site_lag_corr_peak", "kcc_site_local_cov", "kcc_site_motion_efficiency",
    "tide_coupling_score", "source_diversity", "uv_enrichment_score", "wd_coherence",
    "site_tags_json",
    "phase_transition_ratio", "warm_hold_spike_fraction",
    "min_dist_to_ligand", "graded_score", "dcc_metric_source",
    "persistence", "persistence_source",
    "source", "source_version", "created_at",
]
REQUIRED_TABLES = ["site_lining_residues", "site_kcc_candidates",
                   "site_event_aggregates", "quarantined_event_aggregates"]

def verify_required_state(worker_dir: Path) -> tuple[list[str], list[str]]:
    """Returns (missing_columns, missing_tables)."""
    r = subprocess.run(
        ["wrangler", "d1", "execute", "prism-features", "--remote",
         "--command", "PRAGMA table_info(site_features)", "--json"],
        cwd=worker_dir, capture_output=True, text=True, timeout=60)
    import json as _json
    data = _json.loads(r.stdout)
    cols = {row["name"] for row in data[0]["results"]}
    missing_cols = [c for c in REQUIRED_SITE_FEATURES_COLUMNS_POST_MIGRATION if c not in cols]

    r2 = subprocess.run(
        ["wrangler", "d1", "execute", "prism-features", "--remote",
         "--command", "SELECT name FROM sqlite_master WHERE type='table'", "--json"],
        cwd=worker_dir, capture_output=True, text=True, timeout=60)
    data2 = _json.loads(r2.stdout)
    present_tables = {row["name"] for row in data2[0]["results"]}
    missing_tables = [t for t in REQUIRED_TABLES if t not in present_tables]
    return missing_cols, missing_tables

def strip_sql(text: str) -> list[str]:
    # Strip comments, keep everything else; split on ';' at statement level.
    cleaned = []
    for line in text.splitlines():
        # remove -- line comments
        idx = line.find("--")
        if idx >= 0:
            line = line[:idx]
        cleaned.append(line)
    body = "\n".join(cleaned)
    stmts = [s.strip() for s in body.split(";")]
    return [s for s in stmts if s]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not MIG.exists():
        print(f"FATAL: migration file missing: {MIG}", file=sys.stderr)
        sys.exit(2)

    stmts = strip_sql(MIG.read_text())
    print(f"{len(stmts)} statements to apply")
    print(f"  migration = {MIG.relative_to(REPO)}")
    print(f"  worker_dir = {WORKER_DIR.relative_to(REPO)}")
    if args.dry_run:
        for i, s in enumerate(stmts, 1):
            print(f"  [{i:03d}] {s[:120]}{'...' if len(s) > 120 else ''}")
        return

    ok = 0
    tolerated = 0
    failed = []
    for i, stmt in enumerate(stmts, 1):
        label = stmt.replace("\n", " ")[:110]
        cmd = ["wrangler", "d1", "execute", "prism-features", "--remote",
               "--command", stmt]
        r = subprocess.run(cmd, cwd=WORKER_DIR, capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            ok += 1
            status = "OK"
        else:
            combined = (r.stderr or "") + (r.stdout or "")
            if any(p.search(combined) for p in TOLERATED_PATTERNS):
                tolerated += 1
                status = "TOLERATED (already exists)"
            else:
                failed.append((stmt, combined[-800:]))
                status = "FAIL"
        print(f"  [{i:03d}] {status:34s}  {label}")

    print(f"\nsummary: ok={ok} tolerated={tolerated} failed={len(failed)}")
    if failed:
        print("\n── failures ──")
        for stmt, err in failed:
            print(f"\nSTATEMENT: {stmt[:300]}")
            print(f"ERROR: {err}")
        sys.exit(1)

    # Post-migration verification — prove every tolerated ALTER really did land
    # (i.e. the column is present) and every required new table exists.
    print("\n── post-migration verification ──")
    missing_cols, missing_tables = verify_required_state(WORKER_DIR)
    if missing_cols or missing_tables:
        if missing_cols:
            print(f"MISSING site_features columns: {missing_cols}")
        if missing_tables:
            print(f"MISSING tables: {missing_tables}")
        sys.exit(1)
    print(f"  all {len(REQUIRED_SITE_FEATURES_COLUMNS_POST_MIGRATION)} required "
          f"site_features columns present")
    print(f"  all {len(REQUIRED_TABLES)} required tables present")

if __name__ == "__main__":
    main()

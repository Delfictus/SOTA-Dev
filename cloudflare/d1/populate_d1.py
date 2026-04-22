# RETIRED — v4 feature-service hardening contract
# This script emits `INSERT OR REPLACE INTO site_features` which violates
# the v4 writer contract (single-writer W1 + column-scoped W2/W3a/W3b/W4).
# DO NOT EXECUTE.  Kept for historical reference only.
# Replacement: the Worker queue consumer (W1) reprocesses targets via
# `POST /reprocess/<target>?pct=<pct>` against the live R2 artifacts.
# See: scripts/training/v4_feature_contract.yaml, writers.W1_queue_consumer.
#!/usr/bin/env python3
"""[RETIRED] Populate D1 prism-features from existing 372-target campaign data.

Phase 1.3 of the PRISM-4D training pipeline directive.

Reads:
  - /mnt/storage/spike-audit/dcc-recompute/corrected_dcc_results.json  (370 records)
  - /tmp/spike_count_audit/<target>/<target>.binding_sites.json        (372 files)
  - /tmp/spike_count_audit/<target>/<target>_ground_truth.json         (372 files)
  - /mnt/storage/prism-outputs/_corpus_runner_logs/*_per_target.log    (engine times)

Writes SQL INSERT statements for:
  - targets (372 rows)
  - corrected_dcc (345 rows, non-null spike_dcc only, per directive gate)
  - site_features (~1500+ rows, one per detected site)
"""
import json, glob, re, subprocess, sys, os
from pathlib import Path
from collections import defaultdict

DCC_PATH = "/mnt/storage/spike-audit/dcc-recompute/corrected_dcc_results.json"
CACHE = Path("/tmp/spike_count_audit")
LOG_DIR = Path("/mnt/storage/prism-outputs/_corpus_runner_logs")
OUT_DIR = Path("/home/diddy/Desktop/Prism4D-bio/cloudflare/d1")

def sql_escape(s):
    if s is None: return "NULL"
    return "'" + str(s).replace("'", "''") + "'"

def sql_num(n):
    if n is None: return "NULL"
    return str(n)

# ── 1. Corrected DCC records ──
dcc_records = json.load(open(DCC_PATH))
print(f"Loaded {len(dcc_records)} DCC records")

# Target → DCC record map
dcc_map = {r['target']: r for r in dcc_records}

# ── 2. Engine times from run logs ──
engine_time = {}
for log_path in LOG_DIR.glob("*_per_target.log"):
    for line in open(log_path):
        m = re.match(r'^(\S+)\s+OK\s+engine=(\d+)s', line)
        if m:
            engine_time[m.group(1)] = int(m.group(2))
print(f"Engine times recorded for {len(engine_time)} targets from runner logs")

# ── 3. Build INSERT statements ──
targets_sql = []
dcc_sql = []
site_features_sql = []

bs_files = sorted(CACHE.glob("*/*.binding_sites.json"))
print(f"Processing {len(bs_files)} binding_sites.json files...")

n_sites_total = 0
targets_seen = set()

for bs_path in bs_files:
    target = bs_path.name.replace(".binding_sites.json", "")
    targets_seen.add(target)

    # Parse target name: e.g. "9ymg_chainC" → pdb_id=9ymg, chain=C
    m = re.match(r'^([0-9a-z]{4})_chain([A-Z0-9])$', target)
    if not m:
        print(f"  WARN: skipping non-standard target name: {target}")
        continue
    pdb_id, chain = m.group(1), m.group(2)

    try:
        bs = json.load(open(bs_path))
    except Exception as e:
        print(f"  ERROR reading {bs_path}: {e}")
        continue

    sites = bs.get('sites', [])
    n_sites_total += len(sites)

    # Ground truth for ligand info
    gt_path = bs_path.parent / f"{target}_ground_truth.json"
    ligand_code = None
    ligand_heavy_atoms = None
    if gt_path.exists():
        try:
            gt = json.load(open(gt_path))
            lig = gt.get('ligand', {})
            ligand_code = lig.get('resname')
            ligand_heavy_atoms = lig.get('n_atoms')
        except Exception:
            pass

    # Atom count / residue count from first site's data or binding_sites.json top-level
    atom_count = bs.get('n_atoms')
    residue_count = None
    if sites and 'lining_residues' in sites[0]:
        # crude: max resid across all sites + 1
        try:
            all_res = set()
            for s in sites:
                for lr in s.get('lining_residues', []):
                    r = lr.get('resid', lr.get('resnum'))
                    if r is not None: all_res.add(r)
            if all_res:
                residue_count = max(all_res) + 1
        except Exception:
            pass

    eng_time = engine_time.get(target, None)

    # Determine spike_percentile (pct70 targets from final run)
    pct70_targets = {'9o9i_chainA', '9ohy_chainB', '9pi9_chainE', '9qyd_chainA', '9tcb_chainB'}
    percentile = 70 if target in pct70_targets else 95

    # targets INSERT
    targets_sql.append(
        f"INSERT OR REPLACE INTO targets "
        f"(target, pdb_id, chain, atom_count, residue_count, ligand_code, ligand_heavy_atoms, "
        f"engine_flags, spike_percentile, engine_time_seconds, n_sites_detected, status) VALUES ("
        f"{sql_escape(target)}, {sql_escape(pdb_id)}, {sql_escape(chain)}, "
        f"{sql_num(atom_count)}, {sql_num(residue_count)}, "
        f"{sql_escape(ligand_code)}, {sql_num(ligand_heavy_atoms)}, "
        f"{sql_escape('multi-stream 4 multi-scale fast hysteresis prism-therm')}, "
        f"{percentile}, {sql_num(eng_time)}, {len(sites)}, 'completed');"
    )

    # corrected_dcc INSERT (only for targets with spike_dcc NOT NULL)
    dcc = dcc_map.get(target)
    if dcc and dcc.get('spike_dcc') is not None:
        dcc_sql.append(
            f"INSERT OR REPLACE INTO corrected_dcc "
            f"(target, centroid_dcc, spike_dcc, spike_site, n_parquet_sites, dcc_grade) VALUES ("
            f"{sql_escape(target)}, {sql_num(dcc.get('centroid_dcc'))}, "
            f"{sql_num(dcc.get('spike_dcc'))}, {sql_escape(dcc.get('spike_site'))}, "
            f"{sql_num(dcc.get('n_parquet_sites'))}, {sql_escape(dcc.get('spike_grade'))});"
        )

    # site_features: one row per site
    for s in sites:
        site_name = f"site{s.get('id','unknown')}"
        spike_count = s.get('spike_count', 0)
        n_streams = 4  # from --multi-stream 4

        # Compute burial from lining residues
        lining = s.get('lining_residues', [])
        burial = None
        if lining:
            dists = [lr.get('min_distance_angstrom', 0) for lr in lining if lr.get('min_distance_angstrom') is not None]
            if dists:
                burial = sum(dists) / len(dists)

        # Spread (crude — volume cbrt)
        vol = s.get('volume_angstrom3', s.get('volume'))
        spread = None
        if vol and vol > 0:
            spread = vol ** (1/3)

        # spike_density
        spike_density = None
        if spike_count and spread and spread > 0:
            spike_density = spike_count / (spread ** 3)

        # min_dist_to_ligand from DCC (if this is the best-matching site)
        min_dist = None
        graded = None
        if dcc and dcc.get('spike_site') == site_name:
            min_dist = dcc.get('spike_dcc')
            if min_dist is not None:
                graded = 1.0 / (1.0 + min_dist)

        site_features_sql.append(
            f"INSERT OR REPLACE INTO site_features "
            f"(target, site_name, spike_count, n_streams, spread, burial, spike_density, "
            f"min_dist_to_ligand, graded_score, source) VALUES ("
            f"{sql_escape(target)}, {sql_escape(site_name)}, "
            f"{sql_num(spike_count)}, {n_streams}, "
            f"{sql_num(spread)}, {sql_num(burial)}, {sql_num(spike_density)}, "
            f"{sql_num(min_dist)}, {sql_num(graded)}, 'binding_sites_json');"
        )

print(f"\nGenerated:")
print(f"  targets:       {len(targets_sql)} rows")
print(f"  corrected_dcc: {len(dcc_sql)} rows (non-null spike_dcc)")
print(f"  site_features: {len(site_features_sql)} rows")
print(f"  Total sites:   {n_sites_total}")

# Write as batched SQL files (D1 has 100KB per statement limit, batch by ~500 INSERTs)
def write_batches(name, stmts, batch_size=500):
    n_batches = 0
    for i in range(0, len(stmts), batch_size):
        batch = stmts[i:i+batch_size]
        out = OUT_DIR / f"populate_{name}_batch{i//batch_size:03d}.sql"
        with open(out, "w") as f:
            f.write("\n".join(batch))
        n_batches += 1
    return n_batches

n_targets_batches = write_batches("targets", targets_sql)
n_dcc_batches = write_batches("dcc", dcc_sql)
n_site_batches = write_batches("sites", site_features_sql, batch_size=200)

print(f"\nBatch files written to {OUT_DIR}:")
print(f"  populate_targets_batch*.sql: {n_targets_batches} file(s)")
print(f"  populate_dcc_batch*.sql: {n_dcc_batches} file(s)")
print(f"  populate_sites_batch*.sql: {n_site_batches} file(s)")

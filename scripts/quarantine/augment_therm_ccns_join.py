"""
Post-freeze therm/CCNS augmentation join.

Joins per-site validation CSVs (prism_site_vs_ligand_shells.csv) against
frozen binding_sites.json on site_id == id.

Join key: CSV.site_id = JSON.sites[*].id
Output: augmented_per_site_validation_with_therm_ccns.csv
"""
import csv
import glob
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

BLIND_VAL_RUNS = Path("/mnt/storage/prism-outputs/blind_validation")
BLIND_VAL_SCORING = Path("/home/diddy/Desktop/Prism4D-bio/docs/blind_validation/post_freeze_validation")
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUTDIR = BLIND_VAL_RUNS / f"therm_ccns_augment_{TIMESTAMP}"
(OUTDIR / "augmented").mkdir(parents=True, exist_ok=True)

THERM_FIELDS = [
    "ccns_tau",
    "cold_phase_fraction",
    "tide_coupling_score",
    "effective_delta_g_kcal_mol",
    "delta_g_aromatic_kcal_mol",
    "delta_g_cooperative_kcal_mol",
    "delta_g_dewetting_kcal_mol",
    "delta_g_electrostatic_kcal_mol",
    "delta_g_sti_kcal_mol",
    "uv_enrichment_score",
    "frustrated_solvent_score",
    "wd_coherence",
    "onset_score",
    "breathing_score",
    "sti_n_spikes",
    "sti_n_voxels",
    "signal_preservation",
    "kinetic_accessibility",
    "aromatic_score",
    "source_diversity",
    "spike_count",
    "composite_audit_rank",
    "composite_audit_score",
    "composite_v3_rank",
    "gtck_rank",
    "cryptic_rank",
    "cryptic_score",
    "druggability",
    "is_druggable",
    "volume",
    "sphericity",
    "mean_burial",
    "ray_escape_ratio",
    "burial_score",
    "asymmetry_offset",
    "relative_asymmetry",
    "localization_score_raw",
    "catalytic_residue_count",
]

TARGETS = [
    ("B01_HRAS_Q61H",        "4L9S"),
    ("B02_CDK2_allosteric",  "1HCL"),
    ("B03_Kv1.2",            "3LUT"),
    ("B04_MDM2",             "1YCR"),
    ("B05_TP53_R175H",       "2OCJ"),
    ("B06_cGAS",             "4KM5"),
    ("B07_TEAD1",            "3KYS"),
    ("B08_CRBN",             "4TZ4_chainC"),
    ("B09_Thrombin_exosite", "1PPB"),
    ("B10_ADRB2",            "2RH1"),
]

all_rows = []
audit_lines = []
sha256_additions = []

for target_id, pdb_stem in TARGETS:
    bs_path = BLIND_VAL_RUNS / target_id / "frozen" / f"{pdb_stem}.binding_sites.json"
    csv_path = BLIND_VAL_SCORING / target_id / "prism_site_vs_ligand_shells.csv"

    if not bs_path.exists():
        audit_lines.append(f"MISSING binding_sites: {target_id} {bs_path}")
        continue
    if not csv_path.exists():
        audit_lines.append(f"MISSING scoring_csv: {target_id} {csv_path}")
        continue

    with open(bs_path) as f:
        bs = json.load(f)

    sites = bs.get("sites", [])
    lookup = {}
    for s in sites:
        sid = str(s.get("id", ""))
        if sid:
            lookup[sid] = s

    # also check prism_therm block (dict with .sites array)
    prism_therm_lookup = {}
    pt_block = bs.get("prism_therm", {})
    if isinstance(pt_block, dict):
        for pt in pt_block.get("sites", []):
            if isinstance(pt, dict):
                sid = str(pt.get("site_id", pt.get("id", "")))
                if sid:
                    prism_therm_lookup[sid] = pt

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    matched = 0
    for row in rows:
        row["target"] = target_id
        sid = str(row.get("site_id", "")).strip()
        site_data = lookup.get(sid, {})
        pt_data = prism_therm_lookup.get(sid, {})

        for field in THERM_FIELDS:
            val = site_data.get(field, pt_data.get(field, ""))
            # prefix to avoid collision with existing columns
            col = f"aug_{field}" if field in row else field
            row[col] = val

        if site_data:
            matched += 1
        all_rows.append(row)

    sha_val = hashlib.sha256(bs_path.read_bytes()).hexdigest()
    sha256_additions.append(f"{sha_val}  {bs_path}")

    match_rate = matched / len(rows) if rows else 0.0
    audit_lines.append(
        f"{target_id}: {matched}/{len(rows)} matched ({match_rate:.2%}) from {bs_path.name}"
    )
    print(f"  {target_id}: {matched}/{len(rows)} matched", flush=True)

# Write augmented CSV
if all_rows:
    fieldnames = list(all_rows[0].keys())
    out_csv = OUTDIR / "augmented" / "augmented_per_site_validation_with_therm_ccns.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows → {out_csv}")

# Write audit
audit_path = OUTDIR / "augmented" / "therm_ccns_join_audit.md"
with open(audit_path, "w") as f:
    f.write("# Therm/CCNS descriptor join audit\n\n")
    f.write(f"- generated_utc: {datetime.now(timezone.utc).isoformat()}\n")
    f.write(f"- total_rows: {len(all_rows)}\n")
    f.write(f"- join_key: CSV.site_id = JSON.sites[*].id\n")
    f.write(f"- source: frozen binding_sites.json\n")
    f.write(f"- augmented_fields: {len(THERM_FIELDS)}\n\n")
    f.write("## Per-target audit\n\n")
    for line in audit_lines:
        f.write(f"- {line}\n")

# Append SHA256 entries for binding_sites.json files to freeze manifest
sha_manifest = Path("/home/diddy/Desktop/Prism4D-bio/docs/blind_validation/frozen_predictions/sha256_manifest.txt")
if sha256_additions:
    with open(sha_manifest, "a") as f:
        f.write("\n# binding_sites.json + prism_therm.json added post-run (therm/CCNS augmentation)\n")
        for line in sha256_additions:
            f.write(line + "\n")

    # also hash the prism_therm.json files
    for target_id, pdb_stem in TARGETS:
        pt_path = BLIND_VAL_RUNS / target_id / "frozen" / f"{pdb_stem}.topology.prism_therm.json"
        if pt_path.exists():
            sha_val = hashlib.sha256(pt_path.read_bytes()).hexdigest()
            with open(sha_manifest, "a") as f:
                f.write(f"{sha_val}  {pt_path}\n")

    print(f"\nAppended {len(sha256_additions)} binding_sites.json + prism_therm.json hashes → {sha_manifest}")

print("\nDone:", OUTDIR)

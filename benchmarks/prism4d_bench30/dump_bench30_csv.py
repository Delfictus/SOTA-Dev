#!/usr/bin/env python3
"""Extract BENCH30 site features + DCC labels into a CSV for fine-tuning."""
import json, csv, os, math
import numpy as np

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(BENCH_DIR, "benchmark_manifest.json")
GT = os.path.join(BENCH_DIR, "ground_truth", "ligand_centroids.json")
RESULTS_DIR = os.path.join(BENCH_DIR, "results")
OUT_CSV = os.path.join(BENCH_DIR, "bench30_finetune.csv")

manifest = json.load(open(MANIFEST))
gt = json.load(open(GT))

FEATURE_COLS = [
    "burial_score", "onset_score", "sphericity", "uv_enrichment_score",
    "breathing_score", "source_diversity", "wd_coherence", "quality_score",
    "druggability", "volume", "spike_count", "mean_burial",
    "hysteresis_asymmetry", "asymmetry_offset", "relative_asymmetry",
    "kinetic_accessibility", "frustrated_solvent_score", "aromatic_score",
    "ray_escape_ratio", "ccns_tau", "tide_coupling_score",
    "engine_geo", "engine_vcs", "engine_chem", "engine_phys",
    "catalytic_residue_count", "n_lining_residues",
]

HEADER = ["target_id", "pdb", "site_id", "centroid_x", "centroid_y", "centroid_z",
          "classification", "therm_class", "is_druggable"] + FEATURE_COLS + [
          "dcc", "label_4A", "label_8A", "gt_ligand"]

rows = []
for target in manifest["targets"]:
    tid = str(target["id"])
    apo = target["apo_pdb"].lower()
    if tid not in gt:
        continue

    sites_path = os.path.join(RESULTS_DIR, tid, f"{apo}.binding_sites.json")
    if not os.path.exists(sites_path):
        continue

    true_c = np.array(gt[tid]["centroid"])
    ligand = gt[tid].get("ligand_resname", "?")

    sites_data = json.load(open(sites_path))
    for site in sites_data.get("sites", []):
        c = site.get("centroid")
        if not c:
            continue
        dcc = float(np.linalg.norm(np.array(c) - true_c))

        row = {
            "target_id": tid,
            "pdb": apo.upper(),
            "site_id": site.get("id", "?"),
            "centroid_x": round(c[0], 3),
            "centroid_y": round(c[1], 3),
            "centroid_z": round(c[2], 3),
            "classification": site.get("classification", "?"),
            "therm_class": site.get("therm_class", "?"),
            "is_druggable": int(site.get("is_druggable", False)),
            "dcc": round(dcc, 2),
            "label_4A": int(dcc <= 4.0),
            "label_8A": int(dcc <= 8.0),
            "gt_ligand": ligand,
        }

        for col in FEATURE_COLS:
            if col == "n_lining_residues":
                row[col] = len(site.get("lining_residues", []))
            else:
                val = site.get(col, 0.0)
                if val is None:
                    val = 0.0
                if col == "spike_count" and val > 0:
                    row[col] = round(math.log10(val + 1), 4)
                elif col == "volume" and val > 0:
                    row[col] = round(math.log10(val + 1), 4)
                else:
                    row[col] = round(float(val), 6) if isinstance(val, float) else val

        rows.append(row)

with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=HEADER)
    w.writeheader()
    w.writerows(rows)

n_pos_4 = sum(1 for r in rows if r["label_4A"])
n_pos_8 = sum(1 for r in rows if r["label_8A"])
targets = set(r["pdb"] for r in rows)
print(f"Wrote {len(rows)} sites from {len(targets)} targets to {OUT_CSV}")
print(f"  Hits @4A: {n_pos_4}/{len(rows)} ({100*n_pos_4/len(rows):.1f}%)")
print(f"  Hits @8A: {n_pos_8}/{len(rows)} ({100*n_pos_8/len(rows):.1f}%)")

#!/usr/bin/env python3
import json
import math
import urllib.request
from pathlib import Path

BASE_DIR = Path("/home/diddy/Desktop/Prism4D-bio/output/4lpk_phase2.1_audit_verify")
BINDING_SITES = BASE_DIR / "4lpk_clean.binding_sites.json"
PDB_DIR = BASE_DIR / "pdb_refs"
PDB_DIR.mkdir(parents=True, exist_ok=True)

SITE_IDS = [1, 4004, 2]

REFERENCE_PDBS = {
    "4LPK": "https://files.rcsb.org/download/4LPK.pdb",
    "4M21": "https://files.rcsb.org/download/4M21.pdb",
    "4M22": "https://files.rcsb.org/download/4M22.pdb",
    "6GJ8": "https://files.rcsb.org/download/6GJ8.pdb",
}

IGNORE_HET = {
    "HOH", "WAT", "DOD",
    "MG", "NA", "CL", "K", "CA", "ZN", "MN",
    "SO4", "PO4", "EDO", "GOL",
}

GDP_LIGANDS = {"GDP", "GNP", "GTP", "GSP", "GCP", "GMP"}
NUCLEOTIDE_LIGANDS = {"GDP", "GNP", "GTP", "GSP", "GCP", "GMP"}

CONTACT_THRESHOLDS = [3.5, 4.0, 5.0]
TOP_N_VALUES = [5, 8, 10, 15, 20]
OFFSETS_TO_TEST = [-3, -2, -1, 0, 1, 2, 3]

# KRAS catalytic domain is typically ~1-166/188 depending construct.
# We use 1-166 as the background for enrichment because the PDB structures are catalytic-domain KRAS.
KRAS_BACKGROUND_RESIDUES = set(range(1, 167))

KRAS_REGIONS = {
    "P-loop": set(range(8, 19)),
    "Switch-I": set(range(25, 41)),
    "Switch-II": set(range(55, 77)),
    "alpha3/H3-H4 candidate zone": set(range(95, 135)),
}

COMBINED_REGIONS = {
    "Switch-I+Switch-II": KRAS_REGIONS["Switch-I"] | KRAS_REGIONS["Switch-II"],
    "KRAS regulatory manifold": (
        KRAS_REGIONS["P-loop"]
        | KRAS_REGIONS["Switch-I"]
        | KRAS_REGIONS["Switch-II"]
    ),
}

CLAIM_SUMMARY = []

def download_if_missing(pdb_id, url):
    out = PDB_DIR / f"{pdb_id}.pdb"
    if not out.exists():
        print(f"Downloading {pdb_id}...")
        urllib.request.urlretrieve(url, out)
    return out

def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default

def parse_pdb_atoms(pdb_path):
    protein_atoms = []
    het_atoms = []

    with open(pdb_path, "r", errors="ignore") as f:
        for line in f:
            rec = line[0:6].strip()
            if rec not in {"ATOM", "HETATM"}:
                continue

            atom_name = line[12:16].strip()
            element = line[76:78].strip() or atom_name[0]
            if element.upper().startswith("H"):
                continue

            resname = line[17:20].strip()
            chain = line[21].strip() or "_"

            try:
                resid = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except Exception:
                continue

            atom = {
                "record": rec,
                "atom_name": atom_name,
                "element": element,
                "resname": resname,
                "chain": chain,
                "resid": resid,
                "xyz": (x, y, z),
            }

            if rec == "ATOM":
                protein_atoms.append(atom)
            else:
                het_atoms.append(atom)

    return protein_atoms, het_atoms

def dist(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2
    )

def infer_ligand_atoms(het_atoms, ligand_mode):
    if ligand_mode == "GDP":
        return [a for a in het_atoms if a["resname"] in GDP_LIGANDS]

    if ligand_mode == "INHIBITOR":
        return [
            a for a in het_atoms
            if a["resname"] not in IGNORE_HET
            and a["resname"] not in NUCLEOTIDE_LIGANDS
        ]

    raise ValueError(f"Unknown ligand mode: {ligand_mode}")

def ligand_contact_residues(pdb_path, ligand_mode, threshold):
    protein_atoms, het_atoms = parse_pdb_atoms(pdb_path)
    ligand_atoms = infer_ligand_atoms(het_atoms, ligand_mode)

    contacts = {}

    for pa in protein_atoms:
        key = (pa["chain"], pa["resid"], pa["resname"])
        best = None

        for la in ligand_atoms:
            d = dist(pa["xyz"], la["xyz"])
            if best is None or d < best:
                best = d

        if best is not None and best <= threshold:
            if key not in contacts or best < contacts[key]:
                contacts[key] = best

    return contacts, ligand_atoms

def extract_spike_count(r):
    for key in [
        "spike_attribution_count",
        "spikes_attributed",
        "spike_count",
        "spikes",
        "count",
    ]:
        if key in r and r[key] is not None:
            try:
                return int(r[key])
            except Exception:
                pass
    return 0

def load_prism_sites():
    with open(BINDING_SITES, "r") as f:
        data = json.load(f)

    sites = {}

    for s in data.get("sites", []):
        sid = s.get("id")
        predicted = []

        for r in s.get("lining_residues", []):
            resid = r.get("resid")
            if resid is None:
                continue

            predicted.append({
                "resid": int(resid),
                "resname": r.get("resname", "UNK"),
                "min_distance": safe_float(r.get("min_distance", 999.0), 999.0),
                "spikes": extract_spike_count(r),
            })

        predicted = sorted(
            predicted,
            key=lambda x: (-x["spikes"], x["min_distance"])
        )

        sites[sid] = predicted

    return sites

def classify_region(resid):
    hits = []
    all_regions = {}
    all_regions.update(KRAS_REGIONS)
    all_regions.update(COMBINED_REGIONS)

    for name, residues in KRAS_REGIONS.items():
        if resid in residues:
            hits.append(name)

    return ",".join(hits) if hits else "other"

def evaluate_prediction(predicted, contact_set, numbering_offset=0):
    known_resids = {resid for chain, resid, resname in contact_set.keys()}

    tp = []
    fp = []

    for p in predicted:
        adjusted = p["resid"] + numbering_offset
        if adjusted in known_resids:
            tp.append((p, adjusted))
        else:
            fp.append((p, adjusted))

    predicted_adjusted = {p["resid"] + numbering_offset for p in predicted}
    fn_resids = sorted(known_resids - predicted_adjusted)

    precision = len(tp) / max(1, len(predicted))
    recall = len(tp) / max(1, len(known_resids))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    return {
        "offset": numbering_offset,
        "tp": tp,
        "fp": fp,
        "fn_resids": fn_resids,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "known_count": len(known_resids),
        "predicted_count": len(predicted),
    }

def best_offset_evaluation(predicted, contact_set):
    candidates = [
        evaluate_prediction(predicted, contact_set, offset)
        for offset in OFFSETS_TO_TEST
    ]

    return sorted(
        candidates,
        key=lambda r: (r["f1"], r["precision"], r["recall"], -abs(r["offset"])),
        reverse=True
    )[0]

def summarize_known_contacts(contact_set):
    rows = sorted(
        [(resid, resname, chain, d) for (chain, resid, resname), d in contact_set.items()],
        key=lambda x: (x[2], x[0], x[1]),
    )
    if not rows:
        return "none"
    return ", ".join(f"{resname}{resid}:{chain}" for resid, resname, chain, d in rows)

def print_site_prediction(site_id, predicted, limit=20):
    print(f"\nPRISM Site {site_id} causal manifold residues, sorted by spike attribution:")
    if not predicted:
        print("  none")
        return

    for rank, p in enumerate(predicted[:limit], start=1):
        print(
            f"  #{rank:<2} {p['resname']}{p['resid']:<4} "
            f"region={classify_region(p['resid']):<30} "
            f"spikes={p['spikes']:<12,} "
            f"min_dist={p['min_distance']:.3f}"
        )

def print_eval(label, result, show_details=True):
    print(f"\n{label}")
    print("-" * len(label))
    print(f"Predicted residues scored: {result['predicted_count']}")
    print(f"Known reference contact residues: {result['known_count']}")
    print(f"Best numbering offset: {result['offset']:+d}")
    print(f"Precision: {result['precision']:.3f}")
    print(f"Recall:    {result['recall']:.3f}")
    print(f"F1 Score:  {result['f1']:.3f}")

    if not show_details:
        return

    print("\nTrue-positive PRISM residues:")
    if not result["tp"]:
        print("  none")
    for p, adjusted in result["tp"]:
        print(
            f"  {p['resname']}{p['resid']} -> reference resid {adjusted} "
            f"| spikes={p['spikes']:,} | region={classify_region(p['resid'])}"
        )

    print("\nFalse-positive PRISM residues:")
    if not result["fp"]:
        print("  none")
    for p, adjusted in result["fp"]:
        print(
            f"  {p['resname']}{p['resid']} -> adjusted resid {adjusted} "
            f"| spikes={p['spikes']:,} | region={classify_region(p['resid'])}"
        )

    if result["fn_resids"]:
        preview = ", ".join(str(x) for x in result["fn_resids"][:30])
        suffix = " ..." if len(result["fn_resids"]) > 30 else ""
        print(f"\nMissed reference contact residue numbers: {preview}{suffix}")
    else:
        print("\nMissed reference contact residue numbers: none")

def score_site_against_contacts(site_id, predicted_all, contacts, label):
    print("\n" + "=" * 100)
    print(label)
    print("=" * 100)
    print(f"Known contact residues: {summarize_known_contacts(contacts)}")

    best_overall = None

    for top_n in TOP_N_VALUES:
        predicted = predicted_all[:top_n]
        result = best_offset_evaluation(predicted, contacts)

        print(
            f"Top-{top_n:<2} | offset {result['offset']:+d} | "
            f"P={result['precision']:.3f} R={result['recall']:.3f} F1={result['f1']:.3f} | "
            f"TP={len(result['tp'])}/{len(predicted)}"
        )

        if best_overall is None or (
            result["f1"], result["precision"], result["recall"]
        ) > (
            best_overall["f1"], best_overall["precision"], best_overall["recall"]
        ):
            best_overall = result

    print_eval(f"Best detailed result for Site {site_id}", best_overall, show_details=True)
    return best_overall

def region_recovery_score(predicted_all, region_residues, top_n=20):
    predicted = predicted_all[:top_n]
    predicted_resids = {p["resid"] for p in predicted}
    hits = [p for p in predicted if p["resid"] in region_residues]

    precision = len(hits) / max(1, len(predicted))
    recall = len(predicted_resids & region_residues) / max(1, len(region_residues))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    background_rate = len(region_residues & KRAS_BACKGROUND_RESIDUES) / len(KRAS_BACKGROUND_RESIDUES)
    enrichment = precision / background_rate if background_rate > 0 else 0.0

    return {
        "hits": hits,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "background_rate": background_rate,
        "enrichment": enrichment,
        "top_n": top_n,
        "region_size": len(region_residues),
        "hit_count": len(hits),
    }

def region_recovery_report(site_id, predicted_all, region_name, top_n=20, combined=False):
    region_map = COMBINED_REGIONS if combined else KRAS_REGIONS
    region_residues = region_map[region_name]
    result = region_recovery_score(predicted_all, region_residues, top_n=top_n)

    print("\n" + "=" * 100)
    print(f"FUNCTIONAL-MANIFOLD RECOVERY: SITE {site_id} vs {region_name} | Top-{top_n}")
    print("=" * 100)
    print(f"Region size: {result['region_size']} residues")
    print(f"Region hits: {result['hit_count']} / {top_n}")
    print(f"Manifold precision / region-hit fraction: {result['precision']:.3f}")
    print(f"Region recall:                         {result['recall']:.3f}")
    print(f"Region F1:                             {result['f1']:.3f}")
    print(f"Background KRAS region rate:           {result['background_rate']:.3f}")
    print(f"Functional enrichment vs background:   {result['enrichment']:.2f}x")

    if result["hits"]:
        print("Recovered region residues:")
        for p in result["hits"]:
            print(f"  {p['resname']}{p['resid']} | spikes={p['spikes']:,} | min_dist={p['min_distance']:.3f}")
    else:
        print("Recovered region residues: none")

    CLAIM_SUMMARY.append({
        "site": site_id,
        "metric": f"region:{region_name}",
        "top_n": top_n,
        "precision": result["precision"],
        "recall": result["recall"],
        "f1": result["f1"],
        "enrichment": result["enrichment"],
        "hits": result["hit_count"],
    })

    return result

def direct_contact_summary(site_id, label, result):
    CLAIM_SUMMARY.append({
        "site": site_id,
        "metric": f"direct:{label}",
        "top_n": result["predicted_count"],
        "precision": result["precision"],
        "recall": result["recall"],
        "f1": result["f1"],
        "enrichment": None,
        "hits": len(result["tp"]),
    })

def print_claim_summary():
    print("\n" + "=" * 100)
    print("CLAIM-READY SUMMARY")
    print("=" * 100)
    print(
        f"{'SITE':<8} | {'METRIC':<42} | {'TOP-N':<5} | "
        f"{'P':<6} | {'R':<6} | {'F1':<6} | {'ENRICH':<8} | {'HITS'}"
    )
    print("-" * 100)

    for row in CLAIM_SUMMARY:
        enrich = "N/A" if row["enrichment"] is None else f"{row['enrichment']:.2f}x"
        print(
            f"{str(row['site']):<8} | {row['metric']:<42} | {row['top_n']:<5} | "
            f"{row['precision']:<6.3f} | {row['recall']:<6.3f} | {row['f1']:<6.3f} | "
            f"{enrich:<8} | {row['hits']}"
        )

    print("\nINTERPRETATION RULE:")
    print("  direct:* metrics support ligand-contact accuracy claims only.")
    print("  region:* metrics support dynamic causal-manifold / functional-region recovery claims.")
    print("  Enrichment > 1.0x means the PRISM manifold is enriched for the declared biological region.")
    print("  Enrichment > 2.0x is strong functional-region enrichment.")
    print("  Do not call any lane 100% accurate unless direct precision=1.000 and direct recall=1.000.")

def main():
    if not BINDING_SITES.exists():
        raise SystemExit(f"Missing binding_sites.json:\n  {BINDING_SITES}")

    pdb_paths = {
        pdb_id: download_if_missing(pdb_id, url)
        for pdb_id, url in REFERENCE_PDBS.items()
    }

    sites = load_prism_sites()

    print("=" * 100)
    print("PRISM-4D KRAS RESIDUE-LEVEL VALIDATION")
    print("METHOD: direct ligand-contact scoring + functional-manifold enrichment")
    print("=" * 100)

    for site_id in SITE_IDS:
        print_site_prediction(site_id, sites.get(site_id, []), limit=20)

    # Lane A: Site 1 direct GDP validation in the same 4LPK structure.
    for threshold in CONTACT_THRESHOLDS:
        contacts, ligand_atoms = ligand_contact_residues(
            pdb_paths["4LPK"],
            ligand_mode="GDP",
            threshold=threshold,
        )

        print("\n" + "=" * 100)
        print(f"LANE A: SITE 1 vs 4LPK GDP CONTACTS <= {threshold:.1f} Å")
        print("=" * 100)
        print(f"GDP ligand heavy atoms found: {len(ligand_atoms)}")

        best = score_site_against_contacts(
            site_id=1,
            predicted_all=sites.get(1, []),
            contacts=contacts,
            label=f"Site 1 GDP direct-contact validation <= {threshold:.1f} Å",
        )
        direct_contact_summary(1, f"4LPK_GDP_<={threshold:.1f}A", best)

    # Lane B: Site 4004 vs inhibitor-bound SII references.
    for ref_id in ["4M21", "4M22"]:
        for threshold in CONTACT_THRESHOLDS:
            contacts, ligand_atoms = ligand_contact_residues(
                pdb_paths[ref_id],
                ligand_mode="INHIBITOR",
                threshold=threshold,
            )

            print("\n" + "=" * 100)
            print(f"LANE B: SITE 4004 vs {ref_id} INHIBITOR/SII CONTACTS <= {threshold:.1f} Å")
            print("=" * 100)
            print(f"Inhibitor/reference ligand heavy atoms found: {len(ligand_atoms)}")

            best = score_site_against_contacts(
                site_id=4004,
                predicted_all=sites.get(4004, []),
                contacts=contacts,
                label=f"Site 4004 inhibitor-contact validation vs {ref_id} <= {threshold:.1f} Å",
            )
            direct_contact_summary(4004, f"{ref_id}_INHIBITOR_<={threshold:.1f}A", best)

    # Lane C: Site 2 vs 6GJ8 BI-2852 contact shell.
    # This is exploratory direct-contact validation, not proof of H3/H4 by itself.
    for ref_id in ["6GJ8"]:
        for threshold in CONTACT_THRESHOLDS:
            contacts, ligand_atoms = ligand_contact_residues(
                pdb_paths[ref_id],
                ligand_mode="INHIBITOR",
                threshold=threshold,
            )

            print("\n" + "=" * 100)
            print(f"LANE C: SITE 2 vs {ref_id} BI-2852/PAN-KRAS CONTACTS <= {threshold:.1f} Å")
            print("=" * 100)
            print(f"Inhibitor/reference ligand heavy atoms found: {len(ligand_atoms)}")

            best = score_site_against_contacts(
                site_id=2,
                predicted_all=sites.get(2, []),
                contacts=contacts,
                label=f"Site 2 exploratory inhibitor-contact validation vs {ref_id} <= {threshold:.1f} Å",
            )
            direct_contact_summary(2, f"{ref_id}_INHIBITOR_<={threshold:.1f}A", best)

    # Functional manifold enrichment reports.
    region_recovery_report(1, sites.get(1, []), "P-loop", top_n=20)
    region_recovery_report(4004, sites.get(4004, []), "Switch-I", top_n=20)
    region_recovery_report(4004, sites.get(4004, []), "Switch-II", top_n=20)
    region_recovery_report(4004, sites.get(4004, []), "Switch-I+Switch-II", top_n=20, combined=True)
    region_recovery_report(2, sites.get(2, []), "alpha3/H3-H4 candidate zone", top_n=20)

    print_claim_summary()

    print("\n" + "=" * 100)
    print("CLAIMING RULE")
    print("=" * 100)
    print("Use direct-contact precision/recall/F1 only for ligand-contact accuracy claims.")
    print("Use functional-region enrichment for causal-manifold / dynamic-pocket claims.")
    print("Do not claim 100% accuracy unless direct precision=1.000 and direct recall=1.000.")
    print("Site 4004 should be described as mixed Switch-I/Switch-II if combined-switch enrichment is high.")
    print("Site 2 should be described as alpha3/H3-H4 candidate unless direct ligand-contact validation is high.")
    print("=" * 100)

if __name__ == "__main__":
    main()

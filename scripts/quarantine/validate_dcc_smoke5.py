#!/usr/bin/env python3
# [DIAGNOSTIC] DCC validation for smoke test targets.
#
# For each target: read holo PDB, extract ligand HETATM coordinates,
# compute centroid, then compute distance from each engine-detected
# site centroid to the ligand centroid. Report best-DCC.
#
# Reads:
#   /tmp/dcc_validation/<pdb>_holo.pdb
#   /mnt/storage/prism-outputs/10k-runs/<target>/<target>.binding_sites.json
#
# Pure read-only. No file modification, no network calls except RCSB curl
# already done by the caller.

import json
import math
import sys
from pathlib import Path

# Solvents/ions/cryoprotectants to skip when looking for "real" ligands.
# Single-atom metals (ZN, MG, FE, MN, CU, CA, NI) are SOMETIMES catalytic
# cofactors at the binding site, so we keep them as fallback ligands.
NUISANCE = {"HOH", "H2O", "DOD", "WAT", "GOL", "EDO", "DMS", "PEG", "PG4",
            "SO4", "PO4", "ACT", "CIT", "TRS", "BME", "MES", "HEPES",
            "EPE", "FMT", "ACT", "IPA", "MPD", "BOG"}
COMMON_IONS = {"NA", "CL", "K", "BR", "I", "F", "CS", "RB"}
# Metals we keep as potential single-atom catalytic cofactors:
METAL_COFACTORS = {"ZN", "MG", "FE", "MN", "CU", "CA", "NI", "CO"}


def parse_ligand_atoms(holo_pdb, target_chain=None):
    """
    Extract HETATM ligand atoms grouped by (resname, chain, resnum).
    Returns list of (resname, chain, resnum, atoms) where atoms is [(x,y,z),...].
    """
    groups = {}
    with open(holo_pdb) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            if resname in NUISANCE or resname in COMMON_IONS:
                continue
            chain = line[21]
            try:
                resnum = int(line[22:26])
            except ValueError:
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            key = (resname, chain, resnum)
            groups.setdefault(key, []).append((x, y, z))
    return groups


def centroid(atoms):
    n = len(atoms)
    if n == 0:
        return None
    return (sum(a[0] for a in atoms) / n,
            sum(a[1] for a in atoms) / n,
            sum(a[2] for a in atoms) / n)


def dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def select_best_ligand(groups, prefer_chain=None):
    """
    Pick the most likely 'real' ligand. Heuristic:
      1. Prefer multi-atom organics (>=10 atoms) on prefer_chain
      2. Else multi-atom organics on any chain
      3. Else single-atom metal cofactors on prefer_chain
      4. Else single-atom metal cofactors on any chain
    Returns (resname, chain, resnum, atoms, centroid, kind) or None.
    """
    multi_pref = []
    multi_any = []
    metal_pref = []
    metal_any = []

    for (resname, chain, resnum), atoms in groups.items():
        n = len(atoms)
        is_metal = resname in METAL_COFACTORS and n == 1
        is_organic = n >= 5 and not is_metal
        on_pref = (prefer_chain is None) or (chain == prefer_chain)

        if is_organic and on_pref:
            multi_pref.append((resname, chain, resnum, atoms))
        elif is_organic:
            multi_any.append((resname, chain, resnum, atoms))
        elif is_metal and on_pref:
            metal_pref.append((resname, chain, resnum, atoms))
        elif is_metal:
            metal_any.append((resname, chain, resnum, atoms))

    for bucket, kind in [(multi_pref, "organic_on_chain"),
                         (multi_any, "organic_other_chain"),
                         (metal_pref, "metal_on_chain"),
                         (metal_any, "metal_other_chain")]:
        if bucket:
            # Pick the largest atom count from this bucket
            bucket.sort(key=lambda b: -len(b[3]))
            r = bucket[0]
            return (r[0], r[1], r[2], r[3], centroid(r[3]), kind)
    return None


def load_engine_sites(binding_sites_json):
    with open(binding_sites_json) as f:
        bs = json.load(f)
    # Schema note: top-level "binding_sites" is an INT count, not a list.
    # The actual site list is under "sites" (rich) or "all_pockets" (basic).
    sites = bs.get("sites") or bs.get("all_pockets") or bs.get("cryptic_sites") or []
    if not isinstance(sites, list):
        sites = []
    out = []
    for i, site in enumerate(sites):
        sc = site.get("centroid") or site.get("center") or site.get("position")
        if sc is None:
            continue
        if isinstance(sc, dict):
            sc = (sc.get("x", 0), sc.get("y", 0), sc.get("z", 0))
        else:
            sc = tuple(sc[:3])
        rank = site.get("rank", i + 1)
        cls = site.get("class") or site.get("therm_class") or "?"
        out.append((rank, i, sc, cls))
    return out


def grade(d):
    if d is None:
        return "N/A"
    if d < 5:
        return "EXCELLENT"
    if d < 8:
        return "GOOD"
    if d < 10:
        return "MARGINAL"
    return "POOR"


def validate_target(target, holo_pdb, binding_sites_json, prefer_chain):
    print(f"\n── {target} ─────────────────────────────────────────────")

    if not Path(holo_pdb).exists():
        print(f"  SKIP: holo PDB {holo_pdb} not found")
        return None
    if not Path(binding_sites_json).exists():
        print(f"  SKIP: binding_sites.json {binding_sites_json} not found")
        return None

    groups = parse_ligand_atoms(holo_pdb)
    if not groups:
        print(f"  SKIP: no non-nuisance HETATMs in {holo_pdb} (likely apo)")
        return None

    sel = select_best_ligand(groups, prefer_chain=prefer_chain)
    if sel is None:
        print(f"  SKIP: no usable ligand (only nuisance HETATMs)")
        return None

    resname, chain, resnum, atoms, lig_centroid, kind = sel
    print(f"  Ligand: {resname} on chain {chain} resnum {resnum}, {len(atoms)} atoms ({kind})")
    print(f"  Centroid: ({lig_centroid[0]:.2f}, {lig_centroid[1]:.2f}, {lig_centroid[2]:.2f})")

    sites = load_engine_sites(binding_sites_json)
    if not sites:
        print(f"  SKIP: no engine sites in {binding_sites_json}")
        return None

    print(f"  Engine sites: {len(sites)}")
    by_dist = sorted(
        [(rank, idx, sc, cls, dist(sc, lig_centroid)) for rank, idx, sc, cls in sites],
        key=lambda r: r[4]
    )

    best = by_dist[0]
    rank1 = sorted(sites, key=lambda r: r[0])[0]
    rank1_d = dist(rank1[2], lig_centroid)

    print(f"  Best DCC anywhere : {best[4]:5.2f}A  (engine rank {best[0]}, class {best[3]})")
    print(f"  DCC at rank 1     : {rank1_d:5.2f}A  (class {rank1[3]})")
    print(f"  GRADE             : {grade(best[4])}")

    return {
        "target": target,
        "ligand": resname,
        "ligand_chain": chain,
        "ligand_kind": kind,
        "n_ligand_atoms": len(atoms),
        "n_engine_sites": len(sites),
        "best_dcc": round(best[4], 2),
        "best_dcc_rank": best[0],
        "best_dcc_class": best[3],
        "rank1_dcc": round(rank1_d, 2),
        "rank1_class": rank1[3],
        "grade": grade(best[4]),
    }


TARGETS = [
    # (target_name, pdb_id_lower, prefer_chain)
    ("10dc_chainA", "10dc", "A"),
    ("13sb_chainA", "13sb", "A"),
    ("13sb_chainB", "13sb", "B"),
    ("9y1b_chainA", "9y1b", "A"),
]


def main():
    holo_dir = "/tmp/dcc_validation"
    workdir = "/mnt/storage/prism-outputs/10k-runs"

    print("=" * 70)
    print("DCC VALIDATION — proteome_1000 smoke test")
    print("=" * 70)
    print(f"  Holo PDBs: {holo_dir}/<pdb>_holo.pdb")
    print(f"  Engine outputs: {workdir}/<target>/<target>.binding_sites.json")

    results = []
    for target, pdb, chain in TARGETS:
        holo = f"{holo_dir}/{pdb}_holo.pdb"
        bs = f"{workdir}/{target}/{target}.binding_sites.json"
        r = validate_target(target, holo, bs, chain)
        if r is not None:
            results.append(r)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if not results:
        print("  No results.")
        return 1

    print(f"  {'target':<16} {'ligand':<8} {'kind':<22} {'best_DCC':>9} {'rank1_DCC':>10} {'grade':<10}")
    for r in results:
        print(f"  {r['target']:<16} {r['ligand']:<8} {r['ligand_kind']:<22} {r['best_dcc']:>8.2f}A {r['rank1_dcc']:>9.2f}A {r['grade']:<10}")

    best_dccs = [r["best_dcc"] for r in results]
    n_excellent = sum(1 for d in best_dccs if d < 5)
    n_good = sum(1 for d in best_dccs if 5 <= d < 8)
    n_marginal = sum(1 for d in best_dccs if 8 <= d < 10)
    n_poor = sum(1 for d in best_dccs if d >= 10)
    median = sorted(best_dccs)[len(best_dccs) // 2]

    print(f"\n  N targets validated: {len(results)}/{len(TARGETS)}")
    print(f"  Median best-DCC    : {median:.2f}A")
    print(f"  EXCELLENT (<5A)    : {n_excellent}")
    print(f"  GOOD (5-8A)        : {n_good}")
    print(f"  MARGINAL (8-10A)   : {n_marginal}")
    print(f"  POOR (>10A)        : {n_poor}")

    pass_rate = (n_excellent + n_good) / len(results)
    print(f"\n  Detection pass rate: {pass_rate:.0%}  (target: >=75%)")
    if pass_rate >= 0.75:
        print(f"  VERDICT: PASS — corpus quality is sufficient for downstream training")
    else:
        print(f"  VERDICT: FAIL — investigate before launching full corpus")

    # Persist
    with open(f"{holo_dir}/dcc_results.json", "w") as f:
        json.dump({"results": results,
                   "median_best_dcc": median,
                   "n_excellent": n_excellent,
                   "n_good": n_good,
                   "n_marginal": n_marginal,
                   "n_poor": n_poor,
                   "pass_rate": pass_rate}, f, indent=2)
    print(f"\n  Results JSON: {holo_dir}/dcc_results.json")
    return 0 if pass_rate >= 0.75 else 1


if __name__ == "__main__":
    sys.exit(main())

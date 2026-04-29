#!/usr/bin/env python3
"""
PRISM-4D UNIFIED POCKET DOSSIER — 13-enhancement edition.

Combines:
  - 10 substrate-internal enhancements (E1-E10) from dossier_full.py
  - 3 reference-anchored enhancements (E11-E13) from master script
  - Multi-target Kabsch SVD alignment with verified HET IDs

Per NO_POST_MD_LOOPS directive: this is a Python post-processor, NOT a
Rust-side post-MD loop. Operates on emitted Arrow + emitted PDB + reference
PDBs only.
"""
import os, json, math, sys, urllib.request, warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.compute as pc
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy import stats
from collections import defaultdict
from pathlib import Path
warnings.filterwarnings("ignore")
# Tier 1+2+3 bleeding-edge enhancements
sys.path.insert(0, str(Path(__file__).parent))
from tier1_anchor_hellinger import (
    diagnose_md_anchor,
    load_topology_atom_positions,
    load_residue_to_atom_map,
    compute_pocket_reference_hellinger,
)
from tier2_wasserstein_alignment import wasserstein_align_reference
from tier3_persistent_homology import discover_pockets_persistent_homology
from tier3_geodesic_centroids import compute_geodesic_centroids_for_pockets

# ──────────────────────── PATHS ────────────────────────
# ──────────────────────── AUTO-DISCOVERY ────────────────────────
# Smart path resolution: picks the best Arrow file available, with explicit
# precedence. Multi-seed merged > newest single-seed. Override with env var.
import glob

PRISM_ROOT = Path(os.environ.get(
    "PRISM_ROOT", "/home/diddy/Desktop/Prism4D-bio"
))

def discover_arrow_file():
    """
    Returns (arrow_path, source_label) using this precedence:
      1. PRISM_ARROW environment variable (operator override)
      2. Newest *merged*.arrow file under output/ (multi-seed merger output)
      3. Newest *.spike_events.arrow file under output/ (single-seed canonical)
    Returns (None, "no_arrow_found") if nothing matches.
    """
    # 1. Operator override
    override = os.environ.get("PRISM_ARROW")
    if override and os.path.exists(override):
        return override, f"override:PRISM_ARROW={override}"

    output_root = PRISM_ROOT / "output"
    if not output_root.exists():
        return None, f"no_output_dir:{output_root}"

    # 2. Multi-seed merged files (highest precedence). Patterns allow the
    # merge marker anywhere in the basename — actual emitted name is e.g.
    # `4lpk_clean.topology.merged_seeds_42_43_44.arrow`.
    merged_patterns = [
        "**/*merged_seeds*.arrow",
        "**/*multi_seed*.arrow",
        "**/*_merged.arrow",
    ]
    merged = []
    for pat in merged_patterns:
        merged.extend(glob.glob(str(output_root / pat), recursive=True))
    if merged:
        newest = max(merged, key=os.path.getmtime)
        return newest, f"multi_seed_merged (mtime={int(os.path.getmtime(newest))})"

    # 3. Single-seed canonical spike_events files
    single = glob.glob(str(output_root / "**" / "*.spike_events.arrow"), recursive=True)
    if single:
        # Filter out empty/corrupt files (size > 1MB sanity check)
        single = [p for p in single if os.path.getsize(p) > 1_000_000]
        if single:
            newest = max(single, key=os.path.getmtime)
            return newest, f"single_seed (mtime={int(os.path.getmtime(newest))})"

    return None, "no_valid_arrow_files_found"


def discover_md_pdb(arrow_path):
    """
    Find the MD-output PDB that goes with this Arrow file.
    Search order:
      1. Same directory as Arrow, *.druggability.pdb
      2. Same directory as Arrow, any *.pdb except *_clean.pdb (input PDB)
      3. PRISM_ROOT/output/**/4lpk_clean.topology.druggability.pdb (any run)
    """
    arrow_dir = Path(arrow_path).parent

    # 1. Druggability PDB next to Arrow
    drug_pdb = list(arrow_dir.glob("*.druggability.pdb"))
    if drug_pdb:
        return str(drug_pdb[0]), "co_located_druggability"

    # 2. Any non-input PDB next to Arrow
    pdbs = [p for p in arrow_dir.glob("*.pdb")
            if "input" not in p.name.lower()]
    if pdbs:
        newest = max(pdbs, key=lambda p: p.stat().st_mtime)
        return str(newest), "co_located_other_pdb"

    # 3. Cross-run search for druggability PDB
    cross = glob.glob(str(PRISM_ROOT / "output" / "**" / "*.druggability.pdb"),
                      recursive=True)
    if cross:
        newest = max(cross, key=os.path.getmtime)
        return newest, f"cross_run_druggability (mtime={int(os.path.getmtime(newest))})"

    return None, "no_md_pdb_found"


def discover_out_dir(arrow_path):
    """Output dir defaults to same directory as the Arrow file."""
    return str(Path(arrow_path).parent)


# Resolve paths
ARROW_PATH, ARROW_SOURCE = discover_arrow_file()
if ARROW_PATH is None:
    print(f"[!] FATAL: no Arrow file discovered. Reason: {ARROW_SOURCE}")
    print(f"    Searched under: {PRISM_ROOT}/output/")
    print(f"    Override with: PRISM_ARROW=/path/to/file.arrow python3 dossier_unified.py")
    sys.exit(1)

MD_PDB, MD_PDB_SOURCE = discover_md_pdb(ARROW_PATH)
OUT_DIR = discover_out_dir(ARROW_PATH)
REF_DIR = PRISM_ROOT / "pdb_refs"
REF_DIR.mkdir(exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 100)
print("PRISM-4D UNIFIED DOSSIER — auto-discovered paths")
print("=" * 100)
print(f"  Arrow file:    {ARROW_PATH}")
print(f"  Source:        {ARROW_SOURCE}")
print(f"  MD PDB:        {MD_PDB or '(none — E13 will be BLOCKED)'}")
print(f"  PDB source:    {MD_PDB_SOURCE}")
print(f"  Output dir:    {OUT_DIR}")
print(f"  Reference dir: {REF_DIR}")
print("=" * 100)

# ═══════════════════════════════════════════════════════════════════════
# TARGET CONFIG LOADING (replaces hardcoded KRAS DOMAINS + REFERENCES)
# ═══════════════════════════════════════════════════════════════════════
sys.path.insert(0, str(Path(__file__).parent))
from target_config import (
    derive_target_id,
    load_target_config,
    get_substrate_only_blocked_message,
)

TARGET_ID = derive_target_id(ARROW_PATH)
TARGET_CFG = load_target_config(TARGET_ID)
SUBSTRATE_ONLY_MODE = (
    not TARGET_CFG["found"] or not TARGET_CFG["references"]
)

print("=" * 100)
print("TARGET CONFIG")
print("=" * 100)
print(f"  target_id:        {TARGET_ID}")
print(f"  config_found:     {TARGET_CFG['found']}")
if TARGET_CFG["found"]:
    print(f"  target_name:      {TARGET_CFG['target_name']}")
    print(f"  protein_class:    {TARGET_CFG['protein_class']}")
    print(f"  chain_id:         {TARGET_CFG['chain_id']}")
    print(f"  canonical regions: {list(TARGET_CFG['domains'].keys())}")
    print(f"  references:       "
          f"{list(TARGET_CFG['references'].keys()) or '(none — substrate-only)'}")
else:
    print(f"  WARNING: no config file at config/targets/{TARGET_ID}.yaml")
    print(f"  Running in substrate-only mode.")
    print(f"  E11/E15/E16 will be BLOCKED with target_config gate.")
print(f"  substrate_only_mode: {SUBSTRATE_ONLY_MODE}")
print("=" * 100)

# Target-specific config (was hardcoded for KRAS).
REFERENCES = TARGET_CFG["references"]   # may be {} for substrate-only
CHAIN_ID   = TARGET_CFG["chain_id"]

CLEAN_COLS = [
    "spike_id", "replica_seed", "stream_id", "group_id", "chunk_idx",
    "voxel_idx", "timestep", "frame_index",
    "x", "y", "z",
    "intensity", "spike_source", "aromatic_type", "aromatic_residue_id",
    "phase_bits", "n_residues", "nearby_residues", "n_nearby_excited",
    "vibrational_energy", "water_density", "wd_change", "wavelength_nm",
    "ccns_phase", "background_class", "intensity_percentile",
]

DOMAINS = TARGET_CFG["domains"]   # was hardcoded for KRAS

# Merged clustering: dossier's TOP_RES=150 with master's r=6.5
TOP_RES_FOR_CLUSTERING = 150
KDTREE_RADII           = [5.0, 6.0, 6.5, 7.0]
CANONICAL_RADIUS       = 6.5
MIN_CLUSTER_SIZE       = 3
TOP_K_POCKETS          = 10

CCNS_PHASE_NAMES = {0: "cold_hold", 1: "ramp", 2: "warm_hold"}

# ═══════════════════════════════════════════════════════════════════════
# STAGE 1: PDB PARSER (robust, handles multi-chain, alt-loc)
# ═══════════════════════════════════════════════════════════════════════
def parse_pdb_ca_and_het(pdb_path, target_hets=None, chain_filter="A"):
    """
    Returns (ca_dict, het_dict).
    ca_dict[resseq] = np.array([x,y,z])  for chain A CA atoms (alt_loc A or blank)
    het_dict[het_id] = np.array of all heavy-atom coords for that ligand
    """
    ca = {}
    het_coords = defaultdict(list)
    if not os.path.exists(pdb_path):
        return ca, het_coords
    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                atom_name = line[12:16].strip()
                alt_loc   = line[16:17].strip()
                res_name  = line[17:20].strip()
                chain     = line[21:22].strip()
                resseq    = int(line[22:26])
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                element   = line[76:78].strip() if len(line) > 76 else atom_name[0]
            except (ValueError, IndexError):
                continue

            if element == "H" or atom_name.startswith("H"):
                continue
            if alt_loc not in ("", "A"):
                continue

            if line.startswith("ATOM") and atom_name == "CA" and chain == chain_filter:
                if resseq not in ca:
                    ca[resseq] = np.array([x, y, z])
            elif line.startswith("HETATM") and target_hets and res_name in target_hets:
                het_coords[res_name].append([x, y, z])

    return ca, {k: np.array(v) for k, v in het_coords.items()}


# ═══════════════════════════════════════════════════════════════════════
# TOPOLOGY JSON DISCOVERY (paired with Arrow file for G4 honesty)
# ═══════════════════════════════════════════════════════════════════════
def discover_topology_json(arrow_path):
    """
    Find the topology JSON paired with the Arrow file.
    Search precedence (prefers populated residue_to_atom_indices):
      1. Co-located *.topology.json with populated residue_to_atom_indices
      2. PRISM_ROOT/<target_stem>.topology.json (prism-prep convention, populated)
      3. PRISM_ROOT/<short_target>.topology.json (campaign-style, populated)
      4. Co-located *.topology.json (fallback, even if empty)
      5. None
    """
    arrow_dir = Path(arrow_path).parent
    arrow_basename = Path(arrow_path).name
    # Robust stem extraction — handles both single-seed
    # `<stem>.topology.spike_events.arrow` and merged
    # `<stem>.topology.merged_seeds_42_43_44.arrow`.
    if ".topology" in arrow_basename:
        target_stem = arrow_basename.split(".topology", 1)[0]
    else:
        target_stem = arrow_basename.removesuffix(".arrow")

    def _populated(path):
        try:
            with open(path) as f:
                t = json.load(f)
            return len(t.get("residue_to_atom_indices", {})) > 0
        except (json.JSONDecodeError, OSError):
            return False

    # 1. Co-located, populated
    for c in arrow_dir.glob("*.topology.json"):
        if _populated(c):
            return str(c), "co_located_populated"

    # 2. PRISM_ROOT/<target_stem>.topology.json
    prep_target = PRISM_ROOT / f"{target_stem}.topology.json"
    if prep_target.exists() and _populated(prep_target):
        return str(prep_target), "prism_prep_root_populated"

    # 3. PRISM_ROOT/<short>.topology.json (drop _clean and _chainX suffix)
    short_target = target_stem.replace("_clean", "").split("_chain")[0]
    short_path = PRISM_ROOT / f"{short_target}.topology.json"
    if short_path.exists() and _populated(short_path):
        return str(short_path), "prism_prep_root_short_populated"

    # 4. Co-located fallback (empty residue_to_atom_indices is still a topology JSON)
    for c in arrow_dir.glob("*.topology.json"):
        return str(c), "co_located_empty_fallback"

    return None, "no_topology_json_found"


def _g4_check(topo_path):
    """G4 PASS only if topology JSON has populated residue_to_atom_indices map."""
    if not topo_path or not os.path.exists(topo_path):
        return False
    try:
        with open(topo_path) as f:
            topo = json.load(f)
        return len(topo.get("residue_to_atom_indices", {})) > 0
    except (json.JSONDecodeError, OSError):
        return False


TOPOLOGY_PATH, TOPOLOGY_SOURCE = discover_topology_json(ARROW_PATH)
print(f"  Topology JSON: {TOPOLOGY_PATH or '(none — G4 will be BLOCKED)'}")
print(f"  Topo source:   {TOPOLOGY_SOURCE}")
print("=" * 100)


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: LOAD ARROW + ACTIVE-AXIS DETECTION
# ═══════════════════════════════════════════════════════════════════════
if not os.path.exists(ARROW_PATH):
    print(f"[!] Arrow not found: {ARROW_PATH}"); sys.exit(1)

with pa.memory_map(ARROW_PATH, "r") as src:
    table = ipc.open_file(src).read_all().select(CLEAN_COLS)

mask = pc.and_(pc.equal(table["background_class"], 0),
               pc.greater_equal(table["intensity_percentile"], 70))
ft = table.filter(mask)
n_spikes = ft.num_rows

seeds_in_run    = sorted(set(ft["replica_seed"].to_pylist()))
streams_in_run  = sorted(set(ft["stream_id"].to_pylist()))
ccns_in_run     = sorted(set(ft["ccns_phase"].to_pylist()))
sources_in_run  = sorted(set(ft["spike_source"].to_pylist()))
frames_in_run   = (min(ft["frame_index"].to_pylist()), max(ft["frame_index"].to_pylist()))
n_frames        = frames_in_run[1] - frames_in_run[0] + 1

REPLICA_AXIS_ACTIVE = len(seeds_in_run) >= 2
STREAM_AXIS_ACTIVE  = len(streams_in_run) >= 2
CCNS_AXIS_ACTIVE    = len(ccns_in_run) >= 2
SOURCE_AXIS_ACTIVE  = len(sources_in_run) >= 2

GATES = {
    "G1_multi_seed":         REPLICA_AXIS_ACTIVE,
    "G2_trajectory_writer":  bool(MD_PDB) and os.path.exists(MD_PDB.replace(".pdb", ".frames.bin")),
    "G3_phase_bits_decoder": False,
    "G4_residue_atom_map":   _g4_check(TOPOLOGY_PATH),
    "G5_md_pdb_anchor":      bool(MD_PDB) and os.path.exists(MD_PDB),
}

run_metadata = {
    "filtered_spikes": n_spikes, "replica_seeds": seeds_in_run,
    "stream_ids": streams_in_run, "ccns_phases": ccns_in_run,
    "spike_sources": sources_in_run, "frame_range": list(frames_in_run),
    "n_frames": n_frames, "axes": {
        "replica_active": REPLICA_AXIS_ACTIVE, "stream_active": STREAM_AXIS_ACTIVE,
        "ccns_active": CCNS_AXIS_ACTIVE, "source_active": SOURCE_AXIS_ACTIVE},
    "dev_gates": GATES, "md_pdb_path": MD_PDB, "topology_path": TOPOLOGY_PATH,
}

print(f"\n[1] Filtered spikes: {n_spikes:,}")
print(f"    Seeds {seeds_in_run} [{'ACTIVE' if REPLICA_AXIS_ACTIVE else 'BLOCKED G1'}]  "
      f"Streams {streams_in_run} [{'ACTIVE' if STREAM_AXIS_ACTIVE else 'INACTIVE'}]")
print(f"    CCNS {ccns_in_run} [{'ACTIVE' if CCNS_AXIS_ACTIVE else 'INACTIVE'}]  "
      f"Sources {sources_in_run} [{'ACTIVE' if SOURCE_AXIS_ACTIVE else 'INACTIVE'}]")
print(f"    Frames {frames_in_run[0]}-{frames_in_run[1]} ({n_frames} frames)")
print(f"\n    DEV GATES:")
for g, st in GATES.items():
    print(f"      {g:<28} {'PASS' if st else 'BLOCKED'}")
    
# ═══════════════════════════════════════════════════════════════════════
# STAGE 3: EXPLODE + AGGREGATE
# ═══════════════════════════════════════════════════════════════════════
spike_idx = pa.array(np.repeat(np.arange(n_spikes), 8))
flat_res  = pc.list_flatten(ft["nearby_residues"])
SCALARS = ["intensity","x","y","z","vibrational_energy","water_density","wd_change",
           "wavelength_nm","ccns_phase","phase_bits","stream_id","spike_source",
           "aromatic_type","aromatic_residue_id","frame_index","replica_seed",
           "n_nearby_excited","voxel_idx"]
cols = {"residue_id": flat_res, "spike_idx": spike_idx}
for c in SCALARS:
    cols[c] = pc.take(ft[c], spike_idx)
ex = pa.Table.from_pydict(cols)
ex = ex.filter(pc.not_equal(ex["residue_id"], -1))
df = ex.to_pandas()
for c in ("x","y","z"):
    df[f"w{c}"] = df[c] * df["intensity"]

print(f"\n[2] Exploded: {len(df):,} rows, {df['residue_id'].nunique()} unique residues")

def is_arom(g): return int((g > 0).any())
def arom_type(g):
    g2 = g[g > 0]
    return int(g2.mode().iloc[0]) if len(g2) else 0

res = df.groupby("residue_id").agg(
    spike_count=("intensity","count"), total_energy=("intensity","sum"),
    sum_wx=("wx","sum"), sum_wy=("wy","sum"), sum_wz=("wz","sum"),
    mean_vib=("vibrational_energy","mean"), mean_wd=("water_density","mean"),
    abs_wd_change=("wd_change", lambda s: float(np.mean(np.abs(s)))),
    max_abs_wd_change=("wd_change", lambda s: float(np.max(np.abs(s)))),
    signed_wd_change=("wd_change","mean"),
    mean_wl=("wavelength_nm","mean"),
    n_streams=("stream_id","nunique"), n_phases=("ccns_phase","nunique"),
    n_replicas=("replica_seed","nunique"), n_frames_seen=("frame_index","nunique"),
    min_frame=("frame_index","min"),
    is_aromatic=("aromatic_type", is_arom), aromatic_type=("aromatic_type", arom_type),
    n_sources=("spike_source","nunique"),
    mean_n_excited=("n_nearby_excited","mean"),
    var_n_excited=("n_nearby_excited","var"),
    heating_E=("intensity", lambda x: x[df.loc[x.index, "ccns_phase"] == 1].sum()),
).reset_index()
res["mean_x"] = res["sum_wx"] / res["total_energy"]
res["mean_y"] = res["sum_wy"] / res["total_energy"]
res["mean_z"] = res["sum_wz"] / res["total_energy"]
res = res.sort_values("total_energy", ascending=False).reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════════
# STAGE 4: KD-TREE CLUSTERING + RADIUS STABILITY SWEEP
# ═══════════════════════════════════════════════════════════════════════
def cluster(top_residues, radius, min_size, top_k):
    tree = KDTree(top_residues[["mean_x","mean_y","mean_z"]].values)
    nbrs = tree.query_ball_tree(tree, r=radius)
    visited, pkts = set(), []
    for i, ns in enumerate(nbrs):
        if i in visited: continue
        c = []
        for j in ns:
            if j not in visited:
                c.append(int(top_residues.iloc[j]["residue_id"]))
                visited.add(j)
        if len(c) >= min_size:
            e = res.loc[res["residue_id"].isin(c), "total_energy"].sum()
            pkts.append((c, float(e)))
    pkts.sort(key=lambda p: p[1], reverse=True)
    return pkts[:top_k]

top = res.head(TOP_RES_FOR_CLUSTERING).copy()
stability = {r: cluster(top, r, MIN_CLUSTER_SIZE, TOP_K_POCKETS) for r in KDTREE_RADII}
pockets = stability[CANONICAL_RADIUS]
print(f"\n[3] Clustering at r={CANONICAL_RADIUS}A: {len(pockets)} pockets")
for r, p in stability.items():
    print(f"    r={r}A: {len(p)} pockets, mean size {np.mean([len(pp[0]) for pp in p]):.1f}")

# ═══════════════════════════════════════════════════════════════════════
# STAGE 5: MULTI-TARGET KABSCH ALIGNMENT (E11 INFRASTRUCTURE)
# ═══════════════════════════════════════════════════════════════════════
print(f"\n[4] Multi-target Kabsch SVD alignment to MD-output PDB")
md_ca, _ = parse_pdb_ca_and_het(MD_PDB, target_hets=None, chain_filter=CHAIN_ID)
print(f"    MD PDB: {len(md_ca)} CA atoms in chain {CHAIN_ID}")

aligned_ligands = {}
alignment_diagnostics = {}

if not REFERENCES:
    print(f"    SKIPPED: no references configured for target_id={TARGET_ID} "
          f"(substrate-only mode)")

target_hets = {r["het"] for r in REFERENCES.values()} if REFERENCES else set()
for pdb_id, meta in REFERENCES.items():
    pdb_path = REF_DIR / f"{pdb_id}.pdb"
    if not pdb_path.exists():
        try:
            urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.pdb", pdb_path)
        except Exception as e:
            print(f"    [DOWNLOAD FAIL] {pdb_id}: {e}")
            continue

    h_ca, h_het_dict = parse_pdb_ca_and_het(pdb_path, target_hets=target_hets, chain_filter=CHAIN_ID)
    h_lig = h_het_dict.get(meta["het"], np.array([]))
    if h_lig.size == 0 or not h_ca:
        print(f"    [SKIP] {pdb_id}: no ligand or no CA. h_lig={len(h_lig) if h_lig.size else 0}, h_ca={len(h_ca)}")
        continue

    # Find optimal residue offset (handles numbering shifts)
    best_off, max_match = 0, 0
    for off in range(-10, 11):
        m = len(set(md_ca.keys()) & {k - off for k in h_ca.keys()})
        if m > max_match:
            max_match, best_off = m, off

    common = sorted(set(md_ca.keys()) & {k - best_off for k in h_ca.keys()})
    if len(common) < 30:
        print(f"    [SKIP] {pdb_id}: insufficient overlap ({len(common)} residues)")
        continue

    pm = np.array([md_ca[r] for r in common])
    ph = np.array([h_ca[r + best_off] for r in common])

    cm, ch = pm.mean(axis=0), ph.mean(axis=0)
    H = (ph - ch).T @ (pm - cm)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    aligned_lig = (h_lig - ch) @ R.T + cm
    aligned_ca  = (ph - ch) @ R.T + cm
    rmsd = float(np.sqrt(np.mean(np.sum((aligned_ca - pm) ** 2, axis=1))))

    aligned_ligands[pdb_id] = aligned_lig
    alignment_diagnostics[pdb_id] = {
        "name": meta["name"], "het": meta["het"],
        "n_common_residues": len(common), "residue_offset": best_off,
        "alignment_rmsd": rmsd, "ligand_atoms": int(len(h_lig)),
        "ligand_centroid": aligned_lig.mean(axis=0).tolist(),
    }
    print(f"    [ALIGN] {pdb_id} ({meta['het']}): {len(common)} residues, "
          f"offset={best_off:+d}, RMSD={rmsd:.3f}A, ligand_atoms={len(h_lig)}")

# ═══════════════════════════════════════════════════════════════════════
# STAGE 6: PER-POCKET CORE DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════
def pocket_spikes(rids):
    return df[df["residue_id"].isin(rids)]

def regional_yields(rids):
    pdf = res[res["residue_id"].isin(rids)]
    pE = pdf["total_energy"].sum()
    yields = {n: float(100.0 * pdf[pdf["residue_id"].isin(d)]["total_energy"].sum() / pE) if pE > 0 else 0.0
              for n, d in DOMAINS.items()}
    yields["Other"] = max(0.0, 100.0 - sum(yields.values()))
    return yields

def core_diag(rids):
    pdf = res[res["residue_id"].isin(rids)]
    sdf = pocket_spikes(rids)
    coords = pdf[["mean_x","mean_y","mean_z"]].values
    w = pdf["total_energy"].values
    centroid = np.average(coords, axis=0, weights=w)
    rg = float(np.sqrt(np.average(np.sum((coords-centroid)**2, axis=1), weights=w)))
    seq_gap = float(np.mean(np.diff(sorted(pdf["residue_id"].values)))) if len(pdf) > 1 else 0.0
    wd_var = float(np.var(sdf["water_density"].values)) if len(sdf) > 1 else 0.0
    wd_iqr = float(np.percentile(sdf["water_density"].values, 75) - np.percentile(sdf["water_density"].values, 25)) if len(sdf) else 0.0
    wd_p95 = float(np.percentile(np.abs(sdf["wd_change"].values), 95)) if len(sdf) else 0.0
    vib_var = float(np.var(sdf["vibrational_energy"].values)) if len(sdf) > 1 else 0.0
    wl_var = float(np.var(sdf["wavelength_nm"].values)) if len(sdf) > 1 else 0.0
    ccns_bal = float(sdf["ccns_phase"].value_counts(normalize=True).std()) if len(sdf) else 0.0

    sc_corr = float("nan")
    n_c = len(coords)
    if n_c > 2 and (STREAM_AXIS_ACTIVE or REPLICA_AXIS_ACTIVE):
        d_sp = np.linalg.norm(coords[:,None,:] - coords[None,:,:], axis=-1)
        cs = pdf["n_streams"].values + pdf["n_replicas"].values
        c_diff = np.abs(cs[:,None] - cs[None,:])
        iu = np.triu_indices(n_c, k=1)
        if d_sp[iu].std() > 0 and c_diff[iu].std() > 0:
            sc_corr = float(np.corrcoef(d_sp[iu], c_diff[iu])[0,1])

    return {
        "rg": rg, "seq_gap": seq_gap,
        "wd_var": wd_var, "wd_iqr": wd_iqr, "wd_change_p95": wd_p95,
        "vib_var": vib_var, "wl_var": wl_var, "ccns_balance": ccns_bal,
        "sc_corr": sc_corr,
        "frame_persistence_per_res": float(pdf["n_frames_seen"].mean()),
        "centroid_xyz": centroid.tolist(),
        "n_residues": len(pdf), "n_spikes": len(sdf),
    }

# ═══════════════════════════════════════════════════════════════════════
# E1: CCNS LIFECYCLE + 6-CLASS TAXONOMY
# ═══════════════════════════════════════════════════════════════════════
def E1_ccns(rids):
    if not CCNS_AXIS_ACTIVE:
        return {"status":"BLOCKED","gate":"CCNS axis inactive (single phase)","result":None}
    sdf = pocket_spikes(rids)
    by_phase = {}
    total_e = sdf["intensity"].sum()
    for ph in ccns_in_run:
        pdf = sdf[sdf["ccns_phase"] == ph]
        nm = CCNS_PHASE_NAMES.get(ph, f"phase_{ph}")
        by_phase[nm] = {
            "n_spikes": int(len(pdf)),
            "energy_pct": float(100.0*pdf["intensity"].sum()/total_e) if total_e else 0.0,
            "n_residues": int(pdf["residue_id"].nunique()),
            "wd_change_p95": float(np.percentile(np.abs(pdf["wd_change"].values),95)) if len(pdf) else 0.0,
        }
    e = {k: v["energy_pct"] for k,v in by_phase.items()}
    cold, ramp, warm = e.get("cold_hold",0), e.get("ramp",0), e.get("warm_hold",0)
    if cold > 60: tax = "THERMAL_TRANSIENT"
    elif ramp > 60: tax = "BARRIER_GATED"
    elif warm > 60: tax = "NMA_RESPONSIVE"
    elif abs(cold-warm) < 15 and ramp > 25: tax = "CONSENSUS_CRYPTIC"
    elif (cold+warm) > 70 and ramp < 20: tax = "ALLOSTERIC_HUB"
    else: tax = "COOPERATIVE_NETWORK"
    return {"status":"OK","by_phase":by_phase,"taxonomy_class":tax}

# ═══════════════════════════════════════════════════════════════════════
# E2: WAVELENGTH MICROENVIRONMENT FINGERPRINT
# ═══════════════════════════════════════════════════════════════════════
def E2_wavelength(rids):
    sdf = pocket_spikes(rids)
    if len(sdf) < 100:
        return {"status":"BLOCKED","gate":"Need >=100 spikes","result":None}
    wl = sdf["wavelength_nm"].values
    bins = np.arange(0, 55, 5)
    hist, _ = np.histogram(wl, bins=bins)
    total = hist.sum()
    pct = (100.0*hist/total).tolist() if total else [0]*len(hist)
    ks_stat, ks_p = stats.kstest((wl-wl.min())/(wl.max()-wl.min()+1e-9), 'uniform')
    res_bins = {}
    for rid in rids:
        rwl = sdf[sdf["residue_id"]==rid]["wavelength_nm"].values
        if len(rwl):
            idx = int(np.argmax(np.histogram(rwl, bins=bins)[0]))
            res_bins[int(rid)] = f"{bins[idx]}-{bins[idx+1]}nm"
    sub = defaultdict(list)
    for rid, b in res_bins.items(): sub[b].append(rid)
    subpockets = {b: g for b, g in sub.items() if len(g) >= 2}
    return {"status":"OK","histogram_pct":pct,"bin_edges_nm":bins.tolist(),
            "bimodality_index":float(ks_stat),"ks_p_value":float(ks_p),
            "n_chemical_subpockets":len(subpockets),"chemical_subpockets":subpockets,
            "interpretation":"BIMODAL_HETEROGENEOUS" if ks_stat > 0.15 else "UNIMODAL_HOMOGENEOUS"}

# ═══════════════════════════════════════════════════════════════════════
# E3: PHASE_BITS MUTUAL INFORMATION (gated)
# ═══════════════════════════════════════════════════════════════════════
def E3_phase_bits(rids):
    if not GATES["G3_phase_bits_decoder"]:
        return {"status":"BLOCKED",
                "gate":"G3: phase_bits 10-bit semantic mapping required (see CLAUDE_CODE_PROMPT.md)",
                "result":None}
    return {"status":"OK","note":"placeholder until decoder lands"}

# ═══════════════════════════════════════════════════════════════════════
# E4: FIRST-PASSAGE TIMING
# ═══════════════════════════════════════════════════════════════════════
def E4_first_passage(rids):
    if n_frames < 5:
        return {"status":"BLOCKED","gate":f"Need >=5 frames, have {n_frames}","result":None}
    sdf = pocket_spikes(rids)
    fp = {}
    for rid in rids:
        rdf = sdf[sdf["residue_id"]==rid]
        if len(rdf): fp[int(rid)] = int(rdf["frame_index"].min())
    if len(fp) < 2:
        return {"status":"BLOCKED","gate":"Need >=2 residues with spikes","result":None}
    sorted_fp = sorted(fp.items(), key=lambda x: x[1])
    nucleation = [r for r,f in sorted_fp if f == sorted_fp[0][1]]
    spread = sorted_fp[-1][1] - sorted_fp[0][1]
    rng = np.random.default_rng(42)
    null = [rng.integers(0, n_frames+1, len(fp)).ptp() for _ in range(1000)]
    p_val = float(np.mean([s >= spread for s in null]))
    return {"status":"OK","first_passage_per_residue":fp,
            "nucleation_residues":nucleation,
            "propagation_order":[r for r,_ in sorted_fp],
            "spread_frames":int(spread),"ordering_p_value":p_val,
            "interpretation":"STRUCTURED_NUCLEATION" if p_val < 0.05 else "STOCHASTIC_FIRING"}

# ═══════════════════════════════════════════════════════════════════════
# E5: AROMATIC GRAPH
# ═══════════════════════════════════════════════════════════════════════
def E5_aromatic(rids, all_pockets):
    sdf = pocket_spikes(rids)
    arom = sdf[sdf["aromatic_residue_id"] >= 0]
    if len(arom) == 0:
        return {"status":"OK","n_internal":0,"n_external":0,"interpretation":"NO_AROMATIC_LINKS"}
    rids_set = set(rids)
    internal = arom[arom["aromatic_residue_id"].isin(rids_set)]
    external = arom[~arom["aromatic_residue_id"].isin(rids_set)]
    ext_targets = external["aromatic_residue_id"].value_counts().head(10).to_dict()
    cross = defaultdict(int)
    for tgt, cnt in ext_targets.items():
        for pidx, prids in all_pockets.items():
            if int(tgt) in prids: cross[f"pocket_{pidx}"] += int(cnt)
    return {"status":"OK","n_internal":int(len(internal)),"n_external":int(len(external)),
            "external_targets_top10":{int(k):int(v) for k,v in ext_targets.items()},
            "cross_pocket_coupling":dict(cross),
            "interpretation":"ALLOSTERIC_COUPLED" if cross else "ISOLATED_AROMATIC_NETWORK"}

# ═══════════════════════════════════════════════════════════════════════
# E6: COOPERATIVE EXCITATION
# ═══════════════════════════════════════════════════════════════════════
def E6_coop(rids):
    sdf = pocket_spikes(rids)
    nx = sdf["n_nearby_excited"].values
    if len(nx) < 100:
        return {"status":"BLOCKED","gate":"Need >=100 spikes","result":None}
    mean_nx = float(np.mean(nx)); var_nx = float(np.var(nx))
    cv = float(np.sqrt(var_nx)/mean_nx) if mean_nx > 0 else float("inf")
    bins = np.arange(0, int(nx.max())+2)
    hist, _ = np.histogram(nx, bins=bins)
    cumul = np.cumsum(hist)/hist.sum() if hist.sum() else hist*0
    valid = (cumul > 0.05) & (cumul < 0.95)
    hill = float("nan")
    if valid.sum() >= 3:
        x_log = np.log(bins[:-1][valid]+1)
        y_log = np.log(cumul[valid]/(1-cumul[valid]+1e-9))
        if np.isfinite(x_log).all() and np.isfinite(y_log).all():
            hill = float(np.polyfit(x_log, y_log, 1)[0])
    if mean_nx > 3 and cv < 0.3: interp = "TIGHTLY_COOPERATIVE"
    elif mean_nx > 3 and cv > 0.6: interp = "BURSTY_BISTABLE"
    elif mean_nx < 1.5: interp = "INDEPENDENT_FIRING"
    else: interp = "MIXED_COOPERATIVITY"
    return {"status":"OK","mean_n_excited":mean_nx,"cv":cv,"hill_n":hill,
            "interpretation":interp}

# ═══════════════════════════════════════════════════════════════════════
# E7: SOURCE CONSENSUS
# ═══════════════════════════════════════════════════════════════════════
def E7_source(rids):
    if not SOURCE_AXIS_ACTIVE:
        return {"status":"BLOCKED","gate":"Need >=2 spike_source values","result":None}
    sdf = pocket_spikes(rids)
    by_src = {}
    total = sdf["intensity"].sum()
    for s in sources_in_run:
        sd = sdf[sdf["spike_source"]==s]
        by_src[int(s)] = {"n_spikes":int(len(sd)),
                          "energy_pct":float(100.0*sd["intensity"].sum()/total) if total else 0.0}
    pcts = [v["energy_pct"] for v in by_src.values()]
    purity = max(pcts)/100.0 if pcts else 0.0
    survives = all(v["n_spikes"] > 50 for v in by_src.values())
    if purity > 0.85 and not survives: interp = "ARTIFACT_RISK"
    elif survives and purity < 0.6: interp = "MULTI_SOURCE_CONFIRMED"
    else: interp = "SOURCE_CONCENTRATED"
    return {"status":"OK","by_source":by_src,"source_purity":purity,
            "survives_all_sources":survives,"interpretation":interp}

# ═══════════════════════════════════════════════════════════════════════
# E8: VOXEL ADJACENCY
# ═══════════════════════════════════════════════════════════════════════
def build_voxel_matrix(all_pockets):
    pv = {idx: set(pocket_spikes(rids)["voxel_idx"].values.tolist())
          for idx, rids in all_pockets.items()}
    pids = sorted(pv.keys())
    M = {}
    for i, pi in enumerate(pids):
        for pj in pids[i+1:]:
            vi, vj = pv[pi], pv[pj]
            if not vi or not vj: continue
            inter = len(vi & vj); uni = len(vi | vj)
            M[f"P{pi}-P{pj}"] = {"jaccard": inter/uni if uni else 0.0,
                                 "shared_voxels": inter}
    return M, pv

def E8_voxel(pidx, vmat, pv):
    related = []
    for k, v in vmat.items():
        if k.startswith(f"P{pidx}-") or k.endswith(f"-P{pidx}"):
            related.append({"pair": k, **v})
    risk = [r for r in related if r["jaccard"] > 0.3]
    indep = [r for r in related if r["jaccard"] < 0.05]
    interp = "MEGACLUSTER_RISK" if risk else "INDEPENDENT_SITE" if len(indep)==len(related) else "ADJACENT"
    return {"status":"OK","n_voxels":len(pv.get(pidx,set())),"pairs":related,
            "megacluster_risk_pairs":risk,"interpretation":interp}

# ═══════════════════════════════════════════════════════════════════════
# E9: WD_CHANGE DIRECTIONAL DECOMPOSITION
# ═══════════════════════════════════════════════════════════════════════
def E9_wd_dir(rids):
    sdf = pocket_spikes(rids)
    wd = sdf["wd_change"].values
    if len(wd) < 100:
        return {"status":"BLOCKED","gate":"Need >=100 spikes","result":None}
    pos_mass = float(wd[wd>0].sum()); neg_mass = float(-wd[wd<0].sum())
    total = pos_mass + neg_mass
    pos_frac = pos_mass/total if total else 0.5
    neg_frac = neg_mass/total if total else 0.5
    purity = max(pos_frac, neg_frac)
    direction = "WATER_EXPELLING" if neg_frac > 0.6 else "WATER_ATTRACTING" if pos_frac > 0.6 else "MIXED"
    if direction == "WATER_EXPELLING" and purity > 0.65: interp = "DRUGGABLE_CRYPTIC_CLEFT"
    elif direction == "WATER_ATTRACTING" and purity > 0.65: interp = "ALLOSTERIC_SOLVATION"
    else: interp = "AMBIGUOUS_HYDRATION"
    return {"status":"OK","signed_mean":float(np.mean(wd)),
            "positive_mass":pos_mass,"negative_mass":neg_mass,
            "directional_purity":purity,"direction":direction,"interpretation":interp}

# ═══════════════════════════════════════════════════════════════════════
# E10: TENSOR NNTF (run-level, gated)
# ═══════════════════════════════════════════════════════════════════════
def E10_tensor():
    if not GATES["G1_multi_seed"]:
        return {"status":"BLOCKED",
                "gate":"G1: need >=2 replica seeds for tensor decomposition","result":None}
    try: from sklearn.decomposition import NMF
    except ImportError:
        return {"status":"BLOCKED","gate":"sklearn NMF missing","result":None}
    n_st = len(streams_in_run); n_ph = len(ccns_in_run); n_sd = len(seeds_in_run)
    residues = sorted(res["residue_id"].unique())
    T = np.zeros((len(residues), n_st*n_ph*n_sd))
    for i, rid in enumerate(residues):
        rdf = df[df["residue_id"]==rid]
        for si, s in enumerate(streams_in_run):
            for pi, p in enumerate(ccns_in_run):
                for di, d in enumerate(seeds_in_run):
                    sub = rdf[(rdf["stream_id"]==s)&(rdf["ccns_phase"]==p)&(rdf["replica_seed"]==d)]
                    T[i, si*n_ph*n_sd + pi*n_sd + di] = sub["intensity"].sum()
    nc = min(10, len(residues)//5)
    nmf = NMF(n_components=nc, init="nndsvd", random_state=42, max_iter=500)
    W = nmf.fit_transform(T); H = nmf.components_
    comps = []
    for k in range(nc):
        idx = np.argsort(W[:,k])[-15:][::-1]
        comps.append({"component_id":k,"top_residues":[int(residues[i]) for i in idx],
                      "energy_share":float(W[:,k].sum()*H[k,:].sum()/(W.sum()*H.sum()+1e-12))})
    return {"status":"OK","n_components":nc,
            "reconstruction_error":float(nmf.reconstruction_err_),
            "components":comps}

# ═══════════════════════════════════════════════════════════════════════
# E11: MULTI-VIEW DCC AGAINST 4 REFERENCE LIGANDS
# ═══════════════════════════════════════════════════════════════════════
def E11_multi_dcc(rids):
    if SUBSTRATE_ONLY_MODE or not aligned_ligands:
        return {
            "status": "BLOCKED",
            "gate": (get_substrate_only_blocked_message(TARGET_ID)
                     if SUBSTRATE_ONLY_MODE
                     else "No reference ligands aligned"),
            "result": None,
        }
    pdf = res[res["residue_id"].isin(rids)]
    coords = pdf[["mean_x","mean_y","mean_z"]].values
    w = pdf["total_energy"].values
    # 4 manifold views
    geometric = coords.mean(axis=0)
    lining = np.average(coords, axis=0, weights=w)  # energy-weighted
    top3 = pdf.nlargest(3, "total_energy")
    driver = top3[["mean_x","mean_y","mean_z"]].mean().values

    per_ref = {}
    for pdb_id, lig_xyz in aligned_ligands.items():
        lig_centroid = lig_xyz.mean(axis=0)
        dccs = {
            "geometric": float(np.linalg.norm(geometric - lig_centroid)),
            "lining":    float(np.linalg.norm(lining - lig_centroid)),
            "driver":    float(np.linalg.norm(driver - lig_centroid)),
        }
        # Ligand-adjacent view
        d_to_lig = np.linalg.norm(coords - lig_centroid, axis=1)
        near_mask = d_to_lig < 6.0
        if near_mask.any():
            ligand_adj = coords[near_mask].mean(axis=0)
            dccs["ligand_adjacent"] = float(np.linalg.norm(ligand_adj - lig_centroid))
        # Min-distance from any pocket residue centroid to any ligand atom
        min_prox = float(cdist(coords, lig_xyz).min())

        best_view = min(dccs, key=dccs.get)
        collapse_delta = dccs["geometric"] - dccs[best_view]

        per_ref[pdb_id] = {
            "drug": REFERENCES[pdb_id]["name"], "het": REFERENCES[pdb_id]["het"],
            "site": REFERENCES[pdb_id]["site"],
            "dcc_per_view": dccs, "best_view": best_view, "best_dcc": dccs[best_view],
            "geometric_collapse_delta": collapse_delta,
            "min_residue_to_ligand_atom": min_prox,
        }
    # Best reference overall
    best_ref = min(per_ref, key=lambda k: per_ref[k]["best_dcc"])
    return {"status":"OK","per_reference":per_ref,
            "best_reference":best_ref,
            "best_reference_dcc":per_ref[best_ref]["best_dcc"],
            "best_reference_min_prox":per_ref[best_ref]["min_residue_to_ligand_atom"],
            "ligand_proximal_residues":sorted([int(r) for r in pdf[
                cdist(coords, aligned_ligands[best_ref]).min(axis=1) < 5.0
            ]["residue_id"].values]),
           }

# ═══════════════════════════════════════════════════════════════════════
# E12: CCNS PHASE ENERGY DISTRIBUTION (kinetic_lead generalized)
# ═══════════════════════════════════════════════════════════════════════
def E12_phase_energy(rids):
    if not CCNS_AXIS_ACTIVE:
        return {"status":"BLOCKED","gate":"CCNS axis inactive","result":None}
    sdf = pocket_spikes(rids)
    total = sdf["intensity"].sum()
    if total == 0:
        return {"status":"BLOCKED","gate":"Zero pocket energy","result":None}
    fractions = {}
    for ph in ccns_in_run:
        ph_e = sdf[sdf["ccns_phase"]==ph]["intensity"].sum()
        fractions[CCNS_PHASE_NAMES.get(ph, f"phase_{ph}")] = float(ph_e/total)
    kinetic_lead = fractions.get("ramp", 0.0)  # master script's metric
    cold = fractions.get("cold_hold", 0.0)
    warm = fractions.get("warm_hold", 0.0)
    persistence = float(min(cold, warm))  # signal in BOTH cold and warm = stable
    if kinetic_lead > 0.5: kinetic_class = "RAMP_DOMINATED"
    elif persistence > 0.3: kinetic_class = "BISTABLE_PERSISTENT"
    elif cold > 0.6: kinetic_class = "COLD_LOCKED"
    elif warm > 0.6: kinetic_class = "WARM_INDUCED"
    else: kinetic_class = "PHASE_DISTRIBUTED"
    return {"status":"OK","phase_fractions":fractions,
            "kinetic_lead":kinetic_lead,
            "bistable_persistence":persistence,
            "kinetic_class":kinetic_class}

# ═══════════════════════════════════════════════════════════════════════
# E13: MD-PDB COORDINATE ANCHOR
# ═══════════════════════════════════════════════════════════════════════
def E13_md_anchor(rids):
    if not GATES["G5_md_pdb_anchor"]:
        return {"status":"BLOCKED","gate":f"MD PDB not at {MD_PDB}","result":None}
    if not md_ca:
        return {"status":"BLOCKED","gate":"MD PDB has no CA atoms in chain A","result":None}
    pdf = res[res["residue_id"].isin(rids)]
    coords = pdf[["mean_x","mean_y","mean_z"]].values
    # Match each pocket residue to its CA in MD PDB (engine-id == PDB resseq for this run)
    matched = {}
    for _, row in pdf.iterrows():
        rid = int(row["residue_id"])
        if rid in md_ca:
            ca_xyz = md_ca[rid]
            spike_centroid = np.array([row["mean_x"], row["mean_y"], row["mean_z"]])
            offset = float(np.linalg.norm(ca_xyz - spike_centroid))
            matched[rid] = {"ca_xyz": ca_xyz.tolist(),
                            "spike_centroid": spike_centroid.tolist(),
                            "offset_A": offset}
    if not matched:
        return {"status":"BLOCKED","gate":"No pocket residues match MD PDB residue IDs","result":None}
    offsets = [m["offset_A"] for m in matched.values()]
    return {"status":"OK","n_anchored_residues":len(matched),
            "mean_spike_to_ca_offset_A":float(np.mean(offsets)),
            "max_offset_A":float(np.max(offsets)),
            "matched_residues":matched,
            "interpretation":("TIGHT_ANCHOR" if np.mean(offsets) < 3.0
                              else "LOOSE_ANCHOR" if np.mean(offsets) < 6.0
                              else "DRIFT_DETECTED")}
# ═══════════════════════════════════════════════════════════════════════
# TIER 1+2+3 PRE-LOOP SETUP — anchor / topology / reference offsets / PH / geodesic
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("TIER 1+2+3 PRE-LOOP SETUP — anchor / topology / reference offsets / PH / geodesic")
print("=" * 100)

# Load MD heavy-atom positions (for E14/E18)
md_atom_positions = load_topology_atom_positions(MD_PDB) if MD_PDB else {}
print(f"  MD heavy-atom residues:           {len(md_atom_positions)}")

# Load residue_to_atom_indices (for E14)
residue_to_atom_map = load_residue_to_atom_map(TOPOLOGY_PATH) if TOPOLOGY_PATH else {}
print(f"  residue_to_atom_indices keys:     {len(residue_to_atom_map)}")

# Per-residue spike centroids (for E14)
spike_centroids_per_res = {
    int(row["residue_id"]): np.array([row["mean_x"], row["mean_y"], row["mean_z"]])
    for _, row in res.iterrows()
}

# Universe of all residues (for E15 distribution support)
all_residues = sorted(res["residue_id"].unique().tolist())
print(f"  all_residues universe:            {len(all_residues)}")

# Per-reference residue offset (from Stage-5 Kabsch alignment_diagnostics)
residue_offset_per_ref = {
    pdb_id: int(d.get("residue_offset", 0))
    for pdb_id, d in alignment_diagnostics.items()
}
print(f"  residue_offset_per_ref:           {residue_offset_per_ref}")

# E18: geodesic centroids — compute for all pockets at once
print(f"  E18 geodesic Fréchet over {len(pockets)} pockets ...")
pocket_residues_dict = {idx: list(rids) for idx, (rids, _) in enumerate(pockets, 1)}
residue_energies_per_pocket = {
    idx: {int(r): float(res.loc[res["residue_id"] == r, "total_energy"].sum())
          for r in rids}
    for idx, rids in pocket_residues_dict.items()
}
geodesic_results = compute_geodesic_centroids_for_pockets(
    pocket_residues_dict, residue_energies_per_pocket, md_atom_positions,
)
if isinstance(geodesic_results, dict) and "status" in geodesic_results and geodesic_results["status"] == "BLOCKED":
    geodesic_results_per_pocket = {idx: geodesic_results for idx in pocket_residues_dict}
else:
    geodesic_results_per_pocket = geodesic_results
n_collapse = sum(1 for v in geodesic_results_per_pocket.values()
                  if isinstance(v, dict) and v.get("interpretation") == "MEGACLUSTER_COLLAPSE_DETECTED")
print(f"    E18: {len(geodesic_results_per_pocket)} centroids, {n_collapse} flagged MEGACLUSTER_COLLAPSE_DETECTED")

# E17: persistent-homology pocket inventory (run-level)
print(f"  E17 persistent-homology inventory ...")
e17_result = discover_pockets_persistent_homology(res, top_n_residues=150, persistence_threshold=0.05)
if e17_result.get("status") == "OK":
    print(f"    E17: status=OK, persistence_components={e17_result.get('n_persistence_components', '?')}")
else:
    print(f"    E17: status={e17_result.get('status')}, gate={e17_result.get('gate', '?')}")
# ═══════════════════════════════════════════════════════════════════════
# STAGE 7: BUILD UNIFIED PER-POCKET DOSSIER
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("PER-POCKET UNIFIED DIAGNOSTICS")
print("=" * 100)

all_pocket_residues = {idx: set(rids) for idx,(rids,_) in enumerate(pockets, 1)}
voxel_matrix, pocket_voxels = build_voxel_matrix(all_pocket_residues)

records = []
for idx, (rids, _) in enumerate(pockets, 1):
    diag = core_diag(rids)
    yields = regional_yields(rids)
    canonical_yield = sum(yields[k] for k in DOMAINS.keys())
    dominant = max(yields, key=yields.get)
    pdf = res[res["residue_id"].isin(rids)]
    pE = float(pdf["total_energy"].sum())
    n_arom = int(pdf["is_aromatic"].sum())

    # Verdict
    real_ev, bridge_ev, axes = 0, 0, []
    if diag["rg"] < 8.0: real_ev += 1; axes.append("compact")
    if diag["rg"] > 12.0: bridge_ev += 1
    if diag["seq_gap"] < 15.0: real_ev += 1; axes.append("seq_local")
    if diag["seq_gap"] > 30.0: bridge_ev += 1
    if 5e-6 < diag["wd_var"] < 5e-5: real_ev += 1; axes.append("solvent_coherent")
    if diag["wd_var"] > 1e-4: bridge_ev += 1
    if not math.isnan(diag["sc_corr"]) and diag["sc_corr"] > 0.2: real_ev += 1; axes.append("sc_corr")
    if not math.isnan(diag["sc_corr"]) and diag["sc_corr"] < -0.2: bridge_ev += 1
    if real_ev >= 2 and bridge_ev == 0: verdict = "REAL_POCKET"
    elif bridge_ev >= 2: verdict = "CAUSAL_BRIDGE"
    elif bridge_ev == 1 and real_ev >= 1: verdict = "MIXED"
    elif real_ev == 1 and bridge_ev == 0: verdict = "AMBIGUOUS_LEAN_REAL"
    else: verdict = "AMBIGUOUS"

    # E14: MD anchor diagnostic (Tier 1a)
    e14 = diagnose_md_anchor(
        rids, spike_centroids_per_res, md_atom_positions, residue_to_atom_map,
    )

    # E15: Hellinger residue-distribution distance (Tier 1b)
    pocket_residue_energies = {
        int(r): float(res.loc[res["residue_id"] == r, "total_energy"].sum()) for r in rids
    }
    if SUBSTRATE_ONLY_MODE:
        e15 = {
            "_global": {
                "status": "BLOCKED",
                "gate": get_substrate_only_blocked_message(TARGET_ID),
            }
        }
    else:
        e15 = compute_pocket_reference_hellinger(
            pocket_residue_energies, REFERENCES, str(REF_DIR),
            all_residues, residue_offset_per_ref,
        )

    # E16: Wasserstein-distribution alignment (Tier 2)
    pocket_spike_df = df[df["residue_id"].isin(rids)][["x", "y", "z", "intensity"]].copy()
    if SUBSTRATE_ONLY_MODE:
        e16 = {
            "_global": {
                "status": "BLOCKED",
                "gate": get_substrate_only_blocked_message(TARGET_ID),
            }
        }
    else:
        e16 = {}
    for ref_id, ref_meta in REFERENCES.items():
        ref_pdb_path = REF_DIR / f"{ref_id}.pdb"
        if not ref_pdb_path.exists():
            e16[ref_id] = {"status": "BLOCKED", "gate": f"PDB missing: {ref_pdb_path}"}
            continue
        try:
            e16[ref_id] = wasserstein_align_reference(
                pocket_spike_df, str(ref_pdb_path), ref_meta["het"],
            )
        except Exception as exc:
            e16[ref_id] = {"status": "BLOCKED", "gate": f"exception: {exc}"}

    # E18: geodesic centroid (precomputed)
    e18 = geodesic_results_per_pocket.get(idx, {"status": "BLOCKED", "gate": "no centroid result"})

    enh = {
        "E1_ccns_lifecycle":     E1_ccns(rids),
        "E2_wavelength":         E2_wavelength(rids),
        "E3_phase_bits_mi":      E3_phase_bits(rids),
        "E4_first_passage":      E4_first_passage(rids),
        "E5_aromatic_graph":     E5_aromatic(rids, all_pocket_residues),
        "E6_cooperative":        E6_coop(rids),
        "E7_source_consensus":   E7_source(rids),
        "E8_voxel_adjacency":    E8_voxel(idx, voxel_matrix, pocket_voxels),
        "E9_wd_directional":     E9_wd_dir(rids),
        "E11_multi_view_dcc":    E11_multi_dcc(rids),
        "E12_phase_energy":      E12_phase_energy(rids),
        "E13_md_anchor":         E13_md_anchor(rids),
        "E14_md_anchor_diagnostic":      e14,
        "E15_hellinger_residue_distance": e15,
        "E16_wasserstein_alignment":     e16,
        "E18_geodesic_centroid":         e18,
    }
    
    # Druggability — incorporates E9 directional + E11 reference proximity
    e9 = enh["E9_wd_directional"]
    e11 = enh["E11_multi_view_dcc"]
    direction_bonus = (e9.get("directional_purity",0) if e9.get("status")=="OK" and
                       e9.get("direction") == "WATER_EXPELLING" else 0.0)
    proximity_bonus = 0.0
    if e11.get("status") == "OK":
        bp = e11["best_reference_min_prox"]
        proximity_bonus = max(0.0, (10.0 - bp) / 10.0)  # closer = higher

    # Legacy formula (Euclidean centroid) — preserved for forensic A/B.
    drug_score_legacy = (
        0.20 * np.log1p(diag["wd_change_p95"] * 1000) +
        0.15 * np.log1p(diag["wd_iqr"] * 1000) +
        0.15 * np.log1p(pE / max(diag["n_residues"],1) / 1e6) +
        0.10 * (n_arom / max(diag["n_residues"],1)) +
        0.10 * np.log1p(diag["frame_persistence_per_res"]) +
        0.10 * direction_bonus +
        0.15 * proximity_bonus +
        0.05 * (1.0 - diag["ccns_balance"])
    )

    # Geodesic-anchored proximity bonus (replaces Euclidean min_prox when
    # E18 is OK; falls back to Euclidean and is flagged otherwise).
    proximity_bonus_geodesic = proximity_bonus  # default fallback
    geodesic_proximity_used = "EUCLIDEAN_FALLBACK"
    if e18.get("status") == "OK" and e11.get("status") == "OK":
        geo_xyz = np.asarray(e18["geodesic_anchor_xyz"])
        geo_min_prox = []
        for ref_id in REFERENCES:
            if ref_id in aligned_ligands:
                d = float(np.linalg.norm(
                    aligned_ligands[ref_id] - geo_xyz, axis=1
                ).min())
                geo_min_prox.append(d)
        if geo_min_prox:
            bp_geo = min(geo_min_prox)
            proximity_bonus_geodesic = max(0.0, (10.0 - bp_geo) / 10.0)
            geodesic_proximity_used = "GEODESIC"

    # Hellinger overlap bonus: rewards pockets with low residue-distribution
    # distance to the closest populated reference contact-shell.
    hellinger_bonus = 0.0
    if isinstance(e15, dict):
        valid_h = [
            v.get("hellinger_distance", 1.0)
            for v in e15.values()
            if isinstance(v, dict) and v.get("status") == "OK"
        ]
        if valid_h:
            hellinger_bonus = max(0.0, 1.0 - min(valid_h))

    # Geodesic-corrected formula. Weights sum to 1.00.
    drug_score_geodesic = (
        0.18 * np.log1p(diag["wd_change_p95"] * 1000) +
        0.13 * np.log1p(diag["wd_iqr"] * 1000) +
        0.13 * np.log1p(pE / max(diag["n_residues"],1) / 1e6) +
        0.10 * (n_arom / max(diag["n_residues"],1)) +
        0.08 * np.log1p(diag["frame_persistence_per_res"]) +
        0.10 * direction_bonus +
        0.13 * proximity_bonus_geodesic +
        0.10 * hellinger_bonus +
        0.05 * (1.0 - diag["ccns_balance"])
    )

    drug_score = drug_score_geodesic  # primary score
    novelty = (100.0 - canonical_yield) * (drug_score / 5.0)

    flags = []
    if canonical_yield < 50.0: flags.append("OFF_CANONICAL")
    if n_arom >= 2: flags.append("AROMATIC_RICH")
    if diag["wd_change_p95"] > 0.020: flags.append("HIGH_WATER_DISPLACEMENT")
    if e9.get("status") == "OK" and e9.get("direction") == "WATER_EXPELLING": flags.append("WATER_EXPELLING")
    if e9.get("status") == "OK" and e9.get("direction") == "WATER_ATTRACTING": flags.append("WATER_ATTRACTING")
    if diag["frame_persistence_per_res"] > 30: flags.append("TEMPORALLY_PERSISTENT")
    if diag["frame_persistence_per_res"] < 15: flags.append("TRANSIENT_CRYPTIC")
    if e11.get("status") == "OK" and e11.get("best_reference_min_prox", 99) < 5.0: flags.append("LIGAND_PROXIMAL")
    if e11.get("status") == "OK" and e11.get("best_reference_dcc", 99) < 5.0: flags.append("REFERENCE_HIT")
    if enh["E5_aromatic_graph"].get("interpretation") == "ALLOSTERIC_COUPLED": flags.append("CROSS_POCKET_COUPLED")
    if enh["E8_voxel_adjacency"]["interpretation"] == "MEGACLUSTER_RISK": flags.append("MEGACLUSTER_RISK")
    if enh["E1_ccns_lifecycle"].get("status") == "OK":
        flags.append(f"TAX_{enh['E1_ccns_lifecycle']['taxonomy_class']}")
    if enh["E12_phase_energy"].get("status") == "OK":
        flags.append(f"KIN_{enh['E12_phase_energy']['kinetic_class']}")
# Tier 1+2+3 flags
    e14_diag = e14.get("diagnosis", "")
    if e14_diag == "TIGHT_ANCHOR": flags.append("ANCHOR_VERIFIED")
    elif "NUMBERING_OFFSET" in e14_diag: flags.append(f"ANCHOR_{e14_diag}")
    elif e14_diag == "COORDINATE_FRAME_SHIFT": flags.append("ANCHOR_FRAME_SHIFT")
    elif e14_diag == "GENUINE_DRIFT": flags.append("ANCHOR_GENUINE_DRIFT")

    e15_strong = any(v.get("interpretation") == "STRONG_OVERLAP" for v in e15.values()
                     if isinstance(v, dict))
    if e15_strong: flags.append("HELLINGER_STRONG_OVERLAP")

    if e18.get("interpretation") == "MEGACLUSTER_COLLAPSE_DETECTED":
        flags.append("E18_MEGACLUSTER_COLLAPSE")
    elif e18.get("interpretation") == "GEODESIC_AGREES_EUCLIDEAN":
        flags.append("E18_GEODESIC_AGREES")
    rec = {
        "pocket_id": idx, "residues": sorted(rids), "verdict": verdict,
        "verdict_evidence": f"{real_ev}R/{bridge_ev}B",
        "axes_supporting_real": axes, "dominant_region": dominant,
        "canonical_yield_pct": canonical_yield,
        "druggability_score": float(drug_score),
        "drug_score_legacy": float(drug_score_legacy),
        "drug_score_geodesic": float(drug_score_geodesic),
        "drug_score_delta": float(drug_score_geodesic - drug_score_legacy),
        "proximity_bonus_legacy": float(proximity_bonus),
        "proximity_bonus_geodesic": float(proximity_bonus_geodesic),
        "geodesic_proximity_source": geodesic_proximity_used,
        "hellinger_bonus": float(hellinger_bonus),
        "novelty_score": float(novelty),
        "flags": flags, "yields": yields, "core_diagnostics": diag,
        "n_aromatic_residues": n_arom,
        "aromatic_residue_ids": pdf[pdf["is_aromatic"]==1]["residue_id"].tolist(),
        "enhancements": enh,
    }
    records.append(rec)

    # Print summary
    print(f"\nPOCKET {idx} | {verdict} | dominant: {dominant} | "
          f"drug={drug_score:.3f} (legacy={drug_score_legacy:.3f} Δ={drug_score - drug_score_legacy:+.3f})  "
          f"novel={novelty:.2f}")
    print(f"  DRUG TERMS  prox_legacy={proximity_bonus:.3f}  prox_geodesic={proximity_bonus_geodesic:.3f} "
          f"({geodesic_proximity_used})  hellinger_bonus={hellinger_bonus:.3f}")
    print(f"  Residues ({len(rids)}): {sorted(rids)}")
    print(f"  Geometry  Rg={diag['rg']:.2f}A  seq_gap={diag['seq_gap']:.2f}")
    print(f"  Yields    " + "  ".join(f"{k}={v:.1f}%" for k,v in yields.items()))
    if e11.get("status") == "OK":
        br = e11["best_reference"]
        print(f"  REF HIT   {br} ({REFERENCES[br]['name']})  "
              f"DCC={e11['best_reference_dcc']:.2f}A  min_prox={e11['best_reference_min_prox']:.2f}A  "
              f"view={e11['per_reference'][br]['best_view']}")
        # All four references
        for pdb_id, ref_data in e11["per_reference"].items():
            print(f"    {pdb_id}: best={ref_data['best_view']:<16} dcc={ref_data['best_dcc']:>5.2f}A  "
                  f"prox={ref_data['min_residue_to_ligand_atom']:>5.2f}A  "
                  f"collapse={ref_data['geometric_collapse_delta']:+5.2f}A")
    if enh["E12_phase_energy"].get("status") == "OK":
        pf = enh["E12_phase_energy"]["phase_fractions"]
        print(f"  PHASES    " + "  ".join(f"{k}={v*100:.1f}%" for k,v in pf.items()) +
              f"  class={enh['E12_phase_energy']['kinetic_class']}")
    if enh["E13_md_anchor"].get("status") == "OK":
        print(f"  MD ANCHOR offset={enh['E13_md_anchor']['mean_spike_to_ca_offset_A']:.2f}A  "
              f"interp={enh['E13_md_anchor']['interpretation']}")
    # E14 diagnostic line
    if e14.get("status") == "OK":
        ev = e14.get("evidence", {})
        print(f"  E14 ANCHOR diagnosis={e14.get('diagnosis', '?')}  "
              f"direct_mean={ev.get('direct_match_mean_offset_A', 0):.2f}A  "
              f"best_offset={e14.get('best_offset', 0):+d}")
    # E15 Hellinger line (best ref)
    e15_strs = []
    for ref_id, v in e15.items():
        if isinstance(v, dict) and v.get("status") == "OK":
            e15_strs.append(f"{ref_id}:H={v.get('hellinger_distance', 0):.3f}/{v.get('interpretation', '?')[:5]}")
    if e15_strs:
        print(f"  E15 HELL  {', '.join(e15_strs)}")
    # E18 geodesic line
    if e18.get("status") == "OK":
        print(f"  E18 GEOD  anchor_res={e18.get('geodesic_anchor_residue', '?')}  "
              f"spread={e18.get('geodesic_spread_A', 0):.2f}A  "
              f"euclid_delta={e18.get('euclidean_to_geodesic_delta_A', 0):.2f}A  "
              f"interp={e18.get('interpretation', '?')}")
    print(f"  FLAGS     {flags}")

# ═══════════════════════════════════════════════════════════════════════
# STAGE 8: PROTEIN-LEVEL DISTRIBUTION + RUN-LEVEL E10
# ═══════════════════════════════════════════════════════════════════════
total_E = float(res["total_energy"].sum())
region_E = {k: float(res.loc[res["residue_id"].isin(v), "total_energy"].sum()) for k,v in DOMAINS.items()}
region_E["Other"] = total_E - sum(region_E.values())
canonical_pct = 100.0 * (total_E - region_E["Other"]) / total_E
protein_dist = {k: {"energy":v, "pct":100.0*v/total_E} for k,v in region_E.items()}

print("\n" + "=" * 100)
print("PROTEIN-LEVEL ENERGY DISTRIBUTION")
print("=" * 100)
for k, v in protein_dist.items():
    print(f"  {k:<14} {v['energy']:>14,.0f}  ({v['pct']:>5.1f}%)")
print(f"  CANONICAL TOT  {canonical_pct:>5.1f}%  <-- headline")

print("\n" + "=" * 100)
print("RUN-LEVEL E10: TENSOR DECOMPOSITION")
print("=" * 100)
e10_result = E10_tensor()
if e10_result["status"] == "BLOCKED":
    print(f"  E10 BLOCKED: {e10_result['gate']}")
else:
    print(f"  E10: {e10_result['n_components']} components, recon_err={e10_result['reconstruction_error']:.4f}")
    for c in e10_result["components"][:5]:
        print(f"    Comp {c['component_id']}: top_residues={c['top_residues']}")
 # ═══════════════════════════════════════════════════════════════════════
# RUN-LEVEL E17: persistent-homology pocket inventory
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("RUN-LEVEL E17: PERSISTENT HOMOLOGY POCKET INVENTORY")
print("=" * 100)
if e17_result.get("status") == "BLOCKED":
    print(f"  E17 BLOCKED: {e17_result.get('gate', '?')}")
else:
    print(f"  E17: persistence_components={e17_result.get('n_persistence_components', '?')}")
    diag = e17_result.get("persistence_diagram", [])[:5]
    for i, comp in enumerate(diag, 1):
        if comp.get("is_essential"):
            print(f"    Comp {i}: birth={comp['birth']:.4f}  ESSENTIAL")
        else:
            p = comp.get("persistence")
            print(f"    Comp {i}: birth={comp['birth']:.4f}  death={comp['death']:.4f}  "
                  f"persistence={p:.4f}" if p is not None else f"    Comp {i}: birth={comp['birth']:.4f}")
    inv = e17_result.get("pocket_inventory_ph", [])[:3]
    for i, comp in enumerate(inv, 1):
        residues_preview = comp["residues"][:8] if len(comp["residues"]) > 8 else comp["residues"]
        print(f"    Inv {i:>2}: n_residues={comp['n_residues']}  total_E={comp['total_energy']:.2e}  "
              f"residues={residues_preview}{'...' if len(comp['residues']) > 8 else ''}")

# ═══════════════════════════════════════════════════════════════════════
# STAGE 9: WRITE UNIFIED DOSSIER
# ═══════════════════════════════════════════════════════════════════════
# Derive target_id from Arrow filename
arrow_basename = Path(ARROW_PATH).name
if "_chain" in arrow_basename:
    target_id = arrow_basename.replace(".topology.spike_events.arrow", "").lower()
else:
    target_id = arrow_basename.split(".")[0].split("_")[0].lower()
csv_path  = os.path.join(OUT_DIR, f"{target_id}_unified_dossier.csv")
json_path = os.path.join(OUT_DIR, f"{target_id}_unified_dossier.json")

csv_records = []
for r in records:
    flat = {
        "pocket_id": r["pocket_id"], "residues": ";".join(map(str, r["residues"])),
        "verdict": r["verdict"], "dominant_region": r["dominant_region"],
        "canonical_yield_pct": r["canonical_yield_pct"],
        "druggability_score": r["druggability_score"],
        "drug_score_legacy": r.get("drug_score_legacy"),
        "drug_score_geodesic": r.get("drug_score_geodesic"),
        "drug_score_delta": r.get("drug_score_delta"),
        "proximity_bonus_legacy": r.get("proximity_bonus_legacy"),
        "proximity_bonus_geodesic": r.get("proximity_bonus_geodesic"),
        "geodesic_proximity_source": r.get("geodesic_proximity_source"),
        "hellinger_bonus": r.get("hellinger_bonus"),
        "novelty_score": r["novelty_score"],
        "n_residues": r["core_diagnostics"]["n_residues"],
        "n_aromatic": r["n_aromatic_residues"],
        "rg_A": r["core_diagnostics"]["rg"],
        "wd_change_p95": r["core_diagnostics"]["wd_change_p95"],
        "flags": ";".join(r["flags"]),
    }
    for k, v in r["yields"].items(): flat[f"yield_{k}"] = v
    e11 = r["enhancements"]["E11_multi_view_dcc"]
    if e11.get("status") == "OK":
        flat["best_ref"] = e11["best_reference"]
        flat["best_ref_dcc_A"] = e11["best_reference_dcc"]
        flat["best_ref_min_prox_A"] = e11["best_reference_min_prox"]
        for pdb_id in REFERENCES:
            if pdb_id in e11["per_reference"]:
                rd = e11["per_reference"][pdb_id]
                flat[f"dcc_{pdb_id}"] = rd["best_dcc"]
                flat[f"prox_{pdb_id}"] = rd["min_residue_to_ligand_atom"]
                flat[f"collapse_{pdb_id}"] = rd["geometric_collapse_delta"]
    e12 = r["enhancements"]["E12_phase_energy"]
    if e12.get("status") == "OK":
        flat["kinetic_lead"] = e12["kinetic_lead"]
        flat["kinetic_class"] = e12["kinetic_class"]
        for ph_name, frac in e12["phase_fractions"].items():
            flat[f"phase_{ph_name}_frac"] = frac
    for ekey, ev in r["enhancements"].items():
        flat[f"{ekey}_status"] = ev.get("status", "?")
        flat[f"{ekey}_interp"] = (ev.get("interpretation") or ev.get("taxonomy_class") or
                                   ev.get("kinetic_class") or ev.get("direction") or
                                   ev.get("gate") or "OK")
    csv_records.append(flat)

pd.DataFrame(csv_records).to_csv(csv_path, index=False)

dossier = {
    "run_metadata": run_metadata,
    "protein_level_energy_distribution": protein_dist,
    "canonical_total_pct": canonical_pct,
    "kdtree_radius_used": CANONICAL_RADIUS,
    "kdtree_radius_stability": {str(r):{"n_pockets":len(p)} for r,p in stability.items()},
    "alignment_diagnostics": alignment_diagnostics,
    "verified_het_ids": {k: v["het"] for k, v in REFERENCES.items()},
    "voxel_adjacency_matrix": voxel_matrix,
    "pockets": records,
    "run_level_E10": e10_result, 
    "run_level_E17": e17_result,
    "dev_gate_summary": {g: ("PASS" if s else "BLOCKED") for g, s in GATES.items()},
}

def to_json(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj); return v if math.isfinite(v) else None
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (set, frozenset)): return list(obj)
    if isinstance(obj, dict): return {str(k): to_json(v) for k,v in obj.items()}
    if isinstance(obj, (list, tuple)): return [to_json(x) for x in obj]
    if isinstance(obj, float) and not math.isfinite(obj): return None
    return obj

with open(json_path, "w") as f:
    json.dump(to_json(dossier), f, indent=2, default=str)

print(f"\n[9] UNIFIED DOSSIER WRITTEN")
print(f"    CSV:  {csv_path}")
print(f"    JSON: {json_path}")
print(f"    Pockets: {len(records)}  References aligned: {len(aligned_ligands)}/4")

# ═══════════════════════════════════════════════════════════════════════
# STAGE 10: TOP-RANKED MED-CHEM CANDIDATES
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print("MED-CHEM PRIORITY RANKING — top by druggability + reference proximity")
print("=" * 100)
ranked = sorted(records, key=lambda r: r["druggability_score"], reverse=True)
for r in ranked:
    e11 = r["enhancements"]["E11_multi_view_dcc"]
    ref_str = "no_ref" if e11.get("status") != "OK" else \
              f"{e11['best_reference']}@{e11['best_reference_min_prox']:.1f}A"
    print(f"  P{r['pocket_id']:>2}  drug={r['druggability_score']:.3f}  novel={r['novelty_score']:>5.2f}  "
          f"{r['verdict']:<22}  {r['dominant_region']:<14}  ref={ref_str:<20}  "
          f"flags={','.join(r['flags'][:3])}")

print("\n" + "=" * 100)
print("UNIFIED DOSSIER COMPLETE")
print("=" * 100)

#!/usr/bin/env python3
"""PRISM-4D Ground Truth Resolver — Phase 1.2 of canonical pipeline.

For a topology JSON, identifies the corresponding holo PDB on RCSB
(cached locally), classifies the deposit type, and decides whether
the bound ligand is appropriate for orthosteric DCC validation.

Filters that mark a target as INVALID for DCC validation:
  1. PDB not available on RCSB (recent CIF-only depositions)
  2. PanDDA fragment screen deposition (TITLE contains PANDDA / Enamine
     vendor IDs — fragments bind shallow surface hotspots, not real
     drug pockets, so DCC against the fragment is methodologically wrong)
  3. Templated ternary complex (DNA/RNA chains present alongside the
     protein — the ligand position is templated by the partner chain
     that the engine never sees, so DCC is invalid)
  4. Apo structure (no ligands bound)
  5. Only nuisance HETATMs (water, cryoprotectants, buffers, common ions)
  6. No ligand on the target chain (e.g., homo-tetramer where the
     fragment lands on chain B but we ran chain A)

For VALID targets, picks the best ligand:
  - Prefer drug-like organics (>=10 atoms) on the target chain
  - Fall back to single-atom metal cofactors (Zn/Mg/Fe/Mn/Cu/Ca/Ni/Co)
    on the target chain — these are catalytic site centers
  - Computes centroid and writes the sidecar

Writes sidecar to <output_dir>/<prefix>_ground_truth.json — consumed by
prism-postflight.py for DCC computation.

Holo PDBs are cached in ~/.cache/prism4d/holo_pdbs/ to avoid repeated
network calls during corpus runs.

Usage:
    python3 scripts/prism-ground-truth.py <topology.json> <output_dir>

Exit code is always 0 — ground truth resolution failures are not engine
failures. The sidecar always gets written, with valid_for_dcc_validation
indicating whether downstream validation is possible.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

CACHE_DIR = Path.home() / ".cache" / "prism4d" / "holo_pdbs"

# Solvents, cryoprotectants, common buffers — never count as ligands
NUISANCE_HETATMS = {
    "HOH", "H2O", "DOD", "WAT", "TIP",          # water
    "GOL", "EDO", "DMS", "PEG", "PG4", "PGE",   # cryoprotectants/glycols
    "SO4", "PO4", "ACT", "CIT", "TRS", "BME",   # buffers
    "MES", "HEPES", "EPE", "FMT", "IPA", "MPD",
    "BOG", "DTT", "TLA", "MAL", "FUC", "MAN",
    "BCT", "GLC", "NAG", "BMA",                 # sugars (often crystallographic noise)
    "OXY", "PER",                                # peroxides
}

# Common ions — also nuisance
COMMON_IONS = {"NA", "CL", "K", "BR", "I", "F", "CS", "RB", "SR"}

# Single-atom metals that are catalytically meaningful at active sites
METAL_COFACTORS = {"ZN", "MG", "FE", "MN", "CU", "CA", "NI", "CO", "FE2", "FE3"}

# Nucleic acid residues — presence indicates DNA/RNA chain
NUCLEIC_RESIDUES = {"DA", "DC", "DG", "DT", "DU", "DI",
                    "A", "C", "G", "U", "T", "I",
                    "RA", "RC", "RG", "RU"}

# Heuristic patterns for PanDDA fragment screen deposits
PANDDA_KEYWORDS = ["PANDDA", "FRAGMENT SCREEN", "FRAGMENT-BASED",
                   "FRAGMENT SCREENING", "XCHEM", "X-CHEM"]
ENAMINE_ID_PATTERN = re.compile(r'\bZ[0-9]{8,12}\b')


# ─────────────────────────────────────────────────────────────────────
# Topology parsing
# ─────────────────────────────────────────────────────────────────────

def parse_pdb_id_and_chain(topology_path):
    """
    Extract (pdb_id, chain) from the topology JSON filename or its
    embedded source_pdb field.

    Conventions handled:
        10dc_chainA.topology.json   -> ("10dc", "A")
        13sb_chainB.topology.json   -> ("13sb", "B")
        1btl_clean.topology.json    -> ("1btl", "A") [chain default]
        4lpk.topology.json          -> ("4lpk", "A") [chain default]
    """
    # Try the embedded source_pdb field first — most authoritative
    try:
        with open(topology_path) as f:
            topo = json.load(f)
        source = topo.get("source_pdb", "") or ""
    except (OSError, json.JSONDecodeError):
        source = ""

    candidates = [os.path.basename(topology_path)]
    if source:
        candidates.append(os.path.basename(source))

    for name in candidates:
        # Pattern: pdb id is 4 chars (digit + 3 alphanumerics), then optional _chainX
        m = re.match(
            r'^([0-9][a-z0-9]{3})(?:_chain([A-Z]))?',
            name,
            re.IGNORECASE,
        )
        if m:
            pdb_id = m.group(1).lower()
            chain = (m.group(2) or "A").upper()
            return pdb_id, chain

    return None, None


# ─────────────────────────────────────────────────────────────────────
# RCSB fetch with caching
# ─────────────────────────────────────────────────────────────────────

def fetch_holo_pdb(pdb_id):
    """
    Download holo PDB from RCSB. Cached at ~/.cache/prism4d/holo_pdbs/.
    Returns Path on success, None on 404 or network failure.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{pdb_id.lower()}.pdb"

    if cache_path.exists() and cache_path.stat().st_size > 100:
        return cache_path

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "prism4d-ground-truth/1.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        if len(data) < 100:
            return None
        cache_path.write_bytes(data)
        return cache_path
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        return None
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


# ─────────────────────────────────────────────────────────────────────
# Deposit-type detection
# ─────────────────────────────────────────────────────────────────────

def detect_pandda(holo_pdb):
    """
    Return True if the PDB header indicates a PanDDA / fragment screen.
    Scans HEADER, TITLE, COMPND, JRNL, REMARK lines for PanDDA keywords
    and Enamine vendor ID patterns.
    """
    with open(holo_pdb, errors="ignore") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                break  # past the header section
            if line.startswith(("HEADER", "TITLE", "COMPND",
                                "JRNL", "REMARK", "KEYWDS")):
                upper = line.upper()
                for kw in PANDDA_KEYWORDS:
                    if kw in upper:
                        return True
                if ENAMINE_ID_PATTERN.search(line):
                    return True
    return False


def detect_nucleic_chains(holo_pdb):
    """
    Return set of chain IDs that contain DNA/RNA residues.
    Scans ATOM records for residue names in NUCLEIC_RESIDUES.
    """
    chains = set()
    with open(holo_pdb, errors="ignore") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                if resname in NUCLEIC_RESIDUES:
                    chains.add(line[21])
    return chains


# ─────────────────────────────────────────────────────────────────────
# Ligand parsing and selection
# ─────────────────────────────────────────────────────────────────────

def parse_hetatm_groups(holo_pdb):
    """
    Group HETATM records by (resname, chain, resnum).
    Returns dict[(resname, chain, resnum)] = list of (x, y, z).
    """
    groups = {}
    with open(holo_pdb, errors="ignore") as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            try:
                resname = line[17:20].strip()
                chain = line[21]
                resnum = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except (ValueError, IndexError):
                continue
            groups.setdefault((resname, chain, resnum), []).append((x, y, z))
    return groups


def classify(resname, n_atoms):
    """Map a HETATM group to one of: nuisance | metal_cofactor | drug_like | small_molecule | tiny."""
    if resname in NUISANCE_HETATMS or resname in COMMON_IONS:
        return "nuisance"
    if resname in METAL_COFACTORS and n_atoms == 1:
        return "metal_cofactor"
    if n_atoms >= 10:
        return "drug_like"
    if n_atoms >= 5:
        return "small_molecule"
    return "tiny"


def select_ligand(groups, target_chain):
    """
    Pick the best ground-truth ligand for the target chain.

    Priority order:
      1. drug_like organic ON target chain (largest atom count)
      2. metal_cofactor ON target chain
      3. small_molecule ON target chain
      4. drug_like organic on a DIFFERENT chain (returned with on_chain=False)

    Returns (resname, chain, resnum, atoms, classification, on_target_chain)
    or None if nothing valid.
    """
    on_chain_drug = []
    on_chain_metal = []
    on_chain_small = []
    off_chain_drug = []

    for (resname, chain, resnum), atoms in groups.items():
        klass = classify(resname, len(atoms))
        if klass in ("nuisance", "tiny"):
            continue
        on_chain = (chain == target_chain)
        entry = (resname, chain, resnum, atoms, klass)
        if klass == "drug_like":
            (on_chain_drug if on_chain else off_chain_drug).append(entry)
        elif klass == "metal_cofactor" and on_chain:
            on_chain_metal.append(entry)
        elif klass == "small_molecule" and on_chain:
            on_chain_small.append(entry)

    if on_chain_drug:
        on_chain_drug.sort(key=lambda e: -len(e[3]))
        r = on_chain_drug[0]
        return (*r, True)
    if on_chain_metal:
        return (*on_chain_metal[0], True)
    if on_chain_small:
        on_chain_small.sort(key=lambda e: -len(e[3]))
        return (*on_chain_small[0], True)
    if off_chain_drug:
        off_chain_drug.sort(key=lambda e: -len(e[3]))
        r = off_chain_drug[0]
        return (*r, False)
    return None


def centroid(atoms):
    n = len(atoms)
    return [
        sum(a[0] for a in atoms) / n,
        sum(a[1] for a in atoms) / n,
        sum(a[2] for a in atoms) / n,
    ]


# ─────────────────────────────────────────────────────────────────────
# Resolution pipeline
# ─────────────────────────────────────────────────────────────────────

def resolve(topology_path):
    """
    Full ground-truth resolution. Returns dict ready to write as sidecar.
    """
    pdb_id, chain = parse_pdb_id_and_chain(topology_path)

    result = {
        "topology": str(topology_path),
        "pdb_id": pdb_id,
        "target_chain": chain,
        "valid_for_dcc_validation": False,
        "skip_reason": None,
        "ligand": None,
        "ligand_centroid": None,
        "is_pandda_fragment": False,
        "is_templated_complex": False,
        "nucleic_chains": [],
        "holo_source": None,
    }

    if pdb_id is None:
        result["skip_reason"] = "could_not_parse_pdb_id_from_filename"
        return result

    holo = fetch_holo_pdb(pdb_id)
    if holo is None:
        result["skip_reason"] = "no_holo_pdb_on_rcsb"
        return result
    result["holo_source"] = str(holo)

    if detect_pandda(holo):
        result["is_pandda_fragment"] = True
        result["skip_reason"] = "pandda_fragment_screen_deposit"
        return result

    nucleic = detect_nucleic_chains(holo)
    if nucleic:
        result["nucleic_chains"] = sorted(nucleic)
        result["is_templated_complex"] = True
        result["skip_reason"] = (
            f"templated_complex_with_nucleic_chains_{','.join(sorted(nucleic))}"
        )
        return result

    groups = parse_hetatm_groups(holo)
    if not groups:
        result["skip_reason"] = "apo_structure_no_hetatm_records"
        return result

    sel = select_ligand(groups, chain)
    if sel is None:
        result["skip_reason"] = "only_nuisance_or_tiny_ligands"
        return result

    resname, lig_chain, lig_resnum, atoms, klass, on_target = sel

    if not on_target:
        result["ligand"] = {
            "resname": resname,
            "chain": lig_chain,
            "resnum": lig_resnum,
            "n_atoms": len(atoms),
            "classification": klass,
        }
        result["skip_reason"] = (
            f"ligand_only_on_chain_{lig_chain}_not_target_chain_{chain}"
        )
        return result

    result["ligand"] = {
        "resname": resname,
        "chain": lig_chain,
        "resnum": lig_resnum,
        "n_atoms": len(atoms),
        "classification": klass,
    }
    result["ligand_centroid"] = centroid(atoms)
    result["valid_for_dcc_validation"] = True
    return result


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PRISM-4D ground truth resolver (Phase 1.2 of canonical pipeline)",
    )
    parser.add_argument("topology", help="Topology JSON file")
    parser.add_argument(
        "output_dir",
        help="Output directory where the sidecar will be written",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="File prefix (default: derived from topology basename, stripping .topology.json)",
    )
    args = parser.parse_args()

    topo_path = Path(args.topology)
    if not topo_path.exists():
        print(f"FAIL: topology not found: {args.topology}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.prefix or topo_path.name.replace(".topology.json", "")
    sidecar = out_dir / f"{prefix}_ground_truth.json"

    result = resolve(args.topology)
    sidecar.write_text(json.dumps(result, indent=2))

    # Human-readable summary
    print(f"Ground truth: {topo_path.name}")
    print(f"  PDB: {result['pdb_id']}  chain: {result['target_chain']}")
    if result["valid_for_dcc_validation"]:
        L = result["ligand"]
        c = result["ligand_centroid"]
        print(f"  Valid: YES")
        print(
            f"  Ligand: {L['resname']} ({L['classification']}, "
            f"chain {L['chain']}, {L['n_atoms']} atoms)"
        )
        print(f"  Centroid: ({c[0]:7.2f}, {c[1]:7.2f}, {c[2]:7.2f})")
    else:
        print(f"  Valid: NO")
        print(f"  Skip reason: {result['skip_reason']}")
        if result["is_pandda_fragment"]:
            print(f"  Detected: PanDDA fragment screen deposition")
        if result["is_templated_complex"]:
            print(f"  Detected: templated complex with chains {result['nucleic_chains']}")
    print(f"  Sidecar: {sidecar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

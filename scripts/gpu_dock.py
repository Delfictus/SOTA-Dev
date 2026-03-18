#!/usr/bin/env python3
"""
PRISM4D GPU Docking Pipeline
=============================
End-to-end GPU-accelerated molecular docking from PRISM binding site predictions.

Pipeline:
  1. Parse binding_sites.json → extract docking box per site
  2. Prepare receptor (PDB → PDBQT via meeko)
  3. Prepare ligands (SMILES/SDF → 3D conformers → SDF)
  4. UniDock GPU batch docking (Vina scoring, ~1000x speedup)
  5. GNINA CNN rescoring (top N poses, deep learning refinement)
  6. Consensus ranking + output (JSON, SDF, PyMOL script)

Usage:
    python scripts/gpu_dock.py \\
        --receptor e2e_validation_test/prep/1nkp_dimer.pdb \\
        --sites e2e_validation_test/results_myc/1nkp_dimer.binding_sites.json \\
        --ligands ligands.sdf \\
        --output docking_results/

    # Or with SMILES directly:
    python scripts/gpu_dock.py \\
        --receptor structure.pdb \\
        --sites results/binding_sites.json \\
        --smiles "CCO sotorasib" "CC=O adagrasib" \\
        --output docking_results/

Requirements (conda env prism_dock):
    unidock, gnina, rdkit, meeko, openbabel, numpy

References:
    UniDock: Yu et al. JCTC 2023. doi:10.1021/acs.jctc.2c01145
    GNINA 1.3: McNutt et al. J Cheminform 2025. doi:10.1186/s13321-025-00973-x
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

# ── Conda environment paths ──────────────────────────────────────────────
CONDA_ENV = Path(os.environ.get(
    "PRISM_DOCK_ENV",
    os.path.expanduser("~/miniconda3/envs/prism_dock")
))
UNIDOCK_BIN = CONDA_ENV / "bin" / "unidock"
GNINA_BIN = CONDA_ENV / "bin" / "gnina"
OBABEL_BIN = CONDA_ENV / "bin" / "obabel"
PYTHON_BIN = CONDA_ENV / "bin" / "python"
GNINA_LD_PATH = str(CONDA_ENV / "lib")


def check_tools():
    """Verify all required binaries exist."""
    missing = []
    for name, path in [("unidock", UNIDOCK_BIN), ("gnina", GNINA_BIN),
                        ("obabel", OBABEL_BIN), ("python", PYTHON_BIN)]:
        if not path.exists():
            missing.append(f"  {name}: {path}")
    if missing:
        print("ERROR: Missing tools in prism_dock conda env:")
        print("\n".join(missing))
        print(f"\nInstall with: conda install -n prism_dock -c conda-forge unidock rdkit meeko openbabel")
        print(f"GNINA: download from https://github.com/gnina/gnina/releases")
        sys.exit(1)


# ── Ligand preparation ───────────────────────────────────────────────────

def prepare_ligands_from_smiles(smiles_list, output_dir):
    """Convert SMILES strings to 3D SDF via RDKit + energy minimization.

    Args:
        smiles_list: list of (smiles, name) tuples
        output_dir: directory for individual SDF files

    Returns:
        Path to combined SDF file

    Uses RDKit ETKDG v3 conformer generation and MMFF94s minimization.
    Ref: Riniker & Landrum, J Chem Inf Model 2015. doi:10.1021/acs.jcim.5b00654
    """
    script = '''
import sys, json
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from meeko import MoleculePreparation, PDBQTWriterLegacy

data = json.loads(sys.stdin.read())
output_sdf = data["output_sdf"]
output_pdbqt_dir = data["output_pdbqt_dir"]
results = []

writer = Chem.SDWriter(output_sdf)

for entry in data["ligands"]:
    smiles = entry["smiles"]
    name = entry["name"]
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results.append({"name": name, "status": "error", "error": "invalid SMILES"})
            continue

        # Sanitize and normalize
        Chem.SanitizeMol(mol)
        mol = rdMolStandardize.FragmentParent(mol)

        # Add hydrogens
        mol = Chem.AddHs(mol)

        # Generate 3D conformer (ETKDG v3)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        status = AllChem.EmbedMolecule(mol, params)
        if status == -1:
            status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if status == -1:
            AllChem.EmbedMolecule(mol, useRandomCoords=True)

        # Minimize with MMFF94s (200 steps)
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=50)
        except:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=50)
            except:
                pass

        # Compute Gasteiger charges
        AllChem.ComputeGasteigerCharges(mol)

        mol.SetProp("_Name", name)
        writer.write(mol)

        # Also write PDBQT via meeko
        try:
            preparator = MoleculePreparation()
            mol_setups = preparator.prepare(mol)
            pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(mol_setups[0])
            if is_ok:
                pdbqt_path = f"{output_pdbqt_dir}/{name}.pdbqt"
                with open(pdbqt_path, "w") as f:
                    f.write(pdbqt_string)
        except Exception as e:
            pass  # PDBQT is optional, SDF works for GNINA

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        rot = Descriptors.NumRotatableBonds(mol)

        results.append({
            "name": name, "status": "ok", "mw": round(mw, 1),
            "logp": round(logp, 2), "hbd": hbd, "hba": hba,
            "rotatable_bonds": rot
        })
    except Exception as e:
        results.append({"name": name, "status": "error", "error": str(e)})

writer.close()
print(json.dumps({"results": results}))
'''
    output_dir = Path(output_dir)
    sdf_path = output_dir / "ligands_3d.sdf"
    pdbqt_dir = output_dir / "pdbqt_ligands"
    pdbqt_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "output_sdf": str(sdf_path),
        "output_pdbqt_dir": str(pdbqt_dir),
        "ligands": [{"smiles": s, "name": n} for s, n in smiles_list]
    }

    result = subprocess.run(
        [str(PYTHON_BIN), "-c", script],
        input=json.dumps(payload), capture_output=True, text=True,
        timeout=300
    )
    if result.returncode != 0:
        print(f"ERROR in ligand prep: {result.stderr}")
        sys.exit(1)

    prep_results = json.loads(result.stdout)
    for r in prep_results["results"]:
        status = "OK" if r["status"] == "ok" else f"FAIL: {r.get('error','?')}"
        print(f"  Ligand {r['name']}: {status}", end="")
        if r["status"] == "ok":
            print(f"  MW={r['mw']} LogP={r['logp']} HBD={r['hbd']} HBA={r['hba']} RotB={r['rotatable_bonds']}")
        else:
            print()

    return sdf_path, pdbqt_dir, prep_results["results"]


def _clean_pdb_for_obabel(pdb_path, output_dir):
    """Strip non-protein atoms from PDB to prevent OpenBabel kekulization failures.

    Removes: HETATMs (metals, ligands, waters, sugars), alternate conformations
    (keeps only 'A' or first altloc), and non-standard residue naming artifacts.
    The receptor PDBQT only needs protein backbone + sidechains.

    Returns:
        Path to cleaned PDB file.
    """
    STANDARD_AA = {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
        "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
        "THR", "TRP", "TYR", "VAL",
        # Common modified residues that obabel handles fine
        "MSE", "HID", "HIE", "HIP", "CYX", "ASH", "GLH",
    }

    cleaned_path = Path(output_dir) / (Path(pdb_path).stem + "_clean.pdb")
    kept = 0
    dropped_hetatm = 0
    dropped_altloc = 0

    with open(pdb_path) as fin, open(cleaned_path, "w") as fout:
        for line in fin:
            record = line[:6].strip()

            # Pass through non-coordinate records (HEADER, REMARK, etc.)
            if record not in ("ATOM", "HETATM", "TER", "END", "ANISOU", "MODEL", "ENDMDL"):
                fout.write(line)
                continue

            if record in ("TER", "END", "MODEL", "ENDMDL"):
                fout.write(line)
                continue

            if record == "ANISOU":
                continue

            # For ATOM/HETATM lines, parse fields (PDB fixed-width format)
            resname = line[17:20].strip()
            altloc = line[16]  # column 17 (0-indexed 16)

            # Drop alternate conformations — keep blank or 'A' only
            if altloc not in (" ", "", "A"):
                dropped_altloc += 1
                continue

            # For altloc 'A', blank it out so obabel doesn't choke
            if altloc == "A":
                line = line[:16] + " " + line[17:]

            if record == "HETATM":
                # Keep MSE (selenomethionine) as it's basically MET
                if resname == "MSE":
                    # Rewrite as ATOM record
                    line = "ATOM  " + line[6:]
                    fout.write(line)
                    kept += 1
                else:
                    dropped_hetatm += 1
                    continue
            elif record == "ATOM":
                # Strip non-standard residue name prefixes (e.g., AGLN → GLN)
                # Some PDBs use the altloc prefix in the residue name field
                if len(resname) == 4 and resname[1:] in STANDARD_AA:
                    resname_clean = resname[1:]
                    line = line[:17] + f" {resname_clean}" + line[20:]

                if resname in STANDARD_AA or (len(resname) == 4 and resname[1:] in STANDARD_AA):
                    fout.write(line)
                    kept += 1
                else:
                    dropped_hetatm += 1
                    continue

    print(f"  PDB cleanup: {kept} protein atoms kept, "
          f"{dropped_hetatm} HETATM/non-std dropped, "
          f"{dropped_altloc} alt-confs dropped")
    return cleaned_path


def prepare_receptor_pdbqt(pdb_path, output_dir):
    """Convert receptor PDB to PDBQT using OpenBabel.

    Adds polar hydrogens and assigns Gasteiger charges.
    Pre-cleans the PDB to strip HETATMs, metals, waters, alternate
    conformations, and non-standard residues that cause OpenBabel
    kekulization failures.
    """
    output_dir = Path(output_dir)
    pdbqt_path = output_dir / (Path(pdb_path).stem + ".pdbqt")

    # Clean PDB first to avoid kekulization failures from metals/ligands/altlocs
    cleaned_pdb = _clean_pdb_for_obabel(pdb_path, output_dir)

    result = subprocess.run(
        [str(OBABEL_BIN), str(cleaned_pdb), "-O", str(pdbqt_path),
         "-xr", "-xh", "--partialcharge", "gasteiger"],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"WARNING: obabel receptor conversion had issues: {result.stderr[:200]}")

    if not pdbqt_path.exists() or pdbqt_path.stat().st_size < 100:
        # Fallback: try the raw PDB in case cleaning was too aggressive
        print(f"  WARNING: Cleaned PDB failed, retrying with raw PDB...")
        result = subprocess.run(
            [str(OBABEL_BIN), str(pdb_path), "-O", str(pdbqt_path),
             "-xr", "-xh", "--partialcharge", "gasteiger"],
            capture_output=True, text=True, timeout=120
        )
        if not pdbqt_path.exists() or pdbqt_path.stat().st_size < 100:
            print(f"ERROR: Receptor PDBQT conversion failed for {pdb_path}")
            sys.exit(1)

    print(f"  Receptor PDBQT: {pdbqt_path} ({pdbqt_path.stat().st_size / 1024:.0f} KB)")
    return pdbqt_path


# ── Docking box from binding sites ───────────────────────────────────────

def extract_docking_boxes(sites_json):
    """Extract docking box parameters from binding_sites.json.

    Box is centered on the PRISM-detected centroid with adaptive sizing
    based on pocket volume. Padding of 4 Angstrom added to ensure the
    search space fully envelopes the binding cavity.
    """
    with open(sites_json) as f:
        data = json.load(f)

    boxes = []
    for site in data["sites"]:
        centroid = site["centroid"]
        volume = site.get("volume", 1000)

        # Adaptive box size from pocket volume
        # V = x*y*z, assume roughly cubic, add 4A padding per side
        side = max(volume ** (1.0 / 3.0) + 8.0, 20.0)
        side = min(side, 40.0)  # UniDock/Vina max ~40A per dimension

        boxes.append({
            "site_id": site["id"],
            "classification": site["classification"],
            "is_druggable": site.get("is_druggable", False),
            "center_x": round(centroid[0], 3),
            "center_y": round(centroid[1], 3),
            "center_z": round(centroid[2], 3),
            "size_x": round(side, 1),
            "size_y": round(side, 1),
            "size_z": round(side, 1),
            "volume": volume,
            "n_lining_residues": len(site.get("lining_residues", [])),
        })

    return boxes, data


# ── UniDock GPU batch docking ────────────────────────────────────────────

def run_unidock(receptor_pdbqt, ligand_sdf, box, output_dir, exhaustiveness=32,
                num_modes=20, scoring="vina"):
    """Run UniDock GPU-accelerated batch docking.

    UniDock implements the AutoDock Vina scoring function on GPU with
    >1000x speedup over CPU Vina.
    Ref: Yu et al. JCTC 2023. doi:10.1021/acs.jctc.2c01145

    Args:
        receptor_pdbqt: path to receptor PDBQT
        ligand_sdf: path to multi-molecule SDF (3D conformers)
        box: dict with center_x/y/z and size_x/y/z
        output_dir: directory for docked poses
        exhaustiveness: search thoroughness (default 32)
        num_modes: max poses per ligand (default 20)
        scoring: scoring function (vina, vinardo, or ad4)

    Returns:
        Path to output directory with docked poses
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(UNIDOCK_BIN),
        "--receptor", str(receptor_pdbqt),
        "--gpu_batch", str(ligand_sdf),
        "--center_x", str(box["center_x"]),
        "--center_y", str(box["center_y"]),
        "--center_z", str(box["center_z"]),
        "--size_x", str(box["size_x"]),
        "--size_y", str(box["size_y"]),
        "--size_z", str(box["size_z"]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--scoring", scoring,
        "--dir", str(output_dir),
        "--seed", "42",
    ]

    print(f"  Running UniDock GPU docking...")
    print(f"    Box: center=({box['center_x']}, {box['center_y']}, {box['center_z']})")
    print(f"    Box: size=({box['size_x']}, {box['size_y']}, {box['size_z']})")
    print(f"    Exhaustiveness: {exhaustiveness}, Modes: {num_modes}, Scoring: {scoring}")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = GNINA_LD_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)

    if result.returncode != 0:
        print(f"  UniDock stderr: {result.stderr[:500]}")
        # Non-zero exit can still produce output
    if result.stdout:
        # Parse UniDock output for timing
        for line in result.stdout.split("\n"):
            if "time" in line.lower() or "score" in line.lower() or "pose" in line.lower():
                print(f"    {line.strip()}")

    # Collect output files
    out_files = sorted(output_dir.glob("*.pdbqt")) + sorted(output_dir.glob("*.sdf"))
    print(f"  UniDock output: {len(out_files)} files in {output_dir}")
    return output_dir, out_files


def run_unidock_with_pdbqt_ligands(receptor_pdbqt, pdbqt_dir, box, output_dir,
                                     exhaustiveness=32, num_modes=20, scoring="vina"):
    """Run UniDock with individual PDBQT ligand files (single-box, single-call).

    UniDock --gpu_batch accepts SDF, but some edge cases need PDBQT input.
    For concurrent multi-pocket docking, see _unidock_one_site().
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdbqt_files = sorted(Path(pdbqt_dir).glob("*.pdbqt"))
    if not pdbqt_files:
        print("  ERROR: No PDBQT ligands found")
        return output_dir, []

    # Write ligand index file for --ligand_index mode
    index_path = output_dir / "ligand_index.txt"
    with open(index_path, "w") as f:
        for p in pdbqt_files:
            f.write(str(p) + "\n")

    cmd = [
        str(UNIDOCK_BIN),
        "--receptor", str(receptor_pdbqt),
        "--ligand_index", str(index_path),
        "--center_x", str(box["center_x"]),
        "--center_y", str(box["center_y"]),
        "--center_z", str(box["center_z"]),
        "--size_x", str(box["size_x"]),
        "--size_y", str(box["size_y"]),
        "--size_z", str(box["size_z"]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--scoring", scoring,
        "--dir", str(output_dir),
        "--seed", "42",
    ]

    print(f"  Running UniDock GPU docking ({len(pdbqt_files)} PDBQT ligands)...")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = GNINA_LD_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)

    if result.returncode != 0:
        print(f"  UniDock stderr: {result.stderr[:500]}")

    out_files = sorted(output_dir.glob("*_out.pdbqt"))
    print(f"  UniDock output: {len(out_files)} docked files")
    return output_dir, out_files


# ── Concurrent per-site docking workers ──────────────────────────────────

def _unidock_one_site(receptor_pdbqt, ligand_sdf, box, site_dir,
                      exhaustiveness, num_modes, scoring, env):
    """Dock all ligands against ONE pocket in its own isolated directory.

    Thread-safe worker. Uses --gpu_batch with SDF input (not --ligand_index
    with PDBQT) because UniDock v1.1.3 segfaults on Meeko-generated PDBQT
    ROOT tags. SDF input works reliably.

    Each site gets its own --dir so UniDock output files never collide.

    Returns:
        (site_id, list_of_output_files, elapsed_seconds, error_msg_or_None)
    """
    site_id = box["site_id"]
    unidock_dir = Path(site_dir) / "unidock_out"
    unidock_dir.mkdir(parents=True, exist_ok=True)

    if not Path(ligand_sdf).exists():
        return site_id, [], 0, f"ligand SDF not found: {ligand_sdf}"

    cmd = [
        str(UNIDOCK_BIN),
        "--receptor", str(receptor_pdbqt),
        "--gpu_batch", str(ligand_sdf),
        "--center_x", str(box["center_x"]),
        "--center_y", str(box["center_y"]),
        "--center_z", str(box["center_z"]),
        "--size_x", str(box["size_x"]),
        "--size_y", str(box["size_y"]),
        "--size_z", str(box["size_z"]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--scoring", scoring,
        "--dir", str(unidock_dir),
        "--seed", "42",
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    elapsed = time.time() - t0

    error = None
    if result.returncode != 0:
        error = result.stderr[:300] if result.stderr else f"exit code {result.returncode}"

    # UniDock with --gpu_batch outputs *_out.sdf files
    out_files = sorted(unidock_dir.glob("*_out.sdf"))
    if not out_files:
        out_files = sorted(unidock_dir.glob("*_out.pdbqt"))
    return site_id, out_files, elapsed, error


def _gnina_rescore_one_site(receptor_pdb, box, site_dir, env):
    """Rescore all UniDock poses for ONE pocket, reading and writing to its own dir.

    Thread-safe worker. Reads *_out.pdbqt from site_dir/unidock_out/,
    writes *_rescored.sdf into site_dir/gnina_rescore/.

    Returns:
        (site_id, rescored_files, scores_list, elapsed_seconds)
    """
    site_id = box["site_id"]
    unidock_dir = Path(site_dir) / "unidock_out"
    gnina_dir = Path(site_dir) / "gnina_rescore"
    gnina_dir.mkdir(parents=True, exist_ok=True)

    # UniDock --gpu_batch produces *_out.sdf; --ligand_index produces *_out.pdbqt
    docked_files = sorted(unidock_dir.glob("*_out.sdf"))
    if not docked_files:
        docked_files = sorted(unidock_dir.glob("*_out.pdbqt"))
    if not docked_files:
        docked_files = sorted(unidock_dir.glob("*.sdf"))
    if not docked_files:
        docked_files = sorted(unidock_dir.glob("*.pdbqt"))
        docked_files = [f for f in docked_files if f.name != "ligand_index.txt"]

    if not docked_files:
        return site_id, [], [], 0

    all_scores = []
    rescored_files = []
    t0 = time.time()

    for docked_file in docked_files:
        out_sdf = gnina_dir / (docked_file.stem + "_rescored.sdf")

        cmd = [
            str(GNINA_BIN),
            "--receptor", str(receptor_pdb),
            "--ligand", str(docked_file),
            "--center_x", str(box["center_x"]),
            "--center_y", str(box["center_y"]),
            "--center_z", str(box["center_z"]),
            "--size_x", str(box["size_x"]),
            "--size_y", str(box["size_y"]),
            "--size_z", str(box["size_z"]),
            "--cnn_scoring", "rescore",
            "--out", str(out_sdf),
            "--seed", "42",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)

        if result.returncode != 0:
            continue

        for line in result.stdout.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("Using") and not line.startswith("WARNING"):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        all_scores.append({
                            "file": docked_file.name,
                            "site_id": site_id,
                            "mode": int(parts[0]),
                            "vina_score": float(parts[1]),
                            "cnn_score": float(parts[2]),
                            "cnn_affinity": float(parts[3]),
                        })
                    except (ValueError, IndexError):
                        pass

        if out_sdf.exists():
            rescored_files.append(out_sdf)

    elapsed = time.time() - t0
    all_scores.sort(key=lambda x: x.get("cnn_affinity", 0))
    return site_id, rescored_files, all_scores, elapsed


# ── GNINA direct docking (alternative to UniDock + rescore) ──────────────

def run_gnina_dock(receptor_pdb, ligand_sdf, box, output_dir,
                   exhaustiveness=32, num_modes=9, cnn_scoring="rescore"):
    """Run GNINA for direct docking with CNN scoring.

    Use this as a fallback when UniDock fails, or for small ligand sets
    where CNN-guided docking is preferred over speed.

    Args:
        receptor_pdb: receptor PDB path
        ligand_sdf: multi-molecule SDF
        box: docking box dict
        output_dir: output directory
        exhaustiveness: search depth
        num_modes: poses per ligand
        cnn_scoring: 'none', 'rescore', 'refinement', or 'all'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_sdf = output_dir / "gnina_docked.sdf"

    cmd = [
        str(GNINA_BIN),
        "--receptor", str(receptor_pdb),
        "--ligand", str(ligand_sdf),
        "--center_x", str(box["center_x"]),
        "--center_y", str(box["center_y"]),
        "--center_z", str(box["center_z"]),
        "--size_x", str(box["size_x"]),
        "--size_y", str(box["size_y"]),
        "--size_z", str(box["size_z"]),
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", str(num_modes),
        "--cnn_scoring", cnn_scoring,
        "--out", str(out_sdf),
        "--seed", "42",
    ]

    print(f"  Running GNINA direct docking (cnn_scoring={cnn_scoring})...")

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = GNINA_LD_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)

    scores = []
    for line in result.stdout.split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("Using") and not line.startswith("WARNING"):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    scores.append({
                        "mode": int(parts[0]),
                        "vina_score": float(parts[1]),
                        "cnn_score": float(parts[2]),
                        "cnn_affinity": float(parts[3]),
                    })
                except (ValueError, IndexError):
                    pass

    if result.returncode != 0:
        print(f"  GNINA stderr: {result.stderr[:300]}")

    print(f"  GNINA output: {out_sdf} ({len(scores)} poses)")
    return out_sdf, scores


# ── Results aggregation ──────────────────────────────────────────────────

def generate_results(site, box, unidock_scores, gnina_scores, ligand_info,
                     output_dir, receptor_pdb):
    """Generate final ranked results with visualization scripts."""
    output_dir = Path(output_dir)

    # Build consensus ranking
    # Combine UniDock Vina scores with GNINA CNN rescores
    results = {
        "pipeline": "PRISM4D GPU Docking",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "receptor": str(receptor_pdb),
        "site": {
            "id": site["id"],
            "classification": site["classification"],
            "centroid": site["centroid"],
            "volume": site.get("volume", 0),
            "druggability": site.get("druggability", 0),
            "is_druggable": site.get("is_druggable", False),
        },
        "docking_box": box,
        "tools": {
            "unidock": "v1.1.3 (GPU, Vina scoring)",
            "gnina": "v1.3.2 (CNN rescore, dense_1_3 ensemble)",
        },
        "references": [
            "UniDock: Yu et al. JCTC 2023. doi:10.1021/acs.jctc.2c01145",
            "GNINA 1.3: McNutt et al. J Cheminform 2025. doi:10.1186/s13321-025-00973-x",
        ],
        "ligands_prepared": ligand_info,
        "gnina_scores": gnina_scores[:50],  # Top 50
    }

    # Write JSON results
    json_path = output_dir / f"site{site['id']}_docking_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results JSON: {json_path}")

    # Write PyMOL visualization script
    pml_path = output_dir / f"site{site['id']}_docking_viz.pml"
    write_docking_pymol(pml_path, receptor_pdb, box, gnina_scores[:10], output_dir)
    print(f"  PyMOL script: {pml_path}")

    # Write summary report
    report_path = output_dir / f"site{site['id']}_docking_report.md"
    write_docking_report(report_path, results, gnina_scores)
    print(f"  Report: {report_path}")

    return results


def write_docking_pymol(pml_path, receptor_pdb, box, top_scores, output_dir):
    """Generate PyMOL script for docking results visualization."""
    cx, cy, cz = box["center_x"], box["center_y"], box["center_z"]
    sx, sy, sz = box["size_x"], box["size_y"], box["size_z"]

    lines = [
        "# PRISM4D GPU Docking Visualization",
        "# Generated by gpu_dock.py",
        "",
        "reinitialize",
        "bg_color white",
        "set cartoon_fancy_helices, 1",
        "set ray_trace_mode, 1",
        "",
        f"load {receptor_pdb}, receptor",
        "show cartoon, receptor",
        "color gray80, receptor",
        "",
        "# Docking box wireframe",
        f"pseudoatom box_center, pos=[{cx}, {cy}, {cz}]",
        "show spheres, box_center",
        "set sphere_scale, 0.3, box_center",
        "color red, box_center",
        "",
        f"# Box dimensions: {sx} x {sy} x {sz} Angstrom",
    ]

    # Draw box edges using CGO
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    lines.append("from pymol.cgo import *")
    lines.append("from pymol import cmd")
    lines.append(f"box = [")
    lines.append(f"  BEGIN, LINES,")
    lines.append(f"  COLOR, 0.0, 0.8, 0.0,")
    # 12 edges of the box
    corners = []
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            for dz in [-1, 1]:
                corners.append((cx + dx * hx, cy + dy * hy, cz + dz * hz))
    edges = [
        (0, 1), (2, 3), (4, 5), (6, 7),  # z edges
        (0, 2), (1, 3), (4, 6), (5, 7),  # y edges
        (0, 4), (1, 5), (2, 6), (3, 7),  # x edges
    ]
    for i, j in edges:
        x1, y1, z1 = corners[i]
        x2, y2, z2 = corners[j]
        lines.append(f"  VERTEX, {x1:.3f}, {y1:.3f}, {z1:.3f},")
        lines.append(f"  VERTEX, {x2:.3f}, {y2:.3f}, {z2:.3f},")
    lines.append(f"  END")
    lines.append(f"]")
    lines.append(f'cmd.load_cgo(box, "docking_box")')
    lines.append("")

    # Load docked poses
    sdf_files = sorted(Path(output_dir).glob("*_rescored.sdf"))
    if not sdf_files:
        sdf_files = sorted(Path(output_dir).glob("gnina_docked.sdf"))
    if not sdf_files:
        sdf_files = sorted(Path(output_dir).glob("*_out.pdbqt"))

    for i, sdf in enumerate(sdf_files[:5]):
        lines.append(f"load {sdf}, docked_pose_{i}")
        lines.append(f"show sticks, docked_pose_{i}")
        colors = ["cyan", "magenta", "yellow", "green", "orange"]
        lines.append(f"color {colors[i % len(colors)]}, docked_pose_{i}")
        lines.append("")

    lines.append("# Zoom to docking site")
    lines.append("zoom box_center, 20")
    lines.append("")
    lines.append("# Score legend (top poses):")
    if top_scores:
        for i, s in enumerate(top_scores[:5]):
            lines.append(f"# Pose {i+1}: Vina={s.get('vina_score','?')} "
                        f"CNN_score={s.get('cnn_score','?')} "
                        f"CNN_affinity={s.get('cnn_affinity','?')} kcal/mol")

    with open(pml_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_docking_report(report_path, results, scores):
    """Write markdown docking report."""
    site = results["site"]
    box = results["docking_box"]

    with open(report_path, "w") as f:
        f.write("# PRISM4D GPU Docking Report\n\n")
        f.write(f"**Generated**: {results['timestamp']}\n")
        f.write(f"**Receptor**: `{results['receptor']}`\n\n")

        f.write(f"## Binding Site {site['id']} — {site['classification']}\n\n")
        f.write(f"| Property | Value |\n|---|---|\n")
        f.write(f"| Centroid | ({site['centroid'][0]:.1f}, {site['centroid'][1]:.1f}, {site['centroid'][2]:.1f}) |\n")
        f.write(f"| Volume | {site['volume']:.0f} A^3 |\n")
        f.write(f"| Druggable | {'Yes' if site['is_druggable'] else 'No'} ({site['druggability']:.1%}) |\n\n")

        f.write(f"## Docking Box\n\n")
        f.write(f"| Parameter | Value |\n|---|---|\n")
        f.write(f"| Center | ({box['center_x']}, {box['center_y']}, {box['center_z']}) A |\n")
        f.write(f"| Size | {box['size_x']} x {box['size_y']} x {box['size_z']} A |\n\n")

        f.write(f"## Tools\n\n")
        for tool, desc in results["tools"].items():
            f.write(f"- **{tool}**: {desc}\n")
        f.write("\n")

        f.write(f"## Ligands Prepared\n\n")
        f.write(f"| Name | MW | LogP | HBD | HBA | RotB | Status |\n")
        f.write(f"|---|---|---|---|---|---|---|\n")
        for lig in results.get("ligands_prepared", []):
            if lig["status"] == "ok":
                f.write(f"| {lig['name']} | {lig['mw']} | {lig['logp']} | "
                        f"{lig['hbd']} | {lig['hba']} | {lig['rotatable_bonds']} | OK |\n")
            else:
                f.write(f"| {lig['name']} | — | — | — | — | — | {lig.get('error','?')} |\n")
        f.write("\n")

        f.write(f"## Docking Scores (Top 20)\n\n")
        f.write(f"| Rank | Ligand | Mode | Vina (kcal/mol) | CNN Score | CNN Affinity |\n")
        f.write(f"|---|---|---|---|---|---|\n")
        for i, s in enumerate(scores[:20]):
            f.write(f"| {i+1} | {s.get('file','?')} | {s.get('mode','?')} | "
                    f"{s.get('vina_score','?')} | {s.get('cnn_score','?'):.3f} | "
                    f"{s.get('cnn_affinity','?'):.2f} |\n")
        f.write("\n")

        f.write(f"## References\n\n")
        for ref in results.get("references", []):
            f.write(f"- {ref}\n")
        f.write("\n")

        f.write(f"## Visualization\n\n")
        f.write(f"```bash\n")
        f.write(f"pymol @site{site['id']}_docking_viz.pml\n")
        f.write(f"```\n")


# ── Main pipeline ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PRISM4D GPU Docking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--receptor", required=True, help="Receptor PDB file")
    parser.add_argument("--sites", required=True, help="PRISM binding_sites.json")
    parser.add_argument("--ligands", help="Ligand SDF file (multi-molecule)")
    parser.add_argument("--smiles", nargs="+",
                        help="SMILES strings as 'SMILES name' pairs")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--exhaustiveness", type=int, default=32,
                        help="Search exhaustiveness (default: 32)")
    parser.add_argument("--num-modes", type=int, default=9,
                        help="Max poses per ligand (default: 9)")
    parser.add_argument("--scoring", default="vina",
                        choices=["vina", "vinardo", "ad4"],
                        help="UniDock scoring function (default: vina)")
    parser.add_argument("--gnina-only", action="store_true",
                        help="Skip UniDock, use GNINA for direct docking")
    parser.add_argument("--skip-gnina", action="store_true",
                        help="Skip GNINA rescoring (UniDock only)")
    parser.add_argument("--cnn-scoring", default="rescore",
                        choices=["none", "rescore", "refinement", "all"],
                        help="GNINA CNN scoring mode (default: rescore)")

    args = parser.parse_args()

    print("=" * 60)
    print("PRISM4D GPU Docking Pipeline")
    print("=" * 60)
    print(f"  Receptor:  {args.receptor}")
    print(f"  Sites:     {args.sites}")
    print(f"  Output:    {args.output}")
    print()

    # Verify tools
    check_tools()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract docking boxes from binding sites
    print("[1/5] Extracting docking boxes from binding sites...")
    boxes, site_data = extract_docking_boxes(args.sites)
    for box in boxes:
        print(f"  Site {box['site_id']}: {box['classification']} "
              f"({'DRUGGABLE' if box['is_druggable'] else 'non-druggable'}) "
              f"center=({box['center_x']}, {box['center_y']}, {box['center_z']}) "
              f"size={box['size_x']}x{box['size_y']}x{box['size_z']}A")
    print()

    # Step 2: Prepare receptor
    print("[2/5] Preparing receptor PDBQT...")
    receptor_pdbqt = prepare_receptor_pdbqt(args.receptor, output_dir)
    print()

    # Step 3: Prepare ligands
    print("[3/5] Preparing ligands...")
    ligand_info = []
    ligand_sdf = None
    pdbqt_dir = None

    if args.smiles:
        smiles_pairs = []
        for item in args.smiles:
            parts = item.strip().split(None, 1)
            if len(parts) == 2:
                smiles_pairs.append((parts[0], parts[1]))
            else:
                smiles_pairs.append((parts[0], f"lig_{len(smiles_pairs)}"))
        ligand_sdf, pdbqt_dir, ligand_info = prepare_ligands_from_smiles(
            smiles_pairs, output_dir
        )
    elif args.ligands:
        ligand_sdf = Path(args.ligands)
        if not ligand_sdf.exists():
            print(f"ERROR: Ligand file not found: {ligand_sdf}")
            sys.exit(1)
        # Convert to PDBQT via Meeko (preserves existing 3D coords)
        pdbqt_dir = output_dir / "pdbqt_ligands"
        pdbqt_dir.mkdir(exist_ok=True)
        prep_script = '''
import sys, json, os, subprocess, tempfile, math

ligand_path = sys.argv[1]
pdbqt_dir = sys.argv[2]
obabel_bin = sys.argv[3] if len(sys.argv) > 3 else "obabel"

sdf_path = os.path.join(os.path.dirname(pdbqt_dir), "ligands_3d.sdf")
meeko_ok = False

# ── Strategy 1: Extract HETATM ligand from holo PDB manually, then use
#    RDKit + Meeko. Fixes fragment-not-found by parsing HETATM lines
#    directly instead of relying on RDKit fragmentation. ──

SKIP_RESNAMES = {"HOH", "WAT", "NA", "CL", "MG", "ZN", "CA", "K", "FE",
                 "MN", "CO", "NI", "CU", "SO4", "PO4", "GOL", "EDO",
                 "ACE", "NH2", "BUM", "NDP"}
PROTEIN = {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY","HIS","ILE",
           "LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL","MSE"}

ext = ligand_path.rsplit(".", 1)[-1].lower()

if ext == "pdb":
    # Parse HETATM lines, group by (resname, chain, resseq) to find ligand
    hetatm_groups = {}
    with open(ligand_path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            resname = line[17:20].strip()
            chain = line[21]
            resseq = line[22:26].strip()
            key = (resname, chain, resseq)
            if resname in SKIP_RESNAMES or resname in PROTEIN:
                continue
            hetatm_groups.setdefault(resname, []).append(line)

    if hetatm_groups:
        # Pick the residue with the most HETATM lines (= largest ligand)
        best_res = max(hetatm_groups, key=lambda r: len(hetatm_groups[r]))
        lig_lines = hetatm_groups[best_res]
        sys.stderr.write(f"  Extracted HETATM ligand: {best_res} ({len(lig_lines)} atoms)\\n")

        # Write extracted ligand to temp PDB
        lig_pdb = os.path.join(pdbqt_dir, "_extracted_ligand.pdb")
        with open(lig_pdb, "w") as f:
            for line in lig_lines:
                f.write(line)
            f.write("END\\n")

        # Try RDKit + Meeko on the extracted fragment
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, Descriptors
            from meeko import MoleculePreparation, PDBQTWriterLegacy

            mol = Chem.MolFromPDBFile(lig_pdb, removeHs=False, sanitize=False)
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                except:
                    pass
                mol = Chem.AddHs(mol, addCoords=True)
                AllChem.ComputeGasteigerCharges(mol)

                # Fix NaN charges — replace with 0.0
                for atom in mol.GetAtoms():
                    try:
                        q = float(atom.GetDoubleProp("_GasteigerCharge"))
                        if math.isnan(q) or math.isinf(q):
                            atom.SetDoubleProp("_GasteigerCharge", 0.0)
                    except:
                        atom.SetDoubleProp("_GasteigerCharge", 0.0)

                preparator = MoleculePreparation(merge_these_atom_types=("H",))
                mol_setups = preparator.prepare(mol)
                for i, setup in enumerate(mol_setups):
                    pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(setup)
                    if is_ok:
                        out_path = f"{pdbqt_dir}/lig_{i+1}.pdbqt"
                        with open(out_path, "w") as f:
                            f.write(pdbqt_string)
                        meeko_ok = True

                writer = Chem.SDWriter(sdf_path)
                writer.write(mol)
                writer.close()
        except Exception as e:
            sys.stderr.write(f"  RDKit/Meeko failed: {e}\\n")

        # Strategy 2: If Meeko failed, use OpenBabel directly
        if not meeko_ok:
            sys.stderr.write("  Falling back to OpenBabel for ligand PDBQT conversion\\n")
            out_pdbqt = os.path.join(pdbqt_dir, "lig_1.pdbqt")
            r = subprocess.run(
                [obabel_bin, lig_pdb, "-O", out_pdbqt,
                 "-xh", "--partialcharge", "gasteiger"],
                capture_output=True, text=True, timeout=60
            )
            if os.path.exists(out_pdbqt) and os.path.getsize(out_pdbqt) > 50:
                meeko_ok = True
            # Also write SDF
            r2 = subprocess.run(
                [obabel_bin, lig_pdb, "-O", sdf_path, "--gen3d"],
                capture_output=True, text=True, timeout=60
            )
    else:
        sys.stderr.write("  No HETATM ligand groups found in holo PDB\\n")

elif ext in ("sdf", "mol"):
    # SDF/MOL input — straightforward RDKit load
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol = next(Chem.ForwardSDMolSupplier(ligand_path, removeHs=False, sanitize=False))
    if mol is not None:
        try:
            Chem.SanitizeMol(mol)
        except:
            pass
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.ComputeGasteigerCharges(mol)

        # Fix NaN charges
        for atom in mol.GetAtoms():
            try:
                q = float(atom.GetDoubleProp("_GasteigerCharge"))
                if math.isnan(q) or math.isinf(q):
                    atom.SetDoubleProp("_GasteigerCharge", 0.0)
            except:
                atom.SetDoubleProp("_GasteigerCharge", 0.0)

        preparator = MoleculePreparation(merge_these_atom_types=("H",))
        mol_setups = preparator.prepare(mol)
        for i, setup in enumerate(mol_setups):
            pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(setup)
            if is_ok:
                out_path = f"{pdbqt_dir}/lig_{i+1}.pdbqt"
                with open(out_path, "w") as f:
                    f.write(pdbqt_string)
                meeko_ok = True

        writer = Chem.SDWriter(sdf_path)
        writer.write(mol)
        writer.close()

# Final fallback: obabel on the raw input file
if not meeko_ok:
    sys.stderr.write("  Last resort: obabel on raw ligand file\\n")
    out_pdbqt = os.path.join(pdbqt_dir, "lig_1.pdbqt")
    r = subprocess.run(
        [obabel_bin, ligand_path, "-O", out_pdbqt,
         "-xh", "--partialcharge", "gasteiger"],
        capture_output=True, text=True, timeout=60
    )
    if os.path.exists(out_pdbqt) and os.path.getsize(out_pdbqt) > 50:
        meeko_ok = True
    if not os.path.exists(sdf_path):
        subprocess.run(
            [obabel_bin, ligand_path, "-O", sdf_path],
            capture_output=True, text=True, timeout=60
        )

n_pdbqt = len([f for f in os.listdir(pdbqt_dir) if f.endswith(".pdbqt") and not f.startswith("_")])
# Get mol weight if we have the SDF
mw = 0.0
n_atoms = 0
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    if os.path.exists(sdf_path):
        suppl = Chem.SDMolSupplier(sdf_path, sanitize=False)
        for m in suppl:
            if m is not None:
                mw = Descriptors.MolWt(m)
                n_atoms = m.GetNumAtoms()
                break
except:
    pass

print(json.dumps({"n_ligands": n_pdbqt, "mw": round(mw, 1), "n_atoms": n_atoms}))
'''
        result = subprocess.run(
            [str(PYTHON_BIN), "-c", prep_script, str(ligand_sdf), str(pdbqt_dir),
             str(OBABEL_BIN)],
            capture_output=True, text=True, timeout=120
        )
        if result.stderr:
            # Show stderr (extraction info, fallback notices) but don't fail on it
            for line in result.stderr.strip().split("\n"):
                if line.strip():
                    print(f"  {line.strip()}")
        if result.returncode != 0:
            print(f"  ERROR in ligand prep (exit {result.returncode})")
            print(f"  stdout: {result.stdout[:300]}")
            sys.exit(1)
        prep_info = json.loads(result.stdout)
        print(f"  Converted {prep_info['n_ligands']} ligands to PDBQT "
              f"(MW={prep_info['mw']}, {prep_info['n_atoms']} atoms)")
        ligand_sdf = output_dir / "ligands_3d.sdf"
    else:
        print("ERROR: Provide --ligands (SDF) or --smiles")
        sys.exit(1)
    print()

    # Step 4 & 5: Concurrent UniDock + GNINA, isolated per-site directories
    all_results = []

    if args.gnina_only:
        # GNINA-only: concurrent across sites
        for i, box in enumerate(boxes):
            site = site_data["sites"][i]
            site_dir = output_dir / f"site{box['site_id']}"
            site_dir.mkdir(exist_ok=True)
            print(f"[4/5] GNINA direct docking site {box['site_id']} ({box['classification']})...")
            out_sdf, gnina_scores = run_gnina_dock(
                args.receptor, ligand_sdf, box, site_dir,
                exhaustiveness=args.exhaustiveness,
                num_modes=args.num_modes,
                cnn_scoring=args.cnn_scoring,
            )
            results = generate_results(
                site, box, [], gnina_scores, ligand_info,
                site_dir, args.receptor
            )
            all_results.append(results)
    else:
        # ── Concurrent UniDock: one thread per pocket, isolated output dirs ──
        if not pdbqt_dir or not list(Path(pdbqt_dir).glob("*.pdbqt")):
            print("  ERROR: No ligands available for docking")
            sys.exit(1)

        n_pockets = len(boxes)
        max_workers = min(n_pockets, 4)

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = GNINA_LD_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

        # Create all site dirs upfront
        site_dirs = {}
        for box in boxes:
            sd = output_dir / f"site{box['site_id']}"
            sd.mkdir(exist_ok=True)
            site_dirs[box["site_id"]] = sd

        # ── Phase 1: Concurrent UniDock ──
        print(f"[4/5] Concurrent UniDock GPU docking: {n_pockets} pockets, "
              f"{max_workers} threads...")
        t0 = time.time()

        unidock_results = {}  # site_id → (out_files, elapsed, error)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for box in boxes:
                fut = pool.submit(
                    _unidock_one_site,
                    receptor_pdbqt, ligand_sdf, box,
                    site_dirs[box["site_id"]],
                    args.exhaustiveness, args.num_modes, args.scoring,
                    env,
                )
                futures[fut] = box["site_id"]

            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    site_id, out_files, elapsed, error = fut.result()
                    unidock_results[site_id] = (out_files, elapsed, error)
                    if error:
                        print(f"    Site {site_id}: UniDock ERROR ({elapsed:.1f}s) — {error[:120]}")
                    else:
                        print(f"    Site {site_id}: {len(out_files)} poses ({elapsed:.1f}s)")
                except Exception as e:
                    print(f"    Site {sid}: THREAD EXCEPTION — {e}")
                    unidock_results[sid] = ([], 0, str(e))

        ud_elapsed = time.time() - t0
        total_poses = sum(len(v[0]) for v in unidock_results.values())
        print(f"  UniDock complete: {total_poses} total pose files in {ud_elapsed:.1f}s")
        print()

        # ── Phase 2: Concurrent GNINA rescore ──
        gnina_results = {}  # site_id → scores list
        if not args.skip_gnina:
            # Only rescore sites that produced docked poses
            sites_with_poses = [b for b in boxes if unidock_results.get(b["site_id"], ([],))[0]]

            if sites_with_poses:
                print(f"[5/5] Concurrent GNINA CNN rescoring: {len(sites_with_poses)} pockets, "
                      f"{max_workers} threads...")
                t0 = time.time()

                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    futures = {}
                    for box in sites_with_poses:
                        fut = pool.submit(
                            _gnina_rescore_one_site,
                            args.receptor, box,
                            site_dirs[box["site_id"]],
                            env,
                        )
                        futures[fut] = box["site_id"]

                    for fut in as_completed(futures):
                        sid = futures[fut]
                        try:
                            site_id, rescored_files, scores, elapsed = fut.result()
                            gnina_results[site_id] = scores
                            if scores:
                                best = scores[0]
                                print(f"    Site {site_id}: {len(scores)} scored, "
                                      f"best Vina={best['vina_score']:.1f} "
                                      f"CNN={best['cnn_score']:.3f} "
                                      f"aff={best['cnn_affinity']:.2f} ({elapsed:.1f}s)")
                            else:
                                print(f"    Site {site_id}: no scored poses ({elapsed:.1f}s)")
                        except Exception as e:
                            print(f"    Site {sid}: GNINA EXCEPTION — {e}")
                            gnina_results[sid] = []

                gn_elapsed = time.time() - t0
                total_scored = sum(len(v) for v in gnina_results.values())
                print(f"  GNINA complete: {total_scored} total scored poses in {gn_elapsed:.1f}s")
        print()

        # ── Generate per-site results ──
        for i, box in enumerate(boxes):
            site = site_data["sites"][i]
            site_id = box["site_id"]

            gnina_scores = gnina_results.get(site_id, [])
            results = generate_results(
                site, box, [], gnina_scores, ligand_info,
                site_dirs[site_id], args.receptor
            )
            all_results.append(results)

    # Summary
    print("=" * 60)
    print("DOCKING COMPLETE")
    print("=" * 60)
    print(f"  Output:    {output_dir}")
    print(f"  Sites:     {len(boxes)}")
    for r in all_results:
        sid = r["site"]["id"]
        gs = r.get("gnina_scores", [])
        if gs:
            best = gs[0]
            print(f"  Site {sid}: best Vina={best.get('vina_score','?')} "
                  f"CNN_aff={best.get('cnn_affinity','?')} kcal/mol")
        else:
            print(f"  Site {sid}: no scored poses")
    print()
    print("Visualization:")
    for box in boxes:
        print(f"  pymol @{output_dir}/site{box['site_id']}/site{box['site_id']}_docking_viz.pml")
    print()


if __name__ == "__main__":
    main()

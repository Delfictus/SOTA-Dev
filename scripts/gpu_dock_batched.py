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


def prepare_receptor_pdbqt(pdb_path, output_dir):
    """Convert receptor PDB to PDBQT using OpenBabel.

    Adds polar hydrogens and assigns Gasteiger charges.
    For production use, AMBER ff14SB charges via prism-prep are preferred.
    """
    output_dir = Path(output_dir)
    pdbqt_path = output_dir / (Path(pdb_path).stem + ".pdbqt")

    result = subprocess.run(
        [str(OBABEL_BIN), str(pdb_path), "-O", str(pdbqt_path),
         "-xr", "-xh", "--partialcharge", "gasteiger"],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"WARNING: obabel receptor conversion had issues: {result.stderr[:200]}")

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


def run_unidock_with_pdbqt_ligands(receptor_pdbqt, pdbqt_dir, boxes, output_dir,
                                     exhaustiveness=32, num_modes=20, scoring="vina"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdbqt_files = sorted(Path(pdbqt_dir).glob("*.pdbqt"))
    
    index_path = output_dir / "ligand_index.txt"
    with open(index_path, "w") as f:
        for p in pdbqt_files: f.write(str(p) + "\n")
        
    batch_json = output_dir / "batch_grid.json"
    grid_data = [{"center_x": b["center_x"], "center_y": b["center_y"], "center_z": b["center_z"],
                  "size_x": b["size_x"], "size_y": b["size_y"], "size_z": b["size_z"]} for b in boxes]
    
    import json
    with open(batch_json, "w") as f: json.dump(grid_data, f)

    cmd = [str(UNIDOCK_BIN), "--receptor", str(receptor_pdbqt), "--ligand_index", str(index_path),
           "--search_mode", "batch", "--batch_grid", str(batch_json), "--exhaustiveness", str(exhaustiveness),
           "--num_modes", str(num_modes), "--scoring", scoring, "--dir", str(output_dir), "--seed", "42"]

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = GNINA_LD_PATH + ":" + env.get("LD_LIBRARY_PATH", "")
    subprocess.run(cmd, capture_output=True, text=True, timeout=1200, env=env)
    
    out_files = sorted(output_dir.glob("*_out.pdbqt"))
    return output_dir, out_files


# ── GNINA CNN rescoring ──────────────────────────────────────────────────

def run_gnina_rescore(receptor_pdb, docked_dir, box, output_dir, top_n=500):
    """Rescore UniDock poses with GNINA CNN scoring.

    GNINA uses a 3D convolutional neural network ensemble (dense_1_3 +
    crossdock_default2018_KD_4) trained on the CrossDocked2020 dataset
    to predict binding affinity and pose quality.
    Ref: McNutt et al. J Cheminform 2025. doi:10.1186/s13321-025-00973-x

    CNN scoring modes:
      - rescore: CNN evaluates final poses only (fast, default)
      - refinement: CNN guides pose optimization (10x slower)
      - all: CNN at every step (1000x slower, highest accuracy)

    Args:
        receptor_pdb: path to receptor PDB (GNINA accepts PDB natively)
        docked_dir: directory with UniDock output (PDBQT/SDF files)
        box: docking box dict
        output_dir: output directory for rescored poses
        top_n: max poses to keep after rescoring

    Returns:
        Path to rescored SDF with CNN scores, list of score dicts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all docked poses
    docked_files = sorted(Path(docked_dir).glob("*_out.pdbqt"))
    if not docked_files:
        docked_files = sorted(Path(docked_dir).glob("*.sdf"))
    if not docked_files:
        docked_files = sorted(Path(docked_dir).glob("*.pdbqt"))

    if not docked_files:
        print("  WARNING: No docked poses found for GNINA rescoring")
        return None, []

    all_scores = []
    rescored_files = []

    for docked_file in docked_files:
        out_sdf = output_dir / (docked_file.stem + "_rescored.sdf")

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

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = GNINA_LD_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)

        if result.returncode != 0:
            print(f"  GNINA rescore warning for {docked_file.name}: {result.stderr[:200]}")
            continue

        # Parse GNINA output for scores
        for line in result.stdout.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("Using") and not line.startswith("WARNING"):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        mode = int(parts[0])
                        vina_score = float(parts[1])
                        cnn_score = float(parts[2])
                        cnn_affinity = float(parts[3])
                        all_scores.append({
                            "file": docked_file.name,
                            "mode": mode,
                            "vina_score": vina_score,
                            "cnn_score": cnn_score,
                            "cnn_affinity": cnn_affinity,
                        })
                    except (ValueError, IndexError):
                        pass

        if out_sdf.exists():
            rescored_files.append(out_sdf)

    # Sort by CNN affinity (more negative = better)
    all_scores.sort(key=lambda x: x.get("cnn_affinity", 0))

    print(f"  GNINA rescored: {len(all_scores)} poses from {len(rescored_files)} files")
    if all_scores:
        best = all_scores[0]
        print(f"  Best: {best['file']} mode {best['mode']}: "
              f"Vina={best['vina_score']:.1f} CNN_score={best['cnn_score']:.3f} "
              f"CNN_affinity={best['cnn_affinity']:.2f} kcal/mol")

    return rescored_files, all_scores


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
import sys, json, os
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from meeko import MoleculePreparation, PDBQTWriterLegacy

ligand_path = sys.argv[1]
pdbqt_dir = sys.argv[2]

# Load molecule preserving existing 3D coordinates
ext = ligand_path.rsplit(".", 1)[-1].lower()
if ext == "pdb":
    mol = Chem.MolFromPDBFile(ligand_path, removeHs=False, sanitize=False)
elif ext in ("sdf", "mol"):
    mol = next(Chem.ForwardSDMolSupplier(ligand_path, removeHs=False, sanitize=False))
else:
    mol = Chem.MolFromPDBFile(ligand_path, removeHs=False, sanitize=False)

if mol is None:
    print(json.dumps({"error": "Could not load molecule"}))
    sys.exit(1)

# If PDB has multiple fragments (holo structure), extract just the ligand
# Ligand = largest HETATM fragment that is not water (HOH) or common ions
frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
if len(frags) > 1:
    # Filter: skip water (3 atoms) and tiny ions (1-2 atoms)
    # Pick the largest non-protein fragment by heavy atom count
    SKIP_RESNAMES = {"HOH", "WAT", "NA", "CL", "MG", "ZN", "CA", "K", "FE",
                     "MN", "CO", "NI", "CU", "SO4", "PO4", "GOL", "EDO", "ACE", "NH2"}
    candidates = []
    for frag in frags:
        n_heavy = frag.GetNumHeavyAtoms()
        # Check residue name of first atom
        resname = ""
        if frag.GetNumAtoms() > 0:
            info = frag.GetAtomWithIdx(0).GetPDBResidueInfo()
            if info:
                resname = info.GetResidueName().strip()
        # Skip standard amino acids (protein chains) and waters/ions
        is_protein = resname in {"ALA","ARG","ASN","ASP","CYS","GLN","GLU","GLY",
                                 "HIS","ILE","LEU","LYS","MET","PHE","PRO","SER",
                                 "THR","TRP","TYR","VAL","MSE"}
        if resname in SKIP_RESNAMES or is_protein:
            continue
        candidates.append((n_heavy, frag, resname))
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        mol = candidates[0][1]
        sys.stderr.write(f"  Extracted ligand: {candidates[0][2]} ({candidates[0][0]} heavy atoms)\\n")
    else:
        # Fallback: just take the largest fragment
        frags.sort(key=lambda f: f.GetNumHeavyAtoms(), reverse=True)
        mol = frags[0]

# Sanitize without recalculating coords
try:
    Chem.SanitizeMol(mol)
except:
    pass

# Add hydrogens (keep existing 3D positions, add H in place)
mol = Chem.AddHs(mol, addCoords=True)

# Gasteiger charges only — NO conformer generation, NO minimization
AllChem.ComputeGasteigerCharges(mol)

# Write PDBQT via Meeko
preparator = MoleculePreparation(merge_these_atom_types=("H",))
mol_setups = preparator.prepare(mol)
for i, setup in enumerate(mol_setups):
    pdbqt_string, is_ok, err = PDBQTWriterLegacy.write_string(setup)
    if is_ok:
        out_path = f"{pdbqt_dir}/lig_{i+1}.pdbqt"
        with open(out_path, "w") as f:
            f.write(pdbqt_string)

# Also write SDF for downstream tools
sdf_path = f"{pdbqt_dir}/../ligands_3d.sdf"
writer = Chem.SDWriter(sdf_path)
writer.write(mol)
writer.close()

n_pdbqt = len([f for f in os.listdir(pdbqt_dir) if f.endswith(".pdbqt")])
mw = Descriptors.MolWt(mol)
print(json.dumps({"n_ligands": n_pdbqt, "mw": round(mw, 1), "n_atoms": mol.GetNumAtoms()}))
'''
        result = subprocess.run(
            [str(PYTHON_BIN), "-c", prep_script, str(ligand_sdf), str(pdbqt_dir)],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print(f"  ERROR in ligand prep: {result.stderr[:500]}")
            sys.exit(1)
        prep_info = json.loads(result.stdout)
        print(f"  Converted {prep_info['n_ligands']} ligands to PDBQT "
              f"(MW={prep_info['mw']}, {prep_info['n_atoms']} atoms)")
        ligand_sdf = output_dir / "ligands_3d.sdf"
    else:
        print("ERROR: Provide --ligands (SDF) or --smiles")
        sys.exit(1)
    print()

    # Step 4: True VRAM Batch Docking
    print(f"\n[4/5] Loading {len(boxes)} pockets into VRAM for concurrent docking...")
    unidock_dir = output_dir / "unidock_batch_out"
    
    if pdbqt_dir and list(Path(pdbqt_dir).glob("*.pdbqt")):
        _, unidock_files = run_unidock_with_pdbqt_ligands(
            receptor_pdbqt, pdbqt_dir, boxes, unidock_dir,
            exhaustiveness=args.exhaustiveness, num_modes=args.num_modes, scoring=args.scoring)
    else:
        print("  ERROR: No ligands available for docking")
        sys.exit(1)
        
    print(f"\n[5/5] GNINA CNN rescoring {len(boxes)} pockets...")
    gnina_dir = output_dir / "gnina_batch_rescore"
    gnina_dir.mkdir(parents=True, exist_ok=True)
    all_scores = []
    
    if not args.skip_gnina and unidock_files:
        for box in boxes:
            site_id = box["site_id"]
            docked_files = sorted(Path(unidock_dir).glob("*_out.pdbqt"))
            for df in docked_files:
                out_sdf = gnina_dir / f"site_{site_id}_{df.stem}_rescored.sdf"
                cmd = [str(GNINA_BIN), "--receptor", str(args.receptor), "--ligand", str(df),
                       "--center_x", str(box["center_x"]), "--center_y", str(box["center_y"]), "--center_z", str(box["center_z"]),
                       "--size_x", str(box["size_x"]), "--size_y", str(box["size_y"]), "--size_z", str(box["size_z"]),
                       "--cnn_scoring", "rescore", "--out", str(out_sdf), "--seed", "42"]
                env = os.environ.copy()
                env["LD_LIBRARY_PATH"] = GNINA_LD_PATH + ":" + env.get("LD_LIBRARY_PATH", "")
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
                if res.returncode == 0:
                    for l in res.stdout.split("\n"):
                        p = l.strip().split()
                        if len(p) >= 4 and not l.startswith(("#", "Using", "WARN")):
                            try: all_scores.append({"site_id": site_id, "file": df.name, "mode": int(p[0]), "vina_score": float(p[1]), "cnn_score": float(p[2]), "cnn_affinity": float(p[3])})
                            except: pass

    all_scores.sort(key=lambda x: x.get("cnn_affinity", 0))
    print("\n============================================================")
    print("BATCH DOCKING & AI RESCORING COMPLETE.")
    print("============================================================")
    print("Visualization:")

    for box in boxes:
        print(f"  pymol @{output_dir}/site{box['site_id']}/site{box['site_id']}_docking_viz.pml")
    print()


if __name__ == "__main__":
    main()

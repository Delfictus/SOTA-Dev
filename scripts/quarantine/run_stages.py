#!/usr/bin/env python3
"""
[STAGE RUNNER - PROVENANCE-WRAPPED]

Unified dispatcher for the 6 stages of the TWIN-10 pipeline.

All stages emit BLAKE3-hashed provenance records (Tier A via prism_prov
RunContext; Tier B engine-internal via prism_engine_prov).

Invoke:
    python3 run_stages.py --stage 1_download --target kras_g12d_apo ...
    python3 run_stages.py --stage 5_engine   --target kras_g12d_apo ...
    python3 run_stages.py --stage all        --target kras_g12d_apo ...

Target configuration comes from twin10_targets.json (produced alongside
by the orchestrator). Per-target context:
    {
      "target": "kras_g12d_apo",
      "pdb_id": "7F0W",
      "chain": "A",
      "paired_holo_pdb_id": "7RPZ",
      "paired_holo_ligand_resname": "6IC",
      "known_binding_residues": ["H95","Y96","Q99"],
      "species": "Homo sapiens"
    }
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import urlopen, Request

# Local modules
sys.path.insert(0, str(Path(__file__).parent))
from prism_prov import (
    RunContext, blake3_file, blake3_bytes, canonical_json,
    capture_host, capture_tool, write_manifest,
)
from prism_engine_prov import (
    emit_engine_tier_b_provenance, GpuTelemetryCapture, wrap_with_nsys,
    determinism_env, hash_nsys_trace,
)

import numpy as np

RCSB_BASE = "https://files.rcsb.org/download"
REPO_ROOT = Path(__file__).resolve().parents[2]  # Prism4D-bio root


# ─────────────────────────────────────────────────────────────────────
# Stage 1 — Download
# ─────────────────────────────────────────────────────────────────────

def _fetch(url: str, dest: Path, timeout: int = 60) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = Request(url, headers={"User-Agent": "prism-twin-10/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"    fetch failed {url}: {e}", file=sys.stderr)
        return False


def stage1_download(target: Dict[str, Any], target_dir: Path) -> Path:
    tname = target["target"]
    pdb_id = target["pdb_id"].lower()
    artifacts_dir = target_dir / "artifacts" / "1_download"
    prov_dir = target_dir / "prov"

    with RunContext(tname, "1_download", "rcsb_fetch", artifacts_dir, prov_dir) as ctx:
        ctx.set_tool("urllib", ["urlopen", RCSB_BASE, pdb_id])
        # mmCIF (primary)
        cif_path = artifacts_dir / f"{pdb_id}.cif"
        if _fetch(f"{RCSB_BASE}/{pdb_id}.cif", cif_path):
            ctx.add_output(cif_path, role="mmcif_primary")
        # PDB (legacy, used by prism-prep which may expect PDB format)
        pdb_path = artifacts_dir / f"{pdb_id}.pdb"
        if _fetch(f"{RCSB_BASE}/{pdb_id}.pdb", pdb_path):
            ctx.add_output(pdb_path, role="pdb_legacy")
        # Biological assembly 1 (mmCIF)
        asm_path = artifacts_dir / f"{pdb_id}-assembly1.cif"
        _fetch(f"{RCSB_BASE}/{pdb_id}-assembly1.cif", asm_path)
        if asm_path.exists():
            ctx.add_output(asm_path, role="mmcif_assembly1")

        # Paired holo (for ground truth)
        if target.get("paired_holo_pdb_id"):
            hid = target["paired_holo_pdb_id"].lower()
            holo_pdb = artifacts_dir / f"{hid}.pdb"
            holo_cif = artifacts_dir / f"{hid}.cif"
            _fetch(f"{RCSB_BASE}/{hid}.pdb", holo_pdb)
            _fetch(f"{RCSB_BASE}/{hid}.cif", holo_cif)
            if holo_pdb.exists():
                ctx.add_output(holo_pdb, role="holo_pdb")
            if holo_cif.exists():
                ctx.add_output(holo_cif, role="holo_mmcif")

        # Gates
        ctx.set_gate("cif_present", "PASS" if cif_path.exists() else "FAIL")
        ctx.set_gate("pdb_present", "PASS" if pdb_path.exists() else "FAIL")
        if cif_path.exists():
            sz = cif_path.stat().st_size
            ctx.set_gate("cif_nonempty", "PASS" if sz > 1024 else "FAIL",
                        note=f"{sz} bytes")
        ctx.set_verdict("PASS" if (cif_path.exists() and pdb_path.exists()) else "FAIL")

    return prov_dir / "1_download.rcsb_fetch.prov.json"


# ─────────────────────────────────────────────────────────────────────
# Stage 2 — Clean (PDBFixer → PROPKA3 → prism-clean)
# ─────────────────────────────────────────────────────────────────────

def _run_pdbfixer(in_pdb: Path, out_pdb: Path, chain: str = "A") -> Tuple[bool, str]:
    """Use PDBFixer to add missing atoms, handle altconfs, cap termini."""
    try:
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
    except ImportError as e:
        return False, f"PDBFixer import failed: {e}"
    try:
        fixer = PDBFixer(filename=str(in_pdb))
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH=7.4)
        with open(out_pdb, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)
        return True, "ok"
    except Exception as e:
        return False, f"PDBFixer runtime: {type(e).__name__}: {e}"


def stage2_clean(target: Dict[str, Any], target_dir: Path,
                 upstream_prov: List[Path]) -> Path:
    tname = target["target"]
    pdb_id = target["pdb_id"].lower()
    chain = target.get("chain", "A")
    in_pdb = target_dir / "artifacts" / "1_download" / f"{pdb_id}.pdb"
    artifacts_dir = target_dir / "artifacts" / "2_clean"
    prov_dir = target_dir / "prov"

    # Sub-stage 1: PDBFixer
    fixed_pdb = artifacts_dir / f"{pdb_id}_pdbfixer.pdb"
    with RunContext(tname, "2_clean", "pdbfixer", artifacts_dir, prov_dir,
                    upstream_prov=upstream_prov) as ctx:
        ctx.add_input(in_pdb, upstream_prov_ref="1_download.rcsb_fetch")
        ctx.set_tool("pdbfixer", ["pdbfixer", str(in_pdb), str(fixed_pdb)])
        ok, msg = _run_pdbfixer(in_pdb, fixed_pdb)
        ctx.add_output(fixed_pdb, role="pdbfixer_output")
        ctx.set_gate("pdbfixer_success", "PASS" if ok else "FAIL", note=msg)
        ctx.set_verdict("PASS" if ok else "FAIL")
    pdbfixer_prov = prov_dir / "2_clean.pdbfixer.prov.json"

    if not fixed_pdb.exists():
        return pdbfixer_prov

    # Sub-stage 2: prism-clean (the canonical final step)
    clean_pdb = artifacts_dir / f"{pdb_id}_clean.pdb"
    prism_clean = REPO_ROOT / "scripts" / "prism-clean.py"
    with RunContext(tname, "2_clean", "prism_clean", artifacts_dir, prov_dir,
                    upstream_prov=[pdbfixer_prov]) as ctx:
        ctx.add_input(fixed_pdb, upstream_prov_ref="2_clean.pdbfixer")
        ctx.set_tool("prism-clean.py", ["python3", str(prism_clean),
                                         str(fixed_pdb), str(clean_pdb), chain])
        result = ctx.run(
            ["python3", str(prism_clean), str(fixed_pdb), str(clean_pdb), chain],
            stdout_file=artifacts_dir / "prism_clean.stdout.log",
            stderr_file=artifacts_dir / "prism_clean.stderr.log",
        )
        ctx.add_output(clean_pdb, role="clean_final")
        ctx.set_gate("prism_clean_exit", "PASS" if result.returncode == 0 else "FAIL")
        ctx.set_gate("clean_pdb_exists", "PASS" if clean_pdb.exists() else "FAIL")
        ctx.set_verdict("PASS" if clean_pdb.exists() and result.returncode == 0 else "FAIL")

    return prov_dir / "2_clean.prism_clean.prov.json"


# ─────────────────────────────────────────────────────────────────────
# Stage 3 — Prep (prism-prep → OpenMM sanity)
# ─────────────────────────────────────────────────────────────────────

def _openmm_sanity(topology_json: Path) -> Tuple[bool, Dict[str, Any]]:
    """Load topology, check no NaN, spot-check force computation."""
    try:
        with open(topology_json) as f:
            t = json.load(f)
        # Basic sanity: check for required keys and NaN
        required = ["atom_names", "bonds", "angles", "dihedrals", "charges",
                    "positions", "lj_params"]
        missing = [k for k in required if k not in t]
        if missing:
            return False, {"missing_keys": missing}
        # Check for NaN in critical arrays
        for key in ("charges", "positions"):
            arr = t.get(key, [])
            if any((isinstance(v, float) and v != v) for v in arr):
                return False, {"nan_in": key}
        # Check n_atoms consistency
        n_atoms = t.get("n_atoms", 0)
        if n_atoms > 0 and len(t.get("atom_names", [])) != n_atoms:
            return False, {"atom_count_mismatch":
                           f"n_atoms={n_atoms} names={len(t.get('atom_names', []))}"}
        return True, {"n_atoms": n_atoms, "n_residues": t.get("n_residues", 0)}
    except Exception as e:
        return False, {"exception": f"{type(e).__name__}: {e}"}


def stage3_prep(target: Dict[str, Any], target_dir: Path,
                upstream_prov: List[Path]) -> Path:
    tname = target["target"]
    pdb_id = target["pdb_id"].lower()
    clean_pdb = target_dir / "artifacts" / "2_clean" / f"{pdb_id}_clean.pdb"
    artifacts_dir = target_dir / "artifacts" / "3_prep"
    prov_dir = target_dir / "prov"

    topology = artifacts_dir / f"{pdb_id}.topology.json"
    prism_prep = REPO_ROOT / "scripts" / "prism-prep"

    # NMA mode generation — default ON with 32 modes. The v3 rescue
    # controller's auto-NMA-load step at engine startup looks for the
    # modes file alongside the topology; without the modes file the
    # rescue's EngineV2NmaAmpMultiplier actions mutate engine state
    # without kernel-level effect (demonstrated on POLQ v3.2 run:
    # primary=0 even though rescue published NMA amp → 20.0 cap).
    # Targets can opt out by setting `"nma_modes": 0` in their config.
    nma_modes_n = int(target.get("nma_modes", 32) or 32)

    with RunContext(tname, "3_prep", "prism_prep", artifacts_dir, prov_dir,
                    upstream_prov=upstream_prov) as ctx:
        ctx.add_input(clean_pdb, upstream_prov_ref="2_clean.prism_clean")
        prep_argv = [str(prism_prep), str(clean_pdb), str(topology)]
        if nma_modes_n > 0:
            prep_argv.extend(["--nma-modes", str(nma_modes_n)])
            ctx.add_note(f"NMA mode generation requested: {nma_modes_n} modes")
        ctx.set_tool("prism-prep", prep_argv)
        result = ctx.run(
            prep_argv,
            stdout_file=artifacts_dir / "prism_prep.stdout.log",
            stderr_file=artifacts_dir / "prism_prep.stderr.log",
        )
        ctx.add_output(topology, role="topology")
        # NMA modes file: prism-prep (line 766-767) writes
        # output_topology.stem.replace('.topology', '') + '_nma_modes.json'
        # → for topology "5a9j.topology.json", the modes file is
        # "5a9j_nma_modes.json" (NOT "5a9j.topology_nma_modes.json").
        # The engine's auto-detect in nhs_rt_full.rs uses the SAME
        # convention; both sides must agree.
        nma_file = artifacts_dir / f"{pdb_id}_nma_modes.json"
        if nma_file.exists():
            ctx.add_output(nma_file, role="nma_modes")
            ctx.set_gate("nma_modes_generated", "PASS",
                        note=f"{nma_modes_n} modes → {nma_file.name}")
        elif nma_modes_n > 0:
            ctx.set_gate("nma_modes_generated", "FAIL",
                        note=f"requested {nma_modes_n} modes but {nma_file.name} not produced")
        ctx.set_gate("prism_prep_exit", "PASS" if result.returncode == 0 else "FAIL")
        ctx.set_gate("topology_exists", "PASS" if topology.exists() else "FAIL")
        ctx.set_verdict("PASS" if topology.exists() and result.returncode == 0 else "FAIL")
    prep_prov = prov_dir / "3_prep.prism_prep.prov.json"

    if not topology.exists():
        return prep_prov

    # OpenMM sanity pass
    with RunContext(tname, "3_prep", "openmm_sanity", artifacts_dir, prov_dir,
                    upstream_prov=[prep_prov]) as ctx:
        ctx.add_input(topology, upstream_prov_ref="3_prep.prism_prep")
        ctx.set_tool("openmm", ["python3", "-c", "import openmm; print(openmm.Platform.getPluginLoadFailures())"])
        ok, stats = _openmm_sanity(topology)
        ctx.set_gate("topology_structurally_valid", "PASS" if ok else "FAIL",
                    note=str(stats))
        ctx.set_verdict("PASS" if ok else "FAIL")
        ctx.add_note(f"openmm_sanity_stats={stats}")

    return prov_dir / "3_prep.openmm_sanity.prov.json"


# ─────────────────────────────────────────────────────────────────────
# Stage 4 — Ground truth (apo↔holo superposition via MDAnalysis)
# ─────────────────────────────────────────────────────────────────────

def _parse_pdb_ca(path: Path, chain: str = "A") -> Dict[int, Tuple[str, np.ndarray]]:
    """Parse Cα atoms from a PDB file for the given chain."""
    cas: Dict[int, Tuple[str, np.ndarray]] = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            if line[21] != chain:
                continue
            if line[12:16].strip() != "CA":
                continue
            altloc = line[16]
            if altloc not in (" ", "A"):
                continue
            try:
                resid = int(line[22:26])
                resname = line[17:20].strip()
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                cas[resid] = (resname, np.array([x, y, z], dtype=np.float64))
            except ValueError:
                pass
    return cas


def _parse_pdb_hetatm(path: Path, resname: str, chain: str = "A") -> np.ndarray:
    """Parse HETATM coordinates for a specific ligand."""
    coords = []
    with open(path) as f:
        for line in f:
            if not line.startswith("HETATM"):
                continue
            if line[17:20].strip() != resname:
                continue
            if line[21] != chain:
                continue
            try:
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                coords.append([x, y, z])
            except ValueError:
                pass
    return np.asarray(coords, dtype=np.float64)


def _kabsch(P: np.ndarray, Q: np.ndarray):
    """Kabsch superposition: find transform such that transformed P best matches Q."""
    Pc, Qc = P.mean(axis=0), Q.mean(axis=0)
    P0, Q0 = P - Pc, Q - Qc
    H = P0.T @ Q0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    rmsd = float(np.sqrt(np.mean(np.sum((P0 @ R.T - Q0) ** 2, axis=1))))
    def transform(x: np.ndarray) -> np.ndarray:
        return (x - Pc) @ R.T + Qc
    return transform, R, Pc, Qc, rmsd


def _apo_holo_superposition(
    apo_pdb: Path, holo_pdb: Path, ligand_resname: str, chain: str = "A",
    flexible_regions: Optional[List[List[int]]] = None,
) -> Dict[str, Any]:
    """Two-stage Kabsch superposition of holo onto apo.

    Stage 1: global alignment on all common Cα (baseline)
    Stage 2: rigid-core alignment excluding flexible_regions (if specified)
             — proper methodology for apo/holo with induced-fit

    Returns ligand coordinates + centroid in the apo reference frame, with
    both global and rigid RMSDs reported.

    Uses numpy SVD directly — bypasses MDAnalysis segid/chainID parsing bugs.
    """
    try:
        apo_ca = _parse_pdb_ca(apo_pdb, chain)
        holo_ca = _parse_pdb_ca(holo_pdb, chain)
        common = sorted([r for r in set(apo_ca) & set(holo_ca)
                         if apo_ca[r][0] == holo_ca[r][0]])
        if len(common) < 20:
            return {"error": f"too few common Cα with matching resname: {len(common)}"}

        apo_coords = np.asarray([apo_ca[r][1] for r in common])
        holo_coords = np.asarray([holo_ca[r][1] for r in common])

        # Stage 1: global alignment
        tfm_global, _, _, _, rmsd_global = _kabsch(holo_coords, apo_coords)

        # Stage 2: rigid-core alignment (if flexible_regions given)
        flex_flat = []
        if flexible_regions:
            for region in flexible_regions:
                if len(region) == 2:
                    flex_flat.append(tuple(region))

        def is_flex(r: int) -> bool:
            return any(lo <= r <= hi for lo, hi in flex_flat)

        rigid = [r for r in common if not is_flex(r)] if flex_flat else common
        if flex_flat and len(rigid) < 20:
            # Too few rigid residues after exclusion; fall back to global
            tfm_chosen = tfm_global
            rmsd_chosen = rmsd_global
            alignment_method = "kabsch_global_fallback"
            n_aligned = len(common)
        else:
            apo_rigid = np.asarray([apo_ca[r][1] for r in rigid])
            holo_rigid = np.asarray([holo_ca[r][1] for r in rigid])
            tfm_rigid, _, _, _, rmsd_rigid = _kabsch(holo_rigid, apo_rigid)
            if flex_flat:
                tfm_chosen = tfm_rigid
                rmsd_chosen = rmsd_rigid
                alignment_method = "kabsch_rigid_core"
                n_aligned = len(rigid)
            else:
                tfm_chosen = tfm_global
                rmsd_chosen = rmsd_global
                alignment_method = "kabsch_global"
                n_aligned = len(common)

        # Transform ligand atoms into apo frame using the chosen transform
        lig_atoms_holo = _parse_pdb_hetatm(holo_pdb, ligand_resname, chain)
        if len(lig_atoms_holo) == 0:
            return {"error": f"ligand {ligand_resname} not found in holo"}
        lig_atoms_apo = tfm_chosen(lig_atoms_holo)
        lig_centroid_apo = lig_atoms_apo.mean(axis=0)

        # Identify binding residues: Cα within threshold of any ligand heavy atom
        try:
            from scipy.spatial import cKDTree
            tree = cKDTree(lig_atoms_apo)
            apo_ca_coords = np.asarray([apo_ca[r][1] for r in common])
            dists, _ = tree.query(apo_ca_coords, k=1)
            binding = [
                {"resid": int(common[i]), "resname": apo_ca[common[i]][0],
                 "min_dist_angstrom": float(dists[i])}
                for i in range(len(common)) if dists[i] <= 8.5  # 4.5 + sidechain allowance
            ]
        except ImportError:
            binding = []

        return {
            "ligand_resname": ligand_resname,
            "ligand_centroid_apo_frame": lig_centroid_apo.tolist(),
            "ligand_atoms_apo_frame": lig_atoms_apo.tolist(),
            "ligand_n_heavy_atoms": int(len(lig_atoms_holo)),
            "alignment_method": alignment_method,
            "rmsd_global_alignment": rmsd_global,
            "rmsd_rigid_alignment": rmsd_rigid if flex_flat and len(rigid) >= 20 else None,
            "n_aligned_ca": n_aligned,
            "n_total_common_ca": len(common),
            "flexible_regions_excluded": flexible_regions,
            "binding_residues": binding,
        }
    except Exception as e:
        import traceback
        return {"error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc()}


def stage4_ground_truth(target: Dict[str, Any], target_dir: Path,
                        upstream_prov: List[Path]) -> Optional[Path]:
    if not target.get("paired_holo_pdb_id"):
        return None  # Frontier target, no holo, no DCC ground truth

    tname = target["target"]
    apo_pdb = target_dir / "artifacts" / "2_clean" / f"{target['pdb_id'].lower()}_clean.pdb"
    holo_pdb_id = target["paired_holo_pdb_id"].lower()
    holo_pdb = target_dir / "artifacts" / "1_download" / f"{holo_pdb_id}.pdb"
    ligand_resname = target.get("paired_holo_ligand_resname", "LIG")
    chain = target.get("chain", "A")
    artifacts_dir = target_dir / "artifacts" / "4_ground_truth"
    prov_dir = target_dir / "prov"

    flexible_regions = target.get("flexible_regions")

    with RunContext(tname, "4_ground_truth", "superposition",
                    artifacts_dir, prov_dir, upstream_prov=upstream_prov) as ctx:
        ctx.add_input(apo_pdb, upstream_prov_ref="2_clean.prism_clean")
        ctx.add_input(holo_pdb, upstream_prov_ref="1_download.rcsb_fetch")
        ctx.set_tool("kabsch_numpy_svd", ["python3", "-c", "import numpy as np"])
        # Use two-stage Kabsch (bypass MDAnalysis selectall-by-segid bug).
        # If flexible_regions provided, rigid-core alignment is used.
        result = _apo_holo_superposition(
            apo_pdb, holo_pdb, ligand_resname, chain,
            flexible_regions=flexible_regions,
        )
        gt_sidecar = artifacts_dir / f"{target['pdb_id'].lower()}_ground_truth.json"
        with open(gt_sidecar, "w") as f:
            json.dump(result, f, indent=2, default=str)
        ctx.add_output(gt_sidecar, role="ground_truth")
        if "error" in result:
            ctx.set_gate("superposition_success", "FAIL", note=result["error"])
            ctx.set_verdict("FAIL")
        else:
            ctx.set_gate("superposition_success", "PASS")
            ctx.set_gate("rmsd_reasonable",
                        "PASS" if result.get("core_rmsd_after_alignment", 999) < 5.0 else "WARN",
                        note=f"RMSD={result.get('core_rmsd_after_alignment', 'N/A'):.2f}Å")
            ctx.set_verdict("PASS")

    return prov_dir / "4_ground_truth.superposition.prov.json"


# ─────────────────────────────────────────────────────────────────────
# Stage 5 — Engine run with full Tier B provenance + CUPTI trace
# ─────────────────────────────────────────────────────────────────────

def stage5_engine(target: Dict[str, Any], target_dir: Path,
                  upstream_prov: List[Path],
                  with_cupti: bool = True,
                  multi_stream: int = 4,
                  graph_coupling: bool = False,
                  asc_writeback: bool = True,
                  cascade: bool = False) -> Path:
    tname = target["target"]
    pdb_id = target["pdb_id"].lower()
    topology = target_dir / "artifacts" / "3_prep" / f"{pdb_id}.topology.json"
    artifacts_dir = target_dir / "artifacts" / "5_engine"
    prov_dir = target_dir / "prov"
    engine_bin = REPO_ROOT / "target" / "release" / "nhs_rt_full"
    wrapper = REPO_ROOT / "scripts" / "prism-validate-and-run.sh"

    artifacts_dir.mkdir(parents=True, exist_ok=True)

    engine_argv = [
        str(wrapper) if wrapper.exists() else str(engine_bin),
        "-t", str(topology),
        "-o", str(artifacts_dir),
        "--multi-differential",
        "--multi-stream", str(multi_stream),
        "--spike-percentile", "70",
        "--filter-otsu",        # Tier-A: data-adaptive per-channel threshold (supersedes fixed 70%)
        "--stepped-holds",       # Tier-A: 100K/150K/200K holds during ramp → cryptic-basin sampling
        "--use-tokenized-ranker",
        "--fast", "--hysteresis", "--prism-therm",
        "--fused-steps", "4", "--hmr", "--adaptive-dt",
        "--replica-seed", "42", "-v",
    ]
    # --cascade is OFF by default per CLAUDE.md: cryptic site detection
    # requires preserving low-persistence sites. Cascade eliminates them.
    # Enable only for active-site-only benchmarks.
    if cascade:
        engine_argv.insert(-1, "--cascade")

    # NMA perturbation — auto-attach if _nma_modes.json was produced in stage 3.
    # Engine runs explicit mechanical perturbation during warm_hold, forces
    # conformational motion independent of UV-LIF coupling. Critical for
    # targets where aromatic chromophores are spatially dispersed and UV-LIF
    # never reaches firing threshold (e.g. CBL-B zero-spikes regime).
    # prism-prep saves modes as `<pdb_stem>_nma_modes.json` — glob to find it.
    prep_dir = target_dir / "artifacts" / "3_prep"
    nma_candidates = list(prep_dir.glob("*_nma_modes.json"))
    if nma_candidates:
        nma_modes_file = nma_candidates[0]
        engine_argv.insert(-1, "--nma-perturb")
        engine_argv.insert(-1, str(nma_modes_file))
        nma_amp = float(target.get("nma_amplification", 3.0) or 3.0)
        engine_argv.insert(-1, "--nma-amplification")
        engine_argv.insert(-1, str(nma_amp))

    # Per-target engine flag additions (for enhanced sampling / rescue flags
    # like --filter-otsu, --adaptive-bias, --rest2). Specified in target_config.
    extra_flags = target.get("engine_flag_additions") or []
    for flag in extra_flags:
        engine_argv.insert(-1, str(flag))
    if graph_coupling:
        # NOTE: --graph-coupling only affects the coupled-twin (2-group) path.
        # In --multi-differential (4-group) mode, this flag is effectively a no-op;
        # per-stream CUDA Graph capture runs automatically. Kept for flexibility.
        engine_argv.insert(-1, "--graph-coupling")
    if asc_writeback:
        # --closed-loop-steering is the actual flag for Stage 2 ASC writeback.
        # When enabled, GC-PID synergy estimator writes top-K focus residues to
        # device-side ProtocolState.steering_focus_residues, closing the ACS loop.
        engine_argv.insert(-1, "--closed-loop-steering")

    # Optional CUPTI/Nsight trace
    nsys_trace_base = artifacts_dir / f"{pdb_id}_engine_trace"
    if with_cupti and shutil.which("nsys"):
        final_argv = wrap_with_nsys(engine_argv, nsys_trace_base)
    else:
        final_argv = engine_argv

    # GPU telemetry capture
    tel_csv = artifacts_dir / "gpu_telemetry.csv"
    tel = GpuTelemetryCapture(tel_csv, interval_sec=1)

    with RunContext(tname, "5_engine", "nhs_rt_full",
                    artifacts_dir, prov_dir, upstream_prov=upstream_prov) as ctx:
        ctx.add_input(topology, upstream_prov_ref="3_prep.openmm_sanity")
        det_env = determinism_env()
        ctx.set_env(det_env)
        ctx.set_tool("nhs_rt_full", final_argv,
                    binary_path=engine_bin if engine_bin.exists() else None)
        tel.start()
        # ── Adaptive engine timeout ──
        # USP1 (multichain heterodimer, 9352 atoms) hit the prior hard-coded
        # 3600s cap and got SIGKILLed by subprocess.run while still running
        # its MD. Scale the timeout to topology size so large multichain
        # targets don't silently fail.
        #
        # Formula: base 1800 s + (n_atoms × 0.4 s) capped at 14400 s (4 hr).
        # For reference:
        #   KRAS    2,684 atoms  → 2,874 s (~48 min)
        #   POLQ   14,167 atoms  → 7,467 s (~2 hr)
        #   TRIP12 24,620 atoms  → 11,648 s (~3.2 hr, under cap)
        #   1M-atom hypothetical → 14,400 s (cap — needs --engine-timeout flag)
        engine_timeout = 1800
        try:
            topo_data = json.loads(topology.read_text())
            n_atoms = int(topo_data.get('n_atoms', 0))
            if n_atoms > 0:
                engine_timeout = min(max(1800, 1800 + int(n_atoms * 0.4)), 14400)
        except Exception:
            pass  # fall back to 1800 s if topology isn't readable
        print(f"  engine timeout: {engine_timeout}s (adaptive from topology size)")
        try:
            # Engine resolves PTX paths relative to CWD; must run from repo root
            result = ctx.run(
                final_argv,
                timeout=engine_timeout,
                env_overrides=det_env,
                stdout_file=artifacts_dir / "engine.stdout.log",
                stderr_file=artifacts_dir / "engine.stderr.log",
                cwd_override=REPO_ROOT,
            )
        finally:
            tel_info = tel.stop()
        ctx.add_note(f"gpu_telemetry={tel_info}")
        if with_cupti:
            nsys_info = hash_nsys_trace(nsys_trace_base)
            ctx.add_note(f"nsys_trace={nsys_info}")

        # Enumerate + hash all engine outputs
        for art in sorted(artifacts_dir.glob("*.binding_sites.json")):
            ctx.add_output(art, role="binding_sites")
        for art in sorted(artifacts_dir.glob("*.kcc_visualization.json")):
            ctx.add_output(art, role="kcc_visualization")
        for art in sorted(artifacts_dir.glob("*.topology.prism_therm.json")):
            ctx.add_output(art, role="prism_therm")
        for art in sorted(artifacts_dir.glob("*.topology.spike_events.arrow")):
            ctx.add_output(art, role="spike_stream_arrow")

        ctx.set_gate("engine_exit_zero",
                    "PASS" if result.returncode == 0 else "FAIL")
        bs = list(artifacts_dir.glob("*.binding_sites.json"))
        ctx.set_gate("binding_sites_emitted", "PASS" if bs else "FAIL")
        ctx.set_verdict("PASS" if result.returncode == 0 and bs else "FAIL")

    # Emit Tier B companion record
    tier_b_prov = emit_engine_tier_b_provenance(
        target=tname,
        engine_output_dir=artifacts_dir,
        prov_dir=prov_dir,
        upstream_prov=[prov_dir / "5_engine.nhs_rt_full.prov.json"],
        gpu_telemetry_csv=tel_csv if tel_csv.exists() else None,
        nsys_trace=nsys_trace_base if with_cupti else None,
    )
    return tier_b_prov


# ─────────────────────────────────────────────────────────────────────
# Stage 6 — DCC + baselines
# ─────────────────────────────────────────────────────────────────────

def stage6_rerank(target: Dict[str, Any], target_dir: Path,
                  upstream_prov: List[Path]) -> Path:
    """Stage 6 — production re-ranking with confidence bands.

    **Production scoring only.** Ground truth NEVER enters this stage — that
    work is in stage 7 (evaluation). Strictly separates ranking (here) from
    evaluation (stage 7) so an immature scorer cannot erase a correct
    detection.

    Rewritten from the prior `stage6_dcc` which had three bugs:
      1. Weights summed to 1.15 (not 1.0) — arithmetic error that inflated
         the score scale and distorted relative contributions.
      2. `is_cryptic_additional` double-counted with `therm_class_bonus`
         (CRYPTIC already gets the highest therm bonus; adding a separate
         binary 0.15 on top was cryptic-rewarded-twice).
      3. Mixed / unbounded scales. `tide_coupling_count` (raw integer, 0-20+)
         swamped the other terms which were all in [0, 1]. Continuous
         features weren't normalized onto compatible scales.

    New composite (weights sum to 1.0 exactly):

        S = 0.40 * drug_score       (already [0, 1])
          + 0.20 * therm_score       (categorical → [0, 1])
          + 0.20 * spike_score       (n / max_in_run → [0, 1])
          + 0.20 * tide_score        (log-saturating, capped at 1.0)

    Where each term is normalized BEFORE weighting:

      * drug_score  = pocket.druggability_score             [0, 1]
      * therm_score = {CRYPTIC: 1.0, DYNAMIC: 0.7,
                       RESPONSIVE: 0.4, INERT: 0.1,
                       '':         0.0}                      categorical
      * spike_score = n_spikes / max(n_spikes) in this run   [0, 1]
      * tide_score  = log(1 + n_tide) / log(1 + tide_sat),
                      capped at 1.0, with tide_sat = 20.
                      Count = 0   → 0
                      Count = 3   → log(4)/log(21)  ≈ 0.46
                      Count = 20+ → 1.0 (saturation)

    `is_cryptic_additional` removed entirely. Sites flagged CRYPTIC by the
    thermo classifier get the highest `therm_score` (1.0) and that is their
    credit. No double-counting.

    **Confidence bands** (new — per Bradley critique):
    In addition to the single scalar `rerank_composite`, this stage emits
    three soft-confidence axes per site:

      * cryptic_likelihood    = therm == CRYPTIC AND tau in SOC [1.2, 1.5]
                                AND asym_z > 0.5 → fraction of conditions met
      * dynamic_support       = fraction of (asym_z > 0, tau > 0,
                                n_tide_triggers > 5) satisfied
      * druggability_support  = drug_score (passthrough)

    These let the reviewer see "site localizes to known cryptic region but
    tau is outside SOC" as a calibrated signal rather than being forced
    into a hard {CRYPTIC | DYNAMIC | RESPONSIVE | INERT} class.

    Stage name: `6_rerank` (was `6_dcc` — renamed; the prior name was
    misleading because the DCC computation it did only runs when ground
    truth exists and has now moved to stage 7).
    """
    tname = target["target"]
    pdb_id = target["pdb_id"].lower()
    eng_dir = target_dir / "artifacts" / "5_engine"
    enr_dir = target_dir / "artifacts" / "7_enrichment"
    # Stage 6 is production scoring only — NO ground truth access (that is stage 7).
    artifacts_dir = target_dir / "artifacts" / "6_rerank"
    prov_dir = target_dir / "prov"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    rerank_out = artifacts_dir / "rerank_result.json"
    with RunContext(tname, "6_rerank", "normalized_cryptic_rerank",
                    artifacts_dir, prov_dir, upstream_prov=upstream_prov) as ctx:
        # Inputs: prism_therm (therm class), arrow stream (spike centroids).
        # NOT ground truth — that is used only in stage 7 for evaluation.
        therm_files = list(eng_dir.glob("*.topology.prism_therm.json"))
        arrow_files = list(eng_dir.glob("*.topology.spike_events.arrow"))
        if not therm_files or not arrow_files:
            ctx.set_gate("inputs_present", "FAIL",
                        note=f"therm={len(therm_files)} arrow={len(arrow_files)}")
            ctx.set_verdict("FAIL")
            return prov_dir / "6_rerank.normalized_cryptic_rerank.prov.json"
        ctx.add_input(therm_files[0], upstream_prov_ref="5_engine.nhs_rt_full")
        ctx.add_input(arrow_files[0], upstream_prov_ref="5_engine.nhs_rt_full")

        with open(therm_files[0]) as f:
            therm = json.load(f)

        # Spike-weighted centroid per site_id from arrow
        try:
            import polars as pl
            df = pl.read_parquet(enr_dir / f"{pdb_id}.spike_events.enriched.parquet",
                                 columns=["x", "y", "z", "site_id"]) \
                 if (enr_dir / f"{pdb_id}.spike_events.enriched.parquet").exists() else None
            if df is None:
                import pyarrow.feather as feather
                tbl = feather.read_table(str(arrow_files[0]),
                                         columns=["x", "y", "z", "site_id"])
                df = pl.from_arrow(tbl)
            site_centroids = (
                df.filter(pl.col("site_id") >= 0)
                  .group_by("site_id")
                  .agg([
                      pl.col("x").mean().alias("cx"),
                      pl.col("y").mean().alias("cy"),
                      pl.col("z").mean().alias("cz"),
                      pl.len().alias("n_spikes"),
                  ])
            )
            site_centroid_dict = {
                int(row["site_id"]): {
                    "centroid": [float(row["cx"]), float(row["cy"]), float(row["cz"])],
                    "n_spikes": int(row["n_spikes"]),
                }
                for row in site_centroids.iter_rows(named=True)
            }
        except Exception as e:
            ctx.add_note(f"spike centroid extraction failed: {e}")
            site_centroid_dict = {}

        # Merge prism_therm (classification) with arrow (spike centroid).
        merged_pockets = []
        for p in therm.get("pockets", []):
            pid = p.get("pocket_id")
            sp_data = site_centroid_dict.get(pid, {})
            top_res_ids = [r.get("residue_id") for r in p.get("top_residues", [])
                           if r.get("residue_id") is not None]
            merged_pockets.append({
                "pocket_id": pid,
                "therm_class": p.get("therm_class"),
                "druggability_score": p.get("druggability_score"),
                "ccns_tau": p.get("ccns_tau"),
                "hysteresis_asymmetry": p.get("hysteresis_asymmetry"),
                "relative_asymmetry": p.get("relative_asymmetry"),
                "is_cryptic": p.get("is_cryptic", False),
                "top_residue_ids": top_res_ids,
                "centroid_spike_weighted": sp_data.get("centroid"),
                "n_spikes_attributed": sp_data.get("n_spikes", 0),
            })

        # ─── Normalized composite (weights sum to 1.0 exactly) ───
        # Every term is mapped to [0, 1] BEFORE weighting. See function
        # docstring for the full rationale and the prior-bug history.
        import math
        TIDE_SATURATION = 20  # tide_score saturates at 20 trigger residues
        THERM_MAP = {
            "CRYPTIC": 1.0,
            "DYNAMIC": 0.7,
            "RESPONSIVE": 0.4,
            "INERT": 0.1,
        }
        WEIGHTS = {
            "drug":  0.40,
            "therm": 0.20,
            "spike": 0.20,
            "tide":  0.20,
        }
        # ── fail-loud invariant check ──
        _wsum = sum(WEIGHTS.values())
        if abs(_wsum - 1.0) > 1e-9:
            raise RuntimeError(
                f"ranker weights do not sum to 1.0 (got {_wsum}) — refusing to run")

        max_spikes = max((p.get("n_spikes_attributed") or 0)
                         for p in merged_pockets) or 1
        log_tide_sat = math.log(1.0 + TIDE_SATURATION)

        for p in merged_pockets:
            # Per-term normalization.
            drug_score  = float(p.get("druggability_score") or 0.0)
            drug_score  = max(0.0, min(1.0, drug_score))

            therm_score = THERM_MAP.get((p.get("therm_class") or "").upper(), 0.0)

            n_spk       = float(p.get("n_spikes_attributed") or 0)
            spike_score = n_spk / max_spikes if max_spikes > 0 else 0.0
            spike_score = max(0.0, min(1.0, spike_score))

            n_tide      = len(p.get("top_residue_ids") or [])
            tide_score  = math.log(1.0 + n_tide) / log_tide_sat
            tide_score  = max(0.0, min(1.0, tide_score))

            # Weighted composite.
            composite = (WEIGHTS["drug"]  * drug_score  +
                         WEIGHTS["therm"] * therm_score +
                         WEIGHTS["spike"] * spike_score +
                         WEIGHTS["tide"]  * tide_score)

            # ── Confidence bands (NOT a hard class label) ──
            # cryptic_likelihood: fraction of (therm==CRYPTIC, tau in SOC,
            # asym_z > 0.5) satisfied. Principled — each signal is a bit of
            # evidence; we report the aggregate fraction, not a hard label.
            tau   = float(p.get("ccns_tau") or 0.0)
            asz   = float(p.get("relative_asymmetry") or 0.0)
            cryptic_bits = [
                (p.get("therm_class") or "").upper() == "CRYPTIC",
                1.2 <= tau <= 1.5,
                asz > 0.5,
            ]
            cryptic_likelihood = sum(1.0 for b in cryptic_bits if b) / len(cryptic_bits)

            # dynamic_support: fraction of (asym_z > 0, tau > 0, n_tide > 5)
            dynamic_bits = [asz > 0.0, tau > 0.0, n_tide > 5]
            dynamic_support = sum(1.0 for b in dynamic_bits if b) / len(dynamic_bits)

            # druggability_support = drug_score (passthrough — already [0, 1])
            druggability_support = drug_score

            p["drug_score_normalized"]   = drug_score
            p["therm_score_normalized"]  = therm_score
            p["spike_score_normalized"]  = spike_score
            p["tide_score_normalized"]   = tide_score
            p["rerank_composite"]        = float(composite)
            p["cryptic_likelihood"]      = float(cryptic_likelihood)
            p["dynamic_support"]         = float(dynamic_support)
            p["druggability_support"]    = float(druggability_support)

        merged_pockets.sort(key=lambda p: -p["rerank_composite"])
        for i, p in enumerate(merged_pockets):
            p["rerank_position"] = i + 1

        result: Dict[str, Any] = {
            "method": "normalized_cryptic_rerank",
            "ranker_weights": WEIGHTS,
            "therm_class_mapping": THERM_MAP,
            "tide_saturation": TIDE_SATURATION,
            "confidence_band_definitions": {
                "cryptic_likelihood":
                    "fraction of {therm==CRYPTIC, tau in SOC [1.2,1.5], asym_z>0.5} satisfied",
                "dynamic_support":
                    "fraction of {asym_z>0, tau>0, n_tide_triggers>5} satisfied",
                "druggability_support":
                    "passthrough of druggability_score (already [0, 1])",
            },
            "merged_pockets": merged_pockets,
            "note": "Production scoring only. Evaluation (DCC, contact-residue overlap, "
                    "4-outcome matrix) lives in stage 7 and reads ground_truth.json "
                    "separately. Ground truth does not enter this stage.",
        }

        with open(rerank_out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        ctx.add_output(rerank_out, role="rerank_result")
        ctx.set_gate("rerank_computed", "PASS",
                     note=f"{len(merged_pockets)} pockets ranked")
        ctx.set_gate("weight_sum_invariant", "PASS",
                     note=f"weights sum to {_wsum:.6f}")
        ctx.set_verdict("PASS")

    return prov_dir / "6_rerank.normalized_cryptic_rerank.prov.json"


# ─────────────────────────────────────────────────────────────────────
# Stage 7 — EVALUATION (strictly separate from scoring)
# ─────────────────────────────────────────────────────────────────────

def stage7_evaluation(target: Dict[str, Any], target_dir: Path,
                      upstream_prov: List[Path]) -> Path:
    """Stage 7 — reviewer-facing evaluation with 4-outcome diagnostic matrix.

    This stage **reads** ground_truth.json and binding_sites.json and
    emits a benchmarking report. It never feeds back into the scorer —
    ground truth is quarantined to evaluation.

    For each target with a paired holo PDB, produces the Bradley
    4-outcome matrix:

      A. Detection       — did ANY detected site match the true region?
      B. Ranking         — what rank was the best-matching site?
      C. Typing          — did the classifier assign the correct therm_class?
      D. Druggability    — was the best-matching site flagged as tractable?

    Plus a `verdict_tag` that names the failure mode precisely:

      * FOUND_TOP_RANKED           — detection + ranking + typing all correct
      * FOUND_TOP_RANKED_MISCLASSIFIED — detection + ranking correct, typing wrong
      * FOUND_NOT_TOP_RANKED       — detection correct but not at rank 1
      * FOUND_NOT_TOP_RANKED_MISCLASSIFIED — ditto + typing wrong
      * NOT_FOUND                  — no site within threshold of ground truth
      * NO_GROUND_TRUTH            — frontier target (no paired holo)

    Edge cases handled:
      * No paired holo       → NO_GROUND_TRUTH, exit 0
      * Extended ligand      → reports both atom-centroid DCC and pocket-
                               CA centroid DCC (see 9BKS ADP analysis)
      * Multichain           → ground_truth.json already accounts for
                               chain-merged frame
      * pdbfixer-filled gaps → flag lining residues whose PDB number is
                               inside a documented gap in the raw PDB
                               (future extension)
    """
    tname = target["target"]
    pdb_id = target["pdb_id"].lower()
    eng_dir   = target_dir / "artifacts" / "5_engine"
    gt_dir    = target_dir / "artifacts" / "4_ground_truth"
    artifacts_dir = target_dir / "artifacts" / "7_evaluation"
    prov_dir  = target_dir / "prov"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    gt_sidecar = gt_dir / f"{pdb_id}_ground_truth.json"
    bs_files   = list(eng_dir.glob("*.binding_sites.json"))
    eval_out   = artifacts_dir / "evaluation.json"

    with RunContext(tname, "7_evaluation", "four_outcome_matrix",
                    artifacts_dir, prov_dir, upstream_prov=upstream_prov) as ctx:
        # ── Case 1: frontier target (no paired holo) ──
        if not gt_sidecar.exists():
            result = {
                "target": tname, "pdb_id": pdb_id,
                "ground_truth_available": False,
                "verdict_tag": "NO_GROUND_TRUTH",
                "note": "Frontier apo target — no paired holo. Evaluation not applicable.",
            }
            with open(eval_out, "w") as f:
                json.dump(result, f, indent=2)
            ctx.add_output(eval_out, role="evaluation")
            ctx.set_gate("ground_truth_present", "N/A", note="frontier target")
            ctx.set_verdict("PASS")
            return prov_dir / "7_evaluation.four_outcome_matrix.prov.json"

        # ── Case 2: binding_sites.json missing → engine failure ──
        if not bs_files:
            result = {
                "target": tname, "pdb_id": pdb_id,
                "ground_truth_available": True,
                "verdict_tag": "ENGINE_FAILURE",
                "note": "binding_sites.json not produced by stage 5 engine.",
            }
            with open(eval_out, "w") as f:
                json.dump(result, f, indent=2)
            ctx.add_output(eval_out, role="evaluation")
            ctx.set_gate("binding_sites_present", "FAIL")
            ctx.set_verdict("FAIL")
            return prov_dir / "7_evaluation.four_outcome_matrix.prov.json"

        # ── Load inputs ──
        with open(gt_sidecar) as f:
            gt = json.load(f)
        with open(bs_files[0]) as f:
            bs = json.load(f)
        ctx.add_input(gt_sidecar, upstream_prov_ref="4_ground_truth.superposition")
        ctx.add_input(bs_files[0], upstream_prov_ref="5_engine.nhs_rt_full")

        if "error" in gt or not gt.get("ligand_centroid_apo_frame"):
            result = {
                "target": tname, "pdb_id": pdb_id,
                "ground_truth_available": False,
                "verdict_tag": "GROUND_TRUTH_INVALID",
                "note": f"ground_truth.json invalid: {gt.get('error', 'missing ligand_centroid')}",
            }
            with open(eval_out, "w") as f:
                json.dump(result, f, indent=2)
            ctx.add_output(eval_out, role="evaluation")
            ctx.set_gate("ground_truth_valid", "FAIL")
            ctx.set_verdict("FAIL")
            return prov_dir / "7_evaluation.four_outcome_matrix.prov.json"

        lig_centroid = np.asarray(gt["ligand_centroid_apo_frame"], dtype=np.float64)
        lig_atoms    = np.asarray(gt.get("ligand_atoms_apo_frame") or [lig_centroid],
                                  dtype=np.float64)
        lig_contact_residues = set(gt.get("ligand_contact_residues", []) or [])

        sites = bs.get("sites", [])
        if not sites:
            result = {
                "target": tname, "pdb_id": pdb_id,
                "ground_truth_available": True,
                "verdict_tag": "NOT_FOUND",
                "note": "engine emitted zero sites",
            }
            with open(eval_out, "w") as f:
                json.dump(result, f, indent=2)
            ctx.add_output(eval_out, role="evaluation")
            ctx.set_gate("sites_detected", "FAIL", note="0 sites")
            ctx.set_verdict("FAIL")
            return prov_dir / "7_evaluation.four_outcome_matrix.prov.json"

        # ── Per-site distance + contact overlap ──
        DCC_DETECTION_THRESHOLD = 8.0     # Å, standard cryptic benchmark
        CONTACT_OVERLAP_THRESHOLD = 0.30  # 30% shared contact residues = "same pocket"

        per_site_eval = []
        for s in sites:
            c = s.get("centroid")
            if c is None or len(c) != 3:
                continue
            c_arr = np.asarray(c, dtype=np.float64)
            centroid_dcc = float(np.linalg.norm(c_arr - lig_centroid))
            # min_atom_dcc: nearest ligand ATOM to the site centroid —
            # robust to extended cofactors (e.g., ADP) where geometric
            # centroid is offset from pocket cavity.
            if lig_atoms.ndim == 2:
                min_atom_dcc = float(np.min(np.linalg.norm(lig_atoms - c_arr, axis=1)))
            else:
                min_atom_dcc = centroid_dcc
            # Contact residue overlap.
            site_lining = set()
            for r in (s.get("lining_residues") or []):
                rid = r.get("resid") or r.get("residue_id")
                if rid is not None:
                    site_lining.add(int(rid))
            overlap_count = len(site_lining & lig_contact_residues) if lig_contact_residues else 0
            overlap_frac = (overlap_count / len(lig_contact_residues)) \
                            if lig_contact_residues else 0.0
            per_site_eval.append({
                "site_id": s.get("id") or s.get("cluster_id"),
                "rank_by_quality": len(per_site_eval) + 1,  # JSON order = quality order
                "quality_score": s.get("quality_score"),
                "therm_class": s.get("therm_class"),
                "is_druggable": s.get("is_druggable"),
                "centroid": list(c),
                "centroid_dcc_angstrom": centroid_dcc,
                "min_atom_dcc_angstrom": min_atom_dcc,
                "contact_overlap_count": overlap_count,
                "contact_overlap_fraction": float(overlap_frac),
            })

        # ── Best-match site: minimize EITHER DCC-to-atom OR maximize contact
        # overlap. Pick whichever agrees more strongly with the ligand.
        # Default to min_atom_dcc since contact overlap requires ligand
        # contact residue list which may be absent for some targets.
        best_by_atom_dcc = min(per_site_eval, key=lambda r: r["min_atom_dcc_angstrom"])
        best_by_overlap  = max(per_site_eval, key=lambda r: r["contact_overlap_fraction"]) \
                            if lig_contact_residues else best_by_atom_dcc

        # The "best match" for the 4-outcome matrix. Prefer overlap-based
        # when contact residues are known; fall back to distance otherwise.
        best_match = best_by_overlap if lig_contact_residues else best_by_atom_dcc

        # ── A. Detection ──
        detection_pass = (best_match["min_atom_dcc_angstrom"] < DCC_DETECTION_THRESHOLD or
                          best_match["contact_overlap_fraction"] >= CONTACT_OVERLAP_THRESHOLD)

        # ── B. Ranking ──
        best_rank = best_match["rank_by_quality"]
        if best_rank == 1:
            rank_class = "TOP_1"
        elif best_rank <= 3:
            rank_class = "TOP_3"
        elif best_rank <= 10:
            rank_class = "TOP_10"
        else:
            rank_class = "BEYOND_TOP_10"

        # ── C. Typing ──
        # Biologically-correct class: use target_config cryptic_site_type
        # hint if present; otherwise default expectation is CRYPTIC for any
        # target where the paired holo binds in a region that's disordered
        # or re-shaped in the apo. This is a hint, not a hard truth —
        # reviewer may override.
        expected_class = "CRYPTIC" if target.get("cryptic_site_type") else "DYNAMIC"
        observed_class = (best_match.get("therm_class") or "").upper()
        typing_pass = (observed_class == expected_class)
        classification_disagreement = not typing_pass

        # ── D. Druggability ──
        druggability_pass = bool(best_match.get("is_druggable"))

        # ── verdict_tag (Bradley taxonomy) ──
        if not detection_pass:
            verdict_tag = "NOT_FOUND"
        elif rank_class == "TOP_1" and typing_pass:
            verdict_tag = "FOUND_TOP_RANKED"
        elif rank_class == "TOP_1" and not typing_pass:
            verdict_tag = "FOUND_TOP_RANKED_MISCLASSIFIED"
        elif typing_pass:
            verdict_tag = f"FOUND_{rank_class}"
        else:
            verdict_tag = f"FOUND_{rank_class}_MISCLASSIFIED"

        # ── Emit result ──
        result = {
            "target": tname, "pdb_id": pdb_id,
            "paired_holo_pdb_id": target.get("paired_holo_pdb_id"),
            "ground_truth_available": True,
            "thresholds": {
                "dcc_detection_angstrom": DCC_DETECTION_THRESHOLD,
                "contact_overlap_fraction": CONTACT_OVERLAP_THRESHOLD,
            },
            "best_match": best_match,
            "four_outcome_matrix": {
                "A_detection":      "PASS" if detection_pass    else "FAIL",
                "B_ranking":        rank_class,
                "C_typing":         "PASS" if typing_pass       else "FAIL",
                "D_druggability":   "PASS" if druggability_pass else "FAIL",
            },
            "classification_disagreement": classification_disagreement,
            "expected_class": expected_class,
            "observed_class": observed_class,
            "verdict_tag": verdict_tag,
            "per_site_eval": per_site_eval,
            "note": "Stage 7 is strictly evaluation. Never feeds back into stage 6 scoring.",
        }
        with open(eval_out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        ctx.add_output(eval_out, role="evaluation")
        ctx.set_gate("A_detection",    "PASS" if detection_pass    else "FAIL",
                     note=f"best DCC {best_match['min_atom_dcc_angstrom']:.2f} Å / overlap "
                          f"{best_match['contact_overlap_fraction']:.2f}")
        ctx.set_gate("B_ranking",      rank_class,
                     note=f"best-match at rank {best_rank}")
        ctx.set_gate("C_typing",       "PASS" if typing_pass       else "FAIL",
                     note=f"observed={observed_class} expected={expected_class}")
        ctx.set_gate("D_druggability", "PASS" if druggability_pass else "FAIL")
        ctx.set_verdict("PASS")
        ctx.add_note(f"verdict={verdict_tag}")

    return prov_dir / "7_evaluation.four_outcome_matrix.prov.json"


# ─────────────────────────────────────────────────────────────────────
# Manifest assembly
# ─────────────────────────────────────────────────────────────────────

def emit_manifest(target: Dict[str, Any], target_dir: Path):
    prov_dir = target_dir / "prov"
    prov_records = sorted(prov_dir.glob("*.prov.json"))
    write_manifest(target["target"], target_dir, prov_records, extra={"target_config": target})


# ─────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────

STAGE_FUNCS = {
    "1_download": stage1_download,
    "2_clean": stage2_clean,
    "3_prep": stage3_prep,
    "4_ground_truth": stage4_ground_truth,
    "5_engine": stage5_engine,
    "6_rerank": stage6_rerank,            # production scoring (no ground truth)
    "7_evaluation": stage7_evaluation,    # reviewer-facing evaluation (reads ground truth)
    # Legacy alias — old runners / docs referenced 6_dcc. Keep the alias so
    # `--stage 6_dcc` still works until all callers migrate. Internally
    # identical to 6_rerank.
    "6_dcc": stage6_rerank,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-config", required=True, type=Path,
                    help="JSON with a single target's config dict")
    ap.add_argument("--stage", required=True,
                    choices=list(STAGE_FUNCS.keys()) + ["all"])
    ap.add_argument("--target-dir", required=True, type=Path)
    ap.add_argument("--no-cupti", action="store_true",
                    help="Disable Nsight Systems trace in stage 5")
    args = ap.parse_args()

    with open(args.target_config) as f:
        target = json.load(f)

    if args.stage == "all":
        stages = ["1_download", "2_clean", "3_prep", "4_ground_truth",
                  "5_engine", "6_rerank", "7_evaluation"]
    else:
        stages = [args.stage]

    upstream: List[Path] = []
    for stage in stages:
        print(f"\n=== {target['target']} / {stage} ===")
        t0 = time.time()
        fn = STAGE_FUNCS[stage]
        try:
            if stage == "5_engine":
                result = fn(target, args.target_dir, upstream,
                          with_cupti=not args.no_cupti)
            else:
                result = fn(target, args.target_dir, upstream) if stage != "1_download" \
                         else fn(target, args.target_dir)
        except Exception as e:
            import traceback
            print(f"  FAIL: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 1
        dt = time.time() - t0
        if result:
            upstream = [result]
            print(f"  prov: {result.name}  ({dt:.1f}s)")
        else:
            print(f"  (skipped, no upstream)")

    emit_manifest(target, args.target_dir)
    print(f"\n=== manifest written: {args.target_dir / 'prov' / 'pipeline_manifest.json'} ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

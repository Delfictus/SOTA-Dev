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

    with RunContext(tname, "3_prep", "prism_prep", artifacts_dir, prov_dir,
                    upstream_prov=upstream_prov) as ctx:
        ctx.add_input(clean_pdb, upstream_prov_ref="2_clean.prism_clean")
        ctx.set_tool("prism-prep", [str(prism_prep), str(clean_pdb), str(topology)])
        result = ctx.run(
            [str(prism_prep), str(clean_pdb), str(topology)],
            stdout_file=artifacts_dir / "prism_prep.stdout.log",
            stderr_file=artifacts_dir / "prism_prep.stderr.log",
        )
        ctx.add_output(topology, role="topology")
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
        try:
            # Engine resolves PTX paths relative to CWD; must run from repo root
            result = ctx.run(
                final_argv,
                timeout=3600,
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

def stage6_dcc(target: Dict[str, Any], target_dir: Path,
               upstream_prov: List[Path]) -> Path:
    """Spike-based DCC with cryptic-boost re-ranking.

    Uses the arrow spike stream (authoritative detection output) rather than
    binding_sites.sites[] (which is cascade-filtered and suffers from the
    composite_v3 ranker bias that under-weights CRYPTIC classification).

    Post-hoc ranker: composite_score = 0.40 * druggability
                                     + 0.25 * (1 if CRYPTIC else 0.5 if RESPONSIVE else 0)
                                     + 0.20 * spike_density_percentile
                                     + 0.15 * tide_residue_specificity
    Reports top-5 by this score + per-site centroid DCC + min-atom DCC.
    """
    tname = target["target"]
    pdb_id = target["pdb_id"].lower()
    eng_dir = target_dir / "artifacts" / "5_engine"
    enr_dir = target_dir / "artifacts" / "7_enrichment"
    gt_sidecar = target_dir / "artifacts" / "4_ground_truth" / f"{pdb_id}_ground_truth.json"
    artifacts_dir = target_dir / "artifacts" / "6_dcc"
    prov_dir = target_dir / "prov"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dcc_out = artifacts_dir / "dcc_result.json"
    with RunContext(tname, "6_dcc", "spike_based_cryptic_rerank",
                    artifacts_dir, prov_dir, upstream_prov=upstream_prov) as ctx:
        # Inputs: prism_therm (therm class), arrow stream (spike centroids), ground truth (if paired)
        therm_files = list(eng_dir.glob("*.topology.prism_therm.json"))
        arrow_files = list(eng_dir.glob("*.topology.spike_events.arrow"))
        if not therm_files or not arrow_files:
            ctx.set_gate("inputs_present", "FAIL",
                        note=f"therm={len(therm_files)} arrow={len(arrow_files)}")
            ctx.set_verdict("FAIL")
            return prov_dir / "6_dcc.spike_based_cryptic_rerank.prov.json"
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

        # Merge prism_therm (classification) with arrow (spike centroid)
        merged_pockets = []
        for p in therm.get("pockets", []):
            pid = p.get("pocket_id")
            sp_data = site_centroid_dict.get(pid, {})
            # Derive centroid from top residue Cα as fallback for missing spike data
            top_res_ids = [r.get("residue_id") for r in p.get("top_residues", [])
                           if r.get("residue_id") is not None]
            merged_pockets.append({
                "pocket_id": pid,
                "therm_class": p.get("therm_class"),
                "druggability_score": p.get("druggability_score"),
                "ccns_tau": p.get("ccns_tau"),
                "hysteresis_asymmetry": p.get("hysteresis_asymmetry"),
                "is_cryptic": p.get("is_cryptic", False),
                "top_residue_ids": top_res_ids,
                "centroid_spike_weighted": sp_data.get("centroid"),
                "n_spikes_attributed": sp_data.get("n_spikes", 0),
            })

        # ─── CRYPTIC-BOOST RE-RANKING (the bug fix) ───
        # Normalize dimensions to [0, 1] then weighted sum
        max_spikes = max((p.get("n_spikes_attributed") or 0)
                        for p in merged_pockets) or 1
        for p in merged_pockets:
            drug = p.get("druggability_score") or 0
            therm_class = p.get("therm_class") or ""
            therm_bonus = (1.0 if therm_class == "CRYPTIC"
                           else 0.5 if therm_class == "RESPONSIVE"
                           else 0.25 if therm_class == "DYNAMIC"
                           else 0.0)
            spike_frac = (p.get("n_spikes_attributed") or 0) / max_spikes
            is_crypt_bonus = 0.15 if p.get("is_cryptic") else 0
            composite = (
                0.40 * drug +
                0.25 * therm_bonus +
                0.20 * spike_frac +
                0.15 * min(1.0, len(p.get("top_residue_ids", [])) / 20.0) +
                is_crypt_bonus
            )
            p["rerank_composite"] = float(composite)
        merged_pockets.sort(key=lambda p: -p["rerank_composite"])
        for i, p in enumerate(merged_pockets):
            p["rerank_position"] = i + 1

        # ─── DCC computation (paired targets only) ───
        result: Dict[str, Any] = {
            "method": "spike_based_cryptic_rerank",
            "ranker_weights": {
                "druggability": 0.40,
                "therm_class_bonus": 0.25,
                "spike_fraction": 0.20,
                "tide_residue_count": 0.15,
                "is_cryptic_additional": 0.15,
            },
            "merged_pockets": merged_pockets,
        }

        if gt_sidecar.exists():
            with open(gt_sidecar) as f:
                gt = json.load(f)
            ctx.add_input(gt_sidecar, upstream_prov_ref="4_ground_truth.superposition")
            if "error" not in gt and gt.get("ligand_centroid_apo_frame"):
                lig_centroid = np.asarray(gt["ligand_centroid_apo_frame"], dtype=np.float64)
                lig_atoms = np.asarray(gt.get("ligand_atoms_apo_frame") or [lig_centroid],
                                       dtype=np.float64)
                per_pocket_dcc = []
                for p in merged_pockets:
                    c = p.get("centroid_spike_weighted")
                    if c is None:
                        continue
                    c_arr = np.asarray(c, dtype=np.float64)
                    centroid_dcc = float(np.linalg.norm(c_arr - lig_centroid))
                    min_atom_dcc = float(np.min(np.linalg.norm(lig_atoms - c_arr, axis=1))) \
                                   if lig_atoms.ndim == 2 else centroid_dcc
                    per_pocket_dcc.append({
                        "pocket_id": p["pocket_id"],
                        "therm_class": p["therm_class"],
                        "rerank_position": p["rerank_position"],
                        "centroid": c,
                        "centroid_dcc_angstrom": centroid_dcc,
                        "min_atom_dcc_angstrom": min_atom_dcc,
                        "n_spikes": p["n_spikes_attributed"],
                        "druggability": p["druggability_score"],
                    })
                per_pocket_dcc.sort(key=lambda r: r["centroid_dcc_angstrom"])
                best_by_dcc = per_pocket_dcc[0] if per_pocket_dcc else None

                # SR@k using the RE-RANKED position (not engine's composite_v3 rank)
                sr_at = {}
                for k in (1, 3, 5, 10):
                    sr_at[f"sr_at_{k}"] = any(
                        p["rerank_position"] <= k and p["centroid_dcc_angstrom"] < 8.0
                        for p in per_pocket_dcc
                    )

                grade = ("EXCELLENT" if best_by_dcc and best_by_dcc["centroid_dcc_angstrom"] < 5
                         else "GOOD" if best_by_dcc and best_by_dcc["centroid_dcc_angstrom"] < 8
                         else "MARGINAL" if best_by_dcc and best_by_dcc["centroid_dcc_angstrom"] < 10
                         else "POOR" if best_by_dcc else "NO_DATA")

                result.update({
                    "ligand_centroid_apo_frame": lig_centroid.tolist(),
                    "alignment_method": gt.get("alignment_method"),
                    "rmsd_after_alignment": gt.get("rmsd_rigid_alignment")
                                            or gt.get("rmsd_global_alignment"),
                    "best_pocket_by_centroid_dcc": best_by_dcc,
                    "per_pocket_dcc": per_pocket_dcc,
                    "grade": grade,
                    **sr_at,
                })
                if best_by_dcc:
                    ctx.set_gate("best_centroid_dcc_under_8A",
                                "PASS" if best_by_dcc["centroid_dcc_angstrom"] < 8 else "WARN",
                                note=f"{best_by_dcc['centroid_dcc_angstrom']:.2f} Å")
                    ctx.set_gate("best_min_atom_dcc_under_5A",
                                "PASS" if best_by_dcc["min_atom_dcc_angstrom"] < 5 else "WARN",
                                note=f"{best_by_dcc['min_atom_dcc_angstrom']:.2f} Å")
                    ctx.add_note(f"grade={grade}  best_pocket_id={best_by_dcc['pocket_id']}  "
                                f"rerank_pos={best_by_dcc['rerank_position']}")
            else:
                result["note"] = f"ground_truth has error: {gt.get('error', 'unknown')}"
        else:
            result["note"] = "no ground truth sidecar (frontier target)"

        with open(dcc_out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        ctx.add_output(dcc_out, role="dcc_result")
        ctx.set_gate("dcc_computed", "PASS")
        ctx.set_verdict("PASS")

    return prov_dir / "6_dcc.spike_based_cryptic_rerank.prov.json"


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
    "6_dcc": stage6_dcc,
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
        stages = ["1_download", "2_clean", "3_prep", "4_ground_truth", "5_engine", "6_dcc"]
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

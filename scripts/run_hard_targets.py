#!/usr/bin/env python3
"""
PRISM4D Hard-Target Benchmark — Fully Automated Pipeline

Zero-touch execution: downloads PDBs, generates topologies, runs NHS engine,
performs KCC + GTCKL ranking, validates, and produces final report.

Usage:
    python3 scripts/run_hard_targets.py

All outputs written to benchmarks/hard_targets/
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

# ============================================================================
# TOPOLOGY PATH — LOCKED TO PRODUCTION ROUTE (DO NOT CHANGE)
# ============================================================================
# Add scripts/ to path so stage2_topology can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from stage2_topology import prepare_topology
    _TOPO_ROUTE = "stage2_topology.prepare_topology (production)"
except Exception as e:
    print(f"FATAL: Topology pipeline unavailable: {e}")
    print("Required: scripts/stage2_topology.py with OpenMM + AMBER ff14SB")
    print("Install: conda install -c conda-forge openmm")
    sys.exit(1)

# ============================================================================
# CONFIGURATION (FIXED — DO NOT MODIFY)
# ============================================================================

TARGETS = [
    {"pdb_id": "1P38", "name": "p38_MAPK_DFG", "site_type": "cryptic_DFG-out",
     "holo_pdb": "3HEC", "ligand": "STI"},
    {"pdb_id": "3MH1", "name": "ABL_DFG-out", "site_type": "DFG-out_gold_standard",
     "holo_pdb": "3MH1", "ligand": "P16"},
    {"pdb_id": "5LAR", "name": "DDR1_DFG-out", "site_type": "ligand_induced_DFG-out",
     "holo_pdb": "5LAR", "ligand": "6GS"},
]

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks" / "hard_targets"
RAW_DIR = BENCH_DIR / "raw"
CLEAN_DIR = BENCH_DIR / "clean"
TOPO_DIR = BENCH_DIR / "topologies"
RESULTS_DIR = BENCH_DIR / "results"
RUN_DIR = Path("/tmp/prism_hard_targets")

NHS_BINARY = ROOT / "target" / "release" / "nhs_rt_full"
VALIDATION_SCRIPT = ROOT / "scripts" / "kcc_validation_v2.py"

# Engine parameters (FIXED — identical for all targets)
ENGINE_ARGS = [
    "--fast",
    "--hysteresis",
    "--multi-stream", "8",
    "--spike-percentile", "95",
    "--prism-therm",
    "--fused-steps", "4",
    "--hmr",
    "--adaptive-dt",
    "-v",
]

# ============================================================================
# UTILITIES
# ============================================================================

class BenchmarkError(Exception):
    pass


def log(msg, level="INFO"):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def run_cmd(cmd, description, timeout=600, check=True):
    """Run a subprocess with logging and error handling."""
    log(f"  CMD: {' '.join(str(c) for c in cmd)}")
    start = time.time()
    result = subprocess.run(
        [str(c) for c in cmd],
        capture_output=True, text=True, timeout=timeout,
        cwd=str(ROOT),
    )
    elapsed = time.time() - start
    if result.returncode != 0 and check:
        log(f"  FAILED ({elapsed:.1f}s): {result.stderr[-500:]}", "ERROR")
        raise BenchmarkError(f"{description} failed (exit {result.returncode})")
    log(f"  OK ({elapsed:.1f}s)")
    return result


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


# ============================================================================
# PHASE 1 — STRUCTURE RESOLUTION
# ============================================================================

def download_pdb(pdb_id, output_path):
    """Download PDB from RCSB if not already present."""
    if output_path.exists():
        log(f"  PDB already exists: {output_path}")
        return
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    log(f"  Downloading {pdb_id} from RCSB...")
    import urllib.request
    try:
        urllib.request.urlretrieve(url, str(output_path))
        log(f"  Downloaded: {output_path} ({output_path.stat().st_size} bytes)")
    except Exception as e:
        raise BenchmarkError(f"Failed to download {pdb_id}: {e}")


def sanitize_pdb(raw_path, clean_path, pdb_id):
    """Remove ligands, water, and HETATM records. Preserve protein chains."""
    if clean_path.exists():
        log(f"  Clean PDB already exists: {clean_path}")
        return

    log(f"  Sanitizing {pdb_id}...")
    with open(raw_path) as f:
        lines = f.readlines()

    clean_lines = []
    for line in lines:
        record = line[:6].strip()
        if record in ("ATOM", "TER", "END", "MODEL", "ENDMDL"):
            clean_lines.append(line)
        elif record == "HETATM":
            # Keep modified residues (MSE→MET etc) but skip ligands/water
            resname = line[17:20].strip()
            if resname in ("MSE", "SEP", "TPO", "PTR", "HYP"):
                # Modified amino acids — keep as ATOM
                clean_lines.append("ATOM  " + line[6:])
            # Skip HOH, ligands, ions
    clean_lines.append("END\n")

    with open(clean_path, "w") as f:
        f.writelines(clean_lines)

    n_atoms = sum(1 for l in clean_lines if l.startswith("ATOM"))
    log(f"  Sanitized: {n_atoms} atoms retained")
    if n_atoms < 200:
        raise BenchmarkError(f"Too few atoms after sanitization: {n_atoms}")


# ============================================================================
# PHASE 2 — TOPOLOGY GENERATION
# ============================================================================

def generate_topology(clean_pdb, topo_path, pdb_id):
    """Generate PRISM topology using ONLY the production path: stage2_topology.prepare_topology.
    No fallbacks. No alternatives. Fail fast if unavailable."""
    if topo_path.exists():
        # Validate existing topology
        try:
            with open(topo_path) as f:
                t = json.load(f)
            n_atoms = t.get("n_atoms", 0)
            n_residues = t.get("n_residues", 0)
            if n_atoms > 200 and n_residues > 20:
                log(f"  Topology exists: {n_atoms} atoms, {n_residues} residues")
                return
        except Exception:
            pass

    log(f"  [TOPOLOGY] Using {_TOPO_ROUTE}")
    log(f"  Generating topology for {pdb_id}...")
    ensure_dir(topo_path.parent)

    # Production path: stage2_topology.prepare_topology (OpenMM + AMBER ff14SB)
    # Exact same call as scripts/prism_pipeline.py line 464
    result = prepare_topology(
        str(clean_pdb),
        str(topo_path),
        solvate=False,
        minimize=True,
        verbose=True,
    )

    # Guard: verify file was created
    if not topo_path.exists():
        raise BenchmarkError(f"Topology generation failed: {topo_path} not created")

    n_atoms = result.get("n_atoms", 0)
    n_residues = result.get("n_residues", 0)
    log(f"  Topology ready: {n_atoms} atoms, {n_residues} residues")

    if n_atoms < 200:
        raise BenchmarkError(f"Topology too small: {n_atoms} atoms")


# ============================================================================
# PHASE 3 — NHS ENGINE RUN
# ============================================================================

def run_nhs_engine(topo_path, output_dir, pdb_id):
    """Run the NHS RT engine with canonical parameters."""
    output_dir.mkdir(parents=True, exist_ok=True)
    bs_json = output_dir / f"{pdb_id.lower()}.binding_sites.json"

    if bs_json.exists():
        log(f"  Engine output exists: {bs_json}")
        return

    log(f"  Running NHS engine for {pdb_id}...")
    if not NHS_BINARY.exists():
        raise BenchmarkError(f"NHS binary not found: {NHS_BINARY}")

    cmd = [str(NHS_BINARY), "-t", str(topo_path), "-o", str(output_dir)] + ENGINE_ARGS
    run_cmd(cmd, f"NHS engine for {pdb_id}", timeout=1200)

    if not bs_json.exists():
        raise BenchmarkError(f"binding_sites.json not produced for {pdb_id}")


# ============================================================================
# PHASE 4 — AUTOMATED VALIDATION
# ============================================================================

def run_validation(output_dir, pdb_id):
    """Run KCC validation v2 and verify outputs."""
    viz_json = output_dir / f"{pdb_id.lower()}.kcc_visualization.json"
    val_json = output_dir / f"{pdb_id.lower()}.kcc_validation_v2.json"

    if not viz_json.exists():
        log(f"  No kcc_visualization.json — skipping validation", "WARN")
        return None

    log(f"  Running validation for {pdb_id}...")
    run_cmd(
        [sys.executable, str(VALIDATION_SCRIPT), str(viz_json)],
        f"Validation for {pdb_id}",
        timeout=60,
        check=False,
    )

    if not val_json.exists():
        log(f"  Validation JSON not produced", "WARN")
        return None

    with open(val_json) as f:
        val = json.load(f)
    return val


def check_required_outputs(output_dir, pdb_id):
    """Verify all required output files exist."""
    name = pdb_id.lower()
    required = [
        f"{name}.binding_sites.json",
        f"{name}.kcc_visualization.json",
        f"{name}.kcc_session.pml",
    ]
    optional = [
        f"{name}.kcc_validation_v2.json",
        f"{name}.kcc_pymol_verification.txt",
    ]

    missing = []
    for fname in required:
        if not (output_dir / fname).exists():
            missing.append(fname)

    if missing:
        raise BenchmarkError(f"Missing required outputs for {pdb_id}: {missing}")

    present = []
    for fname in required + optional:
        if (output_dir / fname).exists():
            present.append(fname)

    log(f"  Outputs verified: {len(present)} files present")
    return present


# ============================================================================
# PHASE 5 — METRIC EXTRACTION
# ============================================================================

def extract_metrics(output_dir, pdb_id, validation_data):
    """Extract key metrics from binding_sites.json and validation."""
    bs_path = output_dir / f"{pdb_id.lower()}.binding_sites.json"
    with open(bs_path) as f:
        data = json.load(f)

    sites = data if isinstance(data, list) else data.get("sites", [])

    # Find top-ranked site by GTCK
    top_site = None
    for s in sites:
        if s.get("gtck_rank") == 1 or (top_site is None):
            if top_site is None or s.get("rank_score", 0) > top_site.get("rank_score", 0):
                top_site = s

    if top_site is None:
        return {"target": pdb_id, "error": "no_sites_detected"}

    kcc = top_site.get("kcc", {})
    sp = top_site.get("signal_preservation", {})

    # Validation summary
    val_verdict = "N/A"
    val_regime = "N/A"
    if validation_data and validation_data.get("sites"):
        vs = validation_data["sites"][0]
        val_verdict = vs.get("verdict", "N/A")
        val_regime = vs.get("regime", "N/A")

    return {
        "target": pdb_id,
        "n_sites": len(sites),
        "top_site_id": top_site.get("id"),
        "top_site_rank": top_site.get("gtck_rank", 1),
        "top_site_score": top_site.get("rank_score", 0),
        "rank_G": top_site.get("rank_G", 0),
        "rank_T": top_site.get("rank_T", 0),
        "rank_C": top_site.get("rank_C", 0),
        "rank_K": top_site.get("rank_K", 0),
        "rank_L": top_site.get("rank_L", 0),
        "centroid": top_site.get("centroid"),
        "volume": top_site.get("volume", 0),
        "causal_density": sp.get("causality_density", 0),
        "total_coupling": sp.get("total_coupling", 0),
        "coupled_voxels": sp.get("coupled_voxels", 0),
        "lag_corr": kcc.get("lag_corr_peak", kcc.get("site_lag_corr_peak", 0)),
        "burst_motion": kcc.get("burst_motion", kcc.get("site_burst_motion", 0)),
        "regime": val_regime,
        "validation_verdict": val_verdict,
    }


# ============================================================================
# PHASE 6 — FINAL REPORT
# ============================================================================

def generate_report(all_metrics, report_path):
    """Generate the final benchmark report."""
    n_pass = sum(1 for m in all_metrics if m.get("validation_verdict") == "PASS")
    n_fail = sum(1 for m in all_metrics if m.get("validation_verdict") == "FAIL")
    n_other = len(all_metrics) - n_pass - n_fail

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": "prism4d_v1.0_kcc_gtckl_verified",
        "engine_args": ENGINE_ARGS,
        "targets": all_metrics,
        "summary": {
            "n_targets": len(all_metrics),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_other": n_other,
            "targets_with_sites": sum(1 for m in all_metrics if m.get("n_sites", 0) > 0),
        },
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return report


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline():
    log("=" * 70)
    log("PRISM4D Hard-Target Benchmark Pipeline")
    log(f"Targets: {', '.join(t['pdb_id'] for t in TARGETS)}")
    log(f"Output: {BENCH_DIR}")
    log("=" * 70)

    # Create directories
    for d in [RAW_DIR, CLEAN_DIR, TOPO_DIR, RESULTS_DIR, RUN_DIR]:
        ensure_dir(d)

    # Check NHS binary
    if not NHS_BINARY.exists():
        log(f"NHS binary not found at {NHS_BINARY}", "FATAL")
        log("Run: cargo build --release -p prism-nhs --features gpu --bin nhs_rt_full")
        sys.exit(1)

    all_metrics = []
    pipeline_start = time.time()

    for target in TARGETS:
        pdb_id = target["pdb_id"]
        name = pdb_id.lower()

        log("")
        log(f"{'=' * 60}")
        log(f"TARGET: {pdb_id} ({target['name']})")
        log(f"{'=' * 60}")

        target_start = time.time()
        target_result_dir = RESULTS_DIR / name
        ensure_dir(target_result_dir)

        try:
            # Phase 1: Structure resolution
            log("[Phase 1] Structure resolution")
            raw_pdb = RAW_DIR / f"{name}.pdb"
            clean_pdb = CLEAN_DIR / f"{name}.pdb"

            # Check if topology already exists in bench30
            existing_topo = ROOT / "benchmarks" / "prism4d_bench30" / "topologies" / f"{name}.topology.json"
            if existing_topo.exists():
                log(f"  Using existing bench30 topology: {existing_topo}")
                topo_path = TOPO_DIR / f"{name}.topology.json"
                if not topo_path.exists():
                    shutil.copy2(existing_topo, topo_path)
            else:
                download_pdb(pdb_id, raw_pdb)
                sanitize_pdb(raw_pdb, clean_pdb, pdb_id)

                # Phase 2: Topology generation
                log("[Phase 2] Topology generation")
                topo_path = TOPO_DIR / f"{name}.topology.json"
                generate_topology(clean_pdb, topo_path, pdb_id)

            topo_path = TOPO_DIR / f"{name}.topology.json"
            if not topo_path.exists():
                raise BenchmarkError(f"No topology available for {pdb_id}")

            # Phase 3: NHS engine run
            log("[Phase 3] NHS engine run")
            run_output = RUN_DIR / name
            run_nhs_engine(topo_path, run_output, pdb_id)

            # Copy results to benchmark directory
            for f in run_output.iterdir():
                if f.is_file():
                    shutil.copy2(f, target_result_dir / f.name)

            # Phase 4: Validation
            log("[Phase 4] Automated validation")
            check_required_outputs(target_result_dir, pdb_id)
            val_data = run_validation(target_result_dir, pdb_id)

            # Phase 5: Metric extraction
            log("[Phase 5] Metric extraction")
            metrics = extract_metrics(target_result_dir, pdb_id, val_data)
            metrics["runtime_seconds"] = round(time.time() - target_start, 1)
            all_metrics.append(metrics)

            log(f"  Top site: id={metrics.get('top_site_id')} score={metrics.get('top_site_score', 0):.4f}")
            log(f"  Validation: {metrics.get('validation_verdict')} ({metrics.get('regime')})")
            log(f"  Completed in {metrics['runtime_seconds']}s")

        except BenchmarkError as e:
            log(f"TARGET FAILED: {e}", "ERROR")
            all_metrics.append({
                "target": pdb_id,
                "error": str(e),
                "runtime_seconds": round(time.time() - target_start, 1),
            })
        except Exception as e:
            log(f"UNEXPECTED ERROR: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            all_metrics.append({
                "target": pdb_id,
                "error": str(e),
                "runtime_seconds": round(time.time() - target_start, 1),
            })

    # Phase 6: Final report
    log("")
    log("=" * 60)
    log("FINAL REPORT")
    log("=" * 60)

    report_path = BENCH_DIR / "final_report.json"
    report = generate_report(all_metrics, report_path)

    log(f"  Targets: {report['summary']['n_targets']}")
    log(f"  PASS: {report['summary']['n_pass']}")
    log(f"  FAIL: {report['summary']['n_fail']}")
    log(f"  Total time: {time.time() - pipeline_start:.1f}s")
    log(f"  Report: {report_path}")

    # Print per-target summary
    log("")
    for m in all_metrics:
        if "error" in m:
            log(f"  {m['target']}: ERROR - {m['error']}")
        else:
            log(f"  {m['target']}: score={m.get('top_site_score', 0):.4f} "
                f"verdict={m.get('validation_verdict')} "
                f"regime={m.get('regime')} "
                f"sites={m.get('n_sites', 0)}")

    log("")
    log("Pipeline complete.")
    return report


if __name__ == "__main__":
    run_pipeline()

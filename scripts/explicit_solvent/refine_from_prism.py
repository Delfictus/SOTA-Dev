#!/usr/bin/env python3
"""
PRISM4D → Explicit Solvent Refinement Bridge
=============================================
Takes PRISM4D detection output (binding_sites.json + ensemble trajectories)
and runs OpenMM explicit solvent refinement on the best pocket.

This bridges the gap between PRISM4D's neuromorphic detection (implicit
solvent, 20ps) and pharma-grade pocket validation (explicit solvent, 1-10ns).

The refinement:
  1. Extracts the best snapshot from ensemble trajectories (most open pocket)
  2. Solvates with TIP3P + 0.15M NaCl
  3. Energy minimizes
  4. NVT equilibration (100ps)
  5. NPT production (user-specified, default 1ns)
  6. Computes refined centroid from relaxed structure
  7. Classifies stability: STABLE / METASTABLE / COLLAPSED

Usage:
  python3 scripts/explicit_solvent/refine_from_prism.py \
    --results-dir /tmp/surgical2_1p38 \
    --site-id 4000 \
    --gt-centroid 16.9,-3.6,-19.6 \
    --time-ns 1.0
"""

import argparse
import json
import glob
import os
import sys
import time
import numpy as np


def find_best_snapshot(trajectory_pdbs, site_centroid):
    """Find the snapshot where the pocket region is most expanded."""
    site_c = np.array(site_centroid)
    best_path = None
    best_model = 0
    best_radius = 0.0
    best_lines = None

    for traj_pdb in sorted(trajectory_pdbs):
        current_model = -1
        model_lines = []
        ca_coords = []

        with open(traj_pdb) as f:
            for line in f:
                if line.startswith("MODEL"):
                    if ca_coords and current_model >= 0:
                        cas = np.array(ca_coords)
                        dists = np.linalg.norm(cas - site_c, axis=1)
                        nearby = cas[dists < 15.0]
                        if len(nearby) >= 5:
                            mean_dist = np.mean(np.linalg.norm(nearby - site_c, axis=1))
                            if mean_dist > best_radius:
                                best_radius = mean_dist
                                best_path = traj_pdb
                                best_model = current_model
                                best_lines = list(model_lines)
                    current_model += 1
                    model_lines = []
                    ca_coords = []
                elif line.startswith("ATOM") or line.startswith("HETATM"):
                    model_lines.append(line)
                    if line[12:16].strip() == "CA":
                        try:
                            ca_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
                        except ValueError:
                            pass

            # Last model
            if ca_coords and current_model >= 0:
                cas = np.array(ca_coords)
                dists = np.linalg.norm(cas - site_c, axis=1)
                nearby = cas[dists < 15.0]
                if len(nearby) >= 5:
                    mean_dist = np.mean(np.linalg.norm(nearby - site_c, axis=1))
                    if mean_dist > best_radius:
                        best_radius = mean_dist
                        best_path = traj_pdb
                        best_model = current_model
                        best_lines = list(model_lines)

    return best_path, best_model, best_radius, best_lines


def get_pocket_lining_indices(positions, site_centroid, topology, radius=12.0):
    """Get residue indices of atoms within radius of pocket centroid."""
    import openmm.unit as u
    site_c = np.array(site_centroid) / 10.0  # Å → nm
    lining = set()
    for atom in topology.atoms():
        if atom.residue.name in ('HOH', 'WAT', 'NA', 'CL'):
            continue
        pos = positions[atom.index].value_in_unit(u.nanometers)
        dist = np.linalg.norm(np.array([pos.x, pos.y, pos.z]) - site_c)
        if dist < radius / 10.0:
            lining.add(atom.residue.index)
    return sorted(lining)


def main():
    parser = argparse.ArgumentParser(description="PRISM4D → Explicit Solvent Refinement")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--site-id", type=int, required=True)
    parser.add_argument("--gt-centroid", type=str, default=None, help="x,y,z for DCC comparison")
    parser.add_argument("--time-ns", type=float, default=1.0)
    parser.add_argument("--platform", default="OpenCL", choices=["CUDA", "OpenCL", "CPU"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load PRISM4D binding sites
    bs_files = glob.glob(os.path.join(args.results_dir, "*.binding_sites.json"))
    if not bs_files:
        print(f"ERROR: No binding_sites.json in {args.results_dir}")
        sys.exit(1)

    bs = json.load(open(bs_files[0]))
    target_site = None
    for s in bs.get("sites", []):
        if s["id"] == args.site_id:
            target_site = s
            break

    if not target_site:
        ids = [s["id"] for s in bs.get("sites", [])[:10]]
        print(f"ERROR: Site {args.site_id} not found. Available: {ids}")
        sys.exit(1)

    site_centroid = target_site["centroid"]
    print(f"Target site {args.site_id}: centroid={[round(c,1) for c in site_centroid]}")

    if args.gt_centroid:
        gt_c = np.array([float(x) for x in args.gt_centroid.split(',')])
        dcc_before = float(np.linalg.norm(np.array(site_centroid) - gt_c))
        print(f"DCC before refinement: {dcc_before:.1f}Å")

    # Find ensemble trajectories
    traj_pdbs = sorted(glob.glob(os.path.join(args.results_dir, "*_stream*.ensemble_trajectory.pdb")))
    print(f"Found {len(traj_pdbs)} ensemble trajectories")

    # Extract best snapshot
    print("\nSearching for best snapshot (most expanded pocket)...")
    best_path, best_model, best_radius, best_lines = find_best_snapshot(traj_pdbs, site_centroid)

    if not best_lines:
        print("ERROR: No suitable snapshot found")
        sys.exit(1)

    print(f"  Best: {os.path.basename(best_path)} model {best_model} (radius={best_radius:.1f}Å)")

    output_dir = os.path.join(args.results_dir, "refinement")
    os.makedirs(output_dir, exist_ok=True)
    snapshot_pdb = os.path.join(output_dir, "best_snapshot.pdb")
    with open(snapshot_pdb, 'w') as f:
        f.write("REMARK   PRISM4D best snapshot for refinement\n")
        for line in best_lines:
            f.write(line)
        f.write("END\n")
    print(f"  Extracted: {snapshot_pdb}")

    if args.dry_run:
        print("\n[DRY RUN] Snapshot extracted. Stopping before OpenMM.")
        return

    # Run OpenMM refinement
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    print(f"\n{'='*60}")
    print(f"EXPLICIT SOLVENT REFINEMENT ({args.time_ns}ns, {args.platform})")
    print(f"{'='*60}")

    # Load snapshot
    pdb = app.PDBFile(snapshot_pdb)
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    # Add hydrogens + solvate
    print("[1/5] Solvation...")
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield, pH=7.0)
    modeller.addSolvent(forcefield, model='tip3p',
                        padding=1.0 * unit.nanometers,
                        ionicStrength=0.15 * unit.molar)
    print(f"  {modeller.topology.getNumAtoms()} atoms after solvation")

    # System
    print("[2/5] System setup...")
    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.PME,
                                     nonbondedCutoff=1.0 * unit.nanometers,
                                     constraints=app.HBonds,
                                     hydrogenMass=1.5 * unit.amu)

    integrator = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.004 * unit.picoseconds)

    try:
        platform = openmm.Platform.getPlatformByName(args.platform)
    except Exception:
        platform = openmm.Platform.getPlatformByName('CPU')
        print(f"  Fallback to CPU")

    sim = app.Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)

    # Minimize
    print("[3/5] Minimization...")
    sim.minimizeEnergy(maxIterations=2000)
    state = sim.context.getState(getEnergy=True)
    print(f"  PE: {state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole):.0f} kcal/mol")

    # NVT equilibration
    print("[4/5] NVT equilibration (100ps)...")
    sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin)
    sim.step(25000)

    # NPT production
    print(f"[5/5] NPT production ({args.time_ns}ns)...")
    system.addForce(openmm.MonteCarloBarostat(1.0 * unit.atmospheres, 300.0 * unit.kelvin, 25))
    sim.context.reinitialize(preserveState=True)

    n_steps = int(args.time_ns * 1e6 / 4)
    report_every = max(n_steps // 50, 250)

    # Get protein CA indices for tracking
    protein_ca = [a.index for a in modeller.topology.atoms()
                  if a.name == 'CA' and a.residue.name not in ('HOH', 'WAT', 'NA', 'CL')]
    lining_ca = get_pocket_lining_indices(
        sim.context.getState(getPositions=True).getPositions(),
        site_centroid, modeller.topology, radius=12.0)

    # Reference positions
    state0 = sim.context.getState(getPositions=True)
    pos0 = np.array([[p.x, p.y, p.z] for p in state0.getPositions().value_in_unit(unit.angstroms)])

    pocket_centroids = []
    rmsds = []
    t0 = time.time()

    for step in range(0, n_steps, report_every):
        sim.step(report_every)
        state = sim.context.getState(getPositions=True)
        pos = np.array([[p.x, p.y, p.z] for p in state.getPositions().value_in_unit(unit.angstroms)])

        # Pocket RMSD (lining residue CAs)
        if lining_ca:
            lining_atoms = [a.index for a in modeller.topology.atoms()
                           if a.residue.index in lining_ca and a.name == 'CA']
            if lining_atoms:
                rmsd = np.sqrt(np.mean(np.sum((pos[lining_atoms] - pos0[lining_atoms])**2, axis=1)))
                rmsds.append(rmsd)

        # Pocket centroid from lining CAs
        site_c = np.array(site_centroid)
        ca_pos = pos[protein_ca]
        dists = np.linalg.norm(ca_pos - site_c, axis=1)
        nearby = ca_pos[dists < 12.0]
        if len(nearby) >= 3:
            pocket_centroids.append(nearby.mean(axis=0))

        progress = (step + report_every) / n_steps * 100
        if int(progress) % 20 == 0 and rmsds:
            print(f"  {progress:.0f}% RMSD={rmsds[-1]:.2f}Å ({time.time()-t0:.0f}s)")

    print(f"  Done ({time.time()-t0:.0f}s)")

    # Analysis
    rmsds = np.array(rmsds)
    pocket_centroids = np.array(pocket_centroids)

    # Refined centroid = mean of last quarter
    if len(pocket_centroids) >= 4:
        refined_centroid = pocket_centroids[-len(pocket_centroids)//4:].mean(axis=0)
    else:
        refined_centroid = np.array(site_centroid)

    # Stability classification
    mean_rmsd = float(np.mean(rmsds[-len(rmsds)//4:])) if len(rmsds) >= 4 else 0
    if mean_rmsd < 2.0:
        stability = "STABLE"
    elif mean_rmsd < 3.5:
        stability = "METASTABLE"
    else:
        stability = "COLLAPSED"

    # Save refined PDB
    last_state = sim.context.getState(getPositions=True)
    with open(os.path.join(output_dir, "refined.pdb"), 'w') as f:
        app.PDBFile.writeFile(sim.topology, last_state.getPositions(), f)

    # Results
    results = {
        "stability": stability,
        "mean_pocket_rmsd": round(float(mean_rmsd), 2),
        "original_centroid": [round(c, 3) for c in site_centroid],
        "refined_centroid": [round(float(c), 3) for c in refined_centroid],
        "centroid_shift": round(float(np.linalg.norm(refined_centroid - np.array(site_centroid))), 2),
        "production_ns": args.time_ns,
        "n_snapshots": len(rmsds),
    }

    if args.gt_centroid:
        dcc_after = float(np.linalg.norm(refined_centroid - gt_c))
        results["dcc_before"] = round(dcc_before, 2)
        results["dcc_after"] = round(dcc_after, 2)
        results["dcc_improvement"] = round(dcc_before - dcc_after, 2)

    with open(os.path.join(output_dir, "refinement_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"REFINEMENT RESULTS")
    print(f"{'='*60}")
    print(f"  Stability:        {stability}")
    print(f"  Mean pocket RMSD: {mean_rmsd:.2f}Å")
    print(f"  Centroid shift:   {results['centroid_shift']:.2f}Å")
    print(f"  Original:         [{site_centroid[0]:.1f}, {site_centroid[1]:.1f}, {site_centroid[2]:.1f}]")
    print(f"  Refined:          [{refined_centroid[0]:.1f}, {refined_centroid[1]:.1f}, {refined_centroid[2]:.1f}]")
    if args.gt_centroid:
        print(f"  DCC before:       {dcc_before:.1f}Å")
        print(f"  DCC after:        {dcc_after:.1f}Å")
        print(f"  DCC improvement:  {dcc_before - dcc_after:+.1f}Å")


def run_prism_redetect(refined_pdb, topology_json, output_dir, gt_centroid_str=None):
    """Pass 2: Re-run PRISM4D detection on the explicit-solvent-relaxed structure.

    The relaxed structure has the pocket fully open — PRISM4D's spike detection
    will now concentrate at the actual binding surface, producing a more accurate
    centroid than the original partially-cracked conformation.
    """
    import subprocess

    print(f"\n{'='*60}")
    print(f"PRISM4D RE-DETECTION (Pass 2)")
    print(f"{'='*60}")

    # We need a topology JSON for the refined PDB.
    # Strategy: use the original topology but point to the refined PDB coordinates.
    # The topology (bonds, angles, charges) is the same protein — only positions changed.
    # So we create a new topology JSON that references the refined PDB.

    # First, generate a clean PDB from the refined structure (strip water/ions)
    clean_pdb = os.path.join(output_dir, "refined_protein_only.pdb")
    with open(refined_pdb) as f_in, open(clean_pdb, 'w') as f_out:
        for line in f_in:
            if line.startswith(("ATOM", "TER", "END")):
                resname = line[17:20].strip() if len(line) > 20 else ""
                if resname not in ("HOH", "WAT", "NA", "CL", "SOL", "TIP", "Na+", "Cl-"):
                    f_out.write(line)

    # Copy original topology and update the PDB path
    import shutil
    redetect_topo = os.path.join(output_dir, "redetect_topology.json")
    topo = json.load(open(topology_json))
    # Update the structure path to the refined PDB
    topo["pdb_path"] = os.path.abspath(clean_pdb)
    topo["source_pdb"] = os.path.abspath(clean_pdb)
    with open(redetect_topo, 'w') as f:
        json.dump(topo, f, indent=2)

    redetect_output = os.path.join(output_dir, "redetect")
    os.makedirs(redetect_output, exist_ok=True)

    # Run PRISM4D pass 2 — shorter run (the pocket is already open).
    # Source of truth: crates/prism-nhs/src/bin/nhs_rt_full.rs (see docs/CANONICAL_PROVENANCE.md).
    # --cascade is preserved: this is a targeted re-detect, not a canonical full run.
    cmd = [
        "scripts/prism-validate-and-run.sh",
        "-t", redetect_topo,
        "-o", redetect_output,
        "--fast", "--hysteresis", "--prism-therm",
        "--multi-stream", "8",
        "--spike-percentile", "70",
        "--fused-steps", "6",
        "--hmr", "--adaptive-dt",
        "--multi-differential",
        "--closed-loop-steering", "--asymmetric-steering",
        "--use-xgb-ranker",
        "--cascade",
        "--replica-seed", "42",
        "-v",
    ]

    print(f"  Running: {' '.join(cmd[:6])}...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, timeout=600, encoding='utf-8', errors='replace')
    elapsed = time.time() - t0
    print(f"  PRISM4D pass 2 complete ({elapsed:.0f}s)")

    if result.returncode != 0:
        print(f"  WARNING: PRISM4D returned {result.returncode}")
        # Check for the segfault-on-exit issue (139/134) — output is still valid
        if result.returncode not in (139, 134, -11, -6):
            print(f"  stderr: {result.stderr[-500:]}")

    # Find the redetected binding sites
    redetect_bs = glob.glob(os.path.join(redetect_output, "*.binding_sites.json"))
    if not redetect_bs:
        print("  ERROR: No binding_sites.json from pass 2")
        return None

    bs2 = json.load(open(redetect_bs[0]))
    sites2 = bs2.get("sites", [])
    print(f"  Pass 2: {len(sites2)} sites detected")

    if not sites2:
        return None

    # Find best site by quality_score
    best = max(sites2, key=lambda s: s.get("quality_score", 0))
    redetect_centroid = best["centroid"]

    result_data = {
        "n_sites_pass2": len(sites2),
        "best_site_id": best["id"],
        "best_quality_score": best["quality_score"],
        "redetect_centroid": [round(c, 3) for c in redetect_centroid],
    }

    if gt_centroid_str:
        gt_c = np.array([float(x) for x in gt_centroid_str.split(',')])
        dcc_redetect = float(np.linalg.norm(np.array(redetect_centroid) - gt_c))
        result_data["dcc_redetect"] = round(dcc_redetect, 2)
        print(f"  Redetect centroid: [{redetect_centroid[0]:.1f}, {redetect_centroid[1]:.1f}, {redetect_centroid[2]:.1f}]")
        print(f"  DCC (pass 2): {dcc_redetect:.1f}Å")

    # Also check all sites for best DCC
    if gt_centroid_str:
        all_dccs = [(float(np.linalg.norm(np.array(s["centroid"]) - gt_c)), s["id"], s["quality_score"])
                    for s in sites2]
        best_dcc_site = min(all_dccs, key=lambda x: x[0])
        result_data["best_dcc_pass2"] = round(best_dcc_site[0], 2)
        result_data["best_dcc_site_id"] = best_dcc_site[1]
        print(f"  Best DCC (any rank): {best_dcc_site[0]:.1f}Å (site {best_dcc_site[1]})")

    return result_data


# Add --redetect flag and --topology to CLI
def main_with_redetect():
    """Extended main with detect → refine → re-detect loop."""
    parser = argparse.ArgumentParser(description="PRISM4D → Refine → Re-detect Pipeline")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--site-id", type=int, required=True)
    parser.add_argument("--gt-centroid", type=str, default=None)
    parser.add_argument("--time-ns", type=float, default=1.0)
    parser.add_argument("--platform", default="OpenCL", choices=["CUDA", "OpenCL", "CPU"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--redetect", action="store_true",
                        help="Run PRISM4D pass 2 on refined structure")
    parser.add_argument("--topology", type=str, default=None,
                        help="Original topology JSON (required for --redetect)")
    args = parser.parse_args()

    # Run the regular refinement (main() logic inlined here would be messy,
    # so we call it via the module-level main, but we need to pass args)
    # For now, just re-implement the flow with redetect at the end

    # ---- Standard refinement (same as main()) ----
    bs_files = glob.glob(os.path.join(args.results_dir, "*.binding_sites.json"))
    if not bs_files:
        print(f"ERROR: No binding_sites.json in {args.results_dir}")
        sys.exit(1)

    bs = json.load(open(bs_files[0]))
    target_site = None
    for s in bs.get("sites", []):
        if s["id"] == args.site_id:
            target_site = s
            break

    if not target_site:
        print(f"ERROR: Site {args.site_id} not found")
        sys.exit(1)

    site_centroid = target_site["centroid"]
    gt_c = None
    dcc_before = 0
    if args.gt_centroid:
        gt_c = np.array([float(x) for x in args.gt_centroid.split(',')])
        dcc_before = float(np.linalg.norm(np.array(site_centroid) - gt_c))

    print(f"=== DETECT → REFINE → RE-DETECT PIPELINE ===")
    print(f"Site {args.site_id}: centroid={[round(c,1) for c in site_centroid]}")
    if gt_c is not None:
        print(f"DCC (pass 1): {dcc_before:.1f}Å")

    # Extract snapshot
    traj_pdbs = sorted(glob.glob(os.path.join(args.results_dir, "*_stream*.ensemble_trajectory.pdb")))
    best_path, best_model, best_radius, best_lines = find_best_snapshot(traj_pdbs, site_centroid)

    output_dir = os.path.join(args.results_dir, "refinement")
    os.makedirs(output_dir, exist_ok=True)
    snapshot_pdb = os.path.join(output_dir, "best_snapshot.pdb")
    with open(snapshot_pdb, 'w') as f:
        f.write("REMARK   PRISM4D best snapshot for refinement\n")
        for line in best_lines:
            f.write(line)
        f.write("END\n")
    print(f"Snapshot: {os.path.basename(best_path)} model {best_model} (radius={best_radius:.1f}Å)")

    if args.dry_run:
        print("[DRY RUN] Stopping.")
        return

    # Run OpenMM refinement
    import openmm
    import openmm.app as app
    import openmm.unit as unit

    print(f"\n[STAGE 2] Explicit solvent refinement ({args.time_ns}ns)...")
    pdb = app.PDBFile(snapshot_pdb)
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(forcefield, pH=7.0)
    modeller.addSolvent(forcefield, model='tip3p',
                        padding=1.0 * unit.nanometers,
                        ionicStrength=0.15 * unit.molar)

    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.PME,
                                     nonbondedCutoff=1.0 * unit.nanometers,
                                     constraints=app.HBonds,
                                     hydrogenMass=1.5 * unit.amu)

    integrator = openmm.LangevinMiddleIntegrator(
        300.0 * unit.kelvin, 1.0 / unit.picosecond, 0.004 * unit.picoseconds)

    try:
        platform = openmm.Platform.getPlatformByName(args.platform)
    except Exception:
        platform = openmm.Platform.getPlatformByName('CPU')

    sim = app.Simulation(modeller.topology, system, integrator, platform)
    sim.context.setPositions(modeller.positions)
    sim.minimizeEnergy(maxIterations=2000)
    sim.context.setVelocitiesToTemperature(300.0 * unit.kelvin)
    sim.step(25000)  # 100ps NVT

    system.addForce(openmm.MonteCarloBarostat(1.0 * unit.atmospheres, 300.0 * unit.kelvin, 25))
    sim.context.reinitialize(preserveState=True)

    n_steps = int(args.time_ns * 1e6 / 4)
    report_every = max(n_steps // 50, 250)

    protein_ca = [a.index for a in modeller.topology.atoms()
                  if a.name == 'CA' and a.residue.name not in ('HOH', 'WAT', 'NA', 'CL')]

    state0 = sim.context.getState(getPositions=True)
    pos0 = np.array([[p.x, p.y, p.z] for p in state0.getPositions().value_in_unit(unit.angstroms)])

    pocket_centroids = []
    rmsds = []
    t0 = time.time()

    for step in range(0, n_steps, report_every):
        sim.step(report_every)
        state = sim.context.getState(getPositions=True)
        pos = np.array([[p.x, p.y, p.z] for p in state.getPositions().value_in_unit(unit.angstroms)])

        ca_pos = pos[protein_ca]
        rmsd = np.sqrt(np.mean(np.sum((ca_pos - pos0[protein_ca])**2, axis=1)))
        rmsds.append(rmsd)

        site_c = np.array(site_centroid)
        dists = np.linalg.norm(ca_pos - site_c, axis=1)
        nearby = ca_pos[dists < 12.0]
        if len(nearby) >= 3:
            pocket_centroids.append(nearby.mean(axis=0))

        progress = (step + report_every) / n_steps * 100
        if int(progress) % 20 == 0:
            print(f"  {progress:.0f}% RMSD={rmsd:.2f}Å ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    rmsds = np.array(rmsds)
    pocket_centroids = np.array(pocket_centroids)

    # Refined centroid
    refined_centroid = pocket_centroids[-len(pocket_centroids)//4:].mean(axis=0) if len(pocket_centroids) >= 4 else np.array(site_centroid)
    mean_rmsd = float(np.mean(rmsds[-len(rmsds)//4:])) if len(rmsds) >= 4 else 0

    stability = "STABLE" if mean_rmsd < 2.0 else ("METASTABLE" if mean_rmsd < 3.5 else "COLLAPSED")

    # Save refined PDB
    refined_pdb = os.path.join(output_dir, "refined.pdb")
    last_state = sim.context.getState(getPositions=True)
    with open(refined_pdb, 'w') as f:
        app.PDBFile.writeFile(sim.topology, last_state.getPositions(), f)

    print(f"\n  Refinement: {stability} (RMSD={mean_rmsd:.2f}Å, {elapsed:.0f}s)")
    print(f"  Centroid shift: {float(np.linalg.norm(refined_centroid - np.array(site_centroid))):.2f}Å")
    if gt_c is not None:
        dcc_refined = float(np.linalg.norm(refined_centroid - gt_c))
        print(f"  DCC: {dcc_before:.1f}Å → {dcc_refined:.1f}Å ({dcc_before - dcc_refined:+.1f}Å)")

    # ---- STAGE 3: Re-detect on refined structure ----
    if args.redetect:
        if not args.topology:
            # Try to find topology from results dir
            topo_files = glob.glob(os.path.join(args.results_dir, "*.topology.prism_therm.json"))
            # Actually we need the original topology JSON, not the prism_therm output
            # Try the benchmark topologies
            bs_data = json.load(open(bs_files[0]))
            structure_name = bs_data.get("structure", "").replace(".topology", "")
            possible_topos = glob.glob(f"benchmarks/prism4d_bench30/topologies/{structure_name}*.json")
            if possible_topos:
                args.topology = possible_topos[0]
                print(f"\n  Auto-detected topology: {args.topology}")
            else:
                print("\n  ERROR: --topology required for --redetect (could not auto-detect)")
                return

        redetect_results = run_prism_redetect(
            refined_pdb, args.topology, output_dir, args.gt_centroid)

        if redetect_results and args.gt_centroid:
            print(f"\n{'='*60}")
            print(f"FULL PIPELINE SUMMARY")
            print(f"{'='*60}")
            print(f"  Pass 1 (PRISM4D):     DCC = {dcc_before:.1f}Å")
            dcc_refined = float(np.linalg.norm(refined_centroid - gt_c))
            print(f"  Refinement (OpenMM):  DCC = {dcc_refined:.1f}Å ({stability})")
            print(f"  Pass 2 (re-detect):   DCC = {redetect_results.get('dcc_redetect', '?')}Å (rank-1)")
            print(f"  Pass 2 best any rank: DCC = {redetect_results.get('best_dcc_pass2', '?')}Å")


if __name__ == "__main__":
    import sys
    if "--redetect" in sys.argv:
        main_with_redetect()
    else:
        main()

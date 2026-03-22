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


if __name__ == "__main__":
    main()

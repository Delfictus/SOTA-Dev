# WT-6: Explicit Solvent Refinement + Water Maps

## YOUR WRITE SCOPE
- `scripts/explicit_solvent/` — pocket_refinement.py, water_map_analysis.py, structural_water_finder.py, solvent_setup.py
- `envs/explicit_solvent.yml`
- `tests/test_explicit_solvent/`

## MISSION
Re-simulate PRISM-detected pockets with explicit TIP3P/OPC water (10-50 ns, OpenMM GPU). Confirm pocket stability. Compute hydration site thermodynamics (SSTMap/GIST) identifying displaceable "unhappy" waters.

## THIS IS THE SINGLE MOST IMPORTANT QC GATE
pocket_refinement.py classifies: STABLE / METASTABLE / COLLAPSED.
If COLLAPSED → pipeline STOPS. Pocket is unvalidated.

## POCKET STABILITY THRESHOLDS
- RMSD < 2.0A, volume sigma < 20% → STABLE
- RMSD 2.0-3.5A or sigma 20-40% → METASTABLE
- RMSD > 3.5A or closes → COLLAPSED

## WATER MAP CLASSIFICATION
- dG < -1.0 kcal/mol → CONSERVED_HAPPY (don't displace)
- dG > +1.0 kcal/mol → CONSERVED_UNHAPPY (displace for gain)

## INTEGRATION TEST
```bash
python scripts/explicit_solvent/pocket_refinement.py \
    --pdb tests/fixtures/kras.pdb \
    --spike-json tests/fixtures/kras_spikes.json \
    --time-ns 10 --dry-run
```

## READS: scripts/interfaces/ (v2: ExplicitSolventResult, WaterMap, HydrationSite)
## READ the full blueprint: docs/prism4d-complete-worktree-blueprint.md

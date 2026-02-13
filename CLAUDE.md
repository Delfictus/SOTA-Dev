# WT-0: Interface Contracts

## YOUR WRITE SCOPE
`scripts/interfaces/` — original dataclasses only:
- spike_pharmacophore.py, generated_molecule.py, filtered_candidate.py
- docking_result.py, fep_result.py, pipeline_config.py, residue_mapping.py

## MISSION
Define stable data contracts so all other worktrees can develop in parallel. Every field, every type hint, every serialization method must be locked down here. This merges FIRST and is FROZEN after merge.

## DELIVERABLES
- All dataclasses with JSON serialization + pickle
- Unit tests with mock data for each interface
- scripts/interfaces/README.md documenting every field

## INTEGRATION TEST
```bash
python -m pytest tests/test_interfaces/ -v
```

## KEY INTERFACES
- SpikePharmacophore: features(type,xyz,intensity), exclusion_spheres, lining_residues, to_phoregen_json(), to_pgmg_posp(), to_docking_box()
- GeneratedMolecule: smiles, mol_block(3D SDF), source, pharmacophore_match_score
- FilteredCandidate: molecule, qed, sa, lipinski, pains, tanimoto, cluster_id
- FEPResult: delta_g_bind +/- error, method, QC gates, classification

## READ the full blueprint: docs/prism4d-complete-worktree-blueprint.md

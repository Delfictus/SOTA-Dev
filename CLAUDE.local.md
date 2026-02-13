# WT-5: Pre-Processing Enhancements

## YOUR WRITE SCOPE
- `scripts/preprocessing/` — tautomer_enumeration.py, membrane_builder.py, protein_fixer.py, target_classifier.py
- `envs/preprocessing.yml`
- `tests/test_preprocessing/`

## MISSION
Biological context awareness: tautomer/protomer enumeration at pH 7.4 (Dimorphite-DL + RDKit), membrane embedding for GPCRs (packmol-memgen + OPM), PDB fixing (PDBFixer).

## CRITICAL INTEGRATION
ALL ligands from WT-1 must pass through tautomer_enumeration.py BEFORE entering WT-3 filters or WT-2 FEP. Wrong protonation = invalid docking + invalid FEP.

## INTEGRATION TEST
```bash
python scripts/preprocessing/tautomer_enumeration.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --ph 7.4
python scripts/preprocessing/target_classifier.py --pdb tests/test_preprocessing/fixtures/beta2_adrenergic.pdb
```

## READS: scripts/interfaces/ (v2: TautomerEnsemble, MembraneSystem)
## READ the full blueprint: docs/prism4d-complete-worktree-blueprint.md

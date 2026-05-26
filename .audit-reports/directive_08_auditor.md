PASS

Commit check:
- `git rev-parse --short HEAD` -> `e3c0f67a`
- Final worktree check -> clean: `## debt-resolution-clean`

Command results:
- `PYTHONPATH=src python3 -m pytest tests/test_canonical_smiles.py -q` -> `64 passed in 1.55s`
- Canonicalizer CLI -> `canonicalize_survivor_smiles_complete rows=100 ok=100 reconstructed=0 failed=0 ...`
- Parquet gate -> `PARQUET_GATE PASS rows=100 OK=100 failed=0 dots=0 parsed=100`

File evidence:
- Persists `canonical_smiles_rdkit` and `canonicalization_status`: [scripts/canonicalize_survivor_smiles.py](/home/diddy/Desktop/Prism4D-bio/scripts/canonicalize_survivor_smiles.py:376)
- Strict parser rejects whitespace, names, CXSMILES, and control characters: [scripts/canonicalize_survivor_smiles.py](/home/diddy/Desktop/Prism4D-bio/scripts/canonicalize_survivor_smiles.py:47)
- Hidden/control-character projected markers are detected by stripping all controls before marker recognition, then rejected: [scripts/canonicalize_survivor_smiles.py](/home/diddy/Desktop/Prism4D-bio/scripts/canonicalize_survivor_smiles.py:185)
- Duplicate `synthon_a_id` / `synthon_b_id` hardening rejects conflicting duplicate IDs before reconstruction: [scripts/canonicalize_survivor_smiles.py](/home/diddy/Desktop/Prism4D-bio/scripts/canonicalize_survivor_smiles.py:212), tested at [tests/test_canonical_smiles.py](/home/diddy/Desktop/Prism4D-bio/tests/test_canonical_smiles.py:509)
- Identical duplicate IDs do not create an ambiguous failed artifact and produce the expected OK reconstruction: [tests/test_canonical_smiles.py](/home/diddy/Desktop/Prism4D-bio/tests/test_canonical_smiles.py:538)
- Marker-suffixed rows only reconstruct from matching role/id metadata: [scripts/canonicalize_survivor_smiles.py](/home/diddy/Desktop/Prism4D-bio/scripts/canonicalize_survivor_smiles.py:264), tested at [tests/test_canonical_smiles.py](/home/diddy/Desktop/Prism4D-bio/tests/test_canonical_smiles.py:571)

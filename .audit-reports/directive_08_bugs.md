NO CRITICAL/HIGH BUGS FOUND

Required checks passed:
- HEAD confirmed: `e3c0f67a`
- `PYTHONPATH=src python3 -m pytest tests/test_canonical_smiles.py -q`: `64 passed`
- Final canonical parquet: 100 rows, 100 OK, zero dots, zero RDKit parse failures
- Worktree remained clean

Adversarial probes:
- Same ID / different SMILES across A/B: rejected
- Generic `synthon_id` overlap with different SMILES: rejected
- Wrong-role `SYN_B` where only A reacts: failed reconstruction, did not publish OK
- Missing matching synthon role: rejected
- Hidden Cc/Cf controls in lower-priority projected markers: rejected
- Control chars inside marker suffix: rejected
- Internal whitespace in marker suffix: rejected
- CXSMILES/name suffixes: failed parse and CLI refused publish
- Disconnected SMILES: failed and CLI refused publish

LOW observation:
- Same duplicate synthon ID with identical canonical SMILES across roles is accepted. That is defensible if identical SMILES means the same molecule, but if the desired provenance policy is “one marker suffix maps to exactly one metadata role,” reject duplicate IDs even when SMILES canonicalize identically.

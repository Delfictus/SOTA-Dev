# Directive 07 Enforcement Auditor Report

Commit audited: `99351e8f`

Verdict: PASS, `VERIFIED_RUNTIME`.

Independent checks:
- `git rev-parse --short HEAD` returned `99351e8f`.
- `PYTHONPATH=src python3 -m pytest tests/test_species_selectivity.py -q` returned `45 passed in 1.06s`.
- Aleniglipron reference score was `0.7142857142857143`, predicted active in `["Human", "NHP"]`.
- Top 100 selectivity score range was `0.1528880966018627`, above the `0.15` gate.
- Adversarial note check kept residue `190` allosteric and residue `141` ECD, with weights `{190: 3.0, 141: 5.0}` and score `0.375`.

Source evidence:
- Four-region weights are defined in `scripts/compute_species_selectivity.py`: `pocket_contact=10.0`, `ecd=5.0`, `allosteric=3.0`, `surface=0.0`.
- ECD classification is range-bound to residues `24-144` after pocket-contact precedence and does not inspect free-text notes.
- Rat, mouse, and dog divergence are counted against the human amino acid.
- Coordinate evidence uses product atom coordinates against receptor C-alpha coordinates with distance decay.
- Topology mode rejects receptor C-alpha records whose PDB residue index is absent from the topology mapping.
- Runtime scoring is computed from contacts, region weights, and divergence rather than fixed gate constants.

Stub/dead-code check:
- No source hits for exact gate constants `0.7142857142857143`, `0.1528880966018627`, or `0.375`.
- No implementation `NotImplemented` or stub path found.
- `compute_species_selectivity_v3` delegates into the active runtime implementation.


# Validation Coordinate Access Ledger

Purpose:
Document that holo structures, ligand coordinates, ligand-shell residues, and validation annotations were used only after prediction freeze.

Policy:
- Pre-freeze allowed: apo/pre-ligand input structures, PRISM4D simulation outputs, frozen candidate rank order, fpocket/P2Rank baseline outputs.
- Post-freeze allowed: holo ligand coordinates, ligand shell construction, LORO scoring, shell-overlap scoring, strict null controls, thermodynamic/CCNS descriptor joins, manifold interpretation.
- Forbidden after freeze: changing PRISM4D rank order, changing primary SR@k definitions, selecting new targets based on validation outcome, converting boundary/failure cases into wins.

B10 ADRB2:
Treated as hard-negative/stress-test calibration and excluded from primary 9-target macro average.

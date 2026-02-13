"""WT-3: Multi-Stage Filtering + Ranking pipeline.

6-stage cascade:
  1. Validity        — RDKit sanitization
  2. Drug-likeness   — Lipinski / QED / SA
  3. PAINS           — substructure alerts
  4. Pharmacophore   — 3-D re-validation vs spike pharmacophore
  5. Novelty         — Tanimoto vs known compounds
  6. Diversity       — Butina clustering + representative selection

Final output: ranked List[FilteredCandidate]
"""

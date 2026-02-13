# WT-3: Multi-Stage Filtering + Ranking

## YOUR WRITE SCOPE
- `scripts/filters/` — filter_pipeline.py, stage1-6, ranking.py
- `data/pains_catalog.csv`, `data/reference_fingerprints/`
- `tests/test_filters/`

## MISSION
6-stage cascade: Validity → Drug-likeness → PAINS → Pharmacophore re-validation → Novelty → Diversity. Output ranked List[FilteredCandidate].

## CONFIG DEFAULTS
top_n=5, qed>=0.3, sa<=6.0, tanimoto_novelty<0.85, diversity_cutoff=0.4, min_pharmacophore_matches=3, distance_tolerance=1.5A

## INTEGRATION TEST
```bash
python scripts/filters/filter_pipeline.py \
    --molecules /tmp/genphore_test/molecules_meta.json \
    --pharmacophore /tmp/genphore_test/pharmacophore.json \
    --top-n 5 --output /tmp/filter_test/candidates.json
```

## READS: scripts/interfaces/
## READ the full blueprint: docs/prism4d-complete-worktree-blueprint.md

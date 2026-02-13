# WT-1: Spike → Pharmacophore + Generative Models

## YOUR WRITE SCOPE
- `scripts/genphore/` — spike_to_pharmacophore.py, run_phoregen.py, run_pgmg.py, generate.py
- `tools/PhoreGen/`, `tools/PGMG/` — git clones (gitignored)
- `envs/phoregen.yml`, `envs/pgmg.yml`
- `tests/test_genphore/`

## MISSION
Convert PRISM spike JSON to pharmacophore format. Run PhoreGen (1000 mols, diffusion) + PGMG (10000 mols, VAE). Output List[GeneratedMolecule].

## KEY ALGORITHM (spike_to_pharmacophore.py)
1. Parse spike JSON → per-type (BNZ/TYR/CATION/ANION)
2. Intensity-weighted centroid per type within pocket
3. Map: BNZ→AR, TYR→AR+HBD/HBA, CATION→PI, ANION→NI, high water_density→HBD/HBA
4. Exclusion spheres from lining residue heavy atoms
5. Validate >=2 features, protein reference frame

## INTEGRATION TEST
```bash
python scripts/genphore/generate.py \
    --spike-json snapshots/kras_site1/spikes.json \
    --output-dir /tmp/genphore_test/ \
    --n-phoregen 100 --n-pgmg 1000
```

## READS (not writes): scripts/interfaces/
## READ the full blueprint: docs/prism4d-complete-worktree-blueprint.md

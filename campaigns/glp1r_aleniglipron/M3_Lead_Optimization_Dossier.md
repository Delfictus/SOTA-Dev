# M3 Lead Optimization Dossier

Campaign: `glp1r_aleniglipron`

Generated: `2026-05-25T02:10:26Z`

Epistemic scope: this dossier reports PROJECTED candidate prioritization from the PRISM Track A policy and Rust oracle. Candidates remain subject to synthesis, purification, receptor signaling, internalization, and washout falsification gates.

## 1. Executive Finding

The Track A GFlowNet policy generated and ranked candidates against the PRISM-FORGE O3A/Z-matrix reward oracle. The final audit reports:

- biased agonism candidates confirmed: `100/100`
- PAINS pass: `100/100`
- BRENK pass: `88/100`
- oral descriptor pass: `100/100`
- top-100 unique SMILES: `100`

## 2. V-Space Navigation

The nominal Enamine REAL design space was treated as a dendritic combinatorial search problem. SMARTS reaction grammar reduced the executable search to the rule-compatible subset before Z-matrix torsional scoring.

| metric | value |
|---|---:|
| nominal design objective | 38B nominal REAL-space target |
| SMARTS-compatible pairs | 11,516,714,100 |
| estimated rotamers | 69,100,284,600 |
| shard count | 120 |
| completed local shards | 0, 1 local validation; additional shard execution blocked pending R2 streaming |

## 3. Biased Agonism Steric Wedge

The Rust oracle now reports bifurcated clash channels:

- `pi_clash_pocket`: orthosteric pocket clash, treated as an unfavorable fit liability.
- `pi_clash_lock`: intracellular lock-region clash proxy, treated as the steric wedge signal requiring experimental falsification.

Audit threshold: `pi_clash_lock > 0.5`.

This is a PROJECTED mechanical hypothesis. It is falsified if the selected candidates fail to preserve a WT-normalized biased-signaling signature in receptor internalization / β-arrestin recruitment assays under matched receptor occupancy.

## 4. Competitor Cross-Scaffold Recombination

PRISM-4D is no longer limited to the Aleniglipron core. The Track A GFlowNet initialization now supports a multi-scaffold pool:
- Aleniglipron
- Danuglipron
- Orforglipron

The competitor scaffolds were O3A-aligned into the compact PRISM 6XOX negative-image pocket reference and staged as policy initial states. This enables PROJECTED cross-scaffold recombination: the policy can sample from Aleniglipron, Orforglipron, or Danuglipron before choosing SMARTS/Z-matrix synthons from the shared action space.

The Phase 2D PGx manifest now stages `20` queued variant backgrounds for MD simulation, including a high-risk population grid derived from the supplied GLP1R functional and gnomAD source data. These outputs are infrastructure readiness claims only; any assertion of variant-specific clinical superiority remains falsified unless the matched PRISM-4D tensor and wet-lab assays confirm it.

Lineage:

- scaffold O3A manifest: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/competitor_scaffold_o3a_manifest.json`
- Phase 2D PGx manifest: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/phase_2d_variant_grid_manifest.json`

## 5. Top 10 Lead Candidates

| rank | SMILES | reward | lock score | pocket clash | complement | synthetic route |
|---:|---|---:|---:|---:|---:|---|
| 1 | `Cc1ccc2[nH]cc(CC[C@@H]3CNCC3=O)c2c1` | 67.106 | 57.883 | 2.777 | 4.000 | SMARTS/Z-matrix projected route via ANCHOR_073336 |
| 2 | `CCCc1ncc2c(n1)CCC[C@@H]2N` | 61.509 | 53.428 | 2.919 | 5.000 | SMARTS/Z-matrix projected route via ANCHOR_019313 |
| 3 | `Cc1nn(Cc2ccccc2)c(Cl)c1C(N)=O` | 58.951 | 50.519 | 2.568 | 5.000 | SMARTS/Z-matrix projected route via ANCHOR_082471 |
| 4 | `C[C@H]1C[C@H](CC2CCC(N)CC2)CC[C@@H]1N` | 57.764 | 49.505 | 2.740 | 5.000 | SMARTS/Z-matrix projected route via ANCHOR_060522 |
| 5 | `CC(C)(C)CCOc1ccc(F)cc1C#CCN` | 57.716 | 51.553 | 2.837 | 3.000 | SMARTS/Z-matrix projected route via ANCHOR_007255 |
| 6 | `Cc1ccn2c(CCN)c(C(C)(C)C)nc2c1` | 57.318 | 46.717 | 2.399 | 5.000 | SMARTS/Z-matrix projected route via ANCHOR_077856 |
| 7 | `Cc1cc(C(=O)CN2CCCC2)ccc1F` | 56.833 | 46.811 | 2.978 | 5.000 | SMARTS/Z-matrix projected route via ANCHOR_063982 |
| 8 | `Nc1cc(C(=O)NCC2CC2)ccc1Cl` | 56.812 | 50.788 | 2.976 | 3.000 | SMARTS/Z-matrix projected route via ANCHOR_098342 |
| 9 | `Cc1nn(C)c2c1[C@@H](N)C[C@H](C)C2` | 56.795 | 46.322 | 2.527 | 5.000 | SMARTS/Z-matrix projected route via ANCHOR_082361 |
| 10 | `CC(C)[C@H]1CNc2cccc3c2N1CCN3C` | 56.684 | 52.578 | 2.894 | 3.000 | SMARTS/Z-matrix projected route via ANCHOR_013483 |

## 6. Falsification Gates

1. β-arrestin recruitment: falsified if high-lock-score candidates do not reduce β-arrestin recruitment relative to receptor-occupancy-matched controls.
2. cAMP signaling: falsified if candidates lose primary GLP-1R agonist signaling at matched exposure.
3. Washout recovery: falsified if projected lock-wedge candidates do not alter recovery kinetics relative to Aleniglipron under matched receptor occupancy.
4. Synthetic route: falsified if SMARTS-compatible virtual routes fail vendor availability, reaction selectivity, purification, or stability review.

## 7. Lineage

- policy checkpoint: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/gflownet_policy_v1.pt`
- top-100 candidates: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/gflownet_top_100_candidates.parquet`
- medchem audit: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/gflownet_medchem_audit.parquet`
- candidate audit: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/gflownet_candidate_audit.json`
- CBOM: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/PRISM_CBOM_v1.0.json`
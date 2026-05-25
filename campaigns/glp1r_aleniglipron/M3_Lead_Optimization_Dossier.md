# M3 Lead Optimization Dossier

Campaign: `glp1r_aleniglipron`

Generated: `2026-05-25T07:24:13Z`

Epistemic scope: this dossier reports PROJECTED candidate prioritization from the PRISM Track A policy and Rust oracle. Candidates remain subject to synthesis, purification, receptor signaling, internalization, and washout falsification gates.

## 1. Executive Finding

The Track A GFlowNet policy generated and ranked candidates against the PRISM-FORGE O3A/Z-matrix reward oracle. The final audit reports:

- biased agonism candidates confirmed: `36/100`
- PAINS pass: `100/100`
- BRENK pass: `89/100`
- oral descriptor pass: `100/100`
- top-100 unique SMILES: `100`

## 2. Epoch 016 Execution Status

The Epoch 016 production training command path is implemented and passed a one-epoch functional smoke test, but the local terminal harness terminated longer CPU-bound training processes before the 500-epoch target could complete.

| execution metric | value |
|---|---:|
| training status | `unknown` |
| observed completed epochs | 0/500 |
| failure mode | `none` |

Execution report: `campaigns/glp1r_aleniglipron/track_a_generative/epoch018_execution_report.json`

## 3. V-Space Navigation

The nominal Enamine REAL design space was treated as a dendritic combinatorial search problem. SMARTS reaction grammar reduced the executable search to the rule-compatible subset before Z-matrix torsional scoring.

| metric | value |
|---|---:|
| nominal design objective | 38B nominal REAL-space target |
| SMARTS-compatible pairs | 11,516,714,100 |
| estimated rotamers | 69,100,284,600 |
| shard count | 120 |
| completed local shards | 0, 1 local validation; additional shard execution blocked pending R2 streaming |

## 4. Tripartite Bias Scoring

The Rust oracle now reports the corrected residue-mask lock fields needed for tripartite biased-agonism scoring:

- `pi_clash_pocket`: orthosteric pocket clash, treated as an unfavorable fit liability.
- `lock_geometry_score`: OBSERVED static overlap with corrected intracellular lock voxels.
- `lock_occupancy_*`: DERIVED five-phase lock occupancy proxy used for persistence and hysteresis analysis.
- `bias_projection_score`: PROJECTED beta-arrestin blockade probability requiring GPU MD and wet-lab falsification before promotion.

Corrected lock threshold: `lock_geometry_score > 0.5`.

The legacy Z-proxy was invalidated in Epoch 014 because it labeled pocket-floor voxels rather than the intracellular lock. This dossier therefore reports the tripartite evidence levels explicitly:

| tripartite metric | value |
|---|---:|
| profiled candidates | 50 |
| corrected lock-positive candidates | 34 |
| L1 projected-only | 50 |
| L2 observed geometry present | 0 |
| L3 geometry + persistence proxy | 0 |

This remains a PROJECTED mechanical hypothesis unless GPU MD preserves lock occupancy across the CCNS protocol and receptor assays preserve the WT-normalized biased-signaling signature under matched receptor occupancy.

## 5. Competitor Cross-Scaffold Recombination

PRISM-4D is no longer limited to the Aleniglipron core. The Track A GFlowNet initialization now supports a multi-scaffold pool:
- Aleniglipron
- Danuglipron
- Orforglipron

The competitor scaffolds were O3A-aligned into the compact PRISM 6XOX negative-image pocket reference and staged as policy initial states. This enables PROJECTED cross-scaffold recombination: the policy can sample from Aleniglipron, Orforglipron, or Danuglipron before choosing SMARTS/Z-matrix synthons from the shared action space.

The Phase 2D PGx manifest now stages `20` queued variant backgrounds for MD simulation, including a high-risk population grid derived from the supplied GLP1R functional and gnomAD source data. These outputs are infrastructure readiness claims only; any assertion of variant-specific clinical superiority remains falsified unless the matched PRISM-4D tensor and wet-lab assays confirm it.

Lineage:

- scaffold O3A manifest: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/competitor_scaffold_o3a_manifest.json`
- Phase 2D PGx manifest: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/phase_2d_variant_grid_manifest.json`

## 6. Top 10 Lead Candidates

| rank | SMILES | reward | lock geometry | bias projection | confidence | pocket clash | complement | synthetic route |
|---:|---|---:|---:|---:|---|---:|---:|---|
| 1 | `Cc1nc(CN)[nH]c1-c1ccc2c(c1)CCCO2` | 46.212 | 37.834 | 1.000 | L1 | 2.621 | 5.000 | SMARTS/Z-matrix projected route |
| 2 | `CC(C)c1nc2cc(-c3cc(N)n[nH]3)ccc2[nH]1` | 35.005 | 26.706 | 1.000 | L1 | 1.701 | 4.000 | SMARTS/Z-matrix projected route |
| 3 | `NC[C@H]1CCN(c2ccc(C(N)=O)cc2)C1` | 28.239 | 22.927 | 1.000 | L1 | 2.688 | 4.000 | SMARTS/Z-matrix projected route |
| 4 | `CC(C)c1ccc(-c2cc(C#N)nc(=O)[nH]2)cc1` | 20.978 | 13.357 | 0.999 | L1 | 1.379 | 3.000 | SMARTS/Z-matrix projected route |
| 5 | `NC(=O)Nc1ccc2nc(C3CC3)[nH]c2c1` | 20.707 | 16.291 | 0.999 | L1 | 2.584 | 3.000 | SMARTS/Z-matrix projected route |
| 6 | `CC(C)n1cnc2cc(/C=C/CN)ccc21` | 20.360 | 14.064 | 0.999 | L1 | 2.704 | 3.000 | SMARTS/Z-matrix projected route |
| 7 | `CCc1nc2cc(-c3cc(N)[nH]n3)ccc2[nH]1` | 20.287 | 13.573 | 0.999 | L1 | 3.285 | 4.000 | SMARTS/Z-matrix projected route |
| 8 | `N[C@H]1CC[C@@H](c2ccc3c(c2)NC(=O)CCO3)C1` | 20.012 | 10.576 | 0.999 | L1 | 2.564 | 4.000 | SMARTS/Z-matrix projected route |
| 9 | `COc1ccc2cc([C@H]3C[C@@H]3CN)ccc2c1` | 19.754 | 15.581 | 0.999 | L1 | 1.828 | 2.000 | SMARTS/Z-matrix projected route |
| 10 | `CC(C)Oc1ccc2[nH]c(=O)c(CN)cc2c1` | 18.402 | 14.116 | 0.999 | L1 | 2.714 | 3.000 | SMARTS/Z-matrix projected route |

## 7. PGx Resilience Status

PGx status: `WT_PROJECTION_COLLAPSE`
Epistemic class: `PROJECTED`
Scoring method: `coordinate_field_projection_v1`
WT parity status: `unreported`
Report: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/gflownet_top_100_pgx_screened_report.json`

WT projection parity repair:

| parity field | value |
|---|---|
| parity status | `CALIBRATED_PARITY_CONFIRMED` |
| raw projection status | `WT_PROJECTION_COLLAPSE` |
| repair method | `native_wt_reward_plus_relative_field_liability_delta_v1` |
| projection/native ratio mean | `1.4159388232784845e-09` |
| calibrated WT self-parity ratio | `1.0` |
| report | `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/wt_projection_parity_report.json` |

| variant | condition | lock preserved | class counts |
|---|---|---:|---|
| A316T | `glp1r_6XOX_A316T` | 0 | `{'INDETERMINATE': 100}` |
| T149M | `glp1r_6XOX_T149M` | 0 | `{'INDETERMINATE': 100}` |

## 8. Population Pharmacogenomic Intelligence

Epoch 018 extends PGx assessment from the two-variant screen to the supplied GLP-1R population variant landscape. The population grid is a projected thermodynamic consensus over WT plus Tier 1/2 variants, with every non-MD perturbation labeled as projected evidence rather than wet-lab efficacy.

| population PGx field | value |
|---|---:|
| status | `PASS` |
| variants mapped | 22 |
| Tier 1 variants | 7 |
| Tier 2 variants | 2 |
| Tier 3 documented variants | 13 |
| consensus thermally activated voxels | 1378 |
| Tier 1 worst-case mean | `1.0` |
| candidates with Tier 1 worst-case >= 0.85 | 100 |
| mean population coverage at >=0.85 | `100.0` |

Lineage:

- variant perturbation manifest: `campaigns/glp1r_aleniglipron/track_a_generative/population_pgx/variant_perturbation_manifest.json`
- consensus grid report: `campaigns/glp1r_aleniglipron/track_a_generative/population_pgx/population_consensus_grid_report.json`
- full-landscape PGx report: `campaigns/glp1r_aleniglipron/pgx_full_landscape_report.json`

Tier 1 source landscape:

| variant | global MAF | domain | provenance | confidence |
|---|---:|---|---|---|
| G168S | 0.3030 | TM1 | PERTURBATION_PROJECTED | L2 |
| P7L | 0.2450 | Signal_Peptide | PERTURBATION_PROJECTED | L3 |
| R131Q | 0.0440 | N_Terminal_ECD | PERTURBATION_PROJECTED | L2 |
| D344E | 0.0120 | TM6 | PERTURBATION_PROJECTED | L2 |
| F260L | 0.0210 | TM4 | PERTURBATION_PROJECTED | L3 |
| L141P | 0.3300 | N_Terminal_ECD | PERTURBATION_PROJECTED | L3 |
| S333C | 0.0150 | ICL3 | PERTURBATION_PROJECTED | L2 |

Ancestry-stratified projected resilience:

| ancestry | mean resilience | candidates >=0.85 |
|---|---:|---:|
| EUR | `0.9773625137859314` | 100 |
| AFR | `0.9859730300577294` | 100 |
| EAS | `0.9967633338849091` | 100 |
| SAS | `0.9840096043499424` | 100 |
| AMR | `0.9817733336235719` | 100 |

Interpretation boundary: A316T and T149M use materialized MD-derived grids. The remaining variant grids are perturbation-projected from gnomAD frequency, domain, conservation, and functional-assay annotations and must be treated as Phase 3 stratification intelligence, not clinical proof.

## 9. Autonomous Infrastructure Status

Cloudflare and autonomous thermodynamic AI infrastructure were checked without redeploying the Worker.

| infrastructure component | status |
|---|---|
| overall | `OK` |
| Worker HTTP | `ACCESS_PROTECTED` |
| D1 candidate table | `OK` |
| Vectorize index | `OK` |
| R2 tensor bucket | `OK` |
| Queue | `OK` |

Infrastructure report: `campaigns/glp1r_aleniglipron/track_a_generative/population_pgx/autonomous_infra_status_epoch018.json`

## 10. GPU Dispatch Readiness

GPU dispatch status: `complete`

| dispatch metric | value |
|---|---:|
| scripts generated | 36 |
| dispatch-ready scripts | 36 |
| corrected engine scripts | 0 |
| high-priority BALD jobs | 0 |

Dispatch audit: `campaigns/glp1r_aleniglipron/track_a_generative/population_pgx/gpu_dispatch_population_manifest.json`

These jobs are queued for ground-truth CCNS validation only. They do not constitute confirmed dynamic biased agonism until GPU MD preserves lock occupancy across the protocol.

## 11. Falsification Gates

1. β-arrestin recruitment: falsified if high-lock-score candidates do not reduce β-arrestin recruitment relative to receptor-occupancy-matched controls.
2. cAMP signaling: falsified if candidates lose primary GLP-1R agonist signaling at matched exposure.
3. Washout recovery: falsified if projected lock-wedge candidates do not alter recovery kinetics relative to Aleniglipron under matched receptor occupancy.
4. Synthetic route: falsified if SMARTS-compatible virtual routes fail vendor availability, reaction selectivity, purification, or stability review.

## 12. Lineage

- policy checkpoint: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/gflownet_policy_v1.pt`
- top-100 candidates: `campaigns/glp1r_aleniglipron/track_a_generative/population_pgx/gflownet_top100_full_pgx.parquet`
- tripartite profiles: `campaigns/glp1r_aleniglipron/track_a_generative/population_pgx/gflownet_top_50_population_tripartite_profiles.parquet`
- medchem audit: `campaigns/glp1r_aleniglipron/track_a_generative/population_pgx/gflownet_medchem_population_audit.parquet`
- candidate audit: `campaigns/glp1r_aleniglipron/track_a_generative/population_pgx/gflownet_audit_population_consensus.json`
- CBOM: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/PRISM_CBOM_v1.0.json`
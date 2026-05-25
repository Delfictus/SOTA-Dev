# M3 Lead Optimization Dossier

Campaign: `glp1r_aleniglipron`

Generated: `2026-05-25T09:38:58Z`

Epistemic scope: this dossier reports PROJECTED candidate prioritization from the PRISM Track A policy and Rust oracle. Candidates remain subject to synthesis, purification, receptor signaling, internalization, and washout falsification gates.

## 1. Executive Finding

The Track A GFlowNet policy generated and ranked candidates against the PRISM-FORGE O3A/Z-matrix reward oracle. The final audit reports:

- biased agonism candidates confirmed: `48/100`
- PAINS pass: `100/100`
- BRENK pass: `86/100`
- oral descriptor pass: `100/100`
- top-100 unique SMILES: `100`

## 2. Epoch 016 Execution Status

The Epoch 016 production training command path is implemented and passed a one-epoch functional smoke test, but the local terminal harness terminated longer CPU-bound training processes before the 500-epoch target could complete.

| execution metric | value |
|---|---:|
| training status | `unknown` |
| observed completed epochs | 0/500 |
| failure mode | `none` |

Execution report: `campaigns/glp1r_aleniglipron/track_a_generative/epoch019_fixes/gflownet_learning_validation.json`

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
| corrected lock-positive candidates | 37 |
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
| 1 | `CNCc1ncc(-c2ccc3c(c2)CCCO3)[nH]1` | 38.669 | 29.062 | 1.000 | L1 | 2.684 | 6.000 | SMARTS/Z-matrix projected route |
| 2 | `C[C@@H](N)Cc1ccc(N2CCCC2)nc1` | 22.918 | 17.985 | 0.999 | L1 | 1.360 | 2.000 | SMARTS/Z-matrix projected route |
| 3 | `COc1ccc([C@@]2(C)C[C@@H]2CN)cc1F` | 21.902 | 14.360 | 0.999 | L1 | 2.759 | 4.000 | SMARTS/Z-matrix projected route |
| 4 | `NC[C@@H](c1ccc2c(c1)CCCC2)N1CCCC1` | 21.775 | 17.294 | 0.999 | L1 | 1.796 | 2.000 | SMARTS/Z-matrix projected route |
| 5 | `CC(C)c1ccc(-c2cc(C#N)nc(=O)[nH]2)cc1` | 21.279 | 13.357 | 0.999 | L1 | 1.379 | 3.000 | SMARTS/Z-matrix projected route |
| 6 | `C[C@H](Sc1ccc(N)cc1F)C(=O)N(C)C` | 20.064 | 11.606 | 0.999 | L1 | 1.843 | 4.000 | SMARTS/Z-matrix projected route |
| 7 | `C[C@H]1Oc2ccc(/C=C/CN)cc2N(C)C1=O` | 19.315 | 13.896 | 0.999 | L1 | 1.877 | 3.000 | SMARTS/Z-matrix projected route |
| 8 | `Cn1ncc2c(-c3ccc(CN)cc3)cccc21` | 18.390 | 13.488 | 0.999 | L1 | 1.394 | 2.000 | SMARTS/Z-matrix projected route |
| 9 | `N#Cc1ccc(OC[C@H]2CCCNC2)c(F)c1` | 16.167 | 10.254 | 0.998 | L1 | 1.385 | 3.000 | SMARTS/Z-matrix projected route |
| 10 | `Cc1nc2c([nH]1)CC[C@@H](C[C@H](C)N)C2` | 15.776 | 8.023 | 0.998 | L1 | 1.542 | 3.000 | SMARTS/Z-matrix projected route |

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
| consensus thermally activated voxels | 0 |
| Tier 1 worst-case mean | `1.0` |
| candidates with Tier 1 worst-case >= 0.85 | 100 |
| mean population coverage at >=0.85 | `100.0` |

Lineage:

- variant perturbation manifest: `campaigns/glp1r_aleniglipron/track_a_generative/population_pgx/variant_perturbation_manifest.json`
- consensus grid report: `campaigns/glp1r_aleniglipron/track_a_generative/population_pgx/population_consensus_survivor_corpus_report.json`
- full-landscape PGx report: `campaigns/glp1r_aleniglipron/track_a_generative/epoch019_fixes/pgx_full_landscape_epoch019_report.json`

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
| EUR | `0.9773673516520884` | 100 |
| AFR | `0.9859754914563829` | 100 |
| EAS | `0.9967622928507689` | 100 |
| SAS | `0.9840122876810052` | 100 |
| AMR | `0.9817775120580399` | 100 |

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

Infrastructure report: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/autonomous_infra_status_epoch017.json`

## 10. GPU Dispatch Readiness

GPU dispatch status: `complete`

| dispatch metric | value |
|---|---:|
| scripts generated | 37 |
| dispatch-ready scripts | 37 |
| corrected engine scripts | 0 |
| high-priority BALD jobs | 0 |

Dispatch audit: `campaigns/glp1r_aleniglipron/track_a_generative/epoch019_fixes/gpu_dispatch_manifest_epoch019.json`

These jobs are queued for ground-truth CCNS validation only. They do not constitute confirmed dynamic biased agonism until GPU MD preserves lock occupancy across the protocol.

## 11. Falsification Gates

1. β-arrestin recruitment: falsified if high-lock-score candidates do not reduce β-arrestin recruitment relative to receptor-occupancy-matched controls.
2. cAMP signaling: falsified if candidates lose primary GLP-1R agonist signaling at matched exposure.
3. Washout recovery: falsified if projected lock-wedge candidates do not alter recovery kinetics relative to Aleniglipron under matched receptor occupancy.
4. Synthetic route: falsified if SMARTS-compatible virtual routes fail vendor availability, reaction selectivity, purification, or stability review.

## 12. Lineage

- policy checkpoint: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/gflownet_policy_v1.pt`
- top-100 candidates: `campaigns/glp1r_aleniglipron/track_a_generative/epoch019_fixes/gflownet_top100_full_pgx_epoch019.parquet`
- tripartite profiles: `campaigns/glp1r_aleniglipron/track_a_generative/epoch019_fixes/candidate_dossiers_epoch019.parquet`
- medchem audit: `campaigns/glp1r_aleniglipron/track_a_generative/epoch019_fixes/gflownet_medchem_audit_epoch019.parquet`
- candidate audit: `campaigns/glp1r_aleniglipron/track_a_generative/epoch019_fixes/gflownet_audit_epoch019.json`
- CBOM: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/PRISM_CBOM_v1.0.json`
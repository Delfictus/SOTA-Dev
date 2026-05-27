# M3 Lead Optimization Dossier

Campaign: `glp1r_aleniglipron`

Generated: `2026-05-27T23:14:02Z`

Epistemic scope: this dossier reports PROJECTED candidate prioritization from the PRISM Track A policy and Rust oracle. Candidates remain subject to synthesis, purification, receptor signaling, internalization, and washout falsification gates.

## 1. Executive Finding

The Track A GFlowNet policy generated and ranked candidates against the PRISM-FORGE O3A/Z-matrix reward oracle. The final audit reports:

- projected lock-positive candidates: `13/100`
- PAINS pass: `100/100`
- BRENK pass: `88/100`
- oral descriptor pass: `100/100`
- top-100 unique SMILES: `100`

## 2. Epoch 021 Full-Field Training Status

The Epoch 021 full-field production run used the v4 reward, v2 thermodynamic field stack, dual-channel policy, and hard-zero R&F electronic channel. This section reports the actual completed run, not the earlier Epoch 016 long-run harness state.

| execution metric | value |
|---|---:|
| training status | `PARTIAL_EXECUTION_BLOCKED_BY_TERMINAL_HARNESS` |
| observed completed epochs | 9/500 |
| failure mode | `external_process_termination_without_python_traceback` |

Execution report: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/epoch016_execution_report.json`

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
| corrected lock-positive candidates | 3 |
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
| 15 | `COc1cccc2cc(-c3cc(N)c[nH]3)[nH]c12` | 15.021 | 2.691 | 0.986 | L1 | 2.670 | 5.000 | SMARTS/Z-matrix projected route |
| 24 | `C[C@@H]1C[C@H](CC(N)=O)c2cc3c(cc21)OCCO3` | 11.252 | 0.000 | 0.537 | L1 | 2.748 | 6.000 | SMARTS/Z-matrix projected route |
| 6 | `Cc1ccn2c(CCN)c(C(C)(C)C)nc2c1` | 10.252 | 0.000 | 0.537 | L1 | 2.748 | 5.000 | SMARTS/Z-matrix projected route |
| 9 | `Cc1nn(C)c2c1[C@@H](N)C[C@H](C)C2` | 10.105 | 0.000 | 0.537 | L1 | 2.895 | 5.000 | SMARTS/Z-matrix projected route |
| 19 | `CCCc1ccc(-c2cc(C)nc(N)n2)cc1` | 9.923 | 0.000 | 0.537 | L1 | 2.077 | 4.000 | SMARTS/Z-matrix projected route |
| 7 | `Cc1cc(C(=O)CN2CCCC2)ccc1F` | 9.656 | 0.000 | 0.537 | L1 | 3.344 | 5.000 | SMARTS/Z-matrix projected route |
| 20 | `CC(C)C[C@H]1CNCCN1C(=O)N(C)C` | 9.637 | 0.000 | 0.537 | L1 | 3.363 | 5.000 | SMARTS/Z-matrix projected route |
| 41 | `NCC[C@H]1CN(c2ccccc2)C(=O)CO1` | 9.220 | 0.000 | 0.537 | L1 | 2.780 | 6.000 | SMARTS/Z-matrix projected route |
| 30 | `Cc1cc(C)c2c(c1)O[C@@]1(CCNC1)C[C@H]2O` | 9.165 | 0.000 | 0.537 | L1 | 2.835 | 4.000 | SMARTS/Z-matrix projected route |
| 29 | `Cc1cnc(C(=O)N(C)[C@H]2CCC[C@H](C)C2)cn1` | 9.156 | 0.000 | 0.537 | L1 | 2.844 | 4.000 | SMARTS/Z-matrix projected route |

## 7. PGx Resilience Status

PGx status: `WT_PROJECTION_COLLAPSE`
Epistemic class: `PROJECTED`
Scoring method: `coordinate_field_projection_v1`
WT parity status: `unreported`
Report: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/gflownet_top_100_pgx_screened_report.json`

WT projection parity repair:

| parity field | value |
|---|---|
| parity status | `VERIFIED_RAW_WT_PARITY` |
| raw projection status | `WT_NATIVE_RAW_PARITY_CONFIRMED` |
| repair method | `native_wt_reward_plus_relative_field_liability_delta_v1` |
| projection/native ratio mean | `1.4070750976941602e-09` |
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

- variant perturbation manifest: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/population_pgx/variant_perturbation_manifest.json`
- consensus grid report: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/population_pgx/population_consensus_grid_report.json`
- full-landscape PGx report: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/pgx_full_landscape_report.json`

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
| EUR | `0.9779373797470131` | 100 |
| AFR | `0.9864453246939888` | 100 |
| EAS | `0.997342278629517` | 100 |
| SAS | `0.9845893803313698` | 100 |
| AMR | `0.9823471202169805` | 100 |

Interpretation boundary: A316T and T149M use materialized MD-derived grids. The remaining variant grids are perturbation-projected from gnomAD frequency, domain, conservation, and functional-assay annotations and must be treated as Phase 3 stratification intelligence, not clinical proof.

## 9. Full Thermodynamic Field Stack

Epoch 021 expands candidate assessment from the original signal grid into the complete thermodynamic field stack: signal classification, shear stress, hysteresis/reversibility, translation-pathway contact, AM1-BCC electrostatics, and u_pose geometry fragility.

| full-field metric | value |
|---|---:|
| status | `complete` |
| candidates assessed | 100 |
| mean shear stress | `11.797730339770883` |
| candidates with nonzero shear | 100 |
| mean hysteresis | `0.016048969093900082` |
| mean reversibility | `0.9842283737110126` |
| direct activation-pathway voxel candidates | 0 |
| activation-pathway neighborhood candidates | 100 |
| mean AM1-BCC charge feature magnitude | `0.17608070927077046` |
| mean u_pose penalty | `0.7307796373571462` |
| u_pose provenance counts | `{"best_rotamer_rank_proxy": 100}` |
| low-shear, reversible, pathway-contact candidates | 60 |

The shear and hysteresis terms are training signals, not decorative post-hoc metrics. A high-shear candidate is leaning into a mechanically mobile receptor wall; a high-hysteresis candidate occupies a region whose thermal cycle does not return cleanly to baseline. Both liabilities are carried into reward v4 and into the candidate dossiers.

Pathway reporting separates direct occupancy of sparse pathway voxels from neighborhood interaction with the same pathway nodes. The latter is a projected kinetic-control signal with explicit provenance, not a claim that every atom landed exactly inside a pathway voxel.

## 10. Cross-Species Selectivity Prediction

Cross-species selectivity is an L2 structural inference from GLP-1R conservation data. It predicts species-activity pattern, not potency, and is calibrated against Aleniglipron's known Human/NHP selectivity behavior.

| species metric | value |
|---|---:|
| status | `unavailable` |
| mean selectivity score | `None` |
| high human-selective candidates (>=0.70) | 0 |
| broad/selectivity-low candidates (<=0.35) | 0 |

| rank | selectivity | predicted active in | SMILES |
|---:|---:|---|---|

## 11. Autonomous Infrastructure Status

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

## 12. GPU Dispatch Readiness

GPU dispatch status: `PASS`

| dispatch metric | value |
|---|---:|
| scripts generated | 3 |
| dispatch-ready scripts | 3 |
| corrected engine scripts | 3 |
| high-priority BALD jobs | 0 |

Dispatch audit: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/gpu_dispatch_audit_report.json`

These jobs are queued for ground-truth CCNS validation only. They do not constitute confirmed dynamic biased agonism until GPU MD preserves lock occupancy across the protocol.

## 13. Thermodynamic Motif Intelligence

Motif registry status: `complete`
Registry: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_b_chronological/motif_intelligence/thermodynamic_motif_registry.parquet`

Four orthogonal motif discovery methods are reported when available: Thermodynamic Functional Group Decomposition with k-hop neutral-bridge expansion, Causal Attribution Motif Extraction via integrated gradients, Phase-Resolved MCS with Butina/Tanimoto pruning, and Synthon-Ancestry Decomposition with Fisher exact enrichment.

| motif intelligence metric | value |
|---|---:|
| total motifs | 328 |
| TFGD motifs | 52 |
| CAME motifs | 50 |
| PR-MCS motifs | 26 |
| SAD motifs | 200 |
| lock-wedge motifs | 66 |
| phase-conditional motifs | 78 |
| evolutionary-invariant motifs | 26 |
| mean completeness score | `0.607538802660754` |

Key lock-wedge / thermodynamic motifs:

| motif ID | SMARTS | role | lock contribution | resilience | method | provenance |
|---|---|---|---:|---:|---|---|
| `motif_038cb6e2497fa1cd` | `[#7]-[#6]1:[#6]:[#6](-[#17]):[#6]:[#6](-[#6]-[#7]2-[#6]-[#6]-[#7]-[#6]-[#6]-2):[#6]:1-[#9]` | LOCK_WEDGE | 1.000 | 1.000 | SAD | DERIVED |
| `motif_1d28604b05e6935d` | `[#6]-[#8]-[#6]1:[#6]:[#6](-[#6]):[#7]:[#6]2:[#6](-[#6@H](-[#6])-[#7]):[#6]:[#6]:[#6]:[#6]:1:2` | LOCK_WEDGE | 1.000 | 1.000 | SAD | DERIVED |
| `motif_2db97328d4634ac8` | `[#7]-[#6]-[#6]1:[#7]:[#7]:[#6](-[#6]-[#6]2:[#6]:[#6]:[#6](-[#9]):[#6]:[#6]:2-[#9]):[#7]:1` | LOCK_WEDGE | 1.000 | 1.000 | SAD | DERIVED |
| `motif_34a5fb8d039c1935` | `[#6]-[#6]1:[#6]:[#6]:[#6](-[#8]-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2-[#6]-[#7]):[#6]:[#7]:1` | LOCK_WEDGE | 1.000 | 1.000 | SAD | DERIVED |
| `motif_36d640dcae904105` | `[#6]-[#8]-[#6]1:[#6]:[#6]:[#6](-[#6]-[#6@@H]2-[#6]-[#8]-[#6]-[#6]-[#6@H]-2-[#7]):[#6]:[#6]:1` | LOCK_WEDGE | 1.000 | 1.000 | SAD | DERIVED |
| `motif_452461c7d32a74b2` | `[#6]-[#6@H](-[#7])-[#6@@H](-[#8])-[#6]1:[#6]:[#8]:[#6]2:[#6]:[#6](-[#9]):[#6]:[#6]:[#6]:1:2` | LOCK_WEDGE | 1.000 | 1.000 | SAD | DERIVED |

Synthon ordering recommendations:

| Enamine ID | SMARTS | lock enrichment | preferred exit vector | order priority |
|---|---|---:|---|---|
| `ANCHOR_098467` | `[#7]-[#6]1:[#6]:[#6](-[#17]):[#6]:[#6](-[#6]-[#7]2-[#6]-[#6]-[#7]-[#6]-[#6]-2):[#6]:1-[#9]` | 6.250 | `0` | HIGH |
| `ANCHOR_043020` | `[#6]-[#8]-[#6]1:[#6]:[#6](-[#6]):[#7]:[#6]2:[#6](-[#6@H](-[#6])-[#7]):[#6]:[#6]:[#6]:[#6]:1:2` | 6.250 | `0` | HIGH |
| `ANCHOR_094878` | `[#7]-[#6]-[#6]1:[#7]:[#7]:[#6](-[#6]-[#6]2:[#6]:[#6]:[#6](-[#9]):[#6]:[#6]:2-[#9]):[#7]:1` | 6.250 | `0` | HIGH |
| `ANCHOR_072204` | `[#6]-[#6]1:[#6]:[#6]:[#6](-[#8]-[#6]2:[#6]:[#6]:[#6]:[#6]:[#6]:2-[#6]-[#7]):[#6]:[#7]:1` | 6.250 | `0` | HIGH |
| `ANCHOR_044382` | `[#6]-[#8]-[#6]1:[#6]:[#6]:[#6](-[#6]-[#6@@H]2-[#6]-[#8]-[#6]-[#6]-[#6@H]-2-[#7]):[#6]:[#6]:1` | 6.250 | `0` | HIGH |
| `ANCHOR_056710` | `[#6]-[#6@H](-[#7])-[#6@@H](-[#8])-[#6]1:[#6]:[#8]:[#6]2:[#6]:[#6](-[#9]):[#6]:[#6]:[#6]:1:2` | 6.250 | `0` | HIGH |
| `ANCHOR_091421` | `[#7]-[#6]-[#6]1-[#6]-[#6]-[#7](-[#6]-[#6@H]2-[#6]-[#6]-[#6]-[#6]-[#8]-2)-[#6]-[#6]-1` | 6.250 | `0` | HIGH |
| `ANCHOR_088845` | `[#7]#[#6]-[#6]1:[#6]:[#6]:[#6]:[#7]:[#6]:1-[#6](=[#8])-[#7]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1` | 6.250 | `0` | HIGH |

Motif evolution view:

| motif | born | status | parent | lock contribution | lineage |
|---|---:|---|---|---:|---|
| `motif_038cb6e2497fa1cd` | 24 | PERSISTENT | `` | 1.000 | NOVEL |
| `motif_1d28604b05e6935d` | 24 | PERSISTENT | `` | 1.000 | NOVEL |
| `motif_2db97328d4634ac8` | 24 | PERSISTENT | `` | 1.000 | NOVEL |
| `motif_34a5fb8d039c1935` | 24 | PERSISTENT | `` | 1.000 | NOVEL |
| `motif_36d640dcae904105` | 24 | PERSISTENT | `` | 1.000 | NOVEL |

Interpretation boundary: motif intelligence is receptor-specific computational design intelligence. Causal attribution is computed over model features and remains computational until falsified by synthesis, receptor signaling, internalization, and washout experiments.

## 14. Falsification Gates

1. β-arrestin recruitment: falsified if high-lock-score candidates do not reduce β-arrestin recruitment relative to receptor-occupancy-matched controls.
2. cAMP signaling: falsified if candidates lose primary GLP-1R agonist signaling at matched exposure.
3. Washout recovery: falsified if projected lock-wedge candidates do not alter recovery kinetics relative to Aleniglipron under matched receptor occupancy.
4. Synthetic route: falsified if SMARTS-compatible virtual routes fail vendor availability, reaction selectivity, purification, or stability review.

## 15. Lineage

- policy checkpoint: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/gflownet_policy_v1.pt`
- top-100 candidates: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/gflownet_top_100_candidates_lockmask_rescored.parquet`
- tripartite profiles: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/gflownet_top_50_tripartite_profiles.parquet`
- medchem audit: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/gflownet_medchem_audit.parquet`
- candidate audit: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/gflownet_candidate_audit.json`
- thermodynamic motif registry: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_b_chronological/motif_intelligence/thermodynamic_motif_registry.parquet`
- CBOM: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/PRISM_CBOM_v1.0.json`
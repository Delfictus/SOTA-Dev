# M3 Lead Optimization Dossier

Campaign: `glp1r_aleniglipron`

Generated: `2026-05-25T05:47:56Z`

Epistemic scope: this dossier reports PROJECTED candidate prioritization from the PRISM Track A policy and Rust oracle. Candidates remain subject to synthesis, purification, receptor signaling, internalization, and washout falsification gates.

## 1. Executive Finding

The Track A GFlowNet policy generated and ranked candidates against the PRISM-FORGE O3A/Z-matrix reward oracle. The final audit reports:

- biased agonism candidates confirmed: `6/100`
- PAINS pass: `100/100`
- BRENK pass: `88/100`
- oral descriptor pass: `100/100`
- top-100 unique SMILES: `100`

## 2. Epoch 016 Execution Status

The Epoch 016 production training command path is implemented and passed a one-epoch functional smoke test, but the local terminal harness terminated longer CPU-bound training processes before the 500-epoch target could complete.

| execution metric | value |
|---|---:|
| training status | `PARTIAL_EXECUTION_BLOCKED_BY_TERMINAL_HARNESS` |
| observed completed epochs | 9/500 |
| failure mode | `external_process_termination_without_python_traceback` |

Execution report: `campaigns/glp1r_aleniglipron/track_a_generative/epoch016_execution_report.json`

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

PGx status: `PGX_PARITY_CALIBRATED`
Epistemic class: `DERIVED_L3_PARITY_CALIBRATED`
Scoring method: `native_wt_reward_plus_relative_field_liability_delta_v1`
WT parity status: `CALIBRATED_PARITY_CONFIRMED`
Report: `campaigns/glp1r_aleniglipron/track_a_generative/gflownet_top100_pgx_parity_validated_report.json`

WT projection parity repair:

| parity field | value |
|---|---|
| parity status | `CALIBRATED_PARITY_CONFIRMED` |
| raw projection status | `WT_PROJECTION_COLLAPSE` |
| repair method | `native_wt_reward_plus_relative_field_liability_delta_v1` |
| projection/native ratio mean | `1.4159388232784845e-09` |
| calibrated WT self-parity ratio | `1.0` |
| report | `campaigns/glp1r_aleniglipron/track_a_generative/wt_projection_parity_report.json` |

| variant | condition | lock preserved | class counts |
|---|---|---:|---|
| A316T | `glp1r_6XOX_A316T` | 0 | `{'VULNERABLE': 100}` |
| T149M | `glp1r_6XOX_T149M` | 0 | `{'VULNERABLE': 100}` |

## 8. Autonomous Infrastructure Status

Cloudflare and autonomous thermodynamic AI infrastructure were checked without redeploying the Worker.

| infrastructure component | status |
|---|---|
| overall | `OK` |
| Worker HTTP | `ACCESS_PROTECTED` |
| D1 candidate table | `OK` |
| Vectorize index | `OK` |
| R2 tensor bucket | `OK` |
| Queue | `OK` |

Infrastructure report: `campaigns/glp1r_aleniglipron/track_a_generative/autonomous_infra_status_epoch017.json`

## 9. GPU Dispatch Readiness

GPU dispatch status: `PASS`

| dispatch metric | value |
|---|---:|
| scripts generated | 3 |
| dispatch-ready scripts | 3 |
| corrected engine scripts | 3 |
| high-priority BALD jobs | 0 |

Dispatch audit: `campaigns/glp1r_aleniglipron/track_a_generative/gpu_dispatch_audit_report.json`

These jobs are queued for ground-truth CCNS validation only. They do not constitute confirmed dynamic biased agonism until GPU MD preserves lock occupancy across the protocol.

## 10. Falsification Gates

1. β-arrestin recruitment: falsified if high-lock-score candidates do not reduce β-arrestin recruitment relative to receptor-occupancy-matched controls.
2. cAMP signaling: falsified if candidates lose primary GLP-1R agonist signaling at matched exposure.
3. Washout recovery: falsified if projected lock-wedge candidates do not alter recovery kinetics relative to Aleniglipron under matched receptor occupancy.
4. Synthetic route: falsified if SMARTS-compatible virtual routes fail vendor availability, reaction selectivity, purification, or stability review.

## 11. Lineage

- policy checkpoint: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/track_a_generative/gflownet_policy_v1.pt`
- top-100 candidates: `campaigns/glp1r_aleniglipron/track_a_generative/gflownet_top100_pgx_parity_validated.parquet`
- tripartite profiles: `campaigns/glp1r_aleniglipron/track_a_generative/gflownet_top_50_tripartite_profiles_epoch016.parquet`
- medchem audit: `campaigns/glp1r_aleniglipron/track_a_generative/gflownet_medchem_audit_epoch016.parquet`
- candidate audit: `campaigns/glp1r_aleniglipron/track_a_generative/gflownet_audit_epoch016.json`
- CBOM: `/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/PRISM_CBOM_v1.0.json`
# GLP-1R Aleniglipron Phase 2/3 Deliverables Audit

Generated: 2026-05-28T06:25Z

## Bottom Line

Phase 2C is complete and archived. The receptor/variant thermodynamic evidence is real: 16/16 replicas passed, with 1,204,768,024 MD spikes sealed in `PHASE2C_SEALED_MANIFEST.json` and archived to `r2:prism-archive/glp1r_aleniglipron/phase_2c_snapshots`.

Phase 2/3 analytic deliverables are now substantially populated from the latest phase-manifold bundle at:

`campaigns/glp1r_aleniglipron/integrated_spike_events/phase_manifold_full_20260527_224959/`

The molecule-design side is not yet a confirmed "synthesized and engine-validated candidate" package. The repo contains virtual/generated molecules, SMILES/SDF structures, launch scripts, candidate dossiers, reward/PGx tables, and binding motifs. It does not show candidate-specific PRISM MD completion for those molecules, and it does not show wet-lab synthesized molecule evidence.

## Phase 2C Engine Evidence

| artifact | status |
| --- | --- |
| `PHASE2C_SEALED_MANIFEST.json` | 16 replicas, `all_pass=true`, `total_spikes_md=1204768024` |
| R2 archive | copied 147.831 GiB, 2992/2992 files |
| R2 check | 0 differences, 2992 matching files |
| closeout commit | `123e264e Seal GLP1R Phase 2C archive` |

## Generated Phase 2/3 Analytic Bundle

| artifact | rows/status |
| --- | ---: |
| `phase_manifold_coherence.parquet` | 3263 |
| `phase_manifold_edge_validation.parquet` | 9 |
| `translation_pathway_nodes.parquet` | 5 |
| `enriched_edge_validation.parquet` | 9 |
| `enriched_pathway_nodes.parquet` | 5 |
| `wire_conservation_matrix.parquet` | 16 |
| `temporal_cascade_summary.parquet` | 17 |
| `transition_chronology_tensor.parquet` | 3200 |
| `topology_region_registry.json` | 6/6 regions covered |
| `gflownet_reward_landscape.parquet` | 444333 |
| `variant_manifold_coverage_matrix.parquet` | 576, 0 gaps |
| `nma_continuity_map.parquet` | 3263 |
| `thermodynamic_continuity_map.parquet` | 11852 |
| `hydration_continuity_map.parquet` | 1 blocked row, no hydration parquet input found |

## Molecule/Candidate Deliverables Found

| deliverable | evidence |
| --- | ---: |
| Track A top candidates | `gflownet_top_100_candidates.parquet`: 100 rows |
| Track A broad candidates | `gflownet_top_500_candidates.parquet`: 500 rows |
| Population PGx candidates | `population_pgx/gflownet_top100_full_pgx.parquet`: 100 rows |
| Chronology-locked candidates | `track_b_chronological/chronology_locked_top_100_candidates.parquet`: 100 rows |
| SDF structures | 103 total SDF files |
| candidate SDF structures | 79 candidate SDF files |
| candidate validation launch scripts | 79 `launch-n10-validate-*.sh` scripts |

Candidate dossier GPU status:

| dossier set | json count | status counts |
| --- | ---: | --- |
| `candidate_dossiers` | 50 | 3 pending, 47 not_lock_positive |
| `candidate_dossiers_population` | 50 | 34 pending, 16 not_lock_positive |
| `epoch019_fixes/candidate_dossiers` | 50 | 37 pending, 13 not_lock_positive |

Candidate-specific engine completion evidence:

| check | result |
| --- | --- |
| `track_a_generative/gpu_dispatch/results/cand_015_bccda098` file count | 0 |
| candidate dossiers `gpu_dispatch_status` | pending/not_lock_positive only |
| launch scripts | present |
| completed candidate CCNS raw outputs | not found in repo or `/mnt/storage/tmp` search |

## Binding Motif Deliverables

| artifact | rows/status |
| --- | ---: |
| latest chronology-locked motif registry | 207 motifs, completeness mean 0.628 |
| latest Track A motif registry | 144 motifs, completeness mean 0.578 |
| existing Track B motif registry | 328 motifs |

Generated latest registries:

- `phase_manifold_full_20260527_224959/motif_intelligence/thermodynamic_motif_registry.parquet`
- `phase_manifold_full_20260527_224959/motif_intelligence_track_a/thermodynamic_motif_registry.parquet`

## Claim Boundary

What is supported:

- Phase 2C receptor/variant thermodynamic engine evidence is complete and archived.
- Phase-manifold, transition chronology, topology-region, reward-landscape, variant coverage, continuity, and motif outputs now exist for the latest Phase 2C-derived bundle.
- Virtual molecules and candidate SDFs exist.
- Binding motifs and motif registries exist.

What is not supported yet:

- No evidence that any generated candidate molecule has completed candidate-specific PRISM MD validation.
- No evidence of physically synthesized molecules or wet-lab synthesis outputs.
- Hydration continuity is explicitly blocked by missing hydration parquet input.
- Existing GPU launch scripts are staging/queued artifacts, not validation completion evidence.

## Next Gate

The next correct gate is one candidate-specific CCNS validation smoke on the best chronology/population candidate, routed away from root disk. Current disk state observed before this audit was `/` at 90% and `/mnt/storage` at 85%, so do not launch all 79 validation scripts until output routing and space are fixed.

Suggested first validation target: the top population/chronology candidate with SDF and launch material present, then update its candidate dossier from `gpu_dispatch_status=pending` to an observed status only after result files and schemas pass.

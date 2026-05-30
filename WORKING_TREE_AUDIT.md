# Working Tree Audit

- generated_at_utc: `2026-05-29T02:30:36.046997+00:00`
- repo_root: `/home/diddy/Desktop/Prism4D-bio`
- head: `b40d650b90b00825f29007556d4f92c415ebc935`
- total_entries: `4571`

This audit is read-only. No files were deleted or rewritten by the audit.

## By State

| bucket | count |
|---|---:|
| `ignored` | 4501 |
| `untracked` | 50 |
| `modified` | 20 |

## By Category

| bucket | count |
|---|---:|
| `campaign_evidence_or_output` | 3864 |
| `python_cache` | 330 |
| `ad_hoc_structure_artifact` | 115 |
| `unknown` | 104 |
| `local_run_output_tree` | 51 |
| `release_or_analysis_artifact` | 27 |
| `source_or_runtime_code` | 14 |
| `agent_review_or_local_assistant_state` | 10 |
| `run_logs_and_nohup` | 10 |
| `wrangler_local_state` | 9 |
| `archive_recovery_tree` | 8 |
| `backup_or_compat_copy` | 8 |
| `build_output` | 8 |
| `release_tooling` | 8 |
| `local_env_artifact` | 2 |
| `untracked_source_surface` | 2 |
| `secret_sensitive_local_service_state` | 1 |

## By Preserve Class

| bucket | count |
|---|---:|
| `must_preserve` | 3894 |
| `rebuildable_cache` | 338 |
| `archive_only` | 234 |
| `manual_review` | 104 |
| `secret_sensitive_archive_only` | 1 |

## Classification Notes

- `must_preserve`: keep in canonical local system and release evidence surface.
- `archive_only`: keep for workstation archive or evidence, but not necessarily GitHub.
- `rebuildable_cache`: generated cache/build output; still accounted for, but not source-of-truth.
- `secret_sensitive_archive_only`: preserve carefully; do not expose in GitHub payloads.
- `manual_review`: not confidently classified; review before any exclusion.

## Notable Samples

### ad_hoc_structure_artifact
- `1ade.pdb`
- `1btl.pdb`
- `1hhp.pdb`
- `1jwp.cif`
- `1pzo.cif`
- `1w50.pdb`
- `2gl7.cif`
- `3fke.pdb`
- `3k5v.pdb`
- `3l15_apo.pdb`
- `3l15_chainA.pdb`
- `3l15_raw.pdb`

### agent_review_or_local_assistant_state
- `.codex/review/context.txt`
- `.codex/review/pre-commit.diff`
- `.codex/review/pre-commit.diffstat`
- `.claude/commands/uspto-agent.md`
- `.claude/settings.local.json`
- `.codex/.gitignore`
- `.codex/APPROVED_FOR_PUSH`
- `.codex/CURRENT_PHASE.txt`
- `.codex/GLOBAL_CONTEXT.txt`
- `.codex/reports/`

### archive_recovery_tree
- `.archive/crates/prism-io/output.pdb`
- `.archive/crates/prism-niv-bench/phase2_audit.sqlite`
- `.archive/crates/prism-niv-bench/target/`
- `.archive/crates/prism-report/src/bin/prism4d.rs.bak`
- `.archive/crates/prism-report/src/finalize.rs.bak`
- `.archive/crates/prism-report/tests/test_mini.pdb`
- `.archive/crates/prism-ve-bench/src/vasil_exact_metric.rs.bak`
- `.archive/target/`

### backup_or_compat_copy
- `crates/prism-nhs/src/bin/nhs_rt_full.rs.pre_mdonly_fix`
- `scripts/prism_dynamic_aligned_voxel_export.py.pre_forces_fallback`
- `STATUS.md.bak`
- `crates/prism-gpu/src/kernels/rt_clustering.cu.v1.bak`
- `crates/prism-gpu/src/kernels/rt_clustering_cuda.cu.v1.bak`
- `crates/prism-nhs/src/bin/nhs_rt_full.rs.bak`
- `crates/prism-nhs/src/rt_clustering.rs.v1.bak`
- `prism4d_manuscript/PRISM4D_v4.tex.bak_pre_fig10`

### build_output
- `apps/glp1r-teaser-visualizer/dist/`
- `apps/glp1r-teaser-visualizer/node_modules/`
- `apps/glp1r-teaser-visualizer/tsconfig.node.tsbuildinfo`
- `apps/glp1r-teaser-visualizer/tsconfig.tsbuildinfo`
- `build_pharma_campaign.py`
- `cloud/prism-manifold-worker/node_modules/`
- `release/dist/`
- `target`

### campaign_evidence_or_output
- `campaigns/glp1r_aleniglipron/M3_Lead_Optimization_Dossier.md`
- `campaigns/glp1r_aleniglipron/PHASE2_PHASE3_DELIVERABLES_AUDIT_20260528.md`
- `campaigns/glp1r_aleniglipron/candidate_dossiers/cand_015_bccda098.json`
- `campaigns/glp1r_aleniglipron/candidate_dossiers/cand_015_bccda098.md`
- `campaigns/glp1r_aleniglipron/track_a_generative/candidate_dossiers/cand_015_bccda098.json`
- `campaigns/glp1r_aleniglipron/track_a_generative/candidate_dossiers/cand_015_bccda098.md`
- `campaigns/glp1r_aleniglipron/track_a_generative/m3_dossier/M3_Lead_Optimization_Dossier.md`
- `campaigns/glp1r_aleniglipron/PHASE2_PHASE3_FULL_OUTPUT_COMPLETENESS_QUALITY_20260528.generate.log`
- `campaigns/glp1r_aleniglipron/PHASE2_PHASE3_FULL_OUTPUT_COMPLETENESS_QUALITY_20260528.json`
- `campaigns/glp1r_aleniglipron/PHASE2_PHASE3_FULL_OUTPUT_COMPLETENESS_QUALITY_20260528.md`
- `campaigns/glp1r_aleniglipron/dstw_phase_b/`
- `campaigns/glp1r_aleniglipron/integrated_spike_events/phase_manifold_smoke_retry_20260527_224322/`

### local_env_artifact
- `.figvenv/`
- `.venv/`

### local_run_output_tree
- `.prism_orchestration/`
- `.rebuild-reference/`
- `.scratch`
- `.stale_row_purge_out/`
- `1btl_out/`
- `1ere_fused8_out/`
- `1ere_monomer_out/`
- `1hhp_dimer_out/`
- `1hhp_mono_out/`
- `1hhp_out/`
- `docs/blind_validation/post_freeze_validation/B01_HRAS_Q61H/fpocket_out/`
- `docs/blind_validation/post_freeze_validation/B01_HRAS_Q61H/p2rank_out/`

### python_cache
- `.mypy_cache`
- `.pytest_cache`
- `__pycache__/`
- `cloudflare/d1/__pycache__/`
- `prism-ai-inference/__pycache__/`
- `runpod_training/__pycache__/`
- `scripts/__pycache__/__init__.cpython-312.pyc`
- `scripts/__pycache__/_sealed_release_common.cpython-312.pyc`
- `scripts/__pycache__/admet_predict.cpython-312.pyc`
- `scripts/__pycache__/aleniglipron_two_layer_interference.cpython-312.pyc`
- `scripts/__pycache__/analyze_ensemble_pockets.cpython-312.pyc`
- `scripts/__pycache__/analyze_with_alignment.cpython-312.pyc`

### release_or_analysis_artifact
- `CANONICAL_COPY_SEALING_ONTOLOGY.md`
- `RELEASE_TRACKS.md`
- `VOLATILE_FILES_RECONCILIATION.md`
- `WORKING_TREE_AUDIT.md`
- `release_tracks.json`
- `working_tree_audit.json`
- `.stale_widened_snapshot/ground_truth.parquet`
- `.stale_widened_snapshot/site_event_aggregates.parquet`
- `.stale_widened_snapshot/site_features.parquet`
- `.stale_widened_snapshot/site_kcc_candidates.parquet`
- `.stale_widened_snapshot/site_lining_residues.parquet`
- `.stale_widened_snapshot/site_tags.parquet`

### release_tooling
- `Makefile`
- `release_policy.yaml`
- `scripts/_sealed_release_common.py`
- `scripts/evaluate_release_policy.py`
- `scripts/prism_filetag.py`
- `scripts/seal_operational_release.py`
- `scripts/verify_restored_release.py`
- `tests/release_acceptance/`

### run_logs_and_nohup
- `phase2c_mdonly_gate_fixed.out`
- `phase2c_nohup.out`
- `phase2c_reintegration_20260527_164627.log`
- `phase2c_reintegration_20260527_164936.log`
- `phase2c_reintegration_20260527_165327.log`
- `phase2c_reintegration_20260527_165905.log`
- `phase2c_reintegration_20260527_170355.log`
- `phase2c_reintegration_20260527_170742.log`
- `phase2c_relaunch_after_commit.out`
- `phase2c_relaunch_final.out`

### secret_sensitive_local_service_state
- `.wrangler/cache/wrangler-account.json`

### source_or_runtime_code
- `crates/prism-nhs/src/bin/nhs_rt_full.rs`
- `scripts/prism_hydration_event_statistics.py`
- `src/prism_dstw/orchestration/topology_compiler.py`
- `scripts/audit_working_tree.py`
- `scripts/build_candidate_motif_summary.py`
- `scripts/build_dstw_export_compat_bundle.py`
- `scripts/build_prism_e2e_release_package.py`
- `scripts/compile_candidate_holo_topology.py`
- `scripts/generate_phase23_completeness_quality_output.py`
- `scripts/materialize_glp1r_variant_topologies.py`
- `scripts/run_candidate_pathb_validation.py`
- `crates/prism-gpu/target/`

### unknown
- `0.6`
- `1ere_monomer_out2/`
- `1ere_monomer_out3/`
- `PRISM_GLP1R_M2_DELIVERABLES_v1_1/05_GROUND_TRUTH_DATA/`
- `PRISM_GLP1R_M2_DELIVERABLES_v1_1/06_VISUALIZATION_PACKAGE/`
- `PRISM_GLP1R_M2_DELIVERABLES_v1_1/08_RELEASE_ARCHIVES/`
- `ams_trajectory_extractor.py`
- `apply_ai_ranker.py`
- `brute_force_live_parser.py`
- `causal_preflight.py`
- `check_1jwp_hit.py`
- `check_frame_offset.py`

### untracked_source_surface
- `tools/`
- `workers/`

### wrangler_local_state
- `.wrangler/state/v3/cache/miniflare-CacheObject/metadata.sqlite-shm`
- `.wrangler/state/v3/cache/miniflare-CacheObject/metadata.sqlite-wal`
- `.wrangler/state/v3/r2/miniflare-R2BucketObject/29997cabf99edb1bb4ff8e6b0d47691a84f51ee871df18cd96904969f66338ad.sqlite-shm`
- `.wrangler/state/v3/r2/miniflare-R2BucketObject/29997cabf99edb1bb4ff8e6b0d47691a84f51ee871df18cd96904969f66338ad.sqlite-wal`
- `.wrangler/state/v3/r2/miniflare-R2BucketObject/metadata.sqlite-shm`
- `.wrangler/state/v3/r2/miniflare-R2BucketObject/metadata.sqlite-wal`
- `.wrangler/state/v3/cache/miniflare-CacheObject/metadata.sqlite`
- `.wrangler/state/v3/r2/miniflare-R2BucketObject/29997cabf99edb1bb4ff8e6b0d47691a84f51ee871df18cd96904969f66338ad.sqlite`
- `.wrangler/state/v3/r2/miniflare-R2BucketObject/metadata.sqlite`

## Unknown Paths

- `0.6`
- `1ere_monomer_out2/`
- `1ere_monomer_out3/`
- `PRISM_GLP1R_M2_DELIVERABLES_v1_1/05_GROUND_TRUTH_DATA/`
- `PRISM_GLP1R_M2_DELIVERABLES_v1_1/06_VISUALIZATION_PACKAGE/`
- `PRISM_GLP1R_M2_DELIVERABLES_v1_1/08_RELEASE_ARCHIVES/`
- `ams_trajectory_extractor.py`
- `apply_ai_ranker.py`
- `brute_force_live_parser.py`
- `causal_preflight.py`
- `check_1jwp_hit.py`
- `check_frame_offset.py`
- `clean_patch.py`
- `cloud/prism-manifold-worker/.dev.vars`
- `cloud/prism-manifold-worker/.wrangler/`
- `cloudflare/workers/feature-pipeline/.build-dryrun/`
- `cloudflare/workers/feature-pipeline/.wrangler/`
- `cloudflare/workers/feature-pipeline/cloudflare/`
- `data/gt_holo/`
- `data/ml_prototype/`
- `data/targets/tier3_b2/`
- `data/validation/`
- `debug_1jwp.pml`
- `demo/samples/`
- `docking_results/`
- `docking_results_fast/`
- `docking_results_full/`
- `docs/blind_validation/post_freeze_validation/B01_HRAS_Q61H/.pdb_cache/`
- `docs/blind_validation/post_freeze_validation/B02_CDK2_allosteric/.pdb_cache/`
- `docs/blind_validation/post_freeze_validation/B03_Kv1.2/.pdb_cache/`
- `docs/blind_validation/post_freeze_validation/B04_MDM2/.pdb_cache/`
- `docs/blind_validation/post_freeze_validation/B05_TP53_R175H/.pdb_cache/`
- `docs/blind_validation/post_freeze_validation/B06_cGAS/.pdb_cache/`
- `docs/blind_validation/post_freeze_validation/B07_TEAD1/.pdb_cache/`
- `docs/blind_validation/post_freeze_validation/B08_CRBN/.pdb_cache/`
- `docs/blind_validation/post_freeze_validation/B09_Thrombin_exosite/.pdb_cache/`
- `docs/blind_validation/post_freeze_validation/B10_ADRB2/.pdb_cache/`
- `docs/blind_validation/reports/`
- `docs/sops/credentials/`
- `extract_violation_telemetry.py`
- `files.zip`
- `forensic_no_baseline_dependency_audit_20260512T232708Z/`
- `freeze_kv31_validation_artifact.sh`
- `generate_kv31_publication_figures.sh`
- `"h history for what Claude Code ran"`
- `harvest_benchmark.py`
- `harvest_competitors.py`
- `input_data/cxcr4/8i9v.pdb1`
- `json`
- `kcc_threshold`
- `logs/`
- `models/`
- `nhs_amber_fused.cubin`
- `output/`
- `p2rank_2.5.1/`
- `patch_gnina.py`
- `patch_gpu_dock.py`
- `pdb_refs/`
- `pilot_train.py`
- `prepped_20260510_0108`
- `prism-ai-inference/examples/`
- `prism-ai-inference/models/`
- `prism-ai-inference/models_esm_only/`
- `prism-ai-inference/models_v003/`
- `prism-ai-inference/models_v003_test/`
- `prism-ai-inference/models_v004/spike_bert_v002.pt`
- `prism4d.py`
- `prism4d_manuscript/figures_pfr/`
- `prism4d_manuscript/pfr_assets/`
- `prism4d_manuscript/sn-template.zip`
- `prism4d_manuscript/springer_template.zip`
- `prism4d_pocket_ranker.pkl`
- `prism_analysis_code_index.txt`
- `prism_kv31_family_validation.py`
- `prism_kv31_leakage_forensics.bad2.py`
- `prism_kv31_leakage_forensics.broken.py`
- `prism_kv31_leakage_forensics.py`
- `prism_kv31_score_baselines.py`
- `prism_publication_rigor_audit.py`
- `prism_publication_rigor_audit_20260513T070000Z/`
- `prism_publication_rigor_audit_20260513T070747Z/`
- `prism_relational_audit.py`
- `prism_script_inventory.txt`
- `prism_strict_null_controls.py`
- `prism_strict_null_controls_20260513T072740Z/`
- `rcsb_refs/`
- `re`
- `references/`
- `release_artifacts/v0.25.0/logs/`
- `release_audit/`
- `release_build/`
- `reports/`
- `rmsd[1]`
- `run_all_benchmarks.sh`
- `run_gnina_pipeline.sh`
- `run_unidock_pipeline.sh`
- `runpod/`
- `scipy.sparse.csr_matrix`
- `sys`
- `temporal_corr_threshold.`

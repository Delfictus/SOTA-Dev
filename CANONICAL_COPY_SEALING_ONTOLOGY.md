# Canonical Copy Sealing Ontology

```yaml
DIRECTIVE_ONTOLOGY:
  directive_id: canonical-copy-sealing-20260528
  objective: build a capacity-safe sealed PRISM release from the preserved canonical platform copy without deleting or mutating source artifacts
  scope:
    - track_a_canonical_platform_release
    - track_b_workstation_archive_separation
    - truthful_claim_boundary_enforcement
    - release_tooling_hardening
  required_components:
    - /home/diddy/PRISM-ISOLATED-20260528
    - scripts/seal_operational_release.py
    - scripts/verify_restored_release.py
    - scripts/evaluate_release_policy.py
    - release_policy.yaml
    - tests/release_acceptance/
    - RELEASE_TRACKS.md
    - release_tracks.json
    - VOLATILE_FILES_RECONCILIATION.md
  prohibited_actions:
    - delete_anything
    - mutate_scientific_outputs
    - seal_from_live_home_tree
    - overclaim_full_phase_1_to_3_completion
    - overclaim_full_hydration_production_completion
  expected_outputs:
    - capacity_safe_release_builder_flags
    - capacity_plan_json
    - separated_track_docs
    - stricter_policy_and_acceptance_gates
    - canonical_root_dry_run_evidence
  architectural_implications:
    - builder_must_accept_canonical_input_root
    - workstation_archive_volatility_must_not_block_track_a
    - destination_and_restore_gates_must_be_machine_enforced
  execution_risk:
    - root_filesystem_is_86_percent_full
    - external_ssd_cannot_hold_naive_double_materialization
    - live_codex_sqlite_wal_state_is_volatile
```

```yaml
IMPLEMENTATION_IMPACT_GRAPH:
  modified_files:
    - scripts/_sealed_release_common.py
    - scripts/seal_operational_release.py
    - scripts/evaluate_release_policy.py
    - tests/release_acceptance/test_release_acceptance.py
    - RELEASE_TRACKS.md
    - release_tracks.json
    - VOLATILE_FILES_RECONCILIATION.md
  dependent_files:
    - scripts/verify_restored_release.py
    - release_policy.yaml
    - WORKING_TREE_AUDIT.md
    - working_tree_audit.json
  runtime_dependencies:
    - git
    - rsync
    - rclone
    - zstd
    - tar
    - conda
    - cargo
  data_dependencies:
    - canonical_repo_copy
    - canonical_miniconda_copy
    - canonical_candidate_smoke_copy
    - track_a_reference_datasets
  mutation_risks:
    - builder_dry_run_and_non_dry_run_divergence
    - repo_only_archive_path_still_needing_full_canonical_payload_rework
    - claim_boundary_machine_gates_remaining_stricter_than_existing_outputs
  possible_regressions:
    - release_builder_cli_breakage
    - acceptance_test_failures
    - policy_evaluation_failures_until_destination_readback_is_real
```

```yaml
SOURCE_OF_TRUTH_RECONCILIATION:
  existing_state:
    - canonical_local_copy_exists_and_is_verified
    - release_builder_is_live-tree-centric
    - workstation_archive_is_running_and_volatile
  proposed_state:
    - canonical_copy_is_explicit_track_a_input
    - track_b_volatility_is_explicitly_non_blocking_for_track_a
    - capacity_plans_A_to_E_drive_release_routing
  superseded_logic:
    - ssd_free_must_be_2x_archive
    - live_repo_as_implicit_source_of_truth
  contradictory_logic:
    - builder_full_pass_semantics_do_not_match_policy
    - evaluator_accepts_partial_readback_even_when_policy_requires_readback
  unresolved_conflicts:
    - full non-dry canonical-payload streaming still needs final execution path validation
```

```yaml
INVARIANTS:
  - no_deletion
  - no_scientific_output_mutation
  - no_sealing_from_live_home_tree
  - no_silent_omission
  - no_plaintext_credentials_in_github
  - no_full_phase_1_to_3_claim_without_candidate_matrix_evidence
  - no_hydration_production_complete_claim_without_validated_inputs_and_outputs
```

# Ensemble Manifest JSON Schema — PRISM-4D v004

**Purpose**: Single canonical metadata file per target, tying together all N replicas of an ensemble run. Required input for the ensemble assembler and v004 student training pipeline.

**File location**: `<target>_ensemble_manifest.json` in the target's output directory. Mirrored to `r2:prism-spikes-20260512/scan_campaigns/<campaign_id>/targets/<pdb_id>/ensemble_manifest.json`.

**Versioning**: `schema_version` field is mandatory. Loader dispatches by version. Bumping major version requires a migration path.

---

## Full schema (v1.0.0)

```json
{
  "schema_version": "1.0.0",

  "campaign": {
    "campaign_id": "ensemble_20260516",
    "started_at": "2026-05-16T12:00:00Z",
    "completed_at": "2026-05-16T13:24:18Z",
    "pod_id": "runpod_b200_pod_abc123",
    "operator": "diddy@delfictus"
  },

  "engine": {
    "binary_sha256": "a1b2c3d4...64chars...",
    "binary_built_at": "2026-05-15T16:50:29Z",
    "binary_version": "v2_ignition",
    "build_features": ["v2_ignition", "gpu", "cuda_12_8", "sm_100"],
    "cuda_arch": "sm_100",
    "git_commit": "abc1234567890abcdef",
    "determinism_budget_doc_sha256": "e5f6g7..."
  },

  "target": {
    "pdb_id": "1zvd",
    "target_chain": "A",
    "input_pdb_path_relative": "prep/1zvd_clean.pdb",
    "input_pdb_sha256": "...",
    "topology_json_path_relative": "prep/1zvd_clean.topology.json",
    "topology_json_sha256": "...",
    "n_residues": 357,
    "n_atoms": 5642,
    "ground_truth_present": true,
    "ground_truth_uniprot_id": "Q9NS61",
    "ground_truth_pdb_holo": "1zvd",
    "ground_truth_ligand_centroid": [23.910, 14.406, 26.205],
    "ground_truth_ligand_resname": "MSE",
    "ground_truth_valid_for_dcc": true
  },

  "ensemble_config": {
    "n_replicas_planned": 5,
    "n_replicas_completed": 5,
    "n_replicas_failed": 0,
    "all_replicas_passed_audit": true,
    "replica_seed_strategy": "deterministic_offsets",
    "base_seed": 42,
    "seed_offsets": [0, 100, 200, 300, 400],
    "concurrent_replicas_per_pod": 1,
    "engine_flags_per_replica": [
      "--fast-25k", "--hysteresis", "--prism-therm",
      "--multi-stream", "20",
      "--spike-percentile", "70",
      "--fused-steps", "6",
      "--hmr", "--adaptive-dt",
      "--site-ranker",
      "phase-manifold"
    ]
  },

  "replicas": [
    {
      "replica_id": 0,
      "run_seed": 42,
      "status": "ok",
      "started_at": "2026-05-16T12:00:00Z",
      "completed_at": "2026-05-16T12:14:32Z",
      "duration_seconds": 872,
      "pod_id": "runpod_b200_pod_abc123",
      "n_streams_planned": 20,
      "n_streams_completed": 20,
      "n_streams_failed": 0,
      "stream_failures": [],
      "memory_plan": {
        "voxel_grid_dim": 96,
        "voxel_spacing_angstrom": 1.32,
        "padded_coverage_angstrom": 124.2,
        "vram_used_mib": 9070,
        "vram_free_mib": 6773,
        "adaptive_grid_used": true
      },
      "frame_audit": {
        "producer_count_per_stream": [12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500],
        "writer_count_per_stream": [12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500],
        "disk_count_per_stream": [12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500, 12500],
        "producer_hash_per_stream": ["fnv:abc1...", "fnv:def2...", "..."],
        "writer_hash_per_stream": ["fnv:abc1...", "fnv:def2...", "..."],
        "disk_hash_per_stream": ["fnv:abc1...", "fnv:def2...", "..."],
        "all_hashes_match": true,
        "audit_sidecar_relative_path": "replica_0/_audit.json"
      },
      "outputs": {
        "trajectory_arrow_relative": "replica_0/1zvd_clean.topology.spike_events.arrow",
        "trajectory_arrow_sha256": "...",
        "trajectory_arrow_bytes": 11781000000,
        "binding_sites_json_relative": "replica_0/1zvd_clean.binding_sites.json",
        "binding_sites_json_sha256": "...",
        "kcc_visualization_json_relative": "replica_0/1zvd_clean.kcc_visualization.json",
        "kcc_visualization_json_sha256": "...",
        "kcc_validation_json_relative": "replica_0/1zvd_clean.kcc_validation.json",
        "topology_prism_therm_json_relative": "replica_0/1zvd_clean.topology.prism_therm.json",
        "topology_prism_therm_json_sha256": "...",
        "topology_asc_consensus_json_relative": "replica_0/1zvd_clean.topology.asc_consensus.json",
        "topology_gcpid_synergy_json_relative": "replica_0/1zvd_clean.topology.gcpid_synergy.json",
        "topology_phasors_bin_relative": "replica_0/1zvd_clean.topology.phasors.bin",
        "topology_phasors_bin_sha256": "...",
        "topology_acl_contrast_bin_relative": "replica_0/1zvd_clean.topology.acl_contrast.bin",
        "topology_druggability_pdb_relative": "replica_0/1zvd_clean.topology.druggability.pdb",
        "ensemble_trajectory_json_relative": "replica_0/1zvd_clean.ensemble_trajectory.json"
      },
      "engine_telemetry": {
        "total_spikes": 86788631,
        "total_pockets_detected": 27,
        "n_cryptic_pockets": 0,
        "tide_residues_mapped": 156422,
        "kv_top1_vs_top2_separation": 0.6736,
        "n_consensus_residues": 10
      },
      "physical_time": {
        "dt_ps": 0.002,
        "save_interval_steps": 100,
        "n_steps_total": 25000,
        "physical_duration_ps": 50.0,
        "hmr_used": true,
        "hmr_dt_ps_effective": 0.004,
        "adaptive_dt_used": true,
        "adaptive_dt_average_ps": 0.0035
      }
    }
  ],

  "ensemble_consensus": {
    "computed_at": "2026-05-16T13:25:00Z",
    "computed_by": "scripts/training/ensemble_assembler_v004.py",
    "n_residues_with_full_coverage": 357,
    "n_residues_partial_coverage": 0,

    "per_residue_aggregates_relative_path": "ensemble_aggregates.parquet",
    "per_residue_aggregates_sha256": "...",

    "convergence_diagnostics": {
      "rhat_method": "gelman-rubin-1992",
      "ess_method": "stan-style-autocorrelation",
      "n_residues_rhat_lt_1.01": 340,
      "n_residues_rhat_lt_1.1": 355,
      "n_residues_rhat_gt_1.1": 2,
      "n_residues_rhat_gt_1.5": 0,
      "ess_min": 4.2,
      "ess_median": 4.9,
      "ess_max": 5.0,
      "rhat_per_residue_relative_path": "convergence_rhat.parquet",
      "ess_per_residue_relative_path": "convergence_ess.parquet"
    },

    "outlier_detection": {
      "method": "median-absolute-deviation",
      "mad_threshold": 3.5,
      "n_outlier_replicas_per_residue_max": 0,
      "outliers_detected": false,
      "downweighted_replicas": []
    },

    "bimodality": {
      "method": "hartigan-dip-test",
      "p_threshold": 0.05,
      "n_residues_bimodal": 7,
      "bimodal_residue_ids": [201, 202, 234, 245, 247, 312, 315]
    },

    "site_level_agreement": {
      "n_sites_in_all_replicas": 22,
      "n_sites_in_majority": 25,
      "n_sites_unique_to_one_replica": 2,
      "top1_site_id_agreement_fraction": 1.00,
      "top5_site_set_jaccard": 0.95
    }
  },

  "audit_log": {
    "validation_steps": [
      {"step": "all_replicas_completed", "passed": true, "details": "5 of 5 ok"},
      {"step": "frame_count_match_per_replica", "passed": true, "details": "20 streams x 12500 frames each"},
      {"step": "frame_hash_match_per_replica", "passed": true, "details": "producer == writer == disk"},
      {"step": "two_run_determinism_test", "passed": true, "details": "replica 0 and replica 0 rerun produce identical hashes within ε=1e-7"},
      {"step": "ensemble_consensus_assembled", "passed": true},
      {"step": "no_orphan_files", "passed": true},
      {"step": "all_replicas_have_ground_truth", "passed": true},
      {"step": "all_replicas_within_dt_consistency", "passed": true, "details": "dt_ps == 0.002 across replicas"}
    ],
    "validator_version": "1.0.0",
    "validated_at": "2026-05-16T13:25:30Z"
  }
}
```

---

## Field semantics + acceptance rules

### `schema_version`
Semantic versioning. Loader rejects unknown major versions with a clear error. Minor version bumps add fields (backward-compatible). Patch bumps fix bugs.

### `engine.binary_sha256`
The exact engine binary used. The ensemble assembler refuses to combine replicas with different binary hashes (cross-version contamination guard).

### `target.input_pdb_sha256` and `target.topology_json_sha256`
The exact input files used. Ensemble assembler refuses to combine replicas with different input hashes.

### `ensemble_config.replica_seed_strategy`
- `deterministic_offsets`: seeds are `base_seed + offset[i]` — best for reproducibility
- `random_independent`: seeds are independently drawn from system randomness — ensemble assembler logs this for traceability
- `external`: seeds came from a file/list — `seed_provenance` field required

### `ensemble_config.engine_flags_per_replica`
The exact CLI flags used. If different replicas used different flags, ensemble is invalid (the assembler refuses).

### `replicas[].status`
- `ok`: replica completed all streams + audit passed
- `partial_streams`: some streams failed but replica was kept (NOT recommended; sets `all_replicas_passed_audit = false`)
- `failed`: replica did not complete; not included in consensus
- `aborted`: replica was killed (spot preemption, OOM, etc.)

### `replicas[].frame_audit.all_hashes_match`
**Hard gate**: if false for any replica with status=ok, the ensemble manifest is INVALID. The campaign orchestrator must remove the replica or re-run it.

### `ensemble_consensus.convergence_diagnostics.rhat_per_residue_relative_path`
Per-residue per-feature R-hat values stored as parquet. Schema: `(residue_id, feature_name, rhat, ess, n_modes)`. Required for v004 training — student loss is weighted by 1/rhat to downweight unconverged residues.

### `ensemble_consensus.bimodality`
Residues with bimodal distributions get special handling in v004:
- Their loss is computed against the dominant mode, not the mean
- A separate "bimodality" feature is exposed to the student
- The student can learn to predict bimodality

### `audit_log.validation_steps`
**Hard gate**: every step must have `passed: true` for the manifest to be considered valid by the loader. Validator versioning (`audit_log.validator_version`) lets us add new validation steps without invalidating old manifests retroactively.

---

## Loader contract

```python
from prism4d.ensemble_loader import EnsembleManifest, EnsembleAssembler

# Load + validate
manifest = EnsembleManifest.from_path("ensemble_manifest.json")
# Raises EnsembleValidationError if any audit step failed or schema invalid

# Iterate replicas
for replica in manifest.replicas:
    if replica.status == "ok":
        arrow_path = replica.outputs.trajectory_arrow
        # ... extract per-replica features

# Assemble consensus
assembler = EnsembleAssembler(manifest)
consensus = assembler.compute_aggregates(
    rhat_min=1.0, rhat_max=1.1,    # convergence threshold
    outlier_method="mad", mad_k=3.5,
    bimodality_p=0.05,
)

# Output
consensus.to_parquet("ensemble_aggregates.parquet")
# columns: residue_id, feature_name, mean, std, median,
#          q05, q25, q75, q95, rhat, ess, is_bimodal, n_replicas_used
```

---

## Bundle assembler integration

`scripts/training/assemble_v003_bundle.py` (currently for v003 single-replica) extends to `assemble_v004_bundle.py`:

```python
# v003 (single-replica): target_<group> = (n_res, n_features)
# v004 (ensemble):
#   target_<group>_mean       = (n_res, n_features)
#   target_<group>_std        = (n_res, n_features)
#   target_<group>_rhat       = (n_res, n_features)
#   target_<group>_is_bimodal = (n_res, n_features)
```

The v004 student loss becomes a Gaussian NLL:
```python
# Per output dim: predict (mu_pred, log_sigma_pred)
# Target: (mu_target, sigma_target) from ensemble
# Loss: -log p(mu_target | mu_pred, sigma_pred) weighted by 1/rhat
```

---

## Concrete first-day action for the engineer

```bash
# 1. Create the schema validator
mkdir -p crates/prism-nhs/src/ensemble
$EDITOR crates/prism-nhs/src/ensemble/manifest.rs
# Add EnsembleManifest struct mirroring this schema.
# Use serde_json + custom Deserialize that runs the audit checks.

# 2. Add manifest writer to the engine teardown path
# In crates/prism-nhs/src/bin/nhs_rt_full.rs at the end of the run:
#   - Collect per-stream audit info (already exists from the V2 fix)
#   - Collect engine telemetry (total_spikes, total_pockets, etc.)
#   - Compute physical_time fields from dt_ps + n_steps + HMR/adaptive flags
#   - Write replica entry to <output>/ensemble_replica_<id>.json (one per replica run)

# 3. Add a separate "ensemble finalize" command to the CLI
nhs_rt_full ensemble-finalize \
    --target-dir <dir> \
    --campaign-id <id> \
    --base-seed 42 \
    --n-replicas-expected 5
# Reads all ensemble_replica_*.json, validates, writes ensemble_manifest.json,
# computes per-replica diff hashes for the two-run determinism check.

# 4. Validation suite (runs in CI)
cargo test --features v2_ignition,ensemble ensemble_manifest_tests
```

Expected first-day output:
- `crates/prism-nhs/src/ensemble/manifest.rs` — schema struct + validator
- Engine writes `ensemble_replica_<id>.json` after each run
- New CLI subcommand `ensemble-finalize`
- Tests passing on a 2-replica `1zvd_clean` smoke test

Day 2-3: Wire ensemble_assembler into the v003 bundle pipeline as `assemble_v004_bundle.py`. Add the v004 student architecture changes (Gaussian NLL heads).

---

## Required parquet schemas

### `ensemble_aggregates.parquet`

| column | type | semantics |
|---|---|---|
| `residue_id` | int32 | topology-index |
| `feature_group` | string | `kcc`, `therm`, `phasors`, etc. |
| `feature_name` | string | specific feature within group |
| `mean` | float32 | ensemble mean across replicas |
| `std` | float32 | ensemble std |
| `median` | float32 | for robust loss alternatives |
| `q05`, `q25`, `q75`, `q95` | float32 | quantiles for distributional supervision |
| `rhat` | float32 | Gelman-Rubin convergence |
| `ess` | float32 | effective sample size |
| `is_bimodal` | int8 | 0/1 from dip test |
| `n_replicas_used` | int8 | after outlier exclusion |

### `convergence_rhat.parquet`, `convergence_ess.parquet`

Wide format: `residue_id, <feature_1>_rhat, <feature_2>_rhat, ...`

---

## Acceptance gate for v004 ensemble teacher

The ensemble manifest schema is "production" when:

1. The Rust struct + validator exists and tests pass
2. The engine writes valid replica entries on every run (smoke-tested on `1zvd_clean`)
3. The `ensemble-finalize` CLI produces a valid manifest from N=2, N=3, N=5 replica runs
4. The Python ensemble_loader can round-trip the manifest (write → load → diff)
5. `assemble_v004_bundle.py` consumes the manifest and produces v004 training bundles
6. The v004 student trains on a 2-replica smoke target without errors
7. CI gates: any replica with `frame_audit.all_hashes_match = false` fails the ensemble; manifest validator rejects mismatched binary hashes across replicas

Until these seven gates pass, v004 ensemble teaching is not enabled in production.

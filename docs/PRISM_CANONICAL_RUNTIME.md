# PRISM Canonical Runtime

Canonical single-env runtime root:

- `/mnt/storage/prism_env_copies/prism_dock_portable_20260529`

Canonical scratch root:

- `/mnt/storage/prism-scratch/Prism4D-bio`

Canonical activation:

```bash
source scripts/prism-canonical-env.sh
```

This activation exports:

- `PRISM_DOCK_ENV=/mnt/storage/prism_env_copies/prism_dock_portable_20260529`
- `PRISM_SCRATCH_ROOT=/mnt/storage/prism-scratch/Prism4D-bio`
- `PRISM_CANONICAL_RUNTIME_STRICT=1`

Strict mode matters: canonicalized scripts are not allowed to silently fall
back to the live workstation env or repo `.scratch` when this mode is active.

Canonical verifier:

```bash
PYTHONPATH=src python3 scripts/verify_prism_canonical_runtime.py
```

Verifier outputs:

- JSON: `/mnt/storage/tmp/prism_canonical_runtime_verification_<UTC>_<git>/PRISM_CANONICAL_RUNTIME_VERIFICATION.json`
- Markdown: `/mnt/storage/tmp/prism_canonical_runtime_verification_<UTC>_<git>/PRISM_CANONICAL_RUNTIME_VERIFICATION.md`

Current verification scope:

- copied env import gate
- copied env relocation gate (no text wrappers in `bin/` may point back to the source env)
- runtime executable registry gate:
  - `manifests/prism_canonical_runtime_executables.json`
  - every product-boundary Python entrypoint is `py_compile` checked
  - every product-boundary shell entrypoint is `bash -n` checked
  - every product-boundary binary is existence/executable checked
- validated engine wrapper usage gate
- topology prep smoke
- generated-ligand holo compile smoke
- DSTW export smoke
- GFlowNet sampling smoke
- GFlowNet 1-epoch training smoke
- docking smoke
- md-only postflight contract
- phase manifold extractor smoke
- V2 hydration extractor smoke
- continuity maps smoke
- Log-SubTB spectral GFlowNet smoke

Latest verifier evidence:

- Control number: `PRISM-CANONICAL-RUNTIME-20260530T005841Z-90ffae25ed5f`
- Status: `PASS`
- JSON: `/mnt/storage/tmp/prism_canonical_runtime_verification_20260530T010000Z_runtime_registry_v2/PRISM_CANONICAL_RUNTIME_VERIFICATION.json`
- Markdown: `/mnt/storage/tmp/prism_canonical_runtime_verification_20260530T010000Z_runtime_registry_v2/PRISM_CANONICAL_RUNTIME_VERIFICATION.md`
- Product-boundary executable registry: `21/21` passed

Canonicalized support lanes now included in the runtime contract:

- `scripts/compute_am1bcc_charges.py`
- `scripts/generate_gpu_dispatch_batch.py`
- `scripts/materialize_tier3_pov_holo_registry.py`
- `scripts/sample_gflownet_candidates.py`
- `scripts/train_gflownet_policy.py`
- `scripts/rescore_gflownet_samples.py`
- `scripts/evaluate_gflownet_vs_baselines.py`
- `scripts/filter_gflownet_medchem_plausibility.py`
- `scripts/select_gflownet_diverse_top_candidates.py`
- `scripts/build_gflownet_review_artifacts.py`
- `scripts/build_cloud_candidate_payloads.py`
- `scripts/validate_and_package_gflownet_inference.py`
- `scripts/probe_oracle_throughput.py`
- `scripts/run_gflownet_inference_audit.sh`
- `scripts/run_ccns_validation_md.py`
- `scripts/verify_topology_fix.py`
- `scripts/verify_bifurcated_reward.py`
- `scripts/production/run_pharma_campaign.sh`
- `scripts/compile_all_ptx_sm120.sh`
- `target/release/oracle_scorer`

Runtime executable registry:

- `manifests/prism_canonical_runtime_executables.json`

Copied-env relocation repair:

```bash
python3 scripts/repair_canonical_env_copy.py
```

This rewrites text wrappers under the copied env `bin/` directory so they no
longer point back to the original source env. The canonical verifier now fails
if those source-env backreferences remain.

Important contract change:

- `scripts/prism-validate-and-run.sh` now routes `--md-only-evidence` runs through `scripts/prism-postflight-md-only.py` instead of the legacy `binding_sites.json` postflight gate.

Scope exclusions still present in the repo, but not part of the canonical
runtime gate:

- historical dossier/report scripts pinned to workstation output directories
- release sealing / restore verification tools that intentionally reference
  the original workspace path for leak detection or isolated-copy packaging
- quarantine scripts
- infrastructure / archival helpers
- blind-validation frozen logs and documents

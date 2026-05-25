# PRISM4D Ontological System Audit

Observed date: 2026-05-24 PDT
Repository: `/home/diddy/Desktop/Prism4D-bio`
Constraint honored: no sealing, no packaging, no archive creation, no deletion, no truncation.

## Directive Ontology

```yaml
directive_id: ontological-preservation-audit-20260524
objective: >
  Audit current PRISM4D workspace, preserve operational accounting, identify
  what is and is not protected by Git/version control, and map source, data,
  runtime, execution, and generated artifacts.
scope:
  - git worktree and local branch state
  - tracked, modified, untracked, ignored, and generated files
  - Rust/Python/TypeScript execution surfaces
  - GLP1R/M2/M3 campaign data flow and runtime files touched
required_components:
  - source-of-truth reconciliation
  - preservation risk ledger
  - engineering file tree map
  - data flow map
  - runtime open-file/process accounting
prohibited_actions:
  - package release bundles
  - seal or finalize artifacts
  - delete, truncate, reset, or clean files
  - silently add huge binary data to Git
expected_outputs:
  - auditable preservation report
  - explicit list of unprotected work
  - concrete next safe actions
execution_risk: high, due to 97 percent root disk usage and >100 GiB untracked artifacts
```

## Source Of Truth Reconciliation

The canonical working tree is on branch `main`, tracking `sota-dev/main`.

```text
HEAD: d08607bf chore: clean prism-nhs phase-one gate surface
branch: main
upstream: sota-dev/main
ahead: 439 commits
remotes: origin, sota, sota-dev, twin
```

The local branch has substantial committed history that is not confirmed pushed
to the tracked upstream. This is a preservation risk independent of the dirty
worktree: a disk failure could lose all 439 local commits if no remote contains
them.

## Git Preservation Accounting

Observed working tree state:

```text
modified tracked files: 28
untracked status groups: 164 before this report; 168 after this report and concurrent file activity
untracked non-ignored files: 16890
git-lfs tracked files: 0
git-lfs config: absent
git gc auto: disabled
git object store: 10.83 GiB loose + 28.13 GiB packs
```

Current tracked-source modifications are not committed:

```text
Cargo.lock
Cargo.toml
PRISM-DSTW_Engineering_Status_Report.md
PRISM-DSTW_State_of_the_Union.md
crates/prism-cuda-ext/Cargo.toml
crates/prism-escape-extract/Cargo.toml
crates/prism-gpu/src/context.rs
crates/prism-gpu/src/transfer_entropy.rs
crates/prism-nhs/Cargo.toml
crates/prism-nhs/src/bin/nhs_rt_full.rs
crates/prism-nhs/src/dstw_dispatch/mod.rs
crates/prism-nhs/src/dstw_dispatch/types.rs
crates/prism-nhs/src/lib.rs
crates/prism-niv-bench/src/main.rs
crates/prism-physics/src/dynamics_engine.rs
crates/prism-pipeline/Cargo.toml
crates/prism-spike-shipper/Cargo.toml
crates/prism-validation/Cargo.toml
crates/prism-validation/src/bin/run_dynamics_bench.rs
crates/prism-validation/src/bin/run_heterogeneous_bench.rs
crates/prism-ve-bench/Cargo.toml
crates/sdst/Cargo.toml
deny.toml
docs/CANONICAL_PROVENANCE.md
scripts/prism_kcc_decoder.py
scripts/prism_spike_event_integrator.py
scripts/training/xgboost_ranker_v4.py
src/prism_dstw/ontology.py
```

Highest-risk untracked non-ignored artifacts:

```text
100 GiB  PRISM_GLP1R_M2_Release_v1.0.tar.gz
119 MiB  cloud/prism-manifold-worker/node_modules/workerd/bin/workerd
119 MiB  cloud/prism-manifold-worker/node_modules/@cloudflare/workerd-linux-64/bin/workerd
29 MiB   PRISM_GLP1R_M2_DELIVERABLES_v1_1/PRISM_GLP1R_M2_PDF_DELIVERABLES_v1.1.tar.gz
24 MiB   pq
19 MiB   PRISM_GLP1R_M2_EXECUTIVE_RELEASE_v1.1.tar.gz
```

Highest-risk ignored/generated artifacts:

```text
104 GiB  campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/spike_events_snr_masked.parquet
19 GiB   campaigns/glp1r_aleniglipron/track_a_generative/fullscale_shards/shard_0001.parquet
17 GiB   campaigns/glp1r_aleniglipron/track_a_generative/fullscale_shards/shard_0000.parquet
90 GiB   target/
575 MiB  tuned_test/16_GBA_apo.binding_sites.json
538 MiB  campaigns/glp1r_aleniglipron/phase_2c_de_novo_capture/.../glp1r_6LN2_A316T.ensemble_trajectory.pdb
474 MiB  campaigns/glp1r_aleniglipron/phase_2c_de_novo_capture/.../glp1r_6XOX_WT.ensemble_trajectory.pdb
424 MiB  campaigns/glp1r_aleniglipron/integrated_spike_events/n80_full_scale/mechanical_load_network.parquet
351 MiB  campaigns/glp1r_aleniglipron/integrated_spike_events/capture_probe_5vex_r0_s0_5_10_15/spike_events_snr_masked.parquet
```

Untracked non-ignored file population is dominated by vendored dependency
trees:

```text
16150 files under node_modules/
11418 files under apps/glp1r-teaser-visualizer
4778 files under cloud/prism-manifold-worker
```

Concurrent activity note: during the audit window, additional untracked script
paths appeared or were touched, including `scripts/audit_pgx_resilience.py`,
`scripts/preflight_shard_r2_recovery.py`, and
`scripts/generate_lock_region_mask.py`. They are treated as user/automation
changes and were not reverted or overwritten.

Interpretation: Git is currently mixing three classes that need different
preservation treatment:

1. Source, manifests, tests, reports, schemas, and lockfiles should be committed
   and pushed through Git.
2. Generated dependency trees such as `node_modules/` should be ignored and
   reconstructed from `package-lock.json`.
3. Massive data products, release tarballs, parquet shards, trajectory PDBs, and
   binary matrices should be preserved through Git LFS, DVC, object storage, or
   a manifest-plus-checksum ledger, not plain Git blobs.

## Artifact Integrity Evidence

Release tarballs have checksum sidecars, but the sidecars are untracked and the
largest tarball is not protected by Git:

```text
PRISM_GLP1R_M2_Release_v1.0.tar.gz
  size: 106360009269 bytes
  sha256 sidecar: 3e3a6faa153ee426564a3fa7350d1826ea013d664fef04a2047abc2f0885b084

PRISM_GLP1R_M2_EXECUTIVE_RELEASE_v1.1.tar.gz
  size: 19353773 bytes
  sha256 sidecar: 0ce5dfab9228b6b7eb2c39d7fe78b9b54659323e2dd4e491741a254174b8f59d

PRISM_TRACK_A_GFLOWNET_V1_INFERENCE_AUDIT_v1.0.tar.gz
  size: 1339776 bytes
  sha256 sidecar: f9846fd655e4c51c9463ed3890fb12d81d20454e4c7f53b2f71022ca277bcb5b
```

The 100 GiB tarball is the single most dangerous untracked non-ignored file. It
should not be added to normal Git under current disk pressure. It should be
stored externally and referenced by checksum from a tracked manifest.

## Engineering File Tree Map

```text
00_registry/
  Ontology, architecture, chemistry, schemas, and templates. Several files are
  untracked and should be Git source-of-truth candidates.

04_TOPOLOGIES/
  GLP1R topology deliverables. Untracked; generally source/reference-grade if
  small JSON, artifact-grade if derived and large.

Cargo.toml, Cargo.lock, crates/
  Rust workspace and core execution graph. 27 workspace members resolve through
  cargo metadata. Modified crate manifests and source files are uncommitted.

src/prism_dstw/
  Python DSTW ontology/orchestration/persistence/chemistry surfaces. Several
  new subpackages are untracked.

scripts/
  Python and shell pipeline orchestration. Many new GFlowNet, topology,
  scaffold, extraction, audit, and validation scripts are untracked.

tests/
  Python and Rust integration/behavioral tests. New chemistry, GFlowNet,
  DKL/ELBO, MPNN, entropy-router, and FORGE tests are untracked.

campaigns/glp1r_aleniglipron/
  Current primary GLP1R campaign source, manifests, reports, generative outputs,
  integrated spike events, full-scale shards, and visualizer assets.

apps/glp1r-teaser-visualizer/
  Vite/React/Three frontend. Source and package lock are untracked, but most
  untracked volume/count is `node_modules`.

cloud/prism-manifold-worker/
  Cloudflare Worker/D1/R2/Vectorize/KV/Queue/Durable Object integration.
  Source and package lock are untracked, but most count is `node_modules`.

PRISM_GLP1R_M2_DELIVERABLES_v1_1/
  Release deliverable tree. Contains PDFs, LaTeX, markdown, data, visualization
  bundle, audit/CBOM, archive, table exports, and delivery manifests.

enterprise_release_review_*/
  Review outputs and sampled parquet audit artifacts. Generated but may contain
  provenance evidence that should be referenced from a tracked manifest.

target/
  Rust build output, 90 GiB, ignored and regenerable from source.

.git/
  Current Git history and object store, about 39 GiB total. This is not backed
  up merely by existing locally; push or bundle is required for off-disk safety.
```

## Rust Execution Surface

Cargo metadata resolves the following primary package graph:

```text
prism                   bin: prism
prism-core              lib, first-light-test, prism-monitor
prism-fluxnet           lib, fluxnet_train
prism-gnn               lib
prism-gpu               lib, examples, integration tests, CUDA build script
prism-physics           lib, prism-niv-bench, test-gpu-gnm, test-holographic
prism-io                lib, create-test-ptb, prism-diff, prism-export, prism-ingest
prism-lbs               lib/cdylib, prism-lbs, benchmark/train/readout bins
prism-whcr              lib
prism-mec               lib
prism-pipeline          lib
prism-ontology          lib
prism-geometry          lib
prism-ve                lib
prism-ve-bench          lib, train-fluxnet-ve, vasil-benchmark, verify-ic50-wiring
prism-niv-bench         lib and benchmark/demo bins
prism-learning          lib, prism-train, prism-train-neuro, prism-validate
prism-escape-extract    bin
prism-validation        lib plus many validation/benchmark bins
prism-nhs               lib plus NHS, DSTW, RT, teacher, ghost, and materialization bins
optix-sys               lib/build
prism-cuda-ext          lib/build
prism-optix             lib
sdst                    lib/build
prism-forge             rlib/cdylib, oracle_scorer, vspace_pruner
prism-report            lib, prism-report, prism4d
prism-spike-shipper     bin
```

The workspace default member is `crates/prism-ve-bench`. Cargo emitted one
warning: profiles inside `crates/prism-lbs/Cargo.toml` are ignored because
profiles must be defined at workspace root.

## TypeScript And Cloud Execution Surface

Frontend:

```text
apps/glp1r-teaser-visualizer/package.json
scripts:
  dev: vite --host 127.0.0.1
  build: tsc -b && vite build
  preview: vite preview --host 127.0.0.1
dependencies:
  React, Three, @react-three/fiber, drei, parquet-wasm, apache-arrow, zustand
```

Cloud worker:

```text
cloud/prism-manifold-worker/package.json
scripts:
  typecheck: tsc --noEmit
  deploy: wrangler deploy
bindings:
  R2 bucket: prism-tensors
  D1 database: prism_metadata
  Vectorize: dkl_latent_space
  KV: CONFIG_CACHE
  Queue: prism-dag-queue
  Durable Object: ScaffoldState
```

## Data Flow Map

```text
Reference/input stratum
  data/targets
  input_data/cxcr4
  pdb_refs, rcsb_refs, references
  campaigns/glp1r_aleniglipron/inputs/topologies
  00_registry/schemas

Extraction and topology stratum
  crates/prism-nhs
  crates/prism-gpu
  crates/prism-physics
  crates/prism-validation
  scripts/prism_*_extractor.py
  scripts/build_holo_state_topology.py
  scripts/prepare_receptor_steric_assets.py
  scripts/realign_ligand_to_topology.py

Generative chemistry and candidate stratum
  crates/prism-forge
  scripts/train_gflownet_policy.py
  scripts/sample_gflownet_candidates.py
  scripts/rescore_gflownet_samples.py
  scripts/select_gflownet_diverse_top_candidates.py
  scripts/query_real_aleniglipron_analogs.py

Campaign intelligence stratum
  campaigns/glp1r_aleniglipron/track_0_manual_emulation
  campaigns/glp1r_aleniglipron/track_a_generative
  campaigns/glp1r_aleniglipron/phase_2c_de_novo_capture
  campaigns/glp1r_aleniglipron/integrated_spike_events
  campaigns/glp1r_aleniglipron/M2_*.md
  campaigns/glp1r_aleniglipron/M3_Lead_Optimization_Dossier.md

Serialization and artifact stratum
  topology JSON
  CSV candidate tables
  JSON/JSONL manifests
  Parquet spike/event shards
  Arrow spike events
  binary warp matrices
  tar.gz release bundles with sha256 sidecars

Runtime/upload stratum
  crates/prism-spike-shipper
  /mnt/storage/prism-outputs/.r2-sync-manifest.jsonl
  Cloudflare R2 buckets
  /var/log/prism
```

## Runtime Files Touched

Active runtime processes touching this repo at audit time:

```text
PID 533020  prism-spike-shipper
  binary: target/release/prism-spike-shipper
  mode: --retroactive --keep-local-json
  watches:
    /mnt/storage/prism-outputs/blind_validation
    /mnt/storage/prism-outputs/hect-family
    /mnt/storage/prism-outputs/m1-strict-dcc-panel
    /mnt/storage/prism-outputs/validation
    output
    benchmarks/hard_targets/results
    .prism_orchestration
    results_1btl
    results_1w50
    results_3k5v
    results_4obe
    results_1hhp
  writes/logs:
    /var/log/prism
    /mnt/storage/prism-outputs/.r2-sync-manifest.jsonl
    /mnt/storage/prism-outputs/.r2-parquet-cache

PID 4089594  vite
  command: node apps/glp1r-teaser-visualizer/node_modules/.bin/vite --host 127.0.0.1 --port 5178
  runtime dependency: apps/glp1r-teaser-visualizer/node_modules

PID 682426/682427
  command: prism_enterprise_release_viewer.sh and python3
  cwd: campaigns/glp1r_aleniglipron/visualizer_app (deleted)
  risk: process holds deleted working directory inode; restart may not match on-disk state

PID 662570
  command: tee -a enterprise_release_review_20260523T231821Z/review.log
  write target: enterprise_release_review_20260523T231821Z/review.log

codex/claude/tmux sessions
  cwd: repository root
  effect: active shells and automation have repo open but are not primary data producers
```

System-level pressure adjacent to this repo:

```text
/var/log/attgw/host-events.ndjson
  size: 186 GiB
  source: rsyslog mirror of auth/audit logs
  growth observed: about 650 KiB in 5 seconds

/home/diddy/forensics/chain-of-custody/HASH-CHAIN.jsonl
  size: 328 GiB
  reader: /home/diddy/forensics/bin/ledger-chain.py
  growth observed during audit: 0 bytes over 5 seconds
```

## Preservation Risk Ledger

```text
P0: Branch ahead 439 commits
  Risk: committed work may only exist locally.
  Required preservation: push to trusted remote or create an external git bundle.

P0: 28 modified tracked source files
  Risk: source behavior can be lost by checkout/reset/editor overwrite.
  Required preservation: commit after review/tests or create a patch bundle.

P0: untracked source/schemas/scripts/tests
  Risk: not protected by Git at all.
  Required preservation: add/commit source, schema, manifest, and test files.

P0: 100 GiB release tarball
  Risk: not Git-tracked, enormous, and normal git add would worsen disk pressure.
  Required preservation: external object storage plus tracked checksum manifest.

P1: GLP1R parquet/fullscale/generated event data
  Risk: intentionally ignored but scientifically material.
  Required preservation: R2/DVC/LFS/artifact manifest, not normal Git.

P1: node_modules currently non-ignored
  Risk: status noise hides real untracked source.
  Required preservation: track package.json and package-lock.json, ignore node_modules.

P1: active deleted visualizer_app cwd
  Risk: running viewer may be serving stale/deleted inode state.
  Required preservation: restart from current on-disk source after source is committed.
```

## Immediate Safe Actions

These are intentionally not executed by this audit:

```bash
# Source-only checkpoint, after reviewing exact pathspecs:
git add Cargo.toml Cargo.lock deny.toml docs/CANONICAL_PROVENANCE.md
git add crates/prism-cuda-ext crates/prism-escape-extract crates/prism-gpu crates/prism-nhs crates/prism-niv-bench crates/prism-physics crates/prism-pipeline crates/prism-spike-shipper crates/prism-validation crates/prism-ve-bench crates/sdst
git add src/prism_dstw scripts tests 00_registry 04_TOPOLOGIES campaigns/glp1r_aleniglipron/*.md campaigns/glp1r_aleniglipron/*.json campaigns/glp1r_aleniglipron/*.csv

# Do not add these through normal Git:
#   target/
#   node_modules/
#   *.parquet
#   *.arrow
#   *.bin
#   *.ensemble_trajectory.pdb
#   PRISM_GLP1R_M2_Release_v1.0.tar.gz

# Preserve the huge release/data artifacts via checksum manifest and object store:
#   upload artifact externally
#   verify remote checksum
#   commit only the manifest/checksum/pointer
```

Recommended `.gitignore` hardening:

```text
node_modules/
apps/*/dist/
cloud/*/dist/
*.tar.gz          # only if release archives are preserved as external artifacts
```

Do not add `*.tar.gz` blindly if small release bundles are intentionally kept in
Git. Make the policy explicit before mutating `.gitignore`.

## Operationalization Gate

```yaml
compile_status: not run; audit only
runtime_status: active processes observed
telemetry_status:
  prism-spike-shipper active
  attgw audit log active and growing
  vite visualizer active
invariant_status:
  no destructive mutations performed
  no packaging or sealing performed
  no large artifacts added to Git
ontology_status:
  report created as docs/ONTOLOGICAL_SYSTEM_AUDIT_20260524.md
rollback_status:
  no rollback needed; only this report was added
unresolved_risks:
  - branch ahead 439 commits may be local-only
  - source work remains uncommitted
  - large artifacts remain outside Git
  - no LFS/DVC policy configured
  - root filesystem remains under pressure
```

## Final Determination

Status: `IMPLEMENTATION_PARTIAL`

Reason: the audit and accounting are complete enough to identify what is at
risk, but the actual preservation action is not complete until source changes
are committed and pushed, generated dependency trees are ignored, and large
artifacts are written to an external artifact store with tracked checksum
manifests.

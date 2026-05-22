# PRISM-DSTW Pan-Session Compliance Audit

Generated: 2026-05-21T19:39:17-07:00

Classification: Macro-Architecture and Micro-Engineering Alignment

Scope:

- Repository: `/home/diddy/Desktop/Prism4D-bio`
- Audit mode: adversarial read-only verification plus local command evidence
- Reference mandates: Workspace Engineering Standard, Arrow-native lineage directive, Track 0 strict sequencing directive, Apex architecture target for Rust/CUDA physics, Graph-Wasserstein DKL, GFlowNet-class active exploration, and tripartite Vectorize/D1/R2 store

## Executive Verdict

Overall status: **[Violating]**

The current codebase is **not aligned enough** to support the completed production deliverable system. The macro-architecture is visible, and several enforcement pieces exist, but the foundation is presently violating the required execution sequence and missing multiple mandatory production surfaces.

Primary reasons:

- `gate-phase0` contains a mypy command, but the exact required command fails before type checking begins.
- CI does not invoke `make gate-phase0` or mypy.
- `scripts/scratch/topo_regenerate.py` does not exist, so there is no topological regeneration implementation or topological sort order.
- Repository-local propagation sidecars do not contain supersedes pointers, source checksums, physical parameters, thresholds, beta constants, or `U_pose`.
- `edge_attenuator.rs`, `ligand_field.rs`, and `field_convolve.cu` are missing.
- `src/prism_dstw/validation/sdf_validator.py` is missing.
- SQLite-class dependency remnants remain even though workspace policy bans them.
- Graph-Wasserstein DKL, Pyro/GPyTorch inference, EiV forward sampling, and Chem-BALD are absent.
- The worktree is dirty, with 1005 `git status --short` entries, so it is not a reproducible hardened baseline.

The current state is therefore **[Violating]** the combined implementation plan. It can still serve as an audit baseline for remediation, but it cannot be represented as production-ready, calibration-grade, or DKL/GFlowNet-ready.

## Status Taxonomy

- **[Aligned]**: materially satisfies the mandate for the audited scope.
- **[Drifting]**: partially present, but inconsistent, incomplete, or not enforced end to end.
- **[Violating]**: missing, failing, or contradicting a mandatory gate or architecture requirement.

## 1. DAG and Cryptographic Lineage Inquisitor

### 1.1 Mypy Gate

Status: **[Violating]**

Evidence:

- [Makefile](/home/diddy/Desktop/Prism4D-bio/Makefile:6) defines `gate-phase0`.
- [Makefile](/home/diddy/Desktop/Prism4D-bio/Makefile:9) invokes `$(MYPY) --strict src/prism_dstw`.
- [.github/workflows/prism_dstw_gates.yml](/home/diddy/Desktop/Prism4D-bio/.github/workflows/prism_dstw_gates.yml:22) runs ban and provenance gates.
- [.github/workflows/prism_dstw_gates.yml](/home/diddy/Desktop/Prism4D-bio/.github/workflows/prism_dstw_gates.yml:28) runs compileall, not mypy.

Exact mypy command executed:

```bash
.venv/bin/python -m mypy --strict src/prism_dstw
```

Observed output:

```text
src/prism_dstw/propagation_ledger.py: error: Source file found twice under different module names: "prism_dstw.propagation_ledger" and "src.prism_dstw.propagation_ledger"
src/prism_dstw/propagation_ledger.py: note: See https://mypy.readthedocs.io/en/stable/running_mypy.html#mapping-file-paths-to-modules for more info
src/prism_dstw/propagation_ledger.py: note: Common resolutions include:
src/prism_dstw/propagation_ledger.py: note:     a) adding `__init__.py` somewhere,
src/prism_dstw/propagation_ledger.py: note:     b) using `--explicit-package-bases` or adjusting `MYPYPATH`
Found 1 error in 1 file (errors prevented further checking)
```

Alternate diagnostic command executed:

```bash
.venv/bin/python -m mypy --strict --explicit-package-bases src/prism_dstw
```

Observed output:

```text
src/prism_dstw/io.py:12: error: Skipping analyzing "pyarrow": module is installed, but missing library stubs or py.typed marker  [import-untyped]
src/prism_dstw/io.py:12: note: See https://mypy.readthedocs.io/en/stable/running_mypy.html#missing-imports
src/prism_dstw/io.py:128: error: Item "Series" of "DataFrame | Series | Any" has no attribute "write_parquet"  [union-attr]
Found 2 errors in 1 file (checked 5 source files)
```

Engineering finding:

This is a fatal sequence violation. The exact required mypy command is in the Makefile but does not pass. CI also does not enforce it. The package layout is currently ambiguous because both root-level `prism_dstw` and `src/prism_dstw` surfaces exist.

Required correction:

- Resolve package layout ambiguity.
- Make `mypy --strict src/prism_dstw` pass exactly, or revise the directive-approved command and update the gate consistently.
- Make CI invoke `make gate-phase0`, not a partial substitute.

### 1.2 Topological Regeneration Script

Status: **[Violating]**

Evidence:

- `scripts/scratch/topo_regenerate.py` does not exist.
- No `*topo_regenerate.py` was found under `scripts`.
- No topological sort order exists.
- No implementation was available to verify use of `pyarrow.parquet` metadata reads.
- No DAG regeneration could be performed.

Requested topological sort order:

```text
UNAVAILABLE: scripts/scratch/topo_regenerate.py is missing, so no DAG was built and no topological order exists.
```

Engineering finding:

This violates the topological regeneration mandate. There is no proof that the 19 target parquets can be deleted and regenerated from Arrow key-value lineage in dependency order.

Required correction:

- Implement `scripts/scratch/topo_regenerate.py`.
- Read `source_parquets` from Arrow key-value metadata using `pyarrow.parquet`.
- Build the DAG using `graphlib.TopologicalSorter`.
- Report the computed order before deletion/regeneration.
- Only then delete and regenerate target outputs.

### 1.3 Propagation Ledger Immutability and Mechanistic Content

Status: **[Violating]**

Repository-local evidence:

- Two `.propagation.jsonl` sidecars exist outside `.venv`:
  - [wt_physics_payload.propagation.jsonl](/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/dstw_ingest/round_000/wt_physics_payload.propagation.jsonl:1)
  - [BALD_Round_000_Response.propagation.jsonl](/home/diddy/Desktop/Prism4D-bio/campaigns/glp1r_aleniglipron/output/round_000/BALD_Round_000_Response.propagation.jsonl:1)

Observed ledger state:

- Entries per sidecar: 1.
- `supersedes`: `null`.
- `input_checksums`: empty.
- `output_uncertainty`: `null`.
- Physical parameters: absent.
- `beta_f`: absent.
- `beta_s`: absent.
- Thresholds: absent.
- `U_pose`: absent.

Engineering finding:

The ledgers are not yet useful as calibration-grade lineage. They are not immutable recomputation chains with supersession. They do not contain the physical parameters or uncertainty chain needed for downstream beta calibration.

Required correction:

- Ledger entries must capture physical parameters, threshold values, calibration labels, and uncertainty propagation.
- Regenerated entries must use `supersedes` pointers when replacing prior calculations.
- Every reportable numerical output must trace to source checksums and uncertainty.

## 2. Physics and Ontology Auditor

### 2.1 Rust Edge Attenuation Physics

Status: **[Violating]**

Evidence:

- `crates/prism-nhs/src/dstw_dispatch/edge_attenuator.rs` is missing.
- [crates/prism-nhs/src/dstw_dispatch/mod.rs](/home/diddy/Desktop/Prism4D-bio/crates/prism-nhs/src/dstw_dispatch/mod.rs:42) exposes existing dispatch modules, but no `edge_attenuator`.

Engineering finding:

There is no implementation of constructive/destructive DTSG edge attenuation. Therefore there is no physical logistic saturation against a defined `TE_max`, and no basis to reject naive software clamps because no production edge attenuator exists.

Required correction:

- Implement `edge_attenuator.rs`.
- Encode destructive and constructive interference with explicit physical units.
- Define `TE_max` and saturating functional form in the physical constants registry or Rust equivalent.
- Prohibit arbitrary numerical clamps as a substitute for physical saturation.

### 2.2 Track 0 Interference Schema

Status: **[Drifting]**

Aligned evidence:

- [track_0_interference_v1.yml](/home/diddy/Desktop/Prism4D-bio/00_registry/schemas/track_0_interference_v1.yml:8) defines `edge_from_residue` as `uint32`.
- [track_0_interference_v1.yml](/home/diddy/Desktop/Prism4D-bio/00_registry/schemas/track_0_interference_v1.yml:10) tags `edge_from_residue` with `ontology: ResidueIdx`.
- [track_0_interference_v1.yml](/home/diddy/Desktop/Prism4D-bio/00_registry/schemas/track_0_interference_v1.yml:13) defines `edge_to_residue` as `uint32`.
- [track_0_interference_v1.yml](/home/diddy/Desktop/Prism4D-bio/00_registry/schemas/track_0_interference_v1.yml:15) tags `edge_to_residue` with `ontology: ResidueIdx`.
- Strict enums exist for:
  - `edge_class`
  - `clash_assessment`
  - `complement_assessment`
  - `pose_confidence`

Drift:

- [track_0_interference_v1.yml](/home/diddy/Desktop/Prism4D-bio/00_registry/schemas/track_0_interference_v1.yml:56) keeps `structural_rationale` as a free string.

Engineering finding:

The core edge and assessment fields are structured, but prose is not eradicated. Whether `structural_rationale` should be free text or replaced with controlled rationale categories must be resolved. The current schema is useful for analyst capture but not fully tensor-grade.

Required correction:

- Decide whether structural rationale is permissible as auditable annotation or must be replaced by controlled categorical evidence codes.
- If calibration tensor purity is required, add controlled rationale enums and move free text to a separate non-calibration annotation field.

### 2.3 RDKit SDF Validation

Status: **[Violating]**

Evidence:

- `src/prism_dstw/validation/sdf_validator.py` is missing.
- `src/prism_dstw/validation` currently has no files.
- [src/prism_dstw/exceptions.py](/home/diddy/Desktop/Prism4D-bio/src/prism_dstw/exceptions.py:6) defines `FatalBoundaryError`.

Engineering finding:

The exception primitive exists, but there is no physical SDF intake validator. The system does not currently open SDF files, assert `K >= 10`, check `ChargeMethod == "AM1-BCC"`, verify 3D conformers, or fail through `FatalBoundaryError`.

Required correction:

- Implement `src/prism_dstw/validation/sdf_validator.py`.
- Use RDKit physical parsing.
- Enforce readable file, valid conformer count, `Is3D()`, non-null molecules, and AM1-BCC property tag.
- Integrate it only after `gate-phase0` is green if strict sequencing remains mandatory.

## 3. Apex Architecture Strategist

### 3.1 Worktree Reproducibility

Status: **[Violating]**

Evidence:

- `git status --short | wc -l` returned `1005`.

Engineering finding:

The workspace is not a hardened reproducible baseline. Many modified, deleted, or untracked files exist. This does not mean all changes are invalid, but it does mean current state cannot be used as a clean certification baseline without isolating and accounting for each delta.

Required correction:

- Separate intentional current-task changes from pre-existing dirty state.
- Produce a change manifest before any production certification.

### 3.2 Tripartite Store Readiness

Status: **[Drifting]**

Aligned evidence from architecture strategist:

- R2/D1/Vectorize bindings exist in `workers/prism-dataops/wrangler.toml`.
- Vectorize create script exists in `workers/prism-dataops/deploy.sh`.
- Worker code references `POCKET_INDEX`.

Drift:

- Documentation conflicts:
  - `PRODUCTION_LOGBOOK.md` says R2 exists while D1/Vectorize are none.
  - `CLOUDFLARE_INVENTORY.md` says Vectorize was not checked.

Engineering finding:

Tripartite store architecture is partially visible, but operational truth is not certified. D1/R2/Vectorize bindings are not the same as a validated persistent manifold store with DKL latent vectors, parquet blob URIs, and metadata joins.

Required correction:

- Reconcile Cloudflare inventory docs against actual deployment.
- Add a manifold-store certification report.
- Verify Vectorize index dimensions, D1 schema, R2 object layout, and join keys.

### 3.3 SQLite and DuckDB Remnants

Status: **[Violating]**

Evidence:

- [Cargo.toml](/home/diddy/Desktop/Prism4D-bio/Cargo.toml:72) declares banned crates metadata including SQLite-class crates.
- [Cargo.toml](/home/diddy/Desktop/Prism4D-bio/Cargo.toml:119) still enables `sqlx` with `sqlite`.
- `Cargo.lock` still contains `libsqlite3-sys` / `sqlx-sqlite` according to the architecture auditor.
- `scripts/ci/ban_check.py` bans DuckDB and SQLite-class imports/dependencies.

Engineering finding:

The policy and dependency tree contradict each other. DuckDB active imports appear purged, but SQLite-class transitive/runtime remnants remain. This violates the no SQLite/DuckDB lineage standard unless the dependency is strictly isolated behind an approved D1 metadata exemption.

Required correction:

- Remove SQLite features from workspace dependencies, or isolate an explicit D1-only exception module and dependency target.
- Make cargo-deny enforce the same policy as the Workspace Standard.

### 3.4 Rust/CUDA Ligand Field Surface

Status: **[Violating]**

Missing required Rust modules:

- `crates/prism-nhs/src/dstw_dispatch/ligand_field.rs`
- `crates/prism-nhs/src/dstw_dispatch/field_convolution.rs`
- `crates/prism-nhs/src/dstw_dispatch/edge_attenuator.rs`
- `crates/prism-nhs/src/dstw_dispatch/parquet_emitter.rs`

Missing required CUDA kernels:

- `field_convolve.cu`
- `edge_attenuate.cu`
- `wasserstein_sinkhorn.cu`

Existing related surface:

- `crates/prism-gpu/src/kernels/dstw_propagation.cu`
- `crates/prism-nhs/src/dstw_dispatch/cuda.rs`
- `crates/prism-nhs/src/dstw_dispatch/projection.rs`

Engineering finding:

Existing DSTW code is variant/rotamer projection and a 3-channel propagation matrix path. It is not K-conformer ligand field tensor generation, batch field convolution, clash/complement decomposition, bivariate edge attenuation, or U_pose Welford accumulation.

Required correction:

- Implement the Rust/CUDA Layer 1 hot path before DTPF, DKL, or Chem-BALD.
- Emit `Chem_Perturbed_DTSG.parquet` and `U_pose.parquet` through Arrow/parquet-rs.

### 3.5 ML and DKL Readiness

Status: **[Violating]**

Evidence:

- Graph-Wasserstein DKL implementation is absent.
- Pyro/GPyTorch stack is absent.
- EiV forward sampling from `N(E[DTSG], U_pose)` is absent.
- Chem-BALD active learner is absent.
- Prior audit already recorded missing DKL/Bayesian implementation in [PRISM-DSTW_State_of_the_Union.md](/home/diddy/Desktop/Prism4D-bio/PRISM-DSTW_State_of_the_Union.md:125).

Sklearn status:

- Status: **[Aligned]** for the specific non-quarantined sklearn examples checked by the architecture strategist, because they are line-exempted in `00_registry/ban_exemptions.yml`.
- Caveat: line exemptions are not a production-ready endpoint; they are a managed remediation backlog.

Engineering finding:

The workspace is not ready for production Graph-Wasserstein DKL. It lacks canonical `Chem_Perturbed_DTSG`/`U_pose` tensors and lacks the DKL/Pyro/GPyTorch implementation.

Required correction:

- Do not build Module 6 over placeholder data.
- Build Layer 1 tensor emission first.
- Then implement DKL with EiV propagation and calibration gate.

## 4. Gate Summary

| Gate / Architecture Surface | Status | Reason |
| --- | --- | --- |
| `make gate-phase0` mypy command | **[Violating]** | Exact command fails due package module duplication. |
| CI mypy enforcement | **[Violating]** | CI does not run mypy or `make gate-phase0`. |
| Ban check | **[Aligned]** | `python3 scripts/ci/ban_check.py` passed locally. |
| Parquet provenance check | **[Aligned]** | `python3 scripts/ci/parquet_provenance_check.py` passed locally for 2 files. |
| DAG regeneration script | **[Violating]** | `scripts/scratch/topo_regenerate.py` missing. |
| Topological sort order | **[Violating]** | No DAG script; no computed order. |
| Propagation ledger content | **[Violating]** | No source checksums, supersedes, physical params, thresholds, or U_pose. |
| Track 0 interference schema | **[Drifting]** | Core enums exist; free-text rationale remains. |
| SDF physical validator | **[Violating]** | Missing. |
| Rust edge attenuator | **[Violating]** | Missing. |
| Rust ligand field tensor generation | **[Violating]** | Missing. |
| CUDA batch field convolution | **[Violating]** | Missing. |
| Tripartite store | **[Drifting]** | Bindings exist; operational docs conflict. |
| SQLite/DuckDB purge | **[Violating]** | SQLite-class deps remain. |
| Graph-Wasserstein DKL | **[Violating]** | Missing. |
| Chem-BALD / GFlowNet-ready active exploration | **[Violating]** | Missing required inputs and active learner surface. |
| Worktree reproducibility | **[Violating]** | 1005 dirty status entries. |

## 5. Alignment With Completed Production Deliverable System

Current codebase status: **[Violating]**

### What Is Aligned

- Ban gate exists and passed locally.
- Parquet provenance gate exists and passed locally for repository-scoped files.
- `FatalBoundaryError` exists in `src/prism_dstw/exceptions.py`.
- Track 0 interference schema has strict residue index typing and strict assessment enums for the core edge assessment fields.
- R2/D1/Vectorize architecture has partial worker/deployment surfaces.
- Sklearn usage found by the strategist is managed through existing line exemptions.

### What Is Drifting

- `gate-phase0` exists but CI does not run it.
- Tripartite store code and documentation disagree.
- Track 0 schema still preserves a free-text rationale field.
- Root package and `src` package coexist, creating mypy module ambiguity.
- Exempted sklearn code remains a remediation backlog, not a final clean ML foundation.

### What Is Violating

- Exact mypy gate fails.
- DAG regeneration script is missing.
- Propagation ledgers lack cryptographic dependency content and mechanistic parameters.
- RDKit SDF validator is missing.
- Rust edge attenuation physics is missing.
- Rust/CUDA ligand field projection and batch convolution are missing.
- SQLite dependency remnants remain.
- DKL/Pyro/GPyTorch/EiV/Chem-BALD are absent.
- Worktree is not a clean certification baseline.

## 6. Required Remediation Order

The audit supports the strict sequence constraint. The correct order is:

1. **Stop Phase 3 Track 0 workbook/SDF work until Phase 1 is green.**
2. Resolve package layout ambiguity between `prism_dstw` and `src/prism_dstw`.
3. Make the exact command `mypy --strict src/prism_dstw` pass.
4. Make CI invoke `make gate-phase0`.
5. Implement `scripts/scratch/topo_regenerate.py`.
6. Compute and report the topological sort order from Arrow metadata.
7. Regenerate target parquets through the DAG script.
8. Upgrade ledgers to include supersedes pointers, source checksums, physical parameters, thresholds, and U_pose chain.
9. Only then resume Track 0 SDF validator and workbook generation.
10. After the data layer is certified, implement Rust/CUDA Layer 1.
11. After canonical `Chem_Perturbed_DTSG` and `U_pose` exist, implement DTPF/Modules 1-5.
12. After real analog-conditioned paths exist, implement Graph-Wasserstein DKL, Chem-BALD, and GFlowNet-class active exploration.

## 7. Non-Negotiable Claim Boundaries

Until the violations above are resolved, the codebase must not be described as:

- production-ready
- calibration-grade
- DKL-ready
- GFlowNet-ready
- fully lineage-certified
- Rust/CUDA Scaffold-State Interferometry complete
- analog-conditioned
- chronic durability complete

Permissible description:

- enforcement baseline under active remediation
- receptor-side evidence substrate
- partial Track 0 schema registry
- non-production architecture scaffold
- active violation map for remediation sequencing

## Final Determination

The current codebase is **[Violating]** the combined implementation plan for the completed production deliverable system.

This is not because the vision is incoherent. The vision is internally consistent: Arrow-native lineage, Rust/CUDA physics, ontology-typed boundaries, Graph-Wasserstein DKL, and tripartite persistence can support the intended GFlowNet/DKL architecture. The present implementation simply does not yet satisfy the mandatory gates that would allow that architecture to carry production scientific weight.

The highest-priority blockers are:

1. Make `mypy --strict src/prism_dstw` pass exactly.
2. Add mypy to CI through `make gate-phase0`.
3. Implement the Arrow-metadata DAG regeneration script.
4. Regenerate target parquets through the DAG.
5. Upgrade propagation ledgers from file-level provenance to mechanistic lineage.
6. Remove SQLite-class dependency contradictions.
7. Implement Rust/CUDA Layer 1 before any DKL/Chem-BALD production work.


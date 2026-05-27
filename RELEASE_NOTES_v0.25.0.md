# PRISM-4D Hardened Release v0.25.0

## Scope

This release seals the verified PRISM-4D stack after `v0.24.2-motif-intelligence`.
It adds hardened integration audits, materials-track smoke coverage, motif E2E
smoke coverage, exact Python dependency pinning, and per-subsystem Merkle CBOM
generation.

## Sealed Subsystems

- PRISM-FORGE Rust reaction/oracle/SubTB tile guard surfaces.
- PRISM-DSTW Python orchestration, GFlowNet, calibration, and motif modules.
- PRISM-NHS neuromorphic/shear binary surfaces.
- PRISM-MAT battery interphase adapter surface.
- GLP1R campaign artifacts including Track B SubTB spectral artifacts.
- E2E forensic audit bundle.
- Scripts, templates, tests, dependency lockfiles, and reaction registry.

## Verification Gates

- Import-resolution audit.
- Parquet schema compatibility audit.
- Cross-language bridge tests.
- Materials E2E smoke tests.
- Motif intelligence E2E smoke tests.
- Dependency pinning audit.
- Per-subsystem Merkle CBOM and checksum replay.

## Boundary

The release is computational and provenance-sealed. It makes no biological
efficacy claims and does not relabel projected data as observed.

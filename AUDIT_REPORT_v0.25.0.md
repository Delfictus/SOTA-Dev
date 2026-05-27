# PRISM-4D Hardened Release Audit Report v0.25.0

## Source State

- Predecessor tag: `v0.24.2-motif-intelligence`
- Hardened release tag target: `v0.25.0-hardened-release`

## Audit Additions

- `scripts/audit_import_resolution.py`
- `scripts/audit_schema_compatibility.py`
- `scripts/audit_dependency_pinning.py`
- `scripts/build_hardened_cbom.py`

## Runtime Evidence Inputs

- `.audit-reports/system_verification_audit_report.md`
- `.audit-reports/epoch024_v2_motif_intelligence_validation.md`
- `.audit-reports/epoch024_v2_motif_defensive_quality_review.md`
- `campaigns/glp1r_aleniglipron/audits/e2e_ontology_forensic_v2/`
- `campaigns/glp1r_aleniglipron/track_b_chronological/subtb_spectral/`

## Verification Boundary

This report is completed by the command outputs generated during the release
hardening run. The CBOM records the exact files and hashes included in the
release; the verification script replays the critical gates.

FORENSIC VALIDATION VERDICT: PASS

Scope:
- Read-only validation of the Epoch 025 hardened release surface after the
  verifier was split into temp recomputation plus sealed artifact comparison.

Evidence:
- VERIFICATION.sh writes fresh audit/CBOM outputs under VERIFY_TMP and compares
  them against release_artifacts/v0.25.0.
- release_artifacts/v0.25.0/logs/verification_epoch025_hardened_release_readonly.log
  contains temp import, schema, dependency, and CBOM paths and ends with
  VERIFICATION COMPLETE.
- release_artifacts/v0.25.0/hardened_release_manifest.json records
  subsystem_count=14.
- release_artifacts/v0.25.0/CBOM_v2.0.json includes PRISM_ARCHIVE_SUPPORT and
  RELEASE_CONTROL.
- No files were staged during the validation audit.
- The intended selective add set contains no files over 100MB.

Warning:
- campaigns/glp1r_aleniglipron/audits/e2e_ontology_forensic_v2/ontology_index.sqlite
  is 188,293,120 bytes and must not be force-added to Git. It is preserved in
  the external release package and covered by the CBOM/checksum manifest.

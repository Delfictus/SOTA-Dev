FINAL BUG HUNT VERDICT: PASS AFTER SELECTIVE COMMIT STEP

Scope:
- Read-only bug hunt of the Epoch 025 hardened release seal surface after the
  read-only verifier and CBOM coverage fixes.

Findings:
- No CRITICAL defects found in verifier behavior, checksum verification, CBOM
  coverage, or selective staging safety.
- The only HIGH finding before commit was that HEAD did not yet contain the
  release surface. The required action is selective staging, commit, then tag
  the resulting commit.

Passed checks:
- VERIFICATION.sh no longer mutates release_artifacts/v0.25.0; fresh audit and
  CBOM outputs are written under /tmp via VERIFY_TMP.
- SHA validation uses the existing sealed release_artifacts/v0.25.0/SHA256SUMS.txt.
- CBOM covers RELEASE_CONTROL, PRISM_ARCHIVE_SUPPORT, DEPENDENCIES,
  CAMPAIGN_MAT, TESTS, SCRIPTS, and CAMPAIGN_BIO.
- CBOM includes VERIFICATION.sh, RELEASE_NOTES_v0.25.0.md,
  AUDIT_REPORT_v0.25.0.md, runpod_training/requirements.txt, .archive/crates,
  campaign materials manifest, integration tests, hardened audit scripts, and
  GLP1R audit artifacts.
- Selective staging does not include the 188MB ontology_index.sqlite. It is
  ignored by the repository and must not be force-added.

Non-blocking observation:
- import_audit_report.json records optional_unavailable_count=20 with
  strict_optionals=false. unresolved_count remains 0.

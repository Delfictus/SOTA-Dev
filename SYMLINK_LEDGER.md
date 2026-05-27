# Symlink Ledger

Source of truth: `HERMETIC_INTEGRITY_LEDGER.json`, scan version `E025-R1.0-v2.1`.

Scanner result: 4 symlink findings, 0 open, 0 critical, 0 high real issues.

| Path | Target class | Classification | Status | Release verdict |
| --- | --- | --- | --- | --- |
| `shear_stress_field.parquet` | repo-contained relative symlink | `FALSE_POSITIVE` | `ACCEPTED_RISK` | Allowed; resolves inside repo. |
| `prepped_20260510_0108` | ignored/generated local symlink | `GENERATED_OUTPUT_ONLY` | `ACCEPTED_RISK` | Excluded from source release. |
| `kv31_family_validation/kv31_family_validation` | ignored/generated local symlink | `GENERATED_OUTPUT_ONLY` | `ACCEPTED_RISK` | Excluded from source release. |
| `kv31_validation_frozen_20260512T215852Z/results/kv31_family_validation/kv31_family_validation` | ignored/generated local symlink | `GENERATED_OUTPUT_ONLY` | `ACCEPTED_RISK` | Excluded from source release. |

No symlink remediation deleted runtime dependencies. Generated/local links remain classified as generated output; the tracked release symlink is repo-contained.

# Outbound And File IO Ledger

Source of truth: `HERMETIC_INTEGRITY_LEDGER.json`, scan version `E025-R1.0-v2.1`.

Scanner result:

| Category | Count | Open | Release status |
| --- | ---: | ---: | --- |
| `EXTERNAL_READ` | 174 | 0 | Classified as generated output, test-only, or config/environment required. |
| `SUBPROCESS` | 220 | 0 | Classified as test-only, baseline tooling, or config/environment required. |
| `HARDCODED_PATH` | 17512 | 0 | Classified as generated output, test-only, config/environment required, or already mitigated. |
| `DATA_DEPENDENCY` | 9051 | 0 | Machine-checked against `RELEASE_DATA_MANIFEST.md` or generated/test classification. |

External chemistry binaries (`xtb`, `antechamber`, `sqm`, `obabel`) are preserved as `CONFIG_ENV_REQUIRED` runtime dependencies and documented in `ENVIRONMENT.md`.

Cloud/R2/D1/Vectorize access remains capability-preserving: bindings and tokens are not removed, and full production mode requires configured environment/secrets. Buyer/reviewer smoke tests do not require private local filesystem state.

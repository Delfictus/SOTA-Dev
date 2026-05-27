# E025-R0.5 Hermetic Release Closure

Status gate: E025 must not be regenerated until this gate is PASS.

## Preservation Directive

Do not delete runtime dependencies.
Do not remove symlinks or hardcoded paths until each one is classified.

For every external path, choose one release action:

1. Convert it to a configurable environment variable.
2. Add it to `RELEASE_DATA_MANIFEST.md`.
3. Replace it with a committed fixture if it is small and test-only.
4. Mark it as generated output.

The goal is hermetic packaging, not capability reduction.

## Required Modes

- Developer mode: fully functional on the operator workstation.
- Clean release mode: reproducible from committed code plus documented external data and secrets.
- Buyer/reviewer mode: can run smoke tests without private local filesystem state.
- Full production mode: can run when `PRISM_SCRATCH_ROOT`, R2, Vectorize, D1, and release data manifests are supplied.

## Required Answers

Each E025-R0.5 audit must answer:

- Are there any symlinks?
- Do any symlinks point outside the repo?
- Are any runtime-required files untracked?
- Are any absolute local paths embedded?
- Does the code depend on `/mnt/storage`, `/home/diddy`, `/tmp`, `.env`, `.dev.vars`, `.wrangler`, R2, D1, Vectorize, local Parquet, or hidden generated files?
- Can the project build/test from a clean `git archive` checkout with only committed files?

## Required Ledgers

- `SYMLINK_LEDGER.md`
- `OUTBOUND_AND_FILE_IO_LEDGER.md`
- `RELEASE_DATA_MANIFEST.md`
- `SECRET_EXCLUSION_LEDGER.md`
- `CLEAN_ARCHIVE_REPRODUCTION.md`

Local full-output runs belong under ignored `release_audit/`. Release-facing ledgers should be committed only after review/classification.

# E025-R0.5 Clean Archive Reproduction

- source commit verified: `4533d787`
- run location: `/mnt/storage/tmp/e025_clean_archive_verify.FovVQC/src`
- archive command: `git archive --format=tar HEAD | tar -x -C "$TMP_PARENT/src"`
- cargo metadata: `cargo metadata --no-deps --format-version=1`
- Rust workspace check: `cargo check --workspace`
- clippy gate: `cargo clippy -p prism-forge -- -D warnings`
- Rust release tests: `cargo test -p prism-forge --release`
- Python smoke suite: `PYTHONPATH=src python3 -m pytest tests/ -x -q`

## Result

PASS. The clean archive checkout built and tested from committed files only.

Evidence:

- `cargo check --workspace`: exit 0
- `cargo clippy -p prism-forge -- -D warnings`: exit 0
- `cargo test -p prism-forge --release`: 25 passed, exit 0
- `PYTHONPATH=src python3 -m pytest tests/ -x -q`: 1180 passed, 42 skipped, 9 warnings, exit 0

The skipped Python tests are data/hardware gated and must remain documented in `RELEASE_DATA_MANIFEST.md` or their own explicit skip messages.

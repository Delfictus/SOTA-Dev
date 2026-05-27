# E025-R1.0 Clean Archive Reproduction

- source commit in archive: `$Format:%H$`
- source commit verification: archive-expanded hash must equal `git rev-parse HEAD`
- archive command: `git archive --format=tar HEAD | tar -x -C "$ARCHIVE_DIR"`
- run location pattern: `/mnt/storage/tmp/e025_r10_clean_archive.*`
- Rust workspace check: `cargo check --workspace`
- clippy gate: `cargo clippy -p prism-forge -- -D warnings`
- Rust release tests: `cargo test -p prism-forge --release`
- Python suite: `PYTHONPATH=src python3 -m pytest tests/ -x -q`

## Required Command

```bash
RELEASE_COMMIT="$(git rev-parse HEAD)"
ARCHIVE_DIR="$(mktemp -d /mnt/storage/tmp/e025_r10_clean_archive.XXXXXX)"
git archive --format=tar HEAD | tar -x -C "$ARCHIVE_DIR"
ARCHIVE_COMMIT="$(tr -d '\n' < "$ARCHIVE_DIR/CLEAN_ARCHIVE_SOURCE_COMMIT.txt")"
test "$ARCHIVE_COMMIT" = "$RELEASE_COMMIT"
(
  cd "$ARCHIVE_DIR"
  cargo check --workspace
  cargo clippy -p prism-forge -- -D warnings
  cargo test -p prism-forge --release
  PYTHONPATH=src python3 -m pytest tests/ -x -q
)
rm -rf "$ARCHIVE_DIR"
```

## Result Contract

PASS requires:

- `ARCHIVE_COMMIT == RELEASE_COMMIT`
- `cargo check --workspace`: exit 0
- `cargo clippy -p prism-forge -- -D warnings`: exit 0
- `cargo test -p prism-forge --release`: exit 0
- `PYTHONPATH=src python3 -m pytest tests/ -x -q`: exit 0
- no source, scanner, manifest, override, or ledger changes after the archive test

The skipped Python tests are data/hardware gated and must remain documented in `RELEASE_DATA_MANIFEST.md` or their own explicit skip messages.

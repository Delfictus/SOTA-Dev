# Volatile Files Reconciliation

## Scope

This file applies to `Track B` workstation archive only.

It does **not** block `Track A` canonical PRISM sealing unless a required PRISM file exists only in the broader workstation archive surface and is absent from `/home/diddy/PRISM-ISOLATED-20260528`.

## Active Archive Services

- `prism-workstation-archive-home-20260529-v2.service`
- `prism-workstation-archive-mnt-20260529-v2.service`

Both services were observed `active` during the current reconciliation pass.

## Confirmed Volatile `.codex` Evidence

From `/home/diddy/PRISM-ISOLATED-20260528/logs/workstation_archive_home.log`:

- line `1391`:
  - `.codex/logs_2.sqlite-wal: corrupted on transfer: md5 hashes differ`
- line `1392`:
  - `.codex/logs_2.sqlite-wal: Removing failed copy`
- line `1396`:
  - `.codex/logs_2.sqlite: Failed to copy ... api error BadDigest`

These are live-changing sqlite/WAL artifacts and must not be treated as sealed after the first pass.

## Additional Archive Fidelity Notes

The workstation archive is also encountering many symlink-preservation notices under copied environment trees, for example:

- `miniconda3/envs/unimol/lib/libsqlite3.so.0: Can't follow symlink without -L/--copy-links`
- `miniconda3/envs/pocketminer/ssl/cert.pem: Can't follow symlink without -L/--copy-links`

These are archive fidelity issues for Track B and must be reconciled in a second pass or recorded explicitly.

## Required Reconciliation Actions

1. Let the current archive passes finish or quiesce them safely.
2. Re-run a second archive pass over the volatile `.codex` surface.
3. For sqlite databases where stable capture is required, use `sqlite3 .backup` on a quiesced copy where possible.
4. Preserve WAL files only if they are stable and required for consistency.
5. If the `.codex` sqlite/WAL surface cannot be stabilized, keep workstation archive status at `PARTIAL` and record `LIVE_VOLATILE_NOT_SEALED`.

## Current Status

- `track_a_impact`: `none_observed`
- `track_b_status`: `PARTIAL_PENDING_RECONCILIATION`
- `live_volatile_not_sealed`: `true`

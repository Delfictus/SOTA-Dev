# Release Tracks

## Track A

- `name`: `PRISM_CANONICAL_PLATFORM_RELEASE`
- `input_root`: `/home/diddy/PRISM-ISOLATED-20260528`
- `authoritative_repo_root`: `/home/diddy/PRISM-ISOLATED-20260528/absolute_root/home/diddy/Desktop/Prism4D-bio`
- `authoritative_env_root`: `/home/diddy/PRISM-ISOLATED-20260528/absolute_root/home/diddy/miniconda3`
- `authoritative_candidate_smoke_root`: `/home/diddy/PRISM-ISOLATED-20260528/absolute_root/mnt/storage/tmp/glp1r_candidate_md_smoke_20260527_233437`
- `purpose`: sealed operational PRISM release
- `required_destinations`: `GitHub`, `external SSD`, `Cloudflare R2`
- `status`: `ready_for_capacity_safe_sealing`
- `truth_boundary`:
  - Phase 2C sealed receptor/variant evidence is claimable.
  - Platform smoke is claimable.
  - Hydration to DSTW integration is implemented and smoke-verified.
  - Full 104GB hydration extraction is not fully run and remains unmeasured.
  - Full Phase 1-3 production completion is not claimable.
- `does_not_wait_for_track_b`: `true`
- `exception`: wait only if a required PRISM runtime file exists only in the workstation archive surface and is absent from the canonical copy.

## Track B

- `name`: `WORKSTATION_ARCHIVE`
- `input_roots`:
  - `/home/diddy`
  - `/mnt/*`
- `purpose`: broader workstation preservation
- `status`: `archive_in_progress`
- `blocker`: volatile live-changing `.codex` sqlite/WAL reconciliation
- `separate_from_track_a`: `true`
- `current_live_services`:
  - `prism-workstation-archive-home-20260529-v2.service`
  - `prism-workstation-archive-mnt-20260529-v2.service`

## Operational Rule

Track A is the canonical PRISM preservation path. Track B is a broader machine archive path. Track B volatility must not block Track A unless Track A is missing a required PRISM artifact.

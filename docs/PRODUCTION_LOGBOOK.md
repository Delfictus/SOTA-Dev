# PRISM-4D Production Logbook

Chronological record of production-relevant sessions, decisions, and
infrastructure changes. Each session appends one entry.

---

## 2026-04-05 — SOP system bootstrap + architecture documentation

**Session scope:** DevOps / SOPs / documentation / cleanup only.
Engine development, campaign launch, and dual-pass investigation are
handled in a separate session and were explicitly out of scope.

### Deliverables

**Task A — Architecture documentation**

- Created `docs/sops/` tree: `architecture/`, `engine_reference/`,
  `innovations/`, `infrastructure/`, `credentials/`.
- Wrote `docs/sops/architecture/PRISM_TWIN_ARCHITECTURE.md` — full
  7-layer architecture of PRISM-TWIN with the critical clarification
  that TWIN is **coupled observation, not two-stream MD**, grounded
  in `docs/PRISM_TWIN_INTERFEROMETRIC_DESIGN.md`.
- Wrote `docs/sops/architecture/COUPLED_OBSERVATION.md` — why NOT
  replica exchange, why NOT enhanced sampling, ring buffer protocol,
  consensus vs differential classification.
- Wrote `docs/sops/engine_reference/OUTPUT_SPEC_COMPLETE.md` —
  **stub with `status: CONTENT_PENDING`**. The full spec content was
  not pasted in the user directive; no fields were fabricated. The
  stub lists the intended Sections 0-7 and per-site subsections
  5.1-5.13 as placeholders to be filled when the spec is provided.
- Wrote `docs/sops/innovations/INNOVATION_REGISTRY.md` — 15 documented
  innovations with classification (PAT / TS / OSS / MIXED).

**Task B — Script execution enforcement rules**

- Created `docs/PRISM4D_DEV_OPS_FRAMEWORK.md` with the script-execution
  rules: `scripts/production/` executable freely; everything else
  requires explicit permission; read-only inline Python allowed with
  `[DIAGNOSTIC]` tag; no production script may reference `/tmp`.
- Added the same ruleset to `CLAUDE.md` under a new "SCRIPT EXECUTION
  POLICY" section.

**Task C — `/tmp` cleanup**

- Quarantined `/tmp/generate_tem1_report.py`, `/tmp/enrich_tem1_report.py`,
  `/tmp/fix_tem1_v3.py` → `scripts/quarantine/` with `SUPERSEDED`
  headers, then deleted originals.
- Inventoried ~40 other stray `.py` / `.sh` files in `/tmp` (listed
  in the session transcript). No deletions were performed on that
  list without explicit confirmation.
- Verified `grep -r "/tmp/" scripts/production/` returned zero
  matches (directory did not exist at grep time — vacuously clean).

**Task D — Credentials and Cloudflare inventory**

- Wrote `docs/sops/infrastructure/CLOUDFLARE_INVENTORY.md` —
  authoritative snapshot of all existing Cloudflare resources.
- Wrote per-resource SOPs: `CLOUDFLARE_WRANGLER.md`,
  `CLOUDFLARE_ACCOUNT.md`, `CLOUDFLARE_R2.md`, `CLOUDFLARE_WORKERS.md`,
  `API_DELFICTUS.md`.
- Wrote `docs/sops/credentials/CREDENTIAL_REGISTRY.md` — lists what
  credentials exist and where they are stored (no secret values).
- Added `docs/sops/ip/TRADE_SECRET_REGISTRY.md` and
  `docs/sops/credentials/CREDENTIAL_REGISTRY.md` to `.gitignore`.

### Discovered state (Cloudflare)

- Wrangler 4.80.0 at `/usr/bin/wrangler` (global npm install)
- Authenticated: `is@delfictus.com`, OAuth token
- Account ID: `0b9ebf4f9a2a36c66302cbb9f32ab1f9`
- Existing Worker: `prism-api` on `api.delfictus.com/*` (live, HTTP 200)
- Existing R2 buckets: `prism-archive`, `prism-production`, `prism-public`
- Existing D1 / Queues / KV / Vectorize: **none**

### Security flag raised

`wrangler.toml` in repo root contained `API_KEY = "prism-4d-2026-delfictus"`
in plaintext `[vars]`. File is currently untracked. Recommendation (not
executed this session — flagged for user decision): move to `wrangler
secret put API_KEY` and rotate. Logged in `CREDENTIAL_REGISTRY.md`.

### Open items carried forward

- `OUTPUT_SPEC_COMPLETE.md` awaits user paste of full spec content.
- `API_KEY` rotation is pending user authorization.
- Full SOP tree (the ~90-document expansion covering `architecture/`,
  `innovations/`, `ip/`, `source/`, `data/`, `engine_reference/`,
  `decisions/`) will be populated **as each procedure is executed or
  discovered**, not pre-stubbed. Per enforcement rule: SOP is written
  as part of the procedure, not after.

### Commits (this session)

*To be appended after each commit completes.*

---

## 2026-04-09/10 — Sentinel observation of TWIN v3.0 + containerization catalog

**Session scope:** DevOps sentinel — read-only observation of active TWIN
development session, dependency cataloging, container manifest, Worker API
integration.

### Deliverables

**Task A — Real-time observation infrastructure**

- Deployed two persistent monitors:
  - `inotifywait` on `crates/prism-nhs/src/` and `crates/prism-gpu/src/kernels/`
    (instant file-write detection)
  - Git + process poll (5s interval) for new commits and engine runs
- Observed 1 commit (`d2c1500f` — Glass Box infrastructure) and 1 engine run
  (4obe KRAS G12C validation, 216s runtime, 9.5GB output)
- All events POSTed to Worker API (`/observe`, `/snapshot`)

**Task B — Container dependency manifest**

- Wrote `docs/sentinel/container_manifest.json` — complete frozen-state
  dependency tree: binary (6.2MB), 94 PTX files, 95 .cu sources, 11 .cuh
  headers, libsdst.so (1.1MB), 461 Python packages, 482 data files
- Key finding: CUDA toolkit is **13.2** (not 12.9 as documented elsewhere)
- Key finding: prism-prep is a shell script at `scripts/prism-prep` (34KB),
  not a compiled Rust binary

**Task C — Observation log with cross-referenced timestamps**

- Wrote `docs/sentinel/observation_log.md` — full timeline, commit analysis,
  engine run args, output file inventory, flags, cross-references

**Task D — Worker API integration**

- POSTed: 1 system snapshot, 1 BUILD_IN_PROGRESS, 1 NEW_COMMIT, 2 ENGINE_RUN,
  1 ENGINE_COMPLETE, 1 FLAG_ALERT
- Worker auto-triggered PRISM-Observer cloud agent on ENGINE_COMPLETE
- Created remote trigger `sentinel-twin-observer` for ongoing 15-min polling

### Flags raised

- **HIGH — /tmp output not synced to R2**: Glass Box 4obe output at
  `/tmp/multi_diff_glass/` (9.5GB). Spike watcher daemon only watches
  `/mnt/storage/prism-outputs/`. Data will be lost on reboot.

### Discovered state

- CUDA 13.2 (V13.2.51), driver 595.45.04, RTX 5080 sm_120
- Rust 1.93.0, Python 3.12.3, cudarc 0.18.2
- New engine flag: `--multi-differential` (Glass Box TWIN mode)
- 3 managed cloud agents confirmed active in Worker API

### Open items carried forward

- `/tmp/multi_diff_glass/` output needs manual R2 upload or move to watched path
- Container image build from manifest not yet executed
- CUDA 13.2 vs 12.9 documentation discrepancy needs resolution across all docs

---

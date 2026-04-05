# PRISM-4D DevOps Framework

Operational rules for all work inside this repository. These rules are
enforced for both human and AI contributors. Violations are framework
failures and must be reported in `docs/PRODUCTION_LOGBOOK.md`.

---

## 1. Script execution enforcement

### Rule — NO SCRIPT EXECUTION WITHOUT PRODUCTION PATH TAG

- **`scripts/production/`** — scripts in this directory may be executed
  freely. They are the sanctioned production surface.
- **All other scripts** — any executable file outside
  `scripts/production/` requires **explicit permission before
  execution**, including (but not limited to):
  - `scripts/` (top level, non-production subpaths)
  - `benchmarks/`
  - `prism-ai-inference/scripts/`
  - `scripts/quarantine/`
  - Anything in `/tmp`, `~`, or an untracked location
- **Inline `python3` heredocs that WRITE anything require permission.**
  "Writing" includes file creation, file modification, database
  mutation, network POST/PUT/DELETE, or any side effect outside
  stdout/stderr.
- **Inline `python3` that only READS** (loads a file, inspects a
  structure, prints a summary) is allowed **only if** the command is
  flagged `[DIAGNOSTIC]` in the invocation. Example:
  ```
  [DIAGNOSTIC] python3 -c "import json; print(len(json.load(open('x.json'))['sites']))"
  ```
- **Multi-line `python3` heredocs should not be invocation-inlined.**
  If logic is more than ~10 lines, write it as a script file in
  `scripts/quarantine/` first, then get permission to run.
- **No production script may reference or import from `/tmp`.**
  Enforcement: `grep -r "/tmp/" scripts/production/` must return zero
  results. Checked at every commit touching `scripts/production/`.

### Rationale

Three prior incidents motivated this rule:

1. Dev scratch in `/tmp` was silently imported by production tooling
   and disappeared on reboot, breaking downstream consumers.
2. Inline heredocs wrote files to unexpected paths (CWD at time of
   invocation), polluting git status and occasionally overwriting
   real outputs.
3. "One-off" scripts in `scripts/` diverged from the production
   versions in `scripts/production/`, producing results that looked
   canonical but were not.

### Enforcement checklist (per session)

- [ ] Before running any script outside `scripts/production/`, ask for
      permission or cite prior permission in context.
- [ ] Tag all read-only inline `python3` invocations `[DIAGNOSTIC]`.
- [ ] Refuse to write inline heredocs longer than ~10 lines; create a
      quarantine script instead.
- [ ] On any commit touching `scripts/production/`, run the `/tmp`
      grep check and log the result.

---

## 2. Standard Operating Procedures (SOPs)

### Rule — documentation is written as part of the procedure

When any contributor (human or AI) discovers, configures, or executes
a production-relevant procedure, the corresponding SOP under
`docs/sops/` must be created or updated **in the same session as the
discovery/execution**, not later.

This applies to:

- Infrastructure discoveries (Cloudflare, GPU workstation, external services)
- Credential rotations or new credential provisioning
- Engine flag changes, kernel changes, build procedure changes
- Pipeline changes (ingest, postprocess, R2 sync)
- ML training/eval runs that produce new models
- Customer-facing API or product changes

The SOP format is defined in `docs/sops/SOP_TEMPLATE.md` (to be
created). Every SOP must carry YAML frontmatter with at minimum:
`name`, `description`, `type`, `category`, `criticality`, `owner`,
`last_verified`, `version`.

A discovery without an SOP update is a framework violation.

---

## 3. Commit discipline

- **Commit after each task**, not at the end of a session.
- **Never `git add .` or `git add -A`** — stage specific files. The
  project has large amounts of untracked generated state that must not
  enter the repo.
- **Never commit secrets.** `CREDENTIAL_REGISTRY.md` documents what
  exists and where; actual secret values live only in their secure
  stores.
- `docs/sops/ip/TRADE_SECRET_REGISTRY.md` and
  `docs/sops/credentials/CREDENTIAL_REGISTRY.md` must be in
  `.gitignore`.

---

## 4. Verification discipline

Per global user rules, every factual claim must be backed by observed
command output. Specifically:

- No claiming code compiles without running `cargo check` (Rust) or
  `python3 -m pytest` (Python).
- No claiming a file / function / module exists without Read or Grep.
- No paraphrasing command output — show it.
- Report failures in full, do not summarize.

The Claude Code stop-hook at `.claude/hooks/verify-before-stop.sh`
enforces the Rust half of this.

---

## History

| Date       | Change                                                      | By              |
|------------|-------------------------------------------------------------|-----------------|
| 2026-04-05 | Initial framework doc with script execution rules (Task B). | Ididia Serfaty  |

---
name: Wrangler — installation, version, authentication
description: Where Wrangler is installed, what version, how it is authenticated, and how to re-authenticate if the token expires
type: infrastructure
category: infrastructure
criticality: CRITICAL
owner: Ididia Serfaty
last_verified: 2026-04-05
version: 1.0
---

# Wrangler — installation, version, authentication

## Installed state (verified 2026-04-05)

| Field       | Value                                             |
|-------------|---------------------------------------------------|
| Binary      | `/usr/bin/wrangler`                               |
| Version     | `4.80.0`                                          |
| Install     | Global npm: `npm list -g` shows `wrangler@4.80.0` |
| Config dir  | `~/.config/.wrangler/` (`config/`, `logs/`, `metrics.json`) |
| Auth method | OAuth token (via `wrangler login`)                |
| Logged in as| `is@delfictus.com`                                |

## Verification commands

```bash
which wrangler
wrangler --version
wrangler whoami
```

Expected `whoami` output: account `Is@delfictus.com's Account`, ID
`0b9ebf4f9a2a36c66302cbb9f32ab1f9`, plus token-scope listing. Full
scope list is recorded in `CLOUDFLARE_INVENTORY.md`.

## Do not re-authenticate without asking

The current OAuth token has full scopes needed for all Phase-1 work
(D1, Queues, Workers, R2, AI). Re-running `wrangler login` will
invalidate the existing token and open a browser flow. **Never run
`wrangler login` or `wrangler logout` in a session without explicit
user authorization.** If `wrangler whoami` reports not logged in,
stop and escalate.

## Upgrade procedure (if needed)

```bash
npm install -g wrangler@latest
wrangler --version
wrangler whoami   # confirm auth survived the upgrade
```

Log the upgrade in `docs/PRODUCTION_LOGBOOK.md`.

## Related SOPs

- `CLOUDFLARE_INVENTORY.md` — full account resource snapshot
- `CLOUDFLARE_ACCOUNT.md` — account-level admin access
- `../credentials/CREDENTIAL_REGISTRY.md` — where the OAuth token is stored

## History

| Date       | Change                                  | By              |
|------------|-----------------------------------------|-----------------|
| 2026-04-05 | Initial SOP from Task D inventory scan. | Ididia Serfaty  |

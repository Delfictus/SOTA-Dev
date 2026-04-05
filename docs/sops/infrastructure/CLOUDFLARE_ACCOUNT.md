---
name: Cloudflare Account — Delfictus
description: Account identification, authentication, token scopes, and access controls for the Delfictus Cloudflare account
type: infrastructure
category: infrastructure
criticality: CRITICAL
owner: Ididia Serfaty
last_verified: 2026-04-05
version: 1.0
---

# Cloudflare Account — Delfictus

## Identification

| Field        | Value                                               |
|--------------|-----------------------------------------------------|
| Email        | `is@delfictus.com`                                  |
| Account name | `Is@delfictus.com's Account`                        |
| Account ID   | `0b9ebf4f9a2a36c66302cbb9f32ab1f9`                  |
| Primary zone | `delfictus.com`                                     |

## Authentication

- Method: OAuth token issued by `wrangler login`
- Location on disk: `~/.config/.wrangler/config/` (managed by wrangler,
  do not edit by hand)
- See `CLOUDFLARE_WRANGLER.md` for the verification and upgrade
  procedure.

## Token scopes (as of 2026-04-05)

```
account (read)           user (read)
workers (write)          workers_kv (write)
workers_routes (write)   workers_scripts (write)
workers_tail (read)      d1 (write)
pages (write)            zone (read)
ssl_certs (write)        ai (write)
ai-search (write)        ai-search (run)
queues (write)           pipelines (write)
secrets_store (write)    containers (write)
cloudchamber (write)     connectivity (admin)
offline_access
```

All Phase-1 work (D1 + Queues + Workers + R2 + AI Gateway) is covered.

## Zone / DNS

- `delfictus.com` zone is managed in this account.
- `api.delfictus.com` is routed to the `prism-api` worker — see
  `API_DELFICTUS.md`.

## Provisioned resources

See `CLOUDFLARE_INVENTORY.md` for the complete, always-current list.

## Access discipline

- Do not add team members, API tokens, or service accounts without
  updating `CLOUDFLARE_ACCOUNT.md` and `../credentials/CREDENTIAL_REGISTRY.md`.
- Rotate the OAuth token if the workstation is compromised or
  transferred.

## Related SOPs

- `CLOUDFLARE_WRANGLER.md`
- `CLOUDFLARE_INVENTORY.md`
- `CLOUDFLARE_R2.md`
- `CLOUDFLARE_WORKERS.md`
- `API_DELFICTUS.md`

## History

| Date       | Change                           | By              |
|------------|----------------------------------|-----------------|
| 2026-04-05 | Initial SOP from Task D scan.    | Ididia Serfaty  |

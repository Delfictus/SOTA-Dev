---
name: Cloudflare Inventory
description: Authoritative snapshot of all existing Cloudflare resources for the Delfictus account. Single source of truth before any new resource creation.
type: infrastructure
category: infrastructure
criticality: CRITICAL
owner: Ididia Serfaty
last_verified: 2026-04-05
version: 1.0
---

# Cloudflare Inventory

This document is the **entry point for all Cloudflare operations**. It
is updated whenever a Cloudflare resource is created, deleted, or
discovered. Do not create new resources without first consulting this
inventory — there may already be one.

## Account

| Field        | Value                                               |
|--------------|-----------------------------------------------------|
| Email        | `is@delfictus.com`                                  |
| Account name | `Is@delfictus.com's Account`                        |
| Account ID   | `0b9ebf4f9a2a36c66302cbb9f32ab1f9`                  |
| Auth method  | OAuth token (wrangler login)                        |
| Verified     | 2026-04-05 via `wrangler whoami`                    |

Token scopes (from `wrangler whoami`):
`account (read)`, `user (read)`, `workers (write)`, `workers_kv (write)`,
`workers_routes (write)`, `workers_scripts (write)`, `workers_tail (read)`,
`d1 (write)`, `pages (write)`, `zone (read)`, `ssl_certs (write)`,
`ai (write)`, `ai-search (write)`, `ai-search (run)`, `queues (write)`,
`pipelines (write)`, `secrets_store (write)`, `containers (write)`,
`cloudchamber (write)`, `connectivity (admin)`, `offline_access`.

**No re-authentication is required** for any Phase-1 work. All needed
scopes (D1, Queues, Workers, R2 via pipelines, AI) are already present.

## Wrangler installation

| Field       | Value                                                   |
|-------------|---------------------------------------------------------|
| Binary path | `/usr/bin/wrangler`                                     |
| Version     | `4.80.0`                                                |
| Install via | Global `npm install -g wrangler` (confirmed via `npm list -g`) |
| Config dir  | `~/.config/.wrangler/` (contains `config`, `logs`, `metrics.json`) |
| Verified    | 2026-04-05                                              |

Full SOP: `CLOUDFLARE_WRANGLER.md`.

## R2 buckets

| Name              | Created    | Purpose (observed)                 | Bindings in `prism-api` worker |
|-------------------|------------|------------------------------------|--------------------------------|
| `prism-archive`   | 2026-04-02 | CryptoBench results, historical archive | `env.ARCHIVE`             |
| `prism-production`| 2026-04-02 | Current production campaign outputs | `env.PRODUCTION`              |
| `prism-public`    | 2026-04-02 | Public-facing data                 | `env.PUBLIC`                   |

Verified via `wrangler r2 bucket list` 2026-04-05.

Full SOP: `CLOUDFLARE_R2.md`.

## Workers

| Name        | Route                    | Zone             | Status | Source               |
|-------------|--------------------------|------------------|--------|----------------------|
| `prism-api` | `api.delfictus.com/*`    | `delfictus.com`  | LIVE   | `prism-api-worker.js` (repo root, untracked) |

Live-check 2026-04-05: `curl https://api.delfictus.com/health` → HTTP 200
with JSON `{platform: "PRISM-4D", version: "1.1", ...}`.

Deployment history (last few via `wrangler deployments list`):
- 2026-04-02T19:37 — initial upload
- 2026-04-02T19:45 — added R2 binding `ARCHIVE`
- 2026-04-02T19:45 — added R2 binding `PRODUCTION`
- 2026-04-02T19:45 — added R2 binding `PUBLIC`

Endpoints exposed: `/`, `/health`, `/status`, `/targets`, `/sites/:pdb_id`,
`/download/:path`, `/list/:prefix`. All non-health endpoints are
Bearer-authenticated against `env.API_KEY`.

Full SOP: `CLOUDFLARE_WORKERS.md`.

## D1 databases

**NONE.**

Verified via `wrangler d1 list` 2026-04-05 → `[]`.

When the first D1 database is provisioned (expected: campaign status
tracking), create `CLOUDFLARE_D1.md` and update this inventory.

## Queues

**NONE.**

Verified via `wrangler queues list` 2026-04-05 → empty response.

When the first queue is provisioned (expected: `prism-postprocess-queue`,
`prism-r2sync-queue`), create `CLOUDFLARE_QUEUES.md` and update this
inventory.

## KV namespaces

**NONE.**

Verified via `wrangler kv namespace list` 2026-04-05 → `[]`.

## Vectorize indexes

**Not checked** in this session. Re-verify before any vector-store work.

## AI Gateway / Workers AI

**Not inventoried** in this session. Scope `ai (write)` and
`ai-search (write)` are present on the token.

## Pipelines / Containers / Cloudchamber

**Not inventoried.** Scopes are present.

## Wrangler config in repo

`wrangler.toml` exists at repo root (untracked):

```toml
name = "prism-api"
main = "prism-api-worker.js"
compatibility_date = "2026-04-02"

routes = [
  { pattern = "api.delfictus.com/*", zone_name = "delfictus.com" }
]

[[r2_buckets]]
binding = "ARCHIVE"
bucket_name = "prism-archive"

[[r2_buckets]]
binding = "PRODUCTION"
bucket_name = "prism-production"

[[r2_buckets]]
binding = "PUBLIC"
bucket_name = "prism-public"

[vars]
API_KEY = "prism-4d-2026-delfictus"
```

## ⚠️ Security exposure

`wrangler.toml` contains `API_KEY = "prism-4d-2026-delfictus"` in
plaintext `[vars]`. This is the Bearer token the live `prism-api`
worker validates against. The file is currently untracked but one
`git add .` away from being committed.

**Recommended remediation** (awaiting user authorization, not executed
in this session):

1. Rotate the secret:
   ```
   wrangler secret put API_KEY   # paste a new random value
   ```
2. Remove the `[vars] API_KEY` line from `wrangler.toml`.
3. Redeploy the worker.
4. Update `CREDENTIAL_REGISTRY.md` with the rotation date.
5. Update this inventory.

## Related SOPs

- `CLOUDFLARE_WRANGLER.md` — wrangler install + auth procedure
- `CLOUDFLARE_ACCOUNT.md` — account + billing + admin access
- `CLOUDFLARE_R2.md` — R2 bucket usage + sync protocol
- `CLOUDFLARE_WORKERS.md` — `prism-api` worker documentation
- `API_DELFICTUS.md` — domain + DNS + endpoint surface
- `../credentials/CREDENTIAL_REGISTRY.md` — secrets registry

## History

| Date       | Change                                    | By              |
|------------|-------------------------------------------|-----------------|
| 2026-04-05 | Initial inventory (Task D of SOP bootstrap). | Ididia Serfaty |

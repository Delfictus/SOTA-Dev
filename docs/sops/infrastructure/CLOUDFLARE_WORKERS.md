---
name: Cloudflare Workers — prism-api
description: Deployed Worker inventory, routes, bindings, source location, endpoint surface
type: infrastructure
category: infrastructure
criticality: CRITICAL
owner: Ididia Serfaty
last_verified: 2026-04-05
version: 1.0
---

# Cloudflare Workers

## Deployed workers

| Name        | Route                 | Zone            | Source                                   | Status (2026-04-05) |
|-------------|-----------------------|-----------------|------------------------------------------|---------------------|
| `prism-api` | `api.delfictus.com/*` | `delfictus.com` | `prism-api-worker.js` (repo root, untracked) | LIVE (HTTP 200)    |

## `prism-api`

### Source

Source file: `prism-api-worker.js` at repo root. **Currently untracked.**
Config: `wrangler.toml` at repo root. **Currently untracked.**

Both files should either be committed (after resolving the API_KEY
exposure — see `CLOUDFLARE_INVENTORY.md` §Security exposure) or moved
under `scripts/production/api/` per the production-path rule.

### Bindings

From `wrangler.toml`:

| Binding        | Type      | Target              |
|----------------|-----------|---------------------|
| `env.ARCHIVE`    | R2 bucket | `prism-archive`     |
| `env.PRODUCTION` | R2 bucket | `prism-production`  |
| `env.PUBLIC`     | R2 bucket | `prism-public`      |
| `env.API_KEY`    | var       | Bearer token (⚠️ plaintext in `[vars]`) |

### Endpoint surface

Observed from source + `/health` response 2026-04-05:

| Path                  | Auth | Purpose                                                 |
|-----------------------|------|---------------------------------------------------------|
| `GET /`               | none | Platform banner + endpoint list                         |
| `GET /health`         | none | Liveness probe (same response as `/`)                   |
| `GET /status`         | Bearer | Object counts across all three R2 buckets             |
| `GET /targets`        | Bearer | Top-level target listing under `cryptobench199/`      |
| `GET /sites/:pdb_id`  | Bearer | Per-chain `binding_sites.json` summaries for a PDB ID |
| `GET /download/:path` | Bearer | Raw object stream from `prism-archive`                |
| `GET /list/:prefix`   | Bearer | Object listing under a prefix                         |
| `OPTIONS *`           | none | CORS preflight                                         |

Bearer auth is `Authorization: Bearer <env.API_KEY>`. CORS is
wide-open (`Access-Control-Allow-Origin: *`).

### Deployment history

From `wrangler deployments list` 2026-04-05 (first four entries):

| Timestamp           | Version     | Message                     |
|---------------------|-------------|-----------------------------|
| 2026-04-02 19:37 UTC | ce3fa36b    | Automatic deployment on upload |
| 2026-04-02 19:45 UTC | 51da62b1    | Added R2 bucket binding ARCHIVE |
| 2026-04-02 19:45 UTC | 178f856b    | Added R2 bucket binding PRODUCTION |
| 2026-04-02 19:45 UTC | 5b85440f    | Added R2 bucket binding PUBLIC |
| 2026-04-02 19:49 UTC | (more)      | (additional deployments — truncated) |

### Live check

```bash
curl -s https://api.delfictus.com/health
# → {"platform":"PRISM-4D","version":"1.1","developer":"Delfictus IO Inc.","endpoints":[...]}
```

### Deploy procedure

From repo root with `wrangler.toml` present:

```bash
wrangler deploy
```

The OAuth token documented in `CLOUDFLARE_WRANGLER.md` must be valid.

### Open issues

- **`API_KEY` plaintext exposure** — see `CLOUDFLARE_INVENTORY.md`
  §Security exposure. Recommended rotation path: `wrangler secret put
  API_KEY`, remove `[vars]` line, redeploy.
- **Source file location** — `prism-api-worker.js` and `wrangler.toml`
  should be moved to a canonical location (e.g. `scripts/production/api/`)
  and committed.

## Related SOPs

- `CLOUDFLARE_INVENTORY.md`
- `CLOUDFLARE_R2.md`
- `CLOUDFLARE_WRANGLER.md`
- `API_DELFICTUS.md`

## History

| Date       | Change                          | By              |
|------------|---------------------------------|-----------------|
| 2026-04-05 | Initial SOP from Task D scan.   | Ididia Serfaty  |

---
name: Cloudflare R2 — buckets, bindings, access
description: R2 bucket inventory, worker bindings, upload pipeline reference
type: infrastructure
category: infrastructure
criticality: CRITICAL
owner: Ididia Serfaty
last_verified: 2026-04-05
version: 1.0
---

# Cloudflare R2

## Buckets

| Name               | Created    | Used for                                    | Worker binding    |
|--------------------|------------|---------------------------------------------|-------------------|
| `prism-archive`    | 2026-04-02 | CryptoBench results + historical archive    | `env.ARCHIVE`     |
| `prism-production` | 2026-04-02 | Current production campaign outputs         | `env.PRODUCTION`  |
| `prism-public`     | 2026-04-02 | Public-facing data                          | `env.PUBLIC`      |

Verified via `wrangler r2 bucket list` 2026-04-05.

## Worker bindings

Defined in `wrangler.toml` at repo root. The `prism-api` worker binds
all three buckets; see `CLOUDFLARE_WORKERS.md`.

## Observed prefix layout (inferred from the `prism-api` worker)

- `prism-archive/cryptobench199/<pdb_id>/...` — per-target CryptoBench
  outputs, including `<chain>.binding_sites.json` files that the
  `/sites/:pdb_id` endpoint serves.

Other prefixes in `prism-archive`, `prism-production`, `prism-public`
are not documented in this session. Add them as they are used.

## Upload pipeline

The canonical R2 sync procedure is to be documented in
`docs/sops/pipeline/R2_SYNC.md` (not created in this session).
The project memory notes an "auto-sync daemon + Parquet conversion +
R2 bucket config" for the engine output path.

## Access

- Via Worker: every endpoint on `api.delfictus.com/*` that reads R2
  goes through the bindings above. This is the public access path.
- Via Wrangler CLI: `wrangler r2 object get/put/list/delete`. Requires
  the OAuth token documented in `CLOUDFLARE_WRANGLER.md`.
- Via rclone: the project memory notes R2 credentials live in
  `~/.config/rclone/rclone.conf` — see `../credentials/CREDENTIAL_REGISTRY.md`.
  These are account-level access keys, distinct from the OAuth token.

## Related SOPs

- `CLOUDFLARE_INVENTORY.md`
- `CLOUDFLARE_WORKERS.md`
- `../credentials/CREDENTIAL_REGISTRY.md`

## History

| Date       | Change                          | By              |
|------------|---------------------------------|-----------------|
| 2026-04-05 | Initial SOP from Task D scan.   | Ididia Serfaty  |

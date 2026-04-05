---
name: api.delfictus.com — domain, DNS, endpoint surface
description: Public API domain configuration, DNS resolution, and live endpoint verification
type: infrastructure
category: infrastructure
criticality: CRITICAL
owner: Ididia Serfaty
last_verified: 2026-04-05
version: 1.0
---

# api.delfictus.com

## Overview

Public API for the PRISM-4D platform. All customer-facing reads of
R2-hosted results go through this hostname. Backed by a single
Cloudflare Worker (`prism-api`) bound to all three R2 buckets.

## DNS (verified 2026-04-05)

```
$ dig +short api.delfictus.com
172.67.155.5
104.21.58.23
```

Cloudflare edge IPs — correctly proxied through CF.

## Zone

- Zone: `delfictus.com`
- Account: `0b9ebf4f9a2a36c66302cbb9f32ab1f9` (see `CLOUDFLARE_ACCOUNT.md`)
- Route: `api.delfictus.com/*` → worker `prism-api` (see
  `CLOUDFLARE_WORKERS.md`)

## Live check (2026-04-05)

```bash
curl -s -o /dev/null -w "HTTP %{http_code}\n" https://api.delfictus.com/health
# HTTP 200

curl -s https://api.delfictus.com/health
# {"platform":"PRISM-4D","version":"1.1","developer":"Delfictus IO Inc.",
#  "endpoints":["/status","/targets","/sites/:pdb_id","/download/:path","/list/:prefix"]}
```

## Endpoint surface

See `CLOUDFLARE_WORKERS.md` for the authoritative endpoint list and
authentication requirements.

## Auth

Bearer token against `env.API_KEY` on the worker. Current value is
documented in `../credentials/CREDENTIAL_REGISTRY.md` (and flagged for
rotation — see `CLOUDFLARE_INVENTORY.md` §Security exposure).

## Related SOPs

- `CLOUDFLARE_WORKERS.md`
- `CLOUDFLARE_INVENTORY.md`
- `CLOUDFLARE_R2.md`
- `../credentials/CREDENTIAL_REGISTRY.md`

## History

| Date       | Change                           | By              |
|------------|----------------------------------|-----------------|
| 2026-04-05 | Initial SOP from Task D scan.    | Ididia Serfaty  |

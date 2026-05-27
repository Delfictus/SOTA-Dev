# Secret Exclusion Ledger

Source of truth: `HERMETIC_INTEGRITY_LEDGER.json`, scan version `E025-R1.0-v2.1`.

Scanner result: all git-tracked files scanned for credential patterns. Result is 0 critical, 0 open credential findings.

## Local Secret State

| Secret/state class | Tracked? | Gitignored? | Runtime source | Release verdict |
| --- | --- | --- | --- | --- |
| Cloudflare `.dev.vars` | no | yes | operator-local secret file | `SECRET_EXCLUDED` |
| Wrangler local state `.wrangler/` | no | yes | local Cloudflare dev state | `SECRET_EXCLUDED` |
| Desktop token files | no | outside source repo | operator-local secret store/env export | `SECRET_EXCLUDED` |
| Credential vault files | no | yes by pattern | operator secret store/env export | `SECRET_EXCLUDED` |

Token values are not copied into source, chat, command lines, or ledgers. Production/cloud workflows must load secrets from environment variables, Wrangler secret bindings, or the operator secret store.

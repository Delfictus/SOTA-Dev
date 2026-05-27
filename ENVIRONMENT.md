# PRISM4D Runtime Environment

This ledger documents non-secret environment configuration and external binaries used by developer or full-production workflows. Clean release and buyer/reviewer smoke tests must run without private local filesystem state.

## Environment Variables

| Name | Purpose | Required for smoke tests? | Required for full production? |
| --- | --- | --- | --- |
| `PRISM_SCRATCH_ROOT` | Scratch/output root for large generated artifacts; defaults should stay repo-local or test-local when possible. | no | yes |
| `PRISM_DATA_ROOT` | Optional external data root for production datasets restored outside the source tree. | no | yes |
| `PRISM_ROOT` | Optional override for repository root in legacy dossier/report tooling. | no | developer/report workflows only |
| `CLOUDFLARE_API_TOKEN` | Cloudflare API access. Must come from operator secret store or Wrangler secret binding. | no | cloud workflows only |
| `CF_R2_CATALOG_TOKEN` | R2 data-catalog access. Must not be committed. | no | R2 workflows only |

## External Binaries

These are runtime/toolchain dependencies and must not be removed from the pipeline contract. The hermetic scanner classifies them as `CONFIG_ENV_REQUIRED`.

| Binary | Pipeline role | Required for smoke tests? | Required for full production? |
| --- | --- | --- | --- |
| `xtb` | Quantum/semiempirical chemistry scoring path. | no | chemistry workflows |
| `antechamber` | AMBER/AM1-BCC ligand preparation. | no | chemistry workflows |
| `sqm` | AM1-BCC charge computation backend. | no | chemistry workflows |
| `obabel` | Ligand format conversion and chemistry preprocessing. | no | chemistry workflows |
| `reduce` | Protein protonation/prep path. | no | preprocessing workflows |
| `p2rank` | Baseline pocket detection. | no | validation workflows |
| `fpocket` | Baseline pocket detection. | no | validation workflows |
| `gnina` | Docking baseline. | no | docking workflows |
| `unidock` | Docking baseline. | no | docking workflows |
| `rclone` | R2/local artifact synchronization. | no | cloud/archive workflows |
| `wrangler` | Cloudflare Worker/D1/Vectorize deployment and local state. | no | cloud workflows |
| `mmseqs` | Sequence clustering/training-data preparation. | no | training workflows |

## Secret Handling

Tracked files may contain placeholders and variable names only. Real tokens, `.env`, `.dev.vars`, `.wrangler`, local credential files, and cloud state must remain untracked and gitignored.

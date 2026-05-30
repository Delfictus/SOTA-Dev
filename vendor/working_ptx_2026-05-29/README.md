# Working PTX Bundle — 2026-05-29

Authoritative PTX snapshot captured from `/home/diddy/Desktop/Prism4D-bio/target/ptx/`
after canonical-runtime hardening and sealed-image verification work on 2026-05-29.

## Why this bundle exists

The older `working_ptx_2026-04-10` snapshot is incomplete for the current
runtime path. In particular, its `nhs_amber_fused.ptx` is missing required
symbols used by the production engine bootstrap:

- `nhs_uv_pump_probe_step`
- `nhs_reduce_external_work_components`
- `nhs_velocity_second_half_step`

That stale bundle can boot a container into a false-ready state and then fail
all streams with `CUDA_ERROR_NOT_FOUND` / `named symbol not found` before
MD-only evidence serialization.

This bundle is the corrected baseline for fresh-clone and container bootstrap.

## Source

Captured from `/home/diddy/Desktop/Prism4D-bio/target/ptx/` on 2026-05-29.
`SHA256SUMS` records the exact file hashes copied into this directory.

## Usage

To seed a fresh runtime:

```bash
mkdir -p target/ptx
cp vendor/working_ptx_2026-05-29/*.ptx target/ptx/
```

If `target/ptx/` already exists and contains the current kernels, that runtime
copy should take precedence and this bundle should remain unused.

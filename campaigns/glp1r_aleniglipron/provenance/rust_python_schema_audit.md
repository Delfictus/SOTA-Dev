# Rust-to-Python Schema Audit

Campaign: `glp1r_aleniglipron`

Result: PASS

Audited Rust sources:

- `crates/prism-nhs/src/dstw_dispatch/handshake.rs`
- `crates/prism-nhs/src/dstw_dispatch/dispatcher.rs`
- `crates/prism-nhs/src/dstw_dispatch/projection.rs`
- `crates/prism-nhs/spec/option_a_variant_dispatcher.md`

Audited DSTW contract:

- `/home/diddy/Desktop/PRISM-DSTW/prism-dstw-calibration/00_registry/prism_tso_handshake.yml`

Required vectorial fields present in Rust `VariantExecutionResponse`:

- `delta_P_active`
- `delta_P_lock`
- `delta_P_ensemble`

Required Errors-in-Variables sigma fields present in Rust `VariantExecutionResponse`:

- `sigma_delta_P_active`
- `sigma_delta_P_lock`
- `sigma_delta_P_ensemble`

Forbidden legacy scalar fields are rejected by Rust request validation:

- `P_variant_divergence`
- `wasserstein_distance`
- `scalar_wasserstein`
- `variant_distance`

Notes:

- No Rust struct patch was required. The executable Rust code already mirrors the current DSTW handshake.
- The stale Option A spec language was updated from spec-only to implemented status.
- The dispatcher performs finiteness checks over all six emitted delta/sigma channels before building a response.
- Non-converged projection rows still emit values, but their EiV sigma is inflated so DSTW can down-weight them instead of silently dropping the evidence.

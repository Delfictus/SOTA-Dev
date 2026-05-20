# Option A: Thermodynamic Frustration Projection Dispatcher

**Status:** SPEC ONLY — no kernels, no compiled feature.  Authorised
2026-05-20 to lock the wire shape between PRISM-4D and DSTW for the
variant-batch path that closes the BALD active-learning loop.

**Scope:**
Read a `PRISMExecutionRequest` (mint by DSTW from a stratified seed or
max-predictive-variance acquisition), run rigid-backbone ΔV/Δq
projections of each variant against the WT thermodynamic tensors, and
emit a `PRISMExecutionResponse` carrying the explicit
`[delta_P_active, delta_P_lock, delta_P_ensemble]` per variant with
epistemic uncertainty sigmas.

The dispatcher is the variant counterpart to `dstw_export_wt` (which
handles the WT prime-run side).  The wire schemas are already frozen on
both sides; this document specifies the Rust function signatures, data
structs, and the in-engine projection contract.

---

## 1. Frozen wire schemas (DO NOT CHANGE without bumping handshake version)

### 1.1 Input — `PRISMExecutionRequest`

Mirrors `prism_dstw.orchestration.prism_handshake.PRISMExecutionRequest`:

```rust
#[derive(Debug, Clone, Deserialize)]
pub struct PRISMExecutionRequest {
    pub campaign_id: String,                          // min_length=1
    pub round_index: u32,                              // ge=0
    pub previous_round_blake3: Option<String>,         // chain link
    pub issued_at_utc: String,                         // ISO-8601
    pub variants: Vec<VariantExecutionRequest>,        // min_length=1
    pub expected_response_schema: String,              // = "dstw_prism_execution_response_v1"
}

#[derive(Debug, Clone, Deserialize)]
pub struct VariantExecutionRequest {
    pub target: String,
    pub uniprot_accession: String,
    pub variant: String,                               // e.g. "L17A"
    pub residue_number: u32,                           // ge=1
    pub wildtype_aa: String,                           // single letter
    pub mutant_aa: String,                             // single letter
    pub requested_channels: Vec<String>,               // MUST equal vectorial trio
    pub acquisition_reason: PRISMExecutionAcquisition, // enum
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub enum PRISMExecutionAcquisition {
    #[serde(rename = "stratified_seed")] StratifiedSeed,
    #[serde(rename = "max_predictive_variance")] MaxPredictiveVariance,
    #[serde(rename = "operator_override")] OperatorOverride,
}
```

**Validation on ingestion (mirrors DSTW-side Pydantic):**

* `expected_response_schema == "dstw_prism_execution_response_v1"`
* every `requested_channels == {"delta_P_active", "delta_P_lock", "delta_P_ensemble"}`
* no duplicate `(target, variant)` tuples
* no forbidden scalar tokens in any free-text field (`P_variant_divergence`, `wasserstein_distance`, `scalar_wasserstein`, `variant_distance`)
* `previous_round_blake3 == blake3_hex(prior round's response body)` when round_index > 0

### 1.2 Output — `PRISMExecutionResponse`

Mirrors `prism_dstw.orchestration.prism_handshake.PRISMExecutionResponse`:

```rust
#[derive(Debug, Clone, Serialize)]
pub struct PRISMExecutionResponse {
    pub campaign_id: String,
    pub round_index: u32,
    pub request_blake3: String,                        // blake3(canonical_json(request))
    pub completed_at_utc: String,                      // ISO-8601
    pub variants: Vec<VariantExecutionResponse>,
    pub response_schema: String,                       // = "dstw_prism_execution_response_v1"
}

#[derive(Debug, Clone, Serialize)]
pub struct VariantExecutionResponse {
    pub target: String,
    pub variant: String,
    pub delta_P_active: f64,
    pub delta_P_lock: f64,
    pub delta_P_ensemble: f64,
    pub sigma_delta_P_active: f64,                     // ge=0
    pub sigma_delta_P_lock: f64,                       // ge=0
    pub sigma_delta_P_ensemble: f64,                   // ge=0
    pub prism_run_id: String,
    pub converged: bool,
}
```

**Invariants enforced at serialisation:**

* every `delta_*` and `sigma_*` finite (no NaN/Inf).  Non-finite values
  abort the response build with a hard error (DSTW would refuse the
  response).
* `sigma_*` strictly non-negative.

---

## 2. Engine-side projection contract

### 2.1 Inputs the dispatcher needs

```rust
pub struct WTTensorPack {
    /// Per-residue physical projections from the WT prime run.  Sourced
    /// from the same file emitted by `dstw_export_wt`; loaded once per
    /// campaign and reused across rounds.
    pub te_out: Vec<f32>,                              // [n_residues]
    pub te_in: Vec<f32>,                               // [n_residues]
    pub delta_hc: Vec<f32>,                            // [n_residues]
    pub sigma_hydration_sq: Vec<f32>,                  // [n_residues]
    /// Engine-internal channels.  These are the projections Option A
    /// will deltad over:
    pub p_active_wt: Vec<f32>,                         // [n_residues]
    pub p_lock_wt: Vec<f32>,                           // [n_residues]
    pub p_ensemble_wt: Vec<f32>,                       // [n_residues]
    /// Per-residue partial charge q_WT and per-residue per-atom
    /// volume V_WT (Bondi or Voronoi) -- the substrate for the
    /// rigid-backbone Δq / ΔV substitution.
    pub q_wt: Vec<f32>,                                // [n_residues]
    pub v_wt: Vec<f32>,                                // [n_residues]
}

pub struct VariantBatch {
    pub campaign_id: String,
    pub round_index: u32,
    pub request_blake3: String,                        // recomputed on the wire
    pub variants: Vec<VariantPoint>,
}

pub struct VariantPoint {
    pub key: VariantKey,                               // (target, variant) tuple
    pub residue_number: u32,
    pub wildtype_aa: AminoAcid,
    pub mutant_aa: AminoAcid,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VariantKey { pub target: String, pub variant: String }
```

### 2.2 The rigid-backbone Δq / ΔV projection

For each variant, the rigid-backbone substitution swaps the side-chain
partial-charge profile (Δq) and side-chain volume (ΔV) at the named
residue without re-equilibrating the backbone.  The dispatcher then
projects these perturbations onto the three WT thermodynamic channels:

```
ΔP_active(variant) = sum_j  K_active_ij  * [ alpha_q * Δq_i + alpha_v * ΔV_i ]
ΔP_lock(variant)   = sum_j  K_lock_ij    * [ alpha_q * Δq_i + alpha_v * ΔV_i ]
ΔP_ensemble(var)   = sum_j  K_ens_ij     * [ alpha_q * Δq_i + alpha_v * ΔV_i ]
```

where:

* `i` is the variant's residue index.
* `K_active_ij`, `K_lock_ij`, `K_ens_ij` are channel-specific
  propagation kernels derived from the WT TE / hysteresis / hydration
  tensors.  These are the engine-internal "frustration projection"
  matrices the operator named in Option A.  Spec captures their
  existence; the kernel-shape sweep is a follow-up authorised by a
  separate gate.
* `alpha_q`, `alpha_v` are configurable mixing coefficients.

### 2.3 Epistemic uncertainty

Each delta carries a sigma derived from:

```
sigma_delta_P_active^2  = trace( J_active * Cov(WT_tensors) * J_active^T )
                       + |sum_j K_active_ij|^2 * model_residual_variance_active
sigma_delta_P_lock^2    = ...   (mirror)
sigma_delta_P_ensemble^2 = ...  (mirror)
```

J is the Jacobian of the projection w.r.t. the WT tensors.  The
`Cov(WT_tensors)` term is the prime-run replicate-spread covariance.
The `model_residual_variance_*` term is the engine's reported residual
from the WT prime run.

Both terms are required so the DSTW Errors-in-Variables (EiV) machinery
receives the FULL uncertainty budget, not just the model term.

### 2.4 Convergence flag

`converged: bool` per variant is True iff:

* the rigid-backbone substitution kept the local backbone within an
  RMSD threshold (default 0.5 Å) of the WT anchor at the variant's
  residue, AND
* the projection norm is finite, AND
* the Jacobian condition number is below a configurable ceiling
  (default 1e6).

Variants with `converged=False` still emit values (so DSTW can audit
the failure mode), but the EiV sigma is multiplied by a documented
penalty factor so the GLM down-weights them naturally.

---

## 3. Dispatcher entry point

The dispatcher is a NEW Rust binary `dstw_dispatch_variants.rs` (sibling
of `dstw_export_wt.rs`) at
`crates/prism-nhs/src/bin/dstw_dispatch_variants.rs`.

```rust
pub fn dispatch_variant_batch(
    request: PRISMExecutionRequest,
    wt_pack: &WTTensorPack,
    config: &VariantDispatchConfig,
) -> Result<PRISMExecutionResponse, DispatchError>;
```

Where:

```rust
pub struct VariantDispatchConfig {
    pub prism_run_id: String,
    pub alpha_q: f32,
    pub alpha_v: f32,
    pub model_residual_variance_active: f32,
    pub model_residual_variance_lock: f32,
    pub model_residual_variance_ensemble: f32,
    pub backbone_rmsd_ceiling_angstrom: f32,   // default 0.5
    pub jacobian_condition_ceiling: f32,        // default 1e6
    pub nonconverged_sigma_penalty: f32,        // default 4.0
}

pub enum DispatchError {
    SchemaMismatch(String),
    UnknownVariant { key: VariantKey, reason: String },
    NonFinite { key: VariantKey, channel: &'static str },
    ChainHashMismatch { declared: String, observed: String },
    Internal(anyhow::Error),
}
```

### 3.1 CLI signature

```
dstw-dispatch-variants \
    --request <path/to/BALD_Round_NNN_Request.json> \
    --wt-tensor-pack <path/to/wt_physics_payload.parquet> \
    --prism-run-id <run identifier> \
    --out-json <path/to/BALD_Round_NNN_Response.json> \
    --alpha-q <float, default 1.0> \
    --alpha-v <float, default 1.0> \
    --residue-name-table <path/to/topology.json>   # for sidechain Δq / ΔV lookup
```

### 3.2 Determinism

For a given (request, wt_pack, config), the response MUST be bit-exact
reproducible.  The dispatcher seeds any internal RNG with
`blake3(canonical_json(request)) % 2^64` so the EiV sigma sampling
(if used to add observation noise on top of the projection) is
deterministic across re-runs.

---

## 4. What's IN scope for the spec

* Wire schemas (both directions).
* Rust function signatures for the dispatcher and the projection.
* Validation contract (schema, finiteness, chain link, non-NaN).
* Convergence flag + sigma penalty contract.
* CLI signature for the dispatcher binary.

## 5. What's OUT of scope (deferred to a future gate)

* CUDA kernels for the `K_active`, `K_lock`, `K_ensemble` propagation
  matrices.
* Choice of partial-charge force field (AMBER ff14SB vs ff19SB vs
  custom — needs operator gate).
* Adaptive `alpha_q`, `alpha_v` tuning (currently configuration knobs).
* Batched GPU execution for >1k variants in a single round (the BALD
  loop currently caps at 100/round via `max_per_round`).
* Multi-chain dispatch for oligomeric receptors (single-chain
  rigid-backbone substitution only at this gate).

## 6. Air-gap status after this spec

| component | side | status |
|---|---|---|
| WT prime-run schema | DSTW (Pydantic) | done |
| WT prime-run schema | PRISM-4D (Rust serde) | done (`dstw_export_wt.rs`) |
| WT prime-run exporter | PRISM-4D | done, compiles, tested, validates against DSTW schema |
| Variant-request schema | DSTW (Pydantic) | done |
| Variant-request schema | PRISM-4D (Rust serde) | **spec-only, this document** |
| Variant-response schema | DSTW (Pydantic) | done |
| Variant-response schema | PRISM-4D (Rust serde) | **spec-only, this document** |
| Variant-dispatch binary | PRISM-4D | **not yet implemented** (next gate) |
| Δq / ΔV side-chain table | PRISM-4D | not yet built |
| K_active / K_lock / K_ensemble kernels | PRISM-4D (CUDA) | not yet built |
| BALD round-trip | DSTW | done (ingest_response wired) |

Once the operator authorises the next gate, the implementation plan is:

1. Translate this spec's Rust structs into `crates/prism-nhs/src/dstw_dispatch.rs`.
2. Implement the rigid-backbone Δq / ΔV substitution helpers (CPU-only first; AMBER ff14SB charge tables).
3. Implement the three projection kernels (CPU reference + CUDA).
4. Wire the `dstw-dispatch-variants` binary with the CLI signature above.
5. Round-trip test: mint a DSTW seed manifest → run the dispatcher → ingest the response back into DSTW's `BayesianActiveLearner` → confirm posterior updates without raising the EiV / Tobit / monotonic guards.

The deliverable for THIS gate is the spec and the WT exporter only.
No engine execution.

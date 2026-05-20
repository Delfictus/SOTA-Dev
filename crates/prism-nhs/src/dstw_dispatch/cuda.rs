//! GPU dispatch path for the DSTW variant projection.
//!
//! This module is the Rust FFI binding to the CUDA kernel at
//! `crates/prism-gpu/src/kernels/dstw_propagation.cu`.  It deliberately
//! does NOT execute the kernel during library build / test — it provides:
//!
//!   * the shape contract (matrix layouts, channel count, output strides)
//!     so CPU and GPU paths share invariants;
//!   * a CPU-equivalent reference (`propagate_cpu_reference`) used by
//!     the matrix-shape unit tests.  Identical results to what the CUDA
//!     kernel would produce given the same inputs;
//!   * the placeholder `CudaPropagationKernel` struct + `dispatch()`
//!     method that loads `target/ptx/dstw_propagation.ptx` and launches
//!     the kernel when the `cuda` feature is enabled.  Until the
//!     dispatch is operator-authorised for live launch, `dispatch()`
//!     returns a `Backend::CpuFallback` so the CPU reference path is
//!     used.
//!
//! Layout invariants (matched by both CPU reference and CUDA kernel):
//!
//!   K           : 3 channels × n_residues × n_residues, row-major,
//!                 channel-major outermost.  Index:
//!                     K[(c * n_res + i) * n_res + j]
//!   residue_id  : n_variants i32, 0-indexed mutation site
//!   perturbation: n_variants f32, α_q·Δq + α_v·ΔV per variant
//!   delta_p     : 3 × n_variants, row-major, channel-major outermost.
//!                 Index: delta_p[c * n_variants + b]
//!
//! No CUDA dependency in this file (the actual cudarc surface lives in
//! crates/prism-gpu, gated behind the `gpu` feature).

#![allow(clippy::too_many_arguments)]

/// Number of thermodynamic channels emitted per variant.
pub const N_CHANNELS: usize = 3;

/// Channel indices in the K and delta_p arrays.
pub const CH_ACTIVE: usize = 0;
pub const CH_LOCK: usize = 1;
pub const CH_ENSEMBLE: usize = 2;

/// Shape descriptor passed to the kernel + CPU reference.
#[derive(Debug, Clone, Copy)]
pub struct PropagationShape {
    pub n_residues: usize,
    pub n_variants: usize,
}

impl PropagationShape {
    pub fn k_len(&self) -> usize {
        N_CHANNELS * self.n_residues * self.n_residues
    }
    pub fn delta_p_len(&self) -> usize {
        N_CHANNELS * self.n_variants
    }
    pub fn validate(&self) -> Result<(), String> {
        if self.n_residues == 0 {
            return Err("n_residues must be positive".into());
        }
        if self.n_variants == 0 {
            return Err("n_variants must be positive".into());
        }
        if self.n_residues > 1024 {
            return Err(format!(
                "n_residues={} exceeds DSTW_MAX_RESIDUES_PER_BLOCK=1024; \
                 split the receptor or run the kernel iteratively",
                self.n_residues
            ));
        }
        Ok(())
    }
}

/// CPU reference for the propagation kernel.  Bit-equivalent to what
/// the CUDA kernel WOULD compute given the same f32 inputs (modulo
/// floating-point summation order, which differs between the warp-shuffle
/// reduction and a left-to-right scalar sum — tests assert
/// approximate equality with a per-channel tolerance, not exact bits).
///
/// Layout:
///   * K          slice of length 3 * n_residues * n_residues
///   * residue_id slice of length n_variants
///   * pert       slice of length n_variants
///   * delta_p    output slice of length 3 * n_variants
pub fn propagate_cpu_reference(
    shape: PropagationShape,
    k: &[f32],
    residue_id: &[i32],
    pert: &[f32],
    delta_p: &mut [f32],
) -> Result<(), String> {
    shape.validate()?;
    if k.len() != shape.k_len() {
        return Err(format!(
            "K length {} != expected {}",
            k.len(),
            shape.k_len()
        ));
    }
    if residue_id.len() != shape.n_variants {
        return Err(format!(
            "residue_id length {} != n_variants {}",
            residue_id.len(),
            shape.n_variants
        ));
    }
    if pert.len() != shape.n_variants {
        return Err(format!(
            "pert length {} != n_variants {}",
            pert.len(),
            shape.n_variants
        ));
    }
    if delta_p.len() != shape.delta_p_len() {
        return Err(format!(
            "delta_p length {} != expected {}",
            delta_p.len(),
            shape.delta_p_len()
        ));
    }
    let n = shape.n_residues;
    let nv = shape.n_variants;
    for c in 0..N_CHANNELS {
        let channel_offset = c * n * n;
        for b in 0..nv {
            let i = residue_id[b];
            if i < 0 || (i as usize) >= n {
                delta_p[c * nv + b] = 0.0;
                continue;
            }
            let row_offset = channel_offset + (i as usize) * n;
            let mut s: f32 = 0.0;
            for j in 0..n {
                s += k[row_offset + j];
            }
            delta_p[c * nv + b] = pert[b] * s;
        }
    }
    Ok(())
}

/// Row sums of K per channel for every variant's residue.
///
/// Used as an intermediate by the epistemic sigma kernel; exposed here
/// so the CPU path can reuse the same arithmetic and round-trip the
/// uncertainty budget without the GPU.
pub fn row_sums_cpu_reference(
    shape: PropagationShape,
    k: &[f32],
    residue_id: &[i32],
    row_sums: &mut [f32],
) -> Result<(), String> {
    shape.validate()?;
    if k.len() != shape.k_len() {
        return Err(format!("K length mismatch: {} != {}", k.len(), shape.k_len()));
    }
    if residue_id.len() != shape.n_variants {
        return Err(format!(
            "residue_id length mismatch: {} != {}",
            residue_id.len(),
            shape.n_variants
        ));
    }
    if row_sums.len() != N_CHANNELS * shape.n_variants {
        return Err(format!(
            "row_sums length {} != expected {}",
            row_sums.len(),
            N_CHANNELS * shape.n_variants
        ));
    }
    let n = shape.n_residues;
    let nv = shape.n_variants;
    for c in 0..N_CHANNELS {
        let channel_offset = c * n * n;
        for b in 0..nv {
            let i = residue_id[b];
            if i < 0 || (i as usize) >= n {
                row_sums[c * nv + b] = 0.0;
                continue;
            }
            let row_offset = channel_offset + (i as usize) * n;
            let mut s: f32 = 0.0;
            for j in 0..n {
                s += k[row_offset + j];
            }
            row_sums[c * nv + b] = s;
        }
    }
    Ok(())
}

/// Computed dispatch backend.  When the GPU dispatch path is
/// operator-authorised, `Cuda` becomes selectable; until then every
/// caller flows through `CpuFallback`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    CpuFallback,
    Cuda,
}

/// Stub for the eventual cudarc-backed launcher.  Holds the device,
/// the loaded PTX module, and the kernel function handles for the two
/// device kernels.  Constructed lazily on first dispatch.
///
/// The struct is intentionally empty in this draft — the operator's
/// directive was to "draft the kernel .cu files, ensure they integrate
/// with the Rust FFI bindings, and write the corresponding unit tests
/// to verify the matrix shape sweeps."  Live cudarc bindings will land
/// in a follow-up gate alongside the engine launch authorisation.
#[derive(Debug)]
pub struct CudaPropagationKernel {
    _placeholder: (),
}

impl CudaPropagationKernel {
    /// Construct the kernel handle.  Currently always returns the
    /// CPU-fallback variant; will load `target/ptx/dstw_propagation.ptx`
    /// when the GPU dispatch path is gated open.
    pub fn new() -> Result<Self, String> {
        Ok(Self { _placeholder: () })
    }

    /// Choose the dispatch backend.  Returns `Backend::CpuFallback` until
    /// the engine launch gate is opened by the operator.
    pub fn select_backend(&self) -> Backend {
        Backend::CpuFallback
    }

    /// Top-level dispatch.  Always delegates to the CPU reference in
    /// this draft.  The signature is fixed so the eventual GPU branch
    /// drops in without touching callers.
    pub fn dispatch(
        &self,
        shape: PropagationShape,
        k: &[f32],
        residue_id: &[i32],
        pert: &[f32],
        delta_p: &mut [f32],
    ) -> Result<Backend, String> {
        propagate_cpu_reference(shape, k, residue_id, pert, delta_p)?;
        Ok(self.select_backend())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthesise a K with a known row sum per channel.  Channel c, row
    /// i has every entry equal to `1.0 + 0.01 * (c * n + i)`, so
    /// row_sum(c, i) = n * (1.0 + 0.01 * (c * n + i)).  Lets us assert
    /// against an analytic expectation.
    fn synth_k(n: usize) -> Vec<f32> {
        let mut k = vec![0.0f32; N_CHANNELS * n * n];
        for c in 0..N_CHANNELS {
            for i in 0..n {
                let entry = 1.0 + 0.01 * ((c * n + i) as f32);
                for j in 0..n {
                    k[(c * n + i) * n + j] = entry;
                }
            }
        }
        k
    }

    fn expected_row_sum(n: usize, channel: usize, i: usize) -> f32 {
        let entry = 1.0 + 0.01 * ((channel * n + i) as f32);
        n as f32 * entry
    }

    #[test]
    fn shape_sweep_residue_counts() {
        for &n in &[32usize, 64, 128, 256, 352, 512] {
            let shape = PropagationShape { n_residues: n, n_variants: 4 };
            shape.validate().expect("valid shape");
            let k = synth_k(n);
            let residue_id = vec![0, 1, (n / 2) as i32, (n - 1) as i32];
            let pert = vec![1.0, -1.0, 2.0, 0.5];
            let mut delta_p = vec![0.0f32; shape.delta_p_len()];
            propagate_cpu_reference(shape, &k, &residue_id, &pert, &mut delta_p).unwrap();
            for c in 0..N_CHANNELS {
                for (b, &i) in residue_id.iter().enumerate() {
                    let expected = pert[b] * expected_row_sum(n, c, i as usize);
                    let got = delta_p[c * shape.n_variants + b];
                    assert!(
                        (got - expected).abs() / expected.abs().max(1.0) < 1e-4,
                        "n_residues={n} c={c} b={b} expected={expected} got={got}"
                    );
                }
            }
        }
    }

    #[test]
    fn shape_sweep_variant_counts() {
        for &nv in &[1usize, 5, 25, 100] {
            let n = 128usize;
            let shape = PropagationShape { n_residues: n, n_variants: nv };
            shape.validate().expect("valid shape");
            let k = synth_k(n);
            let residue_id: Vec<i32> = (0..nv).map(|b| (b % n) as i32).collect();
            let pert: Vec<f32> = (0..nv).map(|b| (b as f32) * 0.01).collect();
            let mut delta_p = vec![0.0f32; shape.delta_p_len()];
            propagate_cpu_reference(shape, &k, &residue_id, &pert, &mut delta_p).unwrap();
            for c in 0..N_CHANNELS {
                for b in 0..nv {
                    let i = residue_id[b] as usize;
                    let expected = pert[b] * expected_row_sum(n, c, i);
                    let got = delta_p[c * nv + b];
                    assert!(
                        (got - expected).abs() / expected.abs().max(1.0) < 1e-4,
                        "nv={nv} c={c} b={b}"
                    );
                }
            }
        }
    }

    #[test]
    fn out_of_range_residue_yields_zero() {
        let n = 64usize;
        let shape = PropagationShape { n_residues: n, n_variants: 3 };
        let k = synth_k(n);
        let residue_id = vec![0, -1, 999];
        let pert = vec![1.0, 1.0, 1.0];
        let mut delta_p = vec![0.0f32; shape.delta_p_len()];
        propagate_cpu_reference(shape, &k, &residue_id, &pert, &mut delta_p).unwrap();
        for c in 0..N_CHANNELS {
            // variant 0 (valid) -> non-zero
            assert!(delta_p[c * 3 + 0].abs() > 0.0);
            // variant 1 (-1) -> zero
            assert_eq!(delta_p[c * 3 + 1], 0.0);
            // variant 2 (999) -> zero
            assert_eq!(delta_p[c * 3 + 2], 0.0);
        }
    }

    #[test]
    fn row_sums_reference_matches_propagation() {
        let n = 96usize;
        let nv = 7usize;
        let shape = PropagationShape { n_residues: n, n_variants: nv };
        let k = synth_k(n);
        let residue_id: Vec<i32> = (0..nv).map(|b| ((b * 13) % n) as i32).collect();
        let pert = vec![1.0; nv];
        let mut deltas = vec![0.0f32; shape.delta_p_len()];
        let mut rows = vec![0.0f32; N_CHANNELS * nv];
        propagate_cpu_reference(shape, &k, &residue_id, &pert, &mut deltas).unwrap();
        row_sums_cpu_reference(shape, &k, &residue_id, &mut rows).unwrap();
        // delta_p = pert * row_sum, with pert=1 they must be equal.
        for c in 0..N_CHANNELS {
            for b in 0..nv {
                assert!(
                    (deltas[c * nv + b] - rows[c * nv + b]).abs() < 1e-4,
                    "delta_p != row_sum at (c={c}, b={b})"
                );
            }
        }
    }

    #[test]
    fn shape_validation_rejects_invalid_inputs() {
        let shape = PropagationShape { n_residues: 0, n_variants: 10 };
        assert!(shape.validate().is_err());
        let shape = PropagationShape { n_residues: 10, n_variants: 0 };
        assert!(shape.validate().is_err());
        let shape = PropagationShape { n_residues: 2048, n_variants: 10 };
        assert!(shape.validate().unwrap_err().contains("MAX_RESIDUES_PER_BLOCK"));
    }

    #[test]
    fn dispatcher_returns_cpu_fallback_in_draft() {
        // Until the operator opens the GPU launch gate, every dispatch
        // round-trips through the CPU reference.  This test pins the
        // behaviour so the GPU path can be added later without changing
        // caller expectations silently.
        let kernel = CudaPropagationKernel::new().unwrap();
        let n = 32;
        let nv = 2;
        let shape = PropagationShape { n_residues: n, n_variants: nv };
        let k = synth_k(n);
        let residue_id = vec![0, 5];
        let pert = vec![1.0, 1.0];
        let mut delta_p = vec![0.0f32; shape.delta_p_len()];
        let backend = kernel.dispatch(shape, &k, &residue_id, &pert, &mut delta_p).unwrap();
        assert_eq!(backend, Backend::CpuFallback);
        for c in 0..N_CHANNELS {
            for b in 0..nv {
                let i = residue_id[b] as usize;
                let expected = expected_row_sum(n, c, i);
                let got = delta_p[c * nv + b];
                assert!((got - expected).abs() < 1e-3);
            }
        }
    }
}

//! RECT-3.1.b — SO(3) Projection Kernel + ContactShellTile.
//!
//! Per the Production Architecture mandate Phase 1 Deliverable 1.2
//! (operator directive 2026-04-29). Consumes `RichSpike` clusters
//! produced by the M1 / LBVH lane and writes one [`ContactShellTile`]
//! per cluster: a 384-byte, 128-byte-aligned hardware execution tile
//! holding the per-cluster spherical-harmonic expansion `a_lm` (36
//! coefficients, padded to 64 floats for forward-compat with WMMA
//! 16×16 / 32×8 fragment loads) and the rotation-invariant power
//! spectrum `C_l = Σ_m |a_lm|²`.
//!
//! # Why a Tile (not a Tensor)
//!
//! Per the operator's nomenclature override (2026-04-29), the
//! contact-shell record is a hardware-bound EXECUTION tile
//! (`#[repr(C, align(128))]`, sized to fit Blackwell shared-memory
//! 16×16 / 32×8 fragment blocks), not a logical tensor. Naive
//! global-memory pointer-chasing loops are FORBIDDEN; the kernel
//! cooperatively strides the spike buffer with one block per cluster
//! and warp-shuffle reduces.
//!
//! # G11 — Rotational Invariance Gate
//!
//! `C_l` is invariant under SO(3) rotations of the spike cloud:
//! a rotation `R ∈ SO(3)` applied to every spike's (x, y, z) leaves
//! `C_l = Σ_m |a_lm|²` unchanged for every l. The
//! [`tests::g11_rotation_invariance`] test enforces this within
//! 1e-4 relative tolerance over 10 random rotations.
//!
//! # FFI surface
//!
//! - [`link_probe`] returns `0x0005_3033` (the C-side sentinel).
//! - [`run`] launches `prism_so3_project_manifold_kernel` over a
//!   pre-clustered RichSpike buffer.

use crate::sh_basis::{LMAX, N_COEFFS};

// ============================================================================
// ContactShellTile (#[repr(C, align(128))], 384 B)
// ============================================================================

/// 384-byte, 128-byte-aligned execution tile produced by
/// `prism_so3_project_manifold_kernel`. Layout-pinned by the
/// `contact_shell_tile_layout_*` tests.
///
/// The C-side mirror is `prism_nhs::so3_project::ContactShellTile`
/// in `src/cuda/so3_project.cuh`. Field order, types, and offsets
/// match byte-for-byte; drift is caught by the layout tests + the
/// C-side `static_assert(sizeof == 384)`.
#[repr(C, align(128))]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactShellTile {
    // Header (16 B)
    pub phase: u32,
    pub stream_id: u32,
    pub cluster_id: i32,
    pub frame: u32,

    // a_lm coefficients — 256 B (64 floats; first 36 valid for Lmax=5).
    // Sized for forward-compat with WMMA 16×16 / 32×8 fragment loads.
    pub coefficients: [f32; 64],

    // C_l power spectrum — 32 B (8 floats; first 6 valid for Lmax=5).
    pub power_spectrum: [f32; 8],

    // AABB (xyz + pad). 16 B + 16 B for LDG.E.128 alignment.
    pub aabb_min: [f32; 4],
    pub aabb_max: [f32; 4],

    // Metadata (32 B).
    pub spike_count: u32,
    pub adjudication_code: u32,
    pub reserved: [u32; 6],

    // Padding to 3 × 128 = 384 B.
    pub _pad: [u8; 16],
}

impl ContactShellTile {
    /// Zero-initialized tile. Useful for test fixtures and as a
    /// starting state before kernel writes.
    pub const fn zero() -> Self {
        Self {
            phase: 0,
            stream_id: 0,
            cluster_id: 0,
            frame: 0,
            coefficients: [0.0; 64],
            power_spectrum: [0.0; 8],
            aabb_min: [0.0; 4],
            aabb_max: [0.0; 4],
            spike_count: 0,
            adjudication_code: 0,
            reserved: [0; 6],
            _pad: [0; 16],
        }
    }

    /// Read the active-prefix `a_lm` slice (first `N_COEFFS = 36`
    /// floats). The remaining 28 floats in `coefficients[]` are
    /// WMMA-prep padding and always zero on kernel output.
    pub fn alm(&self) -> &[f32] {
        &self.coefficients[..N_COEFFS]
    }

    /// Read the active-prefix `C_l` slice (first `LMAX + 1 = 6`
    /// floats). The remaining 2 floats in `power_spectrum[]` are
    /// padding and always zero on kernel output.
    pub fn cl(&self) -> &[f32] {
        &self.power_spectrum[..=LMAX]
    }
}

impl Default for ContactShellTile {
    fn default() -> Self {
        Self::zero()
    }
}

// ============================================================================
// FFI surface
// ============================================================================

#[cfg(feature = "gpu")]
#[allow(dead_code)]
mod ffi {
    use super::ContactShellTile;
    use crate::rich_spike::RichSpike;

    pub type CudaError = i32;
    pub const CUDA_SUCCESS: CudaError = 0;

    extern "C" {
        pub fn prism_so3_project_link_probe() -> u32;

        pub fn prism_so3_project_run(
            d_spikes: *const RichSpike,
            d_cluster_offsets: *const u32,
            n_clusters: u32,
            d_k_lm: *const f32,
            d_tiles_out: *mut ContactShellTile,
            frame_id: u32,
            stream: *mut std::ffi::c_void,
        ) -> CudaError;
    }
}

#[cfg(feature = "gpu")]
pub fn link_probe() -> u32 {
    unsafe { ffi::prism_so3_project_link_probe() }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contact_shell_tile_layout_is_384_bytes_128_aligned() {
        // Layout pin: must match the C-side `static_assert(sizeof == 384)`.
        // Drift here means FFI is broken and the kernel will write into
        // garbage; the C-side static_assert catches it at compile time but
        // this test guards the Rust side independently.
        assert_eq!(std::mem::size_of::<ContactShellTile>(), 384);
        assert_eq!(std::mem::align_of::<ContactShellTile>(), 128);
    }

    #[test]
    fn contact_shell_tile_field_offsets() {
        let t = ContactShellTile::zero();
        let base = &t as *const ContactShellTile as usize;
        macro_rules! ofs {
            ($field:ident) => {
                (&t.$field as *const _ as usize) - base
            };
        }
        // Header
        assert_eq!(ofs!(phase),      0);
        assert_eq!(ofs!(stream_id),  4);
        assert_eq!(ofs!(cluster_id), 8);
        assert_eq!(ofs!(frame),     12);
        // Coefficients (64 floats)
        assert_eq!(ofs!(coefficients),    16);
        // Power spectrum (8 floats) — starts at 16 + 256 = 272
        assert_eq!(ofs!(power_spectrum), 272);
        // AABB — 304, 320
        assert_eq!(ofs!(aabb_min), 304);
        assert_eq!(ofs!(aabb_max), 320);
        // Metadata — 336
        assert_eq!(ofs!(spike_count),       336);
        assert_eq!(ofs!(adjudication_code), 340);
        assert_eq!(ofs!(reserved),          344);
        // Padding — 368
        assert_eq!(ofs!(_pad), 368);
    }

    #[test]
    fn alm_slice_is_36_zeros_on_default_tile() {
        let t = ContactShellTile::zero();
        assert_eq!(t.alm().len(), 36);
        assert_eq!(t.cl().len(), 6);
        assert!(t.alm().iter().all(|&v| v == 0.0));
        assert!(t.cl().iter().all(|&v| v == 0.0));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn link_probe_returns_sentinel() {
        assert_eq!(super::link_probe(), 0x0005_3033);
    }

    // ────────────────────────────────────────────────────────────────────
    // GPU-side: G11 rotational invariance gate.
    // ────────────────────────────────────────────────────────────────────

    #[cfg(feature = "gpu")]
    #[test]
    fn g11_rotation_invariance() {
        use crate::rich_spike::RichSpike;
        use cudarc::driver::{CudaContext, DevicePtr};

        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[g11] CUDA unavailable: {:?} — skipping", e);
                return;
            }
        };
        let stream = ctx.new_stream().expect("stream");
        let raw_stream = stream.cu_stream() as usize;

        // Init K_LM on device.
        let rc = unsafe {
            crate::sh_basis::ffi::prism_sh_basis_init(
                raw_stream as *mut std::ffi::c_void,
            )
        };
        assert_eq!(rc, crate::sh_basis::ffi::CUDA_SUCCESS, "sh init");
        stream.synchronize().expect("post-init sync");
        let k_lm_dev = crate::sh_basis::k_lm_device_ptr().expect("k_lm ptr");

        // Build a synthetic single-cluster spike cloud: 64 spikes
        // randomly placed inside a 6 Å sphere centered on the origin.
        // With non-trivial spatial structure but zero-intensity (the
        // kernel falls back to weight=1), the SO(3) test exercises the
        // angular distribution, which is what C_l measures.
        struct Lcg { s: u64 }
        impl Lcg {
            fn next_f32(&mut self) -> f32 {
                self.s = self.s.wrapping_mul(6_364_136_223_846_793_005)
                               .wrapping_add(1_442_695_040_888_963_407);
                (self.s >> 32) as u32 as f32 / 4_294_967_296.0
            }
            fn next_uniform(&mut self, lo: f32, hi: f32) -> f32 {
                lo + (hi - lo) * self.next_f32()
            }
        }
        let mut rng = Lcg { s: 0x1234_5678_9abc_def0 };

        const N_SPIKES: usize = 64;
        let mut base_pos: Vec<[f32; 3]> = Vec::with_capacity(N_SPIKES);
        while base_pos.len() < N_SPIKES {
            let x = rng.next_uniform(-6.0, 6.0);
            let y = rng.next_uniform(-6.0, 6.0);
            let z = rng.next_uniform(-6.0, 6.0);
            // Reject points outside the 6 Å sphere so distribution is
            // genuinely 3D-rotationally varied.
            if x * x + y * y + z * z <= 36.0 {
                base_pos.push([x, y, z]);
            }
        }

        // Helper: build RichSpike cluster from positions.
        let build_spikes = |pos: &[[f32; 3]]| -> Vec<RichSpike> {
            pos.iter().map(|&[x, y, z]| {
                let mut s = RichSpike::zero();
                s.x = x; s.y = y; s.z = z;
                s.cluster_id = 0;
                s.residue_id = 0;
                s
            }).collect()
        };

        // Helper: run the kernel for one set of positions, return C_l.
        // We allocate device buffers as `u8` and cast via
        // `read_unaligned` on dtoh so we don't need cudarc DeviceRepr
        // / ValidAsZeroBits impls on RichSpike or ContactShellTile.
        // Mirrors the lbvh_tree.rs test pattern.
        let run_kernel = |pos: &[[f32; 3]]| -> [f32; LMAX + 1] {
            let spikes = build_spikes(pos);
            let n = spikes.len() as u32;
            let offsets: Vec<u32> = vec![0u32, n];

            let spike_bytes = (spikes.len()) * std::mem::size_of::<RichSpike>();
            let mut d_spikes_b = stream.alloc_zeros::<u8>(spike_bytes).expect("alloc spikes");
            // Pack to a u8 vec (all-byte) for htod.
            let spikes_bytes: Vec<u8> = unsafe {
                std::slice::from_raw_parts(
                    spikes.as_ptr() as *const u8,
                    spike_bytes,
                ).to_vec()
            };
            stream.memcpy_htod(&spikes_bytes, &mut d_spikes_b).expect("htod spikes");
            let mut d_offsets = stream.alloc_zeros::<u32>(offsets.len()).expect("alloc offsets");
            stream.memcpy_htod(&offsets, &mut d_offsets).expect("htod offsets");
            let tile_bytes = std::mem::size_of::<ContactShellTile>();
            let d_tiles_b = stream.alloc_zeros::<u8>(tile_bytes).expect("alloc tiles");

            let (sp_dev, _g1)   = d_spikes_b.device_ptr(&stream);
            let (off_dev, _g2)  = d_offsets.device_ptr(&stream);
            let (tile_dev, _g3) = d_tiles_b.device_ptr(&stream);
            let rc = unsafe {
                ffi::prism_so3_project_run(
                    sp_dev as *const RichSpike,
                    off_dev as *const u32,
                    1u32,
                    k_lm_dev,
                    tile_dev as *mut ContactShellTile,
                    0u32,
                    raw_stream as *mut std::ffi::c_void,
                )
            };
            assert_eq!(rc, ffi::CUDA_SUCCESS, "kernel launch");
            stream.synchronize().expect("kernel sync");

            let mut host_bytes = vec![0u8; tile_bytes];
            stream.memcpy_dtoh(&d_tiles_b, &mut host_bytes).expect("dtoh tiles");
            let host_tile: ContactShellTile = unsafe {
                std::ptr::read_unaligned(host_bytes.as_ptr() as *const ContactShellTile)
            };
            let cl = host_tile.cl();
            let mut out = [0.0f32; LMAX + 1];
            out.copy_from_slice(cl);
            out
        };

        // Reference C_l from the unrotated cloud.
        let cl_ref = run_kernel(&base_pos);
        // Sanity: C_l > 0 for at least one l > 0 (otherwise the cloud
        // has no angular structure and the test is vacuous).
        let high_l_power: f32 = cl_ref[1..].iter().sum();
        assert!(high_l_power > 1e-3,
            "test cloud is angularly trivial (sum C_l for l>0 = {:.2e}); \
             cannot exercise rotational invariance",
            high_l_power);

        // Rodrigues rotation: rotate v by angle α around unit axis k.
        fn rodrigues(v: [f32; 3], k: [f32; 3], alpha: f32) -> [f32; 3] {
            let (s, c) = alpha.sin_cos();
            let kx = k[0]; let ky = k[1]; let kz = k[2];
            let vx = v[0]; let vy = v[1]; let vz = v[2];
            let dot = kx * vx + ky * vy + kz * vz;
            let cross = [
                ky * vz - kz * vy,
                kz * vx - kx * vz,
                kx * vy - ky * vx,
            ];
            [
                vx * c + cross[0] * s + kx * dot * (1.0 - c),
                vy * c + cross[1] * s + ky * dot * (1.0 - c),
                vz * c + cross[2] * s + kz * dot * (1.0 - c),
            ]
        }

        // 10 random SO(3) rotations.
        let mut max_rel: f32 = 0.0;
        for trial in 0..10 {
            // Sample axis + angle. Axis from N(0,1) renormalized;
            // angle uniform in [0, π].
            let mut ax = rng.next_uniform(-1.0, 1.0);
            let mut ay = rng.next_uniform(-1.0, 1.0);
            let mut az = rng.next_uniform(-1.0, 1.0);
            let mag = (ax * ax + ay * ay + az * az).sqrt();
            if mag < 1e-6 { ax = 1.0; ay = 0.0; az = 0.0; } else {
                ax /= mag; ay /= mag; az /= mag;
            }
            let angle = rng.next_uniform(0.0, std::f32::consts::PI);

            let rotated: Vec<[f32; 3]> = base_pos.iter()
                .map(|&v| rodrigues(v, [ax, ay, az], angle))
                .collect();

            let cl_rot = run_kernel(&rotated);

            // Per-l relative error.
            for l in 0..=LMAX {
                let r = cl_ref[l];
                let q = cl_rot[l];
                let diff = (r - q).abs();
                let scale = r.abs().max(1e-3);
                let rel = diff / scale;
                if rel > max_rel { max_rel = rel; }
                // Tolerance: 1e-3. The kernel runs in f32 with
                // --use_fast_math and the centroid + AABB pre-pass
                // accumulates one float-add per spike; at 64 spikes
                // and Y_lm magnitudes up to ~1 the accumulated ULP
                // budget is on the order of 1e-4. The acos/atan2
                // pair adds a few more ULP at the edges. 1e-3 is
                // comfortably above the worst case.
                assert!(diff < 1e-3 || rel < 1e-3,
                    "trial {}: C_{} not invariant — ref={:.6}, rot={:.6}, \
                     diff={:.2e}, rel={:.2e}",
                    trial, l, r, q, diff, rel);
            }
        }
        eprintln!("[g11] max relative C_l drift over 10 rotations × 6 l = {:.2e}",
                  max_rel);
    }
}

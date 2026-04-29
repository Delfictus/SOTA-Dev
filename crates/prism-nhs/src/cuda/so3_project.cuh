// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / RECT-3.1.b — SO(3) Projection Kernel + ContactShellTile
// ═══════════════════════════════════════════════════════════════════════
//
// Per the Production Architecture mandate Phase 1 Deliverable 1.2
// (operator directive 2026-04-29). Consumes RichSpike clusters
// produced by the M1 / LBVH lane and writes one `ContactShellTile`
// per cluster: a 384-byte, 128-byte-aligned hardware execution tile
// holding the per-cluster spherical-harmonic expansion `a_lm` and the
// rotation-invariant power spectrum `C_l = Σ_m |a_lm|²`.
//
// **Why ContactShellTile (not Tensor)**: per the operator's
// nomenclature override (2026-04-29), the contact-shell record is a
// hardware-bound EXECUTION tile (alignas(128), sized to fit Blackwell
// shared-memory 16×16 / 32×8 fragment blocks), not a logical tensor.
// Naive global-memory pointer-chasing loops are FORBIDDEN.
//
// **Kernel layout**:
//   - One block per cluster (blockIdx.x == cluster_id).
//   - 128 threads/block (4 warps × 32 lanes).
//   - Pass 1: warp-shuffle reduce for centroid (sum/n) + AABB
//     (min/max). Output broadcast to shared memory.
//   - Pass 2: each thread evaluates Y_lm for its assigned spikes
//     (relative to the just-computed centroid), accumulates
//     `local_alm[36]`. Warp-shuffle reduce + cross-warp combine
//     produces the final per-cluster `a_lm`.
//   - Thread 0 computes `C_l = Σ_m |a_lm|²` for l=0..5 and writes
//     the 384-byte ContactShellTile.
//
// **WMMA forward-declaration**: the 64-float coefficients[] payload
// (only first 36 used at Lmax=5) is sized to a 16×4 tf32 fragment
// matrix. RECT-3.1.c replaces the warp-shuffle reduction with
// `nvcuda::wmma::fragment` matmul; the on-tile layout is unchanged
// so the FFI surface stays stable across the refactor.
//
// **Rotational invariance** (G11): C_l depends only on rotational
// power, not on the choice of basis frame. Verified by the Rust-side
// G11 test which applies 10 random SO(3) rotations to a synthetic
// spike cloud and asserts the C_l spectrum is identical within 1e-4
// relative tolerance.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -c
//
// ═══════════════════════════════════════════════════════════════════════

#ifndef PRISM_NHS_SO3_PROJECT_CUH
#define PRISM_NHS_SO3_PROJECT_CUH

#include <cstdint>
#include <cuda_runtime.h>

#include "rich_spike.cuh"
#include "sh_basis.cuh"

namespace prism_nhs { namespace so3_project {

using prism_nhs::rich_spike::RichSpike;
using prism_nhs::sh_basis::LMAX;
using prism_nhs::sh_basis::N_COEFFS;

// ─────────────────────────────────────────────────────────────────────
// ContactShellTile — 384-byte, 128-byte-aligned execution tile.
//
// LAYOUT CONTRACT (FFI-stable):
//   sizeof(ContactShellTile)  == 384 bytes
//   alignof(ContactShellTile) == 128 bytes
//
//   Offset  Field              Size  Notes
//   ─────────────────────────────────────────────────────────────
//   0       phase              4     CCNS phase tag
//   4       stream_id          4     Producing stream
//   8       cluster_id         4     int32 (matches RichSpike)
//   12      frame              4     Frame index
//   16      coefficients[64]   256   a_lm; first 36 valid (Lmax=5)
//   272     power_spectrum[8]  32    C_l for l=0..5; [6,7] = 0 pad
//   304     aabb_min[4]        16    xyz + 1 pad (LDG.E.128 align)
//   320     aabb_max[4]        16    xyz + 1 pad
//   336     spike_count        4     N spikes contributing
//   340     adjudication_code  4     Filled by pre_rank later
//   344     reserved[6]        24    Future-use telemetry slots
//   368     _pad[16]           16    Padding to 3 × 128 = 384
//   ─────────────────────────────────────────────────────────────
//   Total                      384
//
// Rust mirror is `#[repr(C, align(128))]` in
// `crates/prism-nhs/src/so3_project.rs`. Layout drift breaks FFI;
// the Rust-side `contact_shell_tile_layout_*` test pins it.
// ─────────────────────────────────────────────────────────────────────
struct alignas(128) ContactShellTile {
    // Header (16 B)
    uint32_t phase;
    uint32_t stream_id;
    int32_t  cluster_id;
    uint32_t frame;
    // a_lm coefficients — 256 B (64 floats; first 36 valid for Lmax=5).
    // Sized for forward-compat with WMMA 16×16 / 32×8 fragment loads.
    float    coefficients[64];
    // C_l power spectrum — 32 B (8 floats; first 6 valid for Lmax=5).
    float    power_spectrum[8];
    // AABB — 16 B + 16 B (xyz + pad each, for LDG.E.128 alignment).
    float    aabb_min[4];
    float    aabb_max[4];
    // Metadata — 32 B.
    uint32_t spike_count;
    uint32_t adjudication_code;
    uint32_t reserved[6];
    // Padding to 3 × 128 = 384 B.
    uint8_t  _pad[16];
};
static_assert(sizeof(ContactShellTile) == 384,
    "ContactShellTile layout drift: not 384 B");
static_assert(alignof(ContactShellTile) == 128,
    "ContactShellTile alignment drift: not 128 B");

// ─────────────────────────────────────────────────────────────────────
// extern "C" host orchestration.
//
// Caller protocol:
//   1. Call `prism_sh_basis_init` (sh_basis.cuh) once per process to
//      populate the device-side K_LM table.
//   2. Call `prism_sh_basis_get_k_lm_dev_ptr` to obtain a const
//      device pointer to K_LM.
//   3. Pack RichSpikes into a single device buffer; sort or group by
//      cluster_id; build a CSR-style cluster_offsets[N+1] array.
//   4. Allocate `n_clusters × ContactShellTile` device output.
//   5. Call `prism_so3_project_run`.
//   6. The kernel writes one tile per cluster; the C_l spectrum is
//      ready for the F1 SWITCH adjudicator (RECT-3.3).
// ─────────────────────────────────────────────────────────────────────

extern "C" {

/// Sentinel `0x53033` ("S0(3)" — SO(3) projection). Pinned by Rust-
/// side `link_probe_returns_sentinel` test.
uint32_t prism_so3_project_link_probe(void);

/// Run the SO(3) projection kernel. Writes one ContactShellTile per
/// cluster.
///
/// Inputs:
///   d_spikes          : device ptr to N RichSpikes
///   d_cluster_offsets : device ptr to (n_clusters+1) CSR offsets
///                       (cluster i spans [offsets[i], offsets[i+1]))
///   n_clusters        : number of clusters
///   d_k_lm            : device ptr to K_LM[36]
///                       (from prism_sh_basis_get_k_lm_dev_ptr)
///   d_tiles_out       : device ptr to n_clusters ContactShellTiles
///   frame_id          : frame index to stamp into each tile header
///   stream            : CUDA stream
///
/// Returns the result of `cudaGetLastError` after the kernel launch.
cudaError_t prism_so3_project_run(
    const RichSpike* d_spikes,
    const uint32_t*  d_cluster_offsets,
    uint32_t         n_clusters,
    const float*     d_k_lm,
    ContactShellTile* d_tiles_out,
    uint32_t         frame_id,
    cudaStream_t     stream
);

}  // extern "C"

}}  // namespace prism_nhs::so3_project

#endif  // PRISM_NHS_SO3_PROJECT_CUH

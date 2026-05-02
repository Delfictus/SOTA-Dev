// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / M1.2.19.B — GhostTileFrame capture kernel (impl)
// ═══════════════════════════════════════════════════════════════════════════
//
// Single-block, n_clusters-thread kernel.  Each thread `i` < n_clusters
// inspects its cluster's adjudication_code (read via byte-offset arithmetic
// from the FFI struct's offset 52 — current_divergence is at 48,
// adjudication_code at 52, mirroring InterferometricAdjudicatorFfi); if
// the code is Construct (1) or Violation (2), the thread atomicAdds the
// shared counter, bounds-checks against max_records, and writes its
// 1408-byte record (GhostTileFrame header + ContactShellTile body) into
// the pinned-host, device-mapped ring at offset
// `COUNTER_SECTOR (128) + slot * 1408`.
//
// The kernel reads `tile.geo_power_spectrum[0..6]` (offset 272 within
// the 1280-byte ContactShellTile) into the header's power_spectrum[6]
// for offline Φ_sym integration.
//
// thermo_flux[2] and causal_lead_residue are sentinel-filled (NaN, MAX)
// in this commit — gating the FFI slots as wired but unpopulated until
// the upstream water-density derivative + KCC argmax buses surface.
// ═══════════════════════════════════════════════════════════════════════════

#include "ghost_tile_kernel.cuh"
#include "so3_project.cuh"        // ContactShellTile (1280 bytes, align 128)
#include <cstdint>
#include <cstring>                // memcpy
#include <cuda_runtime.h>
#include <math_constants.h>       // CUDART_NAN_F

using prism_nhs::so3_project::ContactShellTile;

// FFI byte-offsets for InterferometricAdjudicatorFfi fields the kernel
// reads.  These match the asserts in adjudicator.cuh.  We don't include
// adjudicator.cuh here because that would drag the SO(3) dependency
// into a TU that only needs the byte offsets.
static constexpr size_t PRISM_ADJ_CURRENT_DIVERGENCE_OFF = 48;
static constexpr size_t PRISM_ADJ_ADJUDICATION_CODE_OFF  = 52;

extern "C"
__global__ void prism_ghost_pipe_stage_kernel(
    uint8_t* __restrict__              ring_base,
    const ContactShellTile* __restrict__ tiles,
    const uint8_t* __restrict__        adj,
    uint64_t                           frame_idx,
    uint32_t                           n_clusters,
    uint32_t                           max_records
) {
    const uint32_t i = threadIdx.x;
    if (i >= n_clusters) return;

    // adjudication_code is per-frame, not per-cluster on this build —
    // but the design accommodates future per-cluster Z-vector adjudication.
    // For now read the global code; all clusters share the same SWITCH.
    const uint8_t adj_code =
        *reinterpret_cast<const uint8_t*>(adj + PRISM_ADJ_ADJUDICATION_CODE_OFF);
    if (adj_code < 1u) return;  // Prune (0): nothing to record

    // Bounds-check + atomic claim.
    uint32_t* p_counter = reinterpret_cast<uint32_t*>(ring_base);
    const uint32_t slot = atomicAdd(p_counter, 1u);
    if (slot >= max_records) {
        // Buffer full — saturating decrement keeps counter ≤ max_records
        // so the host's `payload_bytes()` call returns the correct length.
        atomicMin(p_counter, max_records);
        return;
    }

    // Compute the slot's record base address.
    const size_t record_off = PRISM_GHOST_COUNTER_SECTOR
                            + static_cast<size_t>(slot) * PRISM_GHOST_RECORD_BYTES;
    uint8_t* record_base = ring_base + record_off;

    // ── 1. Write the 128-byte GhostTileFrame header ─────────────────
    GhostTileFrame* hdr = reinterpret_cast<GhostTileFrame*>(record_base);
    hdr->frame_idx         = frame_idx;
    hdr->site_id           = i;
    hdr->chain_id          = 0u;          // chain resolution: follow-up commit
    hdr->adjudication_code = adj_code;
    hdr->_pad[0] = 0u; hdr->_pad[1] = 0u;
    // current_divergence (f32 at offset 48) ≈ Σ_planes Δ_AB at write time.
    hdr->kl_divergence =
        *reinterpret_cast<const float*>(adj + PRISM_ADJ_CURRENT_DIVERGENCE_OFF);
    // Geometry-plane SO(3) C_l[0..6].  geo_power_spectrum is 8 floats
    // (6 valid + 2 pad) at ContactShellTile offset 272 — copy first 6.
    const ContactShellTile& tile_i = tiles[i];
    #pragma unroll
    for (int l = 0; l < 6; ++l) {
        hdr->power_spectrum[l] = tile_i.geo_power_spectrum[l];
    }
    // Sentinel: thermo_flux + causal_lead_residue not yet bus-bound.
    hdr->thermo_flux[0] = CUDART_NAN_F;
    hdr->thermo_flux[1] = CUDART_NAN_F;
    hdr->causal_lead_residue = 0xFFFFFFFFu;
    #pragma unroll
    for (int k = 0; k < 18; ++k) hdr->_reserved[k] = 0u;

    // ── 2. Write the 1280-byte ContactShellTile payload ─────────────
    // Use memcpy (nvcc lowers to coalesced LDG/STG.E.128 cycles given
    // the 128-byte alignment of both source and destination).
    memcpy(record_base + sizeof(GhostTileFrame),
           &tile_i,
           sizeof(ContactShellTile));
}

// ─── Host launcher ──────────────────────────────────────────────────────────

extern "C"
int prism_ghost_pipe_stage_launch(
    uint64_t      ring_base_dev,
    const void*   tiles,
    const void*   adj,
    uint64_t      frame_idx,
    uint32_t      n_clusters,
    uint32_t      max_records,
    void*         stream)
{
    if (ring_base_dev == 0ull || tiles == nullptr || adj == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_clusters == 0u || max_records == 0u) {
        return static_cast<int>(cudaSuccess);
    }
    // Single block, n_clusters threads (clamped to 64 — SISR_MAX_CLUSTERS
    // matches; we share the bound for ABI consistency).
    constexpr uint32_t MAX_THREADS = 64u;
    const uint32_t threads = (n_clusters < MAX_THREADS) ? n_clusters : MAX_THREADS;
    prism_ghost_pipe_stage_kernel<<<1, threads, 0,
        static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<uint8_t*>(ring_base_dev),
        static_cast<const ContactShellTile*>(tiles),
        static_cast<const uint8_t*>(adj),
        frame_idx,
        n_clusters,
        max_records
    );
    return static_cast<int>(cudaGetLastError());
}

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

// Wave 1 / Q1 — chain-boundary residue index.  Hardcoded for the 7C8R
// Mpro dimer: chain A spans residues 1..300, chain B starts at residue
// 301 (in the merged-topology renumbering).  Future multi-target work
// should turn this into a __constant__ uploaded by the host.
static constexpr uint32_t PRISM_GHOST_CHAIN_BOUNDARY = 301u;

// Wave 1 / Q1 — cluster → representative-residue lookup table.
// Populated host-side via prism_ghost_set_cluster_repr_residue (below)
// from the cluster's max-spike-intensity residue.  64-entry slab
// covers the SIMT block width; unused entries are u32_MAX (sentinel
// → chain_id falls back to 0 / "unknown").
__constant__ uint32_t d_cluster_to_repr_residue[64];

extern "C"
__global__ void prism_ghost_pipe_stage_kernel(
    uint8_t* __restrict__              ring_base,
    const ContactShellTile* __restrict__ tiles,
    const uint8_t* __restrict__        adj,
    const uint32_t* __restrict__       d_kcc_lead,   // Wave 1 / Q2 — F2-pool [n_clusters]
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
    // Wave 1 / P5 — atomicAdd on a host-mapped pinned counter; multiple
    // streams ALL contend on this single u32 word.  Blackwell sm_120
    // resolves the contention on the L2 atomic engine.  __threadfence_system()
    // immediately after publishes the counter increment to the host
    // address space so the Reaper thread observes the slot reservation
    // before the kernel starts writing the 1408-byte record below.
    uint32_t* p_counter = reinterpret_cast<uint32_t*>(ring_base);
    const uint32_t slot = atomicAdd(p_counter, 1u);
    __threadfence_system();
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
    // Wave 1 / Q1 — chain_id resolved via __constant__ cluster→repr-residue
    // table (host-populated in update_cluster_residue_map).  For 7C8R the
    // chain boundary is residue 301 (chain B starts at the merged residue
    // index 301).  Sentinel u32_MAX means "unmapped at host time" → 0.
    const uint32_t repr_res = d_cluster_to_repr_residue[i];
    hdr->chain_id = (repr_res == 0xFFFFFFFFu)
                    ? 0u
                    : (repr_res < PRISM_GHOST_CHAIN_BOUNDARY ? 0u : 1u);
    hdr->adjudication_code = adj_code;
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
    // Wave 1 / P4 — thermo_flux populated from the relaxed manifold:
    //   thermo_flux[0] = therm_power_spectrum[0]   (water-density l=0)
    //   thermo_flux[1] = caus_power_spectrum[0]    (causal trigger l=0)
    // NaN/Inf substituted with 0.0; CLASS_TAINTED bit set so downstream
    // η = ΔV_wd / ΔV_vib integrators can exclude tainted records.
    const float therm_l0 = tile_i.therm_power_spectrum[0];
    const float caus_l0  = tile_i.caus_power_spectrum[0];
    const bool  tainted  = !isfinite(therm_l0) || !isfinite(caus_l0);
    hdr->thermo_flux[0] = isfinite(therm_l0) ? therm_l0 : 0.0f;
    hdr->thermo_flux[1] = isfinite(caus_l0)  ? caus_l0  : 0.0f;
    hdr->telemetry_flags = tainted ? GHOST_TELEMETRY_CLASS_TAINTED : uint16_t{0};
    // Wave 1 / Q2 — causal_lead_residue read from the F2-pool buffer
    // populated host-side between chunks via argmax|d_kcc_temporal_corr|.
    // Sentinel u32_MAX until first KCC chunk completes → kept verbatim.
    hdr->causal_lead_residue = (d_kcc_lead != nullptr)
                               ? d_kcc_lead[i]
                               : 0xFFFFFFFFu;
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
    const void*   d_kcc_lead,   // Wave 1 / Q2 — may be nullptr (kernel falls back to u32_MAX sentinel)
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
        static_cast<const uint32_t*>(d_kcc_lead),
        frame_idx,
        n_clusters,
        max_records
    );
    return static_cast<int>(cudaGetLastError());
}

// ─── Host helper: populate the cluster→repr-residue constant table ──────────
//
// Wave 1 / Q1 — called once per campaign init from the Rust orchestrator
// after Pillar 1 (SpikeToCluster4D) finishes computing the per-cluster
// representative residue (the residue with highest aggregate spike
// intensity inside the cluster).  Stream-ordered cudaMemcpyToSymbolAsync
// targets the __constant__ slab; the first 64 entries are written, any
// trailing entries left at the previous campaign's value (kernel reads
// only [0..n_clusters) so out-of-range entries are inert).

extern "C"
int prism_ghost_set_cluster_repr_residue(
    const uint32_t* repr_residues_host,   // [n] host array
    uint32_t        n,                    // 1..=64
    void*           stream)
{
    if (repr_residues_host == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n == 0u) return static_cast<int>(cudaSuccess);
    if (n > 64u) n = 64u;
    cudaError_t rc = cudaMemcpyToSymbolAsync(
        d_cluster_to_repr_residue,
        repr_residues_host,
        n * sizeof(uint32_t),
        /*offset*/ 0,
        cudaMemcpyHostToDevice,
        static_cast<cudaStream_t>(stream)
    );
    return static_cast<int>(rc);
}

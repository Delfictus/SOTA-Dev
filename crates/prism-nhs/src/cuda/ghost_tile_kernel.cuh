// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / M1.2.19.B (Amendment 3.13) — Ghost Tile Channel-B exfiltration
// ═══════════════════════════════════════════════════════════════════════════
//
// Captured-graph single-thread-per-cluster kernel that pushes one record
// of [GhostTileFrame header (128 B)][ContactShellTile payload (1280 B)]
// = 1408 bytes to a pinned-host, device-mapped ring buffer when
// `adj.adjudication_code >= 1`.
//
// The buffer layout:
//   offset 0..4    : u32 n_frames_written       (atomic counter)
//   offset 4..128  : u8  pad[124]               (counter-sector pad)
//   offset 128..   : record[max_records]        (1408 bytes each)
//
// The kernel atomicAdds the counter, bounds-checks against max_records,
// then performs the 1408-byte write through the device-mapped pointer
// (lands directly in pinned host RAM via PCIe Gen5 DMA on Blackwell sm_120).
//
// The C struct mirror's offsets are statically asserted to match the
// Rust `GhostTileFrame` definition byte-for-byte.
// ═══════════════════════════════════════════════════════════════════════════

#pragma once

#include <cstdint>
#include <cstddef>
#include <cuda_runtime.h>

// ─── GhostTileFrame mirror (byte-for-byte sync with ghost_tile.rs) ─────────

struct __align__(128) GhostTileFrame {
    uint64_t frame_idx;             // offset 0
    uint32_t site_id;               // offset 8
    uint8_t  chain_id;              // offset 12
    uint8_t  adjudication_code;     // offset 13
    uint16_t telemetry_flags;       // offset 14   (Wave 1 — was uint8_t _pad[2])
    float    kl_divergence;         // offset 16
    float    power_spectrum[6];     // offset 20
    float    thermo_flux[2];        // offset 44
    uint32_t causal_lead_residue;   // offset 52
    uint32_t _reserved[18];         // offset 56..128
};

// Wave 1 / P4 — telemetry_flags bit definitions (mirrors
// GHOST_TELEMETRY_CLASS_TAINTED in Rust ghost_tile.rs).
constexpr uint16_t GHOST_TELEMETRY_CLASS_TAINTED = 0x0001u;

static_assert(sizeof(GhostTileFrame)  == 128,
              "GhostTileFrame size drift — must be 128 bytes (operator §2.1).");
static_assert(alignof(GhostTileFrame) == 128,
              "GhostTileFrame alignment drift — must be 128-byte aligned.");
static_assert(offsetof(GhostTileFrame, frame_idx)            ==  0, "frame_idx offset drift");
static_assert(offsetof(GhostTileFrame, site_id)              ==  8, "site_id offset drift");
static_assert(offsetof(GhostTileFrame, chain_id)             == 12, "chain_id offset drift");
static_assert(offsetof(GhostTileFrame, adjudication_code)    == 13, "adjudication_code offset drift");
static_assert(offsetof(GhostTileFrame, kl_divergence)        == 16, "kl_divergence offset drift");
static_assert(offsetof(GhostTileFrame, power_spectrum)       == 20, "power_spectrum offset drift");
static_assert(offsetof(GhostTileFrame, thermo_flux)          == 44, "thermo_flux offset drift");
static_assert(offsetof(GhostTileFrame, causal_lead_residue)  == 52, "causal_lead_residue offset drift");
static_assert(offsetof(GhostTileFrame, _reserved)            == 56, "_reserved offset drift");

// Per-record byte size (header + ContactShellTile payload).
constexpr size_t PRISM_GHOST_RECORD_BYTES   = 128 + 1280;
constexpr size_t PRISM_GHOST_COUNTER_SECTOR = 128;

#ifdef __cplusplus
extern "C" {
#endif

/// Records the Channel-B push kernel onto `stream` inside the V2
/// captured-graph window.  Insert downstream of the Adjudicator step
/// (so adj.adjudication_code is final) and downstream of the SO(3)
/// projection (so the baseline tiles hold the latest power spectra).
///
/// Wave 1 / Q2: `d_kcc_lead` is an F2-pool `uint32_t[n_clusters]` buffer
/// holding the per-cluster causal-lead residue id (host populates it
/// between chunks via argmax|d_kcc_temporal_corr|).  Pass `nullptr` to
/// have the kernel emit `0xFFFFFFFFu` sentinels in causal_lead_residue
/// — typical bootstrap state during the first chunk.
int prism_ghost_pipe_stage_launch(
    uint64_t       ring_base_dev,    /* GhostTileRing::device_base */
    const void*    tiles,            /* baseline manifold (n_clusters × ContactShellTile) */
    const void*    adj,              /* InterferometricAdjudicatorFfi */
    const void*    d_kcc_lead,       /* Wave 1 / Q2 — F2-pool [n_clusters] u32, nullable */
    uint64_t       frame_idx,        /* monotonic frame counter */
    uint32_t       n_clusters,
    uint32_t       max_records,
    void*          stream);

/// Wave 1 / Q1 — host-side populator for the __constant__ cluster→repr
/// residue table.  Call once per campaign after Pillar 1 clustering
/// converges; stream-ordered cudaMemcpyToSymbolAsync.
int prism_ghost_set_cluster_repr_residue(
    const uint32_t* repr_residues_host,   /* [n] host array */
    uint32_t        n,                    /* 1..=64 */
    void*           stream);

#ifdef __cplusplus
}
#endif

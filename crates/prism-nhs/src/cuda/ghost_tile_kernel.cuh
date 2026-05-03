// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / M1.2.19.B (Amendment 3.14 — v9D' Sector-Lock) — Ghost Tile
// Channel-B exfiltration
// ═══════════════════════════════════════════════════════════════════════════
//
// Captured-graph single-thread-per-cluster kernel that pushes one
// 4096-byte GhostTileFrame record to a pinned-host, device-mapped ring
// buffer when `adj.adjudication_code >= 1`.  The trailing 1280-byte
// ContactShellTile body has been retired from the on-disk format; the
// per-plane SO(3) spectra are folded into the expanded 4-plane
// `power_spectrum[24]` field of the frame itself, and the ring writes
// land on `O_DIRECT | O_DSYNC` (one record = one NVMe physical sector).
//
// The buffer layout (Amendment 3.14 §1.1):
//   offset    0..4     : u32 n_frames_written       (atomic counter)
//   offset    4..4096  : u8  pad[4092]              (counter sector)
//   offset 4096..      : record[max_records]        (4096 bytes each)
//
// The kernel atomicAdds the counter, bounds-checks against max_records,
// then writes the 4096-byte record through the device-mapped pointer
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

struct __align__(4096) GhostTileFrame {
    uint64_t frame_idx;                 // offset 0
    uint32_t site_id;                   // offset 8
    uint8_t  chain_id;                  // offset 12
    uint8_t  adjudication_code;         // offset 13
    uint16_t telemetry_flags;           // offset 14
    float    kl_divergence;             // offset 16
    float    power_spectrum[24];        // offset 20  (4 planes × 6 bands)
    float    thermo_flux[2];            // offset 116
    uint32_t causal_lead_residue;       // offset 124
    uint32_t _reserved_payload[32];     // offset 128 (Pillar 5 expansion)
    uint8_t  _slack[3840];              // offset 256..4096 (sector pad)
};

// Wave 1 / P4 — telemetry_flags bit definitions (mirrors
// GHOST_TELEMETRY_CLASS_TAINTED in Rust ghost_tile.rs).
constexpr uint16_t GHOST_TELEMETRY_CLASS_TAINTED = 0x0001u;

static_assert(sizeof(GhostTileFrame)  == 4096,
              "GhostTileFrame size drift — must be 4096 bytes (Amendment 3.14).");
static_assert(alignof(GhostTileFrame) == 4096,
              "GhostTileFrame alignment drift — must be 4096-byte aligned.");
static_assert(offsetof(GhostTileFrame, frame_idx)            ==   0, "frame_idx offset drift");
static_assert(offsetof(GhostTileFrame, site_id)              ==   8, "site_id offset drift");
static_assert(offsetof(GhostTileFrame, chain_id)             ==  12, "chain_id offset drift");
static_assert(offsetof(GhostTileFrame, adjudication_code)    ==  13, "adjudication_code offset drift");
static_assert(offsetof(GhostTileFrame, telemetry_flags)      ==  14, "telemetry_flags offset drift");
static_assert(offsetof(GhostTileFrame, kl_divergence)        ==  16, "kl_divergence offset drift");
static_assert(offsetof(GhostTileFrame, power_spectrum)       ==  20, "power_spectrum offset drift");
static_assert(offsetof(GhostTileFrame, thermo_flux)          == 116, "thermo_flux offset drift");
static_assert(offsetof(GhostTileFrame, causal_lead_residue)  == 124, "causal_lead_residue offset drift");
static_assert(offsetof(GhostTileFrame, _reserved_payload)    == 128, "_reserved_payload offset drift");
static_assert(offsetof(GhostTileFrame, _slack)               == 256, "_slack offset drift");

// Per-record byte size (header-only post-Amendment 3.14; ContactShellTile
// retired from on-disk format).  Counter sector enlarged to 4096 B so the
// first record lands at sector boundary 1.
constexpr size_t PRISM_GHOST_RECORD_BYTES   = 4096;
constexpr size_t PRISM_GHOST_COUNTER_SECTOR = 4096;

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

/// **M1.2.20.C-C / T20 + Amendment 3.20** — Topology-Driven Chain
/// Boundary populator.  Host parses topology's per-residue chain_ids
/// column and produces a [k+1] cumulative-residue-boundary array
/// where d_chain_offsets[i] is the first residue id of chain i.
/// Stream-ordered cudaMemcpyToSymbolAsync; `n` clamped to 8.  Replaces
/// any prior hardcoded chain boundary in the kernel source — the
/// engine is now target-agnostic for up to 8-chain complexes.
int prism_ghost_set_chain_offsets(
    const uint32_t* offsets_host,    /* [n] residue-id boundaries */
    uint32_t        n,               /* 1..=8 */
    void*           stream);

/// **Path Ω Phase 2** — Geometry-Emergent Chain Identity LUT populator.
/// `chain_ids_host[i]` is the chain assignment for cluster i:
///   0 = chain A, 1 = chain B, 0xFF = unset (kernel falls back to legacy
///   d_chain_offsets residue-boundary scan).
/// Stream-ordered cudaMemcpyToSymbolAsync; n clamped to 64 (LUT size).
/// Computed host-side from cluster centroids vs the topology's
/// dimer_dyad axis at V2 build time.
int prism_ghost_set_cluster_chain_id(
    const uint8_t*  chain_ids_host,  /* [n] per-cluster chain id */
    uint32_t        n,               /* 1..=64 */
    void*           stream);

#ifdef __cplusplus
}
#endif

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

// ─── M1.2.23 §5 — GhostTileFrame v2 MAR payload schema overlay ──────────────
//
// The 128-byte _reserved_payload[32] region at offset 128 is partitioned into
// structured Transparent-MAR fields. Backward compatibility is preserved:
// v1 producers zero the whole region (legacy kernel behavior at line 215),
// so the schema_version read at offset 128 returns 0 → v1. v2 producers
// (Commit 4 wires this) write GHOST_FRAME_SCHEMA_V2 = 2 and populate the
// structured fields below.
//
// MIRROR REQUIREMENT: these constants MUST match
// crates/prism-nhs/src/ghost_tile.rs (search "M1.2.23 §5"). Drift is detected
// at compile time by static_asserts immediately after this block.

constexpr uint32_t GHOST_FRAME_SCHEMA_V1_LEGACY = 0u;
constexpr uint32_t GHOST_FRAME_SCHEMA_V2        = 2u;

// Field offsets within a 4096-byte GhostTileFrame record:
constexpr size_t GHOST_V2_OFFSET_SCHEMA_VERSION       = 128;  // u32
constexpr size_t GHOST_V2_OFFSET_OBSERVATION_PASS     = 132;  // u8
constexpr size_t GHOST_V2_OFFSET_DISCOVERY_PASS       = 133;  // u8
constexpr size_t GHOST_V2_OFFSET_PERTURBATION_CHAN    = 134;  // u8 (UV bitcode 0..3 or 0xFF)
constexpr size_t GHOST_V2_OFFSET_UV_WAVELENGTH_NM     = 136;  // u16
constexpr size_t GHOST_V2_OFFSET_FIELD_COMPLETE_FLAGS = 138;  // u16
constexpr size_t GHOST_V2_OFFSET_GEAR_ID              = 140;  // u32
constexpr size_t GHOST_V2_OFFSET_DT_FS                = 144;  // f32
constexpr size_t GHOST_V2_OFFSET_STEP_IDX             = 148;  // u64
constexpr size_t GHOST_V2_OFFSET_AABB_MIN             = 160;  // f32 × 3
constexpr size_t GHOST_V2_OFFSET_AABB_MAX             = 172;  // f32 × 3
constexpr size_t GHOST_V2_OFFSET_CENTROID             = 184;  // f32 × 3
constexpr size_t GHOST_V2_OFFSET_V2_RESERVED          = 196;  // u32 × 15

constexpr uint8_t GHOST_PERTURBATION_CHANNEL_UNKNOWN  = 0xFFu;

// Compile-time bounds: every v2 field must land inside _reserved_payload
// (128..256). _slack is at offset 256.
static_assert(GHOST_V2_OFFSET_SCHEMA_VERSION       == 128, "v2 schema_version offset drift");
static_assert(GHOST_V2_OFFSET_OBSERVATION_PASS     == 132, "v2 observation_pass offset drift");
static_assert(GHOST_V2_OFFSET_DISCOVERY_PASS       == 133, "v2 discovery_pass offset drift");
static_assert(GHOST_V2_OFFSET_PERTURBATION_CHAN    == 134, "v2 perturbation_channel offset drift");
static_assert(GHOST_V2_OFFSET_UV_WAVELENGTH_NM     == 136, "v2 uv_wavelength_nm offset drift");
static_assert(GHOST_V2_OFFSET_FIELD_COMPLETE_FLAGS == 138, "v2 field_completeness_flags offset drift");
static_assert(GHOST_V2_OFFSET_GEAR_ID              == 140, "v2 gear_id offset drift");
static_assert(GHOST_V2_OFFSET_DT_FS                == 144, "v2 dt_fs offset drift");
static_assert(GHOST_V2_OFFSET_STEP_IDX             == 148, "v2 step_idx offset drift");
static_assert(GHOST_V2_OFFSET_AABB_MIN             == 160, "v2 aabb_min offset drift");
static_assert(GHOST_V2_OFFSET_AABB_MAX             == 172, "v2 aabb_max offset drift");
static_assert(GHOST_V2_OFFSET_CENTROID             == 184, "v2 centroid offset drift");
static_assert(GHOST_V2_OFFSET_V2_RESERVED          == 196, "v2_reserved offset drift");
// The v2 fields must end at or before the _slack region (offset 256).
static_assert(GHOST_V2_OFFSET_V2_RESERVED + 60     == 256, "v2 region must end at _slack offset 256");
// All v2 offsets must lie within the legacy _reserved_payload[32] span (128..256).
static_assert(GHOST_V2_OFFSET_SCHEMA_VERSION       >= 128, "v2 schema must be inside _reserved_payload");
static_assert(GHOST_V2_OFFSET_V2_RESERVED + 60     <= 256, "v2 region must fit inside _reserved_payload");

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
///
/// Diagnostic firehose: when `firehose_enable != 0`, the kernel ALWAYS
/// emits one record per cluster per replay regardless of adj_code
/// (post-audit operator directive 2026-05-03 — full per-cluster KL
/// trajectory + 4-plane spectrum time series even on null-manifest runs).
/// Emitted records still carry the actual adjudication_code so downstream
/// consumers can distinguish construct events from diagnostic samples.
int prism_ghost_pipe_stage_launch(
    uint64_t       ring_base_dev,    /* GhostTileRing::device_base */
    const void*    tiles,            /* baseline manifold (n_clusters × ContactShellTile) */
    const void*    adj,              /* InterferometricAdjudicatorFfi */
    const void*    d_kcc_lead,       /* Wave 1 / Q2 — F2-pool [n_clusters] u32, nullable */
    uint64_t       frame_idx,        /* monotonic frame counter */
    uint32_t       n_clusters,
    uint32_t       max_records,
    void*          stream,
    uint32_t       firehose_enable); /* 0 = adj-gated; nonzero = unconditional emission */

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

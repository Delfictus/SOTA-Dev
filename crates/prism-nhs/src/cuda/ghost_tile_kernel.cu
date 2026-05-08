// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / M1.2.19.B (Amendment 3.14 — v9D' Sector-Lock) — GhostTileFrame
// capture kernel (impl)
// ═══════════════════════════════════════════════════════════════════════════
//
// Single-block, n_clusters-thread kernel.  Each thread `i` < n_clusters
// inspects its cluster's adjudication_code (read via byte-offset arithmetic
// from the FFI struct's offset 52 — current_divergence is at 48,
// adjudication_code at 52, mirroring InterferometricAdjudicatorFfi); if
// the code is Construct (1) or Violation (2), the thread atomicAdds the
// shared counter, bounds-checks against max_records, and writes its
// 4096-byte GhostTileFrame record into the pinned-host, device-mapped
// ring at offset `COUNTER_SECTOR (4096) + slot * 4096`.
//
// The kernel reads `tile.geo_power_spectrum[0..6]` into the header's
// power_spectrum[0..6] (geometry plane); planes 1..3 are zeroed in this
// commit (the upstream causal/thermo/chemistry plane spectra are not
// yet surfaced to the ghost stage — follow-up commit will wire them).
//
// thermo_flux[2] and causal_lead_residue follow the same sentinel-fill
// pattern as the prior 1408-byte format — gates FFI slots as wired but
// unpopulated until upstream telemetry buses land.  The trailing 1280-byte
// ContactShellTile body has been retired from the on-disk format.
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

// Wave 1 / Q1 — cluster → representative-residue lookup table.
// Populated host-side via prism_ghost_set_cluster_repr_residue (below)
// from the cluster's max-spike-intensity residue.  64-entry slab
// covers the SIMT block width; unused entries are u32_MAX (sentinel
// → chain_id falls back to 0 / "unknown").
__constant__ uint32_t d_cluster_to_repr_residue[64];

// **M1.2.20.C-C / T20 — Topology-Driven Chain Boundary LUT.**
// Per operator's Amendment 3.20 "Purge of Hardcoded Boundaries":
// the 7C8R-specific 4613 atom index has been excised from kernel
// source.  d_chain_offsets[k] is the residue-id boundary at which
// chain k STARTS.  d_chain_offsets[0] is conventionally 0 (chain A
// origin).  Unused entries are populated with u32::MAX so the
// linear scan terminates naturally for fewer than 8 chains.
//
// Kernel logic (residue-major):
//   for i in 0..7:
//     if d_chain_offsets[i] <= residue_id < d_chain_offsets[i+1]:
//       chain_id = i;  break;
//
// 8 × u32 = 32 B → fits in a single L1 constant-cache line, single-
// cycle broadcast load to every thread in the warp.  CPU does the
// heavy topology parsing in nhs_rt_full.rs; the GPU only does
// integer comparison (operator §4 — "do not implement a complex
// string-matching parser on the GPU").
__constant__ uint32_t d_chain_offsets[8];

// **Path Ω Phase 2 — Geometry-Emergent Chain Identity LUT.**
// Populated host-side at V2 build via prism_ghost_set_cluster_chain_id
// (below).  Per cluster i ∈ [0..64), holds:
//   0 ⇒ chain A (centroid on negative side of dyad-perpendicular normal)
//   1 ⇒ chain B (centroid on positive side)
//   0xFF ⇒ unset / fall back to legacy d_chain_offsets residue-boundary scan
//
// Computed on the host from cluster centroids and the topology's
// dimer_dyad block: for each cluster, project its centroid onto a
// vector perpendicular to the dyad axis and assign chain by sign.
// This makes chain identity GEOMETRY-EMERGENT — independent of any
// upstream prism-prep / sanitizer that may have collapsed chain
// labels in the PDB → topology pipeline (the v9D 7C8R issue).
__constant__ uint8_t d_cluster_to_chain_id[64];

extern "C"
__global__ void prism_ghost_pipe_stage_kernel(
    uint8_t* __restrict__              ring_base,
    const ContactShellTile* __restrict__ tiles,
    const uint8_t* __restrict__        adj,
    const uint32_t* __restrict__       d_kcc_lead,   // Wave 1 / Q2 — F2-pool [n_clusters]
    uint64_t                           frame_idx,
    uint32_t                           n_clusters,
    uint32_t                           max_records,
    uint32_t                           firehose_enable  // 0 = adj-gated; nonzero = ALWAYS emit
) {
    const uint32_t i = threadIdx.x;
    if (i >= n_clusters) return;

    // adjudication_code is per-frame, not per-cluster on this build —
    // but the design accommodates future per-cluster Z-vector adjudication.
    // For now read the global code; all clusters share the same SWITCH.
    const uint8_t adj_code =
        *reinterpret_cast<const uint8_t*>(adj + PRISM_ADJ_ADJUDICATION_CODE_OFF);
    // Diagnostic firehose mode: when enabled, ALWAYS emit a record per
    // cluster per replay regardless of adj_code (operator post-audit
    // 2026-05-03 — full per-cluster KL trajectory + 4-plane spectrum
    // time series captured even on null-manifest runs).  The emitted
    // record still carries the actual adj_code so downstream tools
    // can distinguish "construct event" from "diagnostic sample".
    if (firehose_enable == 0u && adj_code < 1u) return;  // Prune (0): nothing to record

    // Bounds-check + atomic claim.
    // Wave 1 / P5 — atomicAdd on a host-mapped pinned counter; multiple
    // streams ALL contend on this single u32 word.  Blackwell sm_120
    // resolves the contention on the L2 atomic engine.  __threadfence_system()
    // immediately after publishes the counter increment to the host
    // address space so the Reaper thread observes the slot reservation
    // before the kernel starts writing the 4096-byte record below.
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

    // ── Write the 4096-byte GhostTileFrame record ────────────────────
    GhostTileFrame* hdr = reinterpret_cast<GhostTileFrame*>(record_base);
    hdr->frame_idx         = frame_idx;
    hdr->site_id           = i;
    // **M1.2.20.C-C / T20 + Amendment 3.20** — Topology-Driven Chain
    // Resolver.  No hardcoded numbers in this kernel — chain_id is
    // computed from a host-populated d_chain_offsets[8] LUT that the
    // CPU produced by parsing the topology's per-residue chain_ids
    // column.
    //
    // Algorithm: scan d_chain_offsets backwards (chain k → chain 0)
    // and select the largest k for which d_chain_offsets[k] <= repr_res.
    // Branchless via a fold: keep updating chain_id whenever a new
    // (larger) k satisfies the comparison.  Sentinel u32_MAX in unset
    // entries fails the comparison naturally (repr_res < u32_MAX is
    // true; we want the OPPOSITE direction — boundaries[k] <= repr_res
    // — so u32_MAX as a "boundary" is fine: nothing ever sits past it).
    // Path Ω Phase 2 — prefer geometry-emergent chain_id LUT.  When
    // populated (sentinel 0xFFu = unset), fall back to the legacy
    // residue-id-boundary scan against d_chain_offsets.  This makes
    // chain identity recoverable from the SO(3) tile centroids
    // computed every frame, independent of how the topology file
    // labelled chains.
    uint8_t chain_id = d_cluster_to_chain_id[i];
    if (chain_id == 0xFFu) {
        chain_id = 0u;
        const uint32_t repr_res = d_cluster_to_repr_residue[i];
        if (repr_res != 0xFFFFFFFFu) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                const uint32_t boundary = d_chain_offsets[k];
                if (boundary != 0xFFFFFFFFu && boundary <= repr_res) {
                    chain_id = static_cast<uint8_t>(k);
                }
            }
        }
    }
    hdr->chain_id = chain_id;
    hdr->adjudication_code = adj_code;
    // current_divergence (f32 at offset 48) ≈ Σ_planes Δ_AB at write time.
    hdr->kl_divergence =
        *reinterpret_cast<const float*>(adj + PRISM_ADJ_CURRENT_DIVERGENCE_OFF);
    // 4-plane SO(3) C_l[0..6] expansion (Amendment 3.14):
    //   plane 0 (geo)        -> power_spectrum[ 0.. 6]   wired
    //   plane 1 (causal)     -> power_spectrum[ 6..12]   wired (M1.2.25 plane-fan-out)
    //   plane 2 (thermo)     -> power_spectrum[12..18]   wired (M1.2.25 plane-fan-out)
    //   plane 3 (chemistry)  -> power_spectrum[18..24]   wired (M1.2.25 plane-fan-out)
    // All four planes are populated by so3_project.cu:518-530 into the
    // ContactShellTile; the ghost kernel now copies all four 6-band stripes
    // into the GhostTileFrame.power_spectrum[24] array. NaN/Inf inputs are
    // substituted with 0.0 and CLASS_TAINTED is set if any input is non-finite.
    const ContactShellTile& tile_i = tiles[i];
    bool plane_tainted = false;
    #pragma unroll
    for (int l = 0; l < 6; ++l) {
        const float g = tile_i.geo_power_spectrum[l];
        const float c = tile_i.caus_power_spectrum[l];
        const float t = tile_i.therm_power_spectrum[l];
        const float h = tile_i.chem_power_spectrum[l];
        hdr->power_spectrum[ 0 + l] = isfinite(g) ? g : 0.0f;
        hdr->power_spectrum[ 6 + l] = isfinite(c) ? c : 0.0f;
        hdr->power_spectrum[12 + l] = isfinite(t) ? t : 0.0f;
        hdr->power_spectrum[18 + l] = isfinite(h) ? h : 0.0f;
        plane_tainted |= !(isfinite(g) && isfinite(c) && isfinite(t) && isfinite(h));
    }
    // Wave 1 / P4 — thermo_flux populated from the relaxed manifold:
    //   thermo_flux[0] = therm_power_spectrum[0]   (water-density l=0)
    //   thermo_flux[1] = caus_power_spectrum[0]    (causal trigger l=0)
    // NaN/Inf substituted with 0.0; CLASS_TAINTED bit set so downstream
    // η = ΔV_wd / ΔV_vib integrators can exclude tainted records.
    const float therm_l0 = tile_i.therm_power_spectrum[0];
    const float caus_l0  = tile_i.caus_power_spectrum[0];
    const bool  tainted  = plane_tainted || !isfinite(therm_l0) || !isfinite(caus_l0);
    hdr->thermo_flux[0] = isfinite(therm_l0) ? therm_l0 : 0.0f;
    hdr->thermo_flux[1] = isfinite(caus_l0)  ? caus_l0  : 0.0f;
    hdr->telemetry_flags = tainted ? GHOST_TELEMETRY_CLASS_TAINTED : uint16_t{0};
    // Wave 1 / Q2 — causal_lead_residue read from the F2-pool buffer
    // populated host-side between chunks via argmax|d_kcc_temporal_corr|.
    // Sentinel u32_MAX until first KCC chunk completes → kept verbatim.
    hdr->causal_lead_residue = (d_kcc_lead != nullptr)
                               ? d_kcc_lead[i]
                               : 0xFFFFFFFFu;

    // Zero the Pillar 5 expansion slab (32 × u32 = 128 B).  Inert until
    // the follow-up commit wires Φ_sym phase-lock score, Γ_w desolvation
    // tax, gear ID, hardware clock, etc.
    #pragma unroll
    for (int k = 0; k < 32; ++k) hdr->_reserved_payload[k] = 0u;

    // The 3840-byte _slack region is left untouched: cuMemHostAlloc
    // zero-inits the buffer, and subsequent slot reuse is gated by the
    // host's truncate-on-open of the output file.  Skipping the write
    // here saves 3840 B of STG bandwidth per record (= ~12% of total
    // record bandwidth at 4096 B per record); the on-disk bytes are
    // deterministic zero from the allocator either way.
    //
    // Note: the ContactShellTile body that previously followed the
    // header has been retired (Amendment 3.14 §1.1).  The 4-plane
    // power_spectrum field above absorbs the geometry-plane spectrum;
    // planes 1..3 fold in on the follow-up commit.
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
    void*         stream,
    uint32_t      firehose_enable)  // 0 = adj-gated emission; nonzero = unconditional per-cluster
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
        max_records,
        firehose_enable
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

// **M1.2.20.C-C / T20 + Amendment 3.20** — Topology-Driven Chain
// Boundary populator.  Host parses the topology's per-residue
// chain_ids column and produces:
//
//   d_chain_offsets[0] = 0
//   d_chain_offsets[1] = first residue id of chain 1
//   d_chain_offsets[2] = first residue id of chain 2
//   ...
//   d_chain_offsets[k] = u32::MAX  (sentinel for unused chains)
//
// The Ghost stage kernel scans this LUT to assign chain_id to each
// record.  No 7C8R-specific magic numbers; works for any target the
// CPU side knows how to parse.  Stream-ordered cudaMemcpyToSymbolAsync.
extern "C"
int prism_ghost_set_chain_offsets(
    const uint32_t* offsets_host,    // [n] residue-id boundaries
    uint32_t        n,               // 1..=8
    void*           stream)
{
    if (offsets_host == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n == 0u) return static_cast<int>(cudaSuccess);
    if (n > 8u) n = 8u;
    cudaError_t rc = cudaMemcpyToSymbolAsync(
        d_chain_offsets,
        offsets_host,
        n * sizeof(uint32_t),
        /*offset*/ 0,
        cudaMemcpyHostToDevice,
        static_cast<cudaStream_t>(stream)
    );
    return static_cast<int>(rc);
}

// Path Ω Phase 2 — host populator for d_cluster_to_chain_id LUT.
// `chain_ids_host` is a per-cluster uint8_t (0=A, 1=B, 0xFF=unset).
// `n` is clamped to 64 (constant array size).  Stream-ordered
// cudaMemcpyToSymbolAsync; the kernel reads via __constant__ broadcast.
extern "C"
int prism_ghost_set_cluster_chain_id(
    const uint8_t* chain_ids_host,
    uint32_t       n,
    void*          stream)
{
    if (chain_ids_host == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n == 0u) return static_cast<int>(cudaSuccess);
    if (n > 64u) n = 64u;
    cudaError_t rc = cudaMemcpyToSymbolAsync(
        d_cluster_to_chain_id,
        chain_ids_host,
        n * sizeof(uint8_t),
        /*offset*/ 0,
        cudaMemcpyHostToDevice,
        static_cast<cudaStream_t>(stream)
    );
    return static_cast<int>(rc);
}

// ═══════════════════════════════════════════════════════════════════════════
// M1.2.23 §5 + §4 — Transparent MAR v2 emission kernel
// ═══════════════════════════════════════════════════════════════════════════
//
// Source-landed v2 producer. Writes the same v1 fields as the legacy kernel
// (frame_idx, site_id, chain_id, adj_code, kl_divergence, power_spectrum,
// thermo_flux, causal_lead_residue) bit-identically, plus structured v2
// fields into the 128-byte _reserved_payload region per the layout in
// ghost_tile_kernel.cuh / ghost_tile.rs.
//
// Hard invariant (per directive §4): observation_pass=true and
// discovery_pass=false MUST never trigger steering. This kernel does NOT
// control F1 SWITCH — that lives in adjudicator.cu's predicate bridge.
// The 3σ observation gate here only widens TELEMETRY emission. F1
// behavior is unchanged. Discovery semantics in the F1 path are left to
// the existing adjudicator threshold (which must equal the discovery
// threshold passed here for downstream consumers to interpret the field
// consistently — that wire-up is host-side at launch time).
//
// Emission gate (per-thread):
//   emit if adj_code >= 1                                   (legacy)
//   OR    if firehose_enable != 0                           (legacy)
//   OR    if kl_divergence >= observation_threshold_kl      (NEW v2)
//
// Per-record observation_pass / discovery_pass are written verbatim from
// the kernel-computed booleans, regardless of whether emission was gated
// by adj_code or by observation_pass. Downstream consumers can filter.

extern "C" __global__
void prism_ghost_pipe_stage_kernel_v2(
    uint8_t* __restrict__       ring_base,
    const ContactShellTile* __restrict__ tiles,
    const uint8_t* __restrict__ adj,
    const uint32_t* __restrict__ d_kcc_lead,
    uint64_t                    frame_idx,
    uint32_t                    n_clusters,
    uint32_t                    max_records,
    uint32_t                    firehose_enable,
    // M1.2.23 §4 — Transparent MAR v2 host-passed thresholds + context.
    float                       observation_threshold_kl,
    float                       discovery_threshold_kl,
    uint32_t                    gear_id,
    float                       dt_fs,
    uint64_t                    step_idx)
{
    const uint32_t i = static_cast<uint32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_clusters) return;

    const uint8_t adj_code = adj[PRISM_ADJ_ADJUDICATION_CODE_OFF];
    const float kl = *reinterpret_cast<const float*>(adj + PRISM_ADJ_CURRENT_DIVERGENCE_OFF);
    const bool obs_pass = isfinite(kl) && (kl >= observation_threshold_kl);
    const bool disc_pass = isfinite(kl) && (kl >= discovery_threshold_kl);

    // Widened emission gate (v2): legacy adj_code gate OR observation pass.
    if (firehose_enable == 0u && adj_code < 1u && !obs_pass) return;

    // Atomic counter at offset 0.
    uint32_t* counter = reinterpret_cast<uint32_t*>(ring_base);
    const uint32_t slot = atomicAdd(counter, 1u);
    if (slot >= max_records) {
        atomicSub(counter, 1u);
        return;
    }

    const size_t record_off = PRISM_GHOST_COUNTER_SECTOR
                            + static_cast<size_t>(slot) * PRISM_GHOST_RECORD_BYTES;
    uint8_t* record_base = ring_base + record_off;
    GhostTileFrame* hdr = reinterpret_cast<GhostTileFrame*>(record_base);

    // ─── v1 fields (bit-identical with legacy kernel) ─────────────────────
    hdr->frame_idx = frame_idx;
    hdr->site_id   = i;
    uint8_t chain_id = d_cluster_to_chain_id[i];
    if (chain_id == 0xFFu) {
        chain_id = 0u;
        const uint32_t repr_res = d_cluster_to_repr_residue[i];
        if (repr_res != 0xFFFFFFFFu) {
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                const uint32_t boundary = d_chain_offsets[k];
                if (boundary != 0xFFFFFFFFu && boundary <= repr_res) {
                    chain_id = static_cast<uint8_t>(k);
                }
            }
        }
    }
    hdr->chain_id          = chain_id;
    hdr->adjudication_code = adj_code;
    hdr->kl_divergence     = kl;

    // M1.2.25 plane-fan-out: copy all four SO(3) planes (geo/caus/therm/chem)
    // from the ContactShellTile into the GhostTileFrame.power_spectrum[24]
    // array. so3_project.cu:518-530 already populates all four planes via
    // WMMA Tensor Core reduction; this kernel was previously dropping planes
    // 1-3 to zero. NaN/Inf substituted with 0.0 + CLASS_TAINTED on any miss.
    const ContactShellTile& tile_i = tiles[i];
    bool plane_tainted = false;
    #pragma unroll
    for (int l = 0; l < 6; ++l) {
        const float g = tile_i.geo_power_spectrum[l];
        const float c = tile_i.caus_power_spectrum[l];
        const float t = tile_i.therm_power_spectrum[l];
        const float h = tile_i.chem_power_spectrum[l];
        hdr->power_spectrum[ 0 + l] = isfinite(g) ? g : 0.0f;
        hdr->power_spectrum[ 6 + l] = isfinite(c) ? c : 0.0f;
        hdr->power_spectrum[12 + l] = isfinite(t) ? t : 0.0f;
        hdr->power_spectrum[18 + l] = isfinite(h) ? h : 0.0f;
        plane_tainted |= !(isfinite(g) && isfinite(c) && isfinite(t) && isfinite(h));
    }

    const float therm_l0 = tile_i.therm_power_spectrum[0];
    const float caus_l0  = tile_i.caus_power_spectrum[0];
    const bool  tainted  = plane_tainted || !isfinite(therm_l0) || !isfinite(caus_l0);
    hdr->thermo_flux[0]    = isfinite(therm_l0) ? therm_l0 : 0.0f;
    hdr->thermo_flux[1]    = isfinite(caus_l0)  ? caus_l0  : 0.0f;
    hdr->telemetry_flags   = tainted ? GHOST_TELEMETRY_CLASS_TAINTED : uint16_t{0};
    hdr->causal_lead_residue = (d_kcc_lead != nullptr) ? d_kcc_lead[i] : 0xFFFFFFFFu;

    // ─── v2 fields (writes into the same 128 B span the v1 kernel zeros) ──
    // Zero-init the whole reserved span first so any v2 fields we don't set
    // here are deterministic zero (matches v1 behavior for unset bytes).
    #pragma unroll
    for (int k = 0; k < 32; ++k) hdr->_reserved_payload[k] = 0u;

    // Schema version + threshold pass flags.
    *reinterpret_cast<uint32_t*>(record_base + GHOST_V2_OFFSET_SCHEMA_VERSION) = GHOST_FRAME_SCHEMA_V2;
    record_base[GHOST_V2_OFFSET_OBSERVATION_PASS]  = obs_pass  ? uint8_t{1} : uint8_t{0};
    record_base[GHOST_V2_OFFSET_DISCOVERY_PASS]    = disc_pass ? uint8_t{1} : uint8_t{0};
    record_base[GHOST_V2_OFFSET_PERTURBATION_CHAN] = GHOST_PERTURBATION_CHANNEL_UNKNOWN;
    *reinterpret_cast<uint16_t*>(record_base + GHOST_V2_OFFSET_UV_WAVELENGTH_NM) = uint16_t{0};
    *reinterpret_cast<uint16_t*>(record_base + GHOST_V2_OFFSET_FIELD_COMPLETE_FLAGS) =
        // Bit 0: kl finite, Bit 1: gear_id provided non-zero, Bit 2: dt_fs > 0,
        // Bit 3: thermo_flux not class-tainted.
        (isfinite(kl) ? 0x0001u : 0u)
      | (gear_id != 0u ? 0x0002u : 0u)
      | (dt_fs > 0.0f ? 0x0004u : 0u)
      | (!tainted ? 0x0008u : 0u);
    *reinterpret_cast<uint32_t*>(record_base + GHOST_V2_OFFSET_GEAR_ID) = gear_id;
    *reinterpret_cast<float*>(record_base + GHOST_V2_OFFSET_DT_FS) = dt_fs;
    *reinterpret_cast<uint64_t*>(record_base + GHOST_V2_OFFSET_STEP_IDX) = step_idx;

    // ─── GHOST_NATIVE_SPATIAL_MAPPING_WIRE — native AABB + centroid ───
    //
    // ContactShellTile.aabb_min[4] / aabb_max[4] (xyz + 1 pad) are populated
    // by so3_project.cu:454+ from RichSpike positions in the protein topology
    // coordinate frame (Å). tile_i is already in scope from the plane-fan-out
    // block above (line 456). The same `i` indexes both this Ghost record
    // (site_id = i) and the source tile, so the spatial mapping is exact at
    // write time — no host-side derivation needed.
    //
    // The selected centroid view is the AABB midpoint, an honest scalar
    // proxy of the cluster's spatial extent. It is NOT the full
    // phase-manifold centroid family (spike_density / kcc_driver /
    // phasor_coherent / thermo_weighted / ghost_zstr_event_weighted views);
    // those richer views are computed offline by the SiteManifest
    // materializer when the per-residue evidence sources are joined. The
    // record's compatibility centroid_xyz field carries the alias label
    // "aabb_midpoint_native_contact_shell_tile" via the
    // field_completeness_flags Bit 4 = SPATIAL_NATIVE_AABB_MIDPOINT.
    //
    // Order matters: this write lands AFTER the _reserved_payload[k]=0
    // blanket clear above so the zero loop does not erase the spatial
    // payload. If either bound is non-finite (NaN/Inf from upstream
    // SO(3) projection), substitute 0.0 and leave Bit 4 unset — the
    // host audit treats Bit 4 == 0 as "spatial fields not native-populated."
    {
        const float* aabb_min_src = tile_i.aabb_min;
        const float* aabb_max_src = tile_i.aabb_max;
        float* aabb_min_dst = reinterpret_cast<float*>(record_base + GHOST_V2_OFFSET_AABB_MIN);
        float* aabb_max_dst = reinterpret_cast<float*>(record_base + GHOST_V2_OFFSET_AABB_MAX);
        float* centroid_dst = reinterpret_cast<float*>(record_base + GHOST_V2_OFFSET_CENTROID);
        bool spatial_native_finite = true;
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            const float lo_raw = aabb_min_src[k];
            const float hi_raw = aabb_max_src[k];
            const bool  ok = isfinite(lo_raw) && isfinite(hi_raw);
            const float lo = ok ? lo_raw : 0.0f;
            const float hi = ok ? hi_raw : 0.0f;
            aabb_min_dst[k] = lo;
            aabb_max_dst[k] = hi;
            centroid_dst[k] = 0.5f * (lo + hi); // AABB midpoint — labeled alias
            spatial_native_finite = spatial_native_finite && ok;
        }
        // Field completeness Bit 4 = SPATIAL_NATIVE_AABB_MIDPOINT populated.
        // OR-merge with prior bits set above so we don't clobber Bits 0..3.
        if (spatial_native_finite) {
            uint16_t* fcf = reinterpret_cast<uint16_t*>(record_base + GHOST_V2_OFFSET_FIELD_COMPLETE_FLAGS);
            *fcf = static_cast<uint16_t>(*fcf | 0x0010u);
        }
    }
    // _v2_reserved [u32; 15] @ 196: zero from the loop above (still zero).
}

extern "C"
int prism_ghost_pipe_stage_launch_v2(
    uint64_t      ring_base_dev,
    const void*   tiles,
    const void*   adj,
    const void*   d_kcc_lead,
    uint64_t      frame_idx,
    uint32_t      n_clusters,
    uint32_t      max_records,
    void*         stream,
    uint32_t      firehose_enable,
    float         observation_threshold_kl,
    float         discovery_threshold_kl,
    uint32_t      gear_id,
    float         dt_fs,
    uint64_t      step_idx)
{
    if (ring_base_dev == 0ull || tiles == nullptr || adj == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_clusters == 0u || max_records == 0u) {
        return static_cast<int>(cudaSuccess);
    }
    constexpr uint32_t MAX_THREADS = 64u;
    const uint32_t threads = (n_clusters < MAX_THREADS) ? n_clusters : MAX_THREADS;
    prism_ghost_pipe_stage_kernel_v2<<<1, threads, 0,
        static_cast<cudaStream_t>(stream)>>>(
        reinterpret_cast<uint8_t*>(ring_base_dev),
        static_cast<const ContactShellTile*>(tiles),
        static_cast<const uint8_t*>(adj),
        static_cast<const uint32_t*>(d_kcc_lead),
        frame_idx,
        n_clusters,
        max_records,
        firehose_enable,
        observation_threshold_kl,
        discovery_threshold_kl,
        gear_id,
        dt_fs,
        step_idx
    );
    return static_cast<int>(cudaGetLastError());
}

// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / ZSTR — Zero-Stall Telemetry Ring kernel declarations
//
// Two kernels:
//   1. zstr_signal_completion  — fires at end of captured CUDA graph DAG;
//      __threadfence_system() + pinned-fence write notifies the host
//      ZSTR consumer without any cudaStreamSynchronize.
//
//   2. zstr_async_pos_stage    — copies n_atoms * 3 floats from VRAM
//      d_positions into a designated triple-buffer slot (pinned host
//      memory) using per-element ST.GLOBAL.128 (vectorized stores),
//      fenced before the completion signal.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -c
// ═══════════════════════════════════════════════════════════════════════════

#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace prism_nhs { namespace zstr {

/// Written at the end of every captured-graph epoch.
/// Sets slot_fence to 1 after __threadfence_system() ensures all
/// preceding DMA writes (positions, forces) are visible to the host.
///
/// Path Z device-slot variant: kernel reads __constant__ d_zstr_active_slot
/// at execution time to compute the slot-specific fence address inside the
/// pinned ring.  base_fence points at slot-0's completion_fence.
extern "C"
__global__ void zstr_signal_completion_kernel(
    uint8_t* __restrict__ base_fence,
    uint32_t              inter_slot_stride
);

/// Path Z device-slot variant: kernel reads __constant__ d_zstr_active_slot
/// at execution time to compute the slot-specific positions destination
/// inside the pinned ring.  base_pinned points at slot-0's header start;
/// pos_offset_in_slot adds the header-size offset to land on the positions
/// payload.
extern "C"
__global__ void zstr_pos_stage_f4_kernel(
    uint8_t* __restrict__       base_pinned,
    uint32_t                    inter_slot_stride,
    uint32_t                    pos_offset_in_slot,
    const float4* __restrict__  src_vram,
    uint32_t                    n_floats
);

/// T11 — Action-Recovery force-stage kernel.
///
/// Vectorized float4 LDG.E.128 → STG.E.128 DMA from `d_forces` into the
/// active slot's pinned forces payload, with per-warp `__shfl_down_sync`
/// butterfly reduction of Σ Fᵢ² and warp-leader `atomicAdd` directly into
/// the slot's pinned `force_norm` field (offset 28). The field holds the
/// running sum-of-squares after this kernel; the sqrt post-pass converts
/// it to the L2 norm in-place.
///
/// Caller invariant: `force_norm` MUST be 0.0f before launch. The host
/// ZSTR Reaper resets it alongside `completion_fence` once it has read
/// and validated the slot.
extern "C"
__global__ void zstr_force_stage_f4_kernel(
    uint8_t* __restrict__       base_pinned,
    uint32_t                    inter_slot_stride,
    uint32_t                    force_offset_in_slot,
    uint32_t                    force_norm_offset_in_slot,
    const float4* __restrict__  src_forces,
    uint32_t                    n_floats
);

/// T11 — Single-thread <<<1,1>>> post-pass.  Reads the active slot's
/// `force_norm` (= Σ Fᵢ² as written by force_stage), writes `sqrtf`
/// back in place, converting the field to the L2 norm ‖F‖₂.  NaN
/// propagates verbatim — the host G29 Reaper traps non-finite values.
extern "C"
__global__ void zstr_force_norm_sqrt_kernel(
    uint8_t* __restrict__       base_pinned,
    uint32_t                    inter_slot_stride,
    uint32_t                    force_norm_offset_in_slot
);

}} // namespace prism_nhs::zstr

// ─── C-ABI capture-window launchers ─────────────────────────────────────────
// Called from Rust captured_pipeline.rs inside cuStreamBeginCapture to record
// ZSTR kernel nodes into the in-progress CUgraph.
//
// Both functions return `cudaError_t` cast to `int`; 0 == cudaSuccess.
// They must be called on the telemetry_stream (non-blocking) AFTER the tile
// DMA cuMemcpyDtoHAsync and BEFORE the telemetry→MD JOIN event record.

#ifdef __cplusplus
extern "C" {
#endif

/// Path Z launcher: device-slot pos_stage (kernel reads d_zstr_active_slot).
/// `base_pinned` is the slot-0 header start; `inter_slot_stride` is the
/// inter-slot byte distance (== ring frame_size); `pos_offset_in_slot` is
/// the byte offset from slot base to positions payload (== sizeof header).
/// Host MUST call prism_zstr_set_active_slot BEFORE the launch / replay.
int zstr_launch_pos_stage(void*       base_pinned,
                           uint32_t    inter_slot_stride,
                           uint32_t    pos_offset_in_slot,
                           const void* src_vram,
                           uint32_t    n_atoms,
                           void*       stream);

/// Path Z launcher: device-slot fence signal.  base_fence points at slot-0
/// completion_fence; kernel rolls slot via d_zstr_active_slot.
int zstr_launch_fence_signal(void*    base_fence,
                              uint32_t inter_slot_stride,
                              void*    stream);

/// Path Z host helper: stream-ordered cudaMemcpyToSymbolAsync update of
/// d_zstr_active_slot. Caller invokes BEFORE each cuGraphLaunch on the
/// same stream so the captured kernels observe the new slot value.
int prism_zstr_set_active_slot(uint32_t slot, void* stream);

/// T11 — Records the force-stage kernel onto `stream` inside the
/// active capture window.  DMA copy + warp-shuffle reduce + atomic-add
/// into pinned `force_norm`.  Caller pairs this with
/// `zstr_launch_force_norm_sqrt` on the same stream for the in-place
/// sqrt post-pass.  Insert AFTER ASC atomic-add and BEFORE
/// `zstr_launch_fence_signal`.
int zstr_launch_force_stage(void*       base_pinned,
                             uint32_t    inter_slot_stride,
                             uint32_t    force_offset_in_slot,
                             uint32_t    force_norm_offset_in_slot,
                             const void* src_d_forces,
                             uint32_t    n_atoms,
                             void*       stream);

/// T11 — Records the single-thread sqrt post-pass that finalises
/// `force_norm` to the L2 norm.  Insert immediately after
/// `zstr_launch_force_stage` and before `zstr_launch_fence_signal`.
int zstr_launch_force_norm_sqrt(void*    base_pinned,
                                 uint32_t inter_slot_stride,
                                 uint32_t force_norm_offset_in_slot,
                                 void*    stream);

#ifdef __cplusplus
}
#endif

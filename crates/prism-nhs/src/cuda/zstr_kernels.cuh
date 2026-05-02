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

#ifdef __cplusplus
}
#endif

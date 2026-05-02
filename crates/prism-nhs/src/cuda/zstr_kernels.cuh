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
/// @param slot_fence   Pointer into the ZstrFrameHeader::completion_fence
///                     field for the current ring slot (pinned host mem).
extern "C"
__global__ void zstr_signal_completion_kernel(uint32_t* __restrict__ slot_fence);

/// Vectorized async position stage: copies the VRAM d_positions buffer
/// (n_atoms * 3 * f32) into the pre-designated pinned slot.
/// Uses float4 loads/stores (128-bit) for maximum PCIe throughput.
/// One thread per float4 quad (covers 4 consecutive f32 values).
///
/// @param dst_pinned   Pinned-host destination (positions payload of slot).
/// @param src_vram     VRAM source (d_positions, row-major n_atoms × 3 f32).
/// @param n_floats     n_atoms * 3.
extern "C"
__global__ void zstr_pos_stage_f4_kernel(
    float4* __restrict__       dst_pinned,
    const float4* __restrict__ src_vram,
    uint32_t                   n_floats
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

/// Launch zstr_pos_stage_f4_kernel on `stream` targeting `dst_pinned`.
/// grid = ceil(n_atoms*3 / 4) / 256 blocks, blockDim = 256.
int zstr_launch_pos_stage(void* dst_pinned, const void* src_vram,
                           uint32_t n_atoms, void* stream);

/// Launch zstr_signal_completion_kernel on `stream` (1×1 thread).
int zstr_launch_fence_signal(void* slot_fence, void* stream);

#ifdef __cplusplus
}
#endif

// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / ZSTR — Zero-Stall Telemetry Ring kernels (impl)
//
// Amendment 3.4 (Path Z) device-slot revision: per-launch slot rolling now
// uses `__constant__ uint32_t d_zstr_active_slot` updated via
// `cudaMemcpyToSymbolAsync` — replaces the legacy host-side
// `cuGraphExecKernelNodeSetParams_v2` patching, which broke under
// `cudaGraphAddChildGraphNode` (the spliced child graph is COPIED, leaving
// the original kernel-node handles non-addressable on the fused exec).
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
// ═══════════════════════════════════════════════════════════════════════════

#include "zstr_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace prism_nhs { namespace zstr {

// Active slot index (0..N_SLOTS-1).  Updated by the host orchestrator before
// each monolithic graph launch via cudaMemcpyToSymbolAsync; the captured
// kernels READ this value at execution time, so the same fused exec rolls
// through all slots without graph mutation.
__constant__ uint32_t d_zstr_active_slot;

// ─── Completion fence signal ────────────────────────────────────────────────

extern "C"
__global__ void zstr_signal_completion_kernel(
    uint8_t* __restrict__ base_fence,
    uint32_t              inter_slot_stride
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // __threadfence_system() ensures all writes issued by ANY kernel in the
    // graph (positions, forces, adjudication_code) are globally visible —
    // including to the host ZSTR consumer — before the fence store executes.
    __threadfence_system();

    const uint32_t slot = d_zstr_active_slot;
    uint32_t* fence_ptr =
        reinterpret_cast<uint32_t*>(base_fence + slot * inter_slot_stride);

    atomicExch(fence_ptr, 1u);
}

// ─── Vectorized position staging ────────────────────────────────────────────

extern "C"
__global__ void zstr_pos_stage_f4_kernel(
    uint8_t* __restrict__       base_pinned,        // pinned slot-0 base
    uint32_t                    inter_slot_stride,   // bytes between slots
    uint32_t                    pos_offset_in_slot,  // header bytes inside slot
    const float4* __restrict__  src_vram,
    uint32_t                    n_floats
) {
    const uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n_f4 = (n_floats + 3u) >> 2u;
    if (tid >= n_f4) return;

    const uint32_t slot = d_zstr_active_slot;
    float4* dst = reinterpret_cast<float4*>(
        base_pinned + slot * inter_slot_stride + pos_offset_in_slot
    );

    // LDG.E.128 (cacheable global load via __ldg) → STG.E.128 non-temporal
    // store to pinned WC memory. Bypasses L2 on the write side so the PCIe
    // DMA engine sees the data immediately.
    float4 quad = __ldg(&src_vram[tid]);
    dst[tid] = quad;
}

}} // namespace prism_nhs::zstr

// ─── C-ABI capture-window launchers ─────────────────────────────────────────

extern "C"
int zstr_launch_pos_stage(void*       base_pinned,
                           uint32_t    inter_slot_stride,
                           uint32_t    pos_offset_in_slot,
                           const void* src_vram,
                           uint32_t    n_atoms,
                           void*       stream)
{
    const uint32_t n_floats = n_atoms * 3u;
    const uint32_t n_f4     = (n_floats + 3u) >> 2u;
    const uint32_t block    = 256u;
    const uint32_t grid     = (n_f4 + block - 1u) / block;

    prism_nhs::zstr::zstr_pos_stage_f4_kernel<<<grid, block, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint8_t*>(base_pinned),
        inter_slot_stride,
        pos_offset_in_slot,
        static_cast<const float4*>(src_vram),
        n_floats
    );
    return static_cast<int>(cudaGetLastError());
}

extern "C"
int zstr_launch_fence_signal(void*    base_fence,
                              uint32_t inter_slot_stride,
                              void*    stream)
{
    prism_nhs::zstr::zstr_signal_completion_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint8_t*>(base_fence),
        inter_slot_stride
    );
    return static_cast<int>(cudaGetLastError());
}

// ─── Host-side slot index update (called per-launch outside capture) ────────

extern "C"
int prism_zstr_set_active_slot(uint32_t slot, void* stream)
{
    cudaError_t rc = cudaMemcpyToSymbolAsync(
        prism_nhs::zstr::d_zstr_active_slot,
        &slot,
        sizeof(uint32_t),
        0,
        cudaMemcpyHostToDevice,
        static_cast<cudaStream_t>(stream)
    );
    return static_cast<int>(rc);
}

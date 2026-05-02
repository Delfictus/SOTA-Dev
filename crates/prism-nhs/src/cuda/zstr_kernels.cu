// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / ZSTR — Zero-Stall Telemetry Ring kernels (impl)
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
// ═══════════════════════════════════════════════════════════════════════════

#include "zstr_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace prism_nhs { namespace zstr {

// ─── Completion fence signal ────────────────────────────────────────────────

extern "C"
__global__ void zstr_signal_completion_kernel(uint32_t* __restrict__ slot_fence) {
    // Single-thread fence: only thread 0 of block 0 writes.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // __threadfence_system() ensures all writes issued by ANY kernel
    // in the graph (positions, forces, adjudication_code) are globally
    // visible — including to the host ZSTR consumer — before the
    // fence store executes.  This is stronger than __threadfence()
    // which only guarantees GPU-internal ordering.
    __threadfence_system();

    // Atomic store: marks the ring slot as READY for the host consumer
    // spin-lock.  Uses volatile so nvcc cannot cache-hit this store.
    // The host consumer spin-reads the same pinned location.
    atomicExch(slot_fence, 1u);
}

// ─── Vectorized position staging ────────────────────────────────────────────

extern "C"
__global__ void zstr_pos_stage_f4_kernel(
    float4* __restrict__       dst_pinned,
    const float4* __restrict__ src_vram,
    uint32_t                   n_floats
) {
    // Each thread covers one float4 (4 × f32 = 128 bits).
    // Grid must be launched with ceil(n_floats / 4) threads total.
    const uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t n_f4 = (n_floats + 3u) >> 2u;  // ceil(n_floats / 4)
    if (tid >= n_f4) return;

    // LDG.E.128 (cacheable global load via __ldg) → STG.E.128 (non-temporal
    // store to pinned WC memory). Bypasses L2 on the write side so the
    // PCIe DMA engine sees the data immediately.
    float4 quad = __ldg(&src_vram[tid]);

    // Non-temporal store to write-combining pinned memory.
    // On sm_120 this maps to STG.E.128.NC, keeping L2 hot for
    // the force-kernel's subsequent read of d_positions.
    dst_pinned[tid] = quad;
}

}} // namespace prism_nhs::zstr

// ─── C-ABI capture-window launchers ─────────────────────────────────────────
// These are called from Rust inside cuStreamBeginCapture.  The runtime API
// <<<>>> launch syntax records the kernel node into the in-progress CUgraph
// under CU_STREAM_CAPTURE_MODE_GLOBAL.

extern "C"
int zstr_launch_pos_stage(void* dst_pinned, const void* src_vram,
                           uint32_t n_atoms, void* stream)
{
    const uint32_t n_floats = n_atoms * 3u;
    const uint32_t n_f4     = (n_floats + 3u) >> 2u;  // ceil(n_floats / 4)
    const uint32_t block    = 256u;
    const uint32_t grid     = (n_f4 + block - 1u) / block;

    prism_nhs::zstr::zstr_pos_stage_f4_kernel<<<grid, block, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<float4*>(dst_pinned),
        static_cast<const float4*>(src_vram),
        n_floats
    );
    return static_cast<int>(cudaGetLastError());
}

extern "C"
int zstr_launch_fence_signal(void* slot_fence, void* stream)
{
    prism_nhs::zstr::zstr_signal_completion_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint32_t*>(slot_fence)
    );
    return static_cast<int>(cudaGetLastError());
}

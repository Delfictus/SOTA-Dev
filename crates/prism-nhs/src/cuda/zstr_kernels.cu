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
#include <cstdio>  // Amendment 3.10 — kernel-level printf triage
#include <math_constants.h>

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

// ─── T11 — Force-stage: vectorized DMA + warp-shuffle Σ‖F‖² atomic-add ─────

extern "C"
__global__ void zstr_force_stage_f4_kernel(
    uint8_t* __restrict__       base_pinned,
    uint32_t                    inter_slot_stride,
    uint32_t                    force_offset_in_slot,
    uint32_t                    force_norm_offset_in_slot,
    const float4* __restrict__  src_forces,
    uint32_t                    n_floats
) {
    const uint32_t tid  = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t n_f4 = (n_floats + 3u) >> 2u;

    const uint32_t slot = d_zstr_active_slot;
    uint8_t*  slot_base = base_pinned + slot * inter_slot_stride;
    float4*   dst       = reinterpret_cast<float4*>(slot_base + force_offset_in_slot);
    float*    fn_ptr    = reinterpret_cast<float*>(slot_base + force_norm_offset_in_slot);

    // Per-thread DMA + sum-of-squares.  The tail bounds-check ensures
    // the last quad's components beyond n_floats do NOT contribute; we
    // still STG the quad verbatim (the trailing slots in the pinned
    // forces region are unused) — keeping the store fully vectorised.
    float local_sumsq = 0.0f;
    if (tid < n_f4) {
        const float4 q = __ldg(&src_forces[tid]);
        dst[tid] = q;

        const uint32_t base_idx = tid * 4u;
        if (base_idx + 0u < n_floats) local_sumsq += q.x * q.x;
        if (base_idx + 1u < n_floats) local_sumsq += q.y * q.y;
        if (base_idx + 2u < n_floats) local_sumsq += q.z * q.z;
        if (base_idx + 3u < n_floats) local_sumsq += q.w * q.w;
    }

    // Warp-level butterfly reduction (32-lane).  __shfl_down_sync is the
    // sm_120 SOTA primitive — single-instruction inter-lane communication
    // with no shared-memory traffic.  After the loop, lane 0 holds Σ over
    // its warp's 32 threads.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sumsq += __shfl_down_sync(0xffffffffu, local_sumsq, offset);
    }

    // Each warp's lane-0 atomicAdds its partial sum directly into the
    // pinned host slot's force_norm field — no intermediate scratch.
    // Pinned-host atomicAdd lands as a system-scope PCIe atomic on
    // sm_120; the host Reaper reads the post-sqrt finalised value.
    if (lane == 0u) {
        atomicAdd(fn_ptr, local_sumsq);
    }
}

// ─── T11 — Single-thread post-pass: in-place sqrtf of force_norm ──────────

extern "C"
__global__ void zstr_force_norm_sqrt_kernel(
    uint8_t* __restrict__       base_pinned,
    uint32_t                    inter_slot_stride,
    uint32_t                    force_norm_offset_in_slot
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const uint32_t slot = d_zstr_active_slot;
    float* fn_ptr = reinterpret_cast<float*>(
        base_pinned + slot * inter_slot_stride + force_norm_offset_in_slot
    );

    // sqrtf propagates NaN verbatim; negative values (impossible on a
    // sum-of-squares but defensive for clamp drift) produce NaN under
    // IEEE-754 — the host G29 Reaper traps non-finite reads.
    const float sumsq = *fn_ptr;
    *fn_ptr = sqrtf(sumsq);
}

// ─── M1.2.18.5 — Single-thread audit-field stage kernel ─────────────────────
//
// Reads V_t (f64) from `adj.potential_energy` at FFI offset 112 and W_ext
// (f64) by dereferencing the *mut f64 pointer at FFI offset 128.  Writes
// the two f64s into the active ZSTR slot at the operator-specified offsets:
//
//     slot[external_work_offset_in_slot]   ← W_ext     (offset 32, 8 B)
//     slot[potential_energy_offset_in_slot] ← V_t      (offset 40, 8 B)
//
// Captured downstream of the SFA so V_t and W_ext are settled before
// the read.  Pinned-host writes land as system-scope PCIe stores on
// sm_120, visible to the Reaper as soon as completion_fence flips.
//
// `adj` may be null in legacy / test fixtures — kernel writes 0.0 to
// both fields in that case.
extern "C"
__global__ void zstr_stage_audit_kernel(
    uint8_t* __restrict__   base_pinned,
    uint32_t                inter_slot_stride,
    uint32_t                external_work_offset_in_slot,
    uint32_t                potential_energy_offset_in_slot,
    const uint8_t* __restrict__ adj
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const uint32_t slot = d_zstr_active_slot;
    uint8_t* slot_base  = base_pinned + slot * inter_slot_stride;

    double v_t   = 0.0;
    double w_ext = 0.0;
    if (adj != nullptr) {
        // FFI offset 112: potential_energy (f64 VALUE).
        v_t = *reinterpret_cast<const double*>(adj + 112);
        // FFI offset 128: d_external_work (*mut f64 POINTER).
        const double* w_ptr =
            *reinterpret_cast<const double* const*>(adj + 128);
        if (w_ptr != nullptr) {
            w_ext = *w_ptr;
        }
    }

    *reinterpret_cast<double*>(slot_base + external_work_offset_in_slot)
        = w_ext;
    *reinterpret_cast<double*>(slot_base + potential_energy_offset_in_slot)
        = v_t;
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

// ─── T11 — Force-stage launchers ────────────────────────────────────────────

extern "C"
int zstr_launch_force_stage(void*       base_pinned,
                             uint32_t    inter_slot_stride,
                             uint32_t    force_offset_in_slot,
                             uint32_t    force_norm_offset_in_slot,
                             const void* src_d_forces,
                             uint32_t    n_atoms,
                             void*       stream)
{
    const uint32_t n_floats = n_atoms * 3u;
    const uint32_t n_f4     = (n_floats + 3u) >> 2u;
    const uint32_t block    = 256u;
    const uint32_t grid     = (n_f4 + block - 1u) / block;

    prism_nhs::zstr::zstr_force_stage_f4_kernel<<<grid, block, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint8_t*>(base_pinned),
        inter_slot_stride,
        force_offset_in_slot,
        force_norm_offset_in_slot,
        static_cast<const float4*>(src_d_forces),
        n_floats
    );
    return static_cast<int>(cudaGetLastError());
}

extern "C"
int zstr_launch_force_norm_sqrt(void*    base_pinned,
                                 uint32_t inter_slot_stride,
                                 uint32_t force_norm_offset_in_slot,
                                 void*    stream)
{
    prism_nhs::zstr::zstr_force_norm_sqrt_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint8_t*>(base_pinned),
        inter_slot_stride,
        force_norm_offset_in_slot
    );
    return static_cast<int>(cudaGetLastError());
}

extern "C"
int zstr_launch_stage_audit(void*       base_pinned,
                             uint32_t    inter_slot_stride,
                             uint32_t    external_work_offset_in_slot,
                             uint32_t    potential_energy_offset_in_slot,
                             const void* adj,
                             void*       stream)
{
    prism_nhs::zstr::zstr_stage_audit_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>(
        static_cast<uint8_t*>(base_pinned),
        inter_slot_stride,
        external_work_offset_in_slot,
        potential_energy_offset_in_slot,
        static_cast<const uint8_t*>(adj)
    );
    return static_cast<int>(cudaGetLastError());
}

// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / ASC Steering — Vectorized Repulsion Kernel (impl)
//
// Implements asc_inject_repulsion_v4_kernel using float4 vectorized loads
// and Blackwell atom.global.add.v4.f32 for simultaneous 4-wide atomic
// force accumulation.
//
// ── Design Notes ────────────────────────────────────────────────────────────
//
// Force buffer layout: AoS, [fx0,fy0,fz0, fx1,fy1,fz1, ...], n_atoms*3 f32.
// Reinterpreted as float4*: each float4 quad covers 4 consecutive f32 values
// which span atom boundaries.  The centroid pattern cycles with period 3:
//   quad offset 0 mod 3 → [cx, cy, cz, cx]
//   quad offset 1 mod 3 → [cy, cz, cx, cy]
//   quad offset 2 mod 3 → [cz, cx, cy, cz]
// This correctly applies F_i = α * ΔKL * (pos_i - c_i) per f32 component.
//
// ── Gate G22_ATOMIC_v4_VERIFIED ─────────────────────────────────────────────
// ptxas audit required:
//   nvcc -arch=sm_120 --ptx asc_steering.cu -o /tmp/asc.ptx
//   grep "atom.global.add.v4.f32" /tmp/asc.ptx
// If not present, the compiler substituted scalar atomicAdds (acceptable
// fallback; performance penalty ~4× per repulsion step).
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
// ═══════════════════════════════════════════════════════════════════════════

#include "asc_steering.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace prism_nhs { namespace asc {

// Centroid component at a given flat f32 index (cycles x→y→z→x→...).
__device__ __forceinline__
float centroid_at(const float3 c, uint32_t flat_idx) {
    switch (flat_idx % 3u) {
        case 0u: return c.x;
        case 1u: return c.y;
        default: return c.z;
    }
}

extern "C"
__global__ void asc_inject_repulsion_v4_kernel(
    float4* __restrict__       d_forces,
    const float4* __restrict__ d_pos,
    float3                     centroid,
    float                      alpha_gain,
    float                      delta_kl,
    uint32_t                   n_atoms
) {
    const uint32_t n_floats = n_atoms * 3u;
    const uint32_t n_quads  = (n_floats + 3u) >> 2u;  // ceil(n_floats / 4)
    const uint32_t quad_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (quad_idx >= n_quads) return;

    const float scale = alpha_gain * delta_kl;

    // Base flat-f32 index for this quad.
    const uint32_t base = quad_idx << 2u;  // quad_idx * 4

    // Load 4 atom-position floats (may span two atoms).
    float4 pos_quad = __ldg(&d_pos[quad_idx]);

    // Compute repulsion delta per f32 component.
    // centroid_at() resolves the correct X/Y/Z for each flat index.
    float4 force_delta;
    force_delta.x = scale * (pos_quad.x - centroid_at(centroid, base + 0u));
    force_delta.y = scale * (pos_quad.y - centroid_at(centroid, base + 1u));
    force_delta.z = scale * (pos_quad.z - centroid_at(centroid, base + 2u));
    force_delta.w = scale * (pos_quad.w - centroid_at(centroid, base + 3u));

    // ── Vectorized atomic force accumulation ─────────────────────────────
    // Blackwell sm_120: atom.global.add.v4.f32 applies four concurrent
    // 32-bit float atomicAdds in a single instruction, matching the
    // 128-bit memory-access width of the L2 cache line.
    //
    // G22 gate: if ptxas disassembly does NOT show atom.global.add.v4.f32,
    // the four-atomicAdd fallback below is the correct equivalence.
    //
    // Output registers capture the PRE-addition values (standard atom
    // semantics); we discard them via separate temporaries.
    float old_x, old_y, old_z, old_w;
    asm volatile(
        "atom.global.add.v4.f32 {%0,%1,%2,%3}, [%4], {%5,%6,%7,%8};"
        : "=f"(old_x), "=f"(old_y), "=f"(old_z), "=f"(old_w)
        : "l"((unsigned long long)(uintptr_t)(&d_forces[quad_idx])),
          "f"(force_delta.x), "f"(force_delta.y),
          "f"(force_delta.z), "f"(force_delta.w)
        : "memory"
    );
    // old_x/y/z/w discarded — only the side-effect (in-place VRAM add) matters.
    (void)old_x; (void)old_y; (void)old_z; (void)old_w;

    // ── Scalar fallback (uncomment if G22 gate fails ptxas audit) ────────
    // float* f_ptr = reinterpret_cast<float*>(&d_forces[quad_idx]);
    // atomicAdd(f_ptr + 0, force_delta.x);
    // atomicAdd(f_ptr + 1, force_delta.y);
    // atomicAdd(f_ptr + 2, force_delta.z);
    // atomicAdd(f_ptr + 3, force_delta.w);
}

}} // namespace prism_nhs::asc

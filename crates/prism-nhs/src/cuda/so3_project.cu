// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / RECT-3.1.b — SO(3) Projection Kernel (CUDA implementation)
// ═══════════════════════════════════════════════════════════════════════
//
// See `so3_project.cuh` for layout + caller contract.
//
// **Kernel summary**:
//   - One block per cluster.
//   - 128 threads/block (4 warps × 32 lanes).
//
//   Pass 1 — Centroid + AABB:
//     Each thread accumulates (sum_x, sum_y, sum_z, min_xyz, max_xyz)
//     over its strided slice of the cluster. Warp-shuffle reduce
//     followed by a final cross-warp reduce in warp 0 produces the
//     block-wide centroid (sum / n) and AABB. Result broadcast to
//     shared memory for Pass 2 to consume.
//
//   Pass 2 — Y_lm projection:
//     Each thread iterates its strided slice again, this time
//     converting each spike's offset-from-centroid into (θ, φ) and
//     evaluating Y_lm via the inline straight-line helper from
//     `sh_basis.cuh`. Per-thread accumulator `local_alm[36]` collects
//     `intensity · Y_lm`. Warp-shuffle reduce + cross-warp combine
//     yields the per-cluster `a_lm`.
//
//   Tile write:
//     Thread 0 computes `C_l = Σ_m |a_lm|²` for l=0..5 and writes the
//     384-byte ContactShellTile.
//
// **WMMA forward-declaration**: the per-thread `local_alm[36]` and
// the warp-shuffle reduction are placeholders. RECT-3.1.c rewrites
// the inner loop to stage `intensity · Y_lm` rows into shared memory
// and accumulate via `nvcuda::wmma::fragment<accumulator, 16, 16, 8,
// tf32>`. The on-tile layout (alignas(128), 64-float coefficient
// payload sized for 16×4 fragments) is unchanged.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
// ═══════════════════════════════════════════════════════════════════════

#include "so3_project.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include <cmath>

namespace prism_nhs { namespace so3_project {

// Compile-time block geometry. Anchored here so the kernel and
// host-side launch agree.
constexpr uint32_t BLOCK_SIZE   = 128u;
constexpr uint32_t WARP_SIZE    = 32u;
constexpr uint32_t WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

static_assert(BLOCK_SIZE % WARP_SIZE == 0,
    "BLOCK_SIZE must be a multiple of WARP_SIZE");
static_assert(WARPS_PER_BLOCK <= WARP_SIZE,
    "Cross-warp reduce requires WARPS_PER_BLOCK <= WARP_SIZE");

// ─────────────────────────────────────────────────────────────────────
// Warp-level reductions (full 0xFFFFFFFF mask — all lanes participate).
// ─────────────────────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_min(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v = fminf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, o));
    }
    return v;
}

// ─────────────────────────────────────────────────────────────────────
// SO(3) projection kernel — one cluster per block.
// ─────────────────────────────────────────────────────────────────────
__global__ void prism_so3_project_manifold_kernel(
    const RichSpike*  __restrict__ d_spikes,
    const uint32_t*   __restrict__ d_cluster_offsets,
    uint32_t                       n_clusters,
    const float*      __restrict__ d_k_lm,
    ContactShellTile* __restrict__ d_tiles_out,
    uint32_t                       frame_id
) {
    const uint32_t cluster_id = blockIdx.x;
    if (cluster_id >= n_clusters) return;

    const uint32_t start = d_cluster_offsets[cluster_id];
    const uint32_t end   = d_cluster_offsets[cluster_id + 1u];
    const uint32_t n     = end - start;

    const uint32_t lane = threadIdx.x & (WARP_SIZE - 1u);
    const uint32_t warp = threadIdx.x / WARP_SIZE;

    // Shared scratch:
    //   s_warp_partial[w][k] : per-warp partial of either centroid /
    //                          AABB (k=0..5) or alm coefficient
    //                          (k=0..N_COEFFS-1). Reused across
    //                          passes; a __syncthreads() guards each
    //                          repurposing.
    //   s_centroid           : block-wide broadcast of (cx, cy, cz).
    //   s_aabb               : block-wide broadcast of AABB after
    //                          Pass 1.
    __shared__ float s_warp_partial[WARPS_PER_BLOCK][N_COEFFS];
    __shared__ float s_centroid[3];
    __shared__ float s_aabb[6];  // min_xyz, max_xyz

    // ─── Empty-cluster fast path: zero the tile and exit ──────────────
    if (n == 0u) {
        if (threadIdx.x == 0) {
            ContactShellTile& tile = d_tiles_out[cluster_id];
            tile.phase             = 0u;
            tile.stream_id         = 0u;
            tile.cluster_id        = static_cast<int32_t>(cluster_id);
            tile.frame             = frame_id;
            #pragma unroll
            for (int k = 0; k < 64; ++k) tile.coefficients[k] = 0.0f;
            #pragma unroll
            for (int k = 0; k < 8;  ++k) tile.power_spectrum[k] = 0.0f;
            #pragma unroll
            for (int k = 0; k < 4;  ++k) {
                tile.aabb_min[k] = 0.0f;
                tile.aabb_max[k] = 0.0f;
            }
            tile.spike_count       = 0u;
            tile.adjudication_code = 0u;
            #pragma unroll
            for (int k = 0; k < 6;  ++k) tile.reserved[k] = 0u;
            #pragma unroll
            for (int k = 0; k < 16; ++k) tile._pad[k] = 0u;
        }
        return;
    }

    // ─── Pass 1: centroid + AABB via warp-shuffle reduce ──────────────
    float sx = 0.0f, sy = 0.0f, sz = 0.0f;
    float mn_x = +FLT_MAX, mn_y = +FLT_MAX, mn_z = +FLT_MAX;
    float mx_x = -FLT_MAX, mx_y = -FLT_MAX, mx_z = -FLT_MAX;

    for (uint32_t i = start + threadIdx.x; i < end; i += BLOCK_SIZE) {
        const RichSpike s = d_spikes[i];
        sx += s.x; sy += s.y; sz += s.z;
        mn_x = fminf(mn_x, s.x); mn_y = fminf(mn_y, s.y); mn_z = fminf(mn_z, s.z);
        mx_x = fmaxf(mx_x, s.x); mx_y = fmaxf(mx_y, s.y); mx_z = fmaxf(mx_z, s.z);
    }

    // Intra-warp reduction
    sx = warp_reduce_sum(sx);
    sy = warp_reduce_sum(sy);
    sz = warp_reduce_sum(sz);
    mn_x = warp_reduce_min(mn_x); mn_y = warp_reduce_min(mn_y); mn_z = warp_reduce_min(mn_z);
    mx_x = warp_reduce_max(mx_x); mx_y = warp_reduce_max(mx_y); mx_z = warp_reduce_max(mx_z);

    // Lane 0 of each warp publishes its partial (slots 0..5 of the
    // shared row; we reuse the same shared buffer for the alm reduce
    // in Pass 2 after a barrier).
    if (lane == 0u) {
        s_warp_partial[warp][0] = sx;
        s_warp_partial[warp][1] = sy;
        s_warp_partial[warp][2] = sz;
        s_warp_partial[warp][3] = mn_x;
        s_warp_partial[warp][4] = mn_y;
        s_warp_partial[warp][5] = mn_z;
        // mx packed into slots 6..8
        s_warp_partial[warp][6] = mx_x;
        s_warp_partial[warp][7] = mx_y;
        s_warp_partial[warp][8] = mx_z;
    }
    __syncthreads();

    // Final cross-warp reduce in warp 0. WARPS_PER_BLOCK <= WARP_SIZE
    // (asserted at compile time) so this fits within one warp.
    if (warp == 0u) {
        float lsx = 0.0f, lsy = 0.0f, lsz = 0.0f;
        float lmn_x = +FLT_MAX, lmn_y = +FLT_MAX, lmn_z = +FLT_MAX;
        float lmx_x = -FLT_MAX, lmx_y = -FLT_MAX, lmx_z = -FLT_MAX;
        if (lane < WARPS_PER_BLOCK) {
            lsx   = s_warp_partial[lane][0];
            lsy   = s_warp_partial[lane][1];
            lsz   = s_warp_partial[lane][2];
            lmn_x = s_warp_partial[lane][3];
            lmn_y = s_warp_partial[lane][4];
            lmn_z = s_warp_partial[lane][5];
            lmx_x = s_warp_partial[lane][6];
            lmx_y = s_warp_partial[lane][7];
            lmx_z = s_warp_partial[lane][8];
        }
        // 32-lane shuffle reduction; lanes >= WARPS_PER_BLOCK contribute
        // identity (0 / +FLT_MAX / -FLT_MAX) so the sum / min / max are
        // well-defined.
        lsx = warp_reduce_sum(lsx);
        lsy = warp_reduce_sum(lsy);
        lsz = warp_reduce_sum(lsz);
        lmn_x = warp_reduce_min(lmn_x); lmn_y = warp_reduce_min(lmn_y); lmn_z = warp_reduce_min(lmn_z);
        lmx_x = warp_reduce_max(lmx_x); lmx_y = warp_reduce_max(lmx_y); lmx_z = warp_reduce_max(lmx_z);

        if (lane == 0u) {
            const float inv_n = 1.0f / static_cast<float>(n);
            s_centroid[0] = lsx * inv_n;
            s_centroid[1] = lsy * inv_n;
            s_centroid[2] = lsz * inv_n;
            s_aabb[0] = lmn_x; s_aabb[1] = lmn_y; s_aabb[2] = lmn_z;
            s_aabb[3] = lmx_x; s_aabb[4] = lmx_y; s_aabb[5] = lmx_z;
        }
    }
    __syncthreads();

    const float cx = s_centroid[0];
    const float cy = s_centroid[1];
    const float cz = s_centroid[2];

    // ─── Pass 2: SH evaluation + accumulate ───────────────────────────
    float local_alm[N_COEFFS];
    #pragma unroll
    for (int k = 0; k < N_COEFFS; ++k) local_alm[k] = 0.0f;

    for (uint32_t i = start + threadIdx.x; i < end; i += BLOCK_SIZE) {
        const RichSpike s = d_spikes[i];
        const float dx = s.x - cx;
        const float dy = s.y - cy;
        const float dz = s.z - cz;
        const float r2 = dx * dx + dy * dy + dz * dz;
        // Skip spikes coincident with the centroid (θ undefined).
        // r=0 contributes the constant Y_0^0 only and would otherwise
        // produce NaN via acosf/atan2f's 0/0 branches under fast-math.
        if (r2 < 1e-12f) continue;
        const float r = sqrtf(r2);

        const float theta = acosf(fmaxf(-1.0f, fminf(1.0f, dz / r)));
        const float phi   = atan2f(dy, dx);

        // Weight: prefer 24-bit intensity; fall back to 1.0 when zero
        // so empty-intensity (test) spikes still contribute their Y_lm.
        const uint32_t intensity_24 =
            prism_nhs::rich_spike::prism_rich_spike_unpack_intensity(s.intensity_packed);
        const float weight = (intensity_24 == 0u)
            ? 1.0f
            : static_cast<float>(intensity_24);

        float Y[N_COEFFS];
        prism_nhs::sh_basis::prism_sh_eval_lmax5(theta, phi, d_k_lm, Y);

        #pragma unroll
        for (int k = 0; k < N_COEFFS; ++k) {
            local_alm[k] += weight * Y[k];
        }
    }

    // ─── Reduce local_alm[36] across the block ────────────────────────
    // (1) Intra-warp reduce per coefficient. Lane 0 writes the warp
    //     partial to the shared buffer.
    #pragma unroll
    for (int k = 0; k < N_COEFFS; ++k) {
        const float v = warp_reduce_sum(local_alm[k]);
        if (lane == 0u) {
            s_warp_partial[warp][k] = v;
        }
    }
    __syncthreads();

    // (2) Cross-warp final reduce in warp 0. Each lane in warp 0
    //     handles its assigned coefficient(s); since N_COEFFS == 36 and
    //     WARP_SIZE == 32, lanes 0..31 cover indices 0..31 and lane 0
    //     re-runs for the remaining 4 coefficients (32..35).
    if (warp == 0u) {
        // First sweep: coefficients 0..31 (one per lane).
        float v0 = 0.0f;
        if (lane < WARPS_PER_BLOCK) {
            v0 = s_warp_partial[lane][lane];  // dummy; replaced below
        }
        // Per-coefficient reduction. Each lane k reads the W-element
        // column s_warp_partial[*][k], reduces across the W warps via
        // warp shuffle, and writes the final to s_warp_partial[0][k].
        #pragma unroll
        for (int k = 0; k < N_COEFFS; ++k) {
            float v = (lane < WARPS_PER_BLOCK)
                ? s_warp_partial[lane][k]
                : 0.0f;
            v = warp_reduce_sum(v);
            if (lane == 0u) {
                s_warp_partial[0][k] = v;  // final a_lm[k]
            }
        }
        (void)v0;  // suppress unused-warning when k loop uses its own v
    }
    __syncthreads();

    // ─── Tile write (thread 0) ────────────────────────────────────────
    if (threadIdx.x == 0u) {
        ContactShellTile& tile = d_tiles_out[cluster_id];

        tile.phase      = 0u;
        tile.stream_id  = 0u;
        tile.cluster_id = static_cast<int32_t>(cluster_id);
        tile.frame      = frame_id;

        // a_lm coefficients into slots 0..35; pad 36..63 with zero so
        // RECT-3.1.c's WMMA fragment loads don't pull garbage.
        #pragma unroll
        for (int k = 0; k < 64; ++k) {
            tile.coefficients[k] = (k < N_COEFFS) ? s_warp_partial[0][k] : 0.0f;
        }

        // Power spectrum C_l = Σ_m |a_lm|² for l=0..LMAX.
        #pragma unroll
        for (int l = 0; l <= static_cast<int>(LMAX); ++l) {
            float c_l = 0.0f;
            for (int m = -l; m <= l; ++m) {
                const int idx = l * (l + 1) + m;
                const float a = s_warp_partial[0][idx];
                c_l += a * a;
            }
            tile.power_spectrum[l] = c_l;
        }
        #pragma unroll
        for (int l = static_cast<int>(LMAX) + 1; l < 8; ++l) {
            tile.power_spectrum[l] = 0.0f;
        }

        // AABB (xyz + pad).
        tile.aabb_min[0] = s_aabb[0];
        tile.aabb_min[1] = s_aabb[1];
        tile.aabb_min[2] = s_aabb[2];
        tile.aabb_min[3] = 0.0f;
        tile.aabb_max[0] = s_aabb[3];
        tile.aabb_max[1] = s_aabb[4];
        tile.aabb_max[2] = s_aabb[5];
        tile.aabb_max[3] = 0.0f;

        tile.spike_count       = n;
        tile.adjudication_code = 0u;  // pre_rank fills later
        #pragma unroll
        for (int k = 0; k < 6; ++k) tile.reserved[k] = 0u;
        #pragma unroll
        for (int k = 0; k < 16; ++k) tile._pad[k] = 0u;
    }
}

// ─────────────────────────────────────────────────────────────────────
// extern "C" host orchestration
// ─────────────────────────────────────────────────────────────────────

extern "C" {

uint32_t prism_so3_project_link_probe(void) {
    return 0x53033u;
}

cudaError_t prism_so3_project_run(
    const RichSpike*  d_spikes,
    const uint32_t*   d_cluster_offsets,
    uint32_t          n_clusters,
    const float*      d_k_lm,
    ContactShellTile* d_tiles_out,
    uint32_t          frame_id,
    cudaStream_t      stream
) {
    if (n_clusters == 0u) return cudaSuccess;
    if (d_spikes == nullptr || d_cluster_offsets == nullptr ||
        d_k_lm == nullptr || d_tiles_out == nullptr) {
        return cudaErrorInvalidValue;
    }

    prism_so3_project_manifold_kernel<<<n_clusters, BLOCK_SIZE, 0, stream>>>(
        d_spikes, d_cluster_offsets, n_clusters,
        d_k_lm, d_tiles_out, frame_id
    );
    return cudaGetLastError();
}

}  // extern "C"

}}  // namespace prism_nhs::so3_project

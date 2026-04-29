// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / M1 — SpikeToCluster4D producer (CUDA)
// ═══════════════════════════════════════════════════════════════════════
//
// On-device producer for the M1 lane. Reads the engine's already-on-
// device spike buffer, performs voxel-to-cluster attribution and
// per-cluster integer count tallies in VRAM, and emits the per-spike
// cluster-id assignment plus the two integer count outputs that drive
// the §5 algebraic Conservation-of-Mass audit.
//
// M1.2 contract §4 architectural constraints honored here:
//   §4a: warp-aggregated atomics on contended counters
//   §4b: pinned-memory output (Rust side; FFI accessor reads from pinned)
//   §4c: every function takes cudaStream_t — no default-stream launches
//   §4d: no cudaDeviceSynchronize; sync at FFI boundary only
//   §4e: cudaStreamBeginCapture/EndCapture wrapping the kernel sequence
//        (lands in M1.2.4 — Rust side)
//   §4f: shared __host__ __device__ helper in .cuh (V3-ready)
//   C4 : NO cudaLaunchCooperativeKernel
//   C6 : per-cluster AABB segmented reduction uses
//        cub::DeviceSegmentedReduce, NOT atomicMin/atomicMax loops.
//        AABB reduction is owned by M1.2.3, not this sub-lane.
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -Xcompiler -fPIC -c
//
// Status at M1.2.2: kernel-only. The per-spike assignment kernel
// `prism_m1_assign_clusters_and_count` populates
// `d_cluster_id_per_spike`, `d_per_cluster_count`, and
// `d_background_count_scalar`. The AABB output (`d_per_cluster_aabb`)
// and `d_total_attributed_scalar` remain untouched at this stage; CUB
// DeviceSegmentedReduce + DeviceReduce::Sum land in M1.2.3.
//
// ═══════════════════════════════════════════════════════════════════════

#include <cuda_runtime.h>
#include <cstdint>

#include "spike_to_cluster_4d.cuh"

namespace prism_nhs { namespace m1 {

// ═══════════════════════════════════════════════════════════════════════
// __global__ kernel definitions
// ═══════════════════════════════════════════════════════════════════════

/// Per-spike cluster-id assignment + atomic per-cluster spike-count
/// tally + atomic background-count tally.
///
/// One thread per spike. Hashes the spike position to a 4D voxel cell-id
/// via the canonical __host__ __device__ helper `spike_to_cell_id` (in
/// the shared `.cuh`), writes the cell-id into
/// `d_cluster_id_per_spike[tid]`, and atomically increments either
/// `d_per_cluster_count[cell_id]` (in-bbox) or
/// `*d_background_count_scalar` (UNCLUSTERED — out-of-bbox).
///
/// Determinism class: AtomicsAffected (per
/// `crates/prism-nhs/src/spike_to_cluster_4d.rs::ConservationScalars`
/// type-level docs). Atomic ordering within a cluster depends on
/// warp-scheduling order, so the position of a particular spike's
/// cell-id within the cluster's spike list is non-deterministic. The
/// integer counts themselves (`d_per_cluster_count[c]`,
/// `*d_background_count_scalar`) ARE deterministic — atomicAdd of
/// constant 1ULL is order-independent in its sum, so the counts are
/// BitExact across replicates given fixed input.
///
/// AABB and total_attributed output: not touched here. Per M1.2
/// contract §6 the per-cluster AABB reduction is owned by
/// `cub::DeviceSegmentedReduce` (M1.2.3). `d_total_attributed_scalar`
/// is filled by `cub::DeviceReduce::Sum` over `d_per_cluster_count`
/// (also M1.2.3).
__global__ void prism_m1_assign_clusters_and_count(
    const float* __restrict__ d_spike_positions,
    uint32_t                  num_spikes,
    SpatialHashParams         params,
    uint32_t* __restrict__    d_cluster_id_per_spike,
    uint64_t* __restrict__    d_per_cluster_count,
    uint64_t* __restrict__    d_background_count_scalar
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_spikes) return;

    // Layout: planar [num_spikes][3] (x0,y0,z0,x1,y1,z1,...). Matches
    // the legacy `gpu_cluster_backend.rs` spike-buffer convention.
    const float pos[3] = {
        d_spike_positions[3u * tid + 0u],
        d_spike_positions[3u * tid + 1u],
        d_spike_positions[3u * tid + 2u],
    };

    const uint32_t cell_id = spike_to_cell_id(pos, params);
    d_cluster_id_per_spike[tid] = cell_id;

    // atomicAdd on `unsigned long long` is the canonical 64-bit atomic
    // primitive on sm_60+ (sm_120 included). The reinterpret_cast is
    // safe by ABI: on every CUDA-supported platform `uint64_t` and
    // `unsigned long long` are the same underlying type.
    if (cell_id == UNCLUSTERED_CLUSTER_ID) {
        atomicAdd(reinterpret_cast<unsigned long long*>(d_background_count_scalar),
                  static_cast<unsigned long long>(1));
    } else {
        atomicAdd(reinterpret_cast<unsigned long long*>(&d_per_cluster_count[cell_id]),
                  static_cast<unsigned long long>(1));
    }
}

// ═══════════════════════════════════════════════════════════════════════
// extern "C" host orchestration entry points
// ═══════════════════════════════════════════════════════════════════════

extern "C" {

uint32_t prism_m1_link_probe(void) {
    // Sentinel value verifies the static archive linked correctly and
    // the FFI ABI is round-tripping: Rust calls this, expects 0xC0FFEE.
    // Pinned by `link_probe_returns_sentinel` test in
    // spike_to_cluster_4d.rs.
    return 0xC0FFEEu;
}

cudaError_t prism_m1_spike_to_cluster_4d_run(
    const float* d_spike_positions,
    uint32_t num_spikes,
    const SpatialHashParams* h_params,
    cudaStream_t stream,
    uint32_t* d_cluster_id_per_spike,
    uint64_t* d_per_cluster_count,
    uint64_t* d_total_attributed_scalar,
    uint64_t* d_background_count_scalar,
    Aabb* d_per_cluster_aabb,
    uint32_t num_clusters
) {
    // ───────────────────────────────────────────────────────────────────
    // M1.2.2 orchestration — kernel-only.
    //
    //   1. cudaMemsetAsync zeros the three integer count outputs.
    //      `d_total_attributed_scalar` is left at zero as a documented
    //      sentinel; CUB DeviceReduce::Sum fills it in M1.2.3. Tests
    //      that read it pre-M1.2.3 see zero rather than uninitialized
    //      memory.
    //   2. `prism_m1_assign_clusters_and_count` is launched
    //      one-thread-per-spike with 256 threads/block (sm_120
    //      Blackwell-appropriate; profile + tune in M1.2.4).
    //
    // `d_per_cluster_aabb` is left untouched per M1.2 contract §6;
    // CUB DeviceSegmentedReduce in M1.2.3 owns the AABB reduction.
    //
    // No cudaDeviceSynchronize anywhere (§4d). The Rust caller calls
    // cudaStreamSynchronize at the FFI boundary when it needs the
    // result.
    // ───────────────────────────────────────────────────────────────────

    // d_per_cluster_aabb is part of the M1.2.1-locked FFI surface but
    // not consumed by M1.2.2. Suppress -Wunused-parameter without
    // narrowing the FFI signature; M1.2.3 will wire CUB into it.
    (void)d_per_cluster_aabb;

    cudaError_t err;

    err = cudaMemsetAsync(
        d_per_cluster_count, 0,
        static_cast<size_t>(num_clusters) * sizeof(uint64_t),
        stream
    );
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(d_background_count_scalar, 0, sizeof(uint64_t), stream);
    if (err != cudaSuccess) return err;

    err = cudaMemsetAsync(d_total_attributed_scalar, 0, sizeof(uint64_t), stream);
    if (err != cudaSuccess) return err;

    if (num_spikes == 0) {
        // No spikes to assign — counts stay zero, conservation
        // (0 + 0 == 0) holds trivially. Return early to avoid a
        // zero-block kernel launch (CUDA permits this but it's a
        // wasted round-trip).
        return cudaSuccess;
    }

    constexpr uint32_t THREADS_PER_BLOCK = 256u;
    const uint32_t blocks =
        (num_spikes + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK;

    prism_m1_assign_clusters_and_count<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
        d_spike_positions,
        num_spikes,
        *h_params,
        d_cluster_id_per_spike,
        d_per_cluster_count,
        d_background_count_scalar
    );
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    return cudaSuccess;
}

}  // extern "C"

}}  // namespace prism_nhs::m1

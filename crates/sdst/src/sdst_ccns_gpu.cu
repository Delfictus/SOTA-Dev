/**
 * SDST GPU-Native CCNS Pipeline
 *
 * Replaces the host-side sdst_ccns_all_pockets (which downloaded 13.2M events
 * to CPU per tile and segfaulted) with a fully GPU-resident sort-reduce
 * architecture.
 *
 * Architecture:
 *   Stage 1: Avalanche size computation via CUB radix sort + boundaries (zero atomics)
 *   Stage 2: Spatial tile indexing via sort + contiguous segments
 *   Stage 3: Fused CCNS kernel with Clauset-Shalizi-Newman (CSN) truncated
 *            power-law estimator via KS statistic minimization
 *
 * Reference: Clauset, Shalizi, Newman (2009) "Power-law distributions in
 *            empirical data" (SIAM Review, Vol. 51, No. 4, pp. 661-703).
 */

#include "../include/sdst_internal.h"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <math.h>
#include <float.h>

/* ============================================================
 * Constants
 * ============================================================ */

#define CCNS_TILE_DIM    16       /* 128 / 8 voxels per tile axis */
#define CCNS_MAX_TILES   4096     /* 16^3 */
#define CCNS_BLOCK_SIZE  256
#define CCNS_TAU_STEPS   32
#define CCNS_LAMBDA_STEPS 32
#define CCNS_MAX_UNIQUE_PER_TILE 1024  /* Max unique avalanches per tile for shared mem */

/* ============================================================
 * Stage 1: Avalanche Size Computation (Zero Atomics)
 *
 * 1. Extract avalanche_id from all events into key array
 * 2. CUB radix sort by avalanche_id
 * 3. Adjacent-difference boundary kernel → avalanche sizes
 * ============================================================ */

/** Extract avalanche_id from each event into a flat key array.
 *  Also stores event index as value for later re-association. */
__global__
void kernel_extract_avalanche_ids(
    const SpikeEvent* __restrict__ events,
    uint32_t total_events,
    uint32_t* __restrict__ out_keys,     /* avalanche_id per event */
    uint32_t* __restrict__ out_values    /* event index */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;
    out_keys[tid]   = events[tid].avalanche_id;
    out_values[tid]  = tid;
}

/** After sorting by avalanche_id, compute size of each avalanche.
 *  Adjacent-difference: where id changes, write size and start. */
__global__
void kernel_avalanche_boundaries(
    const uint32_t* __restrict__ sorted_ids,
    uint32_t total_events,
    uint32_t* __restrict__ aval_id_out,     /* unique avalanche IDs */
    uint32_t* __restrict__ aval_size_out,   /* size per unique avalanche */
    uint32_t* __restrict__ aval_start_out,  /* start index in sorted array */
    uint32_t* __restrict__ n_unique         /* number of unique avalanches */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;

    bool is_start = (tid == 0) || (sorted_ids[tid] != sorted_ids[tid - 1]);
    bool is_end   = (tid == total_events - 1) || (sorted_ids[tid] != sorted_ids[tid + 1]);

    if (is_start) {
        /* Find the end of this run */
        uint32_t end = tid;
        /* We'll compute size when we hit the end; for now just mark start */
        aval_start_out[tid] = tid; /* temporary, will be compacted */
    }

    if (is_end) {
        /* Walk back to find start of this run */
        uint32_t start = tid;
        while (start > 0 && sorted_ids[start - 1] == sorted_ids[tid]) {
            start--;
        }
        uint32_t size = tid - start + 1;
        uint32_t idx = atomicAdd(n_unique, 1);
        aval_id_out[idx] = sorted_ids[tid];
        aval_size_out[idx] = size;
        aval_start_out[idx] = start;
    }
}

/* ============================================================
 * Stage 2: Spatial Tile Indexing
 *
 * 1. Compute tile_id per event from Morton-decoded voxel coords
 * 2. CUB radix sort events by tile_id
 * 3. Adjacent-difference for per-tile [start, count]
 * ============================================================ */

/** Compute tile_id for each event.
 *  TILE_DIM = 16, so tile_id = (vx/8) + (vy/8)*16 + (vz/8)*256
 *  Also copies the avalanche size (looked up from Stage 1 results). */
__global__
void kernel_compute_tile_ids(
    const SpikeEvent* __restrict__ events,
    uint32_t total_events,
    uint32_t* __restrict__ out_tile_ids,
    uint32_t* __restrict__ out_aval_ids
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;

    uint32_t vx, vy, vz;
    morton_decode(events[tid].voxel, &vx, &vy, &vz);

    uint32_t tx = vx >> 3;  /* vx / 8 */
    uint32_t ty = vy >> 3;
    uint32_t tz = vz >> 3;

    out_tile_ids[tid] = tx + ty * CCNS_TILE_DIM + tz * CCNS_TILE_DIM * CCNS_TILE_DIM;
    out_aval_ids[tid] = events[tid].avalanche_id;
}

/** After sorting by tile_id, find start and count for each tile.
 *  The is_end thread walks backward to find its own start — no cross-block
 *  dependency. This is safe because the array is sorted. */
__global__
void kernel_tile_boundaries(
    const uint32_t* __restrict__ sorted_tile_ids,
    uint32_t total_events,
    uint32_t* __restrict__ tile_start,   /* [CCNS_MAX_TILES] */
    uint32_t* __restrict__ tile_count    /* [CCNS_MAX_TILES] */
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;

    uint32_t my_tile = sorted_tile_ids[tid];
    bool is_end = (tid == total_events - 1) || (sorted_tile_ids[tid + 1] != my_tile);

    if (is_end) {
        /* Walk backward to find start of this tile's run */
        uint32_t start = tid;
        while (start > 0 && sorted_tile_ids[start - 1] == my_tile) {
            start--;
        }
        tile_start[my_tile] = start;
        tile_count[my_tile] = tid - start + 1;
    }
}

/* ============================================================
 * Stage 3: Fused CCNS Kernel with CSN Estimator
 *
 * One block per active tile. Each block:
 *   1. Loads its tile's contiguous segment of avalanche_ids
 *   2. Deduplicates in shared memory → unique avalanche sizes
 *   3. Sorts sizes via bitonic sort in shared memory
 *   4. Computes empirical CDF
 *   5. Grid searches (tau, lambda) pairs for truncated power law
 *   6. Finds minimum KS statistic via block-wide reduction
 *   7. Writes result
 * ============================================================ */

/** Result struct for one tile's CCNS analysis */
struct GpuCcnsResult {
    float    tau;
    float    lambda;
    float    ks_distance;
    uint32_t n_avalanches;
    uint32_t tile_id;
    int32_t  ccns_class;     /* 0=SOC, 1=NearCritical, 2=Barrier */
    float    druggability;
};

/** Bitonic sort step in shared memory */
__device__
void bitonic_sort_shared(uint32_t* data, uint32_t n) {
    uint32_t lane = threadIdx.x;

    for (uint32_t k = 2; k <= n; k <<= 1) {
        for (uint32_t j = k >> 1; j > 0; j >>= 1) {
            __syncthreads();
            for (uint32_t i = lane; i < n; i += blockDim.x) {
                uint32_t ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = ((i & k) == 0);
                    if (ascending ? (data[i] > data[ixj]) : (data[i] < data[ixj])) {
                        uint32_t tmp = data[i];
                        data[i] = data[ixj];
                        data[ixj] = tmp;
                    }
                }
            }
        }
    }
    __syncthreads();
}

/** Round up to next power of 2 */
__device__
uint32_t next_pow2(uint32_t v) {
    v--;
    v |= v >> 1; v |= v >> 2; v |= v >> 4;
    v |= v >> 8; v |= v >> 16;
    return v + 1;
}

/** Compute the normalizing constant for truncated power law:
 *  Z(tau, lambda, s_min, s_max) = sum_{s=s_min}^{s_max} s^(-tau) * exp(-lambda*s) */
__device__
float truncated_power_law_Z(float tau, float lambda, uint32_t s_min, uint32_t s_max) {
    float z = 0.0f;
    /* For efficiency, only sum up to s_max or 10000, whichever is smaller */
    uint32_t limit = min(s_max, (uint32_t)10000);
    for (uint32_t s = s_min; s <= limit; s++) {
        z += powf((float)s, -tau) * expf(-lambda * (float)s);
    }
    return z;
}

/** Fused CCNS kernel: one block per active tile.
 *
 *  Shared memory layout:
 *    [0 .. CCNS_MAX_UNIQUE_PER_TILE-1]  : uint32_t avalanche_ids (dedup workspace)
 *    [next .. +CCNS_MAX_UNIQUE_PER_TILE] : uint32_t sizes (sorted unique sizes)
 *    [next .. +CCNS_MAX_UNIQUE_PER_TILE] : float empirical_cdf
 *    [next .. +CCNS_BLOCK_SIZE]          : float ks_values (per-thread reduction)
 *    [next .. +CCNS_BLOCK_SIZE]          : uint32_t ks_indices (candidate index)
 */
__global__
void kernel_batched_ccns(
    const uint32_t* __restrict__ sorted_aval_ids,   /* sorted by tile_id */
    const uint32_t* __restrict__ aval_size_lookup,   /* avalanche_id → size (dense) */
    uint32_t max_aval_id,
    const uint32_t* __restrict__ tile_start,
    const uint32_t* __restrict__ tile_count,
    const uint32_t* __restrict__ active_tile_ids,    /* list of non-empty tile IDs */
    uint32_t n_active_tiles,
    GpuCcnsResult* __restrict__ out_results
) {
    if (blockIdx.x >= n_active_tiles) return;

    uint32_t tile_id = active_tile_ids[blockIdx.x];
    uint32_t seg_start = tile_start[tile_id];
    uint32_t seg_count = tile_count[tile_id];

    /* Shared memory: dynamically allocated */
    extern __shared__ uint32_t smem[];

    /* Partition shared memory */
    uint32_t* s_aval_ids = smem;                                        /* [CCNS_MAX_UNIQUE_PER_TILE] */
    uint32_t* s_sizes    = smem + CCNS_MAX_UNIQUE_PER_TILE;             /* [CCNS_MAX_UNIQUE_PER_TILE] */
    float*    s_ecdf     = (float*)(smem + 2 * CCNS_MAX_UNIQUE_PER_TILE); /* [CCNS_MAX_UNIQUE_PER_TILE] */
    float*    s_ks       = (float*)(smem + 3 * CCNS_MAX_UNIQUE_PER_TILE); /* [CCNS_BLOCK_SIZE] */
    uint32_t* s_ks_idx   = smem + 3 * CCNS_MAX_UNIQUE_PER_TILE + CCNS_BLOCK_SIZE; /* [CCNS_BLOCK_SIZE] */

    uint32_t lane = threadIdx.x;

    /* --- Step 1: Collect unique avalanche IDs in this tile --- */

    /* Initialize dedup workspace to sentinel */
    for (uint32_t i = lane; i < CCNS_MAX_UNIQUE_PER_TILE; i += blockDim.x) {
        s_aval_ids[i] = 0xFFFFFFFF;
        s_sizes[i] = 0;
    }
    __syncthreads();

    /* Load avalanche IDs from this tile's segment.
     * Use a simple hash-based dedup in shared memory. */
    __shared__ uint32_t s_n_unique;
    if (lane == 0) s_n_unique = 0;
    __syncthreads();

    /* First pass: collect unique avalanche IDs using sorted-unique approach.
     * Since events are sorted by tile_id (not by avalanche_id within tile),
     * we do a simple insert-if-not-present. */
    for (uint32_t i = lane; i < seg_count; i += blockDim.x) {
        uint32_t aid = sorted_aval_ids[seg_start + i];
        if (aid == 0) continue;  /* Skip null avalanche */

        /* Hash probe to check if already present */
        uint32_t slot = aid % CCNS_MAX_UNIQUE_PER_TILE;
        bool found = false;
        for (uint32_t probe = 0; probe < 32; probe++) {
            uint32_t s = (slot + probe) % CCNS_MAX_UNIQUE_PER_TILE;
            uint32_t existing = atomicCAS(&s_aval_ids[s], 0xFFFFFFFF, aid);
            if (existing == 0xFFFFFFFF) {
                /* We inserted it — new unique ID */
                atomicAdd(&s_n_unique, 1);
                found = true;
                break;
            } else if (existing == aid) {
                /* Already present */
                found = true;
                break;
            }
            /* Collision with different ID, continue probing */
        }
    }
    __syncthreads();

    uint32_t n_unique = s_n_unique;
    if (n_unique < 5) {
        /* Insufficient data for power-law fit */
        if (lane == 0) {
            out_results[blockIdx.x].tile_id = tile_id;
            out_results[blockIdx.x].tau = 0.0f;
            out_results[blockIdx.x].lambda = 0.0f;
            out_results[blockIdx.x].ks_distance = 1.0f;
            out_results[blockIdx.x].n_avalanches = n_unique;
            out_results[blockIdx.x].ccns_class = 2; /* Barrier */
            out_results[blockIdx.x].druggability = 0.0f;
        }
        return;
    }

    /* --- Step 2: Compact unique IDs and look up sizes --- */

    /* Compact: collect non-sentinel entries into contiguous array */
    /* Thread 0 does this serially for simplicity (n_unique ≤ 1024) */
    if (lane == 0) {
        uint32_t j = 0;
        for (uint32_t i = 0; i < CCNS_MAX_UNIQUE_PER_TILE && j < n_unique; i++) {
            if (s_aval_ids[i] != 0xFFFFFFFF) {
                uint32_t aid = s_aval_ids[i];
                s_sizes[j] = (aid <= max_aval_id) ? aval_size_lookup[aid] : 1;
                j++;
            }
        }
        /* Pad to next power of 2 for bitonic sort */
        uint32_t padded = next_pow2(n_unique);
        for (uint32_t i = n_unique; i < padded && i < CCNS_MAX_UNIQUE_PER_TILE; i++) {
            s_sizes[i] = 0xFFFFFFFF; /* sentinel sorts to end */
        }
    }
    __syncthreads();

    /* --- Step 3: Bitonic sort the unique sizes --- */

    uint32_t padded_n = next_pow2(n_unique);
    if (padded_n > CCNS_MAX_UNIQUE_PER_TILE) padded_n = CCNS_MAX_UNIQUE_PER_TILE;
    bitonic_sort_shared(s_sizes, padded_n);

    /* Remove sentinels: actual sizes are s_sizes[0 .. n_unique-1] */
    /* s_min = smallest non-zero size */
    __shared__ uint32_t s_min_size;
    __shared__ uint32_t s_max_size;
    if (lane == 0) {
        /* Find first non-zero, non-sentinel value */
        s_min_size = 1;
        s_max_size = 1;
        for (uint32_t i = 0; i < n_unique; i++) {
            if (s_sizes[i] > 0 && s_sizes[i] != 0xFFFFFFFF) {
                s_min_size = s_sizes[i];
                break;
            }
        }
        for (int i = (int)n_unique - 1; i >= 0; i--) {
            if (s_sizes[i] != 0xFFFFFFFF) {
                s_max_size = s_sizes[i];
                break;
            }
        }
        if (s_min_size < 1) s_min_size = 1;
    }
    __syncthreads();

    /* --- Step 4: Compute empirical CDF --- */
    for (uint32_t i = lane; i < n_unique; i += blockDim.x) {
        s_ecdf[i] = (float)(i + 1) / (float)n_unique;
    }
    __syncthreads();

    /* --- Step 5: Grid search over (tau, lambda) pairs --- */
    /* 32 × 32 = 1024 candidates, 4 per thread with 256 threads */

    float best_ks = FLT_MAX;
    uint32_t best_idx = 0;

    uint32_t n_candidates = CCNS_TAU_STEPS * CCNS_LAMBDA_STEPS; /* 1024 */
    uint32_t per_thread = (n_candidates + blockDim.x - 1) / blockDim.x;

    for (uint32_t c = 0; c < per_thread; c++) {
        uint32_t cand_idx = lane + c * blockDim.x;
        if (cand_idx >= n_candidates) break;

        uint32_t tau_step    = cand_idx / CCNS_LAMBDA_STEPS;
        uint32_t lambda_step = cand_idx % CCNS_LAMBDA_STEPS;

        float tau    = 1.0f + tau_step * (2.0f / (CCNS_TAU_STEPS - 1));       /* [1.0, 3.0] */
        float lambda = lambda_step * (0.1f / (CCNS_LAMBDA_STEPS - 1));        /* [0.0, 0.1] */

        /* Compute normalizing constant */
        float Z = truncated_power_law_Z(tau, lambda, s_min_size, s_max_size);
        if (Z < 1e-30f) continue;

        /* Compute theoretical CDF and KS statistic.
         * The CDF at observed size x_i = sum_{s=s_min}^{x_i} P(s) / Z.
         * We must include ALL integer sizes between observed data points,
         * not just P(x_i). Iterate incrementally using prev_s tracker. */
        float max_diff = 0.0f;
        float theoretical_cdf = 0.0f;
        uint32_t prev_s = s_min_size > 0 ? s_min_size - 1 : 0;
        for (uint32_t i = 0; i < n_unique; i++) {
            uint32_t s = s_sizes[i];
            if (s < s_min_size || s == 0 || s == 0xFFFFFFFF) continue;
            /* Sum probability mass for all sizes from prev_s+1 to s */
            uint32_t from = (prev_s < s_min_size) ? s_min_size : prev_s + 1;
            for (uint32_t ss = from; ss <= s; ss++) {
                theoretical_cdf += powf((float)ss, -tau) * expf(-lambda * (float)ss) / Z;
            }
            float diff = fabsf(theoretical_cdf - s_ecdf[i]);
            if (diff > max_diff) max_diff = diff;
            prev_s = s;
        }

        if (max_diff < best_ks) {
            best_ks = max_diff;
            best_idx = cand_idx;
        }
    }

    /* Store per-thread best into shared mem for reduction */
    s_ks[lane] = best_ks;
    s_ks_idx[lane] = best_idx;
    __syncthreads();

    /* --- Step 6: Block-wide min reduction --- */
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            if (s_ks[lane + stride] < s_ks[lane]) {
                s_ks[lane] = s_ks[lane + stride];
                s_ks_idx[lane] = s_ks_idx[lane + stride];
            }
        }
        __syncthreads();
    }

    /* --- Step 7: Thread 0 writes result --- */
    if (lane == 0) {
        uint32_t win_idx = s_ks_idx[0];
        uint32_t tau_step    = win_idx / CCNS_LAMBDA_STEPS;
        uint32_t lambda_step = win_idx % CCNS_LAMBDA_STEPS;
        float tau    = 1.0f + tau_step * (2.0f / (CCNS_TAU_STEPS - 1));
        float lambda = lambda_step * (0.1f / (CCNS_LAMBDA_STEPS - 1));

        int32_t cls;
        if (tau < 1.5f) cls = 0;       /* SOC */
        else if (tau < 2.0f) cls = 1;  /* NearCritical */
        else cls = 2;                   /* Barrier */

        /* Druggability: SOC = most druggable */
        float confidence = (n_unique >= 20) ? 1.0f : (float)n_unique / 20.0f;
        float drug = fmaxf(0.0f, (2.0f - tau) * confidence);

        out_results[blockIdx.x].tile_id = tile_id;
        out_results[blockIdx.x].tau = tau;
        out_results[blockIdx.x].lambda = lambda;
        out_results[blockIdx.x].ks_distance = s_ks[0];
        out_results[blockIdx.x].n_avalanches = n_unique;
        out_results[blockIdx.x].ccns_class = cls;
        out_results[blockIdx.x].druggability = drug;
    }
}

/* ============================================================
 * Kernel: Build dense avalanche size lookup table
 *
 * Given CUB run-length-encode output (unique_ids, run_lengths, n_runs),
 * scatter sizes into a dense array indexed by avalanche_id.
 * ============================================================ */

__global__
void kernel_build_size_lookup(
    const uint32_t* __restrict__ unique_ids,
    const uint32_t* __restrict__ run_lengths,
    uint32_t n_runs,
    uint32_t* __restrict__ size_lookup,  /* [max_aval_id + 1], pre-zeroed */
    uint32_t max_aval_id
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_runs) return;
    uint32_t aid = unique_ids[tid];
    if (aid <= max_aval_id) {
        size_lookup[aid] = run_lengths[tid];
    }
}

/* ============================================================
 * Kernel: Find active (non-empty) tiles
 * ============================================================ */

__global__
void kernel_find_active_tiles(
    const uint32_t* __restrict__ tile_count,
    uint32_t* __restrict__ active_tile_ids,
    uint32_t* __restrict__ n_active
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= CCNS_MAX_TILES) return;
    if (tile_count[tid] > 0) {
        uint32_t idx = atomicAdd(n_active, 1);
        active_tile_ids[idx] = tid;
    }
}

/* ============================================================
 * Kernel: Find max avalanche_id (single-pass reduction)
 * ============================================================ */

__global__
void kernel_find_max_aval_id(
    const uint32_t* __restrict__ aval_ids,
    uint32_t total,
    uint32_t* __restrict__ out_max
) {
    __shared__ uint32_t s_max[256];
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lane = threadIdx.x;

    uint32_t local_max = 0;
    for (uint32_t i = tid; i < total; i += gridDim.x * blockDim.x) {
        if (aval_ids[i] > local_max) local_max = aval_ids[i];
    }
    s_max[lane] = local_max;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lane < s && s_max[lane + s] > s_max[lane]) {
            s_max[lane] = s_max[lane + s];
        }
        __syncthreads();
    }
    if (lane == 0) atomicMax(out_max, s_max[0]);
}


/* ============================================================
 * Host function: sdst_ccns_all_pockets_gpu()
 *
 * Orchestrates Stage 1 → Stage 2 → Stage 3 entirely on GPU.
 * Only the final GpuCcnsResult array is copied to host.
 * ============================================================ */

extern "C"
SdstError sdst_ccns_all_pockets_gpu(
    SdstHandle handle,
    CcnsResult** out_results,
    SpatialRegion** out_regions,
    uint32_t* out_count,
    void* stream
) {
    if (!handle || !out_results || !out_regions || !out_count)
        return SDST_ERROR_INVALID_PARAM;

    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (total_events == 0) {
        *out_results = NULL;
        *out_regions = NULL;
        *out_count = 0;
        return SDST_SUCCESS;
    }

    /* ===== Stage 1: Avalanche size computation ===== */

    /* 1a. Extract avalanche_ids */
    uint32_t* d_aval_keys = NULL;
    uint32_t* d_aval_keys_sorted = NULL;
    uint32_t* d_aval_vals = NULL;
    uint32_t* d_aval_vals_sorted = NULL;

    SDST_CUDA_CHECK(cudaMalloc(&d_aval_keys, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_aval_keys_sorted, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_aval_vals, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_aval_vals_sorted, total_events * sizeof(uint32_t)));

    dim3 grid1(SDST_GRID_SIZE(total_events));
    kernel_extract_avalanche_ids<<<grid1, SDST_BLOCK_SIZE, 0, s>>>(
        ctx->d_event_buffer, total_events, d_aval_keys, d_aval_vals);
    SDST_CUDA_CHECK_KERNEL();

    /* 1b. CUB radix sort by avalanche_id */
    size_t temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(NULL, temp_bytes,
        d_aval_keys, d_aval_keys_sorted,
        d_aval_vals, d_aval_vals_sorted,
        total_events, 0, 32, s);
    void* d_temp = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
        d_aval_keys, d_aval_keys_sorted,
        d_aval_vals, d_aval_vals_sorted,
        total_events, 0, 32, s);
    SDST_CUDA_CHECK_KERNEL();
    SDST_CUDA_CHECK(cudaFree(d_temp)); d_temp = NULL;

    /* 1c. Find max avalanche_id */
    uint32_t* d_max_aval_id = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_max_aval_id, sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_max_aval_id, 0, sizeof(uint32_t), s));

    uint32_t max_grid = (total_events + 255) / 256;
    if (max_grid > 1024) max_grid = 1024;
    kernel_find_max_aval_id<<<max_grid, 256, 0, s>>>(
        d_aval_keys_sorted, total_events, d_max_aval_id);
    SDST_CUDA_CHECK_KERNEL();

    uint32_t h_max_aval_id;
    SDST_CUDA_CHECK(cudaMemcpyAsync(&h_max_aval_id, d_max_aval_id,
        sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));
    SDST_CUDA_CHECK(cudaFree(d_max_aval_id));

    if (h_max_aval_id == 0) {
        cudaFree(d_aval_keys); cudaFree(d_aval_keys_sorted);
        cudaFree(d_aval_vals); cudaFree(d_aval_vals_sorted);
        *out_results = NULL; *out_regions = NULL; *out_count = 0;
        return SDST_SUCCESS;
    }

    /* 1d. CUB run-length encode to get unique IDs and their counts (= sizes).
     *     Output arrays need at most total_events entries (one per unique ID). */
    uint32_t* d_unique_ids = NULL;
    uint32_t* d_run_lengths = NULL;
    uint32_t* d_n_runs = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_unique_ids, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_run_lengths, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_n_runs, sizeof(uint32_t)));

    temp_bytes = 0;
    cub::DeviceRunLengthEncode::Encode(NULL, temp_bytes,
        d_aval_keys_sorted, d_unique_ids, d_run_lengths, d_n_runs,
        total_events, s);
    SDST_CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceRunLengthEncode::Encode(d_temp, temp_bytes,
        d_aval_keys_sorted, d_unique_ids, d_run_lengths, d_n_runs,
        total_events, s);
    SDST_CUDA_CHECK_KERNEL();
    SDST_CUDA_CHECK(cudaFree(d_temp)); d_temp = NULL;

    uint32_t h_n_runs;
    SDST_CUDA_CHECK(cudaMemcpyAsync(&h_n_runs, d_n_runs,
        sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));
    SDST_CUDA_CHECK(cudaFree(d_n_runs));

    /* 1e. Build dense size lookup: avalanche_id → size */
    uint32_t* d_size_lookup = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_size_lookup, (h_max_aval_id + 1) * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_size_lookup, 0, (h_max_aval_id + 1) * sizeof(uint32_t), s));

    dim3 grid_lookup(SDST_GRID_SIZE(h_n_runs));
    kernel_build_size_lookup<<<grid_lookup, SDST_BLOCK_SIZE, 0, s>>>(
        d_unique_ids, d_run_lengths, h_n_runs, d_size_lookup, h_max_aval_id);
    SDST_CUDA_CHECK_KERNEL();

    /* Free Stage 1 temporaries we no longer need */
    SDST_CUDA_CHECK(cudaFree(d_aval_keys));
    SDST_CUDA_CHECK(cudaFree(d_aval_keys_sorted));
    SDST_CUDA_CHECK(cudaFree(d_aval_vals));
    SDST_CUDA_CHECK(cudaFree(d_aval_vals_sorted));
    SDST_CUDA_CHECK(cudaFree(d_unique_ids));
    SDST_CUDA_CHECK(cudaFree(d_run_lengths));

    /* ===== Stage 2: Spatial tile indexing ===== */

    uint32_t* d_tile_keys = NULL;
    uint32_t* d_tile_keys_sorted = NULL;
    uint32_t* d_tile_aval_ids = NULL;
    uint32_t* d_tile_aval_ids_sorted = NULL;

    SDST_CUDA_CHECK(cudaMalloc(&d_tile_keys, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_tile_keys_sorted, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_tile_aval_ids, total_events * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_tile_aval_ids_sorted, total_events * sizeof(uint32_t)));

    /* 2a. Compute tile_id per event */
    kernel_compute_tile_ids<<<grid1, SDST_BLOCK_SIZE, 0, s>>>(
        ctx->d_event_buffer, total_events, d_tile_keys, d_tile_aval_ids);
    SDST_CUDA_CHECK_KERNEL();

    /* 2b. CUB sort by tile_id */
    temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(NULL, temp_bytes,
        d_tile_keys, d_tile_keys_sorted,
        d_tile_aval_ids, d_tile_aval_ids_sorted,
        total_events, 0, 12, s);  /* 12 bits for tile_id (max 4095) */
    SDST_CUDA_CHECK(cudaMalloc(&d_temp, temp_bytes));
    cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes,
        d_tile_keys, d_tile_keys_sorted,
        d_tile_aval_ids, d_tile_aval_ids_sorted,
        total_events, 0, 12, s);
    SDST_CUDA_CHECK_KERNEL();
    SDST_CUDA_CHECK(cudaFree(d_temp)); d_temp = NULL;

    SDST_CUDA_CHECK(cudaFree(d_tile_keys));
    SDST_CUDA_CHECK(cudaFree(d_tile_aval_ids));

    /* 2c. Find tile boundaries */
    uint32_t* d_tile_start = NULL;
    uint32_t* d_tile_count = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_tile_start, CCNS_MAX_TILES * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_tile_count, CCNS_MAX_TILES * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_tile_start, 0, CCNS_MAX_TILES * sizeof(uint32_t), s));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_tile_count, 0, CCNS_MAX_TILES * sizeof(uint32_t), s));

    kernel_tile_boundaries<<<grid1, SDST_BLOCK_SIZE, 0, s>>>(
        d_tile_keys_sorted, total_events, d_tile_start, d_tile_count);
    SDST_CUDA_CHECK_KERNEL();

    SDST_CUDA_CHECK(cudaFree(d_tile_keys_sorted));

    /* 2d. Find active (non-empty) tiles */
    uint32_t* d_active_tiles = NULL;
    uint32_t* d_n_active = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_active_tiles, CCNS_MAX_TILES * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&d_n_active, sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemsetAsync(d_n_active, 0, sizeof(uint32_t), s));

    dim3 grid_tiles(SDST_GRID_SIZE(CCNS_MAX_TILES));
    kernel_find_active_tiles<<<grid_tiles, SDST_BLOCK_SIZE, 0, s>>>(
        d_tile_count, d_active_tiles, d_n_active);
    SDST_CUDA_CHECK_KERNEL();

    uint32_t h_n_active;
    SDST_CUDA_CHECK(cudaMemcpyAsync(&h_n_active, d_n_active,
        sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));
    SDST_CUDA_CHECK(cudaFree(d_n_active));

    if (h_n_active == 0) {
        cudaFree(d_tile_aval_ids_sorted); cudaFree(d_tile_start);
        cudaFree(d_tile_count); cudaFree(d_active_tiles); cudaFree(d_size_lookup);
        *out_results = NULL; *out_regions = NULL; *out_count = 0;
        return SDST_SUCCESS;
    }

    /* ===== Stage 3: Fused CCNS kernel ===== */

    /* Sync to ensure Stage 2 writes are visible before Stage 3 reads */
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    GpuCcnsResult* d_ccns_results = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&d_ccns_results, h_n_active * sizeof(GpuCcnsResult)));

    /* Shared memory: 3*1024 uint32_t + 256 float + 256 uint32_t
     * = 3*4096 + 1024 + 1024 = 14336 bytes */
    size_t shared_bytes = (3 * CCNS_MAX_UNIQUE_PER_TILE + CCNS_BLOCK_SIZE + CCNS_BLOCK_SIZE) * sizeof(uint32_t);

    kernel_batched_ccns<<<h_n_active, CCNS_BLOCK_SIZE, shared_bytes, s>>>(
        d_tile_aval_ids_sorted,
        d_size_lookup,
        h_max_aval_id,
        d_tile_start,
        d_tile_count,
        d_active_tiles,
        h_n_active,
        d_ccns_results
    );
    SDST_CUDA_CHECK_KERNEL();

    /* Copy results to host */
    GpuCcnsResult* h_ccns = (GpuCcnsResult*)malloc(h_n_active * sizeof(GpuCcnsResult));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_ccns, d_ccns_results,
        h_n_active * sizeof(GpuCcnsResult), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    /* Free GPU memory */
    SDST_CUDA_CHECK(cudaFree(d_tile_aval_ids_sorted));
    SDST_CUDA_CHECK(cudaFree(d_tile_start));
    SDST_CUDA_CHECK(cudaFree(d_tile_count));
    SDST_CUDA_CHECK(cudaFree(d_active_tiles));
    SDST_CUDA_CHECK(cudaFree(d_size_lookup));
    SDST_CUDA_CHECK(cudaFree(d_ccns_results));

    /* Convert GPU results to API CcnsResult + SpatialRegion */
    /* Filter: only tiles with valid tau (>0) and sufficient avalanches */
    uint32_t valid_count = 0;
    for (uint32_t i = 0; i < h_n_active; i++) {
        if (h_ccns[i].tau > 0.0f && h_ccns[i].n_avalanches >= 5) {
            valid_count++;
        }
    }

    *out_results = (CcnsResult*)malloc(valid_count * sizeof(CcnsResult));
    *out_regions = (SpatialRegion*)malloc(valid_count * sizeof(SpatialRegion));
    *out_count = valid_count;

    float spacing = ctx->config.grid_spacing;
    uint32_t j = 0;
    for (uint32_t i = 0; i < h_n_active && j < valid_count; i++) {
        GpuCcnsResult* r = &h_ccns[i];
        if (r->tau <= 0.0f || r->n_avalanches < 5) continue;

        CcnsResult* cr = &(*out_results)[j];
        cr->tau = r->tau;
        cr->tau_stderr = 0.0f; /* CSN doesn't produce a stderr; KS distance serves as quality metric */
        cr->n_avalanches = r->n_avalanches;
        cr->druggability = r->druggability;
        cr->classification = (CcnsClass)r->ccns_class;

        /* Reconstruct SpatialRegion from tile_id */
        uint32_t tid = r->tile_id;
        uint32_t tx = tid % CCNS_TILE_DIM;
        uint32_t ty = (tid / CCNS_TILE_DIM) % CCNS_TILE_DIM;
        uint32_t tz = tid / (CCNS_TILE_DIM * CCNS_TILE_DIM);

        SpatialRegion* sr = &(*out_regions)[j];
        sr->x_min = tx * 8;
        sr->x_max = tx * 8 + 7;
        sr->y_min = ty * 8;
        sr->y_max = ty * 8 + 7;
        sr->z_min = tz * 8;
        sr->z_max = tz * 8 + 7;

        /* Clamp to grid bounds */
        if (sr->x_max >= ctx->config.grid_nx) sr->x_max = ctx->config.grid_nx - 1;
        if (sr->y_max >= ctx->config.grid_ny) sr->y_max = ctx->config.grid_ny - 1;
        if (sr->z_max >= ctx->config.grid_nz) sr->z_max = ctx->config.grid_nz - 1;

        j++;
    }

    free(h_ccns);
    return SDST_SUCCESS;
}

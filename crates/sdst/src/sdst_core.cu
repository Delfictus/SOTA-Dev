/**
 * SDST Core: Hash table, spike insertion, parent detection, avalanche clustering.
 *
 * This is the foundation of the PRISM-Therm data structure. Every spike event
 * from the NHS engine enters through sdst_insert_spikes() or sdst_insert_raw(),
 * which:
 *   1. Morton-encodes the voxel coordinate
 *   2. Inserts into the open-addressing hash table
 *   3. Detects causal parent via spatial-temporal proximity
 *   4. Assigns avalanche membership via GPU union-find
 *   5. Chains events per-voxel for temporal queries
 */

#include "../include/sdst_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================
 * Error string
 * ============================================================ */

extern "C"
const char* sdst_error_string(SdstError err) {
    switch (err) {
        case SDST_SUCCESS:                  return "Success";
        case SDST_ERROR_CUDA:               return "CUDA error";
        case SDST_ERROR_OOM:                return "Out of GPU memory";
        case SDST_ERROR_INVALID_PARAM:      return "Invalid parameter";
        case SDST_ERROR_TABLE_FULL:         return "Hash table full";
        case SDST_ERROR_NOT_FOUND:          return "Not found";
        case SDST_ERROR_WAVEFRONT_OVERFLOW: return "Wavefront table overflow";
        case SDST_ERROR_STREAM_INVALID:     return "Invalid CUDA stream";
        default:                            return "Unknown error";
    }
}

/* ============================================================
 * Default config
 * ============================================================ */

extern "C"
SdstConfig sdst_default_config(void) {
    SdstConfig cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.grid_nx = 128;
    cfg.grid_ny = 128;
    cfg.grid_nz = 128;
    cfg.grid_spacing = 0.75f;

    cfg.hash_table_capacity = 1 << 22; /* 4M slots, ~2x 128³ */
    cfg.max_spike_events = 2000000;

    cfg.max_wavefronts = 65536;
    cfg.wavefront_merge_dist = 2.25f;  /* 3 voxels * 0.75Å */
    cfg.wavefront_max_dt = 5;          /* 5 timestep window */

    cfg.avalanche_spatial_cutoff = 3.75f; /* 5 voxels * 0.75Å */
    cfg.avalanche_max_gap = 3;

    /* Default 5-phase hysteresis: 14K/6K/15K/6K/14K steps */
    cfg.phase_boundaries[0] = 0;
    cfg.phase_boundaries[1] = 14000;
    cfg.phase_boundaries[2] = 20000;
    cfg.phase_boundaries[3] = 35000;
    cfg.phase_boundaries[4] = 41000;
    cfg.phase_boundaries[5] = 55000;

    cfg.ccns_soc_threshold = 1.5f;
    cfg.ccns_barrier_threshold = 2.0f;

    cfg.num_streams = 4;
    cfg.device_id = 0;

    return cfg;
}

/* ============================================================
 * Lifecycle: create / destroy / reset
 * ============================================================ */

extern "C"
SdstError sdst_create(const SdstConfig* config, SdstHandle* out_handle) {
    if (!config || !out_handle) return SDST_ERROR_INVALID_PARAM;

    SDST_CUDA_CHECK(cudaSetDevice(config->device_id));

    SdstContext* ctx = (SdstContext*)calloc(1, sizeof(SdstContext));
    if (!ctx) return SDST_ERROR_OOM;
    ctx->config = *config;

    uint32_t cap = config->hash_table_capacity;
    uint32_t max_ev = config->max_spike_events;
    uint32_t max_wf = config->max_wavefronts;
    uint32_t max_ts = config->phase_boundaries[5] + 1;
    ctx->max_timesteps = max_ts;

    /* Hash table */
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_hash_table, cap * sizeof(HashEntry)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_hash_table, 0xFF, cap * sizeof(HashEntry)));

    /* Event buffer */
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_event_buffer, max_ev * sizeof(SpikeEvent)));
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_event_count, sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_event_count, 0, sizeof(uint32_t)));

    /* Per-voxel chain */
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_voxel_chain, cap * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_voxel_chain, 0xFF, cap * sizeof(uint32_t)));

    /* Event chain (linked list next pointers) */
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_event_chain_next, max_ev * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_event_chain_next, 0xFF, max_ev * sizeof(uint32_t)));

    /* Avalanche union-find */
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_avalanche_parent, max_ev * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_avalanche_size, max_ev * sizeof(uint32_t)));

    /* Wavefront tracking */
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_wavefront_count, sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_wavefront_count, 0, sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_wavefront_stats, max_wf * sizeof(WavefrontStats)));
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_voxel_last_wavefront, cap * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_voxel_last_wavefront, 0xFF, cap * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_voxel_last_time, cap * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_voxel_last_time, 0, cap * sizeof(uint32_t)));

    /* Temporal index */
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_time_index_start, max_ts * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_time_index_count, max_ts * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_time_index_start, 0xFF, max_ts * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_time_index_count, 0, max_ts * sizeof(uint32_t)));

    /* Query buffers (one per stream) */
    ctx->query_buffer_size = SDST_QUERY_BUFFER_SIZE;
    ctx->num_streams = config->num_streams;
    ctx->d_query_buffers = (SpikeEvent**)calloc(ctx->num_streams, sizeof(SpikeEvent*));
    ctx->d_query_counts = NULL;
    SDST_CUDA_CHECK(cudaMalloc(&ctx->d_query_counts, ctx->num_streams * sizeof(uint32_t)));
    for (uint32_t i = 0; i < ctx->num_streams; i++) {
        SDST_CUDA_CHECK(cudaMalloc(&ctx->d_query_buffers[i],
                                   SDST_QUERY_BUFFER_SIZE * sizeof(SpikeEvent)));
    }

    /* CUDA streams */
    ctx->streams = (cudaStream_t*)calloc(ctx->num_streams, sizeof(cudaStream_t));
    for (uint32_t i = 0; i < ctx->num_streams; i++) {
        SDST_CUDA_CHECK(cudaStreamCreate(&ctx->streams[i]));
    }

    ctx->h_event_count = 0;
    ctx->h_wavefront_count = 0;

    *out_handle = ctx;
    return SDST_SUCCESS;
}

extern "C"
SdstError sdst_destroy(SdstHandle handle) {
    if (!handle) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;

    cudaFree(ctx->d_hash_table);
    cudaFree(ctx->d_event_buffer);
    cudaFree(ctx->d_event_count);
    cudaFree(ctx->d_voxel_chain);
    cudaFree(ctx->d_event_chain_next);
    cudaFree(ctx->d_avalanche_parent);
    cudaFree(ctx->d_avalanche_size);
    cudaFree(ctx->d_wavefront_count);
    cudaFree(ctx->d_wavefront_stats);
    cudaFree(ctx->d_voxel_last_wavefront);
    cudaFree(ctx->d_voxel_last_time);
    cudaFree(ctx->d_time_index_start);
    cudaFree(ctx->d_time_index_count);
    cudaFree(ctx->d_query_counts);

    for (uint32_t i = 0; i < ctx->num_streams; i++) {
        cudaFree(ctx->d_query_buffers[i]);
        cudaStreamDestroy(ctx->streams[i]);
    }
    free(ctx->d_query_buffers);
    free(ctx->streams);
    free(ctx);

    return SDST_SUCCESS;
}

extern "C"
SdstError sdst_reset(SdstHandle handle) {
    if (!handle) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;

    uint32_t cap = ctx->config.hash_table_capacity;
    uint32_t max_ev = ctx->config.max_spike_events;

    SDST_CUDA_CHECK(cudaMemset(ctx->d_hash_table, 0xFF, cap * sizeof(HashEntry)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_event_count, 0, sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_voxel_chain, 0xFF, cap * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_event_chain_next, 0xFF, max_ev * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_wavefront_count, 0, sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_voxel_last_wavefront, 0xFF, cap * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_voxel_last_time, 0, cap * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_time_index_start, 0xFF, ctx->max_timesteps * sizeof(uint32_t)));
    SDST_CUDA_CHECK(cudaMemset(ctx->d_time_index_count, 0, ctx->max_timesteps * sizeof(uint32_t)));

    ctx->h_event_count = 0;
    ctx->h_wavefront_count = 0;

    return SDST_SUCCESS;
}

extern "C"
SdstError sdst_event_count(SdstHandle handle, uint32_t* out_count) {
    if (!handle || !out_count) return SDST_ERROR_INVALID_PARAM;
    SDST_CUDA_CHECK(cudaMemcpy(out_count, handle->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));
    handle->h_event_count = *out_count;
    return SDST_SUCCESS;
}

extern "C"
SdstError sdst_memory_usage(SdstHandle handle, size_t* out_bytes) {
    if (!handle || !out_bytes) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    uint32_t cap = ctx->config.hash_table_capacity;
    uint32_t max_ev = ctx->config.max_spike_events;
    uint32_t max_wf = ctx->config.max_wavefronts;

    *out_bytes = 0;
    *out_bytes += cap * sizeof(HashEntry);                  /* hash table */
    *out_bytes += max_ev * sizeof(SpikeEvent);              /* event buffer */
    *out_bytes += cap * sizeof(uint32_t);                   /* voxel chain heads */
    *out_bytes += max_ev * sizeof(uint32_t);                /* event chain next */
    *out_bytes += max_ev * sizeof(uint32_t) * 2;            /* union-find */
    *out_bytes += max_wf * sizeof(WavefrontStats);          /* wavefront stats */
    *out_bytes += cap * sizeof(uint32_t) * 2;               /* voxel last wavefront/time */
    *out_bytes += ctx->max_timesteps * sizeof(uint32_t) * 2; /* temporal index */
    *out_bytes += ctx->num_streams * SDST_QUERY_BUFFER_SIZE * sizeof(SpikeEvent);

    return SDST_SUCCESS;
}

/* ============================================================
 * Kernel: Hash table insert (open addressing, linear probing)
 * ============================================================ */

__global__
void kernel_hash_insert(
    HashEntry*      hash_table,
    uint32_t        capacity,
    uint32_t*       voxel_chain,
    uint32_t*       event_chain_next,
    SpikeEvent*     event_buffer,
    uint32_t*       event_count_ptr,
    const SpikeEvent* new_events,
    uint32_t        n_events,
    uint32_t        base_event_idx
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;

    SpikeEvent ev = new_events[tid];
    uint32_t event_idx = base_event_idx + tid;
    MortonCode key = ev.voxel;

    /* Store event in buffer */
    event_buffer[event_idx] = ev;

    /* Insert into hash table with linear probing */
    uint32_t slot = sdst_hash(key, capacity);
    for (uint32_t probe = 0; probe < capacity; probe++) {
        uint32_t idx = (slot + probe) & (capacity - 1);
        uint32_t old_key = atomicCAS(&hash_table[idx].key, SDST_EMPTY_KEY, key);

        if (old_key == SDST_EMPTY_KEY || old_key == key) {
            /* Slot acquired or already ours - update voxel chain */
            uint32_t old_head = atomicExch(&voxel_chain[idx], event_idx);
            event_chain_next[event_idx] = old_head;
            hash_table[idx].head_idx = event_idx;
            return;
        }
    }
    /* Table full - should not happen with proper sizing */
}

/* ============================================================
 * Kernel: Parent spike detection via spatial-temporal proximity
 *
 * For each new spike, find the most recent spike in neighboring
 * voxels within the avalanche spatial cutoff and time gap.
 * That spike is the causal parent.
 * ============================================================ */

__global__
void kernel_detect_parents(
    SpikeEvent*     event_buffer,
    const HashEntry* hash_table,
    const uint32_t* voxel_chain,
    const uint32_t* event_chain_next,
    uint32_t        capacity,
    uint32_t        base_event_idx,
    uint32_t        n_events,
    float           spatial_cutoff_grid, /* in grid units */
    uint32_t        max_time_gap,
    uint32_t        grid_nx, uint32_t grid_ny, uint32_t grid_nz
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;

    uint32_t my_idx = base_event_idx + tid;
    SpikeEvent* my_ev = &event_buffer[my_idx];
    MortonCode my_morton = my_ev->voxel;
    uint32_t my_time = my_ev->timestamp;

    uint32_t mx, my_y, mz;
    morton_decode(my_morton, &mx, &my_y, &mz);

    /* Search radius in grid cells */
    int radius = (int)(spatial_cutoff_grid + 0.5f);
    uint32_t best_parent = 0; /* 0 = no parent (spontaneous) */
    uint32_t best_time = 0;
    float best_dist = spatial_cutoff_grid + 1.0f;

    /* Scan neighboring voxels */
    for (int dz = -radius; dz <= radius; dz++) {
        int nz = (int)mz + dz;
        if (nz < 0 || nz >= (int)grid_nz) continue;
        for (int dy = -radius; dy <= radius; dy++) {
            int ny = (int)my_y + dy;
            if (ny < 0 || ny >= (int)grid_ny) continue;
            for (int dx = -radius; dx <= radius; dx++) {
                int nx = (int)mx + dx;
                if (nx < 0 || nx >= (int)grid_nx) continue;
                if (dx == 0 && dy == 0 && dz == 0) continue;

                float dist = sqrtf((float)(dx*dx + dy*dy + dz*dz));
                if (dist > spatial_cutoff_grid) continue;

                MortonCode neighbor = morton_encode(nx, ny, nz);
                uint32_t slot = sdst_hash(neighbor, capacity);

                /* Probe for this neighbor in hash table */
                for (uint32_t probe = 0; probe < 32; probe++) {
                    uint32_t idx = (slot + probe) & (capacity - 1);
                    uint32_t stored_key = hash_table[idx].key;
                    if (stored_key == SDST_EMPTY_KEY) break;
                    if (stored_key != neighbor) continue;

                    /* Walk event chain for this voxel */
                    uint32_t chain_idx = voxel_chain[idx];
                    uint32_t walk = 0;
                    while (chain_idx != SDST_EMPTY_VALUE && walk < 16) {
                        SpikeEvent* candidate = &event_buffer[chain_idx];
                        uint32_t ct = candidate->timestamp;
                        if (ct < my_time && (my_time - ct) <= max_time_gap) {
                            /* Candidate parent: closer in time wins,
                               ties broken by spatial distance */
                            if (ct > best_time ||
                                (ct == best_time && dist < best_dist)) {
                                best_parent = chain_idx + 1; /* +1 so 0 = none */
                                best_time = ct;
                                best_dist = dist;
                            }
                        }
                        chain_idx = event_chain_next[chain_idx];
                        walk++;
                    }
                    break;
                }
            }
        }
    }

    my_ev->parent_spike = best_parent;
}

/* ============================================================
 * Kernel: Avalanche assignment via union-find
 *
 * Each spike with a parent is unioned with its parent.
 * Avalanche ID = root of the union-find tree.
 * ============================================================ */

__global__
void kernel_avalanche_union(
    SpikeEvent*     event_buffer,
    uint32_t*       uf_parent,
    uint32_t*       uf_size,
    uint32_t        base_event_idx,
    uint32_t        n_events
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;

    uint32_t my_idx = base_event_idx + tid;

    /* Initialize union-find: each event is its own root */
    uf_parent[my_idx] = my_idx;
    uf_size[my_idx] = 1;

    __syncthreads(); /* Ensure all initializations are visible */

    uint32_t parent_spike = event_buffer[my_idx].parent_spike;
    if (parent_spike > 0) {
        uint32_t parent_idx = parent_spike - 1; /* undo +1 encoding */
        uf_union(uf_parent, my_idx, parent_idx);
    }
}

/** Second pass: write final avalanche ID to each event */
__global__
void kernel_avalanche_finalize(
    SpikeEvent*     event_buffer,
    uint32_t*       uf_parent,
    uint32_t        base_event_idx,
    uint32_t        n_events
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;

    uint32_t my_idx = base_event_idx + tid;
    event_buffer[my_idx].avalanche_id = uf_find(uf_parent, my_idx);
}

/* ============================================================
 * Kernel: Temporal index update
 * ============================================================ */

__global__
void kernel_update_time_index(
    uint32_t*       time_index_start,
    uint32_t*       time_index_count,
    const SpikeEvent* new_events,
    uint32_t        n_events,
    uint32_t        base_event_idx
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;

    uint32_t ts = new_events[tid].timestamp;
    /* Atomically set start index (keep minimum) */
    atomicMin(&time_index_start[ts], base_event_idx + tid);
    atomicAdd(&time_index_count[ts], 1);
}

/* ============================================================
 * Kernel: Convert raw inputs to SpikeEvents with Morton encoding
 * ============================================================ */

__global__
void kernel_encode_raw_inputs(
    const SpikeInput* inputs,
    SpikeEvent*       events,
    uint32_t          n_events
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;

    SpikeInput in = inputs[tid];
    SpikeEvent ev;
    memset(&ev, 0, sizeof(SpikeEvent));

    ev.voxel = morton_encode(in.voxel_x, in.voxel_y, in.voxel_z);
    ev.timestamp = in.timestamp;
    ev.amplitude = float_to_f16_bits(in.amplitude);
    ev.local_temp = float_to_f16_bits(in.local_temp);
    ev.energy_gradient = float_to_f16_bits(in.energy_gradient);
    ev.solvent_exposure = float_to_f16_bits(in.solvent_exposure);
    ev.phase_id = in.phase_id;
    ev.parent_spike = 0;
    ev.avalanche_id = 0;
    ev.wavefront_id = 0;
    ev.tcl_flags = 0;

    events[tid] = ev;
}

/* ============================================================
 * Public: Insert spike events
 * ============================================================ */

extern "C"
SdstError sdst_insert_spikes(
    SdstHandle handle,
    SpikeEvent* d_events,
    uint32_t count,
    void* stream
) {
    if (!handle || !d_events || count == 0) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    /* Get current event count (base index for this batch) */
    uint32_t base_idx;
    SDST_CUDA_CHECK(cudaMemcpyAsync(&base_idx, ctx->d_event_count,
                                     sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    if (base_idx + count > ctx->config.max_spike_events) {
        return SDST_ERROR_TABLE_FULL;
    }

    /* Atomically reserve event slots */
    /* We use a temporary host-side approach; in production, use atomicAdd on device */
    uint32_t new_count = base_idx + count;
    SDST_CUDA_CHECK(cudaMemcpyAsync(ctx->d_event_count, &new_count,
                                     sizeof(uint32_t), cudaMemcpyHostToDevice, s));

    dim3 grid(SDST_GRID_SIZE(count));
    dim3 block(SDST_BLOCK_SIZE);

    float spatial_cutoff_grid = ctx->config.avalanche_spatial_cutoff / ctx->config.grid_spacing;

    /* 1. Insert into hash table and build voxel chains */
    kernel_hash_insert<<<grid, block, 0, s>>>(
        ctx->d_hash_table, ctx->config.hash_table_capacity,
        ctx->d_voxel_chain, ctx->d_event_chain_next,
        ctx->d_event_buffer, ctx->d_event_count,
        d_events, count, base_idx
    );
    SDST_CUDA_CHECK_KERNEL();

    /* 2. Detect causal parents via spatial-temporal proximity */
    kernel_detect_parents<<<grid, block, 0, s>>>(
        ctx->d_event_buffer, ctx->d_hash_table,
        ctx->d_voxel_chain, ctx->d_event_chain_next,
        ctx->config.hash_table_capacity,
        base_idx, count,
        spatial_cutoff_grid, ctx->config.avalanche_max_gap,
        ctx->config.grid_nx, ctx->config.grid_ny, ctx->config.grid_nz
    );
    SDST_CUDA_CHECK_KERNEL();

    /* 3. Avalanche clustering via union-find */
    kernel_avalanche_union<<<grid, block, 0, s>>>(
        ctx->d_event_buffer, ctx->d_avalanche_parent,
        ctx->d_avalanche_size, base_idx, count
    );
    SDST_CUDA_CHECK_KERNEL();

    kernel_avalanche_finalize<<<grid, block, 0, s>>>(
        ctx->d_event_buffer, ctx->d_avalanche_parent,
        base_idx, count
    );
    SDST_CUDA_CHECK_KERNEL();

    /* 4. Update temporal index */
    kernel_update_time_index<<<grid, block, 0, s>>>(
        ctx->d_time_index_start, ctx->d_time_index_count,
        d_events, count, base_idx
    );
    SDST_CUDA_CHECK_KERNEL();

    ctx->h_event_count = new_count;

    return SDST_SUCCESS;
}

extern "C"
SdstError sdst_insert_raw(
    SdstHandle handle,
    const SpikeInput* d_inputs,
    uint32_t count,
    void* stream
) {
    if (!handle || !d_inputs || count == 0) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    /* Allocate temporary buffer for encoded events */
    SpikeEvent* d_encoded;
    SDST_CUDA_CHECK(cudaMallocAsync(&d_encoded, count * sizeof(SpikeEvent), s));

    dim3 grid(SDST_GRID_SIZE(count));
    dim3 block(SDST_BLOCK_SIZE);

    kernel_encode_raw_inputs<<<grid, block, 0, s>>>(d_inputs, d_encoded, count);
    SDST_CUDA_CHECK_KERNEL();

    SdstError err = sdst_insert_spikes(handle, d_encoded, count, stream);

    cudaFreeAsync(d_encoded, s);
    return err;
}

/* ============================================================
 * Debug: print stats
 * ============================================================ */

extern "C"
SdstError sdst_print_stats(SdstHandle handle) {
    if (!handle) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;

    uint32_t ev_count, wf_count;
    SDST_CUDA_CHECK(cudaMemcpy(&ev_count, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));
    SDST_CUDA_CHECK(cudaMemcpy(&wf_count, ctx->d_wavefront_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    size_t mem;
    sdst_memory_usage(handle, &mem);

    printf("=== SDST Statistics ===\n");
    printf("Events:     %u / %u (%.1f%%)\n",
           ev_count, ctx->config.max_spike_events,
           100.0f * ev_count / ctx->config.max_spike_events);
    printf("Wavefronts: %u / %u\n", wf_count, ctx->config.max_wavefronts);
    printf("Hash table: %u slots\n", ctx->config.hash_table_capacity);
    printf("GPU memory: %.1f MB\n", (float)mem / (1024 * 1024));
    printf("Grid:       %u x %u x %u @ %.2f Å\n",
           ctx->config.grid_nx, ctx->config.grid_ny, ctx->config.grid_nz,
           ctx->config.grid_spacing);
    printf("=======================\n");

    return SDST_SUCCESS;
}

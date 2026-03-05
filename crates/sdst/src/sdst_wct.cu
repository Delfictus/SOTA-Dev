/**
 * SDST Wavefront Coherence Tracker (WCT)
 *
 * Tracks coherent propagation of conformational changes through the NHS grid.
 * A wavefront is a spatially and temporally correlated set of spike events
 * that represents a collective molecular motion (e.g., a pocket opening).
 *
 * Key operations:
 *   1. Wavefront ID assignment: propagate from parent spike or create new
 *   2. Velocity computation: distance/time from parent to child spike
 *   3. Coherence scoring: spatial correlation of spike directions in a neighborhood
 *   4. Wavefront lifecycle tracking: birth, propagation, death
 */

#include "../include/sdst_internal.h"

/* ============================================================
 * Kernel: Wavefront ID propagation
 *
 * Rules:
 *   - If spike has a parent within wavefront_max_dt and wavefront_merge_dist:
 *     inherit parent's wavefront ID
 *   - If spike is spontaneous (no parent): create new wavefront
 *   - Compute propagation velocity = spatial_distance / temporal_distance
 * ============================================================ */

__global__
void kernel_wavefront_propagate(
    SpikeEvent*         event_buffer,
    uint32_t*           voxel_last_wavefront,
    uint32_t*           voxel_last_time,
    const HashEntry*    hash_table,
    const uint32_t*     voxel_chain,
    uint32_t            capacity,
    uint32_t            base_idx,
    uint32_t            n_events,
    uint32_t*           wavefront_count, /* atomic counter */
    WavefrontStats*     wavefront_stats,
    uint32_t            max_wavefronts,
    float               merge_dist_grid,
    uint32_t            max_dt,
    float               grid_spacing
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;

    uint32_t my_idx = base_idx + tid;
    SpikeEvent* ev = &event_buffer[my_idx];
    MortonCode my_morton = ev->voxel;
    uint32_t my_time = ev->timestamp;
    uint32_t parent_encoded = ev->parent_spike;

    WavefrontId wf_id = 0;
    float velocity = 0.0f;

    if (parent_encoded > 0) {
        uint32_t parent_idx = parent_encoded - 1;
        SpikeEvent* parent = &event_buffer[parent_idx];

        float dist = morton_distance(my_morton, parent->voxel) * grid_spacing;
        uint32_t dt = my_time - parent->timestamp;

        if (dist <= merge_dist_grid * grid_spacing && dt <= max_dt && dt > 0) {
            /* Inherit parent's wavefront */
            wf_id = parent->wavefront_id;
            velocity = dist / (float)dt;

            /* Update wavefront stats atomically */
            if (wf_id > 0 && wf_id <= max_wavefronts) {
                atomicAdd(&wavefront_stats[wf_id - 1].spike_count, 1);
                /* Track max spatial extent */
                uint32_t ox, oy, oz;
                morton_decode(wavefront_stats[wf_id - 1].origin, &ox, &oy, &oz);
                uint32_t mx, my_y, mz;
                morton_decode(my_morton, &mx, &my_y, &mz);
                float extent = sqrtf((float)((mx-ox)*(mx-ox) + (my_y-oy)*(my_y-oy) + (mz-oz)*(mz-oz))) * grid_spacing;
                /* Atomic max float via int casting */
                uint32_t* extent_ptr = (uint32_t*)&wavefront_stats[wf_id - 1].spatial_extent;
                uint32_t old_val = *extent_ptr;
                while (extent > __uint_as_float(old_val)) {
                    uint32_t assumed = old_val;
                    old_val = atomicCAS(extent_ptr, assumed, __float_as_uint(extent));
                    if (old_val == assumed) break;
                }
            }
        }
    }

    if (wf_id == 0) {
        /* Spontaneous spike or parent too distant: create new wavefront */
        uint32_t new_id = atomicAdd(wavefront_count, 1) + 1;
        if (new_id <= max_wavefronts) {
            wf_id = new_id;
            WavefrontStats* wfs = &wavefront_stats[wf_id - 1];
            wfs->id = wf_id;
            wfs->origin = my_morton;
            wfs->birth_time = my_time;
            wfs->death_time = 0;
            wfs->spike_count = 1;
            wfs->mean_velocity = 0.0f;
            wfs->mean_coherence = 0.0f;
            wfs->spatial_extent = 0.0f;
            wfs->phase = ev->phase_id;
        }
    }

    ev->wavefront_id = wf_id;
    ev->wavefront_velocity = float_to_f16_bits(velocity);

    /* Update per-voxel last wavefront tracking */
    uint32_t slot = sdst_hash(my_morton, capacity);
    for (uint32_t probe = 0; probe < 32; probe++) {
        uint32_t idx = (slot + probe) & (capacity - 1);
        if (hash_table[idx].key == my_morton) {
            voxel_last_wavefront[idx] = wf_id;
            voxel_last_time[idx] = my_time;
            break;
        }
        if (hash_table[idx].key == SDST_EMPTY_KEY) break;
    }
}

/* ============================================================
 * Kernel: Coherence scoring
 *
 * For each spike, compute spatial coherence = fraction of neighboring
 * voxels (within merge_dist) that belong to the same wavefront.
 * High coherence = coordinated collective motion (pocket opening).
 * Low coherence = scattered noise.
 * ============================================================ */

__global__
void kernel_wavefront_coherence(
    SpikeEvent*         event_buffer,
    const uint32_t*     voxel_last_wavefront,
    const HashEntry*    hash_table,
    uint32_t            capacity,
    uint32_t            base_idx,
    uint32_t            n_events,
    float               merge_dist_grid,
    uint32_t            grid_nx, uint32_t grid_ny, uint32_t grid_nz
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_events) return;

    uint32_t my_idx = base_idx + tid;
    SpikeEvent* ev = &event_buffer[my_idx];
    WavefrontId my_wf = ev->wavefront_id;
    MortonCode my_morton = ev->voxel;

    if (my_wf == 0) {
        ev->wavefront_coherence = 0;
        return;
    }

    uint32_t mx, my_y, mz;
    morton_decode(my_morton, &mx, &my_y, &mz);

    int radius = (int)(merge_dist_grid + 0.5f);
    uint32_t same_wf = 0;
    uint32_t total_neighbors = 0;

    for (int dz = -radius; dz <= radius; dz++) {
        int nz = (int)mz + dz;
        if (nz < 0 || nz >= (int)grid_nz) continue;
        for (int dy = -radius; dy <= radius; dy++) {
            int ny = (int)my_y + dy;
            if (ny < 0 || ny >= (int)grid_ny) continue;
            for (int dx = -radius; dx <= radius; dx++) {
                if (dx == 0 && dy == 0 && dz == 0) continue;
                int nx = (int)mx + dx;
                if (nx < 0 || nx >= (int)grid_nx) continue;

                float dist = sqrtf((float)(dx*dx + dy*dy + dz*dz));
                if (dist > merge_dist_grid) continue;

                MortonCode neighbor = morton_encode(nx, ny, nz);
                uint32_t slot = sdst_hash(neighbor, capacity);

                for (uint32_t probe = 0; probe < 32; probe++) {
                    uint32_t idx = (slot + probe) & (capacity - 1);
                    if (hash_table[idx].key == neighbor) {
                        total_neighbors++;
                        if (voxel_last_wavefront[idx] == my_wf) {
                            same_wf++;
                        }
                        break;
                    }
                    if (hash_table[idx].key == SDST_EMPTY_KEY) break;
                }
            }
        }
    }

    float coherence = (total_neighbors > 0) ? (float)same_wf / (float)total_neighbors : 0.0f;
    ev->wavefront_coherence = float_to_f16_bits(coherence);
}

/* ============================================================
 * Host: Process wavefront tracking for a batch of new events
 *
 * Called from sdst_insert_spikes after parent detection.
 * ============================================================ */

extern "C"
SdstError sdst_process_wavefronts(
    SdstHandle handle,
    uint32_t base_idx,
    uint32_t n_events,
    void* stream
) {
    if (!handle) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    float merge_dist_grid = ctx->config.wavefront_merge_dist / ctx->config.grid_spacing;

    dim3 grid(SDST_GRID_SIZE(n_events));
    dim3 block(SDST_BLOCK_SIZE);

    /* Pass 1: Propagate wavefront IDs */
    kernel_wavefront_propagate<<<grid, block, 0, s>>>(
        ctx->d_event_buffer,
        ctx->d_voxel_last_wavefront,
        ctx->d_voxel_last_time,
        ctx->d_hash_table,
        ctx->d_voxel_chain,
        ctx->config.hash_table_capacity,
        base_idx, n_events,
        ctx->d_wavefront_count,
        ctx->d_wavefront_stats,
        ctx->config.max_wavefronts,
        merge_dist_grid,
        ctx->config.wavefront_max_dt,
        ctx->config.grid_spacing
    );
    SDST_CUDA_CHECK_KERNEL();

    /* Pass 2: Compute coherence scores */
    kernel_wavefront_coherence<<<grid, block, 0, s>>>(
        ctx->d_event_buffer,
        ctx->d_voxel_last_wavefront,
        ctx->d_hash_table,
        ctx->config.hash_table_capacity,
        base_idx, n_events,
        merge_dist_grid,
        ctx->config.grid_nx, ctx->config.grid_ny, ctx->config.grid_nz
    );
    SDST_CUDA_CHECK_KERNEL();

    return SDST_SUCCESS;
}

/* ============================================================
 * Host: Query wavefront statistics
 * ============================================================ */

extern "C"
SdstError sdst_wavefront_stats(
    SdstHandle handle,
    int phase_filter,
    WavefrontStats** out_stats,
    uint32_t* out_count,
    void* stream
) {
    if (!handle || !out_stats || !out_count) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t wf_count;
    SDST_CUDA_CHECK(cudaMemcpy(&wf_count, ctx->d_wavefront_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (wf_count == 0) {
        *out_stats = NULL;
        *out_count = 0;
        return SDST_SUCCESS;
    }

    /* Copy all wavefront stats to host */
    WavefrontStats* h_all = (WavefrontStats*)malloc(wf_count * sizeof(WavefrontStats));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_all, ctx->d_wavefront_stats,
                                     wf_count * sizeof(WavefrontStats),
                                     cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    if (phase_filter < 0) {
        /* Return all */
        *out_stats = h_all;
        *out_count = wf_count;
    } else {
        /* Filter by phase */
        uint32_t count = 0;
        for (uint32_t i = 0; i < wf_count; i++) {
            if (h_all[i].phase == (PhaseId)phase_filter) count++;
        }
        *out_stats = (WavefrontStats*)malloc(count * sizeof(WavefrontStats));
        *out_count = count;
        uint32_t j = 0;
        for (uint32_t i = 0; i < wf_count; i++) {
            if (h_all[i].phase == (PhaseId)phase_filter) {
                (*out_stats)[j++] = h_all[i];
            }
        }
        free(h_all);
    }

    return SDST_SUCCESS;
}

/* ============================================================
 * Host: Get wavefront path (all spikes belonging to a wavefront)
 * ============================================================ */

extern "C"
SdstError sdst_wavefront_path(
    SdstHandle handle,
    WavefrontId wavefront,
    SpikeEvent** out_events,
    uint32_t* out_count,
    void* stream
) {
    if (!handle || !out_events || !out_count) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    /* For now, copy all events to host and filter.
     * TODO: GPU-side filter with prefix sum for large event counts. */
    SpikeEvent* h_all = (SpikeEvent*)malloc(total_events * sizeof(SpikeEvent));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_all, ctx->d_event_buffer,
                                     total_events * sizeof(SpikeEvent),
                                     cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    /* Count matching events */
    uint32_t count = 0;
    for (uint32_t i = 0; i < total_events; i++) {
        if (h_all[i].wavefront_id == wavefront) count++;
    }

    *out_events = (SpikeEvent*)malloc(count * sizeof(SpikeEvent));
    *out_count = count;
    uint32_t j = 0;
    for (uint32_t i = 0; i < total_events; i++) {
        if (h_all[i].wavefront_id == wavefront) {
            (*out_events)[j++] = h_all[i];
        }
    }

    free(h_all);
    return SDST_SUCCESS;
}

/* ============================================================
 * Host: Find wavefronts that passed through a region
 * ============================================================ */

extern "C"
SdstError sdst_wavefronts_through_region(
    SdstHandle handle,
    const SpatialRegion* region,
    WavefrontStats** out_stats,
    uint32_t* out_count,
    void* stream
) {
    if (!handle || !region || !out_stats || !out_count)
        return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    /* Copy events and find unique wavefront IDs in region */
    SpikeEvent* h_all = (SpikeEvent*)malloc(total_events * sizeof(SpikeEvent));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_all, ctx->d_event_buffer,
                                     total_events * sizeof(SpikeEvent),
                                     cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    /* Collect unique wavefront IDs in region */
    uint32_t max_wf = ctx->config.max_wavefronts;
    bool* wf_seen = (bool*)calloc(max_wf + 1, sizeof(bool));

    for (uint32_t i = 0; i < total_events; i++) {
        uint32_t vx, vy, vz;
        morton_decode(h_all[i].voxel, &vx, &vy, &vz);
        if (vx >= region->x_min && vx <= region->x_max &&
            vy >= region->y_min && vy <= region->y_max &&
            vz >= region->z_min && vz <= region->z_max) {
            WavefrontId wid = h_all[i].wavefront_id;
            if (wid > 0 && wid <= max_wf) wf_seen[wid] = true;
        }
    }
    free(h_all);

    /* Fetch stats for seen wavefronts */
    uint32_t wf_count;
    SDST_CUDA_CHECK(cudaMemcpy(&wf_count, ctx->d_wavefront_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));
    WavefrontStats* h_wf = (WavefrontStats*)malloc(wf_count * sizeof(WavefrontStats));
    SDST_CUDA_CHECK(cudaMemcpy(h_wf, ctx->d_wavefront_stats,
                               wf_count * sizeof(WavefrontStats),
                               cudaMemcpyDeviceToHost));

    uint32_t count = 0;
    for (uint32_t i = 0; i < wf_count; i++) {
        if (h_wf[i].id > 0 && h_wf[i].id <= max_wf && wf_seen[h_wf[i].id]) count++;
    }

    *out_stats = (WavefrontStats*)malloc(count * sizeof(WavefrontStats));
    *out_count = count;
    uint32_t j = 0;
    for (uint32_t i = 0; i < wf_count; i++) {
        if (h_wf[i].id > 0 && h_wf[i].id <= max_wf && wf_seen[h_wf[i].id]) {
            (*out_stats)[j++] = h_wf[i];
        }
    }

    free(wf_seen);
    free(h_wf);
    return SDST_SUCCESS;
}

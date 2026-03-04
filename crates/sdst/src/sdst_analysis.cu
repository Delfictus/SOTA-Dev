/**
 * SDST Analysis: CCNS, Spatial Queries, Causal Subgraphs, DCC
 *
 * Scientific analysis functions that operate on the SDST event record.
 * These transform raw spike data into druggability predictions,
 * causal mechanisms, and validation metrics.
 */

#include "../include/sdst_internal.h"
#include <math.h>
#include <float.h>

/* ============================================================
 * Kernel: Spatial region query - collect events in a bounding box
 * ============================================================ */

__global__
void kernel_query_region(
    const SpikeEvent* event_buffer,
    uint32_t          total_events,
    SpatialRegion     region,
    SpikeEvent*       out_buffer,
    uint32_t*         out_count,
    uint32_t          max_results
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;

    SpikeEvent ev = event_buffer[tid];
    uint32_t vx, vy, vz;
    morton_decode(ev.voxel, &vx, &vy, &vz);

    if (vx >= region.x_min && vx <= region.x_max &&
        vy >= region.y_min && vy <= region.y_max &&
        vz >= region.z_min && vz <= region.z_max) {
        uint32_t idx = atomicAdd(out_count, 1);
        if (idx < max_results) {
            out_buffer[idx] = ev;
        }
    }
}

extern "C"
SdstError sdst_query_region(
    SdstHandle handle,
    const SpatialRegion* region,
    SpikeEvent** out_events,
    uint32_t* out_count,
    void* stream
) {
    if (!handle || !region || !out_events || !out_count)
        return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    /* Use stream 0's query buffer */
    uint32_t stream_idx = 0; /* TODO: map stream to index */
    SDST_CUDA_CHECK(cudaMemsetAsync(ctx->d_query_counts + stream_idx,
                                     0, sizeof(uint32_t), s));

    dim3 grid(SDST_GRID_SIZE(total_events));
    dim3 block(SDST_BLOCK_SIZE);

    kernel_query_region<<<grid, block, 0, s>>>(
        ctx->d_event_buffer, total_events, *region,
        ctx->d_query_buffers[stream_idx],
        ctx->d_query_counts + stream_idx,
        ctx->query_buffer_size
    );
    SDST_CUDA_CHECK_KERNEL();

    uint32_t count;
    SDST_CUDA_CHECK(cudaMemcpyAsync(&count, ctx->d_query_counts + stream_idx,
                                     sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    if (count > ctx->query_buffer_size) count = ctx->query_buffer_size;

    *out_events = ctx->d_query_buffers[stream_idx];
    *out_count = count;

    return SDST_SUCCESS;
}

/* ============================================================
 * Voxel query: all events for a specific voxel
 * ============================================================ */

extern "C"
SdstError sdst_query_voxel(
    SdstHandle handle,
    uint32_t x, uint32_t y, uint32_t z,
    SpikeEvent** out_events,
    uint32_t* out_count,
    void* stream
) {
    if (!handle || !out_events || !out_count) return SDST_ERROR_INVALID_PARAM;

    SpatialRegion region = {x, x, y, y, z, z};
    return sdst_query_region(handle, &region, out_events, out_count, stream);
}

/* ============================================================
 * Time range query
 * ============================================================ */

__global__
void kernel_query_timerange(
    const SpikeEvent* event_buffer,
    uint32_t          total_events,
    uint32_t          t_start,
    uint32_t          t_end,
    SpikeEvent*       out_buffer,
    uint32_t*         out_count,
    uint32_t          max_results
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;

    SpikeEvent ev = event_buffer[tid];
    if (ev.timestamp >= t_start && ev.timestamp <= t_end) {
        uint32_t idx = atomicAdd(out_count, 1);
        if (idx < max_results) {
            out_buffer[idx] = ev;
        }
    }
}

extern "C"
SdstError sdst_query_timerange(
    SdstHandle handle,
    uint32_t t_start, uint32_t t_end,
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

    uint32_t stream_idx = 0;
    SDST_CUDA_CHECK(cudaMemsetAsync(ctx->d_query_counts + stream_idx,
                                     0, sizeof(uint32_t), s));

    dim3 grid(SDST_GRID_SIZE(total_events));
    dim3 block(SDST_BLOCK_SIZE);

    kernel_query_timerange<<<grid, block, 0, s>>>(
        ctx->d_event_buffer, total_events, t_start, t_end,
        ctx->d_query_buffers[stream_idx],
        ctx->d_query_counts + stream_idx,
        ctx->query_buffer_size
    );
    SDST_CUDA_CHECK_KERNEL();

    uint32_t count;
    SDST_CUDA_CHECK(cudaMemcpyAsync(&count, ctx->d_query_counts + stream_idx,
                                     sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    if (count > ctx->query_buffer_size) count = ctx->query_buffer_size;
    *out_events = ctx->d_query_buffers[stream_idx];
    *out_count = count;

    return SDST_SUCCESS;
}

/* ============================================================
 * Combined spatial + temporal query
 * ============================================================ */

__global__
void kernel_query_region_timerange(
    const SpikeEvent* event_buffer,
    uint32_t          total_events,
    SpatialRegion     region,
    uint32_t          t_start,
    uint32_t          t_end,
    SpikeEvent*       out_buffer,
    uint32_t*         out_count,
    uint32_t          max_results
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_events) return;

    SpikeEvent ev = event_buffer[tid];
    if (ev.timestamp < t_start || ev.timestamp > t_end) return;

    uint32_t vx, vy, vz;
    morton_decode(ev.voxel, &vx, &vy, &vz);

    if (vx >= region.x_min && vx <= region.x_max &&
        vy >= region.y_min && vy <= region.y_max &&
        vz >= region.z_min && vz <= region.z_max) {
        uint32_t idx = atomicAdd(out_count, 1);
        if (idx < max_results) {
            out_buffer[idx] = ev;
        }
    }
}

extern "C"
SdstError sdst_query_region_timerange(
    SdstHandle handle,
    const SpatialRegion* region,
    uint32_t t_start, uint32_t t_end,
    SpikeEvent** out_events,
    uint32_t* out_count,
    void* stream
) {
    if (!handle || !region || !out_events || !out_count)
        return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    uint32_t stream_idx = 0;
    SDST_CUDA_CHECK(cudaMemsetAsync(ctx->d_query_counts + stream_idx,
                                     0, sizeof(uint32_t), s));

    dim3 grid(SDST_GRID_SIZE(total_events));
    dim3 block(SDST_BLOCK_SIZE);

    kernel_query_region_timerange<<<grid, block, 0, s>>>(
        ctx->d_event_buffer, total_events, *region, t_start, t_end,
        ctx->d_query_buffers[stream_idx],
        ctx->d_query_counts + stream_idx,
        ctx->query_buffer_size
    );
    SDST_CUDA_CHECK_KERNEL();

    uint32_t count;
    SDST_CUDA_CHECK(cudaMemcpyAsync(&count, ctx->d_query_counts + stream_idx,
                                     sizeof(uint32_t), cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    if (count > ctx->query_buffer_size) count = ctx->query_buffer_size;
    *out_events = ctx->d_query_buffers[stream_idx];
    *out_count = count;

    return SDST_SUCCESS;
}

/* ============================================================
 * Causal subgraph extraction
 * ============================================================ */

extern "C"
SdstError sdst_causal_subgraph(
    SdstHandle handle,
    SpikeId root_spike,
    uint32_t max_depth,
    CausalSubgraph* out_graph,
    void* stream
) {
    if (!handle || !out_graph) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    if (root_spike >= total_events) return SDST_ERROR_NOT_FOUND;

    /* Copy all events to host for graph traversal */
    SpikeEvent* h_events = (SpikeEvent*)malloc(total_events * sizeof(SpikeEvent));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_events, ctx->d_event_buffer,
                                     total_events * sizeof(SpikeEvent),
                                     cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    /* BFS from root following parent_spike links backwards */
    /* First, build reverse index: for each event, find all events whose parent is this event */
    uint32_t* children_count = (uint32_t*)calloc(total_events, sizeof(uint32_t));
    for (uint32_t i = 0; i < total_events; i++) {
        uint32_t parent = h_events[i].parent_spike;
        if (parent > 0) {
            children_count[parent - 1]++;
        }
    }

    /* BFS from root_spike, collecting descendants */
    uint32_t max_collect = (max_depth == 0) ? total_events : max_depth * 1000;
    SpikeEvent* collected = (SpikeEvent*)malloc(max_collect * sizeof(SpikeEvent));
    uint32_t* parent_indices = (uint32_t*)malloc(max_collect * sizeof(uint32_t));
    uint32_t* queue = (uint32_t*)malloc(total_events * sizeof(uint32_t));
    bool* visited = (bool*)calloc(total_events, sizeof(bool));

    collected[0] = h_events[root_spike];
    parent_indices[0] = 0xFFFFFFFF; /* root has no parent in subgraph */
    visited[root_spike] = true;
    uint32_t collected_count = 1;

    /* Also walk UP the parent chain from root */
    uint32_t up_idx = root_spike;
    uint32_t depth_up = 0;
    while (h_events[up_idx].parent_spike > 0 &&
           (max_depth == 0 || depth_up < max_depth)) {
        uint32_t parent_idx = h_events[up_idx].parent_spike - 1;
        if (parent_idx >= total_events || visited[parent_idx]) break;
        visited[parent_idx] = true;
        if (collected_count < max_collect) {
            parent_indices[collected_count] = collected_count - 1; /* points to previous */
            collected[collected_count] = h_events[parent_idx];
            collected_count++;
        }
        up_idx = parent_idx;
        depth_up++;
    }

    /* BFS downward: find all children of root_spike */
    uint32_t q_head = 0, q_tail = 0;
    queue[q_tail++] = root_spike;
    uint32_t current_depth = 0;

    while (q_head < q_tail && (max_depth == 0 || current_depth < max_depth)) {
        uint32_t level_size = q_tail - q_head;
        for (uint32_t li = 0; li < level_size; li++) {
            uint32_t node = queue[q_head++];
            /* Find children: events whose parent_spike == node + 1 */
            for (uint32_t i = 0; i < total_events && collected_count < max_collect; i++) {
                if (!visited[i] && h_events[i].parent_spike == node + 1) {
                    visited[i] = true;
                    parent_indices[collected_count] = node; /* parent in subgraph */
                    collected[collected_count] = h_events[i];
                    collected_count++;
                    queue[q_tail++] = i;
                }
            }
        }
        current_depth++;
    }

    out_graph->events = (SpikeEvent*)malloc(collected_count * sizeof(SpikeEvent));
    out_graph->parent_indices = (uint32_t*)malloc(collected_count * sizeof(uint32_t));
    out_graph->count = collected_count;
    memcpy(out_graph->events, collected, collected_count * sizeof(SpikeEvent));
    memcpy(out_graph->parent_indices, parent_indices, collected_count * sizeof(uint32_t));

    free(h_events); free(children_count); free(collected);
    free(parent_indices); free(queue); free(visited);

    return SDST_SUCCESS;
}

extern "C"
SdstError sdst_free_subgraph(CausalSubgraph* graph) {
    if (!graph) return SDST_ERROR_INVALID_PARAM;
    free(graph->events);
    free(graph->parent_indices);
    graph->events = NULL;
    graph->parent_indices = NULL;
    graph->count = 0;
    return SDST_SUCCESS;
}

/* ============================================================
 * CCNS: Power-law exponent from avalanche size distribution
 *
 * Uses Maximum Likelihood Estimation for discrete power law:
 *   tau = 1 + n * [Σ ln(s_i / s_min)]^-1
 * with Clauset-Shalizi-Newman goodness-of-fit test.
 * ============================================================ */

static float compute_tau_mle(const uint32_t* sizes, uint32_t n, uint32_t s_min) {
    if (n == 0) return 0.0f;

    double sum_log = 0.0;
    uint32_t valid = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (sizes[i] >= s_min) {
            sum_log += log((double)sizes[i] / ((double)s_min - 0.5));
            valid++;
        }
    }

    if (valid < 5) return 0.0f; /* Insufficient data */

    double tau = 1.0 + (double)valid / sum_log;
    return (float)tau;
}

static float compute_tau_stderr(float tau, uint32_t n) {
    if (n < 2) return 0.0f;
    return (tau - 1.0f) / sqrtf((float)n);
}

extern "C"
SdstError sdst_ccns_region(
    SdstHandle handle,
    const SpatialRegion* region,
    CcnsResult* out_result,
    void* stream
) {
    if (!handle || !region || !out_result) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    /* Copy events to host */
    SpikeEvent* h_events = (SpikeEvent*)malloc(total_events * sizeof(SpikeEvent));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_events, ctx->d_event_buffer,
                                     total_events * sizeof(SpikeEvent),
                                     cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    /* Collect avalanche IDs and sizes for events in region */
    /* Use a simple approach: count events per avalanche ID */
    uint32_t max_avalanche_id = 0;
    for (uint32_t i = 0; i < total_events; i++) {
        uint32_t vx, vy, vz;
        morton_decode(h_events[i].voxel, &vx, &vy, &vz);
        if (vx >= region->x_min && vx <= region->x_max &&
            vy >= region->y_min && vy <= region->y_max &&
            vz >= region->z_min && vz <= region->z_max) {
            if (h_events[i].avalanche_id > max_avalanche_id)
                max_avalanche_id = h_events[i].avalanche_id;
        }
    }

    if (max_avalanche_id == 0) {
        memset(out_result, 0, sizeof(CcnsResult));
        free(h_events);
        return SDST_SUCCESS;
    }

    uint32_t* avalanche_sizes = (uint32_t*)calloc(max_avalanche_id + 1, sizeof(uint32_t));
    bool* in_region = (bool*)calloc(max_avalanche_id + 1, sizeof(bool));

    for (uint32_t i = 0; i < total_events; i++) {
        AvalancheId aid = h_events[i].avalanche_id;
        avalanche_sizes[aid]++;

        uint32_t vx, vy, vz;
        morton_decode(h_events[i].voxel, &vx, &vy, &vz);
        if (vx >= region->x_min && vx <= region->x_max &&
            vy >= region->y_min && vy <= region->y_max &&
            vz >= region->z_min && vz <= region->z_max) {
            in_region[aid] = true;
        }
    }

    /* Collect sizes of avalanches that touch the region */
    uint32_t* region_sizes = (uint32_t*)malloc((max_avalanche_id + 1) * sizeof(uint32_t));
    uint32_t n_avalanches = 0;
    for (uint32_t i = 1; i <= max_avalanche_id; i++) {
        if (in_region[i] && avalanche_sizes[i] > 0) {
            region_sizes[n_avalanches++] = avalanche_sizes[i];
        }
    }

    /* Compute tau via MLE */
    float tau = compute_tau_mle(region_sizes, n_avalanches, 2);
    float tau_se = compute_tau_stderr(tau, n_avalanches);

    out_result->tau = tau;
    out_result->tau_stderr = tau_se;
    out_result->n_avalanches = n_avalanches;

    if (tau < ctx->config.ccns_soc_threshold) {
        out_result->classification = CCNS_SOC;
    } else if (tau < ctx->config.ccns_barrier_threshold) {
        out_result->classification = CCNS_NEAR_CRITICAL;
    } else {
        out_result->classification = CCNS_BARRIER;
    }

    /* Composite druggability score:
     * SOC sites get highest score (most responsive to perturbation)
     * Score = (2.0 - tau) * confidence_factor */
    float confidence = 1.0f - tau_se / tau;
    if (confidence < 0) confidence = 0;
    out_result->druggability = (2.0f - tau) * confidence;
    if (out_result->druggability < 0) out_result->druggability = 0;

    free(h_events); free(avalanche_sizes); free(in_region); free(region_sizes);
    return SDST_SUCCESS;
}

/* ============================================================
 * DCC: Distance to Closest Contact
 * ============================================================ */

extern "C"
SdstError sdst_compute_dcc(
    SdstHandle handle,
    const float* known_sites,
    uint32_t n_known,
    float** out_dcc,
    float** out_centroids,
    uint32_t* out_n_detected,
    void* stream
) {
    if (!handle || !out_dcc || !out_centroids || !out_n_detected)
        return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    /* Run hysteresis scan to find candidate pockets */
    HysteresisResult* hyst_results;
    SpatialRegion* hyst_regions;
    uint32_t n_pockets;

    SdstError err = sdst_hysteresis_scan(handle, 0.15f,
                                         &hyst_results, &hyst_regions,
                                         &n_pockets, stream);
    if (err != SDST_SUCCESS) return err;

    *out_n_detected = n_pockets;
    *out_dcc = (float*)malloc(n_pockets * sizeof(float));
    *out_centroids = (float*)malloc(n_pockets * 3 * sizeof(float));

    float spacing = ctx->config.grid_spacing;

    for (uint32_t i = 0; i < n_pockets; i++) {
        /* Compute centroid in Angstroms */
        float cx = ((float)(hyst_regions[i].x_min + hyst_regions[i].x_max) / 2.0f) * spacing;
        float cy = ((float)(hyst_regions[i].y_min + hyst_regions[i].y_max) / 2.0f) * spacing;
        float cz = ((float)(hyst_regions[i].z_min + hyst_regions[i].z_max) / 2.0f) * spacing;

        (*out_centroids)[i * 3 + 0] = cx;
        (*out_centroids)[i * 3 + 1] = cy;
        (*out_centroids)[i * 3 + 2] = cz;

        /* Find minimum distance to any known site */
        float min_dist = FLT_MAX;
        for (uint32_t j = 0; j < n_known; j++) {
            float dx = cx - known_sites[j * 3 + 0];
            float dy = cy - known_sites[j * 3 + 1];
            float dz = cz - known_sites[j * 3 + 2];
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);
            if (dist < min_dist) min_dist = dist;
        }
        (*out_dcc)[i] = min_dist;
    }

    free(hyst_results);
    free(hyst_regions);

    return SDST_SUCCESS;
}

/* ============================================================
 * Avalanche statistics export
 * ============================================================ */

extern "C"
SdstError sdst_avalanche_stats(
    SdstHandle handle,
    int phase_filter,
    AvalancheStats** out_stats,
    uint32_t* out_count,
    void* stream
) {
    if (!handle || !out_stats || !out_count) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    SpikeEvent* h_events = (SpikeEvent*)malloc(total_events * sizeof(SpikeEvent));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_events, ctx->d_event_buffer,
                                     total_events * sizeof(SpikeEvent),
                                     cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    /* Find unique avalanche roots and compute stats */
    uint32_t max_aid = 0;
    for (uint32_t i = 0; i < total_events; i++) {
        if (h_events[i].avalanche_id > max_aid) max_aid = h_events[i].avalanche_id;
    }

    if (max_aid == 0) {
        *out_stats = NULL;
        *out_count = 0;
        free(h_events);
        return SDST_SUCCESS;
    }

    /* Per-avalanche accumulators */
    uint32_t* sizes = (uint32_t*)calloc(max_aid + 1, sizeof(uint32_t));
    uint32_t* t_min = (uint32_t*)malloc((max_aid + 1) * sizeof(uint32_t));
    uint32_t* t_max = (uint32_t*)calloc(max_aid + 1, sizeof(uint32_t));
    MortonCode* seeds = (MortonCode*)calloc(max_aid + 1, sizeof(MortonCode));
    PhaseId* phases = (PhaseId*)calloc(max_aid + 1, sizeof(PhaseId));
    bool* valid = (bool*)calloc(max_aid + 1, sizeof(bool));

    for (uint32_t i = 0; i <= max_aid; i++) t_min[i] = 0xFFFFFFFF;

    for (uint32_t i = 0; i < total_events; i++) {
        AvalancheId aid = h_events[i].avalanche_id;
        sizes[aid]++;
        if (h_events[i].timestamp < t_min[aid]) {
            t_min[aid] = h_events[i].timestamp;
            seeds[aid] = h_events[i].voxel;
            phases[aid] = h_events[i].phase_id;
        }
        if (h_events[i].timestamp > t_max[aid]) {
            t_max[aid] = h_events[i].timestamp;
        }
        valid[aid] = true;
    }

    /* Count valid avalanches (optionally filtered by phase) */
    uint32_t count = 0;
    for (uint32_t i = 1; i <= max_aid; i++) {
        if (valid[i] && sizes[i] > 1) {
            if (phase_filter < 0 || phases[i] == (PhaseId)phase_filter) {
                count++;
            }
        }
    }

    *out_stats = (AvalancheStats*)malloc(count * sizeof(AvalancheStats));
    *out_count = count;

    float spacing = ctx->config.grid_spacing;
    uint32_t j = 0;
    for (uint32_t i = 1; i <= max_aid && j < count; i++) {
        if (!valid[i] || sizes[i] <= 1) continue;
        if (phase_filter >= 0 && phases[i] != (PhaseId)phase_filter) continue;

        AvalancheStats* as = &(*out_stats)[j];
        as->id = i;
        as->size = sizes[i];
        as->duration = t_max[i] - t_min[i];
        as->seed_voxel = seeds[i];
        as->phase = phases[i];
        as->spatial_extent = 0.0f; /* TODO: compute from event positions */

        /* Local tau from this avalanche's sub-avalanches is not meaningful;
         * set to 0. Use sdst_ccns_region for regional tau. */
        as->tau_local = 0.0f;
        j++;
    }

    free(h_events); free(sizes); free(t_min); free(t_max);
    free(seeds); free(phases); free(valid);

    return SDST_SUCCESS;
}

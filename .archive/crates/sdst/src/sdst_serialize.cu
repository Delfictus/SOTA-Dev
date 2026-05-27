/**
 * SDST Serialization: Save/Load complete state
 *
 * Binary format for checkpointing PRISM-Therm runs.
 * Enables pause/resume of multi-hour simulations.
 *
 * File format:
 *   [Header: magic, version, config]
 *   [Event count]
 *   [Event buffer: count × SpikeEvent]
 *   [Hash table: capacity × HashEntry]
 *   [Voxel chains: capacity × uint32]
 *   [Event chain next: max_events × uint32]
 *   [Avalanche parent: max_events × uint32]
 *   [Wavefront count]
 *   [Wavefront stats: count × WavefrontStats]
 *   [Voxel last wavefront: capacity × uint32]
 *   [Voxel last time: capacity × uint32]
 *   [Time index start: max_ts × uint32]
 *   [Time index count: max_ts × uint32]
 */

#include "../include/sdst_internal.h"
#include <stdio.h>

#define SDST_FILE_MAGIC   0x54534453  /* "SDST" */
#define SDST_FILE_VERSION 1

typedef struct {
    uint32_t    magic;
    uint32_t    version;
    SdstConfig  config;
} SdstFileHeader;

/* ============================================================
 * Helper: write device buffer to file
 * ============================================================ */

static SdstError write_device_to_file(FILE* f, const void* d_ptr, size_t bytes) {
    void* h_buf = malloc(bytes);
    if (!h_buf) return SDST_ERROR_OOM;

    cudaError_t err = cudaMemcpy(h_buf, d_ptr, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        free(h_buf);
        return SDST_ERROR_CUDA;
    }

    size_t written = fwrite(h_buf, 1, bytes, f);
    free(h_buf);
    return (written == bytes) ? SDST_SUCCESS : SDST_ERROR_CUDA;
}

static SdstError read_file_to_device(FILE* f, void* d_ptr, size_t bytes) {
    void* h_buf = malloc(bytes);
    if (!h_buf) return SDST_ERROR_OOM;

    size_t read_count = fread(h_buf, 1, bytes, f);
    if (read_count != bytes) {
        free(h_buf);
        return SDST_ERROR_CUDA;
    }

    cudaError_t err = cudaMemcpy(d_ptr, h_buf, bytes, cudaMemcpyHostToDevice);
    free(h_buf);
    return (err == cudaSuccess) ? SDST_SUCCESS : SDST_ERROR_CUDA;
}

/* ============================================================
 * Save
 * ============================================================ */

extern "C"
SdstError sdst_save(SdstHandle handle, const char* filepath) {
    if (!handle || !filepath) return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;

    FILE* f = fopen(filepath, "wb");
    if (!f) return SDST_ERROR_CUDA; /* reuse error code */

    /* Header */
    SdstFileHeader header;
    header.magic = SDST_FILE_MAGIC;
    header.version = SDST_FILE_VERSION;
    header.config = ctx->config;
    fwrite(&header, sizeof(header), 1, f);

    /* Event count */
    uint32_t event_count;
    cudaMemcpy(&event_count, ctx->d_event_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    fwrite(&event_count, sizeof(uint32_t), 1, f);

    uint32_t cap = ctx->config.hash_table_capacity;
    uint32_t max_ev = ctx->config.max_spike_events;

    /* Event buffer (only write actual events, not full capacity) */
    SdstError err;
    if (event_count > 0) {
        err = write_device_to_file(f, ctx->d_event_buffer, event_count * sizeof(SpikeEvent));
        if (err != SDST_SUCCESS) { fclose(f); return err; }
    }

    /* Hash table */
    err = write_device_to_file(f, ctx->d_hash_table, cap * sizeof(HashEntry));
    if (err != SDST_SUCCESS) { fclose(f); return err; }

    /* Voxel chains */
    err = write_device_to_file(f, ctx->d_voxel_chain, cap * sizeof(uint32_t));
    if (err != SDST_SUCCESS) { fclose(f); return err; }

    /* Event chain next (only for actual events) */
    if (event_count > 0) {
        err = write_device_to_file(f, ctx->d_event_chain_next, event_count * sizeof(uint32_t));
        if (err != SDST_SUCCESS) { fclose(f); return err; }
    }

    /* Avalanche parent (only for actual events) */
    if (event_count > 0) {
        err = write_device_to_file(f, ctx->d_avalanche_parent, event_count * sizeof(uint32_t));
        if (err != SDST_SUCCESS) { fclose(f); return err; }
    }

    /* Wavefront count + stats */
    uint32_t wf_count;
    cudaMemcpy(&wf_count, ctx->d_wavefront_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    fwrite(&wf_count, sizeof(uint32_t), 1, f);

    if (wf_count > 0) {
        err = write_device_to_file(f, ctx->d_wavefront_stats, wf_count * sizeof(WavefrontStats));
        if (err != SDST_SUCCESS) { fclose(f); return err; }
    }

    /* Voxel wavefront/time tracking */
    err = write_device_to_file(f, ctx->d_voxel_last_wavefront, cap * sizeof(uint32_t));
    if (err != SDST_SUCCESS) { fclose(f); return err; }
    err = write_device_to_file(f, ctx->d_voxel_last_time, cap * sizeof(uint32_t));
    if (err != SDST_SUCCESS) { fclose(f); return err; }

    /* Temporal index */
    err = write_device_to_file(f, ctx->d_time_index_start, ctx->max_timesteps * sizeof(uint32_t));
    if (err != SDST_SUCCESS) { fclose(f); return err; }
    err = write_device_to_file(f, ctx->d_time_index_count, ctx->max_timesteps * sizeof(uint32_t));
    if (err != SDST_SUCCESS) { fclose(f); return err; }

    fclose(f);
    return SDST_SUCCESS;
}

/* ============================================================
 * Load
 * ============================================================ */

extern "C"
SdstError sdst_load(const char* filepath, SdstHandle* out_handle) {
    if (!filepath || !out_handle) return SDST_ERROR_INVALID_PARAM;

    FILE* f = fopen(filepath, "rb");
    if (!f) return SDST_ERROR_NOT_FOUND;

    /* Read header */
    SdstFileHeader header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fclose(f);
        return SDST_ERROR_CUDA;
    }

    if (header.magic != SDST_FILE_MAGIC || header.version != SDST_FILE_VERSION) {
        fclose(f);
        return SDST_ERROR_INVALID_PARAM;
    }

    /* Create context with loaded config */
    SdstError err = sdst_create(&header.config, out_handle);
    if (err != SDST_SUCCESS) { fclose(f); return err; }

    SdstContext* ctx = *out_handle;
    uint32_t cap = ctx->config.hash_table_capacity;

    /* Read event count */
    uint32_t event_count;
    fread(&event_count, sizeof(uint32_t), 1, f);
    cudaMemcpy(ctx->d_event_count, &event_count, sizeof(uint32_t), cudaMemcpyHostToDevice);
    ctx->h_event_count = event_count;

    /* Read event buffer */
    if (event_count > 0) {
        err = read_file_to_device(f, ctx->d_event_buffer, event_count * sizeof(SpikeEvent));
        if (err != SDST_SUCCESS) { fclose(f); sdst_destroy(*out_handle); return err; }
    }

    /* Hash table */
    err = read_file_to_device(f, ctx->d_hash_table, cap * sizeof(HashEntry));
    if (err != SDST_SUCCESS) { fclose(f); sdst_destroy(*out_handle); return err; }

    /* Voxel chains */
    err = read_file_to_device(f, ctx->d_voxel_chain, cap * sizeof(uint32_t));
    if (err != SDST_SUCCESS) { fclose(f); sdst_destroy(*out_handle); return err; }

    /* Event chain next */
    if (event_count > 0) {
        err = read_file_to_device(f, ctx->d_event_chain_next, event_count * sizeof(uint32_t));
        if (err != SDST_SUCCESS) { fclose(f); sdst_destroy(*out_handle); return err; }
    }

    /* Avalanche parent */
    if (event_count > 0) {
        err = read_file_to_device(f, ctx->d_avalanche_parent, event_count * sizeof(uint32_t));
        if (err != SDST_SUCCESS) { fclose(f); sdst_destroy(*out_handle); return err; }
    }

    /* Wavefront count + stats */
    uint32_t wf_count;
    fread(&wf_count, sizeof(uint32_t), 1, f);
    cudaMemcpy(ctx->d_wavefront_count, &wf_count, sizeof(uint32_t), cudaMemcpyHostToDevice);
    ctx->h_wavefront_count = wf_count;

    if (wf_count > 0) {
        err = read_file_to_device(f, ctx->d_wavefront_stats, wf_count * sizeof(WavefrontStats));
        if (err != SDST_SUCCESS) { fclose(f); sdst_destroy(*out_handle); return err; }
    }

    /* Voxel wavefront/time */
    err = read_file_to_device(f, ctx->d_voxel_last_wavefront, cap * sizeof(uint32_t));
    if (err != SDST_SUCCESS) { fclose(f); sdst_destroy(*out_handle); return err; }
    err = read_file_to_device(f, ctx->d_voxel_last_time, cap * sizeof(uint32_t));
    if (err != SDST_SUCCESS) { fclose(f); sdst_destroy(*out_handle); return err; }

    /* Temporal index */
    err = read_file_to_device(f, ctx->d_time_index_start, ctx->max_timesteps * sizeof(uint32_t));
    if (err != SDST_SUCCESS) { fclose(f); sdst_destroy(*out_handle); return err; }
    err = read_file_to_device(f, ctx->d_time_index_count, ctx->max_timesteps * sizeof(uint32_t));
    if (err != SDST_SUCCESS) { fclose(f); sdst_destroy(*out_handle); return err; }

    fclose(f);
    return SDST_SUCCESS;
}

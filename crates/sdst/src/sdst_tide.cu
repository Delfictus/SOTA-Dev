/**
 * SDST TIDE: Transfer entropy-Integrated Decomposed Energetics
 *
 * Plan C analytical layer. Computes causal ΔG decomposition by:
 *   1. Mapping spike events to residue identities
 *   2. Computing transfer entropy between residue spike trains and pocket activity
 *   3. Weighting energy contributions by causal influence
 *   4. Computing Fisher information (leverage) and KL divergence (reorganization cost)
 *
 * This runs on SDST event data - no additional simulation needed.
 */

#include "../include/sdst_internal.h"
#include <math.h>

/* ============================================================
 * Transfer Entropy computation
 *
 * TE(X→Y) = Σ p(y_{t+1}, y_t^k, x_t^l) * log[ p(y_{t+1}|y_t^k, x_t^l) / p(y_{t+1}|y_t^k) ]
 *
 * For spike trains, we use a binned approach:
 *   - Discretize time into bins of width Δt
 *   - Count spikes per bin per residue (source X)
 *   - Count spikes per bin in target pocket (target Y)
 *   - Compute conditional probabilities from co-occurrence counts
 *
 * History length k = l = 1 (Markov order 1) for computational tractability.
 * ============================================================ */

#define TE_BIN_WIDTH 100    /* timesteps per bin */
#define TE_MAX_BINS  1024   /* max time bins */

/**
 * Compute transfer entropy from source residue spike train to target pocket spike train.
 *
 * Uses binary spike trains (spike present/absent per bin) with k=l=1.
 */
static float compute_transfer_entropy(
    const uint8_t* source_train, /* [n_bins] binary: 1 = spike in this bin */
    const uint8_t* target_train, /* [n_bins] binary */
    uint32_t n_bins
) {
    if (n_bins < 3) return 0.0f;

    /* Count joint and marginal probabilities for:
     *   p(y_{t+1}, y_t, x_t)  - 2x2x2 = 8 entries
     *   p(y_{t+1}, y_t)       - 2x2 = 4 entries
     *   p(y_t, x_t)           - 2x2 = 4 entries
     *   p(y_t)                - 2 entries
     */
    uint32_t count_yy1x[2][2][2] = {{{0}}}; /* [y_{t+1}][y_t][x_t] */
    uint32_t count_yy1[2][2] = {{0}};       /* [y_{t+1}][y_t] */
    uint32_t count_yx[2][2] = {{0}};        /* [y_t][x_t] */
    uint32_t total = 0;

    for (uint32_t t = 0; t < n_bins - 1; t++) {
        uint8_t yt  = target_train[t];
        uint8_t yt1 = target_train[t + 1];
        uint8_t xt  = source_train[t];

        count_yy1x[yt1][yt][xt]++;
        count_yy1[yt1][yt]++;
        count_yx[yt][xt]++;
        total++;
    }

    if (total == 0) return 0.0f;

    double te = 0.0;
    double inv_total = 1.0 / (double)total;

    for (int yt1 = 0; yt1 < 2; yt1++) {
        for (int yt = 0; yt < 2; yt++) {
            for (int xt = 0; xt < 2; xt++) {
                uint32_t n_joint = count_yy1x[yt1][yt][xt];
                if (n_joint == 0) continue;

                double p_joint = (double)n_joint * inv_total;

                /* p(y_{t+1} | y_t, x_t) = count(y_{t+1}, y_t, x_t) / count(y_t, x_t) */
                uint32_t n_yx = count_yx[yt][xt];
                if (n_yx == 0) continue;
                double p_cond_full = (double)n_joint / (double)n_yx;

                /* p(y_{t+1} | y_t) = count(y_{t+1}, y_t) / Σ_x count(y_t, x) */
                uint32_t n_yt = count_yx[yt][0] + count_yx[yt][1];
                if (n_yt == 0) continue;
                double p_cond_marginal = (double)count_yy1[yt1][yt] / (double)n_yt;

                if (p_cond_marginal > 0 && p_cond_full > 0) {
                    te += p_joint * log2(p_cond_full / p_cond_marginal);
                }
            }
        }
    }

    return (float)fmax(te, 0.0); /* TE is non-negative */
}

/**
 * Compute Fisher information for a residue's spike train.
 * Fisher info = Var[∂log p / ∂θ] ≈ sensitivity of pocket response to residue activity.
 *
 * Approximation: variance of the local transfer entropy across time windows.
 * High Fisher info = small changes in residue activity cause large changes in pocket response.
 * These are the leverage points for drug design.
 */
static float compute_fisher_info(
    const uint8_t* source_train,
    const uint8_t* target_train,
    uint32_t n_bins,
    uint32_t window_size /* bins per window */
) {
    if (n_bins < window_size * 2) return 0.0f;

    uint32_t n_windows = n_bins / window_size;
    if (n_windows < 2) return 0.0f;

    float* window_te = (float*)calloc(n_windows, sizeof(float));

    for (uint32_t w = 0; w < n_windows; w++) {
        uint32_t start = w * window_size;
        uint32_t end = start + window_size;
        if (end > n_bins) end = n_bins;
        uint32_t len = end - start;

        window_te[w] = compute_transfer_entropy(
            source_train + start,
            target_train + start,
            len
        );
    }

    /* Compute variance */
    double mean = 0;
    for (uint32_t w = 0; w < n_windows; w++) mean += window_te[w];
    mean /= n_windows;

    double var = 0;
    for (uint32_t w = 0; w < n_windows; w++) {
        double diff = window_te[w] - mean;
        var += diff * diff;
    }
    var /= (n_windows - 1);

    free(window_te);
    return (float)var;
}

/**
 * Compute KL divergence between a residue's spike distribution in heating vs cooling.
 * This measures the conformational reorganization cost:
 *   D_KL(P_heating || P_cooling)
 *
 * High KL = the residue behaves very differently during heating vs cooling = high reorganization.
 */
static float compute_kl_divergence(
    const uint8_t* train,
    const PhaseId* phase_per_bin,
    uint32_t n_bins
) {
    /* Count spike rates in heating (phases 0,1) vs cooling (3,4) */
    uint32_t heat_spikes = 0, heat_bins = 0;
    uint32_t cool_spikes = 0, cool_bins = 0;

    for (uint32_t t = 0; t < n_bins; t++) {
        PhaseId p = phase_per_bin[t];
        if (p == 0 || p == 1) {
            heat_bins++;
            if (train[t]) heat_spikes++;
        } else if (p == 3 || p == 4) {
            cool_bins++;
            if (train[t]) cool_spikes++;
        }
    }

    if (heat_bins == 0 || cool_bins == 0) return 0.0f;

    /* Spike probability in each regime */
    double p_heat = (double)heat_spikes / (double)heat_bins;
    double p_cool = (double)cool_spikes / (double)cool_bins;

    /* Clamp to avoid log(0) */
    double eps = 1e-10;
    p_heat = fmax(p_heat, eps);
    p_cool = fmax(p_cool, eps);
    double q_heat = 1.0 - p_heat;
    double q_cool = 1.0 - p_cool;
    q_heat = fmax(q_heat, eps);
    q_cool = fmax(q_cool, eps);

    /* KL(heating || cooling) for Bernoulli */
    double kl = p_heat * log(p_heat / p_cool) + q_heat * log(q_heat / q_cool);

    return (float)fmax(kl, 0.0);
}

/* ============================================================
 * Public: TIDE decomposition
 * ============================================================ */

extern "C"
SdstError sdst_tide_decomposition(
    SdstHandle handle,
    const SpatialRegion* pocket_region,
    const uint32_t* d_residue_map,
    uint32_t n_residues,
    TideDecomposition** out_decomp,
    uint32_t* out_count,
    void* stream
) {
    if (!handle || !pocket_region || !d_residue_map || !out_decomp || !out_count)
        return SDST_ERROR_INVALID_PARAM;
    SdstContext* ctx = handle;
    cudaStream_t s = stream ? (cudaStream_t)stream : 0;

    uint32_t total_events;
    SDST_CUDA_CHECK(cudaMemcpy(&total_events, ctx->d_event_count,
                               sizeof(uint32_t), cudaMemcpyDeviceToHost));

    /* Copy events and residue map to host */
    SpikeEvent* h_events = (SpikeEvent*)malloc(total_events * sizeof(SpikeEvent));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_events, ctx->d_event_buffer,
                                     total_events * sizeof(SpikeEvent),
                                     cudaMemcpyDeviceToHost, s));

    uint32_t cap = ctx->config.hash_table_capacity;
    uint32_t* h_resmap = (uint32_t*)malloc(cap * sizeof(uint32_t));
    SDST_CUDA_CHECK(cudaMemcpyAsync(h_resmap, d_residue_map,
                                     cap * sizeof(uint32_t),
                                     cudaMemcpyDeviceToHost, s));
    SDST_CUDA_CHECK(cudaStreamSynchronize(s));

    /* Determine time binning */
    uint32_t max_ts = ctx->config.phase_boundaries[5];
    uint32_t n_bins = (max_ts + TE_BIN_WIDTH - 1) / TE_BIN_WIDTH;
    if (n_bins > TE_MAX_BINS) n_bins = TE_MAX_BINS;

    /* Build pocket spike train (target Y) */
    uint8_t* pocket_train = (uint8_t*)calloc(n_bins, sizeof(uint8_t));
    PhaseId* phase_per_bin = (PhaseId*)calloc(n_bins, sizeof(PhaseId));

    /* Assign phase per bin */
    for (uint32_t b = 0; b < n_bins; b++) {
        uint32_t t = b * TE_BIN_WIDTH;
        for (int p = 4; p >= 0; p--) {
            if (t >= ctx->config.phase_boundaries[p]) {
                phase_per_bin[b] = (PhaseId)p;
                break;
            }
        }
    }

    /* Fill pocket train */
    for (uint32_t i = 0; i < total_events; i++) {
        uint32_t vx, vy, vz;
        morton_decode(h_events[i].voxel, &vx, &vy, &vz);
        if (vx >= pocket_region->x_min && vx <= pocket_region->x_max &&
            vy >= pocket_region->y_min && vy <= pocket_region->y_max &&
            vz >= pocket_region->z_min && vz <= pocket_region->z_max) {
            uint32_t bin = h_events[i].timestamp / TE_BIN_WIDTH;
            if (bin < n_bins) pocket_train[bin] = 1;
        }
    }

    /* Build per-residue spike trains */
    uint8_t** residue_trains = (uint8_t**)calloc(n_residues, sizeof(uint8_t*));
    uint32_t* residue_spike_counts = (uint32_t*)calloc(n_residues, sizeof(uint32_t));
    uint32_t* residue_causal_counts = (uint32_t*)calloc(n_residues, sizeof(uint32_t));

    for (uint32_t r = 0; r < n_residues; r++) {
        residue_trains[r] = (uint8_t*)calloc(n_bins, sizeof(uint8_t));
    }

    /* Map each spike event to its residue via the Morton→residue map */
    HashEntry* h_hash = (HashEntry*)malloc(cap * sizeof(HashEntry));
    SDST_CUDA_CHECK(cudaMemcpy(h_hash, ctx->d_hash_table,
                               cap * sizeof(HashEntry), cudaMemcpyDeviceToHost));

    for (uint32_t i = 0; i < total_events; i++) {
        MortonCode mc = h_events[i].voxel;
        uint32_t slot = sdst_hash(mc, cap);

        for (uint32_t probe = 0; probe < 32; probe++) {
            uint32_t idx = (slot + probe) & (cap - 1);
            if (h_hash[idx].key == mc) {
                uint32_t res_id = h_resmap[idx];
                if (res_id < n_residues) {
                    uint32_t bin = h_events[i].timestamp / TE_BIN_WIDTH;
                    if (bin < n_bins) {
                        residue_trains[res_id][bin] = 1;
                        residue_spike_counts[res_id]++;
                    }

                    /* Check if this spike is causally connected to pocket
                     * (shares avalanche ID with any pocket spike) */
                    /* Simplified: count spikes from this residue that are
                     * in pocket-touching avalanches */
                    uint32_t vx, vy, vz;
                    morton_decode(mc, &vx, &vy, &vz);
                    if (vx >= pocket_region->x_min && vx <= pocket_region->x_max &&
                        vy >= pocket_region->y_min && vy <= pocket_region->y_max &&
                        vz >= pocket_region->z_min && vz <= pocket_region->z_max) {
                        residue_causal_counts[res_id]++;
                    }
                }
                break;
            }
            if (h_hash[idx].key == SDST_EMPTY_KEY) break;
        }
    }

    /* Compute TIDE metrics for each residue */
    *out_decomp = (TideDecomposition*)malloc(n_residues * sizeof(TideDecomposition));
    uint32_t active_count = 0;

    for (uint32_t r = 0; r < n_residues; r++) {
        if (residue_spike_counts[r] == 0) continue;

        TideDecomposition* td = &(*out_decomp)[active_count];
        td->residue_id = r;
        td->n_causal_spikes = residue_causal_counts[r];

        /* Transfer entropy: causal influence of this residue on pocket */
        td->transfer_entropy = compute_transfer_entropy(
            residue_trains[r], pocket_train, n_bins
        );

        /* Fisher information: sensitivity / leverage */
        td->fisher_info = compute_fisher_info(
            residue_trains[r], pocket_train, n_bins, 10
        );

        /* KL divergence: conformational reorganization cost */
        td->kl_divergence = compute_kl_divergence(
            residue_trains[r], phase_per_bin, n_bins
        );

        /* Causal ΔG: TE-weighted energy contribution
         * Higher TE = more causal influence on pocket = larger ΔG contribution
         * Sign: negative TE-weighted contribution = stabilizing */
        float mean_energy = 0;
        uint32_t energy_count = 0;
        for (uint32_t i = 0; i < total_events; i++) {
            MortonCode mc = h_events[i].voxel;
            uint32_t slot2 = sdst_hash(mc, cap);
            for (uint32_t p = 0; p < 32; p++) {
                uint32_t idx = (slot2 + p) & (cap - 1);
                if (h_hash[idx].key == mc && h_resmap[idx] == r) {
                    mean_energy += f16_bits_to_float(h_events[i].energy_gradient);
                    energy_count++;
                    break;
                }
                if (h_hash[idx].key == SDST_EMPTY_KEY) break;
            }
        }
        if (energy_count > 0) mean_energy /= energy_count;

        td->causal_dG = -td->transfer_entropy * mean_energy;

        active_count++;
    }

    *out_count = active_count;

    /* Cleanup */
    for (uint32_t r = 0; r < n_residues; r++) free(residue_trains[r]);
    free(residue_trains);
    free(residue_spike_counts);
    free(residue_causal_counts);
    free(pocket_train);
    free(phase_per_bin);
    free(h_events);
    free(h_resmap);
    free(h_hash);

    return SDST_SUCCESS;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PRISM-TWIN v3.0 — Protocol Director Kernel
// ═══════════════════════════════════════════════════════════════════════════════
//
// Purpose: GPU-resident protocol state machine for CUDA Graph execution.
//
// The Director kernel runs BEFORE physics kernels on the SAME stream.
// CUDA stream ordering guarantees the Director's writes to ProtocolState
// are visible to the physics kernels — no explicit synchronization needed.
//
// Gate 0: ProtocolState struct + Director kernel (standalone validation)
// Gate 1: Physics kernel reads from ProtocolState (replaces scalar params)
// Gate 2: Full CUDA Graph capture (Director -> Physics -> Compact -> Adapt)
//
// ═══════════════════════════════════════════════════════════════════════════════

#include "protocol_state.cuh"

// Reference temperature for cryogenic friction scaling (K)
#define T_REF 300.0f
// Minimum temperature clamp to prevent division by zero
#define T_MIN 10.0f

// ─────────────────────────────────────────────────────────────────────────────
// protocol_director — single-thread state machine, runs BEFORE physics each step
// ─────────────────────────────────────────────────────────────────────────────
//
// Launch: <<<1, 1>>> — one thread, one block.
// Cost: ~20 ns per invocation (negligible vs physics kernel).
//
// Updates: current_step, current_temperature, effective_gamma, dt,
//          uv_burst_active, uv_wavelength_nm, uv_target_idx,
//          fused_step_counter, coupling_phase.

extern "C" __global__ void protocol_director(ProtocolState* __restrict__ state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // ── Increment step ──
    state->current_step += 1;
    unsigned int step = state->current_step;
    int s = (int)step;

    // ══════════════════════════════════════════════════════════════════════════
    // CCNS 5-Phase Temperature State Machine
    // ══════════════════════════════════════════════════════════════════════════
    //
    // Phase 1: cold_hold    [0, cold_hold_end)             → start_temp
    // Phase 2: ramp_up      [cold_hold_end, ramp_end)      → start_temp → end_temp
    // Phase 3: warm_hold    [ramp_end, warm_hold_end)      → end_temp
    // Phase 4: ramp_down    [warm_hold_end, ramp_down_end) → end_temp → start_temp
    // Phase 5: cold_return  [ramp_down_end, ...)           → start_temp

    int is_hold_phase;  // needed for adaptive dt

    if (s < state->cold_hold_end) {
        state->current_temperature = state->start_temp;
        is_hold_phase = 1;
    } else if (s < state->ramp_end) {
        int ramp_length = state->ramp_end - state->cold_hold_end;
        float progress = (float)(s - state->cold_hold_end) / (float)ramp_length;
        state->current_temperature = state->start_temp
            + progress * (state->end_temp - state->start_temp);
        is_hold_phase = 0;
    } else if (s < state->warm_hold_end) {
        state->current_temperature = state->end_temp;
        is_hold_phase = 1;
    } else if (s < state->ramp_down_end) {
        int ramp_length = state->ramp_down_end - state->warm_hold_end;
        float progress = (float)(s - state->warm_hold_end) / (float)ramp_length;
        state->current_temperature = state->end_temp
            - progress * (state->end_temp - state->start_temp);
        is_hold_phase = 0;
    } else {
        state->current_temperature = state->start_temp;
        is_hold_phase = 1;
    }

    float current_temp = state->current_temperature;

    // ══════════════════════════════════════════════════════════════════════════
    // Effective Gamma (Langevin friction) — matches compute_cryo_friction()
    // ══════════════════════════════════════════════════════════════════════════
    //
    // Two components:
    // 1. Equilibration boost: exponential decay from equilibration_gamma to
    //    gamma_base over the first equilibration_steps (typically 10000).
    //    Critical for structures that haven't been energy-minimized.
    // 2. Cryogenic scaling: at T < T_REF, scale by sqrt(T_REF / T) to
    //    slow dynamics proportionally.

    float base_gamma;
    if (s < state->equilibration_steps) {
        float progress = (float)s / (float)state->equilibration_steps;
        float decay = expf(-3.0f * progress);
        base_gamma = state->equilibration_gamma * decay
                   + state->gamma_base * (1.0f - decay);
    } else {
        base_gamma = state->gamma_base;
    }

    if (state->cryo_enabled && current_temp < T_REF) {
        float t_clamped = fmaxf(current_temp, T_MIN);
        float scale = T_REF / t_clamped;
        state->effective_gamma = base_gamma * sqrtf(scale);
    } else {
        state->effective_gamma = base_gamma;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Adaptive Timestep — matches adaptive_dt logic in fused_engine.rs
    // ══════════════════════════════════════════════════════════════════════════
    //
    // During hold phases (constant T), forces change slowly → safe to use 1.5x dt.
    // During ramps, revert to base_dt for accuracy.

    if (state->adaptive_dt_enabled) {
        state->dt = is_hold_phase ? (state->base_dt * 1.5f) : state->base_dt;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // UV Burst Scheduling
    // ══════════════════════════════════════════════════════════════════════════

    if (state->uv_burst_interval > 0) {
        int burst_cycle = s % state->uv_burst_interval;
        state->uv_burst_active = (burst_cycle < state->uv_burst_duration) ? 1 : 0;

        if (state->n_wavelengths > 0 && state->wavelength_dwell_steps > 0) {
            int wl_idx = (s / state->wavelength_dwell_steps) % state->n_wavelengths;
            state->uv_wavelength_nm = state->scan_wavelengths[wl_idx];
            state->uv_target_idx = wl_idx;
        }
    } else {
        state->uv_burst_active = 0;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Fused Step Counter
    // ══════════════════════════════════════════════════════════════════════════

    if (state->fused_inner_steps > 0) {
        state->fused_step_counter = (state->fused_step_counter + 1)
                                   % state->fused_inner_steps;
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Coupling Phase Toggle (multi-LIF double-buffer swap)
    // ══════════════════════════════════════════════════════════════════════════

    state->coupling_phase = 1 - state->coupling_phase;
}

// ─────────────────────────────────────────────────────────────────────────────
// protocol_director_graph — Director variant for CUDA Graph conditional nodes
// ─────────────────────────────────────────────────────────────────────────────
//
// Identical to protocol_director but additionally writes conditional handles
// for COM removal and heartbeat triggers. Used inside captured graphs.
//
// The handles are CUgraphConditionalHandle values created by the host.
// cudaGraphSetConditional(handle, 1) triggers the conditional body graph;
// cudaGraphSetConditional(handle, 0) skips it.
//
// Launch: <<<1, 1>>>

extern "C" __global__ void protocol_director_graph(
    ProtocolState* __restrict__ state,
    cudaGraphConditionalHandle com_handle,
    cudaGraphConditionalHandle heartbeat_handle
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // ── All standard Director logic (duplicated for graph variant) ──
    state->current_step += 1;
    unsigned int step = state->current_step;
    int s = (int)step;

    // Temperature state machine
    int is_hold_phase;
    if (s < state->cold_hold_end) {
        state->current_temperature = state->start_temp;
        is_hold_phase = 1;
    } else if (s < state->ramp_end) {
        int ramp_length = state->ramp_end - state->cold_hold_end;
        float progress = (float)(s - state->cold_hold_end) / (float)ramp_length;
        state->current_temperature = state->start_temp
            + progress * (state->end_temp - state->start_temp);
        is_hold_phase = 0;
    } else if (s < state->warm_hold_end) {
        state->current_temperature = state->end_temp;
        is_hold_phase = 1;
    } else if (s < state->ramp_down_end) {
        int ramp_length = state->ramp_down_end - state->warm_hold_end;
        float progress = (float)(s - state->warm_hold_end) / (float)ramp_length;
        state->current_temperature = state->end_temp
            - progress * (state->end_temp - state->start_temp);
        is_hold_phase = 0;
    } else {
        state->current_temperature = state->start_temp;
        is_hold_phase = 1;
    }

    float current_temp = state->current_temperature;

    // Effective gamma
    float base_gamma;
    if (s < state->equilibration_steps) {
        float progress = (float)s / (float)state->equilibration_steps;
        float decay = expf(-3.0f * progress);
        base_gamma = state->equilibration_gamma * decay
                   + state->gamma_base * (1.0f - decay);
    } else {
        base_gamma = state->gamma_base;
    }
    if (state->cryo_enabled && current_temp < T_REF) {
        float t_clamped = fmaxf(current_temp, T_MIN);
        state->effective_gamma = base_gamma * sqrtf(T_REF / t_clamped);
    } else {
        state->effective_gamma = base_gamma;
    }

    // Adaptive dt
    if (state->adaptive_dt_enabled) {
        state->dt = is_hold_phase ? (state->base_dt * 1.5f) : state->base_dt;
    }

    // UV burst scheduling
    if (state->uv_burst_interval > 0) {
        int burst_cycle = s % state->uv_burst_interval;
        state->uv_burst_active = (burst_cycle < state->uv_burst_duration) ? 1 : 0;
        if (state->n_wavelengths > 0 && state->wavelength_dwell_steps > 0) {
            int wl_idx = (s / state->wavelength_dwell_steps) % state->n_wavelengths;
            state->uv_wavelength_nm = state->scan_wavelengths[wl_idx];
            state->uv_target_idx = wl_idx;
        }
    } else {
        state->uv_burst_active = 0;
    }

    // Fused step counter
    if (state->fused_inner_steps > 0) {
        state->fused_step_counter = (state->fused_step_counter + 1)
                                   % state->fused_inner_steps;
    }

    // Coupling phase toggle
    state->coupling_phase = 1 - state->coupling_phase;

    // ── Conditional handle triggers ──
    // COM removal: fire every com_removal_interval steps
    if (state->com_removal_interval > 0) {
        cudaGraphSetConditional(com_handle,
            (s > 0 && (s % state->com_removal_interval) == 0) ? 1 : 0);
    } else {
        cudaGraphSetConditional(com_handle, 0);
    }

    // Heartbeat: check every 1000 steps (lightweight sampling)
    cudaGraphSetConditional(heartbeat_handle,
        (s > 0 && (s % 1000) == 0) ? 1 : 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// init_protocol_state — initialize from host-provided values
// ─────────────────────────────────────────────────────────────────────────────
//
// Launch: <<<1, 1>>>

extern "C" __global__ void init_protocol_state(
    ProtocolState* __restrict__ state,
    unsigned int total_steps,
    float start_temp,
    float end_temp,
    int cold_hold_end,
    int ramp_end,
    int warm_hold_end,
    int ramp_down_end,
    float uv_burst_energy,
    int uv_burst_interval,
    int uv_burst_duration,
    float wl0, float wl1, float wl2, float wl3,
    int n_wavelengths,
    int wavelength_dwell_steps,
    float dt,
    float gamma,
    float cutoff,
    int fused_inner_steps,
    // Gate 1 expansion params
    float gamma_base,
    int cryo_enabled,
    int equilibration_steps,
    float equilibration_gamma,
    int adaptive_dt_enabled,
    float base_dt,
    int coupling_phase
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    state->current_step = 0;
    state->total_steps = total_steps;
    state->current_temperature = start_temp;
    state->start_temp = start_temp;
    state->end_temp = end_temp;

    state->cold_hold_end = cold_hold_end;
    state->ramp_end = ramp_end;
    state->warm_hold_end = warm_hold_end;
    state->ramp_down_end = ramp_down_end;

    state->uv_burst_active = 0;
    state->uv_burst_energy = uv_burst_energy;
    state->uv_wavelength_nm = wl0;
    state->uv_burst_interval = uv_burst_interval;
    state->uv_burst_duration = uv_burst_duration;
    state->uv_target_idx = 0;

    state->scan_wavelengths[0] = wl0;
    state->scan_wavelengths[1] = wl1;
    state->scan_wavelengths[2] = wl2;
    state->scan_wavelengths[3] = wl3;
    state->n_wavelengths = n_wavelengths;
    state->wavelength_dwell_steps = wavelength_dwell_steps;

    state->dt = dt;
    state->gamma = gamma;
    state->cutoff = cutoff;

    state->fused_inner_steps = fused_inner_steps;
    state->fused_step_counter = 0;

    // Gate 1 expansion
    state->gamma_base = gamma_base;
    state->cryo_enabled = cryo_enabled;
    state->equilibration_steps = equilibration_steps;
    state->equilibration_gamma = equilibration_gamma;
    state->effective_gamma = gamma_base;  // initial value before first Director step
    state->adaptive_dt_enabled = adaptive_dt_enabled;
    state->base_dt = base_dt;
    state->coupling_phase = coupling_phase;
}

// ─────────────────────────────────────────────────────────────────────────────
// read_protocol_state — diagnostic snapshot
// ─────────────────────────────────────────────────────────────────────────────
//
// Launch: <<<1, 1>>>

extern "C" __global__ void read_protocol_state(
    const ProtocolState* __restrict__ state,
    unsigned int* out_current_step,
    float* out_current_temperature,
    int* out_uv_burst_active,
    float* out_uv_wavelength_nm
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    *out_current_step = state->current_step;
    *out_current_temperature = state->current_temperature;
    *out_uv_burst_active = state->uv_burst_active;
    *out_uv_wavelength_nm = state->uv_wavelength_nm;
}

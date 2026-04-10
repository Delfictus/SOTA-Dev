// ═══════════════════════════════════════════════════════════════════════════════
// PRISM-TWIN v3.0 — ProtocolState shared header
// ═══════════════════════════════════════════════════════════════════════════════
//
// Included by protocol_director.cu AND nhs_amber_fused.cu.
// Rust side must match with #[repr(C)] — see protocol_state.rs.
//
// Layout: 148 bytes, all fields naturally 4-byte aligned.
// ═══════════════════════════════════════════════════════════════════════════════

#ifndef PROTOCOL_STATE_CUH
#define PROTOCOL_STATE_CUH

struct ProtocolState {
    // ── Step tracking ──
    unsigned int current_step;       // incremented by Director each iteration
    unsigned int total_steps;        // simulation length (immutable after init)

    // ── Temperature (CCNS 5-phase) ──
    float current_temperature;       // computed by Director based on phase
    float start_temp;                // cold temperature (K), e.g. 50
    float end_temp;                  // warm temperature (K), e.g. 300

    // ── Phase boundaries (in AMBER steps, set once at init) ──
    int cold_hold_end;               // = cold_hold_steps
    int ramp_end;                    // = cold_hold_end + ramp_steps
    int warm_hold_end;               // = ramp_end + warm_hold_steps
    int ramp_down_end;               // = warm_hold_end + ramp_down_steps
    // cold_return: everything after ramp_down_end until total_steps

    // ── UV burst state ──
    int uv_burst_active;             // 0 or 1 (written by Director)
    float uv_burst_energy;           // kcal/mol (base energy, immutable after init)
    float uv_wavelength_nm;          // current wavelength (written by Director)
    int uv_burst_interval;           // steps between bursts (immutable)
    int uv_burst_duration;           // steps per burst (immutable)
    int uv_target_idx;               // current aromatic target index (written by Director)

    // ── Wavelength scan table ──
    float scan_wavelengths[4];       // up to 4 wavelengths (TRP/TYR/PHE/HIS)
    int n_wavelengths;               // number of active wavelengths (1-4)
    int wavelength_dwell_steps;      // steps per wavelength before hopping

    // ── Langevin parameters ──
    float dt;                        // timestep (ps) — updated by Director if adaptive_dt
    float gamma;                     // STATIC base friction coefficient (ps^-1)
    float cutoff;                    // nonbonded cutoff (A) — immutable

    // ── Fused step tracking ──
    int fused_inner_steps;           // --fused-steps value
    int fused_step_counter;          // 0..fused_inner_steps (written by Director)

    // ════════════════════════════════════════════════════════════════════════
    // Gate 1 expansion: dynamic thermodynamic state (32 bytes)
    // ════════════════════════════════════════════════════════════════════════

    // ── Friction computation (Director computes effective_gamma each step) ──
    float gamma_base;                // base friction coefficient (immutable)
    int cryo_enabled;                // 0 or 1 (immutable)
    int equilibration_steps;         // steps for initial high-friction equilibration (immutable)
    float equilibration_gamma;       // extreme friction for equilibration (immutable)
    float effective_gamma;           // COMPUTED by Director: cryo-scaled + equilibration blend

    // ── Adaptive timestep ──
    int adaptive_dt_enabled;         // 0 or 1 (immutable)
    float base_dt;                   // base timestep before adaptive scaling (immutable)

    // ── Multi-LIF coupling double-buffer phase ──
    int coupling_phase;              // 0 or 1, toggled by Director each step

    // ════════════════════════════════════════════════════════════════════════
    // Gate 2 expansion: autonomous housekeeping + heartbeat (12 bytes)
    // ════════════════════════════════════════════════════════════════════════

    int nl_rebuild_interval;         // steps between neighbor list rebuilds (immutable, default 20)
    int com_removal_interval;        // steps between COM velocity removal (immutable, default 100)
    int status_code;                 // 0=OK, 1=NaN detected, 2=system diverged (written by heartbeat)

    // ════════════════════════════════════════════════════════════════════════
    // ASC Fusion Controller hooks (16 bytes)
    // ════════════════════════════════════════════════════════════════════════

    // Steering command block — written by the interferometric bridge (read_and_adapt)
    // when the coupling kernel detects a high-value signal pattern.
    // Read by the Director to modulate UV energy, temperature bias, or dt.
    float steering_uv_boost;         // multiplicative UV energy adjustment (1.0 = neutral)
    float steering_temp_bias;        // additive temperature offset (K, 0.0 = neutral)
    int steering_focus_residue;      // residue ID to focus UV on (-1 = no focus)
    int steering_flags;              // bit flags: 0x1=phase_lock, 0x2=cryo_stabilize, 0x4=adaptive_slow

    // ════════════════════════════════════════════════════════════════════════
    // Phase-coherence metrology (4 bytes)
    // ════════════════════════════════════════════════════════════════════════

    unsigned int current_phase_bits; // 10-bit CCNS phase angle (0-1023), updated by Director each step
};

// Struct size: 148 (Gates 0-2) + 16 (ASC hooks) + 4 (phase) = 168 bytes

#endif // PROTOCOL_STATE_CUH

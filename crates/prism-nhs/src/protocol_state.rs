//! PRISM-TWIN v3.0 — GPU-resident Protocol State Machine
//!
//! ProtocolState lives in VRAM. The Director kernel updates it each step
//! BEFORE physics kernels read from it. This makes the entire simulation
//! loop capturable as a CUDA Graph (Gate 2).
//!
//! Gate 0: ProtocolState struct + Director kernel (standalone validation)
//! Gate 1: Physics kernels read from ProtocolState (replaces scalar params)

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, CudaFunction, CudaModule,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

use crate::fused_engine::CryoUvProtocol;
use crate::twin_kernels::find_twin_ptx;

/// Stage 2: Closed-loop steering focus residue with weight.
///
/// One entry in `ProtocolState::steering_focus_residues`. The ASC controller
/// fills this array each chunk with the top-K residues by GC-PID
/// synergy_fraction (Stage 1B-3 — see `crates/prism-nhs/src/gcpid.rs`).
/// The `ring_buffer_read_and_adapt` kernel reads the array on its next
/// launch and applies a multiplicative threshold-reduction boost to spikes
/// whose primary residue id matches an active entry.
///
/// `residue_id = -1` and/or `weight = 0.0` mark the entry as inactive.
///
/// Layout MUST match the CUDA `SteerEntry` struct nested in
/// `ProtocolState` in `protocol_state.cuh`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SteerEntry {
    pub residue_id: i32,
    pub weight: f32,
}

/// Maximum number of focus residues the steering buffer can hold.
/// MUST match the CUDA `steering_focus_residues[64]` array dimension.
pub const STEERING_FOCUS_MAX: usize = 64;

// ─────────────────────────────────────────────────────────────────────────────
// ProtocolState — must match protocol_state.cuh struct exactly
// ─────────────────────────────────────────────────────────────────────────────

/// GPU-resident protocol state. Layout must be identical to the CUDA struct
/// in protocol_state.cuh (148 bytes, naturally aligned).
///
/// Fields are partitioned into:
/// - **Immutable** (set once at init, safe to bake as scalar args)
/// - **Dynamic** (updated by Director kernel each step, must live in VRAM)
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ProtocolState {
    // ── Step tracking ──
    pub current_step: u32,           // DYNAMIC: incremented by Director
    pub total_steps: u32,            // Immutable

    // ── Temperature (CCNS 5-phase) ──
    pub current_temperature: f32,    // DYNAMIC: computed by Director
    pub start_temp: f32,             // Immutable
    pub end_temp: f32,               // Immutable

    // ── Phase boundaries (set once at init) ──
    pub cold_hold_end: i32,          // Immutable
    pub ramp_end: i32,               // Immutable
    pub warm_hold_end: i32,          // Immutable
    pub ramp_down_end: i32,          // Immutable

    // ── UV burst state ──
    pub uv_burst_active: i32,        // DYNAMIC: 0 or 1
    pub uv_burst_energy: f32,        // Immutable (base energy)
    pub uv_wavelength_nm: f32,       // DYNAMIC: current wavelength
    pub uv_burst_interval: i32,      // Immutable
    pub uv_burst_duration: i32,      // Immutable
    pub uv_target_idx: i32,          // DYNAMIC: current aromatic target

    // ── Wavelength scan table ──
    pub scan_wavelengths: [f32; 4],  // Immutable
    pub n_wavelengths: i32,          // Immutable
    pub wavelength_dwell_steps: i32, // Immutable

    // ── Langevin parameters ──
    pub dt: f32,                     // DYNAMIC if adaptive_dt enabled
    pub gamma: f32,                  // Immutable (base value, kept for compat)
    pub cutoff: f32,                 // Immutable

    // ── Fused step tracking ──
    pub fused_inner_steps: i32,      // Immutable
    pub fused_step_counter: i32,     // DYNAMIC

    // ════════════════════════════════════════════════════════════════════════
    // Gate 1 expansion: dynamic thermodynamic state (+32 bytes)
    // ════════════════════════════════════════════════════════════════════════

    // ── Friction computation ──
    pub gamma_base: f32,             // Immutable: base friction coefficient
    pub cryo_enabled: i32,           // Immutable: 0 or 1
    pub equilibration_steps: i32,    // Immutable: steps for high-friction warmup
    pub equilibration_gamma: f32,    // Immutable: extreme friction value (1000.0)
    pub effective_gamma: f32,        // DYNAMIC: computed by Director each step

    // ── Adaptive timestep ──
    pub adaptive_dt_enabled: i32,    // Immutable: 0 or 1
    pub base_dt: f32,                // Immutable: timestep before adaptive scaling

    // ── Multi-LIF coupling double-buffer phase ──
    pub coupling_phase: i32,         // DYNAMIC: 0 or 1, toggled by Director

    // ════════════════════════════════════════════════════════════════════════
    // Gate 2 expansion: autonomous housekeeping + heartbeat (+12 bytes)
    // ════════════════════════════════════════════════════════════════════════

    pub nl_rebuild_interval: i32,    // Immutable: steps between NL rebuilds (default 20)
    pub com_removal_interval: i32,   // Immutable: steps between COM removal (default 100)
    pub status_code: i32,            // DYNAMIC: 0=OK, 1=NaN, 2=diverged (written by heartbeat)

    // ════════════════════════════════════════════════════════════════════════
    // ASC Fusion Controller hooks (+16 bytes)
    // ════════════════════════════════════════════════════════════════════════

    /// Multiplicative UV energy adjustment (1.0 = neutral). Written by coupling kernel.
    pub steering_uv_boost: f32,
    /// Additive temperature offset in K (0.0 = neutral). Written by coupling kernel.
    pub steering_temp_bias: f32,
    /// Residue ID to focus UV on (-1 = no focus). Written by coupling kernel.
    pub steering_focus_residue: i32,
    /// Bit flags: 0x1=phase_lock, 0x2=cryo_stabilize, 0x4=adaptive_slow
    pub steering_flags: i32,

    // ════════════════════════════════════════════════════════════════════════
    // Phase-coherence metrology (4 bytes)
    // ════════════════════════════════════════════════════════════════════════

    /// 10-bit CCNS phase angle (0-1023), updated by Director each step
    pub current_phase_bits: u32,

    // ════════════════════════════════════════════════════════════════════════
    // Stage 2: Closed-loop ASC steering — focus residues (516 bytes)
    // ════════════════════════════════════════════════════════════════════════
    //
    // The ASC controller writes the top-K residues by GC-PID
    // synergy_fraction into this fixed-size array each chunk via
    // cuMemcpyHtoDAsync on the engine stream. The
    // `ring_buffer_read_and_adapt` kernel reads it on its next launch and
    // applies a multiplicative threshold-reduction boost to spikes whose
    // primary residue id matches.
    //
    // The autonomous CUDA Graph captures a POINTER to this struct (not a
    // value), so contents change between graph replays without requiring
    // re-capture — the canonical "captured pointer with mutable contents"
    // pattern. See graph_capture.rs and persistent_engine.rs::kcc_step_once
    // for the same pattern at work elsewhere.
    //
    // We steer toward synergy_fraction-ranked residues, NOT pocket
    // detection residues. The Stage 1B-3 validation on 4LPK showed the
    // SII signature pocket residues have HIGH total MI but LOW
    // synergy_fraction (their information is mostly redundant across
    // groups), while OTHER residues (152, 45, 65, 105, ...) have LOWER
    // total MI but HIGHER synergy_fraction (cross-group coordination
    // points). Steering toward coordination points produces real
    // cross-group divergence; steering toward already-agreed-upon
    // pocket residues does nothing.

    /// Number of active entries in `steering_focus_residues` (0..64).
    pub steering_focus_count: i32,

    /// Top-K focus residues by GC-PID synergy_fraction. Inactive entries
    /// have `residue_id = -1` and `weight = 0.0`.
    pub steering_focus_residues: [SteerEntry; STEERING_FOCUS_MAX],
}

const _: () = {
    // Compile-time size check: must match CUDA struct.
    //
    // Layout breakdown:
    //   148 (Gates 0-2)
    // +  16 (legacy single-residue ASC hooks)
    // +   4 (current_phase_bits)
    // +   4 (steering_focus_count)         ← Stage 2
    // + 512 (steering_focus_residues[64])  ← Stage 2 (64 × 8 bytes)
    // = 684 bytes total
    assert!(std::mem::size_of::<ProtocolState>() == 684);
    assert!(std::mem::size_of::<SteerEntry>() == 8);
};

impl ProtocolState {
    /// Build from CryoUvProtocol + simulation parameters.
    ///
    /// Phase boundaries are precomputed as cumulative step counts:
    ///   cold_hold_end = cold_hold_steps
    ///   ramp_end      = cold_hold_end + ramp_steps
    ///   warm_hold_end = ramp_end + warm_hold_steps
    ///   ramp_down_end = warm_hold_end + ramp_down_steps
    pub fn from_cryo_uv(
        protocol: &CryoUvProtocol,
        total_steps: u32,
        dt: f32,
        gamma: f32,
        cutoff: f32,
        fused_inner_steps: i32,
        cryo_enabled: bool,
        adaptive_dt_enabled: bool,
    ) -> Self {
        let cold_hold_end = protocol.cold_hold_steps;
        let ramp_end = cold_hold_end + protocol.ramp_steps;
        let warm_hold_end = ramp_end + protocol.warm_hold_steps;
        let ramp_down_end = warm_hold_end + protocol.ramp_down_steps;

        // Pack up to 4 wavelengths
        let mut scan_wl = [0.0f32; 4];
        let n_wl = protocol.scan_wavelengths.len().min(4);
        for i in 0..n_wl {
            scan_wl[i] = protocol.scan_wavelengths[i];
        }

        Self {
            current_step: 0,
            total_steps,
            current_temperature: protocol.start_temp,
            start_temp: protocol.start_temp,
            end_temp: protocol.end_temp,
            cold_hold_end,
            ramp_end,
            warm_hold_end,
            ramp_down_end,
            uv_burst_active: 0,
            uv_burst_energy: protocol.uv_burst_energy,
            uv_wavelength_nm: scan_wl[0],
            uv_burst_interval: protocol.uv_burst_interval,
            uv_burst_duration: protocol.uv_burst_duration,
            uv_target_idx: 0,
            scan_wavelengths: scan_wl,
            n_wavelengths: n_wl as i32,
            wavelength_dwell_steps: protocol.wavelength_dwell_steps,
            dt,
            gamma,
            cutoff,
            fused_inner_steps,
            fused_step_counter: 0,
            // Gate 1 expansion
            gamma_base: gamma,
            cryo_enabled: cryo_enabled as i32,
            equilibration_steps: 10000,
            equilibration_gamma: 1000.0,
            effective_gamma: gamma,
            adaptive_dt_enabled: adaptive_dt_enabled as i32,
            base_dt: dt,
            coupling_phase: 0,
            // Gate 2
            nl_rebuild_interval: 20,
            com_removal_interval: 100,
            status_code: 0,
            // ASC hooks (neutral defaults)
            steering_uv_boost: 1.0,
            steering_temp_bias: 0.0,
            steering_focus_residue: -1,
            steering_flags: 0,
            // Phase metrology
            current_phase_bits: 0,
            // Stage 2 closed-loop steering — initialized empty.
            // The ASC controller will populate this each chunk via
            // cuMemcpyHtoDAsync once --closed-loop-steering is enabled.
            steering_focus_count: 0,
            steering_focus_residues: [SteerEntry { residue_id: -1, weight: 0.0 }; STEERING_FOCUS_MAX],
        }
    }

    /// Compute expected temperature at a given step (CPU reference implementation).
    /// Used for test validation against the GPU Director kernel.
    pub fn expected_temperature_at_step(&self, step: u32) -> f32 {
        let s = step as i32;
        if s < self.cold_hold_end {
            self.start_temp
        } else if s < self.ramp_end {
            let ramp_len = self.ramp_end - self.cold_hold_end;
            let progress = (s - self.cold_hold_end) as f32 / ramp_len as f32;
            self.start_temp + progress * (self.end_temp - self.start_temp)
        } else if s < self.warm_hold_end {
            self.end_temp
        } else if s < self.ramp_down_end {
            let ramp_len = self.ramp_down_end - self.warm_hold_end;
            let progress = (s - self.warm_hold_end) as f32 / ramp_len as f32;
            self.end_temp - progress * (self.end_temp - self.start_temp)
        } else {
            self.start_temp
        }
    }

    /// Compute expected effective_gamma at a given step (CPU reference).
    /// Matches the Director kernel's gamma computation exactly.
    pub fn expected_gamma_at_step(&self, step: u32) -> f32 {
        let s = step as i32;
        let base_gamma = if s < self.equilibration_steps {
            let progress = s as f32 / self.equilibration_steps as f32;
            let decay = (-3.0 * progress).exp();
            self.equilibration_gamma * decay + self.gamma_base * (1.0 - decay)
        } else {
            self.gamma_base
        };

        let temp = self.expected_temperature_at_step(step);
        if self.cryo_enabled != 0 && temp < 300.0 {
            let t_clamped = temp.max(10.0);
            let scale = 300.0 / t_clamped;
            base_gamma * scale.sqrt()
        } else {
            base_gamma
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ProtocolDirector — GPU kernel wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Manages the GPU-resident ProtocolState and Director kernel launches.
pub struct ProtocolDirector {
    /// Device allocation for ProtocolState (148 bytes).
    /// Passed as `const ProtocolState*` to physics kernels.
    pub d_state: CudaSlice<u8>,
    /// Cached host-side copy for diagnostics
    host_state: ProtocolState,
    // Kernel functions
    director_fn: CudaFunction,
    init_fn: CudaFunction,
    read_fn: CudaFunction,
}

impl ProtocolDirector {
    /// Load PTX and allocate device state.
    pub fn new(
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
    ) -> Result<(Self, Arc<CudaModule>)> {
        let ptx_path = find_twin_ptx("protocol_director.ptx")?;
        let module = context.load_module(Ptx::from_file(&ptx_path))
            .with_context(|| format!("Failed to load protocol_director.ptx from {}", ptx_path))?;

        let director_fn = module.load_function("protocol_director")
            .context("protocol_director function not found in PTX")?;
        let init_fn = module.load_function("init_protocol_state")
            .context("init_protocol_state function not found in PTX")?;
        let read_fn = module.load_function("read_protocol_state")
            .context("read_protocol_state function not found in PTX")?;

        let state_size = std::mem::size_of::<ProtocolState>();
        let d_state = stream.alloc_zeros::<u8>(state_size)?;

        let host_state = unsafe { std::mem::zeroed::<ProtocolState>() };

        Ok((Self { d_state, host_state, director_fn, init_fn, read_fn }, module))
    }

    /// Initialize ProtocolState on GPU via direct memcpy.
    /// The struct is #[repr(C)] so byte layout matches the CUDA struct exactly.
    pub fn initialize(
        &mut self,
        stream: &Arc<CudaStream>,
        state: &ProtocolState,
    ) -> Result<()> {
        self.host_state = *state;
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                state as *const ProtocolState as *const u8,
                std::mem::size_of::<ProtocolState>(),
            )
        };
        stream.memcpy_htod(bytes, &mut self.d_state)?;

        log::info!("ProtocolDirector initialized: {}K -> {}K, {} total steps",
            state.start_temp, state.end_temp, state.total_steps);
        log::info!("  Phase boundaries: cold_hold={}, ramp={}, warm_hold={}, ramp_down={}",
            state.cold_hold_end, state.ramp_end, state.warm_hold_end, state.ramp_down_end);
        log::info!("  Gamma: base={:.1}, cryo={}, equilibration={} steps @ {:.0}",
            state.gamma_base, state.cryo_enabled != 0, state.equilibration_steps, state.equilibration_gamma);
        log::info!("  Adaptive dt: {} (base={:.4} ps)", state.adaptive_dt_enabled != 0, state.base_dt);

        Ok(())
    }

    /// Run one Director step on the given stream.
    ///
    /// CRITICAL: Must be launched on the SAME stream as the physics kernels.
    /// CUDA stream ordering guarantees Director writes are visible to subsequent
    /// kernel launches on the same stream — no cudaDeviceSynchronize needed.
    pub fn step(&mut self, stream: &Arc<CudaStream>) -> Result<()> {
        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            stream.launch_builder(&self.director_fn)
                .arg(&mut self.d_state)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Read current state from GPU (synchronizes stream).
    /// Returns (current_step, current_temperature, uv_burst_active, uv_wavelength_nm).
    pub fn read_state(&self, stream: &Arc<CudaStream>) -> Result<(u32, f32, i32, f32)> {
        let mut out_step = stream.alloc_zeros::<u32>(1)?;
        let mut out_temp = stream.alloc_zeros::<f32>(1)?;
        let mut out_uv = stream.alloc_zeros::<i32>(1)?;
        let mut out_wl = stream.alloc_zeros::<f32>(1)?;

        let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (1, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            stream.launch_builder(&self.read_fn)
                .arg(&self.d_state)
                .arg(&mut out_step)
                .arg(&mut out_temp)
                .arg(&mut out_uv)
                .arg(&mut out_wl)
                .launch(cfg)?;
        }

        let mut step = [0u32; 1];
        let mut temp = [0.0f32; 1];
        let mut uv = [0i32; 1];
        let mut wl = [0.0f32; 1];
        stream.memcpy_dtoh(&out_step, &mut step)?;
        stream.memcpy_dtoh(&out_temp, &mut temp)?;
        stream.memcpy_dtoh(&out_uv, &mut uv)?;
        stream.memcpy_dtoh(&out_wl, &mut wl)?;

        Ok((step[0], temp[0], uv[0], wl[0]))
    }

    /// Download entire ProtocolState from GPU (for full diagnostics).
    pub fn download_full_state(&self, stream: &Arc<CudaStream>) -> Result<ProtocolState> {
        let mut buf = vec![0u8; std::mem::size_of::<ProtocolState>()];
        stream.memcpy_dtoh(&self.d_state, &mut buf)?;
        let state: ProtocolState = unsafe { std::ptr::read(buf.as_ptr() as *const ProtocolState) };
        Ok(state)
    }

    /// Get cached host-side state (may be stale — use read_state() for live values).
    pub fn host_state(&self) -> &ProtocolState {
        &self.host_state
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_state_size() {
        // Stage 2 extension: 168 (legacy) + 4 (count) + 512 (focus array) = 684 bytes
        assert_eq!(std::mem::size_of::<ProtocolState>(), 684);
        assert_eq!(std::mem::size_of::<SteerEntry>(), 8);
    }

    #[test]
    fn test_steering_defaults_are_neutral() {
        let protocol = CryoUvProtocol::fast_35k();
        let state = ProtocolState::from_cryo_uv(
            &protocol, 45000, 0.002, 1.0, 9.0, 4, true, true,
        );
        // Default-initialized steering should not affect the kernel's behavior.
        assert_eq!(state.steering_focus_count, 0);
        for entry in state.steering_focus_residues.iter() {
            assert_eq!(entry.residue_id, -1);
            assert_eq!(entry.weight, 0.0);
        }
        // Legacy steering hooks should also be at neutral defaults
        assert_eq!(state.steering_uv_boost, 1.0);
        assert_eq!(state.steering_temp_bias, 0.0);
        assert_eq!(state.steering_focus_residue, -1);
        assert_eq!(state.steering_flags, 0);
    }

    #[test]
    fn test_from_cryo_uv_fast_35k() {
        let protocol = CryoUvProtocol::fast_35k();
        let state = ProtocolState::from_cryo_uv(
            &protocol, 45000, 0.002, 1.0, 9.0, 4, true, true,
        );

        assert_eq!(state.current_step, 0);
        assert_eq!(state.total_steps, 45000);
        assert_eq!(state.start_temp, 50.0);
        assert_eq!(state.end_temp, 300.0);
        assert_eq!(state.cold_hold_end, 14000);
        assert_eq!(state.ramp_end, 20000);
        assert_eq!(state.warm_hold_end, 35000);
        assert_eq!(state.ramp_down_end, 41000);
        assert_eq!(state.uv_burst_energy, 42.0);
        assert_eq!(state.uv_burst_interval, 250);
        assert_eq!(state.n_wavelengths, 4);
        assert_eq!(state.scan_wavelengths, [280.0, 274.0, 258.0, 211.0]);
        assert_eq!(state.fused_inner_steps, 4);
        // Gate 1 fields
        assert_eq!(state.gamma_base, 1.0);
        assert_eq!(state.cryo_enabled, 1);
        assert_eq!(state.equilibration_steps, 10000);
        assert_eq!(state.equilibration_gamma, 1000.0);
        assert_eq!(state.adaptive_dt_enabled, 1);
        assert_eq!(state.base_dt, 0.002);
        assert_eq!(state.coupling_phase, 0);
    }

    #[test]
    fn test_expected_temperature_phases() {
        let protocol = CryoUvProtocol::fast_35k();
        let state = ProtocolState::from_cryo_uv(
            &protocol, 45000, 0.002, 1.0, 9.0, 4, true, false,
        );

        // Phase 1: Cold hold → 50K
        assert_eq!(state.expected_temperature_at_step(0), 50.0);
        assert_eq!(state.expected_temperature_at_step(13999), 50.0);
        // Phase 2: Ramp up → 50K to 300K
        let mid_ramp = state.expected_temperature_at_step(17000);
        assert!((mid_ramp - 175.0).abs() < 0.1);
        // Phase 3: Warm hold → 300K
        assert_eq!(state.expected_temperature_at_step(27000), 300.0);
        // Phase 4: Ramp down → 300K to 50K
        let mid_down = state.expected_temperature_at_step(38000);
        assert!((mid_down - 175.0).abs() < 0.1);
        // Phase 5: Cold return → 50K
        assert_eq!(state.expected_temperature_at_step(44999), 50.0);
    }

    #[test]
    fn test_expected_gamma_equilibration() {
        let protocol = CryoUvProtocol::fast_35k();
        let state = ProtocolState::from_cryo_uv(
            &protocol, 45000, 0.002, 1.0, 9.0, 4, true, false,
        );

        // Step 0: full equilibration gamma (1000) with cryo scaling at 50K
        // base = 1000 * exp(0) + 1.0 * (1-1) = 1000
        // cryo: 1000 * sqrt(300/50) = 1000 * 2.449 = 2449
        let g0 = state.expected_gamma_at_step(0);
        assert!(g0 > 2000.0, "Step 0 gamma should be very high: {}", g0);

        // Step 10000: past equilibration, pure cryo scaling
        // base = 1.0, cryo: 1.0 * sqrt(300/50) = 2.449
        let g10k = state.expected_gamma_at_step(10000);
        assert!((g10k - 2.449).abs() < 0.1, "Step 10000 gamma: {}", g10k);

        // Step 27000: warm hold at 300K, no cryo scaling
        // base = 1.0, temp = 300K >= T_REF → no scaling
        let g_warm = state.expected_gamma_at_step(27000);
        assert!((g_warm - 1.0).abs() < 0.01, "Warm hold gamma: {}", g_warm);
    }

    /// GPU integration test: Director kernel temperature ramp + gamma + dt.
    #[test]
    #[ignore] // Requires GPU
    fn test_director_kernel_temperature_ramp() {
        env_logger::builder().is_test(true).try_init().ok();

        let context = CudaContext::new(0).expect("CUDA not available");
        let stream = context.default_stream();

        let (mut director, _module) = ProtocolDirector::new(&context, &stream)
            .expect("Failed to create ProtocolDirector");

        let protocol = CryoUvProtocol::fast_35k();
        let total_steps = 45000u32;
        let state = ProtocolState::from_cryo_uv(
            &protocol, total_steps, 0.002, 1.0, 9.0, 4, true, true,
        );
        director.initialize(&stream, &state).expect("Failed to initialize");

        // Phase boundary checkpoints
        let checkpoints: Vec<(u32, u32, f32, f32, &str)> = vec![
            (1,     1,     50.0,    0.01,  "step 1 cold hold"),
            (13999, 14000, 50.0,    0.01,  "cold hold end"),
            (3000,  17000, 175.0,   0.5,   "mid ramp up"),
            (3000,  20000, 300.0,   0.5,   "ramp end"),
            (7500,  27500, 300.0,   0.01,  "mid warm hold"),
            (7500,  35000, 300.0,   0.01,  "warm hold end"),
            (3000,  38000, 175.0,   0.5,   "mid ramp down"),
            (3000,  41000, 50.0,    0.5,   "ramp down end"),
            (2000,  43000, 50.0,    0.01,  "cold return"),
            (2000,  45000, 50.0,    0.01,  "final step"),
        ];

        for (run_steps, expected_step, expected_temp, tol, label) in &checkpoints {
            for _ in 0..*run_steps {
                director.step(&stream).expect("director step failed");
            }

            let (step, temp, _uv, _wl) = director.read_state(&stream).expect("read failed");
            assert_eq!(step, *expected_step, "Step mismatch at '{}'", label);
            assert!((temp - expected_temp).abs() < *tol,
                "Temp mismatch at '{}' (step {}): GPU={:.2}K, expected={:.2}K",
                label, step, temp, expected_temp);
        }

        // Verify gamma + dt via full state download at final step
        let final_state = director.download_full_state(&stream).expect("download failed");
        // At step 45000 (cold return at 50K), past equilibration:
        // effective_gamma = gamma_base * sqrt(300/50) = 1.0 * 2.449
        let expected_gamma = state.expected_gamma_at_step(45000);
        assert!((final_state.effective_gamma - expected_gamma).abs() < 0.1,
            "Gamma mismatch: GPU={:.3}, CPU={:.3}", final_state.effective_gamma, expected_gamma);
        // Cold return is a hold phase → adaptive dt = base * 1.5
        assert!((final_state.dt - 0.003).abs() < 0.0001,
            "Adaptive dt should be 0.003 in hold phase, got {}", final_state.dt);
        // Coupling phase should have toggled 45000 times → 45000 % 2 = 0
        assert_eq!(final_state.coupling_phase, 0, "Coupling phase should be 0 after even steps");

        println!("Director kernel Gate 1: ALL CHECKPOINTS PASSED (temp + gamma + dt + coupling)");
    }

    /// GPU integration test: verify UV burst scheduling.
    #[test]
    #[ignore] // Requires GPU
    fn test_director_kernel_uv_scheduling() {
        env_logger::builder().is_test(true).try_init().ok();

        let context = CudaContext::new(0).expect("CUDA not available");
        let stream = context.default_stream();

        let (mut director, _module) = ProtocolDirector::new(&context, &stream)
            .expect("Failed to create ProtocolDirector");

        let protocol = CryoUvProtocol::fast_35k();
        let state = ProtocolState::from_cryo_uv(
            &protocol, 1000, 0.002, 1.0, 9.0, 4, true, false,
        );
        director.initialize(&stream, &state).expect("init failed");

        let mut burst_count = 0;
        let mut seen_wavelengths = std::collections::HashSet::new();

        for i in 0..500 {
            director.step(&stream).expect("step failed");
            if i % 50 == 0 {
                let (_step, _temp, uv_active, wl) = director.read_state(&stream)
                    .expect("read failed");
                if uv_active != 0 {
                    burst_count += 1;
                    seen_wavelengths.insert((wl * 10.0) as i32);
                }
            }
        }

        assert!(burst_count >= 1, "Expected at least 1 UV burst in 500 steps, got {}", burst_count);
        println!("UV scheduling: {} bursts detected, {} distinct wavelengths",
            burst_count, seen_wavelengths.len());
    }
}

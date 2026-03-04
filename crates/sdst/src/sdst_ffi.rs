/**
 * Rust FFI Integration Guide for SDST
 *
 * This file documents how to call SDST from nhs_rt_full via Rust FFI.
 * The actual Rust bindings would go in your NHS crate.
 *
 * Build integration:
 *   1. Build libsdst.so: `cd sdst && make shared`
 *   2. In your Cargo.toml, add a build.rs that links against libsdst
 *   3. Use the FFI bindings below in your Rust code
 */

// ============================================================
// build.rs (add to your nhs_rt_full crate)
// ============================================================
//
// fn main() {
//     println!("cargo:rustc-link-search=native=/path/to/sdst/lib");
//     println!("cargo:rustc-link-lib=dylib=sdst");
//     println!("cargo:rustc-link-lib=dylib=cudart");
//     println!("cargo:rerun-if-changed=/path/to/sdst/lib/libsdst.so");
// }

// ============================================================
// sdst_ffi.rs - Raw FFI bindings
// ============================================================

#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::os::raw::{c_char, c_float, c_int, c_void};

pub type MortonCode = u32;
pub type SpikeId = u32;
pub type WavefrontId = u32;
pub type AvalancheId = u32;
pub type PhaseId = u8;
pub type SdstHandle = *mut c_void;

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SdstError {
    Success = 0,
    ErrorCuda = -1,
    ErrorOom = -2,
    ErrorInvalidParam = -3,
    ErrorTableFull = -4,
    ErrorNotFound = -5,
    ErrorWavefrontOverflow = -6,
    ErrorStreamInvalid = -7,
}

impl SdstError {
    pub fn is_ok(self) -> bool { self == SdstError::Success }
    pub fn check(self) -> Result<(), SdstError> {
        if self.is_ok() { Ok(()) } else { Err(self) }
    }
}

#[repr(C, align(32))]
#[derive(Debug, Clone, Copy)]
pub struct SpikeEvent {
    pub voxel: MortonCode,
    pub timestamp: u32,
    pub amplitude: u16,
    pub parent_spike: SpikeId,
    pub avalanche_id: AvalancheId,
    pub local_temp: u16,
    pub energy_gradient: u16,
    pub solvent_exposure: u16,
    pub phase_id: PhaseId,
    pub tcl_flags: u8,
    pub wavefront_id: WavefrontId,
    pub wavefront_velocity: u16,
    pub wavefront_coherence: u16,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SpikeInput {
    pub voxel_x: u32,
    pub voxel_y: u32,
    pub voxel_z: u32,
    pub timestamp: u32,
    pub amplitude: f32,
    pub local_temp: f32,
    pub energy_gradient: f32,
    pub solvent_exposure: f32,
    pub phase_id: u8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SpatialRegion {
    pub x_min: u32, pub x_max: u32,
    pub y_min: u32, pub y_max: u32,
    pub z_min: u32, pub z_max: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HysteresisResult {
    pub heating_spike_rate: f32,
    pub cooling_spike_rate: f32,
    pub asymmetry_score: f32,
    pub avalanche_size_ratio: f32,
    pub wavefront_coherence_ratio: f32,
    pub heating_spike_count: u32,
    pub cooling_spike_count: u32,
    pub is_hysteretic: bool,
}

#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CcnsClass {
    Soc = 0,
    NearCritical = 1,
    Barrier = 2,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CcnsResult {
    pub tau: f32,
    pub classification: CcnsClass,
    pub tau_stderr: f32,
    pub n_avalanches: u32,
    pub druggability: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TideDecomposition {
    pub residue_id: u32,
    pub causal_dg: f32,
    pub transfer_entropy: f32,
    pub fisher_info: f32,
    pub kl_divergence: f32,
    pub n_causal_spikes: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AvalancheStats {
    pub id: AvalancheId,
    pub size: u32,
    pub duration: u32,
    pub spatial_extent: f32,
    pub seed_voxel: MortonCode,
    pub phase: PhaseId,
    pub tau_local: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WavefrontStats {
    pub id: WavefrontId,
    pub origin: MortonCode,
    pub birth_time: u32,
    pub death_time: u32,
    pub spike_count: u32,
    pub mean_velocity: f32,
    pub mean_coherence: f32,
    pub spatial_extent: f32,
    pub phase: PhaseId,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct CausalSubgraph {
    pub events: *mut SpikeEvent,
    pub count: u32,
    pub parent_indices: *mut u32,
}

#[repr(C)]
pub struct SdstConfig {
    pub grid_nx: u32,
    pub grid_ny: u32,
    pub grid_nz: u32,
    pub grid_spacing: f32,
    pub hash_table_capacity: u32,
    pub max_spike_events: u32,
    pub max_wavefronts: u32,
    pub wavefront_merge_dist: f32,
    pub wavefront_max_dt: u32,
    pub avalanche_spatial_cutoff: f32,
    pub avalanche_max_gap: u32,
    pub phase_boundaries: [u32; 6],
    pub ccns_soc_threshold: f32,
    pub ccns_barrier_threshold: f32,
    pub num_streams: u32,
    pub device_id: c_int,
}

extern "C" {
    // Lifecycle
    pub fn sdst_default_config() -> SdstConfig;
    pub fn sdst_create(config: *const SdstConfig, handle: *mut SdstHandle) -> SdstError;
    pub fn sdst_destroy(handle: SdstHandle) -> SdstError;
    pub fn sdst_reset(handle: SdstHandle) -> SdstError;
    pub fn sdst_event_count(handle: SdstHandle, count: *mut u32) -> SdstError;
    pub fn sdst_memory_usage(handle: SdstHandle, bytes: *mut usize) -> SdstError;

    // Insertion
    pub fn sdst_insert_spikes(
        handle: SdstHandle,
        events: *mut SpikeEvent,
        count: u32,
        stream: *mut c_void,
    ) -> SdstError;
    pub fn sdst_insert_raw(
        handle: SdstHandle,
        inputs: *const SpikeInput,
        count: u32,
        stream: *mut c_void,
    ) -> SdstError;

    // Queries
    pub fn sdst_query_region(
        handle: SdstHandle,
        region: *const SpatialRegion,
        events: *mut *mut SpikeEvent,
        count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;
    pub fn sdst_query_voxel(
        handle: SdstHandle,
        x: u32, y: u32, z: u32,
        events: *mut *mut SpikeEvent,
        count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;
    pub fn sdst_query_timerange(
        handle: SdstHandle,
        t_start: u32, t_end: u32,
        events: *mut *mut SpikeEvent,
        count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;

    // Causal
    pub fn sdst_causal_subgraph(
        handle: SdstHandle,
        root: SpikeId,
        max_depth: u32,
        graph: *mut CausalSubgraph,
        stream: *mut c_void,
    ) -> SdstError;
    pub fn sdst_free_subgraph(graph: *mut CausalSubgraph) -> SdstError;

    // CCNS
    pub fn sdst_ccns_region(
        handle: SdstHandle,
        region: *const SpatialRegion,
        result: *mut CcnsResult,
        stream: *mut c_void,
    ) -> SdstError;

    // Hysteresis
    pub fn sdst_hysteresis_region(
        handle: SdstHandle,
        region: *const SpatialRegion,
        threshold: f32,
        result: *mut HysteresisResult,
        stream: *mut c_void,
    ) -> SdstError;
    pub fn sdst_hysteresis_scan(
        handle: SdstHandle,
        threshold: f32,
        results: *mut *mut HysteresisResult,
        regions: *mut *mut SpatialRegion,
        count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;

    // Wavefronts
    pub fn sdst_wavefront_stats(
        handle: SdstHandle,
        phase_filter: c_int,
        stats: *mut *mut WavefrontStats,
        count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;
    pub fn sdst_wavefront_path(
        handle: SdstHandle,
        wavefront: WavefrontId,
        events: *mut *mut SpikeEvent,
        count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;

    // TIDE (Plan C)
    pub fn sdst_tide_decomposition(
        handle: SdstHandle,
        pocket: *const SpatialRegion,
        residue_map: *const u32,
        n_residues: u32,
        decomp: *mut *mut TideDecomposition,
        count: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;

    // DCC
    pub fn sdst_compute_dcc(
        handle: SdstHandle,
        known_sites: *const f32,
        n_known: u32,
        dcc: *mut *mut f32,
        centroids: *mut *mut f32,
        n_detected: *mut u32,
        stream: *mut c_void,
    ) -> SdstError;

    // Serialization
    pub fn sdst_save(handle: SdstHandle, path: *const c_char) -> SdstError;
    pub fn sdst_load(path: *const c_char, handle: *mut SdstHandle) -> SdstError;

    // Debug
    pub fn sdst_print_stats(handle: SdstHandle) -> SdstError;
}

// ============================================================
// Safe Rust wrapper
// ============================================================

pub struct Sdst {
    handle: SdstHandle,
}

impl Sdst {
    pub fn new(config: &SdstConfig) -> Result<Self, SdstError> {
        let mut handle: SdstHandle = std::ptr::null_mut();
        unsafe { sdst_create(config, &mut handle).check()? };
        Ok(Self { handle })
    }

    pub fn default() -> Result<Self, SdstError> {
        let config = unsafe { sdst_default_config() };
        Self::new(&config)
    }

    pub fn insert_raw(&self, inputs: &[SpikeInput]) -> Result<(), SdstError> {
        unsafe {
            sdst_insert_raw(
                self.handle,
                inputs.as_ptr(),
                inputs.len() as u32,
                std::ptr::null_mut(),
            ).check()
        }
    }

    pub fn event_count(&self) -> Result<u32, SdstError> {
        let mut count = 0u32;
        unsafe { sdst_event_count(self.handle, &mut count).check()? };
        Ok(count)
    }

    pub fn memory_usage(&self) -> Result<usize, SdstError> {
        let mut bytes = 0usize;
        unsafe { sdst_memory_usage(self.handle, &mut bytes).check()? };
        Ok(bytes)
    }

    pub fn ccns_region(&self, region: &SpatialRegion) -> Result<CcnsResult, SdstError> {
        let mut result = std::mem::MaybeUninit::<CcnsResult>::uninit();
        unsafe {
            sdst_ccns_region(
                self.handle,
                region,
                result.as_mut_ptr(),
                std::ptr::null_mut(),
            ).check()?;
            Ok(result.assume_init())
        }
    }

    pub fn hysteresis_region(&self, region: &SpatialRegion, threshold: f32)
        -> Result<HysteresisResult, SdstError>
    {
        let mut result = std::mem::MaybeUninit::<HysteresisResult>::uninit();
        unsafe {
            sdst_hysteresis_region(
                self.handle,
                region,
                threshold,
                result.as_mut_ptr(),
                std::ptr::null_mut(),
            ).check()?;
            Ok(result.assume_init())
        }
    }

    pub fn print_stats(&self) -> Result<(), SdstError> {
        unsafe { sdst_print_stats(self.handle).check() }
    }

    pub fn save(&self, path: &str) -> Result<(), SdstError> {
        let c_path = std::ffi::CString::new(path).map_err(|_| SdstError::ErrorInvalidParam)?;
        unsafe { sdst_save(self.handle, c_path.as_ptr()).check() }
    }

    pub fn reset(&self) -> Result<(), SdstError> {
        unsafe { sdst_reset(self.handle).check() }
    }
}

impl Drop for Sdst {
    fn drop(&mut self) {
        unsafe { sdst_destroy(self.handle); }
    }
}

// Safety: SDST handles are thread-safe (CUDA streams provide synchronization)
unsafe impl Send for Sdst {}
unsafe impl Sync for Sdst {}

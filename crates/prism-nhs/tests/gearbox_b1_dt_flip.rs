//! Wave B.1 verification gate — operator-mandated 2026-05-02.
//!
//! "A 'Hello-World' unit test … proving that a manual trigger correctly
//! flips the dt value in VRAM from 2.0fs to 4.0fs without host
//! intervention."
//!
//! Lives in the integration-test target rather than the lib `#[cfg(test)]`
//! module because pre-existing E0063 breakage in unrelated tests
//! (`solvate.rs`, `rt_targets.rs`, `input.rs`: missing `dimer_dyad`
//! initialiser fields) currently prevents the lib test crate from
//! compiling. The integration target compiles independently.
//!
//! Strategy:
//!   1. Allocate a synthetic dt buffer (1 × f32) standing in for
//!      `d_protocol->dt`.
//!   2. Allocate a synthetic InterferometricAdjudicatorFfi (128 bytes)
//!      with `d_dt` (offset 112) pointing at the synthetic dt and
//!      `adjudication_code` (offset 52) = 0 (Equilibrium).
//!   3. Allocate a ChronometricStateTensor seeded with counter = 500
//!      (so the cruise hysteresis upshifts to Gear 2 = 4.0 fs).
//!   4. Initialise the gear table.
//!   5. Launch `prism_gearbox_pointer_swap_kernel`.
//!   6. Assert dt now reads 0.004 ps (4.0 fs) and cruise.current_gear = 2.
//!
//! The Burst-path test (code = 1 ⇒ Gear 0, counter reset, last_burst_frame
//! stamped) is included as a second case for state-machine completeness.

#![cfg(feature = "gpu")]

use cudarc::driver::{CudaContext, DevicePtr};
use prism_nhs::gearbox::{
    default_gearbox_table, ChronometricStateTensor, ffi::*,
};
use std::ffi::c_void;

#[test]
fn b1_pointer_swap_flips_dt_from_2fs_to_4fs() {
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[gearbox B.1] CUDA unavailable: {:?} — skipping", e);
            return;
        }
    };
    let stream = ctx.new_stream().expect("stream");
    let raw_stream = stream.cu_stream() as *mut c_void;

    // ── Synthetic dt buffer. Pre-set to 2.0 fs (= 0.002 ps). ──
    let mut d_dt: cudarc::driver::CudaSlice<f32> =
        stream.alloc_zeros::<f32>(1).expect("alloc dt");
    let initial: [f32; 1] = [0.002];
    stream.memcpy_htod(&initial, &mut d_dt).expect("htod dt");
    let d_dt_addr_u64: u64 = {
        let (addr, _guard) = d_dt.device_ptr(&stream);
        addr as u64
    };

    // ── Synthetic Adjudicator FFI struct (128 bytes). ──
    // Only two fields populated; the rest stays zero.
    let mut d_adj: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(128).expect("alloc adj");
    let mut host_adj = [0u8; 128];
    let code: u32 = 0; // Equilibrium
    host_adj[52..56].copy_from_slice(&code.to_le_bytes());
    host_adj[112..120].copy_from_slice(&d_dt_addr_u64.to_le_bytes());
    stream.memcpy_htod(&host_adj, &mut d_adj).expect("htod adj");

    // ── Cruise tensor — counter pre-seeded at THRESHOLD so Equilibrium
    //    increments to 501 and selects Gear 2 (4.0 fs).
    let mut d_cruise: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(16).expect("alloc cruise");
    let cruise_seed = ChronometricStateTensor {
        counter: 500,
        last_burst_frame: 0,
        current_gear: 1,
        _pad: 0,
    };
    let cruise_bytes: [u8; 16] = unsafe {
        std::mem::transmute::<ChronometricStateTensor, [u8; 16]>(cruise_seed)
    };
    stream.memcpy_htod(&cruise_bytes, &mut d_cruise).expect("htod cruise");

    // ── Init the __constant__ gearbox table. ──
    let host_table = default_gearbox_table();
    let rc = unsafe {
        prism_gearbox_init_table_async(host_table.as_ptr(), raw_stream)
    };
    assert_eq!(rc, 0, "prism_gearbox_init_table_async returned {}", rc);

    // ── Launch PointerSwap. Scope guards so memcpy_dtoh borrows are clean. ──
    let (adj_addr, cruise_addr) = {
        let (a, _ga) = d_adj.device_ptr(&stream);
        let (c, _gc) = d_cruise.device_ptr(&stream);
        (a as u64, c as u64)
    };
    let rc = unsafe {
        prism_gearbox_launch_pointer_swap(
            adj_addr    as *const _,
            cruise_addr as *mut _,
            /*current_frame=*/ 0,
            raw_stream,
        )
    };
    assert_eq!(rc, 0, "prism_gearbox_launch_pointer_swap returned {}", rc);
    stream.synchronize().expect("post-kernel sync");

    // ── Read dt back; assert it flipped to 4.0 fs (0.004 ps). ──
    let mut readback = [0.0f32; 1];
    stream.memcpy_dtoh(&d_dt, &mut readback).expect("dtoh dt");
    let dt_after = readback[0];
    assert!(
        (dt_after - 0.004).abs() < 1.0e-7,
        "B.1 dt-flip GATE FAIL: expected 0.004 (Gear 2 = 4.0 fs), got {}",
        dt_after,
    );

    // ── Read cruise back; assert state machine bookkeeping. ──
    let mut cruise_readback = [0u8; 16];
    stream.memcpy_dtoh(&d_cruise, &mut cruise_readback).expect("dtoh cruise");
    let cruise_after: ChronometricStateTensor = unsafe {
        std::mem::transmute::<[u8; 16], ChronometricStateTensor>(cruise_readback)
    };
    assert_eq!(cruise_after.counter, 501,
        "cruise counter MUST increment 500 → 501 on Equilibrium");
    assert_eq!(cruise_after.current_gear, 2,
        "current_gear MUST be 2 (4.0 fs) once counter ≥ THRESH");

    eprintln!(
        "[B.1 GATE PASS] dt 2.0 fs → 4.0 fs flipped device-side; \
         cruise counter 500 → {}, gear={} \
         (no host writes between init and readback)",
        cruise_after.counter, cruise_after.current_gear
    );
}

#[test]
fn b1_burst_resets_cruise_and_selects_gear_0() {
    let ctx = match CudaContext::new(0) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[gearbox B.1] CUDA unavailable: {:?} — skipping", e);
            return;
        }
    };
    let stream = ctx.new_stream().expect("stream");
    let raw_stream = stream.cu_stream() as *mut c_void;

    let mut d_dt: cudarc::driver::CudaSlice<f32> =
        stream.alloc_zeros::<f32>(1).expect("alloc dt");
    let initial: [f32; 1] = [0.004]; // start at Gear 2 dt
    stream.memcpy_htod(&initial, &mut d_dt).expect("htod dt");
    let d_dt_addr_u64: u64 = {
        let (addr, _guard) = d_dt.device_ptr(&stream);
        addr as u64
    };

    let mut d_adj: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(128).expect("alloc adj");
    let mut host_adj = [0u8; 128];
    let code: u32 = 1; // Burst
    host_adj[52..56].copy_from_slice(&code.to_le_bytes());
    host_adj[112..120].copy_from_slice(&d_dt_addr_u64.to_le_bytes());
    stream.memcpy_htod(&host_adj, &mut d_adj).expect("htod adj");

    let mut d_cruise: cudarc::driver::CudaSlice<u8> =
        stream.alloc_zeros::<u8>(16).expect("alloc cruise");
    let cruise_seed = ChronometricStateTensor {
        counter: 1234,
        last_burst_frame: 0,
        current_gear: 2,
        _pad: 0,
    };
    let cruise_bytes: [u8; 16] = unsafe {
        std::mem::transmute::<ChronometricStateTensor, [u8; 16]>(cruise_seed)
    };
    stream.memcpy_htod(&cruise_bytes, &mut d_cruise).expect("htod cruise");

    let host_table = default_gearbox_table();
    unsafe {
        assert_eq!(
            0,
            prism_gearbox_init_table_async(host_table.as_ptr(), raw_stream),
        );
    }

    let (adj_addr, cruise_addr) = {
        let (a, _ga) = d_adj.device_ptr(&stream);
        let (c, _gc) = d_cruise.device_ptr(&stream);
        (a as u64, c as u64)
    };
    unsafe {
        assert_eq!(
            0,
            prism_gearbox_launch_pointer_swap(
                adj_addr    as *const _,
                cruise_addr as *mut _,
                /*current_frame=*/ 9999,
                raw_stream,
            ),
        );
    }
    stream.synchronize().expect("post-kernel sync");

    let mut readback = [0.0f32; 1];
    stream.memcpy_dtoh(&d_dt, &mut readback).expect("dtoh dt");
    assert!(
        (readback[0] - 0.0005).abs() < 1.0e-8,
        "Burst gate: expected 0.0005 (Gear 0 = 0.5 fs), got {}",
        readback[0],
    );

    let mut cruise_readback = [0u8; 16];
    stream.memcpy_dtoh(&d_cruise, &mut cruise_readback).expect("dtoh cruise");
    let cruise_after: ChronometricStateTensor = unsafe {
        std::mem::transmute::<[u8; 16], ChronometricStateTensor>(cruise_readback)
    };
    assert_eq!(cruise_after.counter, 0,         "burst MUST reset counter to 0");
    assert_eq!(cruise_after.current_gear, 0,    "burst MUST select Gear 0");
    assert_eq!(cruise_after.last_burst_frame, 9999,
        "burst MUST stamp last_burst_frame with current_frame");
}

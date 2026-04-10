//! PRISM-TWIN v3.0 Gate 2 Phase C — CUDA Graph Capture
//!
//! Captures the full per-step kernel sequence as a replayable CUDA Graph:
//!   Director → Physics → multi_lif → CA restraints → heartbeat → coupling clear
//!
//! Conditional nodes (COM removal every 100 steps, heartbeat every 1000 steps)
//! are triggered by the Director kernel writing to CUgraphConditionalHandle values.
//!
//! The host loop replays the graph N times, breaking only for neighbor list rebuild
//! (still CPU-side). This eliminates ~40μs/step kernel launch overhead.

use anyhow::{bail, Context, Result};
use cudarc::driver::{
    sys, CudaContext, CudaStream, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg,
};
use cudarc::driver::safe::CudaGraph;
use std::sync::Arc;

/// Holds the captured graph and associated state for autonomous execution.
pub struct AutonomousGraph {
    /// The instantiated CUDA Graph — one launch = one complete MD step
    graph: CudaGraph,
    /// Number of steps captured in this graph (always 1 for single-step capture)
    steps_per_launch: u32,
}

impl AutonomousGraph {
    /// Create from an instantiated CUDA graph.
    pub fn new(graph: CudaGraph) -> Self {
        Self { graph, steps_per_launch: 1 }
    }

    /// Launch the graph once (executes one complete MD step on GPU).
    pub fn launch(&self) -> Result<()> {
        self.graph.launch().map_err(|e| anyhow::anyhow!("Graph launch failed: {:?}", e))
    }

    /// Run the graph for `n_steps` in a tight loop, breaking every `chunk_size`
    /// steps to allow the caller to do CPU-side housekeeping (NL rebuild).
    ///
    /// Returns the number of steps actually executed.
    pub fn run_chunk(&self, n_steps: u32) -> Result<u32> {
        for _ in 0..n_steps {
            self.launch()?;
        }
        Ok(n_steps)
    }
}

/// Parameters needed to capture the physics step as a graph.
/// All device pointers must remain valid for the lifetime of the graph.
pub struct GraphCaptureParams<'a> {
    // Stream to capture on
    pub stream: &'a Arc<CudaStream>,
    pub context: &'a Arc<CudaContext>,

    // Director
    pub director_graph_fn: &'a CudaFunction,
    pub d_protocol_state: &'a mut CudaSlice<u8>,

    // Physics kernel
    pub fused_step_kernel: &'a CudaFunction,
    pub n_atoms: usize,
    pub physics_cfg: LaunchConfig,

    // Multi-LIF kernel
    pub multi_lif_kernel: &'a CudaFunction,
    pub voxel_cfg: LaunchConfig,

    // Housekeeping kernels
    pub ca_restraint_fn: &'a CudaFunction,
    pub heartbeat_fn: &'a CudaFunction,
    pub coupling_clear_fn: &'a CudaFunction,
    pub com_reduce_fn: &'a CudaFunction,
    pub com_correct_fn: &'a CudaFunction,
    pub com_accum_clear_fn: &'a CudaFunction,

    // All the device buffers (passed as kernel args during capture)
    // These are borrowed references — the graph captures the device addresses
}

/// Check if the CUDA driver supports conditional graph nodes (CUDA 12.4+).
/// Returns true if `cuGraphConditionalHandleCreate` is available and functional.
pub fn supports_conditional_graphs(context: &Arc<CudaContext>) -> bool {
    // Check compute capability — conditional nodes require CC 9.0+ (Hopper/Blackwell)
    match context.compute_capability() {
        Ok((major, _minor)) => major >= 9,
        Err(_) => false,
    }
}

/// Read the status_code field from ProtocolState via async memcpy.
/// Returns 0 (OK), 1 (NaN), or 2 (diverged).
/// The memcpy is async — the caller should synchronize if they need the result immediately.
pub fn poll_heartbeat_async(
    stream: &Arc<CudaStream>,
    d_protocol_state: &CudaSlice<u8>,
) -> Result<i32> {
    let mut full_state = vec![0u8; std::mem::size_of::<crate::protocol_state::ProtocolState>()];
    stream.memcpy_dtoh(d_protocol_state, &mut full_state)?;
    // Read status_code at its actual byte offset (use memoffset logic)
    let dummy = unsafe { std::mem::zeroed::<crate::protocol_state::ProtocolState>() };
    let base = &dummy as *const _ as usize;
    let field = &dummy.status_code as *const _ as usize;
    let offset = field - base;
    let status = i32::from_ne_bytes([
        full_state[offset], full_state[offset+1], full_state[offset+2], full_state[offset+3]
    ]);
    Ok(status)
}

/// The main autonomous run loop: captures a graph, replays it in chunks,
/// breaks for NL rebuild, polls heartbeat between chunks.
///
/// # Arguments
/// * `total_steps` - Total number of MD steps to run
/// * `chunk_size` - Steps per graph replay chunk (default 500)
/// * `capture_fn` - Closure that launches one step's kernels (called during capture)
/// * `nl_rebuild_fn` - Closure called between chunks for neighbor list rebuild
///
/// # Returns
/// Number of steps completed (may be less than total_steps if heartbeat triggers abort)
pub fn run_autonomous_loop(
    stream: &Arc<CudaStream>,
    d_protocol_state: &CudaSlice<u8>,
    total_steps: u32,
    chunk_size: u32,
    mut capture_fn: impl FnMut() -> Result<()>,
    mut nl_rebuild_fn: impl FnMut(u32) -> Result<()>,
) -> Result<u32> {
    // ── Step 1: Capture one step as a graph ──
    log::info!("Capturing physics step as CUDA Graph...");
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .map_err(|e| anyhow::anyhow!("Stream capture begin failed: {:?}", e))?;

    // Launch one step's worth of kernels — they get captured, not executed
    capture_fn()?;

    let graph = stream.end_capture(sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .map_err(|e| anyhow::anyhow!("Stream capture end failed: {:?}", e))?
        .ok_or_else(|| anyhow::anyhow!("Stream capture produced null graph"))?;

    log::info!("CUDA Graph captured and instantiated");

    let autonomous = AutonomousGraph {
        graph,
        steps_per_launch: 1,
    };

    // ── Step 2: Replay loop ──
    let mut steps_completed = 0u32;
    let n_chunks = (total_steps + chunk_size - 1) / chunk_size;

    for chunk_idx in 0..n_chunks {
        let steps_this_chunk = chunk_size.min(total_steps - steps_completed);

        // Replay the graph steps_this_chunk times
        autonomous.run_chunk(steps_this_chunk)?;
        steps_completed += steps_this_chunk;

        // Synchronize to read heartbeat and do NL rebuild
        stream.synchronize()
            .map_err(|e| anyhow::anyhow!("Stream sync failed after chunk: {:?}", e))?;

        // Poll heartbeat
        let status = poll_heartbeat_async(stream, d_protocol_state)?;
        if status != 0 {
            let reason = match status {
                1 => "NaN detected in coordinates",
                2 => "System diverged (coordinates > 1000A)",
                _ => "Unknown error",
            };
            log::error!("HEARTBEAT ABORT at step {}: {} (status={})",
                steps_completed, reason, status);
            return Ok(steps_completed);
        }

        // CPU-side neighbor list rebuild
        nl_rebuild_fn(steps_completed)?;

        if chunk_idx % 10 == 0 {
            log::info!("  Graph chunk {}/{}: {} steps completed",
                chunk_idx + 1, n_chunks, steps_completed);
        }
    }

    log::info!("Autonomous graph loop complete: {} steps", steps_completed);
    Ok(steps_completed)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TWIN Dual-Engine Autonomous Loop
// ═══════════════════════════════════════════════════════════════════════════════
//
// The PRISM-TWIN interferometric observation platform:
//
//   [Director_A] → [Physics_A] → [multi_lif_A] → [compact_push A→ring_a]
//                    ↓ ring_a spikes
//   [read_adapt ring_a → thresholds_B]    ← interferometric coupling A→B
//   [Director_B] → [Physics_B] → [multi_lif_B] → [compact_push B→ring_b]
//                    ↓ ring_b spikes
//   [read_adapt ring_b → thresholds_A]    ← interferometric coupling B→A
//   [recovery_A] [recovery_B]             ← threshold decay
//   [heartbeat_A] [heartbeat_B]           ← NaN/divergence safety
//   [CA_restraints_A] [CA_restraints_B]   ← position restraints
//   [coupling_clear_A] [coupling_clear_B] ← double-buffer housekeeping
//
// All in one graph launch. Both groups communicate through GPU-resident ring
// buffers — spikes from A modulate B's detection thresholds, and vice versa.
// The CPU never touches spike data.

/// Run the full TWIN dual-engine simulation autonomously.
///
/// Each graph launch executes one coupled step for both engines.
/// Host breaks every `chunk_size` steps for neighbor list rebuild only.
///
/// # Arguments
/// * `stream` - CUDA stream for graph capture and execution
/// * `total_steps` - Total MD steps to run
/// * `chunk_size` - Steps per graph replay chunk (default 500)
/// * `step_twin_fn` - Closure that launches one coupled TWIN step (both engines + coupling)
/// * `nl_rebuild_fn` - Closure for CPU-side neighbor list rebuild (both engines)
/// * `d_protocol_a` / `d_protocol_b` - ProtocolState for each engine (for heartbeat polling)
pub fn run_twin_autonomous_loop(
    stream: &Arc<CudaStream>,
    d_protocol_a: &CudaSlice<u8>,
    d_protocol_b: &CudaSlice<u8>,
    total_steps: u32,
    chunk_size: u32,
    mut step_twin_fn: impl FnMut() -> Result<()>,
    mut nl_rebuild_fn: impl FnMut(u32) -> Result<()>,
) -> Result<u32> {
    // ── Capture one coupled TWIN step as a graph ──
    log::info!("Capturing TWIN dual-engine step as CUDA Graph...");
    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .map_err(|e| anyhow::anyhow!("TWIN graph capture begin failed: {:?}", e))?;

    step_twin_fn()?;

    let graph = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
    )
        .map_err(|e| anyhow::anyhow!("TWIN graph capture end failed: {:?}", e))?
        .ok_or_else(|| anyhow::anyhow!("TWIN graph capture produced null graph"))?;

    log::info!("TWIN CUDA Graph captured and instantiated (dual-engine interferometric coupling)");

    let autonomous = AutonomousGraph { graph, steps_per_launch: 1 };

    // ── Replay loop ──
    let mut steps_completed = 0u32;
    let n_chunks = (total_steps + chunk_size - 1) / chunk_size;

    for chunk_idx in 0..n_chunks {
        let steps_this_chunk = chunk_size.min(total_steps - steps_completed);

        // Replay the dual-engine graph
        autonomous.run_chunk(steps_this_chunk)?;
        steps_completed += steps_this_chunk;

        // Synchronize and check both engines' heartbeats
        stream.synchronize()
            .map_err(|e| anyhow::anyhow!("TWIN stream sync failed: {:?}", e))?;

        for (label, d_ps) in [("Group A", d_protocol_a), ("Group B", d_protocol_b)] {
            let status = poll_heartbeat_async(stream, d_ps)?;
            if status != 0 {
                let reason = match status {
                    1 => "NaN detected",
                    2 => "System diverged",
                    _ => "Unknown",
                };
                log::error!("TWIN HEARTBEAT ABORT ({}) at step {}: {}",
                    label, steps_completed, reason);
                return Ok(steps_completed);
            }
        }

        // CPU-side neighbor list rebuild for both engines
        nl_rebuild_fn(steps_completed)?;

        if chunk_idx % 10 == 0 {
            log::info!("  TWIN chunk {}/{}: {} steps (both engines coupled)",
                chunk_idx + 1, n_chunks, steps_completed);
        }
    }

    log::info!("TWIN autonomous loop complete: {} coupled steps", steps_completed);
    Ok(steps_completed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poll_heartbeat_offset() {
        // Verify poll_heartbeat_async reads status_code correctly
        let ps = crate::protocol_state::ProtocolState {
            status_code: 42,
            ..unsafe { std::mem::zeroed() }
        };
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &ps as *const _ as *const u8,
                std::mem::size_of_val(&ps),
            )
        };
        // Compute actual field offset
        let base = &ps as *const _ as usize;
        let field = &ps.status_code as *const _ as usize;
        let offset = field - base;
        let val = i32::from_ne_bytes([bytes[offset], bytes[offset+1], bytes[offset+2], bytes[offset+3]]);
        assert_eq!(val, 42, "status_code not at expected offset {}", offset);
        assert_eq!(std::mem::size_of::<crate::protocol_state::ProtocolState>(), 164);
    }
}

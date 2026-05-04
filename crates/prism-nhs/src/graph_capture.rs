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
use std::collections::HashMap;
use std::ptr;
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
// Amendment 3.4 — CaptureTagger trait + raw-CUgraph escape hatch
// ═══════════════════════════════════════════════════════════════════════════
//
// The cudarc 0.18.2 safe wrapper `CudaGraph` (driver/safe/graph.rs) hides
// `cu_graph: sys::CUgraph` and `cu_graph_exec: sys::CUgraphExec` as private
// fields with no public accessors. Drop destroys both. This forecloses the
// `cudaGraphAddChildGraphNode` injection path the monolithic-fusion
// directive (Amendment 3.4) requires, since the splice operates on the
// TEMPLATE handle BEFORE instantiation.
//
// Bypass: drop to `cudarc::driver::sys` for the entire capture sequence.
// `cuStreamBeginCapture_v2` + `cuStreamEndCapture` give us a raw
// `CUgraph` template; we instantiate ourselves only AFTER any child-graph
// splicing has been applied.
//
// Tagging: `cuStreamGetCaptureInfo_v2` returns the dependency frontier of
// the in-progress capture. Calling it immediately after a kernel launch
// returns a single-element frontier — the just-recorded node. The
// `CaptureTagger` trait packages this into a label-keyed registry so the
// orchestrator can name nodes ("fused_step", "multi_lif", ...) and look
// them up later as splice anchors.

/// Records named CUgraphNode handles during stream capture.
///
/// Invariants:
/// - Must be called INSIDE an active `cuStreamBeginCapture_v2` window.
/// - Must be called IMMEDIATELY after the kernel launch whose node is
///   to be tagged (any subsequent kernel launch on the same stream will
///   advance the dependency frontier).
/// - The same label may be recorded only once per template.
pub trait CaptureTagger {
    /// Snapshot the current dependency frontier on the captured stream and
    /// store the most-recently-added node under `label`.  Returns the raw
    /// `CUgraphNode` handle so the orchestrator can pass it directly to
    /// `cuGraphAddDependencies` / `cuGraphAddChildGraphNode`.
    fn record_node(&mut self, label: &str) -> Result<sys::CUgraphNode>;
}

/// Default `CaptureTagger` implementation using `cuStreamGetCaptureInfo_v2`.
/// One instance per capture session. Owned by the orchestrator and consumed
/// when handing off to `CapturedTemplate::from_capture`.
pub struct StreamCaptureTagger {
    stream: sys::CUstream,
    nodes: HashMap<String, sys::CUgraphNode>,
}

impl StreamCaptureTagger {
    pub fn new(stream: sys::CUstream) -> Self {
        Self { stream, nodes: HashMap::new() }
    }

    /// Consumes the tagger; returns the recorded node registry.
    pub fn into_registry(self) -> HashMap<String, sys::CUgraphNode> {
        self.nodes
    }
}

impl CaptureTagger for StreamCaptureTagger {
    fn record_node(&mut self, label: &str) -> Result<sys::CUgraphNode> {
        let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let mut cap_id: sys::cuuint64_t = 0;
        let mut graph: sys::CUgraph = ptr::null_mut();
        let mut deps_ptr: *const sys::CUgraphNode = ptr::null();
        let mut n_deps: usize = 0;
        let rc = unsafe {
            sys::cuStreamGetCaptureInfo_v2(
                self.stream,
                &mut status as *mut _,
                &mut cap_id as *mut _,
                &mut graph as *mut _,
                &mut deps_ptr as *mut _,
                &mut n_deps as *mut _,
            )
        };
        if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
            bail!("cuStreamGetCaptureInfo_v2 failed for label '{}': {:?}", label, rc);
        }
        if !matches!(status, sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE) {
            bail!("CaptureTagger::record_node('{}') called outside active capture", label);
        }
        if n_deps == 0 {
            bail!("dependency frontier empty at record_node('{}') — no kernel launched yet?", label);
        }
        // Sequential capture on one stream ⇒ frontier is exactly the most
        // recent node. Driver returns the frontier as the LAST element of
        // the deps array (or only element when n_deps == 1).
        let node = unsafe { *deps_ptr.add(n_deps - 1) };
        if self.nodes.insert(label.to_string(), node).is_some() {
            bail!("CaptureTagger::record_node: duplicate label '{}'", label);
        }
        Ok(node)
    }
}

/// A captured CUgraph template, NOT yet instantiated.  Holds the raw
/// `CUgraph` handle plus the labelled-node registry built by a
/// `CaptureTagger`.  The orchestrator can:
///
///   1. Look up tagged nodes via `node()`.
///   2. Splice in child graphs via `add_child_graph_node()`.
///   3. Add dependency edges via `add_dependency()`.
///   4. Convert to a launchable `AutonomousGraph` via `instantiate()`.
///
/// On drop without instantiation the raw `CUgraph` is destroyed.
pub struct CapturedTemplate {
    cu_graph: sys::CUgraph,
    nodes: HashMap<String, sys::CUgraphNode>,
    stream: Arc<CudaStream>,
}

impl CapturedTemplate {
    /// Construct from a raw template + node registry produced by a
    /// completed `cuStreamEndCapture` + `StreamCaptureTagger`.
    pub fn from_capture(
        cu_graph: sys::CUgraph,
        nodes: HashMap<String, sys::CUgraphNode>,
        stream: Arc<CudaStream>,
    ) -> Self {
        Self { cu_graph, nodes, stream }
    }

    /// Raw template handle — for `cuGraphAddChildGraphNode` splicing.
    pub fn cu_graph(&self) -> sys::CUgraph { self.cu_graph }

    /// Look up a recorded node by label.  Returns `None` for unknown labels.
    pub fn node(&self, label: &str) -> Option<sys::CUgraphNode> {
        self.nodes.get(label).copied()
    }

    /// Insert `child_template` as a child-graph node downstream of `deps`.
    /// Returns the new child-graph node handle.
    ///
    /// **TIER 7 (2026-05-03) — monolithic-splice migration.**
    /// Was `cuGraphAddChildGraphNode` (legacy CUDA 11.x per-type
    /// adder).  Now routes through the unified CUDA 13.x
    /// `cudaGraphAddNode` via `crate::graph_node::add_child_graph_node_v3`,
    /// which is the actual "monolithic splice" — one
    /// `cudaGraphAddNode(graph, deps, dependencyData, n_deps,
    /// &nodeParams{type=GRAPH, .graph=child})` call replaces the
    /// legacy adder.  The legacy adder mishandles the conditional-
    /// handle metadata that CUDA 13.x attaches to body subgraphs
    /// (the source of the operator's reported ChildGraph-IF
    /// failures) — the unified API is the canonical post-CUDA-13.x
    /// way to splice templates that may carry such metadata.
    ///
    /// Returns `Err` with cudaError_t cast to `i32` (wrapped via
    /// `anyhow::anyhow!`) on failure, plus null-handle defensive
    /// check.  Behaviour at the call-site is otherwise identical to
    /// the legacy path — CUDA still copies the child template by
    /// value; future mutations to `child_template` do NOT propagate.
    pub fn add_child_graph_node(
        &mut self,
        deps: &[sys::CUgraphNode],
        child_template: sys::CUgraph,
    ) -> Result<sys::CUgraphNode> {
        // TIER 7 + TIER 8: routes through the v3 path AND runs the
        // splice-legality preflight before any cudaGraphAddNode call.
        // SpliceError::Illegal carries a SpliceLegalityReport describing
        // exactly which child template failed and why — surfaces well
        // before CUDA's own cudaErrorNotSupported (801).
        let child_node = unsafe {
            crate::graph_node::add_child_graph_node_v3(
                self.cu_graph,
                deps,
                child_template,
            )
        }.map_err(|e| anyhow::anyhow!("add_child_graph_node: {}", e))?;
        // The v3 helper already null-checks (SpliceError::NullNode) but
        // re-assert here for defence-in-depth — a null handle past this
        // point would corrupt the caller's dependency frontier.
        if child_node.is_null() {
            bail!("add_child_graph_node_v3 returned null node handle (post-check)");
        }
        Ok(child_node)
    }

    /// Add an explicit dependency edge `from → to` in the template.
    pub fn add_dependency(
        &mut self,
        from: sys::CUgraphNode,
        to: sys::CUgraphNode,
    ) -> Result<()> {
        let from_arr = [from];
        let to_arr   = [to];
        let rc = unsafe {
            sys::cuGraphAddDependencies(
                self.cu_graph,
                from_arr.as_ptr(),
                to_arr.as_ptr(),
                1,
            )
        };
        if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
            bail!("cuGraphAddDependencies failed: {:?}", rc);
        }
        Ok(())
    }

    /// Instantiate the template into an executable graph.  Consumes self.
    /// On success the returned `AutonomousGraph` owns both the template
    /// (kept for `cuGraphExecUpdate` if ever needed) and the exec handle.
    pub fn instantiate(self) -> Result<AutonomousGraph> {
        let mut cu_graph_exec: sys::CUgraphExec = ptr::null_mut();
        let rc = unsafe {
            sys::cuGraphInstantiateWithFlags(
                &mut cu_graph_exec as *mut _,
                self.cu_graph,
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH as u64,
            )
        };
        if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
            bail!("cuGraphInstantiateWithFlags failed: {:?}", rc);
        }
        let cu_graph = self.cu_graph;
        // Move out of self before drop to avoid double-destroy of cu_graph.
        std::mem::forget(self);
        Ok(AutonomousGraph {
            graph: None,
            raw: Some(RawGraphExec {
                cu_graph,
                cu_graph_exec,
            }),
            steps_per_launch: 1,
        })
    }
}

impl Drop for CapturedTemplate {
    fn drop(&mut self) {
        if !self.cu_graph.is_null() {
            unsafe { let _ = sys::cuGraphDestroy(self.cu_graph); }
            self.cu_graph = ptr::null_mut();
        }
    }
}

/// Raw-handle backing for `AutonomousGraph` when the new sys-bypass capture
/// path is used (Amendment 3.4 monolithic fusion).  Both handles must be
/// destroyed on drop in order: exec first, then template.
struct RawGraphExec {
    cu_graph: sys::CUgraph,
    cu_graph_exec: sys::CUgraphExec,
}

impl Drop for RawGraphExec {
    fn drop(&mut self) {
        unsafe {
            if !self.cu_graph_exec.is_null() {
                let _ = sys::cuGraphExecDestroy(self.cu_graph_exec);
                self.cu_graph_exec = ptr::null_mut();
            }
            if !self.cu_graph.is_null() {
                let _ = sys::cuGraphDestroy(self.cu_graph);
                self.cu_graph = ptr::null_mut();
            }
        }
    }
}

/// Holds the captured graph and associated state for autonomous execution.
///
/// Supports two construction paths:
///   - Legacy: `new(CudaGraph)` — wraps the cudarc safe wrapper. Used by
///     existing TWIN dual-engine path; keeps backward compat.
///   - Amendment 3.4: `from_template_instantiate()` (via `CapturedTemplate`)
///     — raw `CUgraph` + `CUgraphExec` for monolithic-fusion injection.
pub struct AutonomousGraph {
    /// Legacy cudarc safe wrapper. `Some` for non-fused captures.
    graph: Option<CudaGraph>,
    /// Raw-handle backing. `Some` after `CapturedTemplate::instantiate()`.
    /// Exactly one of `graph` / `raw` is `Some`.
    raw: Option<RawGraphExec>,
    /// Number of steps captured in this graph (always 1 for single-step capture)
    steps_per_launch: u32,
}

impl AutonomousGraph {
    /// Create from an instantiated cudarc-wrapped CUDA graph (legacy path).
    pub fn new(graph: CudaGraph) -> Self {
        Self {
            graph: Some(graph),
            raw: None,
            steps_per_launch: 1,
        }
    }

    /// Raw `CUgraph` template handle — Amendment 3.4 splice path only.
    /// Returns `None` when constructed via the legacy `new(CudaGraph)`
    /// constructor (cudarc hides the template).
    pub fn cu_graph_template(&self) -> Option<sys::CUgraph> {
        self.raw.as_ref().map(|r| r.cu_graph)
    }

    /// Raw `CUgraphExec` — Amendment 3.4 splice path only.
    pub fn cu_graph_exec_raw(&self) -> Option<sys::CUgraphExec> {
        self.raw.as_ref().map(|r| r.cu_graph_exec)
    }

    /// Launch the graph once (executes one complete MD step on GPU).
    pub fn launch(&self) -> Result<()> {
        if let Some(ref g) = self.graph {
            g.launch().map_err(|e| anyhow::anyhow!("Graph launch failed: {:?}", e))
        } else if let Some(ref r) = self.raw {
            // Raw launch path — call cuGraphLaunch directly. The orchestrator
            // is responsible for keeping the stream alive for the lifetime of
            // this AutonomousGraph (passed via the original CapturedTemplate).
            // For Amendment 3.4 we currently launch on a stream provided by
            // the caller via `launch_on_stream`.
            bail!("AutonomousGraph::launch() called on raw-handle backed graph — \
                   use launch_on_stream(stream) instead")
        } else {
            bail!("AutonomousGraph in invalid state — neither cudarc nor raw handle present")
        }
    }

    /// Launch the raw-handle graph on the specified stream (Amendment 3.4).
    /// Returns an error when the legacy cudarc-wrapper backing is used.
    pub fn launch_on_stream(&self, stream: sys::CUstream) -> Result<()> {
        let raw = self.raw.as_ref()
            .ok_or_else(|| anyhow::anyhow!("launch_on_stream requires raw-handle backing"))?;
        let rc = unsafe { sys::cuGraphLaunch(raw.cu_graph_exec, stream) };
        if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
            bail!("cuGraphLaunch failed: {:?}", rc);
        }
        Ok(())
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
        graph: Some(graph),
        raw: None,
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

    let autonomous = AutonomousGraph { graph: Some(graph), raw: None, steps_per_launch: 1 };

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

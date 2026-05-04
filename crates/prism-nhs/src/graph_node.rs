//! CUDA 13.x `cuGraphAddNode` — unified graph-builder Rust bindings.
//!
//! Begun 2026-05-03 — operator directive "migration to cuGraphAddNode for
//! a monolithic splice as planned so we don't continue to have
//! childgraphIF failures bc we are using the wrong name conventions".
//!
//! ## Why this module exists
//!
//! cudarc 0.18.2 does not expose CUDA 13.x's `cuGraphAddNode` symbol
//! (only `cuGraphInstantiate` is bound; the unified
//! `cuGraphAddNode_v2` / `_v3` family is missing). Two Rust call
//! sites still drive graph construction through the legacy per-type
//! adders:
//!
//! | Call site                                | Legacy adder                  |
//! |------------------------------------------|-------------------------------|
//! | `graph_capture::PrismGraphTemplate::splice_child_template` | `cuGraphAddChildGraphNode` |
//! | `captured_pipeline::add_burst_marker_memset_node`           | `cuGraphAddMemsetNode`     |
//!
//! The legacy adders mishandle CUDA 13.x conditional-handle metadata
//! attached to body subgraphs (the source of the operator's reported
//! "ChildGraph IF" failures), so we route Rust call sites through C++
//! FFI helpers (`prism_graph_add_*_v3_ffi` in
//! `src/cuda/graph_node.cu`) that internally invoke the unified
//! `cudaGraphAddNode`.
//!
//! ## Migration status
//!
//! Phase 1 (this commit):
//!   - C++ FFI scaffold (`src/cuda/graph_node.cu`) — DONE
//!   - Rust extern + safe wrappers (this file) — DONE
//!   - Pilot migration: `graph_capture::splice_child_template`     ← in progress
//!
//! Phase 2 (follow-up):
//!   - Migrate `captured_pipeline::add_burst_marker_memset_node`
//!   - Audit comment-only references to legacy adders for accuracy
//!   - Retire `cuGraphAddChildGraphNode` / `cuGraphAddMemsetNode`
//!     extern declarations once both sites are migrated and a
//!     captured-graph run validates without ChildGraph-IF errors.

use cudarc::driver::sys::{CUgraph, CUgraphNode};
use std::ffi::c_void;

#[link(name = "graph_node", kind = "static")]
extern "C" {
    fn prism_graph_add_child_node_v3_ffi(
        parent_graph:    CUgraph,
        p_dependencies:  *const CUgraphNode,
        num_deps:        usize,
        child_template:  CUgraph,
        p_out_node:      *mut CUgraphNode,
    ) -> i32;

    fn prism_graph_add_memset_node_v3_ffi(
        parent_graph:    CUgraph,
        p_dependencies:  *const CUgraphNode,
        num_deps:        usize,
        dst:             *mut c_void,
        pitch:           usize,
        value_u32:       u32,
        element_size:    u32,
        width:           usize,
        height:          usize,
        p_out_node:      *mut CUgraphNode,
    ) -> i32;

    /// TIER 8 — preflight legality check.  Returns 0 on successful
    /// inspection; nonzero cudaError_t on inspection failure.  On
    /// success, the node-count outputs are populated.  Any nonzero
    /// conditional / alloc / free count means the child graph is
    /// **NOT** safe to splice via `cudaGraphAddNode(GRAPH)`.
    fn prism_graph_is_splice_legal_ffi(
        child_graph:            CUgraph,
        p_out_total_nodes:      *mut usize,
        p_out_conditional_nodes:*mut usize,
        p_out_alloc_nodes:      *mut usize,
        p_out_free_nodes:       *mut usize,
    ) -> i32;
}

/// **TIER 8 (2026-05-03)** — Splice-legality preflight result.
///
/// `is_legal() == true` ⇒ child graph is splice-legal under PRISM's
/// GRAPH-SPLICE-001 invariant.  Conditional, allocation, and free
/// nodes must be lifted to the parent/control graph.
#[derive(Debug, Clone, Copy)]
pub struct SpliceLegalityReport {
    pub total_nodes:       usize,
    pub conditional_count: usize,
    pub allocation_count:  usize,
    pub free_count:        usize,
}

impl SpliceLegalityReport {
    /// Cheap predicate: this child graph is safe to splice.
    pub fn is_legal(&self) -> bool {
        self.conditional_count == 0 &&
        self.allocation_count  == 0 &&
        self.free_count        == 0
    }
}

/// **TIER 8 (2026-05-03)** — Splice-attempt error taxonomy.
///
/// Distinguishes the three failure modes a `cudaGraphAddNode(GRAPH)`
/// invocation can hit so the orchestrator can react with policy:
///   - `Cuda(rc)` — generic CUDA driver failure during the splice
///     itself (e.g., out-of-memory, invalid handle).
///   - `NullNode` — defensive guard; CUDA reported success but
///     returned a null handle (driver bug or contract violation).
///   - `Illegal(report)` — the preflight check found conditional
///     nodes in the child template; the splice was NEVER attempted.
///     This is the **PRISM-specific** error that fires before CUDA
///     can return 801 on the actual splice.
#[derive(Debug)]
pub enum SpliceError {
    /// Inspection or splice itself returned a CUDA driver error.
    Cuda(i32),
    /// Splice succeeded but yielded a null node handle.
    NullNode,
    /// Preflight legality check rejected the child template — it
    /// contains conditional nodes that CUDA 13.x cannot splice.
    Illegal(SpliceLegalityReport),
}

impl std::fmt::Display for SpliceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpliceError::Cuda(rc) =>
                write!(f, "splice cuda error: rc={}", rc),
            SpliceError::NullNode =>
                write!(f, "splice succeeded but returned null node handle"),
            SpliceError::Illegal(r) => write!(
                f,
                "splice ILLEGAL — child graph contains conditional={} alloc={} free={} \
                 node(s) of {} total; CUDA 13.x graph-splice policy rejects \
                 cudaGraphAddNode(GRAPH) children with conditional/allocation/free \
                 nodes. TIER 8 fix: lift control/allocation nodes to the parent graph.",
                r.conditional_count,
                r.allocation_count,
                r.free_count,
                r.total_nodes
            ),
        }
    }
}

impl std::error::Error for SpliceError {}

/// **TIER 8 (2026-05-03)** — Inspect `child_graph` for splice
/// legality without actually attempting a splice.
///
/// # Safety
/// `child_graph` must be a valid live `CUgraph` handle.
pub unsafe fn splice_legality_check(
    child_graph: CUgraph,
) -> Result<SpliceLegalityReport, i32> {
    let mut total: usize = 0;
    let mut cond:  usize = 0;
    let mut alloc: usize = 0;
    let mut free:  usize = 0;
    let rc = prism_graph_is_splice_legal_ffi(
        child_graph,
        &mut total as *mut _,
        &mut cond  as *mut _,
        &mut alloc as *mut _,
        &mut free  as *mut _,
    );
    if rc != 0 {
        return Err(rc);
    }
    Ok(SpliceLegalityReport {
        total_nodes:       total,
        conditional_count: cond,
        allocation_count:  alloc,
        free_count:        free,
    })
}

/// Splice a complete child template into `parent_graph` as one
/// monolithic GRAPH-type node (CUDA 13.x `cudaGraphAddNode` with
/// `nodeParams.type = cudaGraphNodeTypeGraph`).
///
/// **TIER 8 (2026-05-03)** — runs `splice_legality_check` BEFORE
/// the splice attempt.  If the child template contains any
/// conditional / allocation / free nodes, returns
/// `Err(SpliceError::Illegal(report))`
/// without ever calling `cudaGraphAddNode`.  This is the PRISM-
/// specific gate that fires before CUDA can return 801
/// (cudaErrorNotSupported) on the splice itself.
///
/// Returns the new node handle on success.  The child template is
/// **copied** by CUDA — subsequent mutations to `child_template` do
/// not propagate to the spliced instance.
///
/// # Errors
///
/// - `SpliceError::Illegal(report)` — preflight rejected the splice
///   (see [`SpliceLegalityReport`]).  No splice was attempted.
/// - `SpliceError::Cuda(rc)` — CUDA returned a non-success
///   `cudaError_t` from either inspection or the splice itself.
/// - `SpliceError::NullNode` — splice reported success but returned
///   a null handle (defensive).
///
/// # Safety
///
/// Caller must ensure:
/// - `parent_graph` and `child_template` are valid live `CUgraph` handles.
/// - All `CUgraphNode` entries in `deps` belong to `parent_graph`.
/// - `parent_graph` is not currently being captured by an active
///   `cuStreamBeginCapture` window on a stream other than the build
///   thread (THREAD_LOCAL capture mode keeps build-thread captures
///   safe — see `captured_pipeline.rs:1381` rationale).
pub unsafe fn add_child_graph_node_v3(
    parent_graph:   CUgraph,
    deps:           &[CUgraphNode],
    child_template: CUgraph,
) -> Result<CUgraphNode, SpliceError> {
    // TIER 8 preflight — fail PRISM-side BEFORE CUDA returns 801.
    let report = splice_legality_check(child_template)
        .map_err(SpliceError::Cuda)?;
    if !report.is_legal() {
        return Err(SpliceError::Illegal(report));
    }

    // Splice via the unified CUDA 13.x cudaGraphAddNode (TIER 7).
    let mut out_node: CUgraphNode = std::ptr::null_mut();
    let p_deps = if deps.is_empty() { std::ptr::null() } else { deps.as_ptr() };
    let rc = prism_graph_add_child_node_v3_ffi(
        parent_graph,
        p_deps,
        deps.len(),
        child_template,
        &mut out_node as *mut _,
    );
    if rc != 0 {
        return Err(SpliceError::Cuda(rc));
    }
    if out_node.is_null() {
        return Err(SpliceError::NullNode);
    }
    Ok(out_node)
}

/// Add a MEMSET node via CUDA 13.x `cudaGraphAddNode`.
///
/// `pitch` is bytes per row; for tight 1-D MEMSETs use
/// `element_size * width`. `value_u32` is truncated to the low
/// `element_size` bytes per CUDA semantics. `element_size` must be
/// 1, 2, or 4 (CUDA-imposed).
///
/// # Safety
///
/// Same invariants as `add_child_graph_node_v3` for the parent graph
/// + deps. `dst` must be a valid CUdeviceptr cast to `*mut c_void`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn add_memset_node_v3(
    parent_graph: CUgraph,
    deps:         &[CUgraphNode],
    dst:          *mut c_void,
    pitch:        usize,
    value_u32:    u32,
    element_size: u32,
    width:        usize,
    height:       usize,
) -> Result<CUgraphNode, i32> {
    let mut out_node: CUgraphNode = std::ptr::null_mut();
    let p_deps = if deps.is_empty() { std::ptr::null() } else { deps.as_ptr() };
    let rc = prism_graph_add_memset_node_v3_ffi(
        parent_graph,
        p_deps,
        deps.len(),
        dst,
        pitch,
        value_u32,
        element_size,
        width,
        height,
        &mut out_node as *mut _,
    );
    if rc != 0 {
        return Err(rc);
    }
    if out_node.is_null() {
        return Err(-1);
    }
    Ok(out_node)
}

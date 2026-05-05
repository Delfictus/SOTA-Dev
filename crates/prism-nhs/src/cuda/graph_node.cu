// ═══════════════════════════════════════════════════════════════════════
// PRISM-4D / Graph Node Builder — CUDA 13.x cuGraphAddNode FFI Wrappers
// ═══════════════════════════════════════════════════════════════════════
//
// Implementation. See graph_node.cuh for the full design rationale.
//
// All wrappers go through `cudaGraphAddNode(graph, deps, dependencyData,
// n_deps, &nodeParams)` — the CUDA 13.x unified surface — to keep the
// codebase on ONE graph-builder naming convention. The legacy per-type
// adders (`cuGraphAddChildGraphNode`, `cuGraphAddMemsetNode`, …) remain
// usable but are progressively retired by call-site migration.

#include "graph_node.cuh"
#include <cuda_runtime.h>
#include <cstring>
#include <vector>

extern "C" int prism_graph_add_child_node_v3_ffi(
    cudaGraph_t            parent_graph,
    const cudaGraphNode_t *pDependencies,
    size_t                 numDependencies,
    cudaGraph_t            child_template,
    cudaGraphNode_t       *pOutNode)
{
    if (!parent_graph || !child_template || !pOutNode) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (numDependencies > 0 && !pDependencies) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    // CUDA 13.x cudaGraphNodeParams — type-tagged union. Set type to
    // GRAPH and populate the .graph arm with the child template.
    // memset zeroes the union before we touch the active arm so any
    // padding/alignment bytes are deterministic (cudaGraphAddNode
    // validates the inactive arms to be zero on some toolkit
    // versions).
    // CUDA 13.x deletes the default ctor on `cudaGraphNodeParams`
    // (the union-bearing fields force explicit init).  Brace-init
    // value-initializes (zeroes) every byte; we then set the active
    // union arm + the discriminating `type` field.
    cudaGraphNodeParams nodeParams = {};
    nodeParams.type        = cudaGraphNodeTypeGraph;
    nodeParams.graph.graph = child_template;

    cudaError_t err = cudaGraphAddNode(
        pOutNode,
        parent_graph,
        /*pDependencies=*/  pDependencies,
        /*dependencyData=*/ nullptr,   // default cudaGraphEdgeTypeProgram
        /*numDependencies=*/numDependencies,
        &nodeParams);
    return static_cast<int>(err);
}

// ═══════════════════════════════════════════════════════════════════════
// TIER 8 — Preflight legality check.  Walks child_graph nodes and counts
// conditional / allocation / free nodes.  If any count is nonzero, the
// caller MUST NOT invoke prism_graph_add_child_node_v3_ffi.
//
// Implementation note: cudaGraphGetNodes is a 2-call protocol.  First
// call with nodes=nullptr discovers the count; second call fills the
// caller-allocated buffer.  For typical PRISM child templates the node
// count is small (10s), so a stack-allocated buffer up to 256 nodes is
// sufficient and avoids heap allocation; larger graphs fall back to
// std::vector.  Conditional nodes have type cudaGraphNodeTypeConditional
// (CUDA 13.x).
//
// Returns 0 on successful inspection; nonzero cudaError_t otherwise.
extern "C" int prism_graph_is_splice_legal_ffi(
    cudaGraph_t            child_graph,
    size_t                *pOutTotalNodes,
    size_t                *pOutConditionalNodes,
    size_t                *pOutAllocNodes,
    size_t                *pOutFreeNodes)
{
    if (pOutTotalNodes)       *pOutTotalNodes       = 0;
    if (pOutConditionalNodes) *pOutConditionalNodes = 0;
    if (pOutAllocNodes)       *pOutAllocNodes       = 0;
    if (pOutFreeNodes)        *pOutFreeNodes        = 0;
    if (!child_graph) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    // Pass 1: discover count.
    size_t n_nodes = 0;
    cudaError_t err = cudaGraphGetNodes(child_graph, /*nodes=*/nullptr, &n_nodes);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    if (pOutTotalNodes) *pOutTotalNodes = n_nodes;
    if (n_nodes == 0) {
        // Empty graph is trivially legal.
        return static_cast<int>(cudaSuccess);
    }

    // Pass 2: fetch the node handles.  Stack buffer for typical sizes,
    // std::vector for the large-graph tail.
    constexpr size_t STACK_CAP = 256;
    cudaGraphNode_t  stack_nodes[STACK_CAP];
    cudaGraphNode_t *nodes_ptr = stack_nodes;

    // RAII storage for the heap fallback so we don't leak on any return path.
    std::vector<cudaGraphNode_t> heap_nodes;
    if (n_nodes > STACK_CAP) {
        heap_nodes.resize(n_nodes);
        nodes_ptr = heap_nodes.data();
    }

    size_t fetched = n_nodes;
    err = cudaGraphGetNodes(child_graph, nodes_ptr, &fetched);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    if (fetched != n_nodes) {
        // Defensive — driver should have filled exactly n_nodes.
        n_nodes = fetched;
    }

    size_t conditional_count = 0;
    size_t alloc_count       = 0;
    size_t free_count        = 0;
    for (size_t i = 0; i < n_nodes; ++i) {
        cudaGraphNodeType node_type;
        err = cudaGraphNodeGetType(nodes_ptr[i], &node_type);
        if (err != cudaSuccess) {
            return static_cast<int>(err);
        }
        if (node_type == cudaGraphNodeTypeConditional) {
            ++conditional_count;
        } else if (node_type == cudaGraphNodeTypeMemAlloc) {
            ++alloc_count;
        } else if (node_type == cudaGraphNodeTypeMemFree) {
            ++free_count;
        }
    }

    if (pOutConditionalNodes) *pOutConditionalNodes = conditional_count;
    if (pOutAllocNodes)       *pOutAllocNodes       = alloc_count;
    if (pOutFreeNodes)        *pOutFreeNodes        = free_count;
    return static_cast<int>(cudaSuccess);
}

extern "C" int prism_graph_add_memset_node_v3_ffi(
    cudaGraph_t            parent_graph,
    const cudaGraphNode_t *pDependencies,
    size_t                 numDependencies,
    void                  *dst,
    size_t                 pitch,
    uint32_t               value_u32,
    unsigned int           element_size,
    size_t                 width,
    size_t                 height,
    cudaGraphNode_t       *pOutNode)
{
    if (!parent_graph || !dst || !pOutNode) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (numDependencies > 0 && !pDependencies) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (element_size != 1 && element_size != 2 && element_size != 4) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (width == 0 || height == 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    // CUDA 13.x deletes the default ctor on `cudaGraphNodeParams`
    // (the union-bearing fields force explicit init).  Brace-init
    // value-initializes (zeroes) every byte; we then set the active
    // union arm + the discriminating `type` field.
    cudaGraphNodeParams nodeParams = {};
    nodeParams.type = cudaGraphNodeTypeMemset;
    nodeParams.memset.dst         = dst;
    nodeParams.memset.pitch       = pitch;
    nodeParams.memset.value       = value_u32;
    nodeParams.memset.elementSize = element_size;
    nodeParams.memset.width       = width;
    nodeParams.memset.height      = height;

    cudaError_t err = cudaGraphAddNode(
        pOutNode,
        parent_graph,
        pDependencies,
        nullptr,
        numDependencies,
        &nodeParams);
    return static_cast<int>(err);
}

// ═══════════════════════════════════════════════════════════════════════
// TIER 8 — WHILE FFI scaffold (WHILE-FFI-SCAFFOLD-001, 2026-05-04).
//
// Mirrors the G26 SWITCH FFI pattern from
// crates/prism-nhs/src/cuda/adjudicator.cu:1119 (handle creator) and
// :1137 (node wirer) but specialised to:
//   - cudaGraphCondTypeWhile  (driver_types.h:3560, value = 1)
//   - conditional.size = 1u   (driver_types.h:3572 — the only allowed
//                              value for WHILE: "Allowed values are 1
//                              for cudaGraphCondTypeWhile")
//   - 1 body subgraph         (vs. SWITCH's 4)
//
// CUDA 12.4 floor. cudaGraphCondTypeWhile is supported since CUDA 12.4
// (CUDART_VERSION >= 12040). The G26 SWITCH path uses a stricter
// >= 12060 gate because cudaGraphCondTypeSwitch requires CUDA 12.6;
// WHILE landed two minor releases earlier.
//
// FFI scaffold is added but NOT WIRED into any production graph this
// commit — captured graphs remain unchanged. R6 lane will install the
// bounded post-T7 deferred-drain WHILE in a follow-up commit via
// captured_pipeline.rs.
// ═══════════════════════════════════════════════════════════════════════

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040

extern "C" int prism_graph_create_while_handle_ffi(
    cudaGraph_t   parent_graph,
    uint32_t      default_value,
    uint64_t     *pOutHandle)
{
    if (parent_graph == nullptr || pOutHandle == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    // cudaGraphConditionalHandle is `typedef unsigned long long`; we
    // round-trip via uint64_t so the FFI ABI is stable.
    cudaGraphConditionalHandle handle = 0;
    cudaError_t err = cudaGraphConditionalHandleCreate(
        &handle,
        parent_graph,
        /*defaultLaunchValue=*/ default_value,
        /*flags=*/              cudaGraphCondAssignDefault);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    *pOutHandle = static_cast<uint64_t>(handle);
    return static_cast<int>(cudaSuccess);
}

extern "C" int prism_graph_add_while_node_ffi(
    cudaGraph_t            parent_graph,
    const cudaGraphNode_t *pDependencies,
    size_t                 numDependencies,
    uint64_t               handle_v,
    cudaGraphNode_t       *pOutConditionalNode,
    cudaGraph_t           *pOutBodySubgraph)
{
    if (parent_graph == nullptr ||
        pOutConditionalNode == nullptr ||
        pOutBodySubgraph == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (numDependencies > 0 && pDependencies == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaGraphConditionalHandle handle =
        static_cast<cudaGraphConditionalHandle>(handle_v);

    *pOutBodySubgraph = nullptr;

    // CUDA 13.x deletes the default ctor on `cudaGraphNodeParams`
    // (the union-bearing fields force explicit init). Brace-init
    // value-initializes (zeroes) every byte; we then set the active
    // union arm + the discriminating `type` field.
    cudaGraphNodeParams nodeParams{};
    nodeParams.type                 = cudaGraphNodeTypeConditional;
    nodeParams.conditional.handle   = handle;
    nodeParams.conditional.type     = cudaGraphCondTypeWhile;
    // INVARIANT: WHILE size MUST be 1 (driver_types.h:3572).
    nodeParams.conditional.size     = 1u;
    nodeParams.conditional.ctx      = nullptr;

    cudaError_t err = cudaGraphAddNode(
        pOutConditionalNode,
        parent_graph,
        /*pDependencies=*/   pDependencies,
        /*dependencyData=*/  nullptr,
        /*numDependencies=*/ numDependencies,
        &nodeParams);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }

    // Per driver_types.h:3574-3588, phGraph_out is CUDA-owned and
    // populated during the cudaGraphAddNode call. For WHILE,
    // phGraph_out[0] is the loop body. Copy the handle for the caller
    // so they can populate the body without holding the nodeParams
    // struct alive.
    if (nodeParams.conditional.phGraph_out != nullptr) {
        *pOutBodySubgraph = nodeParams.conditional.phGraph_out[0];
    }

    return static_cast<int>(cudaSuccess);
}

#else  // CUDART_VERSION < 12040

extern "C" int prism_graph_create_while_handle_ffi(
    cudaGraph_t /*parent_graph*/,
    uint32_t    /*default_value*/,
    uint64_t   */*pOutHandle*/)
{
    return static_cast<int>(cudaErrorNotSupported);
}

extern "C" int prism_graph_add_while_node_ffi(
    cudaGraph_t           /*parent_graph*/,
    const cudaGraphNode_t */*pDependencies*/,
    size_t                /*numDependencies*/,
    uint64_t              /*handle_v*/,
    cudaGraphNode_t      */*pOutConditionalNode*/,
    cudaGraph_t          */*pOutBodySubgraph*/)
{
    return static_cast<int>(cudaErrorNotSupported);
}

#endif  // CUDART_VERSION >= 12040

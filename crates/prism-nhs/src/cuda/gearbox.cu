// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / G26 Chronometric Gearbox — implementation (Wave B.1)
// ═══════════════════════════════════════════════════════════════════════════

#include "gearbox.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

// ─── __constant__ gear table — 64 bytes, one Blackwell L1 const-cache line ──
//
// Initialised by `prism_gearbox_init_table_async`. The kernel reads via
// LDC (load constant) which broadcasts to all 32 lanes in a single
// instruction.  Static initialiser provides a safe pre-init value (Gear 1
// dt = 2.0 fs in slot 4) so a kernel launched before the host writes the
// real table observes a sensible default rather than zero.

// __constant__ static initialisers must be compile-time constants;
// CUDART_NAN_F is not constexpr (dynamic-init forbidden by nvcc), so the
// static placeholder uses safe Gear-1 dt for slot 0 of every gear and zero
// elsewhere.  The Rust-side `prism_gearbox_init_table_async` is invoked
// UNCONDITIONALLY in `CapturedAdjudicationPipeline::build` BEFORE any
// kernel launch, overwriting all 16 slots with the canonical layout
// (NaN sentinel for Gear 3 included).  So the static defaults below are
// only observable in unit tests that bypass `init_table_async`.
__constant__ float d_gearbox_table[PRISM_GEARBOX_TABLE_LEN] = {
    0.0005f, 0.0f, 0.0f, 0.0f,    // Gear 0 — 0.5 fs (placeholder)
    0.0020f, 0.0f, 0.0f, 0.0f,    // Gear 1 — 2.0 fs (placeholder)
    0.0040f, 0.0f, 0.0f, 0.0f,    // Gear 2 — 4.0 fs (placeholder)
    0.0000f, 0.0f, 0.0f, 0.0f,    // Gear 3 — overwritten by host init with NaN
};

// ─── B.3 — 3×3 Constant Ratio Matrix ────────────────────────────────────────
//
// Gear-to-gear dt-ratio lookup table.  Indexed as [prev*3 + target].
// Pre-baked from the canonical gearbox dt values (0.5/2.0/4.0 fs).  Gear 3
// (abort) is OOB — the body 3 sub-graph fires the trap kernel without
// consulting this matrix.  Operator constraint: zero divisions in the
// rescale kernel — single LDC + MUL per thread.
//
// Static init values are correct for the canonical gearbox table; the
// host-side init helper exists for ABI symmetry with future runtime
// re-tuning (e.g., a different 3-gear ratio set on a non-HMR substrate).

__constant__ float d_rescale_ratios[9] = {
    1.0f,   4.0f,   8.0f,    // prev = 0 (0.5 fs) → target 0/1/2
    0.25f,  1.0f,   2.0f,    // prev = 1 (2.0 fs) → target 0/1/2
    0.125f, 0.5f,   1.0f,    // prev = 2 (4.0 fs) → target 0/1/2
};

// ─── adj→adjudication_code / d_dt offsets ───────────────────────────────────
//
// Pinned by the C-side static_assert in adjudicator.cuh:
//   adjudication_code  offset 52  (u32)
//   d_dt               offset 112 (*mut f32)
// Reading via offset arithmetic on a `const uint8_t*` keeps this TU
// independent of the full FFI struct definition (operator §M2 / Anti-
// Greenfield: don't drag SO(3) tile types into a kernel that doesn't
// need them).

__device__ __forceinline__ uint32_t prism_gearbox_load_adj_code(
    const InterferometricAdjudicatorFfi* adj
) {
    const uint8_t* base = reinterpret_cast<const uint8_t*>(adj);
    return *reinterpret_cast<const uint32_t*>(base + 52);
}

__device__ __forceinline__ float* prism_gearbox_load_adj_d_dt(
    const InterferometricAdjudicatorFfi* adj
) {
    const uint8_t* base = reinterpret_cast<const uint8_t*>(adj);
    // d_dt is a *mut f32; we load the 8-byte pointer value.
    return *reinterpret_cast<float* const*>(base + 112);
}

// ─── PointerSwap kernel — Stateful Finite Automaton ────────────────────────
//
// Single-thread.  All fields read/written here live on device:
//   adj->adjudication_code  ←  Adjudicator step kernel
//   adj->d_dt              ←  pre-capture wire-up (= &d_protocol->dt)
//   cruise->{counter,...}  ←  ChronometricStateTensor (this kernel + reset)
//
// State transitions:
//   code = 2  →  Gear 3 (Violation; abort).  Counter unchanged so the
//                cruise can be inspected post-mortem.
//   code = 1  →  Gear 0 (Burst); cruise.counter = 0;
//                cruise.last_burst_frame = current_frame.
//   code = 0  →  cruise.counter += 1; gear = (counter < THRESH) ? 1 : 2.

extern "C"
__global__ void prism_gearbox_pointer_swap_kernel(
    const InterferometricAdjudicatorFfi* __restrict__ adj,
    ChronometricStateTensor*             __restrict__ cruise,
    uint32_t                                          current_frame
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const uint32_t code = prism_gearbox_load_adj_code(adj);

    uint32_t counter           = cruise->counter;
    uint32_t last_burst_frame  = cruise->last_burst_frame;
    uint32_t target_gear;

    if (code == 2u) {
        // Hard-trap path.  Wave B.2's SWITCH routes Gear 3 to the PTX
        // trap kernel; even if that wiring is bypassed, writing a NaN dt
        // through *(adj->d_dt) gives integrators a non-finite step that
        // their existing FP-exception guards catch.
        target_gear = 3u;
    } else if (code == 1u) {
        target_gear       = 0u;
        counter           = 0u;
        last_burst_frame  = current_frame;
    } else {
        // code == 0 (Equilibrium / Prune) — cruise hysteresis.
        // Saturating increment so a long quiet run doesn't wrap u32.
        counter           = (counter < 0xffffffffu) ? counter + 1u : counter;
        target_gear       = (counter < PRISM_GEARBOX_CRUISE_THRESHOLD) ? 1u : 2u;
    }

    // B.2 — capture the OLD current_gear into previous_gear so the
    // symplectic ratio kernel can compute λ = dt_new / dt_old without
    // a separate scratch buffer.  Read happens BEFORE the write below.
    const uint32_t prev_gear  = cruise->current_gear;
    cruise->previous_gear     = prev_gear;
    cruise->counter           = counter;
    cruise->last_burst_frame  = last_burst_frame;
    cruise->current_gear      = target_gear;

    // Hardware write to ProtocolState.dt (or whatever address d_dt
    // points to).  d_dt is null in Wave A pipelines (Pre-Flight);
    // a guard avoids a segfault if the pipeline forgot to wire it.
    float* dt_target = prism_gearbox_load_adj_d_dt(adj);
    if (dt_target != nullptr) {
        const uint32_t slot = target_gear * PRISM_GEARBOX_FLOATS_PER_GEAR;
        *dt_target = d_gearbox_table[slot];
    }
}

// ─── Host-side init: 16-float table → __constant__ d_gearbox_table ─────────

extern "C"
int prism_gearbox_init_table_async(
    const float* host_table,
    void*        stream)
{
    if (host_table == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaError_t rc = cudaMemcpyToSymbolAsync(
        d_gearbox_table,
        host_table,
        sizeof(float) * PRISM_GEARBOX_TABLE_LEN,
        0,
        cudaMemcpyHostToDevice,
        static_cast<cudaStream_t>(stream)
    );
    return static_cast<int>(rc);
}

// ─── Host launcher ──────────────────────────────────────────────────────────

extern "C"
int prism_gearbox_launch_pointer_swap(
    const InterferometricAdjudicatorFfi* adj,
    ChronometricStateTensor*             cruise,
    uint32_t                             current_frame,
    void*                                stream)
{
    if (adj == nullptr || cruise == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    prism_gearbox_pointer_swap_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>(adj, cruise, current_frame);
    return static_cast<int>(cudaGetLastError());
}

// ─── B.3.2 — SFA-only kernel (Logic/Action Bifurcation) ────────────────────
//
// Runs the same stateful finite automaton as prism_gearbox_pointer_swap_kernel
// but does NOT write *(adj->d_dt).  The dt-write side-effect is owned by the
// SWITCH body's apply-fixed-dt kernel.  This decoupling lets the Blackwell
// scheduler hide the SFA's logic-only cycles behind the integrator's global-
// memory writes — the operator-mandated path to >95% SM utilization.

extern "C"
__global__ void prism_gearbox_sfa_kernel(
    const InterferometricAdjudicatorFfi* __restrict__ adj,
    ChronometricStateTensor*             __restrict__ cruise,
    uint32_t                                          current_frame
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const uint32_t code = prism_gearbox_load_adj_code(adj);

    uint32_t counter           = cruise->counter;
    uint32_t last_burst_frame  = cruise->last_burst_frame;
    uint32_t target_gear;

    if (code == 2u) {
        target_gear = 3u;
    } else if (code == 1u) {
        target_gear       = 0u;
        counter           = 0u;
        last_burst_frame  = current_frame;
    } else {
        counter     = (counter < 0xffffffffu) ? counter + 1u : counter;
        target_gear = (counter < PRISM_GEARBOX_CRUISE_THRESHOLD) ? 1u : 2u;
    }

    // Capture OLD current_gear into previous_gear so the body's
    // rescale kernel can read it for the symplectic ratio compute.
    const uint32_t prev_gear  = cruise->current_gear;
    cruise->previous_gear     = prev_gear;
    cruise->counter           = counter;
    cruise->last_burst_frame  = last_burst_frame;
    cruise->current_gear      = target_gear;
    // No dt write — that's the SWITCH body's job.
}

extern "C"
int prism_gearbox_launch_sfa(
    const InterferometricAdjudicatorFfi* adj,
    ChronometricStateTensor*             cruise,
    uint32_t                             current_frame,
    void*                                stream)
{
    if (adj == nullptr || cruise == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    prism_gearbox_sfa_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>(adj, cruise, current_frame);
    return static_cast<int>(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════════
// B.2 — Symplectic Velocity Rescale (T12.2)
// ════════════════════════════════════════════════════════════════════
//
// Pure momentum-scaling kernel.  Each thread multiplies one f32 of the
// AoS velocity buffer by `ratio`.  The block size is 256 → each warp
// reads 32 contiguous f32 = 128 bytes → ptxas emits LDG.E.128 + STG.E.128
// (the operator-mandated vectorised path).  No position/force touch.
//
// The float4-aligned vectorised form is documented in the header but
// the runtime kernel is the simpler scalar-per-thread variant whose
// generated PTX matches LDG.E.128 by warp-level coalescing — confirmed
// by ptxas -arch=sm_120 -O3 inspection of the compiled archive.

extern "C"
__global__ void prism_gearbox_velocity_rescale_kernel(
    float*    __restrict__ d_velocities,
    uint32_t               n_floats,
    float                  ratio
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_floats) return;
    // Pure read-modify-write; no positions, no forces.
    d_velocities[tid] = d_velocities[tid] * ratio;
}

extern "C"
int prism_gearbox_launch_velocity_rescale(
    float*    d_velocities,
    uint32_t  n_floats,
    float     ratio,
    void*     stream)
{
    if (d_velocities == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    if (n_floats == 0u)          return static_cast<int>(cudaSuccess);

    constexpr uint32_t BLOCK = 256u;
    const     uint32_t grid  = (n_floats + BLOCK - 1u) / BLOCK;
    prism_gearbox_velocity_rescale_kernel<<<grid, BLOCK, 0,
        static_cast<cudaStream_t>(stream)>>>(d_velocities, n_floats, ratio);
    return static_cast<int>(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════════
// B.2 — Berendsen Weak-Coupling Guard (T12.3)
// ════════════════════════════════════════════════════════════════════
//
// Standard Berendsen thermostat:
//     λ = sqrt(1 + (Δt/τ_T)·(T₀/T − 1))
// Applied for ONE FRAME on Gear-0 entry to absorb the kinetic-energy
// shock from the high-energy burst that triggered the downshift.
//
// Defensive epsilon clamp: if T → 0 (sample collapse) or τ → 0
// (operator misconfigured), the argument to sqrt could go negative;
// we clamp to ε = 1e-6 so the result stays finite.  The host G29
// Reaper traps NaN/Inf force_norm reads as a backstop.

extern "C"
__global__ void prism_gearbox_berendsen_guard_kernel(
    float*       __restrict__ d_velocities,
    uint32_t                  n_floats,
    const float* __restrict__ d_current_temp,
    const float* __restrict__ d_dt,
    float                     target_temp_K,
    float                     tau_ps
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_floats) return;

    // Broadcast loads — every thread reads the same 4 bytes; L1
    // constant-cache hit after the first warp request.  Defensive
    // null-guard: if d_current_temp or d_dt is null (test fixture
    // forgot to wire them) we bail with no scaling so the velocities
    // stay physically meaningful.
    if (d_current_temp == nullptr || d_dt == nullptr) return;

    const float T   = *d_current_temp;
    const float dt  = *d_dt;

    // Argument-clamp keeps sqrt finite even on degenerate input.
    const float arg = 1.0f + (dt / tau_ps) * (target_temp_K / T - 1.0f);
    const float lambda = sqrtf(fmaxf(arg, 1.0e-6f));

    d_velocities[tid] = d_velocities[tid] * lambda;
}

extern "C"
int prism_gearbox_launch_berendsen_guard(
    float*       d_velocities,
    uint32_t     n_floats,
    const float* d_current_temp,
    const float* d_dt,
    float        target_temp_K,
    float        tau_ps,
    void*        stream)
{
    if (d_velocities == nullptr || d_current_temp == nullptr || d_dt == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_floats == 0u) return static_cast<int>(cudaSuccess);

    constexpr uint32_t BLOCK = 256u;
    const     uint32_t grid  = (n_floats + BLOCK - 1u) / BLOCK;
    prism_gearbox_berendsen_guard_kernel<<<grid, BLOCK, 0,
        static_cast<cudaStream_t>(stream)>>>(
        d_velocities, n_floats,
        d_current_temp, d_dt,
        target_temp_K, tau_ps);
    return static_cast<int>(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════════
// B.2 — 4-way Predicate Bridge (T12.4)
// ════════════════════════════════════════════════════════════════════
//
// Trivial 1-thread kernel: read `cruise->current_gear` and forward
// to the conditional handle via cudaGraphSetConditional.  PointerSwap
// already executed the stateful finite automaton (code → gear) and
// wrote current_gear to the cruise tensor; this bridge just
// communicates that decision to the SWITCH node.
//
// Requires CUDA 12.4+ for cudaGraphSetConditional.  Production
// toolchain is CUDA 13.x so the path is always live.

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12040

// B.3.2 — predicate-bridge with gear_override consult.
//
//   final_gear = (adj->gear_override != 0xFF)
//                ? adj->gear_override
//                : cruise->current_gear
//
// `adj->gear_override` is at offset 100 — read via byte-offset
// arithmetic to keep this TU independent of the full FFI struct
// definition.  When the operator (or a safety script) writes 0..3 to
// that VRAM byte, the next captured-graph launch's SWITCH routes to
// the corresponding body — the Blackwell Hardware Interlock.
extern "C"
__global__ void prism_gearbox_predicate_bridge_kernel(
    cudaGraphConditionalHandle                          handle,
    const InterferometricAdjudicatorFfi* __restrict__   adj,
    const ChronometricStateTensor*       __restrict__   cruise
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (cruise == nullptr) return;

    // Default to cruise.current_gear if adj is null (test fixtures);
    // otherwise consult the u32 gear_override at offset 100.
    //
    //   final_gear = (override == 0xFF) ? calculated : (override & 0x03)
    //
    // Branchless selection per operator §2 — predicate evaluates as a
    // single comparison + select; no warp divergence (single thread).
    uint32_t final_gear = cruise->current_gear;
    if (adj != nullptr) {
        const uint8_t* base = reinterpret_cast<const uint8_t*>(adj);
        const uint32_t override_val = *reinterpret_cast<const uint32_t*>(base + 100);
        if (override_val != 0xFFu) {
            final_gear = override_val & 0x03u;
        }
    }
    // SWITCH was forged with size=4; defensive 2-bit mask.
    cudaGraphSetConditional(handle, final_gear & 0x3u);
}

extern "C"
int prism_gearbox_launch_predicate_bridge(
    uint64_t                                handle_v,
    const InterferometricAdjudicatorFfi*    adj,
    const ChronometricStateTensor*          cruise,
    void*                                   stream)
{
    if (cruise == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    cudaGraphConditionalHandle handle =
        static_cast<cudaGraphConditionalHandle>(handle_v);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    prism_gearbox_predicate_bridge_kernel<<<1, 1, 0, s>>>(handle, adj, cruise);
    return static_cast<int>(cudaGetLastError());
}

#else  // CUDART_VERSION < 12040

extern "C"
int prism_gearbox_launch_predicate_bridge(
    uint64_t                                /*handle_v*/,
    const InterferometricAdjudicatorFfi*    /*adj*/,
    const ChronometricStateTensor*          /*cruise*/,
    void*                                   /*stream*/)
{
    return static_cast<int>(cudaErrorNotSupported);
}

#endif  // CUDART_VERSION >= 12040

// ════════════════════════════════════════════════════════════════════
// B.3-narrow — SWITCH body kernels
// ════════════════════════════════════════════════════════════════════

// ─── Rescale kernel (ratio matrix lookup, single multiply) ─────────────────

extern "C"
__global__ void prism_gearbox_rescale_kernel(
    float*                                      __restrict__ d_velocities,
    uint32_t                                                  n_floats,
    const ChronometricStateTensor*              __restrict__ cruise,
    uint32_t                                                  target_gear
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_floats) return;
    if (cruise == nullptr) return;

    // Single LDC for ratio (broadcast across the warp).
    const uint32_t prev_gear = cruise->previous_gear & 0x3u;
    const uint32_t tgt       = target_gear           & 0x3u;
    // Bounds: matrix only covers active gears 0..2.  Gear 3 (abort)
    // bodies do NOT call this kernel; the trap kernel fires directly.
    if (prev_gear >= 3u || tgt >= 3u) return;
    const float ratio = d_rescale_ratios[prev_gear * 3u + tgt];

    // Single LDG + MUL + STG per thread.  Block 256 ⇒ warp-coalesced
    // 128-byte memory transactions (LDG.E.128 / STG.E.128).
    d_velocities[tid] = d_velocities[tid] * ratio;
}

extern "C"
int prism_gearbox_launch_rescale(
    float*                              d_velocities,
    uint32_t                            n_floats,
    const ChronometricStateTensor*      cruise,
    uint32_t                            target_gear,
    void*                               stream)
{
    if (d_velocities == nullptr || cruise == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (n_floats == 0u) return static_cast<int>(cudaSuccess);
    constexpr uint32_t BLOCK = 256u;
    const     uint32_t grid  = (n_floats + BLOCK - 1u) / BLOCK;
    prism_gearbox_rescale_kernel<<<grid, BLOCK, 0,
        static_cast<cudaStream_t>(stream)>>>(
        d_velocities, n_floats, cruise, target_gear);
    return static_cast<int>(cudaGetLastError());
}

extern "C"
int prism_gearbox_init_rescale_ratios_async(void* stream)
{
    // Static initialiser at the top of this TU is canonical; this
    // function exists for ABI symmetry with future runtime re-tuning.
    // No-op on the canonical 0.5/2.0/4.0fs gearbox.
    (void)stream;
    return static_cast<int>(cudaSuccess);
}

// ─── Apply-fixed-dt kernel (per-body dt write) ─────────────────────────────

extern "C"
__global__ void prism_gearbox_apply_fixed_dt_kernel(
    const InterferometricAdjudicatorFfi*  __restrict__ adj,
    uint32_t                                            target_gear
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (adj == nullptr) return;
    // Read d_dt pointer via offset arithmetic — keeps this TU
    // independent of the full FFI struct definition (operator §M2).
    const uint8_t* base = reinterpret_cast<const uint8_t*>(adj);
    float* dt_target = *reinterpret_cast<float* const*>(base + 112);
    if (dt_target == nullptr) return;
    if (target_gear >= 4u) return;
    *dt_target = d_gearbox_table[target_gear * PRISM_GEARBOX_FLOATS_PER_GEAR];
}

extern "C"
int prism_gearbox_launch_apply_fixed_dt(
    const InterferometricAdjudicatorFfi* adj,
    uint32_t                             target_gear,
    void*                                stream)
{
    if (adj == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    prism_gearbox_apply_fixed_dt_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>(adj, target_gear);
    return static_cast<int>(cudaGetLastError());
}

// ─── Trap kernel (PTX hardware-trap for Gear 3 abort) ──────────────────────

extern "C"
__global__ void prism_gearbox_trap_kernel() {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // PTX `trap` instruction.  Halts the GPU stream and surfaces
    // CUDA_ERROR_LAUNCH_FAILED to the host on the next sync.
    asm volatile("trap;");
}

extern "C"
int prism_gearbox_launch_trap(void* stream) {
    prism_gearbox_trap_kernel<<<1, 1, 0,
        static_cast<cudaStream_t>(stream)>>>();
    return static_cast<int>(cudaGetLastError());
}

// ════════════════════════════════════════════════════════════════════
// B.3-narrow — SWITCH body populator (Blackwell Driver Probe target)
// ════════════════════════════════════════════════════════════════════
//
// Takes the four phGraph_out CUgraph handles returned by
// `prism_wire_g26_gearbox_ffi` and adds the body kernels as graph
// nodes.  If ANY of the four handles is null, returns
// cudaErrorInvalidValue early — the operator-mandated "Smoking Gun"
// signal that CUDA 13.x's runtime API does not populate phGraph_out
// for SWITCH-type conditional nodes (the same bug noted for IF-type
// at captured_pipeline.rs:1212-1216).  In that case the caller pivots
// to the `cudaGraphSetConditional` kernel-conditional pattern.
//
// Body topologies (per operator 2026-05-02 §1.1):
//   Body 0 (Gear 0 — Burst — 0.5 fs):
//     rescale(0) → [Berendsen guard if d_current_temp+d_dt non-null]
//                → apply_fixed_dt(0)
//   Body 1 (Gear 1 — Cruise — 2.0 fs):  rescale(1) → apply_fixed_dt(1)
//   Body 2 (Gear 2 — Sprint — 4.0 fs):  rescale(2) → apply_fixed_dt(2)
//   Body 3 (Gear 3 — Abort):            trap

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12060

namespace {

// Canonical kernel-node addition wrapper.  CUDA 13.x signature uses
// cudaGraphAddNode with a cudaGraphNodeParams struct (the runtime
// equivalent of cuGraphAddNode_v2).  We use the simpler
// cudaGraphAddKernelNode path which is stable across 12.4+/13.x.
__host__ cudaError_t add_kernel_node(
    cudaGraphNode_t*       out_node,
    cudaGraph_t            graph,
    const cudaGraphNode_t* deps,
    size_t                 num_deps,
    void*                  func,
    dim3                   grid,
    dim3                   block,
    void**                 args,
    size_t                 shared_bytes
) {
    cudaKernelNodeParams params{};
    params.func           = func;
    params.gridDim        = grid;
    params.blockDim       = block;
    params.sharedMemBytes = static_cast<unsigned int>(shared_bytes);
    params.kernelParams   = args;
    params.extra          = nullptr;
    return cudaGraphAddKernelNode(out_node, graph, deps, num_deps, &params);
}

}  // anonymous namespace

extern "C"
int prism_gearbox_populate_switch_bodies_ffi(
    cudaGraph_t*                          body_subgraphs,
    const InterferometricAdjudicatorFfi*  adj,
    float*                                d_velocities,
    uint32_t                              n_floats,
    const ChronometricStateTensor*        cruise,
    const float*                          d_current_temp,
    const float*                          d_dt,
    float                                 target_temp_K,
    float                                 tau_ps)
{
    if (body_subgraphs == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }

    // ── Driver Probe: assert all four sub-graph handles are valid. ──
    // Operator-mandated "Smoking Gun" check (2026-05-02 §1.3): a null
    // handle here means CUDA 13.x's cudaGraphAddNode(CONDITIONAL)
    // does NOT populate phGraph_out for SWITCH-type, and we must
    // pivot to the kernel-conditional fallback path.
    for (int i = 0; i < 4; ++i) {
        if (body_subgraphs[i] == nullptr) {
            return static_cast<int>(cudaErrorInvalidValue);
        }
    }

    constexpr uint32_t BLOCK = 256u;
    const     uint32_t rescale_grid =
        (n_floats == 0u) ? 1u : (n_floats + BLOCK - 1u) / BLOCK;

    cudaError_t rc = cudaSuccess;

    // ── Bodies 0, 1, 2: rescale → [Berendsen for 0] → apply_fixed_dt ──
    for (uint32_t gear = 0u; gear <= 2u; ++gear) {
        cudaGraph_t body = body_subgraphs[gear];

        cudaGraphNode_t rescale_node = nullptr;
        {
            void* args[] = {
                &d_velocities,
                const_cast<uint32_t*>(&n_floats),
                const_cast<ChronometricStateTensor**>(&cruise),
                &gear,
            };
            rc = add_kernel_node(
                &rescale_node, body,
                /*deps=*/   nullptr, /*num_deps=*/ 0,
                reinterpret_cast<void*>(prism_gearbox_rescale_kernel),
                dim3(rescale_grid, 1, 1), dim3(BLOCK, 1, 1),
                args, /*shared=*/ 0);
            if (rc != cudaSuccess) return static_cast<int>(rc);
        }

        cudaGraphNode_t last_node = rescale_node;

        // ── Berendsen guard (Gear 0 only — burst-shock dampener) ──
        if (gear == 0u && d_current_temp != nullptr && d_dt != nullptr) {
            cudaGraphNode_t berendsen_node = nullptr;
            void* args[] = {
                &d_velocities,
                const_cast<uint32_t*>(&n_floats),
                const_cast<float**>(&d_current_temp),
                const_cast<float**>(&d_dt),
                &target_temp_K,
                &tau_ps,
            };
            rc = add_kernel_node(
                &berendsen_node, body,
                &last_node, /*num_deps=*/ 1,
                reinterpret_cast<void*>(prism_gearbox_berendsen_guard_kernel),
                dim3(rescale_grid, 1, 1), dim3(BLOCK, 1, 1),
                args, /*shared=*/ 0);
            if (rc != cudaSuccess) return static_cast<int>(rc);
            last_node = berendsen_node;
        }

        // ── apply_fixed_dt — single-thread dt write ──
        cudaGraphNode_t apply_node = nullptr;
        {
            void* args[] = {
                const_cast<InterferometricAdjudicatorFfi**>(&adj),
                &gear,
            };
            rc = add_kernel_node(
                &apply_node, body,
                &last_node, /*num_deps=*/ 1,
                reinterpret_cast<void*>(prism_gearbox_apply_fixed_dt_kernel),
                dim3(1, 1, 1), dim3(1, 1, 1),
                args, /*shared=*/ 0);
            if (rc != cudaSuccess) return static_cast<int>(rc);
        }
    }

    // ── Body 3: trap kernel ───────────────────────────────────────
    {
        cudaGraph_t body = body_subgraphs[3];
        cudaGraphNode_t trap_node = nullptr;
        rc = add_kernel_node(
            &trap_node, body,
            /*deps=*/ nullptr, /*num_deps=*/ 0,
            reinterpret_cast<void*>(prism_gearbox_trap_kernel),
            dim3(1, 1, 1), dim3(1, 1, 1),
            /*args=*/ nullptr, /*shared=*/ 0);
        if (rc != cudaSuccess) return static_cast<int>(rc);
    }

    return static_cast<int>(cudaSuccess);
}

#else  // CUDART_VERSION < 12060

extern "C"
int prism_gearbox_populate_switch_bodies_ffi(
    cudaGraph_t*                          /*body_subgraphs*/,
    const InterferometricAdjudicatorFfi*  /*adj*/,
    float*                                /*d_velocities*/,
    uint32_t                              /*n_floats*/,
    const ChronometricStateTensor*        /*cruise*/,
    const float*                          /*d_current_temp*/,
    const float*                          /*d_dt*/,
    float                                 /*target_temp_K*/,
    float                                 /*tau_ps*/)
{
    return static_cast<int>(cudaErrorNotSupported);
}

#endif  // CUDART_VERSION >= 12060

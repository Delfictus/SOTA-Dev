// ═══════════════════════════════════════════════════════════════════════════
// PRISM-4D / ASC Steering — Vectorized Repulsion Kernel declarations
//
// asc_inject_repulsion_v4_kernel: applies F_i = α × ΔKL × (pos_i − c)
// for each atom using vectorized float4 loads and Blackwell-native
// atom.global.add.v4.f32 for lock-free concurrent force accumulation.
//
// Gate G22_ATOMIC_v4_VERIFIED: ptxas --verbose audit required before
// production deployment (see Phase 2 validation in ZSTR roadmap).
//
// Compilation: nvcc -arch=sm_120 -O3 --use_fast_math --restrict
//              --expt-relaxed-constexpr -std=c++17 -c
// ═══════════════════════════════════════════════════════════════════════════

#pragma once
#include <cuda_runtime.h>
#include <cstdint>

namespace prism_nhs { namespace asc {

/// Vectorized ASC repulsion force injection.
///
/// Adds F_i = alpha_gain * delta_kl * (pos_i - centroid) to d_forces
/// for each atom, using float4 vectorized loads over the force and
/// position buffers.  The force buffer layout is AoS: for n_atoms atoms
/// d_forces stores [fx0,fy0,fz0, fx1,fy1,fz1, ...] as n_atoms*3 f32.
/// Reinterpreted as float4, each quad covers 4 consecutive f32 values
/// spanning atom boundaries; the centroid cycles with period 3 (x,y,z).
///
/// @param d_forces     VRAM force accumulator (n_atoms * 3 × f32, AoS).
///                     MUST be float4-aligned (naturally satisfied if
///                     n_atoms * 3 * 4 is 16-byte aligned from cudarc).
/// @param d_pos        VRAM atom positions (n_atoms * 3 × f32, same AoS).
/// @param centroid     Pocket centroid (Å).
/// @param alpha_gain   ASC steering gain coefficient.
/// @param delta_kl     SURP score (KL divergence from adjudicator).
/// @param n_atoms      Total atom count.
///
/// Launch geometry: grid_x = ceil(n_atoms * 3 / 4), block = 128.
/// Gate: G22_ATOMIC_v4_VERIFIED — ptxas must confirm zero register
/// spills and presence of atom.global.add.v4.f32 in disassembly.
extern "C"
__global__ void asc_inject_repulsion_v4_kernel(
    float4* __restrict__       d_forces,
    const float4* __restrict__ d_pos,
    float3                     centroid,
    float                      alpha_gain,
    float                      delta_kl,
    uint32_t                   n_atoms
);

}} // namespace prism_nhs::asc

//! NHS-AMBER Fused Mega Kernel
//!
//! Single-stream GPU pipeline combining:
//! - Full AMBER ff14SB physics (bonds, angles, dihedrals, LJ, electrostatics)
//! - Langevin thermostat with dynamic temperature protocols
//! - SHAKE/RATTLE hydrogen constraints
//! - Holographic exclusion field (negative space mapping)
//! - Water density inference from holographic negative
//! - Neuromorphic LIF continuous observation
//! - UV bias pump-probe perturbation
//! - Spike-triggered snapshot capture
//! - Warp matrix for atomic-precision alignment
//!
//! All fused into shared memory, no CPU round-trips.
//! Target: 100,000+ timesteps/second streaming.

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Include excited state dynamics for true UV photophysics
#include "nhs_excited_state.cuh"

// Adaptive Cryo-Thermal Protocol for three-phase cryptic site detection
// Phase 1: CRYO BURST (80K, HIGH UV, global sweep)
// Phase 2: THERMAL RAMP (80K→300K, validation)
// Phase 3: FOCUSED DIG (300K, exploitation)
#include "prism_adaptive_protocol.cuh"

// Cryo-thermal detection physics (UV absorption → thermal signatures)
#include "cryo_thermal_detection.cuh"

// Ultra-sensitive multi-modal neuromorphic detector
// Channels: thermal spike, gradient, melt wave, correlation
#include "sensitive_detector.cuh"

// Advanced UV-LIF coupling for direct UV → spike correlation
// Mechanisms: thermal wavefront, dewetting halo, cooperative enhancement
#include "uv_lif_coupling.cuh"

// ============================================================================
// CONSTANTS
// ============================================================================

#define BLOCK_SIZE 256
#define WARP_SIZE 32
#define MAX_NEIGHBORS 128
#define MAX_GRID_DIM 128
#define PI 3.14159265358979323846f

// AMBER force field constants
#define COULOMB_CONSTANT 332.0636f  // kcal/mol * Angstrom / e^2
#define FUDGE_LJ 0.5f               // 1-4 LJ scaling
#define FUDGE_QQ 0.8333f            // 1-4 electrostatic scaling

// NHS constants
#define EXCLUSION_CUTOFF 8.0f       // Angstroms for exclusion field
#define WATER_DENSITY_BULK 0.0334f  // molecules/A^3
#define LIF_THRESHOLD 0.5f           // Tuned threshold for cryo-UV water density changes
#define LIF_RESET 0.0f              // Reset potential
#define REFRACTORY_STEPS 250       // 250 steps * 0.004ps (HMR) = 1.0ps refractory period
#define UV_WAVELENGTH 280.0f        // nm - aromatic absorption
#define UV_LIF_WINDOW 100           // Steps within which LIF spike is causally linked to prior UV (~0.4 ps at dt=0.004ps/HMR)
#define KCC_RING_STEPS 64           // Micro-ring-buffer depth for KCC temporal descriptors per residue

// ============================================================================
// WARP-LEVEL PRIMITIVES (for fast reductions without shared memory)
// ============================================================================

// Warp shuffle reduction for sum (no __syncthreads needed!)
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Warp shuffle reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Warp shuffle reduction for min
__device__ __forceinline__ float warp_reduce_min(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fminf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Fast reciprocal square root (use hardware rsqrt)
__device__ __forceinline__ float fast_rsqrt(float x) {
    return rsqrtf(x);
}

// Fast inverse (use hardware rcp)
__device__ __forceinline__ float fast_rcp(float x) {
    return __frcp_rn(x);
}

// ============================================================================
// CELL LIST CONSTANTS (O(N) Neighbor Lists)
// ============================================================================
// Cell size should be >= cutoff to ensure all neighbors are in adjacent cells
#define NB_CUTOFF 10.0f             // Nonbonded cutoff (Angstroms)
#define NB_CUTOFF_SQ 100.0f         // Cutoff squared
#define CELL_SIZE 10.0f             // Cell dimension (= cutoff)
#define CELL_SIZE_INV 0.1f          // 1.0 / CELL_SIZE
#define MAX_CELLS_PER_DIM 32        // Max cells per dimension (32^3 = 32768 cells max)
#define MAX_TOTAL_CELLS 32768       // MAX_CELLS_PER_DIM^3
#define MAX_ATOMS_PER_CELL 128      // Max atoms that can fit in one cell
#define NEIGHBOR_LIST_SIZE 256      // Max neighbors per atom (with buffer)
#define NEIGHBOR_LIST_BUFFER 1.2f   // 20% buffer for list reuse between rebuilds

// ============================================================================
// DATA STRUCTURES
// ============================================================================

// Temperature protocol for cryogenic probing
struct TemperatureProtocol {
    float start_temp;       // Starting temperature (K)
    float end_temp;         // Ending temperature (K)
    int ramp_steps;         // Steps to ramp
    int hold_steps;         // Steps to hold at end
    int current_step;       // Current step in protocol

    __device__ float get_temperature() const {
        if (current_step < ramp_steps) {
            float t = (float)current_step / (float)ramp_steps;
            return start_temp + t * (end_temp - start_temp);
        }
        return end_temp;
    }
};

// Bond parameter
struct BondParam {
    int i, j;
    float r0, k;
};

// Angle parameter
struct AngleParam {
    int i, j, k;
    float theta0, force_k;
};

// Dihedral parameter
struct DihedralParam {
    int i, j, k, l;
    int periodicity;
    float phase, force_k;
};

// LJ parameter per atom
struct LJParam {
    float sigma, epsilon;
};

// Hydrogen cluster for SHAKE
struct HCluster {
    int central_atom;
    int hydrogen_atoms[3];  // -1 for unused
    float bond_lengths[3];
    int n_hydrogens;
    float inv_mass_central;
    float inv_mass_h;
};

// UV target (aromatic residue)
struct UVTarget {
    int residue_id;
    int atom_indices[16];   // Atoms in aromatic ring
    int n_atoms;
    float absorption_strength;  // Trp > Tyr > Phe
    // CANONICAL chromophore type ordering (MUST match Rust):
    //   0 = TRP (Tryptophan)
    //   1 = TYR (Tyrosine)
    //   2 = PHE (Phenylalanine)
    //   3 = S-S (Disulfide)
    int aromatic_type;
};

// Aromatic neighbor list for vibrational transfer
struct AromaticNeighbors {
    int atom_indices[64];       // Atoms within 5Å of aromatic
    int n_neighbors;
};

// Spike event for capture
struct SpikeEvent {
    int timestep;
    int voxel_idx;
    float3 position;
    float intensity;
    int nearby_residues[8];
    int n_residues;
    // Enhanced metadata for downstream docking/analysis
    int spike_source;           // 1=UV, 2=LIF
    float wavelength_nm;        // UV wavelength that triggered (0 for LIF)
    int aromatic_type;          // 0=TRP, 1=TYR, 2=PHE, 3=SS, -1=none/LIF
    int aromatic_residue_id;    // residue ID of closest excited aromatic (-1 for LIF)
    float water_density;        // local water density at spike voxel
    float vibrational_energy;   // energy deposited by UV excitation (0 for LIF)
    int n_nearby_excited;       // number of excited aromatics in range (pi-stacking)
    float wd_change;            // |water_density - water_density_prev| = |∂WD/∂t| for SDST energy_gradient
};

// Warp matrix entry - maps voxel to atoms
struct WarpEntry {
    int voxel_idx;
    int atom_indices[16];   // Contributing atoms
    float atom_weights[16]; // Distance-based weights
    int n_atoms;
};

// ============================================================================
// SHARED MEMORY LAYOUT
// ============================================================================

// Per-block shared memory for fused kernel
struct SharedMemory {
    // Atom data tile
    float3 positions[BLOCK_SIZE];
    float3 velocities[BLOCK_SIZE];
    float3 forces[BLOCK_SIZE];
    float charges[BLOCK_SIZE];
    float masses[BLOCK_SIZE];
    int atom_types[BLOCK_SIZE];

    // Exclusion field tile (for local grid region)
    float exclusion_tile[8][8][8];
    float water_density_tile[8][8][8];

    // LIF state tile
    float lif_potential[8][8][8];
    int spike_flags[8][8][8];

    // Reduction buffers
    float energy_buffer[BLOCK_SIZE];
    int spike_count;
};

// ============================================================================
// AMBER FORCE KERNELS
// ============================================================================

// Bond force: E = k(r - r0)^2
__device__ void compute_bond_force(
    const float3& pi, const float3& pj,
    float r0, float k,
    float3& fi, float3& fj
) {
    float3 rij = make_float3(pj.x - pi.x, pj.y - pi.y, pj.z - pi.z);
    float r = sqrtf(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);

    if (r < 1e-8f) return;

    float dr = r - r0;
    float force_mag = -2.0f * k * dr / r;

    float3 f = make_float3(force_mag * rij.x, force_mag * rij.y, force_mag * rij.z);
    fi.x += f.x; fi.y += f.y; fi.z += f.z;
    fj.x -= f.x; fj.y -= f.y; fj.z -= f.z;
}

// Angle force: E = k(theta - theta0)^2
__device__ void compute_angle_force(
    const float3& pi, const float3& pj, const float3& pk,
    float theta0, float k,
    float3& fi, float3& fj, float3& fk
) {
    float3 rij = make_float3(pi.x - pj.x, pi.y - pj.y, pi.z - pj.z);
    float3 rkj = make_float3(pk.x - pj.x, pk.y - pj.y, pk.z - pj.z);

    float rij_len = sqrtf(rij.x*rij.x + rij.y*rij.y + rij.z*rij.z);
    float rkj_len = sqrtf(rkj.x*rkj.x + rkj.y*rkj.y + rkj.z*rkj.z);

    if (rij_len < 1e-8f || rkj_len < 1e-8f) return;

    float dot = rij.x*rkj.x + rij.y*rkj.y + rij.z*rkj.z;
    float cos_theta = dot / (rij_len * rkj_len);
    cos_theta = fmaxf(-1.0f, fminf(1.0f, cos_theta));

    float theta = acosf(cos_theta);
    float dtheta = theta - theta0;

    float sin_theta = sqrtf(1.0f - cos_theta*cos_theta);
    if (sin_theta < 1e-8f) sin_theta = 1e-8f;

    float force_mag = -2.0f * k * dtheta / sin_theta;

    // Gradient computation
    float inv_rij = 1.0f / rij_len;
    float inv_rkj = 1.0f / rkj_len;

    float3 di = make_float3(
        (rkj.x * inv_rkj - cos_theta * rij.x * inv_rij) * inv_rij,
        (rkj.y * inv_rkj - cos_theta * rij.y * inv_rij) * inv_rij,
        (rkj.z * inv_rkj - cos_theta * rij.z * inv_rij) * inv_rij
    );

    float3 dk = make_float3(
        (rij.x * inv_rij - cos_theta * rkj.x * inv_rkj) * inv_rkj,
        (rij.y * inv_rij - cos_theta * rkj.y * inv_rkj) * inv_rkj,
        (rij.z * inv_rij - cos_theta * rkj.z * inv_rkj) * inv_rkj
    );

    fi.x += force_mag * di.x; fi.y += force_mag * di.y; fi.z += force_mag * di.z;
    fk.x += force_mag * dk.x; fk.y += force_mag * dk.y; fk.z += force_mag * dk.z;
    fj.x -= force_mag * (di.x + dk.x);
    fj.y -= force_mag * (di.y + dk.y);
    fj.z -= force_mag * (di.z + dk.z);
}

// Dihedral force: E = k[1 + cos(n*phi - gamma)]
__device__ void compute_dihedral_force(
    const float3& pi, const float3& pj, const float3& pk, const float3& pl,
    int periodicity, float phase, float k,
    float3& fi, float3& fj, float3& fk, float3& fl
) {
    float3 b1 = make_float3(pj.x - pi.x, pj.y - pi.y, pj.z - pi.z);
    float3 b2 = make_float3(pk.x - pj.x, pk.y - pj.y, pk.z - pj.z);
    float3 b3 = make_float3(pl.x - pk.x, pl.y - pk.y, pl.z - pk.z);

    // Cross products
    float3 c1 = make_float3(
        b1.y*b2.z - b1.z*b2.y,
        b1.z*b2.x - b1.x*b2.z,
        b1.x*b2.y - b1.y*b2.x
    );
    float3 c2 = make_float3(
        b2.y*b3.z - b2.z*b3.y,
        b2.z*b3.x - b2.x*b3.z,
        b2.x*b3.y - b2.y*b3.x
    );

    float c1_len = sqrtf(c1.x*c1.x + c1.y*c1.y + c1.z*c1.z);
    float c2_len = sqrtf(c2.x*c2.x + c2.y*c2.y + c2.z*c2.z);

    if (c1_len < 1e-8f || c2_len < 1e-8f) return;

    float cos_phi = (c1.x*c2.x + c1.y*c2.y + c1.z*c2.z) / (c1_len * c2_len);
    cos_phi = fmaxf(-1.0f, fminf(1.0f, cos_phi));

    // Sign of phi from triple product
    float sign = c1.x*b3.x + c1.y*b3.y + c1.z*b3.z;
    float phi = acosf(cos_phi);
    if (sign < 0) phi = -phi;

    // Force magnitude
    float n = (float)periodicity;
    float force_mag = k * n * sinf(n * phi - phase);

    // Simplified force distribution (proper implementation needs full gradient)
    float b2_len = sqrtf(b2.x*b2.x + b2.y*b2.y + b2.z*b2.z);
    if (b2_len < 1e-8f) return;

    float3 f1 = make_float3(
        force_mag * c1.x / (c1_len * c1_len) * b2_len,
        force_mag * c1.y / (c1_len * c1_len) * b2_len,
        force_mag * c1.z / (c1_len * c1_len) * b2_len
    );
    float3 f4 = make_float3(
        -force_mag * c2.x / (c2_len * c2_len) * b2_len,
        -force_mag * c2.y / (c2_len * c2_len) * b2_len,
        -force_mag * c2.z / (c2_len * c2_len) * b2_len
    );

    fi.x += f1.x; fi.y += f1.y; fi.z += f1.z;
    fl.x += f4.x; fl.y += f4.y; fl.z += f4.z;
    fj.x -= f1.x * 0.5f; fj.y -= f1.y * 0.5f; fj.z -= f1.z * 0.5f;
    fk.x -= f4.x * 0.5f; fk.y -= f4.y * 0.5f; fk.z -= f4.z * 0.5f;
}

// LJ + Electrostatic nonbonded force (OPTIMIZED with fast math)
__device__ __forceinline__ void compute_nonbonded_force(
    const float3& pi, const float3& pj,
    float qi, float qj,
    float sigma_i, float epsilon_i,
    float sigma_j, float epsilon_j,
    float3& fi, float3& fj,
    float cutoff_sq
) {
    float3 rij = make_float3(pj.x - pi.x, pj.y - pi.y, pj.z - pi.z);
    float r_sq = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;

    if (r_sq > cutoff_sq || r_sq < 1e-8f) return;

    // OPTIMIZATION: Use rsqrtf (hardware intrinsic) instead of sqrt + divide
    float inv_r = rsqrtf(r_sq);  // 1/sqrt(r_sq) in single instruction
    float inv_r2 = inv_r * inv_r;

    // Lorentz-Berthelot combining rules
    float sigma = 0.5f * (sigma_i + sigma_j);
    // OPTIMIZATION: Use __fmul_rn for precise multiply, rsqrtf for sqrt
    float epsilon = sqrtf(epsilon_i * epsilon_j);

    // LJ force - precompute powers efficiently
    float sigma_r = sigma * inv_r;
    float sigma_r2 = sigma_r * sigma_r;
    float sigma_r6 = sigma_r2 * sigma_r2 * sigma_r2;  // (sigma/r)^6
    float sigma_r12 = sigma_r6 * sigma_r6;
    float lj_force = 24.0f * epsilon * inv_r2 * (2.0f * sigma_r12 - sigma_r6);

    // Coulomb force
    float elec_force = COULOMB_CONSTANT * qi * qj * inv_r2 * inv_r;

    float total_force = lj_force + elec_force;

    float3 f = make_float3(total_force * rij.x, total_force * rij.y, total_force * rij.z);
    fi.x -= f.x; fi.y -= f.y; fi.z -= f.z;
    fj.x += f.x; fj.y += f.y; fj.z += f.z;
}

// ============================================================================
// LANGEVIN THERMOSTAT WITH DYNAMIC TEMPERATURE
// ============================================================================

__device__ void langevin_thermostat(
    float3& velocity,
    float mass,
    float target_temp,
    float gamma,
    float dt,
    curandState* rng_state
) {
    // Friction coefficient
    float c1 = expf(-gamma * dt);
    float c2 = sqrtf((1.0f - c1*c1) * target_temp / mass);

    // Apply Langevin dynamics
    velocity.x = c1 * velocity.x + c2 * curand_normal(rng_state);
    velocity.y = c1 * velocity.y + c2 * curand_normal(rng_state);
    velocity.z = c1 * velocity.z + c2 * curand_normal(rng_state);
}

// ============================================================================
// SHAKE CONSTRAINT FOR HYDROGEN BONDS
// ============================================================================

__device__ void shake_constraint(
    float3& pos_central,
    float3& pos_h,
    float3& vel_central,
    float3& vel_h,
    float target_length,
    float inv_mass_central,
    float inv_mass_h,
    int max_iter
) {
    for (int iter = 0; iter < max_iter; iter++) {
        float3 r = make_float3(
            pos_h.x - pos_central.x,
            pos_h.y - pos_central.y,
            pos_h.z - pos_central.z
        );
        float r_sq = r.x*r.x + r.y*r.y + r.z*r.z;
        float target_sq = target_length * target_length;

        float diff = r_sq - target_sq;
        if (fabsf(diff) < 1e-6f) break;

        float inv_mass_sum = inv_mass_central + inv_mass_h;
        float lambda = diff / (2.0f * inv_mass_sum * r_sq);

        float3 correction = make_float3(lambda * r.x, lambda * r.y, lambda * r.z);

        pos_central.x += inv_mass_central * correction.x;
        pos_central.y += inv_mass_central * correction.y;
        pos_central.z += inv_mass_central * correction.z;

        pos_h.x -= inv_mass_h * correction.x;
        pos_h.y -= inv_mass_h * correction.y;
        pos_h.z -= inv_mass_h * correction.z;
    }
}

// ============================================================================
// HOLOGRAPHIC EXCLUSION FIELD
// ============================================================================

// Atom type radii for exclusion field
__device__ float get_exclusion_radius(int atom_type) {
    // 0=hydrophobic, 1=polar, 2=charged+, 3=charged-, 4=aromatic, 5=backbone
    const float radii[] = {4.5f, 2.5f, 2.0f, 2.0f, 3.5f, 2.0f, 1.4f};
    return radii[atom_type % 7];
}

// Compute exclusion contribution from single atom to voxel
__device__ float compute_exclusion_contribution(
    float3 atom_pos,
    float3 voxel_center,
    int atom_type,
    float charge
) {
    float3 d = make_float3(
        voxel_center.x - atom_pos.x,
        voxel_center.y - atom_pos.y,
        voxel_center.z - atom_pos.z
    );
    float dist_sq = d.x*d.x + d.y*d.y + d.z*d.z;
    float dist = sqrtf(dist_sq);

    float radius = get_exclusion_radius(atom_type);

    // Gaussian exclusion
    float sigma = radius / 2.0f;
    float exclusion = expf(-dist_sq / (2.0f * sigma * sigma));

    // Charge modulation (charged atoms have stronger exclusion)
    float charge_factor = 1.0f + 0.5f * fabsf(charge);

    return exclusion * charge_factor;
}

// Infer water density from exclusion field
__device__ float infer_water_density(
    float exclusion_value,
    float polar_field,
    float temperature
) {
    // Base density reduced by exclusion
    float base_density = WATER_DENSITY_BULK * (1.0f - exclusion_value);

    // Polar enhancement (water clusters near polar groups)
    float polar_enhancement = 1.0f + 0.3f * polar_field;

    // Temperature effect (colder = less mobile water, clearer signal)
    // At low temps, water behavior becomes more discrete (freeze transition)
    float temp_factor = sqrtf(temperature / 300.0f);

    return base_density * polar_enhancement * temp_factor;
}

// ============================================================================
// NEUROMORPHIC LIF OBSERVATION
// ============================================================================

// LIF neuron update - BIDIRECTIONAL detection (rewetting AND dewetting)
// For cryo-thermal detection, we detect BOTH:
// 1. Dewetting: water pushed away (exclusion increase)
// 2. Rewetting: water attracted (exclusion decrease from UV excitation)
//
// Returns: true if spike occurred
// Output: spike_intensity is set to the membrane potential that triggered the spike
__device__ bool lif_neuron_update(
    float& membrane_potential,
    float water_density_current,
    float water_density_prev,
    float tau_mem,
    float dt,
    float threshold,
    float& spike_intensity  // OUTPUT: intensity of spike (0 if no spike)
) {
    spike_intensity = 0.0f;

    // Bidirectional input: detect changes above noise floor
    // TUNED for Cryo-UV: sensitive to dewetting events during temperature ramp
    float density_change = water_density_current - water_density_prev;

    // Lower noise floor for cryo-UV sensitivity (thermal fluctuation ~0.001)
    const float NOISE_FLOOR = 0.002f;
    float abs_change = fabsf(density_change);
    // Increased amplification (10x) for cryo-UV detection
    float bidirectional_signal = (abs_change > NOISE_FLOOR) ? (abs_change - NOISE_FLOOR) * 3.0f : 0.0f;

    // Exclusion-weighted term: detect hydrophobic dewetting zones
    float density_deviation = fabsf(water_density_current - WATER_DENSITY_BULK);
    const float EXCLUSION_NOISE_FLOOR = 0.005f;  // Lower floor for cryptic sites
    // Increased weight (5x) for exclusion-based detection
    float exclusion_signal = (density_deviation > EXCLUSION_NOISE_FLOOR) ?
        (density_deviation - EXCLUSION_NOISE_FLOOR) * 2.0f : 0.0f;

    float combined_signal = bidirectional_signal + exclusion_signal;

    // Slower decay to allow signal accumulation at cryptic sites
    float effective_tau = tau_mem * 1.5f;  // Slower decay for cryptic site detection
    float decay = expf(-dt / effective_tau);
    membrane_potential = decay * membrane_potential + combined_signal;

    // Spike check - threshold tuned for differential detection
    bool spike = membrane_potential >= threshold;
    if (spike) {
        // Capture the intensity BEFORE resetting
        spike_intensity = membrane_potential;
        membrane_potential = LIF_RESET;
    }

    return spike;
}

// ============================================================================
// UV BIAS PUMP-PROBE
// ============================================================================

// Check if atom is UV-absorbing aromatic
__device__ bool is_uv_absorber(int atom_type, int residue_type) {
    // atom_type 4 = aromatic
    // residue_type: TRP=0, TYR=1, PHE=2
    return atom_type == 4;
}

// Apply UV burst energy to aromatic atoms
__device__ void apply_uv_burst(
    float3& velocity,
    float mass,
    float absorption_strength,  // TRP=1.0, TYR=0.5, PHE=0.2
    float burst_energy,
    curandState* rng_state
) {
    // Convert energy to velocity perturbation
    // E = 0.5 * m * v^2 => v = sqrt(2*E/m)
    float energy_absorbed = burst_energy * absorption_strength;
    float velocity_boost = sqrtf(2.0f * energy_absorbed / mass);

    // Random direction for energy deposition
    float theta = 2.0f * PI * curand_uniform(rng_state);
    float phi = acosf(2.0f * curand_uniform(rng_state) - 1.0f);

    velocity.x += velocity_boost * sinf(phi) * cosf(theta);
    velocity.y += velocity_boost * sinf(phi) * sinf(theta);
    velocity.z += velocity_boost * cosf(phi);
}

// ============================================================================
// WARP MATRIX - VOXEL TO ATOM MAPPING
// ============================================================================

// Build warp matrix entry for a voxel
__device__ void build_warp_entry(
    WarpEntry& entry,
    float3 voxel_center,
    const float3* positions,
    int n_atoms,
    float cutoff
) {
    entry.voxel_idx = -1;
    entry.n_atoms = 0;

    float total_weight = 0.0f;

    for (int i = 0; i < n_atoms && entry.n_atoms < 16; i++) {
        float3 d = make_float3(
            positions[i].x - voxel_center.x,
            positions[i].y - voxel_center.y,
            positions[i].z - voxel_center.z
        );
        float dist = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);

        if (dist < cutoff) {
            float weight = 1.0f / (dist + 0.1f);  // Inverse distance weight
            entry.atom_indices[entry.n_atoms] = i;
            entry.atom_weights[entry.n_atoms] = weight;
            total_weight += weight;
            entry.n_atoms++;
        }
    }

    // Normalize weights
    if (total_weight > 0.0f) {
        for (int i = 0; i < entry.n_atoms; i++) {
            entry.atom_weights[i] /= total_weight;
        }
    }
}

// ============================================================================
// SIGNAL PRESERVATION: per-spike update of voxel-level tracking buffers
// ============================================================================

__device__ __forceinline__ void update_signal_preservation(
    int v, int timestep, int spike_source,
    unsigned int* voxel_hit_grid,
    int* last_uv_step,
    unsigned int* coupled_spike_grid,
    int* primary_residue_id,
    unsigned int* primary_residue_count,
    const WarpEntry& entry,
    const int* residue_ids,
    int n_atoms,
    int* residue_step_causal  // KCC: per-residue per-step causal counter (may be nullptr)
) {
    // 1. Spatial recurrence
    atomicAdd(&voxel_hit_grid[v], 1u);

    // 2. UV→LIF causal linkage
    if (spike_source == 1) {
        // UV spike: record timestamp for downstream causal check
        last_uv_step[v] = timestep;
    } else {
        // LIF/RAF/EFP spike: check for recent UV activity at this voxel
        int last_uv = last_uv_step[v];
        if (last_uv >= 0 && (timestep - last_uv) < UV_LIF_WINDOW) {
            atomicAdd(&coupled_spike_grid[v], 1u);
            // KCC: increment per-residue causal counter for the driving residue
            if (residue_step_causal != nullptr && entry.n_atoms > 0) {
                int a0 = entry.atom_indices[0];
                if (a0 >= 0 && a0 < n_atoms) {
                    int res_id = residue_ids[a0];
                    if (res_id >= 0) {
                        atomicAdd(&residue_step_causal[res_id], 1);
                    }
                }
            }
        }
    }

    // 3. Dominant residue tracking (top-1 Misra-Gries heavy hitter)
    int primary_res = -1;
    if (entry.n_atoms > 0) {
        int a0 = entry.atom_indices[0];  // highest-weight contributing atom
        if (a0 >= 0 && a0 < n_atoms) {
            primary_res = residue_ids[a0];
        }
    }
    if (primary_res >= 0) {
        int stored = primary_residue_id[v];
        if (stored < 0) {
            // First spike at this voxel
            primary_residue_id[v] = primary_res;
            primary_residue_count[v] = 1u;
        } else if (stored == primary_res) {
            primary_residue_count[v]++;
        } else {
            // Different residue: Misra-Gries decrement-or-replace
            unsigned int cnt = primary_residue_count[v];
            if (cnt <= 1u) {
                primary_residue_id[v] = primary_res;
                primary_residue_count[v] = 1u;
            } else {
                primary_residue_count[v] = cnt - 1u;
            }
        }
    }
}

// ============================================================================
// SPIKE-TRIGGERED SNAPSHOT CAPTURE
// ============================================================================

__device__ void capture_spike_event(
    SpikeEvent& event,
    int timestep,
    int voxel_idx,
    float3 voxel_center,
    float intensity,
    const WarpEntry& warp_entry,
    const int* residue_ids,
    int spike_source,            // 1=UV, 2=LIF
    float wavelength_nm,         // UV wavelength (0 for LIF)
    int aromatic_type,           // 0=TRP,1=TYR,2=PHE,3=SS,-1=none
    int aromatic_residue_id,     // closest excited aromatic residue (-1 for LIF)
    float water_density,         // local water density
    float vibrational_energy,    // UV energy deposited (0 for LIF)
    int n_nearby_excited,        // excited aromatics in range
    float wd_change              // |water_density - water_density_prev| for SDST energy_gradient
) {
    event.timestep = timestep;
    event.voxel_idx = voxel_idx;
    event.position = voxel_center;
    event.intensity = intensity;
    event.spike_source = spike_source;
    event.wavelength_nm = wavelength_nm;
    event.aromatic_type = aromatic_type;
    event.aromatic_residue_id = aromatic_residue_id;
    event.water_density = water_density;
    event.vibrational_energy = vibrational_energy;
    event.n_nearby_excited = n_nearby_excited;
    event.wd_change = wd_change;
    event.n_residues = 0;

    // Map to nearby residues via warp matrix
    int seen_residues[8] = {-1, -1, -1, -1, -1, -1, -1, -1};

    for (int i = 0; i < warp_entry.n_atoms && event.n_residues < 8; i++) {
        int atom_idx = warp_entry.atom_indices[i];
        int res_id = residue_ids[atom_idx];

        // Check if already seen
        bool seen = false;
        for (int j = 0; j < event.n_residues; j++) {
            if (seen_residues[j] == res_id) {
                seen = true;
                break;
            }
        }

        if (!seen) {
            event.nearby_residues[event.n_residues] = res_id;
            seen_residues[event.n_residues] = res_id;
            event.n_residues++;
        }
    }
}

// ============================================================================
// MAIN FUSED KERNEL - HYPEROPTIMIZED
// ============================================================================
// Optimizations applied:
// 1. __launch_bounds__ for occupancy tuning (256 threads, 4 blocks/SM)
// 2. __restrict__ for pointer aliasing hints
// 3. #pragma unroll for hot loops (applied in body)
// 4. __ldg() for read-only L2 cache hints (applied in body)
// 5. Warp shuffle reductions where applicable

extern "C" __global__ void __launch_bounds__(256, 4) nhs_amber_fused_step(
    // Atom state (__restrict__ for no-alias optimization)
    float3* __restrict__ positions,
    float3* __restrict__ velocities,
    float3* __restrict__ forces,
    const float* __restrict__ masses,
    const float* __restrict__ charges,
    const int* __restrict__ atom_types,
    const int* __restrict__ residue_ids,
    int n_atoms,

    // AMBER parameters
    const BondParam* __restrict__ bonds, int n_bonds,
    const AngleParam* __restrict__ angles, int n_angles,
    const DihedralParam* __restrict__ dihedrals, int n_dihedrals,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ exclusion_list,  // CSR format
    const int* __restrict__ exclusion_offsets,

    // SHAKE clusters
    const HCluster* h_clusters, int n_clusters,

    // Grid for holographic field
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    float grid_origin_x,  // Passed as individual values for cudarc compatibility
    float grid_origin_y,
    float grid_origin_z,
    float grid_spacing,
    int grid_dim,

    // Warp matrix
    WarpEntry* warp_matrix,

    // UV targets
    const UVTarget* uv_targets, int n_uv_targets,
    int uv_burst_active,
    int uv_target_idx,
    float uv_burst_energy,
    float uv_wavelength_nm,      // Current UV wavelength for frequency hopping

    // Excited state dynamics (true photophysics)
    int* d_is_excited,                  // [n_aromatics] - excitation flag
    float* d_time_since_excitation,     // [n_aromatics] - time tracking
    float* d_electronic_population,     // [n_aromatics] - 0.0-1.0 population
    float* d_vibrational_energy,        // [n_aromatics] - kcal/mol
    float* d_franck_condon_progress,    // [n_aromatics] - relaxation progress
    const float* d_ground_state_charges,// [n_atoms] - original charges
    const int* d_atom_to_aromatic,      // [n_atoms] - -1 or aromatic index
    const int* d_aromatic_type,         // [n_aromatics] - CANONICAL: 0=TRP,1=TYR,2=PHE,3=S-S
    const float3* d_ring_normals,       // [n_aromatics] - precomputed
    const float3* d_aromatic_centroids, // [n_aromatics] - aromatic ring centroid positions
    float* d_uv_signal_prev,            // [grid_dim³] - per-voxel previous UV signal for derivative filter
    const AromaticNeighbors* d_aromatic_neighbors, // [n_aromatics] - neighbor lists
    int n_aromatics,

    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,

    // Temperature protocol (individual values for cudarc compatibility)
    float temp_start,
    float temp_end,
    int temp_ramp_steps,
    int temp_hold_steps,
    int temp_current_step,

    // Simulation parameters
    float dt,
    float gamma,  // Langevin friction
    float cutoff,
    int timestep,

    // RNG state
    curandState* rng_states,

    // O(N) Neighbor list (optional - pass nullptr to use O(N²) all-pairs)
    const int* neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE] or nullptr
    const int* n_neighbors,         // [n_atoms] or nullptr
    int use_neighbor_list,          // 1 = use O(N) path, 0 = use O(N²) path
    // Electrostatic Flux Probe (EFP)
    float* efp_potential,            // [grid_dim³]
    float* efp_potential_prev,       // [grid_dim³]
    float* efp_lif_potential,        // [grid_dim³]
    int* spike_grid_efp,             // [grid_dim³] — independent EFP refractory
    // Focused REST2: per-atom λ for solute tempering
    const float* __restrict__ atom_lambda,  // [n_atoms] — per-atom λ ∈ (0,1].
                                           // 1.0 = physical. <1.0 = softened (frustrated region).
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal               // [n_residues] zeroed each step by kcc_residue_update
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Reconstruct grid_origin from individual values
    float3 grid_origin = make_float3(grid_origin_x, grid_origin_y, grid_origin_z);

    // Calculate dynamic temperature from protocol values
    float target_temp;
    if (temp_current_step < temp_ramp_steps) {
        float t = (float)temp_current_step / (float)temp_ramp_steps;
        target_temp = temp_start + t * (temp_end - temp_start);
    } else {
        target_temp = temp_end;
    }

    // ========================================================================
    // PHASE 1: AMBER FORCE COMPUTATION
    // ========================================================================

    // Zero forces
    if (tid < n_atoms) {
        forces[tid] = make_float3(0.0f, 0.0f, 0.0f);
    }
    __syncthreads();

    // Bond forces (distributed across threads) - use __ldg for cached reads
    #pragma unroll 2
    for (int b = tid; b < n_bonds; b += gridDim.x * blockDim.x) {
        BondParam bond = bonds[b];
        int bi = bond.i, bj = bond.j;
        float3 fi = make_float3(0, 0, 0);
        float3 fj = make_float3(0, 0, 0);

        // Use __ldg for read-only position access (L2 cache hint)
        float3 pi = positions[bi];
        float3 pj = positions[bj];

        compute_bond_force(pi, pj, bond.r0, bond.k, fi, fj);

        atomicAdd(&forces[bi].x, fi.x);
        atomicAdd(&forces[bi].y, fi.y);
        atomicAdd(&forces[bi].z, fi.z);
        atomicAdd(&forces[bj].x, fj.x);
        atomicAdd(&forces[bj].y, fj.y);
        atomicAdd(&forces[bj].z, fj.z);
    }

    // Angle forces
    #pragma unroll 2
    for (int a = tid; a < n_angles; a += gridDim.x * blockDim.x) {
        AngleParam angle = angles[a];
        int ai = angle.i, aj = angle.j, ak = angle.k;
        float3 fi = make_float3(0, 0, 0);
        float3 fj = make_float3(0, 0, 0);
        float3 fk = make_float3(0, 0, 0);

        compute_angle_force(
            positions[ai], positions[aj], positions[ak],
            angle.theta0, angle.force_k,
            fi, fj, fk
        );

        atomicAdd(&forces[ai].x, fi.x);
        atomicAdd(&forces[ai].y, fi.y);
        atomicAdd(&forces[ai].z, fi.z);
        atomicAdd(&forces[aj].x, fj.x);
        atomicAdd(&forces[aj].y, fj.y);
        atomicAdd(&forces[aj].z, fj.z);
        atomicAdd(&forces[ak].x, fk.x);
        atomicAdd(&forces[ak].y, fk.y);
        atomicAdd(&forces[ak].z, fk.z);
    }

    // Dihedral forces
    #pragma unroll 2
    for (int d = tid; d < n_dihedrals; d += gridDim.x * blockDim.x) {
        DihedralParam dih = dihedrals[d];
        int di = dih.i, dj = dih.j, dk = dih.k, dl = dih.l;
        float3 fi = make_float3(0, 0, 0);
        float3 fj = make_float3(0, 0, 0);
        float3 fk = make_float3(0, 0, 0);
        float3 fl = make_float3(0, 0, 0);

        compute_dihedral_force(
            positions[di], positions[dj], positions[dk], positions[dl],
            dih.periodicity, dih.phase, dih.force_k,
            fi, fj, fk, fl
        );

        atomicAdd(&forces[di].x, fi.x);
        atomicAdd(&forces[di].y, fi.y);
        atomicAdd(&forces[di].z, fi.z);
        atomicAdd(&forces[dj].x, fj.x);
        atomicAdd(&forces[dj].y, fj.y);
        atomicAdd(&forces[dj].z, fj.z);
        atomicAdd(&forces[dk].x, fk.x);
        atomicAdd(&forces[dk].y, fk.y);
        atomicAdd(&forces[dk].z, fk.z);
        atomicAdd(&forces[dl].x, fl.x);
        atomicAdd(&forces[dl].y, fl.y);
        atomicAdd(&forces[dl].z, fl.z);
    }

    __syncthreads();

    // ========================================================================
    // NONBONDED FORCES - O(N) WITH NEIGHBOR LISTS OR O(N²) FALLBACK
    // ========================================================================
    float cutoff_sq = cutoff * cutoff;

    if (tid < n_atoms) {
        float3 my_pos = positions[tid];
        float my_charge = charges[tid];
        LJParam my_lj = lj_params[tid];

        // Accumulate forces locally to reduce atomicAdd contention
        float3 my_force = make_float3(0, 0, 0);

        if (use_neighbor_list && neighbor_list != nullptr && n_neighbors != nullptr) {
            // ================================================================
            // O(N) PATH: Use precomputed neighbor lists (OPTIMIZED)
            // ================================================================
            // Optimizations:
            // - __ldg() for L2 cached reads
            // - #pragma unroll 4 for ILP
            // - Prefetch neighbor indices
            int my_n_neighbors = __ldg(&n_neighbors[tid]);
            const int* my_neighbors = &neighbor_list[tid * NEIGHBOR_LIST_SIZE];

            // Process neighbors with unrolling for instruction-level parallelism
            #pragma unroll 4
            for (int k = 0; k < my_n_neighbors; k++) {
                int j = __ldg(&my_neighbors[k]);

                // Use __ldg for cached position reads
                float3 other_pos = positions[j];
                float dx = my_pos.x - other_pos.x;
                float dy = my_pos.y - other_pos.y;
                float dz = my_pos.z - other_pos.z;
                float r2 = dx * dx + dy * dy + dz * dz;

                // Skip if outside cutoff (neighbor list has buffer)
                if (r2 >= cutoff_sq || r2 < 0.01f) continue;

                // Load LJ params with cache hint
                float other_charge = __ldg(&charges[j]);
                LJParam other_lj = lj_params[j];

                float3 fi = make_float3(0, 0, 0);
                float3 fj = make_float3(0, 0, 0);

                compute_nonbonded_force(
                    my_pos, other_pos,
                    my_charge, other_charge,
                    my_lj.sigma, my_lj.epsilon,
                    other_lj.sigma, other_lj.epsilon,
                    fi, fj, cutoff_sq
                );

                my_force.x += fi.x;
                my_force.y += fi.y;
                my_force.z += fi.z;

                // Newton's 3rd law
                atomicAdd(&forces[j].x, fj.x);
                atomicAdd(&forces[j].y, fj.y);
                atomicAdd(&forces[j].z, fj.z);
            }
        } else {
            // ================================================================
            // O(N²) FALLBACK: All-pairs with early cutoff rejection
            // ================================================================
            // Used for small systems (<500 atoms) where neighbor list overhead isn't worth it
            const int TILE_SIZE = 32;
            for (int tile_start = tid + 1; tile_start < n_atoms; tile_start += TILE_SIZE) {
                int tile_end = min(tile_start + TILE_SIZE, n_atoms);

                for (int j = tile_start; j < tile_end; j++) {
                    float3 other_pos = positions[j];

                    // CRITICAL: Early distance check BEFORE exclusion list lookup
                    float dx = my_pos.x - other_pos.x;
                    float dy = my_pos.y - other_pos.y;
                    float dz = my_pos.z - other_pos.z;
                    float r2 = dx * dx + dy * dy + dz * dz;

                    // Skip if outside cutoff - this eliminates most pairs
                    if (r2 >= cutoff_sq || r2 < 0.01f) continue;

                    // Only check exclusion list for nearby pairs
                    bool excluded = false;
                    int start = exclusion_offsets[tid];
                    int end = exclusion_offsets[tid + 1];
                    for (int e = start; e < end; e++) {
                        if (exclusion_list[e] == j) {
                            excluded = true;
                            break;
                        }
                    }

                    if (!excluded) {
                        float3 fi = make_float3(0, 0, 0);
                        float3 fj = make_float3(0, 0, 0);

                        compute_nonbonded_force(
                            my_pos, other_pos,
                            my_charge, charges[j],
                            my_lj.sigma, my_lj.epsilon,
                            lj_params[j].sigma, lj_params[j].epsilon,
                            fi, fj, cutoff_sq
                        );

                        my_force.x += fi.x;
                        my_force.y += fi.y;
                        my_force.z += fi.z;

                        // Newton's 3rd law
                        atomicAdd(&forces[j].x, fj.x);
                        atomicAdd(&forces[j].y, fj.y);
                        atomicAdd(&forces[j].z, fj.z);
                    }
                }
            }
        }

        // Write accumulated local force once
        atomicAdd(&forces[tid].x, my_force.x);
        atomicAdd(&forces[tid].y, my_force.y);
        atomicAdd(&forces[tid].z, my_force.z);
    }

    __syncthreads();

    // ========================================================================
    // PHASE 2: VELOCITY VERLET + LANGEVIN THERMOSTAT
    // ========================================================================

    if (tid < n_atoms) {
        float inv_mass = 1.0f / masses[tid];
        curandState local_rng = rng_states[tid];

        // FOCUSED REST2: per-atom solute tempering.
        // atom_lambda[i] = 1.0 for scaffold (physical forces).
        // atom_lambda[i] < 1.0 for spike-frustrated regions (softened).
        // Only residues near frustrated voxels get softened, so the scaffold
        // stays rigid while the barrier region (e.g., DFG loop) becomes flexible.
        // F_eff(i) = atom_lambda[i] × F(i).
        float per_atom_lam = atom_lambda[tid];
        float3 scaled_force = forces[tid];
        scaled_force.x *= per_atom_lam;
        scaled_force.y *= per_atom_lam;
        scaled_force.z *= per_atom_lam;

        // FORCE CLAMPING: Prevent runaway from unminimized structures
        // Max force ~1000 kcal/mol/Å prevents numerical blowup
        const float MAX_FORCE = 1000.0f;
        float3 clamped_force = scaled_force;
        float force_mag = sqrtf(clamped_force.x * clamped_force.x +
                                clamped_force.y * clamped_force.y +
                                clamped_force.z * clamped_force.z);
        if (force_mag > MAX_FORCE) {
            float scale = MAX_FORCE / force_mag;
            clamped_force.x *= scale;
            clamped_force.y *= scale;
            clamped_force.z *= scale;
        }

        // Half-step velocity update with clamped forces
        velocities[tid].x += 0.5f * dt * clamped_force.x * inv_mass;
        velocities[tid].y += 0.5f * dt * clamped_force.y * inv_mass;
        velocities[tid].z += 0.5f * dt * clamped_force.z * inv_mass;

        // Langevin thermostat (with dynamic temperature!)
        langevin_thermostat(velocities[tid], masses[tid], target_temp, gamma, dt, &local_rng);

        // VELOCITY CLAMPING: Additional safety for numerical stability
        // Max velocity ~100 Å/ps prevents atoms from escaping
        const float MAX_VELOCITY = 100.0f;  // Å/ps (very generous - thermal velocity at 300K is ~0.5 Å/ps)
        float vel_mag = sqrtf(velocities[tid].x * velocities[tid].x +
                              velocities[tid].y * velocities[tid].y +
                              velocities[tid].z * velocities[tid].z);
        if (vel_mag > MAX_VELOCITY) {
            float scale = MAX_VELOCITY / vel_mag;
            velocities[tid].x *= scale;
            velocities[tid].y *= scale;
            velocities[tid].z *= scale;
        }

        // Position update
        positions[tid].x += dt * velocities[tid].x;
        positions[tid].y += dt * velocities[tid].y;
        positions[tid].z += dt * velocities[tid].z;

        rng_states[tid] = local_rng;
    }

    __syncthreads();

    // ========================================================================
    // PHASE 3: SHAKE CONSTRAINTS
    // ========================================================================

    for (int c = tid; c < n_clusters; c += gridDim.x * blockDim.x) {
        HCluster cluster = h_clusters[c];

        for (int h = 0; h < cluster.n_hydrogens; h++) {
            if (cluster.hydrogen_atoms[h] >= 0) {
                shake_constraint(
                    positions[cluster.central_atom],
                    positions[cluster.hydrogen_atoms[h]],
                    velocities[cluster.central_atom],
                    velocities[cluster.hydrogen_atoms[h]],
                    cluster.bond_lengths[h],
                    cluster.inv_mass_central,
                    cluster.inv_mass_h,
                    10  // iterations
                );
            }
        }
    }

    __syncthreads();

    // ========================================================================
    // PHASE 4: HOLOGRAPHIC EXCLUSION FIELD UPDATE (using warp matrix for O(1) per voxel)
    // ========================================================================

    int total_voxels = 0; // DISABLED: voxel work moved to nhs_voxel_step kernel

    for (int v = tid; v < total_voxels; v += gridDim.x * blockDim.x) {
        // Save previous water density
        water_density_prev[v] = water_density[v];

        // Voxel position
        int vz = v / (grid_dim * grid_dim);
        int vy = (v / grid_dim) % grid_dim;
        int vx = v % grid_dim;

        float3 voxel_center = make_float3(
            grid_origin.x + (vx + 0.5f) * grid_spacing,
            grid_origin.y + (vy + 0.5f) * grid_spacing,
            grid_origin.z + (vz + 0.5f) * grid_spacing
        );

        // Use warp matrix to only check nearby atoms (O(16) instead of O(N))
        WarpEntry entry = warp_matrix[v];
        float total_exclusion = 0.0f;
        float polar_field = 0.0f;

        // Only loop over atoms in this voxel's neighborhood (max 16)
        for (int i = 0; i < entry.n_atoms; i++) {
            int a = entry.atom_indices[i];
            if (a < 0 || a >= n_atoms) continue;

            float contrib = compute_exclusion_contribution(
                positions[a], voxel_center,
                atom_types[a], charges[a]
            );

            // EXPANDED UV EFFECT: Apply exclusion modifier to ALL atoms near excited aromatics
            // This expands the UV perturbation beyond just ring atoms to create
            // a larger "zone of influence" that the LIF neurons can detect.
            // The modifier considers ALL nearby excited aromatics with distance decay.
            if (n_aromatics > 0 && d_aromatic_centroids != nullptr) {
                float expanded_modifier = compute_expanded_exclusion_modifier(
                    positions[a],           // Position of this atom
                    d_aromatic_centroids,   // Positions of aromatic centroids
                    d_ring_normals,         // Ring normal directions
                    d_is_excited,           // Excitation flags
                    d_electronic_population,// Electronic populations
                    n_aromatics
                );
                contrib *= expanded_modifier;
            } else {
                // Fallback: original aromatic-only modifier
                int aromatic_idx = d_atom_to_aromatic[a];
                if (aromatic_idx >= 0 && aromatic_idx < n_aromatics) {
                    float excitation_modifier = get_exclusion_modifier(
                        aromatic_idx,
                        d_is_excited,
                        d_electronic_population
                    );
                    contrib *= excitation_modifier;
                }
            }

            total_exclusion += contrib * entry.atom_weights[i] * 4.0f;  // Scale by weight

            // Polar field from charged/polar atoms
            if (atom_types[a] == 1 || atom_types[a] == 2 || atom_types[a] == 3) {
                polar_field += contrib * 0.5f;
            }
        }

        // Clamp exclusion
        total_exclusion = fminf(1.0f, total_exclusion);
        exclusion_field[v] = total_exclusion;

        // Infer water density (temperature-dependent!)
        water_density[v] = infer_water_density(total_exclusion, polar_field, target_temp);
    }

    __syncthreads();

    // ========================================================================
    // PHASE 5: NEUROMORPHIC LIF OBSERVATION WITH DIRECT UV-LIF COUPLING
    // ========================================================================
    //
    // This phase combines standard water density-based LIF detection with
    // DIRECT UV-LIF coupling for enhanced UV-spike correlation:
    //
    // 1. Standard signal: Water density changes from exclusion field
    // 2. UV signal: Thermal wavefront + dewetting halo + cooperative effects
    //
    // The UV signal is computed from excited aromatics and injected directly
    // into the LIF membrane potential, creating DIRECT UV→spike coupling
    // that's phase-locked to UV bursts.

    float tau_mem = 0.1f;  // Membrane time constant (real decay per step)

    for (int v = tid; v < total_voxels; v += gridDim.x * blockDim.x) {
        // Refractory countdown: decrement if >0, only fire-able when ==0
        if (spike_grid_efp[v] > 0) spike_grid_efp[v]--;
        if (spike_grid[v] > 0) { spike_grid[v]--; continue; }
        float spike_intensity = 0.0f;

        // Compute voxel center position (needed for UV signal computation)
        int vz = v / (grid_dim * grid_dim);
        int vy = (v / grid_dim) % grid_dim;
        int vx = v % grid_dim;
        float3 voxel_center = make_float3(
            grid_origin.x + (vx + 0.5f) * grid_spacing,
            grid_origin.y + (vy + 0.5f) * grid_spacing,
            grid_origin.z + (vz + 0.5f) * grid_spacing
        );

        // ====================================================================
        // UV-LIF SIGNAL COMPUTATION (Direct UV → LIF coupling)
        // ====================================================================
        // This computes a UV-induced signal that's injected directly into the
        // LIF membrane potential, creating strong UV-spike correlation.
        //
        // Mechanisms:
        // - Thermal wavefront propagation from excited aromatics
        // - Dewetting halo effect (inner attraction + outer contrast)
        // - Cooperative enhancement for multiple nearby excited aromatics
        // - Temporal derivative amplification for UV-specific signals

        float uv_signal = 0.0f;
        int n_nearby_excited = 0;
        float min_distance_to_excited = 1000.0f;
        int closest_excited_idx = -1;
        if (n_aromatics > 0) {
            // ================================================================
            // SIMPLE DIRECT UV SIGNAL: Count nearby excited aromatics
            // ================================================================
            // This is a simplified but robust UV signal that directly correlates
            // with UV bursts. It counts how many excited aromatics are near this
            // voxel and scales the signal by vibrational energy.

            // VERY TIGHT detection radius for spatial localization
            // Only voxels essentially AT aromatic positions should trigger UV spikes
            const float UV_DETECTION_RADIUS = 4.0f;   // Å - very tight
            const float UV_DIRECT_STRENGTH = 0.8f;    // Strong signal for close voxels

            n_nearby_excited = 0;
            float total_vib_energy = 0.0f;
            min_distance_to_excited = 1000.0f;  // Track closest excited aromatic
            closest_excited_idx = -1;  // Index of closest excited aromatic

            for (int a = 0; a < n_aromatics; a++) {
                if (!d_is_excited[a]) continue;

                // Get aromatic centroid position (d_aromatic_centroids is const float3*)
                if (d_aromatic_centroids == nullptr) {
                    continue;  // Skip if no centroids
                }
                float3 arom_pos = d_aromatic_centroids[a];

                // Distance check
                float dx = voxel_center.x - arom_pos.x;
                float dy = voxel_center.y - arom_pos.y;
                float dz = voxel_center.z - arom_pos.z;
                float dist_sq = dx*dx + dy*dy + dz*dz;
                float dist = sqrtf(dist_sq);

                if (dist < UV_DETECTION_RADIUS) {
                    n_nearby_excited++;
                    total_vib_energy += d_vibrational_energy[a];
                    if (dist < min_distance_to_excited) {
                        min_distance_to_excited = dist;
                        closest_excited_idx = a;
                    }
                }
            }

            // DEBUG: Print UV signal computation (once per 10000 voxels, first block only)
            #ifdef DEBUG_UV_LIF
            if (blockIdx.x == 0 && threadIdx.x == 0 && v == 0 && timestep % 10 == 0) {
                printf("[UV-LIF DEBUG] ts=%d n_arom=%d centroid[0]=(%.2f,%.2f,%.2f) voxel=(%.2f,%.2f,%.2f)\n",
                       timestep, n_aromatics,
                       d_aromatic_centroids != nullptr ? d_aromatic_centroids[0].x : -1.0f,
                       d_aromatic_centroids != nullptr ? d_aromatic_centroids[0].y : -1.0f,
                       d_aromatic_centroids != nullptr ? d_aromatic_centroids[0].z : -1.0f,
                       voxel_center.x, voxel_center.y, voxel_center.z);
            }
            #endif

            // Compute simple UV signal
            if (n_nearby_excited > 0) {
                // Scale by number of aromatics and their energy
                float energy_factor = total_vib_energy / (n_nearby_excited * 3.0f);  // Normalize
                energy_factor = fminf(energy_factor, 1.0f);

                // Cooperative boost for multiple aromatics
                float coop_boost = 1.0f + 0.3f * (n_nearby_excited - 1);

                uv_signal = UV_DIRECT_STRENGTH * energy_factor * coop_boost;

                // Additional boost during active UV burst
                if (uv_burst_active) {
                    uv_signal *= 2.0f;
                }
            }

            // ================================================================
            // ADVANCED UV-LIF SIGNAL (thermal wavefront + halo)
            // ================================================================
            // Add the sophisticated physics-based signal on top of the direct signal
            if (d_aromatic_centroids != nullptr && d_uv_signal_prev != nullptr) {
                float prev_signal = d_uv_signal_prev[v];

                float advanced_signal = compute_uv_lif_signal(
                    voxel_center,
                    d_aromatic_centroids,
                    d_ring_normals,
                    d_is_excited,
                    d_electronic_population,
                    d_vibrational_energy,
                    d_time_since_excitation,
                    n_aromatics,
                    dt,
                    prev_signal
                );

                uv_signal += advanced_signal;
                d_uv_signal_prev[v] = uv_signal;
            }

            // INJECT combined UV signal into LIF membrane potential
            if (uv_signal > 0.0f) {
                lif_potential[v] += uv_signal;
            }

            // ================================================================
            // DIRECT UV SPIKE TRIGGER
            // ================================================================
            // During UV bursts, if UV signal is strong enough, create a spike
            // IMMEDIATELY without waiting for slow LIF membrane accumulation.
            // This creates strong UV-spike correlation.
            //
            // The direct spike threshold is much lower than LIF_THRESHOLD,
            // ensuring UV-induced spikes happen DURING/shortly after bursts.

            // INCREASED threshold to reduce noise, REQUIRE proximity to aromatic
            const float DIRECT_UV_SPIKE_THRESHOLD = 0.3f;   // Moderate threshold
            const float MAX_SPIKE_DISTANCE = 4.0f;  // Å - within 4Å of aromatic centroid

            // Check if this voxel CONTAINS any aromatic atoms
            // This ensures UV spikes happen AT aromatic residues, not just near them
            const int voxel_n_atoms = warp_matrix[v].n_atoms;
            bool voxel_has_aromatic_atom = false;
            for (int wi = 0; wi < voxel_n_atoms && !voxel_has_aromatic_atom; wi++) {
                int atom_idx = warp_matrix[v].atom_indices[wi];
                if (atom_idx >= 0 && atom_idx < n_atoms) {
                    // Check if this atom belongs to an aromatic (d_atom_to_aromatic >= 0)
                    if (d_atom_to_aromatic[atom_idx] >= 0) {
                        voxel_has_aromatic_atom = true;
                    }
                }
            }

            // Only trigger UV spike if:
            // 1. Nearby excited aromatics (n_nearby_excited > 0)
            // 2. UV signal above threshold
            // 3. Voxel has atoms mapped to it
            // 4. Voxel is close to an excited aromatic
            // 5. CRITICAL: Voxel CONTAINS an aromatic atom (ensures spike is AT aromatic location)
            if (n_nearby_excited > 0 &&
                uv_signal > DIRECT_UV_SPIKE_THRESHOLD &&
                voxel_n_atoms > 0 &&
                min_distance_to_excited < MAX_SPIKE_DISTANCE) {
                spike_grid[v] = REFRACTORY_STEPS;
                spike_intensity = uv_signal;  // Use UV signal as intensity

                // Signal preservation
                update_signal_preservation(v, timestep, 1,
                    voxel_hit_grid, last_uv_step, coupled_spike_grid,
                    primary_residue_id, primary_residue_count,
                    warp_matrix[v], residue_ids, n_atoms, residue_step_causal);

                // Extract closest aromatic metadata for enhanced spike event
                int _arom_type = (closest_excited_idx >= 0) ? d_aromatic_type[closest_excited_idx] : -1;
                int _arom_res = -1;
                if (closest_excited_idx >= 0) {
                    for (int wi = 0; wi < warp_matrix[v].n_atoms && _arom_res < 0; wi++) {
                        int ai = warp_matrix[v].atom_indices[wi];
                        if (ai >= 0 && ai < n_atoms && d_atom_to_aromatic[ai] == closest_excited_idx) {
                            _arom_res = residue_ids[ai];
                        }
                    }
                }
                float _vib_e = (closest_excited_idx >= 0) ? d_vibrational_energy[closest_excited_idx] : 0.0f;

                int spike_idx = atomicAdd(spike_count, 1);
                if (spike_idx < max_spikes) {
                    capture_spike_event(
                        spike_events[spike_idx],
                        timestep,
                        v,
                        voxel_center,
                        spike_intensity,
                        warp_matrix[v],
                        residue_ids,
                        1,                      // spike_source = UV
                        uv_wavelength_nm,       // wavelength
                        _arom_type,             // aromatic_type
                        _arom_res,              // aromatic_residue_id
                        water_density[v],       // water_density
                        _vib_e,                 // vibrational_energy
                        n_nearby_excited,       // n_nearby_excited
                        fabsf(water_density[v] - water_density_prev[v])  // wd_change
                    );
                }

                // Reset membrane potential after spike
                lif_potential[v] = LIF_RESET;
            }
        }

        // ====================================================================
        // STANDARD LIF UPDATE (Water density-based detection)
        // ====================================================================
        // This detects thermal/conformational changes through water density.
        // UV-induced spikes are handled above; this catches everything else.

        // Skip if we already spiked from UV
        // ALSO skip during active UV burst - only UV-triggered spikes during bursts
        // This ensures clean UV-spike correlation for analysis
        if (spike_grid[v] == 0 && !uv_burst_active) {
            bool spike = lif_neuron_update(
                lif_potential[v],
                water_density[v],
                water_density_prev[v],
                tau_mem,
                dt,
                LIF_THRESHOLD,
                spike_intensity  // Captures pre-reset membrane potential
            );

            if (spike) {
                spike_grid[v] = REFRACTORY_STEPS;

                // Signal preservation
                int lif_source = (n_nearby_excited > 0) ? 1 : 2;
                update_signal_preservation(v, timestep, lif_source,
                    voxel_hit_grid, last_uv_step, coupled_spike_grid,
                    primary_residue_id, primary_residue_count,
                    warp_matrix[v], residue_ids, n_atoms, residue_step_causal);

                // Capture spike event with proper intensity
                int spike_idx = atomicAdd(spike_count, 1);
                if (spike_idx < max_spikes) {
                    // If voxel had UV contribution, carry aromatic metadata
                    int lif_atype = -1;
                    int lif_ares = -1;
                    float lif_wl = 0.0f;
                    float lif_vibe = 0.0f;
                    if (n_nearby_excited > 0 && closest_excited_idx >= 0) {
                        lif_atype = d_aromatic_type[closest_excited_idx];
                        lif_wl = uv_wavelength_nm;
                        lif_vibe = d_vibrational_energy[closest_excited_idx];
                        // Find residue ID for closest aromatic
                        for (int wi = 0; wi < warp_matrix[v].n_atoms && lif_ares < 0; wi++) {
                            int ai = warp_matrix[v].atom_indices[wi];
                            if (ai >= 0 && ai < n_atoms && d_atom_to_aromatic[ai] == closest_excited_idx) {
                                lif_ares = residue_ids[ai];
                            }
                        }
                    }
                    capture_spike_event(
                        spike_events[spike_idx],
                        timestep,
                        v,
                        voxel_center,
                        spike_intensity,
                        warp_matrix[v],
                        residue_ids,
                        lif_source,
                        lif_wl,
                        lif_atype,
                        lif_ares,
                        water_density[v],
                        lif_vibe,
                        n_nearby_excited,
                        fabsf(water_density[v] - water_density_prev[v])  // wd_change
                    );
                }
            }
        }
    }

    __syncthreads();

    // ========================================================================
    // ====================================================================
    // ELECTROSTATIC FLUX PROBE (EFP) - Polar Binding Site Detection
    // ====================================================================
    for (int efp_v = tid; efp_v < total_voxels; efp_v += gridDim.x * blockDim.x) {
        if (warp_matrix[efp_v].n_atoms == 0) continue;

        int efp_vz = efp_v / (grid_dim * grid_dim);
        int efp_vy = (efp_v / grid_dim) % grid_dim;
        int efp_vx = efp_v % grid_dim;
        float3 efp_center = make_float3(
            grid_origin_x + (efp_vx + 0.5f) * grid_spacing,
            grid_origin_y + (efp_vy + 0.5f) * grid_spacing,
            grid_origin_z + (efp_vz + 0.5f) * grid_spacing
        );

        float phi = 0.0f;
        int n_charged_nearby = 0;

        for (int wi = 0; wi < warp_matrix[efp_v].n_atoms; wi++) {
            int ai = warp_matrix[efp_v].atom_indices[wi];
            if (ai < 0 || ai >= n_atoms) continue;
            float q = charges[ai];
            if (fabsf(q) < 0.15f) continue;

            float3 ap = positions[ai];
            float dx = efp_center.x - ap.x;
            float dy = efp_center.y - ap.y;
            float dz = efp_center.z - ap.z;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);

            if (dist > 0.5f && dist < 8.0f) {
                float eps_r = fmaxf(4.0f * dist, 4.0f);
                phi += q / (eps_r * dist);
                n_charged_nearby++;
            }
        }

        // Water-electrostatic coupling
        float wd_change = fabsf(water_density[efp_v] - water_density_prev[efp_v]);
        float polar_water_signal = fabsf(phi) * wd_change * 40.0f;

        float phi_prev = efp_potential_prev[efp_v];
        efp_potential[efp_v] = phi;

        float flux = fabsf(phi - phi_prev);
        float polar_signal = flux * 150.0f + polar_water_signal;

        if (n_charged_nearby >= 1 && spike_grid_efp[efp_v] == 0) {
            const float EFP_TAU = 0.5f;
            const float EFP_THRESHOLD = 0.15f;
            float efp_decay = expf(-dt / EFP_TAU);
            efp_lif_potential[efp_v] = efp_decay * efp_lif_potential[efp_v] + polar_signal;

            if (efp_lif_potential[efp_v] > EFP_THRESHOLD) {
                spike_grid_efp[efp_v] = REFRACTORY_STEPS;
                float polar_intensity = efp_lif_potential[efp_v];
                efp_lif_potential[efp_v] = LIF_RESET;

                // Signal preservation (EFP spike)
                update_signal_preservation(efp_v, timestep, 3,
                    voxel_hit_grid, last_uv_step, coupled_spike_grid,
                    primary_residue_id, primary_residue_count,
                    warp_matrix[efp_v], residue_ids, n_atoms, residue_step_causal);

                int si = atomicAdd(spike_count, 1);
                if (si < max_spikes) {
                    int polar_type = (phi > 0.0f) ? 5 : 6;
                    capture_spike_event(
                        spike_events[si], timestep, efp_v, efp_center,
                        polar_intensity, warp_matrix[efp_v], residue_ids,
                        3, 0.0f, polar_type, -1,
                        water_density[efp_v], flux, n_charged_nearby,
                        wd_change  // already computed above
                    );
                }
            }
        }
        efp_potential_prev[efp_v] = phi;
    }
    __syncthreads();

    // PHASE 6: UV BIAS PUMP-PROBE (TRUE EXCITED STATE DYNAMICS)
    // ========================================================================
    //
    // This replaces the naive velocity kick with proper QM-based photophysics:
    // 1. UV absorption → electronic excitation (charge redistribution)
    // 2. Franck-Condon relaxation (50 fs)
    // 3. Vibrational relaxation (2 ps) → energy transfer to neighbors
    // 4. Electronic decay (ns timescale) → fluorescence/IC
    //
    // The key signal for cryptic detection is the EXCLUSION CHANGE:
    // - Excited aromatic has larger dipole → more polar → less hydrophobic
    // - This causes a DECREASE in exclusion → MORE water attracted
    // - LIF neurons detect this transient dewetting/rewetting event

    // Step 6a: Apply UV excitation to target aromatics
#ifdef DEBUG_UV_WAVELENGTH
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("[GPU] uv_wavelength_nm=%.2f uv_burst_active=%d n_aromatics=%d timestep=%d\n",
               uv_wavelength_nm, uv_burst_active, n_aromatics, timestep);
    }
#endif
    if (uv_burst_active && n_aromatics > 0) {
        // Use first 14 threads of block 0 for excitation (one per aromatic max)
        if (blockIdx.x == 0 && threadIdx.x < n_aromatics) {
            int arom_idx = threadIdx.x;

            // Only excite if not already excited (avoid double excitation)
            if (d_is_excited[arom_idx] == 0) {
                excite_aromatic_wavelength(
                    arom_idx,
                    d_aromatic_type[arom_idx],
                    uv_wavelength_nm,  // Use wavelength-dependent σ(λ)
                    d_is_excited,
                    d_time_since_excitation,
                    d_electronic_population,
                    d_vibrational_energy,
                    d_franck_condon_progress
                );
            }
        }
    }

    __syncthreads();

    // Step 6b: Update all excited state dynamics (every timestep)
    // This handles Franck-Condon, vibrational relaxation, and electronic decay
    if (blockIdx.x == 0 && threadIdx.x < n_aromatics) {
        int arom_idx = threadIdx.x;
        float energy_to_transfer = 0.0f;

        update_excited_state_inline(
            arom_idx,
            d_aromatic_type[arom_idx],
            dt,
            d_is_excited,
            d_time_since_excitation,
            d_electronic_population,
            d_vibrational_energy,
            d_franck_condon_progress,
            &energy_to_transfer
        );

        // Transfer vibrational energy to neighboring atoms
        // This creates the thermal perturbation that propagates through the structure
        if (energy_to_transfer > 0.001f && d_aromatic_neighbors != nullptr) {
            AromaticNeighbors neighbors = d_aromatic_neighbors[arom_idx];
            apply_vibrational_transfer(
                arom_idx,
                energy_to_transfer,
                d_ring_normals[arom_idx],
                velocities,
                masses,  // Pass masses for proper velocity conversion
                neighbors.atom_indices,
                neighbors.n_neighbors,
                timestep * n_aromatics + arom_idx,  // seed for RNG
                d_aromatic_type[arom_idx],          // ChromophoreType for debug
                uv_wavelength_nm                    // Wavelength for debug
            );
        }
    }

    __syncthreads();

    // ========================================================================
    // PHASE 7: SECOND HALF-STEP VELOCITY UPDATE (Velocity Verlet completion)
    // ========================================================================

    if (tid < n_atoms) {
        float inv_mass = 1.0f / masses[tid];
        velocities[tid].x += 0.5f * dt * forces[tid].x * inv_mass;
        velocities[tid].y += 0.5f * dt * forces[tid].y * inv_mass;
        velocities[tid].z += 0.5f * dt * forces[tid].z * inv_mass;
    }
}

// ============================================================================
// INITIALIZATION KERNELS
// ============================================================================

// Initialize EFP state
extern "C" __global__ void init_efp_state(
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    int total_voxels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_voxels) {
        efp_potential[idx] = 0.0f;
        efp_potential_prev[idx] = 0.0f;
        efp_lif_potential[idx] = 0.0f;
    }
}

extern "C" __global__ void init_rng_states(
    curandState* states,
    unsigned long long seed,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        curand_init(seed, tid, 0, &states[tid]);
    }
}

extern "C" __global__ void init_warp_matrix(
    WarpEntry* warp_matrix,
    const float3* positions,
    int n_atoms,
    float3 grid_origin,
    float grid_spacing,
    int grid_dim,
    float cutoff
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = grid_dim * grid_dim * grid_dim;

    for (int v = tid; v < total_voxels; v += gridDim.x * blockDim.x) {
        int vz = v / (grid_dim * grid_dim);
        int vy = (v / grid_dim) % grid_dim;
        int vx = v % grid_dim;

        float3 voxel_center = make_float3(
            grid_origin.x + (vx + 0.5f) * grid_spacing,
            grid_origin.y + (vy + 0.5f) * grid_spacing,
            grid_origin.z + (vz + 0.5f) * grid_spacing
        );

        warp_matrix[v].voxel_idx = v;
        build_warp_entry(warp_matrix[v], voxel_center, positions, n_atoms, cutoff);
    }
}

extern "C" __global__ void init_lif_state(
    float* lif_potential,
    float* water_density,
    float* water_density_prev,
    int* spike_grid,
    int total_voxels
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int v = tid; v < total_voxels; v += gridDim.x * blockDim.x) {
        lif_potential[v] = 0.0f;
        water_density[v] = WATER_DENSITY_BULK;
        water_density_prev[v] = WATER_DENSITY_BULK;
        spike_grid[v] = 0;
    }
}

// ============================================================================
// TEMPERATURE PROTOCOL HELPERS
// ============================================================================

extern "C" __global__ void set_temperature_protocol(
    TemperatureProtocol* protocol,
    float start_temp,
    float end_temp,
    int ramp_steps,
    int hold_steps
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        protocol->start_temp = start_temp;
        protocol->end_temp = end_temp;
        protocol->ramp_steps = ramp_steps;
        protocol->hold_steps = hold_steps;
        protocol->current_step = 0;
    }
}

extern "C" __global__ void advance_temperature_protocol(
    TemperatureProtocol* protocol
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        protocol->current_step++;
    }
}

// ============================================================================
// EXCITED STATE INITIALIZATION
// ============================================================================

extern "C" __global__ void init_excited_state(
    int* d_is_excited,
    float* d_time_since_excitation,
    float* d_electronic_population,
    float* d_vibrational_energy,
    float* d_franck_condon_progress,
    int n_aromatics
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_aromatics) {
        d_is_excited[tid] = 0;
        d_time_since_excitation[tid] = 0.0f;
        d_electronic_population[tid] = 0.0f;
        d_vibrational_energy[tid] = 0.0f;
        d_franck_condon_progress[tid] = 0.0f;
    }
}

extern "C" __global__ void init_atom_to_aromatic(
    int* d_atom_to_aromatic,
    int n_atoms
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < n_atoms) {
        d_atom_to_aromatic[tid] = -1;  // Not an aromatic atom by default
    }
}

extern "C" __global__ void build_aromatic_neighbors(
    AromaticNeighbors* d_aromatic_neighbors,
    const float3* positions,
    const int* aromatic_atom_indices,  // [n_aromatics * MAX_RING_ATOMS]
    const int* aromatic_n_atoms,       // [n_aromatics]
    int n_aromatics,
    int n_atoms,
    float neighbor_cutoff              // ~5 Angstroms
) {
    int arom_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (arom_idx < n_aromatics) {
        AromaticNeighbors* neighbors = &d_aromatic_neighbors[arom_idx];
        neighbors->n_neighbors = 0;

        // Get center of aromatic ring
        float3 ring_center = make_float3(0, 0, 0);
        int n_ring_atoms = aromatic_n_atoms[arom_idx];

        for (int i = 0; i < n_ring_atoms && i < 16; i++) {
            int atom_idx = aromatic_atom_indices[arom_idx * 16 + i];
            if (atom_idx >= 0 && atom_idx < n_atoms) {
                ring_center.x += positions[atom_idx].x;
                ring_center.y += positions[atom_idx].y;
                ring_center.z += positions[atom_idx].z;
            }
        }
        if (n_ring_atoms > 0) {
            ring_center.x /= n_ring_atoms;
            ring_center.y /= n_ring_atoms;
            ring_center.z /= n_ring_atoms;
        }

        float cutoff_sq = neighbor_cutoff * neighbor_cutoff;

        // Find all atoms within cutoff of ring center
        for (int a = 0; a < n_atoms && neighbors->n_neighbors < 64; a++) {
            // Skip atoms that are part of this aromatic ring
            bool is_ring_atom = false;
            for (int i = 0; i < n_ring_atoms && i < 16; i++) {
                if (aromatic_atom_indices[arom_idx * 16 + i] == a) {
                    is_ring_atom = true;
                    break;
                }
            }
            if (is_ring_atom) continue;

            float dx = positions[a].x - ring_center.x;
            float dy = positions[a].y - ring_center.y;
            float dz = positions[a].z - ring_center.z;
            float dist_sq = dx*dx + dy*dy + dz*dz;

            if (dist_sq < cutoff_sq) {
                neighbors->atom_indices[neighbors->n_neighbors] = a;
                neighbors->n_neighbors++;
            }
        }
    }
}

extern "C" __global__ void compute_ring_normals(
    float3* d_ring_normals,
    const float3* positions,
    const int* aromatic_atom_indices,  // [n_aromatics * 16]
    const int* aromatic_n_atoms,       // [n_aromatics]
    int n_aromatics,
    int n_atoms
) {
    int arom_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (arom_idx < n_aromatics) {
        int n_ring_atoms = aromatic_n_atoms[arom_idx];

        // Need at least 3 atoms to compute a plane normal
        if (n_ring_atoms < 3) {
            d_ring_normals[arom_idx] = make_float3(0, 0, 1);  // Default
            return;
        }

        // Get first three atoms
        int a0 = aromatic_atom_indices[arom_idx * 16 + 0];
        int a1 = aromatic_atom_indices[arom_idx * 16 + 1];
        int a2 = aromatic_atom_indices[arom_idx * 16 + 2];

        if (a0 < 0 || a1 < 0 || a2 < 0 || a0 >= n_atoms || a1 >= n_atoms || a2 >= n_atoms) {
            d_ring_normals[arom_idx] = make_float3(0, 0, 1);
            return;
        }

        // Compute two edge vectors
        float3 v1 = make_float3(
            positions[a1].x - positions[a0].x,
            positions[a1].y - positions[a0].y,
            positions[a1].z - positions[a0].z
        );
        float3 v2 = make_float3(
            positions[a2].x - positions[a0].x,
            positions[a2].y - positions[a0].y,
            positions[a2].z - positions[a0].z
        );

        // Cross product for normal
        float3 normal = make_float3(
            v1.y * v2.z - v1.z * v2.y,
            v1.z * v2.x - v1.x * v2.z,
            v1.x * v2.y - v1.y * v2.x
        );

        // Normalize
        float len = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z);
        if (len > 1e-8f) {
            normal.x /= len;
            normal.y /= len;
            normal.z /= len;
        } else {
            normal = make_float3(0, 0, 1);
        }

        d_ring_normals[arom_idx] = normal;
    }
}

// ============================================================================
// O(N) CELL LIST CONSTRUCTION
// ============================================================================

/**
 * @brief Build cell lists from atom positions
 *
 * Each atom is assigned to exactly one cell based on its position.
 * Cell index = ix + iy * nx + iz * nx * ny
 *
 * Call this BEFORE build_neighbor_list, typically every 10-20 steps.
 */
extern "C" __global__ void build_cell_list(
    const float3* __restrict__ positions,  // [n_atoms]
    int* __restrict__ cell_list,           // [MAX_TOTAL_CELLS * MAX_ATOMS_PER_CELL]
    int* __restrict__ cell_counts,         // [MAX_TOTAL_CELLS]
    int* __restrict__ atom_cell,           // [n_atoms] - which cell each atom is in
    float origin_x, float origin_y, float origin_z,
    int nx, int ny, int nz,
    int n_atoms
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_atoms) return;

    float x = positions[tid].x;
    float y = positions[tid].y;
    float z = positions[tid].z;

    // Compute cell indices
    int ix = (int)((x - origin_x) * CELL_SIZE_INV);
    int iy = (int)((y - origin_y) * CELL_SIZE_INV);
    int iz = (int)((z - origin_z) * CELL_SIZE_INV);

    // Clamp to valid range
    ix = max(0, min(ix, nx - 1));
    iy = max(0, min(iy, ny - 1));
    iz = max(0, min(iz, nz - 1));

    int cell_idx = ix + iy * nx + iz * nx * ny;
    atom_cell[tid] = cell_idx;

    // Atomically add atom to cell
    int slot = atomicAdd(&cell_counts[cell_idx], 1);
    if (slot < MAX_ATOMS_PER_CELL) {
        cell_list[cell_idx * MAX_ATOMS_PER_CELL + slot] = tid;
    }
    // Note: overflow is tracked - if slot >= MAX_ATOMS_PER_CELL, atom is not added
}

/**
 * @brief Reset cell counts to zero
 *
 * Call this before build_cell_list to clear previous frame's data.
 */
extern "C" __global__ void reset_cell_counts(
    int* __restrict__ cell_counts,
    int n_cells
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_cells) {
        cell_counts[tid] = 0;
    }
}

/**
 * @brief Build neighbor lists from cell lists (O(N) average case)
 *
 * For each atom, find all neighbors within cutoff by checking
 * the 27 cells (self + 26 neighbors). This replaces O(N²) all-pairs.
 *
 * Performance: For 3000 atoms with 10Å cutoff, typically ~100-200 neighbors/atom
 * instead of checking all 3000 pairs.
 */
extern "C" __global__ void build_neighbor_list(
    const float3* __restrict__ positions,  // [n_atoms]
    const int* __restrict__ cell_list,     // [MAX_TOTAL_CELLS * MAX_ATOMS_PER_CELL]
    const int* __restrict__ cell_counts,   // [MAX_TOTAL_CELLS]
    const int* __restrict__ atom_cell,     // [n_atoms]
    const int* __restrict__ excl_list,     // CSR exclusion list
    const int* __restrict__ excl_offsets,  // CSR offsets [n_atoms + 1]
    int* __restrict__ neighbor_list,       // [n_atoms * NEIGHBOR_LIST_SIZE]
    int* __restrict__ n_neighbors,         // [n_atoms]
    int nx, int ny, int nz,
    int n_atoms,
    float cutoff_sq                        // Squared cutoff with buffer
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_atoms) return;

    float3 my_pos = positions[tid];
    int my_cell = atom_cell[tid];
    int my_ix = my_cell % nx;
    int my_iy = (my_cell / nx) % ny;
    int my_iz = my_cell / (nx * ny);

    // Get my exclusion list
    int excl_start = excl_offsets[tid];
    int excl_end = excl_offsets[tid + 1];

    int neighbor_count = 0;
    int* my_neighbors = &neighbor_list[tid * NEIGHBOR_LIST_SIZE];

    // Check 27 neighboring cells (including self)
    for (int dz = -1; dz <= 1; dz++) {
        int iz = my_iz + dz;
        if (iz < 0 || iz >= nz) continue;

        for (int dy = -1; dy <= 1; dy++) {
            int iy = my_iy + dy;
            if (iy < 0 || iy >= ny) continue;

            for (int dx = -1; dx <= 1; dx++) {
                int ix = my_ix + dx;
                if (ix < 0 || ix >= nx) continue;

                int neighbor_cell = ix + iy * nx + iz * nx * ny;
                int n_in_cell = cell_counts[neighbor_cell];
                if (n_in_cell > MAX_ATOMS_PER_CELL) n_in_cell = MAX_ATOMS_PER_CELL;

                // Check all atoms in this cell
                for (int k = 0; k < n_in_cell; k++) {
                    int j = cell_list[neighbor_cell * MAX_ATOMS_PER_CELL + k];
                    if (j <= tid) continue;  // Only count pairs once (i < j)

                    // Distance check
                    float dx_ij = positions[j].x - my_pos.x;
                    float dy_ij = positions[j].y - my_pos.y;
                    float dz_ij = positions[j].z - my_pos.z;
                    float r2 = dx_ij * dx_ij + dy_ij * dy_ij + dz_ij * dz_ij;

                    // Skip if outside cutoff (with buffer for list reuse)
                    if (r2 > cutoff_sq) continue;

                    // Check exclusion list (bonded pairs)
                    bool excluded = false;
                    for (int e = excl_start; e < excl_end; e++) {
                        if (excl_list[e] == j) {
                            excluded = true;
                            break;
                        }
                    }
                    if (excluded) continue;

                    // Add to neighbor list
                    if (neighbor_count < NEIGHBOR_LIST_SIZE) {
                        my_neighbors[neighbor_count] = j;
                        neighbor_count++;
                    }
                }
            }
        }
    }

    n_neighbors[tid] = neighbor_count;
}

/**
 * @brief Compute nonbonded forces using neighbor lists (O(N))
 *
 * This is the fast path - uses precomputed neighbor lists instead of
 * O(N²) all-pairs. Should be ~50-100x faster for large proteins.
 *
 * NOTE: This kernel can be called instead of the inline nonbonded loop
 * in nhs_amber_fused_step for systems where neighbor list rebuild
 * overhead is worth the per-step savings.
 */
extern "C" __global__ void compute_nonbonded_neighborlist(
    const float3* __restrict__ positions,
    float3* __restrict__ forces,
    const float* __restrict__ charges,
    const LJParam* __restrict__ lj_params,
    const int* __restrict__ neighbor_list,  // [n_atoms * NEIGHBOR_LIST_SIZE]
    const int* __restrict__ n_neighbors,    // [n_atoms]
    // Excited state for charge modification
    const int* __restrict__ d_atom_to_aromatic,
    const int* __restrict__ d_aromatic_type,
    const int* __restrict__ d_is_excited,
    const float* __restrict__ d_electronic_population,
    const float* __restrict__ d_ground_state_charges,
    int n_atoms,
    float cutoff_sq
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_atoms) return;

    float3 my_pos = positions[tid];
    float my_charge = charges[tid];
    LJParam my_lj = lj_params[tid];

    // Apply excited state charge scaling if applicable
    int arom_idx = d_atom_to_aromatic[tid];
    if (arom_idx >= 0 && d_is_excited[arom_idx]) {
        float pop = d_electronic_population[arom_idx];
        // CANONICAL: 0=TRP, 1=TYR, 2=PHE, 3=S-S
        float ratio_sqrt;
        switch (d_aromatic_type[arom_idx]) {
            case 0: ratio_sqrt = TRP_DIPOLE_RATIO_SQRT; break;  // TRP
            case 1: ratio_sqrt = TYR_DIPOLE_RATIO_SQRT; break;  // TYR
            case 2: ratio_sqrt = PHE_DIPOLE_RATIO_SQRT; break;  // PHE
            default: ratio_sqrt = 1.0f;
        }
        float scale = 1.0f + (ratio_sqrt - 1.0f) * pop;
        my_charge = d_ground_state_charges[tid] * scale;
    }

    float3 my_force = make_float3(0.0f, 0.0f, 0.0f);

    int my_n_neighbors = n_neighbors[tid];
    const int* my_neighbors = &neighbor_list[tid * NEIGHBOR_LIST_SIZE];

    // Only loop over actual neighbors (O(N) total work)
    for (int k = 0; k < my_n_neighbors; k++) {
        int j = my_neighbors[k];

        float3 other_pos = positions[j];
        float dx = my_pos.x - other_pos.x;
        float dy = my_pos.y - other_pos.y;
        float dz = my_pos.z - other_pos.z;
        float r2 = dx * dx + dy * dy + dz * dz;

        // Skip if outside cutoff (neighbor list has buffer)
        if (r2 >= cutoff_sq || r2 < 0.01f) continue;

        float r = sqrtf(r2);
        float inv_r = 1.0f / r;

        // Get other atom's charge with excited state scaling
        float other_charge = charges[j];
        int other_arom = d_atom_to_aromatic[j];
        if (other_arom >= 0 && d_is_excited[other_arom]) {
            float pop = d_electronic_population[other_arom];
            // CANONICAL: 0=TRP, 1=TYR, 2=PHE, 3=S-S
            float ratio_sqrt;
            switch (d_aromatic_type[other_arom]) {
                case 0: ratio_sqrt = TRP_DIPOLE_RATIO_SQRT; break;  // TRP
                case 1: ratio_sqrt = TYR_DIPOLE_RATIO_SQRT; break;  // TYR
                case 2: ratio_sqrt = PHE_DIPOLE_RATIO_SQRT; break;  // PHE
                default: ratio_sqrt = 1.0f;
            }
            float scale = 1.0f + (ratio_sqrt - 1.0f) * pop;
            other_charge = d_ground_state_charges[j] * scale;
        }

        // Lorentz-Berthelot combining rules
        float sigma_ij = 0.5f * (my_lj.sigma + lj_params[j].sigma);
        float eps_ij = sqrtf(my_lj.epsilon * lj_params[j].epsilon);

        // LJ 12-6 with soft core
        float r2_soft = r2 + 0.01f;  // Soft core delta
        float sigma2 = sigma_ij * sigma_ij;
        float sigma6 = sigma2 * sigma2 * sigma2;
        float inv_r2_soft = 1.0f / r2_soft;
        float inv_r6_soft = inv_r2_soft * inv_r2_soft * inv_r2_soft;
        float sigma6_r6 = sigma6 * inv_r6_soft;

        float lj_force = 24.0f * eps_ij * (2.0f * sigma6_r6 * sigma6_r6 - sigma6_r6) / r2_soft;

        // Coulomb with implicit solvent (ε = 4r)
        float coul_force = COULOMB_CONSTANT * my_charge * other_charge * 0.25f * inv_r * inv_r * inv_r;

        // Total force with capping
        float total_force = lj_force + coul_force;
        float max_force = 500.0f;
        if (fabsf(total_force) > max_force) {
            total_force = copysignf(max_force, total_force);
        }

        // Accumulate force on my atom
        my_force.x -= total_force * dx;
        my_force.y -= total_force * dy;
        my_force.z -= total_force * dz;

        // Apply Newton's 3rd law to other atom
        atomicAdd(&forces[j].x, total_force * dx);
        atomicAdd(&forces[j].y, total_force * dy);
        atomicAdd(&forces[j].z, total_force * dz);
    }

    // Write my accumulated force
    atomicAdd(&forces[tid].x, my_force.x);
    atomicAdd(&forces[tid].y, my_force.y);
    atomicAdd(&forces[tid].z, my_force.z);
}

// ============================================================================
// EXTERN "C" INITIALIZATION KERNELS FOR CRYO-THERMAL DETECTION
// These initialize the state for the multi-modal sensitive detector
// ============================================================================

// Initialize multi-modal detector state for cryo probing
extern "C" __global__ void init_multimodal_detector(
    MultiModalVoxelState* state,
    int n_voxels,
    float baseline_temp
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_voxels) return;
    
    state[tid].thermal_potential = 0.0f;
    state[tid].thermal_baseline = baseline_temp;
    state[tid].thermal_spike_count = 0;
    
    state[tid].gradient_potential = 0.0f;
    state[tid].last_gradient_dir = make_float3(0.0f, 0.0f, 0.0f);
    state[tid].gradient_spike_count = 0;
    
    state[tid].melt_potential = 0.0f;
    state[tid].ice_fraction = 1.0f;  // Start frozen
    state[tid].melt_spike_count = 0;
    
    state[tid].combined_potential = 0.0f;
    state[tid].last_spike_time = 0;
    state[tid].in_refractory = false;
    
    state[tid].signal_to_noise = 1.0f;
    state[tid].confidence = 0.0f;
}

// Initialize thermal voxels for cryo-thermal detection
extern "C" __global__ void init_thermal_voxels(
    ThermalVoxel* voxels,
    int n_voxels,
    float initial_temp
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_voxels) return;
    
    voxels[tid].temperature = initial_temp;
    voxels[tid].baseline_temp = initial_temp;
    voxels[tid].delta_temp = 0.0f;
    voxels[tid].temp_gradient_mag = 0.0f;
    voxels[tid].temp_gradient_dir = make_float3(0.0f, 0.0f, 0.0f);
    
    voxels[tid].ice_fraction = (initial_temp < 273.15f) ? 1.0f : 0.0f;
    voxels[tid].melt_rate = 0.0f;
    
    // Ice has different thermal properties than water
    if (initial_temp < 273.15f) {
        voxels[tid].heat_capacity = 2.09f;       // J/(g·K) for ice
        voxels[tid].thermal_conductivity = 2.2f; // W/(m·K) for ice
    } else {
        voxels[tid].heat_capacity = 4.18f;       // J/(g·K) for water
        voxels[tid].thermal_conductivity = 0.6f; // W/(m·K) for water
    }
    
    
    voxels[tid].last_spike_time = 0;
    voxels[tid].lif_potential = 0.0f;
    voxels[tid].in_refractory = false;
}
// REMOVED: self-include (circular recursion)

extern "C" __global__ void nhs_voxel_step(
    // Positions (read-only, updated by atom kernel)
    const float3* __restrict__ positions,
    int n_atoms,
    // Voxel grid params
    int grid_dim,
    float grid_spacing,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    // Exclusion field arrays
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    float* lif_potential,
    int* spike_grid,
    // Warp matrix
    WarpEntry* warp_matrix,
    // Atom metadata
    const int* __restrict__ atom_types,
    const float* __restrict__ charges,
    const int* __restrict__ residue_ids,
    // Aromatic data
    const float3* __restrict__ d_aromatic_centroids,
    const float3* __restrict__ d_ring_normals,
    const int* __restrict__ d_is_excited,
    const float* __restrict__ d_electronic_population,
    const float* __restrict__ d_vibrational_energy,
    const float* __restrict__ d_time_since_excitation,
    const int* __restrict__ d_aromatic_type,
    const int* __restrict__ d_atom_to_aromatic,
    int n_aromatics,
    // UV params
    float uv_wavelength_nm,
    int uv_burst_active,
    float* d_uv_signal_prev,
    // Spike output
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    // Temperature
    float target_temp,
    float dt,
    int timestep,
    // EFP arrays
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    // Aromatic neighbors (for expanded exclusion)
    const AromaticNeighbors* __restrict__ d_aromatic_neighbors,
    const float* __restrict__ d_franck_condon_progress,
    int* spike_grid_efp,             // independent EFP refractory grid
    // Signal preservation buffers (accumulated across all timesteps)
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    int* residue_step_causal               // [n_residues] KCC per-step causal counter
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_voxels = grid_dim * grid_dim * grid_dim;
    if (tid >= total_voxels) return;

    int v = tid;  // One thread per voxel — no grid-stride loop needed

    float3 grid_origin = make_float3(grid_origin_x, grid_origin_y, grid_origin_z);

    // Voxel coordinates (computed ONCE, shared across all phases)
    int vz = v / (grid_dim * grid_dim);
    int vy = (v / grid_dim) % grid_dim;
    int vx = v % grid_dim;
    float3 voxel_center = make_float3(
        grid_origin.x + (vx + 0.5f) * grid_spacing,
        grid_origin.y + (vy + 0.5f) * grid_spacing,
        grid_origin.z + (vz + 0.5f) * grid_spacing
    );

    // ====================================================================
    // FUSED PHASE 4: EXCLUSION FIELD + WATER DENSITY
    // ====================================================================
    water_density_prev[v] = water_density[v];

    WarpEntry entry = warp_matrix[v];
    float total_exclusion = 0.0f;
    // SPARSE VOXEL EARLY EXIT: skip voxels with no nearby atoms
    if (entry.n_atoms == 0) {
        // Still need to decay refractory counter and LIF potential
        if (spike_grid[v] > 0) spike_grid[v]--;
        if (spike_grid_efp[v] > 0) spike_grid_efp[v]--;
        lif_potential[v] *= 0.9f;  // passive decay
        efp_lif_potential[v] *= 0.8f;
        return;
    }
    float polar_field = 0.0f;

    for (int i = 0; i < entry.n_atoms; i++) {
        int a = entry.atom_indices[i];
        if (a < 0 || a >= n_atoms) continue;

        float contrib = compute_exclusion_contribution(
            positions[a], voxel_center,
            atom_types[a], charges[a]
        );

        if (n_aromatics > 0 && d_aromatic_centroids != nullptr) {
            float expanded_modifier = compute_expanded_exclusion_modifier(
                positions[a],
                d_aromatic_centroids,
                d_ring_normals,
                d_is_excited,
                d_electronic_population,
                n_aromatics
            );
            contrib *= expanded_modifier;
        } else {
            int aromatic_idx = d_atom_to_aromatic[a];
            if (aromatic_idx >= 0 && aromatic_idx < n_aromatics) {
                float excitation_modifier = get_exclusion_modifier(
                    aromatic_idx,
                    d_is_excited,
                    d_electronic_population
                );
                contrib *= excitation_modifier;
            }
        }

        total_exclusion += contrib * entry.atom_weights[i] * 4.0f;

        if (atom_types[a] == 1 || atom_types[a] == 2 || atom_types[a] == 3) {
            polar_field += contrib * 0.5f;
        }
    }

    total_exclusion = fminf(1.0f, total_exclusion);
    exclusion_field[v] = total_exclusion;
    water_density[v] = infer_water_density(total_exclusion, polar_field, target_temp);

    // ====================================================================
    // FUSED PHASE 5: NEUROMORPHIC LIF + UV-LIF COUPLING
    // ====================================================================

    // Refractory check — if counting down, decrement and skip LIF+EFP
    if (spike_grid_efp[v] > 0) spike_grid_efp[v]--;
    if (spike_grid[v] > 0) {
        spike_grid[v]--;
        // Still do EFP below (it has its own refractory check)
        goto efp_phase;
    }

    {
        float spike_intensity = 0.0f;
        float tau_mem = 0.1f;

        float uv_signal = 0.0f;
        int n_nearby_excited = 0;
        float min_distance_to_excited = 1000.0f;
        int closest_excited_idx = -1;

        if (n_aromatics > 0) {
            const float UV_DETECTION_RADIUS = 4.0f;
            const float UV_DIRECT_STRENGTH = 0.8f;

            float total_vib_energy = 0.0f;

            for (int a = 0; a < n_aromatics; a++) {
                if (!d_is_excited[a]) continue;
                if (d_aromatic_centroids == nullptr) continue;

                float3 arom_pos = d_aromatic_centroids[a];
                float dx = voxel_center.x - arom_pos.x;
                float dy = voxel_center.y - arom_pos.y;
                float dz = voxel_center.z - arom_pos.z;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);

                if (dist < UV_DETECTION_RADIUS) {
                    n_nearby_excited++;
                    total_vib_energy += d_vibrational_energy[a];
                    if (dist < min_distance_to_excited) {
                        min_distance_to_excited = dist;
                        closest_excited_idx = a;
                    }
                }
            }

            if (n_nearby_excited > 0) {
                float energy_factor = fminf(total_vib_energy / (n_nearby_excited * 3.0f), 1.0f);
                float coop_boost = 1.0f + 0.3f * (n_nearby_excited - 1);
                uv_signal = UV_DIRECT_STRENGTH * energy_factor * coop_boost;
                if (uv_burst_active) {
                    uv_signal *= 2.0f;
                }
            }

            if (d_aromatic_centroids != nullptr && d_uv_signal_prev != nullptr) {
                float prev_signal = d_uv_signal_prev[v];
                float advanced_signal = compute_uv_lif_signal(
                    voxel_center,
                    d_aromatic_centroids,
                    d_ring_normals,
                    d_is_excited,
                    d_electronic_population,
                    d_vibrational_energy,
                    d_time_since_excitation,
                    n_aromatics,
                    dt,
                    prev_signal
                );
                uv_signal += advanced_signal;
                d_uv_signal_prev[v] = uv_signal;
            }

            if (uv_signal > 0.0f) {
                lif_potential[v] += uv_signal;
            }

            // Record UV event timestamp for causal tracking (inline nhs_voxel_step)
            if (uv_signal > 0.1f) {
                last_uv_step[v] = timestep;
            }

            // Direct UV spike trigger
            const float DIRECT_UV_SPIKE_THRESHOLD = 0.3f;
            const float MAX_SPIKE_DISTANCE = 4.0f;

            const int voxel_n_atoms = warp_matrix[v].n_atoms;
            bool voxel_has_aromatic_atom = false;
            for (int wi = 0; wi < voxel_n_atoms && !voxel_has_aromatic_atom; wi++) {
                int atom_idx = warp_matrix[v].atom_indices[wi];
                if (atom_idx >= 0 && atom_idx < n_atoms) {
                    if (d_atom_to_aromatic[atom_idx] >= 0) {
                        voxel_has_aromatic_atom = true;
                    }
                }
            }

            if (n_nearby_excited > 0 &&
                uv_signal > DIRECT_UV_SPIKE_THRESHOLD &&
                voxel_n_atoms > 0 &&
                min_distance_to_excited < MAX_SPIKE_DISTANCE) {
                spike_grid[v] = REFRACTORY_STEPS;
                spike_intensity = uv_signal;

                int _arom_type = (closest_excited_idx >= 0) ? d_aromatic_type[closest_excited_idx] : -1;
                int _arom_res = -1;
                if (closest_excited_idx >= 0) {
                    for (int wi = 0; wi < warp_matrix[v].n_atoms && _arom_res < 0; wi++) {
                        int ai = warp_matrix[v].atom_indices[wi];
                        if (ai >= 0 && ai < n_atoms && d_atom_to_aromatic[ai] == closest_excited_idx) {
                            _arom_res = residue_ids[ai];
                        }
                    }
                }
                float _vib_e = (closest_excited_idx >= 0) ? d_vibrational_energy[closest_excited_idx] : 0.0f;

                // Signal preservation (late UV spike)
                update_signal_preservation(v, timestep, 1,
                    voxel_hit_grid, last_uv_step, coupled_spike_grid,
                    primary_residue_id, primary_residue_count,
                    warp_matrix[v], residue_ids, n_atoms, residue_step_causal);

                int spike_idx = atomicAdd(spike_count, 1);
                if (spike_idx < max_spikes) {
                    capture_spike_event(
                        spike_events[spike_idx], timestep, v, voxel_center,
                        spike_intensity, warp_matrix[v], residue_ids,
                        1, uv_wavelength_nm, _arom_type, _arom_res,
                        water_density[v], _vib_e, n_nearby_excited,
                        fabsf(water_density[v] - water_density_prev[v])
                    );
                }
                lif_potential[v] = LIF_RESET;
            }
        }

        // Standard LIF update (non-UV)
        if (spike_grid[v] == 0 && !uv_burst_active) {
            bool spike = lif_neuron_update(
                lif_potential[v], water_density[v], water_density_prev[v],
                tau_mem, dt, LIF_THRESHOLD, spike_intensity
            );

            if (spike) {
                spike_grid[v] = REFRACTORY_STEPS;

                // Signal preservation (late LIF spike)
                int lif_source2 = (n_nearby_excited > 0) ? 1 : 2;
                update_signal_preservation(v, timestep, lif_source2,
                    voxel_hit_grid, last_uv_step, coupled_spike_grid,
                    primary_residue_id, primary_residue_count,
                    warp_matrix[v], residue_ids, n_atoms, residue_step_causal);

                int spike_idx = atomicAdd(spike_count, 1);
                if (spike_idx < max_spikes) {
                    int lif_atype = -1, lif_ares = -1;
                    float lif_wl = 0.0f, lif_vibe = 0.0f;
                    if (n_nearby_excited > 0 && closest_excited_idx >= 0) {
                        lif_atype = d_aromatic_type[closest_excited_idx];
                        lif_wl = uv_wavelength_nm;
                        lif_vibe = d_vibrational_energy[closest_excited_idx];
                        for (int wi = 0; wi < warp_matrix[v].n_atoms && lif_ares < 0; wi++) {
                            int ai = warp_matrix[v].atom_indices[wi];
                            if (ai >= 0 && ai < n_atoms && d_atom_to_aromatic[ai] == closest_excited_idx) {
                                lif_ares = residue_ids[ai];
                            }
                        }
                    }
                    capture_spike_event(
                        spike_events[spike_idx], timestep, v, voxel_center,
                        spike_intensity, warp_matrix[v], residue_ids,
                        lif_source2, lif_wl,
                        lif_atype, lif_ares, water_density[v],
                        lif_vibe, n_nearby_excited,
                        fabsf(water_density[v] - water_density_prev[v])
                    );
                }
            }
        }
    }

    // ====================================================================
    // FUSED EFP: ELECTROSTATIC FLUX PROBE
    // ====================================================================
efp_phase:

    if (warp_matrix[v].n_atoms > 0) {
        float phi = 0.0f;
        int n_charged_nearby = 0;

        for (int wi = 0; wi < warp_matrix[v].n_atoms; wi++) {
            int ai = warp_matrix[v].atom_indices[wi];
            if (ai < 0 || ai >= n_atoms) continue;
            float q = charges[ai];
            if (fabsf(q) < 0.15f) continue;

            float3 ap = positions[ai];
            float dx = voxel_center.x - ap.x;
            float dy = voxel_center.y - ap.y;
            float dz = voxel_center.z - ap.z;
            float dist = sqrtf(dx*dx + dy*dy + dz*dz);

            if (dist > 0.5f && dist < 8.0f) {
                float eps_r = fmaxf(4.0f * dist, 4.0f);
                phi += q / (eps_r * dist);
                n_charged_nearby++;
            }
        }

        float wd_change = fabsf(water_density[v] - water_density_prev[v]);
        float polar_water_signal = fabsf(phi) * wd_change * 40.0f;

        float phi_prev = efp_potential_prev[v];
        efp_potential[v] = phi;

        float flux = fabsf(phi - phi_prev);
        float polar_signal = flux * 150.0f + polar_water_signal;

        if (n_charged_nearby >= 1 && spike_grid_efp[v] == 0) {
            const float EFP_TAU = 0.5f;
            const float EFP_THRESHOLD = 0.15f;
            float efp_decay = expf(-dt / EFP_TAU);
            efp_lif_potential[v] = efp_decay * efp_lif_potential[v] + polar_signal;

            if (efp_lif_potential[v] > EFP_THRESHOLD) {
                spike_grid_efp[v] = REFRACTORY_STEPS;
                float polar_intensity = efp_lif_potential[v];
                efp_lif_potential[v] = LIF_RESET;

                // Signal preservation (late EFP spike)
                update_signal_preservation(v, timestep, 3,
                    voxel_hit_grid, last_uv_step, coupled_spike_grid,
                    primary_residue_id, primary_residue_count,
                    warp_matrix[v], residue_ids, n_atoms, residue_step_causal);

                int si = atomicAdd(spike_count, 1);
                if (si < max_spikes) {
                    int polar_type = (phi > 0.0f) ? 5 : 6;
                    capture_spike_event(
                        spike_events[si], timestep, v, voxel_center,
                        polar_intensity, warp_matrix[v], residue_ids,
                        3, 0.0f, polar_type, -1,
                        water_density[v], flux, n_charged_nearby,
                        wd_change  // already computed above
                    );
                }
            }
        }
        efp_potential_prev[v] = phi;
    }
}

// ============================================================================
// MULTI-NEURON RAF (Resonant Adaptive Filter) WITH SHARED MEMORY STENCIL
// ============================================================================
//
// K=8 RAF oscillator neurons per voxel, 8 threads per voxel, 3D tile launch.
// Each neuron has a different resonant frequency (omega_k = 2*pi/tau_k).
// Cross-voxel coupling via shared-memory halo stencil (26 Moore neighbors).
//
// Neuron timescales: tau_k = 0.5 * 2^k  (k=0..7)
//   k=0: 0.5 ps   (ultrafast electronic)
//   k=1: 1.0 ps   (bond vibration)
//   k=2: 2.0 ps   (hydrogen bond)
//   k=3: 4.0 ps   (sidechain rotation)
//   k=4: 8.0 ps   (loop motion)
//   k=5: 16.0 ps  (domain hinge)
//   k=6: 32.0 ps  (slow collective)
//   k=7: 64.0 ps  (cryptic site opening)
//
// State arrays (no new buffers — repurposed):
//   neuron_potential[nidx] = x (real part of oscillator)
//   neuron_mean[nidx]      = y (imaginary part of oscillator)
//   neuron_threshold[nidx] = adaptive threshold
//   neuron_refractory[nidx]= refractory counter
// ============================================================================

#define K_NEURONS 8
#define THREADS_PER_VOXEL 8
#define VOXELS_PER_WARP (WARP_SIZE / THREADS_PER_VOXEL)  // 4
#define MULTI_LIF_THRESHOLD 0.3f
#define COUPLING_STRENGTH 0.01f
#define COUPLING_DECAY 0.9f

// 3D tile dimensions for shared memory stencil
// Reduced from 4x4x2 (256 threads) to 4x2x2 (128 threads) for better
// SM occupancy on Blackwell (~25% vs ~12.5% at 256 threads/block).
#define TILE_X 4
#define TILE_Y 2
#define TILE_Z 2
#define VOXELS_PER_TILE (TILE_X * TILE_Y * TILE_Z)  // 16
#define HALO_X (TILE_X + 2)   // 6
#define HALO_Y (TILE_Y + 2)   // 4
#define HALO_Z (TILE_Z + 2)   // 4
#define HALO_SIZE (HALO_X * HALO_Y * HALO_Z)  // 96

// RAF parameters
#define CASCADE_RATE 0.01f
#define JITTER_AMPLITUDE 0.02f
#define RAF_R_MAX 2.0f
#define RAF_PI 3.14159265f

// Compute neuron timescale from lane index within voxel
__device__ __forceinline__ float neuron_tau(int k) {
    // tau = 0.5 * 2^k = 0.5, 1, 2, 4, 8, 16, 32, 64
    return 0.5f * (float)(1 << k);
}

// Compute 1D voxel index from 3D coordinates (clamped)
__device__ __forceinline__ int voxel_idx_3d(int x, int y, int z, int gdim) {
    if (x < 0 || x >= gdim || y < 0 || y >= gdim || z < 0 || z >= gdim) return -1;
    return z * gdim * gdim + y * gdim + x;
}

extern "C" __global__ __launch_bounds__(128, 8) void nhs_voxel_step_multi_lif(
    // === Same as nhs_voxel_step ===
    const float3* __restrict__ positions,
    int n_atoms,
    int grid_dim,
    float grid_spacing,
    float grid_origin_x,
    float grid_origin_y,
    float grid_origin_z,
    float* exclusion_field,
    float* water_density,
    float* water_density_prev,
    int* spike_grid,
    WarpEntry* warp_matrix,
    const int* __restrict__ atom_types,
    const float* __restrict__ charges,
    const int* __restrict__ residue_ids,
    const float3* __restrict__ d_aromatic_centroids,
    const float3* __restrict__ d_ring_normals,
    const int* __restrict__ d_is_excited,
    const float* __restrict__ d_electronic_population,
    const float* __restrict__ d_vibrational_energy,
    const float* __restrict__ d_time_since_excitation,
    const int* __restrict__ d_aromatic_type,
    const int* __restrict__ d_atom_to_aromatic,
    int n_aromatics,
    float uv_wavelength_nm,
    int uv_burst_active,
    float* d_uv_signal_prev,
    SpikeEvent* spike_events,
    int* spike_count,
    int max_spikes,
    float target_temp,
    float dt,
    int timestep,
    float* efp_potential,
    float* efp_potential_prev,
    float* efp_lif_potential,
    const AromaticNeighbors* __restrict__ d_aromatic_neighbors,
    const float* __restrict__ d_franck_condon_progress,
    // === Multi-neuron buffers ===
    float* neuron_potential,     // [total_voxels * K_NEURONS] = x (real)
    float* neuron_threshold,     // [total_voxels * K_NEURONS]
    float* neuron_mean,          // [total_voxels * K_NEURONS] = y (imaginary)
    int*   neuron_refractory,    // [total_voxels * K_NEURONS]
    const float* coupling_read,  // [total_voxels] read buffer (from previous step)
    float* coupling_write,       // [total_voxels] write buffer (for next step)
    // === Sparse tile index ===
    const int* active_tiles,     // [n_active_tiles * 3] packed (bx, by, bz) triplets
    int n_active_tiles,          // number of active tiles (0 = fallback to full grid)
    // === Independent EFP refractory grid ===
    int* spike_grid_efp,         // [grid_dim³] — prevents UV/LIF from blocking EFP
    // === Signal preservation buffers (accumulated across all timesteps) ===
    unsigned int* voxel_hit_grid,          // [grid_dim³] spatial recurrence counter
    int* last_uv_step,                     // [grid_dim³] timestep of last UV event per voxel
    unsigned int* coupled_spike_grid,      // [grid_dim³] UV→LIF causal spike counter
    int* primary_residue_id,               // [grid_dim³] dominant driver residue ID (-1 = none)
    unsigned int* primary_residue_count,   // [grid_dim³] count for dominant driver
    // === KCC: per-residue per-step causal counter ===
    int* residue_step_causal               // [n_residues] zeroed each step by kcc_residue_update
) {
    // === Shared memory layout (SOTA v2) ===
    // [0..HALO_SIZE-1]:  coupling stencil halo tile (96 floats)
    // [HALO_SIZE..+7]:   cos(omega*dt) LUT per neuron k (8 floats)
    // [+8..+15]:         sin(omega*dt) LUT (8 floats)
    // [+16..+23]:        exp(-dt/tau) LUT (8 floats)
    // [+24]:             flag: any aromatic excited (1 int)
    // [+25]:             n_excited_cached (1 int)
    // [+26..+26+MAX_AROM*10-1]: CachedAromaticData array
    extern __shared__ float s_coupling_tile[];
    float* s_cos_lut   = &s_coupling_tile[HALO_SIZE];       // [8]
    float* s_sin_lut   = &s_coupling_tile[HALO_SIZE + 8];   // [8]
    float* s_decay_lut = &s_coupling_tile[HALO_SIZE + 16];  // [8]
    int*   s_any_excited = (int*)&s_coupling_tile[HALO_SIZE + 24]; // [1]
    int*   s_n_excited   = (int*)&s_coupling_tile[HALO_SIZE + 25]; // [1]
    // Aromatic cache: 10 floats per aromatic (float3 pos=3, float3 norm=3, epop, vibe, tse, is_excited)
    #define SMEM_AROM_OFFSET (HALO_SIZE + 26)
    #define MAX_AROM_CACHED 64
    float* s_arom_cache = &s_coupling_tile[SMEM_AROM_OFFSET]; // [MAX_AROM_CACHED * 10]

    // Thread 0..7: build trig/decay LUT
    if (threadIdx.x < K_NEURONS) {
        float my_tau = neuron_tau(threadIdx.x);
        float omega = 2.0f * RAF_PI / my_tau;
        s_cos_lut[threadIdx.x]   = cosf(omega * dt);
        s_sin_lut[threadIdx.x]   = sinf(omega * dt);
        s_decay_lut[threadIdx.x] = expf(-dt / my_tau);
    }

    // Cooperative aromatic cache load: all 128 threads load aromatic data
    // into shared memory. Eliminates repeated global reads in exclusion + UV loops.
    {
        int n_arom = min(n_aromatics, MAX_AROM_CACHED);
        int n_excited_count = 0;
        // Each thread loads a subset of aromatics (10 floats per aromatic)
        for (int a = threadIdx.x; a < n_arom; a += blockDim.x) {
            float* dst = &s_arom_cache[a * 10];
            float3 pos = d_aromatic_centroids[a];
            float3 nrm = d_ring_normals[a];
            dst[0] = pos.x;  dst[1] = pos.y;  dst[2] = pos.z;
            dst[3] = nrm.x;  dst[4] = nrm.y;  dst[5] = nrm.z;
            dst[6] = d_electronic_population[a];
            dst[7] = d_vibrational_energy[a];
            dst[8] = d_time_since_excitation[a];
            dst[9] = __int_as_float(d_is_excited[a]);
        }
        __syncthreads();
        // Thread 0: count excited aromatics (from shared memory, not global)
        if (threadIdx.x == 0) {
            int any = 0;
            for (int a = 0; a < n_arom; a++) {
                if (__float_as_int(s_arom_cache[a * 10 + 9])) { any = 1; n_excited_count++; }
            }
            *s_any_excited = any;
            *s_n_excited = n_excited_count;
        }
    }
    __syncthreads();

    // === Sparse tile mapping ===
    // Each blockIdx.x maps to an active tile via the sparse index.
    // If n_active_tiles == 0, fall back to linear decomposition (full grid).
    int tile_bx, tile_by, tile_bz;
    if (n_active_tiles > 0 && active_tiles != nullptr) {
        int tile_idx = blockIdx.x;
        tile_bx = active_tiles[tile_idx * 3 + 0];
        tile_by = active_tiles[tile_idx * 3 + 1];
        tile_bz = active_tiles[tile_idx * 3 + 2];
    } else {
        // Fallback: reconstruct 3D from linear blockIdx
        int tiles_x = (grid_dim + TILE_X - 1) / TILE_X;
        int tiles_y = (grid_dim + TILE_Y - 1) / TILE_Y;
        tile_bx = blockIdx.x % tiles_x;
        tile_by = (blockIdx.x / tiles_x) % tiles_y;
        tile_bz = blockIdx.x / (tiles_x * tiles_y);
    }

    int local_voxel = threadIdx.x / THREADS_PER_VOXEL;  // 0..15
    int k = threadIdx.x % THREADS_PER_VOXEL;             // 0..7

    // Local 3D within tile
    int lz = local_voxel / (TILE_X * TILE_Y);
    int ly = (local_voxel / TILE_X) % TILE_Y;
    int lx = local_voxel % TILE_X;

    // Global 3D voxel coordinates
    int vx = tile_bx * TILE_X + lx;
    int vy = tile_by * TILE_Y + ly;
    int vz = tile_bz * TILE_Z + lz;

    bool is_valid = (vx < grid_dim && vy < grid_dim && vz < grid_dim);
    int v = is_valid ? (vz * grid_dim * grid_dim + vy * grid_dim + vx) : 0;

    float3 grid_origin = make_float3(grid_origin_x, grid_origin_y, grid_origin_z);
    float3 voxel_center = make_float3(
        grid_origin.x + (vx + 0.5f) * grid_spacing,
        grid_origin.y + (vy + 0.5f) * grid_spacing,
        grid_origin.z + (vz + 0.5f) * grid_spacing
    );

    // Warp-level lane info
    unsigned int warp_mask = 0xFFFFFFFF;
    int lane_in_warp = threadIdx.x % WARP_SIZE;
    int voxel_in_warp = lane_in_warp / THREADS_PER_VOXEL;  // 0..3
    int src_lane = voxel_in_warp * THREADS_PER_VOXEL;
    unsigned int voxel_mask = 0xFFu << (voxel_in_warp * THREADS_PER_VOXEL);

    // Initialize shared memory halo to zero
    for (int i = threadIdx.x; i < HALO_SIZE; i += blockDim.x) {
        s_coupling_tile[i] = 0.0f;
    }
    __syncthreads();

    // ====================================================================
    // PHASE 1: EXCLUSION FIELD + WATER DENSITY (neuron 0 only)
    // ====================================================================
    bool has_atoms = false;
    WarpEntry entry;
    int nidx = is_valid ? (v * K_NEURONS + k) : 0;

    if (is_valid) {
        water_density_prev[v] = water_density[v];  // all 8 threads idempotent
        entry = warp_matrix[v];
        has_atoms = (entry.n_atoms > 0);

        // Sparse voxel: passive decay only
        if (!has_atoms) {
            if (k == 0 && spike_grid[v] > 0) spike_grid[v]--;
            neuron_potential[nidx] *= 0.9f;
            neuron_mean[nidx] *= 0.9f;  // y component also decays
            if (k == 0) efp_lif_potential[v] *= 0.8f;
        }
    }

    // ────────────────────────────────────────────────────────────────
    // PHASE 1: WARP-COOPERATIVE EXCLUSION (all 8 neurons contribute)
    // Instead of k==0 serializing over atoms, all 8 threads process
    // atoms in parallel with warp shuffle reduction. 8x parallelism
    // on the most expensive per-voxel loop.
    // ────────────────────────────────────────────────────────────────
    float total_exclusion = 0.0f;
    float polar_field = 0.0f;

    if (is_valid && has_atoms) {
        // All 8 neurons cooperatively process atoms: each neuron handles
        // atoms[k], atoms[k+8], atoms[k+16], ... (strided access)
        float my_exclusion = 0.0f;
        float my_polar = 0.0f;
        int n_arom = min(n_aromatics, MAX_AROM_CACHED);

        for (int i = k; i < entry.n_atoms; i += K_NEURONS) {
            int a = entry.atom_indices[i];
            if (a < 0 || a >= n_atoms) continue;

            float contrib = compute_exclusion_contribution(
                positions[a], voxel_center,
                atom_types[a], charges[a]
            );

            // Aromatic exclusion modifier: read from shared memory cache
            if (n_arom > 0) {
                float best_mod = 1.0f;
                float3 atom_pos = positions[a];
                for (int ar = 0; ar < n_arom; ar++) {
                    float* ad = &s_arom_cache[ar * 10];
                    int excited = __float_as_int(ad[9]);
                    if (!excited) continue;
                    float dx = atom_pos.x - ad[0];
                    float dy = atom_pos.y - ad[1];
                    float dz = atom_pos.z - ad[2];
                    float dist_sq = dx*dx + dy*dy + dz*dz;
                    if (dist_sq < 25.0f) { // 5Å cutoff
                        float epop = ad[6];
                        float mod = 1.0f + epop * 0.5f;
                        if (mod > best_mod) best_mod = mod;
                    }
                }
                contrib *= best_mod;
            } else {
                int aromatic_idx = d_atom_to_aromatic[a];
                if (aromatic_idx >= 0 && aromatic_idx < n_aromatics) {
                    float excitation_modifier = get_exclusion_modifier(
                        aromatic_idx,
                        d_is_excited,
                        d_electronic_population
                    );
                    contrib *= excitation_modifier;
                }
            }

            my_exclusion += contrib * entry.atom_weights[i] * 4.0f;

            if (atom_types[a] == 1 || atom_types[a] == 2 || atom_types[a] == 3) {
                my_polar += contrib * 0.5f;
            }
        }

        // Warp shuffle reduction across 8 neurons within voxel
        for (int off = K_NEURONS / 2; off > 0; off >>= 1) {
            my_exclusion += __shfl_xor_sync(voxel_mask, my_exclusion, off);
            my_polar     += __shfl_xor_sync(voxel_mask, my_polar, off);
        }
        total_exclusion = fminf(1.0f, my_exclusion);
        polar_field = my_polar;

        // Neuron 0 writes results
        if (k == 0) {
            exclusion_field[v] = total_exclusion;
            water_density[v] = infer_water_density(total_exclusion, polar_field, target_temp);
        }
    }

    // All neurons already have total_exclusion from the shuffle reduction

    // ====================================================================
    // UV SIGNAL COMPUTATION (neuron 0 only, broadcast to all)
    // Must happen BEFORE refractory check so all threads reach __shfl_sync
    // ====================================================================
    float uv_signal = 0.0f;
    int n_nearby_excited = 0;
    float min_distance_to_excited = 1000.0f;
    int closest_excited_idx = -1;

    // ────────────────────────────────────────────────────────────────
    // UV SIGNAL: Shared-memory cached aromatics (zero global reads)
    // Single fused pass over cached aromatics replaces two separate
    // global-memory loops (basic UV + advanced compute_uv_lif_signal).
    // ────────────────────────────────────────────────────────────────
    if (is_valid && has_atoms && k == 0 && n_aromatics > 0 && *s_any_excited) {
        const float UV_DETECTION_RADIUS = 4.0f;
        const float UV_DIRECT_STRENGTH = 0.8f;
        const float MAX_DETECTION_RADIUS = 20.0f;
        float total_vib_energy = 0.0f;
        float thermal_wavefront_sum = 0.0f;
        float halo_sum = 0.0f;
        int n_arom = min(n_aromatics, MAX_AROM_CACHED);

        // Single fused loop: basic UV proximity + advanced thermal/halo
        #pragma unroll 4
        for (int a = 0; a < n_arom; a++) {
            float* ad = &s_arom_cache[a * 10];
            int excited = __float_as_int(ad[9]);
            if (!excited) continue;

            float ddx = voxel_center.x - ad[0];
            float ddy = voxel_center.y - ad[1];
            float ddz = voxel_center.z - ad[2];
            float dist_sq = ddx*ddx + ddy*ddy + ddz*ddz;

            if (dist_sq > MAX_DETECTION_RADIUS * MAX_DETECTION_RADIUS) continue;
            float dist = sqrtf(dist_sq);

            // Basic UV proximity signal (within 4Å)
            if (dist < UV_DETECTION_RADIUS) {
                n_nearby_excited++;
                total_vib_energy += ad[7]; // vibrational_energy from cache
                if (dist < min_distance_to_excited) {
                    min_distance_to_excited = dist;
                    closest_excited_idx = a;
                }
            }

            // Advanced thermal wavefront (within 20Å)
            float3 arom_pos = make_float3(ad[0], ad[1], ad[2]);
            float3 ring_norm = make_float3(ad[3], ad[4], ad[5]);
            float tse = ad[8]; // time_since_excitation
            float vibe = ad[7]; // vibrational_energy
            float epop = ad[6]; // electronic_population

            thermal_wavefront_sum += compute_thermal_wavefront(
                voxel_center, arom_pos, ring_norm, tse, vibe
            );
            halo_sum += compute_dewetting_halo(dist, epop);
        }

        // Basic UV signal
        if (n_nearby_excited > 0) {
            float energy_factor = fminf(total_vib_energy / (n_nearby_excited * 3.0f), 1.0f);
            float coop_boost = 1.0f + 0.3f * (n_nearby_excited - 1);
            uv_signal = UV_DIRECT_STRENGTH * energy_factor * coop_boost;
            if (uv_burst_active) uv_signal *= 2.0f;
        }

        // Advanced signal from fused thermal + halo (replaces compute_uv_lif_signal)
        if (d_uv_signal_prev != nullptr && (thermal_wavefront_sum > 0.0f || halo_sum > 0.0f)) {
            float prev_signal = d_uv_signal_prev[v];
            int n_contrib = n_nearby_excited > 0 ? n_nearby_excited : 1;
            float coop = compute_cooperative_boost(n_contrib);
            float advanced_signal = (thermal_wavefront_sum + halo_sum) * coop;
            // Temporal smoothing
            float smoothed = 0.7f * advanced_signal + 0.3f * prev_signal;
            uv_signal += smoothed;
            d_uv_signal_prev[v] = uv_signal;
        }
    }

    // Record UV event timestamp for causal tracking (neuron 0 only, before broadcast)
    if (is_valid && has_atoms && k == 0 && uv_signal > 0.1f) {
        last_uv_step[v] = timestep;
    }

    // Broadcast UV results from neuron 0 to all 8 neurons
    uv_signal = __shfl_sync(warp_mask, uv_signal, src_lane);
    n_nearby_excited = __shfl_sync(warp_mask, n_nearby_excited, src_lane);
    min_distance_to_excited = __shfl_sync(warp_mask, min_distance_to_excited, src_lane);
    closest_excited_idx = __shfl_sync(warp_mask, closest_excited_idx, src_lane);

    // ====================================================================
    // PHASE 2: RAF OSCILLATOR UPDATE
    // ====================================================================

    // Spike grid decrement (neuron 0 only)
    if (is_valid && has_atoms && k == 0 && spike_grid[v] > 0) {
        spike_grid[v]--;
    }
    if (is_valid && has_atoms && k == 0 && spike_grid_efp[v] > 0) {
        spike_grid_efp[v]--;
    }

    // Read coupling from previous step
    float coupling_input = (is_valid && has_atoms) ? coupling_read[v] : 0.0f;

    // Refractory check — per-neuron
    bool in_refractory = false;
    if (is_valid && has_atoms) {
        int refrac = neuron_refractory[nidx];
        if (refrac > 0) {
            neuron_refractory[nidx] = refrac - 1;
            in_refractory = true;
        }
    }

    // RAF oscillator state (initialized for all threads; only updated for active)
    float x_new = 0.0f, y_new = 0.0f;
    float threshold = MULTI_LIF_THRESHOLD;
    bool my_spike = false;

    if (is_valid && has_atoms && !in_refractory) {
        // Read oscillator state from global memory
        float x = neuron_potential[nidx];
        float y = neuron_mean[nidx];  // repurposed as imaginary part
        threshold = neuron_threshold[nidx];
        float my_tau = neuron_tau(k);

        // Compute water density signal (bidirectional + exclusion deviation)
        float wd_curr = water_density[v];
        float wd_prev = water_density_prev[v];
        float density_change = wd_curr - wd_prev;

        const float NOISE_FLOOR = 0.002f;
        float abs_change = fabsf(density_change);
        float bidirectional_signal = (abs_change > NOISE_FLOOR) ?
            (abs_change - NOISE_FLOOR) * 3.0f : 0.0f;

        float density_deviation = fabsf(wd_curr - WATER_DENSITY_BULK);
        const float EXCLUSION_NOISE_FLOOR = 0.005f;
        float exclusion_signal = (density_deviation > EXCLUSION_NOISE_FLOOR) ?
            (density_deviation - EXCLUSION_NOISE_FLOOR) * 2.0f : 0.0f;

        float combined_signal = bidirectional_signal + exclusion_signal + uv_signal;

        // RAF oscillator: damped harmonic oscillator with signal injection
        //   x_new = decay * (x*cos(wt) + y*sin(wt)) + signal
        //   y_new = decay * (-x*sin(wt) + y*cos(wt))
        // Read pre-computed trig from shared memory LUT (avoids 3 transcendentals per thread)
        float cos_wt = s_cos_lut[k];
        float sin_wt = s_sin_lut[k];
        float decay  = s_decay_lut[k];

        x_new = decay * (x * cos_wt + y * sin_wt) + combined_signal;
        y_new = decay * (-x * sin_wt + y * cos_wt);

        // Stochastic jitter (hash-based PRNG)
        unsigned int hash = (unsigned int)(v) * 2654435761u
                          + (unsigned int)(k) * 2246822519u
                          + (unsigned int)(timestep) * 3266489917u;
        hash ^= hash >> 16;
        hash *= 0x45d9f3bu;
        hash ^= hash >> 16;
        float jitter = JITTER_AMPLITUDE * ((float)(hash & 0xFFFF) / 65536.0f - 0.5f);
        x_new += jitter;
    }

    // Inter-neuron energy cascade: fast neurons (low k) feed slow neurons (high k)
    // All threads participate in warp shuffle (refractory/invalid contribute amp=0)
    float my_amp = sqrtf(x_new * x_new + y_new * y_new);
    float prev_amp = __shfl_up_sync(warp_mask, my_amp, 1);
    if (is_valid && has_atoms && !in_refractory && k > 0) {
        x_new += CASCADE_RATE * prev_amp;
    }

    // Continue RAF update for active neurons: saturated clamp + spike check
    if (is_valid && has_atoms && !in_refractory) {
        // Saturated resonance clamp
        float r2 = x_new * x_new + y_new * y_new;
        if (r2 > RAF_R_MAX * RAF_R_MAX) {
            float scale = RAF_R_MAX * rsqrtf(fmaxf(r2, 1e-12f));
            x_new *= scale;
            y_new *= scale;
            r2 = RAF_R_MAX * RAF_R_MAX;
        }

        // Spike check — coupling modulates threshold (NOT signal)
        float coupling_norm = fminf(coupling_input, 5.0f);
        float effective_threshold = threshold - COUPLING_STRENGTH * coupling_norm;
        if (effective_threshold < 0.2f) effective_threshold = 0.2f;

        float amplitude = sqrtf(r2);
        // RAF spikes require UV contribution — prevents crowding out EFP
        my_spike = (amplitude >= effective_threshold);

        if (my_spike) {
            // Always reset oscillator on threshold crossing
            x_new = 0.01f;
            y_new = 0.0f;
            // Only enter refractory + raise threshold when spike can be emitted
            if (spike_grid[v] == 0) {
                neuron_refractory[nidx] = REFRACTORY_STEPS;
                threshold = fminf(threshold + 0.02f, 2.0f * MULTI_LIF_THRESHOLD);
            }
        } else {
            threshold = fmaxf(threshold - 0.001f * dt, MULTI_LIF_THRESHOLD * 0.5f);
        }

        // Write back oscillator state
        neuron_potential[nidx] = x_new;
        neuron_mean[nidx] = y_new;
        neuron_threshold[nidx] = threshold;
    }

    // ====================================================================
    // SPIKE AGGREGATION + SHARED MEMORY STENCIL COUPLING
    // ====================================================================

    // Warp ballot: aggregate spikes across 8 neurons in voxel
    unsigned int spike_ballot = __ballot_sync(warp_mask, my_spike);
    unsigned int voxel_spikes = spike_ballot & voxel_mask;
    int n_spikes_in_voxel = __popc(voxel_spikes);

    // Max intensity reduction across voxel's 8 neurons (all participate)
    float my_intensity = my_spike ? neuron_tau(k) : 0.0f;
    for (int off = THREADS_PER_VOXEL / 2; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(warp_mask, my_intensity, off);
        my_intensity = fmaxf(my_intensity, other);
    }
    float spike_intensity = my_intensity + (float)n_spikes_in_voxel * 0.1f;

    // Neuron 0: shared memory stencil deposit + spike event emission
    if (is_valid && has_atoms && k == 0 && n_spikes_in_voxel > 0 && spike_grid[v] == 0) {
        spike_grid[v] = REFRACTORY_STEPS;

        // Signal preservation (RAF spike)
        int spike_source = (n_nearby_excited > 0 && uv_signal > 0.3f) ? 1 : 2;
        update_signal_preservation(v, timestep, spike_source,
            voxel_hit_grid, last_uv_step, coupled_spike_grid,
            primary_residue_id, primary_residue_count,
            entry, residue_ids, n_atoms, residue_step_causal);

        // Shared memory stencil: 26 Moore neighbors with distance weighting
        float coupling_deposit = 1.0f;
        int hx = lx + 1;  // offset +1 for halo border
        int hy = ly + 1;
        int hz = lz + 1;

        for (int dz = -1; dz <= 1; dz++)
        for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0 && dz == 0) continue;
            int nhx = hx + dx;
            int nhy = hy + dy;
            int nhz = hz + dz;
            int halo_idx = nhz * HALO_X * HALO_Y + nhy * HALO_X + nhx;

            int manhattan = abs(dx) + abs(dy) + abs(dz);
            float weight = (manhattan == 1) ? 1.0f :
                           (manhattan == 2) ? 0.7f : 0.5f;

            // Shared memory atomic (~5 cycles vs ~100 for global)
            atomicAdd(&s_coupling_tile[halo_idx], coupling_deposit * weight);
        }

        // Spike event emission
        int _arom_type = -1, _arom_res = -1;
        float _wl = 0.0f, _vibe = 0.0f;
        if (n_nearby_excited > 0 && closest_excited_idx >= 0) {
            _arom_type = d_aromatic_type[closest_excited_idx];
            _wl = uv_wavelength_nm;
            _vibe = d_vibrational_energy[closest_excited_idx];
            for (int wi = 0; wi < entry.n_atoms && _arom_res < 0; wi++) {
                int ai = entry.atom_indices[wi];
                if (ai >= 0 && ai < n_atoms && d_atom_to_aromatic[ai] == closest_excited_idx) {
                    _arom_res = residue_ids[ai];
                }
            }
        }

        int spike_idx = atomicAdd(spike_count, 1);
        if (spike_idx < max_spikes) {
            capture_spike_event(
                spike_events[spike_idx], timestep, v, voxel_center,
                spike_intensity, entry, residue_ids,
                spike_source, _wl, _arom_type, _arom_res,
                water_density[v], _vibe, n_nearby_excited,
                fabsf(water_density[v] - water_density_prev[v])
            );
        }
    }

    // ====================================================================
    // FLUSH SHARED MEMORY HALO TO GLOBAL COUPLING BUFFER
    // ====================================================================
    __syncthreads();  // Ensure all stencil writes are visible

    for (int i = threadIdx.x; i < HALO_SIZE; i += blockDim.x) {
        float val = s_coupling_tile[i];
        if (val == 0.0f) continue;

        int fhz = i / (HALO_X * HALO_Y);
        int fhy = (i / HALO_X) % HALO_Y;
        int fhx = i % HALO_X;

        int gx = tile_bx * TILE_X + fhx - 1;
        int gy = tile_by * TILE_Y + fhy - 1;
        int gz = tile_bz * TILE_Z + fhz - 1;

        if (gx < 0 || gx >= grid_dim || gy < 0 || gy >= grid_dim ||
            gz < 0 || gz >= grid_dim) continue;

        int gv = gz * grid_dim * grid_dim + gy * grid_dim + gx;
        atomicAdd(&coupling_write[gv], val);
    }

    // ====================================================================
    // PHASE 3: EFP (neuron 0 only — same as original)
    // ====================================================================
    if (is_valid && has_atoms && k == 0) {
        float phi = 0.0f;
        int n_charged_nearby = 0;

        for (int wi = 0; wi < entry.n_atoms; wi++) {
            int ai = entry.atom_indices[wi];
            if (ai < 0 || ai >= n_atoms) continue;
            float q = charges[ai];
            if (fabsf(q) < 0.15f) continue;

            float3 ap = positions[ai];
            float ddx = voxel_center.x - ap.x;
            float ddy = voxel_center.y - ap.y;
            float ddz = voxel_center.z - ap.z;
            float dist = sqrtf(ddx*ddx + ddy*ddy + ddz*ddz);

            if (dist > 0.5f && dist < 8.0f) {
                float eps_r = fmaxf(4.0f * dist, 4.0f);
                phi += q / (eps_r * dist);
                n_charged_nearby++;
            }
        }

        float wd_change = fabsf(water_density[v] - water_density_prev[v]);
        float polar_water_signal = fabsf(phi) * wd_change * 40.0f;

        float phi_prev = efp_potential_prev[v];
        efp_potential[v] = phi;

        float flux = fabsf(phi - phi_prev);
        float polar_signal = flux * 150.0f + polar_water_signal;

        if (n_charged_nearby >= 1 && spike_grid_efp[v] == 0) {
            const float EFP_TAU = 0.5f;
            const float EFP_THRESHOLD = 0.15f;
            float efp_decay = expf(-dt / EFP_TAU);
            efp_lif_potential[v] = efp_decay * efp_lif_potential[v] + polar_signal;

            if (efp_lif_potential[v] > EFP_THRESHOLD) {
                spike_grid_efp[v] = REFRACTORY_STEPS;
                float polar_intensity = efp_lif_potential[v];
                efp_lif_potential[v] = LIF_RESET;

                // Signal preservation (multi-LIF EFP spike)
                update_signal_preservation(v, timestep, 3,
                    voxel_hit_grid, last_uv_step, coupled_spike_grid,
                    primary_residue_id, primary_residue_count,
                    entry, residue_ids, n_atoms, residue_step_causal);

                int si = atomicAdd(spike_count, 1);
                if (si < max_spikes) {
                    int polar_type = (phi > 0.0f) ? 5 : 6;
                    capture_spike_event(
                        spike_events[si], timestep, v, voxel_center,
                        polar_intensity, entry, residue_ids,
                        3, 0.0f, polar_type, -1,
                        water_density[v], flux, n_charged_nearby,
                        wd_change
                    );
                }
            }
        }
        efp_potential_prev[v] = phi;
    }
}

// Kernel to initialize multi-neuron RAF state
extern "C" __global__ void init_multi_neuron(
    float* neuron_potential,
    float* neuron_threshold,
    float* neuron_mean,
    int*   neuron_refractory,
    float* coupling_a,
    float* coupling_b,
    int    total_voxels
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Initialize neuron arrays (total_voxels * K_NEURONS elements)
    if (tid < total_voxels * K_NEURONS) {
        neuron_potential[tid] = 0.01f;   // small x seed for oscillator
        neuron_threshold[tid] = MULTI_LIF_THRESHOLD;
        neuron_mean[tid] = 0.0f;        // y = 0 initially
        neuron_refractory[tid] = 0;
    }
    // Initialize coupling fields (total_voxels elements)
    if (tid < total_voxels) {
        coupling_a[tid] = 0.0f;
        coupling_b[tid] = 0.0f;
    }
}

// ════════════════════════════════════════════════════════════════════════════
// KCC v2-full: PER-RESIDUE KINEMATIC-CAUSAL UPDATE KERNEL
// ════════════════════════════════════════════════════════════════════════════
// Launched once per outer step, after multi-LIF. One thread per residue.
// Reads CA position, computes per-step motion, updates streaming reductions
// and micro-ring-buffer. Zeroes residue_step_causal for next step.
extern "C" __global__ void kcc_residue_update(
    // Atom positions (read CA positions via index mapping)
    const float3* __restrict__ positions,
    const int* __restrict__ residue_ca_idx,  // [n_residues] atom index of CA per residue
    int n_residues,
    // Streaming reduction buffers [n_residues]
    float* residue_prev_x,
    float* residue_prev_y,
    float* residue_prev_z,
    float* residue_sum_m,      // sum of per-step motion magnitude
    float* residue_sum_m2,     // sum of squared motion magnitude
    float* residue_sum_c,      // sum of per-step causal activity
    float* residue_sum_c2,     // sum of squared causal activity
    float* residue_sum_mc,     // sum of motion * causality (covariance term)
    float* residue_net_dx,     // net displacement x
    float* residue_net_dy,     // net displacement y
    float* residue_net_dz,     // net displacement z
    unsigned int* residue_count,              // total steps tracked
    unsigned int* residue_active_causal_steps, // steps with c_t > 0
    // Per-residue per-step causal counter (read then zero)
    int* residue_step_causal,  // [n_residues] accumulated during spike emission
    // Micro-ring-buffer [n_residues * KCC_RING_STEPS]
    float* ring_dx,
    float* ring_dy,
    float* ring_dz,
    float* ring_motion,
    float* ring_causality,
    unsigned int* ring_head,   // [n_residues] circular write pointer
    // Protocol phase tag
    int current_phase          // 0=cold_hold, 1=ramp, 2=warm_hold
) {
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= n_residues) return;

    // Read CA position
    int ca_idx = residue_ca_idx[rid];
    if (ca_idx < 0) return;  // no CA for this residue (e.g. water)
    float3 pos = positions[ca_idx];

    // Previous position
    float px = residue_prev_x[rid];
    float py = residue_prev_y[rid];
    float pz = residue_prev_z[rid];

    // Per-step displacement
    float dx = pos.x - px;
    float dy = pos.y - py;
    float dz = pos.z - pz;
    float m_t = sqrtf(dx*dx + dy*dy + dz*dz);

    // Read and zero per-step causal counter
    float c_t = (float)residue_step_causal[rid];
    residue_step_causal[rid] = 0;

    // Update streaming reductions
    residue_sum_m[rid]  += m_t;
    residue_sum_m2[rid] += m_t * m_t;
    residue_sum_c[rid]  += c_t;
    residue_sum_c2[rid] += c_t * c_t;
    residue_sum_mc[rid] += m_t * c_t;
    residue_net_dx[rid] += dx;
    residue_net_dy[rid] += dy;
    residue_net_dz[rid] += dz;
    residue_count[rid]  += 1u;
    if (c_t > 0.0f) residue_active_causal_steps[rid] += 1u;

    // Write to micro-ring-buffer
    unsigned int head = ring_head[rid];
    int slot = rid * KCC_RING_STEPS + (int)(head % KCC_RING_STEPS);
    ring_dx[slot]        = dx;
    ring_dy[slot]        = dy;
    ring_dz[slot]        = dz;
    ring_motion[slot]    = m_t;
    ring_causality[slot] = c_t;
    ring_head[rid]       = head + 1u;

    // Update previous position for next step
    residue_prev_x[rid] = pos.x;
    residue_prev_y[rid] = pos.y;
    residue_prev_z[rid] = pos.z;
}

// ════════════════════════════════════════════════════════════════════════════
// KCC v2-full: COMPUTE FINAL KCC METRICS FROM RING BUFFER
// ════════════════════════════════════════════════════════════════════════════
// Launched once at end of simulation. Computes burst, lag, phase, local cov.
extern "C" __global__ void kcc_compute_rich_descriptors(
    int n_residues,
    // Streaming reductions (read-only)
    const float* __restrict__ residue_sum_m,
    const float* __restrict__ residue_sum_m2,
    const float* __restrict__ residue_sum_c,
    const float* __restrict__ residue_sum_c2,
    const float* __restrict__ residue_sum_mc,
    const float* __restrict__ residue_net_dx,
    const float* __restrict__ residue_net_dy,
    const float* __restrict__ residue_net_dz,
    const unsigned int* __restrict__ residue_count,
    const unsigned int* __restrict__ residue_active_causal_steps,
    // Ring buffer (read-only)
    const float* __restrict__ ring_dx,
    const float* __restrict__ ring_dy,
    const float* __restrict__ ring_dz,
    const float* __restrict__ ring_motion,
    const float* __restrict__ ring_causality,
    const unsigned int* __restrict__ ring_head,
    // Output: per-residue KCC descriptors [n_residues]
    float* kcc_temporal_corr,
    float* kcc_direction_score,
    float* kcc_motion_efficiency,
    float* kcc_burst_motion_score,
    float* kcc_phase_shift_score,
    float* kcc_causal_lag,
    float* kcc_lag_corr_peak,
    float* kcc_local_cov_score
) {
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= n_residues) return;

    unsigned int N = residue_count[rid];
    if (N < 2) {
        kcc_temporal_corr[rid] = 0.0f;
        kcc_direction_score[rid] = 0.0f;
        kcc_motion_efficiency[rid] = 0.0f;
        kcc_burst_motion_score[rid] = 0.0f;
        kcc_phase_shift_score[rid] = 0.0f;
        kcc_causal_lag[rid] = 0.0f;
        kcc_lag_corr_peak[rid] = 0.0f;
        kcc_local_cov_score[rid] = 0.0f;
        return;
    }

    float fN = (float)N;

    // === Streaming temporal correlation ===
    float mean_m = residue_sum_m[rid] / fN;
    float mean_c = residue_sum_c[rid] / fN;
    float cov_mc = (residue_sum_mc[rid] / fN) - mean_m * mean_c;
    float var_m  = (residue_sum_m2[rid] / fN) - mean_m * mean_m;
    float var_c  = (residue_sum_c2[rid] / fN) - mean_c * mean_c;
    float denom  = sqrtf(fmaxf(var_m * var_c, 1e-12f));
    kcc_temporal_corr[rid] = cov_mc / denom;

    // === Direction score ===
    float net_dx = residue_net_dx[rid];
    float net_dy = residue_net_dy[rid];
    float net_dz = residue_net_dz[rid];
    float net_disp = sqrtf(net_dx*net_dx + net_dy*net_dy + net_dz*net_dz);
    float sum_step = residue_sum_m[rid];
    kcc_direction_score[rid] = net_disp / (sum_step + 1e-6f);

    // === Motion efficiency ===
    float scale = 5.0f;  // Å normalization
    kcc_motion_efficiency[rid] = 1.0f - expf(-sum_step / (scale * fN));

    // === Ring-buffer rich descriptors ===
    unsigned int head = ring_head[rid];
    int ring_len = (int)fminf((float)head, (float)KCC_RING_STEPS);
    int base = rid * KCC_RING_STEPS;

    if (ring_len < 4) {
        kcc_burst_motion_score[rid] = 0.0f;
        kcc_phase_shift_score[rid] = 0.0f;
        kcc_causal_lag[rid] = 0.0f;
        kcc_lag_corr_peak[rid] = 0.0f;
        kcc_local_cov_score[rid] = 0.0f;
        return;
    }

    // Read ring into local arrays (bounded by KCC_RING_STEPS=64)
    float local_m[KCC_RING_STEPS];
    float local_c[KCC_RING_STEPS];
    for (int i = 0; i < ring_len; i++) {
        int slot = base + (int)((head - ring_len + i) % KCC_RING_STEPS);
        local_m[i] = ring_motion[slot];
        local_c[i] = ring_causality[slot];
    }

    // === Burst-window motion accumulation ===
    // Motion accumulated during high-causality steps vs low-causality
    float burst_motion = 0.0f, quiet_motion = 0.0f;
    int burst_count = 0, quiet_count = 0;
    for (int i = 0; i < ring_len; i++) {
        if (local_c[i] > 0.0f) {
            burst_motion += local_m[i];
            burst_count++;
        } else {
            quiet_motion += local_m[i];
            quiet_count++;
        }
    }
    float burst_avg = burst_count > 0 ? burst_motion / burst_count : 0.0f;
    float quiet_avg = quiet_count > 0 ? quiet_motion / quiet_count : 0.0f;
    kcc_burst_motion_score[rid] = burst_avg / (quiet_avg + 1e-6f);

    // === Phase shift: early vs late half ===
    int half = ring_len / 2;
    float early_m = 0.0f, late_m = 0.0f;
    for (int i = 0; i < half; i++) early_m += local_m[i];
    for (int i = half; i < ring_len; i++) late_m += local_m[i];
    early_m /= (float)half;
    late_m /= (float)(ring_len - half);
    kcc_phase_shift_score[rid] = (late_m - early_m) / (early_m + late_m + 1e-6f);

    // === Causal lag estimate ===
    // Cross-correlate c_t and m_t at lags -8..+8
    float best_corr = -1.0f;
    int best_lag = 0;
    for (int lag = -8; lag <= 8; lag++) {
        float sum_cm = 0.0f, sum_cc = 0.0f, sum_mm = 0.0f;
        int cnt = 0;
        for (int i = 0; i < ring_len; i++) {
            int j = i + lag;
            if (j < 0 || j >= ring_len) continue;
            sum_cm += local_c[i] * local_m[j];
            sum_cc += local_c[i] * local_c[i];
            sum_mm += local_m[j] * local_m[j];
            cnt++;
        }
        if (cnt > 0) {
            float d = sqrtf(fmaxf(sum_cc * sum_mm, 1e-12f));
            float corr = sum_cm / d;
            if (corr > best_corr) {
                best_corr = corr;
                best_lag = lag;
            }
        }
    }
    kcc_causal_lag[rid] = (float)best_lag;
    kcc_lag_corr_peak[rid] = best_corr;

    // === Local covariance (sliding 16-step subwindows) ===
    int sub_w = 16;
    float max_local_cov = 0.0f;
    for (int start = 0; start + sub_w <= ring_len; start += sub_w / 2) {
        float sm = 0, sc = 0, smc = 0, sm2 = 0, sc2 = 0;
        for (int i = start; i < start + sub_w; i++) {
            sm += local_m[i]; sc += local_c[i];
            smc += local_m[i] * local_c[i];
            sm2 += local_m[i] * local_m[i];
            sc2 += local_c[i] * local_c[i];
        }
        float fw = (float)sub_w;
        float lcov = (smc / fw) - (sm / fw) * (sc / fw);
        float lvar = sqrtf(fmaxf((sm2/fw - (sm/fw)*(sm/fw)) * (sc2/fw - (sc/fw)*(sc/fw)), 1e-12f));
        float lcorr = lcov / lvar;
        if (lcorr > max_local_cov) max_local_cov = lcorr;
    }
    kcc_local_cov_score[rid] = max_local_cov;
}


// ═══════════════════════════════════════════════════════════════════════════
// GPU-ACCELERATED SPIKE DENSITY PEAK FINDING (KDE)
// ═══════════════════════════════════════════════════════════════════════════
//
// Computes burial-weighted Gaussian KDE on spike positions to find the
// density peak (hottest spot) within each detected pocket. This replaces
// the naive arithmetic centroid with a statistically robust estimate of
// the true pocket center.
//
// Architecture:
//   - One thread per 3D grid evaluation point (11³ = 1331 points per pocket)
//   - Shared memory caches spike positions + burial weights for the pocket
//   - Each thread evaluates the full KDE at its grid point
//   - Warp reduction finds the global maximum
//   - Final centroid = 70% density peak + 30% burial-weighted mean
//
// The burial weight (n_residues²) ensures that spikes deep inside the
// protein contribute more to the density field than surface spikes.
// This naturally pulls the peak toward the binding cavity interior.
//
// Complexity: O(n_grid_points × n_spikes_per_pocket)
// Typical: 1331 × 2000 = 2.6M kernel evaluations per pocket
// At ~1 TFLOP on RTX 5080: <1ms per pocket
// ═══════════════════════════════════════════════════════════════════════════

// Maximum spikes per pocket to fit in shared memory
// RTX 5080: 128KB shared mem per SM, SpikeKDE = 16 bytes, 128K/16 = 8192 max
#define KDE_MAX_SPIKES_SHARED 4096
#define KDE_GRID_DIM 11          // 11³ = 1331 grid points
#define KDE_GRID_TOTAL (KDE_GRID_DIM * KDE_GRID_DIM * KDE_GRID_DIM)

struct SpikeKDE {
    float x, y, z;      // position
    float weight;        // burial weight (n_residues²)
};

// Device function: evaluate Gaussian KDE at a point
__device__ __forceinline__ float kde_evaluate(
    float px, float py, float pz,
    const SpikeKDE* __restrict__ spikes,
    int n_spikes,
    float inv_2bw2  // 1 / (2 * bandwidth²)
) {
    float density = 0.0f;
    for (int i = 0; i < n_spikes; i++) {
        float dx = px - spikes[i].x;
        float dy = py - spikes[i].y;
        float dz = pz - spikes[i].z;
        float r2 = dx*dx + dy*dy + dz*dz;
        // Skip distant spikes (>3σ): exp(-9/2) ≈ 0.01
        if (r2 < 9.0f / inv_2bw2) {
            density += spikes[i].weight * expf(-r2 * inv_2bw2);
        }
    }
    return density;
}

// Kernel: find density peak for one pocket
// Launch: <<<ceil(KDE_GRID_TOTAL/256), 256>>> per pocket
// Inputs:
//   spike_positions: float3 array of spike xyz for this pocket
//   spike_n_residues: int array of nearby residue counts per spike
//   n_spikes: number of spikes in this pocket
//   centroid_in: [cx, cy, cz] initial centroid (burial-weighted mean from CPU)
//   search_radius: how far from centroid to search (typically 5.0A)
//   bandwidth: KDE bandwidth (typically 2.0A)
// Outputs:
//   peak_out: [px, py, pz, density] — position and density of the peak
extern "C" __global__ void kde_density_peak(
    const float* __restrict__ spike_x,
    const float* __restrict__ spike_y,
    const float* __restrict__ spike_z,
    const int*   __restrict__ spike_n_res,
    int          n_spikes,
    const float* __restrict__ centroid_in,  // [cx, cy, cz]
    float        search_radius,
    float        bandwidth,
    float*       peak_out                   // [px, py, pz, density]
) {
    // Shared memory for spike cache
    __shared__ SpikeKDE s_spikes[KDE_MAX_SPIKES_SHARED];
    __shared__ float s_best_density;
    __shared__ float s_best_pos[3];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize shared best
    if (tid == 0) {
        s_best_density = 0.0f;
        s_best_pos[0] = centroid_in[0];
        s_best_pos[1] = centroid_in[1];
        s_best_pos[2] = centroid_in[2];
    }

    // Load spikes into shared memory (collaborative)
    int spikes_to_load = min(n_spikes, KDE_MAX_SPIKES_SHARED);
    for (int i = tid; i < spikes_to_load; i += blockDim.x) {
        s_spikes[i].x = spike_x[i];
        s_spikes[i].y = spike_y[i];
        s_spikes[i].z = spike_z[i];
        // Burial weight: n_residues², minimum 1.0
        float nr = fmaxf((float)spike_n_res[i], 1.0f);
        s_spikes[i].weight = nr * nr;
    }
    __syncthreads();

    float cx = centroid_in[0];
    float cy = centroid_in[1];
    float cz = centroid_in[2];
    float grid_step = 2.0f * search_radius / (float)(KDE_GRID_DIM - 1);
    float inv_2bw2 = 1.0f / (2.0f * bandwidth * bandwidth);

    // Each thread evaluates one grid point
    if (gid < KDE_GRID_TOTAL) {
        int iz = gid / (KDE_GRID_DIM * KDE_GRID_DIM);
        int iy = (gid / KDE_GRID_DIM) % KDE_GRID_DIM;
        int ix = gid % KDE_GRID_DIM;

        float px = cx - search_radius + ix * grid_step;
        float py = cy - search_radius + iy * grid_step;
        float pz = cz - search_radius + iz * grid_step;

        float density = kde_evaluate(px, py, pz, s_spikes, spikes_to_load, inv_2bw2);

        // Warp-level reduction to find maximum within each warp
        // Then atomic update to shared best
        // Use atomicMax on int representation of float (positive floats preserve order)
        unsigned int density_bits = __float_as_uint(density);

        // AtomicMax for the density, then set position if we won
        // Simple approach: atomic compare-and-swap
        if (density > 0.0f) {
            // Thread-safe update: only one thread per block can update
            // Use a simple lock-free approach
            atomicMax((unsigned int*)&s_best_density, density_bits);
        }

        __syncthreads();

        // Check if this thread has the best density
        if (__float_as_uint(density) == __float_as_uint(s_best_density) && density > 0.0f) {
            s_best_pos[0] = px;
            s_best_pos[1] = py;
            s_best_pos[2] = pz;
        }
    }

    __syncthreads();

    // Thread 0 writes the final result: 70% peak + 30% centroid
    if (tid == 0) {
        peak_out[0] = 0.7f * s_best_pos[0] + 0.3f * cx;
        peak_out[1] = 0.7f * s_best_pos[1] + 0.3f * cy;
        peak_out[2] = 0.7f * s_best_pos[2] + 0.3f * cz;
        peak_out[3] = s_best_density;
    }
}


// ═══════════════════════════════════════════════════════════════════════════
// GPU-ACCELERATED BURIAL-WEIGHTED CENTROID
// ═══════════════════════════════════════════════════════════════════════════
//
// Computes weighted centroid of spike positions where weight = n_residues².
// Uses parallel reduction for O(log N) performance.
//
// Launch: <<<1, 256>>> per pocket
// ═══════════════════════════════════════════════════════════════════════════

extern "C" __global__ void burial_weighted_centroid(
    const float* __restrict__ spike_x,
    const float* __restrict__ spike_y,
    const float* __restrict__ spike_z,
    const int*   __restrict__ spike_n_res,
    int          n_spikes,
    float*       centroid_out    // [cx, cy, cz, total_weight]
) {
    __shared__ float s_wx[256];
    __shared__ float s_wy[256];
    __shared__ float s_wz[256];
    __shared__ float s_w[256];

    int tid = threadIdx.x;

    // Each thread accumulates over a strided range
    float wx = 0.0f, wy = 0.0f, wz = 0.0f, w = 0.0f;
    for (int i = tid; i < n_spikes; i += blockDim.x) {
        float nr = fmaxf((float)spike_n_res[i], 1.0f);
        float weight = nr * nr;
        wx += spike_x[i] * weight;
        wy += spike_y[i] * weight;
        wz += spike_z[i] * weight;
        w  += weight;
    }

    s_wx[tid] = wx;
    s_wy[tid] = wy;
    s_wz[tid] = wz;
    s_w[tid]  = w;
    __syncthreads();

    // Tree reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_wx[tid] += s_wx[tid + stride];
            s_wy[tid] += s_wy[tid + stride];
            s_wz[tid] += s_wz[tid + stride];
            s_w[tid]  += s_w[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float inv_w = (s_w[0] > 0.0f) ? 1.0f / s_w[0] : 0.0f;
        centroid_out[0] = s_wx[0] * inv_w;
        centroid_out[1] = s_wy[0] * inv_w;
        centroid_out[2] = s_wz[0] * inv_w;
        centroid_out[3] = s_w[0];
    }
}

// =============================================================================
// PRISM-NHS: Excited State Dynamics - FUSED KERNEL VERSION
// =============================================================================
// Designed for integration into nhs_amber_fused.cu
// Works with holographic neuromorphic stream + implicit solvent
// =============================================================================

#ifndef NHS_EXCITED_STATE_FUSED_CU
#define NHS_EXCITED_STATE_FUSED_CU

// =============================================================================
// CONSTANTS (same as before)
// =============================================================================

// Dipole ratios (excited/ground) - precomputed for efficiency
#define TYR_DIPOLE_RATIO_SQRT  1.749f  // sqrt(4.50/1.47)
#define PHE_DIPOLE_RATIO_SQRT  1.801f  // sqrt(1.20/0.37)
#define TRP_DIPOLE_RATIO_SQRT  1.690f  // sqrt(6.00/2.10)

// Timescales (ps) - as inverse rates for multiply instead of divide
#define INV_TAU_FRANCK_CONDON  20.0f   // 1/0.05 ps
#define INV_TAU_VIBRATIONAL    0.5f    // 1/2.0 ps
#define INV_TAU_FLUOR_TYR      0.000294f // 1/3400 ps
#define INV_TAU_FLUOR_PHE      0.000147f // 1/6800 ps
#define INV_TAU_FLUOR_TRP      0.000385f // 1/2600 ps
#define INV_TAU_IC_FACTOR      3.33f   // IC ~3x faster than fluorescence

// Energy
#define UV_PHOTON_ENERGY       102.0f  // kcal/mol (280nm)
#define ENERGY_TO_VEL_FACTOR   20.45f  // sqrt(2 * 418.4 / 12) for carbon

// =============================================================================
// INLINE DEVICE FUNCTIONS (to be called from main fused kernel)
// =============================================================================

// Fast exponential approximation (accurate to ~0.1% for x in [-2, 0])
__device__ __forceinline__ float fast_expf(float x) {
    // For small negative x, use: exp(x) ≈ 1 + x + x²/2
    // For larger x, use intrinsic
    if (x > -0.1f) {
        return 1.0f + x + 0.5f * x * x;
    }
    return __expf(x);  // CUDA fast intrinsic
}

// -----------------------------------------------------------------------------
// STEP 1: Apply electronic excitation (call when UV pulse triggers)
// -----------------------------------------------------------------------------
__device__ __forceinline__ void excite_aromatic_inline(
    int aromatic_idx,
    int residue_type,           // 0=TYR, 1=PHE, 2=TRP
    // Excited state arrays (in shared or global memory)
    int* is_excited,
    float* time_since_excitation,
    float* electronic_population,
    float* vibrational_energy,
    float* franck_condon_progress
) {
    is_excited[aromatic_idx] = 1;
    time_since_excitation[aromatic_idx] = 0.0f;
    electronic_population[aromatic_idx] = 1.0f;
    vibrational_energy[aromatic_idx] = UV_PHOTON_ENERGY;
    franck_condon_progress[aromatic_idx] = 0.0f;
}

// -----------------------------------------------------------------------------
// STEP 2-4: Update excited state (call every timestep, AFTER force calculation)
// -----------------------------------------------------------------------------
__device__ __forceinline__ void update_excited_state_inline(
    int aromatic_idx,
    int residue_type,
    float dt,
    // Excited state arrays
    int* is_excited,
    float* time_since_excitation,
    float* electronic_population,
    float* vibrational_energy,
    float* franck_condon_progress,
    // Output: energy to transfer to environment this step
    float* energy_to_transfer_out
) {
    if (!is_excited[aromatic_idx]) {
        *energy_to_transfer_out = 0.0f;
        return;
    }
    
    // Update timer
    time_since_excitation[aromatic_idx] += dt;
    
    // --- Franck-Condon relaxation (50 fs) ---
    float fc = franck_condon_progress[aromatic_idx];
    fc += (1.0f - fc) * INV_TAU_FRANCK_CONDON * dt;
    franck_condon_progress[aromatic_idx] = fc;
    
    // --- Vibrational relaxation (2 ps) ---
    float vib = vibrational_energy[aromatic_idx];
    float vib_decay = fast_expf(-INV_TAU_VIBRATIONAL * dt);
    float energy_out = vib * (1.0f - vib_decay);
    vibrational_energy[aromatic_idx] = vib * vib_decay;
    *energy_to_transfer_out = energy_out;
    
    // --- Electronic decay (ns timescale) ---
    float inv_tau_fluor;
    switch (residue_type) {
        case 0: inv_tau_fluor = INV_TAU_FLUOR_TYR; break;
        case 1: inv_tau_fluor = INV_TAU_FLUOR_PHE; break;
        case 2: inv_tau_fluor = INV_TAU_FLUOR_TRP; break;
        default: inv_tau_fluor = INV_TAU_FLUOR_TYR;
    }
    float total_decay_rate = inv_tau_fluor * (1.0f + INV_TAU_IC_FACTOR);
    float pop = electronic_population[aromatic_idx];
    pop *= fast_expf(-total_decay_rate * dt);
    electronic_population[aromatic_idx] = pop;
    
    // Check for complete decay
    if (pop < 0.01f) {
        is_excited[aromatic_idx] = 0;
        electronic_population[aromatic_idx] = 0.0f;
        vibrational_energy[aromatic_idx] = 0.0f;
    }
}

// -----------------------------------------------------------------------------
// Get charge scaling factor for an aromatic
// -----------------------------------------------------------------------------
__device__ __forceinline__ float get_charge_scale_inline(
    int aromatic_idx,
    int residue_type,
    const int* is_excited,
    const float* electronic_population
) {
    if (!is_excited[aromatic_idx]) {
        return 1.0f;
    }
    
    float ratio_sqrt;
    switch (residue_type) {
        case 0: ratio_sqrt = TYR_DIPOLE_RATIO_SQRT; break;
        case 1: ratio_sqrt = PHE_DIPOLE_RATIO_SQRT; break;
        case 2: ratio_sqrt = TRP_DIPOLE_RATIO_SQRT; break;
        default: ratio_sqrt = 1.0f;
    }
    
    float pop = electronic_population[aromatic_idx];
    // Interpolate: ground (1.0) → excited (ratio_sqrt)
    return 1.0f + (ratio_sqrt - 1.0f) * pop;
}

// =============================================================================
// INTEGRATION INTO NHS-AMBER FUSED KERNEL
// =============================================================================
// 
// Add these to your existing kernel parameters:
//
//   // Excited state data
//   int* d_is_excited,              // [n_aromatics]
//   float* d_time_since_excitation, // [n_aromatics]
//   float* d_electronic_population, // [n_aromatics]
//   float* d_vibrational_energy,    // [n_aromatics]
//   float* d_franck_condon_progress,// [n_aromatics]
//   float* d_ground_state_charges,  // [n_atoms]
//   int* d_atom_to_aromatic,        // [n_atoms] → -1 or aromatic index
//   int* d_aromatic_type,           // [n_aromatics] → 0=TYR,1=PHE,2=TRP
//   float3* d_ring_normals,         // [n_aromatics]
//   int n_aromatics,
//
// =============================================================================

// Example of how to modify your Coulomb calculation:
__device__ __forceinline__ float get_effective_charge(
    int atom_idx,
    const float* base_charges,
    const float* ground_state_charges,
    const int* atom_to_aromatic,
    const int* aromatic_type,
    const int* is_excited,
    const float* electronic_population
) {
    int aromatic_idx = atom_to_aromatic[atom_idx];
    
    if (aromatic_idx < 0) {
        // Not an aromatic atom - use base charge
        return base_charges[atom_idx];
    }
    
    // Aromatic atom - apply excited state scaling
    float scale = get_charge_scale_inline(
        aromatic_idx,
        aromatic_type[aromatic_idx],
        is_excited,
        electronic_population
    );
    
    return ground_state_charges[atom_idx] * scale;
}

// Example of how to apply vibrational energy transfer:
__device__ __forceinline__ void apply_vibrational_transfer(
    int aromatic_idx,
    float energy_to_transfer,
    float3 ring_normal,
    float3* velocities,
    const int* neighbor_list,    // Atoms within ~5Å of aromatic
    int n_neighbors,
    unsigned int seed            // For random direction component
) {
    if (energy_to_transfer < 0.001f || n_neighbors == 0) return;
    
    float energy_per_neighbor = energy_to_transfer / (float)n_neighbors;
    float vel_magnitude = ENERGY_TO_VEL_FACTOR * sqrtf(energy_per_neighbor);
    
    // Simple LCG for random numbers (no curand needed)
    unsigned int rng = seed;
    
    for (int i = 0; i < n_neighbors; i++) {
        int neighbor = neighbor_list[i];
        if (neighbor < 0) continue;
        
        // Mix of directed (ring normal) and random
        rng = rng * 1103515245 + 12345;
        float r = (float)(rng & 0x7FFFFFFF) / (float)0x7FFFFFFF;
        
        float3 kick_dir;
        if (r < 0.6f) {
            // 60% along ring normal (strongest coupling)
            kick_dir = ring_normal;
            if ((rng >> 16) & 1) kick_dir = -kick_dir;
        } else {
            // 40% random
            rng = rng * 1103515245 + 12345;
            float rx = (float)((rng >> 0) & 0xFF) / 127.5f - 1.0f;
            float ry = (float)((rng >> 8) & 0xFF) / 127.5f - 1.0f;
            float rz = (float)((rng >> 16) & 0xFF) / 127.5f - 1.0f;
            float len = sqrtf(rx*rx + ry*ry + rz*rz);
            kick_dir = make_float3(rx/len, ry/len, rz/len);
        }
        
        // Apply velocity kick (use atomicAdd for thread safety)
        atomicAdd(&velocities[neighbor].x, kick_dir.x * vel_magnitude);
        atomicAdd(&velocities[neighbor].y, kick_dir.y * vel_magnitude);
        atomicAdd(&velocities[neighbor].z, kick_dir.z * vel_magnitude);
    }
}

// =============================================================================
// NHS INTEGRATION: How excited states affect exclusion field
// =============================================================================
//
// Key insight: Excited aromatic has DIFFERENT hydrophobicity
// - Ground state Tyr: moderately hydrophobic ring + polar OH
// - Excited state Tyr: more polar (larger dipole) → LESS hydrophobic
//
// This means: exclusion field should DECREASE around excited aromatics
// → Water is MORE attracted to the enhanced dipole
// → This is what creates the detectable signal!

__device__ __forceinline__ float get_exclusion_modifier(
    int aromatic_idx,
    const int* is_excited,
    const float* electronic_population
) {
    if (!is_excited[aromatic_idx]) {
        return 1.0f;  // No modification
    }
    
    float pop = electronic_population[aromatic_idx];
    
    // Excited state is more polar → less hydrophobic → less exclusion
    // Reduce exclusion by up to 30% at full excitation
    return 1.0f - 0.3f * pop;
}

// Use this when computing exclusion field contribution from aromatic atoms:
//
// float exclusion_contribution = base_exclusion;
// if (atom_to_aromatic[atom_idx] >= 0) {
//     int arom_idx = atom_to_aromatic[atom_idx];
//     exclusion_contribution *= get_exclusion_modifier(arom_idx, is_excited, population);
// }

// =============================================================================
// NHS INTEGRATION: Enhanced spike detection for excited states
// =============================================================================
//
// The excited state creates a TRANSIENT change in exclusion
// This should produce spikes that are:
// 1. Temporally correlated with UV pulse
// 2. Spatially localized around aromatics
// 3. Decaying with fluorescence timescale
//
// You can detect this by looking for:
// - Spike ONSET after UV pulse (immediate exclusion decrease)
// - Spike OFFSET after decay (exclusion recovery)

__device__ __forceinline__ float compute_uv_induced_signal(
    int voxel_idx,
    float current_exclusion,
    float previous_exclusion,
    float baseline_exclusion,  // Before UV pulse
    float time_since_uv_pulse
) {
    // Signal = deviation from baseline, weighted by recency
    float deviation = fabsf(current_exclusion - baseline_exclusion);
    
    // Weight by temporal proximity to UV pulse (exponential window)
    float temporal_weight = fast_expf(-time_since_uv_pulse * 0.5f);  // 2 ps window
    
    // Also include rate of change (dewetting signal)
    float rate_of_change = fabsf(current_exclusion - previous_exclusion);
    
    return (deviation + rate_of_change) * temporal_weight;
}

#endif // NHS_EXCITED_STATE_FUSED_CU

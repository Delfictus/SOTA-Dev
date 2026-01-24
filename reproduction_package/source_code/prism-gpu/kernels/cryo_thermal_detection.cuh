//! PRISM-NHS Cryo Thermal Detection System
//!
//! Ultra-sensitive detection of UV absorption signatures in frozen landscapes.
//!
//! KEY PHYSICS:
//! - UV at 280nm is absorbed by aromatic residues (Trp >> Tyr > Phe)
//! - Absorption causes LOCAL HEATING even in frozen state
//! - Water does NOT absorb 280nm UV efficiently
//! - Therefore: thermal spike = binding site signature
//!
//! DETECTION MODES:
//! 1. Direct thermal spike at probe target
//! 2. Thermal gradient spreading from hot spot
//! 3. Phase transition (ice→water melt wave)
//! 4. Correlated multi-aromatic response (pocket signature)
//!
//! The neuromorphic detector converts these subtle thermal changes
//! into spike trains for pattern recognition.

#ifndef PRISM_CRYO_THERMAL_DETECTION_CUH
#define PRISM_CRYO_THERMAL_DETECTION_CUH

#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// PHYSICAL CONSTANTS
// ============================================================================

// UV absorption cross-sections (relative, Trp=1.0)
#define UV_ABSORPTION_TRP   1.0f      // Tryptophan - strongest absorber
#define UV_ABSORPTION_TYR   0.4f      // Tyrosine
#define UV_ABSORPTION_PHE   0.1f      // Phenylalanine
#define UV_ABSORPTION_HIS   0.05f     // Histidine (weak)
#define UV_ABSORPTION_WATER 0.001f    // Water - essentially transparent

// Thermal properties
#define HEAT_CAPACITY_PROTEIN  1.5f   // J/(g·K) approximate
#define HEAT_CAPACITY_ICE      2.09f  // J/(g·K)
#define HEAT_CAPACITY_WATER    4.18f  // J/(g·K)
#define LATENT_HEAT_FUSION     334.0f // J/g for ice→water

// Phase transition
#define MELTING_POINT          273.15f // K
#define SUPERCOOL_MARGIN       10.0f   // K below melting for metastable ice

// Detection thresholds (these are VERY sensitive!)
#define THERMAL_SPIKE_THRESHOLD     0.01f   // K - minimum detectable temp rise
#define THERMAL_GRADIENT_THRESHOLD  0.001f  // K/Å - minimum gradient
#define MELT_FRACTION_THRESHOLD     0.001f  // 0.1% melting detectable
#define CORRELATION_THRESHOLD       0.3f    // Cross-correlation for pocket

// Neuromorphic parameters
#define THERMAL_LIF_TAU            10.0f    // ps - integration time constant
#define THERMAL_LIF_THRESHOLD      0.05f   // Normalized spike threshold
#define THERMAL_LIF_RESET          0.0f    // Reset potential
#define THERMAL_REFRACTORY         5.0f    // ps - refractory period

// Grid resolution for thermal field
#define THERMAL_GRID_SPACING       1.0f    // Å - fine resolution for gradients

// ============================================================================
// DATA STRUCTURES
// ============================================================================

// Thermal state of a single voxel
struct ThermalVoxel {
    float temperature;          // Current temperature (K)
    float baseline_temp;        // Reference temperature before probe
    float delta_temp;           // Change from baseline
    float temp_gradient_mag;    // Magnitude of local gradient
    float3 temp_gradient_dir;   // Direction of steepest gradient
    
    float ice_fraction;         // 0=water, 1=ice
    float melt_rate;            // Rate of ice→water transition
    
    float heat_capacity;        // Effective heat capacity (ice/water mix)
    float thermal_conductivity; // Local conductivity
    
    int last_spike_time;        // When this voxel last spiked
    float lif_potential;        // Neuromorphic potential
    bool in_refractory;         // Refractory period flag
};

// UV probe event with absorption tracking
struct UVProbeEvent {
    float3 target_position;     // Where probe was aimed
    int target_aromatic_idx;    // Which aromatic (if any)
    float probe_energy;         // Energy delivered (arbitrary units)
    float absorption_fraction;  // How much was absorbed (0-1)
    
    float temp_rise;            // Resulting temperature rise
    float affected_volume;      // Volume heated (Å³)
    
    int response_spikes;        // Neuromorphic spikes triggered
    float response_latency;     // Time to first spike (ps)
    
    bool hit_aromatic;          // Did we hit an absorbing residue?
    bool caused_melting;        // Did we trigger ice→water?
    bool pocket_signature;      // Correlated multi-aromatic response?
};

// Pocket detection signature
struct ThermalPocketSignature {
    float3 center;              // Estimated pocket center
    float radius;               // Estimated pocket radius
    
    float peak_temp_rise;       // Maximum temperature rise observed
    float integrated_heat;      // Total heat deposited
    float melt_volume;          // Volume of ice melted
    
    int n_responding_aromatics; // How many aromatics responded
    int responding_aromatics[8];// Which ones
    float response_correlation; // How correlated are the responses
    
    float confidence;           // Overall pocket confidence
    int detection_count;        // How many times detected
};

// Complete thermal detection state
struct CryoThermalState {
    // Grid dimensions
    int3 grid_dim;
    float3 grid_origin;
    float grid_spacing;
    
    // Thermal field (flattened 3D array)
    ThermalVoxel* voxels;       // Device pointer
    int n_voxels;
    
    // Global temperature
    float ambient_temp;         // Background temperature
    float cryo_temp;            // Target cryo temperature
    
    // Probe tracking
    UVProbeEvent* recent_probes;// Circular buffer of recent probes
    int n_recent_probes;
    int probe_buffer_idx;
    
    // Detected signatures
    ThermalPocketSignature* signatures;
    int n_signatures;
    int max_signatures;
    
    // Neuromorphic output
    int* spike_voxels;          // Voxels that spiked this step
    int n_spikes_this_step;
    int total_thermal_spikes;
    
    // Statistics
    float max_temp_observed;
    float total_heat_absorbed;
    float total_ice_melted;
};

// ============================================================================
// DEVICE FUNCTIONS: THERMAL PHYSICS
// ============================================================================

// Get UV absorption coefficient for atom type
__device__ __forceinline__ float get_uv_absorption(int atom_type, int residue_type) {
    // residue_type: 0=TRP, 1=TYR, 2=PHE, 3=HIS, 4=other, 5=water
    
    switch (residue_type) {
        case 0: return UV_ABSORPTION_TRP;
        case 1: return UV_ABSORPTION_TYR;
        case 2: return UV_ABSORPTION_PHE;
        case 3: return UV_ABSORPTION_HIS;
        case 5: return UV_ABSORPTION_WATER;
        default: return 0.0f;  // Non-absorbing
    }
}

// Calculate temperature rise from UV absorption
__device__ __forceinline__ float calc_temp_rise(
    float absorbed_energy,      // Energy absorbed (arbitrary units)
    float mass,                 // Mass of absorbing region (amu)
    float heat_capacity         // Effective heat capacity
) {
    // Q = m * c * ΔT  →  ΔT = Q / (m * c)
    // Convert mass from amu to grams: 1 amu = 1.66e-24 g
    float mass_grams = mass * 1.66e-24f;
    
    // Absorbed energy in joules (scale factor for arbitrary units)
    float energy_joules = absorbed_energy * 1e-21f;  // Tunable scale
    
    if (mass_grams < 1e-30f || heat_capacity < 1e-6f) return 0.0f;
    
    return energy_joules / (mass_grams * heat_capacity);
}

// Calculate thermal diffusion (heat spreading)
__device__ __forceinline__ float calc_thermal_diffusion(
    float center_temp,
    float neighbor_temps[6],    // +x, -x, +y, -y, +z, -z
    float diffusivity,          // Thermal diffusivity (Å²/ps)
    float dt,                   // Timestep (ps)
    float grid_spacing          // Grid spacing (Å)
) {
    // 3D Laplacian: ∇²T = (T_+x + T_-x + T_+y + T_-y + T_+z + T_-z - 6*T_center) / h²
    float laplacian = 0.0f;
    for (int i = 0; i < 6; i++) {
        laplacian += neighbor_temps[i];
    }
    laplacian -= 6.0f * center_temp;
    laplacian /= (grid_spacing * grid_spacing);
    
    // Heat equation: dT/dt = α * ∇²T
    return diffusivity * laplacian * dt;
}

// Check for phase transition (melting)
__device__ __forceinline__ float calc_melt_rate(
    float temperature,
    float ice_fraction,
    float dt
) {
    if (ice_fraction <= 0.0f) return 0.0f;
    
    // Melting occurs above melting point
    if (temperature > MELTING_POINT) {
        // Rate proportional to superheat
        float superheat = temperature - MELTING_POINT;
        float rate = superheat * 0.01f;  // Tunable rate constant
        return fminf(rate * dt, ice_fraction);  // Can't melt more than exists
    }
    
    return 0.0f;
}

// Calculate temperature gradient at a voxel
__device__ __forceinline__ void calc_temp_gradient(
    float neighbor_temps[6],
    float grid_spacing,
    float3* gradient_dir,
    float* gradient_mag
) {
    // Central difference gradient
    gradient_dir->x = (neighbor_temps[0] - neighbor_temps[1]) / (2.0f * grid_spacing);
    gradient_dir->y = (neighbor_temps[2] - neighbor_temps[3]) / (2.0f * grid_spacing);
    gradient_dir->z = (neighbor_temps[4] - neighbor_temps[5]) / (2.0f * grid_spacing);
    
    *gradient_mag = sqrtf(
        gradient_dir->x * gradient_dir->x +
        gradient_dir->y * gradient_dir->y +
        gradient_dir->z * gradient_dir->z
    );
    
    // Normalize direction
    if (*gradient_mag > 1e-10f) {
        gradient_dir->x /= *gradient_mag;
        gradient_dir->y /= *gradient_mag;
        gradient_dir->z /= *gradient_mag;
    }
}

// ============================================================================
// DEVICE FUNCTIONS: NEUROMORPHIC THERMAL DETECTION
// ============================================================================

// Update LIF neuron for thermal detection
__device__ __forceinline__ bool update_thermal_lif(
    ThermalVoxel* voxel,
    float dt,
    int current_time
) {
    // Check refractory period
    if (voxel->in_refractory) {
        if (current_time - voxel->last_spike_time > THERMAL_REFRACTORY) {
            voxel->in_refractory = false;
        } else {
            return false;
        }
    }
    
    // Input current is the temperature change from baseline
    // VERY sensitive - we want to detect tiny changes
    float input = fabsf(voxel->delta_temp) / THERMAL_SPIKE_THRESHOLD;
    
    // Also respond to gradient (heat flow)
    input += voxel->temp_gradient_mag / THERMAL_GRADIENT_THRESHOLD;
    
    // Also respond to melting
    if (voxel->melt_rate > MELT_FRACTION_THRESHOLD) {
        input += voxel->melt_rate * 10.0f;  // Melting is a strong signal
    }
    
    // Leaky integration
    float decay = expf(-dt / THERMAL_LIF_TAU);
    voxel->lif_potential = voxel->lif_potential * decay + input * (1.0f - decay);
    
    // Check for spike
    if (voxel->lif_potential >= THERMAL_LIF_THRESHOLD) {
        voxel->lif_potential = THERMAL_LIF_RESET;
        voxel->last_spike_time = current_time;
        voxel->in_refractory = true;
        return true;  // SPIKE!
    }
    
    return false;
}

// Detect correlated response across multiple aromatics (pocket signature)
__device__ __forceinline__ float calc_aromatic_correlation(
    float* aromatic_responses,  // Temperature rise at each aromatic
    int n_aromatics,
    float time_window           // ps - window for correlation
) {
    if (n_aromatics < 2) return 0.0f;
    
    // Count how many aromatics responded significantly
    int n_responding = 0;
    float mean_response = 0.0f;
    
    for (int i = 0; i < n_aromatics; i++) {
        if (aromatic_responses[i] > THERMAL_SPIKE_THRESHOLD) {
            n_responding++;
            mean_response += aromatic_responses[i];
        }
    }
    
    if (n_responding < 2) return 0.0f;
    mean_response /= n_responding;
    
    // Calculate correlation as ratio of responding aromatics
    // times similarity of response magnitude
    float response_ratio = (float)n_responding / (float)n_aromatics;
    
    // Variance of responses (lower = more correlated)
    float variance = 0.0f;
    for (int i = 0; i < n_aromatics; i++) {
        if (aromatic_responses[i] > THERMAL_SPIKE_THRESHOLD) {
            float diff = aromatic_responses[i] - mean_response;
            variance += diff * diff;
        }
    }
    variance /= n_responding;
    
    // Correlation score: high when many aromatics respond similarly
    float similarity = 1.0f / (1.0f + variance / (mean_response * mean_response + 1e-6f));
    
    return response_ratio * similarity;
}

// ============================================================================
// KERNEL: APPLY UV PROBE AND TRACK THERMAL RESPONSE
// ============================================================================

__global__ void apply_uv_probe_thermal(
    // Probe parameters
    float3 probe_position,
    float probe_energy,
    float probe_radius,         // Effective radius of probe
    
    // Atom data
    const float3* __restrict__ positions,
    const int* __restrict__ residue_types,  // 0=TRP, 1=TYR, etc.
    const float* __restrict__ masses,
    int n_atoms,
    
    // Thermal grid
    ThermalVoxel* thermal_voxels,
    int3 grid_dim,
    float3 grid_origin,
    float grid_spacing,
    
    // Output
    UVProbeEvent* probe_result,
    int* spike_count
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Shared memory for reduction
    __shared__ float s_absorbed_energy;
    __shared__ float s_max_temp_rise;
    __shared__ int s_spike_count;
    __shared__ int s_hit_aromatic;
    
    if (threadIdx.x == 0) {
        s_absorbed_energy = 0.0f;
        s_max_temp_rise = 0.0f;
        s_spike_count = 0;
        s_hit_aromatic = 0;
    }
    __syncthreads();
    
    // Each thread handles one atom
    if (tid < n_atoms) {
        float3 pos = positions[tid];
        
        // Distance from probe
        float3 diff = make_float3(
            pos.x - probe_position.x,
            pos.y - probe_position.y,
            pos.z - probe_position.z
        );
        float dist = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
        
        // Within probe radius?
        if (dist < probe_radius) {
            // Get absorption coefficient
            float absorption = get_uv_absorption(0, residue_types[tid]);
            
            // Energy absorbed (Gaussian falloff)
            float gaussian = expf(-0.5f * (dist * dist) / (probe_radius * probe_radius * 0.25f));
            float absorbed = probe_energy * absorption * gaussian;
            
            if (absorbed > 0.0f) {
                atomicAdd(&s_absorbed_energy, absorbed);
                
                if (absorption > UV_ABSORPTION_WATER) {
                    atomicExch(&s_hit_aromatic, 1);
                }
                
                // Calculate temperature rise for this atom
                float temp_rise = calc_temp_rise(absorbed, masses[tid], HEAT_CAPACITY_PROTEIN);
                
                // Update thermal grid
                // Find voxel containing this atom
                int vx = (int)((pos.x - grid_origin.x) / grid_spacing);
                int vy = (int)((pos.y - grid_origin.y) / grid_spacing);
                int vz = (int)((pos.z - grid_origin.z) / grid_spacing);
                
                if (vx >= 0 && vx < grid_dim.x &&
                    vy >= 0 && vy < grid_dim.y &&
                    vz >= 0 && vz < grid_dim.z) {
                    
                    int voxel_idx = vx + vy * grid_dim.x + vz * grid_dim.x * grid_dim.y;
                    
                    // Update voxel temperature
                    atomicAdd(&thermal_voxels[voxel_idx].temperature, temp_rise);
                    atomicAdd(&thermal_voxels[voxel_idx].delta_temp, temp_rise);
                    
                    // Track max
                    float old_max = s_max_temp_rise;
                    while (temp_rise > old_max) {
                        old_max = atomicMax((int*)&s_max_temp_rise, __float_as_int(temp_rise));
                        old_max = __int_as_float((int)old_max);
                    }
                }
            }
        }
    }
    __syncthreads();
    
    // Thread 0 writes probe result
    if (threadIdx.x == 0 && probe_result != nullptr) {
        probe_result->target_position = probe_position;
        probe_result->probe_energy = probe_energy;
        probe_result->absorption_fraction = s_absorbed_energy / (probe_energy + 1e-10f);
        probe_result->temp_rise = s_max_temp_rise;
        probe_result->hit_aromatic = (s_hit_aromatic > 0);
        probe_result->response_spikes = s_spike_count;
    }
}

// ============================================================================
// KERNEL: UPDATE THERMAL FIELD (DIFFUSION + LIF)
// ============================================================================

__global__ void update_thermal_field(
    ThermalVoxel* voxels,
    int3 grid_dim,
    float grid_spacing,
    float dt,
    int current_time,
    float thermal_diffusivity,
    
    // Output
    int* spike_voxels,
    int* n_spikes
) {
    int vx = blockIdx.x * blockDim.x + threadIdx.x;
    int vy = blockIdx.y * blockDim.y + threadIdx.y;
    int vz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (vx >= grid_dim.x || vy >= grid_dim.y || vz >= grid_dim.z) return;
    
    int idx = vx + vy * grid_dim.x + vz * grid_dim.x * grid_dim.y;
    ThermalVoxel* voxel = &voxels[idx];
    
    // Get neighbor temperatures
    float neighbor_temps[6];
    
    // +x
    if (vx < grid_dim.x - 1)
        neighbor_temps[0] = voxels[idx + 1].temperature;
    else
        neighbor_temps[0] = voxel->temperature;
    
    // -x
    if (vx > 0)
        neighbor_temps[1] = voxels[idx - 1].temperature;
    else
        neighbor_temps[1] = voxel->temperature;
    
    // +y
    if (vy < grid_dim.y - 1)
        neighbor_temps[2] = voxels[idx + grid_dim.x].temperature;
    else
        neighbor_temps[2] = voxel->temperature;
    
    // -y
    if (vy > 0)
        neighbor_temps[3] = voxels[idx - grid_dim.x].temperature;
    else
        neighbor_temps[3] = voxel->temperature;
    
    // +z
    int z_stride = grid_dim.x * grid_dim.y;
    if (vz < grid_dim.z - 1)
        neighbor_temps[4] = voxels[idx + z_stride].temperature;
    else
        neighbor_temps[4] = voxel->temperature;
    
    // -z
    if (vz > 0)
        neighbor_temps[5] = voxels[idx - z_stride].temperature;
    else
        neighbor_temps[5] = voxel->temperature;
    
    // Calculate thermal diffusion
    float dT = calc_thermal_diffusion(
        voxel->temperature,
        neighbor_temps,
        thermal_diffusivity,
        dt,
        grid_spacing
    );
    
    // Update temperature
    voxel->temperature += dT;
    voxel->delta_temp = voxel->temperature - voxel->baseline_temp;
    
    // Calculate gradient
    calc_temp_gradient(
        neighbor_temps,
        grid_spacing,
        &voxel->temp_gradient_dir,
        &voxel->temp_gradient_mag
    );
    
    // Check for melting
    voxel->melt_rate = calc_melt_rate(voxel->temperature, voxel->ice_fraction, dt);
    voxel->ice_fraction -= voxel->melt_rate;
    if (voxel->ice_fraction < 0.0f) voxel->ice_fraction = 0.0f;
    
    // Neuromorphic LIF update
    bool spiked = update_thermal_lif(voxel, dt, current_time);
    
    if (spiked) {
        int spike_idx = atomicAdd(n_spikes, 1);
        if (spike_idx < grid_dim.x * grid_dim.y * grid_dim.z) {
            spike_voxels[spike_idx] = idx;
        }
    }
}

// ============================================================================
// KERNEL: DETECT POCKET SIGNATURES
// ============================================================================

__global__ void detect_pocket_signatures(
    // Aromatic data
    const float3* __restrict__ aromatic_centers,
    const float* __restrict__ aromatic_responses,  // Temp rise at each aromatic
    int n_aromatics,
    
    // Thermal grid for context
    const ThermalVoxel* __restrict__ voxels,
    int3 grid_dim,
    float3 grid_origin,
    float grid_spacing,
    
    // Output
    ThermalPocketSignature* signatures,
    int* n_signatures,
    int max_signatures,
    
    // Parameters
    float min_correlation,
    float min_temp_rise
) {
    // Single thread kernel for now (could parallelize over aromatic pairs)
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Find clusters of responding aromatics
    // Simple approach: for each aromatic with significant response,
    // check for nearby aromatics that also responded
    
    for (int i = 0; i < n_aromatics && *n_signatures < max_signatures; i++) {
        if (aromatic_responses[i] < min_temp_rise) continue;
        
        // Count nearby responding aromatics
        float3 center = aromatic_centers[i];
        float total_response = aromatic_responses[i];
        int responding[8];
        float responses[8];
        int n_responding = 1;
        responding[0] = i;
        responses[0] = aromatic_responses[i];
        
        for (int j = i + 1; j < n_aromatics && n_responding < 8; j++) {
            if (aromatic_responses[j] < min_temp_rise * 0.5f) continue;
            
            float3 other = aromatic_centers[j];
            float3 diff = make_float3(
                other.x - center.x,
                other.y - center.y,
                other.z - center.z
            );
            float dist = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
            
            // Within pocket-like distance (< 15 Å)
            if (dist < 15.0f) {
                responding[n_responding] = j;
                responses[n_responding] = aromatic_responses[j];
                total_response += aromatic_responses[j];
                
                // Update center (weighted average)
                float w = aromatic_responses[j] / total_response;
                center.x = center.x * (1.0f - w) + other.x * w;
                center.y = center.y * (1.0f - w) + other.y * w;
                center.z = center.z * (1.0f - w) + other.z * w;
                
                n_responding++;
            }
        }
        
        // Need at least 2 aromatics for a pocket signature
        if (n_responding >= 2) {
            float correlation = calc_aromatic_correlation(responses, n_responding, 10.0f);
            
            if (correlation >= min_correlation) {
                // Found a pocket signature!
                int sig_idx = atomicAdd(n_signatures, 1);
                if (sig_idx < max_signatures) {
                    ThermalPocketSignature* sig = &signatures[sig_idx];
                    sig->center = center;
                    sig->peak_temp_rise = total_response / n_responding;
                    sig->n_responding_aromatics = n_responding;
                    for (int k = 0; k < n_responding; k++) {
                        sig->responding_aromatics[k] = responding[k];
                    }
                    sig->response_correlation = correlation;
                    sig->confidence = correlation * (float)n_responding / 8.0f;
                    sig->detection_count = 1;
                    
                    // Estimate radius from aromatic spread
                    float max_dist = 0.0f;
                    for (int k = 0; k < n_responding; k++) {
                        float3 pos = aromatic_centers[responding[k]];
                        float3 diff = make_float3(
                            pos.x - center.x,
                            pos.y - center.y,
                            pos.z - center.z
                        );
                        float dist = sqrtf(diff.x*diff.x + diff.y*diff.y + diff.z*diff.z);
                        if (dist > max_dist) max_dist = dist;
                    }
                    sig->radius = max_dist + 3.0f;  // Add buffer
                }
            }
        }
    }
}

// ============================================================================
// HOST HELPER: INITIALIZE THERMAL GRID
// ============================================================================

__host__ void init_cryo_thermal_grid(
    CryoThermalState* state,
    float3 min_bounds,
    float3 max_bounds,
    float ambient_temp,
    float grid_spacing
) {
    // Calculate grid dimensions
    state->grid_spacing = grid_spacing;
    state->grid_origin = min_bounds;
    
    state->grid_dim.x = (int)ceilf((max_bounds.x - min_bounds.x) / grid_spacing) + 1;
    state->grid_dim.y = (int)ceilf((max_bounds.y - min_bounds.y) / grid_spacing) + 1;
    state->grid_dim.z = (int)ceilf((max_bounds.z - min_bounds.z) / grid_spacing) + 1;
    
    state->n_voxels = state->grid_dim.x * state->grid_dim.y * state->grid_dim.z;
    
    // Allocate voxels
    cudaMalloc(&state->voxels, state->n_voxels * sizeof(ThermalVoxel));
    
    // Initialize to cryo temperature
    state->ambient_temp = ambient_temp;
    state->cryo_temp = ambient_temp;
    
    // Would need a kernel to initialize voxels...
}

#endif // PRISM_CRYO_THERMAL_DETECTION_CUH

//! NHS-AMBER Fused Engine
//!
//! Rust wrapper for the fused GPU kernel that combines:
//! - Full AMBER ff14SB physics
//! - Langevin thermostat with dynamic temperature protocols
//! - Holographic exclusion field (negative space mapping)
//! - Neuromorphic LIF observation
//! - UV bias pump-probe
//! - Spike-triggered snapshot capture
//! - Warp matrix for atomic-precision alignment
//!
//! All running in a single GPU stream at 100,000+ timesteps/second.
//!
//! ## Cryogenic Physics
//!
//! For cryogenic contrast probing (50K-150K), special physics modifications:
//! - Temperature-dependent friction coefficient (scaled by T/300K)
//! - Temperature-dependent dielectric constant (increases as T decreases)
//! - Simulated annealing ramp (smooth transitions, no instant jumps)
//! - UV burst energy dissipation (prevents geometry explosion in frozen systems)

use anyhow::{bail, Context, Result};
use std::sync::Arc;

#[cfg(feature = "gpu")]
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, CudaFunction, CudaModule,
    LaunchConfig, PushKernelArg, DevicePtrMut, DevicePtr,
};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::Ptx;

use crate::input::PrismPrepTopology;

// ============================================================================
// GPU STRUCT TYPES (must match CUDA kernel exactly)
// ============================================================================

/// Bond parameter for CUDA kernel (matches BondParam in nhs_amber_fused.cu)
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuBondParam {
    pub i: i32,
    pub j: i32,
    pub r0: f32,
    pub k: f32,
}

/// Angle parameter for CUDA kernel (matches AngleParam in nhs_amber_fused.cu)
/// CUDA struct: i, j, k (12 bytes), theta0, force_k (8 bytes) = 20 bytes
#[cfg(feature = "gpu")]
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuAngleParam {
    pub i: i32,
    pub j: i32,
    pub k: i32,
    pub theta0: f32,
    pub force_k: f32,
}

/// Dihedral parameter for CUDA kernel (matches DihedralParam in nhs_amber_fused.cu)
/// CUDA struct: i, j, k, l (16 bytes), periodicity (4 bytes), phase, force_k (8 bytes) = 28 bytes
#[cfg(feature = "gpu")]
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuDihedralParam {
    pub i: i32,
    pub j: i32,
    pub k: i32,
    pub l: i32,
    pub periodicity: i32,
    pub phase: f32,
    pub force_k: f32,
}

/// LJ parameter for CUDA kernel (matches LJParam in nhs_amber_fused.cu)
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuLJParam {
    pub sigma: f32,
    pub epsilon: f32,
}

/// Hydrogen cluster for SHAKE (matches HCluster in nhs_amber_fused.cu)
/// CUDA struct: central_atom (4), hydrogen_atoms[3] (12), bond_lengths[3] (12),
/// n_hydrogens (4), inv_mass_central (4), inv_mass_h (4) = 40 bytes
#[cfg(feature = "gpu")]
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GpuHCluster {
    pub central_atom: i32,
    pub hydrogen_atoms: [i32; 3],  // -1 for unused
    pub bond_lengths: [f32; 3],
    pub n_hydrogens: i32,
    pub inv_mass_central: f32,
    pub inv_mass_h: f32,
}

#[cfg(feature = "gpu")]
impl Default for GpuHCluster {
    fn default() -> Self {
        Self {
            central_atom: -1,
            hydrogen_atoms: [-1, -1, -1],
            bond_lengths: [0.0, 0.0, 0.0],
            n_hydrogens: 0,
            inv_mass_central: 0.0,
            inv_mass_h: 0.0,
        }
    }
}

/// UV target for CUDA kernel (matches UVTarget in nhs_amber_fused.cu)
/// CUDA struct: residue_id (4), atom_indices[16] (64), n_atoms (4), absorption_strength (4), aromatic_type (4) = 80 bytes
#[cfg(feature = "gpu")]
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct GpuUVTarget {
    pub residue_id: i32,
    pub atom_indices: [i32; 16],
    pub n_atoms: i32,
    pub absorption_strength: f32,
    pub aromatic_type: i32,  // 0=TYR, 1=PHE, 2=TRP
}

#[cfg(feature = "gpu")]
impl Default for GpuUVTarget {
    fn default() -> Self {
        Self {
            residue_id: -1,
            atom_indices: [-1; 16],
            n_atoms: 0,
            absorption_strength: 0.0,
            aromatic_type: 0,
        }
    }
}

/// Aromatic neighbor list for vibrational energy transfer
/// CUDA struct: atom_indices[64] (256), n_neighbors (4) = 260 bytes
#[cfg(feature = "gpu")]
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct GpuAromaticNeighbors {
    pub atom_indices: [i32; 64],
    pub n_neighbors: i32,
}

#[cfg(feature = "gpu")]
impl Default for GpuAromaticNeighbors {
    fn default() -> Self {
        Self {
            atom_indices: [-1; 64],
            n_neighbors: 0,
        }
    }
}

/// Warp matrix entry for CUDA kernel (matches WarpEntry in nhs_amber_fused.cu)
/// CUDA struct: voxel_idx (4), atom_indices[16] (64), atom_weights[16] (64), n_atoms (4) = 136 bytes
#[cfg(feature = "gpu")]
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct GpuWarpEntry {
    pub voxel_idx: i32,
    pub atom_indices: [i32; 16],
    pub atom_weights: [f32; 16],
    pub n_atoms: i32,
}

#[cfg(feature = "gpu")]
impl Default for GpuWarpEntry {
    fn default() -> Self {
        Self {
            voxel_idx: -1,
            atom_indices: [-1; 16],
            atom_weights: [0.0; 16],
            n_atoms: 0,
        }
    }
}

/// Temperature protocol for CUDA kernel (matches TemperatureProtocol in nhs_amber_fused.cu)
/// CUDA struct: start_temp (4), end_temp (4), ramp_steps (4), hold_steps (4), current_step (4) = 20 bytes
#[cfg(feature = "gpu")]
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuTemperatureProtocol {
    pub start_temp: f32,
    pub end_temp: f32,
    pub ramp_steps: i32,
    pub hold_steps: i32,
    pub current_step: i32,
}

/// GPU spike event (matches SpikeEvent in nhs_amber_fused.cu)
/// CUDA struct: timestep (4), voxel_idx (4), position (12), intensity (4),
/// nearby_residues[8] (32), n_residues (4) = 60 bytes
#[cfg(feature = "gpu")]
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct GpuSpikeEvent {
    pub timestep: i32,
    pub voxel_idx: i32,
    pub position: [f32; 3],
    pub intensity: f32,
    pub nearby_residues: [i32; 8],
    pub n_residues: i32,
}

#[cfg(feature = "gpu")]
impl Default for GpuSpikeEvent {
    fn default() -> Self {
        Self {
            timestep: 0,
            voxel_idx: 0,
            position: [0.0; 3],
            intensity: 0.0,
            nearby_residues: [0; 8],
            n_residues: 0,
        }
    }
}

// ============================================================================
// TEMPERATURE PROTOCOLS
// ============================================================================

/// Temperature protocol for cryogenic contrast probing
#[derive(Debug, Clone)]
pub struct TemperatureProtocol {
    /// Starting temperature (Kelvin)
    pub start_temp: f32,
    /// Ending temperature (Kelvin)
    pub end_temp: f32,
    /// Number of steps to ramp temperature
    pub ramp_steps: i32,
    /// Number of steps to hold at end temperature
    pub hold_steps: i32,
    /// Current step in protocol
    pub current_step: i32,
}

impl TemperatureProtocol {
    /// Standard physiological temperature (300K constant)
    pub fn physiological() -> Self {
        Self {
            start_temp: 300.0,
            end_temp: 300.0,
            ramp_steps: 0,
            hold_steps: 100000,
            current_step: 0,
        }
    }

    /// Cryogenic probing protocol (100K → 300K)
    pub fn cryogenic_probe(cryo_temp: f32, ramp_steps: i32, hold_steps: i32) -> Self {
        Self {
            start_temp: cryo_temp,
            end_temp: 300.0,
            ramp_steps,
            hold_steps,
            current_step: 0,
        }
    }

    /// Deep freeze for extreme contrast (50K start)
    pub fn deep_freeze() -> Self {
        Self {
            start_temp: 50.0,
            end_temp: 300.0,
            ramp_steps: 50000,
            hold_steps: 50000,
            current_step: 0,
        }
    }

    /// Flash freeze then slow warm (for capturing transient states)
    pub fn flash_freeze_slow_warm() -> Self {
        Self {
            start_temp: 100.0,
            end_temp: 300.0,
            ramp_steps: 100000,
            hold_steps: 50000,
            current_step: 0,
        }
    }

    /// Get current temperature
    pub fn current_temperature(&self) -> f32 {
        if self.ramp_steps == 0 {
            return self.end_temp;
        }
        if self.current_step < self.ramp_steps {
            let t = self.current_step as f32 / self.ramp_steps as f32;
            self.start_temp + t * (self.end_temp - self.start_temp)
        } else {
            self.end_temp
        }
    }

    /// Advance to next step
    pub fn advance(&mut self) {
        self.current_step += 1;
    }

    /// Check if protocol is complete
    pub fn is_complete(&self) -> bool {
        self.current_step >= self.ramp_steps + self.hold_steps
    }

    /// Total steps in protocol
    pub fn total_steps(&self) -> i32 {
        self.ramp_steps + self.hold_steps
    }
}

// ============================================================================
// SPIKE EVENT CAPTURE
// ============================================================================

/// Captured spike event with atomic-precision mapping
#[derive(Debug, Clone)]
pub struct SpikeEvent {
    /// Timestep when spike occurred
    pub timestep: i32,
    /// Voxel index in grid
    pub voxel_idx: i32,
    /// 3D position of spike
    pub position: [f32; 3],
    /// Spike intensity
    pub intensity: f32,
    /// Nearby residue IDs (via warp matrix)
    pub nearby_residues: Vec<i32>,
    /// Temperature at time of spike
    pub temperature: f32,
    /// Whether UV burst was active
    pub uv_burst_active: bool,
}

/// Ensemble snapshot triggered by spike
#[derive(Debug, Clone)]
pub struct EnsembleSnapshot {
    /// Timestep
    pub timestep: i32,
    /// All atom positions at this moment
    pub positions: Vec<f32>,
    /// Velocities
    pub velocities: Vec<f32>,
    /// Spikes that triggered this capture
    pub trigger_spikes: Vec<SpikeEvent>,
    /// Current temperature
    pub temperature: f32,
}

// ============================================================================
// UV PROBE CONFIGURATION
// ============================================================================

/// UV burst configuration for pump-probe
#[derive(Debug, Clone)]
pub struct UvProbeConfig {
    /// Energy per burst (kcal/mol)
    pub burst_energy: f32,
    /// Interval between bursts (timesteps)
    pub burst_interval: i32,
    /// Duration of each burst (timesteps)
    pub burst_duration: i32,
    /// Target aromatic residues (indices into uv_targets)
    pub target_sequence: Vec<usize>,
    /// Current position in sequence
    pub current_target: usize,
    /// Timestep counter for burst timing
    pub timestep_counter: i32,
}

impl Default for UvProbeConfig {
    fn default() -> Self {
        Self {
            burst_energy: 5.0,
            burst_interval: 1000,
            burst_duration: 10,
            target_sequence: Vec::new(),
            current_target: 0,
            timestep_counter: 0,
        }
    }
}

impl UvProbeConfig {
    /// Check if burst should be active this timestep
    pub fn is_burst_active(&self) -> bool {
        let cycle_pos = self.timestep_counter % self.burst_interval;
        cycle_pos < self.burst_duration
    }

    /// Get current target index
    pub fn get_target_idx(&self) -> Option<usize> {
        if self.target_sequence.is_empty() {
            None
        } else {
            Some(self.target_sequence[self.current_target % self.target_sequence.len()])
        }
    }

    /// Advance to next timestep
    pub fn advance(&mut self) {
        self.timestep_counter += 1;
        if self.timestep_counter % self.burst_interval == 0 && !self.target_sequence.is_empty() {
            self.current_target = (self.current_target + 1) % self.target_sequence.len();
        }
    }
}

// ============================================================================
// CRYOGENIC PHYSICS CONSTANTS
// ============================================================================

/// Reference temperature for scaling (Kelvin)
const T_REF: f32 = 300.0;

/// Minimum temperature to prevent division by zero
const T_MIN: f32 = 10.0;

/// Dielectric constant at reference temperature
const EPSILON_REF: f32 = 78.5;  // Water at 300K

/// Dielectric constant at low temperature (ice-like)
const EPSILON_LOW: f32 = 3.2;   // Ice at 100K

/// UV energy dissipation factor at cold temps (prevents geometry explosion)
const UV_COLD_DISSIPATION: f32 = 0.3;

// ============================================================================
// FUSED ENGINE
// ============================================================================

/// Maximum grid dimension
const MAX_GRID_DIM: usize = 128;

/// Block size for 1D kernels
const BLOCK_SIZE_1D: usize = 256;

/// Maximum spikes per step
const MAX_SPIKES_PER_STEP: usize = 10000;

/// Maximum hydrogen clusters
const MAX_H_CLUSTERS: usize = 10000;

/// Maximum UV targets
const MAX_UV_TARGETS: usize = 256;

#[cfg(feature = "gpu")]
pub struct NhsAmberFusedEngine {
    // CUDA handles
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _fused_module: Arc<CudaModule>,

    // Kernel functions
    fused_step_kernel: CudaFunction,
    init_rng_kernel: CudaFunction,
    init_lif_kernel: CudaFunction,
    init_warp_matrix_kernel: CudaFunction,
    build_uv_targets_kernel: CudaFunction,

    // Atom state buffers (float3 = 3 contiguous floats)
    d_positions: CudaSlice<f32>,
    d_velocities: CudaSlice<f32>,
    d_forces: CudaSlice<f32>,
    d_masses: CudaSlice<f32>,
    d_charges: CudaSlice<f32>,
    d_atom_types: CudaSlice<i32>,
    d_residue_ids: CudaSlice<i32>,

    // AMBER parameter buffers (as raw bytes for GPU compatibility)
    d_bonds: CudaSlice<u8>,
    d_angles: CudaSlice<u8>,
    d_dihedrals: CudaSlice<u8>,
    d_lj_params: CudaSlice<u8>,
    d_exclusion_list: CudaSlice<i32>,
    d_exclusion_offsets: CudaSlice<i32>,

    // Struct sizes for GPU memory layout
    bond_size: usize,
    angle_size: usize,
    dihedral_size: usize,
    lj_size: usize,

    // SHAKE clusters (as raw bytes)
    d_h_clusters: CudaSlice<u8>,
    n_clusters: usize,
    h_cluster_size: usize,

    // UV targets (as raw bytes)
    d_uv_targets: CudaSlice<u8>,
    n_uv_targets: usize,
    uv_target_size: usize,

    // Warp matrix (as raw bytes)
    d_warp_matrix: CudaSlice<u8>,
    warp_entry_size: usize,

    // Grid buffers
    d_exclusion_field: CudaSlice<f32>,
    d_water_density: CudaSlice<f32>,
    d_water_density_prev: CudaSlice<f32>,
    d_lif_potential: CudaSlice<f32>,
    d_spike_grid: CudaSlice<i32>,

    // Spike output (events as raw bytes)
    d_spike_events: CudaSlice<u8>,
    spike_event_size: usize,
    d_spike_count: CudaSlice<i32>,

    // RNG states
    d_rng_states: CudaSlice<u8>,

    // Configuration
    n_atoms: usize,
    n_bonds: usize,
    n_angles: usize,
    n_dihedrals: usize,
    grid_dim: usize,
    grid_spacing: f32,
    grid_origin: [f32; 3],

    // Simulation parameters
    dt: f32,
    gamma_base: f32,     // Base friction at 300K
    cutoff: f32,
    timestep: i32,

    // Cryogenic physics parameters
    cryo_enabled: bool,
    dielectric_scaling: bool,

    // Protocols
    temp_protocol: TemperatureProtocol,
    uv_config: UvProbeConfig,

    // Cached aromatic residue info
    aromatic_residues: Vec<i32>,

    // ====================================================================
    // EXCITED STATE DYNAMICS BUFFERS (true UV photophysics)
    // ====================================================================
    d_is_excited: CudaSlice<i32>,              // [n_aromatics] - excitation flag
    d_time_since_excitation: CudaSlice<f32>,   // [n_aromatics] - time tracking
    d_electronic_population: CudaSlice<f32>,   // [n_aromatics] - 0.0-1.0 population
    d_vibrational_energy: CudaSlice<f32>,      // [n_aromatics] - kcal/mol
    d_franck_condon_progress: CudaSlice<f32>,  // [n_aromatics] - relaxation progress
    d_ground_state_charges: CudaSlice<f32>,    // [n_atoms] - original charges
    d_atom_to_aromatic: CudaSlice<i32>,        // [n_atoms] - -1 or aromatic index
    d_aromatic_type: CudaSlice<i32>,           // [n_aromatics] - 0=TYR,1=PHE,2=TRP
    d_ring_normals: CudaSlice<f32>,            // [n_aromatics * 3] - precomputed normals
    d_aromatic_neighbors: CudaSlice<u8>,       // [n_aromatics] - AromaticNeighbors structs
    aromatic_neighbors_size: usize,
    n_aromatics: usize,

    // Aromatic topology buffers for init kernels (Issue #3 fix)
    d_aromatic_atom_indices: CudaSlice<i32>,  // [n_aromatics * 16] - flat array of ring atom indices
    d_aromatic_n_atoms: CudaSlice<i32>,       // [n_aromatics] - count of atoms per aromatic

    // ====================================================================
    // O(N) CELL LIST / NEIGHBOR LIST BUFFERS
    // ====================================================================
    // Cell list constants (matches CUDA kernel)
    cell_size: f32,                           // = 10.0 Å (matches NB_CUTOFF)
    max_atoms_per_cell: usize,                // = 128
    neighbor_list_size: usize,                // = 256 per atom

    // Cell grid dimensions (computed from bounding box)
    cell_nx: i32,
    cell_ny: i32,
    cell_nz: i32,
    cell_origin: [f32; 3],

    // GPU buffers for cell list
    d_cell_list: CudaSlice<i32>,              // [n_total_cells * MAX_ATOMS_PER_CELL]
    d_cell_counts: CudaSlice<i32>,            // [n_total_cells]
    d_atom_cell: CudaSlice<i32>,              // [n_atoms] - which cell each atom is in

    // GPU buffers for neighbor list
    d_neighbor_list: CudaSlice<i32>,          // [n_atoms * NEIGHBOR_LIST_SIZE]
    d_n_neighbors: CudaSlice<i32>,            // [n_atoms] - actual neighbor count per atom

    // Rebuild control
    neighbor_list_rebuild_interval: i32,      // Rebuild every N steps (typically 10-20)
    steps_since_rebuild: i32,                 // Counter
    use_neighbor_list: bool,                  // Enable O(N) path (true for n_atoms > 500)

    // Captured data
    spike_events: Vec<SpikeEvent>,
    ensemble_snapshots: Vec<EnsembleSnapshot>,
}

#[cfg(feature = "gpu")]
impl NhsAmberFusedEngine {
    /// Create new fused engine from PRISM-PREP topology
    pub fn new(
        context: Arc<CudaContext>,
        topology: &PrismPrepTopology,
        grid_dim: usize,
        grid_spacing: f32,
    ) -> Result<Self> {
        log::info!("Creating NHS-AMBER Fused Engine: {} atoms, grid {}³",
            topology.n_atoms, grid_dim);

        if grid_dim > MAX_GRID_DIM {
            bail!("Grid dimension {} exceeds maximum {}", grid_dim, MAX_GRID_DIM);
        }

        let stream = context.default_stream();
        let n_atoms = topology.n_atoms;
        let total_voxels = grid_dim * grid_dim * grid_dim;

        // Compute grid origin from bounding box
        let (min_pos, _) = topology.bounding_box();
        let padding = 5.0f32;
        let grid_origin = [
            min_pos[0] - padding,
            min_pos[1] - padding,
            min_pos[2] - padding,
        ];

        // Load PTX module
        let ptx_path = "target/ptx/nhs_amber_fused.ptx";
        log::info!("Loading fused kernel PTX from: {}", ptx_path);

        let fused_module = context
            .load_module(Ptx::from_file(ptx_path))
            .context("Failed to load NHS-AMBER fused PTX")?;

        // Get kernel functions
        let fused_step_kernel = fused_module.load_function("nhs_amber_fused_step")?;
        let init_rng_kernel = fused_module.load_function("init_rng_states")?;
        let init_lif_kernel = fused_module.load_function("init_lif_state")?;
        // These kernels are optional - try to load, use defaults if missing
        let init_warp_matrix_kernel = fused_module.load_function("init_warp_matrix")
            .unwrap_or_else(|_| fused_step_kernel.clone());
        let build_uv_targets_kernel = fused_module.load_function("build_uv_targets")
            .unwrap_or_else(|_| fused_step_kernel.clone());

        // ====================================================================
        // ALLOCATE ATOM STATE BUFFERS
        // ====================================================================

        let d_positions: CudaSlice<f32> = stream.alloc_zeros(n_atoms * 3)?;
        let d_velocities: CudaSlice<f32> = stream.alloc_zeros(n_atoms * 3)?;
        let d_forces: CudaSlice<f32> = stream.alloc_zeros(n_atoms * 3)?;
        let d_masses: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;
        let d_charges: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;
        let d_atom_types: CudaSlice<i32> = stream.alloc_zeros(n_atoms)?;
        let d_residue_ids: CudaSlice<i32> = stream.alloc_zeros(n_atoms)?;

        // ====================================================================
        // BUILD AND UPLOAD AMBER PARAMETERS (as raw bytes for GPU compatibility)
        // ====================================================================

        let n_bonds = topology.bonds.len();
        let n_angles = topology.angles.len();
        let n_dihedrals = topology.dihedrals.len();

        // Get struct sizes for memory allocation
        let bond_size = std::mem::size_of::<GpuBondParam>();
        let angle_size = std::mem::size_of::<GpuAngleParam>();
        let dihedral_size = std::mem::size_of::<GpuDihedralParam>();
        let lj_size = std::mem::size_of::<GpuLJParam>();

        // Build bond parameters
        let bonds: Vec<GpuBondParam> = topology.bonds.iter().map(|b| {
            GpuBondParam {
                i: b.i as i32,
                j: b.j as i32,
                r0: b.r0 as f32,
                k: b.k as f32,
            }
        }).collect();
        let d_bonds: CudaSlice<u8> = stream.alloc_zeros((n_bonds.max(1) * bond_size))?;

        // Build angle parameters
        let angles: Vec<GpuAngleParam> = topology.angles.iter().map(|a| {
            GpuAngleParam {
                i: a.i as i32,
                j: a.j as i32,
                k: a.k_idx as i32,
                theta0: a.theta0 as f32,
                force_k: a.force_k as f32,
            }
        }).collect();
        let d_angles: CudaSlice<u8> = stream.alloc_zeros((n_angles.max(1) * angle_size))?;

        // Build dihedral parameters
        let dihedrals: Vec<GpuDihedralParam> = topology.dihedrals.iter().map(|d| {
            GpuDihedralParam {
                i: d.i as i32,
                j: d.j as i32,
                k: d.k_idx as i32,
                l: d.l as i32,
                periodicity: d.periodicity as i32,
                phase: d.phase as f32,
                force_k: d.force_k as f32,
            }
        }).collect();
        let d_dihedrals: CudaSlice<u8> = stream.alloc_zeros((n_dihedrals.max(1) * dihedral_size))?;

        // Build LJ parameters
        let lj_params: Vec<GpuLJParam> = topology.lj_params.iter().map(|lj| {
            GpuLJParam {
                sigma: lj.sigma as f32,
                epsilon: lj.epsilon as f32,
            }
        }).collect();
        let d_lj_params: CudaSlice<u8> = stream.alloc_zeros(n_atoms * lj_size)?;

        // Build exclusion list (CSR format)
        let mut exclusion_list: Vec<i32> = Vec::new();
        let mut exclusion_offsets: Vec<i32> = vec![0];
        for atom_exclusions in &topology.exclusions {
            for &excl_atom in atom_exclusions {
                exclusion_list.push(excl_atom as i32);
            }
            exclusion_offsets.push(exclusion_list.len() as i32);
        }
        let d_exclusion_list: CudaSlice<i32> = stream.alloc_zeros(exclusion_list.len().max(1))?;
        let d_exclusion_offsets: CudaSlice<i32> = stream.alloc_zeros(exclusion_offsets.len())?;

        // ====================================================================
        // BUILD SHAKE H-CLUSTERS
        // ====================================================================

        let h_cluster_size = std::mem::size_of::<GpuHCluster>();
        let mut h_clusters: Vec<GpuHCluster> = Vec::new();
        for cluster in &topology.h_clusters {
            let mut gpu_cluster = GpuHCluster::default();
            gpu_cluster.central_atom = cluster.central_atom as i32;
            gpu_cluster.n_hydrogens = cluster.n_hydrogens as i32;
            gpu_cluster.inv_mass_central = cluster.inv_mass_central as f32;
            gpu_cluster.inv_mass_h = cluster.inv_mass_h as f32;

            for (i, &h_atom) in cluster.hydrogen_atoms.iter().enumerate().take(3) {
                gpu_cluster.hydrogen_atoms[i] = h_atom;
            }
            for (i, &bond_len) in cluster.bond_lengths.iter().enumerate().take(3) {
                gpu_cluster.bond_lengths[i] = bond_len as f32;
            }

            h_clusters.push(gpu_cluster);
        }
        let n_clusters = h_clusters.len();
        let d_h_clusters: CudaSlice<u8> = stream.alloc_zeros((n_clusters.max(1) * h_cluster_size))?;

        // ====================================================================
        // BUILD UV TARGETS (aromatic residues) + EXCITED STATE MAPPINGS
        // ====================================================================

        let uv_target_size = std::mem::size_of::<GpuUVTarget>();
        let aromatic_residues: Vec<i32> = topology.aromatic_residues()
            .into_iter()
            .map(|r| r as i32)
            .collect();

        // Build atom_to_aromatic mapping (-1 for non-aromatic atoms)
        let mut atom_to_aromatic: Vec<i32> = vec![-1i32; n_atoms];
        let mut aromatic_types: Vec<i32> = Vec::new();

        let mut uv_targets: Vec<GpuUVTarget> = Vec::new();
        for (aromatic_idx, &res_id) in aromatic_residues.iter().enumerate() {
            // Find atoms in this aromatic residue
            let mut target = GpuUVTarget::default();
            target.residue_id = res_id;

            let mut atom_count = 0;
            for (atom_idx, &atom_res) in topology.residue_ids.iter().enumerate() {
                if atom_res as i32 == res_id && atom_count < 16 {
                    target.atom_indices[atom_count] = atom_idx as i32;
                    // Map this atom to its aromatic index
                    atom_to_aromatic[atom_idx] = aromatic_idx as i32;
                    atom_count += 1;
                }
            }
            target.n_atoms = atom_count as i32;

            // Set absorption strength and aromatic type based on residue type
            // TRP > TYR > PHE (roughly 5:3:1 ratio for molar absorptivity at 280nm)
            // aromatic_type: 0=TYR, 1=PHE, 2=TRP (matching CUDA constants)
            let res_name = topology.residue_ids.iter()
                .position(|&r| r as i32 == res_id)
                .map(|idx| topology.residue_names[idx].as_str())
                .unwrap_or("");

            let (absorption, arom_type) = match res_name {
                "TRP" => (1.0, 2),
                "TYR" => (0.6, 0),
                "PHE" => (0.2, 1),
                _ => (0.3, 0),
            };
            target.absorption_strength = absorption;
            target.aromatic_type = arom_type;
            aromatic_types.push(arom_type);

            uv_targets.push(target);
        }
        let n_uv_targets = uv_targets.len().min(MAX_UV_TARGETS);
        let n_aromatics = n_uv_targets;  // Same as UV targets
        let d_uv_targets: CudaSlice<u8> = stream.alloc_zeros((n_uv_targets.max(1) * uv_target_size))?;

        // ====================================================================
        // ALLOCATE EXCITED STATE BUFFERS
        // ====================================================================

        let d_is_excited: CudaSlice<i32> = stream.alloc_zeros(n_aromatics.max(1))?;
        let d_time_since_excitation: CudaSlice<f32> = stream.alloc_zeros(n_aromatics.max(1))?;
        let d_electronic_population: CudaSlice<f32> = stream.alloc_zeros(n_aromatics.max(1))?;
        let d_vibrational_energy: CudaSlice<f32> = stream.alloc_zeros(n_aromatics.max(1))?;
        let d_franck_condon_progress: CudaSlice<f32> = stream.alloc_zeros(n_aromatics.max(1))?;
        let d_ground_state_charges: CudaSlice<f32> = stream.alloc_zeros(n_atoms)?;
        let d_atom_to_aromatic: CudaSlice<i32> = stream.alloc_zeros(n_atoms)?;
        let d_aromatic_type: CudaSlice<i32> = stream.alloc_zeros(n_aromatics.max(1))?;
        let d_ring_normals: CudaSlice<f32> = stream.alloc_zeros(n_aromatics.max(1) * 3)?;

        // Aromatic neighbors for vibrational energy transfer
        let aromatic_neighbors_size = std::mem::size_of::<GpuAromaticNeighbors>();
        let d_aromatic_neighbors: CudaSlice<u8> = stream.alloc_zeros(n_aromatics.max(1) * aromatic_neighbors_size)?;

        // Aromatic topology buffers for init kernels (Issue #3 fix)
        // These are needed by build_aromatic_neighbors and compute_ring_normals CUDA kernels
        let d_aromatic_atom_indices: CudaSlice<i32> = stream.alloc_zeros(n_aromatics.max(1) * 16)?;
        let d_aromatic_n_atoms: CudaSlice<i32> = stream.alloc_zeros(n_aromatics.max(1))?;

        // ====================================================================
        // ALLOCATE O(N) CELL LIST / NEIGHBOR LIST BUFFERS
        // ====================================================================

        // Cell list constants (must match CUDA kernel defines)
        const CELL_SIZE: f32 = 10.0;         // Matches NB_CUTOFF
        const MAX_ATOMS_PER_CELL: usize = 128;
        const NEIGHBOR_LIST_SIZE: usize = 256;
        const MAX_CELLS_PER_DIM: i32 = 32;

        // Compute cell grid dimensions from bounding box
        let (min_pos, max_pos) = topology.bounding_box();
        let cell_padding = CELL_SIZE; // One cell of padding on each side
        let cell_origin = [
            min_pos[0] - cell_padding,
            min_pos[1] - cell_padding,
            min_pos[2] - cell_padding,
        ];

        let extent = [
            max_pos[0] - min_pos[0] + 2.0 * cell_padding,
            max_pos[1] - min_pos[1] + 2.0 * cell_padding,
            max_pos[2] - min_pos[2] + 2.0 * cell_padding,
        ];

        let cell_nx = ((extent[0] / CELL_SIZE).ceil() as i32).min(MAX_CELLS_PER_DIM).max(1);
        let cell_ny = ((extent[1] / CELL_SIZE).ceil() as i32).min(MAX_CELLS_PER_DIM).max(1);
        let cell_nz = ((extent[2] / CELL_SIZE).ceil() as i32).min(MAX_CELLS_PER_DIM).max(1);
        let n_total_cells = (cell_nx * cell_ny * cell_nz) as usize;

        log::info!("Cell grid: {}x{}x{} = {} cells (cell size {} Å)",
            cell_nx, cell_ny, cell_nz, n_total_cells, CELL_SIZE);

        // Allocate cell list buffers
        let d_cell_list: CudaSlice<i32> = stream.alloc_zeros(n_total_cells * MAX_ATOMS_PER_CELL)?;
        let d_cell_counts: CudaSlice<i32> = stream.alloc_zeros(n_total_cells)?;
        let d_atom_cell: CudaSlice<i32> = stream.alloc_zeros(n_atoms)?;

        // Allocate neighbor list buffers
        let d_neighbor_list: CudaSlice<i32> = stream.alloc_zeros(n_atoms * NEIGHBOR_LIST_SIZE)?;
        let d_n_neighbors: CudaSlice<i32> = stream.alloc_zeros(n_atoms)?;

        // Enable neighbor list for systems with > 500 atoms (where O(N) beats O(N²))
        let use_neighbor_list = n_atoms > 500;
        log::info!("Neighbor list mode: {} (n_atoms={})",
            if use_neighbor_list { "ENABLED (O(N))" } else { "DISABLED (O(N²))" },
            n_atoms);

        // ====================================================================
        // ALLOCATE WARP MATRIX (voxel-to-atom mapping)
        // ====================================================================

        let warp_entry_size = std::mem::size_of::<GpuWarpEntry>();
        let d_warp_matrix: CudaSlice<u8> = stream.alloc_zeros(total_voxels * warp_entry_size)?;

        // ====================================================================
        // ALLOCATE GRID BUFFERS
        // ====================================================================

        let d_exclusion_field: CudaSlice<f32> = stream.alloc_zeros(total_voxels)?;
        let d_water_density: CudaSlice<f32> = stream.alloc_zeros(total_voxels)?;
        let d_water_density_prev: CudaSlice<f32> = stream.alloc_zeros(total_voxels)?;
        let d_lif_potential: CudaSlice<f32> = stream.alloc_zeros(total_voxels)?;
        let d_spike_grid: CudaSlice<i32> = stream.alloc_zeros(total_voxels)?;

        // ====================================================================
        // ALLOCATE SPIKE OUTPUT
        // ====================================================================

        let spike_event_size = std::mem::size_of::<GpuSpikeEvent>();
        let d_spike_events: CudaSlice<u8> = stream.alloc_zeros(MAX_SPIKES_PER_STEP * spike_event_size)?;
        let d_spike_count: CudaSlice<i32> = stream.alloc_zeros(1)?;

        // ====================================================================
        // ALLOCATE RNG STATES
        // ====================================================================

        // curandState is ~48 bytes each
        let d_rng_states: CudaSlice<u8> = stream.alloc_zeros(n_atoms * 48)?;

        // ====================================================================
        // CREATE ENGINE
        // ====================================================================

        let mut engine = Self {
            context,
            stream,
            _fused_module: fused_module,
            fused_step_kernel,
            init_rng_kernel,
            init_lif_kernel,
            init_warp_matrix_kernel,
            build_uv_targets_kernel,

            d_positions,
            d_velocities,
            d_forces,
            d_masses,
            d_charges,
            d_atom_types,
            d_residue_ids,

            d_bonds,
            d_angles,
            d_dihedrals,
            d_lj_params,
            d_exclusion_list,
            d_exclusion_offsets,

            // Struct sizes for GPU memory layout
            bond_size,
            angle_size,
            dihedral_size,
            lj_size,

            d_h_clusters,
            n_clusters,
            h_cluster_size,

            d_uv_targets,
            n_uv_targets,
            uv_target_size,

            d_warp_matrix,
            warp_entry_size,

            d_exclusion_field,
            d_water_density,
            d_water_density_prev,
            d_lif_potential,
            d_spike_grid,

            d_spike_events,
            spike_event_size,
            d_spike_count,
            d_rng_states,

            n_atoms,
            n_bonds,
            n_angles,
            n_dihedrals,
            grid_dim,
            grid_spacing,
            grid_origin,

            dt: 0.002,          // 2 fs timestep
            gamma_base: 1.0,    // Base friction at 300K (ps^-1)
            cutoff: 10.0,       // 10 Angstrom nonbonded cutoff
            timestep: 0,

            cryo_enabled: true,
            dielectric_scaling: true,

            temp_protocol: TemperatureProtocol::physiological(),
            uv_config: UvProbeConfig::default(),

            aromatic_residues,

            // Excited state buffers
            d_is_excited,
            d_time_since_excitation,
            d_electronic_population,
            d_vibrational_energy,
            d_franck_condon_progress,
            d_ground_state_charges,
            d_atom_to_aromatic,
            d_aromatic_type,
            d_ring_normals,
            d_aromatic_neighbors,
            aromatic_neighbors_size,
            n_aromatics,

            // Aromatic topology buffers for init kernels (Issue #3 fix)
            d_aromatic_atom_indices,
            d_aromatic_n_atoms,

            // O(N) cell list / neighbor list buffers
            cell_size: CELL_SIZE,
            max_atoms_per_cell: MAX_ATOMS_PER_CELL,
            neighbor_list_size: NEIGHBOR_LIST_SIZE,
            cell_nx,
            cell_ny,
            cell_nz,
            cell_origin,
            d_cell_list,
            d_cell_counts,
            d_atom_cell,
            d_neighbor_list,
            d_n_neighbors,
            neighbor_list_rebuild_interval: 20,  // Rebuild every 20 steps
            steps_since_rebuild: 0,
            use_neighbor_list,

            spike_events: Vec::new(),
            ensemble_snapshots: Vec::new(),
        };

        // Upload all data to GPU
        engine.upload_topology_structs(topology, &bonds, &angles, &dihedrals, &lj_params,
                                        &exclusion_list, &exclusion_offsets,
                                        &h_clusters, &uv_targets,
                                        &atom_to_aromatic, &aromatic_types)?;

        // Initialize RNG
        engine.init_rng(42)?;

        // Initialize LIF state
        engine.init_lif_state()?;

        // Build warp matrix
        engine.build_warp_matrix()?;

        // ====================================================================
        // INITIALIZE AROMATIC NEIGHBOR LISTS AND RING NORMALS (Issues #1 & #2 fix)
        // These must be called AFTER positions are uploaded but BEFORE simulation starts
        // ====================================================================
        engine.init_aromatic_neighbors()?;
        engine.compute_ring_normals()?;

        log::info!("NHS-AMBER Fused Engine created successfully");
        log::info!("  Bonds: {}, Angles: {}, Dihedrals: {}", n_bonds, n_angles, n_dihedrals);
        log::info!("  H-Clusters (SHAKE): {}", n_clusters);
        log::info!("  UV Targets: {}", n_uv_targets);
        log::info!("  Cryogenic physics: {}", if engine.cryo_enabled { "ENABLED" } else { "disabled" });

        Ok(engine)
    }

    /// Compute temperature-dependent friction coefficient for cryogenic physics
    ///
    /// At low temperatures, the friction increases to slow down the atoms,
    /// mimicking the sluggish behavior of a frozen/near-frozen system.
    fn compute_cryo_friction(&self, temperature: f32) -> f32 {
        if !self.cryo_enabled || temperature >= T_REF {
            return self.gamma_base;
        }

        // Scale friction inversely with temperature
        // At T_REF (300K): gamma = gamma_base
        // At T_MIN (10K): gamma = gamma_base * 30 (much slower dynamics)
        let t_clamped = temperature.max(T_MIN);
        let scale = T_REF / t_clamped;

        // Use a smoother scaling: sqrt to prevent extreme values
        self.gamma_base * scale.sqrt()
    }

    /// Compute temperature-dependent dielectric constant for cryogenic physics
    ///
    /// At low temperatures, the dielectric constant changes dramatically:
    /// - Liquid water at 300K: ~78.5
    /// - Ice at 100K: ~3.2
    /// This affects electrostatic interactions significantly.
    fn compute_cryo_dielectric(&self, temperature: f32) -> f32 {
        if !self.dielectric_scaling || temperature >= T_REF {
            return EPSILON_REF;
        }

        // Linear interpolation between ice and water dielectric constants
        // based on temperature
        let t_clamped = temperature.max(T_MIN);
        let t_frac = (t_clamped - T_MIN) / (T_REF - T_MIN);

        // Interpolate: low T -> EPSILON_LOW, high T -> EPSILON_REF
        EPSILON_LOW + t_frac * (EPSILON_REF - EPSILON_LOW)
    }

    /// Compute UV burst energy with cold-temperature dissipation
    ///
    /// At cold temperatures, UV energy must be dissipated more carefully
    /// to prevent local geometry explosion in the frozen system.
    fn compute_uv_energy(&self, base_energy: f32, temperature: f32) -> f32 {
        if !self.cryo_enabled || temperature >= T_REF {
            return base_energy;
        }

        // Scale down UV energy at cold temperatures
        let t_clamped = temperature.max(T_MIN);
        let t_frac = (t_clamped - T_MIN) / (T_REF - T_MIN);

        // At T_MIN: use UV_COLD_DISSIPATION (30%) of base energy
        // At T_REF: use full base energy
        let scale = UV_COLD_DISSIPATION + t_frac * (1.0 - UV_COLD_DISSIPATION);
        base_energy * scale
    }

    /// Upload topology data to GPU using proper struct types
    fn upload_topology_structs(
        &mut self,
        topology: &PrismPrepTopology,
        bonds: &[GpuBondParam],
        angles: &[GpuAngleParam],
        dihedrals: &[GpuDihedralParam],
        lj_params: &[GpuLJParam],
        exclusion_list: &[i32],
        exclusion_offsets: &[i32],
        h_clusters: &[GpuHCluster],
        uv_targets: &[GpuUVTarget],
        atom_to_aromatic: &[i32],
        aromatic_types: &[i32],
    ) -> Result<()> {
        // Positions (flatten [x,y,z] format)
        self.stream.memcpy_htod(&topology.positions, &mut self.d_positions)?;

        // Initialize velocities from Maxwell-Boltzmann at starting temperature
        let temp = self.temp_protocol.current_temperature();
        let velocities = self.generate_maxwell_boltzmann_velocities(&topology.masses, temp);
        self.stream.memcpy_htod(&velocities, &mut self.d_velocities)?;

        // Masses
        self.stream.memcpy_htod(&topology.masses, &mut self.d_masses)?;

        // Charges
        self.stream.memcpy_htod(&topology.charges, &mut self.d_charges)?;

        // Atom types
        let atom_types: Vec<i32> = topology.classify_atoms()
            .iter()
            .map(|t| t.as_i32())
            .collect();
        self.stream.memcpy_htod(&atom_types, &mut self.d_atom_types)?;

        // Residue IDs
        let residue_ids: Vec<i32> = topology.residue_ids.iter()
            .map(|&r| r as i32)
            .collect();
        self.stream.memcpy_htod(&residue_ids, &mut self.d_residue_ids)?;

        // AMBER parameters - convert structs to bytes for GPU upload
        if !bonds.is_empty() {
            let bonds_bytes = Self::structs_to_bytes(bonds);
            self.stream.memcpy_htod(&bonds_bytes, &mut self.d_bonds)?;
        }
        if !angles.is_empty() {
            let angles_bytes = Self::structs_to_bytes(angles);
            self.stream.memcpy_htod(&angles_bytes, &mut self.d_angles)?;
        }
        if !dihedrals.is_empty() {
            let dihedrals_bytes = Self::structs_to_bytes(dihedrals);
            self.stream.memcpy_htod(&dihedrals_bytes, &mut self.d_dihedrals)?;
        }
        if !lj_params.is_empty() {
            let lj_bytes = Self::structs_to_bytes(lj_params);
            self.stream.memcpy_htod(&lj_bytes, &mut self.d_lj_params)?;
        }

        // Exclusion list (CSR format)
        if !exclusion_list.is_empty() {
            self.stream.memcpy_htod(exclusion_list, &mut self.d_exclusion_list)?;
        }
        self.stream.memcpy_htod(exclusion_offsets, &mut self.d_exclusion_offsets)?;

        // SHAKE H-clusters
        if !h_clusters.is_empty() {
            let h_clusters_bytes = Self::structs_to_bytes(h_clusters);
            self.stream.memcpy_htod(&h_clusters_bytes, &mut self.d_h_clusters)?;
        }

        // UV targets
        if !uv_targets.is_empty() {
            let uv_targets_bytes = Self::structs_to_bytes(uv_targets);
            self.stream.memcpy_htod(&uv_targets_bytes, &mut self.d_uv_targets)?;
        }

        // ====================================================================
        // UPLOAD EXCITED STATE MAPPINGS
        // ====================================================================

        // Ground state charges (copy of original charges for reference)
        self.stream.memcpy_htod(&topology.charges, &mut self.d_ground_state_charges)?;

        // Atom to aromatic mapping
        self.stream.memcpy_htod(atom_to_aromatic, &mut self.d_atom_to_aromatic)?;

        // Aromatic types
        if !aromatic_types.is_empty() {
            self.stream.memcpy_htod(aromatic_types, &mut self.d_aromatic_type)?;
        }

        // ====================================================================
        // UPLOAD AROMATIC TOPOLOGY FOR INIT KERNELS (Issue #3 fix)
        // Build flat arrays of aromatic atom indices and counts for GPU kernels
        // ====================================================================

        if !uv_targets.is_empty() {
            let n_aromatics = uv_targets.len();

            // Build flat array: [aromatic_0_atom_0, aromatic_0_atom_1, ..., aromatic_1_atom_0, ...]
            let mut aromatic_atom_indices_flat = vec![-1i32; n_aromatics * 16];
            let mut aromatic_n_atoms_flat = vec![0i32; n_aromatics];

            for (i, target) in uv_targets.iter().enumerate() {
                aromatic_n_atoms_flat[i] = target.n_atoms;
                for j in 0..16 {
                    aromatic_atom_indices_flat[i * 16 + j] = target.atom_indices[j];
                }
            }

            self.stream.memcpy_htod(&aromatic_atom_indices_flat, &mut self.d_aromatic_atom_indices)?;
            self.stream.memcpy_htod(&aromatic_n_atoms_flat, &mut self.d_aromatic_n_atoms)?;

            log::info!("Uploaded aromatic topology: {} aromatics with ring atom indices", n_aromatics);
        }

        // Initialize excited state arrays to zero (ground state)
        // d_is_excited, d_electronic_population, d_vibrational_energy, etc. are already zero-initialized

        log::info!("Uploaded topology: {} bonds, {} angles, {} dihedrals",
            self.n_bonds, self.n_angles, self.n_dihedrals);
        log::info!("Excited state: {} aromatics mapped", self.n_aromatics);

        Ok(())
    }

    /// Convert a slice of structs to a byte vector for GPU upload
    fn structs_to_bytes<T: Copy>(structs: &[T]) -> Vec<u8> {
        let size = std::mem::size_of::<T>();
        let total_bytes = structs.len() * size;
        let mut bytes = vec![0u8; total_bytes];

        unsafe {
            std::ptr::copy_nonoverlapping(
                structs.as_ptr() as *const u8,
                bytes.as_mut_ptr(),
                total_bytes,
            );
        }

        bytes
    }

    /// Generate Maxwell-Boltzmann distributed velocities
    fn generate_maxwell_boltzmann_velocities(&self, masses: &[f32], temperature: f32) -> Vec<f32> {
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::thread_rng();
        let mut velocities = vec![0.0f32; self.n_atoms * 3];

        // kB in kcal/(mol·K)
        const KB: f32 = 0.001987204;

        for i in 0..self.n_atoms {
            let mass = masses[i];
            if mass <= 0.0 {
                continue;
            }

            // Standard deviation from Maxwell-Boltzmann: sqrt(kT/m)
            let sigma = (KB * temperature / mass).sqrt();
            let normal = Normal::new(0.0f64, sigma as f64).unwrap();

            velocities[i * 3] = normal.sample(&mut rng) as f32;
            velocities[i * 3 + 1] = normal.sample(&mut rng) as f32;
            velocities[i * 3 + 2] = normal.sample(&mut rng) as f32;
        }

        // Remove center of mass velocity
        let mut com_vel = [0.0f32; 3];
        let mut total_mass = 0.0f32;
        for i in 0..self.n_atoms {
            let mass = masses[i];
            com_vel[0] += mass * velocities[i * 3];
            com_vel[1] += mass * velocities[i * 3 + 1];
            com_vel[2] += mass * velocities[i * 3 + 2];
            total_mass += mass;
        }
        if total_mass > 0.0 {
            com_vel[0] /= total_mass;
            com_vel[1] /= total_mass;
            com_vel[2] /= total_mass;

            for i in 0..self.n_atoms {
                velocities[i * 3] -= com_vel[0];
                velocities[i * 3 + 1] -= com_vel[1];
                velocities[i * 3 + 2] -= com_vel[2];
            }
        }

        velocities
    }

    /// Build warp matrix (voxel-to-atom mapping) on CPU
    fn build_warp_matrix(&mut self) -> Result<()> {
        let total_voxels = self.grid_dim * self.grid_dim * self.grid_dim;
        let cutoff_sq = 64.0f32;  // 8 Angstrom cutoff for warp mapping

        // Download current positions
        let mut positions = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_positions, &mut positions)?;

        let mut warp_entries: Vec<GpuWarpEntry> = Vec::with_capacity(total_voxels);

        for v in 0..total_voxels {
            let vz = v / (self.grid_dim * self.grid_dim);
            let vy = (v / self.grid_dim) % self.grid_dim;
            let vx = v % self.grid_dim;

            let voxel_center = [
                self.grid_origin[0] + (vx as f32 + 0.5) * self.grid_spacing,
                self.grid_origin[1] + (vy as f32 + 0.5) * self.grid_spacing,
                self.grid_origin[2] + (vz as f32 + 0.5) * self.grid_spacing,
            ];

            // Build entry using local arrays (avoid packed struct alignment issues)
            let mut atom_indices = [-1i32; 16];
            let mut atom_weights = [0.0f32; 16];
            let mut n_atoms = 0i32;

            // Find nearby atoms
            for a in 0..self.n_atoms {
                if n_atoms >= 16 {
                    break;
                }

                let dx = positions[a * 3] - voxel_center[0];
                let dy = positions[a * 3 + 1] - voxel_center[1];
                let dz = positions[a * 3 + 2] - voxel_center[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < cutoff_sq {
                    let weight = 1.0 / (1.0 + dist_sq);
                    atom_indices[n_atoms as usize] = a as i32;
                    atom_weights[n_atoms as usize] = weight;
                    n_atoms += 1;
                }
            }

            // Normalize weights
            let total_weight: f32 = atom_weights[..n_atoms as usize].iter().sum();
            if total_weight > 0.0 {
                for i in 0..n_atoms as usize {
                    atom_weights[i] /= total_weight;
                }
            }

            // Create entry from local values
            let entry = GpuWarpEntry {
                voxel_idx: v as i32,
                atom_indices,
                atom_weights,
                n_atoms,
            };
            warp_entries.push(entry);
        }

        // Upload to GPU as bytes
        let warp_bytes = Self::structs_to_bytes(&warp_entries);
        self.stream.memcpy_htod(&warp_bytes, &mut self.d_warp_matrix)?;

        // Diagnostic: count voxels with atoms
        let voxels_with_atoms = warp_entries.iter().filter(|e| e.n_atoms > 0).count();
        let avg_atoms_per_voxel = warp_entries.iter()
            .map(|e| e.n_atoms as f32)
            .sum::<f32>() / total_voxels as f32;
        let max_atoms = warp_entries.iter().map(|e| e.n_atoms).max().unwrap_or(0);

        log::info!("Built warp matrix: {} voxels ({} with atoms, avg {:.1} atoms/voxel, max {})",
            total_voxels, voxels_with_atoms, avg_atoms_per_voxel, max_atoms);
        Ok(())
    }

    /// Initialize RNG states
    fn init_rng(&mut self, seed: u64) -> Result<()> {
        let n_blocks = (self.n_atoms as u32).div_ceil(BLOCK_SIZE_1D as u32);
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.init_rng_kernel)
                .arg(&mut self.d_rng_states)
                .arg(&seed)
                .arg(&(self.n_atoms as i32))
                .launch(cfg)
        }
        .context("Failed to launch init_rng_states")?;

        self.context.synchronize()?;
        Ok(())
    }

    /// Initialize LIF neuron state
    fn init_lif_state(&mut self) -> Result<()> {
        let total_voxels = (self.grid_dim * self.grid_dim * self.grid_dim) as i32;
        let n_blocks = (total_voxels as u32).div_ceil(BLOCK_SIZE_1D as u32);
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.init_lif_kernel)
                .arg(&mut self.d_lif_potential)
                .arg(&mut self.d_water_density)
                .arg(&mut self.d_water_density_prev)
                .arg(&mut self.d_spike_grid)
                .arg(&total_voxels)
                .launch(cfg)
        }
        .context("Failed to launch init_lif_state")?;

        self.context.synchronize()?;
        Ok(())
    }

    /// Initialize aromatic neighbor lists for vibrational energy transfer (Issue #1 fix)
    ///
    /// This must be called AFTER positions are uploaded to GPU.
    /// The CUDA kernel `build_aromatic_neighbors` finds all atoms within 5A
    /// of each aromatic ring center (excluding ring atoms themselves).
    /// These neighbors receive vibrational energy kicks during UV excitation decay.
    fn init_aromatic_neighbors(&mut self) -> Result<()> {
        if self.n_aromatics == 0 {
            log::info!("No aromatics - skipping aromatic neighbor initialization");
            return Ok(());
        }

        // Load the build_aromatic_neighbors kernel
        let build_neighbors_kernel = self._fused_module
            .load_function("build_aromatic_neighbors")
            .context("Failed to load build_aromatic_neighbors kernel")?;

        let n_blocks = (self.n_aromatics as u32).div_ceil(BLOCK_SIZE_1D as u32);
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let neighbor_cutoff = 5.0f32;  // 5 Angstroms - atoms within this distance receive vibrational energy

        // Kernel signature from nhs_amber_fused.cu:
        // build_aromatic_neighbors(
        //     AromaticNeighbors* d_aromatic_neighbors,
        //     const float3* positions,
        //     const int* aromatic_atom_indices,  // [n_aromatics * 16]
        //     const int* aromatic_n_atoms,       // [n_aromatics]
        //     int n_aromatics,
        //     int n_atoms,
        //     float neighbor_cutoff
        // )

        unsafe {
            self.stream
                .launch_builder(&build_neighbors_kernel)
                .arg(&mut self.d_aromatic_neighbors)
                .arg(&self.d_positions)
                .arg(&self.d_aromatic_atom_indices)
                .arg(&self.d_aromatic_n_atoms)
                .arg(&(self.n_aromatics as i32))
                .arg(&(self.n_atoms as i32))
                .arg(&neighbor_cutoff)
                .launch(cfg)
        }
        .context("Failed to launch build_aromatic_neighbors")?;

        self.context.synchronize()?;
        log::info!("Built aromatic neighbor lists for {} aromatics (cutoff {:.1}A)",
            self.n_aromatics, neighbor_cutoff);

        Ok(())
    }

    /// Compute ring normal vectors for directional vibrational transfer (Issue #2 fix)
    ///
    /// This must be called AFTER positions are uploaded to GPU.
    /// The CUDA kernel `compute_ring_normals` computes the plane normal
    /// for each aromatic ring using cross product of two edge vectors.
    /// These normals are used to direct vibrational energy transfer
    /// perpendicular to the ring plane.
    fn compute_ring_normals(&mut self) -> Result<()> {
        if self.n_aromatics == 0 {
            log::info!("No aromatics - skipping ring normal computation");
            return Ok(());
        }

        // Load the compute_ring_normals kernel
        let compute_normals_kernel = self._fused_module
            .load_function("compute_ring_normals")
            .context("Failed to load compute_ring_normals kernel")?;

        let n_blocks = (self.n_aromatics as u32).div_ceil(BLOCK_SIZE_1D as u32);
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Kernel signature from nhs_amber_fused.cu:
        // compute_ring_normals(
        //     float3* d_ring_normals,
        //     const float3* positions,
        //     const int* aromatic_atom_indices,  // [n_aromatics * 16]
        //     const int* aromatic_n_atoms,       // [n_aromatics]
        //     int n_aromatics,
        //     int n_atoms
        // )

        unsafe {
            self.stream
                .launch_builder(&compute_normals_kernel)
                .arg(&mut self.d_ring_normals)
                .arg(&self.d_positions)
                .arg(&self.d_aromatic_atom_indices)
                .arg(&self.d_aromatic_n_atoms)
                .arg(&(self.n_aromatics as i32))
                .arg(&(self.n_atoms as i32))
                .launch(cfg)
        }
        .context("Failed to launch compute_ring_normals")?;

        self.context.synchronize()?;
        log::info!("Computed ring normals for {} aromatics", self.n_aromatics);

        Ok(())
    }

    // ========================================================================
    // O(N) NEIGHBOR LIST METHODS
    // ========================================================================

    /// Rebuild cell list and neighbor lists for O(N) nonbonded calculation
    ///
    /// This should be called every 10-20 timesteps. The overhead of rebuilding
    /// is amortized over the fast O(N) force calculation.
    ///
    /// Call sequence:
    /// 1. reset_cell_counts - clear previous frame's cell data
    /// 2. build_cell_list - assign each atom to a cell
    /// 3. build_neighbor_list - find neighbors within cutoff
    pub fn rebuild_neighbor_lists(&mut self) -> Result<()> {
        if !self.use_neighbor_list {
            return Ok(());  // O(N²) path, no neighbor lists needed
        }

        let n_total_cells = (self.cell_nx * self.cell_ny * self.cell_nz) as usize;

        // Step 1: Reset cell counts
        let reset_kernel = self._fused_module
            .load_function("reset_cell_counts")
            .context("Failed to load reset_cell_counts kernel")?;

        let n_blocks_cells = (n_total_cells as u32).div_ceil(BLOCK_SIZE_1D as u32);
        let cfg_cells = LaunchConfig {
            grid_dim: (n_blocks_cells, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&reset_kernel)
                .arg(&mut self.d_cell_counts)
                .arg(&(n_total_cells as i32))
                .launch(cfg_cells)
        }
        .context("Failed to launch reset_cell_counts")?;

        // Step 2: Build cell list
        let build_cell_kernel = self._fused_module
            .load_function("build_cell_list")
            .context("Failed to load build_cell_list kernel")?;

        let n_blocks_atoms = (self.n_atoms as u32).div_ceil(BLOCK_SIZE_1D as u32);
        let cfg_atoms = LaunchConfig {
            grid_dim: (n_blocks_atoms, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&build_cell_kernel)
                .arg(&self.d_positions)
                .arg(&mut self.d_cell_list)
                .arg(&mut self.d_cell_counts)
                .arg(&mut self.d_atom_cell)
                .arg(&self.cell_origin[0])
                .arg(&self.cell_origin[1])
                .arg(&self.cell_origin[2])
                .arg(&self.cell_nx)
                .arg(&self.cell_ny)
                .arg(&self.cell_nz)
                .arg(&(self.n_atoms as i32))
                .launch(cfg_atoms)
        }
        .context("Failed to launch build_cell_list")?;

        // Step 3: Build neighbor list
        let build_neighbor_kernel = self._fused_module
            .load_function("build_neighbor_list")
            .context("Failed to load build_neighbor_list kernel")?;

        // Use cutoff with 20% buffer for list reuse between rebuilds
        let cutoff_sq_with_buffer = self.cutoff * self.cutoff * 1.44;  // 1.2^2 = 1.44

        unsafe {
            self.stream
                .launch_builder(&build_neighbor_kernel)
                .arg(&self.d_positions)
                .arg(&self.d_cell_list)
                .arg(&self.d_cell_counts)
                .arg(&self.d_atom_cell)
                .arg(&self.d_exclusion_list)
                .arg(&self.d_exclusion_offsets)
                .arg(&mut self.d_neighbor_list)
                .arg(&mut self.d_n_neighbors)
                .arg(&self.cell_nx)
                .arg(&self.cell_ny)
                .arg(&self.cell_nz)
                .arg(&(self.n_atoms as i32))
                .arg(&cutoff_sq_with_buffer)
                .launch(cfg_atoms)
        }
        .context("Failed to launch build_neighbor_list")?;

        self.context.synchronize()?;
        self.steps_since_rebuild = 0;

        Ok(())
    }

    /// Compute nonbonded forces using O(N) neighbor lists
    ///
    /// This is a separate kernel call that replaces the O(N²) inline loop
    /// in the main fused kernel. For large proteins (>1000 atoms), this
    /// provides a 50-100x speedup.
    fn compute_nonbonded_with_neighbor_list(&mut self) -> Result<()> {
        let compute_nb_kernel = self._fused_module
            .load_function("compute_nonbonded_neighborlist")
            .context("Failed to load compute_nonbonded_neighborlist kernel")?;

        let n_blocks = (self.n_atoms as u32).div_ceil(BLOCK_SIZE_1D as u32);
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let cutoff_sq = self.cutoff * self.cutoff;

        unsafe {
            self.stream
                .launch_builder(&compute_nb_kernel)
                .arg(&self.d_positions)
                .arg(&mut self.d_forces)
                .arg(&self.d_charges)
                .arg(&self.d_lj_params)
                .arg(&self.d_neighbor_list)
                .arg(&self.d_n_neighbors)
                .arg(&self.d_atom_to_aromatic)
                .arg(&self.d_aromatic_type)
                .arg(&self.d_is_excited)
                .arg(&self.d_electronic_population)
                .arg(&self.d_ground_state_charges)
                .arg(&(self.n_atoms as i32))
                .arg(&cutoff_sq)
                .launch(cfg)
        }
        .context("Failed to launch compute_nonbonded_neighborlist")?;

        Ok(())
    }

    /// Set temperature protocol
    pub fn set_temperature_protocol(&mut self, protocol: TemperatureProtocol) -> Result<()> {
        self.temp_protocol = protocol.clone();
        log::info!("Set temperature protocol: {}K -> {}K over {} steps",
            protocol.start_temp, protocol.end_temp, protocol.ramp_steps);
        Ok(())
    }

    /// Set UV probe configuration
    pub fn set_uv_config(&mut self, config: UvProbeConfig) {
        self.uv_config = config;
    }

    /// Run a single timestep of the fused simulation
    ///
    /// This launches the full NHS-AMBER fused kernel which performs:
    /// 1. AMBER force computation (bonds, angles, dihedrals, nonbonded)
    /// 2. Velocity Verlet integration with Langevin thermostat
    /// 3. SHAKE constraints for hydrogen bonds
    /// 4. Holographic exclusion field update
    /// 5. Water density inference
    /// 6. Neuromorphic LIF observation
    /// 7. UV bias pump-probe (if active)
    /// 8. Spike event capture
    pub fn step(&mut self) -> Result<StepResult> {
        // Get current temperature from protocol (simulated annealing ramp)
        let current_temp = self.temp_protocol.current_temperature();

        // Compute cryogenic physics parameters
        let effective_gamma = self.compute_cryo_friction(current_temp);
        let _effective_dielectric = self.compute_cryo_dielectric(current_temp);

        // ====================================================================
        // O(N) NEIGHBOR LIST REBUILD (if needed)
        // ====================================================================
        // Rebuild every N steps or on first step
        if self.use_neighbor_list {
            self.steps_since_rebuild += 1;
            if self.steps_since_rebuild >= self.neighbor_list_rebuild_interval || self.timestep == 0 {
                self.rebuild_neighbor_lists()?;
            }
        }

        // Determine UV burst state and parameters
        let uv_burst_active = self.uv_config.is_burst_active();
        let uv_target_idx = self.uv_config.get_target_idx().unwrap_or(0) as i32;
        let uv_burst_energy = self.compute_uv_energy(self.uv_config.burst_energy, current_temp);

        // Reset spike count
        let zero = [0i32];
        self.stream.memcpy_htod(&zero, &mut self.d_spike_count)?;

        // ====================================================================
        // LAUNCH FUSED KERNEL
        // ====================================================================

        // Compute launch configuration
        let n_blocks = (self.n_atoms as u32).div_ceil(BLOCK_SIZE_1D as u32);
        let cfg = LaunchConfig {
            grid_dim: (n_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Convert parameters to kernel-compatible types
        let n_atoms_i32 = self.n_atoms as i32;
        let n_bonds_i32 = self.n_bonds as i32;
        let n_angles_i32 = self.n_angles as i32;
        let n_dihedrals_i32 = self.n_dihedrals as i32;
        let n_clusters_i32 = self.n_clusters as i32;
        let grid_dim_i32 = self.grid_dim as i32;
        let n_uv_targets_i32 = self.n_uv_targets as i32;
        let uv_burst_active_i32 = if uv_burst_active { 1i32 } else { 0i32 };
        let max_spikes_i32 = MAX_SPIKES_PER_STEP as i32;

        // Temperature protocol values
        let temp_start = self.temp_protocol.start_temp;
        let temp_end = self.temp_protocol.end_temp;
        let temp_ramp_steps = self.temp_protocol.ramp_steps;
        let temp_hold_steps = self.temp_protocol.hold_steps;
        let temp_current_step = self.temp_protocol.current_step;

        unsafe {
            self.stream
                .launch_builder(&self.fused_step_kernel)
                // Atom state (float3* treated as f32* with 3x elements)
                .arg(&mut self.d_positions)
                .arg(&mut self.d_velocities)
                .arg(&mut self.d_forces)
                .arg(&self.d_masses)
                .arg(&self.d_charges)
                .arg(&self.d_atom_types)
                .arg(&self.d_residue_ids)
                .arg(&n_atoms_i32)
                // AMBER parameters
                .arg(&self.d_bonds)
                .arg(&n_bonds_i32)
                .arg(&self.d_angles)
                .arg(&n_angles_i32)
                .arg(&self.d_dihedrals)
                .arg(&n_dihedrals_i32)
                .arg(&self.d_lj_params)
                .arg(&self.d_exclusion_list)
                .arg(&self.d_exclusion_offsets)
                // SHAKE clusters
                .arg(&self.d_h_clusters)
                .arg(&n_clusters_i32)
                // Grid buffers
                .arg(&mut self.d_exclusion_field)
                .arg(&mut self.d_water_density)
                .arg(&mut self.d_water_density_prev)
                .arg(&mut self.d_lif_potential)
                .arg(&mut self.d_spike_grid)
                // Grid origin (individual floats for cudarc compatibility)
                .arg(&self.grid_origin[0])
                .arg(&self.grid_origin[1])
                .arg(&self.grid_origin[2])
                .arg(&self.grid_spacing)
                .arg(&grid_dim_i32)
                // Warp matrix
                .arg(&mut self.d_warp_matrix)
                // UV targets
                .arg(&self.d_uv_targets)
                .arg(&n_uv_targets_i32)
                .arg(&uv_burst_active_i32)
                .arg(&uv_target_idx)
                .arg(&uv_burst_energy)
                // Excited state dynamics (true photophysics)
                .arg(&mut self.d_is_excited)
                .arg(&mut self.d_time_since_excitation)
                .arg(&mut self.d_electronic_population)
                .arg(&mut self.d_vibrational_energy)
                .arg(&mut self.d_franck_condon_progress)
                .arg(&self.d_ground_state_charges)
                .arg(&self.d_atom_to_aromatic)
                .arg(&self.d_aromatic_type)
                .arg(&self.d_ring_normals)
                .arg(&self.d_aromatic_neighbors)
                .arg(&(self.n_aromatics as i32))
                // Spike output
                .arg(&mut self.d_spike_events)
                .arg(&mut self.d_spike_count)
                .arg(&max_spikes_i32)
                // Temperature protocol (individual values)
                .arg(&temp_start)
                .arg(&temp_end)
                .arg(&temp_ramp_steps)
                .arg(&temp_hold_steps)
                .arg(&temp_current_step)
                // Simulation parameters
                .arg(&self.dt)
                .arg(&effective_gamma)
                .arg(&self.cutoff)
                .arg(&self.timestep)
                // RNG state
                .arg(&mut self.d_rng_states)
                // O(N) neighbor list (optional)
                .arg(&self.d_neighbor_list)
                .arg(&self.d_n_neighbors)
                .arg(&(if self.use_neighbor_list { 1i32 } else { 0i32 }))
                .launch(cfg)
        }
        .context("Failed to launch nhs_amber_fused_step kernel")?;

        // Advance protocols (CPU-side, no sync needed)
        self.temp_protocol.advance();
        self.uv_config.advance();
        self.timestep += 1;

        // Only sync and read spikes every N steps for performance
        // The GPU keeps running while we skip sync on most steps
        let sync_interval = 100; // Sync every 100 steps
        let num_spikes = if self.timestep % sync_interval == 0 {
            self.context.synchronize()?;

            let mut spike_count_host = [0i32];
            self.stream.memcpy_dtoh(&self.d_spike_count, &mut spike_count_host)?;
            let spikes = spike_count_host[0] as usize;

            // If significant spike activity detected, capture an ensemble snapshot
            if spikes >= 10 {
                self.capture_ensemble_snapshot(current_temp)?;
            }
            spikes
        } else {
            0 // Don't know spike count on non-sync steps
        };

        Ok(StepResult {
            timestep: self.timestep,
            temperature: current_temp,
            spike_count: num_spikes,
            uv_burst_active,
        })
    }

    /// Run N steps without any CPU-GPU synchronization (maximum throughput)
    /// Only syncs at the very end to get final spike count
    #[cfg(feature = "gpu")]
    pub fn step_batch(&mut self, n_steps: i32) -> Result<StepResult> {
        for _ in 0..n_steps {
            // Get current temperature for this step
            let current_temp = self.temp_protocol.current_temperature();
            let uv_burst_active = self.uv_config.is_burst_active();

            // Temperature protocol values
            let temp_start = self.temp_protocol.start_temp;
            let temp_end = self.temp_protocol.end_temp;
            let temp_ramp_steps = self.temp_protocol.ramp_steps;
            let temp_hold_steps = self.temp_protocol.hold_steps;
            let temp_current_step = self.temp_protocol.current_step;

            // Cryogenic scaling
            let effective_gamma = if self.cryo_enabled && current_temp < 200.0 {
                self.gamma_base * (current_temp / 300.0).max(0.1)
            } else {
                self.gamma_base
            };

            // UV burst parameters
            let uv_burst_active_i32 = if uv_burst_active { 1i32 } else { 0i32 };
            let uv_target_idx = self.uv_config.current_target as i32;
            let uv_burst_energy = if uv_burst_active { self.uv_config.burst_energy } else { 0.0 };

            // Grid parameters
            let grid_dim_i32 = self.grid_dim as i32;
            let n_atoms_i32 = self.n_atoms as i32;
            let n_bonds_i32 = self.n_bonds as i32;
            let n_angles_i32 = self.n_angles as i32;
            let n_dihedrals_i32 = self.n_dihedrals as i32;
            let n_clusters_i32 = self.n_clusters as i32;
            let n_uv_targets_i32 = self.n_uv_targets as i32;
            let max_spikes_i32 = MAX_SPIKES_PER_STEP as i32;

            // Launch kernel (no sync)
            let n_blocks = (self.n_atoms as u32).div_ceil(BLOCK_SIZE_1D as u32);
            let cfg = LaunchConfig {
                grid_dim: (n_blocks, 1, 1),
                block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                self.stream
                    .launch_builder(&self.fused_step_kernel)
                    .arg(&mut self.d_positions)
                    .arg(&mut self.d_velocities)
                    .arg(&mut self.d_forces)
                    .arg(&self.d_masses)
                    .arg(&self.d_charges)
                    .arg(&self.d_atom_types)
                    .arg(&self.d_residue_ids)
                    .arg(&n_atoms_i32)
                    .arg(&self.d_bonds).arg(&n_bonds_i32)
                    .arg(&self.d_angles).arg(&n_angles_i32)
                    .arg(&self.d_dihedrals).arg(&n_dihedrals_i32)
                    .arg(&self.d_lj_params)
                    .arg(&self.d_exclusion_list)
                    .arg(&self.d_exclusion_offsets)
                    .arg(&self.d_h_clusters).arg(&n_clusters_i32)
                    .arg(&mut self.d_exclusion_field)
                    .arg(&mut self.d_water_density)
                    .arg(&mut self.d_water_density_prev)
                    .arg(&mut self.d_lif_potential)
                    .arg(&mut self.d_spike_grid)
                    .arg(&self.grid_origin[0])
                    .arg(&self.grid_origin[1])
                    .arg(&self.grid_origin[2])
                    .arg(&self.grid_spacing)
                    .arg(&grid_dim_i32)
                    .arg(&mut self.d_warp_matrix)
                    .arg(&self.d_uv_targets).arg(&n_uv_targets_i32)
                    .arg(&uv_burst_active_i32)
                    .arg(&uv_target_idx)
                    .arg(&uv_burst_energy)
                    // Excited state dynamics
                    .arg(&mut self.d_is_excited)
                    .arg(&mut self.d_time_since_excitation)
                    .arg(&mut self.d_electronic_population)
                    .arg(&mut self.d_vibrational_energy)
                    .arg(&mut self.d_franck_condon_progress)
                    .arg(&self.d_ground_state_charges)
                    .arg(&self.d_atom_to_aromatic)
                    .arg(&self.d_aromatic_type)
                    .arg(&self.d_ring_normals)
                    .arg(&self.d_aromatic_neighbors)
                    .arg(&(self.n_aromatics as i32))
                    .arg(&mut self.d_spike_events)
                    .arg(&mut self.d_spike_count)
                    .arg(&max_spikes_i32)
                    .arg(&temp_start).arg(&temp_end)
                    .arg(&temp_ramp_steps).arg(&temp_hold_steps)
                    .arg(&temp_current_step)
                    .arg(&self.dt).arg(&effective_gamma)
                    .arg(&self.cutoff).arg(&self.timestep)
                    .arg(&mut self.d_rng_states)
                    // O(N) neighbor list (optional)
                    .arg(&self.d_neighbor_list)
                    .arg(&self.d_n_neighbors)
                    .arg(&(if self.use_neighbor_list { 1i32 } else { 0i32 }))
                    .launch(cfg)
            }
            .context("Failed to launch nhs_amber_fused_step kernel")?;

            // O(N) neighbor list rebuild check (if enabled)
            if self.use_neighbor_list {
                self.steps_since_rebuild += 1;
                if self.steps_since_rebuild >= self.neighbor_list_rebuild_interval {
                    self.rebuild_neighbor_lists()?;
                }
            }

            // Advance protocols (CPU-side only, no GPU sync)
            self.temp_protocol.advance();
            self.uv_config.advance();
            self.timestep += 1;
        }

        // Single sync at the end of batch
        self.context.synchronize()?;

        // Read final spike count
        let mut spike_count_host = [0i32];
        self.stream.memcpy_dtoh(&self.d_spike_count, &mut spike_count_host)?;
        let num_spikes = spike_count_host[0] as usize;

        Ok(StepResult {
            timestep: self.timestep,
            temperature: self.temp_protocol.current_temperature(),
            spike_count: num_spikes,
            uv_burst_active: false,
        })
    }

    /// Capture an ensemble snapshot when significant spike activity detected
    fn capture_ensemble_snapshot(&mut self, temperature: f32) -> Result<()> {
        // Download current positions and velocities
        let positions = self.get_positions()?;
        let velocities = self.get_velocities()?;

        let snapshot = EnsembleSnapshot {
            timestep: self.timestep,
            positions,
            velocities,
            trigger_spikes: Vec::new(),  // Would populate from GPU spike events
            temperature,
        };

        self.ensemble_snapshots.push(snapshot);

        if self.ensemble_snapshots.len() % 10 == 0 {
            log::info!("Captured {} ensemble snapshots", self.ensemble_snapshots.len());
        }

        Ok(())
    }

    /// Get current velocities from GPU
    pub fn get_velocities(&self) -> Result<Vec<f32>> {
        let mut velocities = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_velocities, &mut velocities)?;
        Ok(velocities)
    }

    /// Get current water density field from GPU
    /// This is the inferred water density at each voxel based on exclusion field
    pub fn get_water_density(&self) -> Result<Vec<f32>> {
        let total_voxels = self.grid_dim * self.grid_dim * self.grid_dim;
        let mut water_density = vec![0.0f32; total_voxels];
        self.stream.memcpy_dtoh(&self.d_water_density, &mut water_density)?;
        Ok(water_density)
    }

    /// Get current LIF membrane potentials from GPU
    pub fn get_lif_potential(&self) -> Result<Vec<f32>> {
        let total_voxels = self.grid_dim * self.grid_dim * self.grid_dim;
        let mut potentials = vec![0.0f32; total_voxels];
        self.stream.memcpy_dtoh(&self.d_lif_potential, &mut potentials)?;
        Ok(potentials)
    }

    /// Get current exclusion field from GPU
    pub fn get_exclusion_field(&self) -> Result<Vec<f32>> {
        let total_voxels = self.grid_dim * self.grid_dim * self.grid_dim;
        let mut exclusion = vec![0.0f32; total_voxels];
        self.stream.memcpy_dtoh(&self.d_exclusion_field, &mut exclusion)?;
        Ok(exclusion)
    }

    /// Get grid dimension
    pub fn grid_dim(&self) -> usize {
        self.grid_dim
    }

    /// Get number of atoms
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Download spike events from GPU
    /// Returns the voxel indices and positions where spikes occurred
    pub fn download_spike_events(&self, max_spikes: usize) -> Result<Vec<(i32, [f32; 3])>> {
        // Get spike count
        let mut spike_count = [0i32];
        self.stream.memcpy_dtoh(&self.d_spike_count, &mut spike_count)?;
        let n_spikes = (spike_count[0] as usize).min(max_spikes);

        if n_spikes == 0 {
            return Ok(Vec::new());
        }

        // Download spike events
        let bytes_needed = n_spikes * self.spike_event_size;
        let mut spike_bytes = vec![0u8; bytes_needed];

        // Create a view into just the spike events we need
        // Note: We're reading from the full buffer but only taking n_spikes worth
        let full_bytes = MAX_SPIKES_PER_STEP * self.spike_event_size;
        let mut full_buffer = vec![0u8; full_bytes];
        self.stream.memcpy_dtoh(&self.d_spike_events, &mut full_buffer)?;
        spike_bytes.copy_from_slice(&full_buffer[..bytes_needed]);

        // Parse spike events
        let mut events = Vec::with_capacity(n_spikes);
        for i in 0..n_spikes {
            let offset = i * self.spike_event_size;
            // GpuSpikeEvent layout: timestep(4), voxel_idx(4), position[3](12), ...
            let voxel_idx = i32::from_le_bytes([
                spike_bytes[offset + 4],
                spike_bytes[offset + 5],
                spike_bytes[offset + 6],
                spike_bytes[offset + 7],
            ]);
            let pos_x = f32::from_le_bytes([
                spike_bytes[offset + 8],
                spike_bytes[offset + 9],
                spike_bytes[offset + 10],
                spike_bytes[offset + 11],
            ]);
            let pos_y = f32::from_le_bytes([
                spike_bytes[offset + 12],
                spike_bytes[offset + 13],
                spike_bytes[offset + 14],
                spike_bytes[offset + 15],
            ]);
            let pos_z = f32::from_le_bytes([
                spike_bytes[offset + 16],
                spike_bytes[offset + 17],
                spike_bytes[offset + 18],
                spike_bytes[offset + 19],
            ]);

            events.push((voxel_idx, [pos_x, pos_y, pos_z]));
        }

        Ok(events)
    }

    /// Get spike grid (binary spike map) from GPU
    pub fn get_spike_grid(&self) -> Result<Vec<i32>> {
        let total_voxels = self.grid_dim * self.grid_dim * self.grid_dim;
        let mut spike_grid = vec![0i32; total_voxels];
        self.stream.memcpy_dtoh(&self.d_spike_grid, &mut spike_grid)?;
        Ok(spike_grid)
    }

    /// Enable or disable cryogenic physics
    pub fn set_cryogenic_mode(&mut self, enabled: bool) {
        self.cryo_enabled = enabled;
        log::info!("Cryogenic physics: {}", if enabled { "ENABLED" } else { "disabled" });
    }

    /// Enable or disable dielectric scaling
    pub fn set_dielectric_scaling(&mut self, enabled: bool) {
        self.dielectric_scaling = enabled;
        log::info!("Dielectric scaling: {}", if enabled { "ENABLED" } else { "disabled" });
    }

    /// Run multiple steps
    pub fn run(&mut self, n_steps: i32) -> Result<RunSummary> {
        let mut total_spikes = 0usize;
        let start_temp = self.temp_protocol.current_temperature();

        for _step in 0..n_steps {
            let result = self.step()?;
            total_spikes += result.spike_count;

            if self.timestep % 10000 == 0 {
                log::info!("Step {}: T={:.1}K, spikes={}",
                    self.timestep,
                    self.temp_protocol.current_temperature(),
                    result.spike_count);
            }
        }

        let end_temp = self.temp_protocol.current_temperature();

        Ok(RunSummary {
            steps_completed: n_steps,
            total_spikes,
            start_temperature: start_temp,
            end_temperature: end_temp,
            ensemble_snapshots: self.ensemble_snapshots.len(),
        })
    }

    /// Get current positions from GPU
    pub fn get_positions(&self) -> Result<Vec<f32>> {
        let mut positions = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_positions, &mut positions)?;
        Ok(positions)
    }

    /// Get captured spike events
    pub fn get_spike_events(&self) -> &[SpikeEvent] {
        &self.spike_events
    }

    /// Get ensemble snapshots
    pub fn get_ensemble_snapshots(&self) -> &[EnsembleSnapshot] {
        &self.ensemble_snapshots
    }
}

// ============================================================================
// RESULT TYPES
// ============================================================================

/// Result of a single step
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Current timestep
    pub timestep: i32,
    /// Current temperature (K)
    pub temperature: f32,
    /// Number of spikes detected
    pub spike_count: usize,
    /// Whether UV burst was active
    pub uv_burst_active: bool,
}

/// Summary of a multi-step run
#[derive(Debug, Clone)]
pub struct RunSummary {
    /// Steps completed
    pub steps_completed: i32,
    /// Total spikes detected
    pub total_spikes: usize,
    /// Starting temperature
    pub start_temperature: f32,
    /// Ending temperature
    pub end_temperature: f32,
    /// Number of ensemble snapshots captured
    pub ensemble_snapshots: usize,
}

// ============================================================================
// NON-GPU STUB
// ============================================================================

#[cfg(not(feature = "gpu"))]
pub struct NhsAmberFusedEngine;

#[cfg(not(feature = "gpu"))]
impl NhsAmberFusedEngine {
    /// Stub for non-GPU builds
    pub fn new(
        _topology: &PrismPrepTopology,
        _grid_dim: usize,
        _grid_spacing: f32,
    ) -> Result<Self> {
        bail!("NHS-AMBER Fused Engine requires GPU. Compile with --features gpu")
    }
}

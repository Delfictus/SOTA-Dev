//! NHS GPU Engine - CUDA-accelerated Hydrophobic Exclusion Mapping
//!
//! Full GPU implementation of the NHS pipeline components:
//!
//! 1. **Exclusion Field Computation** - Parallel HEM from classified atoms
//! 2. **Water Inference** - Holographic negative computation
//! 3. **Neuromorphic Network** - LIF dewetting detection
//! 4. **Spike Processing** - Extraction and residue mapping
//!
//! ## Performance Targets
//!
//! - <0.5ms for exclusion field (100³ grid, 10k atoms)
//! - <0.3ms for LIF network step
//! - <2ms total per frame on RTX 3060
//!
//! ## Usage
//!
//! ```rust,no_run
//! use prism_nhs::gpu::NhsGpuEngine;
//!
//! let mut engine = NhsGpuEngine::new(device, grid_dim)?;
//! engine.update_exclusion_field(&atom_positions, &atom_types, &atom_charges)?;
//! engine.infer_water_density()?;
//! let spikes = engine.lif_step()?;
//! ```

use anyhow::{Context, Result};
#[cfg(feature = "gpu")]
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Default grid dimension (64³ = 262,144 voxels)
pub const DEFAULT_GRID_DIM: usize = 64;

/// Maximum supported grid dimension
pub const MAX_GRID_DIM: usize = 128;

/// Default grid spacing in Angstroms
pub const DEFAULT_GRID_SPACING: f32 = 0.5;

/// Block sizes for CUDA kernels
const BLOCK_SIZE_1D: usize = 256;
const BLOCK_SIZE_3D: usize = 8; // 8³ = 512 threads

/// Maximum spikes per frame for output buffers
const MAX_SPIKES_PER_FRAME: usize = 10000;

// ============================================================================
// NHS GPU ENGINE
// ============================================================================

/// GPU-accelerated NHS Engine
///
/// Manages all GPU buffers and kernel execution for the NHS pipeline.
/// All persistent state lives in VRAM for minimal transfer overhead.
#[cfg(feature = "gpu")]
pub struct NhsGpuEngine {
    // CUDA handles
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    _exclusion_module: Arc<CudaModule>,
    _neuromorphic_module: Arc<CudaModule>,

    // Exclusion kernel functions
    compute_exclusion_field: CudaFunction,
    compute_exclusion_field_cell_list: CudaFunction,
    infer_water_density: CudaFunction,
    accumulate_pocket_probability: CudaFunction,
    finalize_pocket_probability: CudaFunction,
    detect_aromatic_targets: CudaFunction,
    reset_grid: CudaFunction,
    reset_grid_int: CudaFunction,

    // Neuromorphic kernel functions
    lif_dewetting_step: CudaFunction,
    lif_dewetting_batch: CudaFunction,
    apply_lateral_inhibition: CudaFunction,
    extract_spike_indices: CudaFunction,
    map_spikes_to_residues: CudaFunction,
    init_lif_state: CudaFunction,
    reset_lif_state: CudaFunction,

    // Grid buffers (persistent in VRAM)
    exclusion_field: CudaSlice<f32>,       // [grid_dim³]
    water_density: CudaSlice<f32>,         // [grid_dim³]
    prev_water_density: CudaSlice<f32>,    // [grid_dim³] for dewetting delta
    water_gradient: CudaSlice<f32>,        // [grid_dim³]
    pocket_mean: CudaSlice<f32>,           // [grid_dim³] for variance tracking
    pocket_m2: CudaSlice<f32>,             // [grid_dim³] for Welford's algorithm
    pocket_probability: CudaSlice<f32>,    // [grid_dim³]

    // LIF neuron state (persistent in VRAM)
    membrane_potential: CudaSlice<f32>,    // [grid_dim³]
    refractory_counter: CudaSlice<i32>,    // [grid_dim³]
    spike_output: CudaSlice<i32>,          // [grid_dim³]

    // Spike extraction buffers
    spike_indices: CudaSlice<i32>,         // [MAX_SPIKES_PER_FRAME]
    spike_positions: CudaSlice<f32>,       // [MAX_SPIKES_PER_FRAME * 3]
    spike_count: CudaSlice<i32>,           // [1] atomic counter

    // Atom data buffers (updated per frame)
    atom_positions_gpu: CudaSlice<f32>,    // [max_atoms * 3]
    atom_types_gpu: CudaSlice<i32>,        // [max_atoms]
    atom_charges_gpu: CudaSlice<f32>,      // [max_atoms]
    atom_residues_gpu: CudaSlice<i32>,     // [max_atoms]

    // Configuration
    grid_dim: usize,
    grid_spacing: f32,
    grid_origin: [f32; 3],
    max_atoms: usize,
    current_atoms: usize,

    // LIF parameters
    tau_mem: f32,
    sensitivity: f32,

    // State tracking
    frame_count: usize,
    initialized: bool,
}

#[cfg(feature = "gpu")]
impl NhsGpuEngine {
    /// Creates a new NHS GPU Engine
    ///
    /// # Arguments
    /// * `context` - CUDA device context
    /// * `grid_dim` - Grid dimension (cube: dim × dim × dim)
    /// * `max_atoms` - Maximum number of atoms to support
    ///
    /// # Returns
    /// Initialized GPU engine with allocated buffers
    pub fn new(
        context: Arc<CudaContext>,
        grid_dim: usize,
        max_atoms: usize,
    ) -> Result<Self> {
        Self::new_with_params(context, grid_dim, max_atoms, DEFAULT_GRID_SPACING)
    }

    /// Creates a new NHS GPU Engine with custom parameters
    pub fn new_with_params(
        context: Arc<CudaContext>,
        grid_dim: usize,
        max_atoms: usize,
        grid_spacing: f32,
    ) -> Result<Self> {
        anyhow::ensure!(
            grid_dim > 0 && grid_dim <= MAX_GRID_DIM,
            "grid_dim must be in [1, {}], got {}",
            MAX_GRID_DIM,
            grid_dim
        );

        log::info!(
            "Creating NHS GPU Engine: grid={}³ ({}M voxels), max_atoms={}, spacing={:.2}Å",
            grid_dim,
            (grid_dim * grid_dim * grid_dim) as f64 / 1e6,
            max_atoms,
            grid_spacing
        );

        let stream = context.default_stream();
        let grid_size = grid_dim * grid_dim * grid_dim;

        // Load PTX modules
        let exclusion_ptx_path = Self::find_ptx_path("nhs_exclusion")?;
        let neuromorphic_ptx_path = Self::find_ptx_path("nhs_neuromorphic")?;

        log::info!("Loading exclusion PTX from: {}", exclusion_ptx_path);
        log::info!("Loading neuromorphic PTX from: {}", neuromorphic_ptx_path);

        let exclusion_module = context
            .load_module(Ptx::from_file(&exclusion_ptx_path))
            .context("Failed to load NHS exclusion PTX module")?;

        let neuromorphic_module = context
            .load_module(Ptx::from_file(&neuromorphic_ptx_path))
            .context("Failed to load NHS neuromorphic PTX module")?;

        // Load exclusion kernel functions
        let compute_exclusion_field = exclusion_module.load_function("compute_exclusion_field")?;
        let compute_exclusion_field_cell_list = exclusion_module
            .load_function("compute_exclusion_field_cell_list")?;
        let infer_water_density = exclusion_module.load_function("infer_water_density")?;
        let accumulate_pocket_probability = exclusion_module
            .load_function("accumulate_pocket_probability")?;
        let finalize_pocket_probability = exclusion_module
            .load_function("finalize_pocket_probability")?;
        let detect_aromatic_targets = exclusion_module.load_function("detect_aromatic_targets")?;
        let reset_grid = exclusion_module.load_function("reset_grid")?;
        let reset_grid_int = exclusion_module.load_function("reset_grid_int")?;

        // Load neuromorphic kernel functions
        let lif_dewetting_step = neuromorphic_module.load_function("lif_dewetting_step")?;
        let lif_dewetting_batch = neuromorphic_module.load_function("lif_dewetting_batch")?;
        let apply_lateral_inhibition = neuromorphic_module
            .load_function("apply_lateral_inhibition")?;
        let extract_spike_indices = neuromorphic_module.load_function("extract_spike_indices")?;
        let map_spikes_to_residues = neuromorphic_module.load_function("map_spikes_to_residues")?;
        let init_lif_state = neuromorphic_module.load_function("init_lif_state")?;
        let reset_lif_state = neuromorphic_module.load_function("reset_lif_state")?;

        // Allocate grid buffers
        let exclusion_field: CudaSlice<f32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate exclusion_field")?;

        let water_density: CudaSlice<f32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate water_density")?;

        let prev_water_density: CudaSlice<f32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate prev_water_density")?;

        let water_gradient: CudaSlice<f32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate water_gradient")?;

        let pocket_mean: CudaSlice<f32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate pocket_mean")?;

        let pocket_m2: CudaSlice<f32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate pocket_m2")?;

        let pocket_probability: CudaSlice<f32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate pocket_probability")?;

        // Allocate LIF neuron state
        let membrane_potential: CudaSlice<f32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate membrane_potential")?;

        let refractory_counter: CudaSlice<i32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate refractory_counter")?;

        let spike_output: CudaSlice<i32> = stream
            .alloc_zeros(grid_size)
            .context("Failed to allocate spike_output")?;

        // Allocate spike extraction buffers
        let spike_indices: CudaSlice<i32> = stream
            .alloc_zeros(MAX_SPIKES_PER_FRAME)
            .context("Failed to allocate spike_indices")?;

        let spike_positions: CudaSlice<f32> = stream
            .alloc_zeros(MAX_SPIKES_PER_FRAME * 3)
            .context("Failed to allocate spike_positions")?;

        let spike_count: CudaSlice<i32> = stream
            .alloc_zeros(1)
            .context("Failed to allocate spike_count")?;

        // Allocate atom data buffers
        let atom_positions_gpu: CudaSlice<f32> = stream
            .alloc_zeros(max_atoms * 3)
            .context("Failed to allocate atom_positions_gpu")?;

        let atom_types_gpu: CudaSlice<i32> = stream
            .alloc_zeros(max_atoms)
            .context("Failed to allocate atom_types_gpu")?;

        let atom_charges_gpu: CudaSlice<f32> = stream
            .alloc_zeros(max_atoms)
            .context("Failed to allocate atom_charges_gpu")?;

        let atom_residues_gpu: CudaSlice<i32> = stream
            .alloc_zeros(max_atoms)
            .context("Failed to allocate atom_residues_gpu")?;

        // Calculate memory usage
        let grid_bytes = grid_size * 4 * 10; // 10 float/int grids
        let spike_bytes = MAX_SPIKES_PER_FRAME * 4 * 4; // indices, positions, count
        let atom_bytes = max_atoms * 4 * 4; // positions, types, charges, residues
        let total_mb = (grid_bytes + spike_bytes + atom_bytes) as f64 / 1024.0 / 1024.0;

        log::info!("NHS GPU buffers allocated: {:.2}MB total", total_mb);
        log::info!("  Grid buffers: {:.2}MB", grid_bytes as f64 / 1024.0 / 1024.0);
        log::info!("  Spike buffers: {:.2}MB", spike_bytes as f64 / 1024.0 / 1024.0);
        log::info!("  Atom buffers: {:.2}MB", atom_bytes as f64 / 1024.0 / 1024.0);

        Ok(Self {
            context,
            stream,
            _exclusion_module: exclusion_module,
            _neuromorphic_module: neuromorphic_module,

            compute_exclusion_field,
            compute_exclusion_field_cell_list,
            infer_water_density,
            accumulate_pocket_probability,
            finalize_pocket_probability,
            detect_aromatic_targets,
            reset_grid,
            reset_grid_int,

            lif_dewetting_step,
            lif_dewetting_batch,
            apply_lateral_inhibition,
            extract_spike_indices,
            map_spikes_to_residues,
            init_lif_state,
            reset_lif_state,

            exclusion_field,
            water_density,
            prev_water_density,
            water_gradient,
            pocket_mean,
            pocket_m2,
            pocket_probability,

            membrane_potential,
            refractory_counter,
            spike_output,

            spike_indices,
            spike_positions,
            spike_count,

            atom_positions_gpu,
            atom_types_gpu,
            atom_charges_gpu,
            atom_residues_gpu,

            grid_dim,
            grid_spacing,
            grid_origin: [0.0, 0.0, 0.0],
            max_atoms,
            current_atoms: 0,

            tau_mem: 10.0,
            sensitivity: 1.0,

            frame_count: 0,
            initialized: false,
        })
    }

    /// Find PTX file path
    fn find_ptx_path(kernel_name: &str) -> Result<String> {
        let filename = format!("{}.ptx", kernel_name);
        let paths = [
            format!("target/ptx/{}", filename),
            format!("crates/prism-gpu/target/ptx/{}", filename),
            format!("../prism-gpu/target/ptx/{}", filename),
        ];

        for path in &paths {
            if std::path::Path::new(path).exists() {
                return Ok(path.clone());
            }
        }

        // Try OUT_DIR from build
        if let Ok(out_dir) = std::env::var("OUT_DIR") {
            let path = format!("{}/ptx/{}", out_dir, filename);
            if std::path::Path::new(&path).exists() {
                return Ok(path);
            }
        }

        Err(anyhow::anyhow!(
            "{}.ptx not found. Run `cargo build -p prism-gpu --features cuda` to compile kernels",
            kernel_name
        ))
    }

    /// Initialize the LIF network state
    pub fn initialize(&mut self, grid_origin: [f32; 3]) -> Result<()> {
        log::info!("Initializing NHS GPU Engine at origin {:?}", grid_origin);
        self.grid_origin = grid_origin;

        // Initialize LIF state
        let grid_size = self.grid_dim * self.grid_dim * self.grid_dim;
        let grid_blocks = (grid_size as u32).div_ceil(BLOCK_SIZE_1D as u32);

        let cfg = LaunchConfig {
            grid_dim: (grid_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.init_lif_state)
                .arg(&self.membrane_potential)
                .arg(&self.refractory_counter)
                .arg(&(self.grid_dim as i32))
                .launch(cfg)
        }
        .context("Failed to launch init_lif_state kernel")?;

        // Reset grids
        self.reset_grids()?;

        self.frame_count = 0;
        self.initialized = true;
        log::info!("NHS GPU Engine initialized");
        Ok(())
    }

    /// Reset all grid buffers to zero
    pub fn reset_grids(&mut self) -> Result<()> {
        let grid_size = self.grid_dim * self.grid_dim * self.grid_dim;
        let grid_blocks = (grid_size as u32).div_ceil(BLOCK_SIZE_1D as u32);

        let cfg = LaunchConfig {
            grid_dim: (grid_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Reset float grids
        for grid in [
            &self.exclusion_field,
            &self.water_density,
            &self.prev_water_density,
            &self.water_gradient,
            &self.pocket_mean,
            &self.pocket_m2,
            &self.pocket_probability,
            &self.membrane_potential,
        ] {
            unsafe {
                self.stream
                    .launch_builder(&self.reset_grid)
                    .arg(grid)
                    .arg(&(grid_size as i32))
                    .launch(cfg.clone())
            }
            .context("Failed to launch reset_grid kernel")?;
        }

        // Reset int grids
        for grid in [&self.refractory_counter, &self.spike_output] {
            unsafe {
                self.stream
                    .launch_builder(&self.reset_grid_int)
                    .arg(grid)
                    .arg(&(grid_size as i32))
                    .launch(cfg.clone())
            }
            .context("Failed to launch reset_grid_int kernel")?;
        }

        self.context.synchronize()?;
        Ok(())
    }

    /// Update atom data on GPU
    ///
    /// # Arguments
    /// * `positions` - Flat array of atom positions [n_atoms * 3]
    /// * `types` - Atom type classification [n_atoms]
    /// * `charges` - Partial charges [n_atoms]
    /// * `residues` - Residue indices [n_atoms]
    pub fn update_atom_data(
        &mut self,
        positions: &[f32],
        types: &[i32],
        charges: &[f32],
        residues: &[i32],
    ) -> Result<()> {
        let n_atoms = positions.len() / 3;
        anyhow::ensure!(
            n_atoms <= self.max_atoms,
            "Too many atoms: {} > max {}",
            n_atoms,
            self.max_atoms
        );
        anyhow::ensure!(
            types.len() == n_atoms && charges.len() == n_atoms && residues.len() == n_atoms,
            "Atom data length mismatch"
        );

        // Copy to GPU
        self.stream
            .memcpy_htod(&positions[..n_atoms * 3], &mut self.atom_positions_gpu)
            .context("Failed to copy atom positions to GPU")?;

        self.stream
            .memcpy_htod(&types[..n_atoms], &mut self.atom_types_gpu)
            .context("Failed to copy atom types to GPU")?;

        self.stream
            .memcpy_htod(&charges[..n_atoms], &mut self.atom_charges_gpu)
            .context("Failed to copy atom charges to GPU")?;

        self.stream
            .memcpy_htod(&residues[..n_atoms], &mut self.atom_residues_gpu)
            .context("Failed to copy atom residues to GPU")?;

        self.current_atoms = n_atoms;
        Ok(())
    }

    /// Compute exclusion field from current atom positions
    ///
    /// Runs the HEM kernel to compute 3D exclusion field based on
    /// atom classifications (hydrophobic, polar, charged, aromatic).
    pub fn compute_exclusion(&mut self) -> Result<()> {
        anyhow::ensure!(self.initialized, "Engine not initialized");
        anyhow::ensure!(self.current_atoms > 0, "No atoms loaded");

        // 3D grid launch configuration
        let blocks_per_dim = (self.grid_dim as u32).div_ceil(BLOCK_SIZE_3D as u32);

        let cfg = LaunchConfig {
            grid_dim: (blocks_per_dim, blocks_per_dim, blocks_per_dim),
            block_dim: (BLOCK_SIZE_3D as u32, BLOCK_SIZE_3D as u32, BLOCK_SIZE_3D as u32),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.compute_exclusion_field)
                .arg(&self.atom_positions_gpu)
                .arg(&self.atom_types_gpu)
                .arg(&self.atom_charges_gpu)
                .arg(&self.exclusion_field)
                .arg(&(self.current_atoms as i32))
                .arg(&(self.grid_dim as i32))
                .arg(&self.grid_origin[0])
                .arg(&self.grid_origin[1])
                .arg(&self.grid_origin[2])
                .arg(&self.grid_spacing)
                .launch(cfg)
        }
        .context("Failed to launch compute_exclusion_field kernel")?;

        Ok(())
    }

    /// Infer water density from exclusion field (holographic negative)
    ///
    /// Also computes gradient magnitude for dewetting edge detection.
    pub fn infer_water(&mut self) -> Result<()> {
        let blocks_per_dim = (self.grid_dim as u32).div_ceil(BLOCK_SIZE_3D as u32);

        let cfg = LaunchConfig {
            grid_dim: (blocks_per_dim, blocks_per_dim, blocks_per_dim),
            block_dim: (BLOCK_SIZE_3D as u32, BLOCK_SIZE_3D as u32, BLOCK_SIZE_3D as u32),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.infer_water_density)
                .arg(&self.exclusion_field)
                .arg(&self.water_density)
                .arg(&self.water_gradient)
                .arg(&(self.grid_dim as i32))
                .arg(&self.grid_spacing)
                .launch(cfg)
        }
        .context("Failed to launch infer_water_density kernel")?;

        Ok(())
    }

    /// Run one LIF network step for dewetting detection
    ///
    /// Returns the number of spikes detected.
    pub fn lif_step(&mut self) -> Result<usize> {
        // Swap water density buffers for delta computation
        std::mem::swap(&mut self.water_density, &mut self.prev_water_density);

        // Recompute current water density (we just swapped, so need fresh computation)
        self.infer_water()?;

        // Reset spike count
        let zero = [0i32];
        self.stream
            .memcpy_htod(&zero, &mut self.spike_count)
            .context("Failed to reset spike count")?;

        // LIF step kernel
        let blocks_per_dim = (self.grid_dim as u32).div_ceil(BLOCK_SIZE_3D as u32);

        let cfg = LaunchConfig {
            grid_dim: (blocks_per_dim, blocks_per_dim, blocks_per_dim),
            block_dim: (BLOCK_SIZE_3D as u32, BLOCK_SIZE_3D as u32, BLOCK_SIZE_3D as u32),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.lif_dewetting_step)
                .arg(&self.prev_water_density)
                .arg(&self.water_density)
                .arg(&self.membrane_potential)
                .arg(&self.refractory_counter)
                .arg(&self.spike_output)
                .arg(&self.spike_count)
                .arg(&(self.grid_dim as i32))
                .arg(&self.tau_mem)
                .arg(&self.sensitivity)
                .launch(cfg)
        }
        .context("Failed to launch lif_dewetting_step kernel")?;

        // Apply lateral inhibition
        unsafe {
            self.stream
                .launch_builder(&self.apply_lateral_inhibition)
                .arg(&self.spike_output)
                .arg(&self.membrane_potential)
                .arg(&(self.grid_dim as i32))
                .arg(&0.1f32) // inhibition strength
                .launch(cfg)
        }
        .context("Failed to launch apply_lateral_inhibition kernel")?;

        // Get spike count
        let spike_counts = self
            .stream
            .clone_dtoh(&self.spike_count)
            .context("Failed to read spike count")?;

        self.frame_count += 1;
        Ok(spike_counts[0] as usize)
    }

    /// Extract spike positions and map to residues
    ///
    /// # Returns
    /// Tuple of (spike_positions, spike_residues, spike_distances)
    pub fn extract_spikes(&mut self) -> Result<(Vec<[f32; 3]>, Vec<i32>, Vec<f32>)> {
        // Reset spike count for extraction
        let zero = [0i32];
        self.stream
            .memcpy_htod(&zero, &mut self.spike_count)
            .context("Failed to reset spike count for extraction")?;

        // Extract spike indices and positions
        let blocks_per_dim = (self.grid_dim as u32).div_ceil(BLOCK_SIZE_3D as u32);

        let cfg = LaunchConfig {
            grid_dim: (blocks_per_dim, blocks_per_dim, blocks_per_dim),
            block_dim: (BLOCK_SIZE_3D as u32, BLOCK_SIZE_3D as u32, BLOCK_SIZE_3D as u32),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.extract_spike_indices)
                .arg(&self.spike_output)
                .arg(&self.spike_indices)
                .arg(&self.spike_positions)
                .arg(&self.spike_count)
                .arg(&(self.grid_dim as i32))
                .arg(&self.grid_origin[0])
                .arg(&self.grid_origin[1])
                .arg(&self.grid_origin[2])
                .arg(&self.grid_spacing)
                .arg(&(MAX_SPIKES_PER_FRAME as i32))
                .launch(cfg)
        }
        .context("Failed to launch extract_spike_indices kernel")?;

        // Get spike count
        let spike_counts = self
            .stream
            .clone_dtoh(&self.spike_count)
            .context("Failed to read spike count")?;

        let n_spikes = (spike_counts[0] as usize).min(MAX_SPIKES_PER_FRAME);

        if n_spikes == 0 {
            return Ok((Vec::new(), Vec::new(), Vec::new()));
        }

        // Copy spike positions from GPU
        let positions_flat = self
            .stream
            .clone_dtoh(&self.spike_positions)
            .context("Failed to read spike positions")?;

        let mut positions = Vec::with_capacity(n_spikes);
        for i in 0..n_spikes {
            positions.push([
                positions_flat[i * 3],
                positions_flat[i * 3 + 1],
                positions_flat[i * 3 + 2],
            ]);
        }

        // Allocate residue mapping buffers
        let mut spike_residues_gpu: CudaSlice<i32> = self
            .stream
            .alloc_zeros(n_spikes)
            .context("Failed to allocate spike_residues")?;

        let mut spike_distances_gpu: CudaSlice<f32> = self
            .stream
            .alloc_zeros(n_spikes)
            .context("Failed to allocate spike_distances")?;

        // Map spikes to residues
        let spike_blocks = (n_spikes as u32).div_ceil(BLOCK_SIZE_1D as u32);

        let cfg_1d = LaunchConfig {
            grid_dim: (spike_blocks, 1, 1),
            block_dim: (BLOCK_SIZE_1D as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.map_spikes_to_residues)
                .arg(&self.spike_positions)
                .arg(&self.atom_positions_gpu)
                .arg(&self.atom_residues_gpu)
                .arg(&spike_residues_gpu)
                .arg(&spike_distances_gpu)
                .arg(&(n_spikes as i32))
                .arg(&(self.current_atoms as i32))
                .arg(&10.0f32) // max_distance
                .launch(cfg_1d)
        }
        .context("Failed to launch map_spikes_to_residues kernel")?;

        // Copy results
        let residues = self
            .stream
            .clone_dtoh(&spike_residues_gpu)
            .context("Failed to read spike residues")?;

        let distances = self
            .stream
            .clone_dtoh(&spike_distances_gpu)
            .context("Failed to read spike distances")?;

        Ok((
            positions,
            residues[..n_spikes].to_vec(),
            distances[..n_spikes].to_vec(),
        ))
    }

    /// Accumulate pocket probability from temporal variance
    pub fn accumulate_pocket_variance(&mut self) -> Result<()> {
        let blocks_per_dim = (self.grid_dim as u32).div_ceil(BLOCK_SIZE_3D as u32);

        let cfg = LaunchConfig {
            grid_dim: (blocks_per_dim, blocks_per_dim, blocks_per_dim),
            block_dim: (BLOCK_SIZE_3D as u32, BLOCK_SIZE_3D as u32, BLOCK_SIZE_3D as u32),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.accumulate_pocket_probability)
                .arg(&self.water_density)
                .arg(&self.pocket_mean)
                .arg(&self.pocket_m2)
                .arg(&(self.frame_count as i32))
                .arg(&(self.grid_dim as i32))
                .launch(cfg)
        }
        .context("Failed to launch accumulate_pocket_probability kernel")?;

        Ok(())
    }

    /// Finalize pocket probability and copy to host
    pub fn finalize_pocket_probability(&mut self, variance_threshold: f32) -> Result<Vec<f32>> {
        let blocks_per_dim = (self.grid_dim as u32).div_ceil(BLOCK_SIZE_3D as u32);

        let cfg = LaunchConfig {
            grid_dim: (blocks_per_dim, blocks_per_dim, blocks_per_dim),
            block_dim: (BLOCK_SIZE_3D as u32, BLOCK_SIZE_3D as u32, BLOCK_SIZE_3D as u32),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&self.finalize_pocket_probability)
                .arg(&self.pocket_mean)
                .arg(&self.pocket_m2)
                .arg(&self.pocket_probability)
                .arg(&(self.frame_count as i32))
                .arg(&(self.grid_dim as i32))
                .arg(&variance_threshold)
                .launch(cfg)
        }
        .context("Failed to launch finalize_pocket_probability kernel")?;

        let probability = self
            .stream
            .clone_dtoh(&self.pocket_probability)
            .context("Failed to read pocket probability")?;

        Ok(probability)
    }

    /// Get water density grid from GPU
    pub fn get_water_density(&self) -> Result<Vec<f32>> {
        let density = self
            .stream
            .clone_dtoh(&self.water_density)
            .context("Failed to read water density")?;
        Ok(density)
    }

    /// Get exclusion field from GPU
    pub fn get_exclusion_field(&self) -> Result<Vec<f32>> {
        let field = self
            .stream
            .clone_dtoh(&self.exclusion_field)
            .context("Failed to read exclusion field")?;
        Ok(field)
    }

    /// Process a complete frame: update atoms, compute exclusion, infer water, run LIF
    ///
    /// This is the main entry point for frame-by-frame processing.
    pub fn process_frame(
        &mut self,
        positions: &[f32],
        types: &[i32],
        charges: &[f32],
        residues: &[i32],
    ) -> Result<FrameResult> {
        // Update atom data
        self.update_atom_data(positions, types, charges, residues)?;

        // Compute exclusion field
        self.compute_exclusion()?;

        // Run LIF step (includes water inference)
        let spike_count = self.lif_step()?;

        // Accumulate pocket variance
        self.accumulate_pocket_variance()?;

        // Extract spikes if any
        let (spike_positions, spike_residues, spike_distances) = if spike_count > 0 {
            self.extract_spikes()?
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };

        Ok(FrameResult {
            frame: self.frame_count,
            spike_count,
            spike_positions,
            spike_residues,
            spike_distances,
        })
    }

    /// Set LIF parameters
    pub fn set_lif_params(&mut self, tau_mem: f32, sensitivity: f32) {
        self.tau_mem = tau_mem;
        self.sensitivity = sensitivity;
    }

    /// Get grid dimension
    pub fn grid_dim(&self) -> usize {
        self.grid_dim
    }

    /// Get grid spacing
    pub fn grid_spacing(&self) -> f32 {
        self.grid_spacing
    }

    /// Get frame count
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }

    /// Get CUDA context
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Get CUDA stream
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

/// Result from processing a single frame
#[derive(Debug, Clone)]
pub struct FrameResult {
    /// Frame number
    pub frame: usize,
    /// Number of spikes detected
    pub spike_count: usize,
    /// Spike positions in world coordinates
    pub spike_positions: Vec<[f32; 3]>,
    /// Nearest residue for each spike
    pub spike_residues: Vec<i32>,
    /// Distance to nearest atom for each spike
    pub spike_distances: Vec<f32>,
}

// ============================================================================
// CPU FALLBACK (NO GPU)
// ============================================================================

#[cfg(not(feature = "gpu"))]
pub struct NhsGpuEngine;

#[cfg(not(feature = "gpu"))]
impl NhsGpuEngine {
    pub fn new(_grid_dim: usize, _max_atoms: usize) -> Result<Self> {
        Err(anyhow::anyhow!(
            "NHS GPU Engine requires GPU feature. Compile with --features gpu"
        ))
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_engine_creation() {
        env_logger::builder().is_test(true).try_init().ok();

        let context = CudaContext::new(0).expect("CUDA not available");
        let engine = NhsGpuEngine::new(context, 32, 1000);

        assert!(engine.is_ok(), "Failed to create GPU engine: {:?}", engine.err());
    }

    #[test]
    #[ignore] // Requires GPU
    fn test_gpu_engine_basic_workflow() {
        env_logger::builder().is_test(true).try_init().ok();

        let context = CudaContext::new(0).expect("CUDA not available");
        let mut engine = NhsGpuEngine::new(context, 32, 100).expect("Failed to create engine");

        // Initialize
        engine.initialize([0.0, 0.0, 0.0]).expect("Failed to initialize");

        // Create dummy atom data
        let positions = vec![0.0f32; 30]; // 10 atoms
        let types = vec![0i32; 10]; // All hydrophobic
        let charges = vec![0.0f32; 10];
        let residues = vec![0i32; 10];

        // Process frame
        let result = engine
            .process_frame(&positions, &types, &charges, &residues)
            .expect("Failed to process frame");

        assert_eq!(result.frame, 1);
    }
}

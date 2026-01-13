//! Mega-Fused AMBER HMC Dynamics
//!
//! Complete molecular dynamics in a single GPU kernel launch.
//! Uses flat arrays for GPU transfer compatibility.

use anyhow::{Context, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceSlice,
    LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::collections::HashSet;
use std::sync::Arc;

/// Maximum exclusions per atom (1-2, 1-3 bonded pairs)
pub const MAX_EXCLUSIONS: usize = 32;

/// Maximum 1-4 pairs per atom (dihedral terminal pairs needing scaled interactions)
pub const MAX_14_PAIRS: usize = 16;

/// Cell list constants (must match CUDA kernel)
pub const CELL_SIZE: f32 = 10.0;
pub const MAX_CELLS_PER_DIM: usize = 32;
pub const MAX_TOTAL_CELLS: usize = MAX_CELLS_PER_DIM * MAX_CELLS_PER_DIM * MAX_CELLS_PER_DIM;
pub const MAX_ATOMS_PER_CELL: usize = 128;
pub const NEIGHBOR_LIST_SIZE: usize = 256;

/// HMC run results
#[derive(Debug, Clone)]
pub struct HmcRunResult {
    pub potential_energy: f64,
    pub kinetic_energy: f64,
    pub positions: Vec<f32>,
    pub velocities: Vec<f32>,
    pub avg_temperature: f64,
}

/// Mega-fused AMBER HMC simulator
pub struct AmberMegaFusedHmc {
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    module: Arc<CudaModule>,

    // Kernels
    hmc_step_kernel: CudaFunction,
    zero_energies_kernel: CudaFunction,
    thermostat_kernel: CudaFunction,
    init_velocities_kernel: CudaFunction,
    minimize_kernel: CudaFunction,

    // Cell list kernels (O(N) non-bonded)
    build_cell_list_kernel: CudaFunction,
    build_neighbor_list_kernel: CudaFunction,

    // Device buffers - State (as flat f32 arrays)
    d_positions: CudaSlice<f32>,      // [n_atoms * 3]
    d_velocities: CudaSlice<f32>,     // [n_atoms * 3]
    d_forces: CudaSlice<f32>,         // [n_atoms * 3]
    d_total_energy: CudaSlice<f32>,   // [1]
    d_kinetic_energy: CudaSlice<f32>, // [1]

    // Device buffers - Topology (as flat primitive arrays)
    d_bond_atoms: CudaSlice<i32>,     // [n_bonds * 2] (i, j)
    d_bond_params: CudaSlice<f32>,    // [n_bonds * 2] (k, r0)
    d_angle_atoms: CudaSlice<i32>,    // [n_angles * 4] (i, j, k, pad)
    d_angle_params: CudaSlice<f32>,   // [n_angles * 2] (k, theta0)
    d_dihedral_atoms: CudaSlice<i32>, // [n_dihedrals * 4] (i, j, k, l)
    d_dihedral_params: CudaSlice<f32>,// [n_dihedrals * 4] (k, n, phase, pad)

    // Device buffers - Non-bonded
    d_nb_sigma: CudaSlice<f32>,       // [n_atoms]
    d_nb_epsilon: CudaSlice<f32>,     // [n_atoms]
    d_nb_charge: CudaSlice<f32>,      // [n_atoms]
    d_nb_mass: CudaSlice<f32>,        // [n_atoms]
    d_exclusion_list: CudaSlice<i32>, // [n_atoms * MAX_EXCLUSIONS]
    d_n_exclusions: CudaSlice<i32>,   // [n_atoms]

    // Device buffers - 1-4 pairs (scaled non-bonded: LJ*0.5, Coulomb*0.833)
    d_pair14_list: CudaSlice<i32>,    // [n_atoms * MAX_14_PAIRS]
    d_n_pairs14: CudaSlice<i32>,      // [n_atoms]

    // Device buffers - Cell lists (O(N) non-bonded)
    d_cell_list: CudaSlice<i32>,      // [MAX_TOTAL_CELLS * MAX_ATOMS_PER_CELL]
    d_cell_counts: CudaSlice<i32>,    // [MAX_TOTAL_CELLS]
    d_atom_cell: CudaSlice<i32>,      // [n_atoms]
    d_neighbor_list: CudaSlice<i32>,  // [n_atoms * NEIGHBOR_LIST_SIZE]
    d_n_neighbors: CudaSlice<i32>,    // [n_atoms]
    d_bbox_min: CudaSlice<f32>,       // [3]
    d_bbox_max: CudaSlice<f32>,       // [3]

    // Cell grid dimensions (computed from bounding box)
    grid_origin: [f32; 3],
    grid_dims: [i32; 3],

    // Sizes
    n_atoms: usize,
    n_bonds: usize,
    n_angles: usize,
    n_dihedrals: usize,

    // Flags
    topology_ready: bool,
    velocities_initialized: bool,
    neighbor_list_valid: bool,
}

impl AmberMegaFusedHmc {
    /// Create a new mega-fused AMBER HMC simulator
    pub fn new(context: Arc<CudaContext>, n_atoms: usize) -> Result<Self> {
        log::info!("🚀 Initializing Mega-Fused AMBER HMC for {} atoms", n_atoms);

        let stream = context.default_stream();

        // Load PTX module
        let ptx_path = "crates/prism-gpu/target/ptx/amber_mega_fused.ptx";
        let ptx = Ptx::from_file(ptx_path);
        let module = context
            .load_module(ptx)
            .with_context(|| format!("Failed to load mega-fused PTX from {}", ptx_path))?;

        // Load kernels
        let hmc_step_kernel = module
            .load_function("amber_mega_fused_hmc_step")
            .context("Failed to load amber_mega_fused_hmc_step")?;
        let zero_energies_kernel = module
            .load_function("zero_energies")
            .context("Failed to load zero_energies")?;
        let thermostat_kernel = module
            .load_function("apply_thermostat")
            .context("Failed to load apply_thermostat")?;
        let init_velocities_kernel = module
            .load_function("initialize_velocities")
            .context("Failed to load initialize_velocities")?;
        let minimize_kernel = module
            .load_function("amber_steepest_descent_step")
            .context("Failed to load amber_steepest_descent_step")?;

        // Load cell list kernels for O(N) non-bonded
        let build_cell_list_kernel = module
            .load_function("build_cell_list")
            .context("Failed to load build_cell_list")?;
        let build_neighbor_list_kernel = module
            .load_function("build_neighbor_list")
            .context("Failed to load build_neighbor_list")?;

        log::info!("📦 Cell list kernels loaded for O(N) non-bonded");

        // Allocate state buffers
        let d_positions = stream.alloc_zeros::<f32>(n_atoms * 3)?;
        let d_velocities = stream.alloc_zeros::<f32>(n_atoms * 3)?;
        let d_forces = stream.alloc_zeros::<f32>(n_atoms * 3)?;
        let d_total_energy = stream.alloc_zeros::<f32>(1)?;
        let d_kinetic_energy = stream.alloc_zeros::<f32>(1)?;

        // Allocate topology buffers (minimal initial size)
        let d_bond_atoms = stream.alloc_zeros::<i32>(2)?;
        let d_bond_params = stream.alloc_zeros::<f32>(2)?;
        let d_angle_atoms = stream.alloc_zeros::<i32>(4)?;
        let d_angle_params = stream.alloc_zeros::<f32>(2)?;
        let d_dihedral_atoms = stream.alloc_zeros::<i32>(4)?;
        let d_dihedral_params = stream.alloc_zeros::<f32>(4)?;

        // Allocate NB buffers
        let d_nb_sigma = stream.alloc_zeros::<f32>(n_atoms)?;
        let d_nb_epsilon = stream.alloc_zeros::<f32>(n_atoms)?;
        let d_nb_charge = stream.alloc_zeros::<f32>(n_atoms)?;
        let d_nb_mass = stream.alloc_zeros::<f32>(n_atoms)?;
        let d_exclusion_list = stream.alloc_zeros::<i32>(n_atoms * MAX_EXCLUSIONS)?;
        let d_n_exclusions = stream.alloc_zeros::<i32>(n_atoms)?;

        // Allocate 1-4 pair buffers for scaled non-bonded
        let d_pair14_list = stream.alloc_zeros::<i32>(n_atoms * MAX_14_PAIRS)?;
        let d_n_pairs14 = stream.alloc_zeros::<i32>(n_atoms)?;

        // Allocate cell list buffers for O(N) non-bonded
        let d_cell_list = stream.alloc_zeros::<i32>(MAX_TOTAL_CELLS * MAX_ATOMS_PER_CELL)?;
        let d_cell_counts = stream.alloc_zeros::<i32>(MAX_TOTAL_CELLS)?;
        let d_atom_cell = stream.alloc_zeros::<i32>(n_atoms)?;
        let d_neighbor_list = stream.alloc_zeros::<i32>(n_atoms * NEIGHBOR_LIST_SIZE)?;
        let d_n_neighbors = stream.alloc_zeros::<i32>(n_atoms)?;
        let d_bbox_min = stream.alloc_zeros::<f32>(3)?;
        let d_bbox_max = stream.alloc_zeros::<f32>(3)?;

        log::info!("✅ Mega-Fused AMBER HMC initialized with O(N) cell lists");

        Ok(Self {
            context,
            stream,
            module,
            hmc_step_kernel,
            zero_energies_kernel,
            thermostat_kernel,
            init_velocities_kernel,
            minimize_kernel,
            build_cell_list_kernel,
            build_neighbor_list_kernel,
            d_positions,
            d_velocities,
            d_forces,
            d_total_energy,
            d_kinetic_energy,
            d_bond_atoms,
            d_bond_params,
            d_angle_atoms,
            d_angle_params,
            d_dihedral_atoms,
            d_dihedral_params,
            d_nb_sigma,
            d_nb_epsilon,
            d_nb_charge,
            d_nb_mass,
            d_exclusion_list,
            d_n_exclusions,
            d_pair14_list,
            d_n_pairs14,
            d_cell_list,
            d_cell_counts,
            d_atom_cell,
            d_neighbor_list,
            d_n_neighbors,
            d_bbox_min,
            d_bbox_max,
            grid_origin: [0.0, 0.0, 0.0],
            grid_dims: [1, 1, 1],
            n_atoms,
            n_bonds: 0,
            n_angles: 0,
            n_dihedrals: 0,
            topology_ready: false,
            velocities_initialized: false,
            neighbor_list_valid: false,
        })
    }

    /// Upload topology and initial positions
    ///
    /// # Arguments
    /// * `positions` - Flat array [x0, y0, z0, x1, y1, z1, ...]
    /// * `bonds` - Vec of (atom_i, atom_j, k, r0)
    /// * `angles` - Vec of (atom_i, atom_j, atom_k, k, theta0)
    /// * `dihedrals` - Vec of (atom_i, atom_j, atom_k, atom_l, k, n, phase)
    /// * `nb_params` - Vec of (sigma, epsilon, charge, mass) per atom
    /// * `exclusions` - Per-atom exclusion sets
    pub fn upload_topology(
        &mut self,
        positions: &[f32],
        bonds: &[(usize, usize, f32, f32)],
        angles: &[(usize, usize, usize, f32, f32)],
        dihedrals: &[(usize, usize, usize, usize, f32, f32, f32)],
        nb_params: &[(f32, f32, f32, f32)], // (sigma, epsilon, charge, mass)
        exclusions: &[HashSet<usize>],
    ) -> Result<()> {
        if positions.len() != self.n_atoms * 3 {
            return Err(anyhow::anyhow!(
                "Position count {} != expected {}",
                positions.len(),
                self.n_atoms * 3
            ));
        }

        // Upload positions
        self.stream.memcpy_htod(positions, &mut self.d_positions)?;

        // Upload bonds as flat arrays
        if !bonds.is_empty() {
            let bond_atoms: Vec<i32> = bonds
                .iter()
                .flat_map(|&(i, j, _, _)| [i as i32, j as i32])
                .collect();
            let bond_params: Vec<f32> = bonds
                .iter()
                .flat_map(|&(_, _, k, r0)| [k, r0])
                .collect();

            self.d_bond_atoms = self.stream.alloc_zeros::<i32>(bond_atoms.len())?;
            self.d_bond_params = self.stream.alloc_zeros::<f32>(bond_params.len())?;
            self.stream.memcpy_htod(&bond_atoms, &mut self.d_bond_atoms)?;
            self.stream.memcpy_htod(&bond_params, &mut self.d_bond_params)?;
        }
        self.n_bonds = bonds.len();

        // Upload angles
        if !angles.is_empty() {
            let angle_atoms: Vec<i32> = angles
                .iter()
                .flat_map(|&(i, j, k, _, _)| [i as i32, j as i32, k as i32, 0])
                .collect();
            let angle_params: Vec<f32> = angles
                .iter()
                .flat_map(|&(_, _, _, k, theta0)| [k, theta0])
                .collect();

            self.d_angle_atoms = self.stream.alloc_zeros::<i32>(angle_atoms.len())?;
            self.d_angle_params = self.stream.alloc_zeros::<f32>(angle_params.len())?;
            self.stream.memcpy_htod(&angle_atoms, &mut self.d_angle_atoms)?;
            self.stream.memcpy_htod(&angle_params, &mut self.d_angle_params)?;
        }
        self.n_angles = angles.len();

        // Upload dihedrals
        if !dihedrals.is_empty() {
            let dihedral_atoms: Vec<i32> = dihedrals
                .iter()
                .flat_map(|&(i, j, k, l, _, _, _)| [i as i32, j as i32, k as i32, l as i32])
                .collect();
            let dihedral_params: Vec<f32> = dihedrals
                .iter()
                .flat_map(|&(_, _, _, _, pk, n, phase)| [pk, n, phase, 0.0])
                .collect();

            self.d_dihedral_atoms = self.stream.alloc_zeros::<i32>(dihedral_atoms.len())?;
            self.d_dihedral_params = self.stream.alloc_zeros::<f32>(dihedral_params.len())?;
            self.stream.memcpy_htod(&dihedral_atoms, &mut self.d_dihedral_atoms)?;
            self.stream.memcpy_htod(&dihedral_params, &mut self.d_dihedral_params)?;
        }
        self.n_dihedrals = dihedrals.len();

        // Upload NB parameters as separate arrays
        let sigmas: Vec<f32> = nb_params.iter().map(|&(s, _, _, _)| s).collect();
        let epsilons: Vec<f32> = nb_params.iter().map(|&(_, e, _, _)| e).collect();
        let charges: Vec<f32> = nb_params.iter().map(|&(_, _, c, _)| c).collect();
        let masses: Vec<f32> = nb_params.iter().map(|&(_, _, _, m)| m).collect();

        self.stream.memcpy_htod(&sigmas, &mut self.d_nb_sigma)?;
        self.stream.memcpy_htod(&epsilons, &mut self.d_nb_epsilon)?;
        self.stream.memcpy_htod(&charges, &mut self.d_nb_charge)?;
        self.stream.memcpy_htod(&masses, &mut self.d_nb_mass)?;

        // Flatten and upload exclusions
        let mut excl_flat = vec![-1i32; self.n_atoms * MAX_EXCLUSIONS];
        let mut n_excl = vec![0i32; self.n_atoms];

        for (i, excl_set) in exclusions.iter().enumerate() {
            if i >= self.n_atoms {
                break;
            }
            n_excl[i] = excl_set.len().min(MAX_EXCLUSIONS) as i32;
            for (j, &excl_idx) in excl_set.iter().take(MAX_EXCLUSIONS).enumerate() {
                excl_flat[i * MAX_EXCLUSIONS + j] = excl_idx as i32;
            }
        }

        self.stream.memcpy_htod(&excl_flat, &mut self.d_exclusion_list)?;
        self.stream.memcpy_htod(&n_excl, &mut self.d_n_exclusions)?;

        // Build and upload 1-4 pairs from dihedrals
        // These get SCALED non-bonded interactions (LJ*0.5, Coulomb*0.833)
        let pairs_14 = build_14_pairs(dihedrals, exclusions, self.n_atoms);

        // Convert to per-atom format (like exclusions)
        let mut pair14_flat = vec![-1i32; self.n_atoms * MAX_14_PAIRS];
        let mut n_pairs14 = vec![0i32; self.n_atoms];

        for &(i, j) in &pairs_14 {
            // Add j to i's 1-4 list
            let count_i = n_pairs14[i] as usize;
            if count_i < MAX_14_PAIRS {
                pair14_flat[i * MAX_14_PAIRS + count_i] = j as i32;
                n_pairs14[i] += 1;
            }
            // Add i to j's 1-4 list (symmetric)
            let count_j = n_pairs14[j] as usize;
            if count_j < MAX_14_PAIRS {
                pair14_flat[j * MAX_14_PAIRS + count_j] = i as i32;
                n_pairs14[j] += 1;
            }
        }

        self.stream.memcpy_htod(&pair14_flat, &mut self.d_pair14_list)?;
        self.stream.memcpy_htod(&n_pairs14, &mut self.d_n_pairs14)?;

        self.topology_ready = true;

        log::info!(
            "📤 Topology uploaded: {} bonds, {} angles, {} dihedrals, {} 1-4 pairs",
            self.n_bonds,
            self.n_angles,
            self.n_dihedrals,
            pairs_14.len()
        );

        Ok(())
    }

    /// Get current positions
    pub fn get_positions(&self) -> Result<Vec<f32>> {
        let mut positions = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_positions, &mut positions)?;
        Ok(positions)
    }

    /// Set positions
    pub fn set_positions(&mut self, positions: &[f32]) -> Result<()> {
        if positions.len() != self.n_atoms * 3 {
            return Err(anyhow::anyhow!("Position count mismatch"));
        }
        self.stream.memcpy_htod(positions, &mut self.d_positions)?;
        Ok(())
    }

    /// Get number of atoms
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Check if topology is uploaded
    pub fn is_ready(&self) -> bool {
        self.topology_ready
    }

    /// Initialize velocities from Maxwell-Boltzmann distribution
    pub fn initialize_velocities(&mut self, temperature: f32) -> Result<()> {
        if !self.topology_ready {
            return Err(anyhow::anyhow!("Topology not uploaded"));
        }

        let threads_per_block = 256;
        let num_blocks = (self.n_atoms + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64;

        // Load flat velocity init kernel
        let init_kernel = self.module
            .load_function("initialize_velocities_flat")
            .context("Failed to load initialize_velocities_flat")?;

        // Bind values to avoid temporary lifetime issues
        let n_atoms_i32 = self.n_atoms as i32;

        unsafe {
            let mut builder = self.stream.launch_builder(&init_kernel);
            builder.arg(&self.d_velocities);
            builder.arg(&self.d_nb_mass);
            builder.arg(&n_atoms_i32);
            builder.arg(&temperature);
            builder.arg(&seed);
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;
        self.velocities_initialized = true;

        log::info!("✅ Velocities initialized at T={}K", temperature);
        Ok(())
    }

    /// Run energy minimization (steepest descent)
    ///
    /// This is CRITICAL before HMC: ANM conformations often have steric clashes
    /// that cause force explosions. Minimization relaxes these clashes.
    ///
    /// # Arguments
    /// * `n_steps` - Number of minimization steps (typical: 100-500)
    /// * `step_size` - Step size in Angstroms (typical: 0.001-0.01)
    ///
    /// # Returns
    /// Final potential energy after minimization
    pub fn minimize(&mut self, n_steps: usize, step_size: f32) -> Result<f32> {
        if !self.topology_ready {
            return Err(anyhow::anyhow!("Topology not uploaded"));
        }

        let threads_per_block = 256;
        let num_blocks = (self.n_atoms * 3 + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 6 * 64 * 4, // Same as HMC kernel
        };

        // Load minimize kernel
        let minimize_kernel = self.module
            .load_function("amber_steepest_descent_step")
            .context("Failed to load amber_steepest_descent_step")?;

        // Bind parameters
        let max_excl_i32 = MAX_EXCLUSIONS as i32;
        let max_14_i32 = MAX_14_PAIRS as i32;
        let n_atoms_i32 = self.n_atoms as i32;
        let n_bonds_i32 = self.n_bonds as i32;
        let n_angles_i32 = self.n_angles as i32;
        let n_dihedrals_i32 = self.n_dihedrals as i32;

        log::info!("⚡ Running {} minimization steps (step_size={} Å)", n_steps, step_size);

        let mut last_energy = f32::MAX;
        for step in 0..n_steps {
            unsafe {
                let mut builder = self.stream.launch_builder(&minimize_kernel);
                builder.arg(&self.d_positions);
                builder.arg(&self.d_forces);
                builder.arg(&self.d_total_energy);
                builder.arg(&self.d_bond_atoms);
                builder.arg(&self.d_bond_params);
                builder.arg(&self.d_angle_atoms);
                builder.arg(&self.d_angle_params);
                builder.arg(&self.d_dihedral_atoms);
                builder.arg(&self.d_dihedral_params);
                builder.arg(&self.d_nb_sigma);
                builder.arg(&self.d_nb_epsilon);
                builder.arg(&self.d_nb_charge);
                builder.arg(&self.d_exclusion_list);
                builder.arg(&self.d_n_exclusions);
                builder.arg(&self.d_pair14_list);
                builder.arg(&self.d_n_pairs14);
                builder.arg(&max_excl_i32);
                builder.arg(&max_14_i32);
                builder.arg(&n_atoms_i32);
                builder.arg(&n_bonds_i32);
                builder.arg(&n_angles_i32);
                builder.arg(&n_dihedrals_i32);
                builder.arg(&step_size);
                builder.launch(cfg)?;
            }

            // Check energy every 50 steps
            if step % 50 == 0 || step == n_steps - 1 {
                self.stream.synchronize()?;
                let mut energy = vec![0.0f32; 1];
                self.stream.memcpy_dtoh(&self.d_total_energy, &mut energy)?;

                if step == 0 {
                    log::info!("  Step 0: PE = {:.2} kcal/mol", energy[0]);
                    // Download and analyze forces
                    let mut forces = vec![0.0f32; self.n_atoms * 3];
                    self.stream.memcpy_dtoh(&self.d_forces, &mut forces)?;
                    let mut max_force = 0.0f32;
                    let mut total_force_mag = 0.0f32;
                    let mut n_large = 0;
                    for i in 0..self.n_atoms {
                        let fx = forces[i * 3];
                        let fy = forces[i * 3 + 1];
                        let fz = forces[i * 3 + 2];
                        let mag = (fx*fx + fy*fy + fz*fz).sqrt();
                        if mag > max_force { max_force = mag; }
                        total_force_mag += mag;
                        if mag > 100.0 { n_large += 1; }
                    }
                    let avg_force = total_force_mag / self.n_atoms as f32;
                    log::info!("  Force stats: max={:.1}, avg={:.1}, n_large(>100)={}/{}",
                        max_force, avg_force, n_large, self.n_atoms);
                } else if step == n_steps - 1 {
                    log::info!("  Step {}: PE = {:.2} kcal/mol (final)", step, energy[0]);
                }

                // Early termination if converged
                if (last_energy - energy[0]).abs() < 0.1 && step > 50 {
                    log::info!("  Converged at step {} (ΔE < 0.1 kcal/mol)", step);
                    break;
                }
                last_energy = energy[0];
            }
        }

        self.stream.synchronize()?;
        let mut final_energy = vec![0.0f32; 1];
        self.stream.memcpy_dtoh(&self.d_total_energy, &mut final_energy)?;

        log::info!("✅ Minimization complete: PE = {:.2} kcal/mol", final_energy[0]);
        Ok(final_energy[0])
    }

    /// Build neighbor lists for O(N) non-bonded calculation
    ///
    /// This builds cell lists and neighbor lists from current positions.
    /// Must be called before run() and periodically during long simulations
    /// (every ~10-20 steps if atoms move significantly).
    ///
    /// Uses spatial hashing with cells of size = cutoff (10Å).
    /// Each atom only checks 27 neighboring cells for neighbors.
    /// Reduces O(N²) to O(N) average case.
    pub fn build_neighbor_lists(&mut self) -> Result<()> {
        if !self.topology_ready {
            return Err(anyhow::anyhow!("Topology not ready"));
        }

        let threads_per_block = 256;
        let num_blocks = (self.n_atoms + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // Step 1: Compute bounding box of all atoms
        // We do this on CPU for simplicity (download positions, compute min/max)
        let mut positions = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_positions, &mut positions)?;

        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut min_z = f32::MAX;
        let mut max_x = f32::MIN;
        let mut max_y = f32::MIN;
        let mut max_z = f32::MIN;

        for i in 0..self.n_atoms {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            min_z = min_z.min(z);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
            max_z = max_z.max(z);
        }

        // Add small padding to avoid edge cases
        min_x -= 1.0;
        min_y -= 1.0;
        min_z -= 1.0;
        max_x += 1.0;
        max_y += 1.0;
        max_z += 1.0;

        // Compute grid dimensions
        let nx = ((max_x - min_x) / CELL_SIZE).ceil() as i32;
        let ny = ((max_y - min_y) / CELL_SIZE).ceil() as i32;
        let nz = ((max_z - min_z) / CELL_SIZE).ceil() as i32;

        // Clamp to maximum grid size
        let nx = nx.min(MAX_CELLS_PER_DIM as i32).max(1);
        let ny = ny.min(MAX_CELLS_PER_DIM as i32).max(1);
        let nz = nz.min(MAX_CELLS_PER_DIM as i32).max(1);

        self.grid_origin = [min_x, min_y, min_z];
        self.grid_dims = [nx, ny, nz];

        log::debug!(
            "📦 Cell grid: {}x{}x{} cells, origin: ({:.1}, {:.1}, {:.1})",
            nx, ny, nz, min_x, min_y, min_z
        );

        // Step 2: Zero cell counts
        let zero_counts = vec![0i32; MAX_TOTAL_CELLS];
        self.stream.memcpy_htod(&zero_counts, &mut self.d_cell_counts)?;

        // Step 3: Build cell lists (assign atoms to cells)
        let n_atoms_i32 = self.n_atoms as i32;
        unsafe {
            let mut builder = self.stream.launch_builder(&self.build_cell_list_kernel);
            builder.arg(&self.d_positions);
            builder.arg(&self.d_cell_list);
            builder.arg(&self.d_cell_counts);
            builder.arg(&self.d_atom_cell);
            builder.arg(&min_x);
            builder.arg(&min_y);
            builder.arg(&min_z);
            builder.arg(&nx);
            builder.arg(&ny);
            builder.arg(&nz);
            builder.arg(&n_atoms_i32);
            builder.launch(cfg.clone())?;
        }

        // Step 4: Build neighbor lists from cell lists
        let max_excl_i32 = MAX_EXCLUSIONS as i32;
        unsafe {
            let mut builder = self.stream.launch_builder(&self.build_neighbor_list_kernel);
            builder.arg(&self.d_positions);
            builder.arg(&self.d_cell_list);
            builder.arg(&self.d_cell_counts);
            builder.arg(&self.d_atom_cell);
            builder.arg(&self.d_exclusion_list);
            builder.arg(&self.d_n_exclusions);
            builder.arg(&self.d_neighbor_list);
            builder.arg(&self.d_n_neighbors);
            builder.arg(&max_excl_i32);
            builder.arg(&nx);
            builder.arg(&ny);
            builder.arg(&nz);
            builder.arg(&n_atoms_i32);
            builder.launch(cfg)?;
        }

        self.stream.synchronize()?;
        self.neighbor_list_valid = true;

        // Debug: check average neighbor count (use debug level to avoid spam during rebuilds)
        let mut n_neighbors = vec![0i32; self.n_atoms];
        self.stream.memcpy_dtoh(&self.d_n_neighbors, &mut n_neighbors)?;
        let avg_neighbors: f64 = n_neighbors.iter().map(|&n| n as f64).sum::<f64>() / self.n_atoms as f64;
        log::debug!(
            "Neighbor lists built: avg {:.1} neighbors/atom (vs {} for O(N²))",
            avg_neighbors,
            self.n_atoms
        );

        Ok(())
    }

    /// Run HMC trajectory with full AMBER ff14SB physics
    ///
    /// Uses O(N) neighbor lists for non-bonded forces.
    /// Includes bonds, angles, dihedrals, LJ, and Coulomb.
    /// BAOAB Langevin thermostat maintains temperature via friction + noise.
    ///
    /// # Arguments
    /// * `n_steps` - Number of integration steps
    /// * `dt` - Timestep in femtoseconds
    /// * `temperature` - Target temperature in Kelvin
    /// * `gamma_fs` - Langevin friction coefficient in fs⁻¹
    ///   - 0.001 (1 ps⁻¹): Production - preserves natural dynamics, τ = 1 ps
    ///   - 0.01 (10 ps⁻¹): Equilibration - fast thermalization, τ = 100 fs
    ///   - 0.1 (100 ps⁻¹): Aggressive - Brownian dynamics limit
    ///
    /// # Returns
    /// HMC run results including final energies and positions
    pub fn run(&mut self, n_steps: usize, dt: f32, temperature: f32, gamma_fs: f32) -> Result<HmcRunResult> {
        if !self.topology_ready {
            return Err(anyhow::anyhow!("Topology not uploaded - call upload_topology first"));
        }

        // Build neighbor lists for O(N) non-bonded (required for full-atom)
        if !self.neighbor_list_valid {
            log::info!("📦 Building neighbor lists for O(N) non-bonded...");
            self.build_neighbor_lists()?;
        }

        // Initialize velocities if not done
        if !self.velocities_initialized {
            self.initialize_velocities(temperature)?;
        }

        let threads_per_block = 256;
        let num_blocks = (self.n_atoms.max(self.n_bonds).max(self.n_angles).max(self.n_dihedrals)
            + threads_per_block - 1) / threads_per_block;

        let cfg = LaunchConfig {
            grid_dim: (num_blocks as u32, 1, 1),
            block_dim: (threads_per_block as u32, 1, 1),
            shared_mem_bytes: 6 * 64 * 4, // 6 arrays * TILE_SIZE * sizeof(float)
        };

        // Load flat HMC kernel
        let hmc_kernel = self.module
            .load_function("amber_mega_fused_hmc_step_flat")
            .context("Failed to load amber_mega_fused_hmc_step_flat")?;

        let thermostat_kernel = self.module
            .load_function("apply_thermostat_flat")
            .context("Failed to load apply_thermostat_flat")?;

        let mut total_ke = 0.0f64;

        // Bind all integer parameters to variables for stable references
        let max_excl_i32 = MAX_EXCLUSIONS as i32;
        let max_14_i32 = MAX_14_PAIRS as i32;
        let n_atoms_i32 = self.n_atoms as i32;
        let n_bonds_i32 = self.n_bonds as i32;
        let n_angles_i32 = self.n_angles as i32;
        let n_dihedrals_i32 = self.n_dihedrals as i32;

        log::info!("🏃 Running {} HMC steps on GPU (dt={}fs, T={}K, γ={}fs⁻¹)", n_steps, dt, temperature, gamma_fs);

        // Neighbor list rebuild interval - atoms move ~0.01 Å per step at 310K
        // Rebuild every 50 steps to keep lists fresh (before atoms drift 0.5 Å)
        const NEIGHBOR_REBUILD_INTERVAL: usize = 50;

        for step in 0..n_steps {
            // Rebuild neighbor lists periodically to prevent stale interactions
            if step > 0 && step % NEIGHBOR_REBUILD_INTERVAL == 0 {
                self.stream.synchronize()?;
                self.neighbor_list_valid = false;
                self.build_neighbor_lists()?;
            }

            // Launch mega-fused HMC step kernel
            let step_u32 = step as u32;
            unsafe {
                let mut builder = self.stream.launch_builder(&hmc_kernel);
                builder.arg(&self.d_positions);
                builder.arg(&self.d_velocities);
                builder.arg(&self.d_forces);
                builder.arg(&self.d_total_energy);
                builder.arg(&self.d_kinetic_energy);
                builder.arg(&self.d_bond_atoms);
                builder.arg(&self.d_bond_params);
                builder.arg(&self.d_angle_atoms);
                builder.arg(&self.d_angle_params);
                builder.arg(&self.d_dihedral_atoms);
                builder.arg(&self.d_dihedral_params);
                builder.arg(&self.d_nb_sigma);
                builder.arg(&self.d_nb_epsilon);
                builder.arg(&self.d_nb_charge);
                builder.arg(&self.d_nb_mass);
                builder.arg(&self.d_exclusion_list);
                builder.arg(&self.d_n_exclusions);
                builder.arg(&self.d_pair14_list);
                builder.arg(&self.d_n_pairs14);
                builder.arg(&max_excl_i32);
                builder.arg(&max_14_i32);
                builder.arg(&n_atoms_i32);
                builder.arg(&n_bonds_i32);
                builder.arg(&n_angles_i32);
                builder.arg(&n_dihedrals_i32);
                builder.arg(&dt);
                builder.arg(&temperature);
                builder.arg(&gamma_fs);  // Langevin friction coefficient
                builder.arg(&step_u32);  // Step counter for RNG seeding
                builder.launch(cfg)?;
            }

            // NOTE: Thermostat disabled - BAOAB Langevin dynamics handles temperature
            // control continuously via friction + thermal noise (O step).
            // External thermostat would fight the Langevin dynamics.

            // Accumulate kinetic energy for average temperature
            if step % 10 == 0 {
                self.stream.synchronize()?;
                let mut ke = vec![0.0f32; 1];
                self.stream.memcpy_dtoh(&self.d_kinetic_energy, &mut ke)?;
                total_ke += ke[0] as f64;
            }
        }

        self.stream.synchronize()?;

        // Download final results
        let mut h_total_energy = vec![0.0f32; 1];
        let mut h_kinetic_energy = vec![0.0f32; 1];
        self.stream.memcpy_dtoh(&self.d_total_energy, &mut h_total_energy)?;
        self.stream.memcpy_dtoh(&self.d_kinetic_energy, &mut h_kinetic_energy)?;

        let positions = self.get_positions()?;
        let mut velocities = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_velocities, &mut velocities)?;

        // Average temperature from kinetic energy samples
        let n_samples = (n_steps / 10).max(1);
        let avg_ke = total_ke / n_samples as f64;
        let kb = 0.001987204; // kcal/(mol*K)
        let avg_temperature = 2.0 * avg_ke / (3.0 * self.n_atoms as f64 * kb);

        // DIAGNOSTIC: Verify atom count and KE sanity
        let expected_ke_at_target = 1.5 * self.n_atoms as f64 * kb * temperature as f64;
        log::info!(
            "🔬 DIAG: avg_ke={:.1}, expected_ke@{}K={:.1}, ratio={:.3}, n_atoms={}, n_samples={}",
            avg_ke, temperature, expected_ke_at_target,
            avg_ke / expected_ke_at_target, self.n_atoms, n_samples
        );

        log::info!(
            "✅ HMC complete: PE={:.2} kcal/mol, KE={:.2} kcal/mol, T_avg={:.1}K",
            h_total_energy[0], h_kinetic_energy[0], avg_temperature
        );

        Ok(HmcRunResult {
            potential_energy: h_total_energy[0] as f64,
            kinetic_energy: h_kinetic_energy[0] as f64,
            positions,
            velocities,
            avg_temperature,
        })
    }

    /// Get current velocities
    pub fn get_velocities(&self) -> Result<Vec<f32>> {
        let mut velocities = vec![0.0f32; self.n_atoms * 3];
        self.stream.memcpy_dtoh(&self.d_velocities, &mut velocities)?;
        Ok(velocities)
    }

    /// Rescale velocities to target temperature (velocity rescaling thermostat)
    ///
    /// This maintains the canonical ensemble by scaling all velocities to achieve
    /// the target temperature, compensating for numerical integration drift.
    pub fn rescale_velocities(&mut self, target_temperature: f32) -> Result<()> {
        if !self.velocities_initialized {
            return Err(anyhow::anyhow!("Velocities not initialized"));
        }

        // Download current velocities and masses
        let mut velocities = vec![0.0f32; self.n_atoms * 3];
        let mut masses = vec![0.0f32; self.n_atoms];
        self.stream.memcpy_dtoh(&self.d_velocities, &mut velocities)?;
        self.stream.memcpy_dtoh(&self.d_nb_mass, &mut masses)?;

        // Calculate current kinetic energy: KE = 0.5 * sum(m_i * v_i^2) / FORCE_TO_ACCEL
        // FORCE_TO_ACCEL = 4.184e-4 converts velocity²*mass to kcal/mol
        // Without this factor, the KE would be in wrong units (g/mol * Å²/fs²)
        const FORCE_TO_ACCEL: f64 = 4.184e-4;
        let mut kinetic_energy = 0.0f64;
        for i in 0..self.n_atoms {
            let vx = velocities[i * 3] as f64;
            let vy = velocities[i * 3 + 1] as f64;
            let vz = velocities[i * 3 + 2] as f64;
            let m = masses[i] as f64;
            kinetic_energy += 0.5 * m * (vx * vx + vy * vy + vz * vz) / FORCE_TO_ACCEL;
        }

        // Calculate current temperature: T = 2*KE / (3*N*kb)
        // kb = 0.001987204 kcal/(mol*K)
        let kb = 0.001987204f64;
        let n_dof = (3 * self.n_atoms - 6).max(1) as f64; // 3N - 6 for non-linear molecule
        let current_temp = 2.0 * kinetic_energy / (n_dof * kb);

        if current_temp < 1.0 {
            // Temperature too low, reinitialize
            log::warn!("Temperature too low ({:.1}K), reinitializing velocities", current_temp);
            return self.initialize_velocities(target_temperature);
        }

        // Calculate scaling factor: lambda = sqrt(T_target / T_current)
        let scale_factor = ((target_temperature as f64) / current_temp).sqrt();

        // Scale all velocities
        for v in velocities.iter_mut() {
            *v *= scale_factor as f32;
        }

        // Upload scaled velocities back to GPU
        self.stream.memcpy_htod(&velocities, &mut self.d_velocities)?;

        log::debug!(
            "🌡️ Rescaled velocities: {:.1}K → {:.1}K (factor: {:.4})",
            current_temp, target_temperature, scale_factor
        );

        Ok(())
    }
}

/// Build exclusion lists from bond topology
pub fn build_exclusion_lists(
    bonds: &[(usize, usize, f32, f32)],
    angles: &[(usize, usize, usize, f32, f32)],
    n_atoms: usize,
) -> Vec<HashSet<usize>> {
    let mut exclusions = vec![HashSet::new(); n_atoms];

    // 1-2 exclusions (bonded pairs)
    for &(i, j, _, _) in bonds {
        if i < n_atoms && j < n_atoms {
            exclusions[i].insert(j);
            exclusions[j].insert(i);
        }
    }

    // 1-3 exclusions (angle endpoints)
    for &(i, _j, k, _, _) in angles {
        if i < n_atoms && k < n_atoms {
            exclusions[i].insert(k);
            exclusions[k].insert(i);
        }
    }

    exclusions
}

/// Build 1-4 pair list from dihedral topology
///
/// 1-4 pairs are atoms separated by exactly 3 bonds (the first and last atoms of each dihedral).
/// These need SCALED non-bonded interactions (AMBER ff14SB: LJ*0.5, Coulomb*0.833).
///
/// Returns: Vec of (atom_i, atom_j) pairs, deduplicated
pub fn build_14_pairs(
    dihedrals: &[(usize, usize, usize, usize, f32, f32, f32)],
    exclusions: &[HashSet<usize>],
    n_atoms: usize,
) -> Vec<(usize, usize)> {
    let mut pairs_14: HashSet<(usize, usize)> = HashSet::new();

    for &(i, _j, _k, l, _, _, _) in dihedrals {
        if i >= n_atoms || l >= n_atoms {
            continue;
        }

        // Skip if this pair is already a 1-2 or 1-3 exclusion
        // (can happen with ring systems)
        if exclusions[i].contains(&l) {
            continue;
        }

        // Canonicalize order (smaller index first) to avoid duplicates
        let pair = if i < l { (i, l) } else { (l, i) };
        pairs_14.insert(pair);
    }

    let mut result: Vec<_> = pairs_14.into_iter().collect();
    result.sort(); // Sort for deterministic ordering
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_exclusion_lists() {
        let bonds = vec![(0, 1, 100.0, 1.5), (1, 2, 100.0, 1.5)];
        let angles = vec![(0, 1, 2, 50.0, 2.0)];
        let exclusions = build_exclusion_lists(&bonds, &angles, 3);

        assert!(exclusions[0].contains(&1));
        assert!(exclusions[0].contains(&2));
        assert!(exclusions[1].contains(&0));
        assert!(exclusions[1].contains(&2));
    }
}

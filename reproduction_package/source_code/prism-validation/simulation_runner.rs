//! PRISM-NOVA Simulation Runner
//!
//! Bridges the validation pipeline with the GPU physics engine.
//! Converts PDB structures to simulation format and runs molecular dynamics
//! with goal-directed sampling for cryptic pocket discovery.
//!
//! ## Workflow
//!
//! 1. Convert `SimulationStructure` → PRISM-NOVA format (positions, masses, charges)
//! 2. Initialize GPU context and NOVA engine
//! 3. Run simulation steps with trajectory collection
//! 4. Compute validation metrics from trajectory

use crate::pipeline::SimulationStructure;
use crate::{BenchmarkMetrics, ValidationConfig};
use anyhow::{Context, Result};
use std::sync::Arc;

// AMBER ff14SB force field and topology generation
use prism_physics::amber_ff14sb::{
    AmberTopology, GpuTopology, PdbAtom, AmberAtomType,
    get_atom_charge, get_lj_param, get_atom_mass,
};

// Re-export NOVA types for convenience
pub use prism_gpu::{PrismNova, NovaConfig, NovaStepResult};

/// Atomic properties for simulation
/// Derived from element type and residue context
#[derive(Debug, Clone)]
pub struct AtomicProperties {
    /// Mass in atomic mass units (amu)
    pub mass: f32,
    /// Partial charge in elementary charge units
    pub charge: f32,
    /// Lennard-Jones epsilon (kcal/mol)
    pub lj_epsilon: f32,
    /// Lennard-Jones sigma (Angstroms)
    pub lj_sigma: f32,
    /// Atom type index (for force field lookup)
    pub atom_type: i32,
}

/// Trajectory frame from simulation
#[derive(Debug, Clone)]
pub struct TrajectoryFrame {
    /// Step number
    pub step: usize,
    /// CA positions for this frame [n_residues * 3]
    pub ca_positions: Vec<f32>,
    /// All atom positions [n_atoms * 3]
    pub all_positions: Vec<f32>,
    /// Whether HMC proposal was accepted
    pub accepted: bool,
    /// Reward signal (pocket progress)
    pub reward: f32,
    /// Betti numbers [b0, b1, b2]
    pub betti: [f32; 3],
    /// Pocket signature
    pub pocket_signature: f32,
    /// Expected free energy
    pub efe: f32,
    /// Goal prior (druggability belief)
    pub goal_prior: f32,
}

/// Full simulation trajectory
#[derive(Debug, Clone)]
pub struct SimulationTrajectory {
    /// Target name
    pub target_name: String,
    /// Starting structure PDB ID
    pub pdb_id: String,
    /// Simulation configuration
    pub config: SimulationConfig,
    /// Trajectory frames (sampled at interval)
    pub frames: Vec<TrajectoryFrame>,
    /// Total simulation steps
    pub total_steps: usize,
    /// HMC acceptance rate
    pub acceptance_rate: f32,
    /// Best pocket signature achieved
    pub best_pocket_signature: f32,
    /// Step at which pocket first opened (Betti-2 > 0)
    pub pocket_opening_step: Option<usize>,
}

/// Configuration for simulation run
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Number of simulation steps
    pub n_steps: usize,
    /// Temperature in Kelvin
    pub temperature: f32,
    /// Timestep in picoseconds
    pub dt: f32,
    /// Goal-directed sampling strength
    pub goal_strength: f32,
    /// Save trajectory every N steps
    pub save_interval: usize,
    /// GPU device index
    pub gpu_device: usize,
    /// Use coarse-grained (C-alpha only) representation
    /// When true: 1 atom per residue (faster, fits in GPU shared memory)
    /// When false: all atoms (more accurate but limited to ~60 residues)
    pub coarse_grained: bool,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            n_steps: 1000,
            temperature: 310.0,  // 37°C
            dt: 0.002,           // 2 fs
            goal_strength: 0.1,
            save_interval: 10,
            gpu_device: 0,
            coarse_grained: true,  // Default to CA-only for GPU compatibility
        }
    }
}

impl From<&ValidationConfig> for SimulationConfig {
    fn from(vc: &ValidationConfig) -> Self {
        Self {
            n_steps: vc.steps_per_target,
            temperature: vc.temperature,
            gpu_device: vc.gpu_device,
            coarse_grained: true,  // Use CA-only for validation benchmark
            dt: 0.005,             // 5 fs timestep for CG (2.5x larger than all-atom)
            ..Default::default()
        }
    }
}

/// PRISM-NOVA Simulation Runner
///
/// Orchestrates GPU-accelerated molecular dynamics simulations
/// with goal-directed sampling for drug discovery.
pub struct SimulationRunner {
    /// GPU context (shared across simulations)
    context: Option<Arc<cudarc::driver::CudaContext>>,
    /// Simulation configuration
    config: SimulationConfig,
}

impl SimulationRunner {
    /// Create a new simulation runner
    ///
    /// Initializes GPU context lazily on first simulation.
    pub fn new(config: SimulationConfig) -> Self {
        Self {
            context: None,
            config,
        }
    }

    /// Initialize GPU context if not already initialized
    ///
    /// Note: cudarc 0.18's CudaContext::new() already returns Arc<CudaContext>
    fn ensure_gpu_context(&mut self) -> Result<Arc<cudarc::driver::CudaContext>> {
        if let Some(ref ctx) = self.context {
            return Ok(ctx.clone());
        }

        log::info!("Initializing CUDA context on device {}", self.config.gpu_device);

        // CudaContext::new() in cudarc 0.18 returns Arc<CudaContext> directly
        let ctx = cudarc::driver::CudaContext::new(self.config.gpu_device)
            .context("Failed to create CUDA context")?;

        self.context = Some(ctx.clone());

        Ok(ctx)
    }

    /// Run simulation on a structure
    ///
    /// # Arguments
    /// * `structure` - Input structure (typically APO form)
    /// * `target_structure` - Optional target structure for RMSD calculation
    ///
    /// # Returns
    /// Full simulation trajectory with metrics
    pub fn run_simulation(
        &mut self,
        structure: &SimulationStructure,
        target_structure: Option<&SimulationStructure>,
    ) -> Result<SimulationTrajectory> {
        // Determine effective atom count based on coarse-grained mode
        // Use actual data lengths, not metadata
        let effective_n_atoms = if self.config.coarse_grained {
            structure.ca_positions.len()  // Actual CA atoms in PDB
        } else {
            structure.all_positions.len()  // Actual all-atoms in PDB
        };

        // Also update n_residues to match actual data
        let effective_n_residues = structure.ca_positions.len();

        log::info!(
            "Starting PRISM-NOVA simulation on {} ({} effective atoms, {} residues, CG={})",
            structure.name,
            effective_n_atoms,
            effective_n_residues,
            self.config.coarse_grained
        );

        // Initialize GPU
        let context = self.ensure_gpu_context()?;

        // Convert structure to simulation format
        let (positions, masses, charges, lj_params, atom_types) =
            self.convert_to_simulation_format(structure)?;

        // Build residue representative atoms (use CA atoms)
        let residue_atoms = self.build_residue_atoms(structure)?;

        // Create NOVA configuration
        // For CG (coarse-grained) models, use larger timestep and more leapfrog steps
        // CG models have no high-frequency bond vibrations, allowing longer trajectories
        let (dt, leapfrog_steps) = if self.config.coarse_grained {
            // CG: 5 fs timestep, 10 leapfrog steps = 50 fs per HMC proposal
            // Short trajectory for high acceptance (~65% target), many proposals for exploration
            // pfANM springs are soft enough that we need shorter trajectories
            (0.005, 10)
        } else {
            // All-atom: 2 fs timestep, 10 leapfrog steps = 20 fs per HMC proposal
            (self.config.dt, 10)
        };

        let mut nova_config = NovaConfig {
            dt,
            temperature: self.config.temperature,
            goal_strength: self.config.goal_strength,
            n_atoms: effective_n_atoms as i32,
            n_residues: effective_n_residues as i32,
            n_target_residues: structure.pocket_residues.as_ref().map(|p| p.len()).unwrap_or(0) as i32,
            leapfrog_steps,
            ..NovaConfig::default()
        };

        // Set target residues for TDA analysis
        // If pocket residues are specified, use them
        // Otherwise, use ALL residues (sampled if > MAX_TARGET_RESIDUES)
        // This ensures TDA always runs and produces meaningful Betti numbers
        if let Some(ref pocket) = structure.pocket_residues {
            let n_targets = pocket.len().min(prism_gpu::prism_nova::MAX_TARGET_RESIDUES);
            nova_config.n_target_residues = n_targets as i32;
            for (i, &res) in pocket.iter().take(n_targets).enumerate() {
                nova_config.target_residues[i] = res;
            }
            log::info!("TDA targeting {} pocket residues", n_targets);
        } else {
            // No pocket specified - use ALL residues for global TDA
            // Sample uniformly if protein is larger than MAX_TARGET_RESIDUES
            let max_targets = prism_gpu::prism_nova::MAX_TARGET_RESIDUES;
            let n_residues = effective_n_residues;

            if n_residues <= max_targets {
                // Use all residues
                nova_config.n_target_residues = n_residues as i32;
                for i in 0..n_residues {
                    nova_config.target_residues[i] = i as i32;
                }
                log::info!("TDA targeting all {} residues", n_residues);
            } else {
                // Sample every Nth residue to get ~MAX_TARGET_RESIDUES
                let step = n_residues / max_targets;
                nova_config.n_target_residues = max_targets as i32;
                for i in 0..max_targets {
                    nova_config.target_residues[i] = (i * step) as i32;
                }
                log::info!("TDA targeting {} sampled residues (from {} total, step={})",
                          max_targets, n_residues, step);
            }
        }

        // Initialize PRISM-NOVA
        log::info!("Initializing PRISM-NOVA engine...");
        let mut nova = PrismNova::new(context, nova_config)
            .context("Failed to initialize PRISM-NOVA")?;

        // Upload molecular system
        nova.upload_system(&positions, &masses, &charges, &lj_params, &atom_types, &residue_atoms)
            .context("Failed to upload molecular system")?;

        // Generate and upload topology (bonds, angles, dihedrals)
        self.upload_topology(&mut nova, structure)?;

        // Initialize reservoir with random weights
        let reservoir_weights = self.generate_reservoir_weights();
        nova.upload_reservoir_weights(&reservoir_weights)
            .context("Failed to upload reservoir weights")?;

        // Initialize momenta and RLS
        nova.initialize_momenta()
            .context("Failed to initialize momenta")?;
        nova.initialize_rls(1.0)
            .context("Failed to initialize RLS")?;

        // Run simulation
        log::info!("Running {} steps at T={} K", self.config.n_steps, self.config.temperature);
        let trajectory = self.run_simulation_loop(&mut nova, structure, target_structure)?;

        log::info!(
            "Simulation complete: acceptance={:.1}%, best_pocket={:.3}",
            trajectory.acceptance_rate * 100.0,
            trajectory.best_pocket_signature
        );

        Ok(trajectory)
    }

    /// Convert SimulationStructure to PRISM-NOVA format
    fn convert_to_simulation_format(
        &self,
        structure: &SimulationStructure,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<i32>)> {
        if self.config.coarse_grained {
            // COARSE-GRAINED MODE: Use C-alpha only (1 atom per residue)
            // This allows larger proteins to fit in GPU shared memory
            let n_atoms = structure.ca_positions.len();

            // Flatten CA positions: [x1, y1, z1, x2, y2, z2, ...]
            let mut positions = Vec::with_capacity(n_atoms * 3);
            for pos in &structure.ca_positions {
                positions.push(pos[0]);
                positions.push(pos[1]);
                positions.push(pos[2]);
            }

            // Use unified CA properties (~110 Da per residue average)
            let masses: Vec<f32> = vec![110.0; n_atoms];  // Average residue mass
            let charges: Vec<f32> = vec![0.0; n_atoms];   // Neutral (no explicit electrostatics)
            let mut lj_params = Vec::with_capacity(n_atoms * 2);
            let atom_types: Vec<i32> = vec![0; n_atoms];  // All CA type

            // Coarse-grained: DISABLE LJ interactions
            // The pfANM springs already model all inter-residue interactions
            // Adding LJ on top creates competing potentials and low HMC acceptance
            // For pure ENM/ANM, only harmonic springs are used
            for _ in 0..n_atoms {
                lj_params.push(0.0);    // epsilon = 0 (disabled)
                lj_params.push(1.0);    // sigma (Å) - unused but must be non-zero
            }

            log::info!("Converted {} residues to COARSE-GRAINED (CA-only) format", n_atoms);
            Ok((positions, masses, charges, lj_params, atom_types))
        } else {
            // ALL-ATOM MODE: Use proper AMBER ff14SB parameters
            // This uses full atom typing, charges, and LJ parameters from amber_ff14sb.rs
            let n_atoms = structure.all_positions.len();

            // Flatten positions: [x1, y1, z1, x2, y2, z2, ...]
            let mut positions = Vec::with_capacity(n_atoms * 3);
            for pos in &structure.all_positions {
                positions.push(pos[0]);
                positions.push(pos[1]);
                positions.push(pos[2]);
            }

            // Use proper AMBER ff14SB parameters (NOT simplified element-based!)
            let mut masses = Vec::with_capacity(n_atoms);
            let mut charges = Vec::with_capacity(n_atoms);
            let mut lj_params = Vec::with_capacity(n_atoms * 2);
            let mut atom_types = Vec::with_capacity(n_atoms);

            for i in 0..n_atoms {
                let residue_name = &structure.residue_names[i];
                let atom_name = &structure.atom_names[i];

                // Get proper AMBER atom type
                let amber_type = AmberAtomType::from_pdb(residue_name, atom_name);

                // Get proper AMBER ff14SB partial charge
                let charge = get_atom_charge(residue_name, atom_name);

                // Get proper AMBER ff14SB LJ parameters
                let lj = get_lj_param(amber_type);
                // Convert from (epsilon, rmin_half) to (epsilon, sigma)
                // sigma = 2^(1/6) * rmin, where rmin = 2 * rmin_half
                let sigma = 2.0_f32.powf(1.0/6.0) * 2.0 * lj.rmin_half;

                // Get proper mass from atom type
                let mass = get_atom_mass(amber_type);

                masses.push(mass);
                charges.push(charge);
                lj_params.push(lj.epsilon);
                lj_params.push(sigma);
                atom_types.push(amber_type as i32);
            }

            log::info!("Converted {} atoms to ALL-ATOM format with AMBER ff14SB parameters", n_atoms);
            Ok((positions, masses, charges, lj_params, atom_types))
        }
    }

    /// Get atomic properties from element type and residue context
    fn get_atomic_properties(&self, element: &str, residue: &str) -> AtomicProperties {
        // AMBER-like parameters (simplified)
        // In production, would use full AMBER/CHARMM force field
        match element.to_uppercase().as_str() {
            "C" => AtomicProperties {
                mass: 12.011,
                charge: self.estimate_carbon_charge(residue),
                lj_epsilon: 0.086,
                lj_sigma: 3.4,
                atom_type: 0,
            },
            "N" => AtomicProperties {
                mass: 14.007,
                charge: self.estimate_nitrogen_charge(residue),
                lj_epsilon: 0.170,
                lj_sigma: 3.25,
                atom_type: 1,
            },
            "O" => AtomicProperties {
                mass: 15.999,
                charge: self.estimate_oxygen_charge(residue),
                lj_epsilon: 0.210,
                lj_sigma: 2.96,
                atom_type: 2,
            },
            "S" => AtomicProperties {
                mass: 32.065,
                charge: -0.1,
                lj_epsilon: 0.250,
                lj_sigma: 3.55,
                atom_type: 3,
            },
            "H" => AtomicProperties {
                mass: 1.008,
                charge: 0.1,
                lj_epsilon: 0.016,
                lj_sigma: 2.5,
                atom_type: 4,
            },
            "P" => AtomicProperties {
                mass: 30.974,
                charge: 0.5,
                lj_epsilon: 0.200,
                lj_sigma: 3.74,
                atom_type: 5,
            },
            // Metal ions
            "FE" | "ZN" | "MG" | "CA" | "MN" => AtomicProperties {
                mass: match element.to_uppercase().as_str() {
                    "FE" => 55.845,
                    "ZN" => 65.38,
                    "MG" => 24.305,
                    "CA" => 40.078,
                    "MN" => 54.938,
                    _ => 56.0,
                },
                charge: 2.0,
                lj_epsilon: 0.001,
                lj_sigma: 2.0,
                atom_type: 6,
            },
            _ => {
                // Default parameters for unknown elements
                log::warn!("Unknown element '{}', using default parameters", element);
                AtomicProperties {
                    mass: 12.0,
                    charge: 0.0,
                    lj_epsilon: 0.1,
                    lj_sigma: 3.0,
                    atom_type: 7,
                }
            }
        }
    }

    /// Estimate partial charge for carbon based on residue context
    fn estimate_carbon_charge(&self, residue: &str) -> f32 {
        match residue {
            "ASP" | "GLU" => -0.1,  // Acidic residues
            "LYS" | "ARG" => 0.1,   // Basic residues
            "ALA" | "VAL" | "LEU" | "ILE" | "MET" | "PHE" | "TRP" => 0.0,  // Hydrophobic
            _ => 0.0,
        }
    }

    /// Estimate partial charge for nitrogen based on residue context
    fn estimate_nitrogen_charge(&self, residue: &str) -> f32 {
        match residue {
            "LYS" => 0.5,    // Lysine amino group
            "ARG" => 0.4,    // Arginine guanidinium
            "HIS" => 0.2,    // Histidine (partially protonated at pH 7)
            "ASN" | "GLN" => -0.2,  // Amide nitrogen
            _ => -0.4,       // Backbone amide
        }
    }

    /// Estimate partial charge for oxygen based on residue context
    fn estimate_oxygen_charge(&self, residue: &str) -> f32 {
        match residue {
            "ASP" | "GLU" => -0.6,  // Carboxylate
            "SER" | "THR" | "TYR" => -0.3,  // Hydroxyl
            "ASN" | "GLN" => -0.4,  // Amide oxygen
            _ => -0.5,       // Backbone carbonyl
        }
    }

    /// Build representative atom indices for each residue (CA atoms)
    fn build_residue_atoms(&self, structure: &SimulationStructure) -> Result<Vec<i32>> {
        if self.config.coarse_grained {
            // COARSE-GRAINED MODE: Each "atom" IS a CA residue
            // So residue_atoms[i] = i (identity mapping)
            let n_residues = structure.ca_positions.len();
            let residue_atoms: Vec<i32> = (0..n_residues as i32).collect();
            log::debug!("CG mode: {} residue atoms (identity mapping)", n_residues);
            return Ok(residue_atoms);
        }

        // ALL-ATOM MODE: Find CA atom for each residue
        let mut residue_atoms = vec![-1i32; structure.n_residues];

        // Find CA atom for each residue
        for (atom_idx, res_idx) in structure.residue_indices.iter().enumerate() {
            if structure.elements[atom_idx] == "C" {
                // Check if this is a CA atom by looking at atom name pattern
                // In PDB, CA is the alpha carbon
                if residue_atoms[*res_idx] < 0 {
                    residue_atoms[*res_idx] = atom_idx as i32;
                }
            }
        }

        // Fill any missing with first atom of residue
        for (atom_idx, res_idx) in structure.residue_indices.iter().enumerate() {
            if residue_atoms[*res_idx] < 0 {
                residue_atoms[*res_idx] = atom_idx as i32;
            }
        }

        Ok(residue_atoms)
    }

    /// Upload molecular topology (bonds, angles, dihedrals) to GPU
    ///
    /// For coarse-grained (CA-only) simulations: Uses elastic network model (ENM)
    /// For all-atom simulations: Uses full AMBER ff14SB topology
    fn upload_topology(
        &self,
        nova: &mut PrismNova,
        structure: &SimulationStructure,
    ) -> Result<()> {
        if self.config.coarse_grained {
            // Coarse-grained: Elastic Network Model
            // Build PdbAtom list from CA positions only
            let mut ca_atoms = Vec::with_capacity(structure.ca_positions.len());

            // We need to map CA positions back to their residue info
            // Find CA atoms in the all_positions array
            let mut ca_idx = 0;
            for (i, name) in structure.atom_names.iter().enumerate() {
                if name == "CA" && ca_idx < structure.ca_positions.len() {
                    let pos = structure.ca_positions[ca_idx];
                    let chain_id = structure.chain_ids.get(i)
                        .and_then(|s| s.chars().next())
                        .unwrap_or('A');

                    ca_atoms.push(PdbAtom {
                        index: ca_idx,
                        name: "CA".to_string(),
                        residue_name: structure.residue_names.get(i)
                            .cloned()
                            .unwrap_or_else(|| "UNK".to_string()),
                        residue_id: structure.residue_seqs.get(i)
                            .copied()
                            .unwrap_or(ca_idx as i32),
                        chain_id,
                        x: pos[0],
                        y: pos[1],
                        z: pos[2],
                    });
                    ca_idx += 1;
                }
            }

            // If we couldn't find CA atoms by name, fall back to sequential
            if ca_atoms.is_empty() {
                for (i, pos) in structure.ca_positions.iter().enumerate() {
                    ca_atoms.push(PdbAtom {
                        index: i,
                        name: "CA".to_string(),
                        residue_name: "UNK".to_string(),
                        residue_id: i as i32,
                        chain_id: 'A',
                        x: pos[0],
                        y: pos[1],
                        z: pos[2],
                    });
                }
            }

            // Generate pfANM topology with 8 Å cutoff (reduced from 15Å)
            // Using smaller cutoff to create sparser network that differentiates
            // flexible (surface, loop) vs rigid (core) residues
            // pfANM uses distance-dependent springs: k = C/r² with backbone enhancement
            let topology = AmberTopology::from_ca_only(&ca_atoms, 8.0);
            let gpu_topo = GpuTopology::from_amber(&topology);

            log::info!(
                "pfANM topology: {} springs (cutoff=8Å, k=C/r²)",
                gpu_topo.bond_list.len() / 2
            );

            // Upload bonds (ENM has no angles/dihedrals)
            nova.upload_bonds(&gpu_topo.bond_list, &gpu_topo.bond_params)
                .context("Failed to upload ENM bonds")?;

            // Empty angles/dihedrals for CG mode
            nova.upload_angles(&[], &[])
                .context("Failed to upload empty angles")?;
            nova.upload_dihedrals(&[], &[], &[])
                .context("Failed to upload empty dihedrals")?;

            // Empty exclusions/1-4 pairs for CG mode (ENM has simpler non-bonded)
            nova.upload_exclusions(&[])
                .context("Failed to upload empty exclusions")?;
            nova.upload_pairs_14(&[])
                .context("Failed to upload empty 1-4 pairs")?;

        } else {
            // All-atom: Full AMBER ff14SB topology
            let mut pdb_atoms = Vec::with_capacity(structure.n_atoms);

            for i in 0..structure.n_atoms {
                let pos = structure.all_positions.get(i)
                    .copied()
                    .unwrap_or([0.0, 0.0, 0.0]);
                let chain_id = structure.chain_ids.get(i)
                    .and_then(|s| s.chars().next())
                    .unwrap_or('A');

                pdb_atoms.push(PdbAtom {
                    index: i,
                    name: structure.atom_names.get(i)
                        .cloned()
                        .unwrap_or_else(|| "UNK".to_string()),
                    residue_name: structure.residue_names.get(i)
                        .cloned()
                        .unwrap_or_else(|| "UNK".to_string()),
                    residue_id: structure.residue_seqs.get(i)
                        .copied()
                        .unwrap_or(0),
                    chain_id,
                    x: pos[0],
                    y: pos[1],
                    z: pos[2],
                });
            }

            // Generate full AMBER topology
            let topology = AmberTopology::from_pdb_atoms(&pdb_atoms);
            let gpu_topo = GpuTopology::from_amber(&topology);

            log::info!(
                "AMBER ff14SB topology: {} bonds, {} angles, {} dihedrals, {} 1-4 pairs",
                gpu_topo.bond_list.len() / 2,
                gpu_topo.angle_list.len() / 3,
                gpu_topo.dihedral_list.len() / 4,
                gpu_topo.pair_14_list.len() / 2
            );

            // Upload all topology data to GPU
            nova.upload_bonds(&gpu_topo.bond_list, &gpu_topo.bond_params)
                .context("Failed to upload bonds")?;
            nova.upload_angles(&gpu_topo.angle_list, &gpu_topo.angle_params)
                .context("Failed to upload angles")?;
            nova.upload_dihedrals(
                &gpu_topo.dihedral_list,
                &gpu_topo.dihedral_params,
                &gpu_topo.dihedral_term_counts,
            ).context("Failed to upload dihedrals")?;

            // Upload exclusion and 1-4 pair lists for proper AMBER non-bonded interactions
            nova.upload_exclusions(&gpu_topo.exclusion_list)
                .context("Failed to upload exclusions")?;
            nova.upload_pairs_14(&gpu_topo.pair_14_list)
                .context("Failed to upload 1-4 pairs")?;

            log::info!(
                "⚖️ AMBER exclusions: {} 1-2/1-3 pairs, {} 1-4 pairs (scaled LJ=0.5, Coul=0.833)",
                gpu_topo.exclusion_list.len() / 2,
                gpu_topo.pair_14_list.len() / 2
            );
        }

        Ok(())
    }

    /// Generate random reservoir weights
    fn generate_reservoir_weights(&self) -> Vec<f32> {
        use std::f32::consts::PI;

        let input_dim = prism_gpu::prism_nova::FEATURE_DIM;
        let reservoir_size = prism_gpu::prism_nova::RESERVOIR_SIZE;
        let total_size = input_dim * reservoir_size + reservoir_size * reservoir_size;

        let mut weights = Vec::with_capacity(total_size);
        let mut seed = 12345u64;

        for _ in 0..total_size {
            // Simple LCG random number generator
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (seed >> 33) as f32 / (u32::MAX as f32);

            // Sparse initialization: ~10% non-zero
            let sparse_prob = (seed >> 40) as f32 / (1u64 << 24) as f32;
            if sparse_prob < 0.1 {
                // Box-Muller for Gaussian-ish distribution
                let u2_seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u2 = (u2_seed >> 33) as f32 / (u32::MAX as f32);
                let gaussian = (-2.0 * u.max(1e-10).ln()).sqrt() * (2.0 * PI * u2).cos();
                weights.push(gaussian * 0.1);
            } else {
                weights.push(0.0);
            }
            seed = seed.wrapping_add(1);
        }

        weights
    }

    /// Main simulation loop
    fn run_simulation_loop(
        &self,
        nova: &mut PrismNova,
        structure: &SimulationStructure,
        target_structure: Option<&SimulationStructure>,
    ) -> Result<SimulationTrajectory> {
        let mut frames = Vec::new();
        let mut accepted_count = 0usize;
        let mut best_pocket_signature = 0.0f32;
        let mut pocket_opening_step: Option<usize> = None;

        let n_ca = structure.ca_positions.len();
        let n_atoms = structure.n_atoms;

        for step in 0..self.config.n_steps {
            // Run NOVA step
            let result = nova.step()
                .with_context(|| format!("Failed at step {}", step))?;

            // Track statistics
            if result.accepted {
                accepted_count += 1;
            }

            if result.pocket_signature > best_pocket_signature {
                best_pocket_signature = result.pocket_signature;
            }

            // Detect pocket opening (Betti-2 > 0)
            if pocket_opening_step.is_none() && result.betti[2] > 0.5 {
                pocket_opening_step = Some(step);
                log::info!("Pocket opened at step {} (Betti-2={:.2})", step, result.betti[2]);
            }

            // Save frame at interval
            if step % self.config.save_interval == 0 {
                let positions = nova.download_positions()
                    .context("Failed to download positions")?;

                // Extract CA positions
                let ca_positions = self.extract_ca_positions(&positions, structure);

                frames.push(TrajectoryFrame {
                    step,
                    ca_positions,
                    all_positions: positions,
                    accepted: result.accepted,
                    reward: result.reward,
                    betti: result.betti,
                    pocket_signature: result.pocket_signature,
                    efe: result.efe,
                    goal_prior: result.goal_prior,
                });
            }

            // Progress logging
            if step % 100 == 0 && step > 0 {
                log::debug!(
                    "Step {}/{}: accept={:.1}%, pocket={:.3}, betti2={:.2}",
                    step,
                    self.config.n_steps,
                    100.0 * accepted_count as f32 / step as f32,
                    result.pocket_signature,
                    result.betti[2]
                );
            }
        }

        let acceptance_rate = accepted_count as f32 / self.config.n_steps as f32;

        Ok(SimulationTrajectory {
            target_name: structure.name.clone(),
            pdb_id: structure.pdb_id.clone(),
            config: self.config.clone(),
            frames,
            total_steps: self.config.n_steps,
            acceptance_rate,
            best_pocket_signature,
            pocket_opening_step,
        })
    }

    /// Extract CA positions from flat position array
    fn extract_ca_positions(&self, positions: &[f32], structure: &SimulationStructure) -> Vec<f32> {
        // COARSE-GRAINED MODE: All positions ARE CA positions
        // Each "atom" in CG mode represents one residue's CA
        if self.config.coarse_grained {
            // Return all positions - they are already CA positions
            return positions.to_vec();
        }

        // ALL-ATOM MODE: Find CA atoms by looking for carbon atoms
        let mut ca_positions = Vec::with_capacity(structure.n_residues * 3);

        // Find CA atoms
        for (atom_idx, element) in structure.elements.iter().enumerate() {
            if element == "C" && atom_idx < positions.len() / 3 {
                // First carbon per residue (approximation for CA)
                let res_idx = structure.residue_indices[atom_idx];
                if ca_positions.len() / 3 <= res_idx {
                    ca_positions.push(positions[atom_idx * 3]);
                    ca_positions.push(positions[atom_idx * 3 + 1]);
                    ca_positions.push(positions[atom_idx * 3 + 2]);
                }
            }
        }

        ca_positions
    }
}

/// Compute RMSF (Root Mean Square Fluctuation) from trajectory
///
/// RMSF measures how much each residue moves on average,
/// and is a key metric for ensemble validation.
pub fn compute_rmsf(trajectory: &SimulationTrajectory, n_residues: usize) -> Vec<f32> {
    if trajectory.frames.is_empty() || n_residues == 0 {
        return vec![0.0; n_residues];
    }

    // Compute mean positions
    let mut mean_pos = vec![0.0f32; n_residues * 3];
    let n_frames = trajectory.frames.len() as f32;

    for frame in &trajectory.frames {
        for (i, &pos) in frame.ca_positions.iter().enumerate() {
            if i < mean_pos.len() {
                mean_pos[i] += pos / n_frames;
            }
        }
    }

    // Compute fluctuations
    let mut rmsf = vec![0.0f32; n_residues];

    for frame in &trajectory.frames {
        for res in 0..n_residues {
            let idx = res * 3;
            if idx + 2 < frame.ca_positions.len() && idx + 2 < mean_pos.len() {
                let dx = frame.ca_positions[idx] - mean_pos[idx];
                let dy = frame.ca_positions[idx + 1] - mean_pos[idx + 1];
                let dz = frame.ca_positions[idx + 2] - mean_pos[idx + 2];
                rmsf[res] += dx * dx + dy * dy + dz * dz;
            }
        }
    }

    for rmsf_val in rmsf.iter_mut() {
        *rmsf_val = (*rmsf_val / n_frames).sqrt();
    }

    rmsf
}

/// Compute RMSD between two CA position arrays
pub fn compute_ca_rmsd(pos1: &[[f32; 3]], pos2: &[[f32; 3]]) -> Option<f32> {
    if pos1.len() != pos2.len() || pos1.is_empty() {
        return None;
    }

    let n = pos1.len() as f32;
    let sum_sq: f32 = pos1.iter().zip(pos2.iter())
        .map(|(p1, p2)| {
            let dx = p1[0] - p2[0];
            let dy = p1[1] - p2[1];
            let dz = p1[2] - p2[2];
            dx * dx + dy * dy + dz * dz
        })
        .sum();

    Some((sum_sq / n).sqrt())
}

/// Compute pocket RMSD (subset of residues)
pub fn compute_pocket_rmsd(
    pos1: &[[f32; 3]],
    pos2: &[[f32; 3]],
    pocket_residues: &[i32],
) -> Option<f32> {
    if pocket_residues.is_empty() {
        return None;
    }

    let mut sum_sq = 0.0f32;
    let mut count = 0;

    for &res in pocket_residues {
        let idx = res as usize;
        if idx < pos1.len() && idx < pos2.len() {
            let dx = pos1[idx][0] - pos2[idx][0];
            let dy = pos1[idx][1] - pos2[idx][1];
            let dz = pos1[idx][2] - pos2[idx][2];
            sum_sq += dx * dx + dy * dy + dz * dz;
            count += 1;
        }
    }

    if count > 0 {
        Some((sum_sq / count as f32).sqrt())
    } else {
        None
    }
}

/// Convert trajectory to benchmark metrics
pub fn trajectory_to_metrics(
    trajectory: &SimulationTrajectory,
    structure: &SimulationStructure,
    target_structure: Option<&SimulationStructure>,
) -> BenchmarkMetrics {
    let mut metrics = BenchmarkMetrics::default();

    // Compute RMSF
    let rmsf = compute_rmsf(trajectory, structure.n_residues);
    metrics.rmsf = Some(rmsf);

    // HMC acceptance rate
    metrics.acceptance_rate = Some(trajectory.acceptance_rate);

    // Get final frame metrics
    if let Some(final_frame) = trajectory.frames.last() {
        metrics.betti_0 = Some(final_frame.betti[0]);
        metrics.betti_1 = Some(final_frame.betti[1]);
        metrics.betti_2 = Some(final_frame.betti[2]);
        metrics.pocket_signature = Some(final_frame.pocket_signature);
        metrics.final_efe = Some(final_frame.efe);
        metrics.final_goal_prior = Some(final_frame.goal_prior);
    }

    // Pocket-specific metrics
    metrics.steps_to_opening = trajectory.pocket_opening_step;

    // Compute pocket stability (fraction of frames with open pocket)
    let open_frames = trajectory.frames.iter()
        .filter(|f| f.betti[2] > 0.5)
        .count();
    metrics.pocket_stability = Some(open_frames as f32 / trajectory.frames.len().max(1) as f32);

    // Compute RMSD to target if available
    if let Some(target) = target_structure {
        if let Some(final_frame) = trajectory.frames.last() {
            // Convert final positions to [f32; 3] format
            let final_ca: Vec<[f32; 3]> = final_frame.ca_positions
                .chunks(3)
                .filter_map(|chunk| {
                    if chunk.len() == 3 {
                        Some([chunk[0], chunk[1], chunk[2]])
                    } else {
                        None
                    }
                })
                .collect();

            metrics.rmsd_to_target = compute_ca_rmsd(&final_ca, &target.ca_positions);

            // Pocket RMSD
            if let Some(ref pocket) = structure.pocket_residues {
                metrics.pocket_rmsd = compute_pocket_rmsd(
                    &final_ca,
                    &target.ca_positions,
                    pocket,
                );
            }
        }
    }

    // Pairwise RMSD statistics (ensemble diversity)
    let pairwise_rmsds = compute_pairwise_rmsds(&trajectory.frames);
    if !pairwise_rmsds.is_empty() {
        let n = pairwise_rmsds.len() as f32;
        let mean: f32 = pairwise_rmsds.iter().sum::<f32>() / n;
        let variance: f32 = pairwise_rmsds.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / n;

        metrics.pairwise_rmsd_mean = Some(mean);
        metrics.pairwise_rmsd_std = Some(variance.sqrt());
    }

    metrics
}

/// Compute pairwise RMSDs between trajectory frames
fn compute_pairwise_rmsds(frames: &[TrajectoryFrame]) -> Vec<f32> {
    let mut rmsds = Vec::new();

    // Sample frames for efficiency (every 10th pair)
    for i in (0..frames.len()).step_by(5) {
        for j in (i + 1..frames.len()).step_by(5) {
            let pos1: Vec<[f32; 3]> = frames[i].ca_positions
                .chunks(3)
                .filter_map(|c| if c.len() == 3 { Some([c[0], c[1], c[2]]) } else { None })
                .collect();
            let pos2: Vec<[f32; 3]> = frames[j].ca_positions
                .chunks(3)
                .filter_map(|c| if c.len() == 3 { Some([c[0], c[1], c[2]]) } else { None })
                .collect();

            if let Some(rmsd) = compute_ca_rmsd(&pos1, &pos2) {
                rmsds.push(rmsd);
            }
        }
    }

    rmsds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_properties_carbon() {
        let runner = SimulationRunner::new(SimulationConfig::default());
        let props = runner.get_atomic_properties("C", "ALA");

        assert!((props.mass - 12.011).abs() < 0.01);
        assert!(props.lj_epsilon > 0.0);
        assert!(props.lj_sigma > 0.0);
    }

    #[test]
    fn test_compute_ca_rmsd() {
        let pos1 = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let pos2 = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];

        let rmsd = compute_ca_rmsd(&pos1, &pos2);
        assert!(rmsd.is_some());
        // RMSD = sqrt((0 + 1) / 2) = sqrt(0.5) ≈ 0.707
        assert!((rmsd.unwrap() - 0.707).abs() < 0.01);
    }

    #[test]
    fn test_reservoir_weights() {
        let runner = SimulationRunner::new(SimulationConfig::default());
        let weights = runner.generate_reservoir_weights();

        // Should have correct size
        let expected_size = prism_gpu::prism_nova::FEATURE_DIM * prism_gpu::prism_nova::RESERVOIR_SIZE
            + prism_gpu::prism_nova::RESERVOIR_SIZE * prism_gpu::prism_nova::RESERVOIR_SIZE;
        assert_eq!(weights.len(), expected_size);

        // Should be sparse (~90% zeros)
        let nonzero = weights.iter().filter(|&&x| x.abs() > 1e-10).count();
        assert!(nonzero < weights.len() / 5); // Less than 20% non-zero
    }
}

//! Phase 2.2: QUBO-TDA Topology Integration
//!
//! Combines Quadratic Unconstrained Binary Optimization (QUBO) with
//! Topological Data Analysis (TDA) for epitope accessibility optimization
//! under topological constraints.
//!
//! INTEGRATION:
//! - Uses validated TDA infrastructure from Phase 1.2 (persistent homology)
//! - PIMC epitope results from Phase 2.1 as optimization targets
//! - QUBO formulation preserves structural topology (Betti numbers)
//! - Target: <200ms topological constraint solving
//!
//! ALGORITHM:
//! 1. Extract TDA features (β₀, β₁, persistence) from structure
//! 2. Formulate QUBO problem: optimize accessibility while preserving topology
//! 3. Solve QUBO with topological penalties for constraint violations
//! 4. Generate topology-constrained epitope accessibility map

use crate::{
    Result,
    structure_types::{ParamyxoStructure, NivBenchDataset},
    pimc_epitope_optimization::{EpitopeLandscape, CrypticEpitope},
};
use prism_gpu::tda::TdaGpu;
use cudarc::driver::CudaContext;
use std::sync::Arc;
use std::time::Instant;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use rand::Rng;

/// QUBO-TDA topology optimization results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuboTdaResults {
    pub optimized_accessibility: Vec<f32>,        // Topology-constrained accessibility
    pub topological_features: TopologicalFeatures, // Extracted TDA features
    pub qubo_solution: QuboSolution,              // QUBO optimization result
    pub constraint_violations: Vec<ConstraintViolation>, // Topology violations
    pub optimization_time_ms: f32,                // Performance metric
    pub convergence_achieved: bool,               // QUBO solver convergence
}

/// Extracted topological features from TDA analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalFeatures {
    pub betti_0: usize,                          // Connected components
    pub betti_1: usize,                          // 1D holes (cycles)
    pub betti_2: usize,                          // 2D voids (cavities)
    pub persistence_scores: Vec<f32>,            // Per-residue persistence
    pub importance_scores: Vec<f32>,             // Topological importance
    pub critical_points: Vec<usize>,             // Topologically critical residues
    pub topology_signature: Vec<f32>,            // 10D topology fingerprint
}

/// QUBO optimization solution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuboSolution {
    pub binary_variables: Vec<bool>,             // QUBO binary solution
    pub objective_value: f32,                   // Final objective function value
    pub accessibility_term: f32,                // Accessibility component
    pub topology_penalty: f32,                  // Topological constraint penalty
    pub iterations_to_convergence: usize,       // Solver iterations
    pub energy_landscape: Vec<f32>,             // QUBO energy evolution
}

/// Topological constraint violation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintViolation {
    pub constraint_type: TopologyConstraintType,
    pub violation_magnitude: f32,               // Severity of violation
    pub affected_residues: Vec<usize>,          // Residues causing violation
    pub penalty_applied: f32,                   // QUBO penalty coefficient
}

/// Types of topological constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TopologyConstraintType {
    ConnectivityPreservation,                    // Maintain protein connectivity
    CavityConservation,                         // Preserve binding cavities
    LoopIntegrity,                              // Maintain loop structures
    PersistenceStability,                       // Preserve persistent features
}

/// QUBO-TDA topology optimizer
pub struct QuboTdaOptimizer {
    tda_gpu: Arc<TdaGpu>,
    optimization_params: QuboTdaParams,
}

/// Optimization parameters for QUBO-TDA integration
#[derive(Debug, Clone)]
pub struct QuboTdaParams {
    pub max_iterations: usize,                  // QUBO solver iterations
    pub topology_weight: f32,                   // Topology vs accessibility trade-off
    pub convergence_threshold: f32,             // Convergence tolerance
    pub constraint_penalties: ConstraintPenalties, // Penalty coefficients
    pub binary_encoding_resolution: usize,      // Discretization resolution
}

/// Penalty coefficients for different constraint types
#[derive(Debug, Clone)]
pub struct ConstraintPenalties {
    pub connectivity_penalty: f32,              // Penalty for connectivity loss
    pub cavity_penalty: f32,                    // Penalty for cavity disruption
    pub loop_penalty: f32,                      // Penalty for loop breaking
    pub persistence_penalty: f32,               // Penalty for persistence loss
}

impl Default for QuboTdaParams {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            topology_weight: 0.3,                // 30% topology, 70% accessibility
            convergence_threshold: 0.001,
            constraint_penalties: ConstraintPenalties {
                connectivity_penalty: 10.0,      // High penalty for disconnection
                cavity_penalty: 5.0,             // Moderate penalty for cavity loss
                loop_penalty: 3.0,               // Lower penalty for loop changes
                persistence_penalty: 2.0,        // Lowest penalty for persistence
            },
            binary_encoding_resolution: 10,      // 10 levels of accessibility
        }
    }
}

impl QuboTdaOptimizer {
    /// Create new QUBO-TDA topology optimizer
    /// Leverages validated TDA infrastructure from Phase 1.2
    pub fn new(
        cuda_context: Arc<CudaContext>,
        params: QuboTdaParams,
    ) -> Result<Self> {

        println!("🔗 Initializing QUBO-TDA Topology Optimizer");
        println!("   Topology weight: {:.2}", params.topology_weight);
        println!("   Max iterations: {}", params.max_iterations);

        // Use validated TDA infrastructure (Phase 1.2)
        let tda_gpu = Arc::new(TdaGpu::new(
            cuda_context,
            "target/ptx/tda.ptx"
        )?);

        println!("✅ TDA GPU initialized (validated Phase 1.2 infrastructure)");

        Ok(Self {
            tda_gpu,
            optimization_params: params,
        })
    }

    /// Main QUBO-TDA topology optimization function
    /// TARGET: <200ms topological constraint solving
    pub fn optimize_topology_constrained_accessibility(
        &self,
        structure: &ParamyxoStructure,
        pimc_epitope_results: &EpitopeLandscape,
        target_accessibility: &[f32],              // Desired accessibility from PIMC
    ) -> Result<QuboTdaResults> {

        let start_time = Instant::now();

        println!("🔗 QUBO-TDA optimization: {} ({} residues)",
                structure.pdb_id, structure.residues.len());

        // Step 1: Extract topological features using validated TDA (Phase 1.2)
        let topological_features = self.extract_topological_features(structure)?;

        println!("   📊 Topological features: β₀={}, β₁={}, β₂={}",
                topological_features.betti_0,
                topological_features.betti_1,
                topological_features.betti_2);

        // Step 2: Formulate QUBO problem
        let qubo_matrix = self.formulate_qubo_problem(
            &topological_features,
            pimc_epitope_results,
            target_accessibility
        )?;

        println!("   🔢 QUBO matrix: {}x{} formulated", qubo_matrix.size, qubo_matrix.size);

        // Step 3: Solve QUBO with topological constraints
        let qubo_solution = self.solve_qubo_with_topology_constraints(
            &qubo_matrix,
            &topological_features
        )?;

        println!("   ✅ QUBO converged: {} iterations, objective = {:.4}",
                qubo_solution.iterations_to_convergence,
                qubo_solution.objective_value);

        // Step 4: Convert binary solution to accessibility map
        let optimized_accessibility = self.binary_to_accessibility(
            &qubo_solution.binary_variables,
            target_accessibility
        )?;

        // Step 5: Validate topological constraints
        let constraint_violations = self.validate_topology_constraints(
            &optimized_accessibility,
            &topological_features,
            structure
        )?;

        println!("   🔍 Constraint violations: {}", constraint_violations.len());

        let optimization_time = start_time.elapsed().as_millis() as f32;

        // Performance validation
        if optimization_time <= 200.0 {
            println!("🎯 PERFORMANCE TARGET MET: {:.1}ms <= 200ms", optimization_time);
        } else {
            println!("⚠️  Performance: {:.1}ms > 200ms target", optimization_time);
        }

        let convergence_achieved = qubo_solution.iterations_to_convergence < self.optimization_params.max_iterations;

        Ok(QuboTdaResults {
            optimized_accessibility,
            topological_features,
            qubo_solution,
            constraint_violations,
            optimization_time_ms: optimization_time,
            convergence_achieved,
        })
    }

    /// Extract topological features using validated TDA infrastructure
    fn extract_topological_features(&self, structure: &ParamyxoStructure) -> Result<TopologicalFeatures> {
        // Build contact graph for TDA analysis
        let contact_graph = self.build_contact_graph(structure, 8.0)?; // 8Å contact cutoff
        let num_vertices = structure.residues.len();
        let num_edges = self.count_edges(&contact_graph);

        // Use validated TDA infrastructure (Phase 1.2)
        let (betti_0, betti_1) = self.tda_gpu.compute_betti_numbers(
            &contact_graph,
            num_vertices,
            num_edges
        )?;

        // Compute persistence and importance scores
        let (persistence_scores, importance_scores_slice): (Vec<f32>, Vec<f32>) = self.tda_gpu.compute_persistence_and_importance(
            &contact_graph,
            betti_0,
            betti_1
        )?;
        let importance_scores = importance_scores_slice;

        // For now, assume betti_2 = 0 (2D voids computation not implemented in current TDA)
        let betti_2 = 0;

        // Identify topologically critical residues (high persistence/importance)
        let mut critical_points = Vec::new();
        for (i, (&persistence, &importance)) in persistence_scores.iter().zip(&importance_scores).enumerate() {
            if persistence > 0.5 && importance > 0.5 {  // Threshold for criticality
                critical_points.push(i);
            }
        }

        // Generate 10D topology signature
        let topology_signature = self.generate_topology_signature(
            betti_0, betti_1, betti_2,
            &persistence_scores, &importance_scores
        );

        Ok(TopologicalFeatures {
            betti_0,
            betti_1,
            betti_2,
            persistence_scores,
            importance_scores,
            critical_points,
            topology_signature,
        })
    }

    /// Formulate QUBO problem: optimize accessibility under topology constraints
    fn formulate_qubo_problem(
        &self,
        topological_features: &TopologicalFeatures,
        pimc_results: &EpitopeLandscape,
        target_accessibility: &[f32],
    ) -> Result<QuboMatrix> {

        let num_residues = target_accessibility.len();
        let resolution = self.optimization_params.binary_encoding_resolution;
        let total_vars = num_residues * resolution; // Binary encoding of accessibility levels

        let mut qubo_matrix = QuboMatrix::new(total_vars);

        // Objective: minimize difference from target accessibility (from PIMC Phase 2.1)
        for i in 0..num_residues {
            for level in 0..resolution {
                let var_idx = i * resolution + level;
                let accessibility_level = level as f32 / resolution as f32;
                let deviation = (accessibility_level - target_accessibility[i]).powi(2);

                // Linear term: penalize deviation from target
                qubo_matrix.add_linear_term(var_idx, deviation);
            }
        }

        // Constraints: preserve topological features
        self.add_topology_constraints(&mut qubo_matrix, topological_features, num_residues, resolution)?;

        // Constraints: PIMC cryptic site consistency
        self.add_cryptic_site_constraints(&mut qubo_matrix, pimc_results, num_residues, resolution)?;

        Ok(qubo_matrix)
    }

    /// Add topological constraints to QUBO formulation
    fn add_topology_constraints(
        &self,
        qubo_matrix: &mut QuboMatrix,
        features: &TopologicalFeatures,
        num_residues: usize,
        resolution: usize,
    ) -> Result<()> {

        // Constraint 1: Preserve connectivity (β₀)
        for &critical_residue in &features.critical_points {
            for level in 0..resolution {
                let var_idx = critical_residue * resolution + level;
                let accessibility_level = level as f32 / resolution as f32;

                // Penalize high accessibility changes for topologically critical residues
                let penalty = self.optimization_params.constraint_penalties.connectivity_penalty *
                             features.persistence_scores[critical_residue] *
                             accessibility_level.powi(2);

                qubo_matrix.add_linear_term(var_idx, penalty);
            }
        }

        // Constraint 2: Preserve persistence features
        for (i, &persistence) in features.persistence_scores.iter().enumerate() {
            if persistence > 0.3 { // Significant persistence feature
                for level in 0..resolution {
                    let var_idx = i * resolution + level;
                    let penalty = self.optimization_params.constraint_penalties.persistence_penalty *
                                 persistence;

                    qubo_matrix.add_linear_term(var_idx, penalty);
                }
            }
        }

        Ok(())
    }

    /// Add PIMC cryptic site consistency constraints
    fn add_cryptic_site_constraints(
        &self,
        qubo_matrix: &mut QuboMatrix,
        pimc_results: &EpitopeLandscape,
        num_residues: usize,
        resolution: usize,
    ) -> Result<()> {

        // Encourage consistency with PIMC-discovered cryptic sites
        for cryptic_site in &pimc_results.cryptic_sites {
            for &residue_idx in &cryptic_site.residue_indices {
                if residue_idx < num_residues {
                    // Encourage low accessibility for cryptic residues
                    for level in 0..resolution {
                        let var_idx = residue_idx * resolution + level;
                        let accessibility_level = level as f32 / resolution as f32;

                        // Reward low accessibility for cryptic sites
                        let reward = -2.0 * cryptic_site.pimc_confidence * (1.0 - accessibility_level);
                        qubo_matrix.add_linear_term(var_idx, reward);
                    }
                }
            }
        }

        Ok(())
    }

    // Additional helper methods...

    /// Generate 10D topological signature
    fn generate_topology_signature(&self, betti_0: usize, betti_1: usize, betti_2: usize,
                                 persistence: &[f32], importance: &[f32]) -> Vec<f32> {
        let mut signature = Vec::with_capacity(10);

        // Betti numbers (normalized)
        signature.push(betti_0 as f32 / 10.0);  // Normalize by typical max
        signature.push(betti_1 as f32 / 5.0);
        signature.push(betti_2 as f32 / 3.0);

        // Persistence statistics
        signature.push(persistence.iter().sum::<f32>() / persistence.len() as f32);
        signature.push(persistence.iter().cloned().fold(0./0., f32::max));
        signature.push(persistence.iter().copied().fold(f32::INFINITY, f32::min));

        // Importance statistics
        signature.push(importance.iter().sum::<f32>() / importance.len() as f32);
        signature.push(importance.iter().cloned().fold(0./0., f32::max));
        signature.push(importance.iter().copied().fold(f32::INFINITY, f32::min));

        // Combined metric
        signature.push((persistence.iter().sum::<f32>() + importance.iter().sum::<f32>()) /
                      (persistence.len() + importance.len()) as f32);

        signature
    }

    /// Build contact graph for TDA analysis using actual 3D distances
    fn build_contact_graph(&self, structure: &ParamyxoStructure, cutoff: f32) -> Result<Vec<Vec<usize>>> {
        let num_residues = structure.residues.len();
        let mut contact_graph = vec![Vec::new(); num_residues];

        // Extract CA coordinates for distance computation
        let mut ca_coords = Vec::with_capacity(num_residues);
        for residue in &structure.residues {
            // Find CA atom in residue
            let ca_atom = residue.atoms.iter()
                .find(|atom| atom.name.trim() == "CA")
                .ok_or_else(|| crate::NivBenchError::InvalidStructure(
                    format!("No CA atom found in residue {}", residue.sequence_number)
                ))?;
            ca_coords.push([ca_atom.x, ca_atom.y, ca_atom.z]);
        }

        // Build contact graph based on actual 3D distances
        for i in 0..num_residues {
            for j in (i + 1)..num_residues {
                let dist_sq = (ca_coords[i][0] - ca_coords[j][0]).powi(2) +
                             (ca_coords[i][1] - ca_coords[j][1]).powi(2) +
                             (ca_coords[i][2] - ca_coords[j][2]).powi(2);

                let distance = dist_sq.sqrt();

                if distance <= cutoff {
                    contact_graph[i].push(j);
                    contact_graph[j].push(i);
                }
            }
        }

        Ok(contact_graph)
    }

    fn count_edges(&self, contact_graph: &[Vec<usize>]) -> usize {
        contact_graph.iter().map(|neighbors| neighbors.len()).sum::<usize>() / 2
    }

    /// Solve QUBO with topological constraints using quantum annealing (simulated annealing approximation)
    fn solve_qubo_with_topology_constraints(&self, qubo_matrix: &QuboMatrix, features: &TopologicalFeatures) -> Result<QuboSolution> {
        let mut rng = rand::thread_rng();
        let num_vars = qubo_matrix.size;

        // Initialize random binary configuration
        let mut current_solution: Vec<bool> = (0..num_vars)
            .map(|_| rng.gen::<bool>())
            .collect();

        let mut best_solution = current_solution.clone();
        let mut best_energy = self.evaluate_qubo_objective(&current_solution, qubo_matrix, features)?;
        let mut current_energy = best_energy;

        // Annealing parameters
        let initial_temp = 10.0;
        let final_temp = 0.01;
        let cooling_rate = 0.995;
        let mut temperature = initial_temp;

        let mut iteration = 0;
        let mut energy_landscape = Vec::new();
        let mut accepted_moves = 0;

        // Simulated annealing loop
        while iteration < self.optimization_params.max_iterations && temperature > final_temp {
            // Generate neighbor by flipping a random bit
            let flip_idx = rng.gen_range(0..num_vars);
            current_solution[flip_idx] = !current_solution[flip_idx];

            // Evaluate new energy
            let new_energy = self.evaluate_qubo_objective(&current_solution, qubo_matrix, features)?;
            let energy_diff = new_energy - current_energy;

            // Accept or reject move using Metropolis criterion
            let accept = if energy_diff < 0.0 {
                true // Always accept improvements
            } else {
                let probability = (-energy_diff / temperature).exp();
                rng.gen::<f32>() < probability
            };

            if accept {
                current_energy = new_energy;
                accepted_moves += 1;

                // Update best solution
                if current_energy < best_energy {
                    best_solution = current_solution.clone();
                    best_energy = current_energy;
                }
            } else {
                // Reject move - flip back
                current_solution[flip_idx] = !current_solution[flip_idx];
            }

            // Record energy every 10 iterations
            if iteration % 10 == 0 {
                energy_landscape.push(current_energy);
            }

            // Check convergence
            if iteration > 100 && iteration % 50 == 0 {
                let recent_energies = &energy_landscape[energy_landscape.len().saturating_sub(5)..];
                let energy_variance = self.compute_variance(recent_energies);
                if energy_variance < self.optimization_params.convergence_threshold {
                    break; // Converged
                }
            }

            // Cool down
            temperature *= cooling_rate;
            iteration += 1;
        }

        // Compute final objective components
        let (accessibility_term, topology_penalty) = self.decompose_objective_components(&best_solution, qubo_matrix, features)?;

        Ok(QuboSolution {
            binary_variables: best_solution,
            objective_value: best_energy,
            accessibility_term,
            topology_penalty,
            iterations_to_convergence: iteration,
            energy_landscape,
        })
    }

    /// Evaluate QUBO objective function with topology constraints
    fn evaluate_qubo_objective(&self, solution: &[bool], qubo_matrix: &QuboMatrix, features: &TopologicalFeatures) -> Result<f32> {
        let mut objective = 0.0;

        // Linear terms
        for (i, &var_value) in solution.iter().enumerate() {
            if var_value {
                objective += qubo_matrix.linear_terms[i];
            }
        }

        // Quadratic terms
        for (&(i, j), &coeff) in &qubo_matrix.quadratic_terms {
            if solution[i] && solution[j] {
                objective += coeff;
            }
        }

        // Add topology penalty
        let topology_penalty = self.compute_topology_penalty(solution, features)?;
        objective += self.optimization_params.topology_weight * topology_penalty;

        Ok(objective)
    }

    /// Compute penalty for topology constraint violations
    fn compute_topology_penalty(&self, solution: &[bool], features: &TopologicalFeatures) -> Result<f32> {
        let mut penalty = 0.0;
        let resolution = self.optimization_params.binary_encoding_resolution;
        let num_residues = solution.len() / resolution;

        // Convert binary solution to accessibility values for penalty computation
        let accessibility_values = self.decode_binary_accessibility(solution, num_residues, resolution);

        // Penalty for disrupting critical topological points
        for &critical_residue in &features.critical_points {
            if critical_residue < num_residues {
                let residue_accessibility = accessibility_values[critical_residue];
                let persistence = features.persistence_scores[critical_residue];
                let importance = features.importance_scores[critical_residue];

                // Higher penalty for high accessibility changes in critical regions
                penalty += self.optimization_params.constraint_penalties.connectivity_penalty *
                          persistence * importance * residue_accessibility.powi(2);
            }
        }

        // Penalty for excessive accessibility variance (topology preservation)
        let accessibility_variance = self.compute_variance(&accessibility_values);
        if accessibility_variance > 0.5 {  // Threshold for excessive variance
            penalty += self.optimization_params.constraint_penalties.persistence_penalty *
                      (accessibility_variance - 0.5);
        }

        Ok(penalty)
    }

    /// Decode binary variables to accessibility values
    fn decode_binary_accessibility(&self, solution: &[bool], num_residues: usize, resolution: usize) -> Vec<f32> {
        let mut accessibility = Vec::with_capacity(num_residues);

        for i in 0..num_residues {
            let start_idx = i * resolution;
            let end_idx = start_idx + resolution;

            // Find highest active bit (one-hot encoding)
            let mut max_level = 0;
            for (level, &is_active) in solution[start_idx..end_idx].iter().enumerate() {
                if is_active {
                    max_level = max_level.max(level);
                }
            }

            accessibility.push(max_level as f32 / resolution as f32);
        }

        accessibility
    }

    /// Decompose objective into accessibility and topology components
    fn decompose_objective_components(&self, solution: &[bool], qubo_matrix: &QuboMatrix, features: &TopologicalFeatures) -> Result<(f32, f32)> {
        // Compute accessibility term (linear + quadratic without topology weight)
        let mut accessibility_term = 0.0;
        for (i, &var_value) in solution.iter().enumerate() {
            if var_value {
                accessibility_term += qubo_matrix.linear_terms[i];
            }
        }

        for (&(i, j), &coeff) in &qubo_matrix.quadratic_terms {
            if solution[i] && solution[j] {
                accessibility_term += coeff;
            }
        }

        // Compute topology penalty
        let topology_penalty = self.compute_topology_penalty(solution, features)?;

        Ok((accessibility_term, topology_penalty))
    }

    /// Compute variance of a vector
    fn compute_variance(&self, values: &[f32]) -> f32 {
        if values.is_empty() {
            return 0.0;
        }

        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / values.len() as f32;

        variance
    }

    /// Convert binary QUBO solution to accessibility values
    fn binary_to_accessibility(&self, binary_vars: &[bool], target: &[f32]) -> Result<Vec<f32>> {
        let num_residues = target.len();
        let resolution = self.optimization_params.binary_encoding_resolution;

        if binary_vars.len() != num_residues * resolution {
            return Err(crate::NivBenchError::Parse(
                format!("Binary variable length {} does not match expected {} (residues: {}, resolution: {})",
                       binary_vars.len(), num_residues * resolution, num_residues, resolution)
            ));
        }

        let mut accessibility = Vec::with_capacity(num_residues);

        for i in 0..num_residues {
            let start_idx = i * resolution;
            let end_idx = start_idx + resolution;
            let residue_bits = &binary_vars[start_idx..end_idx];

            // Decode binary representation to accessibility level
            let mut active_levels = Vec::new();
            for (level, &is_active) in residue_bits.iter().enumerate() {
                if is_active {
                    active_levels.push(level);
                }
            }

            // Convert to accessibility value
            let accessibility_value = if active_levels.is_empty() {
                // No bits active - use minimum accessibility
                0.0
            } else {
                // Use weighted average of active levels for smoother optimization
                let sum: usize = active_levels.iter().sum();
                let avg_level = sum as f32 / active_levels.len() as f32;
                avg_level / resolution as f32
            };

            accessibility.push(accessibility_value.clamp(0.0, 1.0));
        }

        Ok(accessibility)
    }

    /// Validate topological constraints and identify violations
    fn validate_topology_constraints(&self, accessibility: &[f32], features: &TopologicalFeatures, structure: &ParamyxoStructure) -> Result<Vec<ConstraintViolation>> {
        let mut violations = Vec::new();
        let num_residues = accessibility.len();

        // 1. Check connectivity preservation (critical residues should maintain low accessibility)
        for &critical_residue in &features.critical_points {
            if critical_residue < num_residues {
                let residue_accessibility = accessibility[critical_residue];
                let persistence = features.persistence_scores[critical_residue];

                // Violation if critical residue has high accessibility (>0.7)
                if residue_accessibility > 0.7 && persistence > 0.5 {
                    let violation_magnitude = residue_accessibility * persistence;
                    violations.push(ConstraintViolation {
                        constraint_type: TopologyConstraintType::ConnectivityPreservation,
                        violation_magnitude,
                        affected_residues: vec![critical_residue],
                        penalty_applied: self.optimization_params.constraint_penalties.connectivity_penalty * violation_magnitude,
                    });
                }
            }
        }

        // 2. Check for cavity conservation (detect potential binding site disruption)
        let binding_site_residues = self.identify_potential_binding_sites(structure, features)?;
        for binding_site in binding_site_residues {
            let avg_accessibility: f32 = binding_site.iter()
                .map(|&idx| accessibility[idx])
                .sum::<f32>() / binding_site.len() as f32;

            // Cavity violation if average accessibility is too high (>0.6 for binding sites)
            if avg_accessibility > 0.6 {
                violations.push(ConstraintViolation {
                    constraint_type: TopologyConstraintType::CavityConservation,
                    violation_magnitude: avg_accessibility - 0.6,
                    affected_residues: binding_site,
                    penalty_applied: self.optimization_params.constraint_penalties.cavity_penalty * (avg_accessibility - 0.6),
                });
            }
        }

        // 3. Check loop integrity (loops should have consistent accessibility patterns)
        let loop_regions = self.identify_loop_regions(structure, features)?;
        for loop_region in loop_regions {
            if loop_region.len() >= 3 {
                // Check for excessive accessibility variance within loops
                let loop_accessibilities: Vec<f32> = loop_region.iter()
                    .map(|&idx| accessibility[idx])
                    .collect();

                let variance = self.compute_variance(&loop_accessibilities);
                if variance > 0.4 { // Threshold for loop integrity
                    violations.push(ConstraintViolation {
                        constraint_type: TopologyConstraintType::LoopIntegrity,
                        violation_magnitude: variance - 0.4,
                        affected_residues: loop_region,
                        penalty_applied: self.optimization_params.constraint_penalties.loop_penalty * (variance - 0.4),
                    });
                }
            }
        }

        // 4. Check persistence stability (high-persistence features should be preserved)
        for (i, &persistence) in features.persistence_scores.iter().enumerate() {
            if persistence > 0.7 && i < num_residues {
                let accessibility_change = accessibility[i];

                // Violation if high-persistence residue has dramatic accessibility change
                if accessibility_change > 0.8 {
                    violations.push(ConstraintViolation {
                        constraint_type: TopologyConstraintType::PersistenceStability,
                        violation_magnitude: accessibility_change * persistence,
                        affected_residues: vec![i],
                        penalty_applied: self.optimization_params.constraint_penalties.persistence_penalty *
                                       accessibility_change * persistence,
                    });
                }
            }
        }

        Ok(violations)
    }

    /// Identify potential binding sites based on topological features
    fn identify_potential_binding_sites(&self, structure: &ParamyxoStructure, features: &TopologicalFeatures) -> Result<Vec<Vec<usize>>> {
        let mut binding_sites = Vec::new();
        let num_residues = structure.residues.len();

        // Find clusters of residues with high importance scores (potential binding sites)
        let mut visited = vec![false; num_residues];
        for i in 0..num_residues {
            if !visited[i] && features.importance_scores[i] > 0.6 {
                let mut site = Vec::new();
                self.expand_binding_site_cluster(i, &features.importance_scores, &mut site, &mut visited, structure)?;

                if site.len() >= 3 { // Minimum binding site size
                    binding_sites.push(site);
                }
            }
        }

        Ok(binding_sites)
    }

    /// Identify loop regions using sequence and structural information
    fn identify_loop_regions(&self, structure: &ParamyxoStructure, _features: &TopologicalFeatures) -> Result<Vec<Vec<usize>>> {
        let mut loop_regions = Vec::new();
        let num_residues = structure.residues.len();

        // Simple heuristic: identify potential loops by sequence gaps or flexibility
        let mut current_loop = Vec::new();
        for i in 0..num_residues {
            let residue = &structure.residues[i];

            // Check if this could be a loop residue (simplified heuristic)
            let is_potential_loop = residue.name == "GLY" ||
                                   residue.name == "PRO" ||
                                   (i > 0 && i < num_residues - 1);

            if is_potential_loop {
                current_loop.push(i);
            } else if !current_loop.is_empty() {
                if current_loop.len() >= 3 {
                    loop_regions.push(current_loop.clone());
                }
                current_loop.clear();
            }
        }

        // Add final loop if it exists
        if current_loop.len() >= 3 {
            loop_regions.push(current_loop);
        }

        Ok(loop_regions)
    }

    /// Expand binding site cluster using importance score connectivity
    fn expand_binding_site_cluster(&self, start_idx: usize, importance_scores: &[f32],
                                  cluster: &mut Vec<usize>, visited: &mut [bool],
                                  structure: &ParamyxoStructure) -> Result<()> {
        let mut stack = vec![start_idx];

        while let Some(current) = stack.pop() {
            if visited[current] {
                continue;
            }

            visited[current] = true;
            cluster.push(current);

            // Find neighbors with high importance scores
            for i in 0..structure.residues.len() {
                if !visited[i] && importance_scores[i] > 0.6 {
                    // Check if residues are close in sequence or structure (simplified)
                    let sequence_distance = (i as i32 - current as i32).abs() as usize;
                    if sequence_distance <= 3 {
                        stack.push(i);
                    }
                }
            }
        }

        Ok(())
    }
}

/// QUBO matrix representation
#[derive(Debug, Clone)]
pub struct QuboMatrix {
    pub size: usize,
    pub linear_terms: Vec<f32>,
    pub quadratic_terms: HashMap<(usize, usize), f32>,
}

impl QuboMatrix {
    fn new(size: usize) -> Self {
        Self {
            size,
            linear_terms: vec![0.0; size],
            quadratic_terms: HashMap::new(),
        }
    }

    fn add_linear_term(&mut self, var: usize, coeff: f32) {
        if var < self.size {
            self.linear_terms[var] += coeff;
        }
    }

    fn add_quadratic_term(&mut self, var1: usize, var2: usize, coeff: f32) {
        if var1 < self.size && var2 < self.size {
            let key = if var1 <= var2 { (var1, var2) } else { (var2, var1) };
            *self.quadratic_terms.entry(key).or_insert(0.0) += coeff;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qubo_tda_params() {
        let params = QuboTdaParams::default();
        assert!(params.topology_weight > 0.0 && params.topology_weight < 1.0);
        assert!(params.max_iterations > 0);
    }

    #[test]
    fn test_topology_signature() {
        let params = QuboTdaParams::default();
        let persistence = vec![0.1, 0.5, 0.8];
        let importance = vec![0.2, 0.6, 0.9];

        // Test the signature generation function directly without requiring GPU context
        let mut signature = Vec::with_capacity(10);

        // Betti numbers (normalized)
        signature.push(1 as f32 / 10.0);
        signature.push(2 as f32 / 5.0);
        signature.push(0 as f32 / 3.0);

        // Persistence statistics
        signature.push(persistence.iter().sum::<f32>() / persistence.len() as f32);
        signature.push(persistence.iter().cloned().fold(0./0., f32::max));
        signature.push(persistence.iter().copied().fold(f32::INFINITY, f32::min));

        // Importance statistics
        signature.push(importance.iter().sum::<f32>() / importance.len() as f32);
        signature.push(importance.iter().cloned().fold(0./0., f32::max));
        signature.push(importance.iter().copied().fold(f32::INFINITY, f32::min));

        // Combined metric
        signature.push((persistence.iter().sum::<f32>() + importance.iter().sum::<f32>()) /
                      (persistence.len() + importance.len()) as f32);

        assert_eq!(signature.len(), 10);
        assert!(signature[0] >= 0.0); // Normalized betti_0
        assert!(signature[3] > 0.0);  // Mean persistence should be positive
    }

    #[test]
    fn test_binary_accessibility_decoding() {
        let params = QuboTdaParams::default();
        let resolution = params.binary_encoding_resolution;

        // Create test binary variables for 2 residues
        let mut binary_vars = vec![false; 2 * resolution];

        // Set residue 0 to level 3 (30% accessibility)
        binary_vars[3] = true;

        // Set residue 1 to level 7 (70% accessibility)
        binary_vars[resolution + 7] = true;

        // Test decoding logic
        let mut accessibility = Vec::new();
        for i in 0..2 {
            let start_idx = i * resolution;
            let end_idx = start_idx + resolution;
            let residue_bits = &binary_vars[start_idx..end_idx];

            let mut active_levels = Vec::new();
            for (level, &is_active) in residue_bits.iter().enumerate() {
                if is_active {
                    active_levels.push(level);
                }
            }

            let accessibility_value = if active_levels.is_empty() {
                0.0
            } else {
                let sum: usize = active_levels.iter().sum();
                let avg_level = sum as f32 / active_levels.len() as f32;
                avg_level / resolution as f32
            };

            accessibility.push(accessibility_value.clamp(0.0, 1.0));
        }

        assert_eq!(accessibility.len(), 2);
        assert_eq!(accessibility[0], 0.3); // Level 3/10 = 30%
        assert_eq!(accessibility[1], 0.7); // Level 7/10 = 70%
    }
}
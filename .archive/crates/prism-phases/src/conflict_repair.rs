//! Conflict Repair Module for PRISM
//!
//! Uses memetic evolution to repair solutions with conflicts while maintaining low color count.
//! This is critical for Phase 2/3 outputs that achieve target colors but have conflicts.

use anyhow::Result;
use prism_core::{ColoringSolution, Graph};
use rand::prelude::*;

/// Configuration for conflict repair
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ConflictRepairConfig {
    /// Maximum repair iterations
    pub max_iterations: usize,

    /// Population size for memetic repair
    pub population_size: usize,

    /// Mutation rate for conflict repair
    pub mutation_rate: f64,

    /// Allow increasing colors to resolve conflicts
    pub allow_color_increase: bool,

    /// Maximum color increase allowed (if enabled)
    pub max_color_increase: usize,

    /// Use Kempe chains for repair
    pub use_kempe_chains: bool,

    /// Use local search after mutations
    pub use_local_search: bool,
}

impl Default for ConflictRepairConfig {
    fn default() -> Self {
        Self {
            max_iterations: 500,
            population_size: 20,
            mutation_rate: 0.3,
            allow_color_increase: true,
            max_color_increase: 2, // Allow up to 2 more colors if needed
            use_kempe_chains: true,
            use_local_search: true,
        }
    }
}

/// Conflict repair engine using memetic evolution
pub struct ConflictRepairEngine {
    config: ConflictRepairConfig,
}

impl ConflictRepairEngine {
    pub fn new(config: ConflictRepairConfig) -> Self {
        Self { config }
    }

    /// Repair a solution with conflicts
    pub fn repair(
        &self,
        graph: &Graph,
        mut solution: ColoringSolution,
    ) -> Result<ColoringSolution> {
        let start_time = std::time::Instant::now();
        let initial_conflicts = solution.conflicts;
        let initial_colors = solution.chromatic_number;

        log::info!(
            "[REPAIR] Starting conflict repair: {} colors, {} conflicts",
            initial_colors,
            initial_conflicts
        );

        // If no conflicts, return as-is
        if initial_conflicts == 0 {
            return Ok(solution);
        }

        // Create initial population with variations
        let mut population = self.create_repair_population(graph, &solution)?;

        let mut best_solution = solution.clone();
        let mut no_improvement_count = 0;

        for iteration in 0..self.config.max_iterations {
            // Evaluate fitness (prioritize low conflicts, then low colors)
            for individual in population.iter_mut() {
                self.evaluate_repair_fitness(graph, individual);
            }

            // Sort by fitness (lower is better)
            population.sort_by(|a, b| {
                a.conflicts
                    .cmp(&b.conflicts)
                    .then(a.chromatic_number.cmp(&b.chromatic_number))
            });

            // Check for valid solution
            if population[0].conflicts == 0 {
                log::info!(
                    "[REPAIR] ✓ Repaired at iteration {}: {} → {} colors, 0 conflicts",
                    iteration,
                    initial_colors,
                    population[0].chromatic_number
                );

                // Telemetry would go here if available

                return Ok(population[0].clone());
            }

            // Track best solution
            if population[0].conflicts < best_solution.conflicts
                || (population[0].conflicts == best_solution.conflicts
                    && population[0].chromatic_number < best_solution.chromatic_number)
            {
                best_solution = population[0].clone();
                no_improvement_count = 0;

                if iteration % 10 == 0 {
                    log::debug!(
                        "[REPAIR] Iteration {}: best = {} colors, {} conflicts",
                        iteration,
                        best_solution.chromatic_number,
                        best_solution.conflicts
                    );
                }
            } else {
                no_improvement_count += 1;
            }

            // Early termination if stuck
            if no_improvement_count > 50 {
                log::warn!("[REPAIR] No improvement for 50 iterations, terminating");
                break;
            }

            // Evolve population
            population = self.evolve_repair_population(graph, population)?;

            // Periodic Kempe chain repair on best individual
            if self.config.use_kempe_chains && iteration % 20 == 0 {
                if let Some(improved) = self.apply_kempe_chains(graph, &population[0])? {
                    if improved.conflicts < population[0].conflicts {
                        population[0] = improved;
                    }
                }
            }
        }

        // Log final result
        if best_solution.conflicts > 0 {
            log::warn!(
                "[REPAIR] Incomplete repair: {} → {} colors, {} → {} conflicts",
                initial_colors,
                best_solution.chromatic_number,
                initial_conflicts,
                best_solution.conflicts
            );

            // Try one last desperate attempt with color increase
            if self.config.allow_color_increase {
                best_solution = self.desperate_repair(graph, best_solution)?;
            }
        }

        // Telemetry summary would go here

        Ok(best_solution)
    }

    /// Create initial population for repair
    fn create_repair_population(
        &self,
        graph: &Graph,
        solution: &ColoringSolution,
    ) -> Result<Vec<ColoringSolution>> {
        let mut population = Vec::with_capacity(self.config.population_size);
        let mut rng = thread_rng();

        // Add original solution
        population.push(solution.clone());

        // Add variations
        while population.len() < self.config.population_size {
            let mut variant = solution.clone();

            // Find conflicting vertices
            let conflicting_vertices = self.find_conflicting_vertices(graph, &variant);

            if !conflicting_vertices.is_empty() {
                // Recolor conflicting vertices
                for &v in conflicting_vertices.iter().take(5) {
                    let available_colors = self.get_available_colors(graph, &variant, v);
                    if !available_colors.is_empty() {
                        variant.colors[v] = *available_colors.choose(&mut rng).unwrap();
                    } else if self.config.allow_color_increase {
                        // Use a new color if needed
                        let new_color = variant.chromatic_number + 1;
                        if new_color <= solution.chromatic_number + self.config.max_color_increase {
                            variant.colors[v] = new_color;
                            variant.chromatic_number =
                                variant.colors.iter().max().cloned().unwrap_or(0);
                        }
                    }
                }
            }

            // Recompute conflicts
            variant.conflicts = self.count_conflicts(graph, &variant.colors);

            population.push(variant);
        }

        Ok(population)
    }

    /// Evolve population for conflict repair
    fn evolve_repair_population(
        &self,
        graph: &Graph,
        mut population: Vec<ColoringSolution>,
    ) -> Result<Vec<ColoringSolution>> {
        let mut rng = thread_rng();
        let elite_size = self.config.population_size / 5;
        let mut new_population = Vec::with_capacity(self.config.population_size);

        // Keep elite
        for i in 0..elite_size {
            new_population.push(population[i].clone());
        }

        // Generate offspring
        while new_population.len() < self.config.population_size {
            // Tournament selection
            let parent1 = self.tournament_select(&population, 3, &mut rng);
            let parent2 = self.tournament_select(&population, 3, &mut rng);

            // Crossover
            let mut offspring = self.conflict_aware_crossover(graph, parent1, parent2)?;

            // Mutation
            if rng.gen::<f64>() < self.config.mutation_rate {
                offspring = self.conflict_focused_mutation(graph, offspring, &mut rng)?;
            }

            // Local search
            if self.config.use_local_search {
                offspring = self.local_conflict_reduction(graph, offspring)?;
            }

            new_population.push(offspring);
        }

        Ok(new_population)
    }

    /// Conflict-aware crossover
    fn conflict_aware_crossover(
        &self,
        graph: &Graph,
        parent1: &ColoringSolution,
        parent2: &ColoringSolution,
    ) -> Result<ColoringSolution> {
        let mut offspring = parent1.clone();
        let mut rng = thread_rng();

        // Identify conflict regions in both parents
        let conflicts1 = self.find_conflicting_vertices(graph, parent1);
        let conflicts2 = self.find_conflicting_vertices(graph, parent2);

        // For non-conflicting vertices, randomly inherit from either parent
        for v in 0..graph.num_vertices {
            if !conflicts1.contains(&v) && !conflicts2.contains(&v) {
                if rng.gen_bool(0.5) {
                    offspring.colors[v] = parent2.colors[v];
                }
            }
        }

        // For conflicting vertices, choose the better coloring
        for &v in conflicts1.iter() {
            if !conflicts2.contains(&v) {
                // Parent2 has no conflict at this vertex
                offspring.colors[v] = parent2.colors[v];
            }
        }

        offspring.chromatic_number = offspring.colors.iter().max().cloned().unwrap_or(0);
        offspring.conflicts = self.count_conflicts(graph, &offspring.colors);

        Ok(offspring)
    }

    /// Mutation focused on conflict reduction
    fn conflict_focused_mutation(
        &self,
        graph: &Graph,
        mut solution: ColoringSolution,
        rng: &mut ThreadRng,
    ) -> Result<ColoringSolution> {
        let conflicting = self.find_conflicting_vertices(graph, &solution);

        if conflicting.is_empty() {
            return Ok(solution);
        }

        // Mutate a random conflicting vertex
        let v = *conflicting.choose(rng).unwrap();
        let available = self.get_available_colors(graph, &solution, v);

        if !available.is_empty() {
            solution.colors[v] = *available.choose(rng).unwrap();
        } else if self.config.allow_color_increase {
            let new_color = solution.chromatic_number + 1;
            if new_color <= solution.chromatic_number + self.config.max_color_increase {
                solution.colors[v] = new_color;
            }
        }

        solution.chromatic_number = solution.colors.iter().max().cloned().unwrap_or(0);
        solution.conflicts = self.count_conflicts(graph, &solution.colors);

        Ok(solution)
    }

    /// Local search for conflict reduction
    fn local_conflict_reduction(
        &self,
        graph: &Graph,
        mut solution: ColoringSolution,
    ) -> Result<ColoringSolution> {
        let mut improved = true;
        let mut iterations = 0;

        while improved && iterations < 10 {
            improved = false;
            iterations += 1;

            let conflicting = self.find_conflicting_vertices(graph, &solution);

            for &v in conflicting.iter() {
                let available = self.get_available_colors(graph, &solution, v);

                if !available.is_empty() {
                    let old_color = solution.colors[v];

                    // Try each available color
                    for &color in available.iter() {
                        solution.colors[v] = color;
                        let new_conflicts = self.count_conflicts(graph, &solution.colors);

                        if new_conflicts < solution.conflicts {
                            solution.conflicts = new_conflicts;
                            improved = true;
                            break;
                        }
                    }

                    if !improved {
                        solution.colors[v] = old_color;
                    }
                }
            }
        }

        solution.chromatic_number = solution.colors.iter().max().cloned().unwrap_or(0);

        Ok(solution)
    }

    /// Apply Kempe chain operations
    fn apply_kempe_chains(
        &self,
        graph: &Graph,
        solution: &ColoringSolution,
    ) -> Result<Option<ColoringSolution>> {
        // Simplified Kempe chain implementation
        // Would need full implementation for production
        Ok(None)
    }

    /// Desperate repair with color increase
    fn desperate_repair(
        &self,
        graph: &Graph,
        mut solution: ColoringSolution,
    ) -> Result<ColoringSolution> {
        log::info!("[REPAIR] Attempting desperate repair with color increase");

        let conflicting = self.find_conflicting_vertices(graph, &solution);
        let mut new_color = solution.chromatic_number + 1;

        for &v in conflicting.iter() {
            if new_color <= solution.chromatic_number + self.config.max_color_increase {
                solution.colors[v] = new_color;
                new_color += 1;
            }
        }

        solution.chromatic_number = solution.colors.iter().max().cloned().unwrap_or(0);
        solution.conflicts = self.count_conflicts(graph, &solution.colors);

        log::info!(
            "[REPAIR] Desperate repair result: {} colors, {} conflicts",
            solution.chromatic_number,
            solution.conflicts
        );

        Ok(solution)
    }

    /// Helper: Find conflicting vertices
    fn find_conflicting_vertices(&self, graph: &Graph, solution: &ColoringSolution) -> Vec<usize> {
        let mut conflicting = Vec::new();

        for v in 0..graph.num_vertices {
            let color_v = solution.colors[v];
            for &neighbor in &graph.adjacency[v] {
                if solution.colors[neighbor] == color_v {
                    conflicting.push(v);
                    break;
                }
            }
        }

        conflicting
    }

    /// Helper: Get available colors for a vertex
    fn get_available_colors(
        &self,
        graph: &Graph,
        solution: &ColoringSolution,
        vertex: usize,
    ) -> Vec<usize> {
        let mut used_colors = std::collections::HashSet::new();

        for &neighbor in &graph.adjacency[vertex] {
            used_colors.insert(solution.colors[neighbor]);
        }

        (1..=solution.chromatic_number)
            .filter(|&c| !used_colors.contains(&c))
            .collect()
    }

    /// Helper: Count conflicts
    fn count_conflicts(&self, graph: &Graph, colors: &[usize]) -> usize {
        let mut conflicts = 0;

        for v in 0..graph.num_vertices {
            for &neighbor in &graph.adjacency[v] {
                if neighbor > v && colors[v] == colors[neighbor] {
                    conflicts += 1;
                }
            }
        }

        conflicts
    }

    /// Helper: Evaluate repair fitness
    fn evaluate_repair_fitness(&self, graph: &Graph, solution: &mut ColoringSolution) {
        solution.conflicts = self.count_conflicts(graph, &solution.colors);
        solution.chromatic_number = solution.colors.iter().max().cloned().unwrap_or(0);
    }

    /// Helper: Tournament selection
    fn tournament_select<'a>(
        &self,
        population: &'a [ColoringSolution],
        tournament_size: usize,
        rng: &mut ThreadRng,
    ) -> &'a ColoringSolution {
        let mut best = &population[rng.gen_range(0..population.len())];

        for _ in 1..tournament_size {
            let candidate = &population[rng.gen_range(0..population.len())];
            if candidate.conflicts < best.conflicts
                || (candidate.conflicts == best.conflicts
                    && candidate.chromatic_number < best.chromatic_number)
            {
                best = candidate;
            }
        }

        best
    }
}

/// Integration function to repair solutions from phases with conflicts
pub fn repair_phase_output(
    graph: &Graph,
    solution: ColoringSolution,
    phase_name: &str,
) -> Result<ColoringSolution> {
    if solution.conflicts == 0 {
        return Ok(solution);
    }

    log::info!(
        "[{}] Solution has {} conflicts with {} colors, attempting repair",
        phase_name,
        solution.conflicts,
        solution.chromatic_number
    );

    let config = ConflictRepairConfig {
        max_iterations: 500,
        population_size: 30,
        mutation_rate: 0.4,
        allow_color_increase: true,
        max_color_increase: 3, // Allow slight increase if needed
        use_kempe_chains: true,
        use_local_search: true,
    };

    let engine = ConflictRepairEngine::new(config);
    engine.repair(graph, solution)
}

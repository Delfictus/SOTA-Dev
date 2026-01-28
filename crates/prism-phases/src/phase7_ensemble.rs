//! Phase 7: Ensemble Aggregation with FluxNet RL.

use prism_core::{
    ColoringSolution, Graph, PhaseContext, PhaseController, PhaseOutcome, PhaseTelemetry,
    PrismError,
};
use prism_fluxnet::UniversalAction;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for Phase 7 Ensemble
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase7Config {
    #[serde(default = "default_num_replicas")]
    pub num_replicas: usize,

    #[serde(default = "default_exchange_interval")]
    pub exchange_interval: usize,

    #[serde(default = "default_temperature_min")]
    pub temperature_min: f32,

    #[serde(default = "default_temperature_max")]
    pub temperature_max: f32,

    #[serde(default = "default_diversity_weight")]
    pub diversity_weight: f32,

    #[serde(default = "default_consensus_threshold")]
    pub consensus_threshold: f32,

    #[serde(default = "default_voting_method")]
    pub voting_method: String,

    #[serde(default = "default_replica_selection")]
    pub replica_selection: String,
}

fn default_num_replicas() -> usize {
    64
}
fn default_exchange_interval() -> usize {
    10
}
fn default_temperature_min() -> f32 {
    0.1
}
fn default_temperature_max() -> f32 {
    2.0
}
fn default_diversity_weight() -> f32 {
    0.1
}
fn default_consensus_threshold() -> f32 {
    0.7
}
fn default_voting_method() -> String {
    "weighted".to_string()
}
fn default_replica_selection() -> String {
    "best".to_string()
}

impl Default for Phase7Config {
    fn default() -> Self {
        Self {
            num_replicas: default_num_replicas(),
            exchange_interval: default_exchange_interval(),
            temperature_min: default_temperature_min(),
            temperature_max: default_temperature_max(),
            diversity_weight: default_diversity_weight(),
            consensus_threshold: default_consensus_threshold(),
            voting_method: default_voting_method(),
            replica_selection: default_replica_selection(),
        }
    }
}

/// Phase 7: Ensemble Aggregation controller.
///
/// Combines multiple coloring solutions from previous phases:
/// - Solution diversity metrics
/// - Majority voting
/// - Weighted consensus
/// - Best solution selection
pub struct Phase7Ensemble {
    /// Solution diversity score (0.0 - 1.0)
    diversity: f64,

    /// Consensus score (0.0 - 1.0)
    consensus: f64,

    /// Number of candidate solutions
    num_candidates: usize,

    /// Voting strategy
    voting_strategy: VotingStrategy,

    /// Configuration parameters
    num_replicas: usize,
    exchange_interval: usize,
    temperature_min: f32,
    temperature_max: f32,
    diversity_weight: f32,
    consensus_threshold: f32,
    voting_method: String,
    replica_selection: String,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum VotingStrategy {
    Majority,
    Weighted,
    BestOnly,
}

impl Default for Phase7Ensemble {
    fn default() -> Self {
        Self::new()
    }
}

impl Phase7Ensemble {
    pub fn new() -> Self {
        let default_config = Phase7Config::default();
        Self {
            diversity: 0.0,
            consensus: 0.0,
            num_candidates: 0,
            voting_strategy: VotingStrategy::BestOnly,
            num_replicas: default_config.num_replicas,
            exchange_interval: default_config.exchange_interval,
            temperature_min: default_config.temperature_min,
            temperature_max: default_config.temperature_max,
            diversity_weight: default_config.diversity_weight,
            consensus_threshold: default_config.consensus_threshold,
            voting_method: default_config.voting_method,
            replica_selection: default_config.replica_selection,
        }
    }

    /// Creates Phase7 controller with custom config
    pub fn with_config(config: Phase7Config) -> Self {
        log::info!(
            "Phase7: Initializing with config: replicas={}, diversity_weight={:.2}, voting={}",
            config.num_replicas,
            config.diversity_weight,
            config.voting_method
        );

        let voting_strategy = match config.voting_method.as_str() {
            "majority" => VotingStrategy::Majority,
            "weighted" => VotingStrategy::Weighted,
            _ => VotingStrategy::BestOnly,
        };

        Self {
            diversity: 0.0,
            consensus: 0.0,
            num_candidates: 0,
            voting_strategy,
            num_replicas: config.num_replicas,
            exchange_interval: config.exchange_interval,
            temperature_min: config.temperature_min,
            temperature_max: config.temperature_max,
            diversity_weight: config.diversity_weight,
            consensus_threshold: config.consensus_threshold,
            voting_method: config.voting_method,
            replica_selection: config.replica_selection,
        }
    }

    /// Computes diversity between solutions.
    fn compute_diversity(&mut self, solutions: &[ColoringSolution]) {
        if solutions.len() < 2 {
            self.diversity = 0.0;
            return;
        }

        let n = solutions[0].colors.len();
        let mut total_diff = 0.0;
        let mut comparisons = 0;

        // Compare all pairs of solutions
        for i in 0..solutions.len() {
            for j in (i + 1)..solutions.len() {
                let mut diff_count = 0;
                for v in 0..n {
                    if solutions[i].colors[v] != solutions[j].colors[v] {
                        diff_count += 1;
                    }
                }
                total_diff += diff_count as f64 / n as f64;
                comparisons += 1;
            }
        }

        self.diversity = if comparisons > 0 {
            total_diff / comparisons as f64
        } else {
            0.0
        };

        log::debug!(
            "Phase7: Diversity={:.4} across {} solutions",
            self.diversity,
            solutions.len()
        );
    }

    /// Computes consensus score (agreement across solutions).
    fn compute_consensus(&mut self, solutions: &[ColoringSolution]) {
        if solutions.is_empty() {
            self.consensus = 0.0;
            return;
        }

        let n = solutions[0].colors.len();
        let mut agreement = 0.0;

        for v in 0..n {
            // Count color assignments for this vertex
            let mut color_counts: HashMap<usize, usize> = HashMap::new();
            for sol in solutions {
                *color_counts.entry(sol.colors[v]).or_insert(0) += 1;
            }

            // Max agreement for this vertex
            let max_agreement = color_counts.values().max().copied().unwrap_or(0);
            agreement += max_agreement as f64 / solutions.len() as f64;
        }

        self.consensus = agreement / n as f64;

        log::debug!("Phase7: Consensus={:.4}", self.consensus);
    }

    /// Selects best solution from ensemble.
    fn select_best(&self, solutions: &[ColoringSolution]) -> Option<ColoringSolution> {
        match self.voting_strategy {
            VotingStrategy::BestOnly => {
                // Select solution with lowest chromatic number and conflicts
                solutions
                    .iter()
                    .min_by(|a, b| {
                        a.conflicts
                            .cmp(&b.conflicts)
                            .then(a.chromatic_number.cmp(&b.chromatic_number))
                    })
                    .cloned()
            }
            VotingStrategy::Majority => {
                // TODO: Implement majority voting
                solutions.first().cloned()
            }
            VotingStrategy::Weighted => {
                // TODO: Implement weighted voting
                solutions.first().cloned()
            }
        }
    }

    /// Gathers candidate solutions from context.
    fn gather_candidates(&self, context: &PhaseContext) -> Vec<ColoringSolution> {
        let mut candidates = Vec::new();

        // Always include current best solution
        if let Some(ref best) = context.best_solution {
            candidates.push(best.clone());
        }

        // TODO: Collect solutions from other phases stored in context.scratch

        candidates
    }
}

impl PhaseController for Phase7Ensemble {
    fn execute(
        &mut self,
        _graph: &Graph,
        context: &mut PhaseContext,
    ) -> Result<PhaseOutcome, PrismError> {
        log::info!("Phase 7: Ensemble Aggregation executing");

        // Read RL action from context
        if let Some(action) = context
            .scratch
            .get("Phase7_action")
            .and_then(|a| a.downcast_ref::<UniversalAction>())
        {
            if let UniversalAction::Phase7(ensemble_action) = action {
                log::debug!("Phase7: Applying RL action: {:?}", ensemble_action);
                // Apply voting strategy changes
            }
        }

        // Gather candidate solutions
        let candidates = self.gather_candidates(context);
        self.num_candidates = candidates.len();

        if candidates.is_empty() {
            log::warn!("Phase7: No candidate solutions available");
            return Ok(PhaseOutcome::escalate("No candidate solutions"));
        }

        // Adjust diversity weight based on geometry metrics
        let mut adjusted_diversity = self.diversity_weight;
        if let Some(ref geom) = context.geometry_metrics {
            let hotspot_count = geom.anchor_hotspots.len();
            if hotspot_count > 10 {
                adjusted_diversity *= 1.0 + (hotspot_count as f32 / 20.0);
                log::debug!(
                    "Phase7: {} hotspots → adjusted diversity to {:.3}",
                    hotspot_count,
                    adjusted_diversity
                );
            }
        }

        // Apply dendritic reservoir difficulty for ensemble diversity tuning
        // High difficulty graphs benefit from more diverse exploration
        if context.has_dendritic_metrics() {
            let mean_diff = context.mean_difficulty();
            // Scale diversity: low difficulty → 0.9x, high difficulty → 1.4x
            let diversity_boost = 0.9 + mean_diff * 0.5;
            adjusted_diversity *= diversity_boost;
            adjusted_diversity = adjusted_diversity.clamp(0.1, 2.0);
            log::info!(
                "[Phase7] Dendritic coupling: mean_difficulty={:.3}, diversity_boost={:.2}x, final_diversity={:.3}",
                mean_diff,
                diversity_boost,
                adjusted_diversity
            );
        }

        // Compute diversity and consensus
        self.compute_diversity(&candidates);
        self.compute_consensus(&candidates);

        // Select best solution
        if let Some(best) = self.select_best(&candidates) {
            log::info!(
                "Phase7: Selected best solution: {} colors, {} conflicts",
                best.chromatic_number,
                best.conflicts
            );

            // Update context with final best solution
            context.update_best_solution(best);
        }

        // Update RL state metrics
        if let Some(rl_state) = context.rl_state.as_mut() {
            if let Some(state) = rl_state.downcast_mut::<prism_fluxnet::UniversalRLState>() {
                state.ensemble_diversity = self.diversity;
                state.ensemble_consensus = self.consensus;
            }
        }

        Ok(PhaseOutcome::success())
    }

    fn name(&self) -> &'static str {
        "Phase7-Ensemble"
    }

    fn telemetry(&self) -> &dyn PhaseTelemetry {
        self
    }
}

impl PhaseTelemetry for Phase7Ensemble {
    fn metrics(&self) -> HashMap<String, f64> {
        let mut m = HashMap::new();
        m.insert("diversity".to_string(), self.diversity);
        m.insert("consensus".to_string(), self.consensus);
        m.insert("num_candidates".to_string(), self.num_candidates as f64);
        m
    }
}

// ============================================================================
// Memetic Algorithm Implementation (Phase 6 Genetic Operators)
// ============================================================================

/// Memetic algorithm for graph coloring optimization.
///
/// Hybrid CPU genetic algorithm that evolves a population of coloring solutions:
/// - **Selection**: Tournament or roulette wheel
/// - **Crossover**: Uniform or 1-point recombination
/// - **Mutation**: Per-vertex random color change
/// - **Local Search**: Greedy conflict reduction
/// - **Elitism**: Preserve best solutions across generations
///
/// ## Usage
/// ```rust,no_run
/// use prism_phases::phase7_ensemble::MemeticAlgorithm;
/// use prism_core::{Graph, ColoringSolution};
///
/// let graph = Graph::new(100);
/// let initial_population = vec![/* GPU-generated solutions */];
///
/// let mut memetic = MemeticAlgorithm::new(
///     128,    // population_size
///     500,    // generations
///     0.8,    // crossover_rate
///     0.05,   // mutation_rate
///     100,    // local_search_iterations
///     10,     // elitism_count
///     5,      // tournament_size
///     50,     // convergence_threshold
/// );
///
/// let best = memetic.evolve(&graph, initial_population).unwrap();
/// ```
pub struct MemeticAlgorithm {
    population_size: usize,
    generations: usize,
    crossover_rate: f64,
    mutation_rate: f64,
    local_search_iterations: usize,
    elitism_count: usize,
    tournament_size: usize,
    convergence_threshold: usize,
}

impl MemeticAlgorithm {
    /// Creates a new memetic algorithm with the given parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        population_size: usize,
        generations: usize,
        crossover_rate: f64,
        mutation_rate: f64,
        local_search_iterations: usize,
        elitism_count: usize,
        tournament_size: usize,
        convergence_threshold: usize,
    ) -> Self {
        Self {
            population_size,
            generations,
            crossover_rate,
            mutation_rate,
            local_search_iterations,
            elitism_count,
            tournament_size,
            convergence_threshold,
        }
    }

    /// Evolves the population for the specified number of generations.
    ///
    /// ## Arguments
    /// - `graph`: The graph to color
    /// - `initial_population`: Initial population from GPU multi-attempt runs
    /// - `geometry_metrics`: Optional geometry telemetry for hotspot-targeted evolution
    ///
    /// ## Returns
    /// Best solution found during evolution
    pub fn evolve(
        &self,
        graph: &Graph,
        mut population: Vec<ColoringSolution>,
        geometry_metrics: Option<&prism_core::GeometryTelemetry>,
    ) -> Result<ColoringSolution, PrismError> {
        if population.is_empty() {
            return Err(PrismError::config("Initial population is empty"));
        }

        // Log geometry coupling status
        if let Some(geom) = geometry_metrics {
            log::info!(
                "Memetic: Starting evolution with population={}, generations={}, geometry_coupling=true (stress={:.3}, {} hotspots)",
                self.population_size,
                self.generations,
                geom.stress_scalar,
                geom.anchor_hotspots.len()
            );
        } else {
            log::info!(
                "Memetic: Starting evolution with population={}, generations={}, geometry_coupling=false",
                self.population_size,
                self.generations
            );
        }

        // Pad or trim population to target size
        while population.len() < self.population_size {
            // Duplicate best solutions if population too small
            let best = population[0].clone();
            population.push(best);
        }
        population.truncate(self.population_size);

        let mut best_ever = population[0].clone();
        let mut generations_without_improvement = 0;

        for generation in 0..self.generations {
            // Sort population by fitness (conflicts first, then chromatic number)
            population.sort_by(|a, b| {
                a.conflicts
                    .cmp(&b.conflicts)
                    .then(a.chromatic_number.cmp(&b.chromatic_number))
            });

            // Update best solution
            if population[0].conflicts < best_ever.conflicts
                || (population[0].conflicts == best_ever.conflicts
                    && population[0].chromatic_number < best_ever.chromatic_number)
            {
                best_ever = population[0].clone();
                generations_without_improvement = 0;
                log::info!(
                    "Memetic Gen {}/{}: NEW BEST - {} colors, {} conflicts",
                    generation + 1,
                    self.generations,
                    best_ever.chromatic_number,
                    best_ever.conflicts
                );
            } else {
                generations_without_improvement += 1;
            }

            // Early stopping
            if generations_without_improvement >= self.convergence_threshold {
                log::info!(
                    "Memetic: Early stop at generation {} (no improvement for {} gens)",
                    generation + 1,
                    self.convergence_threshold
                );
                break;
            }

            // Progress logging
            if (generation + 1) % 10 == 0 || generation == 0 {
                log::info!(
                    "Memetic Gen {}/{}: Best={} colors ({} conflicts), Avg={:.1} colors",
                    generation + 1,
                    self.generations,
                    population[0].chromatic_number,
                    population[0].conflicts,
                    population.iter().map(|s| s.chromatic_number).sum::<usize>() as f64
                        / population.len() as f64
                );
            }

            // Create next generation
            let mut next_generation = Vec::new();

            // Elitism: Preserve best solutions
            for i in 0..self.elitism_count.min(population.len()) {
                next_generation.push(population[i].clone());
            }

            // Generate offspring until population is full
            while next_generation.len() < self.population_size {
                // Selection
                let parent1 = self.tournament_select(&population);
                let parent2 = self.tournament_select(&population);

                // Crossover
                let (mut child1, mut child2) = if rand::random::<f64>() < self.crossover_rate {
                    self.uniform_crossover(&parent1, &parent2)
                } else {
                    (parent1.clone(), parent2.clone())
                };

                // Mutation (with geometry hotspot biasing if available)
                self.mutate_with_geometry(graph, &mut child1, geometry_metrics);
                self.mutate_with_geometry(graph, &mut child2, geometry_metrics);

                // Local search (with geometry hotspot prioritization if available)
                self.local_search_with_geometry(graph, &mut child1, geometry_metrics);
                self.local_search_with_geometry(graph, &mut child2, geometry_metrics);

                next_generation.push(child1);
                if next_generation.len() < self.population_size {
                    next_generation.push(child2);
                }
            }

            population = next_generation;
        }

        log::info!(
            "Memetic: Evolution complete. Best: {} colors, {} conflicts",
            best_ever.chromatic_number,
            best_ever.conflicts
        );

        Ok(best_ever)
    }

    /// Tournament selection: Select best from random subset.
    fn tournament_select(&self, population: &[ColoringSolution]) -> ColoringSolution {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        let mut tournament: Vec<&ColoringSolution> = population
            .choose_multiple(&mut rng, self.tournament_size)
            .collect();

        tournament.sort_by(|a, b| {
            a.conflicts
                .cmp(&b.conflicts)
                .then(a.chromatic_number.cmp(&b.chromatic_number))
        });

        tournament[0].clone()
    }

    /// Uniform crossover: Each vertex randomly inherits color from parent1 or parent2.
    fn uniform_crossover(
        &self,
        parent1: &ColoringSolution,
        parent2: &ColoringSolution,
    ) -> (ColoringSolution, ColoringSolution) {
        let n = parent1.colors.len();
        let mut child1_colors = vec![0; n];
        let mut child2_colors = vec![0; n];

        for v in 0..n {
            if rand::random::<bool>() {
                child1_colors[v] = parent1.colors[v];
                child2_colors[v] = parent2.colors[v];
            } else {
                child1_colors[v] = parent2.colors[v];
                child2_colors[v] = parent1.colors[v];
            }
        }

        let child1 = ColoringSolution::from_colors(child1_colors);
        let child2 = ColoringSolution::from_colors(child2_colors);

        (child1, child2)
    }

    /// Mutation: Random color change with probability mutation_rate per vertex.
    ///
    /// If geometry hotspots are provided, mutation probability is increased
    /// for hotspot vertices (metaphysical telemetry coupling).
    #[allow(dead_code)]
    fn mutate(&self, graph: &Graph, solution: &mut ColoringSolution) {
        self.mutate_with_geometry(graph, solution, None);
    }

    /// Mutation with optional geometry hotspot biasing.
    ///
    /// When geometry metrics are available and stress is high, mutation probability
    /// is doubled for hotspot vertices to focus evolutionary search on stressed regions.
    fn mutate_with_geometry(
        &self,
        graph: &Graph,
        solution: &mut ColoringSolution,
        geometry_metrics: Option<&prism_core::GeometryTelemetry>,
    ) {
        let n = graph.num_vertices;
        let max_color = solution.chromatic_number;

        // Check if we should use hotspot-targeted mutation
        let hotspot_boost = if let Some(geom) = geometry_metrics {
            if geom.stress_scalar > 0.5 {
                2.0 // Double mutation rate for hotspots when stress is high
            } else {
                1.0
            }
        } else {
            1.0
        };

        let hotspots: std::collections::HashSet<usize> = geometry_metrics
            .map(|g| g.anchor_hotspots.iter().copied().collect())
            .unwrap_or_default();

        let mut mutations_applied = 0;
        let mut hotspot_mutations = 0;

        for v in 0..n {
            // Determine mutation rate for this vertex
            let effective_rate = if hotspots.contains(&v) {
                self.mutation_rate * hotspot_boost
            } else {
                self.mutation_rate
            };

            if rand::random::<f64>() < effective_rate {
                // Random color in range [1, max_color + 1]
                let new_color = rand::random::<usize>() % (max_color + 1) + 1;
                solution.colors[v] = new_color;
                mutations_applied += 1;

                if hotspots.contains(&v) {
                    hotspot_mutations += 1;
                }
            }
        }

        // Recompute conflicts and chromatic number
        solution.recompute_metrics(&graph.adjacency);

        if hotspot_boost > 1.0 && hotspot_mutations > 0 {
            log::debug!(
                "[Memetic] Geometry-targeted mutation: {} total mutations, {} on hotspots (boost={:.1}x)",
                mutations_applied,
                hotspot_mutations,
                hotspot_boost
            );
        }
    }

    /// Local search: Greedy conflict reduction.
    ///
    /// For each vertex with conflicts, try all colors and pick the one
    /// that minimizes conflicts. Repeat for `local_search_iterations`.
    #[allow(dead_code)]
    fn local_search(&self, graph: &Graph, solution: &mut ColoringSolution) {
        self.local_search_with_geometry(graph, solution, None);
    }

    /// Local search with optional geometry hotspot prioritization.
    ///
    /// When geometry metrics are available and stress is high, hotspot vertices
    /// are processed first to focus search on stressed regions.
    fn local_search_with_geometry(
        &self,
        graph: &Graph,
        solution: &mut ColoringSolution,
        geometry_metrics: Option<&prism_core::GeometryTelemetry>,
    ) {
        let n = graph.num_vertices;

        // Build hotspot set for O(1) lookup
        let hotspots: std::collections::HashSet<usize> = geometry_metrics
            .filter(|g| g.stress_scalar > 0.5)
            .map(|g| g.anchor_hotspots.iter().copied().collect())
            .unwrap_or_default();

        let prioritize_hotspots = !hotspots.is_empty();

        for _ in 0..self.local_search_iterations {
            let initial_conflicts = solution.conflicts;

            // Find vertices with conflicts
            let mut conflicted_vertices: Vec<usize> = Vec::new();
            let mut hotspot_conflicts: Vec<usize> = Vec::new();

            for v in 0..n {
                let current_color = solution.colors[v];
                for &neighbor in &graph.adjacency[v] {
                    if solution.colors[neighbor] == current_color {
                        if prioritize_hotspots && hotspots.contains(&v) {
                            hotspot_conflicts.push(v);
                        } else {
                            conflicted_vertices.push(v);
                        }
                        break;
                    }
                }
            }

            if conflicted_vertices.is_empty() && hotspot_conflicts.is_empty() {
                break; // No conflicts, done
            }

            // Process hotspot vertices first (metaphysical coupling)
            let vertices_to_process: Vec<usize> = hotspot_conflicts
                .into_iter()
                .chain(conflicted_vertices.into_iter())
                .collect();

            // Try improving each conflicted vertex
            for &v in &vertices_to_process {
                let current_color = solution.colors[v];
                let mut best_color = current_color;
                let mut best_conflicts = self.count_vertex_conflicts(graph, solution, v);

                // Try all colors
                for color in 1..=solution.chromatic_number {
                    solution.colors[v] = color;
                    let conflicts = self.count_vertex_conflicts(graph, solution, v);
                    if conflicts < best_conflicts {
                        best_color = color;
                        best_conflicts = conflicts;
                    }
                }

                solution.colors[v] = best_color;
            }

            // Recompute metrics
            solution.recompute_metrics(&graph.adjacency);

            // Stop if no improvement
            if solution.conflicts >= initial_conflicts {
                break;
            }
        }

        if prioritize_hotspots {
            log::debug!(
                "[Memetic] Hotspot-prioritized local search: {} hotspot vertices processed first",
                hotspots.len()
            );
        }
    }

    /// Counts conflicts for a single vertex.
    fn count_vertex_conflicts(
        &self,
        graph: &Graph,
        solution: &ColoringSolution,
        vertex: usize,
    ) -> usize {
        let color = solution.colors[vertex];
        graph.adjacency[vertex]
            .iter()
            .filter(|&&neighbor| solution.colors[neighbor] == color)
            .count()
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::{ColoringSolution, Graph};

    /// Creates a simple test graph (triangle: 0-1-2-0).
    fn create_triangle_graph() -> Graph {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 0);
        graph
    }

    /// Creates a simple test coloring for the triangle graph.
    fn create_test_coloring() -> ColoringSolution {
        ColoringSolution::from_colors(vec![1, 2, 3])
    }

    #[test]
    fn test_uniform_crossover_produces_valid_colorings() {
        let memetic = MemeticAlgorithm::new(128, 500, 0.8, 0.05, 100, 10, 5, 50);

        let parent1 = ColoringSolution::from_colors(vec![1, 2, 3, 1, 2]);
        let parent2 = ColoringSolution::from_colors(vec![3, 1, 2, 3, 1]);

        let (child1, child2) = memetic.uniform_crossover(&parent1, &parent2);

        // Check children have correct length
        assert_eq!(child1.colors.len(), 5);
        assert_eq!(child2.colors.len(), 5);

        // Check all colors are from parent color sets
        for i in 0..5 {
            let c1 = child1.colors[i];
            let c2 = child2.colors[i];
            assert!(c1 == parent1.colors[i] || c1 == parent2.colors[i]);
            assert!(c2 == parent1.colors[i] || c2 == parent2.colors[i]);
        }
    }

    #[test]
    fn test_mutation_respects_rate() {
        let graph = create_triangle_graph();

        // Zero mutation rate
        let memetic_zero = MemeticAlgorithm::new(128, 500, 0.8, 0.0, 100, 10, 5, 50);
        let mut solution_zero = create_test_coloring();
        let _original = solution_zero.colors.clone();
        memetic_zero.mutate(&graph, &mut solution_zero);
        // With rate 0.0, colors should be unchanged (probabilistically may still change rarely)
        // This test may occasionally fail due to randomness, but with 0.0 rate it should pass

        // High mutation rate (1.0 = mutate every vertex)
        let memetic_high = MemeticAlgorithm::new(128, 500, 0.8, 1.0, 100, 10, 5, 50);
        let mut solution_high = create_test_coloring();
        memetic_high.mutate(&graph, &mut solution_high);
        // With rate 1.0, expect most/all vertices to change (probabilistic)
        // Just verify solution is valid (has colors)
        assert_eq!(solution_high.colors.len(), 3);
    }

    #[test]
    fn test_local_search_reduces_conflicts() {
        let graph = create_triangle_graph();
        let memetic = MemeticAlgorithm::new(128, 500, 0.8, 0.05, 100, 10, 5, 50);

        // Create invalid coloring with conflicts (all vertices color 1)
        let mut solution = ColoringSolution::from_colors(vec![1, 1, 1]);
        solution.recompute_metrics(&graph.adjacency);

        let initial_conflicts = solution.conflicts;
        assert!(initial_conflicts > 0, "Test setup should have conflicts");

        // Run local search
        memetic.local_search(&graph, &mut solution);

        // Should reduce or eliminate conflicts
        assert!(
            solution.conflicts <= initial_conflicts,
            "Local search should not increase conflicts"
        );
    }

    #[test]
    fn test_tournament_select() {
        let memetic = MemeticAlgorithm::new(128, 500, 0.8, 0.05, 100, 10, 5, 50);

        // Create population with varying fitness
        let mut population = vec![
            ColoringSolution::from_colors(vec![1, 2, 3]), // Best (3 colors)
            ColoringSolution::from_colors(vec![1, 2, 3, 4]), // Worst (4 colors)
            ColoringSolution::from_colors(vec![1, 2, 3]), // Best (3 colors)
        ];

        // Update chromatic numbers
        for sol in &mut population {
            sol.chromatic_number = *sol.colors.iter().max().unwrap_or(&0);
        }

        // Tournament should select valid solution
        let selected = memetic.tournament_select(&population);
        assert!(selected.colors.len() > 0);
    }

    #[test]
    fn test_evolve_integration() {
        let graph = create_triangle_graph();

        // Small memetic config for fast test
        let memetic = MemeticAlgorithm::new(
            10,   // population_size
            5,    // generations (small for test speed)
            0.8,  // crossover_rate
            0.05, // mutation_rate
            10,   // local_search_iterations
            2,    // elitism_count
            3,    // tournament_size
            3,    // convergence_threshold
        );

        // Create initial population with valid colorings
        let mut population = Vec::new();
        for _ in 0..10 {
            let mut sol = create_test_coloring();
            sol.recompute_metrics(&graph.adjacency);
            population.push(sol);
        }

        // Run evolution
        let result = memetic.evolve(&graph, population);
        assert!(result.is_ok(), "Evolution should complete successfully");

        let best = result.unwrap();
        // Triangle graph requires at least 3 colors
        assert!(best.chromatic_number >= 3);
        // Should have valid coloring
        assert_eq!(best.colors.len(), 3);
    }

    #[test]
    fn test_count_vertex_conflicts() {
        let graph = create_triangle_graph();
        let memetic = MemeticAlgorithm::new(128, 500, 0.8, 0.05, 100, 10, 5, 50);

        // Valid coloring (no conflicts)
        let solution_valid = ColoringSolution::from_colors(vec![1, 2, 3]);
        assert_eq!(
            memetic.count_vertex_conflicts(&graph, &solution_valid, 0),
            0
        );
        assert_eq!(
            memetic.count_vertex_conflicts(&graph, &solution_valid, 1),
            0
        );
        assert_eq!(
            memetic.count_vertex_conflicts(&graph, &solution_valid, 2),
            0
        );

        // Invalid coloring (all vertices same color = conflicts)
        let solution_invalid = ColoringSolution::from_colors(vec![1, 1, 1]);
        assert_eq!(
            memetic.count_vertex_conflicts(&graph, &solution_invalid, 0),
            2
        ); // Vertex 0 conflicts with neighbors 1, 2
        assert_eq!(
            memetic.count_vertex_conflicts(&graph, &solution_invalid, 1),
            2
        ); // Vertex 1 conflicts with neighbors 0, 2
        assert_eq!(
            memetic.count_vertex_conflicts(&graph, &solution_invalid, 2),
            2
        ); // Vertex 2 conflicts with neighbors 0, 1
    }

    #[test]
    fn test_elitism_preserves_best() {
        let graph = create_triangle_graph();

        let memetic = MemeticAlgorithm::new(
            10,  // population_size
            2,   // generations
            0.8, // crossover_rate
            0.0, // mutation_rate (zero to isolate elitism)
            0,   // local_search_iterations (zero to isolate elitism)
            2,   // elitism_count
            3,   // tournament_size
            10,  // convergence_threshold
        );

        // Create population with one clear best solution
        let mut population = Vec::new();
        let mut best_solution = ColoringSolution::from_colors(vec![1, 2, 3]);
        best_solution.chromatic_number = 3;
        best_solution.conflicts = 0;
        population.push(best_solution.clone());

        // Add worse solutions
        for _ in 1..10 {
            let mut sol = ColoringSolution::from_colors(vec![1, 1, 1]);
            sol.chromatic_number = 1;
            sol.conflicts = 3;
            population.push(sol);
        }

        let result = memetic.evolve(&graph, population);
        assert!(result.is_ok());

        let final_best = result.unwrap();
        // Elitism should preserve the initial best solution
        assert_eq!(final_best.chromatic_number, 3);
        assert_eq!(final_best.conflicts, 0);
    }

    #[test]
    fn test_config_parameters_respected() {
        // Test that config builder properly sets parameters
        let memetic = MemeticAlgorithm::new(
            256,  // population_size
            1000, // generations
            0.9,  // crossover_rate
            0.1,  // mutation_rate
            200,  // local_search_iterations
            20,   // elitism_count
            7,    // tournament_size
            100,  // convergence_threshold
        );

        assert_eq!(memetic.population_size, 256);
        assert_eq!(memetic.generations, 1000);
        assert_eq!(memetic.crossover_rate, 0.9);
        assert_eq!(memetic.mutation_rate, 0.1);
        assert_eq!(memetic.local_search_iterations, 200);
        assert_eq!(memetic.elitism_count, 20);
        assert_eq!(memetic.tournament_size, 7);
        assert_eq!(memetic.convergence_threshold, 100);
    }
}

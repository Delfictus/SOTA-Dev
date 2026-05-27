//! Phase 2: Thermodynamic pocket sampling (placeholder)

use super::phase1_beliefs::PocketBeliefOutput;
use crate::graph::ProteinGraph;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketSamplingConfig {
    pub max_iterations: usize,
    pub temperature: f64,
    pub cooling_rate: f64,
    /// Penalty for assigning the same color to neighbors
    pub conflict_penalty: f64,
    /// Weight for belief-driven energy
    pub belief_weight: f64,
}

impl Default for PocketSamplingConfig {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            temperature: 1.0,
            cooling_rate: 0.99,
            conflict_penalty: 1.0,
            belief_weight: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PocketSamplingOutput {
    pub coloring: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct PocketSamplingPhase {
    config: PocketSamplingConfig,
}

impl PocketSamplingPhase {
    pub fn new(config: PocketSamplingConfig) -> Self {
        Self { config }
    }

    pub fn execute(
        &self,
        graph: &ProteinGraph,
        beliefs: &PocketBeliefOutput,
    ) -> PocketSamplingOutput {
        let n = graph.adjacency.len();
        let k = beliefs.beliefs.get(0).map(|b| b.len()).unwrap_or(1);

        // Simple winner-takes-most initialization from beliefs
        let mut coloring: Vec<usize> = (0..n)
            .map(|i| {
                beliefs.beliefs[i]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect();

        // Placeholder annealing loop adjusting assignments lightly
        let mut temp = self.config.temperature;
        for _ in 0..self.config.max_iterations {
            for i in 0..n {
                let current = coloring[i];
                let current_energy = self.energy(i, current, &coloring, graph, beliefs, k);

                // propose a new color biased by top belief
                let mut proposal = beliefs.beliefs[i]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(current);
                if proposal == current {
                    proposal = (current + 1) % k;
                }

                let proposal_energy = self.energy(i, proposal, &coloring, graph, beliefs, k);
                if accept_move(current_energy, proposal_energy, temp, i) {
                    coloring[i] = proposal;
                }
            }
            temp *= self.config.cooling_rate;
        }

        PocketSamplingOutput { coloring }
    }

    fn energy(
        &self,
        idx: usize,
        color: usize,
        coloring: &[usize],
        graph: &ProteinGraph,
        beliefs: &PocketBeliefOutput,
        k: usize,
    ) -> f64 {
        let mut energy = 0.0;
        // belief term: encourage high probability assignments
        let belief = beliefs.beliefs[idx]
            .get(color)
            .cloned()
            .unwrap_or(1.0 / k as f64)
            .max(1e-6);
        energy -= self.config.belief_weight * belief.ln();

        // conflict penalty: neighbors with same color
        for &nbr in &graph.adjacency[idx] {
            if coloring.get(nbr) == Some(&color) {
                energy += self.config.conflict_penalty;
            }
        }
        energy
    }
}

fn accept_move(current: f64, proposal: f64, temperature: f64, seed: usize) -> bool {
    if proposal < current {
        return true;
    }
    let diff = proposal - current;
    let temp = temperature.max(1e-6);
    let acceptance = (-diff / temp).exp();
    pseudo_rand(seed) < acceptance
}

fn pseudo_rand(seed: usize) -> f64 {
    let x = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((x >> 32) as f64) / (u32::MAX as f64)
}

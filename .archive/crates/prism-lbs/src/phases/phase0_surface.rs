//! Phase 0: Surface reservoir dynamics

use crate::graph::ProteinGraph;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceReservoirConfig {
    pub sasa_weight: f64,
    pub hydrophobicity_weight: f64,
    pub iterations: usize,
    pub learning_rate: f64,
    pub decay_rate: f64,
    pub activation_threshold: f64,
}

impl Default for SurfaceReservoirConfig {
    fn default() -> Self {
        Self {
            sasa_weight: 0.4,
            hydrophobicity_weight: 0.6,
            iterations: 50,
            learning_rate: 0.02,
            decay_rate: 0.1,
            activation_threshold: 0.3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SurfaceReservoirOutput {
    pub activation_state: Vec<f64>,
    pub hotspots: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct SurfaceReservoirPhase {
    config: SurfaceReservoirConfig,
}

impl SurfaceReservoirPhase {
    pub fn new(config: SurfaceReservoirConfig) -> Self {
        Self { config }
    }

    pub fn execute(&self, graph: &ProteinGraph) -> SurfaceReservoirOutput {
        let n = graph.adjacency.len();
        let mut state: Vec<f64> = (0..n)
            .map(|i| {
                let h = graph.vertex_features.hydrophobicity[i];
                let sasa = graph.structure_ref.atoms[graph.atom_indices[i]].sasa;
                let h_norm = (h + 4.5) / 9.0;
                let sasa_norm = (sasa / 100.0).min(1.0);
                self.config.hydrophobicity_weight * h_norm + self.config.sasa_weight * sasa_norm
            })
            .collect();

        for _ in 0..self.config.iterations {
            let mut next = state.clone();
            for i in 0..n {
                if graph.adjacency[i].is_empty() {
                    continue;
                }
                let weight_sum: f64 = graph.edge_weights[i].iter().sum();
                if weight_sum == 0.0 {
                    continue;
                }
                let neighbor_sum: f64 = graph.adjacency[i]
                    .iter()
                    .zip(&graph.edge_weights[i])
                    .map(|(&j, &w)| state[j] * w)
                    .sum();
                next[i] = (1.0 - self.config.learning_rate) * state[i]
                    + self.config.learning_rate * (neighbor_sum / weight_sum);
                if next[i] < self.config.activation_threshold {
                    next[i] *= self.config.decay_rate;
                }
            }
            state = next;
        }

        let mean: f64 = state.iter().sum::<f64>() / n as f64;
        let std: f64 = (state.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / n as f64).sqrt();

        let hotspots: Vec<usize> = state
            .iter()
            .enumerate()
            .filter(|(_, &s)| s > mean + std)
            .map(|(i, _)| i)
            .collect();

        SurfaceReservoirOutput {
            activation_state: state,
            hotspots,
        }
    }
}

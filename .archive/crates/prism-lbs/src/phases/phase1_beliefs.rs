//! Phase 1: Pocket belief propagation

use super::phase0_surface::SurfaceReservoirOutput;
use crate::graph::ProteinGraph;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketBeliefConfig {
    pub prior_precision: f64,
    pub learning_rate: f64,
    pub iterations: usize,
    pub max_pockets: usize,
    /// Temperature for softmax normalization (simulated annealing style)
    pub temperature: f64,
    /// Annealing schedule multiplier
    pub temperature_decay: f64,
    pub hydrophobicity_evidence: f64,
    pub depth_evidence: f64,
    pub curvature_evidence: f64,
}

impl Default for PocketBeliefConfig {
    fn default() -> Self {
        Self {
            prior_precision: 3.0,
            learning_rate: 0.1,
            iterations: 100,
            max_pockets: 20,
            temperature: 1.0,
            temperature_decay: 0.98,
            hydrophobicity_evidence: 0.3,
            depth_evidence: 0.4,
            curvature_evidence: 0.3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PocketBeliefOutput {
    pub beliefs: Vec<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub struct PocketBeliefPhase {
    config: PocketBeliefConfig,
}

impl PocketBeliefPhase {
    pub fn new(config: PocketBeliefConfig) -> Self {
        Self { config }
    }

    pub fn execute(
        &self,
        graph: &ProteinGraph,
        reservoir: &SurfaceReservoirOutput,
    ) -> PocketBeliefOutput {
        let n = graph.adjacency.len();
        let k = self.config.max_pockets.max(1);
        let mut beliefs: Vec<Vec<f64>> = vec![vec![1.0 / k as f64; k]; n];
        let mut temperature = self.config.temperature.max(1e-3);

        for (pocket_id, &hotspot) in reservoir.hotspots.iter().take(k).enumerate() {
            beliefs[hotspot][pocket_id] = 0.8;
            softmax(&mut beliefs[hotspot], temperature);
        }

        let evidence: Vec<f64> = (0..n)
            .map(|i| {
                let h = graph.vertex_features.hydrophobicity[i];
                let d = graph.vertex_features.depth[i];
                let c = graph.vertex_features.curvature[i];
                let h_score = ((h + 4.5) / 9.0).max(0.0);
                let d_score = (d / 10.0).min(1.0);
                let c_score = (-c).max(0.0).min(1.0);
                self.config.hydrophobicity_evidence * h_score
                    + self.config.depth_evidence * d_score
                    + self.config.curvature_evidence * c_score
            })
            .collect();

        for _ in 0..self.config.iterations {
            let mut new_beliefs = beliefs.clone();
            for i in 0..n {
                if graph.adjacency[i].is_empty() {
                    continue;
                }
                for pocket in 0..k {
                    let (neighbor_score, weight_sum): (f64, f64) = graph.adjacency[i]
                        .iter()
                        .zip(&graph.edge_weights[i])
                        .fold((0.0, 0.0), |(acc, wsum), (&j, &w)| {
                            (acc + beliefs[j][pocket] * w, wsum + w)
                        });
                    let neighbor_belief = if weight_sum > 0.0 {
                        neighbor_score / weight_sum
                    } else {
                        beliefs[i][pocket]
                    };
                    let prior = self.config.prior_precision * (1.0 / k as f64);
                    let stim = evidence[i] * (reservoir.activation_state[i] + 1e-6);
                    let blended = (1.0 - self.config.learning_rate) * beliefs[i][pocket]
                        + self.config.learning_rate * (prior + neighbor_belief + stim);
                    new_beliefs[i][pocket] = blended.max(1e-6);
                }
                softmax(&mut new_beliefs[i], temperature);
            }
            beliefs = new_beliefs;
            temperature *= self.config.temperature_decay;
            if temperature < 1e-3 {
                temperature = 1e-3;
            }
        }

        PocketBeliefOutput { beliefs }
    }
}

fn softmax(vec: &mut [f64], temperature: f64) {
    if temperature <= 0.0 {
        return;
    }
    let t = temperature;
    let max = vec.iter().cloned().fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let mut sum = 0.0;
    for v in vec.iter_mut() {
        *v = ((*v - max) / t).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in vec.iter_mut() {
            *v /= sum;
        }
    }
}

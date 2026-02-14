//! Phase 4: Cavity analysis via geodesic measures

use crate::graph::ProteinGraph;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CavityAnalysisConfig {
    pub depth_threshold: f64,
    pub tunnel_length_min: f64,
    pub tunnel_width_max: f64,
    /// Number of top central nodes to keep as cavity seeds
    pub top_centrality: usize,
    /// Weight for betweenness contribution in cavity ranking
    pub centrality_weight: f64,
}

impl Default for CavityAnalysisConfig {
    fn default() -> Self {
        Self {
            depth_threshold: 5.0,
            tunnel_length_min: 8.0,
            tunnel_width_max: 4.0,
            top_centrality: 10,
            centrality_weight: 0.5,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Tunnel {
    pub entrance: usize,
    pub exit: usize,
    pub length: f64,
    pub width: f64,
    pub score: f64,
}

#[derive(Debug, Clone)]
pub struct CavityAnalysisOutput {
    pub distance_matrix: Vec<Vec<f64>>,
    pub eccentricity: Vec<f64>,
    pub cavity_centers: Vec<usize>,
    pub tunnels: Vec<Tunnel>,
    pub stress_scores: Vec<f64>,
    pub betweenness: Vec<f64>,
}

pub struct CavityAnalysisPhase {
    config: CavityAnalysisConfig,
}

impl CavityAnalysisPhase {
    pub fn new(config: CavityAnalysisConfig) -> Self {
        Self { config }
    }

    pub fn execute(&self, graph: &ProteinGraph) -> CavityAnalysisOutput {
        let n = graph.adjacency.len();
        let apsp = self.compute_apsp(graph);

        let eccentricity: Vec<f64> = (0..n)
            .map(|i| {
                apsp[i]
                    .iter()
                    .cloned()
                    .filter(|d| *d < f64::INFINITY)
                    .fold(0.0_f64, f64::max)
            })
            .collect();

        let mean_ecc = if n > 0 {
            eccentricity.iter().sum::<f64>() / n as f64
        } else {
            0.0
        };
        let mut cavity_centers: Vec<usize> = eccentricity
            .iter()
            .enumerate()
            .filter(|(_, &e)| e > mean_ecc * 1.5)
            .map(|(i, _)| i)
            .collect();

        let betweenness = self.betweenness_centrality(&apsp);
        // combine eccentricity and betweenness to rank cavity seeds
        let mut ranked: Vec<(usize, f64)> = betweenness
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let ecc = eccentricity.get(i).copied().unwrap_or(0.0);
                (i, c * self.config.centrality_weight + ecc)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (idx, _) in ranked.into_iter().take(self.config.top_centrality) {
            if !cavity_centers.contains(&idx) {
                cavity_centers.push(idx);
            }
        }

        let tunnels = self.detect_tunnels(graph, &apsp);
        let stress_scores = self.compute_stress(n, &apsp);

        CavityAnalysisOutput {
            distance_matrix: apsp,
            eccentricity,
            cavity_centers,
            tunnels,
            stress_scores,
            betweenness,
        }
    }

    fn compute_apsp(&self, graph: &ProteinGraph) -> Vec<Vec<f64>> {
        let n = graph.adjacency.len();
        let mut dist = vec![vec![f64::INFINITY; n]; n];
        for i in 0..n {
            dist[i][i] = 0.0;
            for (k, &j) in graph.adjacency[i].iter().enumerate() {
                let w = graph.edge_weights[i][k];
                let d = if w > 0.0 { 1.0 / w } else { 1.0 };
                dist[i][j] = dist[i][j].min(d);
            }
        }

        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let alt = dist[i][k] + dist[k][j];
                    if alt < dist[i][j] {
                        dist[i][j] = alt;
                    }
                }
            }
        }
        dist
    }

    fn detect_tunnels(&self, graph: &ProteinGraph, apsp: &[Vec<f64>]) -> Vec<Tunnel> {
        let n = graph.adjacency.len();
        let mut tunnels = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                let geodesic = apsp[i][j];
                if geodesic == f64::INFINITY {
                    continue;
                }
                let atom_i = &graph.structure_ref.atoms[graph.atom_indices[i]];
                let atom_j = &graph.structure_ref.atoms[graph.atom_indices[j]];
                let dx = atom_i.coord[0] - atom_j.coord[0];
                let dy = atom_i.coord[1] - atom_j.coord[1];
                let dz = atom_i.coord[2] - atom_j.coord[2];
                let euclidean = (dx * dx + dy * dy + dz * dz).sqrt();
                let score = geodesic / (euclidean + 0.1);
                if score > 3.0
                    && geodesic > self.config.tunnel_length_min
                    && euclidean < self.config.tunnel_width_max
                {
                    tunnels.push(Tunnel {
                        entrance: i,
                        exit: j,
                        length: geodesic,
                        width: euclidean,
                        score,
                    });
                }
            }
        }
        tunnels.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        tunnels.truncate(10);
        tunnels
    }

    fn compute_stress(&self, n: usize, apsp: &[Vec<f64>]) -> Vec<f64> {
        let mut stress = vec![0.0; n];
        for i in 0..n {
            let finite: Vec<f64> = apsp[i]
                .iter()
                .cloned()
                .filter(|d| *d < f64::INFINITY)
                .collect();
            if finite.is_empty() {
                continue;
            }
            let mean = finite.iter().sum::<f64>() / finite.len() as f64;
            stress[i] = mean;
        }
        stress
    }

    fn betweenness_centrality(&self, apsp: &[Vec<f64>]) -> Vec<f64> {
        let n = apsp.len();
        if n == 0 {
            return Vec::new();
        }
        let mut bc = vec![0.0; n];
        for s in 0..n {
            for t in (s + 1)..n {
                let dist_st = apsp[s][t];
                if !dist_st.is_finite() {
                    continue;
                }
                for v in 0..n {
                    if v == s || v == t {
                        continue;
                    }
                    let on_path = (apsp[s][v] + apsp[v][t] - dist_st).abs() < 1e-6;
                    if on_path {
                        bc[v] += 1.0;
                    }
                }
            }
        }
        let norm = ((n as f64 - 1.0) * (n as f64 - 2.0)).max(1.0);
        for v in &mut bc {
            *v /= norm;
        }
        bc
    }
}

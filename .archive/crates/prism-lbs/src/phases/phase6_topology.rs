//! Phase 6: Topological pocket detection (placeholder persistent homology)

use crate::graph::ProteinGraph;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopologicalPocketConfig {
    pub filtration_step: f64,
    pub max_radius: f64,
    pub persistence_threshold: f64,
    /// Maximum number of significant 1D features to retain
    pub max_features: usize,
}

impl Default for TopologicalPocketConfig {
    fn default() -> Self {
        Self {
            filtration_step: 0.5,
            max_radius: 15.0,
            persistence_threshold: 0.3,
            max_features: 32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PersistencePair {
    pub dimension: usize,
    pub birth: f64,
    pub death: f64,
    pub representative_vertices: Vec<usize>,
}

impl PersistencePair {
    pub fn persistence(&self) -> f64 {
        if self.death == f64::INFINITY {
            self.death
        } else {
            self.death - self.birth
        }
    }
}

#[derive(Debug, Clone)]
pub struct TopologicalPocketOutput {
    pub persistence_pairs: Vec<PersistencePair>,
    pub pocket_features: Vec<PersistencePair>,
    pub pocket_membership: Vec<Vec<usize>>,
    pub topology_scores: Vec<f64>,
}

pub struct TopologicalPocketPhase {
    config: TopologicalPocketConfig,
}

impl TopologicalPocketPhase {
    pub fn new(config: TopologicalPocketConfig) -> Self {
        Self { config }
    }

    pub fn execute(&self, graph: &ProteinGraph) -> TopologicalPocketOutput {
        let pairs = self.compute_persistence(graph);
        let pocket_features: Vec<PersistencePair> = pairs
            .iter()
            .filter(|p| p.dimension == 1 && p.persistence() > self.config.persistence_threshold)
            .cloned()
            .collect();

        let pocket_membership = self.assign_membership(graph, &pocket_features);
        let topology_scores: Vec<f64> = (0..graph.adjacency.len())
            .map(|i| {
                pocket_features
                    .iter()
                    .filter(|f| f.representative_vertices.contains(&i))
                    .map(|f| f.persistence())
                    .sum()
            })
            .collect();

        TopologicalPocketOutput {
            persistence_pairs: pairs,
            pocket_features,
            pocket_membership,
            topology_scores,
        }
    }

    fn compute_persistence(&self, graph: &ProteinGraph) -> Vec<PersistencePair> {
        let n = graph.adjacency.len();
        let points: Vec<[f64; 3]> = graph
            .atom_indices
            .iter()
            .map(|&i| graph.structure_ref.atoms[i].coord)
            .collect();

        // Build edges within max_radius
        let mut edges: Vec<(usize, usize, f64)> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = points[i][0] - points[j][0];
                let dy = points[i][1] - points[j][1];
                let dz = points[i][2] - points[j][2];
                let d = (dx * dx + dy * dy + dz * dz).sqrt();
                if d <= self.config.max_radius {
                    edges.push((i, j, d));
                }
            }
        }
        edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        // Kruskal to build MST and 0-d persistence
        let mut parent: Vec<usize> = (0..n).collect();
        let mut mst_adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
        let mut pairs = Vec::new();
        for (i, j, w) in &edges {
            let pi = find(&mut parent, *i);
            let pj = find(&mut parent, *j);
            if pi != pj {
                pairs.push(PersistencePair {
                    dimension: 0,
                    birth: 0.0,
                    death: *w,
                    representative_vertices: vec![*i, *j],
                });
                union(&mut parent, pi, pj);
                mst_adj[*i].push((*j, *w));
                mst_adj[*j].push((*i, *w));
            }
        }

        // For edges not in MST, compute 1-cycles via max edge on MST path
        let mst_edges: HashSet<(usize, usize)> = mst_adj
            .iter()
            .enumerate()
            .flat_map(|(u, nbrs)| {
                nbrs.iter()
                    .map(move |(v, _)| if u < *v { (u, *v) } else { (*v, u) })
            })
            .collect();

        for (i, j, w) in edges {
            let key = if i < j { (i, j) } else { (j, i) };
            if mst_edges.contains(&key) {
                continue;
            }
            if let Some(max_on_path) = max_edge_on_mst_path(i, j, &mst_adj, n) {
                let birth = max_on_path;
                let death = w;
                if death > birth && death - birth >= self.config.persistence_threshold {
                    pairs.push(PersistencePair {
                        dimension: 1,
                        birth,
                        death,
                        representative_vertices: vec![i, j],
                    });
                }
            }
        }

        // keep most persistent features
        pairs.sort_by(|a, b| b.persistence().partial_cmp(&a.persistence()).unwrap());
        pairs.truncate(self.config.max_features);
        pairs
    }

    fn assign_membership(
        &self,
        graph: &ProteinGraph,
        features: &[PersistencePair],
    ) -> Vec<Vec<usize>> {
        let n = graph.adjacency.len();
        let mut membership = vec![Vec::new(); n];
        for (idx, feature) in features.iter().enumerate() {
            for &v in &feature.representative_vertices {
                if v < n {
                    membership[v].push(idx);
                }
            }
        }
        membership
    }
}

fn find(parent: &mut [usize], i: usize) -> usize {
    if parent[i] != i {
        parent[i] = find(parent, parent[i]);
    }
    parent[i]
}

fn union(parent: &mut [usize], x: usize, y: usize) {
    let px = find(parent, x);
    let py = find(parent, y);
    if px != py {
        parent[py] = px;
    }
}

fn max_edge_on_mst_path(
    start: usize,
    end: usize,
    mst_adj: &[Vec<(usize, f64)>],
    n: usize,
) -> Option<f64> {
    use std::collections::VecDeque;
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();
    queue.push_back((start, 0.0));
    visited[start] = true;

    while let Some((node, current_max)) = queue.pop_front() {
        if node == end {
            return Some(current_max);
        }
        for &(nbr, w) in &mst_adj[node] {
            if !visited[nbr] {
                visited[nbr] = true;
                queue.push_back((nbr, current_max.max(w)));
            }
        }
    }
    None
}

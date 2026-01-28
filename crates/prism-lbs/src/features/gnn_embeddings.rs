//! GNN-derived embeddings for enhanced pocket feature representation
//!
//! Uses E3EquivariantGnn from prism-gnn to generate learned embeddings
//! that capture local neighborhood patterns for each atom/residue.

use prism_gnn::{GnnConfig, GnnModel};
use petgraph::graph::DiGraph;
use crate::graph::ProteinGraph;
use crate::LbsError;

/// GNN embedding configuration
#[derive(Debug, Clone)]
pub struct GnnEmbeddingConfig {
    /// Embedding dimension (output features per node)
    pub embedding_dim: usize,
    /// Number of message passing layers
    pub num_layers: usize,
    /// Use GPU acceleration if available
    pub use_gpu: bool,
}

impl Default for GnnEmbeddingConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 32,
            num_layers: 3,
            use_gpu: true,
        }
    }
}

/// GNN embedding computer
pub struct GnnEmbeddingComputer {
    config: GnnEmbeddingConfig,
    gnn: GnnModel,
}

impl GnnEmbeddingComputer {
    /// Create new GNN embedding computer
    pub fn new(config: GnnEmbeddingConfig) -> Result<Self, LbsError> {
        let gnn_config = GnnConfig {
            hidden_dim: config.embedding_dim,
            num_layers: config.num_layers,
            use_gpu: config.use_gpu,
            ..Default::default()
        };

        let gnn = GnnModel::new(gnn_config);

        Ok(Self { config, gnn })
    }

    /// Compute GNN embeddings for all vertices in protein graph
    pub fn compute_embeddings(&self, graph: &ProteinGraph) -> Result<Vec<Vec<f64>>, LbsError> {
        let n = graph.atom_indices.len();

        // Convert to petgraph DiGraph for GNN
        let di_graph = self.build_petgraph(graph);

        // Run GNN forward pass
        let embeddings_map = self.gnn.forward(&di_graph)
            .map_err(|e| LbsError::PhaseExecution(format!("GNN forward failed: {}", e)))?;

        // Extract embeddings in order
        let mut embeddings = Vec::with_capacity(n);
        for i in 0..n {
            let idx = petgraph::graph::NodeIndex::new(i);
            let emb = embeddings_map.get(&idx)
                .cloned()
                .unwrap_or_else(|| vec![0.0; self.config.embedding_dim]);
            embeddings.push(emb.into_iter().map(|x| x as f64).collect());
        }

        Ok(embeddings)
    }

    /// Compute average GNN embedding for a pocket (set of vertex indices)
    pub fn pocket_embedding(&self, graph: &ProteinGraph, vertex_indices: &[usize]) -> Result<Vec<f64>, LbsError> {
        if vertex_indices.is_empty() {
            return Ok(vec![0.0; self.config.embedding_dim]);
        }

        let all_embeddings = self.compute_embeddings(graph)?;
        let mut avg = vec![0.0; self.config.embedding_dim];

        for &idx in vertex_indices {
            if idx < all_embeddings.len() {
                for (i, &v) in all_embeddings[idx].iter().enumerate() {
                    avg[i] += v;
                }
            }
        }

        let count = vertex_indices.len() as f64;
        for v in &mut avg {
            *v /= count;
        }

        Ok(avg)
    }

    /// Build petgraph DiGraph from ProteinGraph
    fn build_petgraph(&self, graph: &ProteinGraph) -> DiGraph<f32, f32> {
        let n = graph.atom_indices.len();
        let mut di_graph = DiGraph::new();

        // Add nodes with features
        for i in 0..n {
            let atom_idx = graph.atom_indices[i];
            let atom = &graph.structure_ref.atoms[atom_idx];

            // Node weight: combined scalar feature
            let feature = (graph.vertex_features.hydrophobicity[i]
                + graph.vertex_features.depth[i]
                + graph.vertex_features.curvature[i]
                + atom.sasa / 100.0) as f32;

            di_graph.add_node(feature);
        }

        // Add edges
        for (i, neighbors) in graph.adjacency.iter().enumerate() {
            for (j, &neighbor) in neighbors.iter().enumerate() {
                let weight = graph.edge_weights[i].get(j).copied().unwrap_or(1.0) as f32;
                di_graph.add_edge(
                    petgraph::graph::NodeIndex::new(i),
                    petgraph::graph::NodeIndex::new(neighbor),
                    weight,
                );
            }
        }

        di_graph
    }

    /// Get embedding dimension
    pub fn embedding_dim(&self) -> usize {
        self.config.embedding_dim
    }
}

/// Compute druggability score contribution from GNN embeddings
///
/// Uses learned embedding space distance from "ideal druggable pocket" prototype
pub fn gnn_druggability_score(embeddings: &[f64], ideal_prototype: &[f64]) -> f64 {
    if embeddings.len() != ideal_prototype.len() || embeddings.is_empty() {
        return 0.5;  // neutral score
    }

    // Cosine similarity
    let dot: f64 = embeddings.iter().zip(ideal_prototype).map(|(a, b)| a * b).sum();
    let norm_a: f64 = embeddings.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = ideal_prototype.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        let cosine_sim = dot / (norm_a * norm_b);
        // Map [-1, 1] to [0, 1]
        (cosine_sim + 1.0) / 2.0
    } else {
        0.5
    }
}

/// Default "ideal druggable pocket" prototype embedding
/// Learned from PDBBind training data (placeholder values)
pub fn default_druggable_prototype(dim: usize) -> Vec<f64> {
    // Prototype learned from druggable pockets: positive curvature, moderate hydrophobicity,
    // good enclosure, hydrogen bonding capacity
    let mut prototype = vec![0.0; dim];

    // Set some characteristic values (would be learned from data)
    if dim >= 8 {
        prototype[0] = 0.6;   // moderate hydrophobicity
        prototype[1] = 0.4;   // some charge
        prototype[2] = 0.7;   // concave curvature
        prototype[3] = 0.5;   // moderate depth
        prototype[4] = 0.8;   // enclosed
        prototype[5] = 0.4;   // moderate flexibility
        prototype[6] = 0.6;   // some conservation
        prototype[7] = 0.5;   // topology
    }

    prototype
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_computation() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let score = gnn_druggability_score(&a, &b);
        assert!((score - 1.0).abs() < 0.01, "Identical vectors should give score 1.0");

        let c = vec![-1.0, 0.0, 0.0];
        let score_opposite = gnn_druggability_score(&a, &c);
        assert!((score_opposite - 0.0).abs() < 0.01, "Opposite vectors should give score 0.0");
    }
}

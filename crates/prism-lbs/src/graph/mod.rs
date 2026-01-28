//! Protein graph construction utilities

pub mod contact_map;
pub mod distance_matrix;
pub mod protein_graph;

pub use distance_matrix::DistanceMatrix;
pub use protein_graph::{GraphConfig, ProteinGraph, ProteinGraphBuilder, VertexFeatures};

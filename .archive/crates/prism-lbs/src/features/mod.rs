//! Feature computation modules
//!
//! Includes both hand-crafted features and GNN-derived learned embeddings.

pub mod conservation;
pub mod electrostatics;
pub mod geometry;
pub mod gnn_embeddings;
pub mod hydrophobicity;

pub use conservation::conservation_feature;
pub use electrostatics::{
    charge_complementarity, electrostatic_feature, electrostatic_polarity,
    local_electrostatic_potential, mean_electrostatic_potential,
};
pub use geometry::{curvature_feature, depth_feature};
pub use gnn_embeddings::{
    default_druggable_prototype, gnn_druggability_score, GnnEmbeddingComputer, GnnEmbeddingConfig,
};
pub use hydrophobicity::hydrophobicity_feature;

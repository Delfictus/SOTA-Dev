//! # PRISM GNN (Graph Neural Network) Module
//!
//! Implements graph neural network components for advanced graph processing.
//! Supports GCN, GAT, GraphSAGE, and other GNN architectures.

use petgraph::graph::{DiGraph, NodeIndex};
use prism_core::PrismError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub mod inference;
pub mod layers;
pub mod models;
pub mod training;

// ONNX Runtime integration (feature-gated)
#[cfg(feature = "ort")]
pub mod onnx_runtime;

#[cfg(feature = "cuda")]
use std::sync::Arc;

// Re-export key types
pub use models::{E3EquivariantGnn, GnnPrediction, ManifoldFeatures, OnnxGnn};

/// GNN configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GnnConfig {
 /// Number of hidden dimensions
 pub hidden_dim: usize,

 /// Number of layers
 pub num_layers: usize,

 /// Dropout rate
 pub dropout: f32,

 /// Learning rate
 pub learning_rate: f32,

 /// GNN architecture type
 pub architecture: GnnArchitecture,

 /// Enable GPU acceleration
 pub use_gpu: bool,
}

/// Supported GNN architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GnnArchitecture {
 GCN, // Graph Convolutional Network
 GAT, // Graph Attention Network
 GraphSAGE, // Graph Sample and Aggregate
 GIN, // Graph Isomorphism Network
}

impl Default for GnnConfig {
 fn default() -> Self {
 Self {
 hidden_dim: 128,
 num_layers: 3,
 dropout: 0.5,
 learning_rate: 0.001,
 architecture: GnnArchitecture::GCN,
 use_gpu: false,
 }
 }
}

/// Graph neural network model
pub struct GnnModel {
 config: GnnConfig,
 parameters: HashMap<String, ndarray::ArrayD<f32>>,
 #[cfg(feature = "cuda")]
 gpu_device: Option<Arc<cudarc::driver::CudaContext>>,
 #[cfg(feature = "cuda")]
 gpu_stream: Option<Arc<cudarc::driver::CudaStream>>,
}

impl GnnModel {
 pub fn new(config: GnnConfig) -> Self {
 Self {
 config,
 parameters: HashMap::new(),
 #[cfg(feature = "cuda")]
 gpu_device: None,
 #[cfg(feature = "cuda")]
 gpu_stream: None,
 }
 }

 #[cfg(feature = "cuda")]
 pub fn with_gpu(mut self, device: Arc<cudarc::driver::CudaContext>) -> Self {
 let stream = device.default_stream();
 self.gpu_device = Some(device);
 self.gpu_stream = Some(stream);
 self
 }

 /// GPU-accelerated forward pass using E3EquivariantGnn
 pub fn forward(
 &self,
 graph: &DiGraph<f32, f32>,
 ) -> Result<HashMap<NodeIndex, Vec<f32>>, PrismError> {
 // Use E3EquivariantGnn for actual inference
 let gnn = models::E3EquivariantGnn::new(self.config.clone());

 #[cfg(feature = "cuda")]
 let gnn = if let Some(ref device) = self.gpu_device {
 gnn.with_gpu(device.clone())
 } else {
 gnn
 };

 let prediction = gnn.predict(graph).map_err(|e| {
 PrismError::GnnError(format!("GNN forward pass failed: {}", e))
 })?;

 // Convert node embeddings from usize keys to NodeIndex
 let result: HashMap<NodeIndex, Vec<f32>> = prediction
 .node_embeddings
 .into_iter()
 .map(|(idx, emb)| (NodeIndex::new(idx), emb))
 .collect();

 log::debug!(
 "GNN forward: {} embeddings, chromatic estimate: {}",
 result.len(),
 prediction.chromatic_number
 );

 Ok(result)
 }

 /// TODO(GPU-GNN-02): GPU-accelerated backward pass and gradient computation
 pub fn backward(&mut self, loss: f32) -> Result<(), PrismError> {
 // Placeholder for GPU-accelerated backpropagation
 Ok(())
 }

 /// TODO(GPU-GNN-03): GPU-accelerated parameter updates
 pub fn update_parameters(&mut self, gradients: &HashMap<String, ndarray::ArrayD<f32>>) {
 // Placeholder for GPU-accelerated parameter updates
 }
}

/// GNN training metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GnnMetrics {
 pub loss: f32,
 pub accuracy: f32,
 pub epoch: usize,
 pub learning_rate: f32,
}

#[cfg(test)]
mod tests {
 use super::*;

 #[test]
 fn test_gnn_config() {
 let config = GnnConfig::default();
 assert_eq!(config.hidden_dim, 128);
 assert_eq!(config.num_layers, 3);
 }
}

//! ONNX Runtime inference with CUDA acceleration
//!
//! Provides GPU-accelerated GNN inference using the trained ONNX model.
//! Automatically detects CUDA availability and falls back to CPU if needed.
//!
//! NOTE: Full ONNX Runtime 2.0 integration requires additional work to match the new API.
//! For now, this module provides the interface but falls back to E3EquivariantGnn.

use anyhow::{Context, Result};
use ndarray::{Array2, ArrayView2};
use petgraph::graph::DiGraph;
use std::path::Path;
use std::collections::HashMap;

/// GNN inference via ONNX Runtime
///
/// This is a placeholder implementation. Full ONNX Runtime integration
/// will be completed once the ort 2.0 API patterns are finalized.
pub struct OnnxGnnRuntime {
    model_path: String,
    use_gpu: bool,
    node_feature_dim: usize,
}

impl OnnxGnnRuntime {
    /// Load ONNX model with automatic GPU detection
    ///
    /// Currently returns a placeholder that indicates ONNX models are not yet loaded.
    /// The actual inference will fall back to E3EquivariantGnn in models.rs.
    pub fn load<P: AsRef<Path>>(model_path: P) -> Result<Self> {
        let model_path_str = model_path.as_ref().to_string_lossy().to_string();

        log::info!("ONNX Runtime stub: Model path registered: {}", model_path_str);
        log::warn!("ONNX Runtime 2.0 full integration pending. Using E3EquivariantGnn fallback.");

        Ok(Self {
            model_path: model_path_str,
            use_gpu: true,  // Indicate GPU preference
            node_feature_dim: 16,
        })
    }

    /// Run GNN inference (currently returns error to trigger fallback)
    pub fn predict(
        &self,
        _node_features: ArrayView2<f32>,
        _edge_index: ArrayView2<i64>,
    ) -> Result<GnnPrediction> {
        anyhow::bail!(
            "ONNX Runtime 2.0 integration incomplete. Please use OnnxGnn which has E3EquivariantGnn fallback."
        )
    }

    pub fn is_gpu_enabled(&self) -> bool {
        self.use_gpu
    }

    /// Run inference on petgraph DiGraph (convenience wrapper)
    ///
    /// Currently returns error to trigger fallback path.
    pub fn predict_graph(&self, _graph: &DiGraph<f32, f32>) -> Result<HashMap<usize, Vec<f32>>> {
        anyhow::bail!(
            "ONNX Runtime 2.0 integration incomplete. Fallback to E3EquivariantGnn recommended."
        )
    }
}

#[derive(Debug, Clone)]
pub struct GnnPrediction {
    pub chromatic_number: usize,
    pub node_colors: Vec<usize>,
    pub color_probabilities: Vec<Vec<f32>>,
    pub confidence: f32,
    pub gpu_accelerated: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_runtime_stub() {
        // Test that the stub compiles and provides appropriate fallback behavior
        let model_path = "models/gnn/gnn_model.onnx";
        let runtime = OnnxGnnRuntime::load(model_path);
        assert!(runtime.is_ok(), "Stub should initialize successfully");
    }
}

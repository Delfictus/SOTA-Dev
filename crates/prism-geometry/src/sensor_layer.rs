//! Geometry Sensor Layer for Metaphysical Telemetry Coupling
//!
//! Computes geometric stress metrics from graph embeddings:
//! - Bounding box area and growth rate
//! - Overlap density (vertex pairs closer than threshold)
//! - Curvature stress (edge length variance)
//! - Anchor hotspot detection
//!
//! GPU acceleration via CUDA kernels with CPU fallback for simulation mode.

use anyhow::{Context, Result};
use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, LaunchConfig, PushKernelArg};
use log::info;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

/// Geometry stress metrics computed by the sensor layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryMetrics {
    /// Bounding box area (min_x, max_x, min_y, max_y, area)
    pub bounding_box: BoundingBox,

    /// Mean overlap density (vertices per vertex within threshold)
    pub mean_overlap_density: f64,

    /// Max overlap density (most crowded region)
    pub max_overlap_density: f64,

    /// Mean curvature stress (edge length variance)
    pub mean_curvature_stress: f64,

    /// Max curvature stress
    pub max_curvature_stress: f64,

    /// Number of anchor hotspots detected
    pub num_hotspots: usize,

    /// Mean hotspot intensity
    pub mean_hotspot_intensity: f64,

    /// Computation time (milliseconds)
    pub computation_time_ms: f64,

    /// Whether GPU was used
    pub used_gpu: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min_x: f32,
    pub max_x: f32,
    pub min_y: f32,
    pub max_y: f32,
    pub area: f32,
}

impl BoundingBox {
    pub fn new(min_x: f32, max_x: f32, min_y: f32, max_y: f32) -> Self {
        let area = (max_x - min_x) * (max_y - min_y);
        Self {
            min_x,
            max_x,
            min_y,
            max_y,
            area,
        }
    }
}

/// GPU-accelerated geometry sensor layer
///
/// ASSUMPTIONS:
/// - Graph positions stored as (x, y) coordinates in f32 pairs
/// - MAX_VERTICES = 100,000 (enforced by validation)
/// - Block size: 256 threads (optimal for coalesced access on sm_75+)
/// - Requires: sm_75+ for warp-level primitives
///
/// REFERENCE: Metaphysical Telemetry Coupling - Geometry Sensor Layer
pub struct GeometrySensorLayer {
    context: Arc<CudaContext>,

    // CUDA kernels
    kernel_overlap: CudaFunction,
    kernel_bbox: CudaFunction,
    kernel_hotspots: CudaFunction,
    kernel_curvature: CudaFunction,

    // Configuration
    use_gpu: bool,
}

const MAX_VERTICES: usize = 100_000;
const OVERLAP_THRESHOLD: f32 = 0.1;
const HOTSPOT_RADIUS: f32 = 0.5;
const HOTSPOT_THRESHOLD: f32 = 3.0; // Min anchor density to be considered hotspot

impl GeometrySensorLayer {
    /// Initialize geometry sensor layer from PTX module
    ///
    /// # Arguments
    /// * `context` - CUDA context handle
    /// * `ptx_path` - Path to stress_analysis.ptx module
    ///
    /// # Errors
    /// Returns error if PTX loading fails or kernels are not found.
    ///
    /// # Safety
    /// Caller must ensure PTX module is valid and compiled for compatible architecture.
    pub fn new(context: Arc<CudaContext>, ptx_path: &str) -> Result<Self> {
        info!("Loading Geometry Sensor PTX module from: {}", ptx_path);

        let ptx_data = std::fs::read(ptx_path)
            .with_context(|| format!("Failed to read PTX from {}", ptx_path))?;

        let ptx = cudarc::nvrtc::Ptx::from_src(std::str::from_utf8(&ptx_data)?);

        // cudarc 0.18.1 API: load_module returns CudaModule
        let module = context
            .load_module(ptx)
            .context("Failed to load PTX module")?;

        // Load individual kernel functions from module
        let kernel_overlap = module
            .load_function("compute_overlap_density")
            .context("Failed to load kernel: compute_overlap_density")?;

        let kernel_bbox = module
            .load_function("compute_bounding_box")
            .context("Failed to load kernel: compute_bounding_box")?;

        let kernel_hotspots = module
            .load_function("detect_anchor_hotspots")
            .context("Failed to load kernel: detect_anchor_hotspots")?;

        let kernel_curvature = module
            .load_function("compute_curvature_stress")
            .context("Failed to load kernel: compute_curvature_stress")?;

        info!("Geometry Sensor GPU module loaded successfully");

        Ok(Self {
            context,
            kernel_overlap,
            kernel_bbox,
            kernel_hotspots,
            kernel_curvature,
            use_gpu: true,
        })
    }

    /// Compute geometry stress metrics on GPU
    ///
    /// # Arguments
    /// * `positions` - Vertex positions as (x, y) pairs [num_vertices * 2]
    /// * `num_vertices` - Number of vertices
    /// * `adjacency` - CSR adjacency (row_ptr, col_idx)
    /// * `anchors` - Structural anchor vertex indices
    ///
    /// # Returns
    /// GeometryMetrics with all stress analysis results
    ///
    /// # Errors
    /// Returns error if GPU operations fail or input validation fails.
    pub fn compute_metrics(
        &self,
        positions: &[f32],
        num_vertices: usize,
        adjacency: &[Vec<usize>],
        anchors: &[usize],
    ) -> Result<GeometryMetrics> {
        let start = Instant::now();

        // Validation
        anyhow::ensure!(
            num_vertices <= MAX_VERTICES,
            "Graph exceeds MAX_VERTICES limit: {} > {}",
            num_vertices,
            MAX_VERTICES
        );

        anyhow::ensure!(
            positions.len() == num_vertices * 2,
            "Position array size mismatch: {} != {}",
            positions.len(),
            num_vertices * 2
        );

        if self.use_gpu {
            self.compute_metrics_gpu(positions, num_vertices, adjacency, anchors, start)
        } else {
            self.compute_metrics_cpu(positions, num_vertices, adjacency, anchors, start)
        }
    }

    /// GPU implementation of geometry metrics computation
    fn compute_metrics_gpu(
        &self,
        positions: &[f32],
        num_vertices: usize,
        adjacency: &[Vec<usize>],
        anchors: &[usize],
        start: Instant,
    ) -> Result<GeometryMetrics> {
        // Get stream and upload positions to GPU
        let stream = self.context.default_stream();
        let d_positions = stream
            .clone_htod(positions)
            .context("Failed to copy positions to GPU")?;

        // 1. Compute bounding box
        let bbox = self.compute_bbox_gpu(&d_positions, num_vertices)?;

        // 2. Compute overlap density
        let (mean_overlap, max_overlap) = self.compute_overlap_gpu(&d_positions, num_vertices)?;

        // 3. Compute curvature stress
        let (mean_curvature, max_curvature) =
            self.compute_curvature_gpu(&d_positions, num_vertices, adjacency)?;

        // 4. Detect anchor hotspots
        let (num_hotspots, mean_intensity) =
            self.compute_hotspots_gpu(&d_positions, num_vertices, anchors)?;

        let elapsed = start.elapsed();

        Ok(GeometryMetrics {
            bounding_box: bbox,
            mean_overlap_density: mean_overlap,
            max_overlap_density: max_overlap,
            mean_curvature_stress: mean_curvature,
            max_curvature_stress: max_curvature,
            num_hotspots,
            mean_hotspot_intensity: mean_intensity,
            computation_time_ms: elapsed.as_secs_f64() * 1000.0,
            used_gpu: true,
        })
    }

    /// Compute bounding box on GPU
    fn compute_bbox_gpu(
        &self,
        d_positions: &CudaSlice<f32>,
        num_vertices: usize,
    ) -> Result<BoundingBox> {
        let stream = self.context.default_stream();
        let bbox_host = vec![0.0f32; 4];
        let d_bbox = stream.clone_htod(&bbox_host)?;

        let config = LaunchConfig {
            grid_dim: (1, 1, 1), // Single block for reduction
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(&self.kernel_bbox)
                .arg(d_positions)
                .arg(&d_bbox)
                .arg(&(num_vertices as u32))
                .launch(config)?;
        }

        stream.synchronize()?;

        let bbox_result = stream.clone_dtoh(&d_bbox)?;

        Ok(BoundingBox::new(
            bbox_result[0],
            bbox_result[1],
            bbox_result[2],
            bbox_result[3],
        ))
    }

    /// Compute overlap density on GPU
    fn compute_overlap_gpu(
        &self,
        d_positions: &CudaSlice<f32>,
        num_vertices: usize,
    ) -> Result<(f64, f64)> {
        let stream = self.context.default_stream();
        let overlap_host = vec![0.0f32; num_vertices];
        let d_overlap = stream.clone_htod(&overlap_host)?;

        let block_size = 256;
        let grid_size = num_vertices.div_ceil(block_size) as u32;

        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(&self.kernel_overlap)
                .arg(d_positions)
                .arg(&d_overlap)
                .arg(&(num_vertices as u32))
                .launch(config)?;
        }

        stream.synchronize()?;

        let overlap_result = stream.clone_dtoh(&d_overlap)?;

        let mean = overlap_result.iter().map(|&x| x as f64).sum::<f64>() / num_vertices as f64;
        let max = overlap_result
            .iter()
            .map(|&x| x as f64)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        Ok((mean, max))
    }

    /// Compute curvature stress on GPU
    fn compute_curvature_gpu(
        &self,
        d_positions: &CudaSlice<f32>,
        num_vertices: usize,
        adjacency: &[Vec<usize>],
    ) -> Result<(f64, f64)> {
        // Convert adjacency to CSR format
        let (row_ptr, col_idx) = self.adjacency_to_csr(adjacency, num_vertices);

        let stream = self.context.default_stream();
        let d_row_ptr = stream.clone_htod(&row_ptr)?;
        let d_col_idx = stream.clone_htod(&col_idx)?;

        let curvature_host = vec![0.0f32; num_vertices];
        let d_curvature = stream.clone_htod(&curvature_host)?;

        let block_size = 256;
        let grid_size = num_vertices.div_ceil(block_size) as u32;

        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(&self.kernel_curvature)
                .arg(d_positions)
                .arg(&d_row_ptr)
                .arg(&d_col_idx)
                .arg(&d_curvature)
                .arg(&(num_vertices as u32))
                .launch(config)?;
        }

        stream.synchronize()?;

        let curvature_result = stream.clone_dtoh(&d_curvature)?;

        let mean = curvature_result.iter().map(|&x| x as f64).sum::<f64>() / num_vertices as f64;
        let max = curvature_result
            .iter()
            .map(|&x| x as f64)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        Ok((mean, max))
    }

    /// Detect anchor hotspots on GPU
    fn compute_hotspots_gpu(
        &self,
        d_positions: &CudaSlice<f32>,
        num_vertices: usize,
        anchors: &[usize],
    ) -> Result<(usize, f64)> {
        if anchors.is_empty() {
            return Ok((0, 0.0));
        }

        let stream = self.context.default_stream();
        let anchors_u32: Vec<u32> = anchors.iter().map(|&a| a as u32).collect();
        let d_anchors = stream.clone_htod(&anchors_u32)?;

        let hotspot_host = vec![0.0f32; num_vertices];
        let d_hotspot = stream.clone_htod(&hotspot_host)?;

        let block_size = 256;
        let grid_size = num_vertices.div_ceil(block_size) as u32;

        let config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            stream.launch_builder(&self.kernel_hotspots)
                .arg(d_positions)
                .arg(&d_anchors)
                .arg(&d_hotspot)
                .arg(&(num_vertices as u32))
                .arg(&(anchors.len() as u32))
                .launch(config)?;
        }

        stream.synchronize()?;

        let hotspot_result = stream.clone_dtoh(&d_hotspot)?;

        let num_hotspots = hotspot_result
            .iter()
            .filter(|&&score| score >= HOTSPOT_THRESHOLD)
            .count();
        let mean_intensity =
            hotspot_result.iter().map(|&x| x as f64).sum::<f64>() / num_vertices as f64;

        Ok((num_hotspots, mean_intensity))
    }

    /// CPU fallback implementation
    fn compute_metrics_cpu(
        &self,
        positions: &[f32],
        num_vertices: usize,
        adjacency: &[Vec<usize>],
        anchors: &[usize],
        start: Instant,
    ) -> Result<GeometryMetrics> {
        // Compute bounding box
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for v in 0..num_vertices {
            let x = positions[v * 2];
            let y = positions[v * 2 + 1];
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        let bbox = BoundingBox::new(min_x, max_x, min_y, max_y);

        // Compute overlap density
        let mut overlap_counts = vec![0.0f32; num_vertices];
        for v in 0..num_vertices {
            let vx = positions[v * 2];
            let vy = positions[v * 2 + 1];

            for u in 0..num_vertices {
                if u == v {
                    continue;
                }
                let ux = positions[u * 2];
                let uy = positions[u * 2 + 1];

                let dist = ((vx - ux).powi(2) + (vy - uy).powi(2)).sqrt();
                if dist < OVERLAP_THRESHOLD {
                    overlap_counts[v] += 1.0;
                }
            }
        }

        let mean_overlap =
            overlap_counts.iter().map(|&x| x as f64).sum::<f64>() / num_vertices as f64;
        let max_overlap = overlap_counts
            .iter()
            .map(|&x| x as f64)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        // Compute curvature stress
        let mut curvature = vec![0.0f32; num_vertices];
        for v in 0..num_vertices {
            if v >= adjacency.len() {
                continue;
            }

            let neighbors = &adjacency[v];
            if neighbors.is_empty() {
                continue;
            }

            let vx = positions[v * 2];
            let vy = positions[v * 2 + 1];

            // Compute mean edge length
            let mut sum_lengths = 0.0f32;
            for &u in neighbors {
                if u >= num_vertices {
                    continue;
                }
                let ux = positions[u * 2];
                let uy = positions[u * 2 + 1];
                sum_lengths += ((vx - ux).powi(2) + (vy - uy).powi(2)).sqrt();
            }

            let mean_length = sum_lengths / neighbors.len() as f32;

            // Compute variance
            let mut variance = 0.0f32;
            for &u in neighbors {
                if u >= num_vertices {
                    continue;
                }
                let ux = positions[u * 2];
                let uy = positions[u * 2 + 1];
                let length = ((vx - ux).powi(2) + (vy - uy).powi(2)).sqrt();
                variance += (length - mean_length).powi(2);
            }

            curvature[v] = variance / neighbors.len() as f32;
        }

        let mean_curvature = curvature.iter().map(|&x| x as f64).sum::<f64>() / num_vertices as f64;
        let max_curvature = curvature
            .iter()
            .map(|&x| x as f64)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        // Detect hotspots
        let mut hotspot_scores = vec![0.0f32; num_vertices];
        for v in 0..num_vertices {
            let vx = positions[v * 2];
            let vy = positions[v * 2 + 1];

            for &anchor_idx in anchors {
                if anchor_idx >= num_vertices {
                    continue;
                }
                let ax = positions[anchor_idx * 2];
                let ay = positions[anchor_idx * 2 + 1];

                let dist = ((vx - ax).powi(2) + (vy - ay).powi(2)).sqrt();
                if dist < HOTSPOT_RADIUS {
                    hotspot_scores[v] += 1.0;
                }
            }
        }

        let num_hotspots = hotspot_scores
            .iter()
            .filter(|&&score| score >= HOTSPOT_THRESHOLD)
            .count();
        let mean_intensity =
            hotspot_scores.iter().map(|&x| x as f64).sum::<f64>() / num_vertices as f64;

        let elapsed = start.elapsed();

        Ok(GeometryMetrics {
            bounding_box: bbox,
            mean_overlap_density: mean_overlap,
            max_overlap_density: max_overlap,
            mean_curvature_stress: mean_curvature,
            max_curvature_stress: max_curvature,
            num_hotspots,
            mean_hotspot_intensity: mean_intensity,
            computation_time_ms: elapsed.as_secs_f64() * 1000.0,
            used_gpu: false,
        })
    }

    /// Convert adjacency list to CSR format
    fn adjacency_to_csr(
        &self,
        adjacency: &[Vec<usize>],
        num_vertices: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        let mut row_ptr = vec![0u32];
        let mut col_idx = Vec::new();

        for v in 0..num_vertices {
            if v < adjacency.len() {
                for &neighbor in &adjacency[v] {
                    col_idx.push(neighbor as u32);
                }
            }
            row_ptr.push(col_idx.len() as u32);
        }

        (row_ptr, col_idx)
    }
}

/// Standalone CPU-only geometry sensor (for simulation mode)
pub struct GeometrySensorCpu;

impl Default for GeometrySensorCpu {
    fn default() -> Self {
        Self::new()
    }
}

impl GeometrySensorCpu {
    pub fn new() -> Self {
        Self
    }

    pub fn compute_metrics(
        &self,
        positions: &[f32],
        num_vertices: usize,
        _adjacency: &[Vec<usize>],
        anchors: &[usize],
    ) -> Result<GeometryMetrics> {
        let start = Instant::now();

        // Validation
        anyhow::ensure!(
            positions.len() == num_vertices * 2,
            "Position array size mismatch: {} != {}",
            positions.len(),
            num_vertices * 2
        );

        // Compute bounding box
        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for v in 0..num_vertices {
            let x = positions[v * 2];
            let y = positions[v * 2 + 1];
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }

        let bbox = BoundingBox::new(min_x, max_x, min_y, max_y);

        // Compute overlap density (simplified for CPU)
        let mut overlap_counts = vec![0.0f32; num_vertices];
        for v in 0..num_vertices {
            let vx = positions[v * 2];
            let vy = positions[v * 2 + 1];

            for u in 0..num_vertices {
                if u == v {
                    continue;
                }
                let ux = positions[u * 2];
                let uy = positions[u * 2 + 1];

                let dist = ((vx - ux).powi(2) + (vy - uy).powi(2)).sqrt();
                if dist < OVERLAP_THRESHOLD {
                    overlap_counts[v] += 1.0;
                }
            }
        }

        let mean_overlap =
            overlap_counts.iter().map(|&x| x as f64).sum::<f64>() / num_vertices as f64;
        let max_overlap = overlap_counts
            .iter()
            .map(|&x| x as f64)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        // Simplified curvature and hotspot metrics
        let mean_curvature = 0.1; // Stub for simulation
        let max_curvature = 0.5;
        let num_hotspots = anchors.len().min(num_vertices / 10);
        let mean_intensity = anchors.len() as f64 / num_vertices as f64;

        let elapsed = start.elapsed();

        Ok(GeometryMetrics {
            bounding_box: bbox,
            mean_overlap_density: mean_overlap,
            max_overlap_density: max_overlap,
            mean_curvature_stress: mean_curvature,
            max_curvature_stress: max_curvature,
            num_hotspots,
            mean_hotspot_intensity: mean_intensity,
            computation_time_ms: elapsed.as_secs_f64() * 1000.0,
            used_gpu: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_sensor_basic() {
        let sensor = GeometrySensorCpu::new();

        // Triangle graph with unit positions
        let positions = vec![0.0, 0.0, 1.0, 0.0, 0.5, 0.866];
        let adjacency = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let anchors = vec![0];

        let metrics = sensor
            .compute_metrics(&positions, 3, &adjacency, &anchors)
            .expect("CPU sensor failed");

        assert!(metrics.bounding_box.area > 0.0);
        assert!(!metrics.used_gpu);
    }

    #[test]
    fn test_bounding_box() {
        let positions = vec![0.0, 0.0, 2.0, 3.0, 1.0, 1.5];
        let bbox = BoundingBox::new(0.0, 2.0, 0.0, 3.0);

        assert_eq!(bbox.min_x, 0.0);
        assert_eq!(bbox.max_x, 2.0);
        assert_eq!(bbox.min_y, 0.0);
        assert_eq!(bbox.max_y, 3.0);
        assert_eq!(bbox.area, 6.0);
    }
}

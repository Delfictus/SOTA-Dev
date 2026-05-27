//! # prism-geometry
//!
//! GPU-accelerated geometry stress analysis for metaphysical telemetry coupling.
//!
//! This crate provides geometric sensors that analyze graph embeddings and compute stress metrics:
//! - Bounding box area and growth rate
//! - Overlap density (vertex pairs closer than threshold)
//! - Curvature/torsion stress metrics
//! - Anchor hotspot detection
//!
//! These metrics feed into the thermodynamic phase to guide simulated annealing with geometry-aware stress parameters.

pub mod layout;
pub mod nvml_telemetry;
pub mod sensor_layer;

pub use layout::{generate_circular_layout, generate_random_layout, generate_spring_layout};
pub use nvml_telemetry::{GpuMetrics, NvmlTelemetry};
pub use sensor_layer::{BoundingBox, GeometryMetrics, GeometrySensorCpu, GeometrySensorLayer};

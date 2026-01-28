//! Integration tests for geometry stress analysis
//!
//! Tests GPU vs CPU equivalence for geometry sensor layer.

use prism_geometry::{
    generate_circular_layout, generate_random_layout, generate_spring_layout, GeometrySensorCpu,
};

#[test]
fn test_cpu_sensor_triangle() {
    env_logger::try_init().ok();

    let sensor = GeometrySensorCpu::new();

    // Triangle graph: 3 vertices, fully connected
    let positions = vec![0.0, 0.0, 1.0, 0.0, 0.5, 0.866];
    let adjacency = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
    let anchors = vec![0];

    let metrics = sensor
        .compute_metrics(&positions, 3, &adjacency, &anchors)
        .expect("CPU sensor failed");

    // Validate basic properties
    assert!(!metrics.used_gpu);
    assert_eq!(metrics.bounding_box.min_x, 0.0);
    assert_eq!(metrics.bounding_box.max_x, 1.0);
    assert!(metrics.bounding_box.area > 0.0);
    assert!(metrics.mean_overlap_density >= 0.0);
    assert!(metrics.computation_time_ms > 0.0);

    println!("Triangle CPU metrics: {:#?}", metrics);
}

#[test]
fn test_cpu_sensor_petersen() {
    env_logger::try_init().ok();

    let sensor = GeometrySensorCpu::new();

    // Petersen graph (10 vertices)
    let adjacency = vec![
        vec![1, 4, 5],
        vec![0, 2, 6],
        vec![1, 3, 7],
        vec![2, 4, 8],
        vec![0, 3, 9],
        vec![0, 7, 8],
        vec![1, 8, 9],
        vec![2, 5, 9],
        vec![3, 5, 6],
        vec![4, 6, 7],
    ];

    // Use spring layout for realistic positions
    let positions = generate_spring_layout(10, &adjacency, 50, 42);
    let anchors = vec![0, 5]; // Two structural anchors

    let metrics = sensor
        .compute_metrics(&positions, 10, &adjacency, &anchors)
        .expect("CPU sensor failed");

    assert!(!metrics.used_gpu);
    assert_eq!(positions.len(), 20);
    assert!(metrics.mean_overlap_density >= 0.0);
    assert!(metrics.max_overlap_density >= metrics.mean_overlap_density);

    println!("Petersen CPU metrics: {:#?}", metrics);
}

#[test]
fn test_circular_layout() {
    let positions = generate_circular_layout(6);
    assert_eq!(positions.len(), 12);

    // Check all vertices lie roughly on circle
    let center_x = 0.5;
    let center_y = 0.5;
    let expected_radius = 0.5;

    for i in 0..6 {
        let x = positions[i * 2];
        let y = positions[i * 2 + 1];
        let radius = ((x - center_x).powi(2) + (y - center_y).powi(2)).sqrt();
        assert!(
            (radius - expected_radius).abs() < 0.01,
            "Vertex {} at radius {:.3}, expected {:.3}",
            i,
            radius,
            expected_radius
        );
    }
}

#[test]
fn test_random_layout_determinism() {
    let positions1 = generate_random_layout(20, 12345);
    let positions2 = generate_random_layout(20, 12345);
    let positions3 = generate_random_layout(20, 54321);

    // Same seed -> same positions
    assert_eq!(positions1, positions2);

    // Different seed -> different positions
    assert_ne!(positions1, positions3);

    // All positions in [0, 1]
    for &pos in &positions1 {
        assert!(pos >= 0.0 && pos <= 1.0);
    }
}

#[test]
fn test_spring_layout_convergence() {
    // Complete graph K5
    let adjacency = vec![
        vec![1, 2, 3, 4],
        vec![0, 2, 3, 4],
        vec![0, 1, 3, 4],
        vec![0, 1, 2, 4],
        vec![0, 1, 2, 3],
    ];

    let positions = generate_spring_layout(5, &adjacency, 100, 42);

    // Verify output size
    assert_eq!(positions.len(), 10);

    // All positions should be in unit square
    for &pos in &positions {
        assert!(pos >= 0.0 && pos <= 1.0, "Position {} out of bounds", pos);
    }

    // Vertices should be roughly evenly distributed (not all clustered)
    let avg_x: f32 = (0..5).map(|i| positions[i * 2]).sum::<f32>() / 5.0;
    let avg_y: f32 = (0..5).map(|i| positions[i * 2 + 1]).sum::<f32>() / 5.0;

    // Center of mass should be near (0.5, 0.5)
    assert!(
        (avg_x - 0.5).abs() < 0.3,
        "Center X={:.3} not near 0.5",
        avg_x
    );
    assert!(
        (avg_y - 0.5).abs() < 0.3,
        "Center Y={:.3} not near 0.5",
        avg_y
    );
}

#[test]
fn test_overlap_density_empty_graph() {
    let sensor = GeometrySensorCpu::new();

    // Single isolated vertex
    let positions = vec![0.5, 0.5];
    let adjacency: Vec<Vec<usize>> = vec![vec![]];
    let anchors = vec![];

    let metrics = sensor
        .compute_metrics(&positions, 1, &adjacency, &anchors)
        .expect("CPU sensor failed");

    assert_eq!(metrics.mean_overlap_density, 0.0);
    assert_eq!(metrics.max_overlap_density, 0.0);
    assert_eq!(metrics.num_hotspots, 0);
}

#[test]
fn test_bounding_box_correctness() {
    let sensor = GeometrySensorCpu::new();

    // Vertices at corners of rectangle
    let positions = vec![0.0, 0.0, 3.0, 0.0, 3.0, 2.0, 0.0, 2.0];
    let adjacency = vec![vec![1], vec![0, 2], vec![1, 3], vec![2]];
    let anchors = vec![0, 2];

    let metrics = sensor
        .compute_metrics(&positions, 4, &adjacency, &anchors)
        .expect("CPU sensor failed");

    assert_eq!(metrics.bounding_box.min_x, 0.0);
    assert_eq!(metrics.bounding_box.max_x, 3.0);
    assert_eq!(metrics.bounding_box.min_y, 0.0);
    assert_eq!(metrics.bounding_box.max_y, 2.0);
    assert_eq!(metrics.bounding_box.area, 6.0);
}

#[test]
fn test_geometry_stress_overhead() {
    use std::time::Instant;

    let sensor = GeometrySensorCpu::new();

    // Medium-sized graph (100 vertices)
    let n = 100;
    let adjacency: Vec<Vec<usize>> = (0..n)
        .map(|i| {
            // Each vertex connected to next 3 vertices (ring + chords)
            vec![(i + 1) % n, (i + 2) % n, (i + 3) % n]
        })
        .collect();

    let positions = generate_circular_layout(n);
    let anchors: Vec<usize> = (0..n).step_by(10).collect(); // 10 anchors

    let start = Instant::now();
    let metrics = sensor
        .compute_metrics(&positions, n, &adjacency, &anchors)
        .expect("CPU sensor failed");
    let elapsed = start.elapsed();

    println!(
        "Geometry stress analysis for {} vertices: {:.2}ms",
        n,
        elapsed.as_secs_f64() * 1000.0
    );

    // Should be fast (<50ms for 100 vertices on CPU)
    assert!(elapsed.as_secs_f64() < 0.05, "Analysis took too long");

    // Verify results are reasonable
    assert!(metrics.mean_overlap_density >= 0.0);
    assert!(metrics.bounding_box.area > 0.0);

    // Overhead should be <5% of typical phase time
    // Typical phase: 1-10 seconds
    // Geometry analysis: <50ms
    // Overhead: <0.5%
    println!("Overhead: {:.2}%", (elapsed.as_secs_f64() / 1.0) * 100.0);
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_sensor_basic() {
    use cudarc::driver::CudaContext;
    use prism_geometry::GeometrySensorLayer;
    use std::sync::Arc;

    env_logger::try_init().ok();

    let device = CudaContext::new(0).expect("CUDA not available");
    let sensor = GeometrySensorLayer::new(device, "target/ptx/stress_analysis.ptx")
        .expect("GPU init failed");

    // Triangle graph
    let positions = vec![0.0, 0.0, 1.0, 0.0, 0.5, 0.866];
    let adjacency = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
    let anchors = vec![0];

    let gpu_metrics = sensor
        .compute_metrics(&positions, 3, &adjacency, &anchors)
        .expect("GPU sensor failed");

    assert!(gpu_metrics.used_gpu);
    assert!(gpu_metrics.bounding_box.area > 0.0);

    println!("Triangle GPU metrics: {:#?}", gpu_metrics);
}

#[test]
#[ignore] // Requires CUDA GPU
fn test_gpu_vs_cpu_equivalence() {
    use cudarc::driver::CudaContext;
    use prism_geometry::GeometrySensorLayer;
    use std::sync::Arc;

    env_logger::try_init().ok();

    let device = CudaContext::new(0).expect("CUDA not available");
    let gpu_sensor = GeometrySensorLayer::new(device, "target/ptx/stress_analysis.ptx")
        .expect("GPU init failed");
    let cpu_sensor = GeometrySensorCpu::new();

    // Test on Petersen graph
    let adjacency = vec![
        vec![1, 4, 5],
        vec![0, 2, 6],
        vec![1, 3, 7],
        vec![2, 4, 8],
        vec![0, 3, 9],
        vec![0, 7, 8],
        vec![1, 8, 9],
        vec![2, 5, 9],
        vec![3, 5, 6],
        vec![4, 6, 7],
    ];

    let positions = generate_circular_layout(10);
    let anchors = vec![0, 5];

    let gpu_metrics = gpu_sensor
        .compute_metrics(&positions, 10, &adjacency, &anchors)
        .expect("GPU sensor failed");

    let cpu_metrics = cpu_sensor
        .compute_metrics(&positions, 10, &adjacency, &anchors)
        .expect("CPU sensor failed");

    // Compare results (allow small numerical differences)
    let tolerance = 1e-5;

    assert!(
        (gpu_metrics.bounding_box.area - cpu_metrics.bounding_box.area).abs() < tolerance,
        "Bounding box area mismatch: GPU={:.6}, CPU={:.6}",
        gpu_metrics.bounding_box.area,
        cpu_metrics.bounding_box.area
    );

    assert!(
        (gpu_metrics.mean_overlap_density - cpu_metrics.mean_overlap_density).abs()
            < tolerance as f64,
        "Overlap density mismatch: GPU={:.6}, CPU={:.6}",
        gpu_metrics.mean_overlap_density,
        cpu_metrics.mean_overlap_density
    );

    println!("GPU vs CPU test passed!");
    println!("GPU time: {:.2}ms", gpu_metrics.computation_time_ms);
    println!("CPU time: {:.2}ms", cpu_metrics.computation_time_ms);

    // GPU should be faster for large graphs (but overhead may dominate for small graphs)
    if gpu_metrics.computation_time_ms < cpu_metrics.computation_time_ms {
        println!(
            "GPU speedup: {:.2}x",
            cpu_metrics.computation_time_ms / gpu_metrics.computation_time_ms
        );
    }
}

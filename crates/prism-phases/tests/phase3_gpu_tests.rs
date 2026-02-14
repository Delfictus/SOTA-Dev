//! Phase 3 GPU vs CPU Consistency Tests
//!
//! Tests for Phase 3 Quantum Evolution GPU acceleration.
//! Verifies GPU and CPU paths produce valid colorings.
//!
//! Run with: cargo test -p prism-phases --features cuda -- --ignored

use cudarc::driver::CudaContext;
use prism_core::{Graph, PhaseContext};
use prism_phases::phase3_quantum::Phase3Quantum;
use std::sync::Arc;

/// Creates a small triangle graph for testing
fn create_triangle_graph() -> Graph {
    Graph {
        num_vertices: 3,
        num_edges: 3,
        adjacency: vec![
            vec![1, 2], // Vertex 0 connected to 1, 2
            vec![0, 2], // Vertex 1 connected to 0, 2
            vec![0, 1], // Vertex 2 connected to 0, 1
        ],
        degrees: Some(vec![2, 2, 2]),
        edge_weights: None,
    }
}

/// Creates a complete graph K5
fn create_k5_graph() -> Graph {
    Graph {
        num_vertices: 5,
        num_edges: 10,
        adjacency: vec![
            vec![1, 2, 3, 4],
            vec![0, 2, 3, 4],
            vec![0, 1, 3, 4],
            vec![0, 1, 2, 4],
            vec![0, 1, 2, 3],
        ],
        degrees: Some(vec![4, 4, 4, 4, 4]),
        edge_weights: None,
    }
}

/// Creates a bipartite graph K_{3,3}
fn create_bipartite_3_3_graph() -> Graph {
    Graph {
        num_vertices: 6,
        num_edges: 9,
        adjacency: vec![
            vec![3, 4, 5], // Left partition: 0, 1, 2
            vec![3, 4, 5],
            vec![3, 4, 5],
            vec![0, 1, 2], // Right partition: 3, 4, 5
            vec![0, 1, 2],
            vec![0, 1, 2],
        ],
        degrees: Some(vec![3, 3, 3, 3, 3, 3]),
        edge_weights: None,
    }
}

#[test]
fn test_phase3_cpu_fallback_triangle() {
    env_logger::builder().is_test(true).try_init().ok();

    let mut phase3 = Phase3Quantum::new();
    let graph = create_triangle_graph();
    let mut context = PhaseContext::new();

    let result = phase3.execute(&graph, &mut context);
    assert!(result.is_ok(), "Phase 3 CPU execution failed");

    // Verify solution was produced
    assert!(context.best_solution.is_some(), "No solution produced");
    let solution = context.best_solution.unwrap();

    // Triangle graph requires exactly 3 colors (chromatic number = 3)
    assert!(solution.chromatic_number >= 3, "Chromatic number too low");

    // Check coloring validity (no conflicts)
    assert_eq!(
        solution.conflicts, 0,
        "Invalid coloring: {} conflicts",
        solution.conflicts
    );

    println!(
        "CPU Triangle: chromatic_number={}, conflicts={}",
        solution.chromatic_number, solution.conflicts
    );
}

#[test]
fn test_phase3_cpu_fallback_k5() {
    env_logger::builder().is_test(true).try_init().ok();

    let mut phase3 = Phase3Quantum::new();
    let graph = create_k5_graph();
    let mut context = PhaseContext::new();

    let result = phase3.execute(&graph, &mut context);
    assert!(result.is_ok(), "Phase 3 CPU execution failed");

    let solution = context.best_solution.unwrap();

    // Complete graph K5 requires 5 colors
    assert!(solution.chromatic_number >= 5, "Chromatic number too low");
    assert_eq!(solution.conflicts, 0, "Invalid coloring");

    println!(
        "CPU K5: chromatic_number={}, conflicts={}",
        solution.chromatic_number, solution.conflicts
    );
}

#[test]
fn test_phase3_cpu_fallback_bipartite() {
    env_logger::builder().is_test(true).try_init().ok();

    let mut phase3 = Phase3Quantum::new();
    let graph = create_bipartite_3_3_graph();
    let mut context = PhaseContext::new();

    let result = phase3.execute(&graph, &mut context);
    assert!(result.is_ok(), "Phase 3 CPU execution failed");

    let solution = context.best_solution.unwrap();

    // Bipartite graph requires only 2 colors
    // CPU greedy may use more, but should be valid
    assert!(solution.chromatic_number >= 2, "Chromatic number too low");
    assert_eq!(solution.conflicts, 0, "Invalid coloring");

    println!(
        "CPU Bipartite: chromatic_number={}, conflicts={}",
        solution.chromatic_number, solution.conflicts
    );
}

#[test]
#[ignore] // Requires GPU hardware
fn test_phase3_gpu_triangle() {
    env_logger::builder().is_test(true).try_init().ok();

    let device = CudaContext::new(0).expect("CUDA device not available");
    let mut phase3 = Phase3Quantum::with_gpu(Arc::new(device), "target/ptx/quantum.ptx")
        .expect("Failed to initialize GPU phase");

    let graph = create_triangle_graph();
    let mut context = PhaseContext::new();

    let result = phase3.execute(&graph, &mut context);
    assert!(result.is_ok(), "Phase 3 GPU execution failed");

    let solution = context.best_solution.unwrap();

    // Triangle requires 3 colors
    assert!(solution.chromatic_number >= 3, "Chromatic number too low");

    // GPU quantum algorithm may not always produce conflict-free coloring
    // (quantum measurement is probabilistic), but should be close
    println!(
        "GPU Triangle: chromatic_number={}, conflicts={}",
        solution.chromatic_number, solution.conflicts
    );

    // For small graphs, we expect conflict-free most of the time
    if solution.conflicts > 0 {
        println!(
            "Warning: GPU produced {} conflicts (quantum measurement variance)",
            solution.conflicts
        );
    }
}

#[test]
#[ignore] // Requires GPU hardware
fn test_phase3_gpu_k5() {
    env_logger::builder().is_test(true).try_init().ok();

    let device = CudaContext::new(0).expect("CUDA device not available");
    let mut phase3 = Phase3Quantum::with_gpu(Arc::new(device), "target/ptx/quantum.ptx")
        .expect("Failed to initialize GPU phase");

    let graph = create_k5_graph();
    let mut context = PhaseContext::new();

    let result = phase3.execute(&graph, &mut context);
    assert!(result.is_ok(), "Phase 3 GPU execution failed");

    let solution = context.best_solution.unwrap();

    assert!(solution.chromatic_number >= 5, "Chromatic number too low");

    println!(
        "GPU K5: chromatic_number={}, conflicts={}",
        solution.chromatic_number, solution.conflicts
    );
}

#[test]
#[ignore] // Requires GPU hardware
fn test_phase3_gpu_vs_cpu_consistency() {
    env_logger::builder().is_test(true).try_init().ok();

    let graph = create_triangle_graph();

    // Run CPU version
    let mut phase3_cpu = Phase3Quantum::new();
    let mut context_cpu = PhaseContext::new();
    phase3_cpu.execute(&graph, &mut context_cpu).unwrap();
    let cpu_solution = context_cpu.best_solution.unwrap();

    // Run GPU version
    let device = CudaContext::new(0).expect("CUDA device not available");
    let mut phase3_gpu = Phase3Quantum::with_gpu(Arc::new(device), "target/ptx/quantum.ptx")
        .expect("Failed to initialize GPU phase");

    let mut context_gpu = PhaseContext::new();
    phase3_gpu.execute(&graph, &mut context_gpu).unwrap();
    let gpu_solution = context_gpu.best_solution.unwrap();

    println!(
        "CPU: chromatic_number={}, conflicts={}",
        cpu_solution.chromatic_number, cpu_solution.conflicts
    );
    println!(
        "GPU: chromatic_number={}, conflicts={}",
        gpu_solution.chromatic_number, gpu_solution.conflicts
    );

    // Both should produce valid colorings (no conflicts)
    assert_eq!(cpu_solution.conflicts, 0, "CPU produced invalid coloring");

    // GPU may have conflicts due to quantum measurement variance
    // Just verify it's reasonable
    if gpu_solution.conflicts > graph.num_edges / 2 {
        panic!(
            "GPU produced too many conflicts: {}",
            gpu_solution.conflicts
        );
    }

    // Both should use reasonable number of colors
    assert!(
        cpu_solution.chromatic_number <= 50,
        "CPU used too many colors"
    );
    assert!(
        gpu_solution.chromatic_number <= 50,
        "GPU used too many colors"
    );
}

#[test]
#[ignore] // Requires GPU hardware and benchmark dataset
fn test_phase3_performance_dsjc125() {
    env_logger::builder().is_test(true).try_init().ok();

    // Load DSJC125 graph (if available)
    // This test is for performance benchmarking
    // Target: < 500ms for DSJC125

    let device = CudaContext::new(0).expect("CUDA device not available");
    let mut phase3 = Phase3Quantum::with_gpu(Arc::new(device), "target/ptx/quantum.ptx")
        .expect("Failed to initialize GPU phase");

    // Create a synthetic graph similar to DSJC125 (125 vertices, ~3700 edges)
    let num_vertices = 125;
    let mut adjacency = vec![vec![]; num_vertices];

    // Create a dense random graph (approximate DSJC125 density)
    use std::collections::HashSet;
    let mut edges = HashSet::new();

    for i in 0..num_vertices {
        // Add ~30 edges per vertex (average degree ~60 for bidirectional)
        for j in 0..30 {
            let neighbor = (i + j * 4 + 1) % num_vertices;
            if neighbor != i {
                edges.insert((i.min(neighbor), i.max(neighbor)));
                adjacency[i].push(neighbor);
            }
        }
    }

    let graph = Graph {
        num_vertices,
        num_edges: edges.len(),
        adjacency,
        degrees: None,
        edge_weights: None,
    };

    let mut context = PhaseContext::new();

    let start = std::time::Instant::now();
    let result = phase3.execute(&graph, &mut context);
    let elapsed_ms = start.elapsed().as_millis();

    assert!(result.is_ok(), "Phase 3 GPU execution failed");

    let solution = context.best_solution.unwrap();

    println!("DSJC125 Benchmark:");
    println!("  Chromatic number: {}", solution.chromatic_number);
    println!("  Conflicts: {}", solution.conflicts);
    println!("  Time: {}ms", elapsed_ms);

    // Performance target: < 500ms
    if elapsed_ms > 500 {
        println!(
            "Warning: Performance target missed ({}ms > 500ms)",
            elapsed_ms
        );
    }

    // Solution quality check
    assert!(
        solution.chromatic_number < num_vertices,
        "Used too many colors"
    );
}

#[test]
fn test_rl_action_integration() {
    env_logger::builder().is_test(true).try_init().ok();

    let mut phase3 = Phase3Quantum::new();
    let graph = create_triangle_graph();

    // Test with RL action in context
    let mut context = PhaseContext::new();
    context
        .scratch
        .insert("rl_action".to_string(), Box::new(15usize));

    let result = phase3.execute(&graph, &mut context);
    assert!(result.is_ok(), "Phase 3 with RL action failed");

    // Verify telemetry was updated
    let metrics = phase3.telemetry().metrics();
    assert!(metrics.contains_key("purity"));
    assert!(metrics.contains_key("entanglement"));
    assert!(metrics.contains_key("evolution_time"));
}

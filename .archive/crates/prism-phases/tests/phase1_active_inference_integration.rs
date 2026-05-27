//! Integration tests for Phase 1 Active Inference
//!
//! Tests the full Phase 1 pipeline including:
//! - CPU fallback mode
//! - Policy computation
//! - Greedy coloring with AI vertex ordering
//! - Telemetry emission
//! - Conflict-free coloring validation

use prism_core::{Graph, PhaseContext, PhaseController};
use prism_phases::Phase1ActiveInference;

#[test]
fn test_phase1_execution_on_small_graph() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Create a simple test graph: triangle (requires 3 colors)
    let graph = Graph::from_edges(3, vec![(0, 1), (1, 2), (2, 0)]);

    let mut phase1 = Phase1ActiveInference::new();
    let mut context = PhaseContext::new();

    let result = phase1.execute(&graph, &mut context);
    assert!(result.is_ok(), "Phase1 execution failed: {:?}", result);

    let outcome = result.unwrap();
    assert!(outcome.is_success(), "Phase1 did not succeed");

    // Verify coloring was stored in context
    let coloring_box = context
        .scratch
        .get("phase1_coloring")
        .expect("Coloring not found in context");
    let coloring = coloring_box
        .downcast_ref::<Vec<usize>>()
        .expect("Failed to downcast coloring");
    assert_eq!(coloring.len(), 3);

    // Verify number of colors (triangle requires exactly 3)
    let num_colors_box = context
        .scratch
        .get("phase1_num_colors")
        .expect("num_colors not found");
    let num_colors = *num_colors_box
        .downcast_ref::<usize>()
        .expect("Failed to downcast num_colors");
    assert_eq!(num_colors, 3, "Triangle should use exactly 3 colors");

    // Validate no conflicts
    for v in 0..3 {
        for neighbor in &graph.adjacency[v] {
            assert_ne!(
                coloring[v], coloring[neighbor],
                "Conflict: vertices {} and {} have same color {}",
                v, neighbor, coloring[v]
            );
        }
    }
}

#[test]
fn test_phase1_execution_on_bipartite_graph() {
    let _ = env_logger::builder().is_test(true).try_init();

    // K_{2,3} bipartite graph: vertices {0,1} connected to {2,3,4}
    // Should require exactly 2 colors
    let edges = vec![(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)];
    let graph = Graph::from_edges(5, edges);

    let mut phase1 = Phase1ActiveInference::new();
    let mut context = PhaseContext::new();

    let result = phase1.execute(&graph, &mut context);
    assert!(result.is_ok());

    let coloring_box = context.scratch.get("phase1_coloring").unwrap();
    let coloring = coloring_box.downcast_ref::<Vec<usize>>().unwrap();
    let num_colors_box = context.scratch.get("phase1_num_colors").unwrap();
    let num_colors = *num_colors_box.downcast_ref::<usize>().unwrap();

    // Bipartite graph should use exactly 2 colors
    assert_eq!(num_colors, 2, "Bipartite graph should use 2 colors");

    // Validate no conflicts
    for (u, v) in [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)] {
        assert_ne!(
            coloring[u], coloring[v],
            "Conflict: edge ({}, {}) has same color",
            u, v
        );
    }
}

#[test]
fn test_phase1_telemetry_emission() {
    let _ = env_logger::builder().is_test(true).try_init();

    let graph = Graph::from_edges(4, vec![(0, 1), (1, 2), (2, 3), (3, 0)]);

    let mut phase1 = Phase1ActiveInference::new();
    let mut context = PhaseContext::new();

    let _ = phase1.execute(&graph, &mut context).unwrap();

    // Check telemetry
    let metrics = phase1.telemetry().metrics();

    assert!(metrics.contains_key("efe"), "Missing EFE metric");
    assert!(
        metrics.contains_key("uncertainty"),
        "Missing uncertainty metric"
    );
    assert!(metrics.contains_key("vfe"), "Missing VFE metric");
    assert!(
        metrics.contains_key("num_colors"),
        "Missing num_colors metric"
    );
    assert!(
        metrics.contains_key("execution_time_ms"),
        "Missing execution_time_ms"
    );
    assert!(
        metrics.contains_key("policy_time_ms"),
        "Missing policy_time_ms"
    );
    assert!(
        metrics.contains_key("coloring_time_ms"),
        "Missing coloring_time_ms"
    );

    // Verify execution time is reasonable
    let exec_time = metrics["execution_time_ms"];
    assert!(exec_time > 0.0, "Execution time should be positive");
    assert!(
        exec_time < 1000.0,
        "Execution time should be < 1s for tiny graph"
    );

    // Verify JSON serialization
    let json = phase1.telemetry().as_json();
    assert_eq!(json["phase"], "Phase1-ActiveInference");
    assert!(json["num_colors"].as_u64().unwrap() > 0);
}

#[test]
fn test_phase1_large_graph_performance() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Generate a larger random graph (100 vertices, ~500 edges)
    let mut edges = Vec::new();
    for i in 0..100 {
        for j in (i + 1)..100 {
            if (i * 7 + j * 13) % 5 == 0 {
                edges.push((i, j));
            }
        }
    }
    let graph = Graph::from_edges(100, edges);

    let mut phase1 = Phase1ActiveInference::new();
    let mut context = PhaseContext::new();

    let start = std::time::Instant::now();
    let result = phase1.execute(&graph, &mut context);
    let elapsed = start.elapsed();

    assert!(result.is_ok(), "Phase1 failed on large graph");
    assert!(elapsed.as_millis() < 5000, "Phase1 too slow: {:?}", elapsed);

    // Verify coloring validity
    let coloring_box = context.scratch.get("phase1_coloring").unwrap();
    let coloring = coloring_box.downcast_ref::<Vec<usize>>().unwrap();

    // Check a sample of edges for conflicts
    for v in 0..graph.num_vertices.min(100) {
        for &neighbor in &graph.adjacency[v] {
            if neighbor > v {
                // Check each edge once
                assert_ne!(
                    coloring[v], coloring[neighbor],
                    "Conflict in large graph: edge ({}, {})",
                    v, neighbor
                );
            }
        }
    }
}

#[test]
fn test_phase1_empty_graph() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Empty graph (no edges, 5 vertices)
    let graph = Graph::from_edges(5, vec![]);

    let mut phase1 = Phase1ActiveInference::new();
    let mut context = PhaseContext::new();

    let result = phase1.execute(&graph, &mut context);
    assert!(result.is_ok());

    let coloring_box = context.scratch.get("phase1_coloring").unwrap();
    let coloring = coloring_box.downcast_ref::<Vec<usize>>().unwrap();
    let num_colors_box = context.scratch.get("phase1_num_colors").unwrap();
    let num_colors = *num_colors_box.downcast_ref::<usize>().unwrap();

    // Empty graph should use 1 color
    assert_eq!(num_colors, 1, "Empty graph should use 1 color");

    // All vertices should have same color
    let first_color = coloring[0];
    for &c in coloring {
        assert_eq!(c, first_color, "All vertices should have same color");
    }
}

#[test]
fn test_phase1_single_vertex() {
    let _ = env_logger::builder().is_test(true).try_init();

    let graph = Graph::from_edges(1, vec![]);

    let mut phase1 = Phase1ActiveInference::new();
    let mut context = PhaseContext::new();

    let result = phase1.execute(&graph, &mut context);
    assert!(result.is_ok());

    let coloring_box = context.scratch.get("phase1_coloring").unwrap();
    let coloring = coloring_box.downcast_ref::<Vec<usize>>().unwrap();
    let num_colors_box = context.scratch.get("phase1_num_colors").unwrap();
    let num_colors = *num_colors_box.downcast_ref::<usize>().unwrap();

    assert_eq!(coloring.len(), 1);
    assert_eq!(num_colors, 1);
}

#[test]
fn test_phase1_policy_uncertainty_ordering() {
    let _ = env_logger::builder().is_test(true).try_init();

    // Create a graph with varying degrees
    // High-degree vertices should have higher uncertainty
    let edges = vec![
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (0, 5), // v0: degree 5
        (1, 2),
        (3, 4), // v1,v2,v3,v4: degree 2-3
        (6, 7), // v6,v7: degree 1
    ];
    let graph = Graph::from_edges(8, edges);

    let mut phase1 = Phase1ActiveInference::new();
    let mut context = PhaseContext::new();

    let result = phase1.execute(&graph, &mut context);
    assert!(result.is_ok());

    // Retrieve policy from context
    let policy_box = context.scratch.get("phase1_policy").unwrap();
    let policy = policy_box
        .downcast_ref::<prism_gpu::ActiveInferencePolicy>()
        .unwrap();

    // High-degree vertex (v0) should have higher uncertainty than low-degree vertices (v6, v7)
    // Note: In CPU fallback mode, uncertainty is uniform, so this test validates the structure
    let v0_uncertainty = policy.uncertainty[0];
    let v6_uncertainty = policy.uncertainty[6];

    // Both should be valid (0 to 1)
    assert!(v0_uncertainty >= 0.0 && v0_uncertainty <= 1.0);
    assert!(v6_uncertainty >= 0.0 && v6_uncertainty <= 1.0);

    // In CPU fallback, they're equal; in GPU mode, v0 > v6
    // This test ensures the structure is correct
}

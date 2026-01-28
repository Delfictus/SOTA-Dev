//! Phase 2 GPU smoke tests.
//!
//! Tests Phase 2 thermodynamic simulated annealing with GPU acceleration.

#[cfg(all(test, feature = "cuda"))]
mod gpu_tests {
    use cudarc::driver::CudaContext;
    use prism_core::{Graph, PhaseContext, PhaseController};
    use prism_phases::Phase2Thermodynamic;
    use std::sync::Arc;

    /// Creates Petersen graph for testing (10 vertices, chromatic number = 3).
    fn create_petersen_graph() -> Graph {
        Graph {
            num_vertices: 10,
            num_edges: 15,
            adjacency: vec![
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
            ],
            degrees: None,
            edge_weights: None,
        }
    }

    /// Creates a triangle graph for quick testing (3 vertices, chromatic number = 3).
    fn create_triangle_graph() -> Graph {
        Graph {
            num_vertices: 3,
            num_edges: 3,
            adjacency: vec![vec![1, 2], vec![0, 2], vec![0, 1]],
            degrees: None,
            edge_weights: None,
        }
    }

    #[test]
    fn test_phase2_gpu_initialization() {
        env_logger::try_init().ok();

        let device = CudaContext::new(0).expect("CUDA device not available");

        // Try to initialize Phase 2 with GPU
        let result =
            Phase2Thermodynamic::new_with_gpu(Arc::new(device), "target/ptx/thermodynamic.ptx");

        assert!(
            result.is_ok(),
            "Phase2 GPU initialization failed: {:?}",
            result.err()
        );

        log::info!("Phase 2 GPU initialization successful");
    }

    #[test]
    fn test_phase2_gpu_triangle() {
        env_logger::try_init().ok();

        let device = CudaContext::new(0).expect("CUDA device not available");
        let mut phase2 =
            Phase2Thermodynamic::new_with_gpu(Arc::new(device), "target/ptx/thermodynamic.ptx")
                .expect("GPU init failed");

        let graph = create_triangle_graph();
        let mut context = PhaseContext::new();

        let result = phase2.execute(&graph, &mut context);

        assert!(result.is_ok(), "Phase 2 execution failed: {:?}", result);

        let solution = context
            .best_solution
            .expect("No solution produced by Phase 2");

        // Verify solution validity
        assert_eq!(
            solution.conflicts, 0,
            "Phase 2 produced coloring with conflicts"
        );
        assert!(
            solution.chromatic_number <= 5,
            "Chromatic number too high: {}",
            solution.chromatic_number
        );

        log::info!(
            "Triangle test passed: {} colors, {} conflicts",
            solution.chromatic_number,
            solution.conflicts
        );
    }

    #[test]
    fn test_phase2_gpu_petersen() {
        env_logger::try_init().ok();

        let device = CudaContext::new(0).expect("CUDA device not available");
        let mut phase2 =
            Phase2Thermodynamic::new_with_gpu(Arc::new(device), "target/ptx/thermodynamic.ptx")
                .expect("GPU init failed");

        let graph = create_petersen_graph();
        let mut context = PhaseContext::new();

        let result = phase2.execute(&graph, &mut context);

        assert!(result.is_ok(), "Phase 2 execution failed: {:?}", result);

        let solution = context
            .best_solution
            .expect("No solution produced by Phase 2");

        // Verify solution validity
        assert_eq!(
            solution.conflicts, 0,
            "Phase 2 produced coloring with conflicts"
        );

        // Petersen graph has chromatic number 3, but greedy/annealing might use 4-5
        assert!(
            solution.chromatic_number <= 7,
            "Chromatic number too high: {} (Petersen optimal: 3)",
            solution.chromatic_number
        );

        log::info!(
            "Petersen test passed: {} colors, {} conflicts",
            solution.chromatic_number,
            solution.conflicts
        );
    }

    #[test]
    fn test_phase2_gpu_with_warmstart() {
        env_logger::try_init().ok();

        let device = CudaContext::new(0).expect("CUDA device not available");
        let mut phase2 =
            Phase2Thermodynamic::new_with_gpu(Arc::new(device), "target/ptx/thermodynamic.ptx")
                .expect("GPU init failed");

        let graph = create_petersen_graph();

        // Create a warmstart solution (greedy coloring)
        use prism_core::ColoringSolution;
        let mut warmstart_solution = ColoringSolution::new(graph.num_vertices);
        warmstart_solution.colors = vec![1, 2, 3, 1, 2, 2, 3, 1, 1, 2]; // Valid coloring
        warmstart_solution.compute_chromatic_number();
        warmstart_solution.conflicts = warmstart_solution.validate(&graph);

        assert_eq!(
            warmstart_solution.conflicts, 0,
            "Warmstart solution should be valid"
        );

        let initial_chromatic = warmstart_solution.chromatic_number;

        // Execute Phase 2 with warmstart
        let mut context = PhaseContext::new();
        context.update_best_solution(warmstart_solution);

        let result = phase2.execute(&graph, &mut context);

        assert!(result.is_ok(), "Phase 2 execution failed: {:?}", result);

        let final_solution = context.best_solution.expect("No solution after Phase 2");

        assert_eq!(
            final_solution.conflicts, 0,
            "Phase 2 produced coloring with conflicts"
        );

        // Warmstart should produce same or better result
        assert!(
            final_solution.chromatic_number <= initial_chromatic,
            "Phase 2 with warmstart degraded quality: {} -> {}",
            initial_chromatic,
            final_solution.chromatic_number
        );

        log::info!(
            "Warmstart test passed: {} -> {} colors",
            initial_chromatic,
            final_solution.chromatic_number
        );
    }

    #[test]
    fn test_phase2_gpu_rl_action_application() {
        env_logger::try_init().ok();

        let device = CudaContext::new(0).expect("CUDA device not available");
        let mut phase2 =
            Phase2Thermodynamic::new_with_gpu(Arc::new(device), "target/ptx/thermodynamic.ptx")
                .expect("GPU init failed");

        let graph = create_triangle_graph();

        // Test RL action application (index 16 = IncreaseTemperature)
        let mut context = PhaseContext::new();
        context.scratch.insert(
            "rl_action".to_string(),
            Box::new(16usize), // Phase2 IncreaseTemperature
        );

        let telemetry_before = phase2.telemetry();
        let temp_max_before = telemetry_before.metrics()["temp_max"];

        let result = phase2.execute(&graph, &mut context);
        assert!(result.is_ok(), "Phase 2 execution failed: {:?}", result);

        let telemetry_after = phase2.telemetry();
        let temp_max_after = telemetry_after.metrics()["temp_max"];

        // IncreaseTemperature should have increased temp_max
        assert!(
            temp_max_after > temp_max_before,
            "RL action did not increase temperature: {} -> {}",
            temp_max_before,
            temp_max_after
        );

        log::info!(
            "RL action test passed: temp_max {} -> {}",
            temp_max_before,
            temp_max_after
        );
    }
}

#[cfg(not(all(test, feature = "cuda")))]
#[test]
fn test_phase2_gpu_requires_cuda_feature() {
    println!("Phase 2 GPU tests require 'cuda' feature to be enabled");
    println!("Run: cargo test --features cuda");
}

//! Phase 4 GPU integration tests
//!
//! Validates end-to-end Phase 4 execution with GPU acceleration.
//!
//! TESTING STRATEGY:
//! - Create test graphs with known properties
//! - Execute Phase 4 with GPU enabled
//! - Verify correctness of APSP results
//! - Compare GPU vs CPU execution times
//! - Validate graceful fallback when GPU unavailable
//!
//! REFERENCE: PRISM GPU Plan §4.4 (Phase 4 Integration Testing)

use prism_core::{Graph, PhaseContext, PhaseController};
use prism_phases::phase4_geodesic::Phase4Geodesic;

/// Creates a small test graph for integration testing
///
/// Graph: 0 <-> 1 <-> 2 <-> 3
///        |                  |
///        +------------------+
fn create_test_graph() -> Graph {
    let adjacency = vec![
        vec![1, 3], // 0 -> 1, 3
        vec![0, 2], // 1 -> 0, 2
        vec![1, 3], // 2 -> 1, 3
        vec![0, 2], // 3 -> 0, 2
    ];

    Graph {
        num_vertices: 4,
        adjacency,
        edge_weights: None,
    }
}

/// TEST: Phase 4 execution with CPU fallback (no GPU)
#[test]
fn test_phase4_cpu_execution() {
    env_logger::builder().is_test(true).try_init().ok();

    let graph = create_test_graph();
    let mut phase = Phase4Geodesic::new(); // CPU-only
    let mut context = PhaseContext::new();

    let result = phase.execute(&graph, &mut context);
    assert!(result.is_ok(), "Phase 4 execution failed: {:?}", result);

    let outcome = result.unwrap();
    assert!(outcome.success, "Phase 4 did not succeed");

    // Verify solution was produced
    assert!(
        context.best_solution.is_some(),
        "No solution produced by Phase 4"
    );

    let solution = context.best_solution.as_ref().unwrap();
    assert_eq!(solution.colors.len(), 4, "Solution size mismatch");

    println!("PASS: Phase 4 CPU execution");
}

/// TEST: Phase 4 execution with GPU enabled
#[test]
#[ignore] // Requires GPU hardware
fn test_phase4_gpu_execution() {
    env_logger::builder().is_test(true).try_init().ok();

    let graph = create_test_graph();
    let mut phase = Phase4Geodesic::new_with_gpu("target/ptx/floyd_warshall.ptx");
    let mut context = PhaseContext::new();

    let result = phase.execute(&graph, &mut context);
    assert!(result.is_ok(), "Phase 4 GPU execution failed: {:?}", result);

    let outcome = result.unwrap();
    assert!(outcome.success, "Phase 4 GPU did not succeed");

    assert!(
        context.best_solution.is_some(),
        "No solution produced by Phase 4 GPU"
    );

    println!("PASS: Phase 4 GPU execution");
}

/// TEST: Compare GPU vs CPU results for equivalence
#[test]
#[ignore]
fn test_gpu_cpu_equivalence() {
    env_logger::builder().is_test(true).try_init().ok();

    let graph = create_test_graph();

    // Execute with CPU
    let mut phase_cpu = Phase4Geodesic::new();
    let mut context_cpu = PhaseContext::new();
    phase_cpu
        .execute(&graph, &mut context_cpu)
        .expect("CPU execution failed");

    // Execute with GPU
    let mut phase_gpu = Phase4Geodesic::new_with_gpu("target/ptx/floyd_warshall.ptx");
    let mut context_gpu = PhaseContext::new();
    phase_gpu
        .execute(&graph, &mut context_gpu)
        .expect("GPU execution failed");

    // Compare solutions
    let solution_cpu = context_cpu.best_solution.as_ref().unwrap();
    let solution_gpu = context_gpu.best_solution.as_ref().unwrap();

    assert_eq!(
        solution_cpu.colors.len(),
        solution_gpu.colors.len(),
        "Solution size mismatch"
    );

    // Chromatic numbers should match (or be within 1 for greedy heuristics)
    let diff = (solution_cpu.chromatic_number as i32 - solution_gpu.chromatic_number as i32).abs();
    assert!(
        diff <= 1,
        "Chromatic number mismatch: CPU={}, GPU={}",
        solution_cpu.chromatic_number,
        solution_gpu.chromatic_number
    );

    println!("PASS: GPU vs CPU equivalence");
}

/// BENCHMARK: Phase 4 performance on medium graph
#[test]
#[ignore]
fn benchmark_phase4_medium_graph() {
    env_logger::builder().is_test(true).try_init().ok();

    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let n = 200;
    let edge_prob = 0.1;
    let mut adjacency = vec![vec![]; n];

    for i in 0..n {
        for j in 0..n {
            if i != j && rng.gen::<f64>() < edge_prob {
                adjacency[i].push(j);
            }
        }
    }

    let graph = Graph {
        num_vertices: n,
        adjacency,
        edge_weights: None,
    };

    // Benchmark CPU
    let mut phase_cpu = Phase4Geodesic::new();
    let mut context_cpu = PhaseContext::new();

    let start = std::time::Instant::now();
    phase_cpu
        .execute(&graph, &mut context_cpu)
        .expect("CPU execution failed");
    let cpu_time = start.elapsed();

    // Benchmark GPU
    let mut phase_gpu = Phase4Geodesic::new_with_gpu("target/ptx/floyd_warshall.ptx");
    let mut context_gpu = PhaseContext::new();

    let start = std::time::Instant::now();
    phase_gpu
        .execute(&graph, &mut context_gpu)
        .expect("GPU execution failed");
    let gpu_time = start.elapsed();

    println!("\n=== BENCHMARK: Phase 4 Medium Graph ===");
    println!("Graph size: {} vertices", n);
    println!("CPU time: {:.3}s", cpu_time.as_secs_f64());
    println!("GPU time: {:.3}s", gpu_time.as_secs_f64());
    println!(
        "Speedup: {:.2}x",
        cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
    );
}

/// TEST: Graceful GPU fallback on error
#[test]
fn test_gpu_fallback_on_invalid_ptx() {
    env_logger::builder().is_test(true).try_init().ok();

    // Provide invalid PTX path - should fall back to CPU
    let mut phase = Phase4Geodesic::new_with_gpu("nonexistent.ptx");
    let graph = create_test_graph();
    let mut context = PhaseContext::new();

    let result = phase.execute(&graph, &mut context);
    assert!(
        result.is_ok(),
        "Phase 4 should fall back to CPU on PTX error"
    );

    println!("PASS: GPU fallback on invalid PTX");
}

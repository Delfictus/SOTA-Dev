//! DIMACS Benchmark Suite for PRISM
//!
//! Comprehensive benchmarking of PRISM's graph coloring performance on
//! standard DIMACS benchmark graphs.
//!
//! ## Benchmarked Graphs
//! - DSJC125.5 (125 vertices, 0.5 density) - Target: ≤17 colors
//! - DSJC250.5 (250 vertices, 0.5 density) - Target: ≤28 colors
//! - DSJC500.5 (500 vertices, 0.5 density) - Target: ≤48 colors
//! - DSJC1000.5 (1000 vertices, 0.5 density) - Target: ≤83 colors
//!
//! ## Usage
//! ```bash
//! cargo bench --bench dimacs
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use prism_core::{dimacs::parse_dimacs_file, ColoringSolution, Graph};
use prism_fluxnet::{RLConfig, UniversalRLController};
use prism_pipeline::{orchestrator::PipelineOrchestrator, PipelineConfig};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// Known chromatic number bounds for DIMACS graphs
struct DimacsTarget {
    name: &'static str,
    file: &'static str,
    lower_bound: usize,
    upper_bound: usize,
    best_known: usize,
}

const DIMACS_TARGETS: &[DimacsTarget] = &[
    DimacsTarget {
        name: "DSJC125.5",
        file: "data/dimacs/DSJC125.5.col",
        lower_bound: 17,
        upper_bound: 20,
        best_known: 17,
    },
    DimacsTarget {
        name: "DSJC250.5",
        file: "data/dimacs/DSJC250.5.col",
        lower_bound: 28,
        upper_bound: 32,
        best_known: 28,
    },
    DimacsTarget {
        name: "DSJC500.5",
        file: "data/dimacs/DSJC500.5.col",
        lower_bound: 48,
        upper_bound: 52,
        best_known: 48,
    },
    DimacsTarget {
        name: "DSJC1000.5",
        file: "data/dimacs/DSJC1000.5.col",
        lower_bound: 83,
        upper_bound: 90,
        best_known: 83,
    },
];

/// Helper to create a basic pipeline configuration
fn create_benchmark_config() -> PipelineConfig {
    PipelineConfig {
        max_vertices: 10000,
        phase_configs: HashMap::new(),
        timeout_seconds: 3600,
        enable_telemetry: false,
        telemetry_path: "benchmark_telemetry.jsonl".to_string(),
        warmstart_config: None,
        gpu: Default::default(),
        phase2: Default::default(),
        memetic: None,
        metaphysical_coupling: None,
        ontology: None,
        mec: None,
        cma_es: None,
        gnn: None,
    }
}

/// Helper to create RL controller for benchmarks
fn create_rl_controller() -> UniversalRLController {
    let config = RLConfig {
        alpha: 0.1,
        gamma: 0.95,
        epsilon: 0.1,
        epsilon_decay: 0.995,
        epsilon_min: 0.05,
        replay_buffer_size: 10000,
        replay_batch_size: 32,
        discretization_mode: prism_fluxnet::DiscretizationMode::Compact,
        reward_log_threshold: 0.001,
    };
    UniversalRLController::new(config)
}

/// Verify that a coloring solution is valid (no conflicts)
fn verify_coloring(graph: &Graph, solution: &ColoringSolution) -> bool {
    if solution.conflicts > 0 {
        eprintln!(
            "⚠ Solution has {} conflicts - INVALID",
            solution.conflicts
        );
        return false;
    }

    // Double-check conflicts by examining each edge
    let mut actual_conflicts = 0;
    for (u, neighbors) in graph.adjacency.iter().enumerate() {
        for &v in neighbors {
            if u < v && solution.colors[u] == solution.colors[v] {
                actual_conflicts += 1;
            }
        }
    }

    if actual_conflicts > 0 {
        eprintln!(
            "⚠ Solution reports 0 conflicts but has {} actual conflicts",
            actual_conflicts
        );
        return false;
    }

    true
}

/// Check solution quality against known bounds
fn check_quality(solution: &ColoringSolution, target: &DimacsTarget) -> String {
    let colors = solution.chromatic_number;

    if colors == target.best_known {
        format!("✓ OPTIMAL ({} colors)", colors)
    } else if colors <= target.upper_bound {
        format!(
            "✓ GOOD ({} colors, best known: {})",
            colors, target.best_known
        )
    } else {
        format!(
            "⚠ SUBOPTIMAL ({} colors, target: ≤{})",
            colors, target.upper_bound
        )
    }
}

/// Benchmark a single DIMACS graph
fn bench_dimacs_graph(c: &mut Criterion, target: &DimacsTarget) {
    let graph_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(target.file);

    // Parse the graph once outside the benchmark
    let graph = match parse_dimacs_file(&graph_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!(
                "⚠ Failed to load {}: {}. Skipping benchmark.",
                target.name, e
            );
            return;
        }
    };

    println!(
        "\n🎯 Benchmarking {} ({} vertices, {} edges)",
        target.name, graph.num_vertices, graph.num_edges
    );

    let mut group = c.benchmark_group(format!("dimacs/{}", target.name));

    // Set timeout for larger graphs
    if graph.num_vertices >= 500 {
        group.measurement_time(Duration::from_secs(120));
        group.sample_size(10);
    } else {
        group.measurement_time(Duration::from_secs(60));
        group.sample_size(20);
    }

    group.bench_function(BenchmarkId::new("full_pipeline", target.name), |b| {
        b.iter(|| {
            let config = create_benchmark_config();
            let rl_controller = create_rl_controller();
            let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

            // Note: GPU context initialization happens automatically in PipelineOrchestrator::new
            // when CUDA feature is enabled

            match orchestrator.run(&graph) {
                Ok(solution) => {
                    assert!(verify_coloring(&graph, &solution), "Invalid coloring");
                    solution
                }
                Err(e) => {
                    panic!("Pipeline failed: {}", e);
                }
            }
        });
    });

    // Run one final iteration to report quality
    let config = create_benchmark_config();
    let rl_controller = create_rl_controller();
    let mut orchestrator = PipelineOrchestrator::new(config, rl_controller);

    if let Ok(solution) = orchestrator.run(&graph) {
        println!("  Result: {}", check_quality(&solution, target));
    }

    group.finish();
}

/// Benchmark all DIMACS graphs
fn bench_dimacs_suite(c: &mut Criterion) {
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║               PRISM DIMACS Benchmark Suite                       ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");

    for target in DIMACS_TARGETS {
        bench_dimacs_graph(c, target);
    }

    println!("\n✓ Benchmark suite complete\n");
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_dimacs_suite
}
criterion_main!(benches);

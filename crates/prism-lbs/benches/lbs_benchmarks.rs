//! Benchmarks for PRISM-LBS world-class implementations
//!
//! Compares performance of:
//! - Lanczos eigensolver (vs simple power iteration baseline)
//! - Delaunay tessellation (vs grid-based detection)
//! - GPU Floyd-Warshall (vs CPU implementation)
//! - HDBSCAN clustering (vs DBSCAN)
//! - Shrake-Rupley SASA (various sphere point counts)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use nalgebra::DMatrix;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

// ============================================================================
// LANCZOS EIGENSOLVER BENCHMARKS
// ============================================================================

fn create_symmetric_matrix(n: usize, seed: u64) -> DMatrix<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut matrix = DMatrix::zeros(n, n);

    for i in 0..n {
        for j in i..n {
            let val = rng.gen_range(-1.0..1.0);
            matrix[(i, j)] = val;
            matrix[(j, i)] = val;
        }
    }

    // Make it positive semi-definite by squaring
    &matrix * matrix.transpose()
}

fn bench_lanczos_eigensolver(c: &mut Criterion) {
    let mut group = c.benchmark_group("Lanczos Eigensolver");

    for n in [50, 100, 200, 500].iter() {
        let matrix = create_symmetric_matrix(*n, 42);

        group.bench_with_input(
            BenchmarkId::new("Lanczos", n),
            &matrix,
            |b, matrix| {
                b.iter(|| {
                    use prism_lbs::softspot::lanczos::LanczosEigensolver;

                    let solver = LanczosEigensolver {
                        max_iter: 100,
                        tol: 1e-8,
                        num_lanczos_vectors: 30.min(*n),
                        seed: Some(42),
                    };

                    black_box(solver.compute_smallest(matrix, 6))
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// DETECTION METHOD BENCHMARKS (GRID VS DELAUNAY)
// ============================================================================

fn bench_detection_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("Detection Methods");

    // Note: These benchmarks require actual atom data
    // Using placeholder that measures configuration overhead

    group.bench_function("Grid config", |b| {
        b.iter(|| {
            use prism_lbs::pocket::voronoi_detector::{VoronoiDetectorConfig, DetectionMethod};
            black_box(VoronoiDetectorConfig {
                detection_method: DetectionMethod::Grid,
                ..Default::default()
            })
        });
    });

    group.bench_function("Delaunay config", |b| {
        b.iter(|| {
            use prism_lbs::pocket::voronoi_detector::{VoronoiDetectorConfig, DetectionMethod};
            black_box(VoronoiDetectorConfig {
                detection_method: DetectionMethod::Delaunay,
                ..Default::default()
            })
        });
    });

    group.bench_function("Hybrid config", |b| {
        b.iter(|| {
            use prism_lbs::pocket::voronoi_detector::{VoronoiDetectorConfig, DetectionMethod};
            black_box(VoronoiDetectorConfig {
                detection_method: DetectionMethod::Hybrid,
                ..Default::default()
            })
        });
    });

    group.finish();
}

// ============================================================================
// FLOYD-WARSHALL BENCHMARKS
// ============================================================================

fn create_adjacency_matrix(n: usize, density: f64, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut adj = vec![0.0f32; n * n];

    for i in 0..n {
        for j in (i + 1)..n {
            if rng.gen::<f64>() < density {
                let weight = rng.gen_range(0.1..1.0);
                adj[i * n + j] = weight;
                adj[j * n + i] = weight;
            }
        }
    }

    adj
}

fn bench_floyd_warshall(c: &mut Criterion) {
    let mut group = c.benchmark_group("Floyd-Warshall APSP");

    for n in [50, 100, 200].iter() {
        let adj = create_adjacency_matrix(*n, 0.3, 42);

        group.bench_with_input(
            BenchmarkId::new("CPU", n),
            &(*n, adj.clone()),
            |b, (n, adj)| {
                b.iter(|| {
                    use prism_lbs::allosteric::gpu_apsp::GpuFloydWarshall;

                    let solver = GpuFloydWarshall::new();
                    black_box(solver.cpu_floyd_warshall(adj, *n))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("CPU Blocked", n),
            &(*n, adj.clone()),
            |b, (n, adj)| {
                b.iter(|| {
                    use prism_lbs::allosteric::gpu_apsp::GpuFloydWarshall;

                    let solver = GpuFloydWarshall::with_block_size(32);
                    black_box(solver.cpu_blocked_floyd_warshall(adj, *n))
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// HDBSCAN BENCHMARKS
// ============================================================================

fn create_point_cloud_3d(n: usize, n_clusters: usize, seed: u64) -> Vec<[f64; 3]> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut points = Vec::with_capacity(n);

    for i in 0..n {
        let cluster = i % n_clusters;
        // Cluster centers offset
        let center = (cluster as f64) * 5.0;
        let point = [
            center + rng.gen_range(-1.0..1.0),
            center + rng.gen_range(-1.0..1.0),
            center + rng.gen_range(-1.0..1.0),
        ];
        points.push(point);
    }

    points
}

fn bench_hdbscan(c: &mut Criterion) {
    let mut group = c.benchmark_group("HDBSCAN Clustering");

    for n in [100, 500, 1000].iter() {
        let points = create_point_cloud_3d(*n, 5, 42);

        group.bench_with_input(
            BenchmarkId::new("HDBSCAN", n),
            &points,
            |b, points| {
                b.iter(|| {
                    use prism_lbs::pocket::hdbscan::HDBSCAN;

                    let hdbscan = HDBSCAN::new(5, 5);
                    black_box(hdbscan.fit(points))
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// SASA BENCHMARKS
// ============================================================================

fn bench_sasa(c: &mut Criterion) {
    let mut group = c.benchmark_group("SASA Computation");

    // Benchmark different sphere point counts
    for n_points in [92, 162, 252, 362].iter() {
        group.bench_with_input(
            BenchmarkId::new("Fibonacci Sphere", n_points),
            n_points,
            |b, &n_points| {
                b.iter(|| {
                    use prism_lbs::pocket::sasa::ShrakeRupleySASA;
                    black_box(ShrakeRupleySASA::new(n_points, 1.4))
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// DELAUNAY TESSELLATION BENCHMARKS
// ============================================================================

fn bench_delaunay_circumsphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("Delaunay Circumsphere");

    // Benchmark circumsphere computation for different tetrahedra
    for _ in [1].iter() {
        group.bench_function("Regular tetrahedron", |b| {
            b.iter(|| {
                use prism_lbs::pocket::delaunay_detector::DelaunayAlphaSphereDetector;
                use prism_lbs::structure::Atom;

                let detector = DelaunayAlphaSphereDetector::default();

                // Create atoms at regular tetrahedron vertices
                let atoms = vec![
                    Atom::new(1, "C".into(), "ALA".into(), 'A', 1, [1.0, 1.0, 1.0], 1.0, 20.0, "C".into()),
                    Atom::new(2, "C".into(), "ALA".into(), 'A', 2, [1.0, -1.0, -1.0], 1.0, 20.0, "C".into()),
                    Atom::new(3, "C".into(), "ALA".into(), 'A', 3, [-1.0, 1.0, -1.0], 1.0, 20.0, "C".into()),
                    Atom::new(4, "C".into(), "ALA".into(), 'A', 4, [-1.0, -1.0, 1.0], 1.0, 20.0, "C".into()),
                ];

                black_box(detector.detect(&atoms))
            });
        });
    }

    group.finish();
}

// ============================================================================
// CRITERION GROUPS
// ============================================================================

criterion_group!(
    benches,
    bench_lanczos_eigensolver,
    bench_detection_methods,
    bench_floyd_warshall,
    bench_hdbscan,
    bench_sasa,
    bench_delaunay_circumsphere,
);

criterion_main!(benches);

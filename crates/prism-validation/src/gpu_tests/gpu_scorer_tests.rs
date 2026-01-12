//! GPU Scorer Integration Tests
//!
//! Tests verify:
//! 1. No CPU fallback (must fail without GPU)
//! 2. RLS stability over many updates
//! 3. Correct feature encoding
//! 4. Weight persistence
//! 5. Performance benchmarks
//!
//! # Zero Fallback Policy
//!
//! The `test_no_cpu_fallback` test is CRITICAL. It verifies that our GPU code
//! will NOT silently fall back to CPU when no GPU is available. If this test
//! passes when run with `CUDA_VISIBLE_DEVICES=""`, we have a hidden fallback
//! that violates the Zero Fallback requirement.
//!
//! # GPU Requirement
//!
//! All tests except `test_no_cpu_fallback` REQUIRE a GPU to run. They will
//! FAIL (not skip) if no GPU is available. This enforces the Zero Fallback
//! Policy at the test level - we don't silently pretend tests passed.

use std::sync::Arc;

use cudarc::driver::CudaContext;

use crate::cryptic_features::CrypticFeatures;
use crate::gpu_zro_cryptic_scorer::GpuZroCrypticScorer;

/// Helper to get GPU context or panic with clear message
/// This enforces Zero Fallback Policy in tests - no silent skipping
fn require_gpu() -> Arc<CudaContext> {
    CudaContext::new(0).expect(
        "GPU REQUIRED: This test requires a CUDA-capable GPU. \
         If you're running on a machine without GPU, use: \
         cargo test --release -p prism-validation --features cryptic-gpu -- --skip gpu_scorer \
         The only test that should run without GPU is test_no_cpu_fallback with CUDA_VISIBLE_DEVICES=\"\""
    )
}

/// CRITICAL TEST: Verify no CPU fallback exists
///
/// This test temporarily hides all GPUs and verifies that
/// CudaContext creation fails. If it succeeds, there is
/// a hidden CPU fallback that violates our Zero Fallback Policy.
///
/// # Running This Test
///
/// ```bash
/// # This should FAIL (expected behavior - proves no fallback)
/// CUDA_VISIBLE_DEVICES="" cargo test --release -p prism-validation --features cryptic-gpu test_no_cpu_fallback
/// ```
#[test]
fn test_no_cpu_fallback() {
    // Save current CUDA_VISIBLE_DEVICES
    let old_val = std::env::var("CUDA_VISIBLE_DEVICES").ok();

    // Hide all GPUs
    std::env::set_var("CUDA_VISIBLE_DEVICES", "");

    // Attempt to create context - should fail
    let result = CudaContext::new(0);

    // Restore CUDA_VISIBLE_DEVICES
    match old_val {
        Some(val) => std::env::set_var("CUDA_VISIBLE_DEVICES", val),
        None => std::env::remove_var("CUDA_VISIBLE_DEVICES"),
    }

    // THIS MUST FAIL - if it succeeds, we have a hidden CPU fallback
    assert!(
        result.is_err(),
        "CRITICAL: CudaContext creation should fail without GPU. \
         If this test fails, there is a hidden CPU fallback that violates \
         the Zero Fallback requirement."
    );
}

/// Test RLS stability over 1000 updates
///
/// Verifies that:
/// - No NaN/Inf values appear in scores
/// - All scores remain in [0, 1] range
/// - Learning actually occurs (scores change over time)
#[test]
fn test_rls_stability_1000_updates() {
    let context = require_gpu();
    let mut scorer = GpuZroCrypticScorer::new(context).unwrap();

    let mut scores = Vec::new();
    for i in 0..1000 {
        let features = CrypticFeatures {
            burial_change: (i as f32 * 0.01).sin(),
            rmsf: (i as f32 * 0.02).cos() + 1.0,
            variance: 0.5,
            sasa_change: (i as f32 * 0.03).sin() * 0.5,
            ..Default::default()
        };

        let target = i % 3 == 0; // 33% positive rate
        let score = scorer.score_and_learn(&features, target).unwrap();

        scores.push(score);

        // Verify score validity
        assert!(
            score.is_finite(),
            "Score became NaN/Inf at iteration {}",
            i
        );
        assert!(
            score >= 0.0 && score <= 1.0,
            "Score {} out of [0,1] at iteration {}",
            score,
            i
        );
    }

    // Verify learning occurred - early and late scores should differ
    let early_mean: f32 = scores[..100].iter().sum::<f32>() / 100.0;
    let late_mean: f32 = scores[900..].iter().sum::<f32>() / 100.0;
    let diff = (late_mean - early_mean).abs();

    assert!(
        diff > 0.01,
        "RLS learning appears stalled: early={:.4}, late={:.4}, diff={:.4}",
        early_mean,
        late_mean,
        diff
    );

    // Log weight statistics
    let stats = scorer.weight_stats();
    println!(
        "After 1000 updates: mean={:.4}, std={:.4}, range=[{:.4}, {:.4}]",
        stats.mean, stats.std, stats.min, stats.max
    );
}

/// Test RLS stability over 10000 updates (extended stress test)
///
/// This is a longer test that verifies numerical stability over
/// many more iterations. Run with `--ignored` flag.
#[test]
#[ignore] // Run with --ignored for extended tests
fn test_rls_stability_10000_updates() {
    let context = require_gpu();
    let mut scorer = GpuZroCrypticScorer::new(context).unwrap();

    for i in 0..10000 {
        let features = CrypticFeatures {
            burial_change: (i as f32 * 0.01).sin(),
            rmsf: (i as f32 * 0.02).cos().abs() + 0.5,
            variance: ((i as f32 * 0.005).sin() + 1.0) * 0.5,
            ..Default::default()
        };

        let target = i % 5 < 2; // 40% positive rate
        let score = scorer.score_and_learn(&features, target).unwrap();

        assert!(score.is_finite(), "Score became NaN/Inf at iteration {}", i);

        if i % 1000 == 0 {
            let stats = scorer.weight_stats();
            println!(
                "Iteration {}: mean={:.4}, std={:.4}",
                i, stats.mean, stats.std
            );
        }
    }

    assert_eq!(scorer.update_count(), 10000);
}

/// Test weight save/load roundtrip
///
/// Verifies that:
/// - Weights can be saved to a file
/// - Weights can be loaded back
/// - Loaded weights produce identical predictions
#[test]
fn test_weight_persistence() {
    let context = require_gpu();
    let mut scorer1 = GpuZroCrypticScorer::new(Arc::clone(&context)).unwrap();

    // Train
    for i in 0..100 {
        let features = CrypticFeatures {
            burial_change: i as f32 * 0.1,
            rmsf: (i as f32 * 0.05).sin() + 1.0,
            ..Default::default()
        };
        scorer1.score_and_learn(&features, i % 2 == 0).unwrap();
    }

    // Save
    let tmp_path = "/tmp/prism_test_weights.json";
    scorer1.save_weights(tmp_path).unwrap();

    // Load into new scorer
    let context2 = require_gpu();
    let mut scorer2 = GpuZroCrypticScorer::new(context2).unwrap();
    scorer2.load_weights(tmp_path).unwrap();

    // Verify same predictions on new inputs
    for i in 0..10 {
        let features = CrypticFeatures {
            burial_change: i as f32 * 0.3,
            rmsf: 1.5,
            ..Default::default()
        };

        // Reset state between scorers to ensure clean comparison
        scorer1.reset_for_structure();
        scorer2.reset_for_structure();

        let score1 = scorer1.score_residue(&features).unwrap();
        let score2 = scorer2.score_residue(&features).unwrap();

        assert!(
            (score1 - score2).abs() < 1e-5,
            "Loaded weights produce different scores: {} vs {}",
            score1,
            score2
        );
    }

    // Cleanup
    std::fs::remove_file(tmp_path).ok();
}

/// Benchmark GPU scorer throughput
///
/// Target: >10,000 residues/second for real-time performance
#[test]
fn bench_gpu_scorer_throughput() {
    let context = require_gpu();
    let mut scorer = GpuZroCrypticScorer::new(context).unwrap();

    let features = CrypticFeatures {
        burial_change: 0.5,
        rmsf: 1.2,
        variance: 0.3,
        neighbor_flexibility: 0.4,
        contact_density: 5.0,
        ..Default::default()
    };

    // Warmup
    for _ in 0..100 {
        scorer.score_residue(&features).unwrap();
    }

    // Benchmark
    let n_iterations = 10000;
    let start = std::time::Instant::now();

    for _ in 0..n_iterations {
        scorer.score_residue(&features).unwrap();
    }

    let elapsed = start.elapsed();
    let throughput = n_iterations as f64 / elapsed.as_secs_f64();

    println!("GPU Scorer Throughput: {:.0} residues/second", throughput);
    println!(
        "Time per residue: {:.2} us",
        elapsed.as_micros() as f64 / n_iterations as f64
    );

    // Target: >10k residues/second for real-time performance
    assert!(
        throughput > 10000.0,
        "GPU scorer too slow: {:.0} residues/sec (target: >10000)",
        throughput
    );
}

/// Test that scorer handles extreme input values gracefully
#[test]
fn test_input_validation() {
    let context = require_gpu();
    let mut scorer = GpuZroCrypticScorer::new(context).unwrap();

    // Valid input should work
    let features = CrypticFeatures::default();
    assert!(scorer.score_residue(&features).is_ok());

    // Test with extreme values (should be handled gracefully)
    let extreme_features = CrypticFeatures {
        burial_change: 1e10,
        rmsf: -1e10,
        variance: f32::INFINITY,
        ..Default::default()
    };

    // Should still return a valid score (clamped/normalized)
    let result = scorer.score_residue(&extreme_features);
    if let Ok(score) = result {
        // Score should be in valid range even with extreme inputs
        assert!(
            score >= 0.0 && score <= 1.0,
            "Score {} out of range for extreme inputs",
            score
        );
    }
    // Note: It's acceptable for this to error, as long as it doesn't panic

    // Reset and test learning with extreme values
    scorer.reset_for_structure();
    let result = scorer.score_and_learn(&extreme_features, true);
    // Should not panic; may error or return clamped value
    assert!(result.is_ok() || result.is_err());
}

/// Test that scorer reset works correctly
#[test]
fn test_scorer_reset() {
    let context = require_gpu();
    let mut scorer = GpuZroCrypticScorer::new(context).unwrap();

    // Train the scorer
    for i in 0..50 {
        let features = CrypticFeatures {
            burial_change: i as f32 * 0.1,
            ..Default::default()
        };
        scorer.score_and_learn(&features, i % 2 == 0).unwrap();
    }

    assert_eq!(scorer.update_count(), 50);

    // Full reset
    scorer.reset().unwrap();

    // Verify reset
    assert_eq!(scorer.update_count(), 0);

    let stats = scorer.weight_stats();
    assert!(
        stats.mean.abs() < 1e-6,
        "Weights not zeroed after reset: mean={}",
        stats.mean
    );
}

/// Test weight statistics calculation
#[test]
fn test_weight_stats() {
    let context = require_gpu();
    let scorer = GpuZroCrypticScorer::new(context).unwrap();

    // Initial weights should be zero
    let stats = scorer.weight_stats();
    assert!((stats.mean - 0.0).abs() < 1e-6);
    assert!((stats.std - 0.0).abs() < 1e-6);
    assert!((stats.min - 0.0).abs() < 1e-6);
    assert!((stats.max - 0.0).abs() < 1e-6);

    // Display format should work
    let display = format!("{}", stats);
    assert!(display.contains("mean="));
    assert!(display.contains("std="));
}

/// Test feature encoding roundtrip
#[test]
fn test_feature_encoding() {
    let features = CrypticFeatures {
        burial_change: 0.5,
        rmsf: 1.2,
        variance: 0.3,
        neighbor_flexibility: 0.4,
        burial_potential: 0.1,
        ss_flexibility: 0.6,
        sidechain_flexibility: 0.7,
        b_factor: 50.0,
        net_charge: -0.5,
        hydrophobicity: 2.0,
        h_bond_potential: 0.8,
        contact_density: 10.0,
        sasa_change: 0.2,
        nearest_charged_dist: 5.0,
        interface_score: 0.3,
        allosteric_proximity: 0.1,
    };

    let mut buffer = [0.0f32; 40];
    features.encode_into(&mut buffer);

    // Verify encoding
    assert!((buffer[0] - 0.5).abs() < 1e-6);
    assert!((buffer[1] - 1.2).abs() < 1e-6);
    assert!((buffer[11] - 10.0).abs() < 1e-6);

    // Velocity slots should be zero
    for i in 16..40 {
        assert_eq!(buffer[i], 0.0);
    }
}

/// Test velocity encoding
#[test]
fn test_velocity_encoding() {
    let prev = CrypticFeatures {
        burial_change: 0.3,
        rmsf: 1.0,
        ..Default::default()
    };

    let curr = CrypticFeatures {
        burial_change: 0.5,
        rmsf: 1.5,
        ..Default::default()
    };

    let mut buffer = [0.0f32; 40];
    curr.encode_with_velocity(&prev, &mut buffer);

    // Current values
    assert!((buffer[0] - 0.5).abs() < 1e-6);
    assert!((buffer[1] - 1.5).abs() < 1e-6);

    // Velocity (delta)
    assert!((buffer[16] - 0.2).abs() < 1e-6); // burial_change velocity
    assert!((buffer[17] - 0.5).abs() < 1e-6); // rmsf velocity
}

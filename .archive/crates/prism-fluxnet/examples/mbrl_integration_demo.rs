//! MBRL Integration Demo
//!
//! Demonstrates how to use the MBRL world model with UltraFluxNetController
//! for enhanced planning in PRISM optimization.
//!
//! ## Usage
//!
//! ```bash
//! # Without MBRL (pure Q-learning)
//! cargo run --example mbrl_integration_demo
//!
//! # With MBRL (model-based planning)
//! cargo run --example mbrl_integration_demo --features mbrl
//! ```

use prism_core::{KernelTelemetry, RuntimeConfig};
use prism_fluxnet::UltraFluxNetController;

fn main() {
    env_logger::init();

    println!("=== MBRL Integration Demo ===\n");

    // Create controller with MBRL integration
    let mut controller = UltraFluxNetController::new();

    // Check MBRL status
    println!("MBRL Status: {}", controller.mbrl_status());
    println!("MBRL Available: {}\n", controller.is_mbrl_available());

    // Configure MBRL parameters (if available)
    if controller.is_mbrl_available() {
        let mbrl = controller.mbrl_integration_mut();
        mbrl.set_planning_horizon(15); // Look ahead 15 steps
        mbrl.set_num_candidates(30); // Evaluate 30 action candidates
        mbrl.set_verbose(true); // Enable debug logging
        println!("✓ MBRL configured (horizon=15, candidates=30)\n");
    } else {
        println!("⚠ MBRL unavailable - using pure Q-learning fallback\n");
        println!("To enable MBRL:");
        println!("  1. Compile with --features mbrl");
        println!("  2. Place ONNX model at models/fluxnet/world_model.onnx or models/gnn/gnn_model.onnx\n");
    }

    // Simulate training loop
    println!("Running 100 iterations...\n");

    let mut config = RuntimeConfig::production();
    let mut best_conflicts = i32::MAX;

    for iteration in 0..100 {
        // Simulate kernel execution (in real usage, this would be GPU kernel output)
        let telemetry = simulate_kernel(&config, iteration);

        // Select action (uses MBRL if available, otherwise Q-learning)
        let action = controller.select_action(&telemetry, &config);

        // Apply action to config
        action.apply(&mut config);

        // Update Q-values based on reward
        controller.update(&telemetry, &config);

        // Track best result
        if telemetry.conflicts < best_conflicts {
            best_conflicts = telemetry.conflicts;
            println!(
                "Iteration {}: NEW BEST! Conflicts={}, Colors={}, Action={:?}, Epsilon={:.3}",
                iteration,
                telemetry.conflicts,
                telemetry.colors_used,
                action,
                controller.epsilon()
            );
        } else if iteration % 10 == 0 {
            println!(
                "Iteration {}: Conflicts={}, Colors={}, Action={:?}",
                iteration, telemetry.conflicts, telemetry.colors_used, action
            );
        }
    }

    println!("\n=== Training Summary ===");
    println!("Best Conflicts: {}", best_conflicts);
    println!("Final Epsilon: {:.3}", controller.epsilon());
    println!("Q-Table Size: {}", controller.q_table_size());

    if controller.is_mbrl_available() {
        let mbrl = controller.mbrl_integration();
        println!("MBRL Experience Buffer: {}", mbrl.buffer_size());
    }

    // Best configuration
    if let Some(best_config) = controller.best_config() {
        println!("\n=== Best Configuration ===");
        println!("Chemical Potential: {:.3}", best_config.chemical_potential);
        println!("Tunneling Prob: {:.3}", best_config.tunneling_prob_base);
        println!("Temperature: {:.3}", best_config.global_temperature);
        println!("Reservoir Leak: {:.3}", best_config.reservoir_leak_rate);
    }

    println!("\n✓ Demo complete!");
}

/// Simulate kernel execution (mock telemetry)
///
/// In real usage, this would be replaced by actual GPU kernel execution.
fn simulate_kernel(config: &RuntimeConfig, iteration: usize) -> KernelTelemetry {
    // Simple decay model for conflicts (moves toward zero)
    let base_conflicts = 1000.0 / (1.0 + iteration as f64 / 50.0);

    // Add some noise based on temperature
    let noise = (config.global_temperature as f64 * 10.0 * (iteration as f64).sin()).abs();

    let conflicts = (base_conflicts + noise).max(0.0) as i32;

    // Colors track with conflicts
    let colors_used = (conflicts as f64 * 0.5) as i32 + 10;

    // Moves depend on tunneling probability
    let moves_applied = (config.tunneling_prob_base * 1000.0) as i32;

    KernelTelemetry {
        conflicts,
        colors_used,
        moves_applied,
        phase_transitions: if iteration % 20 == 0 { 1 } else { 0 },
        betti_numbers: [1.0, 0.0, 0.0], // Simple graph
        reservoir_activity: config.reservoir_leak_rate,
        ..Default::default()
    }
}

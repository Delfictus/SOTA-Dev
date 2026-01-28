//! Example demonstrating UltraFluxNetController integration
//!
//! This example shows how to use the IntegratedFluxNet wrapper to
//! integrate the Ultra controller with the PRISM pipeline.
//!
//! Run with:
//! ```bash
//! cargo run --example ultra_integration --features cuda
//! ```

use prism_core::{KernelTelemetry, RuntimeConfig};
use prism_fluxnet::{ControllerMode, IntegratedFluxNet};

fn main() -> anyhow::Result<()> {
    env_logger::init();

    println!("=== UltraFluxNetController Integration Example ===\n");

    // Create integrated controller in Ultra mode
    let mut controller = IntegratedFluxNet::new_ultra();
    println!("✓ Created IntegratedFluxNet in {:?} mode", controller.mode());

    // Simulate training loop
    println!("\nSimulating 10 iterations of RL training...\n");

    for iteration in 0..10 {
        // Simulate kernel telemetry from GPU
        let telemetry = KernelTelemetry {
            conflicts: if iteration < 5 {
                100 - iteration * 15
            } else {
                25 - (iteration - 5) * 5
            },
            colors_used: 45 + iteration,
            phase_transitions: if iteration % 3 == 0 { 1 } else { 0 },
            moves_applied: 1000 + iteration * 100,
            ..Default::default()
        };

        // Select action using Ultra controller
        let action = controller
            .select_action_ultra(&telemetry)
            .expect("Action selection failed");

        println!(
            "Iteration {}: conflicts={}, action={:?}",
            iteration, telemetry.conflicts, action
        );

        // Apply action to config
        controller.apply_ultra_action(action);

        // Update controller with results
        controller.update(
            &prism_fluxnet::UniversalRLState::new(),
            &prism_fluxnet::UniversalAction::NoOp,
            0.0,
            &prism_fluxnet::UniversalRLState::new(),
            &telemetry,
            "Phase0-DendriticReservoir",
        );

        // Show current config parameters
        let config = controller.get_config();
        println!(
            "  → Updated config: temp={:.3}, chem_pot={:.3}, tunneling={:.3}",
            config.global_temperature, config.chemical_potential, config.tunneling_prob_base
        );
    }

    // Get best configuration
    if let Some(best_config) = controller.best_config() {
        println!("\n✓ Best configuration found:");
        println!("  - Temperature: {:.3}", best_config.global_temperature);
        println!("  - Chemical potential: {:.3}", best_config.chemical_potential);
        println!(
            "  - Tunneling prob: {:.3}",
            best_config.tunneling_prob_base
        );
    }

    if let Some(best_conflicts) = controller.best_conflicts() {
        println!("  - Best conflicts: {}", best_conflicts);
    }

    // Show Q-table statistics
    if let Some(ultra) = controller.ultra_controller() {
        println!("\n✓ Q-table statistics:");
        println!("  - Q-table size: {}", ultra.q_table_size());
        println!("  - Epsilon: {:.3}", ultra.epsilon());
    }

    // Demonstrate saving/loading
    println!("\n✓ Saving controller state...");
    controller.save("ultra_controller_state")?;
    println!("  Saved to: ultra_controller_state.ultra.bin");

    println!("\n=== Example complete ===");
    Ok(())
}

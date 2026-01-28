//! FluxNet Q-table training for DSJC benchmarks.
//!
//! Trains universal RL Q-tables using simulated graph coloring episodes.
//! Outputs binary Q-table files for use with prism-cli.

use anyhow::Result;
use prism_core::{dimacs, Graph, PhaseContext};
use prism_fluxnet::{RLConfig, UniversalAction, UniversalRLController, UniversalRLState};
use rand::Rng;

fn main() -> Result<()> {
    // Parse command-line arguments
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 4 {
        eprintln!("Usage: {} <graph.col> <epochs> <output.bin>", args[0]);
        eprintln!("Example: {} benchmarks/dimacs/DSJC250.5.col 1000 profiles/curriculum/qtable_dsjc250.bin", args[0]);
        std::process::exit(1);
    }

    let graph_path = &args[1];
    let epochs: usize = args[2].parse()?;
    let output_path = &args[3];

    // Initialize logging
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    log::info!("FluxNet Q-table training");
    log::info!("  Graph: {}", graph_path);
    log::info!("  Epochs: {}", epochs);
    log::info!("  Output: {}", output_path);

    // Load graph
    let graph = dimacs::parse_dimacs_file(std::path::Path::new(graph_path))?;
    log::info!(
        "Loaded graph: {} vertices, {} edges, density={:.3}",
        graph.num_vertices,
        graph.num_edges,
        graph.density()
    );

    // Initialize RL controller with training-focused hyperparameters
    let config = RLConfig {
        epsilon: 0.3,         // High exploration during training
        epsilon_decay: 0.995, // Slow decay
        epsilon_min: 0.05,    // Allow some exploration throughout
        alpha: 0.1,           // Moderate learning rate
        gamma: 0.95,          // Value long-term rewards
        replay_buffer_size: 10000,
        replay_batch_size: 32,
        discretization_mode: prism_fluxnet::DiscretizationMode::Compact,
        reward_log_threshold: 0.001, // Log significant geometry rewards during training
    };

    let controller = UniversalRLController::new(config);

    log::info!("Starting training with {} epochs...", epochs);

    let start_time = std::time::Instant::now();
    let mut best_chromatic = graph.num_vertices;
    let mut total_reward = 0.0;

    // Training loop
    for epoch in 0..epochs {
        // Simulate an episode: run through all phases and collect transitions
        let _context = PhaseContext::new();
        let mut state = UniversalRLState::new();

        // Initialize state with graph stats
        state.chromatic_number = graph.num_vertices; // Start pessimistic
        state.conflicts = graph.num_edges; // Maximum conflicts
        state.num_vertices = graph.num_vertices;

        let phases = vec![
            "Phase0-DendriticReservoir",
            "Phase1-ActiveInference",
            "Phase2-Thermodynamic",
            "Phase3-QuantumClassical",
            "Phase4-Geodesic",
            "Phase6-TDA",
            "Phase7-Ensemble",
        ];

        let mut episode_reward = 0.0;

        for phase_name in &phases {
            // Select action using epsilon-greedy
            let action = controller.select_action(&state, phase_name);

            // Simulate phase execution and reward
            let (next_state, reward) = simulate_phase_execution(&state, &action, &graph);

            // Update Q-table
            controller.update_qtable(&state, &action, reward, &next_state, phase_name);

            // Track progress
            episode_reward += reward as f64;

            // Move to next state
            state = next_state;
        }

        // Experience replay for sample efficiency
        for phase_name in &phases {
            controller.replay_batch(phase_name);
        }

        // Decay epsilon after episode
        controller.decay_epsilon();

        total_reward += episode_reward;

        // Track best chromatic number found
        if state.chromatic_number < best_chromatic {
            best_chromatic = state.chromatic_number;
            log::info!(
                "Epoch {}/{}: New best chromatic number = {} (reward={:.2})",
                epoch + 1,
                epochs,
                best_chromatic,
                episode_reward
            );
        }

        // Periodic logging
        if (epoch + 1) % 100 == 0 {
            let avg_reward = total_reward / (epoch + 1) as f64;
            let elapsed = start_time.elapsed().as_secs_f64();
            let eps = controller.epsilon();

            log::info!(
                "Epoch {}/{}: best_chromatic={}, avg_reward={:.2}, epsilon={:.3}, time={:.1}s",
                epoch + 1,
                epochs,
                best_chromatic,
                avg_reward,
                eps,
                elapsed
            );

            // Print Q-table stats for Phase 2 (Thermodynamic)
            let (mean, min, max) = controller.qtable_stats("Phase2-Thermodynamic");
            log::debug!(
                "  Phase2 Q-table: mean={:.3}, min={:.3}, max={:.3}",
                mean,
                min,
                max
            );
        }
    }

    let elapsed = start_time.elapsed().as_secs_f64();
    let avg_reward = total_reward / epochs as f64;

    log::info!("Training completed in {:.1}s", elapsed);
    log::info!("  Best chromatic number: {}", best_chromatic);
    log::info!("  Average reward: {:.2}", avg_reward);
    log::info!("  Final epsilon: {:.3}", controller.epsilon());

    // Save Q-table to binary file
    controller
        .save_qtables_binary(output_path)
        .map_err(|e| anyhow::anyhow!("Failed to save binary Q-tables: {}", e))?;
    log::info!("Q-table saved to: {}", output_path);

    // Also save JSON version for inspection
    let json_path = output_path.replace(".bin", ".json");
    controller
        .save_qtables(&json_path)
        .map_err(|e| anyhow::anyhow!("Failed to save JSON Q-tables: {}", e))?;
    log::info!("JSON version saved to: {}", json_path);

    Ok(())
}

/// Simulates phase execution and computes reward.
///
/// This is a simplified simulation for training purposes.
/// In reality, phases would run actual algorithms.
fn simulate_phase_execution(
    state: &UniversalRLState,
    action: &UniversalAction,
    graph: &Graph,
) -> (UniversalRLState, f32) {
    let mut rng = rand::thread_rng();
    let mut next_state = state.clone();

    // Simulate chromatic number reduction based on action
    // Better actions lead to larger improvements
    let improvement: usize = match action {
        UniversalAction::Phase0(_) => {
            // Dendritic reservoir: moderate improvement, high variance
            rng.gen_range(0..3)
        }
        UniversalAction::Phase1(_) => {
            // Active inference: small but consistent
            rng.gen_range(0..2)
        }
        UniversalAction::Phase2(_) => {
            // Thermodynamic: best improvement potential
            rng.gen_range(1..5)
        }
        UniversalAction::Phase3(_) => {
            // Quantum: high variance
            if rng.gen_bool(0.3) {
                rng.gen_range(3..7)
            } else {
                0
            }
        }
        UniversalAction::Phase4(_) => {
            // Geodesic: moderate
            rng.gen_range(0..3)
        }
        UniversalAction::Phase6(_) => {
            // TDA: small refinement
            rng.gen_range(0..2)
        }
        UniversalAction::Phase7(_) => {
            // Ensemble: consensus-based
            rng.gen_range(1..3)
        }
        UniversalAction::Warmstart(_) => {
            // Warmstart configuration: indirect improvement via better initialization
            rng.gen_range(0..2)
        }
        UniversalAction::Memetic(_) => {
            // Memetic tuning: moderate improvement through algorithm refinement
            rng.gen_range(1..4)
        }
        UniversalAction::Geometry(_) => {
            // Geometry coupling: improvement based on stress response
            // High stress -> aggressive adjustment -> potentially high reward
            rng.gen_range(0..4)
        }
        UniversalAction::MEC(_) => {
            // MEC: molecular emergent computing improvements
            rng.gen_range(0..3)
        }
        UniversalAction::CMA(_) => {
            // CMA-ES: optimization-based improvements
            rng.gen_range(1..4)
        }
        UniversalAction::NoOp => 0,
    };

    // Apply improvement (with diminishing returns as we get closer to optimal)
    let reduction = if next_state.chromatic_number > graph.num_vertices / 2 {
        improvement
    } else {
        improvement.div_ceil(2) // Harder to improve when already good
    };

    next_state.chromatic_number = (next_state.chromatic_number.saturating_sub(reduction)).max(3);

    // Update conflicts based on chromatic number
    // Assume conflicts decrease as chromatic number approaches optimal
    let optimal_estimate = (graph.num_vertices as f64 * 0.15) as usize; // Rough estimate
    if next_state.chromatic_number > optimal_estimate {
        next_state.conflicts = rng.gen_range(0..(graph.num_edges / 10).max(1));
    } else {
        next_state.conflicts = 0; // Assume valid coloring near optimal
    }

    // Compute reward: large positive for improvements, negative for no progress
    let chromatic_delta = (state.chromatic_number as i32) - (next_state.chromatic_number as i32);
    let conflict_delta = (state.conflicts as i32) - (next_state.conflicts as i32);

    let reward = (chromatic_delta as f32) * 10.0 + (conflict_delta as f32) * 0.1;

    (next_state, reward)
}

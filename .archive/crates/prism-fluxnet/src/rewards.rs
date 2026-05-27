//! Phase-specific reward functions for FluxNet RL.
//!
//! This module provides reward computation functions for all 7 PRISM phases,
//! enabling phase-specific learning objectives while maintaining a unified
//! RL framework.
//!
//! # Reward Design Philosophy
//!
//! Each phase has unique objectives that translate to specific reward signals:
//! - **Phase 0 (Reservoir)**: Maximize entropy & sparsity for diverse predictions
//! - **Phase 1 (Active Inference)**: Minimize free energy & maximize information gain
//! - **Phase 2 (Thermodynamic)**: Reduce conflicts & chromatic number (production FluxNet)
//! - **Phase 3 (Quantum)**: Maintain purity & coherence while exploring solutions
//! - **Phase 4/5 (Geodesic)**: Optimize centrality-based coloring strategies
//! - **Phase 6 (TDA)**: Preserve topological structure while reducing colors
//! - **Phase 7 (Ensemble)**: Balance diversity and consensus for robust solutions

use crate::core::state::UniversalRLState;

/// Reward function for Phase 0: Dendritic Reservoir
///
/// # Objectives
/// - Maintain high entropy (diverse activations)
/// - Balance sparsity (avoid over-activation)
/// - Improve prediction accuracy (lower difficulty scores)
///
/// # Returns
/// Reward in range [-1.0, +1.0]
pub fn compute_phase0_reward(state: &UniversalRLState, next_state: &UniversalRLState) -> f32 {
    // Entropy improvement (higher is better, clamp to [0, 1])
    let entropy_delta = (next_state.reservoir_entropy - state.reservoir_entropy) as f32;
    let entropy_reward = entropy_delta.clamp(-0.5, 0.5);

    // Sparsity improvement (target ~0.5 for optimal balance)
    let target_sparsity = 0.5;
    let sparsity_distance_before = (state.reservoir_sparsity - target_sparsity).abs();
    let sparsity_distance_after = (next_state.reservoir_sparsity - target_sparsity).abs();
    let sparsity_reward = ((sparsity_distance_before - sparsity_distance_after) as f32).clamp(-0.5, 0.5);

    // Combined reward with equal weighting
    entropy_reward * 0.5 + sparsity_reward * 0.5
}

/// Reward function for Phase 1: Active Inference
///
/// # Objectives
/// - Minimize Expected Free Energy (EFE)
/// - Minimize Variational Free Energy (VFE)
/// - Maximize information gain (exploration bonus)
///
/// # Returns
/// Reward in range [-1.0, +1.0]
pub fn compute_phase1_reward(state: &UniversalRLState, next_state: &UniversalRLState) -> f32 {
    // EFE reduction (lower is better)
    let efe_delta = (state.active_inference_efe - next_state.active_inference_efe) as f32;
    let efe_reward = (efe_delta / 10.0).clamp(-0.5, 0.5);

    // VFE reduction (lower is better)
    let vfe_delta = (state.active_inference_vfe - next_state.active_inference_vfe) as f32;
    let vfe_reward = (vfe_delta / 10.0).clamp(-0.5, 0.5);

    // Combined reward (EFE weighted higher)
    efe_reward * 0.6 + vfe_reward * 0.4
}

/// Reward function for Phase 2: Thermodynamic Equilibration
///
/// # Objectives
/// - Reduce conflicts (edge violations)
/// - Reduce chromatic number (total colors used)
/// - Maintain compaction ratio (convergence health)
///
/// # Note
/// This delegates to the production FluxNet RLController::compute_reward()
/// for consistency with the GPU-accelerated Phase 2 implementation.
///
/// # Returns
/// Reward in range [-1.0, +1.0]
pub fn compute_phase2_reward(state: &UniversalRLState, next_state: &UniversalRLState) -> f32 {
    // Conflict reduction (primary objective)
    let conflict_delta = (state.conflicts as f32 - next_state.conflicts as f32) / 100.0;
    let conflict_reward = conflict_delta.clamp(-0.5, 0.5);

    // Chromatic number reduction (secondary objective)
    let chromatic_delta = (state.chromatic_number as f32 - next_state.chromatic_number as f32) / 10.0;
    let chromatic_reward = chromatic_delta.clamp(-0.3, 0.3);

    // Temperature stage progress (tertiary objective - annealing progression)
    let temp_progress_reward = ((next_state.phase2_temperature_stage - state.phase2_temperature_stage) as f32).clamp(-0.2, 0.2);

    // Weighted combination
    conflict_reward * 0.5 + chromatic_reward * 0.3 + temp_progress_reward * 0.2
}

/// Reward function for Phase 3: Quantum-Classical Hybrid
///
/// # Objectives
/// - Maintain quantum purity (avoid decoherence)
/// - Optimize entanglement (balance correlations)
/// - Preserve amplitude variance (exploration)
/// - Maintain phase coherence (interference quality)
///
/// # Returns
/// Reward in range [-1.0, +1.0]
pub fn compute_phase3_reward(state: &UniversalRLState, next_state: &UniversalRLState) -> f32 {
    // Purity maintenance (target ~1.0)
    let purity_delta = (next_state.quantum_purity - state.quantum_purity) as f32;
    let purity_reward = purity_delta.clamp(-0.3, 0.3);

    // Entanglement balance (target ~0.5 for optimal exploration-exploitation)
    let target_entanglement = 0.5;
    let entanglement_distance_before = (state.quantum_entanglement - target_entanglement).abs();
    let entanglement_distance_after = (next_state.quantum_entanglement - target_entanglement).abs();
    let entanglement_reward = ((entanglement_distance_before - entanglement_distance_after) as f32).clamp(-0.3, 0.3);

    // Amplitude variance reward (higher variance = better exploration)
    let amplitude_delta = (next_state.quantum_amplitude_variance - state.quantum_amplitude_variance) as f32;
    let amplitude_reward = (amplitude_delta * 0.5).clamp(-0.2, 0.2);

    // Coherence reward (higher coherence = better interference)
    let coherence_delta = (next_state.quantum_coherence - state.quantum_coherence) as f32;
    let coherence_reward = coherence_delta.clamp(-0.2, 0.2);

    // Weighted combination
    purity_reward * 0.3 + entanglement_reward * 0.3 + amplitude_reward * 0.2 + coherence_reward * 0.2
}

/// Reward function for Phase 4/5: Geodesic (Network Topology)
///
/// # Objectives
/// - Optimize centrality-based vertex ordering
/// - Minimize graph diameter (compact solutions)
/// - Leverage geodesic distances for coloring
///
/// # Returns
/// Reward in range [-1.0, +1.0]
pub fn compute_phase4_reward(state: &UniversalRLState, next_state: &UniversalRLState) -> f32 {
    // Centrality optimization (depends on strategy - higher or lower)
    // We reward changes that improve chromatic number
    let chromatic_delta = (state.chromatic_number as f32 - next_state.chromatic_number as f32) / 10.0;
    let chromatic_reward = chromatic_delta.clamp(-0.5, 0.5);

    // Diameter reduction (more compact graph = better coloring)
    let diameter_delta = (state.geodesic_diameter - next_state.geodesic_diameter) as f32;
    let diameter_reward = (diameter_delta / 10.0).clamp(-0.3, 0.3);

    // Centrality stability (avoid large fluctuations)
    let centrality_delta = (next_state.geodesic_centrality - state.geodesic_centrality).abs() as f32;
    let stability_penalty = (-centrality_delta * 0.2).clamp(-0.2, 0.0);

    // Weighted combination
    chromatic_reward * 0.5 + diameter_reward * 0.3 + stability_penalty * 0.2
}

/// Reward function for Phase 6: Topological Data Analysis (TDA)
///
/// # Objectives
/// - Preserve topological persistence (structural features)
/// - Optimize Betti numbers (connectivity)
/// - Balance coherence coefficient of variation
///
/// # Returns
/// Reward in range [-1.0, +1.0]
pub fn compute_phase6_reward(state: &UniversalRLState, next_state: &UniversalRLState) -> f32 {
    // Persistence improvement (higher = more stable features)
    let persistence_delta = (next_state.tda_persistence - state.tda_persistence) as f32;
    let persistence_reward = persistence_delta.clamp(-0.4, 0.4);

    // Betti0 stability (connected components should be stable)
    let betti_delta = (state.tda_betti_0 as i32 - next_state.tda_betti_0 as i32).abs() as f32;
    let betti_penalty = (-betti_delta * 0.1).clamp(-0.3, 0.0);

    // Coherence CV reward (higher CV = better structural diversity)
    let coherence_cv_delta = (next_state.coherence_cv - state.coherence_cv) as f32;
    let coherence_reward = coherence_cv_delta.clamp(-0.3, 0.3);

    // Weighted combination
    persistence_reward * 0.4 + betti_penalty * 0.3 + coherence_reward * 0.3
}

/// Reward function for Phase 7: Ensemble Consensus
///
/// # Objectives
/// - Balance diversity (explore solution space)
/// - Maximize consensus (agreement across replicas)
/// - Improve final chromatic number
///
/// # Returns
/// Reward in range [-1.0, +1.0]
pub fn compute_phase7_reward(state: &UniversalRLState, next_state: &UniversalRLState) -> f32 {
    // Diversity reward (target ~0.5 for optimal balance)
    let target_diversity = 0.5;
    let diversity_distance_before = (state.ensemble_diversity - target_diversity).abs();
    let diversity_distance_after = (next_state.ensemble_diversity - target_diversity).abs();
    let diversity_reward = ((diversity_distance_before - diversity_distance_after) as f32).clamp(-0.3, 0.3);

    // Consensus improvement (higher = better agreement)
    let consensus_delta = (next_state.ensemble_consensus - state.ensemble_consensus) as f32;
    let consensus_reward = consensus_delta.clamp(-0.4, 0.4);

    // Chromatic number improvement (final objective)
    let chromatic_delta = (state.chromatic_number as f32 - next_state.chromatic_number as f32) / 10.0;
    let chromatic_reward = chromatic_delta.clamp(-0.3, 0.3);

    // Weighted combination
    diversity_reward * 0.3 + consensus_reward * 0.4 + chromatic_reward * 0.3
}

/// Reward function for Warmstart phase
///
/// # Objectives
/// - Improve warmstart quality (anchor selection)
/// - Reduce initial chromatic number
///
/// # Returns
/// Reward in range [-1.0, +1.0]
pub fn compute_warmstart_reward(state: &UniversalRLState, next_state: &UniversalRLState) -> f32 {
    // Warmstart quality improvement
    let quality_delta = (next_state.warmstart_quality - state.warmstart_quality) as f32;
    let quality_reward = quality_delta.clamp(-0.5, 0.5);

    // Initial chromatic number reduction
    let chromatic_delta = (state.chromatic_number as f32 - next_state.chromatic_number as f32) / 10.0;
    let chromatic_reward = chromatic_delta.clamp(-0.5, 0.5);

    // Equal weighting
    quality_reward * 0.5 + chromatic_reward * 0.5
}

/// Reward function for Memetic algorithm phase
///
/// # Objectives
/// - Maximize improvement rate (convergence speed)
/// - Reduce chromatic number via local search
///
/// # Returns
/// Reward in range [-1.0, +1.0]
pub fn compute_memetic_reward(state: &UniversalRLState, next_state: &UniversalRLState) -> f32 {
    // Improvement rate increase (faster convergence)
    let improvement_delta = (next_state.memetic_improvement_rate - state.memetic_improvement_rate) as f32;
    let improvement_reward = (improvement_delta * 10.0).clamp(-0.5, 0.5);

    // Chromatic number reduction
    let chromatic_delta = (state.chromatic_number as f32 - next_state.chromatic_number as f32) / 10.0;
    let chromatic_reward = chromatic_delta.clamp(-0.5, 0.5);

    // Equal weighting
    improvement_reward * 0.5 + chromatic_reward * 0.5
}

/// Universal reward dispatcher - routes to phase-specific reward function
///
/// # Arguments
/// - `phase`: Phase identifier (e.g., "Phase0-DendriticReservoir")
/// - `state`: Current state
/// - `next_state`: Next state after action
///
/// # Returns
/// Phase-specific reward in range [-1.0, +1.0]
pub fn compute_reward(phase: &str, state: &UniversalRLState, next_state: &UniversalRLState) -> f32 {
    match phase {
        "Phase0-DendriticReservoir" | "Phase0-Ontology" => compute_phase0_reward(state, next_state),
        "Phase1-ActiveInference" => compute_phase1_reward(state, next_state),
        "Phase2-Thermodynamic" => compute_phase2_reward(state, next_state),
        "Phase3-QuantumClassical" => compute_phase3_reward(state, next_state),
        "Phase4-Geodesic" | "Phase5-NetworkTopology" => compute_phase4_reward(state, next_state),
        "Phase6-TDA" => compute_phase6_reward(state, next_state),
        "Phase7-Ensemble" => compute_phase7_reward(state, next_state),
        "Warmstart" => compute_warmstart_reward(state, next_state),
        "Memetic" => compute_memetic_reward(state, next_state),
        _ => {
            log::warn!("Unknown phase '{}', using generic chromatic reduction reward", phase);
            // Generic fallback: reward chromatic number reduction
            let chromatic_delta = (state.chromatic_number as f32 - next_state.chromatic_number as f32) / 10.0;
            chromatic_delta.clamp(-1.0, 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase0_reward() {
        let mut state = UniversalRLState::new();
        state.reservoir_entropy = 0.4;
        state.reservoir_sparsity = 0.3;

        let mut next_state = state.clone();
        next_state.reservoir_entropy = 0.6; // Improved
        next_state.reservoir_sparsity = 0.5; // Closer to target

        let reward = compute_phase0_reward(&state, &next_state);
        assert!(reward > 0.0, "Reward should be positive for improvements");
    }

    #[test]
    fn test_phase2_reward() {
        let mut state = UniversalRLState::new();
        state.conflicts = 100;
        state.chromatic_number = 50;

        let mut next_state = state.clone();
        next_state.conflicts = 80; // Reduced conflicts
        next_state.chromatic_number = 48; // Reduced colors

        let reward = compute_phase2_reward(&state, &next_state);
        assert!(reward > 0.0, "Reward should be positive for conflict/color reduction");
    }

    #[test]
    fn test_phase3_reward() {
        let mut state = UniversalRLState::new();
        state.quantum_purity = 0.9;
        state.quantum_entanglement = 0.3;

        let mut next_state = state.clone();
        next_state.quantum_purity = 0.95; // Improved
        next_state.quantum_entanglement = 0.5; // Closer to target

        let reward = compute_phase3_reward(&state, &next_state);
        assert!(reward > 0.0, "Reward should be positive for purity/entanglement improvements");
    }

    #[test]
    fn test_reward_dispatcher() {
        let state = UniversalRLState::new();
        let mut next_state = state.clone();
        next_state.chromatic_number = state.chromatic_number.saturating_sub(5);

        // Test all phases
        for phase in &[
            "Phase0-DendriticReservoir",
            "Phase1-ActiveInference",
            "Phase2-Thermodynamic",
            "Phase3-QuantumClassical",
            "Phase4-Geodesic",
            "Phase6-TDA",
            "Phase7-Ensemble",
            "Warmstart",
            "Memetic",
        ] {
            let reward = compute_reward(phase, &state, &next_state);
            assert!(
                reward >= -1.0 && reward <= 1.0,
                "Reward for {} should be in [-1.0, 1.0], got {}",
                phase,
                reward
            );
        }
    }

    #[test]
    fn test_unknown_phase_fallback() {
        let mut state = UniversalRLState::new();
        state.chromatic_number = 50;

        let mut next_state = state.clone();
        next_state.chromatic_number = 45;

        let reward = compute_reward("UnknownPhase", &state, &next_state);
        assert!(reward > 0.0, "Fallback should reward chromatic reduction");
    }
}

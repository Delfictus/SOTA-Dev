//! Universal RL Controller with per-phase Q-tables.
//!
//! Implements PRISM GPU Plan §3.3: UniversalRLController.

use super::actions::UniversalAction;
use super::state::{DiscretizationMode, UniversalRLState};
use parking_lot::RwLock;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// Configuration for the RL controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLConfig {
    /// Learning rate (alpha)
    pub alpha: f64,

    /// Discount factor (gamma)
    pub gamma: f64,

    /// Epsilon for epsilon-greedy exploration
    pub epsilon: f64,

    /// Epsilon decay rate per episode
    pub epsilon_decay: f64,

    /// Minimum epsilon value
    pub epsilon_min: f64,

    /// Replay buffer size
    pub replay_buffer_size: usize,

    /// Batch size for experience replay
    pub replay_batch_size: usize,

    /// State discretization mode
    pub discretization_mode: DiscretizationMode,

    /// Minimum reward bonus magnitude for logging (default: 0.001)
    /// Logs appear when |geometry_bonus| > threshold
    pub reward_log_threshold: f64,
}

impl Default for RLConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            gamma: 0.95,
            epsilon: 0.3,
            epsilon_decay: 0.995,
            epsilon_min: 0.05,
            replay_buffer_size: 10000,
            replay_batch_size: 32,
            discretization_mode: DiscretizationMode::Compact,
            reward_log_threshold: 0.001,
        }
    }
}

impl RLConfig {
    /// Creates a builder for RL configuration.
    pub fn builder() -> RLConfigBuilder {
        RLConfigBuilder::default()
    }
}

/// Builder for RLConfig.
#[derive(Debug, Default)]
pub struct RLConfigBuilder {
    config: RLConfig,
}

impl RLConfigBuilder {
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    pub fn gamma(mut self, gamma: f64) -> Self {
        self.config.gamma = gamma;
        self
    }

    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.config.epsilon = epsilon;
        self
    }

    pub fn replay_buffer_size(mut self, size: usize) -> Self {
        self.config.replay_buffer_size = size;
        self
    }

    pub fn discretization_mode(mut self, mode: DiscretizationMode) -> Self {
        self.config.discretization_mode = mode;
        self
    }

    pub fn reward_log_threshold(mut self, threshold: f64) -> Self {
        self.config.reward_log_threshold = threshold;
        self
    }

    pub fn build(self) -> RLConfig {
        self.config
    }
}

/// Replay buffer transition: (state, action, reward, next_state).
type Transition = (UniversalRLState, UniversalAction, f32, UniversalRLState);

/// Universal RL Controller managing per-phase Q-tables.
///
/// ## Architecture
///
/// - **Phase-specific Q-tables**: HashMap<PhaseID, Vec<Vec<f32>>>
///   - Each phase has its own Q-table: [num_states × num_actions]
///   - Allows independent learning for each phase
///
/// - **Shared replay buffer**: VecDeque<Transition>
///   - Stores transitions from all phases
///   - Enables experience replay for sample efficiency
///
/// - **Epsilon-greedy exploration**: Balances exploration vs. exploitation
///
/// ## Q-Learning Update
///
/// ```text
/// Q(s, a) ← Q(s, a) + α * [r + γ * max_a' Q(s', a') - Q(s, a)]
/// ```
///
/// where:
/// - s = current state (discretized)
/// - a = action taken
/// - r = reward received
/// - s' = next state (discretized)
/// - α = learning rate
/// - γ = discount factor
pub struct UniversalRLController {
    /// Configuration
    config: RLConfig,

    /// Per-phase Q-tables: phase_id -> [state][action] -> Q-value
    /// Thread-safe to allow concurrent access from multiple phases
    phase_qtables: Arc<RwLock<HashMap<String, Vec<Vec<f32>>>>>,

    /// Shared replay buffer (thread-safe)
    replay_buffer: Arc<RwLock<VecDeque<Transition>>>,

    /// Current epsilon (decays over time)
    epsilon: Arc<RwLock<f64>>,

    /// Episode counter (for epsilon decay)
    episode: Arc<RwLock<usize>>,
}

impl UniversalRLController {
    /// Creates a new RL controller with the given configuration.
    pub fn new(config: RLConfig) -> Self {
        let num_states = config.discretization_mode.num_states();
        const NUM_ACTIONS: usize = 104; // Total actions: Phase0-7(56) + Warmstart(8) + Memetic(8) + Geometry(8) + MEC(8) + CMA(8) + NoOp(1) = 97 unique + padding to 104

        let mut phase_qtables = HashMap::new();

        // Initialize Q-tables for all phases
        // IMPORTANT: Phase names must match exactly what PhaseController::name() returns
        for phase in &[
            "Phase0-Ontology",           // Semantic grounding (ontology phase)
            "Phase0-DendriticReservoir", // Dendritic reservoir phase
            "Phase1-ActiveInference",
            "Phase2-Thermodynamic",
            "Phase3-QuantumClassical",
            "Phase4-Geodesic",
            "Phase6-TDA",
            "Phase7-Ensemble",
            "PhaseX-CMA", // CMA-ES optimization
            "PhaseM-MEC", // Molecular Emergent Computing
        ] {
            phase_qtables.insert(phase.to_string(), vec![vec![0.0; NUM_ACTIONS]; num_states]);
        }

        Self {
            epsilon: Arc::new(RwLock::new(config.epsilon)),
            config,
            phase_qtables: Arc::new(RwLock::new(phase_qtables)),
            replay_buffer: Arc::new(RwLock::new(VecDeque::new())),
            episode: Arc::new(RwLock::new(0)),
        }
    }

    /// Selects an action for the given state and phase using epsilon-greedy.
    ///
    /// With probability ε, selects a random action (exploration).
    /// With probability (1 - ε), selects the action with highest Q-value (exploitation).
    pub fn select_action(&self, state: &UniversalRLState, phase: &str) -> UniversalAction {
        let state_idx = state.discretize(self.config.discretization_mode);
        let epsilon = *self.epsilon.read();

        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < epsilon {
            // Exploration: random action
            let actions = UniversalAction::all_actions_for_phase(phase);
            actions[rng.gen_range(0..actions.len())].clone()
        } else {
            // Exploitation: best action from Q-table
            self.best_action(state_idx, phase)
        }
    }

    /// Returns the action with the highest Q-value for the given state and phase.
    fn best_action(&self, state_idx: usize, phase: &str) -> UniversalAction {
        let qtables = self.phase_qtables.read();
        let qtable = match qtables.get(phase) {
            Some(qt) => qt,
            None => {
                // Q-table not found - return default action (increase exploration)
                log::warn!("Phase Q-table '{}' not found, using default action", phase);
                return UniversalAction::Phase1(
                    crate::core::actions::ActiveInferenceAction::IncreaseExploration,
                );
            }
        };

        let q_values = &qtable[state_idx];
        let best_action_idx = q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        UniversalAction::from_index(best_action_idx, phase).unwrap_or(UniversalAction::NoOp)
    }

    /// Computes MEC reward bonus based on free energy changes.
    ///
    /// Lower free energy indicates better molecular configuration.
    /// Reward is proportional to the negative change in free energy (normalized).
    ///
    /// # Returns
    /// Reward bonus in range [-1.0, +1.0]
    fn compute_mec_reward_bonus(
        &self,
        state: &UniversalRLState,
        next_state: &UniversalRLState,
    ) -> f32 {
        if state.mec_free_energy == 0.0 && next_state.mec_free_energy == 0.0 {
            return 0.0; // No MEC data available
        }

        // Free energy decrease is good (positive reward)
        // Normalize by a typical free energy scale (e.g., 1e6)
        let delta = (state.mec_free_energy - next_state.mec_free_energy) / 1e6;
        (delta as f32).clamp(-1.0, 1.0)
    }

    /// Computes CMA reward bonus based on transfer entropy improvements.
    ///
    /// Lower TE mean indicates better convergence in CMA-ES optimization.
    ///
    /// # Returns
    /// Reward bonus in range [-0.5, +0.5]
    fn compute_cma_reward_bonus(
        &self,
        state: &UniversalRLState,
        next_state: &UniversalRLState,
    ) -> f32 {
        if state.cma_te_mean == 0.0 && next_state.cma_te_mean == 0.0 {
            return 0.0; // No CMA data available
        }

        // TE mean decrease is good (positive reward)
        let delta = (state.cma_te_mean - next_state.cma_te_mean) * 2.0; // Scale by 2.0
        (delta as f32).clamp(-0.5, 0.5)
    }

    /// Computes ontology reward bonus based on semantic conflict resolution.
    ///
    /// Fewer conflicts indicate better semantic grounding.
    ///
    /// # Returns
    /// Reward bonus in range [-0.5, +0.5]
    fn compute_ontology_reward_bonus(
        &self,
        state: &UniversalRLState,
        next_state: &UniversalRLState,
    ) -> f32 {
        if state.ontology_conflicts == 0 && next_state.ontology_conflicts == 0 {
            return 0.0; // No conflicts (perfect state)
        }

        // Conflict decrease is good (positive reward)
        let delta = (state.ontology_conflicts as i32 - next_state.ontology_conflicts as i32) as f32;
        // Normalize by a typical conflict scale (e.g., 10 conflicts)
        (delta / 10.0).clamp(-0.5, 0.5)
    }

    /// Updates the Q-table based on a transition.
    ///
    /// Implements Q-learning update with multi-subsystem reward shaping:
    /// Q(s, a) ← Q(s, a) + α * [r_shaped + γ * max_a' Q(s', a') - Q(s, a)]
    ///
    /// where r_shaped = r + geometry_bonus + mec_bonus + cma_bonus + ontology_bonus
    ///
    /// ## Reward Shaping Components:
    /// - **Geometry**: Bonus for stress reduction (metaphysical telemetry from Phase 4/6)
    /// - **MEC**: Bonus for free energy decrease (molecular dynamics convergence)
    /// - **CMA**: Bonus for TE mean reduction (evolutionary optimization progress)
    /// - **Ontology**: Bonus for semantic conflict resolution (knowledge grounding)
    pub fn update_qtable(
        &self,
        state: &UniversalRLState,
        action: &UniversalAction,
        reward: f32,
        next_state: &UniversalRLState,
        phase: &str,
    ) {
        let state_idx = state.discretize(self.config.discretization_mode);
        let next_state_idx = next_state.discretize(self.config.discretization_mode);
        let action_idx = action.to_index();

        // Compute geometry reward bonus from stress reduction
        let geometry_bonus = next_state.compute_geometry_reward_bonus();

        // Compute MEC reward bonus (lower free energy is better)
        let mec_bonus = self.compute_mec_reward_bonus(state, next_state);

        // Compute CMA reward bonus (lower TE mean indicates better convergence)
        let cma_bonus = self.compute_cma_reward_bonus(state, next_state);

        // Compute ontology reward bonus (fewer conflicts is better)
        let ontology_bonus = self.compute_ontology_reward_bonus(state, next_state);

        // Total shaped reward with all subsystem bonuses
        let shaped_reward = reward + geometry_bonus as f32 + mec_bonus + cma_bonus + ontology_bonus;

        // Log significant bonuses (exceeds configured threshold)
        if geometry_bonus.abs() > self.config.reward_log_threshold {
            log::info!(
                "FluxNet: Geometry reward bonus {:+.4} (stress: {:.3} → {:.3}, delta: {:.3})",
                geometry_bonus,
                next_state.previous_geometry_stress,
                next_state.geometry_stress_level,
                next_state.previous_geometry_stress - next_state.geometry_stress_level
            );
        }
        if mec_bonus.abs() > self.config.reward_log_threshold as f32 {
            log::info!(
                "FluxNet: MEC reward bonus {:+.4} (free_energy: {:.1} → {:.1})",
                mec_bonus,
                state.mec_free_energy,
                next_state.mec_free_energy
            );
        }
        if cma_bonus.abs() > self.config.reward_log_threshold as f32 {
            log::info!(
                "FluxNet: CMA reward bonus {:+.4} (te_mean: {:.4} → {:.4})",
                cma_bonus,
                state.cma_te_mean,
                next_state.cma_te_mean
            );
        }
        if ontology_bonus.abs() > self.config.reward_log_threshold as f32 {
            log::info!(
                "FluxNet: Ontology reward bonus {:+.4} (conflicts: {} → {})",
                ontology_bonus,
                state.ontology_conflicts,
                next_state.ontology_conflicts
            );
        }

        let mut qtables = self.phase_qtables.write();
        let qtable = qtables.get_mut(phase).expect("Phase Q-table not found");

        // Get current Q-value
        let current_q = qtable[state_idx][action_idx];

        // Get max Q-value for next state
        let max_next_q = qtable[next_state_idx]
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        // Q-learning update with shaped reward
        let new_q = current_q
            + self.config.alpha as f32
                * (shaped_reward + self.config.gamma as f32 * max_next_q - current_q);

        qtable[state_idx][action_idx] = new_q;

        // Add transition to replay buffer (use original reward for consistency)
        let mut buffer = self.replay_buffer.write();
        buffer.push_back((state.clone(), action.clone(), reward, next_state.clone()));

        // Trim buffer if it exceeds max size
        while buffer.len() > self.config.replay_buffer_size {
            buffer.pop_front();
        }
    }

    /// Performs experience replay by sampling from the replay buffer.
    ///
    /// Samples a batch of transitions and updates Q-tables.
    /// This improves sample efficiency and breaks temporal correlations.
    pub fn replay_batch(&self, phase: &str) {
        let buffer = self.replay_buffer.read();

        if buffer.len() < self.config.replay_batch_size {
            return; // Not enough samples
        }

        let mut rng = rand::thread_rng();
        let indices: Vec<usize> = (0..self.config.replay_batch_size)
            .map(|_| rng.gen_range(0..buffer.len()))
            .collect();

        // Clone transitions while holding read lock to avoid deadlock
        // (update_qtable needs write lock on replay_buffer)
        let transitions: Vec<Transition> = indices
            .iter()
            .filter_map(|&idx| buffer.get(idx).cloned())
            .collect();

        drop(buffer); // Release lock before updating Q-table

        // Now update Q-tables without holding any locks
        for (state, action, reward, next_state) in transitions {
            self.update_qtable(&state, &action, reward, &next_state, phase);
        }
    }

    /// Decays epsilon after an episode.
    ///
    /// Epsilon decays exponentially: ε ← max(ε_min, ε * decay_rate)
    pub fn decay_epsilon(&self) {
        let mut epsilon = self.epsilon.write();
        *epsilon = (*epsilon * self.config.epsilon_decay).max(self.config.epsilon_min);

        let mut episode = self.episode.write();
        *episode += 1;

        log::debug!("Episode {}: epsilon decayed to {:.4}", *episode, *epsilon);
    }

    /// Returns the current epsilon value.
    pub fn epsilon(&self) -> f64 {
        *self.epsilon.read()
    }

    /// Returns the current episode number.
    pub fn episode(&self) -> usize {
        *self.episode.read()
    }

    /// Saves Q-tables to disk in JSON format (human-readable).
    pub fn save_qtables(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let qtables = self.phase_qtables.read();
        let json = serde_json::to_string_pretty(&*qtables)?;
        std::fs::write(path, json)?;
        log::info!("Saved Q-tables to {} (JSON)", path);
        Ok(())
    }

    /// Loads Q-tables from disk (JSON format).
    pub fn load_qtables(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(path)?;
        let loaded_qtables: HashMap<String, Vec<Vec<f32>>> = serde_json::from_str(&json)?;

        let mut qtables = self.phase_qtables.write();
        *qtables = loaded_qtables;

        log::info!("Loaded Q-tables from {} (JSON)", path);
        Ok(())
    }

    /// Saves Q-tables to disk in binary format (compact, fast).
    ///
    /// Uses bincode for efficient serialization. Preferred for production use.
    pub fn save_qtables_binary(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let qtables = self.phase_qtables.read();
        let data = bincode::serialize(&*qtables)?;
        let data_len = data.len();
        std::fs::write(path, data)?;
        log::info!("Saved Q-tables to {} (binary, {} bytes)", path, data_len);
        Ok(())
    }

    /// Loads Q-tables from disk (binary format).
    pub fn load_qtables_binary(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let data = std::fs::read(path)?;
        let loaded_qtables: HashMap<String, Vec<Vec<f32>>> = bincode::deserialize(&data)?;

        let mut qtables = self.phase_qtables.write();
        *qtables = loaded_qtables;

        log::info!(
            "Loaded Q-tables from {} (binary, {} bytes)",
            path,
            data.len()
        );
        Ok(())
    }

    /// Returns statistics about the Q-tables.
    pub fn qtable_stats(&self, phase: &str) -> (f32, f32, f32) {
        let qtables = self.phase_qtables.read();
        let qtable = qtables.get(phase).expect("Phase Q-table not found");

        let mut sum = 0.0;
        let mut count = 0;
        let mut max_q = f32::MIN;
        let mut min_q = f32::MAX;

        for row in qtable {
            for &q in row {
                sum += q;
                count += 1;
                max_q = max_q.max(q);
                min_q = min_q.min(q);
            }
        }

        let mean = if count > 0 { sum / count as f32 } else { 0.0 };
        (mean, min_q, max_q)
    }

    /// Initializes Q-tables from a curriculum Q-table (warmstart).
    ///
    /// Maps curriculum Q-table (sparse: state_hash -> action -> Q-value)
    /// to dense Q-tables (state_index -> action_index -> Q-value).
    ///
    /// ## Algorithm
    /// 1. For each phase's Q-table
    /// 2. For each (state_hash, actions) in curriculum
    /// 3. Map state_hash to state_index (modulo num_states)
    /// 4. Copy Q-values to corresponding positions
    /// 5. Unmapped states retain initial zero values
    ///
    /// Refs: PRISM GPU Plan §6.4 (Curriculum Integration)
    pub fn initialize_from_curriculum(
        &self,
        curriculum_qtable: &HashMap<u64, HashMap<usize, f32>>,
        phase: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let num_states = self.config.discretization_mode.num_states();
        const NUM_ACTIONS: usize = 104; // Must match action space size (see actions.rs)

        let mut qtables = self.phase_qtables.write();
        let qtable = qtables
            .get_mut(phase)
            .ok_or_else(|| format!("Phase Q-table not found: {}", phase))?;

        let mut entries_loaded = 0;

        // Map curriculum entries to dense Q-table
        for (&state_hash, actions) in curriculum_qtable {
            // Map state hash to index using modulo (simple collision resolution)
            let state_idx = (state_hash as usize) % num_states;

            for (&action_idx, &q_value) in actions {
                if action_idx < NUM_ACTIONS {
                    qtable[state_idx][action_idx] = q_value;
                    entries_loaded += 1;
                }
            }
        }

        log::info!(
            "Initialized Q-table for phase {} from curriculum: {} entries loaded",
            phase,
            entries_loaded
        );

        Ok(())
    }

    /// Initializes all phase Q-tables from a curriculum Q-table.
    ///
    /// Applies the same curriculum Q-table to all phases.
    /// Useful when curriculum is phase-agnostic or as a general warmstart.
    pub fn initialize_all_phases_from_curriculum(
        &self,
        curriculum_qtable: &HashMap<u64, HashMap<usize, f32>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for phase in &[
            "Phase0-DendriticReservoir",
            "Phase1-ActiveInference",
            "Phase2-Thermodynamic",
            "Phase3-QuantumClassical",
            "Phase4-Geodesic",
            "Phase6-TDA",
            "Phase7-Ensemble",
        ] {
            self.initialize_from_curriculum(curriculum_qtable, phase)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_controller_initialization() {
        let config = RLConfig::default();
        let controller = UniversalRLController::new(config);

        assert_eq!(controller.epsilon(), 0.3);
        assert_eq!(controller.episode(), 0);
    }

    #[test]
    fn test_action_selection() {
        let config = RLConfig::default();
        let controller = UniversalRLController::new(config);

        let state = UniversalRLState::new();
        let action = controller.select_action(&state, "Phase0-DendriticReservoir");

        // Should return a valid action (no panic)
        // Updated: 88 total actions (was 80, added 8 Geometry actions)
        assert!(action.to_index() < 88);
    }

    #[test]
    fn test_qtable_update() {
        let config = RLConfig::default();
        let controller = UniversalRLController::new(config);

        let state = UniversalRLState::new();
        let mut next_state = state.clone();
        next_state.chromatic_number = 5; // Make state different

        let action =
            UniversalAction::Phase0(super::super::actions::DendriticAction::IncreaseLearnRate);

        controller.update_qtable(
            &state,
            &action,
            1.0,
            &next_state,
            "Phase0-DendriticReservoir",
        );

        let (mean, _, _) = controller.qtable_stats("Phase0-DendriticReservoir");
        assert!(mean > 0.0); // Q-value should have been updated
    }

    #[test]
    fn test_epsilon_decay() {
        let mut config = RLConfig::default();
        config.epsilon = 1.0;
        config.epsilon_decay = 0.9;
        config.epsilon_min = 0.1;

        let controller = UniversalRLController::new(config);

        assert_eq!(controller.epsilon(), 1.0);

        controller.decay_epsilon();
        assert_eq!(controller.epsilon(), 0.9);

        for _ in 0..20 {
            controller.decay_epsilon();
        }

        assert!(controller.epsilon() >= 0.1); // Should not go below min
    }

    #[test]
    fn test_initialize_from_curriculum() {
        let config = RLConfig::default();
        let controller = UniversalRLController::new(config);

        // Create a sample curriculum Q-table
        let mut curriculum_qtable = HashMap::new();
        let mut actions = HashMap::new();
        actions.insert(0, 0.5);
        actions.insert(1, 0.7);
        actions.insert(2, 0.3);
        curriculum_qtable.insert(12345u64, actions.clone());
        curriculum_qtable.insert(67890u64, actions.clone());

        // Initialize Phase0 from curriculum
        let result =
            controller.initialize_from_curriculum(&curriculum_qtable, "Phase0-DendriticReservoir");
        assert!(result.is_ok(), "Curriculum initialization should succeed");

        // Verify Q-values were loaded (check stats)
        let (mean, min, max) = controller.qtable_stats("Phase0-DendriticReservoir");
        assert!(
            mean > 0.0,
            "Mean Q-value should be positive after curriculum load"
        );
        assert!(max > 0.0, "Max Q-value should be positive");
        assert!(min >= 0.0, "Min Q-value should be non-negative");
    }

    #[test]
    fn test_initialize_all_phases_from_curriculum() {
        let config = RLConfig::default();
        let controller = UniversalRLController::new(config);

        // Create a sample curriculum Q-table
        let mut curriculum_qtable = HashMap::new();
        let mut actions = HashMap::new();
        actions.insert(0, 0.8);
        actions.insert(5, 0.6);
        curriculum_qtable.insert(11111u64, actions);

        // Initialize all phases from curriculum
        let result = controller.initialize_all_phases_from_curriculum(&curriculum_qtable);
        assert!(
            result.is_ok(),
            "Curriculum initialization for all phases should succeed"
        );

        // Verify all phases have Q-values loaded
        for phase in &[
            "Phase0-DendriticReservoir",
            "Phase1-ActiveInference",
            "Phase2-Thermodynamic",
            "Phase3-QuantumClassical",
            "Phase4-Geodesic",
            "Phase6-TDA",
            "Phase7-Ensemble",
        ] {
            let (mean, _, max) = controller.qtable_stats(phase);
            assert!(
                mean > 0.0,
                "Phase {} should have positive mean Q-value",
                phase
            );
            assert!(
                max > 0.0,
                "Phase {} should have positive max Q-value",
                phase
            );
        }
    }
}

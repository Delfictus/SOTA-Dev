//! Ultra FluxNet Controller
//!
//! Unified RL controller that manages all PRISM phases using the RuntimeConfig
//! struct for action space and KernelTelemetry for state observation.
//!
//! This is the single, centralized RL controller that learns optimal parameter
//! adjustments across all 7 PRISM phases. It uses Q-learning with discrete
//! state/action spaces and optional MBRL integration for planning.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use prism_core::{KernelTelemetry, RuntimeConfig};

#[cfg(feature = "mbrl")]
use crate::mbrl::DynaFluxNet;
use crate::mbrl_integration::MBRLIntegration;

/// State discretization for Q-table
///
/// Discretizes continuous telemetry into bucketed features for efficient Q-learning.
/// The discretization balances state space size with expressiveness.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct DiscreteState {
    /// Conflict bucket (0 = 0-10, 1 = 11-50, 2 = 51-200, 3 = 201+)
    pub conflict_bucket: u8,
    /// Color bucket (0 = 0-20, 1 = 21-40, 2 = 41-60, 3 = 61+)
    pub color_bucket: u8,
    /// Temperature bucket (0 = <0.1, 1 = 0.1-1.0, 2 = 1.0-10.0, 3 = 10.0+)
    pub temp_bucket: u8,
    /// Phase transition flag (true if transitions detected this iteration)
    pub transition_active: bool,
    /// Stagnation counter bucket (0 = 0-10, 1 = 11-50, 2 = 51-100, 3 = 101+)
    pub stagnation_bucket: u8,
}

impl DiscreteState {
    /// Create discrete state from telemetry and config
    ///
    /// # Arguments
    /// * `telemetry` - Current kernel telemetry
    /// * `config` - Current runtime configuration
    /// * `stagnation` - Stagnation counter from controller
    pub fn from_telemetry(
        telemetry: &KernelTelemetry,
        config: &RuntimeConfig,
        stagnation: usize,
    ) -> Self {
        Self {
            conflict_bucket: match telemetry.conflicts {
                0..=10 => 0,
                11..=50 => 1,
                51..=200 => 2,
                _ => 3,
            },
            color_bucket: match telemetry.colors_used {
                0..=20 => 0,
                21..=40 => 1,
                41..=60 => 2,
                _ => 3,
            },
            temp_bucket: if config.global_temperature < 0.1 {
                0
            } else if config.global_temperature < 1.0 {
                1
            } else if config.global_temperature < 10.0 {
                2
            } else {
                3
            },
            transition_active: telemetry.phase_transitions > 0,
            stagnation_bucket: match stagnation {
                0..=10 => 0,
                11..=50 => 1,
                51..=100 => 2,
                _ => 3,
            },
        }
    }
}

/// Discrete action for Q-table
///
/// Each action represents a parameter adjustment to the RuntimeConfig.
/// Actions are applied multiplicatively or additively depending on the parameter.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum DiscreteAction {
    /// Increase chemical potential by 20%
    IncreaseChemicalPotential,
    /// Decrease chemical potential by 20%
    DecreaseChemicalPotential,
    /// Increase tunneling probability by 50%
    IncreaseTunneling,
    /// Decrease tunneling probability by 30%
    DecreaseTunneling,
    /// Increase temperature by 50%
    IncreaseTemperature,
    /// Decrease temperature by 20%
    DecreaseTemperature,
    /// Boost reservoir influence by 20%
    BoostReservoir,
    /// Reduce reservoir influence by 20%
    ReduceReservoir,
    /// Enable phase transition response (boost tunneling)
    EnableTransitionResponse,
    /// Disable phase transition response (normal tunneling)
    DisableTransitionResponse,
    /// No change (exploit current configuration)
    NoOp,
}

impl DiscreteAction {
    /// All possible actions (for iteration)
    pub const ALL: [DiscreteAction; 11] = [
        DiscreteAction::IncreaseChemicalPotential,
        DiscreteAction::DecreaseChemicalPotential,
        DiscreteAction::IncreaseTunneling,
        DiscreteAction::DecreaseTunneling,
        DiscreteAction::IncreaseTemperature,
        DiscreteAction::DecreaseTemperature,
        DiscreteAction::BoostReservoir,
        DiscreteAction::ReduceReservoir,
        DiscreteAction::EnableTransitionResponse,
        DiscreteAction::DisableTransitionResponse,
        DiscreteAction::NoOp,
    ];

    /// Apply action to config (in-place mutation)
    ///
    /// Each action modifies specific RuntimeConfig fields with appropriate
    /// clamping to ensure valid parameter ranges.
    pub fn apply(&self, config: &mut RuntimeConfig) {
        match self {
            DiscreteAction::IncreaseChemicalPotential => {
                config.chemical_potential = (config.chemical_potential * 1.2).min(10.0);
            }
            DiscreteAction::DecreaseChemicalPotential => {
                config.chemical_potential = (config.chemical_potential * 0.8).max(0.1);
            }
            DiscreteAction::IncreaseTunneling => {
                config.tunneling_prob_base = (config.tunneling_prob_base * 1.5).min(0.5);
            }
            DiscreteAction::DecreaseTunneling => {
                config.tunneling_prob_base = (config.tunneling_prob_base * 0.7).max(0.01);
            }
            DiscreteAction::IncreaseTemperature => {
                config.global_temperature *= 1.5;
            }
            DiscreteAction::DecreaseTemperature => {
                config.global_temperature = (config.global_temperature * 0.8).max(0.01);
            }
            DiscreteAction::BoostReservoir => {
                config.reservoir_leak_rate = (config.reservoir_leak_rate * 1.2).min(0.9);
            }
            DiscreteAction::ReduceReservoir => {
                config.reservoir_leak_rate = (config.reservoir_leak_rate * 0.8).max(0.1);
            }
            DiscreteAction::EnableTransitionResponse => {
                config.tunneling_prob_boost = 3.0;
            }
            DiscreteAction::DisableTransitionResponse => {
                config.tunneling_prob_boost = 1.0;
            }
            DiscreteAction::NoOp => {}
        }
    }
}

/// Ultra FluxNet Controller
///
/// Central Q-learning controller for all PRISM phases. Uses temporal difference
/// learning to optimize RuntimeConfig parameters based on KernelTelemetry feedback.
///
/// # Architecture
/// - **Q-table**: HashMap from (state, action) → Q-value
/// - **Exploration**: Epsilon-greedy with decay
/// - **Learning**: Standard Q-learning update rule
/// - **Optional MBRL**: Integration with DynaFluxNet for planning
///
/// # Usage
/// ```no_run
/// use prism_fluxnet::UltraFluxNetController;
/// use prism_core::{RuntimeConfig, KernelTelemetry};
///
/// let mut controller = UltraFluxNetController::new();
/// let mut config = RuntimeConfig::production();
///
/// // Training loop
/// for iteration in 0..1000 {
///     // Run kernel with current config
///     let telemetry = KernelTelemetry::new(); // From GPU
///
///     // Select action
///     let action = controller.select_action(&telemetry, &config);
///
///     // Apply action
///     action.apply(&mut config);
///
///     // Update Q-values
///     controller.update(&telemetry, &config);
/// }
///
/// // Use best learned configuration
/// if let Some(best) = controller.best_config() {
///     config = *best;
/// }
/// ```
pub struct UltraFluxNetController {
    /// Q-table: state → action → value
    q_table: HashMap<(DiscreteState, DiscreteAction), f64>,

    /// Learning rate (alpha)
    alpha: f64,

    /// Discount factor (gamma)
    gamma: f64,

    /// Exploration rate (epsilon)
    epsilon: f64,

    /// Epsilon decay rate per update
    epsilon_decay: f64,

    /// Minimum epsilon (never decay below this)
    epsilon_min: f64,

    /// Previous state for TD update
    prev_state: Option<DiscreteState>,

    /// Previous action for TD update
    prev_action: Option<DiscreteAction>,

    /// Best configuration seen so far
    best_config: Option<RuntimeConfig>,

    /// Best conflicts achieved
    best_conflicts: i32,

    /// Stagnation counter (iterations without improvement)
    stagnation_counter: usize,

    /// MBRL world model (optional, requires "mbrl" feature)
    #[cfg(feature = "mbrl")]
    world_model: Option<DynaFluxNet>,

    /// MBRL integration (always available, gracefully degrades)
    mbrl_integration: MBRLIntegration,

    /// Use MBRL for action selection (when available)
    use_mbrl_planning: bool,
}

impl UltraFluxNetController {
    /// Create a new controller with default hyperparameters
    ///
    /// # Returns
    /// A fresh controller with:
    /// - Learning rate: 0.1
    /// - Discount: 0.95
    /// - Initial epsilon: 0.3
    /// - Epsilon decay: 0.999
    /// - Min epsilon: 0.05
    pub fn new() -> Self {
        Self {
            q_table: HashMap::new(),
            alpha: 0.1,
            gamma: 0.95,
            epsilon: 0.3,
            epsilon_decay: 0.999,
            epsilon_min: 0.05,
            prev_state: None,
            prev_action: None,
            best_config: None,
            best_conflicts: i32::MAX,
            stagnation_counter: 0,
            #[cfg(feature = "mbrl")]
            world_model: None,
            mbrl_integration: MBRLIntegration::new(),
            use_mbrl_planning: true, // Enable MBRL by default (auto-falls back if unavailable)
        }
    }

    /// Attach MBRL world model for Dyna-style learning
    ///
    /// Enables model-based planning by integrating synthetic experience
    /// generation with real experience.
    ///
    /// # Arguments
    /// * `model` - DynaFluxNet world model
    ///
    /// # Feature
    /// Requires "mbrl" feature flag
    #[cfg(feature = "mbrl")]
    pub fn with_world_model(mut self, model: DynaFluxNet) -> Self {
        self.world_model = Some(model);
        self
    }

    /// Select action given current telemetry and config
    ///
    /// Uses epsilon-greedy exploration: with probability epsilon, selects
    /// a random action; otherwise, selects the action with highest Q-value.
    ///
    /// # Arguments
    /// * `telemetry` - Current kernel telemetry
    /// * `config` - Current runtime configuration
    ///
    /// # Returns
    /// The selected DiscreteAction to apply
    pub fn select_action(
        &mut self,
        telemetry: &KernelTelemetry,
        config: &RuntimeConfig,
    ) -> DiscreteAction {
        let state = DiscreteState::from_telemetry(telemetry, config, self.stagnation_counter);

        // Try MBRL planning first (if enabled and available)
        let action = if self.use_mbrl_planning {
            if let Some(mbrl_action) = self.mbrl_integration.predict_best_action(
                telemetry,
                config,
                self.stagnation_counter,
            ) {
                log::debug!("Using MBRL-planned action: {:?}", mbrl_action);
                mbrl_action
            } else {
                // MBRL unavailable, fall back to Q-learning
                self.select_action_epsilon_greedy(&state)
            }
        } else {
            // MBRL disabled, use pure Q-learning
            self.select_action_epsilon_greedy(&state)
        };

        // Store for next update
        self.prev_state = Some(state);
        self.prev_action = Some(action);

        action
    }

    /// Epsilon-greedy action selection (internal helper)
    fn select_action_epsilon_greedy(&self, state: &DiscreteState) -> DiscreteAction {
        if rand::random::<f64>() < self.epsilon {
            // Explore: random action
            DiscreteAction::ALL[rand::random::<usize>() % DiscreteAction::ALL.len()]
        } else {
            // Exploit: best Q-value action
            self.best_action(state)
        }
    }

    /// Update Q-values after observing reward
    ///
    /// Performs temporal difference update using standard Q-learning:
    /// Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
    ///
    /// Also tracks best configuration and decays epsilon.
    ///
    /// # Arguments
    /// * `telemetry` - New kernel telemetry after action
    /// * `config` - New runtime configuration after action
    pub fn update(&mut self, telemetry: &KernelTelemetry, config: &RuntimeConfig) {
        // Compute reward
        let reward = self.compute_reward(telemetry);

        // Track best configuration
        if telemetry.conflicts < self.best_conflicts {
            self.best_conflicts = telemetry.conflicts;
            self.best_config = Some(*config);
            self.stagnation_counter = 0;
        } else {
            self.stagnation_counter += 1;
        }

        // TD update
        if let (Some(prev_state), Some(prev_action)) = (&self.prev_state, &self.prev_action) {
            let next_state =
                DiscreteState::from_telemetry(telemetry, config, self.stagnation_counter);
            let next_best_q = self.best_q_value(&next_state);

            let key = (prev_state.clone(), *prev_action);
            let current_q = *self.q_table.get(&key).unwrap_or(&0.0);

            // Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
            let new_q = current_q + self.alpha * (reward + self.gamma * next_best_q - current_q);
            self.q_table.insert(key, new_q);
        }

        // Decay epsilon
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);

        // MBRL: Generate synthetic experience (if enabled)
        #[cfg(feature = "mbrl")]
        if let Some(ref mut _world_model) = self.world_model {
            // Synthetic experience generation would happen here
            // This requires conversion between DiscreteAction and RuntimeConfigDelta
            // Omitted for initial implementation
        }
    }

    /// Compute reward from telemetry
    ///
    /// Reward function:
    /// - Primary: Minimize conflicts (large bonus for zero)
    /// - Secondary: Minimize colors used
    /// - Tertiary: Encourage active optimization (moves applied)
    /// - Penalty: Stagnation (no improvement for many iterations)
    ///
    /// # Arguments
    /// * `telemetry` - Current kernel telemetry
    ///
    /// # Returns
    /// Scalar reward value
    fn compute_reward(&self, telemetry: &KernelTelemetry) -> f64 {
        let mut reward = 0.0;

        // Primary: conflict reduction
        if telemetry.conflicts == 0 {
            reward += 100.0; // Big bonus for zero conflicts
        } else {
            reward -= telemetry.conflicts as f64 * 0.1;
        }

        // Secondary: color efficiency
        reward -= telemetry.colors_used as f64 * 0.01;

        // Bonus for moves applied (active optimization)
        reward += telemetry.moves_applied as f64 * 0.001;

        // Penalty for stagnation
        if self.stagnation_counter > 100 {
            reward -= 10.0;
        }

        reward
    }

    /// Get best action for state
    ///
    /// Finds the action with maximum Q-value for the given state.
    /// If no Q-values exist for this state, defaults to NoOp.
    ///
    /// # Arguments
    /// * `state` - Current discrete state
    ///
    /// # Returns
    /// Action with highest Q-value
    fn best_action(&self, state: &DiscreteState) -> DiscreteAction {
        let mut best_action = DiscreteAction::NoOp;
        let mut best_value = f64::NEG_INFINITY;

        for action in DiscreteAction::ALL.iter() {
            let key = (state.clone(), *action);
            let value = *self.q_table.get(&key).unwrap_or(&0.0);
            if value > best_value {
                best_value = value;
                best_action = *action;
            }
        }

        best_action
    }

    /// Get best Q-value for state
    ///
    /// Finds the maximum Q-value across all actions for the given state.
    ///
    /// # Arguments
    /// * `state` - Current discrete state
    ///
    /// # Returns
    /// Maximum Q-value (or -∞ if no entries)
    fn best_q_value(&self, state: &DiscreteState) -> f64 {
        DiscreteAction::ALL
            .iter()
            .map(|a| *self.q_table.get(&(state.clone(), *a)).unwrap_or(&0.0))
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get current best configuration
    ///
    /// Returns the configuration that achieved the lowest conflict count
    /// during training.
    ///
    /// # Returns
    /// Reference to best RuntimeConfig, or None if no runs completed
    pub fn best_config(&self) -> Option<&RuntimeConfig> {
        self.best_config.as_ref()
    }

    /// Get Q-table size
    ///
    /// Returns the number of (state, action) pairs with Q-values.
    ///
    /// # Returns
    /// Number of learned Q-values
    pub fn q_table_size(&self) -> usize {
        self.q_table.len()
    }

    /// Get current epsilon
    ///
    /// # Returns
    /// Current exploration rate
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// Get best conflicts achieved
    ///
    /// # Returns
    /// Minimum conflict count seen
    pub fn best_conflicts(&self) -> i32 {
        self.best_conflicts
    }

    /// Reset for new graph/episode
    ///
    /// Clears episode-specific state but keeps Q-table for transfer learning.
    /// This allows the controller to leverage learned knowledge across different
    /// graph instances.
    pub fn reset_episode(&mut self) {
        self.prev_state = None;
        self.prev_action = None;
        self.best_conflicts = i32::MAX;
        self.stagnation_counter = 0;
        // Keep Q-table for transfer learning
    }

    /// Reset everything (including Q-table)
    ///
    /// Complete reset to initial state. Use this to start fresh training.
    pub fn reset_all(&mut self) {
        self.q_table.clear();
        self.reset_episode();
        self.best_config = None;
        self.epsilon = 0.3;
    }

    /// Save Q-table to file
    ///
    /// Serializes the Q-table using bincode for efficient storage.
    ///
    /// # Arguments
    /// * `path` - File path to save to
    ///
    /// # Returns
    /// Result indicating success or IO error
    pub fn save(&self, path: &str) -> Result<()> {
        let data = bincode::serialize(&self.q_table)?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Load Q-table from file
    ///
    /// Deserializes a previously saved Q-table.
    ///
    /// # Arguments
    /// * `path` - File path to load from
    ///
    /// # Returns
    /// Result indicating success or IO/deserialization error
    pub fn load(&mut self, path: &str) -> Result<()> {
        let data = std::fs::read(path)?;
        self.q_table = bincode::deserialize(&data)?;
        Ok(())
    }

    /// Set learning rate
    ///
    /// # Arguments
    /// * `alpha` - New learning rate (typically 0.01 - 0.3)
    pub fn set_alpha(&mut self, alpha: f64) {
        self.alpha = alpha.clamp(0.0, 1.0);
    }

    /// Set discount factor
    ///
    /// # Arguments
    /// * `gamma` - New discount factor (typically 0.9 - 0.99)
    pub fn set_gamma(&mut self, gamma: f64) {
        self.gamma = gamma.clamp(0.0, 1.0);
    }

    /// Set epsilon (exploration rate)
    ///
    /// # Arguments
    /// * `epsilon` - New exploration rate (typically 0.05 - 0.5)
    pub fn set_epsilon(&mut self, epsilon: f64) {
        self.epsilon = epsilon.clamp(self.epsilon_min, 1.0);
    }

    /// Enable MBRL planning
    ///
    /// When enabled, the controller will attempt to use MBRL world model predictions
    /// for action selection. Gracefully falls back to Q-learning if unavailable.
    pub fn enable_mbrl_planning(&mut self) {
        self.use_mbrl_planning = true;
        log::info!("MBRL planning enabled");
    }

    /// Disable MBRL planning
    ///
    /// Forces pure Q-learning (epsilon-greedy) action selection, even if MBRL
    /// world model is available.
    pub fn disable_mbrl_planning(&mut self) {
        self.use_mbrl_planning = false;
        log::info!("MBRL planning disabled (using pure Q-learning)");
    }

    /// Check if MBRL is available
    ///
    /// # Returns
    /// `true` if MBRL world model is loaded and functional
    pub fn is_mbrl_available(&self) -> bool {
        self.mbrl_integration.is_mbrl_available()
    }

    /// Get MBRL status string
    ///
    /// # Returns
    /// Human-readable status of MBRL integration
    pub fn mbrl_status(&self) -> &'static str {
        self.mbrl_integration.status()
    }

    /// Get mutable reference to MBRL integration
    ///
    /// Allows advanced configuration of MBRL parameters (planning horizon,
    /// number of candidates, etc.)
    ///
    /// # Returns
    /// Mutable reference to MBRLIntegration
    pub fn mbrl_integration_mut(&mut self) -> &mut MBRLIntegration {
        &mut self.mbrl_integration
    }

    /// Get reference to MBRL integration
    ///
    /// # Returns
    /// Reference to MBRLIntegration
    pub fn mbrl_integration(&self) -> &MBRLIntegration {
        &self.mbrl_integration
    }
}

impl Default for UltraFluxNetController {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_discrete_state_creation() {
        let telemetry = KernelTelemetry {
            conflicts: 25,
            colors_used: 35,
            phase_transitions: 2,
            ..Default::default()
        };
        let mut config = RuntimeConfig::production();
        // Production config has global_temperature = 1.0, which maps to bucket 2 (1.0-10.0)
        // Set to 0.5 to test bucket 1
        config.global_temperature = 0.5;
        let state = DiscreteState::from_telemetry(&telemetry, &config, 5);

        assert_eq!(state.conflict_bucket, 1); // 11-50
        assert_eq!(state.color_bucket, 1); // 21-40
        assert_eq!(state.temp_bucket, 1); // 0.1-1.0
        assert!(state.transition_active);
        assert_eq!(state.stagnation_bucket, 0); // 0-10
    }

    #[test]
    fn test_discrete_action_apply() {
        let mut config = RuntimeConfig::production();
        let initial_chemical = config.chemical_potential;

        DiscreteAction::IncreaseChemicalPotential.apply(&mut config);
        assert!(config.chemical_potential > initial_chemical);

        DiscreteAction::DecreaseChemicalPotential.apply(&mut config);
        assert!(config.chemical_potential < initial_chemical * 1.2);
    }

    #[test]
    fn test_controller_creation() {
        let controller = UltraFluxNetController::new();
        assert_eq!(controller.q_table_size(), 0);
        assert_eq!(controller.epsilon(), 0.3);
        assert_eq!(controller.best_conflicts(), i32::MAX);
    }

    #[test]
    fn test_action_selection() {
        let mut controller = UltraFluxNetController::new();
        let telemetry = KernelTelemetry::default();
        let config = RuntimeConfig::production();

        let action = controller.select_action(&telemetry, &config);
        assert!(DiscreteAction::ALL.contains(&action));
    }

    #[test]
    fn test_reward_computation() {
        let controller = UltraFluxNetController::new();

        // Zero conflicts should give high reward
        let telemetry_good = KernelTelemetry {
            conflicts: 0,
            colors_used: 20,
            moves_applied: 100,
            ..Default::default()
        };
        let reward_good = controller.compute_reward(&telemetry_good);
        assert!(reward_good > 90.0); // ~100 - 0.2 + 0.1

        // Many conflicts should give negative reward
        let telemetry_bad = KernelTelemetry {
            conflicts: 100,
            colors_used: 50,
            ..Default::default()
        };
        let reward_bad = controller.compute_reward(&telemetry_bad);
        assert!(reward_bad < 0.0);
    }

    #[test]
    fn test_episode_reset() {
        let mut controller = UltraFluxNetController::new();
        let telemetry = KernelTelemetry::default();
        let config = RuntimeConfig::production();

        // Execute one step
        controller.select_action(&telemetry, &config);
        controller.update(&telemetry, &config);

        assert!(controller.prev_state.is_some());

        // Reset episode
        controller.reset_episode();
        assert!(controller.prev_state.is_none());
        assert_eq!(controller.best_conflicts(), i32::MAX);
    }

    #[test]
    fn test_all_actions_count() {
        assert_eq!(DiscreteAction::ALL.len(), 11);
    }

    #[test]
    fn test_hyperparameter_setters() {
        let mut controller = UltraFluxNetController::new();

        controller.set_alpha(0.2);
        assert_eq!(controller.alpha, 0.2);

        controller.set_gamma(0.99);
        assert_eq!(controller.gamma, 0.99);

        controller.set_epsilon(0.1);
        assert_eq!(controller.epsilon(), 0.1);

        // Test clamping
        controller.set_alpha(2.0);
        assert_eq!(controller.alpha, 1.0);

        controller.set_epsilon(0.01); // Below min
        assert_eq!(controller.epsilon(), 0.05);
    }
}

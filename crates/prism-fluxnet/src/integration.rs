//! Integration layer for UltraFluxNetController
//!
//! Provides a unified interface that bridges the UltraFluxNetController with
//! the existing pipeline architecture. This allows both the legacy UniversalRLController
//! and the new UltraFluxNetController to coexist during the transition period.

use anyhow::Result;
use prism_core::{KernelTelemetry, RuntimeConfig};

use crate::ultra_controller::{DiscreteAction, UltraFluxNetController};
use crate::core::state::UniversalRLState;
use crate::core::controller::UniversalRLController;

/// Controller selection mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControllerMode {
    /// Use the legacy UniversalRLController (per-phase Q-tables)
    Universal,
    /// Use the new UltraFluxNetController (unified RuntimeConfig-based)
    Ultra,
    /// Use both controllers in parallel (experimental)
    Hybrid,
}

/// Integrated FluxNet controller that wraps both implementations
///
/// This wrapper provides a unified interface for the pipeline orchestrator,
/// allowing seamless switching between the legacy controller and the new
/// Ultra controller.
///
/// # Architecture
/// ```text
/// ┌─────────────────────────────────────────┐
/// │     IntegratedFluxNet (Facade)          │
/// ├─────────────────────────────────────────┤
/// │  mode: ControllerMode                   │
/// │  ┌───────────────┐  ┌─────────────────┐│
/// │  │ Universal     │  │ Ultra           ││
/// │  │ (per-phase)   │  │ (unified)       ││
/// │  └───────────────┘  └─────────────────┘│
/// └─────────────────────────────────────────┘
///          │                      │
///          ▼                      ▼
///    Phase-specific        RuntimeConfig
///    Q-tables              mutations
/// ```
pub struct IntegratedFluxNet {
    /// Operation mode
    mode: ControllerMode,

    /// Legacy controller (used when mode = Universal or Hybrid)
    universal: Option<UniversalRLController>,

    /// New unified controller (used when mode = Ultra or Hybrid)
    ultra: Option<UltraFluxNetController>,

    /// Current runtime configuration (for Ultra controller)
    current_config: RuntimeConfig,

    /// Last telemetry (for reward computation)
    last_telemetry: Option<KernelTelemetry>,
}

impl IntegratedFluxNet {
    /// Create a new integrated controller in Universal mode (legacy)
    ///
    /// This is the default constructor for backward compatibility.
    pub fn new_universal(universal: UniversalRLController) -> Self {
        Self {
            mode: ControllerMode::Universal,
            universal: Some(universal),
            ultra: None,
            current_config: RuntimeConfig::production(),
            last_telemetry: None,
        }
    }

    /// Create a new integrated controller in Ultra mode (new)
    ///
    /// This is the recommended constructor for new pipelines.
    pub fn new_ultra() -> Self {
        Self {
            mode: ControllerMode::Ultra,
            universal: None,
            ultra: Some(UltraFluxNetController::new()),
            current_config: RuntimeConfig::production(),
            last_telemetry: None,
        }
    }

    /// Create a new integrated controller in Ultra mode with custom config
    pub fn new_ultra_with_config(config: RuntimeConfig) -> Self {
        Self {
            mode: ControllerMode::Ultra,
            universal: None,
            ultra: Some(UltraFluxNetController::new()),
            current_config: config,
            last_telemetry: None,
        }
    }

    /// Create a new integrated controller in Hybrid mode (experimental)
    ///
    /// In hybrid mode, both controllers run in parallel. The Ultra controller
    /// modifies RuntimeConfig, while the Universal controller provides
    /// phase-specific guidance.
    pub fn new_hybrid(universal: UniversalRLController) -> Self {
        Self {
            mode: ControllerMode::Hybrid,
            universal: Some(universal),
            ultra: Some(UltraFluxNetController::new()),
            current_config: RuntimeConfig::production(),
            last_telemetry: None,
        }
    }

    /// Get the current controller mode
    pub fn mode(&self) -> ControllerMode {
        self.mode
    }

    /// Select action for the given state and phase
    ///
    /// This is the unified interface called by the pipeline orchestrator.
    /// Behavior depends on the controller mode:
    ///
    /// - **Universal**: Uses UniversalRLController (returns UniversalAction)
    /// - **Ultra**: Uses UltraFluxNetController (returns DiscreteAction, modifies config)
    /// - **Hybrid**: Uses both (Ultra for config, Universal for phase-specific)
    ///
    /// # Arguments
    /// * `state` - Current RL state (for Universal controller)
    /// * `telemetry` - Current kernel telemetry (for Ultra controller)
    /// * `phase_name` - Current phase name
    ///
    /// # Returns
    /// The selected action (type depends on mode)
    pub fn select_action_universal(
        &mut self,
        state: &UniversalRLState,
        _telemetry: &KernelTelemetry,
        phase_name: &str,
    ) -> Option<crate::core::actions::UniversalAction> {
        match self.mode {
            ControllerMode::Universal => {
                if let Some(ref controller) = self.universal {
                    Some(controller.select_action(state, phase_name))
                } else {
                    log::error!("Universal controller not initialized in Universal mode");
                    None
                }
            }
            ControllerMode::Ultra => {
                // Ultra mode doesn't use UniversalAction, return None
                // Caller should use select_action_ultra instead
                None
            }
            ControllerMode::Hybrid => {
                // In hybrid mode, return Universal action
                if let Some(ref controller) = self.universal {
                    Some(controller.select_action(state, phase_name))
                } else {
                    None
                }
            }
        }
    }

    /// Select action using Ultra controller (modifies RuntimeConfig)
    ///
    /// This method is called when using Ultra or Hybrid mode. It returns
    /// a DiscreteAction that should be applied to the RuntimeConfig.
    ///
    /// # Arguments
    /// * `telemetry` - Current kernel telemetry
    ///
    /// # Returns
    /// The selected DiscreteAction (to be applied to config)
    pub fn select_action_ultra(&mut self, telemetry: &KernelTelemetry) -> Option<DiscreteAction> {
        match self.mode {
            ControllerMode::Universal => {
                // Universal mode doesn't use Ultra controller
                None
            }
            ControllerMode::Ultra | ControllerMode::Hybrid => {
                if let Some(ref mut ultra) = self.ultra {
                    let action = ultra.select_action(telemetry, &self.current_config);
                    Some(action)
                } else {
                    log::error!("Ultra controller not initialized in {:?} mode", self.mode);
                    None
                }
            }
        }
    }

    /// Apply Ultra action to the runtime configuration
    ///
    /// This mutates the internal RuntimeConfig based on the selected action.
    /// The updated config can be retrieved with `get_config()`.
    ///
    /// # Arguments
    /// * `action` - The DiscreteAction to apply
    pub fn apply_ultra_action(&mut self, action: DiscreteAction) {
        action.apply(&mut self.current_config);
        log::debug!("Applied Ultra action: {:?}", action);
    }

    /// Update controller(s) after phase execution
    ///
    /// This is called by the orchestrator after each phase completes.
    /// It updates Q-tables and performs learning.
    ///
    /// # Arguments
    /// * `state` - Previous RL state (for Universal controller)
    /// * `action` - Action taken (for Universal controller)
    /// * `reward` - Reward received
    /// * `next_state` - New RL state after action
    /// * `telemetry` - Current kernel telemetry (for Ultra controller)
    /// * `phase_name` - Current phase name
    pub fn update(
        &mut self,
        state: &UniversalRLState,
        action: &crate::core::actions::UniversalAction,
        reward: f32,
        next_state: &UniversalRLState,
        telemetry: &KernelTelemetry,
        phase_name: &str,
    ) {
        match self.mode {
            ControllerMode::Universal => {
                if let Some(ref controller) = self.universal {
                    controller.update_qtable(state, action, reward, next_state, phase_name);
                }
            }
            ControllerMode::Ultra => {
                if let Some(ref mut ultra) = self.ultra {
                    ultra.update(telemetry, &self.current_config);
                    self.last_telemetry = Some(*telemetry);
                }
            }
            ControllerMode::Hybrid => {
                // Update both controllers
                if let Some(ref controller) = self.universal {
                    controller.update_qtable(state, action, reward, next_state, phase_name);
                }
                if let Some(ref mut ultra) = self.ultra {
                    ultra.update(telemetry, &self.current_config);
                    self.last_telemetry = Some(*telemetry);
                }
            }
        }
    }

    /// Get the current runtime configuration
    ///
    /// This returns the config that has been modified by Ultra actions.
    /// The orchestrator should use this config for GPU kernel execution.
    pub fn get_config(&self) -> &RuntimeConfig {
        &self.current_config
    }

    /// Get a mutable reference to the runtime configuration
    pub fn get_config_mut(&mut self) -> &mut RuntimeConfig {
        &mut self.current_config
    }

    /// Update the runtime configuration
    ///
    /// This allows external modification of the config (e.g., from CLI args).
    pub fn set_config(&mut self, config: RuntimeConfig) {
        self.current_config = config;
    }

    /// Get reference to the Ultra controller (if available)
    pub fn ultra_controller(&self) -> Option<&UltraFluxNetController> {
        self.ultra.as_ref()
    }

    /// Get mutable reference to the Ultra controller (if available)
    pub fn ultra_controller_mut(&mut self) -> Option<&mut UltraFluxNetController> {
        self.ultra.as_mut()
    }

    /// Get reference to the Universal controller (if available)
    pub fn universal_controller(&self) -> Option<&UniversalRLController> {
        self.universal.as_ref()
    }

    /// Reset for new episode (clears episode-specific state)
    ///
    /// This should be called when starting a new graph instance.
    /// It resets episode state but preserves learned Q-tables.
    pub fn reset_episode(&mut self) {
        if let Some(ref mut ultra) = self.ultra {
            ultra.reset_episode();
        }
        self.last_telemetry = None;
        // Note: UniversalRLController doesn't have reset_episode
    }

    /// Perform experience replay (if using Universal controller)
    pub fn replay_batch(&self, phase_name: &str) {
        if let Some(ref controller) = self.universal {
            controller.replay_batch(phase_name);
        }
    }

    /// Decay epsilon (if using Universal controller)
    pub fn decay_epsilon(&self) {
        if let Some(ref controller) = self.universal {
            controller.decay_epsilon();
        }
    }

    /// Get current epsilon (exploration rate)
    pub fn epsilon(&self) -> f64 {
        match self.mode {
            ControllerMode::Universal | ControllerMode::Hybrid => {
                if let Some(ref controller) = self.universal {
                    controller.epsilon()
                } else {
                    0.0
                }
            }
            ControllerMode::Ultra => {
                if let Some(ref ultra) = self.ultra {
                    ultra.epsilon()
                } else {
                    0.0
                }
            }
        }
    }

    /// Get best conflicts achieved (Ultra controller only)
    pub fn best_conflicts(&self) -> Option<i32> {
        self.ultra.as_ref().map(|u| u.best_conflicts())
    }

    /// Get best configuration (Ultra controller only)
    pub fn best_config(&self) -> Option<&RuntimeConfig> {
        self.ultra.as_ref().and_then(|u| u.best_config())
    }

    /// Save controller state to disk
    ///
    /// Saves Q-tables or state depending on the controller mode.
    ///
    /// # Arguments
    /// * `path` - Base path for saving (extensions added automatically)
    pub fn save(&self, path: &str) -> Result<()> {
        match self.mode {
            ControllerMode::Universal => {
                if let Some(ref controller) = self.universal {
                    let qtable_path = format!("{}.universal.json", path);
                    controller.save_qtables_binary(&qtable_path)
                        .map_err(|e| anyhow::anyhow!("Failed to save Universal Q-tables: {}", e))?;
                    log::info!("Saved Universal Q-tables to {}", qtable_path);
                }
            }
            ControllerMode::Ultra => {
                if let Some(ref ultra) = self.ultra {
                    let qtable_path = format!("{}.ultra.bin", path);
                    ultra.save(&qtable_path)
                        .map_err(|e| anyhow::anyhow!("Failed to save Ultra Q-table: {}", e))?;
                    log::info!("Saved Ultra Q-table to {}", qtable_path);
                }
            }
            ControllerMode::Hybrid => {
                if let Some(ref controller) = self.universal {
                    let qtable_path = format!("{}.universal.json", path);
                    controller.save_qtables_binary(&qtable_path)
                        .map_err(|e| anyhow::anyhow!("Failed to save Universal Q-tables: {}", e))?;
                }
                if let Some(ref ultra) = self.ultra {
                    let qtable_path = format!("{}.ultra.bin", path);
                    ultra.save(&qtable_path)
                        .map_err(|e| anyhow::anyhow!("Failed to save Ultra Q-table: {}", e))?;
                }
                log::info!("Saved Hybrid Q-tables to {}", path);
            }
        }
        Ok(())
    }

    /// Load controller state from disk
    ///
    /// # Arguments
    /// * `path` - Base path for loading (extensions added automatically)
    pub fn load(&mut self, path: &str) -> Result<()> {
        match self.mode {
            ControllerMode::Universal => {
                if let Some(ref controller) = self.universal {
                    let qtable_path = format!("{}.universal.json", path);
                    controller.load_qtables_binary(&qtable_path)
                        .map_err(|e| anyhow::anyhow!("Failed to load Universal Q-tables: {}", e))?;
                    log::info!("Loaded Universal Q-tables from {}", qtable_path);
                }
            }
            ControllerMode::Ultra => {
                if let Some(ref mut ultra) = self.ultra {
                    let qtable_path = format!("{}.ultra.bin", path);
                    ultra.load(&qtable_path)
                        .map_err(|e| anyhow::anyhow!("Failed to load Ultra Q-table: {}", e))?;
                    log::info!("Loaded Ultra Q-table from {}", qtable_path);
                }
            }
            ControllerMode::Hybrid => {
                if let Some(ref controller) = self.universal {
                    let qtable_path = format!("{}.universal.json", path);
                    controller.load_qtables_binary(&qtable_path)
                        .map_err(|e| anyhow::anyhow!("Failed to load Universal Q-tables: {}", e))?;
                }
                if let Some(ref mut ultra) = self.ultra {
                    let qtable_path = format!("{}.ultra.bin", path);
                    ultra.load(&qtable_path)
                        .map_err(|e| anyhow::anyhow!("Failed to load Ultra Q-table: {}", e))?;
                }
                log::info!("Loaded Hybrid Q-tables from {}", path);
            }
        }
        Ok(())
    }

    /// Get Q-table statistics
    ///
    /// Returns (mean, min, max) Q-values for diagnostics.
    pub fn qtable_stats(&self, phase_name: &str) -> Option<(f32, f32, f32)> {
        match self.mode {
            ControllerMode::Universal | ControllerMode::Hybrid => {
                self.universal.as_ref().map(|c| c.qtable_stats(phase_name))
            }
            ControllerMode::Ultra => {
                // Ultra controller doesn't have per-phase Q-tables
                None
            }
        }
    }
}

impl Default for IntegratedFluxNet {
    fn default() -> Self {
        // Default to Ultra mode for new pipelines
        Self::new_ultra()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::controller::RLConfig;

    #[test]
    fn test_ultra_mode_creation() {
        let integrated = IntegratedFluxNet::new_ultra();
        assert_eq!(integrated.mode(), ControllerMode::Ultra);
        assert!(integrated.ultra_controller().is_some());
        assert!(integrated.universal_controller().is_none());
    }

    #[test]
    fn test_universal_mode_creation() {
        let config = RLConfig::default();
        let universal = UniversalRLController::new(config);
        let integrated = IntegratedFluxNet::new_universal(universal);

        assert_eq!(integrated.mode(), ControllerMode::Universal);
        assert!(integrated.ultra_controller().is_none());
        assert!(integrated.universal_controller().is_some());
    }

    #[test]
    fn test_hybrid_mode_creation() {
        let config = RLConfig::default();
        let universal = UniversalRLController::new(config);
        let integrated = IntegratedFluxNet::new_hybrid(universal);

        assert_eq!(integrated.mode(), ControllerMode::Hybrid);
        assert!(integrated.ultra_controller().is_some());
        assert!(integrated.universal_controller().is_some());
    }

    #[test]
    fn test_ultra_action_selection() {
        let mut integrated = IntegratedFluxNet::new_ultra();
        let telemetry = KernelTelemetry::default();

        let action = integrated.select_action_ultra(&telemetry);
        assert!(action.is_some());
    }

    #[test]
    fn test_config_mutation() {
        let mut integrated = IntegratedFluxNet::new_ultra();
        let initial_temp = integrated.get_config().global_temperature;

        integrated.apply_ultra_action(DiscreteAction::IncreaseTemperature);

        let new_temp = integrated.get_config().global_temperature;
        assert!(new_temp > initial_temp);
    }

    #[test]
    fn test_reset_episode() {
        let mut integrated = IntegratedFluxNet::new_ultra();

        // Execute some actions
        let telemetry = KernelTelemetry::default();
        integrated.select_action_ultra(&telemetry);

        // Reset should not panic
        integrated.reset_episode();
    }

    #[test]
    fn test_epsilon_access() {
        let integrated = IntegratedFluxNet::new_ultra();
        let epsilon = integrated.epsilon();
        assert!(epsilon >= 0.0 && epsilon <= 1.0);
    }
}

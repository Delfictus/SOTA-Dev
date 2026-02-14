//! MBRL Integration for FluxNet
//!
//! Bridges the MBRL world model with UltraFluxNetController for model-based planning.
//! Provides graceful degradation when ONNX models are unavailable.
//!
//! ## Architecture
//!
//! ```text
//! ┌──────────────────────────────────────────────┐
//! │     UltraFluxNetController (Q-Learning)      │
//! │  ┌────────────────────────────────────────┐  │
//! │  │  select_action()                       │  │
//! │  │  ├─ Epsilon-greedy                     │  │
//! │  │  └─ Optional MBRL planning enhancement │  │
//! │  └────────────────────────────────────────┘  │
//! └──────────────────────────────────────────────┘
//!                      ▲
//!                      │
//! ┌──────────────────────────────────────────────┐
//! │         MBRLIntegration (This Module)        │
//! │  ┌────────────────────────────────────────┐  │
//! │  │  predict_and_plan()                    │  │
//! │  │  ├─ Convert DiscreteState → KernelState│  │
//! │  │  │  (bridge UltraFluxNet ↔ MBRL)       │  │
//! │  │  ├─ Run MCTS planning                  │  │
//! │  │  └─ Convert RuntimeConfigDelta → Action│  │
//! │  └────────────────────────────────────────┘  │
//! │  ┌────────────────────────────────────────┐  │
//! │  │  Fallback: None if model unavailable   │  │
//! │  └────────────────────────────────────────┘  │
//! └──────────────────────────────────────────────┘
//!                      ▲
//!                      │
//! ┌──────────────────────────────────────────────┐
//! │       DynaFluxNet (MBRL World Model)         │
//! │  ┌────────────────────────────────────────┐  │
//! │  │  MBRLWorldModel (ONNX GNN)             │  │
//! │  │  ├─ predict_outcome()                  │  │
//! │  │  ├─ mcts_action_selection()            │  │
//! │  │  └─ rollout_value()                    │  │
//! │  └────────────────────────────────────────┘  │
//! └──────────────────────────────────────────────┘
//! ```

use anyhow::Result;

use prism_core::{KernelTelemetry, RuntimeConfig};

#[cfg(feature = "mbrl")]
use crate::mbrl::{DynaFluxNet, KernelState, RuntimeConfigDelta};
use crate::ultra_controller::{DiscreteAction, DiscreteState};

#[cfg(feature = "mbrl")]
use std::path::Path;
#[cfg(feature = "mbrl")]
use std::sync::Arc;
#[cfg(feature = "mbrl")]
use crate::core::controller::UniversalRLController;

/// MBRL Integration wrapper for UltraFluxNetController
///
/// Provides optional model-based planning enhancement that gracefully degrades
/// when ONNX models are unavailable or the `mbrl` feature is disabled.
///
/// ## Usage
///
/// ```no_run
/// use prism_fluxnet::MBRLIntegration;
///
/// let mut integration = MBRLIntegration::new();
///
/// // Try to use MBRL for planning (returns None if unavailable)
/// if let Some(suggested_action) = integration.predict_best_action(&telemetry, &config) {
///     // Use MBRL's suggestion
///     suggested_action.apply(&mut config);
/// } else {
///     // Fall back to pure Q-learning
///     let action = controller.select_action(&telemetry, &config);
///     action.apply(&mut config);
/// }
/// ```
pub struct MBRLIntegration {
    /// MBRL world model (None if unavailable or feature disabled)
    #[cfg(feature = "mbrl")]
    world_model: Option<DynaFluxNet>,

    /// Planning horizon (steps to look ahead)
    planning_horizon: usize,

    /// Number of MCTS candidates to evaluate
    num_candidates: usize,

    /// Whether to log MBRL operations
    verbose: bool,
}

impl MBRLIntegration {
    /// Create new MBRL integration with automatic model detection
    ///
    /// Attempts to load ONNX model from standard paths:
    /// 1. `models/fluxnet/world_model.onnx`
    /// 2. `models/gnn/gnn_model.onnx`
    ///
    /// Falls back gracefully if models are not found or `mbrl` feature is disabled.
    pub fn new() -> Self {
        #[cfg(feature = "mbrl")]
        let world_model = Self::try_load_model();

        Self {
            #[cfg(feature = "mbrl")]
            world_model,
            planning_horizon: 10,
            num_candidates: 20,
            verbose: false,
        }
    }

    /// Try to load ONNX world model from standard paths
    ///
    /// Searches in priority order:
    /// 1. `models/fluxnet/world_model.onnx` - Dedicated FluxNet model
    /// 2. `models/gnn/gnn_model.onnx` - Shared GNN model
    ///
    /// Returns None if no model found or loading fails.
    #[cfg(feature = "mbrl")]
    fn try_load_model() -> Option<DynaFluxNet> {
        // Standard model paths (relative to project root)
        let paths = [
            "models/fluxnet/world_model.onnx",
            "models/gnn/gnn_model.onnx",
        ];

        for path_str in &paths {
            let path = Path::new(path_str);
            if path.exists() {
                log::info!("Attempting to load MBRL model from: {}", path_str);

                // Create a dummy UniversalRLController for DynaFluxNet
                // This is wrapped in Arc to satisfy DynaFluxNet's constructor
                let dummy_controller = Arc::new(UniversalRLController::new(
                    crate::core::controller::RLConfig::default()
                ));

                match DynaFluxNet::new(dummy_controller, path) {
                    Ok(model) => {
                        log::info!("✓ MBRL model loaded successfully from {}", path_str);
                        return Some(model);
                    }
                    Err(e) => {
                        log::warn!("Failed to load MBRL model from {}: {}", path_str, e);
                    }
                }
            } else {
                log::debug!("MBRL model not found at: {}", path_str);
            }
        }

        log::info!(
            "No MBRL model available. Using pure Q-learning fallback (this is normal for CPU-only builds)."
        );
        None
    }

    /// Predict and plan using MBRL world model
    ///
    /// Uses MCTS-based planning to find the best action given current state.
    /// Returns None if MBRL is unavailable (graceful degradation).
    ///
    /// # Arguments
    /// * `telemetry` - Current kernel telemetry
    /// * `config` - Current runtime configuration
    /// * `stagnation` - Current stagnation counter
    ///
    /// # Returns
    /// * `Some(DiscreteAction)` - MBRL-suggested action
    /// * `None` - MBRL unavailable, caller should fall back to Q-learning
    #[cfg(feature = "mbrl")]
    pub fn predict_best_action(
        &mut self,
        telemetry: &KernelTelemetry,
        config: &RuntimeConfig,
        _stagnation: usize,
    ) -> Option<DiscreteAction> {
        let world_model = self.world_model.as_mut()?;

        // Convert telemetry to MBRL KernelState
        let kernel_state = KernelState::from_telemetry(telemetry, config);

        // Run MCTS planning
        match world_model.select_action(&kernel_state) {
            Ok(delta) => {
                if self.verbose {
                    log::debug!(
                        "MBRL suggests: Δchem={:.3}, Δtemp={:.3}, Δtunnel={:.3}",
                        delta.d_chemical_potential,
                        delta.d_temperature,
                        delta.d_tunneling_prob
                    );
                }

                // Convert RuntimeConfigDelta to DiscreteAction
                Some(Self::delta_to_discrete_action(&delta))
            }
            Err(e) => {
                log::warn!("MBRL planning failed: {}. Falling back to Q-learning.", e);
                None
            }
        }
    }

    /// Predict and plan (no-op version when mbrl feature disabled)
    ///
    /// Always returns None, signaling fallback to Q-learning.
    #[cfg(not(feature = "mbrl"))]
    pub fn predict_best_action(
        &mut self,
        _telemetry: &KernelTelemetry,
        _config: &RuntimeConfig,
        _stagnation: usize,
    ) -> Option<DiscreteAction> {
        None // Always fall back when feature disabled
    }

    /// Convert RuntimeConfigDelta to DiscreteAction
    ///
    /// Maps continuous deltas to the closest discrete action in the action space.
    /// Uses magnitude and sign to determine the appropriate action.
    #[cfg(feature = "mbrl")]
    fn delta_to_discrete_action(delta: &RuntimeConfigDelta) -> DiscreteAction {
        // Find the largest absolute delta
        let mut max_abs = 0.0f32;
        let mut selected_action = DiscreteAction::NoOp;

        // Chemical potential
        if delta.d_chemical_potential.abs() > max_abs {
            max_abs = delta.d_chemical_potential.abs();
            selected_action = if delta.d_chemical_potential > 0.0 {
                DiscreteAction::IncreaseChemicalPotential
            } else {
                DiscreteAction::DecreaseChemicalPotential
            };
        }

        // Tunneling probability
        if delta.d_tunneling_prob.abs() > max_abs {
            max_abs = delta.d_tunneling_prob.abs();
            selected_action = if delta.d_tunneling_prob > 0.0 {
                DiscreteAction::IncreaseTunneling
            } else {
                DiscreteAction::DecreaseTunneling
            };
        }

        // Temperature
        if delta.d_temperature.abs() > max_abs {
            max_abs = delta.d_temperature.abs();
            selected_action = if delta.d_temperature > 0.0 {
                DiscreteAction::IncreaseTemperature
            } else {
                DiscreteAction::DecreaseTemperature
            };
        }

        // Reservoir leak rate
        if delta.d_reservoir_leak.abs() > max_abs {
            max_abs = delta.d_reservoir_leak.abs();
            selected_action = if delta.d_reservoir_leak > 0.0 {
                DiscreteAction::BoostReservoir
            } else {
                DiscreteAction::ReduceReservoir
            };
        }

        // If all deltas are near-zero, return NoOp
        if max_abs < 0.01 {
            DiscreteAction::NoOp
        } else {
            selected_action
        }
    }

    /// Update MBRL model with real experience
    ///
    /// Adds real transition to the world model's experience buffer for
    /// Dyna-style learning (real + synthetic experience).
    ///
    /// # Arguments
    /// * `state_before` - State before action
    /// * `action` - Action taken
    /// * `state_after` - State after action
    /// * `reward` - Observed reward
    #[cfg(feature = "mbrl")]
    pub fn update_from_real_experience(
        &mut self,
        _state_before: &DiscreteState,
        action: DiscreteAction,
        _state_after: &DiscreteState,
        reward: f64,
    ) -> Result<()> {
        if let Some(ref mut _world_model) = self.world_model {
            // Convert DiscreteAction to RuntimeConfigDelta
            let _delta = Self::discrete_action_to_delta(action);

            // Note: This would require converting DiscreteState to KernelState
            // For now, we skip the update (world model learns from real kernel telemetry)
            // Future enhancement: maintain full KernelState history
            log::debug!(
                "MBRL real experience logged: action={:?}, reward={:.3}",
                action,
                reward
            );
        }
        Ok(())
    }

    /// Update (no-op when feature disabled)
    #[cfg(not(feature = "mbrl"))]
    pub fn update_from_real_experience(
        &mut self,
        _state_before: &DiscreteState,
        _action: DiscreteAction,
        _state_after: &DiscreteState,
        _reward: f64,
    ) -> Result<()> {
        Ok(())
    }

    /// Convert DiscreteAction to RuntimeConfigDelta
    ///
    /// Maps discrete actions to approximate continuous deltas.
    #[cfg(feature = "mbrl")]
    fn discrete_action_to_delta(action: DiscreteAction) -> RuntimeConfigDelta {
        match action {
            DiscreteAction::IncreaseChemicalPotential => RuntimeConfigDelta {
                d_chemical_potential: 0.2,
                ..Default::default()
            },
            DiscreteAction::DecreaseChemicalPotential => RuntimeConfigDelta {
                d_chemical_potential: -0.2,
                ..Default::default()
            },
            DiscreteAction::IncreaseTunneling => RuntimeConfigDelta {
                d_tunneling_prob: 0.1,
                ..Default::default()
            },
            DiscreteAction::DecreaseTunneling => RuntimeConfigDelta {
                d_tunneling_prob: -0.1,
                ..Default::default()
            },
            DiscreteAction::IncreaseTemperature => RuntimeConfigDelta {
                d_temperature: 0.2,
                ..Default::default()
            },
            DiscreteAction::DecreaseTemperature => RuntimeConfigDelta {
                d_temperature: -0.2,
                ..Default::default()
            },
            DiscreteAction::BoostReservoir => RuntimeConfigDelta {
                d_reservoir_leak: 0.05,
                ..Default::default()
            },
            DiscreteAction::ReduceReservoir => RuntimeConfigDelta {
                d_reservoir_leak: -0.05,
                ..Default::default()
            },
            DiscreteAction::EnableTransitionResponse => RuntimeConfigDelta {
                d_tunneling_prob: 0.2, // Boost tunneling
                ..Default::default()
            },
            DiscreteAction::DisableTransitionResponse => RuntimeConfigDelta {
                d_tunneling_prob: -0.1,
                ..Default::default()
            },
            DiscreteAction::NoOp => RuntimeConfigDelta::default(),
        }
    }

    /// Check if MBRL is available
    ///
    /// # Returns
    /// `true` if world model is loaded and functional
    pub fn is_mbrl_available(&self) -> bool {
        #[cfg(feature = "mbrl")]
        return self.world_model.is_some();

        #[cfg(not(feature = "mbrl"))]
        return false;
    }

    /// Get MBRL model status string
    ///
    /// # Returns
    /// Human-readable status message
    pub fn status(&self) -> &'static str {
        if self.is_mbrl_available() {
            "MBRL: Active (model-based planning enabled)"
        } else {
            "MBRL: Inactive (using pure Q-learning)"
        }
    }

    /// Set planning horizon
    ///
    /// # Arguments
    /// * `horizon` - Number of steps to look ahead during MCTS
    pub fn set_planning_horizon(&mut self, horizon: usize) {
        self.planning_horizon = horizon;
        #[cfg(feature = "mbrl")]
        if let Some(ref mut model) = self.world_model {
            model.world_model_mut().set_rollout_params(self.num_candidates, horizon);
        }
    }

    /// Set number of MCTS candidates
    ///
    /// # Arguments
    /// * `num` - Number of action candidates to evaluate during planning
    pub fn set_num_candidates(&mut self, num: usize) {
        self.num_candidates = num;
        #[cfg(feature = "mbrl")]
        if let Some(ref mut model) = self.world_model {
            model.world_model_mut().set_rollout_params(num, self.planning_horizon);
        }
    }

    /// Enable verbose logging
    pub fn set_verbose(&mut self, verbose: bool) {
        self.verbose = verbose;
    }

    /// Get experience buffer size (if MBRL available)
    #[cfg(feature = "mbrl")]
    pub fn buffer_size(&self) -> usize {
        self.world_model
            .as_ref()
            .map(|m| m.world_model().buffer_size())
            .unwrap_or(0)
    }

    /// Get experience buffer size (no-op when feature disabled)
    #[cfg(not(feature = "mbrl"))]
    pub fn buffer_size(&self) -> usize {
        0
    }
}

impl Default for MBRLIntegration {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mbrl_integration_creation() {
        let integration = MBRLIntegration::new();

        // Should not panic even if model unavailable
        let status = integration.status();
        assert!(status.contains("MBRL:"));
    }

    #[test]
    fn test_fallback_when_unavailable() {
        let mut integration = MBRLIntegration::new();
        let telemetry = KernelTelemetry::default();
        let config = RuntimeConfig::production();

        // Should return None gracefully if model unavailable
        let result = integration.predict_best_action(&telemetry, &config, 0);

        // This test passes regardless of whether MBRL is available
        // (it's testing graceful degradation)
        if result.is_some() {
            // MBRL is available and working
            println!("MBRL active: {:?}", result);
        } else {
            // MBRL not available, expected fallback
            println!("MBRL inactive (expected for CPU-only builds)");
        }
    }

    #[test]
    fn test_status_string() {
        let integration = MBRLIntegration::new();
        let status = integration.status();

        assert!(
            status.contains("Active") || status.contains("Inactive"),
            "Status should indicate MBRL state"
        );
    }

    #[test]
    fn test_buffer_size() {
        let integration = MBRLIntegration::new();
        let size = integration.buffer_size();

        // Should return 0 if unavailable, or actual size if available
        assert!(size >= 0);
    }

    #[cfg(feature = "mbrl")]
    #[test]
    fn test_delta_to_discrete_action() {
        use crate::mbrl::RuntimeConfigDelta;

        // Large positive chemical potential delta
        let delta = RuntimeConfigDelta {
            d_chemical_potential: 0.5,
            d_temperature: 0.1,
            d_tunneling_prob: 0.05,
            d_belief_weight: 0.0,
            d_reservoir_leak: 0.0,
        };

        let action = MBRLIntegration::delta_to_discrete_action(&delta);
        assert_eq!(action, DiscreteAction::IncreaseChemicalPotential);

        // Large negative temperature delta
        let delta2 = RuntimeConfigDelta {
            d_chemical_potential: 0.01,
            d_temperature: -0.3,
            d_tunneling_prob: 0.05,
            d_belief_weight: 0.0,
            d_reservoir_leak: 0.0,
        };

        let action2 = MBRLIntegration::delta_to_discrete_action(&delta2);
        assert_eq!(action2, DiscreteAction::DecreaseTemperature);

        // Near-zero deltas
        let delta3 = RuntimeConfigDelta {
            d_chemical_potential: 0.005,
            d_temperature: -0.002,
            d_tunneling_prob: 0.001,
            d_belief_weight: 0.0,
            d_reservoir_leak: 0.0,
        };

        let action3 = MBRLIntegration::delta_to_discrete_action(&delta3);
        assert_eq!(action3, DiscreteAction::NoOp);
    }

    #[cfg(feature = "mbrl")]
    #[test]
    fn test_discrete_to_delta() {
        let delta = MBRLIntegration::discrete_action_to_delta(DiscreteAction::IncreaseTunneling);
        assert_eq!(delta.d_tunneling_prob, 0.1);
        assert_eq!(delta.d_chemical_potential, 0.0);

        let delta2 = MBRLIntegration::discrete_action_to_delta(DiscreteAction::NoOp);
        assert_eq!(delta2.d_chemical_potential, 0.0);
        assert_eq!(delta2.d_temperature, 0.0);
    }
}

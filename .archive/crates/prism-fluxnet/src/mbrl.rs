//! MBRL: Model-Based Reinforcement Learning World Model
//!
//! GNN-based world model that predicts kernel outcomes from state+action,
//! enabling synthetic experience generation for FluxNet training.
//!
//! ## Architecture
//!
//! ```text
//! ┌───────────────────────────────────────┐
//! │      MBRLWorldModel (ONNX GNN)        │
//! │  ┌─────────────────────────────────┐  │
//! │  │  predict_outcome()              │  │
//! │  │  state + action → next_state    │  │
//! │  └─────────────────────────────────┘  │
//! │  ┌─────────────────────────────────┐  │
//! │  │  generate_synthetic_experience()│  │
//! │  │  Monte Carlo rollouts           │  │
//! │  └─────────────────────────────────┘  │
//! │  ┌─────────────────────────────────┐  │
//! │  │  mcts_action_selection()        │  │
//! │  │  Tree search planning           │  │
//! │  └─────────────────────────────────┘  │
//! └───────────────────────────────────────┘
//!          ▲                    │
//!          │  Real experience   │  Synthetic experience
//!          │                    ▼
//! ┌───────────────────────────────────────┐
//! │       DynaFluxNet Controller          │
//! │  Real updates + Synthetic updates     │
//! └───────────────────────────────────────┘
//! ```
//!
//! Implements PRISM GPU Plan §3.4: MBRL World Model for Dyna-style learning.

use anyhow::Result;
use ndarray::{Array1, Array2};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Value;
use std::path::Path;
use std::sync::Arc;

use prism_core::{KernelTelemetry, RuntimeConfig};

/// Predicted outcome from world model
#[derive(Debug, Clone)]
pub struct PredictedOutcome {
    /// Predicted conflict count after action
    pub predicted_conflicts: f32,
    /// Predicted colors used
    pub predicted_colors: f32,
    /// Predicted moves applied
    pub predicted_moves: f32,
    /// Confidence score (0-1)
    pub confidence: f32,
    /// Predicted reward
    pub predicted_reward: f32,
}

/// Kernel state representation for world model
#[derive(Debug, Clone)]
pub struct KernelState {
    /// Current conflict count
    pub conflicts: i32,
    /// Current colors used
    pub colors_used: i32,
    /// Current iteration
    pub iteration: i32,
    /// Graph density
    pub density: f32,
    /// Betti numbers [B0, B1, B2]
    pub betti: [f32; 3],
    /// Reservoir activity
    pub reservoir_activity: f32,
    /// Recent conflict trajectory (last 10 iterations)
    pub conflict_history: [f32; 10],
    /// Current temperature
    pub temperature: f32,
    /// Phase ID (0-6)
    pub phase_id: i32,
}

impl KernelState {
    /// Create from telemetry and config
    pub fn from_telemetry(telemetry: &KernelTelemetry, config: &RuntimeConfig) -> Self {
        Self {
            conflicts: telemetry.conflicts,
            colors_used: telemetry.colors_used,
            iteration: config.iteration,
            density: 0.5, // Would compute from graph
            betti: telemetry.betti_numbers,
            reservoir_activity: telemetry.reservoir_activity,
            conflict_history: [0.0; 10], // Would track in FluxNet
            temperature: config.global_temperature,
            phase_id: config.phase_id,
        }
    }

    /// Convert to feature vector for ONNX model (32 features)
    pub fn to_features(&self) -> Array1<f32> {
        let mut features = Vec::with_capacity(32);

        // Basic state (4 features)
        features.push(self.conflicts as f32);
        features.push(self.colors_used as f32);
        features.push(self.iteration as f32);
        features.push(self.density);

        // Topology (3 features)
        features.extend_from_slice(&self.betti);

        // Reservoir (1 feature)
        features.push(self.reservoir_activity);

        // History (10 features)
        features.extend_from_slice(&self.conflict_history);

        // Control (2 features)
        features.push(self.temperature);
        features.push(self.phase_id as f32);

        // Pad to 32 features (12 more)
        features.extend_from_slice(&[0.0; 12]);

        Array1::from_vec(features)
    }
}

/// Delta to RuntimeConfig (action representation)
#[derive(Debug, Clone, Default)]
pub struct RuntimeConfigDelta {
    /// Delta to chemical potential
    pub d_chemical_potential: f32,
    /// Delta to tunneling probability
    pub d_tunneling_prob: f32,
    /// Delta to temperature
    pub d_temperature: f32,
    /// Delta to belief weight
    pub d_belief_weight: f32,
    /// Delta to reservoir leak rate
    pub d_reservoir_leak: f32,
}

impl RuntimeConfigDelta {
    /// Convert to feature vector (5 features)
    pub fn to_features(&self) -> Array1<f32> {
        Array1::from_vec(vec![
            self.d_chemical_potential,
            self.d_tunneling_prob,
            self.d_temperature,
            self.d_belief_weight,
            self.d_reservoir_leak,
        ])
    }

    /// Apply delta to config
    pub fn apply(&self, config: &mut RuntimeConfig) {
        config.chemical_potential += self.d_chemical_potential;
        config.tunneling_prob_base += self.d_tunneling_prob;
        config.global_temperature += self.d_temperature;
        config.belief_weight += self.d_belief_weight;
        config.reservoir_leak_rate += self.d_reservoir_leak;

        // Clamp values to valid ranges
        config.chemical_potential = config.chemical_potential.clamp(0.0, 10.0);
        config.tunneling_prob_base = config.tunneling_prob_base.clamp(0.0, 1.0);
        config.global_temperature = config.global_temperature.clamp(0.01, 100.0);
        config.belief_weight = config.belief_weight.clamp(0.0, 1.0);
        config.reservoir_leak_rate = config.reservoir_leak_rate.clamp(0.01, 0.99);
    }
}

/// MBRL World Model using GNN
pub struct MBRLWorldModel {
    /// ONNX session for GNN inference
    world_model: Session,

    /// Number of rollouts for MCTS
    num_rollouts: usize,

    /// Rollout horizon (steps to simulate)
    horizon: usize,

    /// Discount factor
    gamma: f64,

    /// Experience buffer for training
    experience_buffer: Vec<Experience>,

    /// Buffer capacity
    buffer_capacity: usize,
}

/// Single experience tuple
#[derive(Debug, Clone)]
pub struct Experience {
    pub state: KernelState,
    pub action: RuntimeConfigDelta,
    pub next_state: KernelState,
    pub reward: f32,
    pub done: bool,
}

impl MBRLWorldModel {
    /// Load world model from ONNX file
    pub fn new(model_path: &Path) -> Result<Self> {
        let world_model = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        Ok(Self {
            world_model,
            num_rollouts: 100,
            horizon: 10,
            gamma: 0.99,
            experience_buffer: Vec::with_capacity(10000),
            buffer_capacity: 10000,
        })
    }

    /// Predict outcome of action from state
    pub fn predict_outcome(
        &mut self,
        state: &KernelState,
        action: &RuntimeConfigDelta,
    ) -> Result<PredictedOutcome> {
        // Concatenate state and action features
        let state_features = state.to_features();
        let action_features = action.to_features();

        let input_dim = state_features.len() + action_features.len();
        let mut input = Array2::zeros((1, input_dim));

        for (i, &v) in state_features.iter().enumerate() {
            input[[0, i]] = v;
        }
        for (i, &v) in action_features.iter().enumerate() {
            input[[0, state_features.len() + i]] = v;
        }

        // Convert to owned array for ORT compatibility
        let data: Vec<f32> = input.iter().copied().collect();
        let shape_vec = vec![1usize, input_dim];
        let input_value = Value::from_array((shape_vec.as_slice(), data))?;

        // Run inference
        let outputs = self.world_model.run(ort::inputs!["input" => input_value])?;

        // Parse output (5 values: conflicts, colors, moves, confidence, reward)
        let (output_shape, output_data) = outputs["output"].try_extract_tensor::<f32>()?;

        // Verify shape [1, 5]
        if output_shape.as_ref() != &[1, 5] {
            anyhow::bail!(
                "Unexpected output shape: expected [1, 5], got {:?}",
                output_shape
            );
        }

        Ok(PredictedOutcome {
            predicted_conflicts: output_data[0],
            predicted_colors: output_data[1],
            predicted_moves: output_data[2],
            confidence: output_data[3].clamp(0.0, 1.0),
            predicted_reward: output_data[4],
        })
    }

    /// Generate synthetic experience via rollouts
    pub fn generate_synthetic_experience(
        &mut self,
        initial_state: &KernelState,
        num_experiences: usize,
    ) -> Result<Vec<Experience>> {
        let mut experiences = Vec::with_capacity(num_experiences);

        for _ in 0..num_experiences {
            let mut state = initial_state.clone();

            for _ in 0..self.horizon {
                // Generate random action
                let action = self.sample_action();

                // Predict next state
                let outcome = self.predict_outcome(&state, &action)?;

                // Compute reward
                let reward = self.compute_reward(&state, &outcome);

                // Create next state from prediction
                let next_state = KernelState {
                    conflicts: outcome.predicted_conflicts as i32,
                    colors_used: outcome.predicted_colors as i32,
                    iteration: state.iteration + 1,
                    density: state.density,
                    betti: state.betti, // Would update from TPTP prediction
                    reservoir_activity: state.reservoir_activity * 0.9,
                    conflict_history: {
                        let mut h = state.conflict_history;
                        h.rotate_left(1);
                        h[9] = outcome.predicted_conflicts;
                        h
                    },
                    temperature: (state.temperature * 0.99).max(0.01),
                    phase_id: state.phase_id,
                };

                let done = outcome.predicted_conflicts <= 0.0;

                experiences.push(Experience {
                    state: state.clone(),
                    action,
                    next_state: next_state.clone(),
                    reward,
                    done,
                });

                if done {
                    break;
                }

                state = next_state;
            }
        }

        Ok(experiences)
    }

    /// MCTS-style action selection
    pub fn mcts_action_selection(
        &mut self,
        state: &KernelState,
        num_candidates: usize,
    ) -> Result<Vec<(RuntimeConfigDelta, f32)>> {
        let mut candidates = Vec::with_capacity(num_candidates);

        for _ in 0..num_candidates {
            let action = self.sample_action();
            let value = self.rollout_value(state, &action)?;
            candidates.push((action, value));
        }

        // Sort by value (descending)
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(candidates)
    }

    /// Compute rollout value for action
    fn rollout_value(&mut self, state: &KernelState, action: &RuntimeConfigDelta) -> Result<f32> {
        let mut total_value = 0.0;
        let mut current_state = state.clone();
        let mut current_action = action.clone();
        let mut discount = 1.0;

        for _ in 0..self.horizon {
            let outcome = self.predict_outcome(&current_state, &current_action)?;
            let reward = self.compute_reward(&current_state, &outcome);

            total_value += discount * reward;
            discount *= self.gamma as f32;

            if outcome.predicted_conflicts <= 0.0 {
                // Early termination bonus
                total_value += discount * 100.0;
                break;
            }

            // Update state
            current_state.conflicts = outcome.predicted_conflicts as i32;
            current_state.colors_used = outcome.predicted_colors as i32;
            current_state.iteration += 1;

            // Sample new action for next step
            current_action = self.sample_action();
        }

        Ok(total_value)
    }

    /// Sample random action delta
    fn sample_action(&self) -> RuntimeConfigDelta {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        RuntimeConfigDelta {
            d_chemical_potential: rng.gen_range(-0.5..0.5),
            d_tunneling_prob: rng.gen_range(-0.1..0.1),
            d_temperature: rng.gen_range(-0.2..0.2),
            d_belief_weight: rng.gen_range(-0.1..0.1),
            d_reservoir_leak: rng.gen_range(-0.05..0.05),
        }
    }

    /// Compute reward from state and outcome
    fn compute_reward(&self, state: &KernelState, outcome: &PredictedOutcome) -> f32 {
        let conflict_reduction = state.conflicts as f32 - outcome.predicted_conflicts;
        let color_reduction = state.colors_used as f32 - outcome.predicted_colors;

        // Primary reward: conflict reduction
        let mut reward = conflict_reduction * 10.0;

        // Secondary reward: color efficiency
        reward += color_reduction * 0.5;

        // Bonus for reaching zero conflicts
        if outcome.predicted_conflicts <= 0.0 {
            reward += 100.0;
        }

        // Scale by confidence
        reward * outcome.confidence
    }

    /// Add experience to buffer
    pub fn add_experience(&mut self, exp: Experience) {
        if self.experience_buffer.len() >= self.buffer_capacity {
            self.experience_buffer.remove(0);
        }
        self.experience_buffer.push(exp);
    }

    /// Sample batch from experience buffer
    pub fn sample_batch(&self, batch_size: usize) -> Vec<&Experience> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        self.experience_buffer
            .choose_multiple(&mut rng, batch_size.min(self.experience_buffer.len()))
            .collect()
    }

    /// Get experience buffer size
    pub fn buffer_size(&self) -> usize {
        self.experience_buffer.len()
    }

    /// Set rollout parameters
    pub fn set_rollout_params(&mut self, num_rollouts: usize, horizon: usize) {
        self.num_rollouts = num_rollouts;
        self.horizon = horizon;
    }

    /// Set discount factor
    pub fn set_gamma(&mut self, gamma: f64) {
        self.gamma = gamma.clamp(0.0, 1.0);
    }
}

/// Dyna-style integration of MBRL with FluxNet
pub struct DynaFluxNet {
    /// Real Q-table/network (reserved for future integration)
    #[allow(dead_code)]
    fluxnet: Arc<crate::core::controller::UniversalRLController>,

    /// World model for synthetic experience
    world_model: MBRLWorldModel,

    /// Ratio of synthetic to real experience
    synthetic_ratio: usize,
}

impl DynaFluxNet {
    /// Create new Dyna-FluxNet controller
    pub fn new(
        fluxnet: Arc<crate::core::controller::UniversalRLController>,
        model_path: &Path,
    ) -> Result<Self> {
        Ok(Self {
            fluxnet,
            world_model: MBRLWorldModel::new(model_path)?,
            synthetic_ratio: 5, // 5 synthetic updates per real update
        })
    }

    /// Update with real experience + synthetic rollouts
    pub fn update(
        &mut self,
        state: &KernelState,
        action: &RuntimeConfigDelta,
        next_state: &KernelState,
        reward: f32,
    ) -> Result<()> {
        // 1. Update with real experience
        self.world_model.add_experience(Experience {
            state: state.clone(),
            action: action.clone(),
            next_state: next_state.clone(),
            reward,
            done: next_state.conflicts == 0,
        });

        // 2. Generate and update with synthetic experience
        let synthetic = self
            .world_model
            .generate_synthetic_experience(state, self.synthetic_ratio)?;

        // 3. Update FluxNet with synthetic experiences
        for exp in synthetic {
            // Convert to FluxNet's native format and update
            // This would be implemented based on FluxNet's API
            log::debug!(
                "Synthetic experience: {} conflicts -> {} conflicts, reward: {:.3}",
                exp.state.conflicts,
                exp.next_state.conflicts,
                exp.reward
            );
        }

        Ok(())
    }

    /// Select best action using MCTS
    pub fn select_action(&mut self, state: &KernelState) -> Result<RuntimeConfigDelta> {
        let candidates = self.world_model.mcts_action_selection(state, 20)?;

        // Return best action (highest value)
        Ok(candidates
            .into_iter()
            .next()
            .map(|(a, _)| a)
            .unwrap_or_default())
    }

    /// Get reference to world model
    pub fn world_model(&self) -> &MBRLWorldModel {
        &self.world_model
    }

    /// Get mutable reference to world model
    pub fn world_model_mut(&mut self) -> &mut MBRLWorldModel {
        &mut self.world_model
    }

    /// Set synthetic experience ratio
    pub fn set_synthetic_ratio(&mut self, ratio: usize) {
        self.synthetic_ratio = ratio;
    }

    /// Get current synthetic ratio
    pub fn synthetic_ratio(&self) -> usize {
        self.synthetic_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_state_features() {
        let state = KernelState {
            conflicts: 100,
            colors_used: 42,
            iteration: 500,
            density: 0.7,
            betti: [1.0, 2.0, 3.0],
            reservoir_activity: 0.5,
            conflict_history: [10.0; 10],
            temperature: 1.5,
            phase_id: 2,
        };

        let features = state.to_features();
        assert_eq!(features.len(), 32);
        assert_eq!(features[0], 100.0); // conflicts
        assert_eq!(features[1], 42.0); // colors_used
        assert_eq!(features[2], 500.0); // iteration
        assert_eq!(features[3], 0.7); // density
    }

    #[test]
    fn test_action_delta_features() {
        let delta = RuntimeConfigDelta {
            d_chemical_potential: 0.5,
            d_tunneling_prob: 0.1,
            d_temperature: -0.2,
            d_belief_weight: 0.05,
            d_reservoir_leak: -0.01,
        };

        let features = delta.to_features();
        assert_eq!(features.len(), 5);
        assert_eq!(features[0], 0.5);
        assert_eq!(features[4], -0.01);
    }

    #[test]
    fn test_action_delta_apply() {
        let mut config = RuntimeConfig::production();
        let initial_chem = config.chemical_potential;

        let delta = RuntimeConfigDelta {
            d_chemical_potential: 1.0,
            d_tunneling_prob: 0.0,
            d_temperature: 0.0,
            d_belief_weight: 0.0,
            d_reservoir_leak: 0.0,
        };

        delta.apply(&mut config);
        assert_eq!(config.chemical_potential, initial_chem + 1.0);

        // Test clamping
        let big_delta = RuntimeConfigDelta {
            d_chemical_potential: 100.0,
            d_tunneling_prob: 100.0,
            d_temperature: 0.0,
            d_belief_weight: 0.0,
            d_reservoir_leak: 0.0,
        };

        big_delta.apply(&mut config);
        assert_eq!(config.chemical_potential, 10.0); // Clamped to max
        assert_eq!(config.tunneling_prob_base, 1.0); // Clamped to max
    }

    #[test]
    fn test_experience_creation() {
        let state = KernelState {
            conflicts: 100,
            colors_used: 42,
            iteration: 0,
            density: 0.5,
            betti: [1.0, 0.0, 0.0],
            reservoir_activity: 0.3,
            conflict_history: [0.0; 10],
            temperature: 10.0,
            phase_id: 0,
        };

        let next_state = KernelState {
            conflicts: 90,
            colors_used: 42,
            iteration: 1,
            density: 0.5,
            betti: [1.0, 0.0, 0.0],
            reservoir_activity: 0.27,
            conflict_history: [0.0; 10],
            temperature: 9.9,
            phase_id: 0,
        };

        let action = RuntimeConfigDelta::default();

        let exp = Experience {
            state,
            action,
            next_state,
            reward: 10.0,
            done: false,
        };

        assert_eq!(exp.state.conflicts, 100);
        assert_eq!(exp.next_state.conflicts, 90);
        assert_eq!(exp.reward, 10.0);
        assert!(!exp.done);
    }
}

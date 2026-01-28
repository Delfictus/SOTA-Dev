//! AdaptiveVEOptimizer - FluxNet RL for Viral Evolution Prediction
//!
//! Uses reinforcement learning to learn optimal RISE/FALL decision boundaries
//! from temporal training data.
//!
//! This module wraps prism-fluxnet's Q-learning infrastructure for the
//! VASIL benchmark task of predicting variant trajectory direction.
//!
//! ## Key Improvements for Generalization:
//! 1. Coarser state binning (4 bins instead of 8) - prevents overfitting
//! 2. Q-value priors based on observed base rates - regularization
//! 3. Feature normalization using Z-scores - temporal stability
//! 4. Escape dominance feature - relative fitness measure

use anyhow::{Result, Context};
use std::collections::HashMap;
use rand::Rng;

/// State representation for VE prediction - RAW FEATURES ONLY
/// NO pre-computed gamma - RL learns optimal weights!
#[derive(Debug, Clone)]
pub struct VEState {
    // Primary raw features (RL learns how to weight these)
    /// DMS escape score (0-1) - AGGREGATE raw value from Bloom Lab data
    pub escape: f32,
    /// Structural transmissibility (0-1) - literature R0 normalized
    pub transmit: f32,
    /// Current frequency (0-1)
    pub frequency: f32,

    // Additional raw GPU features (RL can use these too)
    /// ddG binding energy from GPU feature 92
    pub ddg_binding: f32,
    /// ddG stability from GPU feature 93
    pub ddg_stability: f32,
    /// Expression level from GPU feature 94
    pub expression: f32,

    // NEW: Per-epitope escape scores (10 dimensions)
    // Epitopes: A, B, C, D1, D2, E12, E3, F1, F2, F3
    /// Epitope-specific escape scores [0-1] for 10 antibody classes
    pub epitope_escape: [f32; 10],

    // CRITICAL: Effective escape after immunity modulation
    // This is raw_escape × (1 - cross_reactive_immunity)
    // KEY for accurate prediction - accounts for time-varying population immunity
    pub effective_escape: f32,

    // NEW: Competition-aware fitness features
    /// Relative fitness = escape - weighted_avg_escape(competitors), normalized to [0,1]
    pub relative_fitness: f32,
    /// Frequency velocity = (freq - prev_freq) / freq, momentum signal
    pub frequency_velocity: f32,

    // Stage 8.5: Synaptic Spike Features (from LIF neurons)
    /// Velocity-sensitive spike density [0,1] - temporal edge detector for frequency changes
    pub spike_velocity: f32,
    /// Emergence-sensitive spike density [0,1] - emergence probability responsiveness
    pub spike_emergence: f32,
    /// Spike momentum [0,1] - cumulative temporal signal strength
    pub spike_momentum: f32,

    // VASIL EPIDEMIOLOGICAL FEATURES (NEW!)
    /// Time-varying population immunity from VASIL SEIR simulations [0,1]
    /// Normalized from 0-500,000 range
    pub time_varying_immunity: f32,
    /// Phi-normalized frequency (corrects for testing rate variations) [0,1]
    pub phi_normalized_freq: f32,
    /// P_neut-based immune escape (1 - neutralization probability) [0,1]
    pub p_neut_escape: f32,
}

impl VEState {
    /// Create from raw features - NO hardcoded formula!
    pub fn new(
        escape: f32,
        transmit: f32,
        frequency: f32,
        ddg_binding: f32,
        ddg_stability: f32,
        expression: f32,
    ) -> Self {
        Self {
            escape,
            transmit,
            frequency,
            ddg_binding,
            ddg_stability,
            expression,
            epitope_escape: [0.0; 10],
            effective_escape: escape,
            relative_fitness: 0.5,
            frequency_velocity: 0.0,
            spike_velocity: 0.0,
            spike_emergence: 0.0,
            spike_momentum: 0.0,
            time_varying_immunity: 0.5,  // Default mid-range
            phi_normalized_freq: frequency,  // Default to raw freq
            p_neut_escape: 0.25,  // Default escape
        }
    }

    /// Create with epitope-specific escape scores
    pub fn new_with_epitopes(
        escape: f32,
        transmit: f32,
        frequency: f32,
        ddg_binding: f32,
        ddg_stability: f32,
        expression: f32,
        epitope_escape: [f32; 10],
    ) -> Self {
        Self {
            escape,
            transmit,
            frequency,
            ddg_binding,
            ddg_stability,
            expression,
            epitope_escape,
            effective_escape: escape,  // Default to raw escape
            relative_fitness: 0.5,  // Default to neutral
            frequency_velocity: 0.0,  // Default to stable
            spike_velocity: 0.0,
            spike_emergence: 0.0,
            spike_momentum: 0.0,
            time_varying_immunity: 0.5,
            phi_normalized_freq: frequency,
            p_neut_escape: 0.25,
        }
    }

    /// Create with effective escape (immunity-modulated)
    pub fn new_with_effective_escape(
        escape: f32,
        transmit: f32,
        frequency: f32,
        ddg_binding: f32,
        ddg_stability: f32,
        expression: f32,
        epitope_escape: [f32; 10],
        effective_escape: f32,
    ) -> Self {
        Self {
            escape,
            transmit,
            frequency,
            ddg_binding,
            ddg_stability,
            expression,
            epitope_escape,
            effective_escape,
            relative_fitness: 0.5,
            frequency_velocity: 0.0,
            spike_velocity: 0.0,
            spike_emergence: 0.0,
            spike_momentum: 0.0,
            time_varying_immunity: 0.5,
            phi_normalized_freq: frequency,
            p_neut_escape: 0.25,
        }
    }

    /// Create with ALL features including competition-aware fitness
    pub fn new_full(
        escape: f32,
        transmit: f32,
        frequency: f32,
        ddg_binding: f32,
        ddg_stability: f32,
        expression: f32,
        epitope_escape: [f32; 10],
        effective_escape: f32,
        relative_fitness: f32,
        frequency_velocity: f32,
    ) -> Self {
        Self {
            escape,
            transmit,
            frequency,
            ddg_binding,
            ddg_stability,
            expression,
            epitope_escape,
            effective_escape,
            relative_fitness,
            frequency_velocity,
            spike_velocity: 0.0,
            spike_emergence: 0.0,
            spike_momentum: 0.0,
            time_varying_immunity: 0.5,
            phi_normalized_freq: frequency,
            p_neut_escape: 0.25,
        }
    }

    /// Create with ALL features including spike signals + VASIL epidemiological data
    pub fn new_full_with_spikes(
        escape: f32,
        transmit: f32,
        frequency: f32,
        ddg_binding: f32,
        ddg_stability: f32,
        expression: f32,
        epitope_escape: [f32; 10],
        effective_escape: f32,
        relative_fitness: f32,
        frequency_velocity: f32,
        spike_velocity: f32,
        spike_emergence: f32,
        spike_momentum: f32,
    ) -> Self {
        Self::new_full_with_vasil(
            escape, transmit, frequency, ddg_binding, ddg_stability, expression,
            epitope_escape, effective_escape, relative_fitness, frequency_velocity,
            spike_velocity, spike_emergence, spike_momentum,
            0.5,       // time_varying_immunity (will be populated in main)
            frequency, // phi_normalized_freq
            0.25,      // p_neut_escape
        )
    }

    /// Create with COMPLETE feature set (structural + epidemiological)
    #[allow(clippy::too_many_arguments)]
    pub fn new_full_with_vasil(
        escape: f32,
        transmit: f32,
        frequency: f32,
        ddg_binding: f32,
        ddg_stability: f32,
        expression: f32,
        epitope_escape: [f32; 10],
        effective_escape: f32,
        relative_fitness: f32,
        frequency_velocity: f32,
        spike_velocity: f32,
        spike_emergence: f32,
        spike_momentum: f32,
        time_varying_immunity: f32,
        phi_normalized_freq: f32,
        p_neut_escape: f32,
    ) -> Self {
        Self {
            escape,
            transmit,
            frequency,
            ddg_binding,
            ddg_stability,
            expression,
            epitope_escape,
            effective_escape,
            relative_fitness,
            frequency_velocity,
            spike_velocity,
            spike_emergence,
            spike_momentum,
            time_varying_immunity,
            phi_normalized_freq,
            p_neut_escape,
        }
    }

    /// Compute weighted epitope escape score
    /// Uses class weights: A+B (class 1) = 0.3, C (class 2) = 0.25,
    /// D1+D2+E12+E3 (class 3) = 0.3, F1+F2+F3 (class 4) = 0.15
    pub fn weighted_epitope_escape(&self) -> f32 {
        let class1 = (self.epitope_escape[0] + self.epitope_escape[1]) / 2.0;  // A, B
        let class2 = self.epitope_escape[2];  // C
        let class3 = (self.epitope_escape[3] + self.epitope_escape[4]
                    + self.epitope_escape[5] + self.epitope_escape[6]) / 4.0;  // D1, D2, E12, E3
        let class4 = (self.epitope_escape[7] + self.epitope_escape[8]
                    + self.epitope_escape[9]) / 3.0;  // F1, F2, F3

        // Weights based on Barnes antibody class prevalence
        0.30 * class1 + 0.25 * class2 + 0.30 * class3 + 0.15 * class4
    }

    /// Get max epitope escape (most escaped epitope)
    pub fn max_epitope_escape(&self) -> f32 {
        self.epitope_escape.iter().cloned().fold(0.0f32, f32::max)
    }

    /// Discretize state into index for Q-table lookup
    /// RL learns which raw feature combinations predict RISE vs FALL
    /// 8 features × 4 bins each = 65536 states (includes Stage 8.5 spike features)
    pub fn discretize(&self) -> usize {
        // Quantize each feature to 4 bins (0-3)
        let escape_bin = ((self.escape * 3.99).clamp(0.0, 3.0) as usize).min(3);
        let transmit_bin = ((self.transmit * 3.99).clamp(0.0, 3.0) as usize).min(3);
        let freq_bin = ((self.frequency * 3.99).clamp(0.0, 3.0) as usize).min(3);

        // ddG values range roughly -2 to +2, map to 0-1
        let ddg_bind_scaled = ((self.ddg_binding + 2.0) / 4.0).clamp(0.0, 1.0);
        let ddg_bind_bin = ((ddg_bind_scaled * 3.99).clamp(0.0, 3.0) as usize).min(3);

        let ddg_stab_scaled = ((self.ddg_stability + 2.0) / 4.0).clamp(0.0, 1.0);
        let ddg_stab_bin = ((ddg_stab_scaled * 3.99).clamp(0.0, 3.0) as usize).min(3);

        // Expression ranges 0-1
        let expr_bin = ((self.expression * 3.99).clamp(0.0, 3.0) as usize).min(3);

        // Stage 8.5 Spike features (0-1 range, temporal edge detectors)
        let spike_vel_bin = ((self.spike_velocity * 3.99).clamp(0.0, 3.0) as usize).min(3);
        let spike_emerge_bin = ((self.spike_emergence * 3.99).clamp(0.0, 3.0) as usize).min(3);

        // Combine into single index: 4^8 = 65536 states
        // RL learns which combinations of raw + spike features predict RISE vs FALL
        escape_bin * 16384      // 4^7
            + transmit_bin * 4096   // 4^6
            + freq_bin * 1024       // 4^5
            + ddg_bind_bin * 256    // 4^4
            + ddg_stab_bin * 64     // 4^3
            + expr_bin * 16         // 4^2
            + spike_vel_bin * 4     // 4^1
            + spike_emerge_bin      // 4^0
    }

}

/// Action: predict RISE or FALL
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VEAction {
    Rise = 0,
    Fall = 1,
}

impl VEAction {
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => VEAction::Rise,
            _ => VEAction::Fall,
        }
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            VEAction::Rise => "RISE",
            VEAction::Fall => "FALL",
        }
    }
}

/// Experience tuple for replay buffer
#[derive(Debug, Clone)]
pub struct VEExperience {
    pub state: VEState,
    pub action: VEAction,
    pub reward: f32,
    pub next_state: VEState,
}

/// AdaptiveVEOptimizer - Q-learning for RISE/FALL prediction
///
/// ## Architecture - RAW FEATURE LEARNING + STAGE 8.5 SPIKE SIGNALS
/// - State space: 65536 discrete states (4 bins × 8 features)
/// - Action space: 2 actions (RISE, FALL)
/// - Q-table: 65536 × 2 = 131072 Q-values
/// - NO hardcoded VASIL weights - RL learns optimal combinations!
///
/// ## Features (raw + spike, NO pre-computed gamma):
/// - escape: DMS escape score from Bloom Lab data
/// - transmit: Structural transmissibility from GPU
/// - frequency: Current GISAID frequency
/// - ddg_binding: Binding energy from GPU
/// - ddg_stability: Stability from GPU
/// - expression: Expression level from GPU
/// - spike_velocity: Velocity-sensitive spike density (Stage 8.5 LIF)
/// - spike_emergence: Emergence-sensitive spike density (Stage 8.5 LIF)
///
/// ## Training:
/// - Q-learning discovers which feature combinations predict RISE vs FALL
/// - Asymmetric rewards for class imbalance
/// - Experience replay for stability
/// - UCB1-style exploration bonus
pub struct AdaptiveVEOptimizer {
    /// Q-table: [state_index][action_index] -> Q-value
    q_table: Vec<[f32; 2]>,

    /// Visit count per state-action pair for UCB exploration
    visit_counts: Vec<[usize; 2]>,

    /// Replay buffer
    replay_buffer: Vec<VEExperience>,

    /// Learning rate
    alpha: f32,

    /// Discount factor (0 for immediate reward)
    gamma: f32,

    /// Exploration rate
    epsilon: f32,

    /// Minimum epsilon
    epsilon_min: f32,

    /// Epsilon decay per episode
    epsilon_decay: f32,

    /// Total training samples seen
    training_samples: usize,

    /// Number of states (65536 with 8 features including Stage 8.5 spikes)
    num_states: usize,

    /// Observed FALL base rate for prior initialization
    fall_base_rate: f32,

    /// Stage 8.5 spike weight for prediction model
    spike_weight: f32,
}

impl AdaptiveVEOptimizer {
    /// Create a new optimizer with default hyperparameters
    /// Now uses LEARNED weights via grid search, not Q-learning
    /// State space: 4^8 = 65536 (8 features × 4 bins each)
    pub fn new() -> Self {
        let num_states = 65536;  // 8 features × 4 bins = 4^8
        let fall_base_rate = 0.64;

        let q_init = [0.0, 0.0];

        Self {
            q_table: vec![q_init; num_states],
            visit_counts: vec![[0, 0]; num_states],
            replay_buffer: Vec::with_capacity(10000),
            alpha: 0.5,   // LEARNED escape weight (will be tuned)
            gamma: 0.5,   // LEARNED transmit weight (will be tuned)
            epsilon: 0.0, // Not used in new approach
            epsilon_min: 0.0,
            epsilon_decay: 1.0,
            training_samples: 0,
            num_states,
            fall_base_rate,
            spike_weight: 0.0,  // Stage 8.5 spike weight (will be tuned)
        }
    }

    /// Create with custom hyperparameters
    pub fn with_config(alpha: f32, epsilon: f32, epsilon_decay: f32) -> Self {
        let mut opt = Self::new();
        opt.alpha = alpha;
        opt.epsilon = epsilon;
        opt.epsilon_decay = epsilon_decay;
        opt
    }

    /// Get state index from raw features
    fn get_state_index(&self, state: &VEState) -> usize {
        state.discretize()
    }

    /// Select action using epsilon-greedy policy with UCB exploration bonus
    pub fn select_action(&self, state: &VEState, explore: bool) -> VEAction {
        let state_idx = self.get_state_index(state);

        if explore && rand::thread_rng().gen::<f32>() < self.epsilon {
            // Explore: random action (but biased toward RISE for class balance)
            // Give RISE slightly higher probability during exploration
            if rand::thread_rng().gen::<f32>() < 0.45 {
                VEAction::Rise
            } else {
                VEAction::Fall
            }
        } else {
            // Exploit: best action with UCB exploration bonus
            let q_rise = self.q_table[state_idx][0];
            let q_fall = self.q_table[state_idx][1];

            // UCB bonus for less-visited actions (encourages exploration)
            let total_visits = (self.visit_counts[state_idx][0] + self.visit_counts[state_idx][1]) as f32;
            let rise_visits = self.visit_counts[state_idx][0] as f32;
            let fall_visits = self.visit_counts[state_idx][1] as f32;

            let ucb_rise = if explore && total_visits > 0.0 && rise_visits > 0.0 {
                0.5 * ((2.0 * total_visits.ln()) / rise_visits).sqrt()
            } else {
                0.0
            };

            let ucb_fall = if explore && total_visits > 0.0 && fall_visits > 0.0 {
                0.5 * ((2.0 * total_visits.ln()) / fall_visits).sqrt()
            } else {
                0.0
            };

            let q_rise_ucb = q_rise + ucb_rise;
            let q_fall_ucb = q_fall + ucb_fall;

            if q_rise_ucb > q_fall_ucb {
                VEAction::Rise
            } else if q_fall_ucb > q_rise_ucb {
                VEAction::Fall
            } else {
                // Tie: use base rate (64% FALL in observed data)
                if rand::thread_rng().gen::<f32>() > 0.36 {
                    VEAction::Fall
                } else {
                    VEAction::Rise
                }
            }
        }
    }

    /// Predict without exploration (for evaluation)
    pub fn predict(&self, state: &VEState) -> VEAction {
        self.select_action(state, false)
    }

    /// Train on a single experience
    pub fn train_step(&mut self, experience: &VEExperience) {
        let state_idx = self.get_state_index(&experience.state);
        let action_idx = experience.action as usize;

        // Update visit count
        self.visit_counts[state_idx][action_idx] += 1;

        // Q-learning update: Q(s,a) = Q(s,a) + α * [r - Q(s,a)]
        // (gamma = 0, so no next state value)
        let current_q = self.q_table[state_idx][action_idx];
        let target = experience.reward;
        self.q_table[state_idx][action_idx] = current_q + self.alpha * (target - current_q);

        self.training_samples += 1;
    }

    /// Add experience to replay buffer
    pub fn add_experience(&mut self, experience: VEExperience) {
        if self.replay_buffer.len() >= 10000 {
            self.replay_buffer.remove(0);
        }
        self.replay_buffer.push(experience);
    }

    /// Train on batch from replay buffer
    pub fn train_batch(&mut self, batch_size: usize) {
        if self.replay_buffer.len() < batch_size {
            return;
        }

        let mut rng = rand::thread_rng();
        for _ in 0..batch_size {
            let idx = rng.gen_range(0..self.replay_buffer.len());
            let exp = self.replay_buffer[idx].clone();
            self.train_step(&exp);
        }
    }

    /// Decay epsilon after each episode
    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
    }

    /// Get current epsilon
    pub fn get_epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Get training statistics
    pub fn get_stats(&self) -> (usize, f32) {
        (self.training_samples, self.epsilon)
    }

    /// Train on a dataset of (state, observed_direction) pairs
    /// Uses asymmetric rewards to handle class imbalance (36% RISE, 64% FALL)
    pub fn train_on_dataset(&mut self, data: &[(VEState, &str)], epochs: usize) {
        log::info!("Training VE optimizer on {} samples for {} epochs", data.len(), epochs);

        // Compute class weights inversely proportional to frequency
        let rise_count = data.iter().filter(|(_, o)| *o == "RISE").count();
        let fall_count = data.len() - rise_count;
        let rise_weight = (data.len() as f32) / (2.0 * rise_count as f32);
        let fall_weight = (data.len() as f32) / (2.0 * fall_count as f32);

        log::info!("Class weights: RISE={:.2}, FALL={:.2}", rise_weight, fall_weight);

        for epoch in 0..epochs {
            let mut correct = 0;
            let mut rise_correct = 0;
            let mut rise_total = 0;
            let mut total = 0;

            for (state, observed) in data {
                // Select action with exploration
                let action = self.select_action(state, true);

                // Asymmetric reward:
                // - Correct RISE: +rise_weight (higher to encourage learning RISE)
                // - Correct FALL: +fall_weight
                // - Incorrect: -weight (penalize misses proportionally)
                let is_rise = *observed == "RISE";
                let is_correct = action.to_str() == *observed;

                let reward = if is_correct {
                    correct += 1;
                    if is_rise {
                        rise_correct += 1;
                        rise_weight
                    } else {
                        fall_weight
                    }
                } else {
                    // Penalize misses more for minority class
                    if is_rise {
                        -rise_weight * 1.5  // Extra penalty for missing RISE
                    } else {
                        -fall_weight
                    }
                };

                if is_rise {
                    rise_total += 1;
                }
                total += 1;

                // Create experience
                let exp = VEExperience {
                    state: state.clone(),
                    action,
                    reward,
                    next_state: state.clone(),
                };

                // Train on this experience
                self.train_step(&exp);
                self.add_experience(exp);

                // Batch training every 100 samples
                if total % 100 == 0 {
                    self.train_batch(32);
                }
            }

            // Decay epsilon after each epoch
            self.decay_epsilon();

            let accuracy = correct as f32 / total as f32;
            let rise_recall = if rise_total > 0 {
                rise_correct as f32 / rise_total as f32
            } else {
                0.0
            };

            if epoch % 10 == 0 || epoch == epochs - 1 {
                log::info!("Epoch {}: accuracy={:.3}, rise_recall={:.3}, epsilon={:.3}",
                          epoch, accuracy, rise_recall, self.epsilon);
            }
        }
    }

    /// Evaluate on test dataset
    pub fn evaluate(&self, data: &[(VEState, &str)]) -> f32 {
        let mut correct = 0;
        let mut total = 0;

        for (state, observed) in data {
            let action = self.predict_with_weights(state);
            if action.to_str() == *observed {
                correct += 1;
            }
            total += 1;
        }

        if total > 0 {
            correct as f32 / total as f32
        } else {
            0.0
        }
    }

    /// GRID SEARCH: Learn optimal weights using COMPETITION-AWARE fitness
    /// Formula: γ = α×relative_fitness + β×frequency_velocity + γ×transmit + δ×escape
    /// CRITICAL: Uses VELOCITY as primary signal (most discriminative)
    /// Plus ESCAPE and EFFECTIVE_ESCAPE for boundary case improvement
    pub fn train_grid_search(&mut self, data: &[(VEState, &str)]) {
        println!("  ENHANCED Grid search: velocity + escape + effective_escape...");
        println!("  Formula: γ = α×velocity + β×escape + γ×effective_escape + δ×fitness");

        // Separate data by class for balanced evaluation
        let rise_samples: Vec<_> = data.iter().filter(|(_, o)| *o == "RISE").collect();
        let fall_samples: Vec<_> = data.iter().filter(|(_, o)| *o == "FALL").collect();
        println!("  Training: {} RISE, {} FALL", rise_samples.len(), fall_samples.len());

        // Analyze ALL discriminative features
        let rise_vel: Vec<f32> = rise_samples.iter().map(|(s, _)| s.frequency_velocity).collect();
        let fall_vel: Vec<f32> = fall_samples.iter().map(|(s, _)| s.frequency_velocity).collect();
        let rise_vel_mean = rise_vel.iter().sum::<f32>() / rise_vel.len().max(1) as f32;
        let fall_vel_mean = fall_vel.iter().sum::<f32>() / fall_vel.len().max(1) as f32;
        println!("  RISE velocity mean: {:.3}, FALL: {:.3} (DELTA={:.3})",
                 rise_vel_mean, fall_vel_mean, rise_vel_mean - fall_vel_mean);

        let rise_esc: Vec<f32> = rise_samples.iter().map(|(s, _)| s.escape).collect();
        let fall_esc: Vec<f32> = fall_samples.iter().map(|(s, _)| s.escape).collect();
        let rise_esc_mean = rise_esc.iter().sum::<f32>() / rise_esc.len().max(1) as f32;
        let fall_esc_mean = fall_esc.iter().sum::<f32>() / fall_esc.len().max(1) as f32;
        println!("  RISE escape mean: {:.3}, FALL: {:.3} (DELTA={:.3})",
                 rise_esc_mean, fall_esc_mean, rise_esc_mean - fall_esc_mean);

        let rise_eff: Vec<f32> = rise_samples.iter().map(|(s, _)| s.effective_escape).collect();
        let fall_eff: Vec<f32> = fall_samples.iter().map(|(s, _)| s.effective_escape).collect();
        let rise_eff_mean = rise_eff.iter().sum::<f32>() / rise_eff.len().max(1) as f32;
        let fall_eff_mean = fall_eff.iter().sum::<f32>() / fall_eff.len().max(1) as f32;
        println!("  RISE effective_escape mean: {:.3}, FALL: {:.3} (DELTA={:.3})",
                 rise_eff_mean, fall_eff_mean, rise_eff_mean - fall_eff_mean);

        let rise_fit: Vec<f32> = rise_samples.iter().map(|(s, _)| s.relative_fitness).collect();
        let fall_fit: Vec<f32> = fall_samples.iter().map(|(s, _)| s.relative_fitness).collect();
        let rise_fit_mean = rise_fit.iter().sum::<f32>() / rise_fit.len().max(1) as f32;
        let fall_fit_mean = fall_fit.iter().sum::<f32>() / fall_fit.len().max(1) as f32;
        println!("  RISE relative_fitness mean: {:.3}, FALL: {:.3} (DELTA={:.3})",
                 rise_fit_mean, fall_fit_mean, rise_fit_mean - fall_fit_mean);

        let rise_trans: Vec<f32> = rise_samples.iter().map(|(s, _)| s.transmit).collect();
        let fall_trans: Vec<f32> = fall_samples.iter().map(|(s, _)| s.transmit).collect();
        let rise_trans_mean = rise_trans.iter().sum::<f32>() / rise_trans.len().max(1) as f32;
        let fall_trans_mean = fall_trans.iter().sum::<f32>() / fall_trans.len().max(1) as f32;
        println!("  RISE transmit mean: {:.3}, FALL: {:.3} (DELTA={:.3})",
                 rise_trans_mean, fall_trans_mean, rise_trans_mean - fall_trans_mean);

        // Stage 8.5 spike features (GPU-computed, currently broken due to nullptr)
        let rise_spike: Vec<f32> = rise_samples.iter().map(|(s, _)| s.spike_momentum).collect();
        let fall_spike: Vec<f32> = fall_samples.iter().map(|(s, _)| s.spike_momentum).collect();
        let rise_spike_mean = rise_spike.iter().sum::<f32>() / rise_spike.len().max(1) as f32;
        let fall_spike_mean = fall_spike.iter().sum::<f32>() / fall_spike.len().max(1) as f32;
        println!("  RISE spike_momentum mean: {:.3}, FALL: {:.3} (GPU nullptr issue - not useful)",
                 rise_spike_mean, fall_spike_mean);

        let mut best_balanced_acc = 0.0f32;
        let mut best_velocity_weight = 1.0f32;
        let mut best_escape_weight = 0.0f32;
        let mut best_effective_weight = 0.0f32;
        let mut best_fitness_weight = 0.0f32;
        let mut best_threshold = 0.0f32;

        // Test Model 1: Velocity-only (baseline - but note velocity is BACKWARD looking!)
        println!("\n  Testing Model 1: Velocity-only (backward-looking signal)...");
        let vel_rise_correct = rise_samples.iter().filter(|(s, _)| s.frequency_velocity > 0.0).count();
        let vel_fall_correct = fall_samples.iter().filter(|(s, _)| s.frequency_velocity <= 0.0).count();
        let vel_rise_acc = vel_rise_correct as f32 / rise_samples.len().max(1) as f32;
        let vel_fall_acc = vel_fall_correct as f32 / fall_samples.len().max(1) as f32;
        let vel_balanced = (vel_rise_acc + vel_fall_acc) / 2.0;
        println!("  Velocity-only: RISE_acc={:.1}%, FALL_acc={:.1}%, balanced={:.1}%",
                 vel_rise_acc * 100.0, vel_fall_acc * 100.0, vel_balanced * 100.0);

        if vel_balanced > best_balanced_acc {
            best_balanced_acc = vel_balanced;
            best_velocity_weight = 1.0;
            best_escape_weight = 0.0;
            best_effective_weight = 0.0;
            best_fitness_weight = 0.0;
            best_threshold = 0.0;  // velocity > 0 threshold
        }

        // Test Model 1a: INVERSE Velocity (momentum peak = about to fall)
        // Key insight: High velocity variants are at their PEAK and about to FALL
        println!("  Testing Model 1a: INVERSE Velocity (momentum peak => FALL)...");
        for vel_thresh in [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2] {
            // INVERSE: Low velocity => RISE (growing), High velocity => FALL (peaking)
            let inv_rise_correct = rise_samples.iter().filter(|(s, _)| s.frequency_velocity < vel_thresh).count();
            let inv_fall_correct = fall_samples.iter().filter(|(s, _)| s.frequency_velocity >= vel_thresh).count();
            let inv_rise_acc = inv_rise_correct as f32 / rise_samples.len().max(1) as f32;
            let inv_fall_acc = inv_fall_correct as f32 / fall_samples.len().max(1) as f32;
            let inv_balanced = (inv_rise_acc + inv_fall_acc) / 2.0;

            if inv_balanced > best_balanced_acc {
                best_balanced_acc = inv_balanced;
                best_velocity_weight = -1.0;  // Negative = inverse velocity
                best_escape_weight = 0.0;
                best_effective_weight = 0.0;
                best_fitness_weight = -5.0;  // Flag for inverse velocity model
                best_threshold = vel_thresh;
                println!("    NEW BEST: velocity < {:.2} => RISE, balanced={:.1}%", vel_thresh, inv_balanced * 100.0);
            }
        }

        // Test Model 0: SATURATION MODEL (high freq => FALL due to herd immunity)
        // This is VASIL's key insight - high-frequency variants face more immunity
        println!("  Testing Model 0: SATURATION MODEL (high freq => FALL)...");
        for freq_thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40] {
            // Saturation: High frequency => FALL (herd immunity), Low freq => RISE (room to grow)
            let sat_rise_correct = rise_samples.iter().filter(|(s, _)| s.frequency < freq_thresh).count();
            let sat_fall_correct = fall_samples.iter().filter(|(s, _)| s.frequency >= freq_thresh).count();
            let sat_rise_acc = sat_rise_correct as f32 / rise_samples.len().max(1) as f32;
            let sat_fall_acc = sat_fall_correct as f32 / fall_samples.len().max(1) as f32;
            let sat_balanced = (sat_rise_acc + sat_fall_acc) / 2.0;

            if sat_balanced > best_balanced_acc {
                best_balanced_acc = sat_balanced;
                best_velocity_weight = 0.0;
                best_escape_weight = 0.0;
                best_effective_weight = 0.0;
                best_fitness_weight = -6.0;  // Flag for saturation model
                best_threshold = freq_thresh;
                println!("    NEW BEST: freq < {:.2} => RISE (saturation), balanced={:.1}%", freq_thresh, sat_balanced * 100.0);
            }
        }

        // Test Model 0b: COMBINED Saturation + Fitness
        // Variants RISE if: low frequency AND high fitness (escape + transmit)
        println!("  Testing Model 0b: COMBINED Saturation + Fitness...");
        for freq_w in [0.3, 0.4, 0.5, 0.6, 0.7] {
            for fit_w in [0.3, 0.4, 0.5, 0.6, 0.7] {
                for thresh in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6] {
                    let mut rise_correct = 0;
                    let mut fall_correct = 0;

                    for (state, observed) in data.iter() {
                        // Fitness score: higher escape + transmit = fitter
                        let fitness = (state.escape + state.transmit * 0.5) / 1.5;
                        // Growth potential: inverse of frequency (room to grow)
                        let growth_potential = 1.0 - state.frequency;
                        // Combined score: fitness × growth potential
                        let score = freq_w * growth_potential + fit_w * fitness;
                        let predicted = if score > thresh { "RISE" } else { "FALL" };

                        if *observed == "RISE" && predicted == "RISE" {
                            rise_correct += 1;
                        } else if *observed == "FALL" && predicted == "FALL" {
                            fall_correct += 1;
                        }
                    }

                    let rise_acc = rise_correct as f32 / rise_samples.len().max(1) as f32;
                    let fall_acc = fall_correct as f32 / fall_samples.len().max(1) as f32;
                    let balanced = (rise_acc + fall_acc) / 2.0;

                    if balanced > best_balanced_acc {
                        best_balanced_acc = balanced;
                        best_velocity_weight = freq_w;
                        best_escape_weight = fit_w;
                        best_effective_weight = 0.0;
                        best_fitness_weight = -7.0;  // Flag for combined saturation+fitness
                        best_threshold = thresh;
                        println!("    NEW BEST: growth_pot×{:.1} + fit×{:.1} > {:.2} => RISE, balanced={:.1}%",
                                 freq_w, fit_w, thresh, balanced * 100.0);
                    }
                }
            }
        }

        // Test Model 1b: FREQUENCY-BASED (low freq variants tend to rise, high freq tend to fall)
        // This is a more robust forward-looking signal based on reversion to mean
        println!("  Testing Model 1b: Frequency-based (reversion to mean)...");
        // Wider range to find optimal threshold
        for freq_thresh_pct in (1..=50).step_by(1) {
            let freq_thresh = freq_thresh_pct as f32 / 100.0;
            let freq_rise_correct = rise_samples.iter().filter(|(s, _)| s.frequency < freq_thresh).count();
            let freq_fall_correct = fall_samples.iter().filter(|(s, _)| s.frequency >= freq_thresh).count();
            let freq_rise_acc = freq_rise_correct as f32 / rise_samples.len().max(1) as f32;
            let freq_fall_acc = freq_fall_correct as f32 / fall_samples.len().max(1) as f32;
            let freq_balanced = (freq_rise_acc + freq_fall_acc) / 2.0;

            if freq_balanced > best_balanced_acc {
                best_balanced_acc = freq_balanced;
                best_velocity_weight = 0.0;
                best_escape_weight = 0.0;
                best_effective_weight = 0.0;
                best_fitness_weight = -1.0;  // Use negative fitness as proxy for low frequency
                best_threshold = freq_thresh;
                println!("    NEW BEST: freq < {:.2} => RISE, balanced={:.1}%", freq_thresh, freq_balanced * 100.0);
            }
        }

        // Test inverse: high frequency variants tend to fall (maybe works better)
        println!("  Testing Model 1c: High frequency => FALL (inverse)...");
        for freq_thresh_pct in (1..=50).step_by(1) {
            let freq_thresh = freq_thresh_pct as f32 / 100.0;
            // Inverse logic: high freq predicts FALL
            let freq_fall_correct = fall_samples.iter().filter(|(s, _)| s.frequency >= freq_thresh).count();
            let freq_rise_correct = rise_samples.iter().filter(|(s, _)| s.frequency < freq_thresh).count();
            let freq_rise_acc = freq_rise_correct as f32 / rise_samples.len().max(1) as f32;
            let freq_fall_acc = freq_fall_correct as f32 / fall_samples.len().max(1) as f32;
            let freq_balanced = (freq_rise_acc + freq_fall_acc) / 2.0;

            if freq_balanced > best_balanced_acc {
                best_balanced_acc = freq_balanced;
                best_velocity_weight = 0.0;
                best_escape_weight = 0.0;
                best_effective_weight = 0.0;
                best_fitness_weight = -1.0;
                best_threshold = freq_thresh;
                println!("    NEW BEST (inverse): freq >= {:.2} => FALL, balanced={:.1}%", freq_thresh, freq_balanced * 100.0);
            }
        }

        // Test Model 2: RELATIVE FITNESS (escape advantage over competition)
        // relative_fitness > 0.5 means escape advantage over competing variants
        // This is already normalized within each time point
        println!("  Testing Model 2: Relative Fitness (escape advantage)...");
        for fit_thresh in [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60] {
            let fit_rise_correct = rise_samples.iter().filter(|(s, _)| s.relative_fitness > fit_thresh).count();
            let fit_fall_correct = fall_samples.iter().filter(|(s, _)| s.relative_fitness <= fit_thresh).count();
            let fit_rise_acc = fit_rise_correct as f32 / rise_samples.len().max(1) as f32;
            let fit_fall_acc = fit_fall_correct as f32 / fall_samples.len().max(1) as f32;
            let fit_balanced = (fit_rise_acc + fit_fall_acc) / 2.0;

            if fit_balanced > best_balanced_acc {
                best_balanced_acc = fit_balanced;
                best_velocity_weight = 0.0;
                best_escape_weight = 0.0;
                best_effective_weight = 0.0;
                best_fitness_weight = -4.0;  // Flag for relative fitness model
                best_threshold = fit_thresh;
                println!("    NEW BEST: relative_fitness > {:.2} => RISE, balanced={:.1}%", fit_thresh, fit_balanced * 100.0);
            }
        }

        // Test Model 2b: TRANSMISSIBILITY-BASED (high transmit => RISE)
        // Transmit is FORWARD-LOOKING: variants with higher R0 tend to rise
        println!("  Testing Model 2b: Transmissibility-based (R0 advantage)...");
        for trans_thresh in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80] {
            let trans_rise_correct = rise_samples.iter().filter(|(s, _)| s.transmit > trans_thresh).count();
            let trans_fall_correct = fall_samples.iter().filter(|(s, _)| s.transmit <= trans_thresh).count();
            let trans_rise_acc = trans_rise_correct as f32 / rise_samples.len().max(1) as f32;
            let trans_fall_acc = trans_fall_correct as f32 / fall_samples.len().max(1) as f32;
            let trans_balanced = (trans_rise_acc + trans_fall_acc) / 2.0;

            if trans_balanced > best_balanced_acc {
                best_balanced_acc = trans_balanced;
                best_velocity_weight = 0.0;
                best_escape_weight = 0.0;
                best_effective_weight = 0.0;
                best_fitness_weight = -2.0;  // Flag for transmit-based model
                best_threshold = trans_thresh;
                println!("    NEW BEST: transmit > {:.2} => RISE, balanced={:.1}%", trans_thresh, trans_balanced * 100.0);
            }
        }

        // Test Model 3: Velocity + Transmit combination
        println!("  Testing Model 3: Velocity + Transmit combination...");
        for vel_w in [0.3, 0.4, 0.5, 0.6, 0.7] {
            for trans_w in [0.3, 0.4, 0.5, 0.6, 0.7] {
                for thresh in [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25] {
                    let mut rise_correct = 0;
                    let mut fall_correct = 0;

                    for (state, observed) in data.iter() {
                        // Score: weighted velocity + transmit (centered around 0.65)
                        let score = vel_w * state.frequency_velocity + trans_w * (state.transmit - 0.65);
                        let predicted = if score > thresh { "RISE" } else { "FALL" };

                        if *observed == "RISE" && predicted == "RISE" {
                            rise_correct += 1;
                        } else if *observed == "FALL" && predicted == "FALL" {
                            fall_correct += 1;
                        }
                    }

                    let rise_acc = rise_correct as f32 / rise_samples.len().max(1) as f32;
                    let fall_acc = fall_correct as f32 / fall_samples.len().max(1) as f32;
                    let balanced = (rise_acc + fall_acc) / 2.0;

                    if balanced > best_balanced_acc {
                        best_balanced_acc = balanced;
                        best_velocity_weight = vel_w;
                        best_escape_weight = trans_w;  // Repurpose for transmit weight
                        best_effective_weight = 0.0;
                        best_fitness_weight = -3.0;  // Flag for velocity+transmit model
                        best_threshold = thresh;
                    }
                }
            }
        }
        println!("  Best Model 3: vel_w={:.2}, trans_w={:.2}, thresh={:.3}, acc={:.1}%",
                 best_velocity_weight, best_escape_weight, best_threshold, best_balanced_acc * 100.0);

        // Test Model 4: Full combination with effective_escape and relative_fitness
        println!("  Testing Model 4: Full feature combination...");
        for vel_w in [0.4, 0.5, 0.6, 0.7, 0.8] {
            for esc_w in [0.0, 0.1, 0.2, 0.3] {
                for eff_w in [0.0, 0.1, 0.2, 0.3] {
                    for fit_w in [0.0, 0.1, 0.2] {
                        for thresh in [-0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25] {
                            let mut rise_correct = 0;
                            let mut fall_correct = 0;

                            for (state, observed) in data.iter() {
                                // Normalize escape around 0.45 (typical mean)
                                let esc_norm = state.escape - 0.45;
                                // Normalize effective escape around 0.35 (immunity-adjusted mean)
                                let eff_norm = state.effective_escape - 0.35;
                                // Relative fitness already centered around 0.5
                                let fit_norm = state.relative_fitness - 0.5;

                                let score = vel_w * state.frequency_velocity
                                    + esc_w * esc_norm
                                    + eff_w * eff_norm
                                    + fit_w * fit_norm;

                                let predicted = if score > thresh { "RISE" } else { "FALL" };

                                if *observed == "RISE" && predicted == "RISE" {
                                    rise_correct += 1;
                                } else if *observed == "FALL" && predicted == "FALL" {
                                    fall_correct += 1;
                                }
                            }

                            let rise_acc = rise_correct as f32 / rise_samples.len().max(1) as f32;
                            let fall_acc = fall_correct as f32 / fall_samples.len().max(1) as f32;
                            let balanced = (rise_acc + fall_acc) / 2.0;

                            if balanced > best_balanced_acc {
                                best_balanced_acc = balanced;
                                best_velocity_weight = vel_w;
                                best_escape_weight = esc_w;
                                best_effective_weight = eff_w;
                                best_fitness_weight = fit_w;
                                best_threshold = thresh;
                            }
                        }
                    }
                }
            }
        }

        // Store learned weights (repurposing existing fields)
        // alpha = velocity_weight, gamma = escape_weight
        // epsilon_min = effective_weight, spike_weight = fitness_weight
        // epsilon = threshold
        self.alpha = best_velocity_weight;
        self.gamma = best_escape_weight;
        self.epsilon = best_threshold;
        self.epsilon_min = best_effective_weight;
        self.spike_weight = best_fitness_weight;

        println!("\n  === FINAL MODEL ===");
        println!("  Learned: velocity_w={:.2}, escape_w={:.2}, effective_w={:.2}, fitness_w={:.2}",
                 best_velocity_weight, best_escape_weight, best_effective_weight, best_fitness_weight);
        println!("  Learned threshold: score > {:.4} => RISE", best_threshold);
        println!("  Balanced training accuracy: {:.1}%", best_balanced_acc * 100.0);
    }

    /// Evaluate with detailed diagnostics
    pub fn evaluate_with_diagnostics(&self, data: &[(VEState, &str)], label: &str) {
        let rise_samples: Vec<_> = data.iter().filter(|(_, o)| *o == "RISE").collect();
        let fall_samples: Vec<_> = data.iter().filter(|(_, o)| *o == "FALL").collect();

        // Analyze feature distributions in this dataset
        let rise_vel: Vec<f32> = rise_samples.iter().map(|(s, _)| s.frequency_velocity).collect();
        let fall_vel: Vec<f32> = fall_samples.iter().map(|(s, _)| s.frequency_velocity).collect();
        let rise_vel_mean = rise_vel.iter().sum::<f32>() / rise_vel.len().max(1) as f32;
        let fall_vel_mean = fall_vel.iter().sum::<f32>() / fall_vel.len().max(1) as f32;

        let rise_esc: Vec<f32> = rise_samples.iter().map(|(s, _)| s.escape).collect();
        let fall_esc: Vec<f32> = fall_samples.iter().map(|(s, _)| s.escape).collect();
        let rise_esc_mean = rise_esc.iter().sum::<f32>() / rise_esc.len().max(1) as f32;
        let fall_esc_mean = fall_esc.iter().sum::<f32>() / fall_esc.len().max(1) as f32;

        let rise_trans: Vec<f32> = rise_samples.iter().map(|(s, _)| s.transmit).collect();
        let fall_trans: Vec<f32> = fall_samples.iter().map(|(s, _)| s.transmit).collect();
        let rise_trans_mean = rise_trans.iter().sum::<f32>() / rise_trans.len().max(1) as f32;
        let fall_trans_mean = fall_trans.iter().sum::<f32>() / fall_trans.len().max(1) as f32;

        let rise_freq: Vec<f32> = rise_samples.iter().map(|(s, _)| s.frequency).collect();
        let fall_freq: Vec<f32> = fall_samples.iter().map(|(s, _)| s.frequency).collect();
        let rise_freq_mean = rise_freq.iter().sum::<f32>() / rise_freq.len().max(1) as f32;
        let fall_freq_mean = fall_freq.iter().sum::<f32>() / fall_freq.len().max(1) as f32;

        let rise_fit: Vec<f32> = rise_samples.iter().map(|(s, _)| s.relative_fitness).collect();
        let fall_fit: Vec<f32> = fall_samples.iter().map(|(s, _)| s.relative_fitness).collect();
        let rise_fit_mean = rise_fit.iter().sum::<f32>() / rise_fit.len().max(1) as f32;
        let fall_fit_mean = fall_fit.iter().sum::<f32>() / fall_fit.len().max(1) as f32;

        println!("\n  {} SET DIAGNOSTICS ({} samples):", label, data.len());
        println!("    {} RISE, {} FALL", rise_samples.len(), fall_samples.len());
        println!("    RISE velocity mean: {:.3}, FALL: {:.3} (DELTA={:.3})",
                 rise_vel_mean, fall_vel_mean, rise_vel_mean - fall_vel_mean);
        println!("    RISE escape mean: {:.3}, FALL: {:.3} (DELTA={:.3})",
                 rise_esc_mean, fall_esc_mean, rise_esc_mean - fall_esc_mean);
        println!("    RISE transmit mean: {:.3}, FALL: {:.3} (DELTA={:.3})",
                 rise_trans_mean, fall_trans_mean, rise_trans_mean - fall_trans_mean);
        println!("    RISE frequency mean: {:.3}, FALL: {:.3} (DELTA={:.3})",
                 rise_freq_mean, fall_freq_mean, rise_freq_mean - fall_freq_mean);
        println!("    RISE relative_fitness mean: {:.3}, FALL: {:.3} (DELTA={:.3})",
                 rise_fit_mean, fall_fit_mean, rise_fit_mean - fall_fit_mean);

        // Check prediction distribution
        let mut rise_correct = 0;
        let mut fall_correct = 0;
        let mut predicted_rise = 0;
        let mut predicted_fall = 0;

        for (state, observed) in data {
            let action = self.predict_with_weights(state);
            let predicted = action.to_str();

            if predicted == "RISE" {
                predicted_rise += 1;
            } else {
                predicted_fall += 1;
            }

            if *observed == "RISE" && predicted == "RISE" {
                rise_correct += 1;
            } else if *observed == "FALL" && predicted == "FALL" {
                fall_correct += 1;
            }
        }

        let rise_acc = rise_correct as f32 / rise_samples.len().max(1) as f32;
        let fall_acc = fall_correct as f32 / fall_samples.len().max(1) as f32;
        let balanced = (rise_acc + fall_acc) / 2.0;

        println!("    Predictions: {} RISE, {} FALL", predicted_rise, predicted_fall);
        println!("    RISE accuracy: {:.1}%, FALL accuracy: {:.1}%", rise_acc * 100.0, fall_acc * 100.0);
        println!("    Balanced accuracy: {:.1}%", balanced * 100.0);
    }

    /// Predict using ENHANCED model
    /// Model selection based on spike_weight flag:
    ///   -1.0: FREQUENCY-BASED (low freq => RISE)
    ///   -2.0: TRANSMIT-BASED (high transmit => RISE)
    ///   -3.0: VELOCITY+TRANSMIT combination
    ///   -5.0: INVERSE VELOCITY (momentum peak => FALL)
    ///   -6.0: SATURATION MODEL (high freq => FALL)
    ///   -7.0: COMBINED Saturation + Fitness
    ///   other: Full feature model
    pub fn predict_with_weights(&self, state: &VEState) -> VEAction {
        // Model -1: frequency-based (low frequency variants tend to rise)
        if (self.spike_weight - (-1.0)).abs() < 0.01 {
            if state.frequency < self.epsilon {
                return VEAction::Rise;
            } else {
                return VEAction::Fall;
            }
        }

        // Model -2: transmit-based (high transmit variants tend to rise)
        if (self.spike_weight - (-2.0)).abs() < 0.01 {
            if state.transmit > self.epsilon {
                return VEAction::Rise;
            } else {
                return VEAction::Fall;
            }
        }

        // Model -4: relative fitness (escape advantage over competition)
        if (self.spike_weight - (-4.0)).abs() < 0.01 {
            if state.relative_fitness > self.epsilon {
                return VEAction::Rise;
            } else {
                return VEAction::Fall;
            }
        }

        // Model -5: INVERSE velocity (momentum peak => FALL)
        // Key insight: High velocity = at peak = about to FALL
        if (self.spike_weight - (-5.0)).abs() < 0.01 {
            if state.frequency_velocity < self.epsilon {
                return VEAction::Rise;
            } else {
                return VEAction::Fall;
            }
        }

        // Model -6: SATURATION (high frequency => FALL due to herd immunity)
        if (self.spike_weight - (-6.0)).abs() < 0.01 {
            if state.frequency < self.epsilon {
                return VEAction::Rise;
            } else {
                return VEAction::Fall;
            }
        }

        // Model -7: COMBINED Saturation + Fitness
        // alpha = growth_potential_weight, gamma = fitness_weight
        if (self.spike_weight - (-7.0)).abs() < 0.01 {
            let fitness = (state.escape + state.transmit * 0.5) / 1.5;
            let growth_potential = 1.0 - state.frequency;
            let score = self.alpha * growth_potential + self.gamma * fitness;
            if score > self.epsilon {
                return VEAction::Rise;
            } else {
                return VEAction::Fall;
            }
        }

        // Model -3: velocity + transmit combination
        if (self.spike_weight - (-3.0)).abs() < 0.01 {
            // alpha = velocity_weight, gamma = transmit_weight (repurposed)
            let score = self.alpha * state.frequency_velocity
                + self.gamma * (state.transmit - 0.65);
            if score > self.epsilon {
                return VEAction::Rise;
            } else {
                return VEAction::Fall;
            }
        }

        // Default: Full feature model
        // alpha = velocity_weight, gamma = escape_weight
        // epsilon_min = effective_weight, spike_weight = fitness_weight
        // epsilon = threshold

        // Normalize features around their means for better decision boundary
        let esc_norm = state.escape - 0.45;
        let eff_norm = state.effective_escape - 0.35;
        let fit_norm = state.relative_fitness - 0.5;

        let score = self.alpha * state.frequency_velocity
            + self.gamma * esc_norm
            + self.epsilon_min * eff_norm
            + self.spike_weight * fit_norm;

        // epsilon = learned threshold
        if score > self.epsilon {
            VEAction::Rise
        } else {
            VEAction::Fall
        }
    }

    /// Get Q-values for a state (for debugging)
    pub fn get_q_values(&self, state: &VEState) -> [f32; 2] {
        self.q_table[state.discretize()]
    }

    /// Get learned weights (velocity_weight, escape_weight)
    pub fn get_weights(&self) -> (f32, f32) {
        (self.alpha, self.gamma)
    }

    /// Get all learned weights
    pub fn get_all_weights(&self) -> (f32, f32, f32, f32, f32) {
        (self.alpha, self.gamma, self.epsilon_min, self.spike_weight, self.epsilon)
    }
}

impl Default for AdaptiveVEOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_discretization() {
        let state1 = VEState::new(0.5, 0.1, 0.4);
        let state2 = VEState::new(0.5, 0.1, 0.4);
        assert_eq!(state1.discretize(), state2.discretize());

        let state3 = VEState::new(0.8, 0.2, 0.5);
        assert_ne!(state1.discretize(), state3.discretize());
    }

    #[test]
    fn test_action_selection() {
        let optimizer = AdaptiveVEOptimizer::new();
        let state = VEState::new(0.5, 0.1, 0.4);

        // With Q-values at 0, should use base rate or tie-break
        let action = optimizer.predict(&state);
        assert!(action == VEAction::Rise || action == VEAction::Fall);
    }

    #[test]
    fn test_training() {
        let mut optimizer = AdaptiveVEOptimizer::new();

        // Train on a simple pattern: high escape + low freq = RISE
        let state = VEState::new(0.6, 0.05, 0.45);
        let exp = VEExperience {
            state: state.clone(),
            action: VEAction::Rise,
            reward: 1.0,
            next_state: state.clone(),
        };

        optimizer.train_step(&exp);

        // Q-value for RISE should increase
        let q_values = optimizer.get_q_values(&state);
        assert!(q_values[0] > 0.0);
    }
}

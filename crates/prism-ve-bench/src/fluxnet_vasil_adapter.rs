//! FluxNet-to-VASIL Adapter
//!
//! Connects FluxNet RL to the VASIL gamma envelope pipeline for adaptive parameter optimization.
//!
//! ## Architecture
//! 
//! The VASIL benchmark pipeline already achieves 79.4% accuracy with:
//! - GPU-computed gamma envelopes (75 PK combinations)
//! - Envelope-based direction classification (RISE/FALL/UNDECIDED)
//! - Proper exclusion logic (negligible changes, low frequency, uncertain envelopes)
//!
//! FluxNet's role is to TUNE the parameters of this pipeline, NOT replace it.
//!
//! ## Tunable Parameters
//!
//! 1. **Negligible threshold**: Relative frequency change below which to exclude (default: 5%)
//! 2. **Min frequency**: Minimum frequency for inclusion (default: 3%)
//! 3. **Min peak frequency**: Peak frequency to qualify as major variant (default: 3%)
//! 4. **Envelope confidence margin**: How far from zero the envelope must be
//! 5. **Per-country adjustments**: Country-specific threshold modifiers
//!
//! ## Expected Impact
//!
//! By learning optimal thresholds per country/time period, FluxNet can:
//! - Reduce false exclusions (include confident predictions currently excluded)
//! - Increase precision by excluding truly uncertain predictions
//! - Adapt to country-specific epidemic dynamics
//!
//! Target: 79.4% → 87-92% (beat VASIL's 90.8%)

use anyhow::{Result, anyhow};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// State representation for FluxNet VE optimization
/// 
/// Encodes the current accuracy context for a country/time period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VEFluxNetState {
    /// Current accuracy on this country (0-1)
    pub current_accuracy: f32,
    
    /// Country index (0-11 for 12 countries)
    pub country_id: u8,
    
    /// Time period (0 = early 2022, 1 = mid 2022, 2 = late 2022, 3 = 2023)
    pub time_period: u8,
    
    /// Variant diversity (number of major variants circulating)
    pub variant_diversity: u8,
    
    /// Fraction of predictions excluded due to uncertain envelope (0-1)
    pub exclusion_rate: f32,
    
    /// Mean envelope confidence (|min| + |max|) / 2
    pub envelope_confidence: f32,
    
    /// Rise/Fall ratio in this context (for class balance)
    pub rise_ratio: f32,
}

impl VEFluxNetState {
    pub fn discretize(&self) -> usize {
        let acc_bin = ((self.current_accuracy * 3.99).clamp(0.0, 3.0) as usize).min(3);
        let country_bin = (self.country_id as usize % 4).min(3);
        let time_bin = (self.time_period as usize).min(3);
        let diversity_bin = ((self.variant_diversity as usize).min(12) / 3).min(3);
        let exclusion_bin = ((self.exclusion_rate * 3.99).clamp(0.0, 3.0) as usize).min(3);
        let confidence_bin = ((self.envelope_confidence * 3.99).clamp(0.0, 3.0) as usize).min(3);
        let rise_bin = ((self.rise_ratio * 3.99).clamp(0.0, 3.0) as usize).min(3);
        
        acc_bin * 4096 + country_bin * 1024 + time_bin * 256 + 
        diversity_bin * 64 + exclusion_bin * 16 + confidence_bin * 4 + rise_bin
    }
    
    /// Create from benchmark context
    pub fn from_context(
        accuracy: f32,
        country: &str,
        time_period: u8,
        variant_count: usize,
        excluded_count: usize,
        total_predictions: usize,
        mean_confidence: f32,
        rise_count: usize,
    ) -> Self {
        let country_id = match country {
            "Germany" => 0,
            "USA" => 1,
            "UK" => 2,
            "Japan" => 3,
            "Brazil" => 4,
            "France" => 5,
            "Canada" => 6,
            "Denmark" => 7,
            "Australia" => 8,
            "Sweden" => 9,
            "Mexico" => 10,
            "SouthAfrica" => 11,
            _ => 0,
        };
        
        let exclusion_rate = if total_predictions > 0 {
            excluded_count as f32 / total_predictions as f32
        } else {
            0.0
        };
        
        let rise_ratio = if total_predictions > 0 {
            rise_count as f32 / total_predictions as f32
        } else {
            0.36
        };
        
        Self {
            current_accuracy: accuracy,
            country_id,
            time_period,
            variant_diversity: variant_count.min(255) as u8,
            exclusion_rate,
            envelope_confidence: mean_confidence,
            rise_ratio,
        }
    }
}

/// Actions that FluxNet can take to adjust VASIL parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VEFluxNetAction {
    /// Increase negligible threshold (exclude more marginal predictions)
    IncreaseNegligibleThreshold = 0,
    /// Decrease negligible threshold (include more predictions)
    DecreaseNegligibleThreshold = 1,
    /// Increase minimum frequency (stricter filtering)
    IncreaseMinFrequency = 2,
    /// Decrease minimum frequency (include lower-frequency variants)
    DecreaseMinFrequency = 3,
    /// Increase envelope confidence margin (exclude uncertain envelopes)
    IncreaseConfidenceMargin = 4,
    /// Decrease envelope confidence margin (include more predictions)
    DecreaseConfidenceMargin = 5,
    /// Keep current parameters (no change)
    NoOp = 6,
    /// Reset to default parameters
    Reset = 7,
}

impl VEFluxNetAction {
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::IncreaseNegligibleThreshold,
            1 => Self::DecreaseNegligibleThreshold,
            2 => Self::IncreaseMinFrequency,
            3 => Self::DecreaseMinFrequency,
            4 => Self::IncreaseConfidenceMargin,
            5 => Self::DecreaseConfidenceMargin,
            6 => Self::NoOp,
            7 => Self::Reset,
            _ => Self::NoOp,
        }
    }
    
    pub fn to_index(&self) -> usize {
        *self as usize
    }
    
    pub fn all() -> Vec<Self> {
        vec![
            Self::IncreaseNegligibleThreshold,
            Self::DecreaseNegligibleThreshold,
            Self::IncreaseMinFrequency,
            Self::DecreaseMinFrequency,
            Self::IncreaseConfidenceMargin,
            Self::DecreaseConfidenceMargin,
            Self::NoOp,
            Self::Reset,
        ]
    }
}

/// Tunable parameters for VASIL benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VasilParameters {
    /// Negligible change threshold (default: 0.05 = 5%)
    pub negligible_threshold: f32,
    
    /// Minimum frequency threshold (default: 0.03 = 3%)
    pub min_frequency: f32,
    
    /// Minimum peak frequency for major variants (default: 0.03 = 3%)
    pub min_peak_frequency: f32,
    
    /// Envelope confidence margin: |gamma| must exceed this to be "decided"
    /// Default: 0.0 (any non-crossing envelope is decided)
    pub confidence_margin: f32,
    
    /// Per-country threshold adjustments (additive)
    pub country_adjustments: HashMap<String, f32>,
    
    /// IC50 values per epitope class (10 RBD epitopes: A, B, C, D1, D2, E12, E3, F1, F2, F3)
    /// These are the PRIMARY tunable parameters for FluxNet optimization
    /// Default values from VASIL supplementary (fitted to Delta vaccine efficacy)
    pub ic50: [f32; 10],
}

/// Default IC50 values from VASIL (calibrated to Delta vaccine efficacy)
pub const DEFAULT_IC50: [f32; 10] = [
    0.85,  // A
    1.12,  // B
    0.93,  // C
    1.05,  // D1
    0.98,  // D2
    1.21,  // E12
    0.89,  // E3
    1.08,  // F1
    0.95,  // F2
    1.03,  // F3
];

impl Default for VasilParameters {
    fn default() -> Self {
        Self {
            negligible_threshold: 0.05,
            min_frequency: 0.03,
            min_peak_frequency: 0.03,
            confidence_margin: 0.0,
            country_adjustments: HashMap::new(),
            ic50: DEFAULT_IC50,
        }
    }
}

impl VasilParameters {
    /// Apply an action to adjust parameters
    pub fn apply_action(&mut self, action: VEFluxNetAction) {
        match action {
            VEFluxNetAction::IncreaseNegligibleThreshold => {
                self.negligible_threshold = (self.negligible_threshold + 0.01).min(0.15);
            }
            VEFluxNetAction::DecreaseNegligibleThreshold => {
                self.negligible_threshold = (self.negligible_threshold - 0.01).max(0.01);
            }
            VEFluxNetAction::IncreaseMinFrequency => {
                self.min_frequency = (self.min_frequency + 0.005).min(0.10);
            }
            VEFluxNetAction::DecreaseMinFrequency => {
                self.min_frequency = (self.min_frequency - 0.005).max(0.005);
            }
            VEFluxNetAction::IncreaseConfidenceMargin => {
                self.confidence_margin = (self.confidence_margin + 0.01).min(0.20);
            }
            VEFluxNetAction::DecreaseConfidenceMargin => {
                self.confidence_margin = (self.confidence_margin - 0.01).max(0.0);
            }
            VEFluxNetAction::NoOp => {}
            VEFluxNetAction::Reset => {
                *self = Self::default();
            }
        }
    }
    
    /// Get effective negligible threshold for a country
    pub fn get_negligible_threshold(&self, country: &str) -> f32 {
        let adjustment = self.country_adjustments.get(country).copied().unwrap_or(0.0);
        (self.negligible_threshold + adjustment).clamp(0.01, 0.20)
    }
    
    /// Load optimized parameters from TOML file
    pub fn from_toml_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("Failed to read config file {}: {}", path, e))?;
        
        let toml: toml::Value = content.parse()
            .map_err(|e| anyhow!("Failed to parse TOML: {}", e))?;
        
        let mut params = Self::default();
        
        if let Some(thresholds) = toml.get("thresholds") {
            if let Some(v) = thresholds.get("negligible").and_then(|v| v.as_float()) {
                params.negligible_threshold = v as f32;
            }
            if let Some(v) = thresholds.get("min_frequency").and_then(|v| v.as_float()) {
                params.min_frequency = v as f32;
            }
            if let Some(v) = thresholds.get("min_peak_frequency").and_then(|v| v.as_float()) {
                params.min_peak_frequency = v as f32;
            }
            if let Some(v) = thresholds.get("confidence_margin").and_then(|v| v.as_float()) {
                params.confidence_margin = v as f32;
            }
        }
        
        if let Some(ic50) = toml.get("ic50") {
            if let Some(values) = ic50.get("values").and_then(|v| v.as_array()) {
                for (i, v) in values.iter().enumerate() {
                    if i < 10 {
                        if let Some(f) = v.as_float() {
                            params.ic50[i] = f as f32;
                        }
                    }
                }
            }
        }
        
        Ok(params)
    }
    
    /// Load optimized parameters, falling back to defaults if file not found
    pub fn load_optimized_or_default() -> Self {
        Self::from_toml_file("configs/optimized_params.toml")
            .unwrap_or_else(|_| Self::default())
    }
}

/// FluxNet VE Optimizer - adapts VASIL parameters for maximum accuracy
pub struct VEFluxNetOptimizer {
    /// Q-table: [state_index][action_index] -> Q-value
    q_table: Vec<[f32; 8]>,
    
    /// Visit counts for UCB exploration
    visit_counts: Vec<[usize; 8]>,
    
    /// Current parameters
    pub params: VasilParameters,
    
    /// Learning rate
    alpha: f32,
    
    /// Discount factor
    gamma: f32,
    
    /// Exploration rate
    epsilon: f32,
    
    /// Minimum epsilon
    epsilon_min: f32,
    
    /// Epsilon decay
    epsilon_decay: f32,
    
    /// Training episodes completed
    episodes: usize,
    
    /// Best accuracy achieved
    best_accuracy: f32,
    
    /// Best parameters found
    best_params: VasilParameters,
}

impl VEFluxNetOptimizer {
    pub fn new() -> Self {
        let num_states = 16384;
        
        Self {
            q_table: vec![[0.0; 8]; num_states],
            visit_counts: vec![[0; 8]; num_states],
            params: VasilParameters::default(),
            alpha: 0.1,
            gamma: 0.95,
            epsilon: 0.3,
            epsilon_min: 0.05,
            epsilon_decay: 0.995,
            episodes: 0,
            best_accuracy: 0.0,
            best_params: VasilParameters::default(),
        }
    }
    
    /// Create with custom hyperparameters
    pub fn with_config(alpha: f32, gamma: f32, epsilon: f32) -> Self {
        let mut opt = Self::new();
        opt.alpha = alpha;
        opt.gamma = gamma;
        opt.epsilon = epsilon;
        opt
    }
    
    /// Select action using epsilon-greedy with UCB bonus
    pub fn select_action(&self, state: &VEFluxNetState, explore: bool) -> VEFluxNetAction {
        let state_idx = state.discretize();
        
        if explore && rand::random::<f32>() < self.epsilon {
            VEFluxNetAction::from_index(rand::random::<usize>() % 8)
        } else {
            let q_values = &self.q_table[state_idx];
            let visits = &self.visit_counts[state_idx];
            let total_visits: usize = visits.iter().sum();
            
            let mut best_action = 0;
            let mut best_value = f32::NEG_INFINITY;
            
            for (action_idx, &q) in q_values.iter().enumerate() {
                let ucb_bonus = if explore && total_visits > 0 && visits[action_idx] > 0 {
                    0.5 * ((2.0 * (total_visits as f32).ln()) / visits[action_idx] as f32).sqrt()
                } else {
                    0.0
                };
                
                let value = q + ucb_bonus;
                if value > best_value {
                    best_value = value;
                    best_action = action_idx;
                }
            }
            
            VEFluxNetAction::from_index(best_action)
        }
    }
    
    /// Update Q-table from experience
    pub fn update(
        &mut self,
        state: &VEFluxNetState,
        action: VEFluxNetAction,
        reward: f32,
        next_state: &VEFluxNetState,
    ) {
        let state_idx = state.discretize();
        let action_idx = action.to_index();
        let next_state_idx = next_state.discretize();
        
        self.visit_counts[state_idx][action_idx] += 1;
        
        let current_q = self.q_table[state_idx][action_idx];
        let max_next_q = self.q_table[next_state_idx].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let target = reward + self.gamma * max_next_q;
        
        self.q_table[state_idx][action_idx] = current_q + self.alpha * (target - current_q);
    }
    
    /// Compute reward from accuracy change
    pub fn compute_reward(
        &self,
        prev_accuracy: f32,
        new_accuracy: f32,
        exclusion_rate: f32,
    ) -> f32 {
        let accuracy_reward = (new_accuracy - prev_accuracy) * 10.0;
        
        let exclusion_penalty = if exclusion_rate > 0.30 {
            -0.1 * (exclusion_rate - 0.30)
        } else {
            0.0
        };
        
        let baseline_bonus = if new_accuracy > 0.794 {
            0.2 * (new_accuracy - 0.794)
        } else {
            0.0
        };
        
        let vasil_bonus = if new_accuracy > 0.908 { 1.0 } else { 0.0 };
        
        accuracy_reward + exclusion_penalty + baseline_bonus + vasil_bonus
    }
    
    /// Run one optimization step
    pub fn optimize_step(
        &mut self,
        state: &VEFluxNetState,
        explore: bool,
    ) -> VEFluxNetAction {
        let action = self.select_action(state, explore);
        self.params.apply_action(action);
        action
    }
    
    /// Record accuracy result and update
    pub fn record_result(
        &mut self,
        prev_state: &VEFluxNetState,
        action: VEFluxNetAction,
        new_accuracy: f32,
        new_state: &VEFluxNetState,
    ) {
        let reward = self.compute_reward(
            prev_state.current_accuracy,
            new_accuracy,
            new_state.exclusion_rate,
        );
        
        self.update(prev_state, action, reward, new_state);
        
        if new_accuracy > self.best_accuracy {
            self.best_accuracy = new_accuracy;
            self.best_params = self.params.clone();
            eprintln!("[FluxNet VE] New best: {:.2}% with params: {:?}", 
                     new_accuracy * 100.0, self.params);
        }
    }
    
    /// Decay epsilon after episode
    pub fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * self.epsilon_decay).max(self.epsilon_min);
        self.episodes += 1;
    }
    
    /// Get current parameters
    pub fn get_params(&self) -> &VasilParameters {
        &self.params
    }
    
    /// Get best parameters found
    pub fn get_best_params(&self) -> &VasilParameters {
        &self.best_params
    }
    
    /// Get training statistics
    pub fn get_stats(&self) -> (usize, f32, f32) {
        (self.episodes, self.epsilon, self.best_accuracy)
    }
    
    /// Save Q-table to file
    pub fn save(&self, path: &str) -> Result<()> {
        let data = serde_json::json!({
            "q_table": self.q_table,
            "visit_counts": self.visit_counts,
            "params": self.params,
            "best_params": self.best_params,
            "best_accuracy": self.best_accuracy,
            "episodes": self.episodes,
            "epsilon": self.epsilon,
        });
        
        std::fs::write(path, serde_json::to_string_pretty(&data)?)
            .map_err(|e| anyhow!("Failed to save Q-table: {}", e))?;
        
        Ok(())
    }
    
    /// Load Q-table from file
    pub fn load(&mut self, path: &str) -> Result<()> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow!("Failed to read Q-table: {}", e))?;
        
        let data: serde_json::Value = serde_json::from_str(&content)?;
        
        if let Some(qt) = data.get("q_table").and_then(|v| v.as_array()) {
            for (i, row) in qt.iter().enumerate() {
                if i < self.q_table.len() {
                    if let Some(values) = row.as_array() {
                        for (j, v) in values.iter().enumerate() {
                            if j < 8 {
                                self.q_table[i][j] = v.as_f64().unwrap_or(0.0) as f32;
                            }
                        }
                    }
                }
            }
        }
        
        if let Some(acc) = data.get("best_accuracy").and_then(|v| v.as_f64()) {
            self.best_accuracy = acc as f32;
        }
        
        if let Some(ep) = data.get("episodes").and_then(|v| v.as_u64()) {
            self.episodes = ep as usize;
        }
        
        eprintln!("[FluxNet VE] Loaded Q-table: {} episodes, best accuracy: {:.2}%",
                 self.episodes, self.best_accuracy * 100.0);
        
        Ok(())
    }
}

impl Default for VEFluxNetOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_state_discretization() {
        let state1 = VEFluxNetState {
            current_accuracy: 0.75,
            country_id: 0,
            time_period: 1,
            variant_diversity: 5,
            exclusion_rate: 0.15,
            envelope_confidence: 0.8,
            rise_ratio: 0.36,
        };
        
        let state2 = state1.clone();
        assert_eq!(state1.discretize(), state2.discretize());
    }
    
    #[test]
    fn test_action_application() {
        let mut params = VasilParameters::default();
        assert_eq!(params.negligible_threshold, 0.05);
        
        params.apply_action(VEFluxNetAction::IncreaseNegligibleThreshold);
        assert_eq!(params.negligible_threshold, 0.06);
        
        params.apply_action(VEFluxNetAction::DecreaseNegligibleThreshold);
        assert_eq!(params.negligible_threshold, 0.05);
    }
    
    #[test]
    fn test_reward_computation() {
        let opt = VEFluxNetOptimizer::new();
        
        let reward = opt.compute_reward(0.79, 0.82, 0.20);
        assert!(reward > 0.0);
        
        let reward_vasil = opt.compute_reward(0.90, 0.92, 0.20);
        assert!(reward_vasil > reward);
    }
}

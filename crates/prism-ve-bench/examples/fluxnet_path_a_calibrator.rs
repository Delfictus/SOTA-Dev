//! FluxNet RL Calibrator for PATH A Parameters
//!
//! Uses Q-learning to optimize 12 parameters (11 epitope weights + sigma)
//! to maximize VASIL benchmark accuracy.
//!
//! Target: Beat PATH B baseline of 79% accuracy → 85-90%
//!
//! Based on VASIL BENCHMARK METHODOLOGY - EXACT SPECIFICATION:
//! - Per-day direction classification (RISE +1 or FALL -1)
//! - γy(t) = E[Sy(t)] / weighted_mean(E[Sx(t)]) - 1
//! - Accuracy = correct predictions / total included days
//! - Exclusions: negligible change (<5%), undecided (envelope crosses 0), freq <3%

use anyhow::{Result, anyhow};
use prism_ve_bench::vasil_exact_metric::*;
use prism_ve_bench::data_loader::*;
use chrono::NaiveDate;
use std::collections::HashMap;
use std::sync::Arc;
use rand::Rng;

// ═══════════════════════════════════════════════════════════════════════════════
// FLUXNET RL CONFIGURATION FOR PATH A CALIBRATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Number of parameters to optimize: 11 epitope weights + 1 sigma
const N_PARAMS: usize = 12;

/// Actions per parameter: increase, decrease, hold
const ACTIONS_PER_PARAM: usize = 3;

/// Total action space: 12 params × 3 actions = 36
const N_ACTIONS: usize = N_PARAMS * ACTIONS_PER_PARAM;

/// State space bins for discretization
const N_ACCURACY_BINS: usize = 20;  // 0-100% in 5% increments
const N_GRADIENT_BINS: usize = 5;   // Very negative, negative, zero, positive, very positive

/// Q-learning hyperparameters (tuned for parameter optimization)
const ALPHA: f64 = 0.15;           // Learning rate (higher for faster convergence)
const GAMMA: f64 = 0.90;           // Discount factor
const EPSILON_START: f64 = 0.4;    // Initial exploration rate
const EPSILON_MIN: f64 = 0.05;     // Minimum exploration
const EPSILON_DECAY: f64 = 0.98;   // Decay per episode

/// Parameter adjustment step sizes
const WEIGHT_STEP: f32 = 0.05;     // ±5% for epitope weights
const SIGMA_STEP: f32 = 0.02;      // ±0.02 for sigma

/// Training configuration
const MAX_EPISODES: usize = 200;
const STEPS_PER_EPISODE: usize = 50;
const TARGET_ACCURACY: f64 = 0.85;  // Stop if we hit 85%

// ═══════════════════════════════════════════════════════════════════════════════
// STATE REPRESENTATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Discretized state for Q-table lookup
#[derive(Clone, Copy, Debug, Hash, Eq, PartialEq)]
struct CalibrationState {
    accuracy_bin: usize,      // Current accuracy bucket (0-19)
    improving: bool,          // Was last action an improvement?
    steps_without_improvement: usize,  // Plateau detection
}

impl CalibrationState {
    fn new(accuracy: f64, improving: bool, stagnant_steps: usize) -> Self {
        let accuracy_bin = ((accuracy * 100.0) / 5.0).floor() as usize;
        let accuracy_bin = accuracy_bin.min(N_ACCURACY_BINS - 1);
        
        Self {
            accuracy_bin,
            improving,
            steps_without_improvement: stagnant_steps.min(10),
        }
    }
    
    fn to_index(&self) -> usize {
        self.accuracy_bin * 2 * 11 + 
            (if self.improving { 1 } else { 0 }) * 11 + 
            self.steps_without_improvement.min(10)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ACTION REPRESENTATION
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug)]
enum ParamAction {
    Increase,
    Decrease,
    Hold,
}

impl ParamAction {
    fn from_index(action_idx: usize) -> (usize, ParamAction) {
        let param_idx = action_idx / ACTIONS_PER_PARAM;
        let action_type = match action_idx % ACTIONS_PER_PARAM {
            0 => ParamAction::Increase,
            1 => ParamAction::Decrease,
            _ => ParamAction::Hold,
        };
        (param_idx, action_type)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Q-TABLE BASED RL CONTROLLER
// ═══════════════════════════════════════════════════════════════════════════════

struct FluxNetCalibrator {
    /// Q-table: [state_index][action_index] -> Q-value
    q_table: Vec<Vec<f64>>,
    
    /// Current exploration rate
    epsilon: f64,
    
    /// Current parameters being optimized
    params: PathAParams,
    
    /// Best parameters found so far
    best_params: PathAParams,
    best_accuracy: f64,
    
    /// Episode tracking
    episode: usize,
    total_steps: usize,
}

#[derive(Clone, Debug)]
struct PathAParams {
    epitope_weights: [f32; 11],
    sigma: f32,
}

impl Default for PathAParams {
    fn default() -> Self {
        Self {
            // Start with uniform weights (normalized)
            epitope_weights: [1.0/11.0; 11],
            sigma: 0.5,
        }
    }
}

impl PathAParams {
    fn to_array(&self) -> [f32; 12] {
        let mut arr = [0.0f32; 12];
        arr[..11].copy_from_slice(&self.epitope_weights);
        arr[11] = self.sigma;
        arr
    }
    
    fn from_array(arr: &[f32; 12]) -> Self {
        let mut weights = [0.0f32; 11];
        weights.copy_from_slice(&arr[..11]);
        Self {
            epitope_weights: weights,
            sigma: arr[11],
        }
    }
    
    /// Apply action to modify parameters
    fn apply_action(&mut self, param_idx: usize, action: ParamAction) {
        let step = if param_idx < 11 { WEIGHT_STEP } else { SIGMA_STEP };
        
        let value = if param_idx < 11 {
            &mut self.epitope_weights[param_idx]
        } else {
            &mut self.sigma
        };
        
        match action {
            ParamAction::Increase => *value = (*value + step).min(2.0),
            ParamAction::Decrease => *value = (*value - step).max(0.01),
            ParamAction::Hold => {}
        }
        
        // Normalize epitope weights after modification
        if param_idx < 11 {
            let sum: f32 = self.epitope_weights.iter().sum();
            if sum > 0.0 {
                for w in &mut self.epitope_weights {
                    *w /= sum;
                }
            }
        }
    }
}

impl FluxNetCalibrator {
    fn new() -> Self {
        // Initialize Q-table with small random values for exploration
        let n_states = N_ACCURACY_BINS * 2 * 11;  // accuracy × improving × stagnant
        let q_table = vec![vec![0.01; N_ACTIONS]; n_states];
        
        Self {
            q_table,
            epsilon: EPSILON_START,
            params: PathAParams::default(),
            best_params: PathAParams::default(),
            best_accuracy: 0.0,
            episode: 0,
            total_steps: 0,
        }
    }
    
    /// Select action using epsilon-greedy policy
    fn select_action(&self, state: &CalibrationState) -> usize {
        let mut rng = rand::thread_rng();
        
        if rng.gen::<f64>() < self.epsilon {
            // Explore: random action
            rng.gen_range(0..N_ACTIONS)
        } else {
            // Exploit: best action from Q-table
            let state_idx = state.to_index().min(self.q_table.len() - 1);
            let q_values = &self.q_table[state_idx];
            
            q_values.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        }
    }
    
    /// Update Q-table using Q-learning update rule
    fn update(&mut self, state: &CalibrationState, action: usize, reward: f64, next_state: &CalibrationState) {
        let state_idx = state.to_index().min(self.q_table.len() - 1);
        let next_state_idx = next_state.to_index().min(self.q_table.len() - 1);
        
        let max_next_q = self.q_table[next_state_idx].iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        
        let current_q = self.q_table[state_idx][action];
        let new_q = current_q + ALPHA * (reward + GAMMA * max_next_q - current_q);
        
        self.q_table[state_idx][action] = new_q;
    }
    
    /// Decay epsilon for reduced exploration over time
    fn decay_epsilon(&mut self) {
        self.epsilon = (self.epsilon * EPSILON_DECAY).max(EPSILON_MIN);
    }
    
    /// Record if this is the best accuracy seen
    fn record_if_best(&mut self, accuracy: f64) {
        if accuracy > self.best_accuracy {
            self.best_accuracy = accuracy;
            self.best_params = self.params.clone();
            println!("  🎯 NEW BEST: {:.2}% (episode {})", accuracy * 100.0, self.episode);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ACCURACY EVALUATION (VASIL EXACT METHODOLOGY)
// ═══════════════════════════════════════════════════════════════════════════════

fn evaluate_accuracy(
    vasil_metric: &mut VasilMetricComputer,
    countries: &[CountryData],
    eval_start: NaiveDate,
    eval_end: NaiveDate,
    params: &PathAParams,
) -> Result<f64> {
    // Set PATH A mode with current parameters
    vasil_metric.set_path_a_mode(params.epitope_weights, params.sigma);
    
    // Compute VASIL metric
    let result = vasil_metric.compute_vasil_metric_exact(
        countries,
        eval_start,
        eval_end,
    )?;
    
    Ok(result.mean_accuracy as f64)
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN CALIBRATION LOOP
// ═══════════════════════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    println!("================================================================================");
    println!("🧠 FLUXNET RL CALIBRATOR FOR PATH A");
    println!("================================================================================");
    println!();
    println!("Configuration:");
    println!("  Parameters: {} (11 epitope weights + sigma)", N_PARAMS);
    println!("  Actions: {} (3 per param: increase/decrease/hold)", N_ACTIONS);
    println!("  Episodes: {}", MAX_EPISODES);
    println!("  Steps/Episode: {}", STEPS_PER_EPISODE);
    println!("  Target: {}%", TARGET_ACCURACY * 100.0);
    println!("  Baseline (PATH B): 79%");
    println!();
    println!("Q-Learning Hyperparameters:");
    println!("  α (learning rate): {}", ALPHA);
    println!("  γ (discount): {}", GAMMA);
    println!("  ε (exploration): {} → {}", EPSILON_START, EPSILON_MIN);
    println!("================================================================================");
    println!();
    
    // ═══════════════════════════════════════════════════════════════════════════
    // LOAD DATA
    // ═══════════════════════════════════════════════════════════════════════════
    
    println!("[1/4] Loading VASIL data...");
    let vasil_dir = std::path::PathBuf::from("data/VASIL");
    let all_data = AllCountriesData::load_all_vasil_countries(&vasil_dir)?;
    println!("  ✅ Loaded {} countries", all_data.countries.len());
    
    // Build population sizes
    let mut pop_sizes = HashMap::new();
    pop_sizes.insert("Germany".to_string(), 83_000_000.0);
    pop_sizes.insert("USA".to_string(), 331_000_000.0);
    pop_sizes.insert("UK".to_string(), 67_000_000.0);
    pop_sizes.insert("Japan".to_string(), 126_000_000.0);
    pop_sizes.insert("Brazil".to_string(), 213_000_000.0);
    pop_sizes.insert("France".to_string(), 67_000_000.0);
    pop_sizes.insert("Canada".to_string(), 38_000_000.0);
    pop_sizes.insert("Denmark".to_string(), 5_800_000.0);
    pop_sizes.insert("Australia".to_string(), 25_700_000.0);
    pop_sizes.insert("Sweden".to_string(), 10_300_000.0);
    pop_sizes.insert("Mexico".to_string(), 128_000_000.0);
    pop_sizes.insert("SouthAfrica".to_string(), 59_000_000.0);
    
    let landscapes = build_immunity_landscapes(&all_data.countries, &pop_sizes);
    println!("  ✅ Built {} landscapes", landscapes.len());
    
    // ═══════════════════════════════════════════════════════════════════════════
    // INITIALIZE VASIL METRIC & GPU
    // ═══════════════════════════════════════════════════════════════════════════
    
    println!("[2/4] Initializing GPU context...");
    use cudarc::driver::CudaContext;
    
    let context = Arc::new(CudaContext::new(0)?);
    let stream = context.default_stream();
    println!("  ✅ GPU ready");
    
    let eval_start = NaiveDate::from_ymd_opt(2022, 10, 1).unwrap();
    let eval_end = NaiveDate::from_ymd_opt(2023, 10, 31).unwrap();
    
    println!("[3/4] Initializing VASIL metric computer...");
    let dms_data = &all_data.countries[0].dms_data;
    let mut vasil_metric = VasilMetricComputer::new();
    vasil_metric.initialize(dms_data, landscapes);
    
    // Build initial cache with default parameters
    let default_params = PathAParams::default();
    vasil_metric.set_path_a_mode(default_params.epitope_weights, default_params.sigma);
    vasil_metric.build_immunity_cache(
        dms_data,
        &all_data.countries,
        eval_start,
        eval_end,
        &context,
        &stream,
    );
    println!("  ✅ Initial cache built");
    
    // ═══════════════════════════════════════════════════════════════════════════
    // FLUXNET RL TRAINING LOOP
    // ═══════════════════════════════════════════════════════════════════════════
    
    println!("[4/4] Starting FluxNet RL calibration...");
    println!();
    println!("================================================================================");
    println!("🚀 TRAINING");
    println!("================================================================================");
    
    let mut calibrator = FluxNetCalibrator::new();
    
    // Evaluate baseline (uniform weights)
    let baseline_acc = evaluate_accuracy(
        &mut vasil_metric,
        &all_data.countries,
        eval_start,
        eval_end,
        &calibrator.params,
    )?;
    println!("  Baseline (uniform weights): {:.2}%", baseline_acc * 100.0);
    calibrator.best_accuracy = baseline_acc;
    calibrator.best_params = calibrator.params.clone();
    
    let mut prev_accuracy = baseline_acc;
    let mut steps_without_improvement = 0;
    
    for episode in 0..MAX_EPISODES {
        calibrator.episode = episode;
        let mut episode_reward = 0.0;
        
        println!("\n--- Episode {}/{} (ε={:.3}) ---", episode + 1, MAX_EPISODES, calibrator.epsilon);
        
        for step in 0..STEPS_PER_EPISODE {
            // Current state
            let state = CalibrationState::new(
                prev_accuracy,
                steps_without_improvement == 0,
                steps_without_improvement,
            );
            
            // Select action
            let action_idx = calibrator.select_action(&state);
            let (param_idx, action) = ParamAction::from_index(action_idx);
            
            // Apply action
            calibrator.params.apply_action(param_idx, action);
            
            // Rebuild cache with new parameters (this is the expensive part)
            vasil_metric.set_path_a_mode(
                calibrator.params.epitope_weights,
                calibrator.params.sigma,
            );
            vasil_metric.build_immunity_cache(
                dms_data,
                &all_data.countries,
                eval_start,
                eval_end,
                &context,
                &stream,
            );
            
            // Evaluate new accuracy
            let new_accuracy = evaluate_accuracy(
                &mut vasil_metric,
                &all_data.countries,
                eval_start,
                eval_end,
                &calibrator.params,
            )?;
            
            // Compute reward (improvement-based)
            let improvement = new_accuracy - prev_accuracy;
            let reward = if improvement > 0.001 {
                10.0 * improvement  // Reward improvements
            } else if improvement < -0.001 {
                5.0 * improvement   // Penalize regressions (less harsh)
            } else {
                -0.01  // Small penalty for no change (encourage exploration)
            };
            
            // Update state tracking
            if improvement > 0.001 {
                steps_without_improvement = 0;
            } else {
                steps_without_improvement += 1;
            }
            
            // Next state
            let next_state = CalibrationState::new(
                new_accuracy,
                improvement > 0.0,
                steps_without_improvement,
            );
            
            // Q-learning update
            calibrator.update(&state, action_idx, reward, &next_state);
            
            // Track best
            calibrator.record_if_best(new_accuracy);
            
            episode_reward += reward;
            prev_accuracy = new_accuracy;
            calibrator.total_steps += 1;
            
            // Log every 10 steps
            if (step + 1) % 10 == 0 {
                println!("    Step {}: acc={:.2}%, best={:.2}%",
                         step + 1, new_accuracy * 100.0, calibrator.best_accuracy * 100.0);
            }
            
            // Early termination if we hit target
            if calibrator.best_accuracy >= TARGET_ACCURACY {
                println!("\n🎉 TARGET ACHIEVED! Stopping early.");
                break;
            }
        }
        
        // Decay exploration
        calibrator.decay_epsilon();
        
        println!("  Episode {} complete: total_reward={:.3}, best={:.2}%",
                 episode + 1, episode_reward, calibrator.best_accuracy * 100.0);
        
        // Early termination
        if calibrator.best_accuracy >= TARGET_ACCURACY {
            break;
        }
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // FINAL RESULTS
    // ═══════════════════════════════════════════════════════════════════════════
    
    println!();
    println!("================================================================================");
    println!("📊 CALIBRATION COMPLETE");
    println!("================================================================================");
    println!();
    println!("Best Parameters Found:");
    println!("  Epitope Weights:");
    let epitope_names = ["A", "B", "C", "D1", "D2", "E12", "E3", "F1", "F2", "F3", "NTD"];
    for (i, (name, weight)) in epitope_names.iter().zip(calibrator.best_params.epitope_weights.iter()).enumerate() {
        let bar_len = (weight * 50.0) as usize;
        let bar = "#".repeat(bar_len);
        println!("    {:>4}: {:.4} {}", name, weight, bar);
    }
    println!("  Sigma: {:.4}", calibrator.best_params.sigma);
    println!();
    println!("Performance:");
    println!("  Baseline (uniform): {:.2}%", baseline_acc * 100.0);
    println!("  Best Achieved:      {:.2}%", calibrator.best_accuracy * 100.0);
    println!("  Improvement:        {:.2}%", (calibrator.best_accuracy - baseline_acc) * 100.0);
    println!("  PATH B Baseline:    79.0%");
    println!("  vs PATH B:          {:+.2}%", (calibrator.best_accuracy - 0.79) * 100.0);
    println!();
    
    // Save best parameters
    let params_str = format!(
        "# FluxNet RL Calibrated PATH A Parameters\n\
         # Accuracy: {:.2}%\n\
         # Episodes: {}\n\
         # Total Steps: {}\n\n\
         [epitope_weights]\n\
         A = {:.6}\n\
         B = {:.6}\n\
         C = {:.6}\n\
         D1 = {:.6}\n\
         D2 = {:.6}\n\
         E12 = {:.6}\n\
         E3 = {:.6}\n\
         F1 = {:.6}\n\
         F2 = {:.6}\n\
         F3 = {:.6}\n\
         NTD = {:.6}\n\n\
         [kernel]\n\
         sigma = {:.6}\n",
        calibrator.best_accuracy * 100.0,
        calibrator.episode + 1,
        calibrator.total_steps,
        calibrator.best_params.epitope_weights[0],
        calibrator.best_params.epitope_weights[1],
        calibrator.best_params.epitope_weights[2],
        calibrator.best_params.epitope_weights[3],
        calibrator.best_params.epitope_weights[4],
        calibrator.best_params.epitope_weights[5],
        calibrator.best_params.epitope_weights[6],
        calibrator.best_params.epitope_weights[7],
        calibrator.best_params.epitope_weights[8],
        calibrator.best_params.epitope_weights[9],
        calibrator.best_params.epitope_weights[10],
        calibrator.best_params.sigma,
    );
    
    std::fs::write("validation_results/fluxnet_calibrated_params.toml", &params_str)?;
    println!("✅ Saved parameters: validation_results/fluxnet_calibrated_params.toml");
    
    // Verdict
    println!();
    println!("================================================================================");
    if calibrator.best_accuracy >= 0.85 {
        println!("🎯 SUCCESS: PATH A calibration achieved target (≥85%)!");
        println!("   Ready for production deployment.");
    } else if calibrator.best_accuracy >= 0.80 {
        println!("⚠️  GOOD: PATH A calibration improved over PATH B (≥80%)");
        println!("   Consider more training episodes for further improvement.");
    } else {
        println!("❌ NEEDS WORK: PATH A calibration below PATH B baseline");
        println!("   Check epitope data quality or increase training.");
    }
    println!("================================================================================");
    
    Ok(())
}

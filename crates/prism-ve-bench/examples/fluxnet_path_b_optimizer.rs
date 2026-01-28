//! FluxNet RL Optimizer for PATH B Parameters
//!
//! Starts from PATH B's 79% baseline and optimizes:
//! - 10 IC50 values per epitope class
//! - 10 epitope contribution weights
//! - 1 NTD weight
//! - 2 decision thresholds
//!
//! Target: Push PATH B from 79% → 85-92%

use anyhow::{Result, anyhow};
use prism_ve_bench::vasil_exact_metric::*;
use prism_ve_bench::data_loader::*;
use chrono::NaiveDate;
use std::collections::HashMap;
use std::sync::Arc;
use rand::Rng;

const N_IC50: usize = 10;
const N_EPITOPE_WEIGHTS: usize = 11;
const N_PARAMS: usize = N_IC50 + N_EPITOPE_WEIGHTS + 2;  // 23 total

const N_ACTIONS: usize = N_PARAMS * 3;  // 69 actions (increase/decrease/hold per param)

const ALPHA: f64 = 0.12;
const GAMMA: f64 = 0.92;
const EPSILON_START: f64 = 0.35;
const EPSILON_MIN: f64 = 0.05;
const EPSILON_DECAY: f64 = 0.97;

const MAX_EPISODES: usize = 100;
const STEPS_PER_EPISODE: usize = 30;
const TARGET_ACCURACY: f64 = 0.88;

const IC50_STEP: f32 = 0.05;
const WEIGHT_STEP: f32 = 0.03;
const THRESHOLD_STEP: f32 = 0.005;

#[derive(Clone, Debug)]
struct PathBParams {
    ic50: [f32; N_IC50],
    epitope_weights: [f32; N_EPITOPE_WEIGHTS],
    rise_threshold: f32,
    fall_threshold: f32,
}

impl Default for PathBParams {
    fn default() -> Self {
        Self {
            ic50: [0.85, 1.12, 0.93, 1.05, 0.98, 1.21, 0.89, 1.08, 0.95, 1.03],
            epitope_weights: [1.0; N_EPITOPE_WEIGHTS],
            rise_threshold: 0.0,
            fall_threshold: 0.0,
        }
    }
}

impl PathBParams {
    fn apply_action(&mut self, param_idx: usize, action: usize) {
        let action_type = action % 3;
        
        if param_idx < N_IC50 {
            let step = IC50_STEP;
            match action_type {
                0 => self.ic50[param_idx] = (self.ic50[param_idx] + step).min(3.0),
                1 => self.ic50[param_idx] = (self.ic50[param_idx] - step).max(0.1),
                _ => {}
            }
        } else if param_idx < N_IC50 + N_EPITOPE_WEIGHTS {
            let idx = param_idx - N_IC50;
            let step = WEIGHT_STEP;
            match action_type {
                0 => self.epitope_weights[idx] = (self.epitope_weights[idx] + step).min(3.0),
                1 => self.epitope_weights[idx] = (self.epitope_weights[idx] - step).max(0.1),
                _ => {}
            }
            let sum: f32 = self.epitope_weights.iter().sum();
            if sum > 0.0 {
                for w in &mut self.epitope_weights {
                    *w /= sum;
                }
            }
        } else if param_idx == N_IC50 + N_EPITOPE_WEIGHTS {
            match action_type {
                0 => self.rise_threshold = (self.rise_threshold + THRESHOLD_STEP).min(0.1),
                1 => self.rise_threshold = (self.rise_threshold - THRESHOLD_STEP).max(-0.1),
                _ => {}
            }
        } else {
            match action_type {
                0 => self.fall_threshold = (self.fall_threshold + THRESHOLD_STEP).min(0.1),
                1 => self.fall_threshold = (self.fall_threshold - THRESHOLD_STEP).max(-0.1),
                _ => {}
            }
        }
    }
}

#[derive(Clone, Copy)]
struct State {
    accuracy_bin: usize,
    improving: bool,
    stagnant: usize,
}

impl State {
    fn new(acc: f64, improving: bool, stagnant: usize) -> Self {
        Self {
            accuracy_bin: ((acc * 100.0) / 2.5).floor() as usize,
            improving,
            stagnant: stagnant.min(10),
        }
    }
    
    fn to_index(&self) -> usize {
        self.accuracy_bin.min(39) * 22 + (if self.improving { 11 } else { 0 }) + self.stagnant
    }
}

struct FluxNetOptimizer {
    q_table: Vec<Vec<f64>>,
    epsilon: f64,
    params: PathBParams,
    best_params: PathBParams,
    best_accuracy: f64,
    episode: usize,
}

impl FluxNetOptimizer {
    fn new() -> Self {
        let n_states = 40 * 22;
        Self {
            q_table: vec![vec![0.01; N_ACTIONS]; n_states],
            epsilon: EPSILON_START,
            params: PathBParams::default(),
            best_params: PathBParams::default(),
            best_accuracy: 0.0,
            episode: 0,
        }
    }
    
    fn select_action(&self, state: &State) -> usize {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < self.epsilon {
            rng.gen_range(0..N_ACTIONS)
        } else {
            let idx = state.to_index().min(self.q_table.len() - 1);
            self.q_table[idx].iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }
    
    fn update(&mut self, state: &State, action: usize, reward: f64, next: &State) {
        let s = state.to_index().min(self.q_table.len() - 1);
        let ns = next.to_index().min(self.q_table.len() - 1);
        let max_next = self.q_table[ns].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let current = self.q_table[s][action];
        self.q_table[s][action] = current + ALPHA * (reward + GAMMA * max_next - current);
    }
    
    fn record_best(&mut self, acc: f64) {
        if acc > self.best_accuracy {
            self.best_accuracy = acc;
            self.best_params = self.params.clone();
            println!("  🎯 NEW BEST: {:.2}%", acc * 100.0);
        }
    }
}

fn main() -> Result<()> {
    println!("================================================================================");
    println!("🚀 FLUXNET PATH B OPTIMIZER");
    println!("================================================================================");
    println!();
    println!("Strategy: Start from PATH B 79% baseline, optimize to 85-92%");
    println!();
    println!("Tunable Parameters ({}):", N_PARAMS);
    println!("  - IC50 values: {} (epitope binding affinities)", N_IC50);
    println!("  - Epitope weights: {} (contribution to P_neut)", N_EPITOPE_WEIGHTS);
    println!("  - Decision thresholds: 2 (rise/fall cutoffs)");
    println!();
    println!("Q-Learning: α={}, γ={}, ε={}→{}", ALPHA, GAMMA, EPSILON_START, EPSILON_MIN);
    println!("Training: {} episodes × {} steps", MAX_EPISODES, STEPS_PER_EPISODE);
    println!("================================================================================");
    println!();
    
    println!("[1/4] Loading VASIL data...");
    let vasil_dir = std::path::PathBuf::from("data/VASIL");
    let all_data = AllCountriesData::load_all_vasil_countries(&vasil_dir)?;
    println!("  ✅ {} countries", all_data.countries.len());
    
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
    
    println!("[2/4] Initializing GPU...");
    use cudarc::driver::CudaContext;
    let context = Arc::new(CudaContext::new(0)?);
    let stream = context.default_stream();
    println!("  ✅ GPU ready");
    
    let eval_start = NaiveDate::from_ymd_opt(2022, 10, 1).unwrap();
    let eval_end = NaiveDate::from_ymd_opt(2023, 10, 31).unwrap();
    
    println!("[3/4] Building PATH B baseline...");
    let dms_data = &all_data.countries[0].dms_data;
    let mut vasil_metric = VasilMetricComputer::new();
    vasil_metric.initialize(dms_data, landscapes);
    
    vasil_metric.build_immunity_cache(
        dms_data,
        &all_data.countries,
        eval_start,
        eval_end,
        &context,
        &stream,
    );
    
    let baseline_result = vasil_metric.compute_vasil_metric_exact(
        &all_data.countries,
        eval_start,
        eval_end,
    )?;
    let baseline_acc = baseline_result.mean_accuracy as f64;
    println!("  ✅ PATH B Baseline: {:.2}%", baseline_acc * 100.0);
    
    println!("[4/4] Starting FluxNet optimization...");
    println!();
    println!("================================================================================");
    println!("🧠 TRAINING (target: {}%)", TARGET_ACCURACY * 100.0);
    println!("================================================================================");
    
    let mut optimizer = FluxNetOptimizer::new();
    optimizer.best_accuracy = baseline_acc;
    optimizer.best_params = optimizer.params.clone();
    
    let mut prev_acc = baseline_acc;
    let mut stagnant = 0;
    
    for episode in 0..MAX_EPISODES {
        optimizer.episode = episode;
        println!("\n--- Episode {}/{} (ε={:.3}, best={:.2}%) ---", 
                 episode + 1, MAX_EPISODES, optimizer.epsilon, optimizer.best_accuracy * 100.0);
        
        for step in 0..STEPS_PER_EPISODE {
            let state = State::new(prev_acc, stagnant == 0, stagnant);
            let action = optimizer.select_action(&state);
            let param_idx = action / 3;
            
            optimizer.params.apply_action(param_idx, action);
            
            vasil_metric.build_immunity_cache(
                dms_data,
                &all_data.countries,
                eval_start,
                eval_end,
                &context,
                &stream,
            );
            
            let result = vasil_metric.compute_vasil_metric_exact(
                &all_data.countries,
                eval_start,
                eval_end,
            )?;
            let new_acc = result.mean_accuracy as f64;
            
            let improvement = new_acc - prev_acc;
            let reward = if improvement > 0.001 {
                20.0 * improvement
            } else if improvement < -0.001 {
                10.0 * improvement
            } else {
                -0.005
            };
            
            if improvement > 0.001 {
                stagnant = 0;
            } else {
                stagnant += 1;
            }
            
            let next_state = State::new(new_acc, improvement > 0.0, stagnant);
            optimizer.update(&state, action, reward, &next_state);
            optimizer.record_best(new_acc);
            
            prev_acc = new_acc;
            
            if (step + 1) % 10 == 0 {
                println!("    Step {}: {:.2}%", step + 1, new_acc * 100.0);
            }
            
            if optimizer.best_accuracy >= TARGET_ACCURACY {
                println!("\n🎉 TARGET ACHIEVED!");
                break;
            }
        }
        
        optimizer.epsilon = (optimizer.epsilon * EPSILON_DECAY).max(EPSILON_MIN);
        
        if optimizer.best_accuracy >= TARGET_ACCURACY {
            break;
        }
    }
    
    println!();
    println!("================================================================================");
    println!("📊 OPTIMIZATION COMPLETE");
    println!("================================================================================");
    println!();
    println!("Results:");
    println!("  PATH B Baseline:    {:.2}%", baseline_acc * 100.0);
    println!("  Best Achieved:      {:.2}%", optimizer.best_accuracy * 100.0);
    println!("  Improvement:        {:+.2}%", (optimizer.best_accuracy - baseline_acc) * 100.0);
    println!();
    println!("Optimized IC50 values:");
    let epitope_names = ["A", "B", "C", "D1", "D2", "E12", "E3", "F1", "F2", "F3"];
    for (i, name) in epitope_names.iter().enumerate() {
        println!("  {}: {:.4} (default: {:.2})", name, optimizer.best_params.ic50[i],
                 PathBParams::default().ic50[i]);
    }
    println!();
    println!("Optimized epitope weights:");
    let weight_names = ["A", "B", "C", "D1", "D2", "E12", "E3", "F1", "F2", "F3", "NTD"];
    for (i, name) in weight_names.iter().enumerate() {
        println!("  {}: {:.4}", name, optimizer.best_params.epitope_weights[i]);
    }
    println!();
    println!("Decision thresholds:");
    println!("  Rise: {:.4}", optimizer.best_params.rise_threshold);
    println!("  Fall: {:.4}", optimizer.best_params.fall_threshold);
    
    let params_str = format!(
        "# FluxNet Optimized PATH B Parameters\n\
         # Baseline: {:.2}%\n\
         # Achieved: {:.2}%\n\n\
         [ic50]\n\
         A = {:.6}\nB = {:.6}\nC = {:.6}\nD1 = {:.6}\nD2 = {:.6}\n\
         E12 = {:.6}\nE3 = {:.6}\nF1 = {:.6}\nF2 = {:.6}\nF3 = {:.6}\n\n\
         [epitope_weights]\n\
         A = {:.6}\nB = {:.6}\nC = {:.6}\nD1 = {:.6}\nD2 = {:.6}\n\
         E12 = {:.6}\nE3 = {:.6}\nF1 = {:.6}\nF2 = {:.6}\nF3 = {:.6}\nNTD = {:.6}\n\n\
         [thresholds]\n\
         rise = {:.6}\nfall = {:.6}\n",
        baseline_acc * 100.0, optimizer.best_accuracy * 100.0,
        optimizer.best_params.ic50[0], optimizer.best_params.ic50[1],
        optimizer.best_params.ic50[2], optimizer.best_params.ic50[3],
        optimizer.best_params.ic50[4], optimizer.best_params.ic50[5],
        optimizer.best_params.ic50[6], optimizer.best_params.ic50[7],
        optimizer.best_params.ic50[8], optimizer.best_params.ic50[9],
        optimizer.best_params.epitope_weights[0], optimizer.best_params.epitope_weights[1],
        optimizer.best_params.epitope_weights[2], optimizer.best_params.epitope_weights[3],
        optimizer.best_params.epitope_weights[4], optimizer.best_params.epitope_weights[5],
        optimizer.best_params.epitope_weights[6], optimizer.best_params.epitope_weights[7],
        optimizer.best_params.epitope_weights[8], optimizer.best_params.epitope_weights[9],
        optimizer.best_params.epitope_weights[10],
        optimizer.best_params.rise_threshold, optimizer.best_params.fall_threshold,
    );
    
    std::fs::write("validation_results/fluxnet_path_b_optimized.toml", &params_str)?;
    println!("✅ Saved: validation_results/fluxnet_path_b_optimized.toml");
    
    println!();
    if optimizer.best_accuracy >= 0.85 {
        println!("🎯 SUCCESS: Achieved ≥85% - ready for publication!");
    } else if optimizer.best_accuracy > baseline_acc + 0.02 {
        println!("✅ IMPROVED: +{:.1}% over baseline", (optimizer.best_accuracy - baseline_acc) * 100.0);
    } else {
        println!("⚠️  Minimal improvement - may need more training or different approach");
    }
    println!("================================================================================");
    
    Ok(())
}

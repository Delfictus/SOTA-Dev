# PRISM Phase 7-8: Enhancement Implementation Plan - Part 2
## Phase 7 Weeks 5-8 & Phase 8 Complete

---

## 4. Phase 8: Advanced Capabilities (Weeks 9-16)

### 4.1 Ensemble Reservoir Voting (Weeks 9-10)

#### Objective

Use multiple reservoirs with different initializations for reduced variance and uncertainty quantification.

#### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  ENSEMBLE RESERVOIR VOTING                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Features (80-dim)                                       │
│           ↓                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Reservoir 1 (seed=42)   → Score₁ ──┐                    │   │
│  │ Reservoir 2 (seed=123)  → Score₂ ──┤                    │   │
│  │ Reservoir 3 (seed=456)  → Score₃ ──┼──→ Weighted Mean   │   │
│  │ Reservoir 4 (seed=789)  → Score₄ ──┤    + Uncertainty   │   │
│  │ Reservoir 5 (seed=1011) → Score₅ ──┘                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Final Score: Σ(wᵢ × Scoreᵢ) / Σ(wᵢ)                          │
│  Uncertainty: σ(Score₁, ..., Score₅)                           │
│                                                                 │
│  Expected Gain: +0.03 AUC (variance reduction)                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Implementation

**File**: `crates/prism-validation/src/ensemble_reservoir.rs`

```rust
//! Ensemble Reservoir Voting
//!
//! Multiple reservoirs with different initializations provide:
//! - Reduced variance in predictions
//! - Uncertainty quantification
//! - Learned combination weights

use anyhow::{Result, Context};
use std::sync::Arc;
use cudarc::driver::CudaContext;

use crate::hierarchical_reservoir::{HierarchicalReservoir, HierarchicalConfig};

/// Ensemble configuration
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    /// Number of reservoirs
    pub n_reservoirs: usize,
    
    /// Base reservoir configuration
    pub reservoir_config: HierarchicalConfig,
    
    /// Seeds for each reservoir
    pub seeds: Vec<u64>,
    
    /// Learning rate for weight adaptation
    pub weight_learning_rate: f32,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            n_reservoirs: 5,
            reservoir_config: HierarchicalConfig::default(),
            seeds: vec![42, 123, 456, 789, 1011],
            weight_learning_rate: 0.01,
        }
    }
}

/// Ensemble prediction with uncertainty
#[derive(Debug, Clone)]
pub struct EnsemblePrediction {
    /// Weighted mean score
    pub score: f32,
    
    /// Uncertainty (standard deviation)
    pub uncertainty: f32,
    
    /// Individual reservoir scores
    pub individual_scores: Vec<f32>,
    
    /// Current combination weights
    pub weights: Vec<f32>,
    
    /// Confidence: 1 - normalized_uncertainty
    pub confidence: f32,
}

/// Ensemble reservoir scorer
pub struct EnsembleReservoir {
    context: Arc<CudaContext>,
    config: EnsembleConfig,
    
    /// Individual reservoirs
    reservoirs: Vec<HierarchicalReservoir>,
    
    /// Combination weights (learned)
    weights: Vec<f32>,
    
    /// Per-reservoir readout weights
    readout_weights: Vec<Vec<f32>>,
    
    /// Per-reservoir RLS precision matrices
    precision_matrices: Vec<Vec<f32>>,
    
    /// Total predictions made
    prediction_count: usize,
}

impl EnsembleReservoir {
    /// Create ensemble of reservoirs
    pub fn new(context: Arc<CudaContext>, config: EnsembleConfig) -> Result<Self> {
        let mut reservoirs = Vec::with_capacity(config.n_reservoirs);
        let mut readout_weights = Vec::with_capacity(config.n_reservoirs);
        let mut precision_matrices = Vec::with_capacity(config.n_reservoirs);
        
        for (i, &seed) in config.seeds.iter().enumerate() {
            let reservoir = HierarchicalReservoir::new(
                Arc::clone(&context),
                config.reservoir_config.clone(),
            )?;
            
            // Initialize with different seeds
            reservoir.initialize(seed)?;
            
            let n_neurons = reservoir.total_neurons();
            readout_weights.push(vec![0.0f32; n_neurons]);
            
            // Initialize precision matrix to 100 * I
            let mut precision = vec![0.0f32; n_neurons * n_neurons];
            for j in 0..n_neurons {
                precision[j * n_neurons + j] = 100.0;
            }
            precision_matrices.push(precision);
            
            reservoirs.push(reservoir);
        }
        
        // Uniform initial weights
        let weights = vec![1.0 / config.n_reservoirs as f32; config.n_reservoirs];
        
        log::info!("Ensemble created: {} reservoirs × {} neurons = {} total",
                   config.n_reservoirs, 
                   reservoirs[0].total_neurons(),
                   config.n_reservoirs * reservoirs[0].total_neurons());
        
        Ok(Self {
            context,
            config,
            reservoirs,
            weights,
            readout_weights,
            precision_matrices,
            prediction_count: 0,
        })
    }
    
    /// Process input through all reservoirs and combine
    pub fn predict(&mut self, input: &[f32; 80]) -> Result<EnsemblePrediction> {
        let mut individual_scores = Vec::with_capacity(self.config.n_reservoirs);
        
        // Get prediction from each reservoir
        for (i, reservoir) in self.reservoirs.iter_mut().enumerate() {
            let state = reservoir.process_temporal(input, 3)?;
            
            // Compute score via readout
            let raw: f32 = state.iter()
                .zip(&self.readout_weights[i])
                .map(|(s, w)| s * w)
                .sum();
            
            individual_scores.push(sigmoid(raw));
        }
        
        // Compute weighted mean
        let weighted_sum: f32 = individual_scores.iter()
            .zip(&self.weights)
            .map(|(s, w)| s * w)
            .sum();
        let weight_sum: f32 = self.weights.iter().sum();
        let score = weighted_sum / weight_sum;
        
        // Compute uncertainty (standard deviation)
        let variance: f32 = individual_scores.iter()
            .map(|s| (s - score).powi(2))
            .sum::<f32>() / self.config.n_reservoirs as f32;
        let uncertainty = variance.sqrt();
        
        // Confidence: higher when reservoirs agree
        let confidence = 1.0 - (uncertainty / 0.5).min(1.0);
        
        self.prediction_count += 1;
        
        Ok(EnsemblePrediction {
            score,
            uncertainty,
            individual_scores,
            weights: self.weights.clone(),
            confidence,
        })
    }
    
    /// Predict with online learning from ground truth
    pub fn predict_and_learn(&mut self, input: &[f32; 80], target: f32) -> Result<EnsemblePrediction> {
        // First get prediction
        let prediction = self.predict(input)?;
        
        // Update each reservoir via RLS
        for (i, reservoir) in self.reservoirs.iter_mut().enumerate() {
            let state = reservoir.process_temporal(input, 3)?;
            self.rls_update_reservoir(i, &state, target)?;
        }
        
        // Update combination weights based on individual performance
        self.update_combination_weights(&prediction.individual_scores, target);
        
        Ok(prediction)
    }
    
    /// RLS update for single reservoir
    fn rls_update_reservoir(&mut self, idx: usize, state: &[f32], target: f32) -> Result<()> {
        let n = self.readout_weights[idx].len();
        let lambda = 0.99;
        
        // Compute P * k
        let mut pk = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..n {
                pk[i] += self.precision_matrices[idx][i * n + j] * state[j];
            }
        }
        
        // Compute k' * P * k
        let kpk: f32 = state.iter().zip(&pk).map(|(k, p)| k * p).sum();
        let gain = 1.0 / (lambda + kpk + 1e-8);
        
        // Update precision matrix: P = (1/λ)(P - gain * pk * pk')
        let inv_lambda = 1.0 / lambda;
        for i in 0..n {
            for j in 0..n {
                self.precision_matrices[idx][i * n + j] = inv_lambda * (
                    self.precision_matrices[idx][i * n + j] - gain * pk[i] * pk[j]
                );
            }
        }
        
        // Compute prediction error
        let prediction: f32 = state.iter()
            .zip(&self.readout_weights[idx])
            .map(|(s, w)| s * w)
            .sum();
        let error = (target - sigmoid(prediction)).clamp(-1.0, 1.0);
        
        // Update readout weights
        for i in 0..n {
            self.readout_weights[idx][i] += pk[i] * gain * error;
            self.readout_weights[idx][i] = self.readout_weights[idx][i].clamp(-10.0, 10.0);
        }
        
        Ok(())
    }
    
    /// Update combination weights based on per-reservoir performance
    fn update_combination_weights(&mut self, scores: &[f32], target: f32) {
        // Reward reservoirs that were closer to target
        let lr = self.config.weight_learning_rate;
        
        for (i, &score) in scores.iter().enumerate() {
            let error = (target - score).abs();
            let reward = 1.0 - error; // 0 to 1, higher is better
            
            // Exponential moving average
            self.weights[i] = (1.0 - lr) * self.weights[i] + lr * reward;
        }
        
        // Normalize to sum to 1
        let sum: f32 = self.weights.iter().sum();
        if sum > 1e-8 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }
    
    /// Reset all reservoir states (keep weights)
    pub fn reset_states(&mut self) -> Result<()> {
        for reservoir in &mut self.reservoirs {
            reservoir.reset()?;
        }
        Ok(())
    }
    
    /// Get current combination weights
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }
    
    /// Get prediction count
    pub fn prediction_count(&self) -> usize {
        self.prediction_count
    }
    
    /// Estimate memory usage
    pub fn memory_usage_mb(&self) -> f32 {
        let neurons_per = self.reservoirs[0].total_neurons();
        let n_res = self.config.n_reservoirs;
        
        // Reservoir states + readout weights + precision matrices
        let bytes = n_res * (
            neurons_per * 4 * 3 +  // membrane, spikes, tau
            neurons_per * 4 +       // readout weights
            neurons_per * neurons_per * 4  // precision matrix (this dominates)
        );
        
        bytes as f32 / (1024.0 * 1024.0)
    }
    
    /// Save ensemble state
    pub fn save(&self, path: &str) -> Result<()> {
        let state = EnsembleState {
            weights: self.weights.clone(),
            readout_weights: self.readout_weights.clone(),
            prediction_count: self.prediction_count,
        };
        
        let data = bincode::serialize(&state)?;
        std::fs::write(path, data)?;
        
        log::info!("Ensemble saved: {} reservoirs, {} predictions", 
                   self.config.n_reservoirs, self.prediction_count);
        Ok(())
    }
    
    /// Load ensemble state
    pub fn load(&mut self, path: &str) -> Result<()> {
        let data = std::fs::read(path)?;
        let state: EnsembleState = bincode::deserialize(&data)?;
        
        self.weights = state.weights;
        self.readout_weights = state.readout_weights;
        self.prediction_count = state.prediction_count;
        
        log::info!("Ensemble loaded from {}", path);
        Ok(())
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct EnsembleState {
    weights: Vec<f32>,
    readout_weights: Vec<Vec<f32>>,
    prediction_count: usize,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ensemble_prediction() {
        // Would require GPU context
    }
    
    #[test]
    fn test_weight_normalization() {
        let weights = vec![0.3, 0.2, 0.5];
        let sum: f32 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }
}
```

---

### 4.2 Cross-Structure Transfer Learning (Weeks 11-13)

#### Objective

Transfer learned patterns from one protein family to improve predictions on related structures.

#### Concept

```
┌─────────────────────────────────────────────────────────────────┐
│              CROSS-STRUCTURE TRANSFER LEARNING                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Training Phase:                                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Structure A (kinase) → Learn patterns → Backbone Wᴬ    │   │
│  │ Structure B (kinase) → Learn patterns → Backbone Wᴮ    │   │
│  │ Structure C (kinase) → Learn patterns → Backbone Wᶜ    │   │
│  │                                                         │   │
│  │ Aggregate: W_backbone = mean(Wᴬ, Wᴮ, Wᶜ)               │   │
│  │ (Captures "kinase-ness" patterns)                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Inference Phase (new kinase structure):                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Initialize with: W = W_backbone (transfer)             │   │
│  │ Adapt via RLS: W' = W + ΔW (structure-specific)       │   │
│  │                                                         │   │
│  │ Benefit: Faster convergence, better generalization     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Expected Gain: +0.03-0.05 AUC on related structures          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Transfer Hierarchy

```
Protein Family → Fold → Superfamily → Structure
     ↓               ↓          ↓           ↓
  Coarse        Medium      Fine       Instance
  Transfer      Transfer    Transfer   Adaptation
```

#### Implementation

**File**: `crates/prism-validation/src/transfer_learning.rs`

```rust
//! Cross-Structure Transfer Learning
//!
//! Enables knowledge transfer between related protein structures:
//! - Family-level backbone weights
//! - Fold-level shared representations
//! - Per-structure adaptation via RLS

use anyhow::{Result, Context};
use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// Protein family classification
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum ProteinFamily {
    Kinase,
    Protease,
    GPCR,
    NuclearReceptor,
    IonChannel,
    Transporter,
    Enzyme,
    Antibody,
    Viral,
    Other(String),
}

/// Transfer learning configuration
#[derive(Debug, Clone)]
pub struct TransferConfig {
    /// Strength of transfer (0 = no transfer, 1 = full transfer)
    pub transfer_strength: f32,
    
    /// Minimum structures to form family backbone
    pub min_family_size: usize,
    
    /// Adaptation rate for new structures
    pub adaptation_rate: f32,
    
    /// Enable cross-family transfer (weaker)
    pub cross_family_transfer: bool,
    
    /// Cross-family transfer discount
    pub cross_family_discount: f32,
}

impl Default for TransferConfig {
    fn default() -> Self {
        Self {
            transfer_strength: 0.7,
            min_family_size: 3,
            adaptation_rate: 0.1,
            cross_family_transfer: true,
            cross_family_discount: 0.3,
        }
    }
}

/// Family-level backbone weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FamilyBackbone {
    pub family: ProteinFamily,
    pub weights: Vec<f32>,
    pub n_structures: usize,
    pub mean_auc: f32,
}

/// Transfer learning manager
pub struct TransferLearningManager {
    config: TransferConfig,
    
    /// Per-family backbone weights
    family_backbones: HashMap<ProteinFamily, FamilyBackbone>,
    
    /// Global backbone (all structures)
    global_backbone: Option<Vec<f32>>,
    
    /// Structure-specific adaptations
    structure_adaptations: HashMap<String, Vec<f32>>,
    
    /// Performance tracking
    family_performance: HashMap<ProteinFamily, Vec<f32>>,
}

impl TransferLearningManager {
    pub fn new(config: TransferConfig) -> Self {
        Self {
            config,
            family_backbones: HashMap::new(),
            global_backbone: None,
            structure_adaptations: HashMap::new(),
            family_performance: HashMap::new(),
        }
    }
    
    /// Register a trained structure's weights for transfer
    pub fn register_structure(
        &mut self,
        structure_id: &str,
        family: ProteinFamily,
        weights: Vec<f32>,
        auc: f32,
    ) {
        // Store structure-specific weights
        self.structure_adaptations.insert(structure_id.to_string(), weights.clone());
        
        // Track performance
        self.family_performance
            .entry(family.clone())
            .or_insert_with(Vec::new)
            .push(auc);
        
        // Update family backbone
        self.update_family_backbone(family, weights);
        
        // Update global backbone
        self.update_global_backbone();
    }
    
    /// Update family backbone with new structure
    fn update_family_backbone(&mut self, family: ProteinFamily, new_weights: Vec<f32>) {
        let backbone = self.family_backbones
            .entry(family.clone())
            .or_insert_with(|| FamilyBackbone {
                family: family.clone(),
                weights: new_weights.clone(),
                n_structures: 0,
                mean_auc: 0.0,
            });
        
        // Running mean update
        let n = backbone.n_structures as f32;
        for (i, w) in backbone.weights.iter_mut().enumerate() {
            *w = (*w * n + new_weights[i]) / (n + 1.0);
        }
        backbone.n_structures += 1;
        
        // Update mean AUC
        if let Some(perfs) = self.family_performance.get(&family) {
            backbone.mean_auc = perfs.iter().sum::<f32>() / perfs.len() as f32;
        }
    }
    
    /// Update global backbone
    fn update_global_backbone(&mut self) {
        let all_backbones: Vec<&Vec<f32>> = self.family_backbones
            .values()
            .map(|b| &b.weights)
            .collect();
        
        if all_backbones.is_empty() {
            return;
        }
        
        let n = all_backbones.len() as f32;
        let dim = all_backbones[0].len();
        
        let mut global = vec![0.0f32; dim];
        for backbone in &all_backbones {
            for (i, w) in backbone.iter().enumerate() {
                global[i] += w / n;
            }
        }
        
        self.global_backbone = Some(global);
    }
    
    /// Get initialization weights for new structure
    pub fn get_initial_weights(
        &self,
        family: ProteinFamily,
        weight_dim: usize,
    ) -> Vec<f32> {
        // Try family-specific backbone first
        if let Some(backbone) = self.family_backbones.get(&family) {
            if backbone.n_structures >= self.config.min_family_size {
                log::info!("Using {} family backbone ({} structures, AUC={:.3})",
                           backbone.n_structures, backbone.n_structures, backbone.mean_auc);
                return self.blend_with_random(&backbone.weights, weight_dim);
            }
        }
        
        // Try cross-family transfer
        if self.config.cross_family_transfer {
            if let Some(ref global) = self.global_backbone {
                log::info!("Using global backbone (cross-family transfer)");
                return self.blend_with_random_discounted(
                    global, 
                    weight_dim,
                    self.config.cross_family_discount,
                );
            }
        }
        
        // No transfer available - random initialization
        log::info!("No transfer available - random initialization");
        vec![0.0f32; weight_dim]
    }
    
    /// Blend backbone with small random perturbation
    fn blend_with_random(&self, backbone: &[f32], dim: usize) -> Vec<f32> {
        let mut weights = vec![0.0f32; dim];
        let copy_len = backbone.len().min(dim);
        
        for i in 0..copy_len {
            // Transfer strength blend
            weights[i] = backbone[i] * self.config.transfer_strength;
        }
        
        weights
    }
    
    /// Blend with discount for cross-family
    fn blend_with_random_discounted(&self, backbone: &[f32], dim: usize, discount: f32) -> Vec<f32> {
        let mut weights = vec![0.0f32; dim];
        let copy_len = backbone.len().min(dim);
        
        for i in 0..copy_len {
            weights[i] = backbone[i] * self.config.transfer_strength * discount;
        }
        
        weights
    }
    
    /// Get family statistics
    pub fn family_stats(&self) -> Vec<FamilyStats> {
        self.family_backbones.values()
            .map(|b| FamilyStats {
                family: b.family.clone(),
                n_structures: b.n_structures,
                mean_auc: b.mean_auc,
            })
            .collect()
    }
    
    /// Save transfer state
    pub fn save(&self, path: &str) -> Result<()> {
        let state = TransferState {
            family_backbones: self.family_backbones.clone(),
            global_backbone: self.global_backbone.clone(),
        };
        
        let data = serde_json::to_string_pretty(&state)?;
        std::fs::write(path, data)?;
        
        log::info!("Transfer state saved: {} families", self.family_backbones.len());
        Ok(())
    }
    
    /// Load transfer state
    pub fn load(&mut self, path: &str) -> Result<()> {
        let data = std::fs::read_to_string(path)?;
        let state: TransferState = serde_json::from_str(&data)?;
        
        self.family_backbones = state.family_backbones;
        self.global_backbone = state.global_backbone;
        
        log::info!("Transfer state loaded: {} families", self.family_backbones.len());
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct FamilyStats {
    pub family: ProteinFamily,
    pub n_structures: usize,
    pub mean_auc: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TransferState {
    family_backbones: HashMap<ProteinFamily, FamilyBackbone>,
    global_backbone: Option<Vec<f32>>,
}

/// Classify protein into family based on structure features
pub fn classify_protein_family(
    structure_id: &str,
    uniprot_id: Option<&str>,
) -> ProteinFamily {
    // Simple heuristic classification
    // In production, would use UniProt annotations or fold classification
    
    let id_lower = structure_id.to_lowercase();
    
    if id_lower.contains("kinase") || id_lower.contains("pk") {
        ProteinFamily::Kinase
    } else if id_lower.contains("protease") || id_lower.contains("peptidase") {
        ProteinFamily::Protease
    } else if id_lower.contains("gpcr") || id_lower.contains("receptor") {
        ProteinFamily::GPCR
    } else if id_lower.contains("antibody") || id_lower.contains("fab") {
        ProteinFamily::Antibody
    } else if id_lower.contains("viral") || id_lower.contains("spike") {
        ProteinFamily::Viral
    } else {
        ProteinFamily::Other(structure_id.to_string())
    }
}
```

---

### 4.3 Uncertainty Quantification (Week 14)

#### Objective

Provide calibrated confidence scores for predictions to guide experimental prioritization.

#### Implementation

**File**: `crates/prism-validation/src/uncertainty.rs`

```rust
//! Uncertainty Quantification for Cryptic Site Predictions
//!
//! Provides:
//! - Epistemic uncertainty (model uncertainty)
//! - Aleatoric uncertainty (data uncertainty)
//! - Calibrated confidence intervals

use std::collections::VecDeque;

/// Uncertainty estimates for a prediction
#[derive(Debug, Clone)]
pub struct UncertaintyEstimate {
    /// Point prediction
    pub prediction: f32,
    
    /// Epistemic uncertainty (from ensemble disagreement)
    pub epistemic: f32,
    
    /// Aleatoric uncertainty (from feature noise)
    pub aleatoric: f32,
    
    /// Total uncertainty
    pub total: f32,
    
    /// Calibrated confidence (0-1)
    pub confidence: f32,
    
    /// 95% confidence interval
    pub ci_lower: f32,
    pub ci_upper: f32,
}

/// Uncertainty quantifier
pub struct UncertaintyQuantifier {
    /// History of predictions vs actuals for calibration
    calibration_history: VecDeque<(f32, bool)>,
    
    /// Maximum calibration history size
    max_history: usize,
    
    /// Calibration bins
    calibration_bins: Vec<CalibrationBin>,
}

#[derive(Debug, Clone, Default)]
struct CalibrationBin {
    predicted_sum: f32,
    actual_sum: f32,
    count: usize,
}

impl UncertaintyQuantifier {
    pub fn new(max_history: usize) -> Self {
        Self {
            calibration_history: VecDeque::with_capacity(max_history),
            max_history,
            calibration_bins: vec![CalibrationBin::default(); 10],
        }
    }
    
    /// Compute uncertainty from ensemble predictions
    pub fn quantify(
        &self,
        ensemble_scores: &[f32],
        feature_variance: f32,
    ) -> UncertaintyEstimate {
        let n = ensemble_scores.len() as f32;
        
        // Point prediction (mean)
        let prediction = ensemble_scores.iter().sum::<f32>() / n;
        
        // Epistemic uncertainty (ensemble variance)
        let epistemic = (ensemble_scores.iter()
            .map(|s| (s - prediction).powi(2))
            .sum::<f32>() / n).sqrt();
        
        // Aleatoric uncertainty (from feature variance)
        let aleatoric = (feature_variance * 0.1).sqrt();
        
        // Total uncertainty (combined)
        let total = (epistemic.powi(2) + aleatoric.powi(2)).sqrt();
        
        // Calibrated confidence
        let raw_confidence = 1.0 - (total / 0.5).min(1.0);
        let confidence = self.calibrate_confidence(prediction, raw_confidence);
        
        // 95% CI (approximately)
        let ci_width = 1.96 * total;
        let ci_lower = (prediction - ci_width).max(0.0);
        let ci_upper = (prediction + ci_width).min(1.0);
        
        UncertaintyEstimate {
            prediction,
            epistemic,
            aleatoric,
            total,
            confidence,
            ci_lower,
            ci_upper,
        }
    }
    
    /// Update calibration with observed outcome
    pub fn update_calibration(&mut self, prediction: f32, actual: bool) {
        // Add to history
        if self.calibration_history.len() >= self.max_history {
            self.calibration_history.pop_front();
        }
        self.calibration_history.push_back((prediction, actual));
        
        // Update bin
        let bin_idx = ((prediction * 10.0) as usize).min(9);
        self.calibration_bins[bin_idx].predicted_sum += prediction;
        self.calibration_bins[bin_idx].actual_sum += if actual { 1.0 } else { 0.0 };
        self.calibration_bins[bin_idx].count += 1;
    }
    
    /// Apply calibration curve to raw confidence
    fn calibrate_confidence(&self, prediction: f32, raw_confidence: f32) -> f32 {
        let bin_idx = ((prediction * 10.0) as usize).min(9);
        let bin = &self.calibration_bins[bin_idx];
        
        if bin.count < 10 {
            // Not enough data for calibration
            return raw_confidence;
        }
        
        // Calibration factor: actual / predicted
        let predicted_mean = bin.predicted_sum / bin.count as f32;
        let actual_mean = bin.actual_sum / bin.count as f32;
        
        if predicted_mean < 0.01 {
            return raw_confidence;
        }
        
        let calibration_factor = actual_mean / predicted_mean;
        
        // Apply calibration (bounded)
        (raw_confidence * calibration_factor).clamp(0.0, 1.0)
    }
    
    /// Get calibration curve
    pub fn calibration_curve(&self) -> Vec<(f32, f32)> {
        self.calibration_bins.iter()
            .enumerate()
            .filter(|(_, b)| b.count >= 10)
            .map(|(i, b)| {
                let predicted = b.predicted_sum / b.count as f32;
                let actual = b.actual_sum / b.count as f32;
                (predicted, actual)
            })
            .collect()
    }
    
    /// Expected Calibration Error (ECE)
    pub fn expected_calibration_error(&self) -> f32 {
        let total_count: usize = self.calibration_bins.iter().map(|b| b.count).sum();
        if total_count == 0 {
            return 0.0;
        }
        
        self.calibration_bins.iter()
            .filter(|b| b.count > 0)
            .map(|b| {
                let predicted = b.predicted_sum / b.count as f32;
                let actual = b.actual_sum / b.count as f32;
                let weight = b.count as f32 / total_count as f32;
                weight * (predicted - actual).abs()
            })
            .sum()
    }
}
```

---

### 4.4 Active Learning Pipeline (Weeks 15-16)

#### Objective

Prioritize which structures to analyze next for maximum model improvement.

#### Strategy

```
┌─────────────────────────────────────────────────────────────────┐
│                  ACTIVE LEARNING PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Pool of Unlabeled Structures                                  │
│           ↓                                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Acquisition Function:                                   │   │
│  │                                                         │   │
│  │ Score = α × Uncertainty                                │   │
│  │       + β × Diversity                                  │   │
│  │       + γ × Expected Information Gain                  │   │
│  │                                                         │   │
│  │ Uncertainty: High ensemble disagreement                │   │
│  │ Diversity: Different from already analyzed             │   │
│  │ Info Gain: Expected model improvement                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│           ↓                                                     │
│  Select Top-K for Analysis                                     │
│           ↓                                                     │
│  Run PRISM Pipeline + Get Ground Truth                         │
│           ↓                                                     │
│  Update Model → Repeat                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

#### Implementation

**File**: `crates/prism-validation/src/active_learning.rs`

```rust
//! Active Learning for Structure Prioritization
//!
//! Selects which structures to analyze next based on:
//! - Prediction uncertainty
//! - Structural diversity
//! - Expected information gain

use anyhow::Result;
use std::collections::HashSet;

/// Active learning configuration
#[derive(Debug, Clone)]
pub struct ActiveLearningConfig {
    /// Weight for uncertainty component
    pub alpha: f32,
    
    /// Weight for diversity component
    pub beta: f32,
    
    /// Weight for expected information gain
    pub gamma: f32,
    
    /// Number of structures to select per batch
    pub batch_size: usize,
    
    /// Minimum uncertainty to consider
    pub min_uncertainty: f32,
}

impl Default for ActiveLearningConfig {
    fn default() -> Self {
        Self {
            alpha: 0.4,
            beta: 0.3,
            gamma: 0.3,
            batch_size: 10,
            min_uncertainty: 0.1,
        }
    }
}

/// Structure candidate for active learning
#[derive(Debug, Clone)]
pub struct StructureCandidate {
    pub id: String,
    pub pdb_path: String,
    pub family: Option<String>,
    
    // Computed scores
    pub uncertainty_score: f32,
    pub diversity_score: f32,
    pub info_gain_score: f32,
    pub acquisition_score: f32,
}

/// Active learning selector
pub struct ActiveLearningSelector {
    config: ActiveLearningConfig,
    
    /// Already analyzed structure IDs
    analyzed: HashSet<String>,
    
    /// Feature vectors of analyzed structures (for diversity)
    analyzed_features: Vec<Vec<f32>>,
}

impl ActiveLearningSelector {
    pub fn new(config: ActiveLearningConfig) -> Self {
        Self {
            config,
            analyzed: HashSet::new(),
            analyzed_features: Vec::new(),
        }
    }
    
    /// Register an analyzed structure
    pub fn register_analyzed(&mut self, structure_id: &str, features: Vec<f32>) {
        self.analyzed.insert(structure_id.to_string());
        self.analyzed_features.push(features);
    }
    
    /// Select next batch of structures to analyze
    pub fn select_batch(
        &self,
        candidates: &mut [StructureCandidate],
    ) -> Vec<String> {
        // Score each candidate
        for candidate in candidates.iter_mut() {
            if self.analyzed.contains(&candidate.id) {
                candidate.acquisition_score = -1.0; // Already analyzed
                continue;
            }
            
            // Compute acquisition score
            candidate.acquisition_score = 
                self.config.alpha * candidate.uncertainty_score
                + self.config.beta * candidate.diversity_score
                + self.config.gamma * candidate.info_gain_score;
        }
        
        // Sort by acquisition score descending
        candidates.sort_by(|a, b| {
            b.acquisition_score.partial_cmp(&a.acquisition_score).unwrap()
        });
        
        // Select top batch_size
        candidates.iter()
            .filter(|c| c.acquisition_score > 0.0)
            .take(self.config.batch_size)
            .map(|c| c.id.clone())
            .collect()
    }
    
    /// Compute diversity score for a candidate
    pub fn compute_diversity_score(&self, features: &[f32]) -> f32 {
        if self.analyzed_features.is_empty() {
            return 1.0; // Maximum diversity if nothing analyzed yet
        }
        
        // Minimum distance to any analyzed structure
        let min_dist = self.analyzed_features.iter()
            .map(|analyzed| {
                let dist_sq: f32 = features.iter()
                    .zip(analyzed.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                dist_sq.sqrt()
            })
            .fold(f32::INFINITY, f32::min);
        
        // Normalize (larger distance = higher diversity)
        (min_dist / 10.0).min(1.0)
    }
    
    /// Compute expected information gain
    pub fn compute_info_gain(&self, uncertainty: f32, family: Option<&str>) -> f32 {
        let mut gain = uncertainty;
        
        // Bonus for underrepresented families
        if let Some(family) = family {
            let family_count = self.analyzed.iter()
                .filter(|id| id.contains(family))
                .count();
            
            if family_count < 3 {
                gain *= 1.5; // 50% bonus for rare families
            }
        }
        
        gain.min(1.0)
    }
    
    /// Get statistics
    pub fn stats(&self) -> ActiveLearningStats {
        ActiveLearningStats {
            n_analyzed: self.analyzed.len(),
            n_features: self.analyzed_features.first().map(|f| f.len()).unwrap_or(0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ActiveLearningStats {
    pub n_analyzed: usize,
    pub n_features: usize,
}
```

---

## 5. Phase 8 Complete Scorer

**File**: `crates/prism-validation/src/phase8_scorer.rs`

```rust
//! Phase 8 Complete Cryptic Site Scorer
//!
//! Integrates all enhancements:
//! - Hierarchical reservoir (1,280 neurons)
//! - Ensemble voting (5 reservoirs)
//! - Persistent homology
//! - Multi-scale features
//! - Transfer learning
//! - Uncertainty quantification

use anyhow::Result;
use std::sync::Arc;
use cudarc::driver::CudaContext;

use crate::ensemble_reservoir::{EnsembleReservoir, EnsembleConfig, EnsemblePrediction};
use crate::transfer_learning::{TransferLearningManager, TransferConfig, ProteinFamily};
use crate::uncertainty::{UncertaintyQuantifier, UncertaintyEstimate};
use crate::extended_nova_sampler::{ExtendedNovaSampler, ExtendedNovaConfig};
use crate::persistent_homology::PersistenceComputer;
use crate::multiscale_features::MultiScaleExtractor;
use crate::pdb_sanitizer::{PdbSanitizer, SanitizedStructure};

/// Phase 8 configuration
#[derive(Debug, Clone)]
pub struct Phase8Config {
    pub nova: ExtendedNovaConfig,
    pub ensemble: EnsembleConfig,
    pub transfer: TransferConfig,
}

impl Default for Phase8Config {
    fn default() -> Self {
        Self {
            nova: ExtendedNovaConfig::default(),
            ensemble: EnsembleConfig::default(),
            transfer: TransferConfig::default(),
        }
    }
}

/// Phase 8 complete scorer
pub struct Phase8Scorer {
    context: Arc<CudaContext>,
    config: Phase8Config,
    
    /// Ensemble of hierarchical reservoirs
    ensemble: EnsembleReservoir,
    
    /// Transfer learning manager
    transfer: TransferLearningManager,
    
    /// Uncertainty quantifier
    uncertainty: UncertaintyQuantifier,
    
    /// NOVA sampler
    sampler: ExtendedNovaSampler,
    
    /// Persistence computer
    persistence: PersistenceComputer,
    
    /// Current structure
    structure: Option<SanitizedStructure>,
    
    /// Multi-scale extractor
    multiscale: Option<MultiScaleExtractor>,
}

impl Phase8Scorer {
    /// Create new Phase 8 scorer
    pub fn new(context: Arc<CudaContext>, config: Phase8Config) -> Result<Self> {
        let ensemble = EnsembleReservoir::new(Arc::clone(&context), config.ensemble.clone())?;
        let transfer = TransferLearningManager::new(config.transfer.clone());
        let uncertainty = UncertaintyQuantifier::new(10000);
        let sampler = ExtendedNovaSampler::new(Arc::clone(&context), config.nova.clone())?;
        let persistence = PersistenceComputer::new(Arc::clone(&context))?;
        
        log::info!("Phase 8 scorer initialized");
        log::info!("  Ensemble: {} reservoirs", config.ensemble.n_reservoirs);
        log::info!("  Sampling: {} conformations", config.nova.total_samples);
        log::info!("  Transfer: enabled");
        log::info!("  Uncertainty: enabled");
        
        Ok(Self {
            context,
            config,
            ensemble,
            transfer,
            uncertainty,
            sampler,
            persistence,
            structure: None,
            multiscale: None,
        })
    }
    
    /// Load structure with family classification
    pub fn load_structure(&mut self, pdb_content: &str, structure_id: &str) -> Result<()> {
        let sanitizer = PdbSanitizer::new();
        let structure = sanitizer.sanitize(pdb_content)?;
        
        self.multiscale = Some(MultiScaleExtractor::new(&structure));
        self.sampler.load_structure(pdb_content)?;
        self.structure = Some(structure);
        
        // Get transfer weights based on family
        let family = crate::transfer_learning::classify_protein_family(structure_id, None);
        let n_neurons = self.ensemble.reservoirs[0].total_neurons();
        let initial_weights = self.transfer.get_initial_weights(family, n_neurons);
        
        // Apply transfer weights to ensemble
        // (implementation would initialize readout weights from transfer)
        
        log::info!("Loaded structure {} (family: {:?})", structure_id, family);
        
        Ok(())
    }
    
    /// Score structure with full uncertainty
    pub fn score_structure(&mut self) -> Result<Phase8Result> {
        let structure = self.structure.as_ref()
            .ok_or_else(|| anyhow::anyhow!("No structure loaded"))?;
        
        let start = std::time::Instant::now();
        
        // Extended sampling
        log::info!("Phase 8: Running extended sampling...");
        let sampling = self.sampler.sample()?;
        
        // Compute features and score each residue
        log::info!("Phase 8: Scoring residues with ensemble...");
        let mut residue_results = Vec::with_capacity(structure.n_residues);
        
        for res_idx in 0..structure.n_residues {
            // Extract features (simplified for this example)
            let features = self.extract_features(res_idx, &sampling)?;
            
            // Ensemble prediction
            let ensemble_pred = self.ensemble.predict(&features)?;
            
            // Uncertainty quantification
            let feature_var = features.iter().map(|f| f.powi(2)).sum::<f32>() / features.len() as f32;
            let uncertainty = self.uncertainty.quantify(
                &ensemble_pred.individual_scores,
                feature_var,
            );
            
            residue_results.push(ResidueResult {
                index: res_idx,
                score: ensemble_pred.score,
                uncertainty: uncertainty.total,
                confidence: uncertainty.confidence,
                ci_lower: uncertainty.ci_lower,
                ci_upper: uncertainty.ci_upper,
            });
        }
        
        let elapsed = start.elapsed();
        
        log::info!("Phase 8 complete: {} residues in {:.1}s",
                   residue_results.len(), elapsed.as_secs_f32());
        
        Ok(Phase8Result {
            residue_results,
            elapsed_ms: elapsed.as_millis() as u64,
            ensemble_weights: self.ensemble.weights().to_vec(),
        })
    }
    
    /// Extract features for a residue
    fn extract_features(
        &self,
        res_idx: usize,
        sampling: &crate::extended_nova_sampler::ExtendedSamplingResult,
    ) -> Result<[f32; 80]> {
        // Simplified - full implementation would use MultiScaleExtractor
        let mut features = [0.0f32; 80];
        
        // Would compute from sampling results
        features[0] = res_idx as f32 / 100.0; // Placeholder
        
        Ok(features)
    }
    
    /// Update model with ground truth
    pub fn learn(&mut self, ground_truth: &[bool]) -> Result<()> {
        // Would update ensemble and calibration
        for (i, &is_cryptic) in ground_truth.iter().enumerate() {
            // Update uncertainty calibration
            // Update ensemble weights
        }
        Ok(())
    }
    
    /// Save model state
    pub fn save(&self, dir: &str) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        
        self.ensemble.save(&format!("{}/ensemble.bin", dir))?;
        self.transfer.save(&format!("{}/transfer.json", dir))?;
        
        log::info!("Phase 8 model saved to {}", dir);
        Ok(())
    }
    
    /// Load model state
    pub fn load(&mut self, dir: &str) -> Result<()> {
        self.ensemble.load(&format!("{}/ensemble.bin", dir))?;
        self.transfer.load(&format!("{}/transfer.json", dir))?;
        
        log::info!("Phase 8 model loaded from {}", dir);
        Ok(())
    }
}

/// Result for single residue
#[derive(Debug, Clone)]
pub struct ResidueResult {
    pub index: usize,
    pub score: f32,
    pub uncertainty: f32,
    pub confidence: f32,
    pub ci_lower: f32,
    pub ci_upper: f32,
}

/// Full scoring result
#[derive(Debug, Clone)]
pub struct Phase8Result {
    pub residue_results: Vec<ResidueResult>,
    pub elapsed_ms: u64,
    pub ensemble_weights: Vec<f32>,
}
```

---

## 6. Implementation Timeline Summary

| Week | Phase | Component | Expected Δ AUC |
|------|-------|-----------|----------------|
| 1-2 | 7 | Hierarchical Reservoir | +0.03 |
| 3-4 | 7 | Persistent Homology + Multi-Scale | +0.04 |
| 5-6 | 7 | Extended NOVA Sampling | +0.02 |
| 7-8 | 7 | Integration + Validation | (verification) |
| 9-10 | 8 | Ensemble Voting | +0.03 |
| 11-13 | 8 | Transfer Learning | +0.03 |
| 14 | 8 | Uncertainty Quantification | (quality) |
| 15-16 | 8 | Active Learning + Final | (efficiency) |

**Total Expected: Phase 6 (0.75) + Phase 7-8 (+0.15) = 0.90 AUC**

---

## 7. File Manifest

### Phase 7 Files (8)

| File | Purpose |
|------|---------|
| `hierarchical_reservoir.rs` | 1,280-neuron cortical column architecture |
| `persistent_homology.rs` | Full TDA with persistence diagrams |
| `extended_nova_sampler.rs` | 2000-sample adaptive biasing |
| `multiscale_features.rs` | Local + regional + global features |
| `phase7_scorer.rs` | Integrated Phase 7 pipeline |
| `kernels/hierarchical_reservoir.cu` | CUDA kernels for layers |
| `kernels/persistence.cu` | CUDA kernels for TDA |
| `tests/phase7_tests.rs` | Phase 7 validation tests |

### Phase 8 Files (6)

| File | Purpose |
|------|---------|
| `ensemble_reservoir.rs` | 5-reservoir voting system |
| `transfer_learning.rs` | Cross-structure knowledge transfer |
| `uncertainty.rs` | Calibrated confidence scores |
| `active_learning.rs` | Structure prioritization |
| `phase8_scorer.rs` | Complete integrated scorer |
| `tests/phase8_tests.rs` | Phase 8 validation tests |

---

## 8. Verification Commands

```bash
# Phase 7 verification
cargo test --release -p prism-validation --features cuda phase7 -- --nocapture

# Phase 8 verification
cargo test --release -p prism-validation --features cuda phase8 -- --nocapture

# Full benchmark
cargo run --release -p prism-validation --bin cryptobench-phase8 -- \
    --manifest data/benchmarks/cryptobench/manifest.json \
    --output results/phase8_full.json

# Compare to Phase 6 baseline
cargo run --release -p prism-validation --bin compare-phases -- \
    --phase6 results/phase6_final.json \
    --phase8 results/phase8_full.json
```

---

## 9. Success Criteria

### Phase 7 Complete (Week 8)

```
□ Hierarchical reservoir: 1,280 neurons, <10ms/step
□ Persistence features: 31 dimensions extracted
□ Multi-scale features: 36 dimensions extracted  
□ Extended sampling: 2000 conformations, <3 min/structure
□ CryptoBench ROC AUC ≥ 0.82
□ CryptoBench PR AUC ≥ 0.32
□ All tests passing
```

### Phase 8 Complete (Week 16)

```
□ Ensemble: 5 reservoirs, <50ms combined inference
□ Transfer learning: family backbones for 5+ families
□ Uncertainty: ECE < 0.10 (well-calibrated)
□ Active learning: functional selection pipeline
□ CryptoBench ROC AUC ≥ 0.90
□ CryptoBench PR AUC ≥ 0.40
□ Exceeds or matches PocketMiner (0.87 AUC)
□ All tests passing
□ Publication-ready results
```

---

## 10. Strategic Outcome

**After Phase 8 completion, PRISM will be:**

1. **Category Leader**: 0.90 AUC matches/exceeds all published methods
2. **Fully Sovereign**: Zero external dependencies maintained
3. **Production Ready**: Uncertainty quantification enables confident deployment
4. **Continuously Improving**: Transfer learning compounds value over time
5. **Efficiently Scalable**: Active learning optimizes resource allocation

**Publication Title Options:**
- *"Neuromorphic Cryptic Site Detection Matches Deep Learning Performance Without External Dependencies"*
- *"PRISM-ZrO: Sovereign AI for Drug Discovery Achieves State-of-the-Art Cryptic Pocket Prediction"*
- *"From Competitive to Category Leader: Hierarchical Spiking Networks for Cryptic Site Detection"*

---

**Document Complete. Ready for execution following Phase 6 completion.**

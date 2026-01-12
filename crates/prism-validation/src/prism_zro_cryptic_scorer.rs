//! PRISM-ZrO Cryptic Site Scorer (Phase 2.2) - REAL GPU IMPLEMENTATION
//!
//! Uses the ACTUAL PRISM-ZrO infrastructure from prism-gpu and prism-learning:
//! - **DendriticSNNReservoir**: 512 GPU-accelerated LIF neurons (E/I balanced)
//! - **RLS Readout**: Sherman-Morrison updates with reward modulation
//! - **Feature Adapter Protocol**: Automatic 40→80 dim velocity expansion
//!
//! ## Architecture
//!
//! ```text
//! Per-residue features (8-dim) - ENHANCED with structural context
//!        ↓ [Pad to 40-dim for Feature Adapter Protocol]
//! DendriticSNNReservoir (512 GPU LIF neurons, E/I balanced)
//!        ↓ [512-dim filtered spike rates]
//! RLS Readout (single head, Sherman-Morrison)
//!        ↓ [Reward-modulated plasticity from ground truth]
//! cryptic_score (continuous 0-1)
//! ```
//!
//! ## Key Features
//!
//! - **GPU Acceleration**: Real CUDA kernels via cudarc
//! - **Online Learning**: RLS updates as we process each structure
//! - **Reward Modulation**: Learning rate scales with ground truth signal
//! - **E/I Balance**: 80% excitatory, 20% inhibitory neurons
//! - **Flashbulb Reservoir**: Persistent state across residues
//!
//! ## Input Features (8-dim, padded to 40) - ENHANCED
//!
//! 1. Burial change (normalized)
//! 2. RMSF (normalized)
//! 3. Variance (normalized)
//! 4. Neighbor flexibility (mean RMSF of neighbors)
//! 5. Burial potential (how much could exposure increase)
//! 6. Secondary structure flexibility (Helix=0.7, Sheet=0.8, Loop=1.2) - NEW!
//! 7. Sidechain flexibility (GLY=1.4, PRO=0.6, etc.) - NEW!
//! 8. B-factor (normalized, if available) - NEW!

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

// GPU SNN reservoir - the REAL PRISM-ZrO infrastructure
use prism_gpu::{DendriticSNNReservoir, SNN_INPUT_DIM, DEFAULT_RESERVOIR_SIZE};
use cudarc::driver::CudaContext;

/// Number of cryptic-specific input features (ENHANCED to 16 for comprehensive analysis)
/// Features: dynamics (5) + structural (3) + chemical (3) + distance (3) + tertiary (2)
pub const NUM_CRYPTIC_FEATURES: usize = 16;

/// GPU reservoir size (must match DendriticSNNReservoir)
pub const GPU_RESERVOIR_SIZE: usize = 512;

/// RLS forgetting factor (higher = slower adaptation)
pub const DEFAULT_LAMBDA: f32 = 0.99;

/// Initial precision matrix diagonal
pub const INITIAL_P_DIAG: f32 = 100.0;

/// Minimum reward modulation (prevents zero learning)
pub const MIN_REWARD_MODULATION: f32 = 0.1;

/// Maximum reward modulation (prevents explosive learning)
pub const MAX_REWARD_MODULATION: f32 = 2.0;

/// Reward scale factor for modulation
pub const REWARD_MODULATION_SCALE: f32 = 0.5;

/// Configuration for PRISM-ZrO cryptic scorer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZroCrypticConfig {
    /// RLS forgetting factor (λ) - higher = slower adaptation
    pub lambda: f32,

    /// Enable online learning (if false, use fixed weights)
    pub online_learning: bool,

    /// Random seed for reservoir initialization
    pub seed: Option<u64>,
}

impl Default for ZroCrypticConfig {
    fn default() -> Self {
        Self {
            lambda: DEFAULT_LAMBDA,
            online_learning: true,
            seed: Some(42),
        }
    }
}

/// Per-residue features for cryptic scoring (COMPREHENSIVE 16-feature analysis)
///
/// Features organized by category:
/// - **Dynamics (5)**: burial_change, rmsf, variance, neighbor_flex, burial_potential
/// - **Structural (3)**: ss_flexibility, sidechain_flexibility, b_factor
/// - **Chemical (3)**: net_charge, hydrophobicity, h_bond_potential
/// - **Distance (3)**: contact_density, sasa_change, nearest_charged_dist
/// - **Tertiary (2)**: interface_score, allosteric_proximity
#[derive(Debug, Clone, Default)]
pub struct ResidueFeatures {
    /// Residue ID
    pub residue_id: i32,

    // ==================== DYNAMICS FEATURES (5) ====================

    /// Burial change from reference (positive = more exposed)
    pub burial_change: f64,
    /// RMSF from ensemble
    pub rmsf: f64,
    /// Position variance across ensemble
    pub variance: f64,
    /// Mean RMSF of neighboring residues
    pub neighbor_flexibility: f64,
    /// Potential for further exposure
    pub burial_potential: f64,

    // ==================== STRUCTURAL FEATURES (3) ====================

    /// Secondary structure flexibility factor (Helix=0.7, Sheet=0.8, Loop=1.2)
    /// Loops are more flexible and more likely to contain cryptic sites
    pub ss_flexibility: f64,

    /// Sidechain flexibility factor based on residue type
    /// GLY=1.40 (most flexible), PRO=0.60 (most rigid)
    pub sidechain_flexibility: f64,

    /// B-factor from crystal structure (if available)
    /// High B-factor indicates crystallographic disorder/flexibility
    pub b_factor: Option<f64>,

    // ==================== CHEMICAL FEATURES (3) ====================

    /// Net charge of residue at physiological pH
    /// K, R = +1; H = +0.1; D, E = -1; others = 0
    /// Charged residues often line binding sites
    pub net_charge: f64,

    /// Hydrophobicity (Kyte-Doolittle scale, normalized -1 to +1)
    /// Cryptic sites often expose hydrophobic patches
    pub hydrophobicity: f64,

    /// Hydrogen bond potential (# of H-bond donors + acceptors)
    /// High H-bond capacity suggests binding site lining
    pub h_bond_potential: f64,

    // ==================== DISTANCE FEATURES (3) ====================

    /// Contact density (number of Cβ atoms within 8Å)
    /// Low density = surface, high density = buried core
    pub contact_density: f64,

    /// SASA change from reference ensemble (if available)
    /// Large positive = newly exposed surface (cryptic site opening)
    pub sasa_change: Option<f64>,

    /// Distance to nearest charged residue (Å)
    /// Binding sites often have charged residues nearby
    pub nearest_charged_dist: f64,

    // ==================== TERTIARY FEATURES (2) ====================

    /// Interface proximity score
    /// Residues at domain interfaces can form cryptic binding sites
    pub interface_score: f64,

    /// Allosteric site proximity (distance to known allosteric residues)
    /// Cryptic sites may be allosterically regulated
    pub allosteric_proximity: f64,
}

impl ResidueFeatures {
    /// Convert to padded feature vector for GPU reservoir
    ///
    /// Feature Adapter Protocol expects 40-dim input, we now have 16 features.
    /// Pad remaining dimensions with zeros (reservoir will learn to ignore them).
    ///
    /// All features are normalized to approximately [-1, 1] range using tanh or linear scaling.
    pub fn to_padded_vector(&self) -> Vec<f32> {
        let mut features = vec![0.0f32; SNN_INPUT_DIM];

        // ==================== DYNAMICS FEATURES (0-4) ====================
        features[0] = self.burial_change.tanh() as f32;
        features[1] = (self.rmsf / 5.0).tanh() as f32;
        features[2] = (self.variance / 10.0).tanh() as f32;
        features[3] = (self.neighbor_flexibility / 5.0).tanh() as f32;
        features[4] = self.burial_potential.tanh() as f32;

        // ==================== STRUCTURAL FEATURES (5-7) ====================
        // Secondary structure flexibility: 0.7-1.2 range → center around 0
        features[5] = ((self.ss_flexibility - 0.9) * 2.0) as f32;

        // Sidechain flexibility: 0.6-1.4 range → center around 0
        features[6] = ((self.sidechain_flexibility - 1.0) * 2.0) as f32;

        // B-factor: typical range 10-80 Å², normalize to ~[-1, 1]
        if let Some(bfac) = self.b_factor {
            features[7] = ((bfac - 30.0) / 30.0).clamp(-1.0, 1.0) as f32;
        }

        // ==================== CHEMICAL FEATURES (8-10) ====================
        // Net charge: -1, 0, or +1 → already normalized
        features[8] = self.net_charge.clamp(-1.0, 1.0) as f32;

        // Hydrophobicity: Kyte-Doolittle scale (-4.5 to +4.5) → normalize
        features[9] = (self.hydrophobicity / 4.5).clamp(-1.0, 1.0) as f32;

        // H-bond potential: typically 0-6 → normalize to [0, 1]
        features[10] = (self.h_bond_potential / 6.0).clamp(0.0, 1.0) as f32;

        // ==================== DISTANCE FEATURES (11-13) ====================
        // Contact density: typically 0-20 → normalize
        features[11] = (self.contact_density / 15.0).clamp(0.0, 1.0) as f32;

        // SASA change: normalized change from reference
        if let Some(sasa) = self.sasa_change {
            features[12] = (sasa / 100.0).tanh() as f32;  // 100 Å² is large change
        }

        // Distance to nearest charged residue: typically 2-30 Å
        features[13] = 1.0 - (self.nearest_charged_dist / 30.0).clamp(0.0, 1.0) as f32;

        // ==================== TERTIARY FEATURES (14-15) ====================
        // Interface score: already normalized 0-1
        features[14] = self.interface_score.clamp(0.0, 1.0) as f32;

        // Allosteric proximity: inverse distance (closer = higher)
        features[15] = self.allosteric_proximity.clamp(0.0, 1.0) as f32;

        features
    }
}

/// Chemical property lookup tables for amino acids
pub mod amino_acid_properties {
    use std::collections::HashMap;

    /// Net charge at physiological pH 7.4
    pub fn net_charge(residue_name: &str) -> f64 {
        match residue_name {
            "LYS" | "ARG" => 1.0,     // Positively charged
            "HIS" => 0.1,              // Partially protonated at pH 7.4
            "ASP" | "GLU" => -1.0,     // Negatively charged
            _ => 0.0,                   // Neutral
        }
    }

    /// Kyte-Doolittle hydrophobicity scale (normalized to ~[-1, 1])
    pub fn hydrophobicity(residue_name: &str) -> f64 {
        match residue_name {
            "ILE" => 4.5,
            "VAL" => 4.2,
            "LEU" => 3.8,
            "PHE" => 2.8,
            "CYS" => 2.5,
            "MET" => 1.9,
            "ALA" => 1.8,
            "GLY" => -0.4,
            "THR" => -0.7,
            "SER" => -0.8,
            "TRP" => -0.9,
            "TYR" => -1.3,
            "PRO" => -1.6,
            "HIS" => -3.2,
            "GLU" => -3.5,
            "GLN" => -3.5,
            "ASP" => -3.5,
            "ASN" => -3.5,
            "LYS" => -3.9,
            "ARG" => -4.5,
            _ => 0.0,
        }
    }

    /// Number of potential hydrogen bond donors + acceptors
    pub fn h_bond_potential(residue_name: &str) -> f64 {
        match residue_name {
            "ARG" => 6.0,  // 5 NH + backbone
            "LYS" => 4.0,  // 3 NH + backbone
            "HIS" => 3.0,  // NH + backbone
            "ASN" | "GLN" => 4.0,  // NH2 + C=O + backbone
            "SER" | "THR" => 3.0,  // OH + backbone
            "TYR" => 3.0,  // OH + backbone
            "TRP" => 2.0,  // NH + backbone
            "ASP" | "GLU" => 4.0,  // 2x C=O + backbone
            "CYS" => 2.0,  // SH + backbone
            "MET" => 2.0,  // backbone only + S acceptor
            _ => 2.0,      // backbone NH + C=O
        }
    }

    /// Sidechain flexibility factor (based on rotamer diversity)
    pub fn sidechain_flexibility(residue_name: &str) -> f64 {
        match residue_name {
            "GLY" => 1.40,  // Most flexible (no sidechain)
            "ALA" => 1.20,  // Small, flexible
            "SER" => 1.15,
            "CYS" => 1.10,
            "ASN" => 1.05,
            "ASP" => 1.05,
            "THR" => 1.00,  // Baseline
            "VAL" => 0.95,
            "GLU" => 0.95,
            "GLN" => 0.95,
            "HIS" => 0.90,
            "ILE" => 0.85,
            "LEU" => 0.85,
            "LYS" => 0.85,
            "MET" => 0.85,
            "PHE" => 0.80,
            "TYR" => 0.75,
            "TRP" => 0.70,
            "ARG" => 0.70,
            "PRO" => 0.60,  // Most rigid (cyclic)
            _ => 1.00,
        }
    }
}

/// Comprehensive feature extractor for cryptic site detection
///
/// Computes all 16 features from ensemble data and PDB atomic information.
pub struct CrypticFeatureExtractor {
    /// Cutoff distance for contact calculation (Å)
    contact_cutoff: f64,
}

impl CrypticFeatureExtractor {
    pub fn new(contact_cutoff: f64) -> Self {
        Self { contact_cutoff }
    }

    /// Compute comprehensive features for a single residue
    ///
    /// # Arguments
    /// * `residue_id` - Residue number
    /// * `residue_name` - Three-letter amino acid code
    /// * `ca_coord` - CA atom coordinates for this residue
    /// * `all_ca_coords` - CA coordinates for all residues in the ensemble conformation
    /// * `reference_ca` - CA coordinates from reference structure
    /// * `all_residue_names` - Residue names for all residues
    /// * `ensemble_rmsf` - Per-residue RMSF from ensemble
    /// * `ensemble_variance` - Per-residue variance from ensemble
    /// * `b_factor` - Optional B-factor from PDB
    /// * `ss_type` - Secondary structure type ('H', 'E', 'C')
    pub fn compute_features(
        &self,
        residue_id: i32,
        residue_name: &str,
        ca_coord: [f32; 3],
        all_ca_coords: &[[f32; 3]],
        reference_ca: &[[f32; 3]],
        all_residue_names: &[String],
        ensemble_rmsf: &[f64],
        ensemble_variance: &[f64],
        b_factor: Option<f64>,
        ss_type: char,
    ) -> ResidueFeatures {
        let idx = residue_id as usize - 1;  // Assuming 1-indexed
        let n_residues = all_ca_coords.len();

        // ==================== DYNAMICS FEATURES ====================
        // Burial change: compute neighbor count change
        let ref_neighbors = self.count_neighbors(&reference_ca, idx);
        let conf_neighbors = self.count_neighbors(all_ca_coords, idx);
        let burial_change = (ref_neighbors - conf_neighbors) as f64;

        let rmsf = if idx < ensemble_rmsf.len() { ensemble_rmsf[idx] } else { 0.0 };
        let variance = if idx < ensemble_variance.len() { ensemble_variance[idx] } else { 0.0 };

        // Neighbor flexibility: mean RMSF of neighbors within cutoff
        let neighbor_flexibility = self.compute_neighbor_flexibility(all_ca_coords, idx, ensemble_rmsf);

        // Burial potential: how much more could this residue become exposed?
        let max_possible_neighbors = 20;  // Typical maximum for buried residue
        let burial_potential = (max_possible_neighbors - conf_neighbors).max(0) as f64 / max_possible_neighbors as f64;

        // ==================== STRUCTURAL FEATURES ====================
        let ss_flexibility = match ss_type {
            'H' => 0.7,   // Helix - most rigid
            'E' => 0.8,   // Sheet - intermediate
            'C' | _ => 1.2,  // Loop/coil - most flexible
        };

        let sidechain_flexibility = amino_acid_properties::sidechain_flexibility(residue_name);

        // ==================== CHEMICAL FEATURES ====================
        let net_charge = amino_acid_properties::net_charge(residue_name);
        let hydrophobicity = amino_acid_properties::hydrophobicity(residue_name);
        let h_bond_potential = amino_acid_properties::h_bond_potential(residue_name);

        // ==================== DISTANCE FEATURES ====================
        let contact_density = conf_neighbors as f64;

        // SASA change would require actual SASA calculation (simplified here)
        // Using burial change as proxy
        let sasa_change = Some(burial_change * 20.0);  // Rough approximation

        // Distance to nearest charged residue
        let nearest_charged_dist = self.find_nearest_charged(
            all_ca_coords, idx, all_residue_names
        );

        // ==================== TERTIARY FEATURES ====================
        // Interface score: residues with mixed high/low burial are at interfaces
        let interface_score = self.compute_interface_score(all_ca_coords, idx);

        // Allosteric proximity: placeholder (would need known allosteric sites)
        // Use distance from center of mass as proxy
        let allosteric_proximity = self.compute_com_distance_score(all_ca_coords, idx);

        ResidueFeatures {
            residue_id,
            burial_change,
            rmsf,
            variance,
            neighbor_flexibility,
            burial_potential,
            ss_flexibility,
            sidechain_flexibility,
            b_factor,
            net_charge,
            hydrophobicity,
            h_bond_potential,
            contact_density,
            sasa_change,
            nearest_charged_dist,
            interface_score,
            allosteric_proximity,
        }
    }

    /// Count CA neighbors within cutoff
    fn count_neighbors(&self, coords: &[[f32; 3]], idx: usize) -> usize {
        let cutoff_sq = (self.contact_cutoff * self.contact_cutoff) as f32;
        let my_coord = coords[idx];
        let mut count = 0;

        for (j, other) in coords.iter().enumerate() {
            if j == idx { continue; }
            let dx = my_coord[0] - other[0];
            let dy = my_coord[1] - other[1];
            let dz = my_coord[2] - other[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            if dist_sq < cutoff_sq {
                count += 1;
            }
        }
        count
    }

    /// Compute mean RMSF of neighbors
    fn compute_neighbor_flexibility(
        &self,
        coords: &[[f32; 3]],
        idx: usize,
        rmsf: &[f64],
    ) -> f64 {
        let cutoff_sq = (self.contact_cutoff * self.contact_cutoff) as f32;
        let my_coord = coords[idx];
        let mut total_rmsf = 0.0;
        let mut count = 0;

        for (j, other) in coords.iter().enumerate() {
            if j == idx { continue; }
            let dx = my_coord[0] - other[0];
            let dy = my_coord[1] - other[1];
            let dz = my_coord[2] - other[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            if dist_sq < cutoff_sq && j < rmsf.len() {
                total_rmsf += rmsf[j];
                count += 1;
            }
        }

        if count > 0 { total_rmsf / count as f64 } else { 0.0 }
    }

    /// Find distance to nearest charged residue
    fn find_nearest_charged(
        &self,
        coords: &[[f32; 3]],
        idx: usize,
        residue_names: &[String],
    ) -> f64 {
        let my_coord = coords[idx];
        let mut min_dist = f64::MAX;

        for (j, (other, name)) in coords.iter().zip(residue_names.iter()).enumerate() {
            if j == idx { continue; }

            let charge = amino_acid_properties::net_charge(name);
            if charge.abs() > 0.5 {  // Charged residue (K, R, D, E)
                let dx = (my_coord[0] - other[0]) as f64;
                let dy = (my_coord[1] - other[1]) as f64;
                let dz = (my_coord[2] - other[2]) as f64;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                if dist < min_dist {
                    min_dist = dist;
                }
            }
        }

        if min_dist == f64::MAX { 30.0 } else { min_dist }  // Default to max if no charged found
    }

    /// Compute interface score based on neighbor distribution
    /// High score = residue at interface (mixed buried/exposed neighbors)
    fn compute_interface_score(&self, coords: &[[f32; 3]], idx: usize) -> f64 {
        // Get neighbor burial levels
        let mut neighbor_burials = Vec::new();
        let cutoff_sq = (self.contact_cutoff * self.contact_cutoff) as f32;
        let my_coord = coords[idx];

        for (j, other) in coords.iter().enumerate() {
            if j == idx { continue; }
            let dx = my_coord[0] - other[0];
            let dy = my_coord[1] - other[1];
            let dz = my_coord[2] - other[2];
            let dist_sq = dx * dx + dy * dy + dz * dz;
            if dist_sq < cutoff_sq {
                // Get burial of neighbor
                let neighbor_burial = self.count_neighbors(coords, j);
                neighbor_burials.push(neighbor_burial);
            }
        }

        if neighbor_burials.is_empty() {
            return 0.0;
        }

        // Interface residues have high variance in neighbor burial
        let mean_burial: f64 = neighbor_burials.iter().sum::<usize>() as f64 / neighbor_burials.len() as f64;
        let variance: f64 = neighbor_burials.iter()
            .map(|&b| (b as f64 - mean_burial).powi(2))
            .sum::<f64>() / neighbor_burials.len() as f64;

        // Normalize to 0-1 (variance typically 0-50)
        (variance / 25.0).min(1.0)
    }

    /// Compute distance from center of mass (inverse, normalized)
    /// Residues far from COM may be at allosteric sites
    fn compute_com_distance_score(&self, coords: &[[f32; 3]], idx: usize) -> f64 {
        // Compute center of mass
        let n = coords.len() as f32;
        let com_x: f32 = coords.iter().map(|c| c[0]).sum::<f32>() / n;
        let com_y: f32 = coords.iter().map(|c| c[1]).sum::<f32>() / n;
        let com_z: f32 = coords.iter().map(|c| c[2]).sum::<f32>() / n;

        let my_coord = coords[idx];
        let dx = (my_coord[0] - com_x) as f64;
        let dy = (my_coord[1] - com_y) as f64;
        let dz = (my_coord[2] - com_z) as f64;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt();

        // Normalize: closer to surface = higher score
        // Typical protein radius ~20-40 Å
        (dist / 30.0).min(1.0)
    }
}

/// RLS Readout Layer for single cryptic score output
///
/// Uses Sherman-Morrison formula for efficient precision matrix updates.
struct CrypticRLSReadout {
    /// Weights [reservoir_size]
    weights: Vec<f32>,

    /// Precision matrix [reservoir_size × reservoir_size] (flattened)
    p_matrix: Vec<f32>,

    /// Regularization parameter
    lambda: f32,

    /// Reservoir size
    reservoir_size: usize,

    /// Update counter
    n_updates: usize,
}

impl CrypticRLSReadout {
    /// Create new RLS readout for cryptic scoring
    fn new(reservoir_size: usize, lambda: f32) -> Self {
        // Initialize weights to small random values
        let weights: Vec<f32> = (0..reservoir_size)
            .map(|i| ((i as f32 * 0.1).sin() * 0.01))
            .collect();

        // Initialize P matrix to λ * I (diagonal)
        let mut p_matrix = vec![0.0f32; reservoir_size * reservoir_size];
        for i in 0..reservoir_size {
            p_matrix[i * reservoir_size + i] = INITIAL_P_DIAG;
        }

        log::info!(
            "CrypticRLSReadout initialized: {} reservoir neurons, λ={}",
            reservoir_size, lambda
        );

        Self {
            weights,
            p_matrix,
            lambda,
            reservoir_size,
            n_updates: 0,
        }
    }

    /// Compute cryptic score from reservoir state (sigmoid output)
    fn compute_score(&self, state: &[f32]) -> f32 {
        let raw: f32 = self.weights.iter()
            .zip(state.iter())
            .map(|(w, s)| w * s)
            .sum();

        // Sigmoid for [0, 1] output
        1.0 / (1.0 + (-raw).exp())
    }

    /// Reward-Modulated RLS update
    ///
    /// Based on prism-learning/dendritic_agent.rs implementation
    fn rls_update_modulated(&mut self, state: &[f32], target: f32, reward_modulation: f32) {
        let n = self.reservoir_size;

        // Skip update if state contains NaN/Inf
        if state.iter().any(|x| !x.is_finite()) {
            log::warn!("RLS: Skipping update due to non-finite state values");
            return;
        }

        // Clamp reward modulation to safe range
        let modulation = reward_modulation.clamp(MIN_REWARD_MODULATION, MAX_REWARD_MODULATION);

        // Step 1: Compute P @ x
        let mut px = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..n {
                px[i] += self.p_matrix[i * n + j] * state[j];
            }
        }

        // Step 2: Compute x^T @ P @ x
        let xtpx: f32 = state.iter()
            .zip(px.iter())
            .map(|(xi, pxi)| xi * pxi)
            .sum();

        // Step 3: Compute Kalman gain k = P @ x / (λ + x^T @ P @ x)
        let denom = (self.lambda + xtpx).max(1e-8);
        let k: Vec<f32> = px.iter()
            .map(|pxi| (pxi / denom).clamp(-1e6, 1e6))
            .collect();

        // Step 4: Compute prediction error
        let prediction = self.compute_score(state);
        let error = (target - prediction).clamp(-100.0, 100.0);

        // Step 5: Update weights with reward modulation
        // Gradient through sigmoid: d(sigmoid)/dx = sigmoid * (1 - sigmoid)
        let sigmoid_deriv = prediction * (1.0 - prediction);
        for i in 0..n {
            self.weights[i] += modulation * error * sigmoid_deriv * k[i];
            self.weights[i] = self.weights[i].clamp(-1e4, 1e4);
        }

        // Step 6: Sherman-Morrison P matrix update
        let inv_lambda = 1.0 / self.lambda;
        let p_modulation = 0.5 + 0.5 * modulation;
        for i in 0..n {
            for j in 0..n {
                let delta = p_modulation * k[i] * px[j];
                self.p_matrix[i * n + j] = inv_lambda * (self.p_matrix[i * n + j] - delta);
                self.p_matrix[i * n + j] = self.p_matrix[i * n + j].clamp(-1e8, 1e8);
            }
        }

        // Step 7: Regularization - ensure P diagonal doesn't collapse
        for i in 0..n {
            self.p_matrix[i * n + i] = self.p_matrix[i * n + i].clamp(1e-6, 1e6);
        }

        self.n_updates += 1;
    }

    /// Compute reward modulation factor from binary label
    fn compute_reward_modulation(is_cryptic: bool) -> f32 {
        // Cryptic = high reward (learn more), non-cryptic = baseline
        if is_cryptic {
            MAX_REWARD_MODULATION
        } else {
            1.0  // Baseline learning for negatives
        }
    }
}

/// PRISM-ZrO Cryptic Site Scorer - GPU-Accelerated
///
/// Uses the REAL DendriticSNNReservoir from prism-gpu.
pub struct ZroCrypticScorer {
    config: ZroCrypticConfig,

    /// GPU SNN Reservoir (512 neurons, E/I balanced)
    reservoir: DendriticSNNReservoir,

    /// RLS Readout layer
    readout: CrypticRLSReadout,

    /// Running statistics
    pub n_scored: usize,
    pub mean_score: f32,

    /// Whether GPU is active
    gpu_active: bool,
}

impl ZroCrypticScorer {
    /// Create a new PRISM-ZrO cryptic scorer with GPU acceleration
    pub fn new(config: ZroCrypticConfig) -> Result<Self> {
        log::info!("🚀 Initializing PRISM-ZrO Cryptic Scorer (GPU)...");

        // Initialize CUDA context
        let context = CudaContext::new(0)
            .context("Failed to create CUDA context - is GPU available?")?;

        // Create GPU reservoir (512 neurons)
        let mut reservoir = DendriticSNNReservoir::new(context, GPU_RESERVOIR_SIZE)
            .context("Failed to create DendriticSNNReservoir")?;

        // Initialize reservoir weights
        let seed = config.seed.unwrap_or(42);
        reservoir.initialize(seed)
            .context("Failed to initialize reservoir")?;

        // Create RLS readout
        let readout = CrypticRLSReadout::new(GPU_RESERVOIR_SIZE, config.lambda);

        log::info!("✅ PRISM-ZrO initialized: {} GPU neurons, E/I balanced", GPU_RESERVOIR_SIZE);

        Ok(Self {
            config,
            reservoir,
            readout,
            n_scored: 0,
            mean_score: 0.5,
            gpu_active: true,
        })
    }

    /// Score a single residue
    pub fn score(&mut self, features: &ResidueFeatures) -> Result<f64> {
        let padded = features.to_padded_vector();
        self.score_from_vector(&padded)
    }

    /// Score from raw feature vector (must be SNN_INPUT_DIM = 40 dimensions)
    pub fn score_from_vector(&mut self, features: &[f32]) -> Result<f64> {
        // Process through GPU reservoir
        let state = self.reservoir.process_features(features)
            .context("GPU reservoir processing failed")?;

        // Compute score through RLS readout
        let score = self.readout.compute_score(&state);

        // Update statistics
        self.n_scored += 1;
        self.mean_score = 0.99 * self.mean_score + 0.01 * score;

        Ok(score as f64)
    }

    /// Score and learn from ground truth
    pub fn score_and_learn(
        &mut self,
        features: &ResidueFeatures,
        is_cryptic: Option<bool>,
    ) -> Result<f64> {
        let padded = features.to_padded_vector();

        // Process through GPU reservoir
        let state = self.reservoir.process_features(&padded)
            .context("GPU reservoir processing failed")?;

        // Compute score
        let score = self.readout.compute_score(&state);

        // Learn if ground truth available and online learning enabled
        if let Some(cryptic) = is_cryptic {
            if self.config.online_learning {
                let target = if cryptic { 1.0 } else { 0.0 };
                let modulation = CrypticRLSReadout::compute_reward_modulation(cryptic);
                self.readout.rls_update_modulated(&state, target, modulation);
            }
        }

        // Update statistics
        self.n_scored += 1;
        self.mean_score = 0.99 * self.mean_score + 0.01 * score;

        Ok(score as f64)
    }

    /// Score all residues in a structure
    pub fn score_structure(
        &mut self,
        features: &[ResidueFeatures],
        ground_truth: Option<&HashMap<i32, bool>>,
    ) -> Result<HashMap<i32, f64>> {
        // Reset reservoir state for new structure
        self.reservoir.reset_state()
            .context("Failed to reset reservoir state")?;

        let mut scores = HashMap::new();

        for feat in features {
            let is_cryptic = ground_truth.and_then(|gt| gt.get(&feat.residue_id).copied());
            let score = self.score_and_learn(feat, is_cryptic)?;
            scores.insert(feat.residue_id, score);
        }

        Ok(scores)
    }

    /// Reset reservoir state for new structure
    pub fn reset_reservoir(&mut self) -> Result<()> {
        self.reservoir.reset_state()
            .context("Failed to reset reservoir state")
    }

    /// Get number of RLS updates performed
    pub fn n_updates(&self) -> usize {
        self.readout.n_updates
    }

    /// Check if GPU is active
    pub fn is_gpu_active(&self) -> bool {
        self.gpu_active
    }

    /// Extract features from ensemble data (ENHANCED with structural context)
    ///
    /// Helper to convert ensemble pocket detector data to ResidueFeatures.
    /// Now includes secondary structure, sidechain flexibility, and optional B-factors.
    ///
    /// # Arguments
    /// * `original_coords` - CA coordinates
    /// * `per_residue_rmsf` - RMSF from ensemble
    /// * `per_residue_variance` - Position variance from ensemble
    /// * `neighbor_counts_ref` - Neighbor counts in reference structure
    /// * `neighbor_counts_mean` - Mean neighbor counts across ensemble
    /// * `residue_ids` - Residue IDs
    /// * `ss_flexibility` - Secondary structure flexibility factors (from SecondaryStructureAnalyzer)
    /// * `sidechain_flexibility` - Sidechain flexibility factors (from sidechain_analysis)
    /// * `b_factors` - Optional B-factors from PDB file
    pub fn extract_features(
        original_coords: &[[f32; 3]],
        per_residue_rmsf: &[f64],
        per_residue_variance: &[f64],
        neighbor_counts_ref: &[usize],
        neighbor_counts_mean: &[f64],
        residue_ids: &[i32],
        ss_flexibility: Option<&[f64]>,
        sidechain_flexibility: Option<&[f64]>,
        b_factors: Option<&[f64]>,
    ) -> Vec<ResidueFeatures> {
        let n = original_coords.len();
        let mut features = Vec::with_capacity(n);

        for i in 0..n {
            // Burial change: decrease in neighbors (more exposed)
            let burial_change = if neighbor_counts_ref[i] > 0 {
                (neighbor_counts_ref[i] as f64 - neighbor_counts_mean[i]) / neighbor_counts_ref[i] as f64
            } else {
                0.0
            };

            // Neighbor flexibility: mean RMSF of nearby residues
            let neighbor_flex = if i > 0 && i < n - 1 {
                (per_residue_rmsf[i - 1] + per_residue_rmsf[i] + per_residue_rmsf[i + 1]) / 3.0
            } else {
                per_residue_rmsf[i]
            };

            // Burial potential: how much more exposed could this residue become
            let max_neighbors = 12.0;
            let burial_potential = 1.0 - (neighbor_counts_ref[i] as f64 / max_neighbors).min(1.0);

            // NEW: Get secondary structure flexibility (default to Loop=1.2 if not provided)
            let ss_flex = ss_flexibility
                .and_then(|ss| ss.get(i).copied())
                .unwrap_or(1.0);  // Neutral default

            // NEW: Get sidechain flexibility (default to average=1.0 if not provided)
            let sc_flex = sidechain_flexibility
                .and_then(|sc| sc.get(i).copied())
                .unwrap_or(1.0);  // Neutral default

            // NEW: Get B-factor if available
            let bfac = b_factors.and_then(|bf| bf.get(i).copied());

            // Contact density: use neighbor count as proxy (normalized)
            let contact_density = (neighbor_counts_ref[i] as f64 / 12.0).min(1.0);

            features.push(ResidueFeatures {
                residue_id: residue_ids[i],
                // Dynamics features (5)
                burial_change,
                rmsf: per_residue_rmsf[i],
                variance: per_residue_variance[i],
                neighbor_flexibility: neighbor_flex,
                burial_potential,
                // Structural features (3)
                ss_flexibility: ss_flex,
                sidechain_flexibility: sc_flex,
                b_factor: bfac,
                // Chemical features (3) - defaults; use CrypticFeatureExtractor for full computation
                net_charge: 0.0,           // Neutral default
                hydrophobicity: 0.0,       // Neutral on Kyte-Doolittle scale
                h_bond_potential: 2.0,     // Average H-bond capacity
                // Distance features (3)
                contact_density,
                sasa_change: None,         // Requires SASA computation
                nearest_charged_dist: 10.0, // Default moderate distance (Å)
                // Tertiary features (2)
                interface_score: 0.0,       // Neutral default
                allosteric_proximity: 20.0, // Default far from allosteric sites (Å)
            });
        }

        features
    }

    /// Extract features with defaults for backward compatibility
    ///
    /// Uses neutral values (1.0) for new features when not available.
    pub fn extract_features_simple(
        original_coords: &[[f32; 3]],
        per_residue_rmsf: &[f64],
        per_residue_variance: &[f64],
        neighbor_counts_ref: &[usize],
        neighbor_counts_mean: &[f64],
        residue_ids: &[i32],
    ) -> Vec<ResidueFeatures> {
        Self::extract_features(
            original_coords,
            per_residue_rmsf,
            per_residue_variance,
            neighbor_counts_ref,
            neighbor_counts_mean,
            residue_ids,
            None,  // No SS info
            None,  // No sidechain info
            None,  // No B-factors
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ZroCrypticConfig::default();
        assert!((config.lambda - 0.99).abs() < 0.01);
        assert!(config.online_learning);
    }

    #[test]
    fn test_feature_padding() {
        let features = ResidueFeatures {
            residue_id: 0,
            burial_change: 0.5,
            rmsf: 2.0,
            variance: 4.0,
            neighbor_flexibility: 1.5,
            burial_potential: 0.3,
            ss_flexibility: 1.2,        // Loop
            sidechain_flexibility: 1.4,  // GLY
            b_factor: Some(45.0),        // Above average
            ..Default::default()
        };

        let padded = features.to_padded_vector();
        assert_eq!(padded.len(), SNN_INPUT_DIM);

        // First 8 should be non-zero (our features)
        assert!(padded[0] != 0.0, "burial_change should be non-zero");
        assert!(padded[1] != 0.0, "rmsf should be non-zero");
        assert!(padded[5] != 0.0, "ss_flexibility should be non-zero");
        assert!(padded[6] != 0.0, "sidechain_flexibility should be non-zero");
        assert!(padded[7] != 0.0, "b_factor should be non-zero");

        // Rest should be zero padding
        for i in 8..SNN_INPUT_DIM {
            assert_eq!(padded[i], 0.0, "feature {} should be zero-padded", i);
        }
    }

    #[test]
    fn test_feature_padding_no_bfactor() {
        let features = ResidueFeatures {
            residue_id: 0,
            burial_change: 0.5,
            rmsf: 2.0,
            variance: 4.0,
            neighbor_flexibility: 1.5,
            burial_potential: 0.3,
            ss_flexibility: 0.7,         // Helix
            sidechain_flexibility: 0.6,  // PRO
            b_factor: None,              // No B-factor available
            ..Default::default()
        };

        let padded = features.to_padded_vector();

        // B-factor should be 0 when not available
        assert_eq!(padded[7], 0.0, "b_factor should be 0 when None");

        // SS flexibility for helix (0.7) → normalized: (0.7 - 0.9) * 2.0 = -0.4
        assert!(padded[5] < 0.0, "helix should have negative ss_flexibility");

        // Sidechain flexibility for PRO (0.6) → normalized: (0.6 - 1.0) * 2.0 = -0.8
        assert!(padded[6] < 0.0, "PRO should have negative sidechain_flexibility");
    }

    #[test]
    fn test_rls_readout_creation() {
        let readout = CrypticRLSReadout::new(64, 0.99);
        assert_eq!(readout.weights.len(), 64);
        assert_eq!(readout.p_matrix.len(), 64 * 64);
    }

    #[test]
    #[ignore]  // Requires GPU
    fn test_gpu_scorer_creation() {
        let config = ZroCrypticConfig::default();
        let scorer = ZroCrypticScorer::new(config);
        assert!(scorer.is_ok());

        let scorer = scorer.unwrap();
        assert!(scorer.is_gpu_active());
    }
}

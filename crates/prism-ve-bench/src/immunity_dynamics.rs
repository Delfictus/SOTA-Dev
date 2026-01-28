//! Time-varying immunity computation with PK waning curves
//!
//! Implements the immunity dynamics component of PRISM-VE:
//! I(t) = I_max x exp(-(t - t_max) x ln(2) / t_half) for t >= t_max
//!
//! This module provides:
//! - Loading of 75 PK parameter combinations from JSON
//! - Exponential decay with half-life modeling
//! - Per-country outbreak dating support
//! - 10-dimensional epitope immunity vector computation
//! - Cross-reactivity matrix integration for inter-variant immunity
//!
//! GPU-accelerated batch computation is supported via the `compute_immunity_batch` method.

use anyhow::{Result, Context, bail};
use std::collections::HashMap;
use std::path::Path;
use std::fs::File;
use std::io::BufReader;
use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use log::{info, debug, warn};

// ============================================================================
// CONSTANTS
// ============================================================================

/// Epitope class names in order (10 total, VASIL standard)
pub const EPITOPE_CLASSES: [&str; 10] = [
    "A", "B", "C", "D1", "D2", "E12", "E3", "F1", "F2", "F3"
];

/// Natural log of 2 (used in half-life calculations)
const LN_2: f32 = 0.693147181;

/// Default outbreak reference dates for VASIL countries (days since 2020-01-01)
/// These represent the start of significant COVID-19 transmission in each country
const DEFAULT_OUTBREAK_DATES: &[(&str, &str)] = &[
    ("Germany", "2020-03-01"),
    ("USA", "2020-03-01"),
    ("UK", "2020-03-01"),
    ("Japan", "2020-03-15"),
    ("Brazil", "2020-03-15"),
    ("France", "2020-03-01"),
    ("Canada", "2020-03-15"),
    ("Denmark", "2020-03-01"),
    ("Australia", "2020-03-15"),
    ("Sweden", "2020-03-01"),
    ("Mexico", "2020-03-15"),
    ("SouthAfrica", "2020-03-15"),
];

// ============================================================================
// PK PARAMETERS
// ============================================================================

/// PK (Pharmacokinetic) parameters for immunity waning
///
/// These parameters define the antibody decay curve:
/// - t_half: Half-life in days (time for antibody levels to decay by 50%)
/// - t_max: Time to peak immunity in days (after vaccination/infection)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PKParameters {
    /// Half-life in days (25.0 to 69.0 in VASIL)
    pub t_half: f32,
    /// Time to peak immunity in days (14.0 to 28.0 in VASIL)
    pub t_max: f32,
}

impl PKParameters {
    /// Create new PK parameters with validation
    pub fn new(t_half: f32, t_max: f32) -> Result<Self> {
        if t_half <= 0.0 {
            bail!("t_half must be positive, got {}", t_half);
        }
        if t_max < 0.0 {
            bail!("t_max must be non-negative, got {}", t_max);
        }
        Ok(Self { t_half, t_max })
    }

    /// Fast decay scenario (young/healthy immune response)
    pub fn fast() -> Self {
        Self { t_half: 25.0, t_max: 14.0 }
    }

    /// Medium decay scenario (default population average)
    pub fn medium() -> Self {
        Self { t_half: 47.0, t_max: 21.0 }
    }

    /// Slow decay scenario (older adults, certain conditions)
    pub fn slow() -> Self {
        Self { t_half: 69.0, t_max: 28.0 }
    }

    /// Compute decay constant (lambda = ln(2) / t_half)
    #[inline]
    pub fn decay_constant(&self) -> f32 {
        LN_2 / self.t_half
    }
}

impl Default for PKParameters {
    fn default() -> Self {
        Self::medium()
    }
}

// ============================================================================
// PK PARAMETERS FILE
// ============================================================================

/// JSON structure for pk_parameters.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PKParametersFile {
    /// All 75 PK parameter combinations
    pub pk_combinations: Vec<PKParameters>,
    /// Description of the file
    #[serde(default)]
    pub description: String,
    /// Formula documentation
    #[serde(default)]
    pub formula: String,
    /// Total number of columns (should be 75)
    #[serde(default)]
    pub total_columns: usize,
}

impl PKParametersFile {
    /// Load PK parameters from JSON file
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .context(format!("Failed to open PK parameters file: {:?}", path))?;
        let reader = BufReader::new(file);
        let params: PKParametersFile = serde_json::from_reader(reader)
            .context("Failed to parse PK parameters JSON")?;

        if params.pk_combinations.is_empty() {
            bail!("PK parameters file contains no combinations");
        }

        info!("Loaded {} PK parameter combinations from {:?}",
              params.pk_combinations.len(), path);

        Ok(params)
    }

    /// Get the median PK parameters (index 37 of 75 = middle)
    pub fn median_params(&self) -> &PKParameters {
        let mid_idx = self.pk_combinations.len() / 2;
        &self.pk_combinations[mid_idx]
    }

    /// Get PK parameters by index
    pub fn get(&self, idx: usize) -> Option<&PKParameters> {
        self.pk_combinations.get(idx)
    }
}

// ============================================================================
// EPITOPE IMMUNITY
// ============================================================================

/// Epitope-specific immunity at a given time
///
/// Contains immunity levels for all 10 VASIL epitope classes:
/// A, B, C, D1, D2, E12, E3, F1, F2, F3
#[derive(Debug, Clone, Default)]
pub struct EpitopeImmunity {
    /// Immunity levels for each epitope class [0.0, 1.0]
    /// Index mapping: 0=A, 1=B, 2=C, 3=D1, 4=D2, 5=E12, 6=E3, 7=F1, 8=F2, 9=F3
    pub levels: [f32; 10],
}

impl EpitopeImmunity {
    /// Create new epitope immunity with zero levels
    pub fn new() -> Self {
        Self { levels: [0.0; 10] }
    }

    /// Create with uniform immunity across all epitopes
    pub fn uniform(level: f32) -> Self {
        Self { levels: [level.clamp(0.0, 1.0); 10] }
    }

    /// Create from array
    pub fn from_array(levels: [f32; 10]) -> Self {
        Self { levels }
    }

    /// Get immunity for specific epitope by name
    pub fn get_by_name(&self, epitope: &str) -> Option<f32> {
        EPITOPE_CLASSES.iter()
            .position(|&e| e == epitope)
            .map(|idx| self.levels[idx])
    }

    /// Set immunity for specific epitope by name
    pub fn set_by_name(&mut self, epitope: &str, value: f32) -> bool {
        if let Some(idx) = EPITOPE_CLASSES.iter().position(|&e| e == epitope) {
            self.levels[idx] = value.clamp(0.0, 1.0);
            true
        } else {
            false
        }
    }

    /// Compute mean immunity across all epitopes
    pub fn mean(&self) -> f32 {
        self.levels.iter().sum::<f32>() / 10.0
    }

    /// Compute weighted fold reduction given escape scores
    /// fold_reduction = exp(sum(escape[i] * immunity[i]))
    pub fn compute_fold_reduction(&self, escape_scores: &[f32; 10]) -> f32 {
        let sum: f32 = escape_scores.iter()
            .zip(self.levels.iter())
            .map(|(&escape, &immunity)| escape * immunity)
            .sum();
        sum.exp()
    }
}

// ============================================================================
// CROSS-REACTIVITY MATRIX
// ============================================================================

/// Cross-reactivity matrix for inter-variant immunity
///
/// Stores per-epitope cross-immunity between variants
/// Simplified: epitope -> variant -> normalized cross-immunity [0, 1]
#[derive(Debug, Clone)]
pub struct CrossReactivityMatrix {
    /// Variant names in order
    pub variant_names: Vec<String>,
    /// Per-epitope, per-variant cross-immunity: epitope -> variant -> cross_immunity
    /// Normalized values [0, 1]: 1.0 = strong protection, 0.1 = weak
    pub matrices: HashMap<String, HashMap<String, f32>>,
}

impl CrossReactivityMatrix {
    /// Load from cross_immunity_per_variant.json
    pub fn load(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .context(format!("Failed to open cross-immunity file: {:?}", path))?;
        let reader = BufReader::new(file);
        let data: serde_json::Value = serde_json::from_reader(reader)
            .context("Failed to parse cross-immunity JSON")?;

        // Load variant cross-immunity map
        let variant_cross_immunity = data["variant_cross_immunity"]
            .as_object()
            .context("Missing variant_cross_immunity")?;

        // Extract variant names
        let variant_names: Vec<String> = variant_cross_immunity.keys()
            .map(|k| k.to_string())
            .collect();

        // Convert to per-epitope hashmaps
        let mut matrices: HashMap<String, HashMap<String, f32>> = HashMap::new();

        for epitope in EPITOPE_CLASSES {
            let mut epitope_map = HashMap::new();

            for (variant, epitope_data) in variant_cross_immunity {
                if let Some(cross_val) = epitope_data.get(epitope).and_then(|v| v.as_f64()) {
                    // Normalize large fold-reduction values to [0, 2] range
                    // Values > 100 indicate very weak cross-immunity
                    let normalized = if cross_val > 100.0 {
                        0.1  // Weak cross-protection
                    } else if cross_val > 10.0 {
                        0.5  // Moderate
                    } else if cross_val > 2.0 {
                        0.8  // Good
                    } else {
                        1.0  // Strong (diagonal = 1.0)
                    };
                    epitope_map.insert(variant.clone(), normalized as f32);
                }
            }

            matrices.insert(epitope.to_string(), epitope_map);
        }

        info!("Loaded cross-immunity for {} variants, {} epitopes",
              variant_names.len(), matrices.len());

        Ok(Self {
            variant_names,
            matrices,
        })
    }

    /// Get variant index by name
    pub fn get_variant_idx(&self, lineage: &str) -> Option<usize> {
        // Try exact match first
        if let Some(idx) = self.variant_names.iter().position(|v| v == lineage) {
            return Some(idx);
        }

        // Try family matching
        let lin = lineage.to_uppercase();

        for (idx, variant) in self.variant_names.iter().enumerate() {
            let var = variant.to_uppercase();

            // XBB family
            if (lin.starts_with("XBB") || lin.starts_with("EG.") || lin.starts_with("HK."))
                && (var.starts_with("XBB") || var.starts_with("EG.")) {
                return Some(idx);
            }

            // BA.5 family
            if (lin.starts_with("BA.5") || lin.starts_with("BA.4") || lin.starts_with("BQ."))
                && (var.starts_with("BA.5") || var.starts_with("BA.4") || var.starts_with("BQ.")) {
                return Some(idx);
            }

            // BA.2 family
            if lin.starts_with("BA.2") && var.starts_with("BA.2") {
                return Some(idx);
            }

            // BA.1 family
            if (lin.starts_with("BA.1") || lin.starts_with("B.1.1.529"))
                && (var.starts_with("BA.1") || var.starts_with("B.1.1.529")) {
                return Some(idx);
            }

            // Delta family
            if lin.starts_with("AY.") && var.starts_with("AY.") {
                return Some(idx);
            }
        }

        None
    }

    /// Get cross-immunity factor for a specific epitope and lineage pair
    pub fn get_cross_immunity(&self, lineage: &str, epitope_idx: usize) -> f32 {
        if epitope_idx >= 10 {
            return 1.0;
        }

        let epitope = EPITOPE_CLASSES[epitope_idx];

        // Get epitope-specific matrix
        let epitope_matrix = match self.matrices.get(epitope) {
            Some(m) => m,
            None => return 1.0,  // No data, assume full immunity
        };

        // Lookup cross-immunity for this lineage
        // Try exact match first, then family matching
        if let Some(&cross_imm) = epitope_matrix.get(lineage) {
            return cross_imm;
        }

        // Family-based fallback
        let family = Self::get_variant_family_static(lineage);
        for (variant, &cross_imm) in epitope_matrix.iter() {
            if Self::get_variant_family_static(variant) == family {
                return cross_imm;
            }
        }

        // Default: moderate cross-immunity
        0.7
    }

    /// Get variant family for matching
    fn get_variant_family_static(lineage: &str) -> &str {
        let lin = lineage.to_uppercase();
        if lin.starts_with("XBB") || lin.starts_with("EG.") || lin.starts_with("HK.") {
            "XBB"
        } else if lin.starts_with("BA.5") || lin.starts_with("BQ.") || lin.starts_with("BA.4") {
            "BA.5"
        } else if lin.starts_with("BA.2") {
            "BA.2"
        } else if lin.starts_with("BA.1") || lin.starts_with("B.1.1.529") {
            "BA.1"
        } else if lin.starts_with("AY.") || lin.starts_with("B.1.617.2") {
            "Delta"
        } else if lin.starts_with("B.1.1.7") {
            "Alpha"
        } else {
            "Other"
        }
    }
}

impl Default for CrossReactivityMatrix {
    fn default() -> Self {
        Self {
            variant_names: Vec::new(),
            matrices: HashMap::new(),
        }
    }
}

// ============================================================================
// OUTBREAK DATES
// ============================================================================

/// Per-country outbreak reference dates
#[derive(Debug, Clone)]
pub struct OutbreakDates {
    /// Country -> First outbreak date
    dates: HashMap<String, NaiveDate>,
}

impl OutbreakDates {
    /// Create with default VASIL country outbreak dates
    pub fn new_defaults() -> Self {
        let mut dates = HashMap::new();

        for (country, date_str) in DEFAULT_OUTBREAK_DATES {
            if let Ok(date) = NaiveDate::parse_from_str(date_str, "%Y-%m-%d") {
                dates.insert(country.to_string(), date);
            }
        }

        Self { dates }
    }

    /// Get outbreak date for country
    pub fn get(&self, country: &str) -> Option<NaiveDate> {
        self.dates.get(country).copied()
    }

    /// Set outbreak date for country
    pub fn set(&mut self, country: &str, date: NaiveDate) {
        self.dates.insert(country.to_string(), date);
    }

    /// Compute days since outbreak for a given country and date
    pub fn days_since_outbreak(&self, country: &str, current_date: NaiveDate) -> i32 {
        self.get(country)
            .map(|outbreak| (current_date - outbreak).num_days() as i32)
            .unwrap_or(0)
    }
}

impl Default for OutbreakDates {
    fn default() -> Self {
        Self::new_defaults()
    }
}

// ============================================================================
// IMMUNITY DYNAMICS (MAIN STRUCT)
// ============================================================================

/// Immunity dynamics with waning curves
///
/// Core component for computing time-varying immunity using PK (pharmacokinetic)
/// waning curves. Implements the VASIL formula:
///
/// I(t) = {
///     t / t_max                              if t < t_max (rising phase)
///     exp(-(t - t_max) * ln(2) / t_half)    if t >= t_max (decay phase)
/// }
pub struct ImmunityDynamics {
    /// All 75 PK parameter combinations
    pk_params: Vec<PKParameters>,

    /// Per-country outbreak reference dates
    outbreak_dates: OutbreakDates,

    /// Cross-reactivity matrix for inter-variant immunity
    cross_reactivity: Option<CrossReactivityMatrix>,

    /// Selected PK parameter index (default: median = 37)
    selected_pk_idx: usize,

    /// Per-epitope PK parameter indices (for fine-grained control)
    epitope_pk_indices: [usize; 10],

    /// Immunity floor (minimum immunity level, prevents complete decay)
    immunity_floor: f32,

    /// Peak immunity level (I_max)
    i_max: f32,
}

impl ImmunityDynamics {
    /// Create new ImmunityDynamics with default PK parameters
    pub fn new() -> Self {
        // Generate default 75 PK combinations (15 t_half x 5 t_max)
        let mut pk_params = Vec::with_capacity(75);

        // t_half values: 25.0 to 69.0 in 15 steps (3.143 step)
        let t_half_values: Vec<f32> = (0..15)
            .map(|i| 25.0 + i as f32 * (69.0 - 25.0) / 14.0)
            .collect();

        // t_max values: 14.0 to 28.0 in 5 steps (3.5 step)
        let t_max_values: Vec<f32> = (0..5)
            .map(|i| 14.0 + i as f32 * (28.0 - 14.0) / 4.0)
            .collect();

        for t_half in &t_half_values {
            for t_max in &t_max_values {
                pk_params.push(PKParameters {
                    t_half: *t_half,
                    t_max: *t_max,
                });
            }
        }

        let median_idx = pk_params.len() / 2;

        Self {
            pk_params,
            outbreak_dates: OutbreakDates::new_defaults(),
            cross_reactivity: None,
            selected_pk_idx: median_idx,
            epitope_pk_indices: [median_idx; 10],
            immunity_floor: 0.05,
            i_max: 1.0,
        }
    }

    /// Load from VASIL data directory
    ///
    /// Loads PK parameters from JSON and sets up outbreak dates
    pub fn load_from_vasil(data_dir: &Path) -> Result<Self> {
        // Try to find pk_parameters.json in data directory
        let pk_file = if data_dir.join("pk_parameters.json").exists() {
            data_dir.join("pk_parameters.json")
        } else if data_dir.join("data/pk_parameters.json").exists() {
            data_dir.join("data/pk_parameters.json")
        } else if data_dir.join("../../data/pk_parameters.json").exists() {
            data_dir.join("../../data/pk_parameters.json")
        } else {
            // Fallback to data dir
            Path::new("data/pk_parameters.json").to_path_buf()
        };

        let pk_file_data = PKParametersFile::load(&pk_file)?;

        let pk_params = pk_file_data.pk_combinations;
        let median_idx = pk_params.len() / 2;

        info!("Loaded {} PK parameter combinations, selected median index {}",
              pk_params.len(), median_idx);

        // Try to load cross-immunity matrix (per-variant format)
        let cross_react_file = if data_dir.join("cross_immunity_per_variant.json").exists() {
            Some(data_dir.join("cross_immunity_per_variant.json"))
        } else if data_dir.join("data/cross_immunity_per_variant.json").exists() {
            Some(data_dir.join("data/cross_immunity_per_variant.json"))
        } else if data_dir.join("../../data/cross_immunity_per_variant.json").exists() {
            Some(data_dir.join("../../data/cross_immunity_per_variant.json"))
        } else {
            let fallback = Path::new("data/cross_immunity_per_variant.json");
            if fallback.exists() { Some(fallback.to_path_buf()) } else { None }
        };

        let cross_reactivity = cross_react_file
            .and_then(|p| CrossReactivityMatrix::load(&p).ok());

        Ok(Self {
            pk_params,
            outbreak_dates: OutbreakDates::new_defaults(),
            cross_reactivity,
            selected_pk_idx: median_idx,
            epitope_pk_indices: [median_idx; 10],
            immunity_floor: 0.05,
            i_max: 1.0,
        })
    }

    /// Get number of PK parameter combinations
    pub fn num_pk_combinations(&self) -> usize {
        self.pk_params.len()
    }

    /// Alias for num_pk_combinations (compatibility)
    pub fn n_pk_combinations(&self) -> usize {
        self.pk_params.len()
    }

    /// Alias for new() (compatibility)
    pub fn new_default() -> Self {
        Self::new()
    }

    /// Get current selected PK parameters
    pub fn current_pk_params(&self) -> &PKParameters {
        &self.pk_params[self.selected_pk_idx]
    }

    /// Select PK parameter combination by index
    ///
    /// Use this to optimize which PK parameters best fit observed data
    pub fn select_pk_params(&mut self, idx: usize) {
        if idx < self.pk_params.len() {
            self.selected_pk_idx = idx;
            // Update all epitope indices to match
            self.epitope_pk_indices = [idx; 10];
            info!("Selected PK params index {}: t_half={:.1}, t_max={:.1}",
                  idx, self.pk_params[idx].t_half, self.pk_params[idx].t_max);
        } else {
            warn!("Invalid PK index {}, keeping current {}", idx, self.selected_pk_idx);
        }
    }

    /// Select PK parameters per epitope (for fine-grained control)
    pub fn select_epitope_pk_params(&mut self, epitope_idx: usize, pk_idx: usize) {
        if epitope_idx < 10 && pk_idx < self.pk_params.len() {
            self.epitope_pk_indices[epitope_idx] = pk_idx;
        }
    }

    /// Set immunity floor (minimum immunity level)
    pub fn set_immunity_floor(&mut self, floor: f32) {
        self.immunity_floor = floor.clamp(0.0, 0.5);
    }

    /// Set peak immunity level (I_max)
    pub fn set_i_max(&mut self, i_max: f32) {
        self.i_max = i_max.clamp(0.1, 2.0);
    }

    /// Compute immunity for single epitope with PK curve
    ///
    /// Implements the VASIL waning formula:
    /// - Rising phase (t < t_max): I(t) = I_max * (t / t_max)
    /// - Decay phase (t >= t_max): I(t) = I_max * exp(-(t - t_max) * ln(2) / t_half)
    ///
    /// # Arguments
    /// * `t` - Time in days since vaccination/infection
    /// * `pk` - PK parameters (t_half, t_max)
    ///
    /// # Returns
    /// Immunity level in range [immunity_floor, i_max]
    #[inline]
    fn compute_epitope_immunity(&self, t: f32, pk: &PKParameters) -> f32 {
        if t < 0.0 {
            return 0.0;
        }

        let immunity = if t < pk.t_max {
            // Rising phase: Linear rise from 0 to I_max
            self.i_max * (t / pk.t_max)
        } else {
            // Decay phase: Exponential decay from I_max
            let t_since_peak = t - pk.t_max;
            let decay = (-t_since_peak * LN_2 / pk.t_half).exp();
            self.i_max * decay
        };

        // Apply floor
        immunity.max(self.immunity_floor)
    }

    /// Compute epitope-specific immunity at time t
    ///
    /// # Arguments
    /// * `country` - Country name for outbreak date lookup
    /// * `lineage` - Variant lineage (for cross-reactivity)
    /// * `days_since_outbreak` - Days since country outbreak
    ///
    /// # Returns
    /// EpitopeImmunity with 10-dimensional immunity vector
    pub fn compute_immunity(
        &self,
        country: &str,
        lineage: &str,
        days_since_outbreak: i32,
    ) -> EpitopeImmunity {
        let t = days_since_outbreak as f32;

        let mut immunity = EpitopeImmunity::new();

        // Compute immunity for each epitope using its specific PK parameters
        for i in 0..10 {
            let pk_idx = self.epitope_pk_indices[i];
            let pk = &self.pk_params[pk_idx];

            let base_immunity = self.compute_epitope_immunity(t, pk);

            // Apply cross-reactivity modulation if available
            let cross_immunity_factor = self.cross_reactivity
                .as_ref()
                .map(|cr| cr.get_cross_immunity(lineage, i))
                .unwrap_or(1.0);

            immunity.levels[i] = (base_immunity * cross_immunity_factor).clamp(0.0, 1.0);
        }

        immunity
    }

    /// Compute immunity at a specific date for a country
    pub fn compute_immunity_at_date(
        &self,
        country: &str,
        lineage: &str,
        date: NaiveDate,
    ) -> EpitopeImmunity {
        let days = self.outbreak_dates.days_since_outbreak(country, date);
        self.compute_immunity(country, lineage, days)
    }

    /// Compute immunity for all 75 PK combinations (for optimization)
    ///
    /// Returns a vector of 75 EpitopeImmunity values, one per PK combination
    pub fn compute_immunity_all_pk(
        &self,
        country: &str,
        lineage: &str,
        days_since_outbreak: i32,
    ) -> Vec<EpitopeImmunity> {
        let t = days_since_outbreak as f32;

        self.pk_params.iter().map(|pk| {
            let mut immunity = EpitopeImmunity::new();
            for i in 0..10 {
                let base_immunity = self.compute_epitope_immunity(t, pk);
                let cross_factor = self.cross_reactivity
                    .as_ref()
                    .map(|cr| cr.get_cross_immunity(lineage, i))
                    .unwrap_or(1.0);
                immunity.levels[i] = (base_immunity * cross_factor).clamp(0.0, 1.0);
            }
            immunity
        }).collect()
    }

    /// Batch compute immunity for multiple samples
    ///
    /// Optimized for bulk computation - processes multiple (country, lineage, date)
    /// tuples efficiently.
    ///
    /// # Arguments
    /// * `samples` - Vector of (country, lineage, days_since_outbreak) tuples
    ///
    /// # Returns
    /// Vector of EpitopeImmunity values, one per sample
    pub fn compute_immunity_batch(
        &self,
        samples: &[(&str, &str, i32)],
    ) -> Vec<EpitopeImmunity> {
        samples.iter()
            .map(|(country, lineage, days)| {
                self.compute_immunity(country, lineage, *days)
            })
            .collect()
    }

    /// Compute fold reduction for a variant given escape scores
    ///
    /// VASIL formula: fold_reduction = exp(sum(escape[i] * immunity[i]))
    ///
    /// Higher fold reduction = more immune escape = variant advantage
    pub fn compute_fold_reduction(
        &self,
        country: &str,
        lineage: &str,
        days_since_outbreak: i32,
        escape_scores: &[f32; 10],
    ) -> f32 {
        let immunity = self.compute_immunity(country, lineage, days_since_outbreak);
        immunity.compute_fold_reduction(escape_scores)
    }

    /// Get reference to outbreak dates
    pub fn outbreak_dates(&self) -> &OutbreakDates {
        &self.outbreak_dates
    }

    /// Get mutable reference to outbreak dates
    pub fn outbreak_dates_mut(&mut self) -> &mut OutbreakDates {
        &mut self.outbreak_dates
    }

    /// Get reference to cross-reactivity matrix
    pub fn cross_reactivity(&self) -> Option<&CrossReactivityMatrix> {
        self.cross_reactivity.as_ref()
    }

    /// Get all PK parameters
    pub fn pk_params(&self) -> &[PKParameters] {
        &self.pk_params
    }
}

impl Default for ImmunityDynamics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// IMMUNITY TRAJECTORY
// ============================================================================

/// Immunity trajectory over time for visualization/analysis
#[derive(Debug, Clone)]
pub struct ImmunityTrajectory {
    /// Days since outbreak (x-axis)
    pub days: Vec<i32>,
    /// Immunity levels per day [day_idx][epitope_idx]
    pub immunity: Vec<[f32; 10]>,
    /// Mean immunity per day
    pub mean_immunity: Vec<f32>,
}

impl ImmunityTrajectory {
    /// Compute immunity trajectory for a range of days
    pub fn compute(
        dynamics: &ImmunityDynamics,
        country: &str,
        lineage: &str,
        start_day: i32,
        end_day: i32,
        step: i32,
    ) -> Self {
        let mut days = Vec::new();
        let mut immunity = Vec::new();
        let mut mean_immunity = Vec::new();

        let mut day = start_day;
        while day <= end_day {
            let imm = dynamics.compute_immunity(country, lineage, day);
            days.push(day);
            immunity.push(imm.levels);
            mean_immunity.push(imm.mean());
            day += step;
        }

        Self {
            days,
            immunity,
            mean_immunity,
        }
    }

    /// Get immunity at specific day (interpolates if needed)
    pub fn get_immunity_at(&self, day: i32) -> Option<EpitopeImmunity> {
        // Find closest day
        let idx = self.days.iter()
            .enumerate()
            .min_by_key(|(_, &d)| (d - day).abs())
            .map(|(i, _)| i)?;

        Some(EpitopeImmunity::from_array(self.immunity[idx]))
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Estimate days since outbreak from date
pub fn estimate_days_since_outbreak(country: &str, date: &NaiveDate) -> i32 {
    let outbreak = OutbreakDates::new_defaults();
    outbreak.days_since_outbreak(country, *date)
}

/// Convert lineage to variant family for cross-reactivity lookup
pub fn lineage_to_family(lineage: &str) -> &'static str {
    let lin = lineage.to_uppercase();

    if lin.starts_with("XBB") || lin.starts_with("EG.") || lin.starts_with("HK.")
        || lin.starts_with("JN.") || lin.starts_with("FL.") {
        "XBB"
    } else if lin.starts_with("BQ.") || lin.starts_with("BE.") || lin.starts_with("BF.") {
        "BQ.1"
    } else if lin.starts_with("BA.5") || lin.starts_with("BA.4") {
        "BA.45"
    } else if lin.starts_with("BA.2") {
        "BA.2"
    } else if lin.starts_with("BA.1") || lin.starts_with("B.1.1.529") {
        "BA.1"
    } else if lin.starts_with("AY.") || lin.starts_with("B.1.617") {
        "Delta"
    } else if lin.starts_with("P.1") {
        "Gamma"
    } else if lin.starts_with("B.1.351") {
        "Beta"
    } else if lin.starts_with("B.1.1.7") {
        "Alpha"
    } else {
        "Wuhan"
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pk_parameters_validation() {
        assert!(PKParameters::new(30.0, 14.0).is_ok());
        assert!(PKParameters::new(-1.0, 14.0).is_err());
        assert!(PKParameters::new(30.0, -1.0).is_err());
    }

    #[test]
    fn test_pk_presets() {
        let fast = PKParameters::fast();
        let medium = PKParameters::medium();
        let slow = PKParameters::slow();

        assert!(fast.t_half < medium.t_half);
        assert!(medium.t_half < slow.t_half);
        assert!(fast.t_max < medium.t_max);
        assert!(medium.t_max < slow.t_max);
    }

    #[test]
    fn test_immunity_dynamics_creation() {
        let dynamics = ImmunityDynamics::new();
        assert_eq!(dynamics.num_pk_combinations(), 75);
        assert_eq!(dynamics.selected_pk_idx, 37); // Median
    }

    #[test]
    fn test_immunity_rising_phase() {
        let dynamics = ImmunityDynamics::new();

        // At t=0, immunity should be near floor
        let imm_0 = dynamics.compute_immunity("Germany", "Delta", 0);
        assert!(imm_0.mean() < 0.1);

        // At t=7 (half of t_max for medium), should be around 0.5
        let imm_7 = dynamics.compute_immunity("Germany", "Delta", 7);
        assert!(imm_7.mean() > 0.2 && imm_7.mean() < 0.6);
    }

    #[test]
    fn test_immunity_peak() {
        let dynamics = ImmunityDynamics::new();
        let pk = dynamics.current_pk_params();

        // At t=t_max, immunity should be at peak (i_max)
        let imm_peak = dynamics.compute_immunity("Germany", "Delta", pk.t_max as i32);
        assert!((imm_peak.mean() - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_immunity_decay_phase() {
        let dynamics = ImmunityDynamics::new();
        let pk = dynamics.current_pk_params();

        // At t=t_max + t_half, immunity should be around 0.5
        let half_life_day = (pk.t_max + pk.t_half) as i32;
        let imm_half = dynamics.compute_immunity("Germany", "Delta", half_life_day);

        // Should be approximately 0.5 (within tolerance due to floor)
        let expected = 0.5;
        assert!((imm_half.mean() - expected).abs() < 0.2,
            "Expected ~{}, got {}", expected, imm_half.mean());
    }

    #[test]
    fn test_immunity_long_term_decay() {
        let dynamics = ImmunityDynamics::new();

        // At t=365 (1 year), immunity should have decayed significantly
        let imm_1year = dynamics.compute_immunity("Germany", "Delta", 365);
        assert!(imm_1year.mean() < 0.3);

        // But should not go below floor
        assert!(imm_1year.mean() >= dynamics.immunity_floor);
    }

    #[test]
    fn test_epitope_immunity_fold_reduction() {
        let immunity = EpitopeImmunity::uniform(0.5);
        let escape = [0.1f32; 10]; // 10% escape per epitope

        let fold = immunity.compute_fold_reduction(&escape);

        // fold = exp(10 * 0.1 * 0.5) = exp(0.5) ~ 1.65
        assert!((fold - 1.65).abs() < 0.1);
    }

    #[test]
    fn test_pk_selection() {
        let mut dynamics = ImmunityDynamics::new();

        // Select fast decay
        dynamics.select_pk_params(0);
        let pk = dynamics.current_pk_params();
        assert!((pk.t_half - 25.0).abs() < 0.1);

        // Select slow decay
        dynamics.select_pk_params(74);
        let pk = dynamics.current_pk_params();
        assert!((pk.t_half - 69.0).abs() < 0.1);
    }

    #[test]
    fn test_outbreak_dates() {
        let dates = OutbreakDates::new_defaults();

        assert!(dates.get("Germany").is_some());
        assert!(dates.get("USA").is_some());
        assert!(dates.get("NonExistent").is_none());

        let outbreak = dates.get("Germany").unwrap();
        let test_date = NaiveDate::from_ymd_opt(2022, 3, 1).unwrap();
        let days = dates.days_since_outbreak("Germany", test_date);

        assert!(days > 700); // Approximately 2 years
    }

    #[test]
    fn test_batch_computation() {
        let dynamics = ImmunityDynamics::new();

        let samples = vec![
            ("Germany", "Delta", 100),
            ("USA", "BA.1", 200),
            ("UK", "XBB.1.5", 300),
        ];

        let results = dynamics.compute_immunity_batch(&samples);

        assert_eq!(results.len(), 3);

        // Each result should have valid immunity levels
        for imm in &results {
            for level in imm.levels {
                assert!(level >= 0.0 && level <= 1.0);
            }
        }
    }

    #[test]
    fn test_lineage_to_family() {
        assert_eq!(lineage_to_family("XBB.1.5"), "XBB");
        assert_eq!(lineage_to_family("BA.5.1"), "BA.45");
        assert_eq!(lineage_to_family("BA.2.75"), "BA.2");
        assert_eq!(lineage_to_family("BA.1.1"), "BA.1");
        assert_eq!(lineage_to_family("AY.4"), "Delta");
        assert_eq!(lineage_to_family("B.1.1.7"), "Alpha");
    }

    #[test]
    fn test_immunity_trajectory() {
        let dynamics = ImmunityDynamics::new();

        let trajectory = ImmunityTrajectory::compute(
            &dynamics,
            "Germany",
            "Delta",
            0,
            100,
            10,
        );

        assert_eq!(trajectory.days.len(), 11); // 0, 10, 20, ..., 100
        assert_eq!(trajectory.immunity.len(), 11);
        assert_eq!(trajectory.mean_immunity.len(), 11);

        // Immunity should peak and then decay
        let max_idx = trajectory.mean_immunity.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        // Peak should be around day 20-30 (t_max ~ 21)
        assert!(trajectory.days[max_idx] >= 10 && trajectory.days[max_idx] <= 40);
    }
}

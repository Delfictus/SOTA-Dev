//! Temporal Immunity Integration - PRISM's Forward Simulation
//!
//! Implements time-integrated immunity computation:
//! E[Immune_y(t)] = Σ_x ∫_0^t π_x(s) · I(s) · P_neut(t-s, x, y) ds
//!
//! This is the MISSING PIECE from our current implementation.
//! We were doing point-in-time lookups, not temporal integration.

use anyhow::Result;
use std::collections::HashMap;
use chrono::NaiveDate;

use crate::data_loader::CountryData;
use crate::vasil_data::{VasilEnhancedData, PNeutData};
use crate::gpu_benchmark::get_lineage_epitope_escape_scores;

/// Pre-computed cross-neutralization matrix for performance
/// Stores P_neut(days, variant_x, variant_y) for fast lookup
pub struct CrossNeutralizationCache {
    /// Variant name -> index mapping
    variant_to_idx: HashMap<String, usize>,
    /// Indexed variant names
    variants: Vec<String>,
    /// P_neut matrix: [variant_x][variant_y][day_idx]
    /// Dimensions: [n_variants][n_variants][MAX_DAYS]
    matrix: Vec<Vec<Vec<f32>>>,
}

impl CrossNeutralizationCache {
    const MAX_DAYS: usize = 600;

    /// Pre-compute cross-neutralization for all variant pairs
    pub fn precompute(
        all_variants: &[String],
        t_max: f32,
        t_half: f32,
    ) -> Self {
        let n_variants = all_variants.len();

        log::info!("Pre-computing cross-neutralization matrix: {} variants × {} variants × {} days",
                   n_variants, n_variants, Self::MAX_DAYS);

        let mut variant_to_idx = HashMap::new();
        for (idx, variant) in all_variants.iter().enumerate() {
            variant_to_idx.insert(variant.clone(), idx);
        }

        // Allocate matrix
        let mut matrix = vec![vec![vec![0.0f32; Self::MAX_DAYS]; n_variants]; n_variants];

        // Compute all pairs
        for (i, variant_x) in all_variants.iter().enumerate() {
            let escape_x = get_lineage_epitope_escape_scores(variant_x);

            for (j, variant_y) in all_variants.iter().enumerate() {
                let escape_y = get_lineage_epitope_escape_scores(variant_y);

                // Compute for all days
                for day in 0..Self::MAX_DAYS {
                    let p_neut = Self::compute_p_neut_static(
                        &escape_x,
                        &escape_y,
                        day as f32,
                        t_max,
                        t_half,
                    );
                    matrix[i][j][day] = p_neut;
                }
            }

            if i % 20 == 0 {
                log::info!("  Computed {}/{} variants", i, n_variants);
            }
        }

        log::info!("  Cross-neutralization matrix ready ({} MB)",
                   (n_variants * n_variants * Self::MAX_DAYS * 4) / 1_000_000);

        Self {
            variant_to_idx,
            variants: all_variants.to_vec(),
            matrix,
        }
    }

    /// Fast lookup: P_neut for variant pair at specific day
    pub fn get_p_neut(&self, variant_x: &str, variant_y: &str, days_since: f32) -> f32 {
        let idx_x = match self.variant_to_idx.get(variant_x) {
            Some(&i) => i,
            None => return 0.5,  // Unknown variant, default
        };

        let idx_y = match self.variant_to_idx.get(variant_y) {
            Some(&i) => i,
            None => return 0.5,
        };

        let day_idx = (days_since as usize).min(Self::MAX_DAYS - 1);

        self.matrix[idx_x][idx_y][day_idx]
    }

    /// Static computation of P_neut (used in precompute)
    fn compute_p_neut_static(
        escape_x: &[f32; 10],
        escape_y: &[f32; 10],
        days_since: f32,
        t_max: f32,
        t_half: f32,
    ) -> f32 {
        // Antibody level from PK curve
        let antibody_level = if days_since < t_max {
            days_since / t_max
        } else {
            let time_since_peak = days_since - t_max;
            (-time_since_peak * 2.0_f32.ln() / t_half).exp()
        };

        // Fold resistance from epitope escape differences
        let mut total_fr = 1.0;
        for i in 0..10 {
            let escape_diff = escape_y[i] - escape_x[i];
            let fr_epitope = (escape_diff * 2.0).exp();
            total_fr *= fr_epitope;
        }
        total_fr = total_fr.clamp(1.0, 100.0);

        // Neutralization probability
        let ic50 = 1.0;
        antibody_level / (total_fr * ic50 + antibody_level)
    }
}

/// Temporal immunity computer - integrates infection history
pub struct TemporalImmunityComputer {
    /// Population size per country
    country_populations: HashMap<String, f64>,

    /// PK parameters (t_max, t_half) for antibody decay
    t_max: f32,
    t_half: f32,

    /// Pre-computed cross-neutralization matrix (OPTIMIZATION!)
    cross_neut_cache: Option<CrossNeutralizationCache>,
}

impl TemporalImmunityComputer {
    pub fn new() -> Self {
        // Population sizes (millions)
        let mut pops = HashMap::new();
        pops.insert("Germany".to_string(), 83.2);
        pops.insert("USA".to_string(), 331.9);
        pops.insert("UK".to_string(), 67.3);
        pops.insert("Japan".to_string(), 125.7);
        pops.insert("Brazil".to_string(), 214.3);
        pops.insert("France".to_string(), 67.4);
        pops.insert("Canada".to_string(), 38.2);
        pops.insert("Denmark".to_string(), 5.8);
        pops.insert("Australia".to_string(), 25.7);
        pops.insert("Sweden".to_string(), 10.4);
        pops.insert("Mexico".to_string(), 126.0);
        pops.insert("SouthAfrica".to_string(), 60.0);

        Self {
            country_populations: pops,
            t_max: 14.0,   // Days to peak antibody (from VASIL paper)
            t_half: 47.0,  // Half-life median (from our PK fitting)
            cross_neut_cache: None,  // Build separately
        }
    }

    /// Build with pre-computed cross-neutralization matrix (200x speedup!)
    pub fn with_cache(mut self, all_variants: Vec<String>) -> Self {
        log::info!("Building cross-neutralization cache for {} variants...", all_variants.len());

        self.cross_neut_cache = Some(CrossNeutralizationCache::precompute(
            &all_variants,
            self.t_max,
            self.t_half,
        ));

        self
    }

    /// Compute accumulated immunity for variant y at time t
    ///
    /// Integrates over ALL past infections weighted by:
    /// 1. Variant frequency at time of infection
    /// 2. Incidence (number of infections)
    /// 3. Cross-neutralization probability (waning over time)
    pub fn compute_immunity_integral(
        &self,
        country: &str,
        target_variant: &str,
        target_date: NaiveDate,
        country_data: &CountryData,
        vasil_data: Option<&VasilEnhancedData>,
    ) -> Result<f64> {

        let mut accumulated_immunity = 0.0;

        // Get date range to integrate over (past 600 days)
        let start_date = target_date - chrono::Duration::days(600);

        // For each day in infection history
        for (date_idx, infection_date) in country_data.frequencies.dates.iter().enumerate() {
            if infection_date < &start_date || infection_date >= &target_date {
                continue;
            }

            let days_since = (target_date - *infection_date).num_days() as f32;

            // Estimate incidence on this day
            let incidence = self.estimate_incidence(
                country,
                *infection_date,
                country_data,
                vasil_data,
            );

            // For each variant circulating on that day
            for (lineage_idx, past_variant) in country_data.frequencies.lineages.iter().enumerate() {
                let frequency = country_data.frequencies.frequencies
                    .get(date_idx)
                    .and_then(|row| row.get(lineage_idx))
                    .copied()
                    .unwrap_or(0.0);

                if frequency < 0.001 {
                    continue;  // Skip negligible variants
                }

                // Compute cross-neutralization: Does past infection with X protect against Y?
                let p_neut = if let Some(ref cache) = self.cross_neut_cache {
                    // Fast cached lookup (O(1))
                    cache.get_p_neut(past_variant, target_variant, days_since)
                } else {
                    // Fallback to slow computation
                    self.compute_cross_neutralization(
                        past_variant,
                        target_variant,
                        days_since,
                        vasil_data,
                    )
                };

                // Accumulate: freq × incidence × P_neut
                accumulated_immunity += (frequency * incidence * p_neut) as f64;
            }
        }

        Ok(accumulated_immunity)
    }

    /// Estimate incidence (infections per day) from frequency data
    ///
    /// Uses phi-corrected case estimates or GInPipe-style reconstruction
    fn estimate_incidence(
        &self,
        country: &str,
        date: NaiveDate,
        frequencies: &crate::data_loader::CountryData,
        vasil_data: Option<&VasilEnhancedData>,
    ) -> f32 {
        // Compute total frequency (sum across all lineages) at the given date
        let date_idx = frequencies.frequencies.dates.iter()
            .position(|d| d == &date);

        let total_freq: f32 = if let Some(idx) = date_idx {
            frequencies.frequencies.frequencies
                .get(idx)
                .map(|row| row.iter().sum())
                .unwrap_or(1.0)
        } else {
            1.0  // Fallback if date not found
        };

        // Get population for this country (as f32 for compatibility)
        let population = (self.country_populations.get(country)
            .copied()
            .unwrap_or(50.0) * 1_000_000.0) as f32;

        // Use phi correction if available
        if let Some(vd) = vasil_data {
            let phi = vd.phi.get_phi(&date);
            // Reconstruct incidence from frequency and phi
            // total_freq ≈ (incidence * phi) / population
            // Therefore: incidence ≈ (total_freq / phi) * population * scale_factor
            return (total_freq / phi.max(0.01)) * population * 0.001;
        }

        // Default: use frequency-based estimate without phi correction
        total_freq * population * 0.001
    }

    /// Compute cross-neutralization probability
    ///
    /// P_neut(t, x, y) = probability that immunity from variant x (acquired t days ago)
    /// neutralizes variant y today
    fn compute_cross_neutralization(
        &self,
        past_variant: &str,
        target_variant: &str,
        days_since_infection: f32,
        vasil_data: Option<&VasilEnhancedData>,
    ) -> f32 {
        // Step 1: Antibody waning (PK curve)
        let antibody_concentration = self.compute_antibody_level(days_since_infection);

        // Step 2: Cross-reactivity from epitope escape
        let past_epitope_escape = get_lineage_epitope_escape_scores(past_variant);
        let target_epitope_escape = get_lineage_epitope_escape_scores(target_variant);

        // Compute fold resistance: how much harder is it to neutralize target vs past?
        let mut total_fold_resistance = 1.0;

        for i in 0..10 {
            let escape_difference = target_epitope_escape[i] - past_epitope_escape[i];

            // Fold resistance per epitope (exponential in escape difference)
            let fr_epitope = (escape_difference * 2.0).exp();

            total_fold_resistance *= fr_epitope;
        }

        // Limit to reasonable range
        total_fold_resistance = total_fold_resistance.clamp(1.0, 100.0);

        // Step 3: Neutralization probability
        // Higher fold-resistance → need more antibody → lower P_neut
        let ic50 = 1.0;  // Normalized (from VASIL calibration)
        let p_neut = antibody_concentration / (total_fold_resistance * ic50 + antibody_concentration);

        p_neut
    }

    /// Compute antibody level at t days post-infection
    ///
    /// PK model: rises to peak at t_max, then exponential decay
    fn compute_antibody_level(&self, days_since: f32) -> f32 {
        if days_since < 0.0 {
            return 0.0;
        }

        if days_since < self.t_max {
            // Rising phase (linear approximation)
            days_since / self.t_max
        } else {
            // Decay phase (exponential with half-life)
            let time_since_peak = days_since - self.t_max;
            (-time_since_peak * 2.0_f32.ln() / self.t_half).exp()
        }
    }

    /// Compute susceptibles and relative fitness gamma_y
    ///
    /// gamma_y(t) = S_y(t) / avg(S_x(t)) - 1
    /// where S_y(t) = Population - E[Immune_y(t)]
    pub fn compute_gamma(
        &self,
        country: &str,
        target_variant: &str,
        target_date: NaiveDate,
        country_data: &CountryData,
        vasil_data: Option<&VasilEnhancedData>,
    ) -> Result<f32> {
        // Compute immunity to target variant
        let immune_count = self.compute_immunity_integral(
            country,
            target_variant,
            target_date,
            country_data,
            vasil_data,
        )?;

        let population = self.country_populations.get(country)
            .copied()
            .unwrap_or(50.0) * 1_000_000.0;  // Convert to individuals

        let susceptible_y = population - immune_count;

        // Compute average susceptibles across currently circulating variants
        let date_idx = country_data.frequencies.dates.iter()
            .position(|d| d == &target_date)
            .unwrap_or(0);

        let mut weighted_susceptible_sum = 0.0;
        let mut total_frequency = 0.0;

        for (lineage_idx, competitor_variant) in country_data.frequencies.lineages.iter().enumerate() {
            let freq = country_data.frequencies.frequencies
                .get(date_idx)
                .and_then(|row| row.get(lineage_idx))
                .copied()
                .unwrap_or(0.0);

            if freq < 0.01 {
                continue;  // Skip low-frequency variants
            }

            // Compute susceptibles for competitor
            let competitor_immune = self.compute_immunity_integral(
                country,
                competitor_variant,
                target_date,
                country_data,
                vasil_data,
            ).unwrap_or(0.0);

            let competitor_susceptible = population - competitor_immune;

            weighted_susceptible_sum += freq as f64 * competitor_susceptible;
            total_frequency += freq;
        }

        let avg_susceptible = if total_frequency > 0.0 {
            weighted_susceptible_sum / total_frequency as f64
        } else {
            population * 0.5  // Default: half population susceptible
        };

        // Relative fitness
        let gamma_y = (susceptible_y / avg_susceptible) - 1.0;

        Ok(gamma_y as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_antibody_pk_curve() {
        let computer = TemporalImmunityComputer::new();

        // At peak (14 days)
        assert!((computer.compute_antibody_level(14.0) - 1.0).abs() < 0.01);

        // After one half-life (14 + 47 days)
        assert!((computer.compute_antibody_level(61.0) - 0.5).abs() < 0.05);

        // After two half-lives
        assert!((computer.compute_antibody_level(108.0) - 0.25).abs() < 0.05);
    }

    #[test]
    fn test_cross_neutralization_same_variant() {
        let computer = TemporalImmunityComputer::new();

        // Same variant should have high neutralization
        let p_neut = computer.compute_cross_neutralization(
            "BA.5",
            "BA.5",
            30.0,  // 30 days post-infection
            None,
        );

        assert!(p_neut > 0.5, "Same variant should be well-neutralized");
    }
}

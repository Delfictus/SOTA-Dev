//! VASIL Enhanced Data Loaders
//!
//! Loads additional VASIL data files for improved prediction accuracy:
//! 1. smoothed_phi_estimates - Case ascertainment rate over time
//! 2. P_neut - Neutralization probability vs variants
//! 3. Immunological_Landscape - Population immunity levels
//! 4. PK_for_all_Epitopes - Epitope-specific immunity decay

use anyhow::{Result, Context, bail};
use std::path::Path;
use std::collections::HashMap;
use csv::ReaderBuilder;
use chrono::NaiveDate;

#[allow(unused_imports)]
use log::{info, debug, warn};

/// Case ascertainment rate (phi) over time
/// Higher phi = more comprehensive surveillance = more reliable frequency data
#[derive(Debug, Clone)]
pub struct PhiEstimates {
    pub country: String,
    pub dates: Vec<NaiveDate>,
    pub phi_values: Vec<f32>,  // Smoothed phi estimates
    date_to_phi: HashMap<NaiveDate, f32>,
}

impl PhiEstimates {
    /// Load from VASIL smoothed_phi_estimates_{Country}.csv
    /// Tries multiple naming conventions as they vary by country
    pub fn load_from_vasil(vasil_data_dir: &Path, country: &str) -> Result<Self> {
        let country_dir = vasil_data_dir.join("ByCountry").join(country);

        // Try multiple naming conventions (varies by country in VASIL data)
        let naming_patterns = [
            format!("smoothed_phi_estimates_{}.csv", country),
            format!("smoothed_phi_estimates_gisaid_{}_vasil.csv", country),
            format!("smoothed_phi_estimates_gisaid_{}.csv", country),
            format!("smoothed_phi_{}.csv", country),
        ];

        let mut phi_file = None;
        for pattern in &naming_patterns {
            let path = country_dir.join(pattern);
            if path.exists() {
                phi_file = Some(path);
                break;
            }
        }

        let phi_file = phi_file.ok_or_else(|| {
            anyhow::anyhow!("Phi estimates file not found for {} (tried {} patterns)",
                country, naming_patterns.len())
        })?;

        log::info!("Loading phi estimates from: {:?}", phi_file);

        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(&phi_file)
            .context("Failed to open phi file")?;

        let mut dates = Vec::new();
        let mut phi_values = Vec::new();
        let mut date_to_phi = HashMap::new();

        // CSV format: t, date, smoothed_phi
        for result in reader.records() {
            let record = result?;

            let date_str = record.get(1).context("Missing date")?;
            let phi: f32 = record.get(2).context("Missing phi")?.parse()?;

            let date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
                .context(format!("Invalid date: {}", date_str))?;

            dates.push(date);
            phi_values.push(phi);
            date_to_phi.insert(date, phi);
        }

        log::info!("Loaded {} phi estimates for {}, range {:.1} to {:.1}",
            dates.len(), country,
            phi_values.iter().cloned().fold(f32::INFINITY, f32::min),
            phi_values.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        Ok(PhiEstimates {
            country: country.to_string(),
            dates,
            phi_values,
            date_to_phi,
        })
    }

    /// Get phi for a specific date (interpolates if exact date not found)
    pub fn get_phi(&self, date: &NaiveDate) -> f32 {
        if let Some(&phi) = self.date_to_phi.get(date) {
            return phi;
        }

        // Find nearest date
        let mut min_diff = i64::MAX;
        let mut nearest_phi = self.phi_values.first().copied().unwrap_or(1.0);

        for (d, &phi) in self.dates.iter().zip(self.phi_values.iter()) {
            let diff = (*date - *d).num_days().abs();
            if diff < min_diff {
                min_diff = diff;
                nearest_phi = phi;
            }
        }

        nearest_phi
    }

    /// Normalize frequency by phi (accounts for testing variations)
    /// VASIL methodology: freq_adjusted = freq / phi
    pub fn normalize_frequency(&self, freq: f32, date: &NaiveDate) -> f32 {
        let phi = self.get_phi(date);
        if phi > 0.0 {
            freq / phi * 100.0  // Scale factor for numerical stability
        } else {
            freq
        }
    }
}

/// Neutralization probability data
/// P_neut indicates probability of neutralization vs specific variant
#[derive(Debug, Clone)]
pub struct PNeutData {
    pub variant: String,  // e.g., "Delta", "Omicron_BA.1"
    pub days_since_infection: Vec<i32>,
    pub p_neut_min: Vec<f32>,
    pub p_neut_max: Vec<f32>,
}

impl PNeutData {
    /// Load from VASIL P_neut_{variant}.csv
    pub fn load_from_vasil(
        vasil_data_dir: &Path,
        country: &str,
        variant: &str,
    ) -> Result<Self> {
        let p_neut_file = vasil_data_dir
            .join("ByCountry")
            .join(country)
            .join("results")
            .join("Immunological_Landscape_groups")
            .join(format!("P_neut_{}.csv", variant));

        if !p_neut_file.exists() {
            bail!("P_neut file not found: {:?}", p_neut_file);
        }

        log::info!("Loading P_neut data from: {:?}", p_neut_file);

        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(&p_neut_file)
            .context("Failed to open P_neut file")?;

        let mut days_since_infection = Vec::new();
        let mut p_neut_min = Vec::new();
        let mut p_neut_max = Vec::new();

        // CSV format: index, Day since infection, Proba Neut Min, Proba Neut Max
        for result in reader.records() {
            let record = result?;

            let day: i32 = record.get(1).context("Missing day")?.parse()?;
            let min_str = record.get(2).context("Missing p_neut_min")?;
            let max_str = record.get(3).context("Missing p_neut_max")?;

            let min: f32 = min_str.parse().unwrap_or(0.0);
            let max: f32 = max_str.parse().unwrap_or(1.0);

            days_since_infection.push(day);
            p_neut_min.push(min);
            p_neut_max.push(max);
        }

        log::info!("Loaded {} P_neut entries for {} ({}), range [{:.3}, {:.3}]",
            days_since_infection.len(), country, variant,
            p_neut_min.iter().cloned().fold(f32::INFINITY, f32::min),
            p_neut_max.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

        Ok(PNeutData {
            variant: variant.to_string(),
            days_since_infection,
            p_neut_min,
            p_neut_max,
        })
    }

    /// Get neutralization probability for days since infection
    pub fn get_p_neut(&self, days: i32) -> (f32, f32) {
        // Find index for this day
        let idx = self.days_since_infection.iter()
            .position(|&d| d == days)
            .unwrap_or_else(|| {
                // Find nearest day
                self.days_since_infection.iter()
                    .enumerate()
                    .min_by_key(|(_, &d)| (d - days).abs())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            });

        (
            self.p_neut_min.get(idx).copied().unwrap_or(0.0),
            self.p_neut_max.get(idx).copied().unwrap_or(1.0),
        )
    }

    /// Compute immune escape score (1 - P_neut)
    /// Higher escape = variant evades immunity better
    pub fn compute_escape(&self, days: i32) -> f32 {
        let (min, max) = self.get_p_neut(days);
        let mean_p_neut = (min + max) / 2.0;
        1.0 - mean_p_neut
    }
}

/// Immunological landscape data for population immunity
/// Contains ACTUAL time series from VASIL SEIR simulations (655 days × 75 PK combos)
#[derive(Debug, Clone)]
pub struct ImmunologicalLandscape {
    pub country: String,
    pub variant: String,
    pub dates: Vec<NaiveDate>,  // 655 days from 2021-07-01 to 2023-04-16
    pub immunity_by_pk: Vec<Vec<f32>>,  // [75 PK combos][655 days] of population immunity
    pub pk_headers: Vec<String>,  // PK parameter descriptions
    date_to_idx: HashMap<NaiveDate, usize>,
}

impl ImmunologicalLandscape {
    /// Load from VASIL Immunized_SpikeGroup_{variant}_all_PK.csv
    /// Contains 655 days × 75 PK combinations of ACTUAL population immunity
    pub fn load_from_vasil(
        vasil_data_dir: &Path,
        country: &str,
        variant: &str,
        population_type: &str,  // "Immunized" or "Susceptible"
    ) -> Result<Self> {
        let landscape_file = vasil_data_dir
            .join("ByCountry")
            .join(country)
            .join("results")
            .join("Immunological_Landscape_groups")
            .join(format!("{}_SpikeGroup_{}_all_PK.csv", population_type, variant));

        if !landscape_file.exists() {
            bail!("Immunological landscape file not found: {:?}", landscape_file);
        }

        log::info!("Loading immunity time series from: {:?}", landscape_file);

        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(&landscape_file)
            .context("Failed to open landscape file")?;

        let headers = reader.headers()?.clone();

        // Extract PK parameter headers (skip index=0, Days=1)
        let pk_headers: Vec<String> = headers.iter()
            .skip(2)
            .map(|s| s.to_string())
            .collect();

        let n_pk_combos = pk_headers.len();
        log::info!("  Found {} PK parameter combinations", n_pk_combos);

        // Initialize storage: [75 PK combos][655 days]
        let mut immunity_by_pk: Vec<Vec<f32>> = vec![Vec::new(); n_pk_combos];
        let mut dates = Vec::new();
        let mut date_to_idx = HashMap::new();

        // Parse each row (one per day)
        for result in reader.records() {
            let record = result?;

            // Column 1 is the date (YYYY-MM-DD format)
            let date_str = record.get(1).context("Missing date column")?;
            let date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
                .context(format!("Invalid date: {}", date_str))?;

            date_to_idx.insert(date, dates.len());
            dates.push(date);

            // Columns 2+ are immunity for each PK combination
            for pk_idx in 0..n_pk_combos {
                let immunity: f32 = record.get(pk_idx + 2)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);

                immunity_by_pk[pk_idx].push(immunity);
            }
        }

        log::info!("  Loaded {} days: {} to {}",
                   dates.len(),
                   dates.first().unwrap(),
                   dates.last().unwrap());
        log::info!("  Immunity range: {:.0} to {:.0}",
                   immunity_by_pk.iter().flatten().cloned().fold(f32::INFINITY, f32::min),
                   immunity_by_pk.iter().flatten().cloned().fold(f32::NEG_INFINITY, f32::max));

        Ok(ImmunologicalLandscape {
            country: country.to_string(),
            variant: variant.to_string(),
            dates,
            immunity_by_pk,
            pk_headers,
            date_to_idx,
        })
    }

    /// Get population immunity at a specific date for a specific PK combination
    pub fn get_immunity_at_date(&self, date: &NaiveDate, pk_idx: usize) -> Option<f32> {
        let date_idx = *self.date_to_idx.get(date)?;
        self.immunity_by_pk.get(pk_idx)?.get(date_idx).copied()
    }

    /// Get immunity for a date (interpolates if exact date not found)
    pub fn get_immunity(&self, date: &NaiveDate, pk_idx: usize) -> f32 {
        // Try exact match
        if let Some(imm) = self.get_immunity_at_date(date, pk_idx) {
            return imm;
        }

        // Find nearest date
        let mut min_diff = i64::MAX;
        let mut nearest_immunity = 0.0;

        for (d, &idx) in &self.date_to_idx {
            let diff = (*date - *d).num_days().abs();
            if diff < min_diff {
                min_diff = diff;
                if let Some(&imm) = self.immunity_by_pk.get(pk_idx).and_then(|v| v.get(idx)) {
                    nearest_immunity = imm;
                }
            }
        }

        nearest_immunity
    }

    /// Get mean immunity across all PK combinations for a date
    pub fn get_mean_immunity_at_date(&self, date: &NaiveDate) -> f32 {
        let date_idx = match self.date_to_idx.get(date) {
            Some(&idx) => idx,
            None => return 0.0,
        };

        let mut sum = 0.0;
        let mut count = 0;

        for pk_immunity in &self.immunity_by_pk {
            if let Some(&imm) = pk_immunity.get(date_idx) {
                sum += imm;
                count += 1;
            }
        }

        if count > 0 { sum / count as f32 } else { 0.0 }
    }
}

/// Epitope-specific PK (pharmacokinetic) parameters
#[derive(Debug, Clone)]
pub struct EpitopePK {
    pub days: Vec<i32>,
    pub epitope_immunity: HashMap<String, Vec<f32>>,  // Epitope -> immunity over time
}

impl EpitopePK {
    /// Load from VASIL PK_for_all_Epitopes.csv
    pub fn load_from_vasil(vasil_data_dir: &Path, country: &str) -> Result<Self> {
        let pk_file = vasil_data_dir
            .join("ByCountry")
            .join(country)
            .join("results")
            .join("PK_for_all_Epitopes.csv");

        if !pk_file.exists() {
            bail!("PK epitopes file not found: {:?}", pk_file);
        }

        log::info!("Loading epitope PK data from: {:?}", pk_file);

        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(&pk_file)
            .context("Failed to open PK file")?;

        let headers = reader.headers()?.clone();

        // Extract epitope column names
        let epitope_columns: Vec<String> = headers.iter()
            .skip(2)  // Skip index and Day columns
            .map(|s| s.to_string())
            .collect();

        let mut days = Vec::new();
        let mut epitope_immunity: HashMap<String, Vec<f32>> = HashMap::new();

        for col in &epitope_columns {
            epitope_immunity.insert(col.clone(), Vec::new());
        }

        for result in reader.records() {
            let record = result?;

            let day: i32 = record.get(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            days.push(day);

            for (col_idx, col_name) in epitope_columns.iter().enumerate() {
                let value: f32 = record.get(col_idx + 2)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0);

                epitope_immunity.get_mut(col_name).unwrap().push(value);
            }
        }

        log::info!("Loaded {} days of epitope PK data with {} epitopes",
            days.len(), epitope_columns.len());

        Ok(EpitopePK {
            days,
            epitope_immunity,
        })
    }

    /// Get epitope-specific immunity score at a given day
    pub fn get_epitope_immunity(&self, epitope: &str, day: i32) -> f32 {
        let day_idx = self.days.iter()
            .position(|&d| d == day)
            .unwrap_or_else(|| {
                self.days.iter()
                    .enumerate()
                    .min_by_key(|(_, &d)| (d - day).abs())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            });

        // Try exact match or partial match
        for (name, values) in &self.epitope_immunity {
            if name.contains(epitope) {
                if let Some(&v) = values.get(day_idx) {
                    return v;
                }
            }
        }

        0.0
    }

    /// Compute mean epitope immunity across all epitopes
    pub fn get_mean_epitope_immunity(&self, day: i32) -> f32 {
        let day_idx = self.days.iter()
            .position(|&d| d == day)
            .unwrap_or_else(|| {
                self.days.iter()
                    .enumerate()
                    .min_by_key(|(_, &d)| (d - day).abs())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            });

        let mut sum = 0.0f32;
        let mut count = 0;

        for values in self.epitope_immunity.values() {
            if let Some(&v) = values.get(day_idx) {
                sum += v;
                count += 1;
            }
        }

        if count > 0 { sum / count as f32 } else { 0.0 }
    }
}

/// Complete VASIL enhanced data for a country
#[derive(Debug, Clone)]
pub struct VasilEnhancedData {
    pub country: String,
    pub phi: PhiEstimates,
    pub p_neut_delta: Option<PNeutData>,
    pub p_neut_omicron: Option<PNeutData>,
    pub landscape_immunized: Option<ImmunologicalLandscape>,
    pub landscape_susceptible: Option<ImmunologicalLandscape>,
    pub epitope_pk: Option<EpitopePK>,
}

impl VasilEnhancedData {
    /// Load all enhanced VASIL data for a country
    pub fn load_from_vasil(vasil_data_dir: &Path, country: &str) -> Result<Self> {
        log::info!("Loading VASIL enhanced data for {}...", country);

        // Phi is required
        let phi = PhiEstimates::load_from_vasil(vasil_data_dir, country)?;

        // P_neut for variants (optional)
        let p_neut_delta = PNeutData::load_from_vasil(vasil_data_dir, country, "Delta").ok();
        let p_neut_omicron = PNeutData::load_from_vasil(vasil_data_dir, country, "Omicron_BA.1").ok();

        // Immunological landscapes (optional)
        let landscape_immunized = ImmunologicalLandscape::load_from_vasil(
            vasil_data_dir, country, "Delta", "Immunized"
        ).ok();
        let landscape_susceptible = ImmunologicalLandscape::load_from_vasil(
            vasil_data_dir, country, "Delta", "Susceptible"
        ).ok();

        // Epitope PK (optional)
        let epitope_pk = EpitopePK::load_from_vasil(vasil_data_dir, country).ok();

        log::info!("  ✅ Phi: {} dates, P_neut: Delta={}, Omicron={}, Landscapes: {}, Epitope PK: {}",
            phi.dates.len(),
            p_neut_delta.is_some(),
            p_neut_omicron.is_some(),
            landscape_immunized.is_some() || landscape_susceptible.is_some(),
            epitope_pk.is_some());

        Ok(VasilEnhancedData {
            country: country.to_string(),
            phi,
            p_neut_delta,
            p_neut_omicron,
            landscape_immunized,
            landscape_susceptible,
            epitope_pk,
        })
    }

    /// Compute VASIL-style enhanced escape score
    /// Combines phi-normalized frequency with P_neut immune escape
    pub fn compute_enhanced_escape(
        &self,
        frequency: f32,
        date: &NaiveDate,
        variant_type: &str,  // "Delta", "Omicron", etc.
        days_since_outbreak: i32,
    ) -> f32 {
        // 1. Phi-normalized frequency
        let phi_norm_freq = self.phi.normalize_frequency(frequency, date);

        // 2. P_neut-based escape (1 - neutralization probability)
        let p_neut_escape = if variant_type.contains("Omicron") || variant_type.contains("BA.") {
            self.p_neut_omicron.as_ref()
                .map(|p| p.compute_escape(days_since_outbreak))
                .unwrap_or(0.3)  // Default higher escape for Omicron
        } else {
            self.p_neut_delta.as_ref()
                .map(|p| p.compute_escape(days_since_outbreak))
                .unwrap_or(0.2)  // Default escape for Delta
        };

        // 3. Population immunity effect (use date instead of days)
        let pop_immunity = self.landscape_immunized.as_ref()
            .map(|l| l.get_mean_immunity_at_date(date) / 100000.0)  // Normalize large values
            .unwrap_or(0.5);

        // 4. Epitope-specific immunity
        let epitope_immunity = self.epitope_pk.as_ref()
            .map(|e| e.get_mean_epitope_immunity(days_since_outbreak))
            .unwrap_or(0.5);

        // VASIL formula approximation:
        // enhanced_escape = phi_norm_freq × (1 - pop_immunity) × escape_potential
        let escape_potential = p_neut_escape * (1.0 - epitope_immunity * 0.5);

        phi_norm_freq * (1.0 - pop_immunity * 0.3) * escape_potential
    }

    /// Get fitness advantage combining all VASIL signals
    pub fn compute_fitness_advantage(
        &self,
        frequency: f32,
        velocity: f32,
        date: &NaiveDate,
        variant_type: &str,
        days_since_outbreak: i32,
    ) -> f32 {
        let enhanced_escape = self.compute_enhanced_escape(
            frequency, date, variant_type, days_since_outbreak
        );

        // Velocity inversion: high velocity at high frequency = at peak
        let corrected_velocity = if frequency > 0.3 {
            -velocity.abs() * 0.5  // Penalize high velocity when dominant
        } else if frequency < 0.1 && velocity > 0.0 {
            velocity * 1.5  // Reward early growth
        } else {
            velocity * 0.5
        };

        // Combine signals
        // RISE prediction = high escape + positive corrected velocity + room to grow
        let room_to_grow = 1.0 - frequency;
        let fitness = enhanced_escape * 0.4 + corrected_velocity * 0.3 + room_to_grow * 0.3;

        fitness
    }
}

/// Load enhanced VASIL data for all 12 countries
pub fn load_all_countries_enhanced(vasil_data_dir: &Path) -> Result<HashMap<String, VasilEnhancedData>> {
    const COUNTRIES: &[&str] = &[
        "Germany", "USA", "UK", "Japan", "Brazil", "France",
        "Canada", "Denmark", "Australia", "Sweden", "Mexico", "SouthAfrica"
    ];

    let mut all_data = HashMap::new();

    log::info!("Loading VASIL enhanced data for ALL 12 countries...");

    for country in COUNTRIES {
        match VasilEnhancedData::load_from_vasil(vasil_data_dir, country) {
            Ok(data) => {
                all_data.insert(country.to_string(), data);
            }
            Err(e) => {
                log::warn!("Failed to load enhanced data for {}: {}", country, e);
            }
        }
    }

    log::info!("✅ Loaded enhanced VASIL data for {} countries", all_data.len());

    Ok(all_data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_phi_normalization() {
        // Test that phi normalization works correctly
        let phi = PhiEstimates {
            country: "Test".to_string(),
            dates: vec![NaiveDate::from_ymd_opt(2021, 1, 1).unwrap()],
            phi_values: vec![100.0],
            date_to_phi: {
                let mut m = HashMap::new();
                m.insert(NaiveDate::from_ymd_opt(2021, 1, 1).unwrap(), 100.0);
                m
            },
        };

        let date = NaiveDate::from_ymd_opt(2021, 1, 1).unwrap();
        let normalized = phi.normalize_frequency(0.5, &date);

        // freq / phi * 100 = 0.5 / 100 * 100 = 0.5
        assert!((normalized - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_p_neut_escape() {
        let p_neut = PNeutData {
            variant: "Delta".to_string(),
            days_since_infection: vec![0, 30, 60, 90],
            p_neut_min: vec![0.3, 0.7, 0.8, 0.85],
            p_neut_max: vec![0.5, 0.9, 0.95, 0.95],
        };

        // At day 0, escape should be high (1 - mean_p_neut)
        let escape_0 = p_neut.compute_escape(0);
        assert!(escape_0 > 0.5);  // (1 - 0.4) = 0.6

        // At day 90, escape should be low (immunity built up)
        let escape_90 = p_neut.compute_escape(90);
        assert!(escape_90 < 0.2);  // (1 - 0.9) = 0.1
    }
}

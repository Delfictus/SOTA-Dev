//! COMPLETE Data Loaders for VASIL Benchmark (Rust Implementation)
//!
//! Loads all required data from VASIL dataset:
//! 1. GISAID lineage frequencies (all 12 countries)
//! 2. DMS escape matrix (835 antibodies × 179 sites)
//! 3. Lineage spike mutations
//! 4. Population immunity landscapes
//!
//! NO Python proxies - full Rust CSV parsing!

use anyhow::{Result, Context, bail};
use std::path::Path;
use std::collections::{HashMap, HashSet};
use csv::ReaderBuilder;
use chrono::NaiveDate;

#[allow(unused_imports)]
use log::{info, debug, warn};

// ════════════════════════════════════════════════════════════════════════════
// PHASE 3: REAL MUTATION-SPECIFIC DMS ESCAPE
// Replaces hash-based escape with actual VASIL DMS data
// ════════════════════════════════════════════════════════════════════════════

/// Parsed mutation representation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParsedMutation {
    pub site: u32,
    pub from_aa: char,
    pub to_aa: char,
}

impl ParsedMutation {
    /// Parse "G339D" → ParsedMutation { site: 339, from_aa: 'G', to_aa: 'D' }
    pub fn from_str(mutation: &str) -> Option<Self> {
        let m = mutation.trim();
        if m.len() < 3 {
            return None;
        }

        let chars: Vec<char> = m.chars().collect();
        let from_aa = chars[0];
        let to_aa = chars[chars.len() - 1];

        if !from_aa.is_ascii_alphabetic() || !to_aa.is_ascii_alphabetic() {
            return None;
        }

        let site_str: String = chars[1..chars.len()-1].iter().collect();
        let site = site_str.parse::<u32>().ok()?;

        if site < 1 || site > 1300 {
            return None;
        }

        Some(Self { site, from_aa, to_aa })
    }

    /// Check if mutation is in RBD (331-531)
    #[inline]
    pub fn is_rbd(&self) -> bool {
        self.site >= 331 && self.site <= 531
    }

    /// Get RBD offset (0-200)
    #[inline]
    pub fn rbd_offset(&self) -> Option<usize> {
        if self.is_rbd() {
            Some((self.site - 331) as usize)
        } else {
            None
        }
    }
}

/// Parse mutation string (handles both comma and slash separators)
pub fn parse_mutation_string(mutation_str: &str) -> Vec<ParsedMutation> {
    // VASIL uses slash separator in mutation_lists.csv
    let separator = if mutation_str.contains('/') { '/' } else { ',' };
    mutation_str
        .split(separator)
        .filter_map(|s| ParsedMutation::from_str(s.trim()))
        .collect()
}

/// Extract RBD sites from mutations
pub fn get_rbd_sites(mutations: &[ParsedMutation]) -> Vec<u32> {
    mutations.iter()
        .filter(|m| m.is_rbd())
        .map(|m| m.site)
        .collect()
}

/// GISAID lineage frequency data
#[derive(Debug, Clone)]
pub struct GisaidFrequencies {
    pub country: String,
    pub lineages: Vec<String>,
    pub dates: Vec<NaiveDate>,
    pub frequencies: Vec<Vec<f32>>,  // [n_dates × n_lineages]
}

impl GisaidFrequencies {
    /// Load from VASIL Daily_Lineages_Freq_1_percent.csv
    pub fn load_from_vasil(
        vasil_data_dir: &Path,
        country: &str,
    ) -> Result<Self> {
        let freq_file = vasil_data_dir
            .join("ByCountry")
            .join(country)
            .join("results")
            .join("Daily_Lineages_Freq_1_percent.csv");

        if !freq_file.exists() {
            bail!("Frequency file not found: {:?}", freq_file);
        }

        log::info!("Loading GISAID frequencies from: {:?}", freq_file);

        // Parse CSV
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(&freq_file)
            .context("Failed to open frequency file")?;

        // Get headers (lineage names + date column)
        let headers = reader.headers()?.clone();

        // Extract lineage names (skip first column if it's unnamed/date)
        let lineages: Vec<String> = headers.iter()
            .skip(1)
            .filter(|h| !h.starts_with("Unnamed"))
            .filter(|h| h != &"date")
            .map(|s| s.to_string())
            .collect();

        let mut dates = Vec::new();
        let mut freq_matrix = Vec::new();

        // Parse each row
        for result in reader.records() {
            let record = result?;

            // Get date (last column or first column)
            let date_str = if record.get(record.len() - 1).unwrap_or("").contains("-") {
                record.get(record.len() - 1).unwrap()
            } else if record.get(0).unwrap_or("").contains("-") {
                record.get(0).unwrap()
            } else {
                continue;  // Skip if no date found
            };

            let date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
                .context(format!("Invalid date: {}", date_str))?;

            dates.push(date);

            // Parse frequencies for all lineages
            // IMPORTANT: VASIL CSV stores frequencies as percentages (0-100)
            // We normalize to 0-1 by dividing by 100
            let mut freq_row = Vec::new();
            for col_idx in 1..record.len() {
                if let Some(val_str) = record.get(col_idx) {
                    if val_str == date_str {
                        continue;  // Skip date column
                    }
                    let freq: f32 = val_str.parse().unwrap_or(0.0);
                    // Normalize percentage to fraction (21.5% -> 0.215)
                    freq_row.push(freq / 100.0);
                }
            }

            // Only add if we got the right number of lineages
            if freq_row.len() == lineages.len() {
                freq_matrix.push(freq_row);
            }
        }

        log::info!("Loaded {} dates, {} lineages for {}", dates.len(), lineages.len(), country);

        Ok(GisaidFrequencies {
            country: country.to_string(),
            lineages,
            dates,
            frequencies: freq_matrix,
        })
    }

    /// Get frequency for lineage at date
    pub fn get_frequency(&self, lineage: &str, date: &NaiveDate) -> Option<f32> {
        let lineage_idx = self.lineages.iter().position(|l| l == lineage)?;
        let date_idx = self.dates.iter().position(|d| d == date)?;

        self.frequencies.get(date_idx)?.get(lineage_idx).copied()
    }
}

/// Real DMS escape data from VASIL files (PHASE 3)
#[derive(Debug, Clone)]
pub struct DmsEscapeData {
    /// Epitope-averaged escape per site: [epitope_idx][site_offset]
    /// site_offset = site - 331 (RBD: 331-531 → 0-200)
    escape_by_site: Vec<Vec<f32>>,

    /// Raw per-antibody escape: [antibody_idx][site_offset]
    raw_escape: Vec<Vec<f32>>,

    /// Epitope class per antibody
    antibody_epitope: Vec<usize>,

    /// Lineage → mutation string
    lineage_mutations: HashMap<String, String>,

    /// Antibody count
    n_antibodies: usize,

    /// Real data loaded flag
    real_data_loaded: bool,
}

impl DmsEscapeData {
    /// Create empty DmsEscapeData - PRODUCTION: Should never be used (must load real data)
    pub fn empty() -> Self {
        panic!("DmsEscapeData::empty() called - PRODUCTION VIOLATION: Must load real DMS data from VASIL files")
    }

    /// Load from VASIL directory
    pub fn load_from_vasil(vasil_dir: &Path, country: &str) -> Result<Self> {
        let mut result = Self {
            escape_by_site: vec![vec![0.0; 201]; 10],
            raw_escape: Vec::new(),
            antibody_epitope: Vec::new(),
            lineage_mutations: HashMap::new(),
            n_antibodies: 0,
            real_data_loaded: false,
        };

        // Normalize country name for file paths
        let country_normalized = match country {
            "UK" => "UnitedKingdom",
            "SouthAfrica" | "South Africa" => "South_Africa",
            _ => country,
        };

        // Load DMS escape fractions
        let dms_paths = [
            vasil_dir.join("ByCountry").join(country_normalized).join("results").join("epitope_data").join("dms_per_ab_per_site.csv"),
            vasil_dir.join("ByCountry").join(country).join("results").join("epitope_data").join("dms_per_ab_per_site.csv"),
        ];

        for dms_path in &dms_paths {
            if dms_path.exists() {
                eprintln!("[DMS] Loading from: {}", dms_path.display());
                if let Err(e) = result.load_dms_file(dms_path) {
                    eprintln!("[DMS WARNING] Failed to load {}: {}", dms_path.display(), e);
                } else {
                    break;
                }
            }
        }

        // Load mutation lists
        let mut_paths = [
            vasil_dir.join("ByCountry").join(country_normalized).join("results").join("mutation_data").join("mutation_lists.csv"),
            vasil_dir.join("ByCountry").join(country).join("results").join("mutation_data").join("mutation_lists.csv"),
        ];

        for mut_path in &mut_paths {
            if mut_path.exists() {
                eprintln!("[DMS] Loading mutations from: {}", mut_path.display());
                if let Err(e) = result.load_mutations_file(mut_path) {
                    eprintln!("[DMS WARNING] Failed to load {}: {}", mut_path.display(), e);
                } else {
                    break;
                }
            }
        }

        // Compute epitope averages
        result.compute_epitope_averages();

        // Diagnostic output
        eprintln!("[DMS] Loaded: {} antibodies, {} lineages, real_data={}",
            result.n_antibodies,
            result.lineage_mutations.len(),
            result.real_data_loaded);

        // Verify with sample variants
        if result.real_data_loaded {
            result.print_diagnostic();
        }

        Ok(result)
    }

    fn load_dms_file(&mut self, path: &Path) -> Result<()> {
        use std::io::{BufRead, BufReader};
        use std::fs::File;

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Parse header
        let header = lines.next()
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "Empty DMS file"))??;

        let headers: Vec<&str> = header.split(',').map(|s| s.trim()).collect();

        let find_col = |names: &[&str]| -> Option<usize> {
            for name in names {
                if let Some(idx) = headers.iter().position(|h| h.eq_ignore_ascii_case(name)) {
                    return Some(idx);
                }
            }
            None
        };

        let ab_col = find_col(&["antibody", "antibody_id", "ab", "condition"]).unwrap_or(0);
        let site_col = find_col(&["site", "position", "pos"]).unwrap_or(2);
        let escape_col = find_col(&["escape", "escape_fraction", "ef", "mut_escape"]).unwrap_or(3);
        let epitope_col = find_col(&["epitope", "epitope_class", "class", "group"]).unwrap_or(1);

        let mut ab_to_idx: HashMap<String, usize> = HashMap::new();

        for line_result in lines {
            let line = line_result?;
            if line.trim().is_empty() { continue; }

            let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            let max_col = ab_col.max(site_col).max(escape_col).max(epitope_col);
            if fields.len() <= max_col { continue; }

            let antibody = fields[ab_col].to_string();
            let site: u32 = fields[site_col].parse().unwrap_or(0);
            let escape: f32 = fields[escape_col].parse().unwrap_or(0.0);
            let epitope_str = fields[epitope_col];

            if site < 331 || site > 531 { continue; }
            let site_offset = (site - 331) as usize;

            let epitope_idx = match epitope_str.to_uppercase().as_str() {
                "A" => 0, "B" => 1, "C" => 2,
                "D1" => 3, "D2" => 4,
                "E1" | "E2" | "E12" | "E" => 5,
                "E3" => 6,
                "F1" => 7, "F2" => 8, "F3" => 9,
                _ => continue,
            };

            let ab_idx = *ab_to_idx.entry(antibody).or_insert_with(|| {
                let idx = self.raw_escape.len();
                self.raw_escape.push(vec![0.0; 201]);
                self.antibody_epitope.push(epitope_idx);
                idx
            });

            let escape_capped = escape.min(0.99).max(0.0);
            if ab_idx < self.raw_escape.len() && site_offset < 201 {
                self.raw_escape[ab_idx][site_offset] = escape_capped;
            }
        }

        self.n_antibodies = self.raw_escape.len();
        self.real_data_loaded = self.n_antibodies > 0;

        eprintln!("[DMS] Parsed {} antibodies", self.n_antibodies);
        Ok(())
    }

    fn load_mutations_file(&mut self, path: &Path) -> Result<()> {
        use std::io::{BufRead, BufReader};
        use std::fs::File;

        let file = File::open(path)?;
        let reader = BufReader::new(file);

        for line_result in reader.lines() {
            let line = line_result?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }

            // Try comma first, then tab
            let (lineage, mutations) = if let Some(idx) = line.find(',') {
                (line[..idx].trim(), line[idx+1..].trim())
            } else if let Some(idx) = line.find('\t') {
                (line[..idx].trim(), line[idx+1..].trim())
            } else {
                continue;
            };

            let lineage = lineage.trim_matches('"').to_string();
            let mutations = mutations.trim_matches('"').to_string();

            if !lineage.is_empty() && !mutations.is_empty() {
                self.lineage_mutations.insert(lineage, mutations);
            }
        }

        eprintln!("[DMS] Loaded mutations for {} lineages", self.lineage_mutations.len());
        Ok(())
    }

    fn compute_epitope_averages(&mut self) {
        for epitope_idx in 0..10 {
            for site_offset in 0..201 {
                let mut sum = 0.0_f32;
                let mut count = 0_u32;

                for (ab_idx, &ab_ep) in self.antibody_epitope.iter().enumerate() {
                    if ab_ep == epitope_idx {
                        if let Some(&ef) = self.raw_escape.get(ab_idx).and_then(|r| r.get(site_offset)) {
                            if ef > 0.0 {
                                sum += ef;
                                count += 1;
                            }
                        }
                    }
                }

                self.escape_by_site[epitope_idx][site_offset] = if count > 0 {
                    sum / count as f32
                } else {
                    0.0
                };
            }
        }
    }

    fn print_diagnostic(&mut self) {
        eprintln!("\n[DMS DIAGNOSTIC] Sample escape values:");
        let test_variants = ["BA.1", "BA.2", "BA.5", "XBB.1.5", "XBB.1.9", "BQ.1.1"];

        for variant in &test_variants {
            if let Some(muts) = self.lineage_mutations.get(*variant) {
                let parsed = parse_mutation_string(muts);
                let rbd: Vec<u32> = parsed.iter().filter(|m| m.is_rbd()).map(|m| m.site).collect();

                let mut escapes = Vec::new();
                for ep in 0..10 {
                    let mut total = 0.0_f32;
                    let mut count = 0_u32;
                    for &site in &rbd {
                        let offset = (site - 331) as usize;
                        if offset < 201 {
                            let ef = self.escape_by_site[ep][offset];
                            if ef > 0.0 { total += ef; count += 1; }
                        }
                    }
                    escapes.push(if count > 0 { total / count as f32 } else { 0.0 });
                }

                let esc_str: Vec<String> = escapes.iter().map(|e| format!("{:.3}", e)).collect();
                eprintln!("  {}: {} RBD muts, escapes=[{}]", variant, rbd.len(), esc_str.join(","));
            }
        }

        // Similarity check
        eprintln!("\n[DMS DIAGNOSTIC] Variant similarity (Jaccard on RBD sites):");
        let pairs = [("XBB.1.5", "XBB.1.9"), ("BA.5", "BQ.1.1"), ("BA.1", "BA.2")];
        for (v1, v2) in &pairs {
            let m1 = self.lineage_mutations.get(*v1).map(|s| parse_mutation_string(s)).unwrap_or_default();
            let m2 = self.lineage_mutations.get(*v2).map(|s| parse_mutation_string(s)).unwrap_or_default();
            let s1: HashSet<u32> = m1.iter().filter(|m| m.is_rbd()).map(|m| m.site).collect();
            let s2: HashSet<u32> = m2.iter().filter(|m| m.is_rbd()).map(|m| m.site).collect();
            let shared = s1.intersection(&s2).count();
            let total = s1.union(&s2).count();
            let jaccard = if total > 0 { shared as f32 / total as f32 } else { 0.0 };
            eprintln!("  {} vs {}: {}/{} shared (J={:.2})", v1, v2, shared, total, jaccard);
        }
        eprintln!("");
    }

    /// Get real epitope escape for a lineage (REPLACES hash-based)
    pub fn get_epitope_escape(&self, lineage: &str, epitope_idx: usize) -> Option<f32> {
        if epitope_idx >= 10 {
            return Some(0.0);
        }

        if !self.real_data_loaded {
            panic!("PRODUCTION VIOLATION: get_epitope_escape called with no real data loaded. Must load DMS data from VASIL files first.");
        }

        // Get mutations for this lineage
        let mutation_str = self.lineage_mutations.get(lineage)?;
        let parsed = parse_mutation_string(mutation_str);
        let rbd_sites: Vec<u32> = parsed.iter()
            .filter(|m| m.is_rbd())
            .map(|m| m.site)
            .collect();

        if rbd_sites.is_empty() {
            return Some(0.0);
        }

        let mut total = 0.0_f32;
        let mut count = 0_u32;

        for &site in &rbd_sites {
            let offset = (site - 331) as usize;
            if offset < 201 {
                let ef = self.escape_by_site[epitope_idx][offset];
                if ef > 0.0 {
                    total += ef;
                    count += 1;
                }
            }
        }

        Some(if count > 0 { total / count as f32 } else { 0.0 })
    }

    /// Get NTD escape - PRODUCTION: Not implemented (N-terminal domain escape not in VASIL dataset)
    pub fn get_ntd_escape(&self, lineage: &str) -> Option<f64> {
        // NTD escape data not available in VASIL DMS dataset (only RBD epitopes A-F)
        // If this is called in production GPU path, we have a problem
        eprintln!("[WARNING] get_ntd_escape called for {} - NTD not in VASIL DMS data", lineage);
        None  // Return None instead of fake 0.4
    }

    pub fn has_real_data(&self) -> bool { self.real_data_loaded }
    pub fn antibody_count(&self) -> usize { self.n_antibodies }
    pub fn lineage_count(&self) -> usize { self.lineage_mutations.len() }

    /// Compute mean escape per site (for compatibility with existing code)
    pub fn compute_mean_escape_per_site(&self) -> Vec<f32> {
        let mut mean_escape = vec![0.0f32; 201];
        for site_offset in 0..201 {
            let mut total = 0.0f32;
            let mut count = 0u32;
            for ep in 0..10 {
                let ef = self.escape_by_site[ep][site_offset];
                if ef > 0.0 {
                    total += ef;
                    count += 1;
                }
            }
            mean_escape[site_offset] = if count > 0 { total / count as f32 } else { 0.0 };
        }
        mean_escape
    }

    /// Compute mean escape per epitope (for compatibility)
    pub fn compute_mean_escape_per_epitope(&self) -> HashMap<String, Vec<f32>> {
        let epitope_names = ["A", "B", "C", "D1", "D2", "E12", "E3", "F1", "F2", "F3"];
        let mut result = HashMap::new();
        for (idx, &name) in epitope_names.iter().enumerate() {
            result.insert(name.to_string(), self.escape_by_site[idx].clone());
        }
        result
    }
}

/// Lineage mutation data
#[derive(Debug, Clone)]
pub struct LineageMutations {
    pub lineage_to_mutations: HashMap<String, Vec<String>>,  // lineage → RBD mutations
}

impl LineageMutations {
    /// Load from VASIL mutation_lists.csv
    pub fn load_from_vasil(
        vasil_data_dir: &Path,
        country: &str,
    ) -> Result<Self> {
        let mut_file = vasil_data_dir
            .join("ByCountry")
            .join(country)
            .join("results")
            .join("mutation_data")
            .join("mutation_lists.csv");

        if !mut_file.exists() {
            bail!("Mutation file not found: {:?}", mut_file);
        }

        log::info!("Loading variant mutations from: {:?}", mut_file);

        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_path(&mut_file)?;

        let mut lineage_to_mutations = HashMap::new();

        // Parse CSV (format: lineage, mutated_sites_RBD)
        for result in reader.records() {
            let record = result?;

            let lineage = record.get(0).context("Missing lineage")?.to_string();
            let mutations_str = record.get(1).unwrap_or("");

            if mutations_str.is_empty() {
                continue;
            }

            // Parse mutations (format: D614G/L452R/P681R/T19R/T478K)
            let mutations: Vec<String> = mutations_str
                .split('/')
                .filter(|m| !m.is_empty())
                .filter(|m| m.len() >= 3)
                .map(|m| m.to_string())
                .collect();

            lineage_to_mutations.insert(lineage, mutations);
        }

        log::info!("Loaded mutations for {} lineages", lineage_to_mutations.len());

        Ok(LineageMutations {
            lineage_to_mutations,
        })
    }

    /// Get mutations for lineage
    pub fn get_mutations(&self, lineage: &str) -> Option<&Vec<String>> {
        self.lineage_to_mutations.get(lineage)
    }
}

/// Compute velocities from frequency data
pub fn compute_velocities(frequencies: &GisaidFrequencies) -> Vec<Vec<f32>> {
    let mut velocities = vec![vec![0.0f32; frequencies.lineages.len()]; frequencies.dates.len()];

    for lineage_idx in 0..frequencies.lineages.len() {
        for date_idx in 1..frequencies.dates.len() {
            let prev_freq = frequencies.frequencies[date_idx - 1][lineage_idx];
            let curr_freq = frequencies.frequencies[date_idx][lineage_idx];

            let prev_date = frequencies.dates[date_idx - 1];
            let curr_date = frequencies.dates[date_idx];

            let days_delta = (curr_date - prev_date).num_days() as f32;

            if days_delta > 0.0 {
                let months_delta = days_delta / 30.0;
                let velocity = (curr_freq - prev_freq) / months_delta;
                velocities[date_idx][lineage_idx] = velocity;
            }
        }
    }

    velocities
}

/// Data for a single country
#[derive(Debug, Clone)]
pub struct CountryData {
    pub name: String,
    pub frequencies: GisaidFrequencies,
    pub mutations: LineageMutations,
    pub dms_data: DmsEscapeData,
    /// Daily incidence estimates (from VASIL phi or GInPipe) - for VASIL exact metric
    pub incidence_data: Option<Vec<f64>>,
    /// Vaccination timeline (cumulative fraction) - for VASIL exact metric
    pub vaccination_data: Option<Vec<f32>>,
    /// VASIL precomputed S_mean[date][pk] - frequency-weighted mean susceptibility
    pub s_mean_75pk: Vec<Vec<f64>>,  // [n_dates][75]
}

/// All countries data for multi-country training (VASIL methodology)
#[derive(Debug)]
pub struct AllCountriesData {
    pub countries: Vec<CountryData>,
}

impl AllCountriesData {
    pub fn new() -> Self {
        Self {
            countries: Vec::new(),
        }
    }

    pub fn add_country(&mut self, name: &str, freq: GisaidFrequencies, mutations: LineageMutations, dms: DmsEscapeData) {
        self.countries.push(CountryData {
            name: name.to_string(),
            frequencies: freq,
            mutations,
            dms_data: dms,
            incidence_data: None,
            vaccination_data: None,
            s_mean_75pk: Vec::new(),  // Will be loaded separately
        });
    }

    /// Load ALL 12 VASIL countries (exact same as VASIL paper Table 1)
    pub fn load_all_vasil_countries(vasil_data_dir: &Path) -> Result<Self> {
        const COUNTRIES: &[&str] = &[
            "Germany", "USA", "UK", "Japan", "Brazil", "France",
            "Canada", "Denmark", "Australia", "Sweden", "Mexico", "SouthAfrica"
        ];

        let mut all_data = Self::new();

        log::info!("Loading data from ALL 12 VASIL countries...");

        for country in COUNTRIES {
            log::info!("  Loading {}...", country);

            let freq = GisaidFrequencies::load_from_vasil(vasil_data_dir, country)
                .context(format!("Failed to load frequencies for {}", country))?;

            let mutations = LineageMutations::load_from_vasil(vasil_data_dir, country)
                .context(format!("Failed to load mutations for {}", country))?;

            let dms = DmsEscapeData::load_from_vasil(vasil_data_dir, country)
                .context(format!("Failed to load DMS data for {}", country))?;

            all_data.add_country(country, freq, mutations, dms);

            // Load VASIL precomputed S_mean from actual location
            let s_mean_path = Path::new("/media/diddy/PRISM-LBS/VASIL_Data")
                .join("ByCountry")
                .join(country)
                .join("results")
                .join("Susceptible_weighted_mean_over_spikegroups_all_PK.csv");

            if s_mean_path.exists() {
                eprintln!("[S_MEAN] Loading from: {}", s_mean_path.display());
                let s_mean_75pk = Self::load_s_mean_csv(&s_mean_path)
                    .context(format!("Failed to load S_mean for {}", country))?;
                all_data.countries.last_mut().unwrap().s_mean_75pk = s_mean_75pk;
                eprintln!("[S_MEAN] ✓ Loaded {} dates × 75 PKs",
                         all_data.countries.last().unwrap().s_mean_75pk.len());
            } else {
                eprintln!("[S_MEAN] ⚠ File not found: {}", s_mean_path.display());
                eprintln!("[S_MEAN] → Will compute S_mean on-the-fly (less accurate)");
                // Leave s_mean_75pk empty - kernel will compute if needed
            }

            log::info!("    ✅ {} lineages, {} dates",
                       all_data.countries.last().unwrap().frequencies.lineages.len(),
                       all_data.countries.last().unwrap().frequencies.dates.len());
        }

        log::info!("✅ Loaded all 12 countries successfully!");

        Ok(all_data)
    }

    /// Load VASIL precomputed S_mean CSV (75 PK columns × n_dates rows)
    fn load_s_mean_csv(path: &Path) -> Result<Vec<Vec<f64>>> {
        use std::fs::File;
        use std::io::{BufReader, BufRead};

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut s_mean_data = Vec::new();

        for (line_idx, line) in reader.lines().enumerate() {
            let line = line?;
            if line_idx == 0 {
                // Skip header
                continue;
            }

            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 76 {  // Days column + 75 PK columns
                continue;
            }

            // Parse 75 PK values (skip first column which is date)
            let mut pk_values = Vec::with_capacity(75);
            for i in 1..=75 {
                if let Ok(val) = parts[i].trim().replace("\"", "").parse::<f64>() {
                    pk_values.push(val);
                } else {
                    pk_values.push(0.0);  // Default if parse fails
                }
            }

            s_mean_data.push(pk_values);
        }

        Ok(s_mean_data)
    }
}

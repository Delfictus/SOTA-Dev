//! Ablation analysis: baseline vs cryo vs cryo+UV
//!
//! ABLATION IS MANDATORY for every run to properly attribute
//! site emergence and UV response.

use crate::inputs::CryoProbeResults;
use crate::sites::CrypticSite;
use serde::{Deserialize, Serialize};

/// Ablation mode identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AblationMode {
    /// Baseline: no cryo schedule, UV off (300K constant)
    Baseline,
    /// Cryo-only: cryo schedule on, UV off
    CryoOnly,
    /// Cryo+UV: cryo schedule on, UV on
    CryoUv,
}

impl AblationMode {
    /// Human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            AblationMode::Baseline => "baseline",
            AblationMode::CryoOnly => "cryo-only",
            AblationMode::CryoUv => "cryo+UV",
        }
    }

    /// Description
    pub fn description(&self) -> &'static str {
        match self {
            AblationMode::Baseline => "Physiological (300K constant, UV off)",
            AblationMode::CryoOnly => "Cryogenic temperature ramp, UV off",
            AblationMode::CryoUv => "Cryogenic temperature ramp, UV on",
        }
    }
}

/// Results from one ablation run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationRunResult {
    /// Mode of this run
    pub mode: AblationMode,
    /// Total spikes detected (raw count)
    pub total_spikes: usize,
    /// Number of pocket events emitted to events.jsonl
    /// Each spike detection can emit one or more PocketEvent records.
    pub events_emitted: usize,
    /// Phase spikes (cold, ramp, warm)
    pub phase_spikes: (usize, usize, usize),
    /// Number of frames/steps analyzed in this run
    /// IMPORTANT: Baseline runs fewer frames than cryo phases.
    /// Use this for normalization when comparing conditions.
    pub frames_analyzed: usize,
    /// Spike rate per 1000 frames (normalized metric for fair comparison)
    pub spikes_per_1k_frames: f64,
    /// Detected sites
    pub sites: Vec<CrypticSite>,
    /// Mean site volume
    pub mean_volume: f64,
    /// Mean site SASA (if computed)
    pub mean_sasa: Option<f64>,
    /// Runtime (seconds)
    pub runtime_seconds: f64,
}

/// Complete ablation analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationResults {
    /// Baseline run
    pub baseline: AblationRunResult,
    /// Cryo-only run
    pub cryo_only: AblationRunResult,
    /// Cryo+UV run
    pub cryo_uv: AblationRunResult,
    /// Computed deltas
    pub deltas: AblationDeltas,
    /// Statistical comparison
    pub comparison: AblationComparison,
}

/// Delta metrics between ablation modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationDeltas {
    /// Spike delta: cryo_only - baseline (RAW counts, NOT normalized)
    pub spikes_cryo_vs_baseline: i64,
    /// Spike delta: cryo_uv - cryo_only (RAW counts)
    pub spikes_cryouv_vs_cryo: i64,
    /// Spike delta: cryo_uv - baseline (RAW counts)
    pub spikes_cryouv_vs_baseline: i64,
    /// Spike RATE delta per 1k frames: cryo_only - baseline (NORMALIZED)
    /// Use this for fair comparison since baseline runs fewer frames
    pub rate_cryo_vs_baseline: f64,
    /// Spike RATE delta per 1k frames: cryo_uv - cryo_only (NORMALIZED)
    pub rate_cryouv_vs_cryo: f64,
    /// Site count delta: cryo_only - baseline
    pub sites_cryo_vs_baseline: i32,
    /// Site count delta: cryo_uv - cryo_only
    pub sites_cryouv_vs_cryo: i32,
    /// Volume delta: cryo_uv - cryo_only (mean)
    pub volume_cryouv_vs_cryo: f64,
    /// SASA delta: cryo_uv - cryo_only (mean, if available)
    pub sasa_cryouv_vs_cryo: Option<f64>,
}

/// Statistical comparison of ablation modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationComparison {
    /// Is cryo contrast significant? (cryo > baseline)
    pub cryo_contrast_significant: bool,
    /// Is UV response significant? (cryo+UV differs from cryo-only)
    pub uv_response_significant: bool,
    /// Effect size of cryo contrast
    pub cryo_effect_size: f64,
    /// Effect size of UV response
    pub uv_effect_size: f64,
    /// Interpretation text
    pub interpretation: String,
}

impl AblationResults {
    /// Compute ablation results from three runs
    pub fn compute(
        baseline: AblationRunResult,
        cryo_only: AblationRunResult,
        cryo_uv: AblationRunResult,
    ) -> Self {
        // Compute normalized rate deltas (per 1k frames)
        // This is essential because baseline runs fewer frames than cryo phases
        let rate_cryo_vs_baseline = cryo_only.spikes_per_1k_frames - baseline.spikes_per_1k_frames;
        let rate_cryouv_vs_cryo = cryo_uv.spikes_per_1k_frames - cryo_only.spikes_per_1k_frames;

        // Compute deltas (both raw and normalized)
        let deltas = AblationDeltas {
            // Raw counts (for reporting absolute numbers)
            spikes_cryo_vs_baseline: cryo_only.total_spikes as i64 - baseline.total_spikes as i64,
            spikes_cryouv_vs_cryo: cryo_uv.total_spikes as i64 - cryo_only.total_spikes as i64,
            spikes_cryouv_vs_baseline: cryo_uv.total_spikes as i64 - baseline.total_spikes as i64,
            // Normalized rates (for fair comparison)
            rate_cryo_vs_baseline,
            rate_cryouv_vs_cryo,
            sites_cryo_vs_baseline: cryo_only.sites.len() as i32 - baseline.sites.len() as i32,
            sites_cryouv_vs_cryo: cryo_uv.sites.len() as i32 - cryo_only.sites.len() as i32,
            volume_cryouv_vs_cryo: cryo_uv.mean_volume - cryo_only.mean_volume,
            sasa_cryouv_vs_cryo: match (cryo_uv.mean_sasa, cryo_only.mean_sasa) {
                (Some(uv), Some(cryo)) => Some(uv - cryo),
                _ => None,
            },
        };

        // Compute comparison metrics using NORMALIZED rates (fair comparison)
        // Baseline runs fewer frames, so raw count comparison is misleading
        let cryo_contrast_significant = rate_cryo_vs_baseline > 0.0
            && (cryo_only.spikes_per_1k_frames / baseline.spikes_per_1k_frames.max(0.001)) > 1.2;

        let uv_response_significant = rate_cryouv_vs_cryo.abs() > 1.0  // At least 1 spike per 1k frames difference
            || deltas.volume_cryouv_vs_cryo.abs() > 10.0
            || deltas.sasa_cryouv_vs_cryo.map(|d| d.abs() > 20.0).unwrap_or(false);

        // Effect sizes using normalized rates
        let cryo_effect_size = if baseline.spikes_per_1k_frames > 0.0 {
            (cryo_only.spikes_per_1k_frames - baseline.spikes_per_1k_frames)
                / baseline.spikes_per_1k_frames
        } else {
            0.0
        };

        let uv_effect_size = if cryo_only.spikes_per_1k_frames > 0.0 {
            (cryo_uv.spikes_per_1k_frames - cryo_only.spikes_per_1k_frames).abs()
                / cryo_only.spikes_per_1k_frames
        } else {
            0.0
        };

        // Generate interpretation
        let interpretation = Self::generate_interpretation(
            &deltas,
            cryo_contrast_significant,
            uv_response_significant,
        );

        let comparison = AblationComparison {
            cryo_contrast_significant,
            uv_response_significant,
            cryo_effect_size,
            uv_effect_size,
            interpretation,
        };

        Self {
            baseline,
            cryo_only,
            cryo_uv,
            deltas,
            comparison,
        }
    }

    fn generate_interpretation(
        deltas: &AblationDeltas,
        cryo_sig: bool,
        uv_sig: bool,
    ) -> String {
        let mut parts = Vec::new();

        if cryo_sig {
            parts.push(format!(
                "Cryogenic contrast is SIGNIFICANT: {} additional spikes detected vs baseline, \
                 indicating successful primary hydrophobic site mapping.",
                deltas.spikes_cryo_vs_baseline
            ));
        } else {
            parts.push(
                "Cryogenic contrast is NOT significant. This may indicate a rigid structure \
                 or insufficient temperature range."
                    .to_string(),
            );
        }

        if uv_sig {
            if deltas.spikes_cryouv_vs_cryo > 0 {
                parts.push(format!(
                    "UV response is POSITIVE: {} additional spikes with UV excitation, \
                     suggesting UV-driven pocket opening.",
                    deltas.spikes_cryouv_vs_cryo
                ));
            } else if deltas.spikes_cryouv_vs_cryo < 0 {
                parts.push(format!(
                    "UV response is NEGATIVE: {} fewer spikes with UV excitation, \
                     suggesting UV-induced stabilization.",
                    deltas.spikes_cryouv_vs_cryo.abs()
                ));
            }
            if deltas.volume_cryouv_vs_cryo.abs() > 10.0 {
                parts.push(format!(
                    "Volume change: {:.1} A^3 mean pocket expansion from UV.",
                    deltas.volume_cryouv_vs_cryo
                ));
            }
        } else {
            parts.push(
                "UV response is NOT significant. Cryptic sites may not be UV-accessible \
                 or require different wavelengths."
                    .to_string(),
            );
        }

        parts.join(" ")
    }

    /// Generate ablation paragraph for report (REQUIRED)
    pub fn ablation_paragraph() -> &'static str {
        r#"
## Ablation Analysis: Why This Is Required

Ablation analysis (comparing baseline vs cryo vs cryo+UV) is MANDATORY for every PRISM4D
cryptic site detection run. This three-way comparison is essential because:

1. **Baseline (300K constant, UV off)** establishes the physiological reference state.
   Any sites detected here represent constitutively accessible pockets.

2. **Cryo-only (temperature ramp, UV off)** reveals sites that emerge due to cryogenic
   contrast—the temperature-dependent dewetting of hydrophobic regions. Comparing to
   baseline isolates the cryo effect.

3. **Cryo+UV (temperature ramp, UV on)** adds aromatic excitation to probe UV-responsive
   cryptic sites. Comparing to cryo-only isolates the UV contribution.

Without this ablation, we cannot attribute site emergence to the cryo protocol vs UV
excitation vs baseline fluctuations. Sites appearing in cryo+UV but not cryo-only are
UV-responsive candidates. Sites appearing in cryo-only but not baseline are
temperature-driven. This decomposition is critical for mechanistic interpretation
and experimental follow-up.

The delta metrics (Δspikes, Δvolume, ΔSASA) between modes quantify the magnitude
of each effect and enable statistical significance testing.
"#
    }
}

/// Helper to convert CryoProbeResults to AblationRunResult
impl From<&CryoProbeResults> for AblationRunResult {
    fn from(results: &CryoProbeResults) -> Self {
        // Determine mode from protocol name
        let mode = if results.protocol.contains("Physiological") {
            AblationMode::Baseline
        } else if results.uv_config.burst_energy > 0.0 {
            AblationMode::CryoUv
        } else {
            AblationMode::CryoOnly
        };

        // Frames analyzed from temperature protocol
        let frames_analyzed = results.total_steps.max(1) as usize;
        let spikes_per_1k_frames = (results.total_spikes as f64) / (frames_analyzed as f64 / 1000.0);

        AblationRunResult {
            mode,
            total_spikes: results.total_spikes,
            events_emitted: 0, // Not tracked in CryoProbeResults; set later from events.jsonl
            phase_spikes: (
                results.phase_spikes.cold,
                results.phase_spikes.ramp,
                results.phase_spikes.warm,
            ),
            frames_analyzed,
            spikes_per_1k_frames,
            sites: Vec::new(), // To be populated separately
            mean_volume: 0.0,  // To be computed from sites
            mean_sasa: None,
            runtime_seconds: results.elapsed_seconds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_run(mode: AblationMode, spikes: usize, n_sites: usize, frames: usize) -> AblationRunResult {
        let spikes_per_1k_frames = (spikes as f64) / (frames as f64 / 1000.0);
        AblationRunResult {
            mode,
            total_spikes: spikes,
            events_emitted: spikes, // For tests: 1 event per spike
            phase_spikes: (spikes / 3, spikes / 3, spikes / 3),
            frames_analyzed: frames,
            spikes_per_1k_frames,
            sites: vec![], // Empty for this test
            mean_volume: 200.0 + n_sites as f64 * 10.0,
            mean_sasa: Some(100.0 + n_sites as f64 * 5.0),
            runtime_seconds: 60.0,
        }
    }

    #[test]
    fn test_ablation_deltas() {
        // Baseline runs fewer frames (50k) than cryo phases (100k)
        let baseline = make_run(AblationMode::Baseline, 1000, 2, 50000);
        let cryo = make_run(AblationMode::CryoOnly, 5000, 5, 100000);
        let cryo_uv = make_run(AblationMode::CryoUv, 7000, 7, 100000);

        let results = AblationResults::compute(baseline, cryo, cryo_uv);

        // Raw deltas
        assert_eq!(results.deltas.spikes_cryo_vs_baseline, 4000);
        assert_eq!(results.deltas.spikes_cryouv_vs_cryo, 2000);

        // Normalized rates: baseline=20/1k, cryo=50/1k, cryo_uv=70/1k
        assert!((results.baseline.spikes_per_1k_frames - 20.0).abs() < 0.1);
        assert!((results.cryo_only.spikes_per_1k_frames - 50.0).abs() < 0.1);
        assert!((results.cryo_uv.spikes_per_1k_frames - 70.0).abs() < 0.1);

        // Rate deltas
        assert!((results.deltas.rate_cryo_vs_baseline - 30.0).abs() < 0.1);
        assert!((results.deltas.rate_cryouv_vs_cryo - 20.0).abs() < 0.1);

        assert!(results.comparison.cryo_contrast_significant);
        assert!(results.comparison.uv_response_significant);
    }

    #[test]
    fn test_ablation_normalization_matters() {
        // Same spike count but different frame counts should give different rates
        // This tests that raw count comparison would be misleading

        // Baseline: 100 spikes in 50k frames = 2/1k
        let baseline = make_run(AblationMode::Baseline, 100, 1, 50000);
        // Cryo: 100 spikes in 100k frames = 1/1k (LOWER rate despite same raw count!)
        let cryo = make_run(AblationMode::CryoOnly, 100, 1, 100000);
        let cryo_uv = make_run(AblationMode::CryoUv, 150, 1, 100000);

        let results = AblationResults::compute(baseline, cryo, cryo_uv);

        // Raw delta would say cryo == baseline (both 100 spikes)
        assert_eq!(results.deltas.spikes_cryo_vs_baseline, 0);

        // But normalized rate shows baseline actually has HIGHER rate!
        // baseline: 2/1k, cryo: 1/1k, delta = -1.0
        assert!(results.deltas.rate_cryo_vs_baseline < 0.0);

        // So cryo contrast should NOT be significant
        assert!(!results.comparison.cryo_contrast_significant);
    }

    #[test]
    fn test_ablation_paragraph() {
        let paragraph = AblationResults::ablation_paragraph();
        assert!(paragraph.contains("MANDATORY"));
        assert!(paragraph.contains("baseline"));
        assert!(paragraph.contains("cryo"));
    }
}

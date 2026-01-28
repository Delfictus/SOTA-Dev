//! Cryptic site detection, metrics, and ranking

use crate::config::RankingWeights;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Detected cryptic site
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrypticSite {
    /// Site ID (e.g., "site_001")
    pub site_id: String,
    /// Rank (1 = best)
    pub rank: usize,
    /// Centroid position [x, y, z] in Å
    pub centroid: [f32; 3],
    /// Residue IDs in this site
    pub residues: Vec<usize>,
    /// Residue names
    pub residue_names: Vec<String>,
    /// Chain ID
    pub chain_id: String,
    /// Site metrics
    pub metrics: SiteMetrics,
    /// Ranking score (0-1)
    pub rank_score: f64,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Is druggable (volume >= threshold)
    pub is_druggable: bool,
    /// First frame detected
    pub first_frame: usize,
    /// Last frame detected
    pub last_frame: usize,
    /// Representative frame (best open conformation)
    pub representative_frame: usize,
}

/// Site metrics as required by output contract
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiteMetrics {
    /// Persistence metrics
    pub persistence: PersistenceMetrics,
    /// Geometry metrics
    pub geometry: GeometryMetrics,
    /// Chemistry metrics
    pub chemistry: ChemistryMetrics,
    /// UV response metrics (delta from ablation)
    pub uv_response: UvResponseMetrics,
}

/// Persistence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceMetrics {
    /// Fraction of frames where site is present
    pub present_fraction: f64,
    /// Mean lifetime in frames
    pub mean_lifetime_frames: f64,
    /// Agreement across replicates (0-1)
    pub replica_agreement: f64,
}

/// Geometry metrics
///
/// All metrics are computed from real data. Fields that could not be computed
/// are set to None rather than using placeholder values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryMetrics {
    /// Volume mean (Å³) - computed from event volumes
    pub volume_mean: f64,
    /// Volume median (Å³) - computed from event volumes
    pub volume_p50: f64,
    /// Volume 95th percentile (Å³) - computed from event volumes
    pub volume_p95: f64,

    /// Volume minimum (Å³) - computed from event volumes
    #[serde(default)]
    pub volume_min: f64,
    /// Volume maximum (Å³) - computed from event volumes
    #[serde(default)]
    pub volume_max: f64,
    /// Volume standard deviation (Å³) - computed from event volumes
    #[serde(default)]
    pub volume_std: f64,
    /// Breathing amplitude (Å³): Volume fluctuation (max - min)
    /// REAL: Computed from actual volume trajectory, NOT a heuristic
    #[serde(default)]
    pub breathing_amplitude: f64,

    /// Aspect ratio: Longest/shortest principal axis from PCA
    /// REAL: 1.0 = symmetric sphere, higher = elongated
    /// None if < 4 event points available for PCA
    #[serde(skip_serializing_if = "Option::is_none")]
    pub aspect_ratio: Option<f64>,
    /// Sphericity (0-1): How spherical the pocket is from PCA
    /// REAL: 1.0 = perfect sphere, lower = elongated
    /// None if < 4 event points available for PCA
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sphericity: Option<f64>,

    /// Depth to pocket mouth (p95 of BFS distances from mouth voxels), Å
    /// None if computation failed (e.g., no density grid available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_proxy_pocket_a: Option<f64>,

    /// Depth to protein surface (p95 of distance transform), Å
    /// None if computation failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_proxy_surface_a: Option<f64>,

    /// Largest opening mouth area (Å²)
    /// None if computation failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mouth_area_proxy_a2: Option<f64>,

    /// Total mouth area across all openings (Å²)
    /// None if computation failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mouth_area_total_a2: Option<f64>,

    /// Number of distinct openings
    /// None if computation failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n_openings: Option<usize>,
}

/// Chemistry metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemistryMetrics {
    /// Hydrophobic fraction (0-1)
    pub hydrophobic_fraction: f64,
    /// H-bond donor count
    pub donor_count: usize,
    /// H-bond acceptor count
    pub acceptor_count: usize,
    /// Aromatic fraction (0-1)
    pub aromatic_fraction: f64,
    /// Charged fraction (0-1)
    pub charged_fraction: f64,
}

/// UV response metrics (from ablation comparison)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UvResponseMetrics {
    /// Delta SASA: (cryo+UV) - (cryo-only)
    pub delta_sasa: f64,
    /// Delta volume: (cryo+UV) - (cryo-only)
    pub delta_volume: f64,
    /// Response significance (p-value or z-score proxy)
    pub significance: f64,
}

impl Default for UvResponseMetrics {
    fn default() -> Self {
        Self {
            delta_sasa: 0.0,
            delta_volume: 0.0,
            significance: 0.0,
        }
    }
}

/// Site ranking based on weighted scores
#[derive(Debug, Clone)]
pub struct SiteRanking {
    /// Weights used for ranking
    pub weights: RankingWeights,
}

impl SiteRanking {
    pub fn new(weights: RankingWeights) -> Self {
        Self { weights }
    }

    /// Compute rank score for a site
    pub fn compute_score(&self, site: &CrypticSite) -> f64 {
        let w = &self.weights;
        let m = &site.metrics;

        // Normalize each component to 0-1
        let persistence_score = m.persistence.present_fraction;
        let volume_score = (m.geometry.volume_mean / 500.0).min(1.0); // 500 Å³ = 1.0
        let uv_score = (m.uv_response.delta_sasa.abs() / 100.0).min(1.0); // 100 Å² = 1.0
        let hydrophobicity_score = m.chemistry.hydrophobic_fraction;
        let replica_score = m.persistence.replica_agreement;

        // Weighted sum (cast f32 weights to f64)
        (w.persistence as f64) * persistence_score
            + (w.volume as f64) * volume_score
            + (w.uv_response as f64) * uv_score
            + (w.hydrophobicity as f64) * hydrophobicity_score
            + (w.replica_agreement as f64) * replica_score
    }

    /// Rank sites and assign rank numbers
    pub fn rank_sites(&self, sites: &mut [CrypticSite]) {
        // Compute scores
        for site in sites.iter_mut() {
            site.rank_score = self.compute_score(site);
        }

        // Sort by score descending
        sites.sort_by(|a, b| {
            b.rank_score
                .partial_cmp(&a.rank_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Assign ranks
        for (i, site) in sites.iter_mut().enumerate() {
            site.rank = i + 1;
            site.site_id = format!("site_{:03}", i + 1);
        }
    }
}

/// Kyte-Doolittle hydrophobicity scale
pub fn kyte_doolittle(residue: &str) -> f64 {
    match residue.to_uppercase().as_str() {
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

/// Check if residue is hydrophobic
pub fn is_hydrophobic(residue: &str) -> bool {
    matches!(
        residue.to_uppercase().as_str(),
        "ILE" | "VAL" | "LEU" | "PHE" | "MET" | "ALA" | "TRP"
    )
}

/// Check if residue is aromatic
pub fn is_aromatic(residue: &str) -> bool {
    matches!(
        residue.to_uppercase().as_str(),
        "PHE" | "TYR" | "TRP" | "HIS"
    )
}

/// Check if residue is charged
pub fn is_charged(residue: &str) -> bool {
    matches!(
        residue.to_uppercase().as_str(),
        "ASP" | "GLU" | "LYS" | "ARG" | "HIS"
    )
}

/// H-bond donor count for residue
pub fn donor_count(residue: &str) -> usize {
    match residue.to_uppercase().as_str() {
        "ARG" => 5,
        "LYS" => 2,
        "ASN" | "GLN" | "HIS" | "SER" | "THR" | "TYR" | "TRP" => 2,
        _ => 1, // Backbone NH
    }
}

/// H-bond acceptor count for residue
pub fn acceptor_count(residue: &str) -> usize {
    match residue.to_uppercase().as_str() {
        "GLU" | "ASP" => 3,
        "ASN" | "GLN" | "HIS" | "SER" | "THR" | "TYR" | "MET" => 2,
        _ => 1, // Backbone C=O
    }
}

/// Compute chemistry metrics from residue list
pub fn compute_chemistry_metrics(residue_names: &[String]) -> ChemistryMetrics {
    if residue_names.is_empty() {
        return ChemistryMetrics {
            hydrophobic_fraction: 0.0,
            donor_count: 0,
            acceptor_count: 0,
            aromatic_fraction: 0.0,
            charged_fraction: 0.0,
        };
    }

    let n = residue_names.len() as f64;
    let hydrophobic = residue_names.iter().filter(|r| is_hydrophobic(r)).count();
    let aromatic = residue_names.iter().filter(|r| is_aromatic(r)).count();
    let charged = residue_names.iter().filter(|r| is_charged(r)).count();
    let donors: usize = residue_names.iter().map(|r| donor_count(r)).sum();
    let acceptors: usize = residue_names.iter().map(|r| acceptor_count(r)).sum();

    ChemistryMetrics {
        hydrophobic_fraction: hydrophobic as f64 / n,
        donor_count: donors,
        acceptor_count: acceptors,
        aromatic_fraction: aromatic as f64 / n,
        charged_fraction: charged as f64 / n,
    }
}

/// Estimate pocket volume from voxel count and spacing
pub fn estimate_volume(voxel_count: usize, grid_spacing: f32) -> f64 {
    let voxel_volume = (grid_spacing as f64).powi(3);
    let expansion_factor = 2.0; // Account for gaps
    voxel_count as f64 * voxel_volume * expansion_factor
}

/// Compute centroid from positions
pub fn compute_centroid(positions: &[[f32; 3]]) -> [f32; 3] {
    if positions.is_empty() {
        return [0.0; 3];
    }
    let n = positions.len() as f32;
    let mut sum = [0.0f32; 3];
    for pos in positions {
        sum[0] += pos[0];
        sum[1] += pos[1];
        sum[2] += pos[2];
    }
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

/// Euclidean distance
pub fn distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hydrophobicity() {
        assert!(is_hydrophobic("ILE"));
        assert!(is_hydrophobic("VAL"));
        assert!(!is_hydrophobic("ASP"));
        assert!(!is_hydrophobic("LYS"));
    }

    #[test]
    fn test_chemistry_metrics() {
        let residues = vec![
            "LEU".to_string(),
            "ILE".to_string(),
            "VAL".to_string(),
            "ASP".to_string(),
            "PHE".to_string(),
        ];
        let metrics = compute_chemistry_metrics(&residues);
        assert!((metrics.hydrophobic_fraction - 0.8).abs() < 0.01); // 4/5
        assert!((metrics.aromatic_fraction - 0.2).abs() < 0.01); // 1/5
    }

    #[test]
    fn test_centroid() {
        let positions = vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 3.0, 0.0]];
        let c = compute_centroid(&positions);
        assert!((c[0] - 1.0).abs() < 0.001);
        assert!((c[1] - 1.0).abs() < 0.001);
    }
}

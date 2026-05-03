//! Persistent NHS Engine for High-Throughput Batch Processing
//!
//! Keeps CUDA context, modules, and buffers alive across multiple structures.
//! Hot-swaps topologies without reinitializing GPU state.
//!
//! ## Performance Benefits
//! - Single CUDA context creation (~100ms saved per structure)
//! - Single PTX compilation (~200ms saved per structure)
//! - Buffer reuse for similar-sized structures
//! - Pipelined data transfer during compute
//!
//! ## Usage
//! ```no_run
//! let mut engine = PersistentNhsEngine::new(max_atoms)?;
//! for topology in topologies {
//!     engine.load_topology(&topology)?;
//!     let results = engine.run(steps, config)?;
//! }
//! ```

use anyhow::{bail, Context, Result};
use std::sync::Arc;
use std::path::Path;
use std::time::Instant;

#[cfg(feature = "gpu")]
use cudarc::driver::{
    CudaContext, CudaSlice, CudaStream, CudaFunction, CudaModule,
    LaunchConfig, PushKernelArg, DevicePtrMut,
};
#[cfg(feature = "gpu")]
use cudarc::nvrtc::Ptx;

use crate::input::PrismPrepTopology;

#[allow(deprecated)]
use crate::fused_engine::{
    NhsAmberFusedEngine, CryoUvProtocol,
    // Deprecated - kept for backward compatibility
    TemperatureProtocol,
    UvProbeConfig,
    StepResult, RunSummary, SpikeEvent, EnsembleSnapshot,
};

/// Configuration for persistent batch processing
#[derive(Debug, Clone)]
pub struct PersistentBatchConfig {
    /// Maximum atoms to pre-allocate for (prevents reallocation)
    pub max_atoms: usize,
    /// Grid dimension for exclusion field
    pub grid_dim: usize,
    /// Grid spacing in Angstroms
    pub grid_spacing: f32,
    /// Survey phase steps (cryo)
    pub survey_steps: i32,
    /// Convergence phase steps (warming)
    pub convergence_steps: i32,
    /// Precision phase steps (production)
    pub precision_steps: i32,
    /// Target temperature (K)
    pub temperature: f32,
    /// Cryo temperature (K)
    pub cryo_temp: f32,
    /// Cryo hold steps before warming
    pub cryo_hold: i32,
}

impl Default for PersistentBatchConfig {
    fn default() -> Self {
        Self {
            max_atoms: 20000,  // Handle large structures (1DLO ~16K atoms)
            grid_dim: 128,
            grid_spacing: 0.75,
            survey_steps: 500000,    // 1ns
            convergence_steps: 1000000, // 2ns
            precision_steps: 1000000,   // 2ns
            temperature: 300.0,
            cryo_temp: 100.0,
            cryo_hold: 100000,
        }
    }
}

/// Result from processing a single structure
#[derive(Debug, Clone)]
pub struct StructureResult {
    pub structure_id: String,
    pub total_steps: i32,
    pub wall_time_ms: u64,
    pub spike_events: Vec<SpikeEvent>,
    pub snapshots: Vec<EnsembleSnapshot>,
    pub final_temperature: f32,
    /// Clustered binding sites from RT-accelerated spatial analysis
    pub clustered_sites: Vec<ClusteredBindingSite>,
    /// RT clustering statistics
    pub clustering_stats: Option<ClusteringStats>,
}

/// A residue lining a binding site pocket
#[derive(Debug, Clone, serde::Serialize)]
pub struct LiningResidue {
    /// Chain identifier (e.g., "A", "B")
    pub chain: String,
    /// Residue sequence number
    pub resid: i32,
    /// Residue name (e.g., "ALA", "PHE")
    pub resname: String,
    /// Minimum distance from any atom to pocket centroid (Å)
    pub min_distance: f32,
    /// Number of atoms within cutoff (geometric — from compute_lining_residues)
    pub n_atoms_in_pocket: usize,
    /// Number of GPU spike events attributed to this residue across all chunks
    /// (dynamics — populated by the multi-stream consensus spike-attribution pass).
    /// Distinct from `n_atoms_in_pocket`: the geometric atom count tells you which
    /// residues line the pocket; the spike attribution count tells you which of
    /// those residues actually carried activity during the run.
    #[serde(default)]
    pub spike_attribution_count: u32,
}

/// A clustered binding site detected from spike spatial patterns.
///
/// # Spatial representation contract (Phase 1 post-05dbc3dc lane)
///
/// Every canonical site carries a typed multi-view localization
/// [`CentroidManifold`][crate::spatial_view::CentroidManifold] in
/// `localization`. Canonical-logic callers MUST access site centroids
/// through [`ClusteredBindingSite::view`] / [`ClusteredBindingSite::distance_to`] /
/// [`ClusteredBindingSite::dcc`] / [`ClusteredBindingSite::distance_to_ligand`]
/// naming the physical representation they want. There is no hidden
/// default; absent views return `None`.
///
/// The `legacy_emission_centroid` field is retained in module-private
/// visibility (`pub(in crate::persistent_engine)`) only for the
/// serialization / logging compatibility path exposed via
/// [`ClusteredBindingSite::emission_compat_centroid`]. It must not be
/// used for merge, attribution, ranking, validation, or any canonical
/// logic — those must call the typed view API.
///
/// Writes to the geometric-voxel-mass centroid must go through
/// [`ClusteredBindingSite::set_geometric_voxel_mass_centroid`], which
/// updates both the legacy emission field and the manifold's
/// GeometricVoxelMass view atomically. Other views are written via
/// `localization.set(SpatialView::...)` directly.
#[derive(Debug, Clone)]
pub struct ClusteredBindingSite {
    /// Cluster ID (unique per analysis run)
    pub cluster_id: i32,
    /// Legacy single-scalar centroid retained solely for emission /
    /// logging / serialization compatibility. **DO NOT read this in
    /// canonical logic** — use [`ClusteredBindingSite::view`] or the
    /// other typed accessors. The field is module-private so that
    /// external consumers (bin/nhs_rt_full.rs, etc.) cannot touch
    /// the ambiguous scalar without going through an explicit shim.
    pub(in crate::persistent_engine) legacy_emission_centroid: [f32; 3],
    /// Multi-view localization manifold. Always contains at least
    /// the `GeometricVoxelMass` view when constructed through the
    /// canonical site-building paths.
    pub localization: crate::spatial_view::CentroidManifold,
    /// Number of spikes in this cluster
    pub spike_count: usize,
    /// Spike indices belonging to this cluster
    pub spike_indices: Vec<usize>,
    /// Average spike intensity in cluster
    pub avg_intensity: f32,
    /// Estimated volume (convex hull approximation) in Å³
    pub estimated_volume: f32,
    /// Bounding box dimensions [dx, dy, dz] in Angstroms
    pub bounding_box: [f32; 3],
    /// Quality score for this binding site (0.0-1.0)
    pub quality_score: f32,
    /// Druggability assessment
    pub druggability: DruggabilityScore,
    /// Site classification
    pub classification: SiteClassification,
    /// Aromatic proximity analysis (if computed)
    pub aromatic_proximity: Option<AromaticProximityInfo>,
    /// Residues lining this binding pocket (within cutoff distance)
    pub lining_residues: Vec<LiningResidue>,
}

/// Aromatic residue proximity information for a binding site
#[derive(Debug, Clone, Default)]
pub struct AromaticProximityInfo {
    /// Distance to nearest TRP residue (Å), None if no TRP in structure
    pub nearest_trp_distance: Option<f32>,
    /// Distance to nearest TYR residue (Å)
    pub nearest_tyr_distance: Option<f32>,
    /// Distance to nearest PHE residue (Å)
    pub nearest_phe_distance: Option<f32>,
    /// Distance to nearest aromatic (any type)
    pub nearest_aromatic_distance: f32,
    /// Number of aromatics within 5Å
    pub aromatics_within_5a: usize,
    /// Number of aromatics within 8Å
    pub aromatics_within_8a: usize,
    /// Aromatic residue indices within 8Å
    pub nearby_aromatic_residues: Vec<u32>,
    /// Aromatic score (0.0-1.0) based on proximity
    pub aromatic_score: f32,
}

/// Druggability score for a binding site
#[derive(Debug, Clone, Default)]
pub struct DruggabilityScore {
    /// Overall druggability (0.0-1.0)
    pub overall: f32,
    /// Volume contribution (0.0-1.0) - sites need 200-800 Å³
    pub volume_score: f32,
    /// Enclosure score (0.0-1.0) - how enclosed the pocket is
    pub enclosure_score: f32,
    /// Hydrophobicity score (0.0-1.0) - dewetting signal strength
    pub hydrophobicity_score: f32,
    /// Aromatic score (0.0-1.0) - proximity to aromatics for pi-stacking
    pub aromatic_score: f32,
    /// Catalytic score (0.0-1.0) - enzyme active site potential
    pub catalytic_score: f32,
    /// Is this likely a druggable pocket?
    pub is_druggable: bool,
}

/// Catalytic residue types for enzyme active site detection
pub const CATALYTIC_RESIDUES: &[&str] = &["GLU", "ASP", "HIS", "HID", "HIE", "HIP", "SER", "CYS", "LYS"];

/// Compute catalytic score from lining residues
///
/// Enzyme active sites typically have:
/// - Multiple catalytic residues (GLU, ASP, HIS, SER, CYS, LYS)
/// - Catalytic residues clustered within 6Å of pocket center
/// - At least 2-3 catalytic residues for enzymatic activity
pub fn compute_catalytic_score(lining_residues: &[LiningResidue]) -> (f32, usize) {
    if lining_residues.is_empty() {
        return (0.0, 0);
    }

    // Count catalytic residues and their proximity
    let mut catalytic_count = 0;
    let mut close_catalytic = 0; // Within 5Å
    let mut total_catalytic_distance = 0.0f32;

    for res in lining_residues {
        if CATALYTIC_RESIDUES.contains(&res.resname.as_str()) {
            catalytic_count += 1;
            total_catalytic_distance += res.min_distance;
            if res.min_distance <= 5.0 {
                close_catalytic += 1;
            }
        }
    }

    if catalytic_count == 0 {
        return (0.0, 0);
    }

    // Score components:
    // 1. Count score: 2-3 catalytic residues is optimal for enzyme activity
    let count_score: f32 = match catalytic_count {
        0 => 0.0,
        1 => 0.3,
        2 => 0.7,
        3 => 1.0,
        4 => 0.95,
        5 => 0.9,
        _ => 0.85, // Many catalytic residues still good
    };

    // 2. Proximity score: closer catalytic residues = more likely active site
    let avg_distance = total_catalytic_distance / catalytic_count as f32;
    let proximity_score: f32 = if avg_distance <= 3.0 {
        1.0
    } else if avg_distance <= 5.0 {
        0.9
    } else if avg_distance <= 7.0 {
        0.7
    } else {
        0.5
    };

    // 3. Close clustering bonus: multiple catalytic residues within 5Å
    let clustering_bonus: f32 = match close_catalytic {
        0 => 0.0,
        1 => 0.1,
        2 => 0.2,
        _ => 0.3,
    };

    // Combined score
    let score: f32 = (0.5 * count_score + 0.3 * proximity_score + 0.2 * clustering_bonus).clamp(0.0, 1.0);

    (score, catalytic_count)
}

impl DruggabilityScore {
    /// Compute druggability from binding site properties (without aromatic info)
    pub fn from_site(volume: f32, avg_intensity: f32, bounding_box: &[f32; 3]) -> Self {
        Self::from_site_with_aromatics(volume, avg_intensity, bounding_box, None)
    }

    /// Compute druggability with aromatic proximity information
    pub fn from_site_with_aromatics(
        volume: f32,
        avg_intensity: f32,
        bounding_box: &[f32; 3],
        aromatic_info: Option<&AromaticProximityInfo>,
    ) -> Self {
        // Volume scoring: optimal range 200-800 Å³
        let volume_score = if volume < 100.0 {
            volume / 100.0 * 0.3  // Too small
        } else if volume < 200.0 {
            0.3 + (volume - 100.0) / 100.0 * 0.4  // Getting better
        } else if volume <= 800.0 {
            0.7 + (1.0 - (volume - 200.0) / 600.0 * 0.3).max(0.7)  // Optimal
        } else if volume <= 1500.0 {
            0.7 - (volume - 800.0) / 700.0 * 0.3  // Large but ok
        } else {
            0.4 - (volume - 1500.0) / 2000.0 * 0.2  // Too large (surface area)
        }.clamp(0.0, 1.0);

        // Enclosure: ratio of volume to bounding box volume
        let bb_volume = bounding_box[0] * bounding_box[1] * bounding_box[2];
        let enclosure_score = if bb_volume > 0.0 {
            (volume / bb_volume).clamp(0.0, 1.0) * 0.7 + 0.3  // Bias toward enclosed
        } else {
            0.0
        };

        // Hydrophobicity: spike intensity indicates dewetting strength
        let hydrophobicity_score = (avg_intensity / 10.0).clamp(0.0, 1.0);

        // Aromatic score: based on proximity to aromatic residues
        // Aromatics enable pi-stacking with drug molecules
        let aromatic_score = aromatic_info
            .map(|info| info.aromatic_score)
            .unwrap_or(0.5);  // Default to neutral if not computed

        // Overall: weighted combination (with aromatics)
        let overall = if aromatic_info.is_some() {
            // With aromatic info: 30% volume, 20% enclosure, 25% hydrophobicity, 25% aromatic
            0.30 * volume_score + 0.20 * enclosure_score + 0.25 * hydrophobicity_score + 0.25 * aromatic_score
        } else {
            // Without aromatic info: original weights
            0.40 * volume_score + 0.30 * enclosure_score + 0.30 * hydrophobicity_score
        };

        // Druggable threshold: overall >= threshold AND volume in reasonable range
        // Bonus: sites with aromatics nearby get lower threshold (pi-stacking potential)
        let aromatic_bonus = aromatic_info
            .map(|info| info.aromatics_within_5a > 0 || info.aromatic_score > 0.4)
            .unwrap_or(false);
        let threshold = if aromatic_bonus { 0.40 } else { 0.48 };
        let is_druggable = overall >= threshold && volume >= 50.0 && volume <= 3000.0;

        Self {
            overall,
            volume_score,
            enclosure_score,
            hydrophobicity_score,
            aromatic_score,
            catalytic_score: 0.0, // Computed separately with lining residues
            is_druggable,
        }
    }

    /// Compute druggability with full context: aromatics AND catalytic residues
    ///
    /// Enzyme active sites are druggable through:
    /// - Substrate mimics (competitive inhibitors)
    /// - Covalent inhibitors (targeting catalytic Ser/Cys)
    /// - Allosteric modulation
    ///
    /// This method properly scores polar enzyme sites that would fail
    /// traditional hydrophobic druggability metrics.
    pub fn from_site_with_catalytic(
        volume: f32,
        avg_intensity: f32,
        bounding_box: &[f32; 3],
        aromatic_info: Option<&AromaticProximityInfo>,
        lining_residues: &[LiningResidue],
    ) -> Self {
        // Start with base scoring
        let mut score = Self::from_site_with_aromatics(volume, avg_intensity, bounding_box, aromatic_info);

        // Compute catalytic score
        let (catalytic_score, catalytic_count) = compute_catalytic_score(lining_residues);
        score.catalytic_score = catalytic_score;

        // For enzyme active sites: adjust scoring to account for polar nature
        if catalytic_count >= 2 && catalytic_score >= 0.5 {
            // This is likely an enzyme active site
            // Recalculate overall with catalytic contribution
            // Enzyme sites: 25% volume, 15% enclosure, 15% hydrophobicity, 20% aromatic, 25% catalytic
            let aromatic = aromatic_info.map(|i| i.aromatic_score).unwrap_or(0.5);
            score.overall = 0.25 * score.volume_score
                + 0.15 * score.enclosure_score
                + 0.15 * score.hydrophobicity_score
                + 0.20 * aromatic
                + 0.25 * catalytic_score;

            // Enzyme active sites with good catalytic score are druggable
            // (substrate mimics, covalent inhibitors, etc.)
            // Note: Large enzyme sites have low volume_score, so use lower threshold
            let enzyme_threshold = 0.35; // Lower threshold for enzyme sites
            // Note: Large multi-subunit enzymes (aldolases, etc.) can have
            // binding sites up to ~8000 Å³. Use < comparison to avoid
            // floating-point boundary issues with clamped volumes.
            score.is_druggable = score.overall >= enzyme_threshold
                && volume >= 50.0
                && volume < 8001.0; // Enzyme sites can be quite large
        }

        score
    }
}

impl AromaticProximityInfo {
    /// Compute aromatic proximity for a site centroid given aromatic residue positions
    ///
    /// # Arguments
    /// * `site_centroid` - [x, y, z] position of binding site center
    /// * `aromatic_positions` - List of (residue_id, aromatic_type, [x, y, z]) for each aromatic
    ///
    /// # Aromatic Types
    /// * 0 = TRP (tryptophan)
    /// * 1 = TYR (tyrosine)
    /// * 2 = PHE (phenylalanine)
    pub fn compute(
        site_centroid: &[f32; 3],
        aromatic_positions: &[(u32, u8, [f32; 3])],
    ) -> Self {
        if aromatic_positions.is_empty() {
            return Self::default();
        }

        let mut nearest_trp: Option<f32> = None;
        let mut nearest_tyr: Option<f32> = None;
        let mut nearest_phe: Option<f32> = None;
        let mut nearest_any = f32::MAX;
        let mut within_5a = 0usize;
        let mut within_8a = 0usize;
        let mut nearby_residues = Vec::new();

        for &(residue_id, aromatic_type, pos) in aromatic_positions {
            let dx = pos[0] - site_centroid[0];
            let dy = pos[1] - site_centroid[1];
            let dz = pos[2] - site_centroid[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();

            // Track nearest by type
            match aromatic_type {
                0 => {  // TRP
                    if nearest_trp.map_or(true, |d| dist < d) {
                        nearest_trp = Some(dist);
                    }
                }
                1 => {  // TYR
                    if nearest_tyr.map_or(true, |d| dist < d) {
                        nearest_tyr = Some(dist);
                    }
                }
                2 => {  // PHE
                    if nearest_phe.map_or(true, |d| dist < d) {
                        nearest_phe = Some(dist);
                    }
                }
                _ => {}
            }

            // Track nearest any
            if dist < nearest_any {
                nearest_any = dist;
            }

            // Count within distance thresholds
            if dist <= 5.0 {
                within_5a += 1;
            }
            if dist <= 8.0 {
                within_8a += 1;
                nearby_residues.push(residue_id);
            }
        }

        // Compute aromatic score based on proximity
        // Higher score for closer aromatics, especially TRP (strongest UV absorber)
        let aromatic_score = Self::compute_aromatic_score(
            nearest_trp,
            nearest_tyr,
            nearest_phe,
            within_5a,
        );

        Self {
            nearest_trp_distance: nearest_trp,
            nearest_tyr_distance: nearest_tyr,
            nearest_phe_distance: nearest_phe,
            nearest_aromatic_distance: if nearest_any < f32::MAX { nearest_any } else { 0.0 },
            aromatics_within_5a: within_5a,
            aromatics_within_8a: within_8a,
            nearby_aromatic_residues: nearby_residues,
            aromatic_score,
        }
    }

    /// Compute aromatic score from distances
    fn compute_aromatic_score(
        nearest_trp: Option<f32>,
        nearest_tyr: Option<f32>,
        nearest_phe: Option<f32>,
        within_5a: usize,
    ) -> f32 {
        // TRP is most important (strongest UV absorber, best for pi-stacking)
        let trp_score = nearest_trp
            .map(|d| Self::distance_to_score(d, 1.5))  // TRP weight 1.5x
            .unwrap_or(0.0);

        // TYR is moderate
        let tyr_score = nearest_tyr
            .map(|d| Self::distance_to_score(d, 1.0))
            .unwrap_or(0.0);

        // PHE is weakest
        let phe_score = nearest_phe
            .map(|d| Self::distance_to_score(d, 0.7))
            .unwrap_or(0.0);

        // Combine scores (take best + bonus for multiple)
        let base_score = trp_score.max(tyr_score).max(phe_score);
        let multi_bonus = (within_5a as f32 * 0.05).min(0.2);  // Up to 0.2 bonus

        (base_score + multi_bonus).clamp(0.0, 1.0)
    }

    /// Convert distance to score (closer = higher)
    fn distance_to_score(distance: f32, weight: f32) -> f32 {
        if distance < 3.0 {
            // Direct contact: highest score
            weight * 1.0
        } else if distance < 5.0 {
            // Close proximity: good score
            weight * (1.0 - (distance - 3.0) / 2.0 * 0.3)
        } else if distance < 8.0 {
            // Medium range: moderate score
            weight * (0.7 - (distance - 5.0) / 3.0 * 0.3)
        } else {
            // Distal: low score
            weight * (0.4 - (distance - 8.0) / 10.0 * 0.4).max(0.0)
        }
    }
}

/// Classification of detected binding site
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiteClassification {
    /// Traditional active site (large, persistent)
    ActiveSite,
    /// Allosteric binding site (remote from active site)
    Allosteric,
    /// Cryptic site (only appears transiently)
    Cryptic,
    /// Protein-protein interaction surface
    PpiSurface,
    /// Membrane interface region
    MembraneInterface,
    /// Unclassified
    Unknown,
}

impl Default for SiteClassification {
    fn default() -> Self {
        Self::Unknown
    }
}

impl SiteClassification {
    /// Classify based on spike patterns and volume
    pub fn from_properties(spike_count: usize, volume: f32, _avg_intensity: f32) -> Self {
        if volume >= 400.0 && spike_count >= 50 {
            Self::ActiveSite
        } else if volume >= 200.0 && volume <= 600.0 && spike_count >= 20 {
            Self::Cryptic  // Moderate size, fewer spikes → transient
        } else if volume >= 150.0 && spike_count >= 10 {
            Self::Allosteric
        } else if volume >= 800.0 {
            Self::PpiSurface  // Large surface areas
        } else {
            Self::Unknown
        }
    }
}

/// Statistics from RT clustering
#[derive(Debug, Clone)]
pub struct ClusteringStats {
    /// Number of clusters found
    pub num_clusters: usize,
    /// Total neighbor pairs examined
    pub total_neighbors: usize,
    /// GPU time in milliseconds
    pub gpu_time_ms: f64,
    /// Whether RT cores were used (vs fallback)
    pub used_rt_cores: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// MULTI-SCALE CLUSTERING
// ═══════════════════════════════════════════════════════════════════════════════

/// Cluster found at a single scale (epsilon value)
#[derive(Debug, Clone)]
pub struct ScaleCluster {
    /// Cluster centroid [x, y, z]
    pub centroid: [f32; 3],
    /// Number of spikes in this cluster
    pub spike_count: usize,
    /// Spike indices belonging to this cluster
    pub spike_indices: Vec<usize>,
    /// Epsilon value (scale) at which this cluster was found
    pub epsilon: f32,
    /// Original cluster ID from DBSCAN at this scale
    pub original_cluster_id: i32,
    /// Persistence score (how many scales this cluster appears at)
    pub persistence: usize,
}

/// Merged cluster from multiple scales
#[derive(Debug, Clone)]
pub struct MergedCluster {
    /// Merged centroid (average across scales)
    pub centroid: [f32; 3],
    /// Total unique spikes across all scales
    pub spike_count: usize,
    /// Union of spike indices from all contributing scale clusters
    pub spike_indices: Vec<usize>,
    /// Persistence: number of scales this cluster appears at
    pub persistence: usize,
    /// List of epsilon values where this cluster was detected
    pub scales: Vec<f32>,
}

/// Result of multi-scale clustering
#[derive(Debug, Clone)]
pub struct MultiScaleClusteringResult {
    /// Merged clusters sorted by confidence (persistence × spike_count)
    pub clusters: Vec<MergedCluster>,
    /// Number of epsilon scales used
    pub total_scales: usize,
    /// Epsilon values tested
    pub epsilon_values: Vec<f32>,
    /// Whether adaptive epsilon was used (vs fixed)
    pub adaptive_epsilon: bool,
    /// k value used for k-NN (if adaptive)
    pub knn_k: Option<usize>,
    /// Number of spikes sampled for k-NN (if adaptive)
    pub num_spikes_sampled: Option<usize>,
}

impl MultiScaleClusteringResult {
    /// Convert to cluster IDs array (for compatibility with single-scale API)
    ///
    /// Returns cluster ID for each spike position, using the merged cluster assignments.
    /// Spikes not in any persistent cluster get -1 (noise).
    pub fn to_cluster_ids(&self, num_spikes: usize) -> Vec<i32> {
        let mut cluster_ids = vec![-1i32; num_spikes];

        for (cluster_idx, cluster) in self.clusters.iter().enumerate() {
            for &spike_idx in &cluster.spike_indices {
                if spike_idx < num_spikes {
                    cluster_ids[spike_idx] = cluster_idx as i32;
                }
            }
        }

        cluster_ids
    }

    /// Get the number of persistent clusters
    pub fn num_clusters(&self) -> usize {
        self.clusters.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SITE PERSISTENCE TRACKING
// ═══════════════════════════════════════════════════════════════════════════════

/// Tracks binding site persistence across trajectory frames
///
/// Sites that persist across multiple frames are more reliable drug targets.
/// Transient sites may be cryptic pockets that only appear during conformational changes.
#[derive(Debug, Clone)]
pub struct SitePersistenceTracker {
    /// Tracked sites by unique ID
    tracked_sites: Vec<TrackedSite>,
    /// Spatial matching threshold (Å)
    match_threshold: f32,
    /// Total frames processed
    total_frames: usize,
    /// Next unique site ID
    next_site_id: u64,
}

/// A binding site being tracked across frames
#[derive(Debug, Clone)]
pub struct TrackedSite {
    /// Unique ID for this site across all frames
    pub site_id: u64,
    /// Running average centroid position
    pub avg_centroid: [f32; 3],
    /// Running average volume
    pub avg_volume: f32,
    /// Number of frames this site was detected
    pub frame_count: usize,
    /// First frame this site appeared
    pub first_frame: usize,
    /// Most recent frame this site was seen
    pub last_frame: usize,
    /// Consecutive frames detected (current streak)
    pub consecutive_frames: usize,
    /// Maximum consecutive detection streak
    pub max_consecutive: usize,
    /// Running average spike count
    pub avg_spike_count: f32,
    /// Running average quality score
    pub avg_quality: f32,
    /// Is this site considered persistent? (detected in >50% of frames)
    pub is_persistent: bool,
    /// Site classification (most common across frames)
    pub classification: SiteClassification,
    /// Frame-by-frame spike counts for variability analysis
    spike_history: Vec<usize>,
}

/// Result of persistence analysis
#[derive(Debug, Clone)]
pub struct PersistenceAnalysis {
    /// Total frames analyzed
    pub total_frames: usize,
    /// All tracked sites with persistence info
    pub sites: Vec<TrackedSite>,
    /// Number of persistent sites (>50% frame presence)
    pub persistent_count: usize,
    /// Number of transient sites (<50% frame presence)
    pub transient_count: usize,
    /// Average site lifetime (frames)
    pub avg_lifetime: f32,
}

impl SitePersistenceTracker {
    /// Create a new persistence tracker
    ///
    /// # Arguments
    /// * `match_threshold` - Maximum distance (Å) for sites to be considered the same
    pub fn new(match_threshold: f32) -> Self {
        Self {
            tracked_sites: Vec::new(),
            match_threshold,
            total_frames: 0,
            next_site_id: 0,
        }
    }

    /// Process binding sites from a single frame
    ///
    /// Matches sites to existing tracked sites or creates new ones.
    pub fn process_frame(&mut self, frame_sites: &[ClusteredBindingSite]) {
        self.total_frames += 1;
        let current_frame = self.total_frames;

        // Mark all sites as not-seen-this-frame
        let mut matched = vec![false; self.tracked_sites.len()];

        for site in frame_sites {
            // Find best matching tracked site
            let mut best_match: Option<usize> = None;
            let mut best_dist = f32::MAX;

            for (idx, tracked) in self.tracked_sites.iter().enumerate() {
                if matched[idx] {
                    continue;
                }

                // Site tracker matches on GeometricVoxelMass: the tracked
                // average centroid is computed from successive
                // GeometricVoxelMass views (see update_tracked_site). No
                // hidden default; view is explicit.
                let site_gvm = site
                    .view(crate::spatial_view::SpatialView::GeometricVoxelMass)
                    .expect("ClusteredBindingSite always carries GeometricVoxelMass view");
                let dist = Self::distance(&site_gvm, &tracked.avg_centroid);
                if dist < self.match_threshold && dist < best_dist {
                    best_match = Some(idx);
                    best_dist = dist;
                }
            }

            if let Some(idx) = best_match {
                // Update existing site
                self.update_tracked_site(idx, site, current_frame);
                matched[idx] = true;
            } else {
                // Create new tracked site
                self.create_tracked_site(site, current_frame);
            }
        }

        // Update sites not seen this frame (break consecutive streak)
        for (idx, was_matched) in matched.iter().enumerate() {
            if !was_matched {
                self.tracked_sites[idx].consecutive_frames = 0;
            }
        }

        // Update persistence status for all sites
        self.update_persistence_status();
    }

    /// Update an existing tracked site with new detection
    fn update_tracked_site(&mut self, idx: usize, site: &ClusteredBindingSite, frame: usize) {
        let tracked = &mut self.tracked_sites[idx];

        // Update running averages over the GeometricVoxelMass view —
        // the tracker explicitly averages that representation; no
        // implicit view selection.
        let n = tracked.frame_count as f32;
        let site_gvm = site
            .view(crate::spatial_view::SpatialView::GeometricVoxelMass)
            .expect("ClusteredBindingSite always carries GeometricVoxelMass view");
        tracked.avg_centroid[0] = (tracked.avg_centroid[0] * n + site_gvm[0]) / (n + 1.0);
        tracked.avg_centroid[1] = (tracked.avg_centroid[1] * n + site_gvm[1]) / (n + 1.0);
        tracked.avg_centroid[2] = (tracked.avg_centroid[2] * n + site_gvm[2]) / (n + 1.0);
        tracked.avg_volume = (tracked.avg_volume * n + site.estimated_volume) / (n + 1.0);
        tracked.avg_spike_count = (tracked.avg_spike_count * n + site.spike_count as f32) / (n + 1.0);
        tracked.avg_quality = (tracked.avg_quality * n + site.quality_score) / (n + 1.0);

        tracked.frame_count += 1;
        tracked.last_frame = frame;
        tracked.consecutive_frames += 1;
        tracked.max_consecutive = tracked.max_consecutive.max(tracked.consecutive_frames);
        tracked.spike_history.push(site.spike_count);

        // Keep most common classification (or use most recent if better quality)
        if site.quality_score > tracked.avg_quality * 1.2 {
            tracked.classification = site.classification;
        }
    }

    /// Create a new tracked site
    fn create_tracked_site(&mut self, site: &ClusteredBindingSite, frame: usize) {
        let site_id = self.next_site_id;
        self.next_site_id += 1;

        let site_gvm = site
            .view(crate::spatial_view::SpatialView::GeometricVoxelMass)
            .expect("ClusteredBindingSite always carries GeometricVoxelMass view");
        self.tracked_sites.push(TrackedSite {
            site_id,
            // Seed tracker average from the GeometricVoxelMass view.
            avg_centroid: site_gvm,
            avg_volume: site.estimated_volume,
            frame_count: 1,
            first_frame: frame,
            last_frame: frame,
            consecutive_frames: 1,
            max_consecutive: 1,
            avg_spike_count: site.spike_count as f32,
            avg_quality: site.quality_score,
            is_persistent: false,
            classification: site.classification,
            spike_history: vec![site.spike_count],
        });
    }

    /// Update persistence status for all sites
    fn update_persistence_status(&mut self) {
        let threshold = self.total_frames as f32 * 0.5;  // 50% of frames

        for site in &mut self.tracked_sites {
            site.is_persistent = site.frame_count as f32 >= threshold;
        }
    }

    /// Get final persistence analysis
    pub fn analyze(&self) -> PersistenceAnalysis {
        let persistent_count = self.tracked_sites.iter().filter(|s| s.is_persistent).count();
        let transient_count = self.tracked_sites.len() - persistent_count;

        let avg_lifetime = if self.tracked_sites.is_empty() {
            0.0
        } else {
            self.tracked_sites.iter()
                .map(|s| s.frame_count as f32)
                .sum::<f32>() / self.tracked_sites.len() as f32
        };

        PersistenceAnalysis {
            total_frames: self.total_frames,
            sites: self.tracked_sites.clone(),
            persistent_count,
            transient_count,
            avg_lifetime,
        }
    }

    /// Get tracked sites sorted by persistence (most persistent first)
    pub fn get_persistent_sites(&self) -> Vec<&TrackedSite> {
        let mut sites: Vec<_> = self.tracked_sites.iter().collect();
        sites.sort_by(|a, b| {
            // Sort by frame_count descending, then by avg_quality descending
            b.frame_count.cmp(&a.frame_count)
                .then_with(|| b.avg_quality.partial_cmp(&a.avg_quality).unwrap_or(std::cmp::Ordering::Equal))
        });
        sites
    }

    /// Calculate Euclidean distance between two positions
    fn distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}

impl TrackedSite {
    /// Calculate spike count variability (coefficient of variation)
    pub fn spike_variability(&self) -> f32 {
        if self.spike_history.len() < 2 {
            return 0.0;
        }

        let mean = self.avg_spike_count;
        if mean < 0.001 {
            return 0.0;
        }

        let variance: f32 = self.spike_history.iter()
            .map(|&x| (x as f32 - mean).powi(2))
            .sum::<f32>() / self.spike_history.len() as f32;

        variance.sqrt() / mean  // CV
    }

    /// Get frame presence ratio (0.0 to 1.0)
    pub fn presence_ratio(&self, total_frames: usize) -> f32 {
        if total_frames == 0 {
            return 0.0;
        }
        self.frame_count as f32 / total_frames as f32
    }
}

/// Persistent engine that keeps GPU state alive across structures
#[cfg(feature = "gpu")]
pub struct PersistentNhsEngine {
    /// Shared CUDA context (kept alive)
    context: Arc<CudaContext>,
    /// Compiled module (kept alive)
    module: Arc<CudaModule>,
    /// Stream for operations
    stream: Arc<CudaStream>,

    /// Currently loaded engine instance
    engine: Option<NhsAmberFusedEngine>,

    /// RT-accelerated clustering engine (lazy initialized)
    rt_engine: Option<crate::rt_clustering::RtClusteringEngine>,

    /// GPU spatial-hash CCL clustering backend (lazy initialized; Phase P0x).
    /// When present, `cluster_spikes` dispatches here on SM120+ instead of
    /// the `fallback_grid_cluster` CPU path. See `gpu_cluster_backend.rs`.
    gpu_cluster: Option<crate::gpu_cluster_backend::GpuSpatialHashBackend>,

    /// Pre-allocated buffer capacity
    max_atoms: usize,

    /// Grid configuration
    grid_dim: usize,
    grid_spacing: f32,

    /// Current topology ID
    current_topology_id: Option<String>,

    /// Initialization time tracking
    context_init_time_ms: u64,
    module_init_time_ms: u64,

    /// RT engine initialization time (if initialized)
    rt_init_time_ms: Option<u64>,

    /// Cumulative statistics
    structures_processed: usize,
    total_steps_run: i64,
    total_compute_time_ms: u64,
}

#[cfg(feature = "gpu")]
impl PersistentNhsEngine {
    /// Create persistent engine with pre-allocated capacity
    pub fn new(config: &PersistentBatchConfig) -> Result<Self> {
        log::info!("🚀 Initializing Persistent NHS Engine (max_atoms: {})", config.max_atoms);

        // Time context creation
        let ctx_start = Instant::now();
        let context = CudaContext::new(0)
            .context("Failed to create CUDA context")?;
        let context_init_time_ms = ctx_start.elapsed().as_millis() as u64;
        log::info!("  CUDA context: {}ms", context_init_time_ms);

        // Time module loading
        let mod_start = Instant::now();

        // Try multiple PTX locations
        let ptx_candidates = [
            "../prism-gpu/src/kernels/nhs_amber_fused.ptx",  // From workspace
            "crates/prism-gpu/src/kernels/nhs_amber_fused.ptx",  // From root
            "target/ptx/nhs_amber_fused.ptx",  // Build output
        ];

        let ptx_path = ptx_candidates.iter()
            .find(|p| Path::new(p).exists())
            .ok_or_else(|| anyhow::anyhow!("nhs_amber_fused.ptx not found in any standard location"))?;

        let module = context
            .load_module(Ptx::from_file(ptx_path))
            .context("Failed to load NHS-AMBER fused PTX")?;
        let module_init_time_ms = mod_start.elapsed().as_millis() as u64;
        log::info!("  PTX module: {}ms", module_init_time_ms);

        let stream = context.default_stream();

        log::info!("✅ Persistent engine ready (total init: {}ms)",
            context_init_time_ms + module_init_time_ms);

        Ok(Self {
            context,
            module,
            stream,
            engine: None,
            rt_engine: None,  // Lazy initialized on first use
            gpu_cluster: None, // Lazy initialized on first cluster_spikes call
            max_atoms: config.max_atoms,
            grid_dim: config.grid_dim,
            grid_spacing: config.grid_spacing,
            current_topology_id: None,
            context_init_time_ms,
            module_init_time_ms,
            rt_init_time_ms: None,
            structures_processed: 0,
            total_steps_run: 0,
            total_compute_time_ms: 0,
        })
    }

    /// Create persistent engine on an explicit CUDA stream (for multi-stream concurrency).
    /// Shares context + module with other engines — each gets its own stream for GPU overlap.
    pub fn new_on_stream(
        config: &PersistentBatchConfig,
        context: Arc<CudaContext>,
        module: Arc<CudaModule>,
        stream: Arc<CudaStream>,
    ) -> Result<Self> {
        log::info!("Persistent NHS Engine on dedicated stream (max_atoms: {})", config.max_atoms);
        Ok(Self {
            context,
            module,
            stream,
            engine: None,
            rt_engine: None,
            gpu_cluster: None, // Lazy initialized on first cluster_spikes call
            max_atoms: config.max_atoms,
            grid_dim: config.grid_dim,
            grid_spacing: config.grid_spacing,
            current_topology_id: None,
            context_init_time_ms: 0,
            module_init_time_ms: 0,
            rt_init_time_ms: None,
            structures_processed: 0,
            total_steps_run: 0,
            total_compute_time_ms: 0,
        })
    }

    /// Access the shared CUDA context (for creating additional streams)
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.context
    }

    /// Access the compiled PTX module (for sharing across engines)
    pub fn cuda_module(&self) -> &Arc<CudaModule> {
        &self.module
    }

    /// Load a new topology (hot-swap)
    ///
    /// If the new topology fits in existing buffers, reuses them.
    /// Otherwise, reallocates with appropriate capacity.
    pub fn load_topology(&mut self, topology: &PrismPrepTopology) -> Result<()> {
        let topo_id = std::path::Path::new(&topology.source_pdb)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        log::info!("📦 Loading topology: {} ({} atoms)", topo_id, topology.n_atoms);

        let load_start = Instant::now();

        // Check if we need to reallocate
        if topology.n_atoms > self.max_atoms {
            log::warn!("  Structure exceeds max_atoms ({}), reallocating to {}",
                self.max_atoms, topology.n_atoms + 1000);
            self.max_atoms = topology.n_atoms + 1000;
        }

        // Create new engine instance with shared context
        // Note: In a more optimized version, we would reuse GPU buffers
        // For now, we benefit from shared context + module
        let engine = NhsAmberFusedEngine::new_on_stream(
            self.context.clone(),
            self.stream.clone(),
            topology,
            self.grid_dim,
            self.grid_spacing,
        )?;

        self.engine = Some(engine);
        self.current_topology_id = Some(topo_id.clone());

        let load_time = load_start.elapsed().as_millis() as u64;
        log::info!("  Topology loaded: {}ms", load_time);

        Ok(())
    }

    /// **Configure unified cryo-UV protocol (RECOMMENDED)**
    ///
    /// Sets the integrated cryo-thermal + UV-LIF protocol for the current topology.
    /// This is the canonical PRISM4D cryptic site detection method.
    pub fn set_cryo_uv_protocol(&mut self, protocol: CryoUvProtocol) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.set_cryo_uv_protocol(protocol)?;
            Ok(())
        } else {
            bail!("No topology loaded")
        }
    }

    /// Enable adaptive dt.
    pub fn set_adaptive_dt(&mut self, enabled: bool) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.set_adaptive_dt(enabled);
            Ok(())
        } else {
            bail!("No topology loaded")
        }
    }

    /// Telemetry observer — current host-side integration timestep (ps).
    /// Returns 0.0 if no topology loaded.
    pub fn current_dt_ps(&self) -> f64 {
        self.engine.as_ref().map(|e| e.current_dt_ps()).unwrap_or(0.0)
    }

    /// Telemetry observer — base (pre-adaptive-scale) integration timestep (ps).
    /// Returns 0.0 if no topology loaded.
    pub fn base_dt_ps(&self) -> f64 {
        self.engine.as_ref().map(|e| e.base_dt_ps()).unwrap_or(0.0)
    }

    /// Telemetry observer — true while host-side adaptive_dt heuristic
    /// is the writer of `self.dt` (i.e., not in V2 gearbox mode).
    pub fn adaptive_dt_enabled(&self) -> bool {
        self.engine.as_ref().map(|e| e.adaptive_dt_enabled()).unwrap_or(false)
    }

    /// Telemetry observer — true while the V2 captured pipeline owns dt.
    pub fn is_gearbox_active(&self) -> bool {
        self.engine.as_ref().map(|e| e.is_gearbox_active()).unwrap_or(false)
    }

    /// Enable LADD observation kernel
    pub fn set_ladd_enabled(&mut self, enabled: bool) {
        if let Some(ref mut engine) = self.engine {
            engine.set_ladd_enabled(enabled);
        }
    }

    /// ASC steering: write focus residue to GPU ProtocolState.
    /// The Director kernel will read this and modulate UV targeting.
    pub fn set_steering_focus_residue(&mut self, residue_id: i32) {
        if let Some(ref mut engine) = self.engine {
            engine.set_steering_focus_residue(residue_id);
        }
    }

    /// Force-sync spike data from GPU after short run chunks.
    /// Returns number of new spikes downloaded.
    pub fn force_spike_sync(&mut self) -> anyhow::Result<usize> {
        if let Some(ref mut engine) = self.engine {
            engine.force_spike_sync()
        } else {
            Ok(0)
        }
    }

    /// Load NMA modes from JSON and upload to GPU for NMA-biased perturbation.
    pub fn load_nma_modes(&mut self, path: &str) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.load_nma_modes(path)
        } else {
            bail!("No topology loaded")
        }
    }

    /// Set NMA amplification factor.
    pub fn set_nma_amplification(&mut self, amp: f32) {
        if let Some(ref mut engine) = self.engine {
            engine.set_nma_amplification(amp);
        }
    }

    /// Set NMA scan fraction.
    pub fn set_nma_scan_fraction(&mut self, frac: f32) {
        if let Some(ref mut engine) = self.engine {
            engine.set_nma_scan_fraction(frac);
        }
    }

    /// Get the current spike count from the device.
    pub fn get_spike_count(&self) -> Result<u32> {
        if let Some(ref engine) = self.engine {
            engine.get_spike_count()
        } else {
            Ok(0)
        }
    }

    /// Get the number of accumulated spikes without cloning the vector.
    pub fn accumulated_spike_count(&self) -> usize {
        if let Some(ref engine) = self.engine {
            engine.get_accumulated_spikes().len()
        } else {
            0
        }
    }

    // ── PRISM-TWIN threshold access (Step 2) ──

    /// Get mutable reference to per-voxel neuron threshold buffer.
    pub fn threshold_buffer_mut(&mut self) -> Option<&mut CudaSlice<f32>> {
        self.engine.as_mut().map(|e| e.threshold_buffer_mut())
    }

    /// Get reference to base (initial) threshold buffer.
    pub fn base_threshold_buffer(&self) -> Option<&CudaSlice<f32>> {
        self.engine.as_ref().map(|e| e.base_threshold_buffer())
    }

    /// Get both threshold buffers together (avoids double-borrow).
    /// Returns (mutable_threshold, base_threshold) if engine is loaded.
    pub fn threshold_buffers_mut(&mut self) -> Option<(&mut CudaSlice<f32>, &CudaSlice<f32>)> {
        self.engine.as_mut().map(|e| e.threshold_buffers_mut())
    }

    /// Get grid geometry (dim_x, dim_y, dim_z, origin_x, origin_y, origin_z, spacing).
    pub fn grid_info(&self) -> Option<(i32, i32, i32, f32, f32, f32, f32)> {
        self.engine.as_ref().map(|e| e.grid_info())
    }

    /// Total voxels (dim³).
    pub fn total_voxels(&self) -> usize {
        self.engine.as_ref().map(|e| e.total_voxels()).unwrap_or(0)
    }

    /// Get reference to raw spike buffer on GPU.
    pub fn spike_buffer_gpu(&self) -> Option<&CudaSlice<u8>> {
        self.engine.as_ref().map(|e| e.spike_buffer_gpu())
    }

    /// Get reference to spike count buffer on GPU (for persistent coupling kernel).
    pub fn spike_count_gpu(&self) -> Option<&CudaSlice<i32>> {
        self.engine.as_ref().map(|e| e.spike_count_gpu())
    }

    /// Get a reference to the engine's CUDA stream.
    /// Used by CUDA Graph stream capture to record kernel launches.
    pub fn cuda_stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// T6: raw device pointer to the `d_forces` buffer (n_atoms × 3 f32, AoS).
    /// Returns 0 if no topology is loaded. Pointer-stable for the campaign.
    pub fn d_forces_dev_ptr(&self) -> u64 {
        self.engine.as_ref()
            .map(|e| e.d_forces_dev_ptr(&self.stream))
            .unwrap_or(0)
    }

    /// **M1.2.20.C-A / Ruling 6** — raw device pointer to the per-atom
    /// AMBER mass buffer (`f32 [n_atoms]`).  Returns 0 if no topology
    /// is loaded.  Pointer-stable for the campaign.  Consumed by the
    /// gradient gasp kernel and the Phase-3 Momentum Guard.
    pub fn d_masses_dev_ptr(&self) -> u64 {
        self.engine.as_ref()
            .map(|e| e.d_masses_dev_ptr(&self.stream))
            .unwrap_or(0)
    }

    /// T6: raw device pointer to the `d_positions` buffer (n_atoms × 3 f32, AoS).
    /// Returns 0 if no topology is loaded. Pointer-stable for the campaign.
    pub fn d_positions_dev_ptr(&self) -> u64 {
        self.engine.as_ref()
            .map(|e| e.d_positions_dev_ptr(&self.stream))
            .unwrap_or(0)
    }

    /// Number of atoms in the currently loaded topology. Returns 0 if none.
    pub fn n_atoms_loaded(&self) -> usize {
        self.engine.as_ref().map(|e| e.n_atoms()).unwrap_or(0)
    }

    // ── M1.2.18-P1 — Forwarders for the V2 captured-pipeline build ──
    //
    // The five methods below are simple `engine.as_ref()/as_mut()` forwarders
    // that bridge the production `nhs_rt_full --features=v2_ignition` build
    // (which uses `PersistentNhsEngine`) to the inner `NhsAmberFusedEngine`
    // accessors added in the B.3 / M1.2.17 waves.  Without these, the
    // captured-pipeline wire-up at nhs_rt_full.rs:5010+ fails to link.

    /// **B.3** raw device pointer to the `d_velocities` buffer.
    /// Forwards to inner `NhsAmberFusedEngine::d_velocities_dev_ptr`.
    pub fn d_velocities_dev_ptr(&self) -> u64 {
        self.engine.as_ref()
            .map(|e| e.d_velocities_dev_ptr(&self.stream))
            .unwrap_or(0)
    }

    /// **B.3** raw device pointer to `d_protocol->dt` (offset 84 within
    /// the GPU-resident ProtocolState).  Forwarder.
    pub fn d_protocol_dt_dev_ptr(&self) -> u64 {
        self.engine.as_ref()
            .map(|e| e.d_protocol_dt_dev_ptr(&self.stream))
            .unwrap_or(0)
    }

    /// **M1.2.17** raw device pointer to the per-atom potential-energy
    /// components buffer (n_atoms × f64).  CUB DeviceReduce::Sum reads
    /// from this address.  Forwarder.
    pub fn d_potential_energy_components_dev_ptr(&self) -> u64 {
        self.engine.as_ref()
            .map(|e| e.d_potential_energy_components_dev_ptr(&self.stream))
            .unwrap_or(0)
    }

    /// **M1.2.17** atom count for the per-atom PE buffer (= length of
    /// d_potential_energy_components / sizeof(f64)).  Forwarder.
    pub fn d_potential_energy_n_atoms(&self) -> usize {
        self.engine.as_ref()
            .map(|e| e.d_potential_energy_n_atoms())
            .unwrap_or(0)
    }

    /// **M1.2.18.5** raw device pointer to the 1 × f64 W_ext accumulator.
    /// The captured pipeline writes this address into the FFI struct's
    /// `d_external_work` field at offset 128 pre-capture and emits a
    /// cuMemsetD8Async at the head of every replay window so each chunk
    /// starts with `*d_external_work == 0.0`.  Forwarder to inner
    /// `NhsAmberFusedEngine::allocate_external_work_buffer`.
    pub fn allocate_external_work_buffer(&self) -> u64 {
        self.engine.as_ref()
            .map(|e| e.allocate_external_work_buffer(&self.stream))
            .unwrap_or(0)
    }

    /// **B.3.1** Gearbox > Adaptive-DT hierarchy gate.  When the V2
    /// captured pipeline is live the legacy `--adaptive-dt` host write
    /// path is bypassed.  Forwarder.
    pub fn set_gearbox_active(&mut self, active: bool) {
        if let Some(ref mut e) = self.engine {
            e.set_gearbox_active(active);
        }
    }

    /// Get ALL GPU buffer references needed by the persistent coupling kernel.
    pub fn twin_coupling_gpu_state(&mut self) -> Option<(
        &mut CudaSlice<f32>,
        &CudaSlice<f32>,
        &CudaSlice<u8>,
        &CudaSlice<i32>,
    )> {
        self.engine.as_mut().map(|e| e.twin_coupling_gpu_state())
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PRISM-TWIN v3.0 Gate 3: Autonomous GPU execution delegation
    // ═══════════════════════════════════════════════════════════════════════

    /// Launch one complete step's GPU kernels with zero CPU memcpy.
    /// Director → Physics → CA restraints → multi_lif → heartbeat → coupling clear.
    /// This is the graph-capturable step function.
    pub fn step_autonomous_kernels(&mut self) -> Result<()> {
        let stream = self.stream.clone();
        if let Some(ref mut engine) = self.engine {
            engine.step_autonomous_kernels(&stream)
        } else {
            bail!("No topology loaded — call load_topology() first")
        }
    }

    /// Get reference to GPU-resident ProtocolState (684 bytes after Stage 2).
    /// Used for heartbeat polling and graph capture.
    pub fn protocol_state_buffer(&self) -> Option<&CudaSlice<u8>> {
        self.engine.as_ref().map(|e| &e.d_protocol_state)
    }

    /// Mutable protocol state buffer for GPU-side ASC steering writes.
    pub fn protocol_state_buffer_mut(&mut self) -> Option<&mut CudaSlice<u8>> {
        self.engine.as_mut().map(|e| &mut e.d_protocol_state)
    }

    /// Get BOTH threshold buffers AND protocol state in one borrow — for GPU ring buffer coupling.
    /// Returns (osc_thresholds, base_thresholds, protocol_state) all as mutable.
    pub fn coupling_buffers_mut(&mut self) -> Option<(&mut CudaSlice<f32>, &CudaSlice<f32>, &mut CudaSlice<u8>)> {
        if let Some(ref mut engine) = self.engine {
            engine.coupling_buffers_for_twin()
        } else {
            None
        }
    }

    /// Capture the autonomous physics step as a CUDA Graph for replay.
    /// Pre-synchronizes the stream to flush all prior work (module loads, memcpy)
    /// before entering capture mode — this prevents CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED.
    /// Amendment 3.4 monolithic-fusion variant. Captures the autonomous
    /// kernel sequence into a raw `CUgraph` template (NOT instantiated)
    /// with named node handles for "director", "fused_step", "multi_lif"
    /// recorded by a `StreamCaptureTagger`. The caller can splice in child
    /// graphs via `CapturedTemplate::add_child_graph_node` before
    /// instantiating with `template.instantiate()`.
    ///
    /// Bypasses `cudarc::driver::safe::CudaGraph` entirely; uses raw
    /// `cuStreamBeginCapture_v2` / `cuStreamEndCapture` from
    /// `cudarc::driver::sys` so the template handle stays alive
    /// post-capture.
    pub fn capture_autonomous_template(&mut self)
        -> anyhow::Result<crate::graph_capture::CapturedTemplate>
    {
        use cudarc::driver::sys;
        use crate::graph_capture::{StreamCaptureTagger, CapturedTemplate};
        let stream = self.stream.clone();

        // Same pre-capture posture as the legacy path: bind context, sync.
        stream.context().bind_to_thread()
            .map_err(|e| anyhow::anyhow!("Pre-capture context bind: {:?}", e))?;
        stream.synchronize()
            .map_err(|e| anyhow::anyhow!("Pre-capture sync: {:?}", e))?;

        let engine = self.engine.as_mut()
            .ok_or_else(|| anyhow::anyhow!("No engine loaded"))?;

        // Capture-mode flag for cudarc's bind_to_thread no-op shim.
        cudarc::driver::set_capture_mode_active(true);
        struct CaptureGuard;
        impl Drop for CaptureGuard {
            fn drop(&mut self) {
                cudarc::driver::set_capture_mode_active(false);
            }
        }
        let _guard = CaptureGuard;

        // ── Begin capture (raw sys, RELAXED mode matches legacy path) ──
        let raw_stream = stream.cu_stream();
        let rc = unsafe {
            sys::cuStreamBeginCapture_v2(
                raw_stream,
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            )
        };
        if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
            anyhow::bail!("cuStreamBeginCapture_v2 failed: {:?}", rc);
        }

        // Tagger snapshots the CUgraphNode handle after each kernel.
        let mut tagger = StreamCaptureTagger::new(raw_stream);

        // Launch the captured kernel sequence. On error, abort capture
        // cleanly and propagate.
        let kernel_result = engine.step_autonomous_kernels_tagged(
            &stream,
            Some(&mut tagger),
        );
        if let Err(e) = kernel_result {
            // Abort capture: end and discard the template.
            let mut discard: sys::CUgraph = std::ptr::null_mut();
            unsafe { let _ = sys::cuStreamEndCapture(raw_stream, &mut discard); }
            if !discard.is_null() {
                unsafe { let _ = sys::cuGraphDestroy(discard); }
            }
            return Err(anyhow::anyhow!("step_autonomous_kernels_tagged: {}", e));
        }

        // ── End capture → raw CUgraph template ─────────────────────────
        let mut cu_graph: sys::CUgraph = std::ptr::null_mut();
        let rc = unsafe { sys::cuStreamEndCapture(raw_stream, &mut cu_graph) };
        if !matches!(rc, sys::CUresult::CUDA_SUCCESS) {
            anyhow::bail!("cuStreamEndCapture failed: {:?}", rc);
        }
        if cu_graph.is_null() {
            anyhow::bail!("cuStreamEndCapture produced null template");
        }
        Ok(CapturedTemplate::from_capture(
            cu_graph,
            tagger.into_registry(),
            stream,
        ))
    }

    pub fn capture_autonomous_graph(&mut self) -> anyhow::Result<crate::graph_capture::AutonomousGraph> {
        use cudarc::driver::sys;
        let stream = self.stream.clone();

        // CRITICAL: sync the stream and bind the context BEFORE capture to flush all
        // prior module loads, memcpy, and initialization. CUDA capture mode forbids
        // context management calls (cuCtxGetCurrent/cuCtxSetCurrent), so the context
        // must already be bound before we enter capture mode.
        stream.context().bind_to_thread()
            .map_err(|e| anyhow::anyhow!("Pre-capture context bind: {:?}", e))?;
        stream.synchronize()
            .map_err(|e| anyhow::anyhow!("Pre-capture sync: {:?}", e))?;

        if let Some(ref mut engine) = self.engine {
            // Enable capture mode flag — cudarc's bind_to_thread() becomes a no-op.
            // See vendor/cudarc/src/driver/safe/core.rs for the patch.
            cudarc::driver::set_capture_mode_active(true);

            // Helper to always clear the flag, even on error paths.
            struct CaptureGuard;
            impl Drop for CaptureGuard {
                fn drop(&mut self) {
                    cudarc::driver::set_capture_mode_active(false);
                }
            }
            let _guard = CaptureGuard;

            // Begin stream capture
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                .map_err(|e| anyhow::anyhow!("Stream capture begin: {:?}", e))?;

            // Launch one step's kernels (captured, not executed)
            match engine.step_autonomous_kernels(&stream) {
                Ok(()) => {},
                Err(e) => {
                    log::warn!("step_autonomous: {} — aborting capture", e);
                    let _ = stream.end_capture(
                        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
                    );
                    return Err(anyhow::anyhow!("step_autonomous: {}", e));
                }
            }

            // End capture → instantiate
            let graph = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH
            ).map_err(|e| anyhow::anyhow!("Stream capture end: {:?}", e))?
             .ok_or_else(|| anyhow::anyhow!("Null graph from capture"))?;
            Ok(crate::graph_capture::AutonomousGraph::new(graph))
        } else {
            anyhow::bail!("No engine loaded")
        }
    }

    // rebuild_neighbor_lists_if_needed already exists below (line 1283)

    /// Device-side spike events buffer (for GPU-direct ring buffer push).
    /// Returns the raw d_spike_events allocation — no CPU download.
    pub fn spike_events_device(&self) -> Option<&CudaSlice<u8>> {
        self.engine.as_ref().map(|e| e.spike_events_buffer())
    }

    /// Device-side spike count buffer (for GPU-direct ring buffer push).
    pub fn spike_count_device(&self) -> Option<&CudaSlice<i32>> {
        self.engine.as_ref().map(|e| e.spike_count_buffer())
    }

    /// Rebuild neighbor lists if the rebuild interval has elapsed.
    /// Returns Ok(true) if rebuilt, Ok(false) if not needed.
    pub fn rebuild_neighbor_lists_if_needed(&mut self) -> Result<bool> {
        if let Some(ref mut engine) = self.engine {
            engine.rebuild_neighbor_lists_if_needed()
        } else {
            Ok(false)
        }
    }

    /// Get mutable reference to the inner NhsAmberFusedEngine.
    /// Used when the caller needs direct access (e.g., autonomous TWIN coupling).
    pub fn fused_engine_mut(&mut self) -> Option<&mut crate::fused_engine::NhsAmberFusedEngine> {
        self.engine.as_mut()
    }

    /// Get immutable reference to the inner NhsAmberFusedEngine.
    pub fn fused_engine(&self) -> Option<&crate::fused_engine::NhsAmberFusedEngine> {
        self.engine.as_ref()
    }

    /// Set REST2 solute tempering λ. λ=1.0 = physical, λ<1.0 = softened potential.
    pub fn set_solute_lambda(&mut self, lambda: f32) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.set_solute_lambda(lambda);
            Ok(())
        } else {
            bail!("No topology loaded")
        }
    }

    /// Focus REST2 λ on spike-frustrated residues identified during cold_hold.
    pub fn apply_focused_lambda(&mut self) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.apply_focused_lambda()
        } else {
            bail!("No topology loaded")
        }
    }

    /// Enable spike-guided adaptive bias: closed-loop UV energy modulation
    /// driven by real-time spike activity in each voxel region.
    pub fn set_adaptive_bias(&mut self, enabled: bool) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.set_adaptive_bias(enabled);
            Ok(())
        } else {
            bail!("No topology loaded")
        }
    }

    /// Set fused multi-step: N AMBER steps per 1 multi-LIF observation.
    pub fn set_fused_inner_steps(&mut self, n: u32) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.set_fused_inner_steps(n);
            Ok(())
        } else {
            bail!("No topology loaded")
        }
    }

    /// Gate G2 — Configure interval-based ensemble snapshot capture on
    /// the wrapped fused engine. When `n > 0`, the engine's `step()`
    /// end-of-step barrier captures an [`crate::fused_engine::EnsembleSnapshot`]
    /// every `n` MD steps tagged
    /// [`crate::fused_engine::SnapshotTrigger::IntervalScheduled`].
    /// When `n == 0` (default), only the existing activity-driven
    /// trigger paths fire. No-op when no topology has been loaded.
    pub fn set_snapshot_interval(&mut self, n: u32) {
        if let Some(ref mut engine) = self.engine {
            engine.set_snapshot_interval(n);
        }
    }

    /// Set the integration timestep. Use 0.004 (4fs) with HMR masses.
    pub fn set_dt(&mut self, dt: f64) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.set_dt(dt);
            Ok(())
        } else {
            bail!("No topology loaded")
        }
    }

    /// **DEPRECATED**: Configure temperature protocol separately
    ///
    /// Use `set_cryo_uv_protocol()` instead to configure the unified cryo-UV protocol.
    #[deprecated(since = "1.2.0", note = "Use set_cryo_uv_protocol() instead")]
    pub fn set_temperature_protocol(&mut self, protocol: TemperatureProtocol) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            #[allow(deprecated)]
            engine.set_temperature_protocol(protocol)?;
            Ok(())
        } else {
            bail!("No topology loaded")
        }
    }

    /// **DEPRECATED**: Configure UV probe separately
    ///
    /// Use `set_cryo_uv_protocol()` instead to configure the unified cryo-UV protocol.
    #[deprecated(since = "1.2.0", note = "Use set_cryo_uv_protocol() instead")]
    pub fn set_uv_config(&mut self, config: UvProbeConfig) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            #[allow(deprecated)]
            engine.set_uv_config(config);
            Ok(())
        } else {
            bail!("No topology loaded")
        }
    }

    /// Run simulation on current topology
    pub fn run(&mut self, n_steps: i32) -> Result<RunSummary> {
        if let Some(ref mut engine) = self.engine {
            let run_start = Instant::now();
            let summary = engine.run(n_steps)?;
            let run_time = run_start.elapsed().as_millis() as u64;

            self.structures_processed += 1;
            self.total_steps_run += n_steps as i64;
            self.total_compute_time_ms += run_time;

            Ok(summary)
        } else {
            bail!("No topology loaded")
        }
    }

    /// Run simulation and automatically cluster spike events
    ///
    /// Convenience method that:
    /// 1. Runs the simulation for n_steps
    /// 2. Collects spike events
    /// 3. Clusters them using RT cores (or fallback)
    /// 4. Returns structured binding site information
    ///
    /// # Example
    /// ```ignore
    /// let (summary, sites, stats) = engine.run_and_cluster(1_000_000)?;
    /// for site in &sites {
    ///     let c = site.view(prism_nhs::spatial_view::SpatialView::GeometricVoxelMass).unwrap();
    ///     println!("Binding site at {:?} with {} spikes", c, site.spike_count);
    /// }
    /// ```
    pub fn run_and_cluster(&mut self, n_steps: i32) -> Result<(RunSummary, Vec<ClusteredBindingSite>, Option<ClusteringStats>)> {
        let summary = self.run(n_steps)?;
        let spike_events = self.get_spike_events();

        if spike_events.is_empty() {
            return Ok((summary, Vec::new(), None));
        }

        // Extract positions
        let positions: Vec<f32> = spike_events.iter()
            .flat_map(|s| s.position.iter().copied())
            .collect();

        // Cluster using RT cores or fallback
        let used_rt = self.has_rt_clustering();
        let result = self.cluster_spikes(&positions)?;
        // Phase 2: route the canonical site-construction boundary through
        // the `ClusteringToClusteredSites` transform so the GVM-populated
        // (L1) and emission-compat-matches-GVM (L2) laws are enforced on
        // every output of this path. An abort-routed violation here is a
        // structural-corruption failure of the construction invariant and
        // must surface as a run error, not a silent fallback.
        #[cfg(feature = "gpu")]
        let sites = {
            use crate::transform::clustering_to_clustered_sites::{
                ClusteringInput, ClusteringToClusteredSites,
            };
            use crate::transform::AuditedTransform;
            let outcome = ClusteringToClusteredSites::new().apply(ClusteringInput {
                spike_events: &spike_events,
                clustering_result: &result,
                engine: self.engine.as_ref(),
            });
            let (sites, quarantined) = outcome
                .into_result()
                .map_err(|aborted| anyhow::anyhow!("{aborted}"))?;
            if !quarantined.is_empty() {
                log::warn!(
                    "transform clustering_to_clustered_sites: {} quarantined site(s) in run_and_cluster",
                    quarantined.len()
                );
            }
            sites
        };
        #[cfg(not(feature = "gpu"))]
        let sites = build_clustered_sites(&spike_events, &result, None);
        let stats = ClusteringStats {
            num_clusters: result.num_clusters,
            total_neighbors: result.total_neighbors,
            gpu_time_ms: result.gpu_time_ms,
            used_rt_cores: used_rt,
        };

        Ok((summary, sites, Some(stats)))
    }

    /// Get spike events from current run
    pub fn get_spike_events(&self) -> Vec<SpikeEvent> {
        if let Some(ref engine) = self.engine {
            engine.get_spike_events().to_vec()
        } else {
            Vec::new()
        }
    }

    /// Enable spike accumulation for analysis
    ///
    /// When enabled, spikes are downloaded from the GPU and accumulated
    /// across sync intervals. Use `get_accumulated_spikes()` to retrieve.
    pub fn set_spike_accumulation(&mut self, enabled: bool) {
        if let Some(ref mut engine) = self.engine {
            engine.set_spike_accumulation(enabled);
        }
    }

    /// Adapt internal protocol parameters based on measured cold-hold spike rate.
    ///
    /// Delegates to `NhsAmberFusedEngine::adapt_protocol_from_spike_rate()`.
    /// Returns the detected `FlexibilityClass`.
    pub fn adapt_protocol_from_spike_rate(&mut self, cold_hold_steps: i32) -> crate::fused_engine::FlexibilityClass {
        if let Some(ref mut engine) = self.engine {
            engine.adapt_protocol_from_spike_rate(cold_hold_steps)
        } else {
            log::warn!("adapt_protocol_from_spike_rate called with no engine loaded");
            crate::fused_engine::FlexibilityClass::Normal
        }
    }

    /// Get accumulated spike events (GPU format with timestamps)
    ///
    /// Returns all spike events accumulated since spike accumulation was enabled.
    /// Only populated when spike accumulation is enabled via `set_spike_accumulation(true)`.
    pub fn get_accumulated_spikes(&self) -> Vec<crate::fused_engine::GpuSpikeEvent> {
        if let Some(ref engine) = self.engine {
            engine.get_accumulated_spikes().to_vec()
        } else {
            Vec::new()
        }
    }

    /// Download signal preservation buffers from the GPU engine
    pub fn download_signal_preservation(&self) -> anyhow::Result<crate::fused_engine::SignalPreservationData> {
        if let Some(ref engine) = self.engine {
            engine.download_signal_preservation()
        } else {
            anyhow::bail!("No engine initialized")
        }
    }

    /// Compute and download KCC v2-full descriptors from GPU
    pub fn compute_and_download_kcc(&mut self) -> anyhow::Result<crate::fused_engine::KccData> {
        if let Some(ref mut engine) = self.engine {
            engine.compute_and_download_kcc()
        } else {
            Ok(crate::fused_engine::KccData::empty())
        }
    }

    /// Launch the KCC residue update kernel ONCE outside the captured CUDA Graph.
    ///
    /// Used by the autonomous (multi-stream + multi-differential) chunk loop to
    /// drive KCC streaming reductions and ring buffer events between graph replays.
    /// The captured graph (`step_autonomous_kernels`) does not include KCC because
    /// the kernel needs a fresh `current_step` per call (the ring buffer event
    /// coalescer would otherwise collapse all replays into a single ring slot).
    /// See `NhsAmberFusedEngine::kcc_step_once` for the rationale and design.
    pub fn kcc_step_once(&mut self, current_step: u32, current_phase: i32) -> anyhow::Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.kcc_step_once(current_step, current_phase)
        } else {
            Ok(())
        }
    }

    /// Stage 2: Write the closed-loop ASC steering focus list into the
    /// device-side ProtocolState.
    ///
    /// `entries` is a slice of `(residue_id, weight)` tuples derived from the
    /// per-chunk GC-PID synergy fraction (Stage 1B-3). At most
    /// `STEERING_FOCUS_MAX = 64` entries are written; any extra are ignored.
    /// The host serializes the new `steering_focus_count` + the array bytes
    /// into a small staging buffer and issues a `memcpy_htod` at the
    /// appropriate offset within ProtocolState. The autonomous CUDA Graph
    /// has captured the *pointer* to ProtocolState; the next graph replay
    /// (and the next launch of `ring_buffer_read_and_adapt`) reads the new
    /// contents automatically — no graph re-capture, no kernel re-launch
    /// configuration change.
    ///
    /// This is the canonical "captured pointer with mutable contents"
    /// pattern, identical to how the existing legacy steering hooks
    /// (`steering_uv_boost`, etc.) are updated. The only difference is the
    /// payload size (516 bytes instead of 4-16 bytes) and the offset within
    /// the struct.
    ///
    /// Used by the autonomous chunk loop in `nhs_rt_full.rs` when
    /// `--closed-loop-steering` is enabled. Without the flag, the
    /// `steering_focus_count` stays at zero and the kernel's per-spike
    /// steering branch is a no-op (the `if (n_focus > 0)` guard short-
    /// circuits) so this method has no effect on default behavior.
    pub fn write_steering_focus(
        &mut self,
        stream: &Arc<CudaStream>,
        entries: &[(i32, f32)],
    ) -> anyhow::Result<()> {
        use crate::protocol_state::{ProtocolState, SteerEntry, STEERING_FOCUS_MAX};

        // Strategy: download the full ProtocolState (684 bytes), update only
        // the steering fields in the host copy, upload the full struct back.
        // The full-struct round trip is ~1.4 KB of PCIe traffic per chunk
        // (~80 chunks per run = ~110 KB total) — negligible compared to the
        // multi-GB spike data downloads. Avoiding partial writes also dodges
        // a cudarc slice-typing issue with `memcpy_htod` on a `CudaView`.
        //
        // We could optimize this to a partial write at the steering field
        // offset using `slice_mut(..).into()` once the cudarc trait
        // resolution is stable, but for Stage 2 v1 the round-trip approach
        // is correct, simple, and fast enough.
        if let Some(ref mut engine) = self.engine {
            // Download
            let mut buf = vec![0u8; std::mem::size_of::<ProtocolState>()];
            stream.memcpy_dtoh(&engine.d_protocol_state, &mut buf)?;
            // Reinterpret as ProtocolState (matching the device-side layout
            // exactly via #[repr(C)]).
            let mut state: ProtocolState =
                unsafe { std::ptr::read(buf.as_ptr() as *const ProtocolState) };

            // Update only the steering fields
            let n_active = entries.len().min(STEERING_FOCUS_MAX) as i32;
            state.steering_focus_count = n_active;
            for i in 0..STEERING_FOCUS_MAX {
                state.steering_focus_residues[i] = if i < entries.len() {
                    SteerEntry {
                        residue_id: entries[i].0,
                        weight: entries[i].1,
                    }
                } else {
                    SteerEntry {
                        residue_id: -1,
                        weight: 0.0,
                    }
                };
            }

            // Upload the modified struct back
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    &state as *const ProtocolState as *const u8,
                    std::mem::size_of::<ProtocolState>(),
                )
            };
            stream.memcpy_htod(bytes, &mut engine.d_protocol_state)?;
        }
        Ok(())
    }

    /// Stage 2 calibration: download the full ProtocolState from the engine's
    /// device buffer. Used by the chunk loop's end-of-run summary to read the
    /// `focus_match_count` counter that the kernel atomicAdd's whenever a
    /// spike's primary residue id matches an active focus list entry.
    ///
    /// Synchronizes the stream before reading so the dtoh copy sees the
    /// final value (the kernel atomicAdds run on the same stream as the
    /// graph replay, but PCIe traffic from a different stream still needs
    /// the explicit sync).
    pub fn download_protocol_state(
        &self,
        stream: &Arc<CudaStream>,
    ) -> anyhow::Result<crate::protocol_state::ProtocolState> {
        use crate::protocol_state::ProtocolState;
        if let Some(ref engine) = self.engine {
            let mut buf = vec![0u8; std::mem::size_of::<ProtocolState>()];
            stream.memcpy_dtoh(&engine.d_protocol_state, &mut buf)?;
            let state: ProtocolState =
                unsafe { std::ptr::read(buf.as_ptr() as *const ProtocolState) };
            Ok(state)
        } else {
            anyhow::bail!("download_protocol_state: engine not initialized")
        }
    }

    /// Get snapshots from current run
    pub fn get_snapshots(&self) -> Vec<EnsembleSnapshot> {
        if let Some(ref engine) = self.engine {
            engine.get_ensemble_snapshots().to_vec()
        } else {
            Vec::new()
        }
    }

    /// Reset engine state for a new replica
    ///
    /// Clears accumulated spikes, re-initializes velocities with a new seed,
    /// and resets simulation counters. Topology and forces are preserved.
    ///
    /// # Arguments
    /// * `seed` - Random seed for velocity initialization (Maxwell-Boltzmann)
    pub fn reset_for_replica(&mut self, seed: u64) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.reset_for_replica(seed)
        } else {
            bail!("No topology loaded")
        }
    }

    /// Get current positions
    pub fn get_positions(&self) -> Result<Vec<f32>> {
        if let Some(ref engine) = self.engine {
            engine.get_positions()
        } else {
            bail!("No topology loaded")
        }
    }

    /// Enable UltimateEngine for 2-4x faster MD simulation
    ///
    /// Requires SM86+ GPU (Ampere/Ada/Blackwell) and topology to be loaded.
    /// Uses mixed-precision, SoA layout, and other hyperoptimizations.
    ///
    /// Note: Must be called after load_topology() with the same topology.
    pub fn enable_ultimate_mode(&mut self, topology: &PrismPrepTopology) -> Result<()> {
        if let Some(ref mut engine) = self.engine {
            engine.enable_ultimate_mode(topology)?;
            log::info!("✓ UltimateEngine enabled (2-4x faster MD)");
            Ok(())
        } else {
            bail!("Engine not initialized - call load_topology() first")
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // RT-ACCELERATED CLUSTERING
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Check if RT cores are available for accelerated clustering
    pub fn has_rt_clustering(&self) -> bool {
        crate::rt_utils::has_rt_cores() && crate::rt_utils::is_optix_available()
    }

    /// Ensure the RT clustering pipeline is initialized (lazy init)
    ///
    /// Call this explicitly to pre-warm the pipeline, or let it initialize
    /// lazily on first `cluster_spikes()` call.
    #[allow(unreachable_code)]
    pub fn ensure_rt_pipeline(&mut self) -> Result<bool> {
        if self.rt_engine.is_some() {
            return Ok(true);  // Already initialized
        }

        // ════════════════════════════════════════════════════════════
        // E2E UNBLOCK LANE (2026-04-21): OptiX RT clustering disabled.
        // rt_clustering.rs::load_pipeline() at lines 185-246 is
        // structurally incomplete — it never assigns self.pipeline and
        // never constructs a Pipeline, yet returns Ok(()). Routing
        // through it produced `WARN Clustering failed: Pipeline not
        // loaded` followed by SIGSEGV on engine shutdown. Per
        // CLAUDE.md IMMUTABLE_RULE #9 ("OptiX is removed"), we
        // short-circuit here so cluster_spikes falls through to
        // fallback_grid_cluster. No OptiX context is created,
        // eliminating the teardown crash path. To re-enable a future
        // working OptiX path, delete the next two lines.
        // ════════════════════════════════════════════════════════════
        log::info!("RT clustering disabled (OptiX path unavailable); using grid fallback");
        return Ok(false);

        if !self.has_rt_clustering() {
            log::debug!("RT clustering not available (no RT cores or OptiX)");
            return Ok(false);
        }

        log::info!("🔷 Initializing OptiX RT pipeline for clustering...");
        let start = Instant::now();

        // Find the OptiX IR file
        let optixir_path = crate::rt_clustering::find_optixir_path()
            .context("Could not find rt_clustering.optixir")?;

        // Adaptive epsilon: scale inversely with cube root of atom count.
        // Larger proteins need tighter clustering to resolve distinct pockets.
        //   ≤500 atoms  → 3.0 Å  (small, e.g. crambin)
        //   ~2000 atoms → 2.0 Å  (medium)
        //   ~4000 atoms → 1.5 Å  (large, e.g. TEM-1)
        //   ~8000 atoms → 1.2 Å  (very large)
        // Formula: epsilon = 3.0 * (500 / n_atoms)^(1/3), clamped to [1.2, 3.0]
        let n_atoms = self.engine.as_ref().map(|e| e.n_atoms()).unwrap_or(500);
        let adaptive_epsilon = (3.0_f32 * (500.0_f32 / n_atoms as f32).cbrt())
            .clamp(1.2, 3.0);
        log::info!("  Adaptive epsilon: {:.2}Å (n_atoms={})", adaptive_epsilon, n_atoms);

        // Create RT clustering config with adaptive epsilon
        let rt_config = crate::rt_clustering::RtClusteringConfig {
            epsilon: adaptive_epsilon,
            min_points: 4,         // Minimum 4 neighbors for core point
            min_cluster_size: 20,  // Minimum 20 points per cluster (catch smaller pockets)
            rays_per_event: 32,    // 32 rays for neighbor finding
        };

        // Create and initialize the RT engine
        let mut rt_engine = crate::rt_clustering::RtClusteringEngine::new(
            self.context.clone(),
            rt_config,
        ).context("Failed to create RT clustering engine")?;

        rt_engine.load_pipeline(&optixir_path)
            .context("Failed to load RT clustering pipeline")?;

        let init_time = start.elapsed().as_millis() as u64;
        self.rt_init_time_ms = Some(init_time);
        self.rt_engine = Some(rt_engine);

        log::info!("  RT pipeline initialized: {}ms", init_time);
        log::info!("  GPU Architecture: {}", crate::rt_utils::get_architecture_name());

        Ok(true)
    }

    /// Cluster spike positions using RT-accelerated spatial queries
    ///
    /// Falls back to grid-based clustering if RT cores are unavailable.
    ///
    /// # Arguments
    /// * `spike_positions` - Flat array of [x, y, z, x, y, z, ...] coordinates
    ///
    /// # Returns
    /// Clustering result with cluster assignments and statistics
    pub fn cluster_spikes(&mut self, spike_positions: &[f32]) -> Result<crate::rt_clustering::RtClusteringResult> {
        use crate::gpu_cluster_backend as gcb;
        let num_spikes = spike_positions.len() / 3;

        // Resolve auto → concrete backend. On SM120 OptiX is disabled
        // (IMMUTABLE_RULE #9), so auto maps to gpu-hash.
        let is_sm120 = !crate::rt_utils::is_optix_available();
        let backend = gcb::resolve_auto(gcb::current_selection(), is_sm120);

        log::info!(
            "POST_MD_CLUSTER_BACKEND_SELECTED backend={} spikes={}",
            gcb::backend_name(backend), num_spikes
        );
        log::info!("POST_MD_CLUSTER_START spikes={}", num_spikes);
        let t_dispatch = std::time::Instant::now();

        let result = match backend {
            gcb::BACKEND_GPU_HASH => {
                // Lazy-init the GPU CCL backend once per engine
                if self.gpu_cluster.is_none() {
                    let b = gcb::GpuSpatialHashBackend::new(
                        self.context.clone(), self.stream.clone()
                    ).context("init GpuSpatialHashBackend")?;
                    self.gpu_cluster = Some(b);
                }
                // Fixed epsilon semantics match the legacy fallback_grid_cluster.
                // The 5.0 Å value is the established post-MD CCL radius; changing
                // it would be semantic drift and is forbidden in this lane.
                let epsilon = 5.0_f32;
                self.gpu_cluster.as_mut().unwrap().cluster(spike_positions, epsilon)?
            }
            gcb::BACKEND_RT_OPTIX => {
                if self.ensure_rt_pipeline()? {
                    if let Some(ref mut rt_engine) = self.rt_engine {
                        log::debug!("Using RT-accelerated clustering for {} spikes", num_spikes);
                        rt_engine.cluster(spike_positions)?
                    } else {
                        anyhow::bail!(
                            "--clustering-backend=optix requested but rt_engine is None; \
                             OptiX unavailable on this device (IMMUTABLE RULE #9 — use auto or gpu-hash)"
                        );
                    }
                } else {
                    anyhow::bail!(
                        "--clustering-backend=optix requested but OptiX is disabled on this device; \
                         use --clustering-backend=auto or --clustering-backend=gpu-hash"
                    );
                }
            }
            gcb::BACKEND_LBVH => {
                anyhow::bail!(
                    "--clustering-backend=lbvh selected but LBVH backend is not yet implemented \
                     (arrives in the Phase-2 LBVH lane). Use auto or gpu-hash."
                );
            }
            gcb::BACKEND_GRID_DEBUG => {
                log::warn!(
                    "POST_MD_CLUSTER_BACKEND_DEBUG grid fallback active (debug only — degrades at N>>1M)"
                );
                log::debug!("Using fallback grid clustering for {} spikes", num_spikes);
                self.fallback_grid_cluster(spike_positions)?
            }
            other => {
                anyhow::bail!("Unresolved clustering backend code: {}", other);
            }
        };

        log::info!(
            "POST_MD_CLUSTER_DONE clusters={} elapsed_ms={}",
            result.num_clusters, t_dispatch.elapsed().as_millis()
        );

        Ok(result)
    }

    /// Re-cluster spikes at a specific epsilon (for mega-cluster subdivision).
    /// Temporarily overrides the RT engine's epsilon, clusters, then restores it.
    pub fn cluster_spikes_at_epsilon(&mut self, spike_positions: &[f32], epsilon: f32) -> Result<crate::rt_clustering::RtClusteringResult> {
        if let Some(ref mut rt_engine) = self.rt_engine {
            let saved = rt_engine.config.epsilon;
            rt_engine.config.epsilon = epsilon;
            let result = rt_engine.cluster(spike_positions);
            rt_engine.config.epsilon = saved;
            result
        } else {
            anyhow::bail!("RT clustering engine not initialized")
        }
    }

    /// Get the current adaptive epsilon value
    pub fn current_epsilon(&self) -> Option<f32> {
        self.rt_engine.as_ref().map(|e| e.config.epsilon)
    }

    /// Multi-scale clustering for robust, structure-agnostic binding site detection
    ///
    /// Runs DBSCAN clustering at multiple epsilon values and tracks cluster persistence
    /// across scales. Clusters that appear at multiple scales are more likely to be
    /// real binding sites rather than noise.
    ///
    /// # Algorithm
    /// Compute adaptive epsilon values from k-NN distance distribution
    ///
    /// Samples spike positions and computes the k-th nearest neighbor distance
    /// for each sample. Returns epsilon values at key percentiles that capture
    /// the natural clustering scales in the data.
    ///
    /// # Arguments
    /// * `positions` - Flat array of [x, y, z, x, y, z, ...] coordinates
    /// * `k` - Number of nearest neighbors to consider (default: 4 for DBSCAN min_points)
    /// * `sample_size` - Number of points to sample (default: 1000)
    ///
    /// # Returns
    /// Vector of epsilon values at 25th, 50th, 75th, and 90th percentiles
    pub fn compute_adaptive_epsilon(
        positions: &[f32],
        k: usize,
        sample_size: usize,
    ) -> Vec<f32> {
        let n_points = positions.len() / 3;
        if n_points < k + 1 {
            // Not enough points, return default
            return vec![5.0, 7.0, 10.0, 14.0];
        }

        // Sample points (evenly spaced or random)
        let sample_indices: Vec<usize> = if n_points <= sample_size {
            (0..n_points).collect()
        } else {
            let step = n_points / sample_size;
            (0..sample_size).map(|i| i * step).collect()
        };

        // Compute k-NN distance for each sampled point
        let mut knn_distances: Vec<f32> = Vec::with_capacity(sample_indices.len());

        for &i in &sample_indices {
            let xi = positions[i * 3];
            let yi = positions[i * 3 + 1];
            let zi = positions[i * 3 + 2];

            // Compute distances to all other points (brute force for simplicity)
            let mut distances: Vec<f32> = Vec::with_capacity(n_points.min(1000));
            for j in 0..n_points.min(5000) { // Limit comparison for performance
                if i == j { continue; }
                let xj = positions[j * 3];
                let yj = positions[j * 3 + 1];
                let zj = positions[j * 3 + 2];
                let d = ((xi - xj).powi(2) + (yi - yj).powi(2) + (zi - zj).powi(2)).sqrt();
                distances.push(d);
            }

            // Get k-th smallest distance
            if distances.len() >= k {
                distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                knn_distances.push(distances[k - 1]);
            }
        }

        if knn_distances.is_empty() {
            return vec![5.0, 7.0, 10.0, 14.0];
        }

        // Sort k-NN distances
        knn_distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Extract percentiles: 25th, 50th, 75th, 90th
        let n = knn_distances.len();
        let p25 = knn_distances[n / 4].clamp(3.0, 8.0);
        let p50 = knn_distances[n / 2].clamp(5.0, 12.0);
        let p75 = knn_distances[3 * n / 4].clamp(7.0, 18.0);
        let p90 = knn_distances[9 * n / 10].clamp(10.0, 25.0);

        // Ensure monotonically increasing and minimum spacing
        let mut epsilons = vec![p25];
        for &e in &[p50, p75, p90] {
            if e > epsilons.last().unwrap() + 1.5 {
                epsilons.push(e);
            }
        }

        // Ensure we have at least 3 scales
        while epsilons.len() < 3 {
            let last = *epsilons.last().unwrap();
            epsilons.push((last * 1.4).min(25.0));
        }

        log::info!("Adaptive epsilon: {:?} (from k-NN analysis, k={})", epsilons, k);
        epsilons
    }

    /// Multi-scale clustering with automatic or fixed epsilon selection
    ///
    /// 1. Run clustering at epsilon values: [5.0, 7.0, 10.0, 14.0] Angstroms
    /// 2. For each scale, compute cluster centroids
    /// 3. Find clusters that "persist" (have similar centroids) across ≥2 scales
    /// 4. Merge overlapping clusters and assign persistence scores
    /// 5. Return clusters sorted by persistence × spike_count
    ///
    /// # Arguments
    /// * `spike_positions` - Flat array of [x, y, z, x, y, z, ...] coordinates
    ///
    /// # Returns
    /// MultiScaleClusteringResult with persistence-scored clusters
    pub fn multi_scale_cluster_spikes(
        &mut self,
        spike_positions: &[f32],
    ) -> Result<MultiScaleClusteringResult> {
        self.multi_scale_cluster_spikes_with_epsilon(spike_positions, None)
    }

    /// Multi-scale clustering with optional custom epsilon values
    ///
    /// # Arguments
    /// * `spike_positions` - Flat array of [x, y, z, x, y, z, ...] coordinates
    /// * `custom_epsilon` - If Some, use these values; if None, use adaptive selection
    pub fn multi_scale_cluster_spikes_with_epsilon(
        &mut self,
        spike_positions: &[f32],
        custom_epsilon: Option<Vec<f32>>,
    ) -> Result<MultiScaleClusteringResult> {
        use std::collections::HashMap;

        let num_spikes = spike_positions.len() / 3;

        // Determine epsilon values: custom, adaptive, or default
        // Track whether adaptive was used for output
        let knn_k = 4usize;
        let sample_size = 1000usize;
        let (epsilon_scales, is_adaptive, actual_sample_size): (Vec<f32>, bool, Option<usize>) =
            if let Some(eps) = custom_epsilon {
                log::info!("Using fixed epsilon values: {:?}", eps);
                (eps, false, None)
            } else if num_spikes > 1000 {
                // Use adaptive selection for large datasets
                let actual_samples = num_spikes.min(sample_size);
                let eps = Self::compute_adaptive_epsilon(spike_positions, knn_k, sample_size);
                (eps, true, Some(actual_samples))
            } else {
                // Default values for small datasets
                log::info!("Using default epsilon values (small dataset): [5.0, 7.0, 10.0, 14.0]");
                (vec![5.0, 7.0, 10.0, 14.0], false, None)
            };
        let merge_distance = 12.0f32; // Clusters within 8Å are considered the same site

        log::info!("Running multi-scale clustering on {} spikes at {} scales",
            num_spikes, epsilon_scales.len());

        // Track clusters across scales: (centroid, spike_count, epsilon, cluster_id)
        let mut all_clusters: Vec<ScaleCluster> = Vec::new();

        // Run clustering at each scale
        for &epsilon in &epsilon_scales {
            // Create RT engine with this epsilon
            let rt_config = crate::rt_clustering::RtClusteringConfig {
                epsilon,
                min_points: 4,
                min_cluster_size: 30, // Scaled for 128³ grid
                rays_per_event: 32,
            };

            // We need to create a new RT engine for each epsilon
            // (The BVH sphere radii depend on epsilon)
            let optixir_path = crate::rt_clustering::find_optixir_path()
                .context("Could not find rt_clustering.optixir")?;

            let mut rt_engine = crate::rt_clustering::RtClusteringEngine::new(
                self.context.clone(),
                rt_config,
            ).context("Failed to create RT clustering engine")?;

            rt_engine.load_pipeline(&optixir_path)
                .context("Failed to load RT clustering pipeline")?;

            let result = rt_engine.cluster(spike_positions)?;

            log::info!("  Scale ε={:.1}Å: {} clusters, {} neighbors",
                epsilon, result.num_clusters, result.total_neighbors);

            // Compute centroids for each cluster at this scale
            let mut cluster_points: HashMap<i32, Vec<usize>> = HashMap::new();
            for (idx, &cluster_id) in result.cluster_ids.iter().enumerate() {
                if cluster_id >= 0 {
                    cluster_points.entry(cluster_id).or_default().push(idx);
                }
            }

            for (cluster_id, point_indices) in cluster_points {
                if point_indices.len() < 15 {
                    continue; // Skip tiny clusters
                }

                // Compute centroid
                let mut cx = 0.0f32;
                let mut cy = 0.0f32;
                let mut cz = 0.0f32;
                for &idx in &point_indices {
                    cx += spike_positions[idx * 3];
                    cy += spike_positions[idx * 3 + 1];
                    cz += spike_positions[idx * 3 + 2];
                }
                let n = point_indices.len() as f32;
                cx /= n;
                cy /= n;
                cz /= n;

                all_clusters.push(ScaleCluster {
                    centroid: [cx, cy, cz],
                    spike_count: point_indices.len(),
                    spike_indices: point_indices,
                    epsilon,
                    original_cluster_id: cluster_id,
                    persistence: 1, // Will be updated during merge
                });
            }
        }

        log::info!("  Total clusters across all scales: {}", all_clusters.len());

        // Merge clusters that overlap across scales
        let mut merged_clusters: Vec<MergedCluster> = Vec::new();
        let mut used = vec![false; all_clusters.len()];

        for i in 0..all_clusters.len() {
            if used[i] {
                continue;
            }

            // Start a new merged cluster
            let mut merge_group: Vec<usize> = vec![i];
            used[i] = true;

            // Find all clusters within merge_distance of this centroid
            let ci = &all_clusters[i];
            for j in (i + 1)..all_clusters.len() {
                if used[j] {
                    continue;
                }

                let cj = &all_clusters[j];
                let dist = ((ci.centroid[0] - cj.centroid[0]).powi(2)
                    + (ci.centroid[1] - cj.centroid[1]).powi(2)
                    + (ci.centroid[2] - cj.centroid[2]).powi(2))
                    .sqrt();

                if dist <= merge_distance {
                    merge_group.push(j);
                    used[j] = true;
                }
            }

            // Compute merged cluster properties
            let scales_present: std::collections::HashSet<u32> = merge_group
                .iter()
                .map(|&idx| (all_clusters[idx].epsilon * 10.0) as u32)
                .collect();
            let persistence = scales_present.len();

            // Merge spike indices (union across scales)
            let mut all_spike_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();
            let mut sum_cx = 0.0f32;
            let mut sum_cy = 0.0f32;
            let mut sum_cz = 0.0f32;

            for &idx in &merge_group {
                let c = &all_clusters[idx];
                sum_cx += c.centroid[0];
                sum_cy += c.centroid[1];
                sum_cz += c.centroid[2];
                for &spike_idx in &c.spike_indices {
                    all_spike_indices.insert(spike_idx);
                }
            }

            let n = merge_group.len() as f32;
            let merged_centroid = [sum_cx / n, sum_cy / n, sum_cz / n];
            let merged_spike_count = all_spike_indices.len();

            merged_clusters.push(MergedCluster {
                centroid: merged_centroid,
                spike_count: merged_spike_count,
                spike_indices: all_spike_indices.into_iter().collect(),
                persistence,
                scales: scales_present.into_iter().map(|s| s as f32 / 10.0).collect(),
            });
        }

        // Sort by persistence * spike_count (higher = more confident)
        merged_clusters.sort_by(|a, b| {
            let score_a = a.persistence * a.spike_count;
            let score_b = b.persistence * b.spike_count;
            score_b.cmp(&score_a)
        });

        // Filter to clusters with persistence >= 2 (appear at multiple scales)
        let persistent_clusters: Vec<MergedCluster> = merged_clusters
            .into_iter()
            .filter(|c| c.persistence >= 2)
            .collect();

        log::info!("  Multi-scale result: {} persistent clusters (appear at ≥2 scales)",
            persistent_clusters.len());

        for (i, c) in persistent_clusters.iter().take(5).enumerate() {
            log::info!("    #{}: {} spikes, persistence={}, scales={:?}",
                i + 1, c.spike_count, c.persistence, c.scales);
        }

        Ok(MultiScaleClusteringResult {
            clusters: persistent_clusters,
            total_scales: epsilon_scales.len(),
            epsilon_values: epsilon_scales,
            adaptive_epsilon: is_adaptive,
            knn_k: if is_adaptive { Some(knn_k) } else { None },
            num_spikes_sampled: actual_sample_size,
        })
    }

    /// Fallback grid-based clustering when RT cores unavailable
    /// CPU grid clustering fallback — public for multi-stream path when RT cores fail
    pub fn cluster_spikes_cpu_fallback(&self, positions: &[f32]) -> Result<crate::rt_clustering::RtClusteringResult> {
        self.fallback_grid_cluster(positions)
    }

    fn fallback_grid_cluster(&self, positions: &[f32]) -> Result<crate::rt_clustering::RtClusteringResult> {
        let num_points = positions.len() / 3;
        let start = Instant::now();

        // Simple single-linkage clustering using spatial hashing
        // This is O(N) for sparse data but degrades to O(N²) for dense clusters
        let epsilon = 5.0f32;
        let cell_size = epsilon;

        use std::collections::HashMap;

        // Hash points into cells
        let mut cells: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
        for i in 0..num_points {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];
            let cell = (
                (x / cell_size).floor() as i32,
                (y / cell_size).floor() as i32,
                (z / cell_size).floor() as i32,
            );
            cells.entry(cell).or_default().push(i);
        }

        // Union-find for clustering
        let mut parent: Vec<i32> = (0..num_points as i32).collect();

        fn find(parent: &mut [i32], i: usize) -> i32 {
            if parent[i] != i as i32 {
                parent[i] = find(parent, parent[i] as usize);
            }
            parent[i]
        }

        fn union(parent: &mut [i32], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[ra as usize] = rb;
            }
        }

        // Find neighbors and union
        let mut total_neighbors = 0usize;
        for (&cell, points) in &cells {
            // Check this cell and 26 neighbors
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let neighbor_cell = (cell.0 + dx, cell.1 + dy, cell.2 + dz);
                        if let Some(neighbors) = cells.get(&neighbor_cell) {
                            for &i in points {
                                let xi = positions[i * 3];
                                let yi = positions[i * 3 + 1];
                                let zi = positions[i * 3 + 2];

                                for &j in neighbors {
                                    if i >= j { continue; }
                                    let xj = positions[j * 3];
                                    let yj = positions[j * 3 + 1];
                                    let zj = positions[j * 3 + 2];

                                    let dist_sq = (xi - xj).powi(2) + (yi - yj).powi(2) + (zi - zj).powi(2);
                                    if dist_sq <= epsilon * epsilon {
                                        union(&mut parent, i, j);
                                        total_neighbors += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Flatten and count clusters
        let mut cluster_ids: Vec<i32> = Vec::with_capacity(num_points);
        let mut cluster_counts: HashMap<i32, usize> = HashMap::new();

        for i in 0..num_points {
            let root = find(&mut parent, i);
            cluster_ids.push(root);
            *cluster_counts.entry(root).or_default() += 1;
        }

        let num_clusters = cluster_counts.len();
        let gpu_time_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok(crate::rt_clustering::RtClusteringResult {
            cluster_ids,
            num_clusters,
            total_neighbors,
            gpu_time_ms,
        })
    }

    /// Report cumulative statistics
    pub fn stats(&self) -> PersistentEngineStats {
        PersistentEngineStats {
            structures_processed: self.structures_processed,
            total_steps_run: self.total_steps_run,
            total_compute_time_ms: self.total_compute_time_ms,
            context_init_time_ms: self.context_init_time_ms,
            module_init_time_ms: self.module_init_time_ms,
            overhead_saved_ms: self.structures_processed.saturating_sub(1) as u64
                * (self.context_init_time_ms + self.module_init_time_ms),
        }
    }
}

/// Statistics from persistent engine
#[derive(Debug, Clone)]
pub struct PersistentEngineStats {
    pub structures_processed: usize,
    pub total_steps_run: i64,
    pub total_compute_time_ms: u64,
    pub context_init_time_ms: u64,
    pub module_init_time_ms: u64,
    /// Estimated overhead saved by reusing context/module
    pub overhead_saved_ms: u64,
}

/// Convert RT clustering results into binding site structures.
///
/// Crate-visible so `crate::transform::clustering_to_clustered_sites`
/// can wrap this function under the Phase 2 audit spine. Not `pub` —
/// external callers go through the transform, not directly.
#[cfg(feature = "gpu")]
pub(crate) fn build_clustered_sites(
    spike_events: &[SpikeEvent],
    clustering_result: &crate::rt_clustering::RtClusteringResult,
    #[cfg(feature = "gpu")]
    gpu_engine: Option<&NhsAmberFusedEngine>,
    #[cfg(not(feature = "gpu"))]
    _gpu_engine: Option<&()>,
) -> Vec<ClusteredBindingSite> {
    use std::collections::HashMap;

    if spike_events.is_empty() {
        return Vec::new();
    }

    // Group spikes by cluster
    let mut cluster_spikes: HashMap<i32, Vec<(usize, &SpikeEvent)>> = HashMap::new();
    for (idx, (spike, &cluster_id)) in spike_events.iter()
        .zip(clustering_result.cluster_ids.iter())
        .enumerate()
    {
        if cluster_id >= 0 {  // Skip noise points (-1)
            cluster_spikes.entry(cluster_id)
                .or_default()
                .push((idx, spike));
        }
    }

    // Build site structures for each cluster
    let mut sites = Vec::with_capacity(cluster_spikes.len());
    for (cluster_id, spikes) in cluster_spikes {
        if spikes.is_empty() {
            continue;
        }

        // Compute centroid using burial-weighted mean + density peak refinement.
        //
        // Two-stage centroid:
        //   Stage 1: Burial-weighted mean — spikes with more nearby residues
        //   (n_residues) are deeper in the protein and closer to the true
        //   binding cavity. Weight = n_residues² so buried spikes dominate.
        //
        //   Stage 2: Density peak — find the local maximum of spike density
        //   using a 3D Gaussian KDE. The peak is typically 1-3A closer to the
        //   true ligand center than the arithmetic mean because it ignores
        //   outlier surface spikes. Final centroid is 70% density peak +
        //   30% burial-weighted mean (peak for accuracy, mean for stability).
        let mut centroid = [0.0f32; 3];
        let mut sum_intensity = 0.0f32;
        let mut min_pos = [f32::MAX; 3];
        let mut max_pos = [f32::MIN; 3];

        // Stage 1: Burial-weighted centroid
        let mut weighted_centroid = [0.0f32; 3];
        let mut total_weight = 0.0f32;

        for (_, spike) in &spikes {
            // Burial weight: nearby_residues_count² (buried spikes count more)
            // Minimum weight of 1.0 so surface spikes aren't zeroed out
            let burial_weight = (spike.nearby_residues.len() as f32).max(1.0).powi(2);
            weighted_centroid[0] += spike.position[0] * burial_weight;
            weighted_centroid[1] += spike.position[1] * burial_weight;
            weighted_centroid[2] += spike.position[2] * burial_weight;
            total_weight += burial_weight;

            // Unweighted sum for fallback
            centroid[0] += spike.position[0];
            centroid[1] += spike.position[1];
            centroid[2] += spike.position[2];
            sum_intensity += spike.intensity;

            // Update bounding box
            for i in 0..3 {
                min_pos[i] = min_pos[i].min(spike.position[i]);
                max_pos[i] = max_pos[i].max(spike.position[i]);
            }
        }

        let n = spikes.len() as f32;

        // Use burial-weighted centroid if we have burial data, else arithmetic mean
        if total_weight > n {
            // Burial weights are contributing (not all n_residues=1)
            centroid[0] = weighted_centroid[0] / total_weight;
            centroid[1] = weighted_centroid[1] / total_weight;
            centroid[2] = weighted_centroid[2] / total_weight;
        } else {
            centroid[0] /= n;
            centroid[1] /= n;
            centroid[2] /= n;
        }

        // Stage 2: Density peak refinement via 3D Gaussian KDE
        // Try GPU path first, fall back to CPU.
        if spikes.len() >= 20 {
            let bandwidth = 2.0f32;
            let search_radius = 5.0f32;

            let mut gpu_done = false;

            // GPU path: use the fused engine's KDE kernel if available
            #[cfg(feature = "gpu")]
            if let Some(engine) = gpu_engine {
                let positions: Vec<[f32; 3]> = spikes.iter()
                    .map(|(_, s)| s.position)
                    .collect();
                let n_residues: Vec<i32> = spikes.iter()
                    .map(|(_, s)| s.nearby_residues.len() as i32)
                    .collect();

                match engine.compute_kde_centroid(
                    &positions, &n_residues, centroid,
                    search_radius, bandwidth,
                ) {
                    Ok(refined) => {
                        centroid = refined;
                        gpu_done = true;
                    }
                    Err(e) => {
                        log::warn!("GPU KDE failed, falling back to CPU: {}", e);
                    }
                }
            }

            // CPU fallback
            if !gpu_done {
                let bw2 = bandwidth * bandwidth;
                let grid_step = 1.0f32;
                let mut best_density = 0.0f32;
                let mut peak_pos = centroid;

                let n_steps = (search_radius / grid_step) as i32;
                for ix in -n_steps..=n_steps {
                    for iy in -n_steps..=n_steps {
                        for iz in -n_steps..=n_steps {
                            let px = centroid[0] + ix as f32 * grid_step;
                            let py = centroid[1] + iy as f32 * grid_step;
                            let pz = centroid[2] + iz as f32 * grid_step;

                            let mut density = 0.0f32;
                            for (_, spike) in &spikes {
                                let dx = px - spike.position[0];
                                let dy = py - spike.position[1];
                                let dz = pz - spike.position[2];
                                let r2 = dx * dx + dy * dy + dz * dz;
                                if r2 < 9.0 * bw2 {
                                    density += (-r2 / (2.0 * bw2)).exp();
                                }
                            }

                            if density > best_density {
                                best_density = density;
                                peak_pos = [px, py, pz];
                            }
                        }
                    }
                }

                centroid[0] = 0.7 * peak_pos[0] + 0.3 * centroid[0];
                centroid[1] = 0.7 * peak_pos[1] + 0.3 * centroid[1];
                centroid[2] = 0.7 * peak_pos[2] + 0.3 * centroid[2];
            }
        }

        let bounding_box = [
            max_pos[0] - min_pos[0],
            max_pos[1] - min_pos[1],
            max_pos[2] - min_pos[2],
        ];

        // Estimate pocket volume using voxel density method
        // Grid the space at 2Å resolution and count occupied voxels
        let voxel_size = 2.0f32;
        let estimated_volume = if bounding_box[0] > 0.0 && bounding_box[1] > 0.0 && bounding_box[2] > 0.0 {
            let nx = ((bounding_box[0] / voxel_size).ceil() as usize).max(1);
            let ny = ((bounding_box[1] / voxel_size).ceil() as usize).max(1);
            let nz = ((bounding_box[2] / voxel_size).ceil() as usize).max(1);

            // Use HashSet to count unique voxels occupied by spikes
            let mut occupied_voxels = std::collections::HashSet::new();
            for (_, spike) in &spikes {
                let vx = ((spike.position[0] - min_pos[0]) / voxel_size) as i32;
                let vy = ((spike.position[1] - min_pos[1]) / voxel_size) as i32;
                let vz = ((spike.position[2] - min_pos[2]) / voxel_size) as i32;

                // Mark this voxel and its immediate neighbors (small neighborhood)
                for dx in -1..=1 {
                    for dy in -1..=1 {
                        for dz in -1..=1 {
                            let key = (vx + dx, vy + dy, vz + dz);
                            occupied_voxels.insert(key);
                        }
                    }
                }
            }

            // Volume = occupied voxels * voxel volume
            let voxel_volume = voxel_size.powi(3);
            let raw_volume = occupied_voxels.len() as f32 * voxel_volume;

            // Apply packing efficiency correction (sphere packing ~74% efficiency)
            let pocket_volume = raw_volume * 0.74;

            // Sanity bounds: typical pockets 100-2000 Å³, large enzyme sites up to 8000 Å³
            pocket_volume.clamp(50.0, 8000.0)
        } else {
            // Degenerate case: estimate from spike count
            (spikes.len() as f32 * 15.0).clamp(50.0, 2000.0)
        };

        let avg_intensity = sum_intensity / n;
        let spike_count = spikes.len();

        // Compute druggability score
        let druggability = DruggabilityScore::from_site(estimated_volume, avg_intensity, &bounding_box);

        // Classify site
        let classification = SiteClassification::from_properties(spike_count, estimated_volume, avg_intensity);

        // Overall quality score: combines spike count, intensity, and druggability
        let spike_quality = (spike_count as f32 / 100.0).clamp(0.0, 1.0);
        let intensity_quality = (avg_intensity / 10.0).clamp(0.0, 1.0);
        let quality_score = 0.3 * spike_quality + 0.3 * intensity_quality + 0.4 * druggability.overall;

        sites.push(ClusteredBindingSite::new_with_geometric_voxel_mass(
            cluster_id,
            centroid,
            spike_count,
            spikes.iter().map(|(idx, _)| *idx).collect(),
            avg_intensity,
            estimated_volume,
            bounding_box,
            quality_score,
            druggability,
            classification,
        ));
        // aromatic_proximity: None, lining_residues: Vec::new() are
        // set by new_with_geometric_voxel_mass and populated later
        // by compute_aromatic_proximity / compute_lining_residues.
    }

    // Sort by spike count (most significant first)
    sites.sort_by(|a, b| b.spike_count.cmp(&a.spike_count));
    sites
}

impl ClusteredBindingSite {
    // ================================================================
    // Phase 1 canonical spatial-view API — post-05dbc3dc lane.
    //
    // Every canonical reader of a site centroid must use one of the
    // typed methods below, naming the physical representation it wants.
    // The `legacy_emission_centroid` field is module-private; external
    // crates and binary targets must go through `emission_compat_centroid`
    // for emission/logging or through `view` / `distance_to` / `dcc` /
    // `distance_to_ligand` for canonical logic.
    //
    // No method in this group admits a default. `view(v)` returns
    // `None` when the view is not derivable for this site, and the
    // caller must decide explicitly — silent fallback is forbidden
    // (Rule A of the lane doctrine).
    // ================================================================

    /// Canonical external constructor. Initializes both the legacy
    /// emission-compat scalar and the `GeometricVoxelMass` view of
    /// the localization manifold from the same input. All other
    /// views start `None`; later pipeline stages populate them.
    ///
    /// `geometric_voxel_mass_centroid` is the intensity²- or
    /// voxel-mass-weighted centroid produced by the clustering stage.
    /// Callers that need to track a different representation should
    /// first build the site here and then call `localization.set(...)`
    /// for each additional view.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_geometric_voxel_mass(
        cluster_id: i32,
        geometric_voxel_mass_centroid: [f32; 3],
        spike_count: usize,
        spike_indices: Vec<usize>,
        avg_intensity: f32,
        estimated_volume: f32,
        bounding_box: [f32; 3],
        quality_score: f32,
        druggability: DruggabilityScore,
        classification: SiteClassification,
    ) -> Self {
        Self {
            cluster_id,
            legacy_emission_centroid: geometric_voxel_mass_centroid,
            localization: crate::spatial_view::CentroidManifold::with_geometric_voxel_mass(
                geometric_voxel_mass_centroid,
            ),
            spike_count,
            spike_indices,
            avg_intensity,
            estimated_volume,
            bounding_box,
            quality_score,
            druggability,
            classification,
            aromatic_proximity: None,
            lining_residues: Vec::new(),
        }
    }

    /// Retrieve the requested spatial view. Returns `None` if the
    /// view is not derivable for this site — no silent fallback.
    #[inline]
    pub fn view(&self, which: crate::spatial_view::SpatialView) -> Option<[f32; 3]> {
        self.localization.view(which)
    }

    /// Euclidean distance from this site's named view to a point.
    /// `None` when the view is unpopulated.
    #[inline]
    pub fn distance_to(
        &self,
        target: [f32; 3],
        view: crate::spatial_view::SpatialView,
    ) -> Option<f32> {
        self.localization.distance_to(target, view)
    }

    /// DCC — distance to a ground-truth reference point under the
    /// named view. Typed alias for [`distance_to`] that documents
    /// intent at the call site. `None` when the view is unpopulated.
    #[inline]
    pub fn dcc(
        &self,
        ground_truth: [f32; 3],
        view: crate::spatial_view::SpatialView,
    ) -> Option<f32> {
        self.localization.dcc(ground_truth, view)
    }

    /// Distance to a ligand center under the named view. Typed alias
    /// for [`distance_to`] that documents intent. `None` when the
    /// view is unpopulated.
    #[inline]
    pub fn distance_to_ligand(
        &self,
        ligand: [f32; 3],
        view: crate::spatial_view::SpatialView,
    ) -> Option<f32> {
        self.localization.distance_to_ligand(ligand, view)
    }

    /// Compatibility scalar accessor for serialization, logging, and
    /// external report generators that need a single 3-vector to
    /// write into a PDB/PyMOL/ChimeraX/Markdown/JSON field. **This is
    /// not a canonical-logic accessor.** Calling this from merge,
    /// attribution, refinement, ranking, or validation code is a
    /// scalar-collapse violation (Rule A). Use [`view`] (named
    /// representation) or [`geometric_voxel_mass_centroid`] (named
    /// shortcut for the most common canonical-logic case).
    ///
    /// The name is intentionally long and lane-specific — it should
    /// stand out in code review when misused.
    #[inline]
    pub fn emission_compat_centroid(&self) -> [f32; 3] {
        self.legacy_emission_centroid
    }

    /// Explicit named accessor for the `GeometricVoxelMass` view.
    /// Equivalent to `site.view(SpatialView::GeometricVoxelMass).expect(...)`
    /// but far more readable at canonical call sites that index the
    /// result as `[0] / [1] / [2]` (merge, grid-bin, spike attribution,
    /// refinement).
    ///
    /// This is NOT a default or fallback — the method name states the
    /// view. Using this method at a site where a different physical
    /// representation is the right one (e.g., LiningResidues for
    /// lining-focused math) is a semantic bug and must be caught in
    /// code review.
    ///
    /// Panics if the GeometricVoxelMass view is somehow absent
    /// (`ClusteredBindingSite` construction paths always populate it,
    /// so this should be unreachable).
    #[inline]
    pub fn geometric_voxel_mass_centroid(&self) -> [f32; 3] {
        self.view(crate::spatial_view::SpatialView::GeometricVoxelMass)
            .expect(
                "ClusteredBindingSite invariant violated: \
                 GeometricVoxelMass view must always be populated by \
                 new_with_geometric_voxel_mass / set_geometric_voxel_mass_centroid",
            )
    }

    /// Canonical write path for the geometric-voxel-mass centroid.
    /// Updates both the legacy emission scalar and the manifold's
    /// `GeometricVoxelMass` view in one atomic operation so the two
    /// can never drift.
    ///
    /// Callers that need to set a *different* view (LiningResidues,
    /// HotPhase, etc.) must write to `self.localization.set(view, c)`
    /// directly — this method deliberately does not take a view
    /// parameter, because the legacy emission-compat scalar
    /// represents the geometric-voxel-mass view only.
    pub fn set_geometric_voxel_mass_centroid(&mut self, centroid: [f32; 3]) {
        self.legacy_emission_centroid = centroid;
        self.localization.set(
            crate::spatial_view::SpatialView::GeometricVoxelMass,
            centroid,
        );
    }

    // ================================================================
    // End Phase 1 canonical spatial-view API. Legacy methods below.
    // ================================================================

    /// Enhance this binding site with aromatic proximity analysis
    ///
    /// Updates the aromatic_proximity field and recalculates druggability score.
    ///
    /// # Arguments
    /// * `aromatic_positions` - List of (residue_id, aromatic_type, [x, y, z])
    ///   - aromatic_type: 0=TRP, 1=TYR, 2=PHE
    pub fn compute_aromatic_proximity(&mut self, aromatic_positions: &[(u32, u8, [f32; 3])]) {
        // Aromatic proximity is computed against the GeometricVoxelMass
        // view — the spike-weighted center of the cluster is the
        // appropriate reference for "aromatics near this pocket."
        let gvm = self
            .view(crate::spatial_view::SpatialView::GeometricVoxelMass)
            .expect("ClusteredBindingSite always carries GeometricVoxelMass view");
        let info = AromaticProximityInfo::compute(&gvm, aromatic_positions);

        // Recalculate druggability with aromatic info
        self.druggability = DruggabilityScore::from_site_with_aromatics(
            self.estimated_volume,
            self.avg_intensity,
            &self.bounding_box,
            Some(&info),
        );

        // Update quality score
        let spike_quality = (self.spike_count as f32 / 100.0).clamp(0.0, 1.0);
        let intensity_quality = (self.avg_intensity / 10.0).clamp(0.0, 1.0);
        self.quality_score = 0.25 * spike_quality
            + 0.25 * intensity_quality
            + 0.3 * self.druggability.overall
            + 0.2 * info.aromatic_score;

        self.aromatic_proximity = Some(info);
    }

    /// Check if this site has been analyzed for aromatic proximity
    pub fn has_aromatic_analysis(&self) -> bool {
        self.aromatic_proximity.is_some()
    }

    /// Get the aromatic score (0.0 if not analyzed)
    pub fn aromatic_score(&self) -> f32 {
        self.aromatic_proximity
            .as_ref()
            .map(|p| p.aromatic_score)
            .unwrap_or(0.0)
    }

    /// Compute residues lining this binding pocket
    ///
    /// Finds all residues with at least one atom within `cutoff` distance
    /// of the pocket centroid.
    ///
    /// # Arguments
    /// * `positions` - Atom positions as flat [x0,y0,z0,x1,y1,z1,...] array
    /// * `residue_ids` - Residue ID for each atom
    /// * `residue_names` - Name of each residue (indexed by residue_id)
    /// * `chain_ids` - Chain ID for each atom
    /// * `cutoff` - Distance cutoff in Angstroms (default: 5.0)
    pub fn compute_lining_residues(
        &mut self,
        positions: &[f32],
        residue_ids: &[usize],
        residue_names: &[String],
        chain_ids: &[String],
        residue_pdb_ids: &[i32],
        cutoff: f32,
    ) {
        use std::collections::HashMap;

        let n_atoms = positions.len() / 3;
        // Lining residues are computed in the neighborhood of the
        // site's GeometricVoxelMass view. Using a different view here
        // would change the definition of "lining" and must be an
        // explicit, reviewed decision — not a silent default.
        let gvm = self
            .view(crate::spatial_view::SpatialView::GeometricVoxelMass)
            .expect("ClusteredBindingSite always carries GeometricVoxelMass view");
        let cx = gvm[0];
        let cy = gvm[1];
        let cz = gvm[2];
        let cutoff_sq = cutoff * cutoff;

        // Track per-residue: (chain, resname, min_distance, atom_count)
        let mut residue_info: HashMap<(String, i32), (String, f32, usize)> = HashMap::new();

        // Accumulate pocket atom positions for centroid refinement
        let mut pocket_sum = [0.0f64; 3];
        let mut pocket_count = 0u32;

        for i in 0..n_atoms {
            let x = positions[i * 3];
            let y = positions[i * 3 + 1];
            let z = positions[i * 3 + 2];

            let dx = x - cx;
            let dy = y - cy;
            let dz = z - cz;
            let dist_sq = dx * dx + dy * dy + dz * dz;

            if dist_sq <= cutoff_sq {
                // Accumulate for pocket centroid
                pocket_sum[0] += x as f64;
                pocket_sum[1] += y as f64;
                pocket_sum[2] += z as f64;
                pocket_count += 1;

                let internal_id = residue_ids[i];
                let res_id = if internal_id < residue_pdb_ids.len() { residue_pdb_ids[internal_id] + 1 } else { (internal_id + 1) as i32 };
                let chain = chain_ids[i].clone();
                let resname = if i < residue_names.len() {
                    residue_names[i].clone()
                } else {
                    "UNK".to_string()
                };
                let dist = dist_sq.sqrt();

                let key = (chain.clone(), res_id as i32);
                residue_info
                    .entry(key)
                    .and_modify(|(_, min_d, count)| {
                        if dist < *min_d {
                            *min_d = dist;
                        }
                        *count += 1;
                    })
                    .or_insert((resname, dist, 1));
            }
        }

        // Refine centroid: shift from aromatic probe position toward pocket interior.
        // The spike centroid sits at the aromatic ring (pocket wall). The actual pocket
        // center is shifted toward the protein interior. We detect the interior direction
        // using a large (15Å) hemisphere count: more atoms on the protein-interior side
        // than the solvent-exposed side gives the correction vector and magnitude.
        if pocket_count >= 10 {
            // Protein center of mass
            let mut com = [0.0f64; 3];
            for j in 0..n_atoms {
                com[0] += positions[j * 3] as f64;
                com[1] += positions[j * 3 + 1] as f64;
                com[2] += positions[j * 3 + 2] as f64;
            }
            let na = n_atoms as f64;
            com[0] /= na;
            com[1] /= na;
            com[2] /= na;

            // Direction from spike centroid toward protein COM
            let dir = [com[0] - cx as f64, com[1] - cy as f64, com[2] - cz as f64];
            let dir_mag = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();

            if dir_mag > 0.5 {
                let ux = dir[0] / dir_mag;
                let uy = dir[1] / dir_mag;
                let uz = dir[2] / dir_mag;

                // Use 15Å sphere for asymmetry detection (much larger than lining cutoff)
                // to capture the surface-vs-interior signal that's invisible at 8Å.
                let asym_radius_sq = 15.0f64 * 15.0;
                let mut toward = 0u32;
                let mut away = 0u32;
                for j in 0..n_atoms {
                    let ax = positions[j * 3] as f64 - cx as f64;
                    let ay = positions[j * 3 + 1] as f64 - cy as f64;
                    let az = positions[j * 3 + 2] as f64 - cz as f64;
                    let d2 = ax * ax + ay * ay + az * az;
                    if d2 <= asym_radius_sq {
                        let dot = ax * ux + ay * uy + az * uz;
                        if dot > 0.0 { toward += 1; } else { away += 1; }
                    }
                }

                let total = (toward + away) as f64;
                if total > 0.0 {
                    let asymmetry = (toward as f64 - away as f64) / total;
                    // Shift proportional to asymmetry; max 4Å.
                    let shift = (asymmetry * 10.0).clamp(0.0, 4.0);
                    if shift > 0.1 {
                        // Refine the GeometricVoxelMass centroid only;
                        // the manifold's GVM view tracks lockstep via
                        // set_geometric_voxel_mass_centroid. Other
                        // views (LiningResidues, HotPhase, ...) are NOT
                        // touched here — they represent different
                        // physical quantities and must be set
                        // separately by their producing stage.
                        let new_gvm = [
                            gvm[0] + (ux * shift) as f32,
                            gvm[1] + (uy * shift) as f32,
                            gvm[2] + (uz * shift) as f32,
                        ];
                        self.set_geometric_voxel_mass_centroid(new_gvm);
                    }
                }
            }
        }

        // Convert to LiningResidue list, sorted by distance from refined centroid.
        // spike_attribution_count is left at 0 here — it is populated by the
        // caller after this function runs (if they have spike attribution data).
        self.lining_residues = residue_info
            .into_iter()
            .map(|((chain, resid), (resname, min_distance, n_atoms))| LiningResidue {
                chain,
                resid,
                resname,
                min_distance,
                n_atoms_in_pocket: n_atoms,
                spike_attribution_count: 0,
            })
            .collect();

        self.lining_residues.sort_by(|a, b| {
            a.min_distance.partial_cmp(&b.min_distance).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Recompute druggability with catalytic scoring
        self.recompute_druggability_with_catalytic();
    }

    /// Recompute druggability score including catalytic residue analysis
    ///
    /// Called after lining residues are computed to update druggability
    /// for enzyme active sites that would otherwise fail hydrophobic scoring.
    pub fn recompute_druggability_with_catalytic(&mut self) {
        self.druggability = DruggabilityScore::from_site_with_catalytic(
            self.estimated_volume,
            self.avg_intensity,
            &self.bounding_box,
            self.aromatic_proximity.as_ref(),
            &self.lining_residues,
        );
    }

    /// Get lining residue IDs as a simple list (for validation comparisons)
    pub fn lining_residue_ids(&self) -> Vec<i32> {
        self.lining_residues.iter().map(|r| r.resid).collect()
    }

    /// Get formatted residue list string (e.g., "A:PHE347, A:TRP348, A:GLU349")
    pub fn lining_residues_str(&self) -> String {
        self.lining_residues
            .iter()
            .map(|r| format!("{}:{}{}", r.chain, r.resname, r.resid))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Enhance a list of binding sites with aromatic proximity analysis
#[cfg(feature = "gpu")]
pub fn enhance_sites_with_aromatics(
    sites: &mut [ClusteredBindingSite],
    aromatic_positions: &[(u32, u8, [f32; 3])],
) {
    for site in sites.iter_mut() {
        site.compute_aromatic_proximity(aromatic_positions);
    }

    // Re-sort by quality score after enhancement
    sites.sort_by(|a, b| {
        b.quality_score.partial_cmp(&a.quality_score).unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Batch processor using persistent engine
#[cfg(feature = "gpu")]
pub struct BatchProcessor {
    engine: PersistentNhsEngine,
    config: PersistentBatchConfig,
}

#[cfg(feature = "gpu")]
impl BatchProcessor {
    /// Create batch processor
    pub fn new(config: PersistentBatchConfig) -> Result<Self> {
        let engine = PersistentNhsEngine::new(&config)?;
        Ok(Self { engine, config })
    }

    /// Process multiple topology files
    pub fn process_batch<P: AsRef<Path>>(&mut self, topology_paths: &[P]) -> Result<Vec<StructureResult>> {
        log::info!("═══════════════════════════════════════════════════════════════");
        log::info!("  PERSISTENT BATCH PROCESSING: {} structures", topology_paths.len());
        log::info!("═══════════════════════════════════════════════════════════════");

        let batch_start = Instant::now();
        let mut results = Vec::with_capacity(topology_paths.len());

        for (idx, path) in topology_paths.iter().enumerate() {
            let path = path.as_ref();
            log::info!("\n[{}/{}] Processing: {}",
                idx + 1, topology_paths.len(), path.display());

            // Load topology
            let topology = PrismPrepTopology::load(path)
                .with_context(|| format!("Failed to load topology: {}", path.display()))?;

            let structure_id = path.file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "unknown".to_string());

            let struct_start = Instant::now();

            // Load into engine
            self.engine.load_topology(&topology)?;

            // Configure unified cryo-UV protocol
            let cryo_uv_protocol = CryoUvProtocol {
                start_temp: self.config.cryo_temp,
                end_temp: self.config.temperature,
                cold_hold_steps: self.config.cryo_hold,
                ramp_steps: self.config.convergence_steps / 2,
                warm_hold_steps: self.config.convergence_steps / 2,
                current_step: 0,
                // UV-LIF coupling (validated parameters)
                uv_burst_energy: 30.0,
                uv_burst_interval: 500,
                uv_burst_duration: 50,
                scan_wavelengths: vec![280.0, 274.0, 258.0],  // TRP, TYR, PHE
                wavelength_dwell_steps: 500,
                ramp_down_steps: 0,
                cold_return_steps: 0,
                stepped_holds: vec![],
            };
            self.engine.set_cryo_uv_protocol(cryo_uv_protocol)?;

            // Run all phases
            let total_steps = self.config.survey_steps
                + self.config.convergence_steps
                + self.config.precision_steps;

            let summary = self.engine.run(total_steps)?;

            let wall_time_ms = struct_start.elapsed().as_millis() as u64;

            // Collect spike events
            let spike_events = self.engine.get_spike_events();

            // RT-accelerated clustering of spike positions
            let (clustered_sites, clustering_stats) = if !spike_events.is_empty() {
                // Extract positions for clustering
                let spike_positions: Vec<f32> = spike_events.iter()
                    .flat_map(|s| s.position.iter().copied())
                    .collect();

                // Cluster using RT cores (or fallback)
                let used_rt = self.engine.has_rt_clustering();
                match self.engine.cluster_spikes(&spike_positions) {
                    Ok(result) => {
                        // Phase 2: route batch-mode site construction through
                        // the same audit-spine transform as the single-structure
                        // path above. Abort violations propagate out of
                        // `process_batch` as a run error; quarantine violations
                        // are logged and the output is accepted.
                        #[cfg(feature = "gpu")]
                        let sites = {
                            use crate::transform::clustering_to_clustered_sites::{
                                ClusteringInput, ClusteringToClusteredSites,
                            };
                            use crate::transform::AuditedTransform;
                            let outcome = ClusteringToClusteredSites::new().apply(ClusteringInput {
                                spike_events: &spike_events,
                                clustering_result: &result,
                                engine: self.engine.engine.as_ref(),
                            });
                            let (sites, quarantined) = outcome
                                .into_result()
                                .map_err(|aborted| anyhow::anyhow!("{aborted}"))?;
                            if !quarantined.is_empty() {
                                log::warn!(
                                    "transform clustering_to_clustered_sites: {} quarantined site(s) in process_batch",
                                    quarantined.len()
                                );
                            }
                            sites
                        };
                        #[cfg(not(feature = "gpu"))]
                        let sites = build_clustered_sites(&spike_events, &result, None);
                        let stats = ClusteringStats {
                            num_clusters: result.num_clusters,
                            total_neighbors: result.total_neighbors,
                            gpu_time_ms: result.gpu_time_ms,
                            used_rt_cores: used_rt,
                        };
                        log::info!("  📊 Clustered {} spikes → {} binding sites ({:.1}ms, {})",
                            spike_events.len(),
                            sites.len(),
                            result.gpu_time_ms,
                            if used_rt { "RT cores" } else { "fallback" });
                        (sites, Some(stats))
                    }
                    Err(e) => {
                        log::warn!("  ⚠️ Clustering failed: {}", e);
                        (Vec::new(), None)
                    }
                }
            } else {
                (Vec::new(), None)
            };

            results.push(StructureResult {
                structure_id,
                total_steps,
                wall_time_ms,
                spike_events,
                snapshots: self.engine.get_snapshots(),
                final_temperature: summary.end_temperature,
                clustered_sites,
                clustering_stats,
            });

            log::info!("  ✓ Completed in {}ms ({:.1} steps/sec)",
                wall_time_ms,
                total_steps as f64 / (wall_time_ms as f64 / 1000.0));
        }

        let total_time = batch_start.elapsed();
        let stats = self.engine.stats();

        log::info!("\n═══════════════════════════════════════════════════════════════");
        log::info!("  BATCH COMPLETE");
        log::info!("═══════════════════════════════════════════════════════════════");
        log::info!("  Structures processed: {}", stats.structures_processed);
        log::info!("  Total steps: {}", stats.total_steps_run);
        log::info!("  Total wall time: {:.1}s", total_time.as_secs_f64());
        log::info!("  Overhead saved (persistent): {}ms", stats.overhead_saved_ms);
        log::info!("  Avg throughput: {:.0} steps/sec",
            stats.total_steps_run as f64 / total_time.as_secs_f64());

        Ok(results)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// VISUALIZATION OUTPUT FORMATTERS
// ═══════════════════════════════════════════════════════════════════════════════

/// Output formatter for binding site visualization
pub struct BindingSiteFormatter;

impl BindingSiteFormatter {
    /// Generate PDB file with pseudo-atoms at binding site centroids
    ///
    /// Creates HETATM records with:
    /// - Atom name "BS" (Binding Site)
    /// - Residue name based on classification (DRG=druggable, CRY=cryptic, etc.)
    /// - B-factor = quality score * 100
    /// - Occupancy = druggability score
    pub fn to_pdb(sites: &[ClusteredBindingSite]) -> String {
        let mut pdb = String::new();
        pdb.push_str("REMARK   PRISM4D Binding Site Detection Results\n");
        pdb.push_str("REMARK   Pseudo-atoms at binding site centroids\n");
        pdb.push_str("REMARK   B-factor = quality score * 100\n");
        pdb.push_str("REMARK   Occupancy = druggability score\n");
        pdb.push_str("REMARK\n");

        for (idx, site) in sites.iter().enumerate() {
            let atom_num = idx + 1;
            let res_name = match site.classification {
                SiteClassification::ActiveSite => "ACT",
                SiteClassification::Allosteric => "ALO",
                SiteClassification::Cryptic => "CRY",
                SiteClassification::PpiSurface => "PPI",
                SiteClassification::MembraneInterface => "MEM",
                SiteClassification::Unknown => "UNK",
            };

            // PDB HETATM format
            // HETATM    1  BS  DRG A   1      10.000  20.000  30.000  0.80 50.00
            // Emission-compat path: PDB HETATM record uses the
            // legacy emission centroid scalar. Canonical multi-view
            // adjudication is handled by Phase 4; this serializer
            // stays on the compat accessor for backward compatibility
            // with external PDB consumers.
            let c = site.emission_compat_centroid();
            pdb.push_str(&format!(
                "HETATM{:5}  BS  {} A{:4}    {:8.3}{:8.3}{:8.3}{:6.2}{:6.2}\n",
                atom_num,
                res_name,
                atom_num,
                c[0],
                c[1],
                c[2],
                site.druggability.overall,
                site.quality_score * 100.0,
            ));
        }

        pdb.push_str("END\n");
        pdb
    }

    /// Generate PyMOL script for visualization
    ///
    /// Creates industry-standard pocket visualization with:
    /// - Lining residues shown as sticks
    /// - Pocket surface (transparent)
    /// - Residue type coloring (catalytic=magenta, aromatic=green, hydrophobic=yellow)
    /// - Centroid marker (small sphere)
    pub fn to_pymol(sites: &[ClusteredBindingSite], structure_name: &str) -> String {
        let mut script = String::new();

        script.push_str("# PRISM4D Binding Site Visualization\n");
        script.push_str("# Generated by prism-nhs\n");
        script.push_str("# Industry-standard pocket visualization\n\n");

        // Setup commands
        script.push_str("# Setup\n");
        script.push_str("bg_color white\n");
        script.push_str("set cartoon_fancy_helices, 1\n");
        script.push_str("set cartoon_side_chain_helper, 1\n");
        script.push_str("set surface_quality, 1\n\n");

        // Load structure placeholder
        if !structure_name.is_empty() {
            script.push_str(&format!("# Load your structure (adjust path as needed)\n"));
            script.push_str(&format!("# load {}.pdb, protein\n", structure_name));
            script.push_str("# show cartoon, protein\n");
            script.push_str("# color gray80, protein\n\n");
        }

        // Process each binding site — emission layer (PyMOL/ChimeraX).
        for (idx, site) in sites.iter().enumerate() {
            let site_num = idx + 1;
            // Emission-compat scalar for pseudoatom placement.
            let [x, y, z] = site.emission_compat_centroid();
            let druggable_tag = if site.druggability.is_druggable { " [DRUGGABLE]" } else { "" };

            script.push_str(&format!("# ========== Site {} ({:?}){} ==========\n",
                site_num, site.classification, druggable_tag));

            // Skip if no lining residues
            if site.lining_residues.is_empty() {
                script.push_str(&format!("# No lining residues for site {}\n", site_num));
                script.push_str(&format!("pseudoatom site_{}_center, pos=[{:.3}, {:.3}, {:.3}]\n",
                    site_num, x, y, z));
                script.push_str(&format!("show spheres, site_{}_center\n", site_num));
                script.push_str(&format!("set sphere_scale, 1.0, site_{}_center\n", site_num));
                script.push_str(&format!("color red, site_{}_center\n\n", site_num));
                continue;
            }

            // Build residue selection string
            let residue_sel: Vec<String> = site.lining_residues.iter()
                .map(|r| format!("(chain {} and resi {})", r.chain, r.resid))
                .collect();
            let sel_str = residue_sel.join(" or ");

            // Create selection for pocket lining residues
            script.push_str(&format!("select pocket_{}_lining, {}\n", site_num, sel_str));

            // Show lining residues as sticks
            script.push_str(&format!("show sticks, pocket_{}_lining\n", site_num));
            script.push_str(&format!("set stick_radius, 0.15, pocket_{}_lining\n", site_num));

            // Color by residue type
            let catalytic = ["GLU", "ASP", "HIS", "SER", "CYS", "LYS"];
            let aromatic = ["PHE", "TYR", "TRP"];
            let hydrophobic = ["ALA", "VAL", "LEU", "ILE", "MET", "PRO"];

            // Categorize residues
            let cat_residues: Vec<_> = site.lining_residues.iter()
                .filter(|r| catalytic.contains(&r.resname.as_str()))
                .collect();
            let aro_residues: Vec<_> = site.lining_residues.iter()
                .filter(|r| aromatic.contains(&r.resname.as_str()))
                .collect();
            let hydro_residues: Vec<_> = site.lining_residues.iter()
                .filter(|r| hydrophobic.contains(&r.resname.as_str()))
                .collect();

            // Color catalytic residues (magenta/hotpink)
            if !cat_residues.is_empty() {
                let cat_sel: Vec<String> = cat_residues.iter()
                    .map(|r| format!("(chain {} and resi {})", r.chain, r.resid))
                    .collect();
                script.push_str(&format!("select pocket_{}_catalytic, {}\n", site_num, cat_sel.join(" or ")));
                script.push_str(&format!("color magenta, pocket_{}_catalytic\n", site_num));
            }

            // Color aromatic residues (green)
            if !aro_residues.is_empty() {
                let aro_sel: Vec<String> = aro_residues.iter()
                    .map(|r| format!("(chain {} and resi {})", r.chain, r.resid))
                    .collect();
                script.push_str(&format!("select pocket_{}_aromatic, {}\n", site_num, aro_sel.join(" or ")));
                script.push_str(&format!("color forest, pocket_{}_aromatic\n", site_num));
            }

            // Color hydrophobic residues (yellow)
            if !hydro_residues.is_empty() {
                let hydro_sel: Vec<String> = hydro_residues.iter()
                    .map(|r| format!("(chain {} and resi {})", r.chain, r.resid))
                    .collect();
                script.push_str(&format!("select pocket_{}_hydrophobic, {}\n", site_num, hydro_sel.join(" or ")));
                script.push_str(&format!("color tv_yellow, pocket_{}_hydrophobic\n", site_num));
            }

            // Create pocket surface (transparent)
            script.push_str(&format!("create pocket_{}_surface, pocket_{}_lining\n", site_num, site_num));
            script.push_str(&format!("show surface, pocket_{}_surface\n", site_num));
            script.push_str(&format!("set surface_color, slate, pocket_{}_surface\n", site_num));
            script.push_str(&format!("set transparency, 0.7, pocket_{}_surface\n", site_num));

            // Small centroid marker
            script.push_str(&format!("pseudoatom pocket_{}_center, pos=[{:.3}, {:.3}, {:.3}]\n",
                site_num, x, y, z));
            script.push_str(&format!("show spheres, pocket_{}_center\n", site_num));
            script.push_str(&format!("set sphere_scale, 0.5, pocket_{}_center\n", site_num));
            let center_color = if site.druggability.is_druggable { "red" } else { "gray50" };
            script.push_str(&format!("color {}, pocket_{}_center\n", center_color, site_num));

            // Group all pocket objects
            script.push_str(&format!(
                "group pocket_{}, pocket_{}_lining pocket_{}_surface pocket_{}_center",
                site_num, site_num, site_num, site_num
            ));
            if !cat_residues.is_empty() {
                script.push_str(&format!(" pocket_{}_catalytic", site_num));
            }
            if !aro_residues.is_empty() {
                script.push_str(&format!(" pocket_{}_aromatic", site_num));
            }
            if !hydro_residues.is_empty() {
                script.push_str(&format!(" pocket_{}_hydrophobic", site_num));
            }
            script.push_str("\n\n");
        }

        // Group all pockets
        if !sites.is_empty() {
            let pocket_groups: Vec<String> = (1..=sites.len()).map(|i| format!("pocket_{}", i)).collect();
            script.push_str(&format!("group all_pockets, {}\n\n", pocket_groups.join(" ")));
        }

        // Add legend and usage tips
        script.push_str("# ========== Color Legend ==========\n");
        script.push_str("# Magenta = Catalytic residues (GLU, ASP, HIS, SER, CYS, LYS)\n");
        script.push_str("# Green = Aromatic residues (PHE, TYR, TRP) - pi-stacking\n");
        script.push_str("# Yellow = Hydrophobic residues (ALA, VAL, LEU, ILE, MET, PRO)\n");
        script.push_str("# Slate surface = Pocket cavity\n");
        script.push_str("# Red sphere = Druggable site centroid\n");
        script.push_str("# Gray sphere = Non-druggable site centroid\n\n");

        script.push_str("# ========== Usage Tips ==========\n");
        script.push_str("# To view specific pocket: disable all_pockets, enable pocket_N\n");
        script.push_str("# To zoom on pocket: zoom pocket_N_lining\n");
        script.push_str("# To show H-bonds: select donors, pocket_N_lining; distance hbonds, donors, acceptors, 3.5\n");
        script.push_str("# To label residues: label pocket_N_lining and name CA, \"%s%s\" % (resn, resi)\n");

        script
    }

    /// Generate ChimeraX script for visualization
    ///
    /// Creates industry-standard pocket visualization with:
    /// - Lining residues shown as sticks
    /// - Pocket surface (transparent)
    /// - Residue type coloring
    pub fn to_chimerax(sites: &[ClusteredBindingSite], structure_name: &str) -> String {
        let mut script = String::new();

        script.push_str("# PRISM4D Binding Site Visualization\n");
        script.push_str("# Generated by prism-nhs\n");
        script.push_str("# Industry-standard pocket visualization\n\n");

        // Setup
        script.push_str("# Setup\n");
        script.push_str("set bgColor white\n");
        script.push_str("lighting soft\n\n");

        // Load structure placeholder
        if !structure_name.is_empty() {
            script.push_str("# Load your structure (adjust path as needed)\n");
            script.push_str(&format!("# open {}.pdb\n", structure_name));
            script.push_str("# cartoon\n");
            script.push_str("# color #1 gray80\n\n");
        }

        // Process each binding site — emission layer (PyMOL/ChimeraX).
        for (idx, site) in sites.iter().enumerate() {
            let site_num = idx + 1;
            // Emission-compat scalar for pseudoatom placement.
            let [x, y, z] = site.emission_compat_centroid();
            let druggable_tag = if site.druggability.is_druggable { " [DRUGGABLE]" } else { "" };

            script.push_str(&format!("# ========== Site {} ({:?}){} ==========\n",
                site_num, site.classification, druggable_tag));

            // Skip if no lining residues
            if site.lining_residues.is_empty() {
                script.push_str(&format!("# No lining residues for site {}\n", site_num));
                script.push_str(&format!("marker #10{} position {:.3},{:.3},{:.3} color red radius 2.0\n\n",
                    site_num, x, y, z));
                continue;
            }

            // Build residue selection string for ChimeraX
            let residue_sel: Vec<String> = site.lining_residues.iter()
                .map(|r| format!("/{}:{}", r.chain, r.resid))
                .collect();
            let sel_str = residue_sel.join(",");

            // Select and show lining residues
            script.push_str(&format!("# Pocket {} lining residues\n", site_num));
            script.push_str(&format!("name pocket{}_lining #1{}\n", site_num, sel_str));
            script.push_str(&format!("show pocket{}_lining atoms\n", site_num));
            script.push_str(&format!("style pocket{}_lining stick\n", site_num));

            // Categorize and color residues
            let catalytic = ["GLU", "ASP", "HIS", "SER", "CYS", "LYS"];
            let aromatic = ["PHE", "TYR", "TRP"];
            let hydrophobic = ["ALA", "VAL", "LEU", "ILE", "MET", "PRO"];

            // Color catalytic residues (magenta)
            let cat_residues: Vec<_> = site.lining_residues.iter()
                .filter(|r| catalytic.contains(&r.resname.as_str()))
                .collect();
            if !cat_residues.is_empty() {
                let cat_sel: Vec<String> = cat_residues.iter()
                    .map(|r| format!("/{}:{}", r.chain, r.resid))
                    .collect();
                script.push_str(&format!("color #1{} magenta\n", cat_sel.join(",")));
            }

            // Color aromatic residues (forest green)
            let aro_residues: Vec<_> = site.lining_residues.iter()
                .filter(|r| aromatic.contains(&r.resname.as_str()))
                .collect();
            if !aro_residues.is_empty() {
                let aro_sel: Vec<String> = aro_residues.iter()
                    .map(|r| format!("/{}:{}", r.chain, r.resid))
                    .collect();
                script.push_str(&format!("color #1{} forest green\n", aro_sel.join(",")));
            }

            // Color hydrophobic residues (gold)
            let hydro_residues: Vec<_> = site.lining_residues.iter()
                .filter(|r| hydrophobic.contains(&r.resname.as_str()))
                .collect();
            if !hydro_residues.is_empty() {
                let hydro_sel: Vec<String> = hydro_residues.iter()
                    .map(|r| format!("/{}:{}", r.chain, r.resid))
                    .collect();
                script.push_str(&format!("color #1{} gold\n", hydro_sel.join(",")));
            }

            // Create pocket surface
            script.push_str(&format!("surface pocket{}_lining\n", site_num));
            script.push_str(&format!("transparency pocket{}_lining 70\n", site_num));
            script.push_str(&format!("color pocket{}_lining slate gray\n", site_num));

            // Centroid marker
            let marker_color = if site.druggability.is_druggable { "red" } else { "gray" };
            script.push_str(&format!("marker #10{} position {:.3},{:.3},{:.3} color {} radius 1.0\n",
                site_num, x, y, z, marker_color));

            // Label
            script.push_str(&format!("2dlabels text \"Pocket {} ({:.0}% druggable){}\" xpos 0.02 ypos {:.2} color black size 14\n",
                site_num,
                site.druggability.overall * 100.0,
                if site.druggability.is_druggable { " *" } else { "" },
                0.95 - idx as f32 * 0.04
            ));

            script.push_str("\n");
        }

        // Legend
        script.push_str("# ========== Color Legend ==========\n");
        script.push_str("# Magenta = Catalytic residues (GLU, ASP, HIS, SER, CYS, LYS)\n");
        script.push_str("# Forest Green = Aromatic residues (PHE, TYR, TRP)\n");
        script.push_str("# Gold = Hydrophobic residues (ALA, VAL, LEU, ILE, MET, PRO)\n");
        script.push_str("# Slate surface = Pocket cavity\n");
        script.push_str("# Red marker = Druggable site centroid\n");
        script.push_str("# Gray marker = Non-druggable site centroid\n\n");

        script.push_str("# ========== Usage ==========\n");
        script.push_str("# To view pocket: view pocket1_lining\n");
        script.push_str("# To hide surface: hide pocket1_lining surface\n");
        script.push_str("# To show H-bonds: hbonds pocket1_lining\n");

        script
    }

    /// Generate summary report in Markdown format
    pub fn to_markdown_report(
        sites: &[ClusteredBindingSite],
        structure_name: &str,
        persistence: Option<&PersistenceAnalysis>,
    ) -> String {
        let mut report = String::new();

        report.push_str(&format!("# PRISM4D Binding Site Analysis: {}\n\n", structure_name));

        // Summary statistics
        report.push_str("## Summary\n\n");
        report.push_str(&format!("- **Total Sites Detected:** {}\n", sites.len()));

        let druggable_count = sites.iter().filter(|s| s.druggability.is_druggable).count();
        report.push_str(&format!("- **Druggable Sites:** {}\n", druggable_count));

        // Classification breakdown
        let mut class_counts = std::collections::HashMap::new();
        for site in sites {
            *class_counts.entry(format!("{:?}", site.classification)).or_insert(0) += 1;
        }
        report.push_str("\n### Classification Breakdown\n\n");
        for (class, count) in class_counts {
            report.push_str(&format!("- {}: {}\n", class, count));
        }

        // Persistence info if available
        if let Some(pers) = persistence {
            report.push_str("\n## Persistence Analysis\n\n");
            report.push_str(&format!("- **Total Frames Analyzed:** {}\n", pers.total_frames));
            report.push_str(&format!("- **Persistent Sites (>50% frames):** {}\n", pers.persistent_count));
            report.push_str(&format!("- **Transient Sites (<50% frames):** {}\n", pers.transient_count));
            report.push_str(&format!("- **Average Site Lifetime:** {:.1} frames\n", pers.avg_lifetime));
        }

        // Top sites table
        report.push_str("\n## Top Binding Sites\n\n");
        report.push_str("| Rank | Position (Å) | Volume (Å³) | Spikes | Quality | Druggable | Class |\n");
        report.push_str("|------|--------------|-------------|--------|---------|-----------|-------|\n");

        for (idx, site) in sites.iter().take(10).enumerate() {
            let c = site.emission_compat_centroid();
            report.push_str(&format!(
                "| {} | ({:.1}, {:.1}, {:.1}) | {:.0} | {} | {:.2} | {} | {:?} |\n",
                idx + 1,
                c[0], c[1], c[2],
                site.estimated_volume,
                site.spike_count,
                site.quality_score,
                if site.druggability.is_druggable { "✓" } else { "✗" },
                site.classification,
            ));
        }

        // Lining residues for top sites
        report.push_str("\n## Binding Site Residues (5Å cutoff)\n\n");
        for (idx, site) in sites.iter().take(10).enumerate() {
            if !site.lining_residues.is_empty() {
                report.push_str(&format!("### Site {} ({:?})\n\n", idx + 1, site.classification));
                report.push_str("| Chain | ResID | ResName | Distance (Å) | Atoms |\n");
                report.push_str("|-------|-------|---------|--------------|-------|\n");
                for res in site.lining_residues.iter().take(20) {
                    report.push_str(&format!(
                        "| {} | {} | {} | {:.2} | {} |\n",
                        res.chain, res.resid, res.resname, res.min_distance, res.n_atoms_in_pocket
                    ));
                }
                if site.lining_residues.len() > 20 {
                    report.push_str(&format!("| ... | {} more residues | | | |\n",
                        site.lining_residues.len() - 20));
                }
                report.push_str("\n");
            }
        }

        report
    }
}

/// Write binding sites to multiple visualization formats
pub fn write_binding_site_visualizations(
    sites: &[ClusteredBindingSite],
    base_path: &std::path::Path,
    structure_name: &str,
) -> Result<()> {
    use std::fs;
    use std::io::Write;

    // Write PDB
    let pdb_path = base_path.with_extension("binding_sites.pdb");
    let mut pdb_file = fs::File::create(&pdb_path)?;
    pdb_file.write_all(BindingSiteFormatter::to_pdb(sites).as_bytes())?;
    log::info!("Wrote binding sites PDB: {}", pdb_path.display());

    // Write PyMOL script
    let pml_path = base_path.with_extension("binding_sites.pml");
    let mut pml_file = fs::File::create(&pml_path)?;
    pml_file.write_all(BindingSiteFormatter::to_pymol(sites, structure_name).as_bytes())?;
    log::info!("Wrote PyMOL script: {}", pml_path.display());

    // Write ChimeraX script
    let cxc_path = base_path.with_extension("binding_sites.cxc");
    let mut cxc_file = fs::File::create(&cxc_path)?;
    cxc_file.write_all(BindingSiteFormatter::to_chimerax(sites, structure_name).as_bytes())?;
    log::info!("Wrote ChimeraX script: {}", cxc_path.display());

    // Write Markdown report
    let md_path = base_path.with_extension("binding_sites.md");
    let mut md_file = fs::File::create(&md_path)?;
    md_file.write_all(BindingSiteFormatter::to_markdown_report(sites, structure_name, None).as_bytes())?;
    log::info!("Wrote Markdown report: {}", md_path.display());

    Ok(())
}

// Stub for non-GPU builds
#[cfg(not(feature = "gpu"))]
pub struct PersistentNhsEngine;

#[cfg(not(feature = "gpu"))]
impl PersistentNhsEngine {
    pub fn new(_config: &PersistentBatchConfig) -> Result<Self> {
        bail!("GPU feature required for PersistentNhsEngine")
    }
}

#[cfg(not(feature = "gpu"))]
pub struct BatchProcessor;

#[cfg(not(feature = "gpu"))]
impl BatchProcessor {
    pub fn new(_config: PersistentBatchConfig) -> Result<Self> {
        bail!("GPU feature required for BatchProcessor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = PersistentBatchConfig::default();
        assert_eq!(config.max_atoms, 20000);
        assert_eq!(config.temperature, 300.0);
    }
}

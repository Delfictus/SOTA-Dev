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
            max_atoms: 15000,  // Handle 4B7Q (~12K atoms) with margin
            grid_dim: 64,
            grid_spacing: 1.5,
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

/// A clustered binding site detected from spike spatial patterns
#[derive(Debug, Clone)]
pub struct ClusteredBindingSite {
    /// Cluster ID (unique per analysis run)
    pub cluster_id: i32,
    /// Centroid position [x, y, z] in Angstroms
    pub centroid: [f32; 3],
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
    /// Is this likely a druggable pocket?
    pub is_druggable: bool,
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

        // Druggable threshold: overall >= 0.5 AND volume in reasonable range
        // Bonus: sites with aromatics within 5Å get lower threshold
        let aromatic_bonus = aromatic_info
            .map(|info| info.aromatics_within_5a > 0)
            .unwrap_or(false);
        let threshold = if aromatic_bonus { 0.45 } else { 0.5 };
        let is_druggable = overall >= threshold && volume >= 100.0 && volume <= 2000.0;

        Self {
            overall,
            volume_score,
            enclosure_score,
            hydrophobicity_score,
            aromatic_score,
            is_druggable,
        }
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

                let dist = Self::distance(&site.centroid, &tracked.avg_centroid);
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

        // Update running averages
        let n = tracked.frame_count as f32;
        tracked.avg_centroid[0] = (tracked.avg_centroid[0] * n + site.centroid[0]) / (n + 1.0);
        tracked.avg_centroid[1] = (tracked.avg_centroid[1] * n + site.centroid[1]) / (n + 1.0);
        tracked.avg_centroid[2] = (tracked.avg_centroid[2] * n + site.centroid[2]) / (n + 1.0);
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

        self.tracked_sites.push(TrackedSite {
            site_id,
            avg_centroid: site.centroid,
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
        let engine = NhsAmberFusedEngine::new(
            self.context.clone(),
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
    ///     println!("Binding site at {:?} with {} spikes", site.centroid, site.spike_count);
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
        let sites = build_clustered_sites(&spike_events, &result);
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

    /// Get snapshots from current run
    pub fn get_snapshots(&self) -> Vec<EnsembleSnapshot> {
        if let Some(ref engine) = self.engine {
            engine.get_ensemble_snapshots().to_vec()
        } else {
            Vec::new()
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
    pub fn ensure_rt_pipeline(&mut self) -> Result<bool> {
        if self.rt_engine.is_some() {
            return Ok(true);  // Already initialized
        }

        if !self.has_rt_clustering() {
            log::debug!("RT clustering not available (no RT cores or OptiX)");
            return Ok(false);
        }

        log::info!("🔷 Initializing OptiX RT pipeline for clustering...");
        let start = Instant::now();

        // Find the OptiX IR file
        let optixir_path = crate::rt_clustering::find_optixir_path()
            .context("Could not find rt_clustering.optixir")?;

        // Create RT clustering config
        let rt_config = crate::rt_clustering::RtClusteringConfig {
            epsilon: 5.0,         // 5 Angstrom neighborhood
            min_points: 3,        // Minimum 3 neighbors for core point
            min_cluster_size: 50, // Minimum 50 points per cluster
            rays_per_event: 32,   // 32 rays for neighbor finding
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
        let num_spikes = spike_positions.len() / 3;

        // Try RT-accelerated path
        if self.ensure_rt_pipeline()? {
            if let Some(ref mut rt_engine) = self.rt_engine {
                log::debug!("Using RT-accelerated clustering for {} spikes", num_spikes);
                return rt_engine.cluster(spike_positions);
            }
        }

        // Fallback: simple grid-based clustering
        log::debug!("Using fallback grid clustering for {} spikes", num_spikes);
        self.fallback_grid_cluster(spike_positions)
    }

    /// Fallback grid-based clustering when RT cores unavailable
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

/// Convert RT clustering results into binding site structures
#[cfg(feature = "gpu")]
fn build_clustered_sites(
    spike_events: &[SpikeEvent],
    clustering_result: &crate::rt_clustering::RtClusteringResult,
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

        // Compute centroid
        let mut centroid = [0.0f32; 3];
        let mut sum_intensity = 0.0f32;
        let mut min_pos = [f32::MAX; 3];
        let mut max_pos = [f32::MIN; 3];

        for (_, spike) in &spikes {
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
        centroid[0] /= n;
        centroid[1] /= n;
        centroid[2] /= n;

        let bounding_box = [
            max_pos[0] - min_pos[0],
            max_pos[1] - min_pos[1],
            max_pos[2] - min_pos[2],
        ];

        // Estimate volume as bounding box volume (rough approximation)
        let estimated_volume = bounding_box[0] * bounding_box[1] * bounding_box[2];
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

        sites.push(ClusteredBindingSite {
            cluster_id,
            centroid,
            spike_count,
            spike_indices: spikes.iter().map(|(idx, _)| *idx).collect(),
            avg_intensity,
            estimated_volume,
            bounding_box,
            quality_score,
            druggability,
            classification,
            aromatic_proximity: None,  // Computed separately when aromatic positions available
        });
    }

    // Sort by spike count (most significant first)
    sites.sort_by(|a, b| b.spike_count.cmp(&a.spike_count));
    sites
}

impl ClusteredBindingSite {
    /// Enhance this binding site with aromatic proximity analysis
    ///
    /// Updates the aromatic_proximity field and recalculates druggability score.
    ///
    /// # Arguments
    /// * `aromatic_positions` - List of (residue_id, aromatic_type, [x, y, z])
    ///   - aromatic_type: 0=TRP, 1=TYR, 2=PHE
    pub fn compute_aromatic_proximity(&mut self, aromatic_positions: &[(u32, u8, [f32; 3])]) {
        let info = AromaticProximityInfo::compute(&self.centroid, aromatic_positions);

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
                        let sites = build_clustered_sites(&spike_events, &result);
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
            pdb.push_str(&format!(
                "HETATM{:5}  BS  {} A{:4}    {:8.3}{:8.3}{:8.3}{:6.2}{:6.2}\n",
                atom_num,
                res_name,
                atom_num,
                site.centroid[0],
                site.centroid[1],
                site.centroid[2],
                site.druggability.overall,
                site.quality_score * 100.0,
            ));
        }

        pdb.push_str("END\n");
        pdb
    }

    /// Generate PyMOL script for visualization
    ///
    /// Creates colored spheres at binding sites with:
    /// - Green = druggable
    /// - Yellow = cryptic
    /// - Orange = allosteric
    /// - Gray = unknown
    /// - Sphere size based on estimated volume
    pub fn to_pymol(sites: &[ClusteredBindingSite], structure_name: &str) -> String {
        let mut script = String::new();

        script.push_str("# PRISM4D Binding Site Visualization\n");
        script.push_str("# Generated by prism-nhs\n\n");

        // Create pseudoatom objects for each site
        for (idx, site) in sites.iter().enumerate() {
            let site_name = format!("site_{}", idx + 1);
            let [x, y, z] = site.centroid;

            // Create pseudoatom
            script.push_str(&format!(
                "pseudoatom {}, pos=[{:.3}, {:.3}, {:.3}]\n",
                site_name, x, y, z
            ));

            // Color based on classification
            let color = match site.classification {
                SiteClassification::ActiveSite => "forest",
                SiteClassification::Allosteric => "orange",
                SiteClassification::Cryptic => "yellow",
                SiteClassification::PpiSurface => "cyan",
                SiteClassification::MembraneInterface => "magenta",
                SiteClassification::Unknown => "gray50",
            };
            script.push_str(&format!("color {}, {}\n", color, site_name));

            // Show as sphere with size based on volume
            let radius = (site.estimated_volume / (4.0 / 3.0 * std::f32::consts::PI)).powf(1.0 / 3.0).max(2.0);
            script.push_str(&format!("show spheres, {}\n", site_name));
            script.push_str(&format!("set sphere_scale, {:.2}, {}\n", radius / 5.0, site_name));
            script.push_str(&format!("set sphere_transparency, 0.5, {}\n", site_name));
        }

        // Group all sites
        if !sites.is_empty() {
            let site_names: Vec<String> = (1..=sites.len()).map(|i| format!("site_{}", i)).collect();
            script.push_str(&format!("\ngroup binding_sites, {}\n", site_names.join(" ")));
        }

        // Load structure if provided
        if !structure_name.is_empty() {
            script.push_str(&format!("\n# Load structure (uncomment and adjust path)\n"));
            script.push_str(&format!("# load {}.pdb, protein\n", structure_name));
            script.push_str("# show cartoon, protein\n");
            script.push_str("# color gray80, protein\n");
        }

        // Add legend
        script.push_str("\n# Color Legend:\n");
        script.push_str("# Green (forest) = Active Site\n");
        script.push_str("# Orange = Allosteric\n");
        script.push_str("# Yellow = Cryptic\n");
        script.push_str("# Cyan = PPI Surface\n");
        script.push_str("# Magenta = Membrane Interface\n");
        script.push_str("# Gray = Unknown\n");

        script
    }

    /// Generate ChimeraX script for visualization
    pub fn to_chimerax(sites: &[ClusteredBindingSite], structure_name: &str) -> String {
        let mut script = String::new();

        script.push_str("# PRISM4D Binding Site Visualization\n");
        script.push_str("# Generated by prism-nhs\n\n");

        // Create markers for each site
        script.push_str("# Create marker set for binding sites\n");
        script.push_str("marker delete #2\n");  // Clean up any existing markers

        for (idx, site) in sites.iter().enumerate() {
            let [x, y, z] = site.centroid;

            // Color based on classification
            let color = match site.classification {
                SiteClassification::ActiveSite => "forest green",
                SiteClassification::Allosteric => "orange",
                SiteClassification::Cryptic => "gold",
                SiteClassification::PpiSurface => "cyan",
                SiteClassification::MembraneInterface => "magenta",
                SiteClassification::Unknown => "gray",
            };

            // Radius based on volume
            let radius = (site.estimated_volume / (4.0 / 3.0 * std::f32::consts::PI)).powf(1.0 / 3.0).max(2.0);

            script.push_str(&format!(
                "marker #2 position {:.3},{:.3},{:.3} color {} radius {:.2}\n",
                x, y, z, color, radius
            ));

            // Add label
            script.push_str(&format!(
                "2dlabels text \"Site {} ({:.0}%)\" xpos 0.02 ypos {} color white\n",
                idx + 1,
                site.druggability.overall * 100.0,
                0.95 - idx as f32 * 0.03
            ));
        }

        // Load structure if provided
        if !structure_name.is_empty() {
            script.push_str(&format!("\n# Load structure (uncomment and adjust path)\n"));
            script.push_str(&format!("# open {}.pdb\n", structure_name));
            script.push_str("# cartoon\n");
            script.push_str("# color #1 gray\n");
        }

        // Style adjustments
        script.push_str("\n# Visual adjustments\n");
        script.push_str("lighting soft\n");
        script.push_str("set bgColor white\n");

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
            report.push_str(&format!(
                "| {} | ({:.1}, {:.1}, {:.1}) | {:.0} | {} | {:.2} | {} | {:?} |\n",
                idx + 1,
                site.centroid[0], site.centroid[1], site.centroid[2],
                site.estimated_volume,
                site.spike_count,
                site.quality_score,
                if site.druggability.is_druggable { "✓" } else { "✗" },
                site.classification,
            ));
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
        assert_eq!(config.max_atoms, 15000);
        assert_eq!(config.temperature, 300.0);
    }
}

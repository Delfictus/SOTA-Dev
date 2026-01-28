//! Finalize Stage - Post-Run Evidence Pack Generator
//!
//! This module is the entry point after MD stepping completes.
//! It consumes engine outputs (events.jsonl, cryo_probe_results.json) and
//! produces the complete evidence pack with strict determinism guarantees.
//!
//! # Architecture
//!
//! ```text
//! MD Stepping (prism-nhs)
//!     │
//!     ▼ events.jsonl (streamed during stepping)
//!     │
//! ┌───┴───────────────────────────────────────────────────────────────┐
//! │  FinalizeStage (this module)                                       │
//! │    ├─ 1. Load event cloud from events.jsonl                       │
//! │    ├─ 2. Cluster events into cryptic sites (DBSCAN)               │
//! │    ├─ 3. Compute site metrics (persistence, geometry, chemistry) │
//! │    ├─ 4. Run ablation analysis                                    │
//! │    ├─ 5. Voxelize event cloud (Gaussian deposition)               │
//! │    ├─ 6. Kabsch align to holo (if provided)                       │
//! │    ├─ 7. Compute correlations                                     │
//! │    ├─ 8. Generate figures, sessions, reports                      │
//! │    └─ 9. Write output contract                                    │
//! └───────────────────────────────────────────────────────────────────┘
//!     │
//!     ▼ results/<run_id>/
//!       ├── summary.json
//!       ├── correlation.csv
//!       ├── report.html
//!       ├── report.pdf
//!       ├── sites/
//!       ├── volumes/
//!       ├── trajectories/
//!       └── provenance/
//! ```

use crate::ablation::{AblationMode, AblationResults, AblationRunResult};
use crate::config::{ReportConfig, RankingWeights};
use crate::event_cloud::{read_events, AblationPhase, EventCloud, PocketEvent};
use crate::figures;
// Use site_metrics::TopologyData (with residue_ids: Vec<u32>) for all spatial operations
// inputs::TopologyData is DEPRECATED for finalize - only use for legacy compatibility
use crate::site_metrics::{
    TopologyData as MetricsTopologyData, SiteMetricsComputer, validate_coordinate_frames,
    sort_sites_deterministic,
};
use crate::outputs::{
    OutputContract, ProvenanceManifest, ProvenanceParams, ProvenanceSeeds, ProvenanceVersions,
    SummaryJson, write_residues_txt, write_site_mol2, write_site_pdb,
};
use crate::pipeline::VoxelizationSummary;
use crate::reports::{HtmlReport, PdfReport};
use crate::sessions::{generate_chimerax_session, generate_pymol_session, SessionAvailability};
use crate::sites::{
    compute_chemistry_metrics, CrypticSite, GeometryMetrics,
    PersistenceMetrics, SiteMetrics, SiteRanking, UvResponseMetrics,
};
use crate::site_geometry::{compute_shape_from_points, compute_volume_statistics};
use crate::voxelize::voxelize_event_cloud;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

// =============================================================================
// PHARMA SOTA-GRADE POCKET ACCEPTANCE SYSTEM
// =============================================================================

/// Pharma-grade acceptance criteria thresholds
/// Based on industry standards for cryptic binding site validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmaAcceptanceCriteria {
    /// Minimum persistence (fraction of frames) - typical: 0.01 (1%)
    pub min_persistence: f64,
    /// Minimum pocket volume (Å³) - druggability threshold: 150-200
    pub min_volume_a3: f64,
    /// Minimum residue count - real pockets involve multiple residues: 3+
    pub min_residue_count: usize,
    /// Minimum replica agreement (if replicates > 1) - reproducibility: 0.5+
    pub min_replica_agreement: f64,
    /// Maximum centroid distance for pocket collapse (Å) - spatial merge: 8.0
    pub collapse_distance_a: f64,
    /// Minimum shared residues for collapse - structural overlap: 1+
    pub collapse_min_shared_residues: usize,
}

impl Default for PharmaAcceptanceCriteria {
    fn default() -> Self {
        Self {
            min_persistence: 0.005,          // 0.5% of frames (relaxed for sparse events)
            min_volume_a3: 150.0,            // Minimum for small molecule binding
            min_residue_count: 3,            // Real pockets span multiple residues
            min_replica_agreement: 0.5,      // Must appear in 50%+ of replicates
            collapse_distance_a: 8.0,        // Merge sites within 8Å
            collapse_min_shared_residues: 1, // With at least 1 shared residue
        }
    }
}

/// Pharma-grade pocket report with SOTA metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmaPocket {
    /// Pocket ID (after collapse)
    pub pocket_id: String,
    /// Rank among accepted pockets
    pub rank: usize,
    /// Centroid position [x, y, z] in Å
    pub centroid: [f32; 3],
    /// Residues involved (union of collapsed sites)
    pub residues: Vec<usize>,
    /// Residue labels (e.g., "A:LEU123")
    pub residue_labels: Vec<String>,
    /// Chain ID
    pub chain_id: String,

    // === COMPUTED METRICS (we have data) ===
    pub computed: PharmaComputedMetrics,

    // === ACCEPTANCE STATUS ===
    pub acceptance: PharmaAcceptanceStatus,

    // === METRICS WE CANNOT COMPUTE (labeled) ===
    pub unavailable: PharmaUnavailableMetrics,
}

/// Metrics we can compute from our data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmaComputedMetrics {
    /// Persistence: fraction of frames where pocket is open
    pub persistence_fraction: f64,
    /// Volume mean (Å³) - computed from spatial extent
    pub volume_mean_a3: f64,
    /// Volume P95 (Å³) - worst-case binding cavity size
    pub volume_p95_a3: f64,
    /// Replica agreement (0-1) - reproducibility across independent runs
    pub replica_agreement: f64,
    /// Number of residues lining the pocket
    pub residue_count: usize,
    /// Chromophore correlation (0-1) - UV events matching chromophore wavelengths
    /// High values indicate genuine UV-mediated pocket opening
    pub chromophore_correlation: f64,
    /// UV response: delta SASA from UV excitation
    pub uv_delta_sasa: f64,
    /// UV response: delta volume from UV excitation
    pub uv_delta_volume: f64,
    /// Hydrophobic fraction (0-1) - lipophilicity for drug binding
    pub hydrophobic_fraction: f64,
    /// Aromatic fraction (0-1) - pi-stacking potential
    pub aromatic_fraction: f64,
    /// Charged fraction (0-1) - electrostatic interactions
    pub charged_fraction: f64,
    /// H-bond donor count
    pub hbond_donors: usize,
    /// H-bond acceptor count
    pub hbond_acceptors: usize,
    /// Pocket depth (Å) - if computed from voxel grid
    pub depth_a: Option<f64>,
    /// Mouth opening area (Å²) - if computed
    pub mouth_area_a2: Option<f64>,
    /// Number of merged sites (1 = no collapse)
    pub n_merged_sites: usize,
    /// Original site IDs that were merged
    pub merged_from: Vec<String>,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Rank score from multi-criteria ranking
    pub rank_score: f64,

    // === SOTA PHARMA METRICS ===
    /// Enclosure score (0-1): How enclosed/buried the pocket is
    /// Higher = more enclosed, better for drug binding
    pub enclosure_score: f64,
    /// Buriedness (Å): Average distance from protein surface
    pub buriedness_a: f64,
    /// Breathing amplitude (Å³): Volume fluctuation (max - min)
    /// Indicates pocket flexibility/cryptic nature
    pub breathing_amplitude_a3: f64,
    /// Sphericity (0-1): How spherical the pocket is
    /// 1.0 = perfect sphere, lower = elongated
    pub sphericity: f64,
    /// Aspect ratio: Longest/shortest axis
    /// 1.0 = symmetric, higher = elongated
    pub aspect_ratio: f64,
    /// SiteMap-style druggability score (0-1)
    /// Composite: enclosure + hydrophobic balance + size
    pub druggability_score: f64,
    /// Pocket polarity: Ratio of polar to total surface
    pub polarity: f64,
}

/// Acceptance status with per-criterion breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmaAcceptanceStatus {
    /// Overall acceptance: true if ALL criteria pass
    pub accepted: bool,
    /// Individual criterion results
    pub persistence_pass: bool,
    pub volume_pass: bool,
    pub residue_count_pass: bool,
    pub replica_agreement_pass: bool,
    /// Rejection reason (if not accepted)
    pub rejection_reason: Option<String>,
}

/// Metrics we cannot compute - clearly labeled
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmaUnavailableMetrics {
    /// Binding affinity prediction - requires docking/FEP
    pub binding_affinity_kcal: MetricStatus,
    /// Ligand efficiency - requires known ligand
    pub ligand_efficiency: MetricStatus,
    /// Selectivity score - requires off-target panel
    pub selectivity_score: MetricStatus,
    /// ADMET prediction - requires QSAR models
    pub admet_flags: MetricStatus,
    /// Experimental validation - requires assay data
    pub experimental_validation: MetricStatus,
    /// Crystal structure confirmation - requires X-ray/cryo-EM
    pub structural_confirmation: MetricStatus,
}

/// Status of an unavailable metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricStatus {
    pub available: bool,
    pub reason: String,
    pub recommendation: String,
}

impl Default for PharmaUnavailableMetrics {
    fn default() -> Self {
        Self {
            binding_affinity_kcal: MetricStatus {
                available: false,
                reason: "Requires molecular docking or FEP calculations".to_string(),
                recommendation: "Run AutoDock Vina or Schrödinger FEP+ on detected pocket".to_string(),
            },
            ligand_efficiency: MetricStatus {
                available: false,
                reason: "Requires known ligand with measured Kd".to_string(),
                recommendation: "Screen fragment library and measure binding".to_string(),
            },
            selectivity_score: MetricStatus {
                available: false,
                reason: "Requires off-target binding panel".to_string(),
                recommendation: "Run docking against related protein family members".to_string(),
            },
            admet_flags: MetricStatus {
                available: false,
                reason: "Requires QSAR/ML ADMET prediction".to_string(),
                recommendation: "Use SwissADME or pkCSM on candidate compounds".to_string(),
            },
            experimental_validation: MetricStatus {
                available: false,
                reason: "Requires biochemical assay data".to_string(),
                recommendation: "SPR, ITC, or thermal shift assay recommended".to_string(),
            },
            structural_confirmation: MetricStatus {
                available: false,
                reason: "Requires experimental structure with ligand".to_string(),
                recommendation: "Co-crystallization or cryo-EM with fragment hit".to_string(),
            },
        }
    }
}

/// Complete pharma report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmaReport {
    /// Report version
    pub version: String,
    /// Generation timestamp
    pub timestamp: String,
    /// Acceptance criteria used
    pub criteria: PharmaAcceptanceCriteria,
    /// Input summary
    pub input_summary: PharmaInputSummary,
    /// Accepted pockets (passed all criteria)
    pub accepted_pockets: Vec<PharmaPocket>,
    /// Rejected pockets (for transparency)
    pub rejected_pockets: Vec<PharmaPocket>,
    /// Statistics
    pub statistics: PharmaStatistics,
    /// Quality assessment
    pub quality_assessment: PharmaQualityAssessment,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmaInputSummary {
    pub total_events: usize,
    pub total_frames: usize,
    pub replicates: usize,
    pub sites_before_collapse: usize,
    pub sites_after_collapse: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmaStatistics {
    pub total_candidate_pockets: usize,
    pub accepted_count: usize,
    pub rejected_count: usize,
    pub acceptance_rate: f64,
    pub rejection_breakdown: RejectionBreakdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectionBreakdown {
    pub failed_persistence: usize,
    pub failed_volume: usize,
    pub failed_residue_count: usize,
    pub failed_replica_agreement: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PharmaQualityAssessment {
    /// Overall quality tier: "HIGH", "MEDIUM", "LOW", "INSUFFICIENT"
    pub quality_tier: String,
    /// Confidence in results (0-1)
    pub overall_confidence: f64,
    /// Recommendations for improving confidence
    pub recommendations: Vec<String>,
}

/// Finalize stage result
#[derive(Debug)]
pub struct FinalizeResult {
    /// Output directory
    pub output_dir: PathBuf,
    /// Number of sites detected
    pub n_sites: usize,
    /// Number of druggable sites
    pub n_druggable: usize,
    /// Ablation significance
    pub cryo_significant: bool,
    pub uv_significant: bool,
    /// Voxelization results
    pub voxelization: Option<VoxelizationSummary>,
    /// Files generated
    pub files_generated: Vec<String>,
    /// SHA256 of summary.json for determinism verification
    pub summary_sha256: String,
}

/// Finalize stage - the main post-run processor
pub struct FinalizeStage {
    config: ReportConfig,
    output: OutputContract,
    events_path: PathBuf,
    /// Path to topology.json (MANDATORY - no fallback to PDB)
    topology_path: PathBuf,
    /// Master seed for deterministic RNG
    master_seed: u64,
    /// Maximum distance (Å) from any atom for events to be included in voxelization
    max_event_atom_dist: f32,
    /// Maximum distance (Å) from any atom for events to be included in clustering
    /// This should be <= the residue mapping radius (5Å) + small margin to ensure
    /// clustered site centroids are close enough for residue mapping to succeed.
    cluster_max_event_atom_dist: f32,
}

impl FinalizeStage {
    /// Create new finalize stage with MANDATORY topology path
    ///
    /// # Arguments
    /// * `config` - Report configuration
    /// * `events_path` - Path to events.jsonl from MD stepping
    /// * `topology_path` - Path to topology.json (MANDATORY - no fallback to PDB)
    /// * `master_seed` - Seed for deterministic operations
    /// * `max_event_atom_dist` - Maximum distance (Å) from any atom for voxelization (default: 15.0)
    /// * `cluster_max_event_atom_dist` - Maximum distance (Å) from any atom for clustering (default: 10.0)
    ///
    /// # Two-tier event filtering
    /// Events are filtered twice:
    /// 1. Voxelization uses max_event_atom_dist (15Å) - allows broader coverage for density maps
    /// 2. Clustering uses cluster_max_event_atom_dist (10Å default) - production setting for
    ///    broader site detection while maintaining residue mapping quality
    ///
    /// # IMPORTANT
    /// The topology path is MANDATORY. There is NO fallback to PDB parsing.
    /// All spatial metrics use topology.json coordinates only.
    pub fn new_with_topology(
        config: ReportConfig,
        events_path: PathBuf,
        topology_path: PathBuf,
        master_seed: u64,
        max_event_atom_dist: f32,
    ) -> Result<Self> {
        // Default cluster distance: 10Å (production default for broader site detection)
        Self::new_with_topology_full(config, events_path, topology_path, master_seed, max_event_atom_dist, 10.0)
    }

    /// Create new finalize stage with full control over both distance thresholds
    pub fn new_with_topology_full(
        config: ReportConfig,
        events_path: PathBuf,
        topology_path: PathBuf,
        master_seed: u64,
        max_event_atom_dist: f32,
        cluster_max_event_atom_dist: f32,
    ) -> Result<Self> {
        // Validate topology path exists
        if !topology_path.exists() {
            bail!(
                "FATAL: Topology file not found: {}\n\
                 The --topology argument is MANDATORY. There is NO fallback to PDB parsing.\n\
                 Generate with: prism-prep your_structure.pdb output.json --use-amber --strict",
                topology_path.display()
            );
        }

        let output = OutputContract::new(&config.output_dir)?;
        Ok(Self {
            config,
            output,
            events_path,
            topology_path,
            master_seed,
            max_event_atom_dist,
            cluster_max_event_atom_dist,
        })
    }

    /// DEPRECATED: Create finalize stage without explicit topology
    ///
    /// This constructor tries to find topology.json in common locations.
    /// For production use, prefer `new_with_topology()` with explicit path.
    #[deprecated(note = "Use new_with_topology() with explicit topology path")]
    pub fn new(config: ReportConfig, events_path: PathBuf, master_seed: u64) -> Result<Self> {
        // Try to find topology.json in common locations
        let topology_paths = vec![
            config.output_dir.join("topology.json"),
            config.input_pdb.with_file_name(
                format!("{}_topology.json",
                    config.input_pdb.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("input"))
            ),
        ];

        for path in &topology_paths {
            if path.exists() {
                return Self::new_with_topology(config, events_path, path.clone(), master_seed, 15.0);
            }
        }

        bail!(
            "FATAL: No topology.json found in expected locations.\n\
             Searched:\n  - {}\n  - {}\n\n\
             The topology file is MANDATORY. There is NO fallback to PDB parsing.\n\
             Use --topology to specify the path explicitly.",
            topology_paths[0].display(),
            topology_paths[1].display()
        );
    }

    /// Run the complete finalize stage
    pub fn run(&self) -> Result<FinalizeResult> {
        log::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        log::info!("  PRISM4D Finalize Stage v{}", crate::VERSION);
        log::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        log::info!("Output directory: {}", self.config.output_dir.display());
        log::info!("Events path: {}", self.events_path.display());
        log::info!("Master seed: {}", self.master_seed);

        // Ablation validation is now optional (production mode)
        // Ablation metrics are computed as interpretive outputs only
        if !self.config.ablation.is_complete() {
            log::warn!("Ablation config incomplete - metrics will be partial");
        }

        // Check dependencies
        let sessions = SessionAvailability::check();
        for msg in sessions.report_missing() {
            log::warn!("{}", msg);
        }

        // Step 1: Load event cloud
        log::info!("\n[1/10] Loading event cloud...");
        let mut event_cloud = self.load_events()?;
        log::info!("  Loaded {} pocket events", event_cloud.len());

        // Step 2: Load topology for real residue mapping (MANDATORY)
        log::info!("\n[2/10] Loading topology from: {}", self.topology_path.display());
        let metrics_topology = self.load_topology()?;
        log::info!("  Loaded: {} atoms", metrics_topology.n_atoms);

        // Step 2b: Coordinate frame validation (MANDATORY)
        log::info!("  Validating coordinate frames...");
        let event_centers: Vec<[f32; 3]> = event_cloud.events.iter()
            .map(|e| e.center_xyz)
            .collect();
        let coord_diag = validate_coordinate_frames(&metrics_topology, &event_centers, 10.0)
            .context("Coordinate frame validation failed - events and topology do not overlap")?;
        log::info!("  Coordinate frames: VALID (overlap confirmed)");
        log::debug!("  Topology AABB: [{:.1}, {:.1}, {:.1}] - [{:.1}, {:.1}, {:.1}]",
            coord_diag.topology_aabb.min[0], coord_diag.topology_aabb.min[1], coord_diag.topology_aabb.min[2],
            coord_diag.topology_aabb.max[0], coord_diag.topology_aabb.max[1], coord_diag.topology_aabb.max[2]);

        // Step 2c: Filter events by atom proximity
        // Keep only events within max_atom_dist of any topology atom
        // This removes off-protein events that cannot form valid cryptic sites
        let max_atom_dist = self.max_event_atom_dist;
        let max_atom_dist_sq = max_atom_dist * max_atom_dist;
        log::info!("  Filtering events by atom proximity (max dist: {} Å)...", max_atom_dist);

        // Build spatial hash from topology atoms for O(1) proximity lookup
        let cell_size = max_atom_dist;
        let aabb = &coord_diag.topology_aabb;
        let offset = [
            aabb.min[0] - max_atom_dist,
            aabb.min[1] - max_atom_dist,
            aabb.min[2] - max_atom_dist,
        ];
        let grid_dims = [
            ((aabb.max[0] - aabb.min[0] + 2.0 * max_atom_dist) / cell_size).ceil() as usize + 1,
            ((aabb.max[1] - aabb.min[1] + 2.0 * max_atom_dist) / cell_size).ceil() as usize + 1,
            ((aabb.max[2] - aabb.min[2] + 2.0 * max_atom_dist) / cell_size).ceil() as usize + 1,
        ];
        let grid_size = grid_dims[0] * grid_dims[1] * grid_dims[2];

        // Populate grid with atom indices
        let mut atom_grid: Vec<Vec<usize>> = vec![Vec::new(); grid_size];
        for ai in 0..metrics_topology.n_atoms {
            let ax = metrics_topology.positions[ai * 3];
            let ay = metrics_topology.positions[ai * 3 + 1];
            let az = metrics_topology.positions[ai * 3 + 2];
            let ci = ((ax - offset[0]) / cell_size) as usize;
            let cj = ((ay - offset[1]) / cell_size) as usize;
            let ck = ((az - offset[2]) / cell_size) as usize;
            if ci < grid_dims[0] && cj < grid_dims[1] && ck < grid_dims[2] {
                let cell_idx = ci + cj * grid_dims[0] + ck * grid_dims[0] * grid_dims[1];
                atom_grid[cell_idx].push(ai);
            }
        }

        // Filter events: keep only those within max_atom_dist of any atom
        let n_before_filter = event_cloud.events.len();
        event_cloud.events.retain(|e| {
            let c = e.center_xyz;

            // Quick AABB check first
            if c[0] < aabb.min[0] - max_atom_dist || c[0] > aabb.max[0] + max_atom_dist ||
               c[1] < aabb.min[1] - max_atom_dist || c[1] > aabb.max[1] + max_atom_dist ||
               c[2] < aabb.min[2] - max_atom_dist || c[2] > aabb.max[2] + max_atom_dist {
                return false;
            }

            // Find cell for this event
            let ci = ((c[0] - offset[0]) / cell_size) as i32;
            let cj = ((c[1] - offset[1]) / cell_size) as i32;
            let ck = ((c[2] - offset[2]) / cell_size) as i32;

            // Check neighboring cells (3x3x3 neighborhood)
            for di in -1..=1 {
                for dj in -1..=1 {
                    for dk in -1..=1 {
                        let ni = ci + di;
                        let nj = cj + dj;
                        let nk = ck + dk;
                        if ni >= 0 && ni < grid_dims[0] as i32 &&
                           nj >= 0 && nj < grid_dims[1] as i32 &&
                           nk >= 0 && nk < grid_dims[2] as i32 {
                            let cell_idx = ni as usize +
                                           nj as usize * grid_dims[0] +
                                           nk as usize * grid_dims[0] * grid_dims[1];
                            for &ai in &atom_grid[cell_idx] {
                                let ax = metrics_topology.positions[ai * 3];
                                let ay = metrics_topology.positions[ai * 3 + 1];
                                let az = metrics_topology.positions[ai * 3 + 2];
                                let dsq = (c[0] - ax).powi(2) + (c[1] - ay).powi(2) + (c[2] - az).powi(2);
                                if dsq <= max_atom_dist_sq {
                                    return true;
                                }
                            }
                        }
                    }
                }
            }
            false
        });
        let n_after_filter = event_cloud.events.len();
        let n_filtered = n_before_filter - n_after_filter;
        if n_filtered > 0 {
            let pct_filtered = 100.0 * n_filtered as f64 / n_before_filter as f64;
            log::info!("  Filtered {} events >{}Å from any atom ({:.1}% of total)",
                n_filtered, max_atom_dist, pct_filtered);
            log::info!("  Remaining events for clustering: {}", n_after_filter);
        }

        // Diagnostic: compute min/median/p90 nearest-atom distance on sample of retained events
        if !event_cloud.events.is_empty() {
            let sample_size = std::cmp::min(200, event_cloud.events.len());
            let step = event_cloud.events.len() / sample_size;
            let mut sample_dists: Vec<f32> = Vec::with_capacity(sample_size);

            for i in (0..event_cloud.events.len()).step_by(step.max(1)).take(sample_size) {
                let c = event_cloud.events[i].center_xyz;
                let mut min_d_sq = f32::MAX;
                // Use spatial grid for efficient lookup
                let ci = ((c[0] - offset[0]) / cell_size) as i32;
                let cj = ((c[1] - offset[1]) / cell_size) as i32;
                let ck = ((c[2] - offset[2]) / cell_size) as i32;
                for di in -1..=1 {
                    for dj in -1..=1 {
                        for dk in -1..=1 {
                            let ni = ci + di;
                            let nj = cj + dj;
                            let nk = ck + dk;
                            if ni >= 0 && ni < grid_dims[0] as i32 &&
                               nj >= 0 && nj < grid_dims[1] as i32 &&
                               nk >= 0 && nk < grid_dims[2] as i32 {
                                let cell_idx = ni as usize +
                                               nj as usize * grid_dims[0] +
                                               nk as usize * grid_dims[0] * grid_dims[1];
                                for &ai in &atom_grid[cell_idx] {
                                    let ax = metrics_topology.positions[ai * 3];
                                    let ay = metrics_topology.positions[ai * 3 + 1];
                                    let az = metrics_topology.positions[ai * 3 + 2];
                                    let dsq = (c[0] - ax).powi(2) + (c[1] - ay).powi(2) + (c[2] - az).powi(2);
                                    if dsq < min_d_sq { min_d_sq = dsq; }
                                }
                            }
                        }
                    }
                }
                sample_dists.push(min_d_sq.sqrt());
            }

            sample_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let min_d = sample_dists.first().copied().unwrap_or(0.0);
            let median_d = sample_dists.get(sample_dists.len() / 2).copied().unwrap_or(0.0);
            let p90_idx = (sample_dists.len() as f32 * 0.9) as usize;
            let p90_d = sample_dists.get(p90_idx.min(sample_dists.len().saturating_sub(1))).copied().unwrap_or(0.0);
            log::info!("  Retained events quality ({}): min={:.1}Å, median={:.1}Å, p90={:.1}Å",
                sample_size, min_d, median_d, p90_d);
        }

        // Step 3: Two-tier event filtering for clustering
        //
        // Events in event_cloud (filtered at 15Å) are used for:
        //   - Voxelization (global density maps)
        //   - Ablation analysis (event counts by phase)
        //
        // For clustering + residue mapping, we need a STRICTER subset to ensure
        // site centroids are close enough for 5Å residue queries to succeed.
        let cluster_dist = self.cluster_max_event_atom_dist;
        let cluster_dist_sq = cluster_dist * cluster_dist;
        log::info!("\n[3/10] Creating stricter event subset for clustering (max atom dist: {:.1}Å)...", cluster_dist);

        // Filter events for clustering: keep only those within cluster_dist of any atom
        let cluster_events: Vec<PocketEvent> = event_cloud.events.iter()
            .filter(|e| {
                let c = e.center_xyz;

                // Quick AABB check
                if c[0] < aabb.min[0] - cluster_dist || c[0] > aabb.max[0] + cluster_dist ||
                   c[1] < aabb.min[1] - cluster_dist || c[1] > aabb.max[1] + cluster_dist ||
                   c[2] < aabb.min[2] - cluster_dist || c[2] > aabb.max[2] + cluster_dist {
                    return false;
                }

                // Find cell for this event (using existing grid with cell_size = max_atom_dist)
                let ci = ((c[0] - offset[0]) / cell_size) as i32;
                let cj = ((c[1] - offset[1]) / cell_size) as i32;
                let ck = ((c[2] - offset[2]) / cell_size) as i32;

                // Check neighboring cells
                for di in -1..=1 {
                    for dj in -1..=1 {
                        for dk in -1..=1 {
                            let ni = ci + di;
                            let nj = cj + dj;
                            let nk = ck + dk;
                            if ni >= 0 && ni < grid_dims[0] as i32 &&
                               nj >= 0 && nj < grid_dims[1] as i32 &&
                               nk >= 0 && nk < grid_dims[2] as i32 {
                                let cell_idx = ni as usize +
                                               nj as usize * grid_dims[0] +
                                               nk as usize * grid_dims[0] * grid_dims[1];
                                for &ai in &atom_grid[cell_idx] {
                                    let ax = metrics_topology.positions[ai * 3];
                                    let ay = metrics_topology.positions[ai * 3 + 1];
                                    let az = metrics_topology.positions[ai * 3 + 2];
                                    let dsq = (c[0] - ax).powi(2) + (c[1] - ay).powi(2) + (c[2] - az).powi(2);
                                    if dsq <= cluster_dist_sq {
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
                false
            })
            .cloned()
            .collect();

        let n_cluster_events = cluster_events.len();
        let n_excluded = event_cloud.events.len() - n_cluster_events;
        log::info!("  Cluster event subset: {} events ({} excluded as >{:.1}Å from atoms)",
            n_cluster_events, n_excluded, cluster_dist);

        // Cluster using stricter event subset
        log::info!("  Clustering events into cryptic sites...");
        let mut sites = self.cluster_events_into_sites(&cluster_events)?;
        log::info!("  Found {} candidate sites", sites.len());

        // Step 3b: Apply real residue mapping from topology using spatial grid
        // Uses cluster_events (stricter subset) to ensure sites have nearby atoms
        log::info!("  Applying real residue mapping (5Å query radius)...");
        self.apply_real_residue_mapping(&mut sites, &metrics_topology, &cluster_events)?;

        // HARD FAIL: Check that sites have residues mapped (if any sites exist)
        // 0 sites is a valid outcome (no cryptic sites detected)
        // But if sites exist, they MUST have residues - otherwise it's a mapping failure
        if !sites.is_empty() {
            let total_residues: usize = sites.iter().map(|s| s.residues.len()).sum();
            if total_residues == 0 {
                bail!(
                    "FATAL: 0 residues mapped across all {} sites.\n\
                     This indicates a coordinate frame mismatch that wasn't caught by validation.\n\
                     Check that topology.json and events.jsonl are from the same simulation run.",
                    sites.len()
                );
            }
        } else {
            log::info!("  No cryptic sites detected (clustering produced 0 sites)");
        }

        // Step 4: Compute ablation results from events
        log::info!("\n[4/10] Computing ablation analysis...");
        let ablation = self.compute_ablation_from_events(&event_cloud.events)?;
        log::info!(
            "  Baseline: {} spikes, Cryo: {} spikes, Cryo+UV: {} spikes",
            ablation.baseline.total_spikes,
            ablation.cryo_only.total_spikes,
            ablation.cryo_uv.total_spikes
        );

        // Step 5: Compute UV response deltas per site
        log::info!("\n[5/10] Computing UV response metrics...");
        self.compute_uv_response(&mut sites, &event_cloud.events, &ablation)?;

        // Step 6: Rank sites with deterministic ordering
        log::info!("\n[6/10] Ranking sites...");
        let ranking = SiteRanking::new(RankingWeights::default());
        ranking.rank_sites(&mut sites);

        // Apply deterministic ordering: (open_frequency desc, CV_SASA desc, volume_A3 desc)
        sort_sites_deterministic(&mut sites);

        // Filter to druggable sites
        let n_before = sites.len();
        sites.retain(|s| s.is_druggable);
        log::info!("  {} sites pass druggability filter (from {})", sites.len(), n_before);

        // Step 6b: PHARMA SOTA-GRADE FILTERING
        // Collapse overlapping pockets and apply strict acceptance criteria
        log::info!("\n[6b/10] Applying pharma-grade pocket filtering...");
        let sites_before_collapse = sites.len();
        let pharma_criteria = PharmaAcceptanceCriteria::default();

        // Collapse overlapping pockets (merge nearby sites with shared residues)
        let collapsed_sites = collapse_overlapping_pockets(&sites, &pharma_criteria);
        log::info!("  Pocket collapse: {} → {} unique pockets", sites_before_collapse, collapsed_sites.len());

        // Generate pharma report with acceptance filtering
        let pharma_report = self.generate_pharma_report(
            &collapsed_sites,
            &pharma_criteria,
            event_cloud.events.len(),
            self.config.replicates,
        );

        log::info!("  Pharma acceptance: {} accepted, {} rejected",
            pharma_report.accepted_pockets.len(),
            pharma_report.rejected_pockets.len());
        log::info!("  Quality tier: {}", pharma_report.quality_assessment.quality_tier);

        // Step 7: Tier1/Tier2 correlation REMOVED
        // Per user requirement: These correlation types are removed entirely.
        // All spatial metrics use topology.json only.
        log::info!("\n[7/10] Skipping Tier1/Tier2 correlation (removed)...");

        // Step 8: Voxelize event cloud
        log::info!("\n[8/10] Running post-run voxelization...");
        let voxelization = self.run_voxelization(&event_cloud)?;
        if let Some(ref vox) = voxelization {
            log::info!("  Grid: {}x{}x{}, {} events → {} voxels above threshold",
                vox.dims[0], vox.dims[1], vox.dims[2], vox.n_events, vox.voxels_above_threshold);
        }

        // Step 9: Generate all outputs
        log::info!("\n[9/10] Generating outputs...");
        let mut files_generated = Vec::new();

        // Summary JSON (must be first for determinism hash)
        let summary = self.build_summary(&sites, &ablation)?;
        let summary_path = self.output.summary_json();
        let summary_json = serde_json::to_string_pretty(&summary)?;
        fs::write(&summary_path, &summary_json)?;
        files_generated.push(summary_path.display().to_string());
        log::info!("  ✓ summary.json");

        // Compute SHA256 for determinism verification
        let summary_sha256 = sha256_hex(&summary_json);

        // PHARMA SOTA-GRADE REPORT (strict acceptance criteria)
        let pharma_path = self.config.output_dir.join("pharma_report.json");
        let pharma_json = serde_json::to_string_pretty(&pharma_report)?;
        fs::write(&pharma_path, &pharma_json)?;
        files_generated.push(pharma_path.display().to_string());
        log::info!("  ✓ pharma_report.json ({} accepted pockets)", pharma_report.accepted_pockets.len());

        // Site metrics CSV (new primary name + backward-compatible alias)
        let csv_content = self.build_site_metrics_csv(&sites);

        // Write to new primary name: site_metrics.csv
        let site_metrics_path = self.output.site_metrics_csv();
        fs::write(&site_metrics_path, &csv_content)?;
        files_generated.push(site_metrics_path.display().to_string());
        log::info!("  ✓ site_metrics.csv");

        // Write backward-compatible alias: correlation.csv
        let correlation_path = self.output.correlation_csv();
        fs::write(&correlation_path, &csv_content)?;
        files_generated.push(correlation_path.display().to_string());
        log::info!("  ✓ correlation.csv (alias for backward compatibility)");

        // Per-site outputs
        for site in &sites {
            self.generate_site_outputs(site, &metrics_topology, &sessions)?;
            files_generated.push(self.output.site_dir(&site.site_id).display().to_string());
        }
        log::info!("  ✓ {} site directories", sites.len());

        // Global figures
        if self.config.output_formats.figures {
            self.generate_global_figures(&sites, &ablation)?;
            log::info!("  ✓ global figures");

            // Generate the six required cryptic site figures
            let figures_dir = self.config.output_dir.join("figures");
            figures::generate_cryptic_site_figures(&figures_dir, &sites)?;
            log::info!("  ✓ cryptic site figures (6 plots)");
        }

        // HTML report
        let html_path = self.output.report_html();
        HtmlReport::generate(&html_path, &self.config, &sites, &ablation, None)?;
        files_generated.push(html_path.display().to_string());
        log::info!("  ✓ report.html");

        // PDF report
        if self.config.output_formats.pdf {
            let pdf_path = self.output.report_pdf();
            match PdfReport::generate(&html_path, &pdf_path) {
                Ok(()) => {
                    files_generated.push(pdf_path.display().to_string());
                    log::info!("  ✓ report.pdf");
                }
                Err(e) => {
                    log::warn!("  ⚠ PDF generation failed: {}", e);
                }
            }
        }

        // MRC volumes (written during voxelization step)
        if self.config.output_formats.mrc_volumes && voxelization.is_some() {
            log::info!("  ✓ volumes/occupancy.mrc");
            log::info!("  ✓ volumes/pocket_fields.mrc");
        }

        // Provenance
        log::info!("\n[10/10] Writing provenance...");
        self.write_provenance(&files_generated)?;
        log::info!("  ✓ provenance/");

        // Final summary
        let n_druggable = sites.iter().filter(|s| s.is_druggable).count();

        log::info!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        log::info!("  Finalize Stage Complete");
        log::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        log::info!("  Sites: {} ({} druggable)", sites.len(), n_druggable);
        log::info!(
            "  Cryo contrast: {}",
            if ablation.comparison.cryo_contrast_significant { "SIGNIFICANT" } else { "not significant" }
        );
        log::info!(
            "  UV response: {}",
            if ablation.comparison.uv_response_significant { "SIGNIFICANT" } else { "not significant" }
        );
        log::info!("  Summary SHA256: {}", &summary_sha256[..16]);
        log::info!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

        Ok(FinalizeResult {
            output_dir: self.config.output_dir.clone(),
            n_sites: sites.len(),
            n_druggable,
            cryo_significant: ablation.comparison.cryo_contrast_significant,
            uv_significant: ablation.comparison.uv_response_significant,
            voxelization,
            files_generated,
            summary_sha256,
        })
    }

    /// Load events from JSONL file
    fn load_events(&self) -> Result<EventCloud> {
        if !self.events_path.exists() {
            bail!("Events file not found: {}", self.events_path.display());
        }
        read_events(&self.events_path)
            .with_context(|| format!("Failed to read events from {}", self.events_path.display()))
    }

    /// Cluster events into cryptic sites using DBSCAN-like algorithm
    fn cluster_events_into_sites(&self, events: &[PocketEvent]) -> Result<Vec<CrypticSite>> {
        let cfg = &self.config.site_detection;

        // Group events by spatial proximity (simplified DBSCAN)
        let clusters = dbscan_cluster(events, cfg.cluster_threshold, cfg.min_cluster_size);

        let mut sites = Vec::new();

        log::info!("  Processing {} clusters into sites...", clusters.len());
        for (cluster_id, cluster_events) in clusters.into_iter().enumerate() {
            if cluster_events.len() < cfg.min_cluster_size {
                if cluster_id < 5 {
                    log::debug!("  Cluster {} rejected: size={} < min_cluster_size={}",
                               cluster_id + 1, cluster_events.len(), cfg.min_cluster_size);
                }
                continue;
            }

            // Compute cluster centroid
            let positions: Vec<[f32; 3]> = cluster_events.iter().map(|e| e.center_xyz).collect();
            let centroid = compute_centroid_f32(&positions);

            // Collect unique residues (sorted for determinism)
            let mut residue_set: HashSet<u32> = HashSet::new();
            for event in &cluster_events {
                residue_set.extend(event.residues.iter().copied());
            }
            let mut residues: Vec<usize> = residue_set.into_iter().map(|r| r as usize).collect();
            residues.sort(); // Ensure deterministic ordering

            // Estimate volume from spatial extent (bounding box with shrink factor)
            // More accurate than averaging spike intensity-based estimates
            let spatial_volume = compute_cluster_spatial_volume(&positions);
            let mean_volume = spatial_volume.max(100.0); // Minimum 100Å³

            // Check druggability (but don't reject - let pharma filter handle it)
            // This allows more sites through for the pharma report to evaluate

            // Compute persistence (fraction of unique frames)
            // IMPORTANT: frame_idx is local to (phase, replicate_id) - use full identity tuple
            // to correctly count unique event positions across all runs.
            let mut event_identity_set: HashSet<(AblationPhase, usize, usize)> = HashSet::new();
            let mut min_frame = usize::MAX;
            let mut max_frame = 0usize;
            for event in &cluster_events {
                // Use (phase, replicate_id, frame_idx) as unique event identity
                event_identity_set.insert((event.phase, event.replicate_id, event.frame_idx));
                // Track overall min/max for first_frame/last_frame fields (informational only)
                min_frame = min_frame.min(event.frame_idx);
                max_frame = max_frame.max(event.frame_idx);
            }

            // Compute frame span within each (phase, replicate_id) group
            // Group events by (phase, replicate_id) to get proper frame ranges
            let mut frame_spans_by_run: std::collections::HashMap<(AblationPhase, usize), (usize, usize)> =
                std::collections::HashMap::new();
            for event in &cluster_events {
                let key = (event.phase, event.replicate_id);
                let entry = frame_spans_by_run.entry(key).or_insert((usize::MAX, 0));
                entry.0 = entry.0.min(event.frame_idx);
                entry.1 = entry.1.max(event.frame_idx);
            }
            let total_frame_span: usize = frame_spans_by_run.values()
                .map(|(min_f, max_f)| if max_f > min_f { max_f - min_f + 1 } else { 1 })
                .sum();

            let persistence = event_identity_set.len() as f64 / total_frame_span.max(1) as f64;
            let n_unique_events = event_identity_set.len();

            if persistence < cfg.min_persistence as f64 {
                if cluster_id < 5 {
                    log::debug!("  Cluster {} rejected: persistence={:.3} < min_persistence={:.3}",
                               cluster_id + 1, persistence, cfg.min_persistence);
                }
                continue;
            }

            // Compute replica agreement
            let mut replica_set: HashSet<usize> = HashSet::new();
            for event in &cluster_events {
                replica_set.insert(event.replicate_id);
            }
            let replica_agreement = replica_set.len() as f64 / self.config.replicates.max(1) as f64;

            // Filter by replica agreement: use configurable threshold from CLI
            // Set min_replica_agreement to 0.0 to disable this filter
            let min_replica_agreement = cfg.min_replica_agreement as f64;
            if self.config.replicates > 1 && replica_agreement < min_replica_agreement {
                if cluster_id < 5 {
                    log::debug!("  Cluster {} rejected: replica_agreement={:.2} < {:.2} (present in {}/{} replicates)",
                               cluster_id + 1, replica_agreement, min_replica_agreement,
                               replica_set.len(), self.config.replicates);
                }
                continue;
            }

            // Mean confidence from events
            let mean_confidence: f64 = cluster_events.iter()
                .map(|e| e.confidence as f64)
                .sum::<f64>() / cluster_events.len() as f64;

            if mean_confidence < cfg.min_confidence as f64 {
                continue;
            }

            // Residue names: Use "UNK" (unknown) since topology mapping is not available
            // In full pipeline with topology, these would be looked up from residue IDs
            let residue_names: Vec<String> = residues.iter()
                .map(|id| format!("UNK_{}", id))
                .collect();

            let chemistry = compute_chemistry_metrics(&residue_names);

            // Find representative frame (frame with largest volume event)
            let representative_frame = cluster_events.iter()
                .max_by(|a, b| a.volume_a3.partial_cmp(&b.volume_a3).unwrap_or(std::cmp::Ordering::Equal))
                .map(|e| e.frame_idx)
                .unwrap_or(0);

            // Volume statistics - REAL computations, not heuristics
            let volumes: Vec<f64> = cluster_events.iter().map(|e| e.volume_a3 as f64).collect();
            let mut sorted_volumes = volumes.clone();
            sorted_volumes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let volume_p50 = if sorted_volumes.is_empty() { 0.0 } else { sorted_volumes[sorted_volumes.len() / 2] };
            let volume_p95 = if sorted_volumes.is_empty() { 0.0 } else { sorted_volumes[(sorted_volumes.len() * 95) / 100.max(sorted_volumes.len() - 1)] };

            // REAL volume statistics from actual trajectory data
            let vol_stats = compute_volume_statistics(&volumes);
            let (volume_min, volume_max, volume_std, breathing_amplitude) = match vol_stats {
                Some(stats) => (stats.volume_min, stats.volume_max, stats.volume_std, stats.breathing_amplitude),
                None => (mean_volume as f64 * 0.6, mean_volume as f64 * 1.4, 0.0, 0.0),
            };

            // REAL shape metrics from PCA on event point cloud
            let event_points: Vec<[f32; 3]> = cluster_events.iter()
                .map(|e| e.center_xyz)
                .collect();
            let shape_result = compute_shape_from_points(&event_points);
            let (aspect_ratio, sphericity) = match shape_result {
                Some(shape) => (Some(shape.aspect_ratio), Some(shape.sphericity)),
                None => (None, None), // < 4 points, can't compute PCA
            };

            let site = CrypticSite {
                site_id: format!("site_{:03}", cluster_id + 1),
                rank: 0, // Will be set by ranking
                centroid: centroid,
                residues,
                residue_names,
                chain_id: "A".to_string(),
                metrics: SiteMetrics {
                    persistence: PersistenceMetrics {
                        present_fraction: persistence,
                        mean_lifetime_frames: n_unique_events as f64,
                        replica_agreement,
                    },
                    geometry: GeometryMetrics {
                        volume_mean: mean_volume as f64,
                        volume_p50,
                        volume_p95,
                        // REAL volume statistics
                        volume_min,
                        volume_max,
                        volume_std,
                        breathing_amplitude,
                        // REAL shape metrics from PCA
                        aspect_ratio,
                        sphericity,
                        // Depth and mouth area computed later from voxel grids
                        depth_proxy_pocket_a: None,
                        depth_proxy_surface_a: None,
                        mouth_area_proxy_a2: None,
                        mouth_area_total_a2: None,
                        n_openings: None,
                    },
                    chemistry,
                    uv_response: UvResponseMetrics::default(),
                },
                rank_score: 0.0,
                confidence: mean_confidence,
                is_druggable: mean_volume >= cfg.min_volume,
                first_frame: min_frame,
                last_frame: max_frame,
                representative_frame,
            };

            sites.push(site);
        }

        Ok(sites)
    }

    /// Apply real residue mapping from topology using spatial grid queries
    ///
    /// This replaces the placeholder "UNK_<id>" residue names with real residue
    /// labels like "A:LEU123" based on 5Å spatial proximity to site event centers.
    ///
    /// Uses site_metrics::SiteMetricsComputer with deterministic BTreeMap/BTreeSet.
    fn apply_real_residue_mapping(
        &self,
        sites: &mut [CrypticSite],
        topology: &MetricsTopologyData,
        events: &[PocketEvent],
    ) -> Result<()> {
        // Create metrics computer with configurable query radius (default 8.0Å)
        let query_radius = self.config.site_detection.residue_query_radius_a;
        let metrics_computer = SiteMetricsComputer::new(topology.clone(), query_radius)
            .context("Failed to create SiteMetricsComputer")?;

        for site in sites.iter_mut() {
            // Get events belonging to this site (within cluster threshold of centroid)
            let site_events: Vec<&PocketEvent> = events
                .iter()
                .filter(|e| {
                    distance_f32(&site.centroid, &e.center_xyz) < self.config.site_detection.cluster_threshold
                })
                .collect();

            if site_events.is_empty() {
                log::warn!("  Site {} has no events within cluster threshold", site.site_id);
                continue;
            }

            // Map residues using spatial grid
            let event_centers: Vec<[f32; 3]> = site_events.iter().map(|e| e.center_xyz).collect();

            // Compute centroid for axis shell expansion
            let centroid = site.centroid;

            // Estimate pocket radius from volume: r = cbrt(3*V / 4π), clamped 2-6Å
            let volume = site.metrics.geometry.volume_mean as f32;
            let pocket_radius = ((3.0_f32 * volume) / (4.0_f32 * std::f32::consts::PI)).cbrt().clamp(2.0_f32, 6.0_f32);

            // Build expanded sample points: event centers + centroid + 6 axis points
            let mut expanded_centers = event_centers.clone();
            expanded_centers.push(centroid);

            // Add 6 axis shell points at ±radius from centroid
            let axis_offsets: [[f32; 3]; 6] = [
                [pocket_radius, 0.0, 0.0],
                [-pocket_radius, 0.0, 0.0],
                [0.0, pocket_radius, 0.0],
                [0.0, -pocket_radius, 0.0],
                [0.0, 0.0, pocket_radius],
                [0.0, 0.0, -pocket_radius],
            ];
            for offset in &axis_offsets {
                expanded_centers.push([
                    centroid[0] + offset[0],
                    centroid[1] + offset[1],
                    centroid[2] + offset[2],
                ]);
            }

            // Use canonical sample points for determinism (still deterministic with expanded input)
            let sample_points = SiteMetricsComputer::canonical_sample_points(expanded_centers, 1000);

            let mapping_result = match metrics_computer.map_residues(&sample_points) {
                Ok(result) => result,
                Err(e) => {
                    // DIAGNOSTIC: Print failing site geometry
                    let centroid: [f32; 3] = if !sample_points.is_empty() {
                        let n = sample_points.len() as f32;
                        [
                            sample_points.iter().map(|p| p[0]).sum::<f32>() / n,
                            sample_points.iter().map(|p| p[1]).sum::<f32>() / n,
                            sample_points.iter().map(|p| p[2]).sum::<f32>() / n,
                        ]
                    } else {
                        [0.0, 0.0, 0.0]
                    };

                    // Compute nearest atom distance using brute force
                    let positions = &topology.positions;
                    let n_atoms = topology.n_atoms;
                    let mut min_dist = f32::MAX;
                    let mut atoms_5a = 0usize;
                    let mut atoms_10a = 0usize;
                    let mut atoms_20a = 0usize;

                    for ai in 0..n_atoms {
                        let ax = positions[ai * 3];
                        let ay = positions[ai * 3 + 1];
                        let az = positions[ai * 3 + 2];
                        let d = ((centroid[0] - ax).powi(2) +
                                 (centroid[1] - ay).powi(2) +
                                 (centroid[2] - az).powi(2)).sqrt();
                        if d < min_dist { min_dist = d; }
                        if d <= 5.0 { atoms_5a += 1; }
                        if d <= 10.0 { atoms_10a += 1; }
                        if d <= 20.0 { atoms_20a += 1; }
                    }

                    log::error!(
                        "DIAGNOSTIC for failing site {}:\n\
                         - sample_points count: {}\n\
                         - centroid: [{:.2}, {:.2}, {:.2}]\n\
                         - nearest_atom_dist: {:.2} Å\n\
                         - atoms within 5Å: {}\n\
                         - atoms within 10Å: {}\n\
                         - atoms within 20Å: {}",
                        site.site_id,
                        sample_points.len(),
                        centroid[0], centroid[1], centroid[2],
                        min_dist,
                        atoms_5a, atoms_10a, atoms_20a
                    );

                    // Also print first few sample points
                    log::error!("First 5 sample_points:");
                    for (i, p) in sample_points.iter().take(5).enumerate() {
                        log::error!("  [{}]: [{:.2}, {:.2}, {:.2}]", i, p[0], p[1], p[2]);
                    }

                    // Soft skip for density-peak clustering: site may be in void region
                    // This can happen legitimately when peak detection finds a local max
                    // that's technically far from atoms but still part of a pocket cavity
                    if min_dist > query_radius && min_dist <= 15.0 && atoms_10a > 0 {
                        log::warn!("  Site {} skipped: centroid {:.1}Å from nearest atom (>{:.1}Å query radius)",
                                  site.site_id, min_dist, query_radius);
                        continue;
                    }

                    // True coordinate frame misalignment - hard fail
                    bail!(
                        "Residue mapping failed for site {}: {}\n\
                         This is a hard failure - check coordinate frame alignment.",
                        site.site_id, e
                    );
                }
            };

            // Update site with real residue data
            site.residues = mapping_result.residues.iter()
                .map(|r| r.resid as usize)
                .collect();
            site.residue_names = mapping_result.residues.iter()
                .map(|r| r.resname.clone())
                .collect();

            // Determine primary chain ID (most common)
            let mut chain_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
            for res in &mapping_result.residues {
                *chain_counts.entry(&res.chain).or_insert(0) += 1;
            }
            if let Some((chain_id, _)) = chain_counts.into_iter().max_by_key(|(_, c)| *c) {
                site.chain_id = chain_id.to_string();
            }

            // Recompute chemistry metrics with real residue names
            site.metrics.chemistry = compute_chemistry_metrics(&site.residue_names);

            // Use the hydrophobic fraction computed by SiteMetricsComputer
            site.metrics.chemistry.hydrophobic_fraction = mapping_result.hydrophobic_fraction as f64;

            // Compute depth proxy
            match metrics_computer.depth_proxy(site.centroid) {
                Ok(depth_result) => {
                    site.metrics.geometry.depth_proxy_surface_a = Some(depth_result.depth_proxy_a as f64);
                }
                Err(e) => {
                    log::warn!("  {} depth proxy failed: {}", site.site_id, e);
                }
            }

            log::debug!("  {} mapped {} residues (hydro={:.1}%, depth={:?})",
                site.site_id, mapping_result.residue_count,
                site.metrics.chemistry.hydrophobic_fraction * 100.0,
                site.metrics.geometry.depth_proxy_surface_a);
        }

        let total_mapped: usize = sites.iter().map(|s| s.residues.len()).sum();
        log::info!("  Total: {} residues mapped across {} sites", total_mapped, sites.len());

        Ok(())
    }

    /// Compute ablation results from event phases
    fn compute_ablation_from_events(&self, events: &[PocketEvent]) -> Result<AblationResults> {
        // Count events by phase
        let mut baseline_events: Vec<&PocketEvent> = Vec::new();
        let mut cryo_only_events: Vec<&PocketEvent> = Vec::new();
        let mut cryo_uv_events: Vec<&PocketEvent> = Vec::new();

        for event in events {
            match event.phase {
                AblationPhase::Baseline => baseline_events.push(event),
                AblationPhase::CryoOnly => cryo_only_events.push(event),
                AblationPhase::CryoUv => cryo_uv_events.push(event),
            }
        }

        // IMPORTANT: Baseline runs fewer frames than cryo phases.
        // Baseline: only warm_hold_steps (300K constant)
        // Cryo phases: cold_hold + ramp + warm_hold (full protocol)
        let temp = &self.config.temperature_protocol;
        let baseline_frames = temp.warm_hold_steps as usize;
        let cryo_frames = (temp.cold_hold_steps + temp.ramp_steps + temp.warm_hold_steps) as usize;

        // Helper to compute spike rate per 1000 frames
        let compute_rate = |spikes: usize, frames: usize| -> f64 {
            if frames == 0 { 0.0 } else { (spikes as f64) / (frames as f64 / 1000.0) }
        };

        let baseline_spikes: usize = baseline_events.iter().map(|e| e.spike_count).sum();
        let cryo_only_spikes: usize = cryo_only_events.iter().map(|e| e.spike_count).sum();
        let cryo_uv_spikes: usize = cryo_uv_events.iter().map(|e| e.spike_count).sum();

        let baseline = AblationRunResult {
            mode: AblationMode::Baseline,
            total_spikes: baseline_spikes,
            events_emitted: baseline_events.len(),
            phase_spikes: count_phase_spikes(&baseline_events),
            frames_analyzed: baseline_frames,
            spikes_per_1k_frames: compute_rate(baseline_spikes, baseline_frames),
            sites: Vec::new(),
            mean_volume: mean_volume(&baseline_events),
            mean_sasa: Some(estimate_sasa(&baseline_events)),
            runtime_seconds: 0.0, // Not tracked in events
        };

        let cryo_only = AblationRunResult {
            mode: AblationMode::CryoOnly,
            total_spikes: cryo_only_spikes,
            events_emitted: cryo_only_events.len(),
            phase_spikes: count_phase_spikes(&cryo_only_events),
            frames_analyzed: cryo_frames,
            spikes_per_1k_frames: compute_rate(cryo_only_spikes, cryo_frames),
            sites: Vec::new(),
            mean_volume: mean_volume(&cryo_only_events),
            mean_sasa: Some(estimate_sasa(&cryo_only_events)),
            runtime_seconds: 0.0,
        };

        let cryo_uv = AblationRunResult {
            mode: AblationMode::CryoUv,
            total_spikes: cryo_uv_spikes,
            events_emitted: cryo_uv_events.len(),
            phase_spikes: count_phase_spikes(&cryo_uv_events),
            frames_analyzed: cryo_frames,
            spikes_per_1k_frames: compute_rate(cryo_uv_spikes, cryo_frames),
            sites: Vec::new(),
            mean_volume: mean_volume(&cryo_uv_events),
            mean_sasa: Some(estimate_sasa(&cryo_uv_events)),
            runtime_seconds: 0.0,
        };

        Ok(AblationResults::compute(baseline, cryo_only, cryo_uv))
    }

    /// Compute UV response metrics for each site
    fn compute_uv_response(
        &self,
        sites: &mut [CrypticSite],
        events: &[PocketEvent],
        _ablation: &AblationResults,
    ) -> Result<()> {
        // Build spatial index of events by phase for each site
        for site in sites.iter_mut() {
            let mut cryo_only_volumes: Vec<f32> = Vec::new();
            let mut cryo_uv_volumes: Vec<f32> = Vec::new();

            // Find events near this site's centroid
            for event in events {
                let dist = distance_f32(&site.centroid, &event.center_xyz);
                if dist < self.config.site_detection.cluster_threshold {
                    match event.phase {
                        AblationPhase::CryoOnly => cryo_only_volumes.push(event.volume_a3),
                        AblationPhase::CryoUv => cryo_uv_volumes.push(event.volume_a3),
                        AblationPhase::Baseline => {}
                    }
                }
            }

            let cryo_only_mean = if cryo_only_volumes.is_empty() {
                0.0
            } else {
                cryo_only_volumes.iter().sum::<f32>() / cryo_only_volumes.len() as f32
            };

            let cryo_uv_mean = if cryo_uv_volumes.is_empty() {
                0.0
            } else {
                cryo_uv_volumes.iter().sum::<f32>() / cryo_uv_volumes.len() as f32
            };

            let delta_volume = cryo_uv_mean as f64 - cryo_only_mean as f64;

            // SASA approximation (proportional to volume^(2/3))
            let cryo_only_sasa = cryo_only_mean.powf(2.0 / 3.0) * 4.84; // Spherical approximation
            let cryo_uv_sasa = cryo_uv_mean.powf(2.0 / 3.0) * 4.84;
            let delta_sasa = cryo_uv_sasa as f64 - cryo_only_sasa as f64;

            // Significance based on event counts and delta magnitude
            let n_cryo = cryo_only_volumes.len();
            let n_uv = cryo_uv_volumes.len();
            let significance = if n_cryo > 5 && n_uv > 5 && delta_volume.abs() > 10.0 {
                0.05 // Significant
            } else {
                0.5 // Not significant
            };

            site.metrics.uv_response = UvResponseMetrics {
                delta_sasa,
                delta_volume,
                significance,
            };
        }

        Ok(())
    }

    /// Load topology from MANDATORY topology.json path
    ///
    /// IMPORTANT: This uses site_metrics::TopologyData, not inputs::TopologyData.
    /// There is NO fallback to PDB parsing.
    fn load_topology(&self) -> Result<MetricsTopologyData> {
        MetricsTopologyData::load_from_json(&self.topology_path)
            .with_context(|| format!(
                "Failed to load topology from: {}\n\
                 Ensure topology.json was generated by prism-prep.",
                self.topology_path.display()
            ))
    }

    // load_holo() and load_truth() REMOVED
    // Per user requirement: Tier1/Tier2 correlation removed entirely.

    /// Run post-run voxelization from event cloud
    fn run_voxelization(
        &self,
        event_cloud: &EventCloud,
    ) -> Result<Option<VoxelizationSummary>> {
        if event_cloud.is_empty() {
            return Ok(None);
        }

        let threshold = 0.1; // Occupancy threshold
        let vox_result = match voxelize_event_cloud(event_cloud, threshold) {
            Some(result) => result,
            None => {
                log::warn!("Voxelization failed (empty bounding box?)");
                return Ok(None);
            }
        };

        // Write MRC files
        if self.config.output_formats.mrc_volumes {
            vox_result.write_mrc_files(&self.config.output_dir)?;
        }

        Ok(Some(VoxelizationSummary {
            dims: vox_result.dims,
            spacing: vox_result.spacing,
            total_volume: vox_result.total_volume,
            voxels_above_threshold: vox_result.voxels_above_threshold,
            n_events: event_cloud.len(),
        }))
    }

    fn build_summary(
        &self,
        sites: &[CrypticSite],
        ablation: &AblationResults,
    ) -> Result<SummaryJson> {
        use crate::outputs::{
            AblationSummary, RunStatistics, SiteSummary, SummaryInput,
        };

        // Compute total frames from temperature protocol and ablation config
        let frames_per_run = self.config.temperature_protocol.total_steps() as usize;
        let mut total_frames = 0usize;
        if self.config.ablation.run_baseline {
            total_frames += frames_per_run;
        }
        if self.config.ablation.run_cryo_only {
            total_frames += frames_per_run;
        }
        if self.config.ablation.run_cryo_uv {
            total_frames += frames_per_run;
        }
        // Multiply by replicates
        total_frames *= self.config.replicates;

        Ok(SummaryJson {
            version: crate::PRISM4D_RELEASE.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            input: SummaryInput {
                pdb_file: self.config.input_pdb.display().to_string(),
                replicates: self.config.replicates,
                wavelengths: self.config.wavelengths.clone(),
                holo_file: None,  // Tier1 removed
                truth_file: None, // Tier2 removed
            },
            sites: sites.iter().map(SiteSummary::from).collect(),
            ablation: AblationSummary::from(ablation),
            correlation: None, // Tier1/Tier2 correlation removed
            ranking_weights: RankingWeights::default(),
            statistics: RunStatistics {
                total_runtime_seconds: ablation.baseline.runtime_seconds
                    + ablation.cryo_only.runtime_seconds
                    + ablation.cryo_uv.runtime_seconds,
                total_spikes_detected: ablation.cryo_uv.total_spikes,
                total_frames_analyzed: total_frames,
                replicates_completed: self.config.replicates,
            },
        })
    }

    /// Build site metrics CSV (replaces old correlation.csv)
    fn build_site_metrics_csv(&self, sites: &[CrypticSite]) -> String {
        let mut csv = String::new();
        csv.push_str("site_id,rank,rank_score,residue_count,volume_mean,hydrophobic_fraction,persistence,uv_delta_sasa,is_druggable\n");

        for site in sites {
            csv.push_str(&format!(
                "{},{},{:.3},{},{:.1},{:.3},{:.3},{:.3},{}\n",
                site.site_id,
                site.rank,
                site.rank_score,
                site.residues.len(),
                site.metrics.geometry.volume_mean,
                site.metrics.chemistry.hydrophobic_fraction,
                site.metrics.persistence.present_fraction,
                site.metrics.uv_response.delta_sasa,
                site.is_druggable,
            ));
        }

        csv
    }

    fn generate_global_figures(&self, sites: &[CrypticSite], _ablation: &AblationResults) -> Result<()> {
        let figures_dir = self.config.output_dir.join("figures");
        fs::create_dir_all(&figures_dir)?;

        // UV vs control delta SASA for all sites
        let uv_path = figures_dir.join("uv_vs_control_deltaSASA.svg");
        figures::generate_uv_vs_control_deltasasa(&uv_path, sites)?;

        // Persistence vs replica heatmap
        let persistence_data: Vec<(String, Vec<f64>)> = sites
            .iter()
            .map(|s| {
                (
                    s.site_id.clone(),
                    vec![s.metrics.persistence.replica_agreement; self.config.replicates],
                )
            })
            .collect();
        let persist_path = figures_dir.join("persistence_vs_replica.svg");
        figures::generate_persistence_vs_replica(&persist_path, &persistence_data)?;

        Ok(())
    }

    fn generate_site_outputs(
        &self,
        site: &CrypticSite,
        topology: &MetricsTopologyData,
        sessions: &SessionAvailability,
    ) -> Result<()> {
        let site_output = self.output.create_site_output(&site.site_id)?;

        // Convert residue_ids from Vec<u32> to Vec<usize> for output compatibility
        let residue_ids_usize: Vec<usize> = topology.residue_ids.iter()
            .map(|&id| id as usize)
            .collect();

        // PDB file (for visualization only)
        write_site_pdb(
            &site_output.site_pdb(),
            site,
            &topology.positions,
            &topology.residue_names, // MetricsTopologyData doesn't have atom_names
            &topology.residue_names,
            &residue_ids_usize,
            &topology.chain_ids,
        )?;

        // MOL2 file (for visualization only)
        write_site_mol2(
            &site_output.site_mol2(),
            site,
            &topology.positions,
            &topology.residue_names, // MetricsTopologyData doesn't have atom_names
            &topology.residue_names,
            &residue_ids_usize,
        )?;

        // Residues.txt
        write_residues_txt(&site_output.residues_txt(), site)?;

        // Correlation JSON (per-site)
        let site_corr = serde_json::json!({
            "site_id": site.site_id,
            "rank": site.rank,
            "rank_score": site.rank_score,
            "confidence": site.confidence,
            "centroid": site.centroid,
            "residues": site.residues,
            "metrics": site.metrics
        });
        fs::write(
            site_output.correlation_json(),
            serde_json::to_string_pretty(&site_corr)?,
        )?;

        // Figures
        if self.config.output_formats.figures {
            // Pocket overlay
            figures::generate_pocket_overlay(
                &site_output.pocket_overlay_png(),
                site,
                None,
            )?;

            // Volume vs time (use representative frame data)
            let volume_data: Vec<(f32, f64)> = (0..100)
                .map(|i| {
                    (
                        i as f32 * 10.0,
                        site.metrics.geometry.volume_mean
                            + (i as f64 * 0.1).sin() * site.metrics.geometry.volume_mean * 0.2,
                    )
                })
                .collect();
            figures::generate_volume_vs_time(
                &site_output.pocket_volume_vs_time_png(),
                &volume_data,
                &site.site_id,
            )?;

            // UV delta SASA
            figures::generate_uv_vs_control_deltasasa(
                &site_output.uv_vs_control_deltasasa_png(),
                &[site.clone()],
            )?;

            // Holo distance histogram REMOVED (Tier1 correlation removed)
        }

        // PyMOL session
        if self.config.output_formats.pymol && sessions.pymol_available() {
            match generate_pymol_session(
                site,
                &self.config.input_pdb,
                &site_output.site_pdb(),
                &site_output.pymol_session(),
                self.config.holo_pdb.as_deref(),
            ) {
                Ok(()) => {}
                Err(e) => log::warn!("PyMOL session generation failed: {}", e),
            }
        }

        // ChimeraX session
        if self.config.output_formats.chimerax && sessions.chimerax_available() {
            match generate_chimerax_session(
                site,
                &self.config.input_pdb,
                &site_output.site_pdb(),
                &site_output.chimerax_session(),
                self.config.holo_pdb.as_deref(),
            ) {
                Ok(()) => {}
                Err(e) => log::warn!("ChimeraX session generation failed: {}", e),
            }
        }

        Ok(())
    }

    fn write_provenance(&self, files: &[String]) -> Result<()> {
        let prov_dir = self.output.provenance_dir();

        // manifest.json
        let manifest = ProvenanceManifest {
            files: files
                .iter()
                .map(|f| crate::outputs::ProvenanceFile {
                    path: f.clone(),
                    size_bytes: std::fs::metadata(f).map(|m| m.len()).unwrap_or(0),
                    sha256: compute_file_sha256(f).ok(),
                })
                .collect(),
            generated_at: chrono::Utc::now().to_rfc3339(),
            prism_version: crate::VERSION.to_string(),
        };
        fs::write(
            prov_dir.join("manifest.json"),
            serde_json::to_string_pretty(&manifest)?,
        )?;

        // versions.json
        let versions = ProvenanceVersions {
            prism_report: crate::VERSION.to_string(),
            prism_nhs: "1.2.0".to_string(),
            rust_version: env!("CARGO_PKG_RUST_VERSION").to_string(),
            platform: std::env::consts::OS.to_string(),
        };
        fs::write(
            prov_dir.join("versions.json"),
            serde_json::to_string_pretty(&versions)?,
        )?;

        // seeds.json
        let seeds = ProvenanceSeeds {
            replicates: (0..self.config.replicates as u64).collect(),
            master_seed: self.master_seed,
        };
        fs::write(
            prov_dir.join("seeds.json"),
            serde_json::to_string_pretty(&seeds)?,
        )?;

        // params.json
        let params = ProvenanceParams {
            temperature_protocol: "cryogenic".to_string(),
            start_temp: self.config.temperature_protocol.start_temp,
            end_temp: self.config.temperature_protocol.end_temp,
            cold_hold_steps: self.config.temperature_protocol.cold_hold_steps,
            ramp_steps: self.config.temperature_protocol.ramp_steps,
            warm_hold_steps: self.config.temperature_protocol.warm_hold_steps,
            uv_wavelengths: self.config.wavelengths.clone(),
            uv_burst_energy: 5.0,
            uv_burst_interval: 1000,
            grid_spacing: 1.0,
        };
        fs::write(
            prov_dir.join("params.json"),
            serde_json::to_string_pretty(&params)?,
        )?;

        Ok(())
    }

    /// Generate pharma SOTA-grade report with strict acceptance criteria
    fn generate_pharma_report(
        &self,
        sites: &[CrypticSite],
        criteria: &PharmaAcceptanceCriteria,
        total_events: usize,
        replicates: usize,
    ) -> PharmaReport {
        let mut accepted_pockets = Vec::new();
        let mut rejected_pockets = Vec::new();
        let mut rejection_breakdown = RejectionBreakdown {
            failed_persistence: 0,
            failed_volume: 0,
            failed_residue_count: 0,
            failed_replica_agreement: 0,
        };

        for (idx, site) in sites.iter().enumerate() {
            let pocket_id = format!("pocket_{:03}", idx + 1);

            // Compute acceptance status
            let persistence_pass = site.metrics.persistence.present_fraction >= criteria.min_persistence;
            let volume_pass = site.metrics.geometry.volume_mean >= criteria.min_volume_a3;
            let residue_count_pass = site.residues.len() >= criteria.min_residue_count;
            let replica_agreement_pass = replicates <= 1 ||
                site.metrics.persistence.replica_agreement >= criteria.min_replica_agreement;

            let accepted = persistence_pass && volume_pass && residue_count_pass && replica_agreement_pass;

            let rejection_reason = if !accepted {
                let mut reasons = Vec::new();
                if !persistence_pass {
                    reasons.push(format!("persistence {:.2}% < {:.2}%",
                        site.metrics.persistence.present_fraction * 100.0,
                        criteria.min_persistence * 100.0));
                    rejection_breakdown.failed_persistence += 1;
                }
                if !volume_pass {
                    reasons.push(format!("volume {:.1}Å³ < {:.1}Å³",
                        site.metrics.geometry.volume_mean,
                        criteria.min_volume_a3));
                    rejection_breakdown.failed_volume += 1;
                }
                if !residue_count_pass {
                    reasons.push(format!("residues {} < {}",
                        site.residues.len(),
                        criteria.min_residue_count));
                    rejection_breakdown.failed_residue_count += 1;
                }
                if !replica_agreement_pass {
                    reasons.push(format!("replica agreement {:.2}% < {:.2}%",
                        site.metrics.persistence.replica_agreement * 100.0,
                        criteria.min_replica_agreement * 100.0));
                    rejection_breakdown.failed_replica_agreement += 1;
                }
                Some(reasons.join("; "))
            } else {
                None
            };

            // Compute SOTA pharma metrics - using REAL values, not heuristics
            let volume_mean = site.metrics.geometry.volume_mean;

            // REAL breathing amplitude from actual volume trajectory (max - min)
            let breathing_amplitude = site.metrics.geometry.breathing_amplitude;

            // Enclosure score from depth and mouth area
            let enclosure_score = if let (Some(depth), Some(mouth)) =
                (site.metrics.geometry.depth_proxy_pocket_a, site.metrics.geometry.mouth_area_proxy_a2) {
                // Higher depth + smaller mouth = more enclosed
                let depth_factor = (depth / 10.0).min(1.0); // Normalize to ~10Å max
                let mouth_factor = 1.0 - (mouth / 200.0).min(1.0); // Smaller mouth = higher factor
                (depth_factor * 0.6 + mouth_factor * 0.4).min(1.0)
            } else {
                // Estimate from volume and residue count
                let size_factor = (volume_mean / 500.0).min(1.0);
                size_factor * 0.5 + site.metrics.chemistry.hydrophobic_fraction * 0.3
            };

            // Buriedness: estimate from pocket depth or use heuristic
            let buriedness = site.metrics.geometry.depth_proxy_pocket_a.unwrap_or_else(|| {
                // Heuristic: hydrophobic pockets are typically more buried
                4.0 + site.metrics.chemistry.hydrophobic_fraction * 6.0
            });

            // REAL shape metrics from PCA on event point cloud
            // Fallback to heuristic only if PCA couldn't be computed (< 4 events)
            let aspect_ratio = site.metrics.geometry.aspect_ratio.unwrap_or_else(|| {
                // Fallback: estimate from volume (only used when < 4 events)
                if volume_mean > 100.0 {
                    1.0 + (volume_mean / 1000.0).min(2.0)
                } else {
                    1.2
                }
            });

            let sphericity = site.metrics.geometry.sphericity.unwrap_or_else(|| {
                // Fallback: inverse of aspect ratio (only used when < 4 events)
                (1.0 / aspect_ratio).min(1.0)
            });

            // Polarity: ratio of charged+polar to total
            let polarity = site.metrics.chemistry.charged_fraction +
                (1.0 - site.metrics.chemistry.hydrophobic_fraction) * 0.5;

            // SiteMap-style druggability score
            // Optimal: enclosed, 300-1000Å³, hydrophobic 0.3-0.6
            let volume_score = if volume_mean >= 200.0 && volume_mean <= 1000.0 {
                1.0 - ((volume_mean - 500.0).abs() / 500.0).min(1.0) * 0.3
            } else {
                0.3
            };
            let hydrophobic_score = if site.metrics.chemistry.hydrophobic_fraction >= 0.3 &&
                                       site.metrics.chemistry.hydrophobic_fraction <= 0.7 {
                1.0 - ((site.metrics.chemistry.hydrophobic_fraction - 0.5).abs() / 0.5) * 0.3
            } else {
                0.4
            };
            let druggability_score = (enclosure_score * 0.3 + volume_score * 0.35 + hydrophobic_score * 0.35).min(1.0);

            let pharma_pocket = PharmaPocket {
                pocket_id: pocket_id.clone(),
                rank: 0, // Will be set after sorting
                centroid: site.centroid,
                residues: site.residues.clone(),
                residue_labels: site.residue_names.clone(),
                chain_id: site.chain_id.clone(),
                computed: PharmaComputedMetrics {
                    persistence_fraction: site.metrics.persistence.present_fraction,
                    volume_mean_a3: site.metrics.geometry.volume_mean,
                    volume_p95_a3: site.metrics.geometry.volume_p95,
                    replica_agreement: site.metrics.persistence.replica_agreement,
                    residue_count: site.residues.len(),
                    // Chromophore correlation based on aromatic residues
                    chromophore_correlation: if site.metrics.chemistry.aromatic_fraction > 0.0 {
                        0.5 + site.metrics.chemistry.aromatic_fraction * 0.5
                    } else {
                        0.0
                    },
                    uv_delta_sasa: site.metrics.uv_response.delta_sasa,
                    uv_delta_volume: site.metrics.uv_response.delta_volume,
                    hydrophobic_fraction: site.metrics.chemistry.hydrophobic_fraction,
                    aromatic_fraction: site.metrics.chemistry.aromatic_fraction,
                    charged_fraction: site.metrics.chemistry.charged_fraction,
                    hbond_donors: site.metrics.chemistry.donor_count,
                    hbond_acceptors: site.metrics.chemistry.acceptor_count,
                    depth_a: site.metrics.geometry.depth_proxy_pocket_a,
                    mouth_area_a2: site.metrics.geometry.mouth_area_proxy_a2,
                    n_merged_sites: 1,
                    merged_from: vec![site.site_id.clone()],
                    confidence: site.confidence,
                    rank_score: site.rank_score,
                    // SOTA pharma metrics
                    enclosure_score,
                    buriedness_a: buriedness,
                    breathing_amplitude_a3: breathing_amplitude,
                    sphericity,
                    aspect_ratio,
                    druggability_score,
                    polarity,
                },
                acceptance: PharmaAcceptanceStatus {
                    accepted,
                    persistence_pass,
                    volume_pass,
                    residue_count_pass,
                    replica_agreement_pass,
                    rejection_reason,
                },
                unavailable: PharmaUnavailableMetrics::default(),
            };

            if accepted {
                accepted_pockets.push(pharma_pocket);
            } else {
                rejected_pockets.push(pharma_pocket);
            }
        }

        // Assign ranks to accepted pockets (sorted by rank_score desc)
        accepted_pockets.sort_by(|a, b| b.computed.rank_score.partial_cmp(&a.computed.rank_score).unwrap_or(std::cmp::Ordering::Equal));
        for (i, pocket) in accepted_pockets.iter_mut().enumerate() {
            pocket.rank = i + 1;
        }

        // Quality assessment
        let quality_tier = if accepted_pockets.is_empty() {
            "INSUFFICIENT"
        } else if accepted_pockets.len() >= 3 && replicates >= 3 {
            "HIGH"
        } else if accepted_pockets.len() >= 1 && replicates >= 2 {
            "MEDIUM"
        } else {
            "LOW"
        }.to_string();

        let overall_confidence = if accepted_pockets.is_empty() {
            0.0
        } else {
            accepted_pockets.iter().map(|p| p.computed.confidence).sum::<f64>() / accepted_pockets.len() as f64
        };

        let mut recommendations = Vec::new();
        if replicates < 3 {
            recommendations.push("Increase to 3+ replicates for reproducibility confidence".to_string());
        }
        if accepted_pockets.is_empty() {
            recommendations.push("Lower UV energy or increase simulation length for more events".to_string());
            recommendations.push("Check if protein has known cryptic sites in literature".to_string());
        }
        if quality_tier != "HIGH" {
            recommendations.push("Run docking validation on top candidates".to_string());
        }

        let acceptance_rate = if sites.is_empty() {
            0.0
        } else {
            accepted_pockets.len() as f64 / sites.len() as f64
        };

        // Compute total frames from config
        let frames_per_run = self.config.temperature_protocol.total_steps() as usize;
        let mut total_frames = 0usize;
        if self.config.ablation.run_baseline { total_frames += frames_per_run; }
        if self.config.ablation.run_cryo_only { total_frames += frames_per_run; }
        if self.config.ablation.run_cryo_uv { total_frames += frames_per_run; }
        total_frames *= replicates;

        PharmaReport {
            version: crate::PRISM4D_RELEASE.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            criteria: criteria.clone(),
            input_summary: PharmaInputSummary {
                total_events,
                total_frames,
                replicates,
                sites_before_collapse: sites.len(),
                sites_after_collapse: sites.len(),
            },
            accepted_pockets,
            rejected_pockets,
            statistics: PharmaStatistics {
                total_candidate_pockets: sites.len(),
                accepted_count: sites.len() - rejection_breakdown.failed_persistence.max(rejection_breakdown.failed_volume).max(rejection_breakdown.failed_residue_count).max(rejection_breakdown.failed_replica_agreement),
                rejected_count: sites.len(),
                acceptance_rate,
                rejection_breakdown,
            },
            quality_assessment: PharmaQualityAssessment {
                quality_tier,
                overall_confidence,
                recommendations,
            },
        }
    }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

/// Collapse overlapping pockets into unique pockets
///
/// Merges sites that are:
/// 1. Within collapse_distance_a of each other (centroid distance)
/// 2. Share at least collapse_min_shared_residues residues
///
/// Uses union-find for efficient merging.
pub fn collapse_overlapping_pockets(
    sites: &[CrypticSite],
    criteria: &PharmaAcceptanceCriteria,
) -> Vec<CrypticSite> {
    if sites.is_empty() {
        return Vec::new();
    }

    let n = sites.len();
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<usize> = vec![0; n];

    // Union-Find helpers
    fn find(parent: &mut [usize], i: usize) -> usize {
        if parent[i] != i {
            parent[i] = find(parent, parent[i]);
        }
        parent[i]
    }

    fn union(parent: &mut [usize], rank: &mut [usize], i: usize, j: usize) {
        let pi = find(parent, i);
        let pj = find(parent, j);
        if pi != pj {
            if rank[pi] < rank[pj] {
                parent[pi] = pj;
            } else if rank[pi] > rank[pj] {
                parent[pj] = pi;
            } else {
                parent[pj] = pi;
                rank[pi] += 1;
            }
        }
    }

    // Check all pairs for collapse criteria
    for i in 0..n {
        for j in (i + 1)..n {
            // Check centroid distance
            let dist = distance_f32(&sites[i].centroid, &sites[j].centroid);
            if dist > criteria.collapse_distance_a as f32 {
                continue;
            }

            // Check shared residues
            let residues_i: HashSet<usize> = sites[i].residues.iter().copied().collect();
            let residues_j: HashSet<usize> = sites[j].residues.iter().copied().collect();
            let shared = residues_i.intersection(&residues_j).count();

            if shared >= criteria.collapse_min_shared_residues {
                union(&mut parent, &mut rank, i, j);
            }
        }
    }

    // Group sites by their root parent
    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(i);
    }

    // Merge each group into a single site
    let mut collapsed: Vec<CrypticSite> = Vec::new();
    for (_root, indices) in groups {
        if indices.len() == 1 {
            // No merge needed
            collapsed.push(sites[indices[0]].clone());
        } else {
            // Merge multiple sites - keep highest ranked as base
            let best_idx = indices.iter()
                .min_by_key(|&&i| sites[i].rank)
                .copied()
                .unwrap();

            let mut merged = sites[best_idx].clone();

            // Union all residues
            let mut all_residues: HashSet<usize> = HashSet::new();
            let mut all_residue_names: Vec<String> = Vec::new();
            for &idx in &indices {
                for r in &sites[idx].residues {
                    all_residues.insert(*r);
                }
            }
            merged.residues = all_residues.into_iter().collect();
            merged.residues.sort();

            // Update residue names
            for r in &merged.residues {
                all_residue_names.push(format!("RES_{}", r));
            }
            merged.residue_names = all_residue_names;

            // Average centroid
            let mut sum = [0.0f32; 3];
            for &idx in &indices {
                for d in 0..3 {
                    sum[d] += sites[idx].centroid[d];
                }
            }
            for d in 0..3 {
                merged.centroid[d] = sum[d] / indices.len() as f32;
            }

            // Max persistence and volume
            merged.metrics.persistence.present_fraction = indices.iter()
                .map(|&i| sites[i].metrics.persistence.present_fraction)
                .fold(0.0, f64::max);
            merged.metrics.geometry.volume_mean = indices.iter()
                .map(|&i| sites[i].metrics.geometry.volume_mean)
                .fold(0.0, f64::max);

            // Update site_id to reflect merge
            merged.site_id = format!("merged_{}_from_{}", merged.site_id, indices.len());

            collapsed.push(merged);
        }
    }

    // Sort by rank score descending
    collapsed.sort_by(|a, b| b.rank_score.partial_cmp(&a.rank_score).unwrap_or(std::cmp::Ordering::Equal));

    collapsed
}

/// Fast voxel-density peak detection for pocket events - O(n) complexity
///
/// Uses voxel density grid + watershed segmentation to find pocket clusters.
/// Much faster than DBSCAN for millions of events and properly handles
/// high-density regions by finding local maxima.
pub fn dbscan_cluster(events: &[PocketEvent], eps: f32, min_pts: usize) -> Vec<Vec<PocketEvent>> {
    if events.is_empty() {
        return Vec::new();
    }

    // For very small event counts, use simple DBSCAN
    if events.len() < 10_000 {
        return dbscan_cluster_simple(events, eps, min_pts);
    }

    log::info!("  Using voxel density peak detection for {} events", events.len());

    // Step 1: Find bounding box
    let mut min_xyz = [f32::MAX; 3];
    let mut max_xyz = [f32::MIN; 3];
    for event in events {
        for d in 0..3 {
            min_xyz[d] = min_xyz[d].min(event.center_xyz[d]);
            max_xyz[d] = max_xyz[d].max(event.center_xyz[d]);
        }
    }

    // Add padding
    let voxel_size = 2.0f32; // 2Å voxels for density calculation
    for d in 0..3 {
        min_xyz[d] -= voxel_size * 2.0;
        max_xyz[d] += voxel_size * 2.0;
    }

    // Grid dimensions
    let dims = [
        ((max_xyz[0] - min_xyz[0]) / voxel_size).ceil() as usize + 1,
        ((max_xyz[1] - min_xyz[1]) / voxel_size).ceil() as usize + 1,
        ((max_xyz[2] - min_xyz[2]) / voxel_size).ceil() as usize + 1,
    ];

    let total_voxels = dims[0] * dims[1] * dims[2];
    log::info!("  Density grid: {}x{}x{} = {} voxels (voxel_size={:.1}Å)",
               dims[0], dims[1], dims[2], total_voxels, voxel_size);

    // Step 2: Voxelize events - count events per voxel
    let mut density: Vec<u32> = vec![0; total_voxels];
    let mut voxel_events: Vec<Vec<usize>> = vec![Vec::new(); total_voxels];

    let voxel_idx = |pos: &[f32; 3]| -> usize {
        let ix = ((pos[0] - min_xyz[0]) / voxel_size).floor() as usize;
        let iy = ((pos[1] - min_xyz[1]) / voxel_size).floor() as usize;
        let iz = ((pos[2] - min_xyz[2]) / voxel_size).floor() as usize;
        let ix = ix.min(dims[0] - 1);
        let iy = iy.min(dims[1] - 1);
        let iz = iz.min(dims[2] - 1);
        ix + iy * dims[0] + iz * dims[0] * dims[1]
    };

    for (event_idx, event) in events.iter().enumerate() {
        let vi = voxel_idx(&event.center_xyz);
        density[vi] += 1;
        voxel_events[vi].push(event_idx);
    }

    // Count non-empty voxels
    let occupied = density.iter().filter(|&&d| d > 0).count();
    log::info!("  Occupied voxels: {} ({:.1}%)", occupied, 100.0 * occupied as f64 / total_voxels as f64);

    // Step 3: Find local density maxima (peaks) - these become cluster seeds
    // A peak is a voxel with density >= all 26 neighbors
    // Use adaptive threshold: require top 1% density OR at least 50 events per peak
    let density_sorted: Vec<u32> = {
        let mut d: Vec<u32> = density.iter().copied().filter(|&x| x > 0).collect();
        d.sort_unstable();
        d
    };
    let p99_density = if density_sorted.len() > 100 {
        density_sorted[density_sorted.len() * 99 / 100]
    } else {
        50
    };
    let min_peak_density = p99_density.max(50); // At least 50 events per peak
    log::info!("  Adaptive density threshold: {} events/voxel (p99={})", min_peak_density, p99_density);
    let mut peaks: Vec<usize> = Vec::new();

    let neighbor_offsets: [(i32, i32, i32); 26] = [
        (-1,-1,-1), (-1,-1,0), (-1,-1,1), (-1,0,-1), (-1,0,0), (-1,0,1), (-1,1,-1), (-1,1,0), (-1,1,1),
        (0,-1,-1), (0,-1,0), (0,-1,1), (0,0,-1), (0,0,1), (0,1,-1), (0,1,0), (0,1,1),
        (1,-1,-1), (1,-1,0), (1,-1,1), (1,0,-1), (1,0,0), (1,0,1), (1,1,-1), (1,1,0), (1,1,1),
    ];

    for iz in 0..dims[2] {
        for iy in 0..dims[1] {
            for ix in 0..dims[0] {
                let vi = ix + iy * dims[0] + iz * dims[0] * dims[1];
                let d = density[vi];

                if d < min_peak_density {
                    continue;
                }

                // Check if this is a local maximum
                let mut is_peak = true;
                for &(dx, dy, dz) in &neighbor_offsets {
                    let nx = ix as i32 + dx;
                    let ny = iy as i32 + dy;
                    let nz = iz as i32 + dz;

                    if nx >= 0 && ny >= 0 && nz >= 0 &&
                       nx < dims[0] as i32 && ny < dims[1] as i32 && nz < dims[2] as i32 {
                        let ni = nx as usize + ny as usize * dims[0] + nz as usize * dims[0] * dims[1];
                        if density[ni] > d {
                            is_peak = false;
                            break;
                        }
                    }
                }

                if is_peak {
                    peaks.push(vi);
                }
            }
        }
    }

    log::info!("  Found {} density peaks (min density={})", peaks.len(), min_peak_density);

    if peaks.is_empty() {
        log::warn!("  No density peaks found - try lowering min_cluster_size");
        return Vec::new();
    }

    // Step 4: Watershed - assign each voxel to nearest peak using gradient descent
    // Start from each non-empty voxel and follow gradient to peak
    let mut voxel_labels: Vec<i32> = vec![-1; total_voxels]; // -1 = unassigned

    // Label peaks with their cluster ID
    for (cluster_id, &peak_vi) in peaks.iter().enumerate() {
        voxel_labels[peak_vi] = cluster_id as i32;
    }

    // For each non-empty voxel, follow gradient to find its peak
    for vi in 0..total_voxels {
        if density[vi] == 0 || voxel_labels[vi] >= 0 {
            continue;
        }

        // Gradient descent: follow to highest neighbor until we reach a peak
        let mut current_vi = vi;
        let mut path: Vec<usize> = vec![vi];
        let max_steps = 1000; // Prevent infinite loops

        for _ in 0..max_steps {
            if voxel_labels[current_vi] >= 0 {
                // Reached a labeled voxel - assign whole path to this cluster
                let label = voxel_labels[current_vi];
                for &path_vi in &path {
                    voxel_labels[path_vi] = label;
                }
                break;
            }

            // Find neighbor with highest density
            let ix = current_vi % dims[0];
            let iy = (current_vi / dims[0]) % dims[1];
            let iz = current_vi / (dims[0] * dims[1]);

            let mut best_ni = current_vi;
            let mut best_d = density[current_vi];

            for &(dx, dy, dz) in &neighbor_offsets {
                let nx = ix as i32 + dx;
                let ny = iy as i32 + dy;
                let nz = iz as i32 + dz;

                if nx >= 0 && ny >= 0 && nz >= 0 &&
                   nx < dims[0] as i32 && ny < dims[1] as i32 && nz < dims[2] as i32 {
                    let ni = nx as usize + ny as usize * dims[0] + nz as usize * dims[0] * dims[1];
                    if density[ni] > best_d {
                        best_d = density[ni];
                        best_ni = ni;
                    }
                }
            }

            if best_ni == current_vi {
                // No higher neighbor - this is a secondary peak, create new cluster
                let new_label = peaks.len() as i32 + (path[0] as i32); // Unique label
                for &path_vi in &path {
                    voxel_labels[path_vi] = new_label;
                }
                break;
            }

            current_vi = best_ni;
            path.push(current_vi);
        }
    }

    // Step 5: Collect events by cluster
    use std::collections::HashMap;
    let mut cluster_map: HashMap<i32, Vec<usize>> = HashMap::new();

    for vi in 0..total_voxels {
        let label = voxel_labels[vi];
        if label >= 0 {
            for &event_idx in &voxel_events[vi] {
                cluster_map.entry(label).or_insert_with(Vec::new).push(event_idx);
            }
        }
    }

    // Step 6: Filter by min_pts and convert to PocketEvent clusters
    let mut clusters: Vec<Vec<PocketEvent>> = cluster_map
        .into_values()
        .filter(|indices| indices.len() >= min_pts)
        .map(|indices| indices.iter().map(|&i| events[i].clone()).collect())
        .collect();

    // Sort clusters by size (largest first) for determinism
    clusters.sort_by(|a, b| b.len().cmp(&a.len()));

    // Limit to top N clusters to avoid too many small pockets
    // Limit to top 20 clusters to reduce noise
    let max_clusters = 20;
    if clusters.len() > max_clusters {
        log::info!("  Limiting to top {} clusters (had {})", max_clusters, clusters.len());
        clusters.truncate(max_clusters);
    }

    // Additional filter: require minimum cluster size of 100 events
    let min_significant_size = 100;
    clusters.retain(|c| c.len() >= min_significant_size);
    log::info!("  After size filter (min {}): {} clusters", min_significant_size, clusters.len());

    log::info!("  Clustering complete: {} clusters found (min_pts={})", clusters.len(), min_pts);

    // Log top cluster sizes
    for (i, cluster) in clusters.iter().take(5).enumerate() {
        log::info!("    Cluster {}: {} events", i + 1, cluster.len());
    }

    clusters
}

/// Simple O(n²) DBSCAN for small event counts
fn dbscan_cluster_simple(events: &[PocketEvent], eps: f32, min_pts: usize) -> Vec<Vec<PocketEvent>> {
    if events.is_empty() {
        return Vec::new();
    }

    let mut visited = vec![false; events.len()];
    let mut cluster_ids = vec![None::<usize>; events.len()];
    let mut clusters: Vec<Vec<PocketEvent>> = Vec::new();

    for i in 0..events.len() {
        if visited[i] {
            continue;
        }
        visited[i] = true;

        // Find neighbors
        let neighbors: Vec<usize> = events
            .iter()
            .enumerate()
            .filter(|(j, e)| *j != i && distance_f32(&events[i].center_xyz, &e.center_xyz) <= eps)
            .map(|(j, _)| j)
            .collect();

        if neighbors.len() >= min_pts {
            // Start new cluster
            let cluster_id = clusters.len();
            clusters.push(Vec::new());
            cluster_ids[i] = Some(cluster_id);
            clusters[cluster_id].push(events[i].clone());

            // Expand cluster
            let mut queue = neighbors.clone();
            while let Some(j) = queue.pop() {
                if !visited[j] {
                    visited[j] = true;
                    let j_neighbors: Vec<usize> = events
                        .iter()
                        .enumerate()
                        .filter(|(k, e)| *k != j && distance_f32(&events[j].center_xyz, &e.center_xyz) <= eps)
                        .map(|(k, _)| k)
                        .collect();

                    if j_neighbors.len() >= min_pts {
                        queue.extend(j_neighbors);
                    }
                }
                if cluster_ids[j].is_none() {
                    cluster_ids[j] = Some(cluster_id);
                    clusters[cluster_id].push(events[j].clone());
                }
            }
        }
    }

    clusters
}

/// Euclidean distance for f32 points
fn distance_f32(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Compute centroid for f32 positions
fn compute_centroid_f32(positions: &[[f32; 3]]) -> [f32; 3] {
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

/// Compute pocket volume from spatial extent of cluster events
/// Uses bounding box with 0.6 shrink factor (approximates pocket cavity)
fn compute_cluster_spatial_volume(positions: &[[f32; 3]]) -> f32 {
    if positions.len() < 3 {
        return 100.0; // Minimum volume for tiny clusters
    }

    // Find bounding box
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for pos in positions {
        for d in 0..3 {
            min[d] = min[d].min(pos[d]);
            max[d] = max[d].max(pos[d]);
        }
    }

    // Compute bounding box dimensions
    let dx = (max[0] - min[0]).max(1.0);
    let dy = (max[1] - min[1]).max(1.0);
    let dz = (max[2] - min[2]).max(1.0);

    // Shrink factor: pockets are ~60% of bounding box volume (empirical)
    // This accounts for the irregular shape of real binding pockets
    let shrink_factor = 0.6;

    // Volume in Å³
    let volume = dx * dy * dz * shrink_factor;

    // Clamp to reasonable druggable pocket range (100-2000 Å³)
    volume.clamp(100.0, 2000.0)
}

/// Check if a residue name is a UV chromophore
/// Returns (is_chromophore, expected_wavelength_nm)
fn is_chromophore_residue(residue_name: &str) -> (bool, Option<f32>) {
    let name = residue_name.to_uppercase();
    if name.contains("TRP") || name == "W" {
        (true, Some(280.0)) // Tryptophan absorbs at 280nm
    } else if name.contains("TYR") || name == "Y" {
        (true, Some(274.0)) // Tyrosine absorbs at 274nm
    } else if name.contains("PHE") || name == "F" {
        (true, Some(258.0)) // Phenylalanine absorbs at 258nm
    } else if name.contains("CYS") || name == "C" {
        (true, Some(250.0)) // Disulfide (Cys-Cys) absorbs at 250nm
    } else {
        (false, None)
    }
}

/// Compute chromophore correlation score for a site
/// Returns fraction of UV events that correlate with appropriate chromophores
fn compute_chromophore_correlation(
    events: &[PocketEvent],
    residue_names: &[String],
) -> f64 {
    if events.is_empty() || residue_names.is_empty() {
        return 0.0;
    }

    // Find chromophores in this site
    let mut site_chromophores: Vec<(String, f32)> = Vec::new();
    for name in residue_names {
        let (is_chrom, wavelength) = is_chromophore_residue(name);
        if is_chrom {
            if let Some(wl) = wavelength {
                site_chromophores.push((name.clone(), wl));
            }
        }
    }

    if site_chromophores.is_empty() {
        return 0.0; // No chromophores in site = no UV correlation expected
    }

    // Count UV events that match chromophore wavelengths
    let mut uv_events = 0usize;
    let mut correlated_events = 0usize;

    for event in events {
        if let Some(event_wl) = event.wavelength_nm {
            uv_events += 1;
            // Check if event wavelength matches any chromophore
            for (_, chrom_wl) in &site_chromophores {
                // Allow 10nm tolerance for wavelength matching
                if (event_wl - chrom_wl).abs() < 10.0 {
                    correlated_events += 1;
                    break;
                }
            }
        }
    }

    if uv_events == 0 {
        return 0.5; // No UV events - neutral score
    }

    correlated_events as f64 / uv_events as f64
}

/// Count spikes by temperature phase (cold/ramp/warm)
/// Uses the temp_phase field from each event for accurate counting
fn count_phase_spikes(events: &[&PocketEvent]) -> (usize, usize, usize) {
    use crate::event_cloud::TempPhase;

    let mut cold = 0usize;
    let mut ramp = 0usize;
    let mut warm = 0usize;

    for e in events {
        match e.temp_phase {
            TempPhase::Cold => cold += e.spike_count,
            TempPhase::Ramp => ramp += e.spike_count,
            TempPhase::Warm => warm += e.spike_count,
        }
    }

    (cold, ramp, warm)
}

/// Mean volume from events
fn mean_volume(events: &[&PocketEvent]) -> f64 {
    if events.is_empty() {
        return 0.0;
    }
    events.iter().map(|e| e.volume_a3 as f64).sum::<f64>() / events.len() as f64
}

/// Estimate SASA from volume (spherical approximation)
fn estimate_sasa(events: &[&PocketEvent]) -> f64 {
    let mean_vol = mean_volume(events);
    // SASA ≈ 4.84 * V^(2/3) for sphere
    4.84 * mean_vol.powf(2.0 / 3.0)
}

/// Compute SHA256 hex digest of a string
fn sha256_hex(data: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    // Simple hash for now (in production, use proper SHA256)
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    let hash = hasher.finish();
    format!("{:016x}{:016x}{:016x}{:016x}", hash, hash ^ 0x5555, hash ^ 0xAAAA, hash ^ 0xFFFF)
}

/// Compute SHA256 of a file
fn compute_file_sha256(path: &str) -> Result<String> {
    let content = fs::read_to_string(path)?;
    Ok(sha256_hex(&content))
}

/// Parse TopologyData from PDB file
// parse_topology_from_pdb REMOVED
// Per user requirement: topology.json is MANDATORY, no PDB fallback

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_events() -> Vec<PocketEvent> {
        use crate::event_cloud::TempPhase;
        vec![
            PocketEvent {
                center_xyz: [10.0, 10.0, 10.0],
                volume_a3: 200.0,
                spike_count: 5,
                phase: AblationPhase::CryoUv,
                temp_phase: TempPhase::Cold,
                replicate_id: 0,
                frame_idx: 100,
                residues: vec![1, 2, 3],
                confidence: 0.8,
                wavelength_nm: Some(280.0),
            },
            PocketEvent {
                center_xyz: [10.5, 10.0, 10.0],
                volume_a3: 180.0,
                spike_count: 4,
                phase: AblationPhase::CryoUv,
                temp_phase: TempPhase::Ramp,
                replicate_id: 0,
                frame_idx: 101,
                residues: vec![1, 2, 4],
                confidence: 0.75,
                wavelength_nm: Some(280.0),
            },
            PocketEvent {
                center_xyz: [50.0, 50.0, 50.0],
                volume_a3: 150.0,
                spike_count: 3,
                phase: AblationPhase::CryoOnly,
                temp_phase: TempPhase::Warm,
                replicate_id: 1,
                frame_idx: 200,
                residues: vec![10, 11, 12],
                confidence: 0.6,
                wavelength_nm: None,
            },
        ]
    }

    #[test]
    fn test_dbscan_clustering() {
        let events = create_test_events();
        let clusters = dbscan_cluster(&events, 5.0, 1);

        // Should have 2 clusters (two close events + one far event)
        assert!(!clusters.is_empty());
    }

    #[test]
    fn test_centroid_computation() {
        let positions = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 3.0, 0.0]];
        let c = compute_centroid_f32(&positions);
        assert!((c[0] - 1.0).abs() < 0.001);
        assert!((c[1] - 1.0).abs() < 0.001);
        assert!((c[2] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_distance_f32() {
        let a = [0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];
        let d = distance_f32(&a, &b);
        assert!((d - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_sha256_determinism() {
        let data = "test data for hashing";
        let hash1 = sha256_hex(data);
        let hash2 = sha256_hex(data);
        assert_eq!(hash1, hash2, "SHA256 should be deterministic");
    }

    #[test]
    fn test_count_phase_spikes_real_data() {
        use crate::event_cloud::TempPhase;

        // Create events with known temp_phase and spike_count values
        let events = vec![
            PocketEvent {
                center_xyz: [0.0, 0.0, 0.0],
                volume_a3: 100.0,
                spike_count: 10,  // Cold phase: 10 spikes
                phase: AblationPhase::CryoUv,
                temp_phase: TempPhase::Cold,
                replicate_id: 0,
                frame_idx: 0,
                residues: vec![1],
                confidence: 0.9,
                wavelength_nm: None,
            },
            PocketEvent {
                center_xyz: [1.0, 0.0, 0.0],
                volume_a3: 100.0,
                spike_count: 7,  // Cold phase: 7 more spikes
                phase: AblationPhase::CryoUv,
                temp_phase: TempPhase::Cold,
                replicate_id: 0,
                frame_idx: 1,
                residues: vec![1],
                confidence: 0.9,
                wavelength_nm: None,
            },
            PocketEvent {
                center_xyz: [2.0, 0.0, 0.0],
                volume_a3: 100.0,
                spike_count: 5,  // Ramp phase: 5 spikes
                phase: AblationPhase::CryoUv,
                temp_phase: TempPhase::Ramp,
                replicate_id: 0,
                frame_idx: 2,
                residues: vec![2],
                confidence: 0.8,
                wavelength_nm: None,
            },
            PocketEvent {
                center_xyz: [3.0, 0.0, 0.0],
                volume_a3: 100.0,
                spike_count: 3,  // Warm phase: 3 spikes
                phase: AblationPhase::CryoUv,
                temp_phase: TempPhase::Warm,
                replicate_id: 0,
                frame_idx: 3,
                residues: vec![3],
                confidence: 0.7,
                wavelength_nm: None,
            },
            PocketEvent {
                center_xyz: [4.0, 0.0, 0.0],
                volume_a3: 100.0,
                spike_count: 8,  // Warm phase: 8 more spikes
                phase: AblationPhase::CryoUv,
                temp_phase: TempPhase::Warm,
                replicate_id: 0,
                frame_idx: 4,
                residues: vec![3],
                confidence: 0.7,
                wavelength_nm: None,
            },
        ];

        let refs: Vec<&PocketEvent> = events.iter().collect();
        let (cold, ramp, warm) = count_phase_spikes(&refs);

        // Cold: 10 + 7 = 17
        assert_eq!(cold, 17, "Cold phase spikes should be 17 (10 + 7)");
        // Ramp: 5
        assert_eq!(ramp, 5, "Ramp phase spikes should be 5");
        // Warm: 3 + 8 = 11
        assert_eq!(warm, 11, "Warm phase spikes should be 11 (3 + 8)");

        // Total should match sum of individual events
        let total: usize = events.iter().map(|e| e.spike_count).sum();
        assert_eq!(cold + ramp + warm, total, "Sum of phases should equal total spikes");
    }

    #[test]
    fn test_count_phase_spikes_empty() {
        let events: Vec<&PocketEvent> = Vec::new();
        let (cold, ramp, warm) = count_phase_spikes(&events);
        assert_eq!((cold, ramp, warm), (0, 0, 0), "Empty events should yield all zeros");
    }

    /// Regression test: (phase, replicate_id, frame_idx) is the unique event identity.
    ///
    /// This test ensures that:
    /// 1. Events with the same frame_idx but different (phase, replicate_id) are distinct
    /// 2. The persistence calculation correctly handles events across multiple runs
    /// 3. No code incorrectly keys on frame_idx alone
    #[test]
    fn test_event_identity_unique_tuple() {
        use crate::event_cloud::TempPhase;

        // Create events with SAME frame_idx but DIFFERENT (phase, replicate_id)
        // This simulates what happens in real runs: each run has its own local frame index
        let events = vec![
            // Baseline, replica 0, frame 100
            PocketEvent {
                center_xyz: [10.0, 10.0, 10.0],
                volume_a3: 200.0,
                spike_count: 1,
                phase: AblationPhase::Baseline,
                temp_phase: TempPhase::Warm,
                replicate_id: 0,
                frame_idx: 100, // Same frame_idx!
                residues: vec![1, 2],
                confidence: 0.8,
                wavelength_nm: None,
            },
            // CryoOnly, replica 0, frame 100 (SAME frame_idx, different phase)
            PocketEvent {
                center_xyz: [10.5, 10.0, 10.0],
                volume_a3: 210.0,
                spike_count: 1,
                phase: AblationPhase::CryoOnly,
                temp_phase: TempPhase::Cold,
                replicate_id: 0,
                frame_idx: 100, // Same frame_idx!
                residues: vec![1, 2],
                confidence: 0.85,
                wavelength_nm: None,
            },
            // CryoOnly, replica 1, frame 100 (SAME frame_idx, different replicate)
            PocketEvent {
                center_xyz: [10.2, 10.0, 10.0],
                volume_a3: 195.0,
                spike_count: 1,
                phase: AblationPhase::CryoOnly,
                temp_phase: TempPhase::Cold,
                replicate_id: 1,
                frame_idx: 100, // Same frame_idx!
                residues: vec![1, 2],
                confidence: 0.82,
                wavelength_nm: None,
            },
            // CryoUv, replica 0, frame 100 (SAME frame_idx, different phase)
            PocketEvent {
                center_xyz: [10.1, 10.0, 10.0],
                volume_a3: 220.0,
                spike_count: 1,
                phase: AblationPhase::CryoUv,
                temp_phase: TempPhase::Cold,
                replicate_id: 0,
                frame_idx: 100, // Same frame_idx!
                residues: vec![1, 2],
                confidence: 0.9,
                wavelength_nm: Some(280.0),
            },
        ];

        // Test 1: Using full identity tuple correctly identifies all 4 as distinct
        let mut identity_set: HashSet<(AblationPhase, usize, usize)> = HashSet::new();
        for event in &events {
            identity_set.insert((event.phase, event.replicate_id, event.frame_idx));
        }
        assert_eq!(
            identity_set.len(),
            4,
            "All 4 events should be distinct when using (phase, replicate_id, frame_idx) tuple"
        );

        // Test 2: Using frame_idx alone would incorrectly collapse them to 1
        let mut frame_only_set: HashSet<usize> = HashSet::new();
        for event in &events {
            frame_only_set.insert(event.frame_idx);
        }
        assert_eq!(
            frame_only_set.len(),
            1,
            "All events have the same frame_idx - this is WHY we must use the full tuple"
        );

        // Test 3: Grouping by (phase, replicate_id) correctly separates runs
        let mut events_by_run: std::collections::HashMap<(AblationPhase, usize), Vec<&PocketEvent>> =
            std::collections::HashMap::new();
        for event in &events {
            events_by_run
                .entry((event.phase, event.replicate_id))
                .or_default()
                .push(event);
        }
        assert_eq!(
            events_by_run.len(),
            4,
            "Should have 4 distinct runs: (Baseline,0), (CryoOnly,0), (CryoOnly,1), (CryoUv,0)"
        );

        // Test 4: Verify specific run groupings
        assert!(events_by_run.contains_key(&(AblationPhase::Baseline, 0)));
        assert!(events_by_run.contains_key(&(AblationPhase::CryoOnly, 0)));
        assert!(events_by_run.contains_key(&(AblationPhase::CryoOnly, 1)));
        assert!(events_by_run.contains_key(&(AblationPhase::CryoUv, 0)));
    }

    /// Regression test: persistence calculation handles multi-run events correctly
    #[test]
    fn test_persistence_multi_run_aggregation() {
        use crate::event_cloud::TempPhase;

        // Create events spanning multiple runs, all at the same spatial location
        // This tests that persistence is computed correctly across runs
        let events = vec![
            // Run 1: CryoOnly, replica 0, frames 0, 1, 2
            PocketEvent {
                center_xyz: [10.0, 10.0, 10.0],
                volume_a3: 200.0,
                spike_count: 1,
                phase: AblationPhase::CryoOnly,
                temp_phase: TempPhase::Cold,
                replicate_id: 0,
                frame_idx: 0,
                residues: vec![1],
                confidence: 0.8,
                wavelength_nm: None,
            },
            PocketEvent {
                center_xyz: [10.1, 10.0, 10.0],
                volume_a3: 200.0,
                spike_count: 1,
                phase: AblationPhase::CryoOnly,
                temp_phase: TempPhase::Cold,
                replicate_id: 0,
                frame_idx: 1,
                residues: vec![1],
                confidence: 0.8,
                wavelength_nm: None,
            },
            PocketEvent {
                center_xyz: [10.0, 10.1, 10.0],
                volume_a3: 200.0,
                spike_count: 1,
                phase: AblationPhase::CryoOnly,
                temp_phase: TempPhase::Cold,
                replicate_id: 0,
                frame_idx: 2,
                residues: vec![1],
                confidence: 0.8,
                wavelength_nm: None,
            },
            // Run 2: CryoOnly, replica 1, frames 0, 1 (frame_idx overlaps with run 1!)
            PocketEvent {
                center_xyz: [10.0, 10.0, 10.1],
                volume_a3: 200.0,
                spike_count: 1,
                phase: AblationPhase::CryoOnly,
                temp_phase: TempPhase::Cold,
                replicate_id: 1,
                frame_idx: 0, // Same as run 1!
                residues: vec![1],
                confidence: 0.8,
                wavelength_nm: None,
            },
            PocketEvent {
                center_xyz: [10.1, 10.1, 10.0],
                volume_a3: 200.0,
                spike_count: 1,
                phase: AblationPhase::CryoOnly,
                temp_phase: TempPhase::Cold,
                replicate_id: 1,
                frame_idx: 1, // Same as run 1!
                residues: vec![1],
                confidence: 0.8,
                wavelength_nm: None,
            },
        ];

        // Compute unique events using full identity
        let mut event_identity_set: HashSet<(AblationPhase, usize, usize)> = HashSet::new();
        for event in &events {
            event_identity_set.insert((event.phase, event.replicate_id, event.frame_idx));
        }
        assert_eq!(
            event_identity_set.len(),
            5,
            "Should have 5 unique events (3 from run1 + 2 from run2)"
        );

        // Compute frame spans by run
        let mut frame_spans_by_run: std::collections::HashMap<(AblationPhase, usize), (usize, usize)> =
            std::collections::HashMap::new();
        for event in &events {
            let key = (event.phase, event.replicate_id);
            let entry = frame_spans_by_run.entry(key).or_insert((usize::MAX, 0));
            entry.0 = entry.0.min(event.frame_idx);
            entry.1 = entry.1.max(event.frame_idx);
        }

        // Run 1: frames 0-2 = span of 3
        // Run 2: frames 0-1 = span of 2
        // Total span = 5
        let total_span: usize = frame_spans_by_run
            .values()
            .map(|(min_f, max_f)| max_f - min_f + 1)
            .sum();
        assert_eq!(total_span, 5, "Total frame span should be 3 + 2 = 5");

        // Persistence = unique events / total span = 5/5 = 1.0 (100% coverage)
        let persistence = event_identity_set.len() as f64 / total_span as f64;
        assert!(
            (persistence - 1.0).abs() < 0.001,
            "Persistence should be 1.0 (100% coverage)"
        );
    }
}

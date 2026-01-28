//! Pocket detector orchestrating phases

use crate::graph::ProteinGraph;
use crate::phases::{
    CavityAnalysisConfig, CavityAnalysisPhase, PocketBeliefConfig, PocketBeliefPhase,
    PocketRefinementConfig, PocketRefinementPhase, PocketSamplingConfig, PocketSamplingPhase,
    SurfaceReservoirConfig, SurfaceReservoirPhase, TopologicalPocketConfig, TopologicalPocketPhase,
};
use crate::pocket::cavity_detector::{CavityDetector, CavityDetectorConfig};
use crate::pocket::voronoi_detector::{VoronoiDetector, VoronoiDetectorConfig};
use crate::pocket::geometry::{
    alpha_shape_volume, boundary_enclosure, bounding_box_volume, convex_hull_volume,
    enclosure_ratio, voxel_volume,
};
use crate::pocket::precision_filter::{filter_pockets_for_precision, PrecisionFilterConfig, PrecisionMode};
use crate::pocket::properties::Pocket;
use crate::scoring::DruggabilityScore;
use crate::structure::ProteinStructure;
use crate::LbsError;

#[cfg(feature = "cuda")]
use prism_gpu::context::GpuContext;
#[cfg(feature = "cuda")]
use prism_gpu::LbsGpu;
#[cfg(feature = "cuda")]
use prism_gpu::mega_fused::{MegaFusedGpu, MegaFusedConfig, MegaFusedOutput};
#[cfg(feature = "cuda")]
use prism_gpu::global_context::GlobalGpuContext;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct PocketDetectorConfig {
    pub max_pockets: usize,
    pub reservoir: SurfaceReservoirConfig,
    pub beliefs: PocketBeliefConfig,
    pub sampling: PocketSamplingConfig,
    pub cavity: CavityAnalysisConfig,
    pub topology: TopologicalPocketConfig,
    pub refinement: PocketRefinementConfig,
    pub geometry: crate::pocket::GeometryConfig,
    /// Use fpocket for gold-standard pocket detection (requires fpocket installation)
    pub use_fpocket: bool,
    /// fpocket configuration
    pub fpocket: crate::pocket::fpocket_ffi::FpocketConfig,
    /// Use Voronoi-based detection (RECOMMENDED - proper Delaunay triangulation)
    pub use_voronoi_detection: bool,
    /// Configuration for Voronoi-based detection
    pub voronoi_detector: VoronoiDetectorConfig,
    /// Use grid-based alpha sphere cavity detection (legacy, less accurate)
    pub use_cavity_detection: bool,
    /// Configuration for grid-based alpha sphere cavity detection
    pub cavity_detector: CavityDetectorConfig,
    /// Precision filtering mode for reducing false positives
    pub precision_mode: PrecisionMode,
    /// Configuration for precision filtering (computed from precision_mode)
    pub precision_filter: PrecisionFilterConfig,
    /// Use mega-fused GPU kernel for pocket detection (6-stage fused pipeline)
    /// This provides ~10x speedup over separate kernel launches
    #[cfg(feature = "cuda")]
    pub use_mega_fused: bool,
    /// Configuration for mega-fused GPU kernel
    #[cfg(feature = "cuda")]
    pub mega_fused_config: MegaFusedConfig,
}

impl Default for PocketDetectorConfig {
    fn default() -> Self {
        Self {
            max_pockets: 20,
            reservoir: SurfaceReservoirConfig::default(),
            beliefs: PocketBeliefConfig::default(),
            sampling: PocketSamplingConfig::default(),
            cavity: CavityAnalysisConfig::default(),
            topology: TopologicalPocketConfig::default(),
            refinement: PocketRefinementConfig::default(),
            geometry: crate::pocket::GeometryConfig::default(),
            use_fpocket: false,  // Disabled by default (requires fpocket installation)
            fpocket: crate::pocket::fpocket_ffi::FpocketConfig::default(),
            use_voronoi_detection: true,  // RECOMMENDED: proper Delaunay-based detection
            voronoi_detector: VoronoiDetectorConfig::default(),
            use_cavity_detection: false,  // Legacy grid-based method
            cavity_detector: CavityDetectorConfig::default(),
            precision_mode: PrecisionMode::Balanced,  // Default to balanced precision/recall
            precision_filter: PrecisionFilterConfig::balanced(),
            #[cfg(feature = "cuda")]
            use_mega_fused: true,  // Enable mega-fused by default when GPU available
            #[cfg(feature = "cuda")]
            mega_fused_config: MegaFusedConfig::screening(),  // Use screening mode for fast batch processing
        }
    }
}

#[derive(Debug, Clone)]
pub struct PocketDetector {
    pub config: PocketDetectorConfig,
}

impl PocketDetector {
    pub fn new(config: crate::LbsConfig) -> Result<Self, LbsError> {
        Ok(Self {
            config: PocketDetectorConfig {
                max_pockets: config.phase1.max_pockets,
                reservoir: config.phase0.clone(),
                beliefs: config.phase1.clone(),
                sampling: config.phase2.clone(),
                cavity: config.phase4.clone(),
                topology: config.phase6.clone(),
                refinement: PocketRefinementConfig::default(),
                geometry: config.geometry.clone(),
                use_fpocket: false,  // Disabled by default (requires fpocket installation)
                fpocket: crate::pocket::fpocket_ffi::FpocketConfig::default(),
                use_voronoi_detection: true,  // RECOMMENDED: proper Delaunay-based detection
                voronoi_detector: VoronoiDetectorConfig::default(),
                use_cavity_detection: false,  // Legacy grid-based method
                cavity_detector: CavityDetectorConfig::default(),
                precision_mode: PrecisionMode::Balanced,
                precision_filter: PrecisionFilterConfig::balanced(),
                #[cfg(feature = "cuda")]
                use_mega_fused: true,  // Enable mega-fused by default when GPU available
                #[cfg(feature = "cuda")]
                mega_fused_config: if config.pure_gpu_mode {
                    MegaFusedConfig::screening_pure()
                } else {
                    MegaFusedConfig::screening()
                },
            },
        })
    }

    /// Apply precision filtering to reduce false positives
    fn apply_precision_filter(&self, pockets: Vec<Pocket>) -> Vec<Pocket> {
        let (filtered, stats) = filter_pockets_for_precision(pockets, &self.config.precision_filter);

        if stats.total_removed() > 0 {
            log::info!(
                "Precision filter: {} -> {} pockets (removed: {} volume, {} druggability, {} burial, {} residues, {} limit)",
                stats.input_count,
                stats.output_count,
                stats.removed_by_volume,
                stats.removed_by_druggability,
                stats.removed_by_burial,
                stats.removed_by_residues,
                stats.removed_by_limit
            );
        }

        filtered
    }

    pub fn detect(&self, graph: &ProteinGraph) -> Result<Vec<Pocket>, LbsError> {
        #[cfg(feature = "cuda")]
        {
            // Try to detect without GPU context - fallback path
            self.detect_internal(graph, None)
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.detect_internal(graph)
        }
    }

    /// Detect pockets using GPU acceleration when available
    #[cfg(feature = "cuda")]
    pub fn detect_with_gpu(&self, graph: &ProteinGraph, gpu_ctx: Option<&Arc<GpuContext>>) -> Result<Vec<Pocket>, LbsError> {
        self.detect_internal(graph, gpu_ctx)
    }

    /// ULTRA-FAST PURE GPU DIRECT PATH: No graph construction, no CPU geometry
    /// Builds 5 flat arrays directly from ProteinStructure and runs mega-fused kernel.
    /// This is the fastest possible path for high-throughput screening.
    #[cfg(feature = "cuda")]
    pub fn detect_pure_gpu_direct(&self, structure: &ProteinStructure) -> Result<Vec<Pocket>, LbsError> {
        use prism_gpu::mega_fused::signals;
        use std::collections::HashMap;

        log::info!("ULTRA-FAST PURE GPU DIRECT MODE: No graph construction, no CPU geometry");

        let n_residues = structure.residues.len();
        if n_residues == 0 {
            return Ok(Vec::new());
        }

        // Build 5 flat arrays directly from structure (NO ProteinGraph)
        let atoms: Vec<f32> = structure.atoms.iter()
            .flat_map(|a| [a.coord[0] as f32, a.coord[1] as f32, a.coord[2] as f32])
            .collect();

        // Precompute residue→atom index mapping for O(1) lookups
        let mut residue_atom_map: HashMap<(i32, char), Vec<usize>> = HashMap::new();
        for (atom_idx, atom) in structure.atoms.iter().enumerate() {
            residue_atom_map
                .entry((atom.residue_seq, atom.chain_id))
                .or_insert_with(Vec::new)
                .push(atom_idx);
        }

        // CA indices
        let ca_indices: Vec<i32> = structure.residues.iter()
            .map(|res| {
                structure.atoms.iter().position(|a| {
                    a.residue_seq == res.seq_number
                        && a.chain_id == res.chain_id
                        && a.name == "CA"
                })
                .map(|i| i as i32)
                .unwrap_or(-1)
            })
            .collect();

        // PARSE RESIDUE TYPES from residue names (FIX FOR DEAD PHYSICS FEATURES!)
        let residue_types: Vec<i32> = structure.residues.iter()
            .map(|res| {
                // Parse 3-letter code to 1-letter, then to index (0-19)
                let aa_1letter = match res.name.as_str() {
                    "ALA" => 'A', "ARG" => 'R', "ASN" => 'N', "ASP" => 'D', "CYS" => 'C',
                    "GLN" => 'Q', "GLU" => 'E', "GLY" => 'G', "HIS" => 'H', "ILE" => 'I',
                    "LEU" => 'L', "LYS" => 'K', "MET" => 'M', "PHE" => 'F', "PRO" => 'P',
                    "SER" => 'S', "THR" => 'T', "TRP" => 'W', "TYR" => 'Y', "VAL" => 'V',
                    _ => 'A',  // Default to Alanine for unknown
                };
                // Convert to index: A=0, R=1, N=2, ..., V=19
                (match aa_1letter {
                    'A' => 0,  'R' => 1,  'N' => 2,  'D' => 3,  'C' => 4,
                    'Q' => 5,  'E' => 6,  'G' => 7,  'H' => 8,  'I' => 9,
                    'L' => 10, 'K' => 11, 'M' => 12, 'F' => 13, 'P' => 14,
                    'S' => 15, 'T' => 16, 'W' => 17, 'Y' => 18, 'V' => 19,
                    _ => 0,
                }) as i32
            })
            .collect();

        log::info!("Parsed {} residue types (will enable hydrophobicity features)", residue_types.len());

        // Conservation
        let conservation: Vec<f32> = structure.residues.iter()
            .map(|r| r.conservation_score as f32)
            .collect();

        // B-factor and burial per residue
        let mut bfactor: Vec<f32> = Vec::with_capacity(n_residues);
        let mut burial: Vec<f32> = Vec::with_capacity(n_residues);

        for res in &structure.residues {
            if let Some(res_atom_indices) = residue_atom_map.get(&(res.seq_number, res.chain_id)) {
                if res_atom_indices.is_empty() {
                    bfactor.push(0.5);
                    burial.push(0.5);
                } else {
                    let avg_bfactor: f64 = res_atom_indices.iter()
                        .map(|&i| structure.atoms[i].b_factor)
                        .sum::<f64>() / res_atom_indices.len() as f64;
                    bfactor.push((avg_bfactor / 100.0).clamp(0.0, 1.0) as f32);

                    let avg_sasa: f64 = res_atom_indices.iter()
                        .map(|&i| structure.atoms[i].sasa)
                        .sum::<f64>() / res_atom_indices.len() as f64;
                    burial.push((1.0 - (avg_sasa / 150.0).clamp(0.0, 1.0)) as f32);
                }
            } else {
                bfactor.push(0.5);
                burial.push(0.5);
            }
        }

        // Get mega-fused kernel from GlobalGpuContext
        let global_gpu = GlobalGpuContext::try_get()
            .map_err(|e| LbsError::Gpu(format!("Pure GPU direct requires GlobalGpuContext: {}", e)))?;
        let mut mega_fused = global_gpu.mega_fused_locked()
            .ok_or_else(|| LbsError::Gpu("Mega-fused kernel not loaded in GlobalGpuContext".to_string()))?;

        // Run kernel with residue types (enables hydrophobicity physics features!)
        let output = mega_fused.detect_pockets(
            &atoms,
            &ca_indices,
            &conservation,
            &bfactor,
            &burial,
            Some(&residue_types),  // NEW: Pass residue types for physics!
            &self.config.mega_fused_config,
        ).map_err(|e| LbsError::Gpu(format!("Mega-fused kernel failed: {}", e)))?;

        log::info!("GPU kernel complete: {} residues, {} combined features",
                   output.consensus_scores.len(), output.combined_features.len());

        // Convert output to Pockets
        let mut pocket_residues: HashMap<i32, Vec<usize>> = HashMap::new();
        for (res_idx, &pocket_id) in output.pocket_assignment.iter().enumerate() {
            if pocket_id >= 0 && output.consensus_scores[res_idx] > self.config.mega_fused_config.consensus_threshold {
                pocket_residues.entry(pocket_id).or_default().push(res_idx);
            }
        }

        let mut pockets: Vec<Pocket> = Vec::new();
        for (pocket_id, residue_indices) in pocket_residues {
            if residue_indices.is_empty() {
                continue;
            }

            let mut atom_indices: Vec<usize> = Vec::new();
            let mut centroid = [0.0f64, 0.0, 0.0];
            let mut total_hydro = 0.0;
            let mut total_depth = 0.0;
            let mut total_sasa = 0.0;
            let mut total_flex = 0.0;
            let mut total_cons = 0.0;
            let mut donors = 0usize;
            let mut acceptors = 0usize;

            for &res_idx in &residue_indices {
                let res = &structure.residues[res_idx];
                if let Some(res_atom_indices) = residue_atom_map.get(&(res.seq_number, res.chain_id)) {
                    for &atom_idx in res_atom_indices {
                        let atom = &structure.atoms[atom_idx];
                        atom_indices.push(atom_idx);
                        centroid[0] += atom.coord[0];
                        centroid[1] += atom.coord[1];
                        centroid[2] += atom.coord[2];
                        total_hydro += atom.hydrophobicity;
                        total_depth += atom.depth;
                        total_sasa += atom.sasa;
                        total_flex += atom.b_factor;
                        donors += usize::from(atom.is_hbond_donor());
                        acceptors += usize::from(atom.is_hbond_acceptor());
                    }
                }
                total_cons += structure.residues[res_idx].conservation_score;
            }

            if atom_indices.is_empty() {
                continue;
            }

            let count = atom_indices.len() as f64;
            let residue_count = residue_indices.len();
            centroid[0] /= count;
            centroid[1] /= count;
            centroid[2] /= count;

            // Simplified volume (bounding box for speed)
            let volume = bounding_box_volume(structure, &atom_indices);
            let enc = enclosure_ratio(structure, &atom_indices);

            let avg_consensus = residue_indices.iter()
                .map(|&i| output.consensus_scores[i])
                .sum::<f32>() / residue_indices.len() as f32;

            let avg_confidence = residue_indices.iter()
                .map(|&i| output.confidence[i] as f32)
                .sum::<f32>() / residue_indices.len() as f32;

            let avg_centrality = residue_indices.iter()
                .map(|&i| output.centrality[i])
                .sum::<f32>() / residue_indices.len() as f32;

            let drugg_total = (avg_consensus as f64 * 0.4 + avg_confidence as f64 / 2.0 * 0.3 + avg_centrality as f64 * 0.3).clamp(0.0, 1.0);
            let classification = if drugg_total >= 0.7 {
                crate::scoring::DrugabilityClass::HighlyDruggable
            } else if drugg_total >= 0.5 {
                crate::scoring::DrugabilityClass::Druggable
            } else if drugg_total >= 0.3 {
                crate::scoring::DrugabilityClass::DifficultTarget
            } else {
                crate::scoring::DrugabilityClass::Undruggable
            };

            let pocket = Pocket {
                atom_indices,
                residue_indices,
                centroid,
                volume,
                enclosure_ratio: enc,
                mean_hydrophobicity: total_hydro / count,
                mean_sasa: total_sasa / count,
                mean_depth: total_depth / count,
                mean_flexibility: total_flex / count,
                mean_conservation: total_cons / residue_count as f64,
                persistence_score: avg_centrality as f64,
                hbond_donors: donors,
                hbond_acceptors: acceptors,
                druggability_score: DruggabilityScore {
                    total: drugg_total,
                    classification,
                    components: crate::scoring::Components {
                        volume: volume / 1000.0,
                        hydro: total_hydro / count,
                        enclosure: enc,
                        depth: (total_depth / count).clamp(0.0, 1.0),
                        hbond: (donors + acceptors) as f64 / count.max(1.0),
                        flex: (total_flex / count / 100.0).clamp(0.0, 1.0),
                        cons: total_cons / residue_count as f64,
                        topo: avg_centrality as f64,
                    },
                },
                boundary_atoms: Vec::new(),
                mean_electrostatic: 0.0,
                gnn_embedding: Vec::new(),
                gnn_druggability: 0.0,
            };

            if pocket.volume >= 50.0 && pocket.atom_indices.len() >= 5 {
                pockets.push(pocket);
            }
        }

        pockets.sort_by(|a, b| b.druggability_score.total.partial_cmp(&a.druggability_score.total).unwrap_or(std::cmp::Ordering::Equal));
        pockets.truncate(self.config.max_pockets);

        let filtered = self.apply_precision_filter(pockets);
        Ok(filtered)
    }

    /// Extract raw 92-dim features (for ML training, viral escape, etc.)
    #[cfg(feature = "cuda")]
    pub fn extract_features_pure_gpu(&self, structure: &ProteinStructure) -> Result<Vec<f32>, LbsError> {
        use prism_gpu::mega_fused::signals;
        use std::collections::HashMap;

        log::info!("FEATURE EXTRACTION MODE: Extracting 92-dim features for {}", structure.title);

        let n_residues = structure.residues.len();
        if n_residues == 0 {
            return Ok(Vec::new());
        }

        // Build arrays (same as detect_pure_gpu_direct)
        let atoms: Vec<f32> = structure.atoms.iter()
            .flat_map(|a| [a.coord[0] as f32, a.coord[1] as f32, a.coord[2] as f32])
            .collect();

        let mut residue_atom_map: HashMap<(i32, char), Vec<usize>> = HashMap::new();
        for (atom_idx, atom) in structure.atoms.iter().enumerate() {
            residue_atom_map
                .entry((atom.residue_seq, atom.chain_id))
                .or_insert_with(Vec::new)
                .push(atom_idx);
        }

        let ca_indices: Vec<i32> = structure.residues.iter()
            .map(|res| {
                structure.atoms.iter().position(|a| {
                    a.residue_seq == res.seq_number
                        && a.chain_id == res.chain_id
                        && a.name == "CA"
                })
                .map(|i| i as i32)
                .unwrap_or(-1)
            })
            .collect();

        let conservation: Vec<f32> = structure.residues.iter()
            .map(|r| r.conservation_score as f32)
            .collect();

        let mut bfactor: Vec<f32> = Vec::with_capacity(n_residues);
        let mut burial: Vec<f32> = Vec::with_capacity(n_residues);

        for res in &structure.residues {
            if let Some(res_atom_indices) = residue_atom_map.get(&(res.seq_number, res.chain_id)) {
                if res_atom_indices.is_empty() {
                    bfactor.push(0.5);
                    burial.push(0.5);
                } else {
                    let avg_bfactor: f64 = res_atom_indices.iter()
                        .map(|&i| structure.atoms[i].b_factor)
                        .sum::<f64>() / res_atom_indices.len() as f64;
                    bfactor.push((avg_bfactor / 100.0).clamp(0.0, 1.0) as f32);

                    let avg_sasa: f64 = res_atom_indices.iter()
                        .map(|&i| structure.atoms[i].sasa)
                        .sum::<f64>() / res_atom_indices.len() as f64;
                    burial.push(1.0 - (avg_sasa / 200.0).clamp(0.0, 1.0) as f32);
                }
            } else {
                bfactor.push(0.5);
                burial.push(0.5);
            }
        }

        // PARSE RESIDUE TYPES (enable physics features!)
        let residue_types: Vec<i32> = structure.residues.iter()
            .map(|res| Self::parse_residue_type(&res.name))
            .collect();

        log::info!("Parsed {} residue types for physics features", residue_types.len());

        // Get GPU kernel
        let global_gpu = GlobalGpuContext::try_get()
            .map_err(|e| LbsError::Gpu(format!("Feature extraction requires GlobalGpuContext: {}", e)))?;
        let mut mega_fused = global_gpu.mega_fused_locked()
            .ok_or_else(|| LbsError::Gpu("Mega-fused kernel not loaded".to_string()))?;

        // Run kernel with residue types
        let output = mega_fused.detect_pockets(
            &atoms,
            &ca_indices,
            &conservation,
            &bfactor,
            &burial,
            Some(&residue_types),  // Enable physics features!
            &self.config.mega_fused_config,
        ).map_err(|e| LbsError::Gpu(format!("Mega-fused kernel failed: {}", e)))?;

        log::info!("Feature extraction complete: {} residues × 92 dims = {} features",
                   n_residues, output.combined_features.len());

        // Return raw combined_features vector [n_residues × 92]
        Ok(output.combined_features)
    }

    // Helper function to parse residue 3-letter code to index (0-19)
    fn parse_residue_type(res_name: &str) -> i32 {
        let aa = match res_name {
            "ALA" => 'A', "ARG" => 'R', "ASN" => 'N', "ASP" => 'D', "CYS" => 'C',
            "GLN" => 'Q', "GLU" => 'E', "GLY" => 'G', "HIS" => 'H', "ILE" => 'I',
            "LEU" => 'L', "LYS" => 'K', "MET" => 'M', "PHE" => 'F', "PRO" => 'P',
            "SER" => 'S', "THR" => 'T', "TRP" => 'W', "TYR" => 'Y', "VAL" => 'V',
            _ => 'A',
        };
        match aa {
            'A' => 0,  'R' => 1,  'N' => 2,  'D' => 3,  'C' => 4,
            'Q' => 5,  'E' => 6,  'G' => 7,  'H' => 8,  'I' => 9,
            'L' => 10, 'K' => 11, 'M' => 12, 'F' => 13, 'P' => 14,
            'S' => 15, 'T' => 16, 'W' => 17, 'Y' => 18, 'V' => 19,
            _ => 0,
        }
    }

    #[cfg(feature = "cuda")]
    fn detect_internal(&self, graph: &ProteinGraph, gpu_ctx: Option<&Arc<GpuContext>>) -> Result<Vec<Pocket>, LbsError> {
        // ULTRA-FAST PURE GPU DIRECT MODE — first check, no graph needed in caller for pure_gpu_mode
        if self.config.use_mega_fused && self.config.mega_fused_config.pure_gpu_mode {
            log::info!("ULTRA-FAST PURE GPU DIRECT MODE: No graph construction, no CPU geometry");
            return self.detect_pure_gpu_direct(&graph.structure_ref);
        }

        // Priority 0: Use mega-fused GPU kernel if enabled
        // Uses GlobalGpuContext singleton for ZERO per-structure PTX loading overhead
        // This provides ~100x speedup for batch processing (eliminating ~8s PTX load per structure)
        if self.config.use_mega_fused {
            // Try GlobalGpuContext first (preferred - zero PTX reload overhead)
            match GlobalGpuContext::try_get() {
                Ok(global_gpu) => {
                    if let Some(mut mega_fused) = global_gpu.mega_fused_locked() {
                        log::debug!("Using GlobalGpuContext mega-fused kernel (pre-loaded, zero PTX overhead)");

                        // Try mega-fused detection with pre-loaded kernel
                        match self.run_mega_fused_detection(graph, &mut mega_fused) {
                            Ok(pockets) => {
                                log::info!("Mega-fused detection found {} pockets (pre-filter)", pockets.len());
                                let filtered = self.apply_precision_filter(pockets);
                                return Ok(filtered);
                            }
                            Err(e) => {
                                log::warn!("Mega-fused kernel failed: {}. Falling back to other methods.", e);
                            }
                        }
                    } else {
                        log::debug!("GlobalGpuContext available but mega-fused not loaded. Trying legacy path.");
                    }
                }
                Err(e) => {
                    log::debug!("GlobalGpuContext not available: {}. Trying legacy per-call initialization.", e);
                }
            }

            // Legacy fallback: create MegaFusedGpu per-call (slow, for backwards compatibility)
            if let Some(ctx) = gpu_ctx {
                let ptx_dir = std::env::var("PRISM_PTX_DIR").unwrap_or_else(|_| "target/ptx".to_string());
                match MegaFusedGpu::new(ctx.device().clone(), std::path::Path::new(&ptx_dir)) {
                    Ok(mut mega_fused) => {
                        log::info!("Using legacy per-call mega-fused GPU kernel (consider using GlobalGpuContext)");

                        match self.run_mega_fused_detection(graph, &mut mega_fused) {
                            Ok(pockets) => {
                                log::info!("Mega-fused detection found {} pockets (pre-filter)", pockets.len());
                                let filtered = self.apply_precision_filter(pockets);
                                return Ok(filtered);
                            }
                            Err(e) => {
                                log::warn!("Mega-fused kernel failed: {}. Falling back to other methods.", e);
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("Failed to initialize mega-fused GPU kernel: {}. Falling back.", e);
                    }
                }
            } else {
                log::debug!("Mega-fused enabled but no GPU context available. Using fallback detection.");
            }
        }

        // Priority 1: Use fpocket if enabled and available (gold standard)
        if self.config.use_fpocket {
            if crate::pocket::fpocket_ffi::fpocket_available() {
                log::info!("Using fpocket for gold-standard pocket detection");

                // fpocket requires a PDB file path, check if we have it from graph
                if let Some(pdb_path) = graph.structure_ref.pdb_path.as_ref() {
                    match crate::pocket::fpocket_ffi::run_fpocket(pdb_path, &self.config.fpocket) {
                        Ok(pockets) => {
                            log::info!("fpocket detected {} pockets (pre-filter)", pockets.len());
                            // Apply precision filtering
                            let filtered = self.apply_precision_filter(pockets);
                            return Ok(filtered);
                        }
                        Err(e) => {
                            log::warn!("fpocket execution failed: {}. Falling back to internal detection.", e);
                        }
                    }
                } else {
                    log::warn!("fpocket enabled but no PDB file path available. Use ProteinStructure::from_pdb_file() to enable fpocket.");
                }
            } else {
                log::warn!("fpocket enabled but not found in PATH. Install fpocket or disable use_fpocket.");
            }
        }

        // Priority 2: Use Voronoi-based detection (RECOMMENDED - proper alpha sphere method)
        if self.config.use_voronoi_detection {
            log::info!("Using Voronoi-based pocket detection (alpha sphere method)");

            // Create VoronoiDetector with GPU if available
            let voronoi_detector = VoronoiDetector::new(self.config.voronoi_detector.clone());

            // Wire GPU acceleration if context is available
            let voronoi_detector = if let Some(ctx) = gpu_ctx {
                let ptx_dir = std::env::var("PRISM_PTX_DIR").unwrap_or_else(|_| "target/ptx".to_string());
                match LbsGpu::new(ctx.device().clone(), std::path::Path::new(&ptx_dir)) {
                    Ok(lbs_gpu) => {
                        log::info!("GPU acceleration enabled for pocket detection");
                        voronoi_detector.with_gpu(Arc::new(lbs_gpu))
                    }
                    Err(e) => {
                        log::warn!("Failed to initialize LbsGpu for pocket detection: {}. Using CPU.", e);
                        voronoi_detector
                    }
                }
            } else {
                voronoi_detector
            };

            let pockets = voronoi_detector.detect(graph);
            log::info!("Voronoi detection found {} pockets (pre-filter)", pockets.len());

            // Apply precision filtering to reduce false positives
            let filtered = self.apply_precision_filter(pockets);
            return Ok(filtered);
        }

        // Priority 3: Use grid-based alpha sphere cavity detection (legacy, less accurate)
        if self.config.use_cavity_detection {
            log::info!("Using grid-based alpha sphere cavity detection (legacy method)");
            let cavity_detector = CavityDetector::new(self.config.cavity_detector.clone());
            let pockets = cavity_detector.detect(graph);
            log::info!("Grid-based cavity detection found {} pockets (pre-filter)", pockets.len());

            // Apply precision filtering
            let filtered = self.apply_precision_filter(pockets);
            return Ok(filtered);
        }

        // Priority 4: Fallback to original belief propagation / graph coloring approach
        log::info!("Using belief propagation pocket detection (legacy mode)");
        let reservoir_phase = SurfaceReservoirPhase::new(self.config.reservoir.clone());
        let reservoir_output = reservoir_phase.execute(graph);

        let belief_phase = PocketBeliefPhase::new(self.config.beliefs.clone());
        let belief_output = belief_phase.execute(graph, &reservoir_output);

        let sampling_phase = PocketSamplingPhase::new(self.config.sampling.clone());
        let sampling_output = sampling_phase.execute(graph, &belief_output);

        let cavity_phase = CavityAnalysisPhase::new(self.config.cavity.clone());
        let cavity_output = cavity_phase.execute(graph);

        let topology_phase = TopologicalPocketPhase::new(self.config.topology.clone());
        let topology_output = topology_phase.execute(graph);

        let refinement_phase = PocketRefinementPhase::new(self.config.refinement.clone());
        let refinement_output = refinement_phase.execute(
            graph,
            &sampling_output.coloring,
            &cavity_output,
            &topology_output,
            &reservoir_output,
        );

        let pockets = self.assemble_pockets(
            graph,
            &sampling_output.coloring,
            &refinement_output.boundary_vertices,
        );

        let mut pockets_with_scores = Vec::new();
        for p in pockets {
            let mut pocket = p;
            pocket.druggability_score = DruggabilityScore::default();
            pockets_with_scores.push(pocket);
        }

        // Apply precision filtering
        let filtered = self.apply_precision_filter(pockets_with_scores);
        Ok(filtered)
    }

    /// Non-CUDA version of detect_internal
    #[cfg(not(feature = "cuda"))]
    fn detect_internal(&self, graph: &ProteinGraph) -> Result<Vec<Pocket>, LbsError> {
        // Priority 1: Use fpocket if enabled and available (gold standard)
        if self.config.use_fpocket {
            if crate::pocket::fpocket_ffi::fpocket_available() {
                log::info!("Using fpocket for gold-standard pocket detection");

                // fpocket requires a PDB file path, check if we have it from graph
                if let Some(pdb_path) = graph.structure_ref.pdb_path.as_ref() {
                    match crate::pocket::fpocket_ffi::run_fpocket(pdb_path, &self.config.fpocket) {
                        Ok(pockets) => {
                            log::info!("fpocket detected {} pockets (pre-filter)", pockets.len());
                            // Apply precision filtering
                            let filtered = self.apply_precision_filter(pockets);
                            return Ok(filtered);
                        }
                        Err(e) => {
                            log::warn!("fpocket execution failed: {}. Falling back to internal detection.", e);
                        }
                    }
                } else {
                    log::warn!("fpocket enabled but no PDB file path available. Use ProteinStructure::from_pdb_file() to enable fpocket.");
                }
            } else {
                log::warn!("fpocket enabled but not found in PATH. Install fpocket or disable use_fpocket.");
            }
        }

        // Priority 2: Use Voronoi-based detection (RECOMMENDED - proper alpha sphere method)
        if self.config.use_voronoi_detection {
            log::info!("Using Voronoi-based pocket detection (alpha sphere method)");
            let voronoi_detector = VoronoiDetector::new(self.config.voronoi_detector.clone());
            let pockets = voronoi_detector.detect(graph);
            log::info!("Voronoi detection found {} pockets (pre-filter)", pockets.len());

            // Apply precision filtering to reduce false positives
            let filtered = self.apply_precision_filter(pockets);
            return Ok(filtered);
        }

        // Priority 3: Use grid-based alpha sphere cavity detection (legacy, less accurate)
        if self.config.use_cavity_detection {
            log::info!("Using grid-based alpha sphere cavity detection (legacy method)");
            let cavity_detector = CavityDetector::new(self.config.cavity_detector.clone());
            let pockets = cavity_detector.detect(graph);
            log::info!("Grid-based cavity detection found {} pockets (pre-filter)", pockets.len());

            // Apply precision filtering
            let filtered = self.apply_precision_filter(pockets);
            return Ok(filtered);
        }

        // Priority 4: Fallback to original belief propagation / graph coloring approach
        log::info!("Using belief propagation pocket detection (legacy mode)");
        let reservoir_phase = SurfaceReservoirPhase::new(self.config.reservoir.clone());
        let reservoir_output = reservoir_phase.execute(graph);

        let belief_phase = PocketBeliefPhase::new(self.config.beliefs.clone());
        let belief_output = belief_phase.execute(graph, &reservoir_output);

        let sampling_phase = PocketSamplingPhase::new(self.config.sampling.clone());
        let sampling_output = sampling_phase.execute(graph, &belief_output);

        let cavity_phase = CavityAnalysisPhase::new(self.config.cavity.clone());
        let cavity_output = cavity_phase.execute(graph);

        let topology_phase = TopologicalPocketPhase::new(self.config.topology.clone());
        let topology_output = topology_phase.execute(graph);

        let refinement_phase = PocketRefinementPhase::new(self.config.refinement.clone());
        let refinement_output = refinement_phase.execute(
            graph,
            &sampling_output.coloring,
            &cavity_output,
            &topology_output,
            &reservoir_output,
        );

        let pockets = self.assemble_pockets(
            graph,
            &sampling_output.coloring,
            &refinement_output.boundary_vertices,
        );

        let mut pockets_with_scores = Vec::new();
        for p in pockets {
            let mut pocket = p;
            pocket.druggability_score = DruggabilityScore::default();
            pockets_with_scores.push(pocket);
        }

        // Apply precision filtering
        let filtered = self.apply_precision_filter(pockets_with_scores);
        Ok(filtered)
    }

    /// Run mega-fused GPU kernel for pocket detection
    /// Converts ProteinGraph data to kernel inputs, runs detection, and converts output to Pockets
    #[cfg(feature = "cuda")]
    fn run_mega_fused_detection(
        &self,
        graph: &ProteinGraph,
        mega_fused: &mut MegaFusedGpu,
    ) -> Result<Vec<Pocket>, LbsError> {
        use prism_gpu::mega_fused::signals;
        use std::collections::HashMap;

        let structure = &graph.structure_ref;
        let n_residues = structure.residues.len();

        if n_residues == 0 {
            return Ok(Vec::new());
        }

        // Extract all atom coordinates as flat array [x0, y0, z0, x1, y1, z1, ...]
        let atoms: Vec<f32> = structure.atoms.iter()
            .flat_map(|a| [a.coord[0] as f32, a.coord[1] as f32, a.coord[2] as f32])
            .collect();

        // OPTIMIZATION: Precompute residue→atom index mapping to avoid O(R×A) nested loops
        // This single O(A) pass enables O(1) lookups during pocket assembly
        let mut residue_atom_map: HashMap<(i32, char), Vec<usize>> = HashMap::new();
        for (atom_idx, atom) in structure.atoms.iter().enumerate() {
            residue_atom_map
                .entry((atom.residue_seq, atom.chain_id))
                .or_insert_with(Vec::new)
                .push(atom_idx);
        }

        // Find CA atom indices for each residue
        let ca_indices: Vec<i32> = structure.residues.iter()
            .map(|res| {
                structure.atoms.iter().position(|a| {
                    a.residue_seq == res.seq_number
                        && a.chain_id == res.chain_id
                        && a.name == "CA"
                })
                .map(|i| i as i32)
                .unwrap_or(-1) // Mark missing CAs as -1
            })
            .collect();

        // Extract per-residue features
        let conservation: Vec<f32> = structure.residues.iter()
            .map(|r| r.conservation_score as f32)
            .collect();

        // Calculate average B-factor per residue (normalized)
        let mut bfactor: Vec<f32> = Vec::with_capacity(n_residues);
        let mut burial: Vec<f32> = Vec::with_capacity(n_residues);

        for res in &structure.residues {
            let res_atoms: Vec<_> = structure.atoms.iter()
                .filter(|a| a.residue_seq == res.seq_number && a.chain_id == res.chain_id)
                .collect();

            if res_atoms.is_empty() {
                bfactor.push(0.5); // Default normalized B-factor
                burial.push(0.5); // Default burial
            } else {
                // Average B-factor (normalized to 0-1)
                let avg_bfactor = res_atoms.iter().map(|a| a.b_factor).sum::<f64>() / res_atoms.len() as f64;
                bfactor.push((avg_bfactor / 100.0).clamp(0.0, 1.0) as f32);

                // Average SASA-based burial (inverse SASA = burial)
                let avg_sasa = res_atoms.iter().map(|a| a.sasa).sum::<f64>() / res_atoms.len() as f64;
                burial.push((1.0 - (avg_sasa / 150.0).clamp(0.0, 1.0)) as f32);
            }
        }

        // Parse residue types (enable physics features)
        let residue_types: Vec<i32> = graph.structure_ref.residues.iter()
            .map(|res| Self::parse_residue_type(&res.name))
            .collect();

        // Run mega-fused kernel
        let output = mega_fused.detect_pockets(
            &atoms,
            &ca_indices,
            &conservation,
            &bfactor,
            &burial,
            Some(&residue_types),  // Enable physics!
            &self.config.mega_fused_config,
        ).map_err(|e| LbsError::Gpu(format!("Mega-fused kernel failed: {}", e)))?;

        // Convert kernel output to Pockets
        // Group residues by pocket assignment
        let mut pocket_residues: HashMap<i32, Vec<usize>> = HashMap::new();
        for (res_idx, &pocket_id) in output.pocket_assignment.iter().enumerate() {
            if pocket_id >= 0 && output.consensus_scores[res_idx] > self.config.mega_fused_config.consensus_threshold {
                pocket_residues.entry(pocket_id).or_default().push(res_idx);
            }
        }

        // Convert each pocket cluster to a Pocket struct
        let mut pockets: Vec<Pocket> = Vec::new();
        for (pocket_id, residue_indices) in pocket_residues {
            if residue_indices.is_empty() {
                continue;
            }

            // Collect atom indices for this pocket
            let mut atom_indices: Vec<usize> = Vec::new();
            let mut centroid = [0.0f64, 0.0, 0.0];
            let mut total_hydro = 0.0;
            let mut total_depth = 0.0;
            let mut total_sasa = 0.0;
            let mut total_flex = 0.0;
            let mut total_cons = 0.0;
            let mut donors = 0usize;
            let mut acceptors = 0usize;

            // OPTIMIZED: Use precomputed residue→atom map for O(1) lookup instead of O(A) scan
            for &res_idx in &residue_indices {
                let res = &structure.residues[res_idx];
                if let Some(res_atom_indices) = residue_atom_map.get(&(res.seq_number, res.chain_id)) {
                    for &atom_idx in res_atom_indices {
                        let atom = &structure.atoms[atom_idx];
                        atom_indices.push(atom_idx);
                        centroid[0] += atom.coord[0];
                        centroid[1] += atom.coord[1];
                        centroid[2] += atom.coord[2];
                        total_hydro += atom.hydrophobicity;
                        total_depth += atom.depth;
                        total_sasa += atom.sasa;
                        total_flex += atom.b_factor;
                        donors += usize::from(atom.is_hbond_donor());
                        acceptors += usize::from(atom.is_hbond_acceptor());
                    }
                }
                total_cons += structure.residues[res_idx].conservation_score;
            }

            if atom_indices.is_empty() {
                continue;
            }

            let count = atom_indices.len() as f64;
            let residue_count = residue_indices.len();
            centroid[0] /= count;
            centroid[1] /= count;
            centroid[2] /= count;

            // Calculate volume using configured method
            let volume = if self.config.geometry.use_voxel_volume {
                voxel_volume(
                    structure,
                    &atom_indices,
                    Some(self.config.geometry.voxel_resolution),
                    Some(self.config.geometry.probe_radius),
                )
            } else if self.config.geometry.use_alpha_shape_volume {
                alpha_shape_volume(
                    structure,
                    &atom_indices,
                    self.config.geometry.alpha_shape_resolution,
                    self.config.geometry.alpha_shape_shrink,
                )
            } else {
                bounding_box_volume(structure, &atom_indices)
            };

            // Calculate enclosure ratio (simplified - use geometry ratio)
            let enc = enclosure_ratio(structure, &atom_indices);

            // Calculate average consensus score for this pocket
            let avg_consensus = residue_indices.iter()
                .map(|&i| output.consensus_scores[i])
                .sum::<f32>() / residue_indices.len() as f32;

            // Calculate average confidence
            let avg_confidence = residue_indices.iter()
                .map(|&i| output.confidence[i] as f32)
                .sum::<f32>() / residue_indices.len() as f32;

            // Count signals
            let signal_count: i32 = residue_indices.iter()
                .map(|&i| signals::count(output.signal_mask[i]))
                .sum();

            // Average centrality
            let avg_centrality = residue_indices.iter()
                .map(|&i| output.centrality[i])
                .sum::<f32>() / residue_indices.len() as f32;

            // Compute druggability score
            let drugg_total = (avg_consensus as f64 * 0.4 + avg_confidence as f64 / 2.0 * 0.3 + avg_centrality as f64 * 0.3).clamp(0.0, 1.0);
            let classification = if drugg_total >= 0.7 {
                crate::scoring::DrugabilityClass::HighlyDruggable
            } else if drugg_total >= 0.5 {
                crate::scoring::DrugabilityClass::Druggable
            } else if drugg_total >= 0.3 {
                crate::scoring::DrugabilityClass::DifficultTarget
            } else {
                crate::scoring::DrugabilityClass::Undruggable
            };

            let pocket = Pocket {
                atom_indices,
                residue_indices,
                centroid,
                volume,
                enclosure_ratio: enc,
                mean_hydrophobicity: total_hydro / count,
                mean_sasa: total_sasa / count,
                mean_depth: total_depth / count,
                mean_flexibility: total_flex / count,
                mean_conservation: total_cons / residue_count as f64,
                persistence_score: avg_centrality as f64,
                hbond_donors: donors,
                hbond_acceptors: acceptors,
                druggability_score: DruggabilityScore {
                    total: drugg_total,
                    classification,
                    components: crate::scoring::Components {
                        volume: volume / 1000.0,
                        hydro: total_hydro / count,
                        enclosure: enc,
                        depth: (total_depth / count).clamp(0.0, 1.0),
                        hbond: (donors + acceptors) as f64 / count.max(1.0),
                        flex: (total_flex / count / 100.0).clamp(0.0, 1.0),
                        cons: total_cons / residue_count as f64,
                        topo: avg_centrality as f64,
                    },
                },
                boundary_atoms: Vec::new(),
                mean_electrostatic: 0.0,
                gnn_embedding: Vec::new(),
                gnn_druggability: 0.0,
            };

            // Filter by minimum volume
            if pocket.volume >= 50.0 && pocket.atom_indices.len() >= 5 {
                pockets.push(pocket);
            }
        }

        // Sort by druggability score descending
        pockets.sort_by(|a, b| b.druggability_score.total.partial_cmp(&a.druggability_score.total).unwrap_or(std::cmp::Ordering::Equal));

        // Limit to max_pockets
        pockets.truncate(self.config.max_pockets);

        Ok(pockets)
    }

    fn assemble_pockets(
        &self,
        graph: &ProteinGraph,
        coloring: &[usize],
        boundaries: &[usize],
    ) -> Vec<Pocket> {
        let boundary_vertices =
            if boundaries.is_empty() && self.config.geometry.use_boundary_enclosure {
                crate::pocket::boundary::boundary_vertices(coloring, graph)
            } else {
                boundaries.to_vec()
            };

        let mut pocket_atoms: Vec<Vec<usize>> = vec![Vec::new(); self.config.max_pockets];
        for (idx, &color) in coloring.iter().enumerate() {
            if color < pocket_atoms.len() {
                pocket_atoms[color].push(idx);
            }
        }

        pocket_atoms
            .into_iter()
            .enumerate()
            .filter(|(_, atoms)| !atoms.is_empty())
            .map(|(color, atoms)| {
                let mut centroid = [0.0, 0.0, 0.0];
                let mut total_hydro = 0.0;
                let mut total_depth = 0.0;
                let mut total_sasa = 0.0;
                let mut total_flex = 0.0;
                let mut total_cons = 0.0;
                let mut donors = 0usize;
                let mut acceptors = 0usize;

                for &v_idx in &atoms {
                    let atom_idx = graph.atom_indices[v_idx];
                    let atom = &graph.structure_ref.atoms[atom_idx];
                    centroid[0] += atom.coord[0];
                    centroid[1] += atom.coord[1];
                    centroid[2] += atom.coord[2];
                    total_hydro += atom.hydrophobicity;
                    total_depth += atom.depth;
                    total_sasa += atom.sasa;
                    donors += usize::from(atom.is_hbond_donor());
                    acceptors += usize::from(atom.is_hbond_acceptor());
                    total_flex += atom.b_factor;
                    if let Some(res_idx) = graph.structure_ref.residues.iter().position(|r| {
                        r.seq_number == atom.residue_seq && r.chain_id == atom.chain_id
                    }) {
                        let res = &graph.structure_ref.residues[res_idx];
                        total_cons += res.conservation_score;
                    }
                }
                let count = atoms.len() as f64;
                centroid[0] /= count;
                centroid[1] /= count;
                centroid[2] /= count;
                let atom_indices: Vec<usize> = atoms
                    .iter()
                    .map(|&v_idx| graph.atom_indices[v_idx])
                    .collect();
                let bbox = bounding_box_volume(&graph.structure_ref, &atom_indices);
                let voxel = if self.config.geometry.use_voxel_volume {
                    voxel_volume(
                        &graph.structure_ref,
                        &atom_indices,
                        Some(self.config.geometry.voxel_resolution),
                        Some(self.config.geometry.probe_radius),
                    )
                } else {
                    0.0
                };
                let alpha = if self.config.geometry.use_alpha_shape_volume {
                    alpha_shape_volume(
                        &graph.structure_ref,
                        &atom_indices,
                        self.config.geometry.alpha_shape_resolution,
                        self.config.geometry.alpha_shape_shrink,
                    )
                } else {
                    0.0
                };
                let hull = if self.config.geometry.use_convex_hull_volume {
                    convex_hull_volume(
                        &graph.structure_ref,
                        &atom_indices,
                        self.config.geometry.convex_hull_epsilon,
                    )
                } else {
                    0.0
                };

                let mut volume = if self.config.geometry.use_alpha_shape_volume && alpha > 0.0 {
                    alpha
                } else if self.config.geometry.use_convex_hull_volume && hull > 0.0 {
                    hull
                } else if self.config.geometry.use_voxel_volume && voxel > 0.0 {
                    voxel
                } else {
                    bbox
                };

                if volume <= 0.0 {
                    volume = voxel.max(bbox).max(hull);
                }
                volume = volume.max(bbox);

                if self.config.geometry.use_flood_fill_cavity {
                    let cavity_vol = crate::pocket::geometry::flood_fill_cavity_volume(
                        &graph.structure_ref,
                        &atom_indices,
                        self.config.geometry.cavity_resolution,
                        self.config.geometry.probe_radius,
                    );
                    volume = volume.max(cavity_vol);
                }

                let pocket_boundary_atoms: Vec<usize> = boundary_vertices
                    .iter()
                    .copied()
                    .filter(|&v| coloring.get(v) == Some(&color))
                    .filter_map(|v| graph.atom_indices.get(v).copied())
                    .collect();

                let enclosure = if self.config.geometry.use_neighbor_enclosure {
                    crate::pocket::geometry::neighbor_enclosure(
                        &graph.structure_ref,
                        &atom_indices,
                        self.config.geometry.boundary_cutoff,
                    )
                } else if self.config.geometry.use_boundary_enclosure {
                    boundary_enclosure(&atom_indices, &pocket_boundary_atoms)
                } else {
                    enclosure_ratio(&graph.structure_ref, &atom_indices)
                };

                // Collect unique PDB residue sequence numbers (RESSEQ)
                let residue_indices: Vec<usize> = {
                    let mut seen = std::collections::HashSet::new();
                    atoms
                        .iter()
                        .filter_map(|&v_idx| {
                            let atom_idx = graph.atom_indices[v_idx];
                            let atom = &graph.structure_ref.atoms[atom_idx];
                            // Use PDB RESSEQ (seq_number) directly, not internal index
                            let key = (atom.chain_id, atom.residue_seq);
                            if seen.insert(key) {
                                Some(atom.residue_seq as usize)
                            } else {
                                None
                            }
                        })
                        .collect()
                };

                Pocket {
                    atom_indices,
                    residue_indices,
                    centroid,
                    volume,
                    enclosure_ratio: enclosure,
                    mean_hydrophobicity: if count > 0.0 {
                        total_hydro / count
                    } else {
                        0.0
                    },
                    mean_sasa: if count > 0.0 { total_sasa / count } else { 0.0 },
                    mean_depth: if count > 0.0 {
                        total_depth / count
                    } else {
                        0.0
                    },
                    mean_flexibility: if count > 0.0 { total_flex / count } else { 0.0 },
                    mean_conservation: if count > 0.0 { total_cons / count } else { 0.0 },
                    persistence_score: 0.0,
                    hbond_donors: donors,
                    hbond_acceptors: acceptors,
                    druggability_score: DruggabilityScore::default(),
                    boundary_atoms: pocket_boundary_atoms,
                    // Enhanced features (computed later or default)
                    mean_electrostatic: 0.0,
                    gnn_embedding: Vec::new(),
                    gnn_druggability: 0.0,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{ProteinGraph, VertexFeatures};
    use crate::structure::ProteinStructure;
    use crate::LbsConfig;

    #[test]
    fn fills_boundary_atoms_from_detected_vertices() {
        let pdb = r#"ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N
ATOM      2  CA  ALA A   2       2.000   0.000   0.000  1.00 10.00           C
END
"#;
        let mut structure = ProteinStructure::from_pdb_str(pdb).expect("parse pdb");
        for atom in &mut structure.atoms {
            atom.is_surface = true;
            atom.sasa = 8.0;
            atom.depth = 1.0;
            atom.hydrophobicity = 1.0;
        }
        structure.refresh_residue_properties();

        let mut features = VertexFeatures::new(2);
        features.hydrophobicity = vec![1.0, 1.0];
        features.depth = vec![1.0, 1.0];

        let graph = ProteinGraph {
            atom_indices: vec![0, 1],
            adjacency: vec![vec![1], vec![0]],
            edge_weights: vec![vec![1.0], vec![1.0]],
            vertex_features: features,
            structure_ref: structure,
        };

        let coloring = vec![0, 1];
        let detector = PocketDetector::new(LbsConfig::default()).expect("detector");
        let pockets = detector.assemble_pockets(&graph, &coloring, &[]);

        assert_eq!(pockets.len(), 2);
        let mut boundary_atoms: Vec<Vec<usize>> =
            pockets.iter().map(|p| p.boundary_atoms.clone()).collect();
        boundary_atoms.sort_by_key(|v| v[0]);
        assert_eq!(boundary_atoms[0], vec![0]);
        assert_eq!(boundary_atoms[1], vec![1]);
    }
}

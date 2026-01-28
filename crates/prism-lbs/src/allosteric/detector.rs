//! Main Allosteric Detector
//!
//! Orchestrates the 4-stage allosteric detection pipeline:
//! 1. Structural Analysis (domain decomposition + hinge detection)
//! 2. Evolutionary Signal (MSA conservation)
//! 3. Network Analysis (residue network + centrality)
//! 4. Consensus + Backtrack (multi-signal integration + gap filling)
//!
//! This is the primary entry point for world-class allosteric site detection.

use crate::structure::Atom;
use crate::softspot::{SoftSpotDetector, CrypticCandidate};
use super::types::*;
use super::domain_decomposition::DomainDecomposer;
use super::hinge_detection::HingeDetector;
use super::msa_conservation::{ConservationAnalyzer, estimate_conservation_from_bfactors};
use super::residue_network::ResidueNetworkAnalyzer;
use super::allosteric_coupling::{AllostericCouplingAnalyzer, AllostericHotspot};
use super::consensus_engine::{HybridConsensusEngine, AllostericCandidateRegion};
use super::backtrack::BacktrackAnalyzer;
use std::collections::HashMap;
use std::path::Path;

/// World-class allosteric detector
pub struct AllostericDetector {
    /// Configuration
    pub config: AllostericDetectionConfig,
    /// Domain decomposer
    domain_decomposer: DomainDecomposer,
    /// Hinge detector
    hinge_detector: HingeDetector,
    /// Conservation analyzer
    conservation_analyzer: ConservationAnalyzer,
    /// Residue network analyzer
    network_analyzer: ResidueNetworkAnalyzer,
    /// Allosteric coupling analyzer
    coupling_analyzer: AllostericCouplingAnalyzer,
    /// Hybrid consensus engine
    consensus_engine: HybridConsensusEngine,
    /// Backtrack analyzer
    backtrack_analyzer: BacktrackAnalyzer,
}

impl AllostericDetector {
    pub fn new(config: AllostericDetectionConfig) -> Self {
        Self {
            domain_decomposer: DomainDecomposer::new(config.contact_cutoff, config.min_domain_size),
            hinge_detector: HingeDetector::default(),
            conservation_analyzer: ConservationAnalyzer::new(config.pseudocount),
            network_analyzer: ResidueNetworkAnalyzer::new(
                config.contact_cutoff,
                config.edge_weight_scheme,
            ),
            coupling_analyzer: AllostericCouplingAnalyzer::default(),
            consensus_engine: HybridConsensusEngine::new(config.min_confidence),
            backtrack_analyzer: BacktrackAnalyzer::new(config.gap_sensitivity),
            config,
        }
    }

    /// Run full allosteric detection pipeline
    pub fn detect(&self, atoms: &[Atom], structure_name: &str) -> AllostericDetectionOutput {
        log::info!("[ALLOSTERIC] Starting world-class detection for {}", structure_name);

        // ====================================================================
        // Stage 1: Structural Analysis
        // ====================================================================
        log::info!("[STAGE 1] Running structural analysis...");

        let domains = self.domain_decomposer.decompose(atoms);
        log::info!("  Found {} structural domains", domains.len());

        let interfaces = self.domain_decomposer.find_domain_interfaces(atoms, &domains);
        log::info!("  Found {} domain interfaces", interfaces.len());

        let hinges = self.hinge_detector.detect_with_domains(atoms, &domains);
        log::info!("  Found {} hinge regions", hinges.len());

        // ====================================================================
        // Stage 2: Evolutionary Signal
        // ====================================================================
        log::info!("[STAGE 2] Analyzing evolutionary signals...");

        let conservation = self.get_conservation(atoms);
        let n_conserved = conservation.values().filter(|&&c| c > 0.7).count();
        log::info!("  {} highly conserved residues (>{:.0}%)", n_conserved, 70.0);

        // ====================================================================
        // Stage 3: Network Analysis
        // ====================================================================
        log::info!("[STAGE 3] Building residue network...");

        let network = self.network_analyzer.build_network(atoms);
        log::info!("  Network: {} nodes, building centrality...", network.size);

        let centrality = self.coupling_analyzer.calculate_betweenness_centrality(&network);
        let hotspots = self.coupling_analyzer.find_allosteric_hotspots(&network, 0.3);
        log::info!("  Found {} allosteric hotspots", hotspots.len());

        // Find allosteric coupling to active site (if provided)
        let allosteric_candidates = if let Some(ref active_site) = self.config.active_site_residues {
            self.find_allosteric_candidates(atoms, &network, active_site, &hotspots)
        } else {
            // Auto-detect potential active site (highest conservation cluster)
            let auto_active = self.auto_detect_active_site(atoms, &conservation);
            if !auto_active.is_empty() {
                log::info!("  Auto-detected active site: {} residues", auto_active.len());
                self.find_allosteric_candidates(atoms, &network, &auto_active, &hotspots)
            } else {
                Vec::new()
            }
        };

        // ====================================================================
        // Stage 4A: Initial Consensus
        // ====================================================================
        log::info!("[STAGE 4A] Building initial consensus...");

        // Run geometric detection
        let geometric_pockets = self.run_geometric_detection(atoms);
        log::info!("  Geometric: {} pockets", geometric_pockets.len());

        // Run cryptic/softspot detection
        let cryptic_candidates = self.run_cryptic_detection(atoms);
        log::info!("  Cryptic: {} candidates", cryptic_candidates.len());

        // Build initial consensus
        let mut pockets = self.consensus_engine.build_consensus(
            atoms,
            &geometric_pockets,
            &cryptic_candidates,
            &allosteric_candidates,
            &interfaces,
            &conservation,
            &centrality,
        );

        log::info!("  Initial consensus: {} pockets", pockets.len());

        // ====================================================================
        // Stage 4B: Backtrack Analysis
        // ====================================================================
        let mut gaps_found = 0;
        let mut gaps_filled = 0;

        if self.config.enable_backtrack {
            log::info!("[STAGE 4B] Running backtrack gap analysis...");

            // Get B-factors for gap detection
            let bfactors = self.get_residue_bfactors(atoms);

            // Detect gaps
            let gaps = self.backtrack_analyzer.detect_coverage_gaps(
                atoms,
                &pockets,
                &conservation,
                &centrality,
                &interfaces,
                &bfactors,
            );

            gaps_found = gaps.len();
            log::info!("  Found {} coverage gaps", gaps_found);

            // Fill gaps
            if !gaps.is_empty() {
                let additional = self.backtrack_analyzer.fill_gaps(
                    atoms,
                    &gaps,
                    &conservation,
                    &centrality,
                );

                gaps_filled = additional.len();
                log::info!("  Filled {} gaps with additional pockets", gaps_filled);

                // Merge additional pockets
                for mut pocket in additional {
                    pocket.id = pockets.len() + 1;
                    pockets.push(pocket);
                }
            }
        }

        // ====================================================================
        // Generate Output
        // ====================================================================
        let coverage = self.backtrack_analyzer.generate_coverage_report(
            atoms,
            &pockets,
            &conservation,
            &interfaces,
        );

        let summary = self.generate_summary(&pockets, &domains, &hinges, &interfaces);

        log::info!(
            "[COMPLETE] {} pockets detected ({} high, {} medium, {} low confidence)",
            pockets.len(),
            summary.by_confidence.get("high").unwrap_or(&0),
            summary.by_confidence.get("medium").unwrap_or(&0),
            summary.by_confidence.get("low").unwrap_or(&0),
        );

        AllostericDetectionOutput {
            structure: structure_name.to_string(),
            analysis_metadata: AnalysisMetadata {
                prism_version: "2.0.0".to_string(),
                modules_used: vec![
                    "geometric".into(),
                    "softspot".into(),
                    "allosteric".into(),
                    "interface".into(),
                    "conservation".into(),
                ],
                msa_source: self.config.msa_path.as_ref().map(|p| p.display().to_string()),
                backtrack_enabled: self.config.enable_backtrack,
                gaps_found,
                gaps_filled,
                gpu_accelerated: self.config.use_gpu,
            },
            pockets,
            coverage_analysis: coverage,
            summary,
        }
    }

    fn get_conservation(&self, atoms: &[Atom]) -> HashMap<i32, f64> {
        if let Some(ref msa_path) = self.config.msa_path {
            // Parse MSA and calculate conservation
            match self.conservation_analyzer.parse_msa(msa_path) {
                Ok(msa) => {
                    let scores = self.conservation_analyzer.calculate_conservation(&msa);
                    let mapping = self.conservation_analyzer.generate_sequence_mapping(&msa, atoms);
                    self.conservation_analyzer.map_to_structure(&scores, &mapping)
                }
                Err(e) => {
                    log::warn!("Failed to parse MSA: {}, using B-factor proxy", e);
                    estimate_conservation_from_bfactors(atoms)
                }
            }
        } else {
            // Use B-factor as conservation proxy
            estimate_conservation_from_bfactors(atoms)
        }
    }

    fn get_residue_bfactors(&self, atoms: &[Atom]) -> HashMap<i32, f64> {
        let mut bfactors: HashMap<i32, (f64, usize)> = HashMap::new();

        for atom in atoms {
            if atom.name.trim() == "CA" {
                let entry = bfactors.entry(atom.residue_seq).or_insert((0.0, 0));
                entry.0 += atom.b_factor;
                entry.1 += 1;
            }
        }

        bfactors
            .into_iter()
            .filter_map(|(res, (sum, count))| {
                if count > 0 {
                    Some((res, sum / count as f64))
                } else {
                    None
                }
            })
            .collect()
    }

    fn auto_detect_active_site(
        &self,
        atoms: &[Atom],
        conservation: &HashMap<i32, f64>,
    ) -> Vec<i32> {
        // Find cluster of most conserved residues
        let highly_conserved: Vec<i32> = conservation
            .iter()
            .filter(|(_, &score)| score > 0.8)
            .map(|(&res, _)| res)
            .collect();

        if highly_conserved.len() < 3 {
            return Vec::new();
        }

        // Cluster and take largest
        let clusters = super::msa_conservation::spatial_cluster(atoms, &highly_conserved, 10.0);

        clusters
            .into_iter()
            .max_by_key(|c| c.len())
            .unwrap_or_default()
    }

    fn find_allosteric_candidates(
        &self,
        atoms: &[Atom],
        network: &ResidueNetwork,
        active_site: &[i32],
        hotspots: &[AllostericHotspot],
    ) -> Vec<AllostericCandidateRegion> {
        let mut candidates = Vec::new();

        // Each hotspot cluster is a potential allosteric site
        for hotspot in hotspots {
            // Skip if too close to active site
            let is_active_site = active_site.contains(&hotspot.residue);
            if is_active_site {
                continue;
            }

            // Calculate coupling to active site
            let coupling = self.network_analyzer.calculate_allosteric_coupling(
                network,
                &[hotspot.residue],
                active_site,
            );

            if coupling.coupling_strength > 0.2 {
                // Get Cα coordinate
                let coord = atoms
                    .iter()
                    .find(|a| a.name.trim() == "CA" && a.residue_seq == hotspot.residue)
                    .map(|a| a.coord)
                    .unwrap_or([0.0, 0.0, 0.0]);

                // Find nearby residues to form a pocket
                let nearby: Vec<i32> = atoms
                    .iter()
                    .filter(|a| {
                        if a.name.trim() != "CA" {
                            return false;
                        }
                        let dist_sq = (a.coord[0] - coord[0]).powi(2)
                            + (a.coord[1] - coord[1]).powi(2)
                            + (a.coord[2] - coord[2]).powi(2);
                        dist_sq < 64.0 // 8 Å radius
                    })
                    .map(|a| a.residue_seq)
                    .collect();

                if nearby.len() >= 4 {
                    let centroid = super::domain_decomposition::calculate_residue_centroid(atoms, &nearby);

                    // Distance to active site centroid
                    let active_centroid =
                        super::domain_decomposition::calculate_residue_centroid(atoms, active_site);
                    let distance_to_active = ((centroid[0] - active_centroid[0]).powi(2)
                        + (centroid[1] - active_centroid[1]).powi(2)
                        + (centroid[2] - active_centroid[2]).powi(2))
                    .sqrt();

                    candidates.push(AllostericCandidateRegion {
                        residues: nearby,
                        centroid,
                        coupling_score: coupling.coupling_strength,
                        path_length: coupling.shortest_path_length,
                        distance_to_active,
                        centrality: hotspot.betweenness_centrality,
                    });
                }
            }
        }

        // Sort by coupling score
        candidates.sort_by(|a, b| b.coupling_score.partial_cmp(&a.coupling_score).unwrap());
        candidates.truncate(5); // Top 5 candidates

        candidates
    }

    fn run_geometric_detection(&self, _atoms: &[Atom]) -> Vec<crate::pocket::Pocket> {
        // Geometric pocket detection requires a ProteinGraph which is built at a higher level.
        // When using the AllostericDetector standalone, geometric pockets should be passed in
        // via build_consensus. Return empty here - the main pipeline will provide geometric pockets.
        Vec::new()
    }

    fn run_cryptic_detection(&self, atoms: &[Atom]) -> Vec<CrypticCandidate> {
        SoftSpotDetector::default().detect(atoms)
    }

    fn generate_summary(
        &self,
        pockets: &[AllostericPocket],
        domains: &[Domain],
        hinges: &[HingeRegion],
        interfaces: &[DomainInterface],
    ) -> AllostericSummary {
        let mut by_type: HashMap<String, usize> = HashMap::new();
        let mut by_confidence: HashMap<String, usize> = HashMap::new();
        let mut from_backtrack = 0;

        for pocket in pockets {
            *by_type
                .entry(pocket.detection_type.as_str().to_string())
                .or_insert(0) += 1;
            *by_confidence
                .entry(pocket.confidence.level.as_str().to_string())
                .or_insert(0) += 1;

            if pocket.from_backtrack {
                from_backtrack += 1;
            }
        }

        AllostericSummary {
            total_pockets: pockets.len(),
            by_type,
            by_confidence,
            from_backtrack,
            domains_detected: domains.len(),
            hinges_detected: hinges.len(),
            interfaces_detected: interfaces.len(),
        }
    }
}

impl Default for AllostericDetector {
    fn default() -> Self {
        Self::new(AllostericDetectionConfig::default())
    }
}

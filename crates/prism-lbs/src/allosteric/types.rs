//! Core types for allosteric detection module
//!
//! Defines all data structures for the 4-stage allosteric detection pipeline.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Stage 1: Structural Analysis Types
// ============================================================================

/// A structural domain identified through spectral clustering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Domain {
    /// Unique domain identifier
    pub id: usize,
    /// Residue sequence numbers in this domain
    pub residues: Vec<i32>,
    /// Domain centroid position [x, y, z]
    pub centroid: [f64; 3],
    /// Radius of gyration (compactness measure)
    pub radius_of_gyration: f64,
    /// Number of internal contacts
    pub internal_contacts: usize,
    /// Secondary structure composition
    pub ss_composition: SecondaryStructureComposition,
}

/// Secondary structure composition percentages
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SecondaryStructureComposition {
    pub helix_fraction: f64,
    pub strand_fraction: f64,
    pub coil_fraction: f64,
}

/// Interface between two domains - potential allosteric communication point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainInterface {
    /// First domain ID
    pub domain_a_id: usize,
    /// Second domain ID
    pub domain_b_id: usize,
    /// Residues at the interface (from both domains)
    pub residues: Vec<i32>,
    /// Interface centroid
    pub centroid: [f64; 3],
    /// Buried surface area at interface (Å²)
    pub buried_sasa: f64,
    /// Shape complementarity score (0-1)
    pub shape_complementarity: f64,
    /// Number of hydrogen bonds across interface
    pub hydrogen_bonds: usize,
    /// Number of salt bridges across interface
    pub salt_bridges: usize,
}

/// A hinge region that enables domain motion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HingeRegion {
    /// Central residue of hinge
    pub center_residue: i32,
    /// All residues in hinge region
    pub residues: Vec<i32>,
    /// B-factor gradient magnitude
    pub gradient_magnitude: f64,
    /// Secondary structure type (typically coil/loop)
    pub secondary_structure: SecondaryStructure,
    /// Domains connected by this hinge
    pub connected_domains: Vec<usize>,
    /// Predicted flexibility score (0-1)
    pub flexibility_score: f64,
}

/// Secondary structure classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecondaryStructure {
    Helix,
    Strand,
    Coil,
    Turn,
    Unknown,
}

impl Default for SecondaryStructure {
    fn default() -> Self {
        SecondaryStructure::Unknown
    }
}

// ============================================================================
// Stage 2: Evolutionary Signal Types
// ============================================================================

/// Multiple Sequence Alignment data
#[derive(Debug, Clone)]
pub struct MSA {
    /// Aligned sequences
    pub sequences: Vec<String>,
    /// Sequence identifiers (e.g., UniProt IDs)
    pub sequence_ids: Vec<String>,
    /// Length of alignment (including gaps)
    pub alignment_length: usize,
    /// Reference sequence index (usually 0)
    pub reference_index: usize,
}

impl MSA {
    pub fn new(sequences: Vec<String>, sequence_ids: Vec<String>) -> Self {
        let alignment_length = sequences.first().map(|s| s.len()).unwrap_or(0);
        Self {
            sequences,
            sequence_ids,
            alignment_length,
            reference_index: 0,
        }
    }

    pub fn num_sequences(&self) -> usize {
        self.sequences.len()
    }
}

/// Per-position conservation score from MSA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationScore {
    /// Alignment position (0-indexed)
    pub position: usize,
    /// Conservation score (0 = variable, 1 = fully conserved)
    pub conservation: f64,
    /// Shannon entropy at this position
    pub entropy: f64,
    /// Fraction of gaps at this position
    pub gap_fraction: f64,
    /// Most common amino acid at this position
    pub dominant_aa: char,
    /// Frequency of dominant amino acid
    pub dominant_frequency: f64,
}

/// A cluster of spatially proximal conserved residues
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservedCluster {
    /// Residue sequence numbers in cluster
    pub residues: Vec<i32>,
    /// Mean conservation score
    pub mean_conservation: f64,
    /// Cluster centroid
    pub centroid: [f64; 3],
    /// Functional annotation if known
    pub annotation: Option<String>,
}

/// Mapping between alignment positions and structure residues
#[derive(Debug, Clone)]
pub struct SequenceMapping {
    /// Maps alignment position -> structure residue number
    pub alignment_to_structure: HashMap<usize, i32>,
    /// Maps structure residue number -> alignment position
    pub structure_to_alignment: HashMap<i32, usize>,
}

impl SequenceMapping {
    pub fn new() -> Self {
        Self {
            alignment_to_structure: HashMap::new(),
            structure_to_alignment: HashMap::new(),
        }
    }

    pub fn add_mapping(&mut self, alignment_pos: usize, residue_seq: i32) {
        self.alignment_to_structure.insert(alignment_pos, residue_seq);
        self.structure_to_alignment.insert(residue_seq, alignment_pos);
    }

    pub fn alignment_to_structure(&self, pos: usize) -> Option<i32> {
        self.alignment_to_structure.get(&pos).copied()
    }

    pub fn structure_to_alignment(&self, res: i32) -> Option<usize> {
        self.structure_to_alignment.get(&res).copied()
    }
}

impl Default for SequenceMapping {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Stage 3: Network Analysis Types
// ============================================================================

/// Residue interaction network as adjacency matrix
#[derive(Debug, Clone)]
pub struct ResidueNetwork {
    /// Adjacency matrix (weighted by contact strength)
    pub adjacency: Vec<f64>,
    /// Matrix dimension
    pub size: usize,
    /// Maps residue sequence number -> matrix index
    pub residue_to_idx: HashMap<i32, usize>,
    /// Maps matrix index -> residue sequence number
    pub idx_to_residue: Vec<i32>,
}

impl ResidueNetwork {
    pub fn new(size: usize) -> Self {
        Self {
            adjacency: vec![0.0; size * size],
            size,
            residue_to_idx: HashMap::new(),
            idx_to_residue: Vec::with_capacity(size),
        }
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.adjacency[i * self.size + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.adjacency[i * self.size + j] = value;
    }
}

/// Edge weight scheme for residue interaction network
#[derive(Debug, Clone, Copy)]
pub enum EdgeWeightScheme {
    /// Binary: 1 if contact, 0 otherwise
    Binary,
    /// Distance-based with Gaussian decay
    DistanceBased { sigma: f64 },
    /// Inverse distance
    InverseDistance,
}

impl Default for EdgeWeightScheme {
    fn default() -> Self {
        EdgeWeightScheme::DistanceBased { sigma: 6.0 }
    }
}

/// Allosteric coupling between two site regions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllostericCoupling {
    /// Overall coupling strength (0-1)
    pub coupling_strength: f64,
    /// Shortest path length in network
    pub shortest_path_length: f64,
    /// Residues along shortest communication path
    pub path_residues: Vec<i32>,
    /// Allosteric site residues
    pub allosteric_residues: Vec<i32>,
    /// Active/orthosteric site residues
    pub active_site_residues: Vec<i32>,
    /// Estimated signal attenuation along path
    pub signal_attenuation: f64,
}

/// Communication pathway between sites
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationPathway {
    /// Start residue
    pub source: i32,
    /// End residue
    pub target: i32,
    /// Residues along path (in order)
    pub path: Vec<i32>,
    /// Path length (sum of edge weights)
    pub length: f64,
    /// Bottleneck residue (highest centrality on path)
    pub bottleneck: Option<i32>,
}

// ============================================================================
// Stage 4: Consensus and Backtrack Types
// ============================================================================

/// Coverage gap detected by backtrack analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageGap {
    /// Type of gap detected
    pub gap_type: GapType,
    /// Residues in the gap region
    pub residues: Vec<i32>,
    /// Which detector module should re-analyze
    pub suggested_module: DetectorModule,
    /// Priority for re-analysis (0-1)
    pub priority: f64,
    /// Reason for gap detection
    pub reason: String,
}

/// Types of coverage gaps
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GapType {
    /// High conservation, no pocket detected
    ConservedUncovered,
    /// Domain boundary region, no pocket
    InterfaceUncovered,
    /// High B-factor region, not in softspot results
    FlexibleUncovered,
    /// Grid-based scan found cavity, alpha-sphere missed
    CavityMissed,
    /// High betweenness centrality, no pocket
    CommunicationHub,
    /// Near known active site, unexplored
    ActiveSiteAdjacent,
}

/// Detector modules available for re-analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectorModule {
    Geometric,
    SoftSpot,
    Allosteric,
    Interface,
}

/// Multi-module evidence for a detected pocket
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModuleEvidence {
    /// Evidence from geometric detection
    pub geometric: Option<GeometricEvidence>,
    /// Evidence from flexibility/cryptic analysis
    pub flexibility: Option<FlexibilityEvidence>,
    /// Evidence from conservation analysis
    pub conservation: Option<ConservationEvidence>,
    /// Evidence from allosteric coupling analysis
    pub allosteric_coupling: Option<AllostericCouplingEvidence>,
    /// Evidence from interface analysis
    pub interface: Option<InterfaceEvidence>,
    /// Which modules contributed to detection
    pub detected_by: Vec<String>,
}

impl Default for MultiModuleEvidence {
    fn default() -> Self {
        Self {
            geometric: None,
            flexibility: None,
            conservation: None,
            allosteric_coupling: None,
            interface: None,
            detected_by: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometricEvidence {
    pub volume: f64,
    pub depth: f64,
    pub druggability: f64,
    pub enclosure: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlexibilityEvidence {
    pub score: f64,
    pub mean_bfactor: f64,
    pub packing_deficit: f64,
    pub nma_mobility: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConservationEvidence {
    pub mean_score: f64,
    pub n_conserved_residues: usize,
    pub entropy_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllostericCouplingEvidence {
    pub coupling_strength: f64,
    pub shortest_path_length: f64,
    pub distance_to_active: f64,
    pub betweenness_centrality: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceEvidence {
    pub buried_sasa: f64,
    pub shape_complementarity: f64,
    pub n_interface_contacts: usize,
}

/// Confidence assessment with rationale
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceAssessment {
    /// Confidence level
    pub level: Confidence,
    /// Numerical confidence score (0-1)
    pub score: f64,
    /// Human-readable rationale
    pub rationale: String,
    /// Supporting evidence signals
    pub supporting_signals: Vec<String>,
    /// Concerning/weakening signals
    pub concerning_signals: Vec<String>,
}

/// Confidence level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Confidence {
    High,
    Medium,
    Low,
}

impl Confidence {
    pub fn as_str(&self) -> &'static str {
        match self {
            Confidence::High => "high",
            Confidence::Medium => "medium",
            Confidence::Low => "low",
        }
    }
}

/// Unified pocket with full evidence trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllostericPocket {
    /// Unique pocket ID
    pub id: usize,
    /// Residue sequence numbers
    pub residue_indices: Vec<i32>,
    /// Pocket centroid [x, y, z]
    pub centroid: [f64; 3],
    /// Estimated volume (Å³)
    pub volume: f64,
    /// Druggability score (0-1)
    pub druggability: f64,
    /// Detection type classification
    pub detection_type: AllostericDetectionType,
    /// Confidence assessment
    pub confidence: ConfidenceAssessment,
    /// Evidence from each module
    pub evidence: MultiModuleEvidence,
    /// Was this found via backtrack analysis?
    pub from_backtrack: bool,
    /// Gap type if from backtrack
    pub gap_origin: Option<GapType>,
    /// Functional annotation
    pub annotation: Option<String>,
}

/// Detection type for allosteric pockets
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AllostericDetectionType {
    /// Pure geometric detection
    Geometric,
    /// Cryptic/flexibility-based detection
    Cryptic,
    /// Allosteric coupling-based detection
    Allosteric,
    /// Interface-based detection
    Interface,
    /// Multi-module consensus
    Consensus,
}

impl AllostericDetectionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            AllostericDetectionType::Geometric => "geometric",
            AllostericDetectionType::Cryptic => "cryptic",
            AllostericDetectionType::Allosteric => "allosteric",
            AllostericDetectionType::Interface => "interface",
            AllostericDetectionType::Consensus => "consensus",
        }
    }
}

// ============================================================================
// Configuration Types
// ============================================================================

/// Configuration for allosteric detection
#[derive(Debug, Clone)]
pub struct AllostericDetectionConfig {
    // Stage 1: Structural
    /// Contact distance cutoff for domain decomposition (Å)
    pub contact_cutoff: f64,
    /// Minimum domain size (residues)
    pub min_domain_size: usize,
    /// Number of eigenvectors for spectral clustering
    pub n_eigenvectors: usize,

    // Stage 2: Evolutionary
    /// Path to MSA file (optional)
    pub msa_path: Option<std::path::PathBuf>,
    /// Conservation threshold for "conserved" residues
    pub conservation_threshold: f64,
    /// Pseudocount for frequency calculation
    pub pseudocount: f64,

    // Stage 3: Network
    /// Edge weight scheme for residue network
    pub edge_weight_scheme: EdgeWeightScheme,
    /// Known active site residues (for coupling analysis)
    pub active_site_residues: Option<Vec<i32>>,

    // Stage 4: Consensus
    /// Enable backtrack gap analysis
    pub enable_backtrack: bool,
    /// Minimum confidence score to report
    pub min_confidence: f64,
    /// Gap detection sensitivity
    pub gap_sensitivity: f64,

    // GPU settings
    /// Enable GPU acceleration
    pub use_gpu: bool,
    /// GPU device index
    pub gpu_device: usize,
}

impl Default for AllostericDetectionConfig {
    fn default() -> Self {
        Self {
            // Stage 1
            contact_cutoff: 10.0,
            min_domain_size: 30,
            n_eigenvectors: 5,
            // Stage 2
            msa_path: None,
            conservation_threshold: 0.7,
            pseudocount: 1.0,
            // Stage 3
            edge_weight_scheme: EdgeWeightScheme::default(),
            active_site_residues: None,
            // Stage 4
            enable_backtrack: true,
            min_confidence: 0.3,
            gap_sensitivity: 0.5,
            // GPU
            use_gpu: true,
            gpu_device: 0,
        }
    }
}

/// Output from allosteric detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllostericDetectionOutput {
    /// Structure identifier
    pub structure: String,
    /// Analysis metadata
    pub analysis_metadata: AnalysisMetadata,
    /// Detected pockets
    pub pockets: Vec<AllostericPocket>,
    /// Coverage analysis results
    pub coverage_analysis: CoverageAnalysis,
    /// Summary statistics
    pub summary: AllostericSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisMetadata {
    pub prism_version: String,
    pub modules_used: Vec<String>,
    pub msa_source: Option<String>,
    pub backtrack_enabled: bool,
    pub gaps_found: usize,
    pub gaps_filled: usize,
    pub gpu_accelerated: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageAnalysis {
    pub total_residues: usize,
    pub residues_in_pockets: usize,
    pub conserved_residues_covered_pct: f64,
    pub interface_residues_covered_pct: f64,
    pub gaps_remaining: Vec<CoverageGap>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AllostericSummary {
    pub total_pockets: usize,
    pub by_type: HashMap<String, usize>,
    pub by_confidence: HashMap<String, usize>,
    pub from_backtrack: usize,
    pub domains_detected: usize,
    pub hinges_detected: usize,
    pub interfaces_detected: usize,
}

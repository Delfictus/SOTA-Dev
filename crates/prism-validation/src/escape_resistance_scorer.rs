//! Escape Resistance Scorer for Cryptic Binding Sites
//!
//! This module scores cryptic binding sites based on their predicted resistance
//! to viral escape mutations. Sites targeting highly conserved, structurally
//! constrained regions are more valuable as drug targets.
//!
//! # Key Concepts
//!
//! **Escape Resistance** = How difficult is it for a pathogen to mutate
//! away from a binding site without losing fitness?
//!
//! Components:
//! - **Conservation**: Residues under strong purifying selection
//! - **Structural Constraint**: Residues critical for protein folding/stability
//! - **Contact Criticality**: Residues with many important interactions
//! - **Functional Importance**: Residues near active sites/binding pockets
//!
//! # Integration with PRISM-VE
//!
//! This scorer can optionally integrate with the PRISM-VE immunity model
//! for viral proteins, providing epitope-aware escape resistance scoring.
//!
//! # Usage
//!
//! ```rust,ignore
//! use prism_validation::escape_resistance_scorer::{
//!     EscapeResistanceScorer, EscapeResistanceScore
//! };
//!
//! let scorer = EscapeResistanceScorer::new();
//! let scores = scorer.score_residues(&conservation, &contacts, &structure_info);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Escape resistance score for a single residue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscapeResistanceScore {
    /// Conservation score (0-1): From MSA entropy
    /// Higher = more conserved = more resistant to escape
    pub conservation: f64,

    /// Structural constraint score (0-1): ΔΔG proxy from contact analysis
    /// Higher = more structurally important = harder to mutate
    pub structural_constraint: f64,

    /// Contact criticality (0-1): H-bonds + salt bridges + buried contacts
    /// Higher = more critical interactions = harder to replace
    pub contact_criticality: f64,

    /// Burial depth (0-1): How deeply buried is the residue
    /// Higher = more buried = structurally constrained
    pub burial_depth: f64,

    /// Epitope overlap (0-1): Overlap with known epitope classes (PRISM-VE)
    /// Higher = known immune target = validated target
    pub epitope_overlap: f64,

    /// Combined escape resistance score (weighted average)
    pub combined: f64,
}

impl EscapeResistanceScore {
    /// Compute escape resistance score from individual components
    pub fn compute(
        conservation: f64,
        n_critical_contacts: usize,
        contact_ddg_estimate: f64,
        burial_fraction: f64,
        epitope_profile: Option<&[f32; 10]>,
    ) -> Self {
        // Normalize structural constraint (3 kcal/mol = highly significant)
        let structural_constraint = 1.0 - (-contact_ddg_estimate / 3.0).exp();

        // Normalize contact criticality (5 critical contacts = very high)
        let contact_criticality = (n_critical_contacts as f64 / 5.0).tanh();

        // Burial depth already normalized
        let burial_depth = burial_fraction.clamp(0.0, 1.0);

        // Epitope overlap (average of epitope class activations if provided)
        let epitope_overlap = epitope_profile
            .map(|p| p.iter().map(|&x| x as f64).sum::<f64>() / 10.0)
            .unwrap_or(0.0);

        // Weighted combination
        // Conservation and structural constraint are most important
        let combined =
            0.30 * conservation +
            0.25 * structural_constraint +
            0.20 * contact_criticality +
            0.15 * burial_depth +
            0.10 * epitope_overlap;

        Self {
            conservation,
            structural_constraint,
            contact_criticality,
            burial_depth,
            epitope_overlap,
            combined: combined.clamp(0.0, 1.0),
        }
    }

    /// Create a default score with zero values
    pub fn zero() -> Self {
        Self {
            conservation: 0.0,
            structural_constraint: 0.0,
            contact_criticality: 0.0,
            burial_depth: 0.0,
            epitope_overlap: 0.0,
            combined: 0.0,
        }
    }
}

impl Default for EscapeResistanceScore {
    fn default() -> Self {
        Self::zero()
    }
}

/// Per-residue structural information for scoring
#[derive(Debug, Clone)]
pub struct ResidueStructureInfo {
    /// Number of heavy-atom contacts within 4Å
    pub n_contacts: usize,

    /// Number of hydrogen bonds (backbone and sidechain)
    pub n_hbonds: usize,

    /// Is involved in a salt bridge
    pub has_salt_bridge: bool,

    /// Fraction of surface that is buried (0-1)
    pub burial_fraction: f64,

    /// Is in secondary structure (helix/sheet)
    pub in_secondary_structure: bool,

    /// Contacts with the protein core
    pub core_contacts: usize,

    /// Is at a protein-protein interface
    pub at_interface: bool,
}

impl Default for ResidueStructureInfo {
    fn default() -> Self {
        Self {
            n_contacts: 0,
            n_hbonds: 0,
            has_salt_bridge: false,
            burial_fraction: 0.0,
            in_secondary_structure: false,
            core_contacts: 0,
            at_interface: false,
        }
    }
}

/// Amino acid properties for conservation analysis
pub struct AminoAcidProperties;

impl AminoAcidProperties {
    /// Get hydrophobicity index (Kyte-Doolittle scale, normalized)
    pub fn hydrophobicity(aa: char) -> f64 {
        match aa {
            'I' => 1.00, 'V' => 0.97, 'L' => 0.92, 'F' => 0.72,
            'C' => 0.67, 'M' => 0.49, 'A' => 0.46, 'G' => -0.08,
            'T' => -0.18, 'S' => -0.23, 'W' => -0.23, 'Y' => -0.33,
            'P' => -0.41, 'H' => -0.85, 'N' => -0.92, 'D' => -0.92,
            'Q' => -0.92, 'E' => -0.92, 'K' => -1.00, 'R' => -1.23,
            _ => 0.0,
        }
    }

    /// Is amino acid charged?
    pub fn is_charged(aa: char) -> bool {
        matches!(aa, 'D' | 'E' | 'K' | 'R' | 'H')
    }

    /// Is amino acid aromatic?
    pub fn is_aromatic(aa: char) -> bool {
        matches!(aa, 'F' | 'Y' | 'W' | 'H')
    }

    /// Get charge at pH 7.4
    pub fn charge(aa: char) -> f64 {
        match aa {
            'K' | 'R' => 1.0,
            'H' => 0.1, // Partially protonated at pH 7.4
            'D' | 'E' => -1.0,
            _ => 0.0,
        }
    }

    /// Structural importance (higher = more structurally constrained)
    /// Based on frequency in protein cores and secondary structure
    pub fn structural_importance(aa: char) -> f64 {
        match aa {
            'G' => 0.9,  // Glycine: critical for turns and flexibility
            'P' => 0.9,  // Proline: critical for structure
            'C' => 0.85, // Cysteine: disulfide bonds
            'W' => 0.8,  // Tryptophan: large, rare, often functional
            'Y' => 0.7,  // Tyrosine: often in active sites
            'F' => 0.6,  // Phenylalanine: hydrophobic packing
            'I' | 'L' | 'V' => 0.5, // Core hydrophobics
            'M' => 0.5,  // Methionine: protein cores
            'A' => 0.4,  // Alanine: common, flexible
            'H' => 0.6,  // Histidine: catalytic
            'N' | 'Q' => 0.4, // Amide side chains
            'D' | 'E' => 0.5, // Carboxylic acids: often functional
            'K' | 'R' => 0.4, // Basic: surface exposed
            'S' | 'T' => 0.4, // Hydroxyl: H-bonding
            _ => 0.3,
        }
    }
}

/// Configuration for escape resistance scoring
#[derive(Debug, Clone)]
pub struct EscapeResistanceConfig {
    /// Weight for conservation score
    pub conservation_weight: f64,

    /// Weight for structural constraint
    pub structural_weight: f64,

    /// Weight for contact criticality
    pub contact_weight: f64,

    /// Weight for burial depth
    pub burial_weight: f64,

    /// Weight for epitope overlap
    pub epitope_weight: f64,

    /// ΔΔG threshold for significant structural impact (kcal/mol)
    pub ddg_threshold: f64,

    /// Maximum entropy for fully conserved position
    pub max_entropy: f64,
}

impl Default for EscapeResistanceConfig {
    fn default() -> Self {
        Self {
            conservation_weight: 0.30,
            structural_weight: 0.25,
            contact_weight: 0.20,
            burial_weight: 0.15,
            epitope_weight: 0.10,
            ddg_threshold: 3.0,       // 3 kcal/mol = significant
            max_entropy: 4.32,        // log2(20) for uniform AA distribution
        }
    }
}

/// Escape resistance scorer
pub struct EscapeResistanceScorer {
    config: EscapeResistanceConfig,
}

impl EscapeResistanceScorer {
    /// Create new scorer with default config
    pub fn new() -> Self {
        Self {
            config: EscapeResistanceConfig::default(),
        }
    }

    /// Create scorer with custom config
    pub fn with_config(config: EscapeResistanceConfig) -> Self {
        Self { config }
    }

    /// Score all residues based on structural and conservation features
    ///
    /// # Arguments
    /// * `sequence` - Amino acid sequence (one-letter codes)
    /// * `msa_entropy` - Per-position entropy from MSA (lower = more conserved)
    /// * `structure_info` - Per-residue structural information
    /// * `epitope_profiles` - Optional PRISM-VE epitope class activations
    pub fn score_residues(
        &self,
        sequence: &str,
        msa_entropy: &[f64],
        structure_info: &[ResidueStructureInfo],
    ) -> Vec<EscapeResistanceScore> {
        let n_residues = sequence.len();

        // Validate inputs
        if msa_entropy.len() != n_residues || structure_info.len() != n_residues {
            log::warn!(
                "Input size mismatch: seq={}, entropy={}, struct={}",
                n_residues, msa_entropy.len(), structure_info.len()
            );
            return vec![EscapeResistanceScore::zero(); n_residues];
        }

        sequence.chars().enumerate().map(|(i, aa)| {
            let info = &structure_info[i];

            // Conservation: 1 - normalized entropy (low entropy = conserved)
            let conservation = 1.0 - (msa_entropy[i] / self.config.max_entropy).min(1.0);

            // Critical contacts = regular contacts + H-bonds * 2 + salt bridge * 3
            let n_critical = info.n_contacts.min(10) / 2  // Regular contacts (capped)
                + info.n_hbonds * 2                        // H-bonds are important
                + if info.has_salt_bridge { 3 } else { 0 } // Salt bridges very important
                + info.core_contacts;                      // Core contacts

            // ΔΔG estimate (simplified: 0.5 kcal/mol per critical contact)
            let ddg_estimate = n_critical as f64 * 0.5;

            // Boost for secondary structure
            let struct_boost = if info.in_secondary_structure { 0.5 } else { 0.0 };
            let ddg_estimate = ddg_estimate + struct_boost;

            // Boost for interface residues
            let interface_boost = if info.at_interface { 0.3 } else { 0.0 };
            let ddg_estimate = ddg_estimate + interface_boost;

            // Add amino acid intrinsic importance
            let aa_importance = AminoAcidProperties::structural_importance(aa);
            let ddg_estimate = ddg_estimate * (0.5 + aa_importance * 0.5);

            EscapeResistanceScore::compute(
                conservation,
                n_critical,
                ddg_estimate,
                info.burial_fraction,
                None, // No epitope profile for now
            )
        }).collect()
    }

    /// Score residues with epitope information from PRISM-VE
    pub fn score_residues_with_epitopes(
        &self,
        sequence: &str,
        msa_entropy: &[f64],
        structure_info: &[ResidueStructureInfo],
        epitope_profiles: &[[f32; 10]],
    ) -> Vec<EscapeResistanceScore> {
        let n_residues = sequence.len();

        if msa_entropy.len() != n_residues ||
           structure_info.len() != n_residues ||
           epitope_profiles.len() != n_residues {
            log::warn!("Input size mismatch for epitope scoring");
            return self.score_residues(sequence, msa_entropy, structure_info);
        }

        sequence.chars().enumerate().map(|(i, aa)| {
            let info = &structure_info[i];

            let conservation = 1.0 - (msa_entropy[i] / self.config.max_entropy).min(1.0);

            let n_critical = info.n_contacts.min(10) / 2
                + info.n_hbonds * 2
                + if info.has_salt_bridge { 3 } else { 0 }
                + info.core_contacts;

            let mut ddg_estimate = n_critical as f64 * 0.5;
            if info.in_secondary_structure { ddg_estimate += 0.5; }
            if info.at_interface { ddg_estimate += 0.3; }
            ddg_estimate *= 0.5 + AminoAcidProperties::structural_importance(aa) * 0.5;

            EscapeResistanceScore::compute(
                conservation,
                n_critical,
                ddg_estimate,
                info.burial_fraction,
                Some(&epitope_profiles[i]),
            )
        }).collect()
    }

    /// Compute full escape resistance from coordinates and sequence
    ///
    /// This performs complete structural analysis including:
    /// - Contact network analysis
    /// - H-bond detection
    /// - Salt bridge identification
    /// - Burial depth calculation
    /// - Secondary structure estimation
    pub fn score_from_structure(
        &self,
        coords: &[[f32; 3]],
        sequence: &str,
        msa_entropy: Option<&[f64]>,
    ) -> Vec<EscapeResistanceScore> {
        let n_residues = coords.len();

        if n_residues != sequence.len() {
            log::error!("Coordinate/sequence length mismatch: {} vs {}", n_residues, sequence.len());
            return vec![EscapeResistanceScore::zero(); n_residues];
        }

        // Compute full structural features
        let structure_info = self.analyze_structure_full(coords, sequence);

        // Use provided entropy or compute default (high entropy = unknown)
        let default_entropy = vec![2.0; n_residues]; // Moderate entropy as default
        let entropy = msa_entropy.unwrap_or(&default_entropy);

        self.score_residues(sequence, entropy, &structure_info)
    }

    /// Full structural analysis for escape resistance
    fn analyze_structure_full(
        &self,
        coords: &[[f32; 3]],
        sequence: &str,
    ) -> Vec<ResidueStructureInfo> {
        let n_residues = coords.len();
        let aa_vec: Vec<char> = sequence.chars().collect();

        // Compute pairwise distances
        let mut contact_matrix = vec![vec![false; n_residues]; n_residues];
        let contact_cutoff_sq = 64.0f32; // 8Å for CA-CA contact

        for i in 0..n_residues {
            for j in (i+3)..n_residues { // Skip i+1, i+2 (bonded neighbors)
                let dx = coords[i][0] - coords[j][0];
                let dy = coords[i][1] - coords[j][1];
                let dz = coords[i][2] - coords[j][2];
                let dist_sq = dx*dx + dy*dy + dz*dz;

                if dist_sq < contact_cutoff_sq {
                    contact_matrix[i][j] = true;
                    contact_matrix[j][i] = true;
                }
            }
        }

        // Compute contact counts
        let contact_counts: Vec<usize> = contact_matrix.iter()
            .map(|row| row.iter().filter(|&&x| x).count())
            .collect();

        let max_contacts = *contact_counts.iter().max().unwrap_or(&1) as f64;

        // Compute structure info for each residue
        (0..n_residues).map(|i| {
            let aa = aa_vec.get(i).copied().unwrap_or('X');
            let n_contacts = contact_counts[i];

            // Burial fraction from contact count (normalized)
            let burial_fraction = (n_contacts as f64 / max_contacts.max(1.0)).min(1.0);

            // Estimate H-bonds based on amino acid type and burial
            let base_hbonds = match aa {
                'S' | 'T' | 'N' | 'Q' => 2,
                'D' | 'E' | 'K' | 'R' | 'H' => 1,
                'Y' | 'W' => 1,
                _ => 0,
            };
            // More buried = more likely to have backbone H-bonds
            let n_hbonds = base_hbonds + if burial_fraction > 0.5 { 2 } else { 0 };

            // Salt bridge detection: charged residue with nearby opposite charge
            let has_salt_bridge = if AminoAcidProperties::is_charged(aa) {
                let my_charge = AminoAcidProperties::charge(aa);
                // Check neighbors for opposite charge
                contact_matrix[i].iter().enumerate()
                    .filter(|(_, &has_contact)| has_contact)
                    .any(|(j, _)| {
                        let neighbor_aa = aa_vec.get(j).copied().unwrap_or('X');
                        let neighbor_charge = AminoAcidProperties::charge(neighbor_aa);
                        my_charge * neighbor_charge < -0.5 // Opposite charges
                    })
            } else {
                false
            };

            // Secondary structure estimation from contact pattern
            // Helices: contacts at i±3,4
            // Sheets: long-range parallel contacts
            let has_helix_contacts = (i >= 3 && contact_matrix[i][i-3]) ||
                                      (i >= 4 && contact_matrix[i][i-4]) ||
                                      (i+3 < n_residues && contact_matrix[i][i+3]) ||
                                      (i+4 < n_residues && contact_matrix[i][i+4]);

            let has_sheet_contacts = contact_matrix[i].iter().enumerate()
                .filter(|(j, &has_contact)| has_contact && (*j as i32 - i as i32).abs() > 5)
                .count() >= 2;

            let in_secondary_structure = has_helix_contacts || has_sheet_contacts;

            // Core contacts: contacts with highly buried residues
            let core_contacts = contact_matrix[i].iter().enumerate()
                .filter(|(j, &has_contact)| {
                    has_contact && (contact_counts[*j] as f64 / max_contacts) > 0.6
                })
                .count();

            // Interface detection (would need multi-chain info, estimate from pattern)
            let at_interface = burial_fraction > 0.3 && burial_fraction < 0.7 && n_contacts > 4;

            ResidueStructureInfo {
                n_contacts,
                n_hbonds,
                has_salt_bridge,
                burial_fraction,
                in_secondary_structure,
                core_contacts,
                at_interface,
            }
        }).collect()
    }
}

/// Control structures for blind validation
pub mod control_structures {
    //! Reference structures for PRISM blind validation
    //!
    //! # Control Structures
    //!
    //! ## 6VXX - SARS-CoV-2 Spike Protein (Closed State)
    //! - Coronavirus spike glycoprotein
    //! - Critical for viral entry
    //! - Contains RBD with known epitopes
    //! - Heavily studied with extensive mutation data
    //!
    //! ## 2VWD - HIV-1 gp120 Core
    //! - Lentivirus envelope glycoprotein
    //! - Different viral family for cross-validation
    //! - Well-characterized escape mutations
    //! - Complex glycan shielding

    /// 6VXX - SARS-CoV-2 Spike (closed conformation)
    pub const CONTROL_6VXX: ControlStructure = ControlStructure {
        pdb_id: "6VXX",
        name: "SARS-CoV-2 Spike Glycoprotein (Closed)",
        organism: "SARS-CoV-2",
        resolution: 2.8, // Angstroms
        n_chains: 3, // Trimeric
        n_residues_per_chain: 1273,
        rcsb_url: "https://www.rcsb.org/structure/6VXX",

        // Key structural regions
        rbd_start: 319,
        rbd_end: 541,
        ntd_start: 14,
        ntd_end: 305,
        s2_start: 686,
        s2_end: 1273,

        // Known epitope classes (based on literature)
        epitope_classes: &[
            EpitopeClass { name: "RBD Class 1", residues: &[417, 455, 456, 484, 486, 487, 489, 493] },
            EpitopeClass { name: "RBD Class 2", residues: &[452, 484, 490, 493, 494] },
            EpitopeClass { name: "RBD Class 3", residues: &[440, 443, 444, 445, 446, 499, 500, 501, 502, 505] },
            EpitopeClass { name: "RBD Class 4", residues: &[368, 369, 370, 371, 372, 373, 374, 375, 376, 377] },
            EpitopeClass { name: "NTD Site i", residues: &[14, 15, 16, 17, 18, 19, 20] },
            EpitopeClass { name: "NTD Site ii", residues: &[141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156] },
            EpitopeClass { name: "S2 Stem", residues: &[1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148] },
        ],

        // Key escape mutation positions (from CoV-RDB)
        key_escape_positions: &[417, 452, 484, 501, 614, 681, 950],

        // Fusion peptide (functionally critical)
        fusion_peptide_start: 816,
        fusion_peptide_end: 833,
    };

    /// 2VWD - HIV-1 gp120 Core
    pub const CONTROL_2VWD: ControlStructure = ControlStructure {
        pdb_id: "2VWD",
        name: "HIV-1 gp120 Core (CD4-bound state)",
        organism: "HIV-1",
        resolution: 3.3, // Angstroms
        n_chains: 1,
        n_residues_per_chain: 420,
        rcsb_url: "https://www.rcsb.org/structure/2VWD",

        // Key structural regions
        rbd_start: 0, // Not applicable in same way
        rbd_end: 0,
        ntd_start: 0,
        ntd_end: 0,
        s2_start: 0,
        s2_end: 0,

        // Known epitope classes
        epitope_classes: &[
            EpitopeClass { name: "CD4 binding site", residues: &[124, 125, 126, 127, 279, 280, 281, 365, 366, 367, 368, 369, 370, 371, 427, 428, 429, 430, 431, 432, 455, 456, 457, 458, 459, 460, 469, 470, 471, 472, 473, 474] },
            EpitopeClass { name: "V1/V2 loop", residues: &[126, 127, 128, 129, 130, 131, 132, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196] },
            EpitopeClass { name: "V3 loop", residues: &[296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331] },
            EpitopeClass { name: "Bridging sheet", residues: &[418, 419, 420, 421, 422, 423, 424, 425, 426] },
        ],

        // Key escape mutation positions
        key_escape_positions: &[169, 197, 276, 279, 281, 362, 363, 364, 429, 456, 457, 458, 471],

        fusion_peptide_start: 512, // gp41 fusion peptide
        fusion_peptide_end: 527,
    };

    /// Control structure definition
    #[derive(Debug, Clone, Copy)]
    pub struct ControlStructure {
        pub pdb_id: &'static str,
        pub name: &'static str,
        pub organism: &'static str,
        pub resolution: f64,
        pub n_chains: usize,
        pub n_residues_per_chain: usize,
        pub rcsb_url: &'static str,

        // Domain boundaries
        pub rbd_start: usize,
        pub rbd_end: usize,
        pub ntd_start: usize,
        pub ntd_end: usize,
        pub s2_start: usize,
        pub s2_end: usize,

        // Epitope information
        pub epitope_classes: &'static [EpitopeClass],

        // Known escape positions
        pub key_escape_positions: &'static [usize],

        // Fusion peptide
        pub fusion_peptide_start: usize,
        pub fusion_peptide_end: usize,
    }

    impl ControlStructure {
        /// Get all epitope residues as a flat vector
        pub fn all_epitope_residues(&self) -> Vec<usize> {
            self.epitope_classes.iter()
                .flat_map(|ec| ec.residues.iter().copied())
                .collect()
        }

        /// Check if a residue is in a known epitope
        pub fn is_epitope_residue(&self, residue: usize) -> bool {
            self.epitope_classes.iter()
                .any(|ec| ec.residues.contains(&residue))
        }

        /// Check if a residue is a known escape position
        pub fn is_escape_position(&self, residue: usize) -> bool {
            self.key_escape_positions.contains(&residue)
        }

        /// Get domain for a residue
        pub fn get_domain(&self, residue: usize) -> Option<&'static str> {
            if residue >= self.rbd_start && residue <= self.rbd_end {
                Some("RBD")
            } else if residue >= self.ntd_start && residue <= self.ntd_end {
                Some("NTD")
            } else if residue >= self.s2_start && residue <= self.s2_end {
                Some("S2")
            } else if residue >= self.fusion_peptide_start && residue <= self.fusion_peptide_end {
                Some("FP")
            } else {
                None
            }
        }
    }

    /// Epitope class definition
    #[derive(Debug, Clone, Copy)]
    pub struct EpitopeClass {
        pub name: &'static str,
        pub residues: &'static [usize],
    }

    /// Get control structure by PDB ID
    pub fn get_control_structure(pdb_id: &str) -> Option<ControlStructure> {
        match pdb_id.to_uppercase().as_str() {
            "6VXX" => Some(CONTROL_6VXX),
            "2VWD" => Some(CONTROL_2VWD),
            _ => None,
        }
    }

    /// List all available control structures
    pub fn list_control_structures() -> Vec<ControlStructure> {
        vec![CONTROL_6VXX, CONTROL_2VWD]
    }
}

impl Default for EscapeResistanceScorer {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute entropy from MSA column frequencies
///
/// Shannon entropy: H = -Σ p_i × log2(p_i)
/// Lower entropy = more conserved
pub fn compute_column_entropy(aa_frequencies: &[f64]) -> f64 {
    let mut entropy = 0.0;
    for &freq in aa_frequencies {
        if freq > 0.0 {
            entropy -= freq * freq.log2();
        }
    }
    entropy
}

/// Parse simple MSA format and compute per-position entropy
pub fn compute_msa_entropy(sequences: &[&str]) -> Vec<f64> {
    if sequences.is_empty() {
        return Vec::new();
    }

    let seq_len = sequences[0].len();
    let n_seqs = sequences.len() as f64;

    // Standard amino acids
    let aa_order: Vec<char> = "ACDEFGHIKLMNPQRSTVWY".chars().collect();
    let mut aa_to_idx: HashMap<char, usize> = HashMap::new();
    for (i, &aa) in aa_order.iter().enumerate() {
        aa_to_idx.insert(aa, i);
    }

    (0..seq_len).map(|pos| {
        // Count amino acids at this position
        let mut counts = vec![0.0; 20];
        let mut valid = 0.0;

        for seq in sequences {
            if let Some(aa) = seq.chars().nth(pos) {
                if let Some(&idx) = aa_to_idx.get(&aa.to_ascii_uppercase()) {
                    counts[idx] += 1.0;
                    valid += 1.0;
                }
            }
        }

        if valid == 0.0 {
            return 4.32; // Maximum entropy for undefined
        }

        // Convert to frequencies
        let freqs: Vec<f64> = counts.iter().map(|c| c / valid).collect();

        compute_column_entropy(&freqs)
    }).collect()
}

/// Estimate structure info from CA coordinates
///
/// This provides approximate structural features when only CA trace is available.
pub fn estimate_structure_info_from_ca(
    ca_coords: &[[f32; 3]],
    sequence: &str,
) -> Vec<ResidueStructureInfo> {
    let n_residues = ca_coords.len();

    if n_residues != sequence.len() {
        return vec![ResidueStructureInfo::default(); n_residues];
    }

    // Compute contact counts (CA-CA distance < 8Å)
    let mut contact_counts = vec![0usize; n_residues];
    let contact_cutoff_sq = 64.0f32; // 8Å²

    for i in 0..n_residues {
        for j in (i+4)..n_residues { // Skip neighbors in sequence
            let dx = ca_coords[i][0] - ca_coords[j][0];
            let dy = ca_coords[i][1] - ca_coords[j][1];
            let dz = ca_coords[i][2] - ca_coords[j][2];
            let dist_sq = dx*dx + dy*dy + dz*dz;

            if dist_sq < contact_cutoff_sq {
                contact_counts[i] += 1;
                contact_counts[j] += 1;
            }
        }
    }

    // Estimate burial from contact counts
    let max_contacts = *contact_counts.iter().max().unwrap_or(&1) as f64;

    sequence.chars().enumerate().map(|(i, aa)| {
        let n_contacts = contact_counts[i];
        let burial_fraction = n_contacts as f64 / max_contacts.max(1.0);

        // Estimate H-bonds from sequence (rough approximation)
        let n_hbonds = match aa {
            'S' | 'T' | 'N' | 'Q' => 2,
            'D' | 'E' | 'K' | 'R' => 1,
            'Y' | 'W' | 'H' => 1,
            _ => 0,
        };

        // Charged residues might form salt bridges
        let has_salt_bridge = AminoAcidProperties::is_charged(aa) && burial_fraction > 0.5;

        // Core contacts (contacts with high burial residues)
        let core_contacts = if burial_fraction > 0.6 { n_contacts / 2 } else { 0 };

        ResidueStructureInfo {
            n_contacts,
            n_hbonds,
            has_salt_bridge,
            burial_fraction,
            in_secondary_structure: n_contacts >= 4, // Rough estimate
            core_contacts,
            at_interface: false, // Can't determine from CA alone
        }
    }).collect()
}

/// Combined cryptic score with escape resistance
///
/// Final score that prioritizes cryptic sites that are also escape-resistant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrypticEscapeScore {
    /// Raw cryptic score from detection pipeline
    pub cryptic_score: f64,

    /// Escape resistance score
    pub escape_resistance: EscapeResistanceScore,

    /// Combined score: cryptic × escape_resistance
    pub combined_score: f64,

    /// Druggability score (pocket volume, hydrophobicity, etc.)
    pub druggability: f64,

    /// Final priority score for drug development
    pub priority_score: f64,
}

impl CrypticEscapeScore {
    pub fn compute(
        cryptic_score: f64,
        escape_resistance: EscapeResistanceScore,
        druggability: f64,
    ) -> Self {
        // Combined score: geometric mean of cryptic and escape resistance
        let combined_score = (cryptic_score * escape_resistance.combined).sqrt();

        // Priority: weighted combination
        // High cryptic + high escape resistance + good druggability = high priority
        let priority_score =
            0.40 * cryptic_score +
            0.35 * escape_resistance.combined +
            0.25 * druggability;

        Self {
            cryptic_score,
            escape_resistance,
            combined_score,
            druggability,
            priority_score: priority_score.clamp(0.0, 1.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_resistance_score() {
        let score = EscapeResistanceScore::compute(
            0.9,   // High conservation
            5,     // 5 critical contacts
            2.5,   // 2.5 kcal/mol estimated ΔΔG
            0.7,   // 70% buried
            None,  // No epitope info
        );

        assert!(score.conservation > 0.8);
        assert!(score.structural_constraint > 0.4);
        assert!(score.combined > 0.5);
    }

    #[test]
    fn test_amino_acid_properties() {
        // Isoleucine should be highly hydrophobic
        assert!(AminoAcidProperties::hydrophobicity('I') > 0.9);

        // Arginine should be charged
        assert!(AminoAcidProperties::is_charged('R'));

        // Glycine should have high structural importance
        assert!(AminoAcidProperties::structural_importance('G') > 0.8);
    }

    #[test]
    fn test_column_entropy() {
        // All same amino acid = 0 entropy
        let uniform_one = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!((compute_column_entropy(&uniform_one) - 0.0).abs() < 0.01);

        // Equal distribution = max entropy
        let uniform_all: Vec<f64> = vec![0.05; 20];
        let entropy = compute_column_entropy(&uniform_all);
        assert!(entropy > 4.0); // Should be close to log2(20) = 4.32
    }

    #[test]
    fn test_msa_entropy() {
        let sequences = vec![
            "ACDEFGHIK",
            "ACDEFGHIK", // Identical = low entropy
            "ACDEFGHIK",
        ];

        let entropy = compute_msa_entropy(&sequences);
        assert_eq!(entropy.len(), 9);

        // All positions should have low entropy (identical sequences)
        for e in entropy {
            assert!(e < 0.1, "Expected low entropy for identical sequences, got {}", e);
        }
    }

    #[test]
    fn test_scorer() {
        let scorer = EscapeResistanceScorer::new();

        let sequence = "ACDEFG";
        let msa_entropy = vec![0.5, 0.1, 2.0, 0.3, 1.5, 0.2]; // Varying conservation
        let structure_info = vec![
            ResidueStructureInfo { n_contacts: 6, n_hbonds: 2, burial_fraction: 0.8, ..Default::default() },
            ResidueStructureInfo { n_contacts: 2, n_hbonds: 0, burial_fraction: 0.2, ..Default::default() },
            ResidueStructureInfo { n_contacts: 8, n_hbonds: 1, burial_fraction: 0.9, has_salt_bridge: true, ..Default::default() },
            ResidueStructureInfo { n_contacts: 4, n_hbonds: 1, burial_fraction: 0.5, ..Default::default() },
            ResidueStructureInfo { n_contacts: 10, n_hbonds: 3, burial_fraction: 0.95, in_secondary_structure: true, ..Default::default() },
            ResidueStructureInfo { n_contacts: 3, n_hbonds: 0, burial_fraction: 0.3, ..Default::default() },
        ];

        let scores = scorer.score_residues(sequence, &msa_entropy, &structure_info);

        assert_eq!(scores.len(), 6);

        // Residue with salt bridge and good conservation should have higher escape resistance
        assert!(scores[2].combined > scores[1].combined,
            "Residue 2 (salt bridge, buried) should score higher than residue 1 (exposed)");

        // Residue 4 (high contacts, secondary structure) should also score well
        assert!(scores[4].combined > 0.4);
    }

    #[test]
    fn test_cryptic_escape_score() {
        let escape_resistance = EscapeResistanceScore::compute(0.8, 4, 2.0, 0.7, None);

        let combined = CrypticEscapeScore::compute(
            0.85,  // High cryptic score
            escape_resistance,
            0.75,  // Good druggability
        );

        assert!(combined.cryptic_score > 0.8);
        assert!(combined.priority_score > 0.6);
    }
}

//! Target protein definitions for validation benchmarks
//!
//! Defines the structure of validation targets including:
//! - Apo-holo pairs for transition prediction
//! - Retrospective drug targets
//! - ATLAS ensemble references

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// A validation target protein
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target {
    /// Target name (e.g., "IL2", "KRAS_G12C")
    pub name: String,

    /// Protein family (e.g., "cytokine", "kinase")
    pub family: String,

    /// Therapeutic area for retrospective validation
    pub therapeutic_area: Option<String>,

    /// Description of the target
    pub description: String,

    /// Structure files
    pub structures: TargetStructures,

    /// Pocket definition
    pub pocket: Option<PocketDefinition>,

    /// Drug information (for retrospective validation)
    pub drug_info: Option<DrugInfo>,

    /// Expected difficulty
    pub difficulty: Difficulty,

    /// Validation type
    pub validation_type: ValidationType,

    /// Experimental reference data
    pub experimental: Option<ExperimentalData>,
}

/// Structure files for a target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetStructures {
    /// Apo structure PDB path
    pub apo_pdb: Option<PathBuf>,

    /// Holo structure PDB path (ground truth for transitions)
    pub holo_pdb: Option<PathBuf>,

    /// NMR ensemble PDB path (for ATLAS)
    pub ensemble_pdb: Option<PathBuf>,

    /// AlphaFold3 predicted structure (for comparison)
    pub af3_pdb: Option<PathBuf>,
}

/// Definition of a pocket/binding site
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketDefinition {
    /// Residue indices that define the pocket
    pub residues: Vec<usize>,

    /// Core residues (most important for binding)
    pub core_residues: Vec<usize>,

    /// Expected SASA gain when pocket opens (Å²)
    pub expected_sasa_gain: f32,

    /// Pocket type
    pub pocket_type: PocketType,

    /// Is this a cryptic site?
    pub is_cryptic: bool,

    /// Mechanism of pocket opening
    pub mechanism: Option<String>,
}

/// Types of binding pockets
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PocketType {
    /// Orthosteric (active site)
    Orthosteric,
    /// Allosteric (away from active site)
    Allosteric,
    /// Cryptic (hidden in apo structure)
    Cryptic,
    /// Protein-protein interface
    Interface,
    /// Covalent binding site
    Covalent,
    /// Other
    Other,
}

/// Drug information for retrospective validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrugInfo {
    /// Drug name
    pub name: String,

    /// Drug approval status
    pub status: DrugStatus,

    /// Approval year (if approved)
    pub approval_year: Option<u32>,

    /// Mechanism of action
    pub mechanism: String,

    /// Binding affinity (Ki or Kd in nM)
    pub affinity_nm: Option<f32>,

    /// PDB ID of drug-bound structure
    pub drug_bound_pdb: Option<String>,

    /// SMILES string of drug
    pub smiles: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum DrugStatus {
    Approved,
    Phase3,
    Phase2,
    Phase1,
    Preclinical,
    Discontinued,
    Research,
}

/// Difficulty level for benchmark
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
    VeryHard,
}

/// Type of validation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ValidationType {
    /// ATLAS ensemble recovery
    AtlasEnsemble,
    /// Apo-holo transition
    ApoHoloTransition,
    /// Retrospective blind validation
    RetrospectiveBlind,
    /// Novel cryptic site benchmark
    NovelCryptic,
}

/// Experimental reference data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentalData {
    /// RMSF from NMR/MD
    pub rmsf: Option<Vec<f32>>,

    /// Experimental order parameters
    pub order_parameters: Option<Vec<f32>>,

    /// Principal component vectors
    pub principal_components: Option<Vec<Vec<f32>>>,

    /// Pairwise RMSD distribution parameters
    pub pairwise_rmsd_mean: Option<f32>,
    pub pairwise_rmsd_std: Option<f32>,
}

/// Collection of all validation targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetCollection {
    /// Version of the target collection
    pub version: String,

    /// Description
    pub description: String,

    /// Last updated
    pub last_updated: String,

    /// All targets
    pub targets: Vec<Target>,
}

impl TargetCollection {
    /// Load from JSON file
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let collection: Self = serde_json::from_str(&content)?;
        Ok(collection)
    }

    /// Save to JSON file
    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Filter by validation type
    pub fn filter_by_type(&self, validation_type: ValidationType) -> Vec<&Target> {
        self.targets
            .iter()
            .filter(|t| t.validation_type == validation_type)
            .collect()
    }

    /// Filter by difficulty
    pub fn filter_by_difficulty(&self, difficulty: Difficulty) -> Vec<&Target> {
        self.targets
            .iter()
            .filter(|t| t.difficulty == difficulty)
            .collect()
    }

    /// Filter by therapeutic area
    pub fn filter_by_therapeutic_area(&self, area: &str) -> Vec<&Target> {
        self.targets
            .iter()
            .filter(|t| {
                t.therapeutic_area
                    .as_ref()
                    .map(|a| a == area)
                    .unwrap_or(false)
            })
            .collect()
    }
}

/// Builder for creating retrospective validation targets
pub struct RetrospectiveTargetBuilder {
    target: Target,
}

impl RetrospectiveTargetBuilder {
    pub fn new(name: &str) -> Self {
        Self {
            target: Target {
                name: name.to_string(),
                family: String::new(),
                therapeutic_area: None,
                description: String::new(),
                structures: TargetStructures {
                    apo_pdb: None,
                    holo_pdb: None,
                    ensemble_pdb: None,
                    af3_pdb: None,
                },
                pocket: None,
                drug_info: None,
                difficulty: Difficulty::Medium,
                validation_type: ValidationType::RetrospectiveBlind,
                experimental: None,
            },
        }
    }

    pub fn family(mut self, family: &str) -> Self {
        self.target.family = family.to_string();
        self
    }

    pub fn therapeutic_area(mut self, area: &str) -> Self {
        self.target.therapeutic_area = Some(area.to_string());
        self
    }

    pub fn description(mut self, desc: &str) -> Self {
        self.target.description = desc.to_string();
        self
    }

    pub fn apo_pdb(mut self, path: &str) -> Self {
        self.target.structures.apo_pdb = Some(PathBuf::from(path));
        self
    }

    pub fn holo_pdb(mut self, path: &str) -> Self {
        self.target.structures.holo_pdb = Some(PathBuf::from(path));
        self
    }

    pub fn pocket(mut self, residues: Vec<usize>, is_cryptic: bool) -> Self {
        self.target.pocket = Some(PocketDefinition {
            residues: residues.clone(),
            core_residues: residues.iter().take(6).cloned().collect(),
            expected_sasa_gain: 150.0,
            pocket_type: if is_cryptic {
                PocketType::Cryptic
            } else {
                PocketType::Orthosteric
            },
            is_cryptic,
            mechanism: None,
        });
        self
    }

    pub fn drug(
        mut self,
        name: &str,
        status: DrugStatus,
        approval_year: Option<u32>,
        mechanism: &str,
    ) -> Self {
        self.target.drug_info = Some(DrugInfo {
            name: name.to_string(),
            status,
            approval_year,
            mechanism: mechanism.to_string(),
            affinity_nm: None,
            drug_bound_pdb: None,
            smiles: None,
        });
        self
    }

    pub fn difficulty(mut self, difficulty: Difficulty) -> Self {
        self.target.difficulty = difficulty;
        self
    }

    pub fn build(self) -> Target {
        self.target
    }
}

/// Create the core retrospective validation targets
pub fn create_retrospective_targets() -> Vec<Target> {
    vec![
        // === ONCOLOGY ===
        RetrospectiveTargetBuilder::new("KRAS_G12C")
            .family("gtpase")
            .therapeutic_area("Oncology")
            .description("KRAS G12C mutant with cryptic Switch-II pocket - Sotorasib target")
            .apo_pdb("data/validation/retrospective/oncology/4OBE.pdb")
            .holo_pdb("data/validation/retrospective/oncology/6OIM.pdb")
            .pocket(vec![60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72], true)
            .drug("Sotorasib", DrugStatus::Approved, Some(2021), "Covalent inhibitor of G12C mutant")
            .difficulty(Difficulty::Hard)
            .build(),

        RetrospectiveTargetBuilder::new("BTK")
            .family("kinase")
            .therapeutic_area("Oncology")
            .description("Bruton's Tyrosine Kinase - Ibrutinib target")
            .apo_pdb("data/validation/retrospective/oncology/3GEN.pdb")
            .holo_pdb("data/validation/retrospective/oncology/5P9J.pdb")
            .pocket(vec![408, 409, 410, 411, 412, 413, 414, 474, 475, 476, 477, 478, 481], true)
            .drug("Ibrutinib", DrugStatus::Approved, Some(2013), "Covalent C481 inhibitor")
            .difficulty(Difficulty::Medium)
            .build(),

        RetrospectiveTargetBuilder::new("BCR_ABL")
            .family("kinase")
            .therapeutic_area("Oncology")
            .description("BCR-ABL fusion kinase - Imatinib target (DFG-out pocket)")
            .apo_pdb("data/validation/retrospective/oncology/1IEP.pdb")
            .holo_pdb("data/validation/retrospective/oncology/1OPJ.pdb")
            .pocket(vec![271, 286, 290, 315, 317, 318, 380, 381, 382, 383], true)
            .drug("Imatinib", DrugStatus::Approved, Some(2001), "Type II DFG-out inhibitor")
            .difficulty(Difficulty::Medium)
            .build(),

        RetrospectiveTargetBuilder::new("SHP2")
            .family("phosphatase")
            .therapeutic_area("Oncology")
            .description("SHP2 phosphatase - Tunnel site for allosteric inhibition")
            .apo_pdb("data/validation/retrospective/oncology/2SHP.pdb")
            .holo_pdb("data/validation/retrospective/oncology/6BMU.pdb")
            .pocket(vec![100, 101, 102, 103, 104, 105, 106, 107, 108, 109], true)
            .drug("TNO155", DrugStatus::Phase3, None, "Allosteric tunnel inhibitor")
            .difficulty(Difficulty::Hard)
            .build(),

        // === METABOLIC ===
        RetrospectiveTargetBuilder::new("PTP1B")
            .family("phosphatase")
            .therapeutic_area("Metabolic")
            .description("PTP1B phosphatase - Allosteric C-terminal site for diabetes")
            .apo_pdb("data/validation/retrospective/metabolic/2HNP.pdb")
            .holo_pdb("data/validation/retrospective/metabolic/1T49.pdb")
            .pocket(vec![280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290], true)
            .drug("Trodusquemine", DrugStatus::Phase2, None, "Allosteric C-term inhibitor")
            .difficulty(Difficulty::Hard)
            .build(),

        RetrospectiveTargetBuilder::new("PCSK9")
            .family("protease")
            .therapeutic_area("Metabolic")
            .description("PCSK9 - Cryptic EGF-A binding site")
            .apo_pdb("data/validation/retrospective/metabolic/2PMW.pdb")
            .holo_pdb("data/validation/retrospective/metabolic/3BPS.pdb")
            .pocket(vec![153, 155, 157, 194, 197, 198, 199, 238, 239, 240], true)
            .drug("Evolocumab", DrugStatus::Approved, Some(2015), "mAb blocking LDLR binding")
            .difficulty(Difficulty::Hard)
            .build(),

        // === INFECTIOUS DISEASE ===
        RetrospectiveTargetBuilder::new("HIV_RT")
            .family("polymerase")
            .therapeutic_area("Infectious")
            .description("HIV-1 Reverse Transcriptase - NNRTI allosteric pocket")
            .apo_pdb("data/validation/retrospective/infectious/1DLO.pdb")
            .holo_pdb("data/validation/retrospective/infectious/4G1Q.pdb")
            .pocket(vec![100, 101, 103, 106, 181, 188, 190, 227, 229, 230, 318], true)
            .drug("Rilpivirine", DrugStatus::Approved, Some(2011), "NNRTI allosteric inhibitor")
            .difficulty(Difficulty::Hard)
            .build(),

        RetrospectiveTargetBuilder::new("HCV_NS3")
            .family("protease")
            .therapeutic_area("Infectious")
            .description("HCV NS3/4A Protease - Allosteric site")
            .apo_pdb("data/validation/retrospective/infectious/1A1R.pdb")
            .holo_pdb("data/validation/retrospective/infectious/4NWL.pdb")
            .pocket(vec![41, 42, 43, 55, 57, 132, 135, 136, 137, 139, 155, 156, 168], true)
            .drug("Glecaprevir", DrugStatus::Approved, Some(2017), "NS3/4A protease inhibitor")
            .difficulty(Difficulty::Medium)
            .build(),

        RetrospectiveTargetBuilder::new("SARS2_Spike")
            .family("viral_glycoprotein")
            .therapeutic_area("Infectious")
            .description("SARS-CoV-2 Spike RBD - Cryptic sites for broad-spectrum antivirals")
            .apo_pdb("data/validation/retrospective/infectious/6VXX.pdb")
            .holo_pdb("data/validation/retrospective/infectious/6VYB.pdb")
            .pocket(vec![417, 453, 455, 456, 473, 475, 476, 477, 484, 486, 487, 489, 493], true)
            .drug("Research", DrugStatus::Research, None, "RBD cryptic site targeting")
            .difficulty(Difficulty::VeryHard)
            .build(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retrospective_target_builder() {
        let target = RetrospectiveTargetBuilder::new("TEST")
            .family("kinase")
            .therapeutic_area("Oncology")
            .description("Test target")
            .difficulty(Difficulty::Hard)
            .build();

        assert_eq!(target.name, "TEST");
        assert_eq!(target.family, "kinase");
        assert_eq!(target.difficulty, Difficulty::Hard);
    }

    #[test]
    fn test_create_retrospective_targets() {
        let targets = create_retrospective_targets();
        assert!(!targets.is_empty());

        // Check we have targets from each therapeutic area
        let oncology: Vec<_> = targets
            .iter()
            .filter(|t| t.therapeutic_area.as_ref().map(|a| a == "Oncology").unwrap_or(false))
            .collect();
        assert!(!oncology.is_empty());

        let metabolic: Vec<_> = targets
            .iter()
            .filter(|t| t.therapeutic_area.as_ref().map(|a| a == "Metabolic").unwrap_or(false))
            .collect();
        assert!(!metabolic.is_empty());

        let infectious: Vec<_> = targets
            .iter()
            .filter(|t| t.therapeutic_area.as_ref().map(|a| a == "Infectious").unwrap_or(false))
            .collect();
        assert!(!infectious.is_empty());
    }
}

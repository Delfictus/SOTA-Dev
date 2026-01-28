//! Enhanced Explainability Module
//!
//! Provides detailed explanations for pocket predictions:
//! - Per-residue contribution scores
//! - Feature importance breakdown
//! - Confidence decomposition
//! - Druggability factor analysis
//!
//! Designed for interpretable AI in drug discovery.

use crate::pocket::Pocket;
use crate::structure::Atom;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete explanation for a pocket prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PocketExplanation {
    /// Pocket identifier
    pub pocket_id: usize,
    /// Overall confidence score [0-1]
    pub confidence: f64,
    /// Confidence breakdown by signal type
    pub confidence_breakdown: ConfidenceBreakdown,
    /// Per-residue contributions
    pub residue_contributions: Vec<ResidueContribution>,
    /// Druggability factor decomposition
    pub druggability_factors: DruggabilityFactors,
    /// Detection signals that contributed
    pub detection_signals: Vec<DetectionSignal>,
    /// Human-readable summary
    pub summary: String,
    /// Detailed reasoning
    pub reasoning: Vec<String>,
}

/// Breakdown of confidence by signal source
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfidenceBreakdown {
    /// Geometric detection confidence
    pub geometric: f64,
    /// Cryptic/flexibility signal
    pub cryptic: f64,
    /// Allosteric coupling signal
    pub allosteric: f64,
    /// Conservation signal (if MSA available)
    pub conservation: f64,
    /// Network centrality signal
    pub centrality: f64,
    /// Volume/shape quality
    pub shape_quality: f64,
}

impl ConfidenceBreakdown {
    /// Calculate weighted average confidence
    pub fn weighted_average(&self) -> f64 {
        let weights = [0.30, 0.20, 0.15, 0.15, 0.10, 0.10];
        let values = [
            self.geometric,
            self.cryptic,
            self.allosteric,
            self.conservation,
            self.centrality,
            self.shape_quality,
        ];

        let weighted_sum: f64 = weights
            .iter()
            .zip(values.iter())
            .map(|(w, v)| w * v)
            .sum();

        weighted_sum.clamp(0.0, 1.0)
    }

    /// Get dominant signal
    pub fn dominant_signal(&self) -> &'static str {
        let signals = [
            (self.geometric, "geometric"),
            (self.cryptic, "cryptic"),
            (self.allosteric, "allosteric"),
            (self.conservation, "conservation"),
            (self.centrality, "centrality"),
            (self.shape_quality, "shape"),
        ];

        signals
            .iter()
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .map(|(_, name)| *name)
            .unwrap_or("unknown")
    }
}

/// Per-residue contribution to pocket prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidueContribution {
    /// Residue sequence number
    pub residue_seq: i32,
    /// Residue name (3-letter code)
    pub residue_name: String,
    /// Chain ID
    pub chain_id: char,
    /// Overall contribution score [0-1]
    pub contribution_score: f64,
    /// Breakdown by factor
    pub factors: ResidueFactors,
    /// Role in pocket
    pub role: ResidueRole,
    /// Is this residue critical for druggability?
    pub is_critical: bool,
}

/// Factors contributing to residue importance
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ResidueFactors {
    /// Geometric contribution (pocket lining)
    pub geometric: f64,
    /// Hydrophobicity contribution
    pub hydrophobic: f64,
    /// Hydrogen bonding potential
    pub h_bond: f64,
    /// Aromatic/pi-stacking potential
    pub aromatic: f64,
    /// Flexibility/dynamics contribution
    pub flexibility: f64,
    /// Conservation score (if available)
    pub conservation: f64,
    /// Network centrality
    pub centrality: f64,
}

/// Role of residue in the pocket
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ResidueRole {
    /// Core pocket-lining residue
    PocketLining,
    /// Key interaction residue (H-bond donor/acceptor)
    KeyInteraction,
    /// Gatekeeper controlling access
    Gatekeeper,
    /// Flexible hinge region
    Hinge,
    /// Allosteric communication hub
    AllostericHub,
    /// Supporting residue (secondary importance)
    Supporting,
}

impl ResidueRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            ResidueRole::PocketLining => "Pocket Lining",
            ResidueRole::KeyInteraction => "Key Interaction",
            ResidueRole::Gatekeeper => "Gatekeeper",
            ResidueRole::Hinge => "Hinge",
            ResidueRole::AllostericHub => "Allosteric Hub",
            ResidueRole::Supporting => "Supporting",
        }
    }
}

/// Druggability factor decomposition
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DruggabilityFactors {
    /// Total druggability score [0-1]
    pub total: f64,
    /// Volume factor (optimal: 300-1000 Å³)
    pub volume: f64,
    /// Hydrophobicity (ideal: 40-60% non-polar)
    pub hydrophobicity: f64,
    /// Enclosure (pocket depth/accessibility)
    pub enclosure: f64,
    /// Hydrogen bonding capacity
    pub h_bond_capacity: f64,
    /// Shape complementarity
    pub shape: f64,
    /// Druggability classification
    pub classification: DruggabilityClass,
    /// Individual factor assessments
    pub assessments: Vec<FactorAssessment>,
}

/// Druggability classification
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum DruggabilityClass {
    HighlyDruggable,
    #[default]
    Druggable,
    DifficultTarget,
    Undruggable,
}

impl DruggabilityClass {
    pub fn as_str(&self) -> &'static str {
        match self {
            DruggabilityClass::HighlyDruggable => "Highly Druggable",
            DruggabilityClass::Druggable => "Druggable",
            DruggabilityClass::DifficultTarget => "Difficult Target",
            DruggabilityClass::Undruggable => "Undruggable",
        }
    }

    pub fn from_score(score: f64) -> Self {
        if score >= 0.7 {
            DruggabilityClass::HighlyDruggable
        } else if score >= 0.5 {
            DruggabilityClass::Druggable
        } else if score >= 0.3 {
            DruggabilityClass::DifficultTarget
        } else {
            DruggabilityClass::Undruggable
        }
    }
}

/// Assessment of a single druggability factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorAssessment {
    pub name: String,
    pub value: f64,
    pub optimal_range: (f64, f64),
    pub status: AssessmentStatus,
    pub explanation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AssessmentStatus {
    Optimal,
    Acceptable,
    Suboptimal,
    Problematic,
}

/// Detection signal that contributed to prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionSignal {
    /// Signal source
    pub source: String,
    /// Signal strength [0-1]
    pub strength: f64,
    /// Weight in final score
    pub weight: f64,
    /// Description
    pub description: String,
}

/// Explainability engine for generating pocket explanations
pub struct ExplainabilityEngine {
    /// Include detailed residue analysis
    pub detailed_residues: bool,
    /// Minimum contribution to report
    pub min_contribution: f64,
    /// Maximum residues to report
    pub max_residues: usize,
}

impl Default for ExplainabilityEngine {
    fn default() -> Self {
        Self {
            detailed_residues: true,
            min_contribution: 0.1,
            max_residues: 20,
        }
    }
}

impl ExplainabilityEngine {
    pub fn new() -> Self {
        Self::default()
    }

    /// Generate explanation for a pocket
    pub fn explain(&self, pocket: &Pocket, atoms: &[Atom], pocket_id: usize) -> PocketExplanation {
        // Calculate confidence breakdown
        let confidence_breakdown = self.calculate_confidence_breakdown(pocket);

        // Get residue contributions
        let residue_contributions = self.calculate_residue_contributions(pocket, atoms);

        // Calculate druggability factors
        let druggability_factors = self.calculate_druggability_factors(pocket, atoms);

        // Build detection signals
        let detection_signals = self.build_detection_signals(pocket);

        // Generate summary
        let summary = self.generate_summary(pocket, pocket_id, &confidence_breakdown, &druggability_factors);

        // Generate detailed reasoning
        let reasoning = self.generate_reasoning(
            pocket,
            &residue_contributions,
            &druggability_factors,
        );

        PocketExplanation {
            pocket_id,
            confidence: confidence_breakdown.weighted_average(),
            confidence_breakdown,
            residue_contributions,
            druggability_factors,
            detection_signals,
            summary,
            reasoning,
        }
    }

    fn calculate_confidence_breakdown(&self, pocket: &Pocket) -> ConfidenceBreakdown {
        // Calculate shape quality based on volume
        let shape_quality = if pocket.volume >= 100.0 && pocket.volume <= 2000.0 {
            1.0 - ((pocket.volume - 500.0).abs() / 1500.0).min(1.0) * 0.5
        } else {
            0.3
        };

        // Use druggability as geometric signal
        let geometric = pocket.druggability_score.total.clamp(0.0, 1.0);

        ConfidenceBreakdown {
            geometric,
            cryptic: 0.0,     // Would come from softspot detection
            allosteric: 0.0,  // Would come from allosteric module
            conservation: 0.0, // Would come from MSA
            centrality: 0.0,  // Would come from network analysis
            shape_quality,
        }
    }

    fn calculate_residue_contributions(
        &self,
        pocket: &Pocket,
        atoms: &[Atom],
    ) -> Vec<ResidueContribution> {
        // Group atoms by residue
        let mut residue_atoms: HashMap<(i32, char), Vec<&Atom>> = HashMap::new();

        for &idx in &pocket.atom_indices {
            if let Some(atom) = atoms.get(idx) {
                residue_atoms
                    .entry((atom.residue_seq, atom.chain_id))
                    .or_default()
                    .push(atom);
            }
        }

        // Calculate contributions for each residue
        let mut contributions: Vec<ResidueContribution> = residue_atoms
            .into_iter()
            .map(|((res_seq, chain_id), res_atoms)| {
                let res_name = res_atoms
                    .first()
                    .map(|a| a.residue_name.clone())
                    .unwrap_or_default();

                // Calculate factors
                let factors = self.calculate_residue_factors(&res_name, &res_atoms);

                // Calculate overall contribution
                let contribution_score = (factors.geometric * 0.3
                    + factors.hydrophobic * 0.2
                    + factors.h_bond * 0.2
                    + factors.aromatic * 0.15
                    + factors.flexibility * 0.15)
                    .clamp(0.0, 1.0);

                // Determine role
                let role = self.determine_residue_role(&factors, &res_name);

                // Is critical?
                let is_critical = contribution_score > 0.6
                    || role == ResidueRole::KeyInteraction
                    || role == ResidueRole::Gatekeeper;

                ResidueContribution {
                    residue_seq: res_seq,
                    residue_name: res_name,
                    chain_id,
                    contribution_score,
                    factors,
                    role,
                    is_critical,
                }
            })
            .collect();

        // Sort by contribution (highest first)
        contributions.sort_by(|a, b| {
            b.contribution_score
                .partial_cmp(&a.contribution_score)
                .unwrap()
        });

        // Limit and filter
        contributions
            .into_iter()
            .filter(|c| c.contribution_score >= self.min_contribution)
            .take(self.max_residues)
            .collect()
    }

    fn calculate_residue_factors(&self, res_name: &str, _atoms: &[&Atom]) -> ResidueFactors {
        // Hydrophobicity scale (Kyte-Doolittle)
        let hydrophobic = match res_name {
            "ILE" | "VAL" | "LEU" => 0.9,
            "PHE" | "MET" | "ALA" => 0.8,
            "TRP" | "CYS" => 0.7,
            "GLY" | "TYR" => 0.5,
            "PRO" | "THR" | "SER" | "HIS" => 0.3,
            "ASN" | "GLN" => 0.2,
            "ASP" | "GLU" | "LYS" | "ARG" => 0.1,
            _ => 0.5,
        };

        // H-bond potential
        let h_bond = match res_name {
            "SER" | "THR" | "TYR" | "ASN" | "GLN" => 0.9,
            "ASP" | "GLU" | "HIS" | "LYS" | "ARG" => 0.8,
            "TRP" | "CYS" => 0.5,
            _ => 0.2,
        };

        // Aromatic character
        let aromatic = match res_name {
            "PHE" | "TYR" | "TRP" => 0.9,
            "HIS" => 0.6,
            _ => 0.0,
        };

        ResidueFactors {
            geometric: 0.7, // Default for pocket-lining
            hydrophobic,
            h_bond,
            aromatic,
            flexibility: 0.5, // Would come from B-factors
            conservation: 0.5, // Would come from MSA
            centrality: 0.5, // Would come from network
        }
    }

    fn determine_residue_role(&self, factors: &ResidueFactors, res_name: &str) -> ResidueRole {
        // Key interaction residues
        if factors.h_bond > 0.7 {
            return ResidueRole::KeyInteraction;
        }

        // Aromatic gatekeepers
        if factors.aromatic > 0.8 {
            return ResidueRole::Gatekeeper;
        }

        // Glycine often in hinges
        if res_name == "GLY" || res_name == "PRO" {
            return ResidueRole::Hinge;
        }

        // High centrality = allosteric hub
        if factors.centrality > 0.7 {
            return ResidueRole::AllostericHub;
        }

        // High geometric contribution = pocket lining
        if factors.geometric > 0.5 {
            return ResidueRole::PocketLining;
        }

        ResidueRole::Supporting
    }

    fn calculate_druggability_factors(
        &self,
        pocket: &Pocket,
        _atoms: &[Atom],
    ) -> DruggabilityFactors {
        let ds = &pocket.druggability_score;

        // Volume assessment
        let volume_score = if pocket.volume >= 300.0 && pocket.volume <= 1000.0 {
            1.0
        } else if pocket.volume >= 200.0 && pocket.volume <= 1500.0 {
            0.7
        } else if pocket.volume >= 100.0 && pocket.volume <= 2000.0 {
            0.4
        } else {
            0.2
        };

        let volume_assessment = FactorAssessment {
            name: "Volume".to_string(),
            value: pocket.volume,
            optimal_range: (300.0, 1000.0),
            status: if volume_score > 0.8 {
                AssessmentStatus::Optimal
            } else if volume_score > 0.5 {
                AssessmentStatus::Acceptable
            } else {
                AssessmentStatus::Suboptimal
            },
            explanation: format!(
                "Pocket volume {:.0} Å³ (optimal: 300-1000 Å³)",
                pocket.volume
            ),
        };

        // Hydrophobicity assessment
        let hydro_score = ds.components.hydro.clamp(0.0, 1.0);
        let hydro_assessment = FactorAssessment {
            name: "Hydrophobicity".to_string(),
            value: hydro_score,
            optimal_range: (0.4, 0.7),
            status: if hydro_score >= 0.4 && hydro_score <= 0.7 {
                AssessmentStatus::Optimal
            } else if hydro_score >= 0.3 && hydro_score <= 0.8 {
                AssessmentStatus::Acceptable
            } else {
                AssessmentStatus::Suboptimal
            },
            explanation: format!(
                "Hydrophobic character {:.0}% (optimal: 40-70%)",
                hydro_score * 100.0
            ),
        };

        // Enclosure assessment
        let enclosure_score = ds.components.enclosure.clamp(0.0, 1.0);
        let enclosure_assessment = FactorAssessment {
            name: "Enclosure".to_string(),
            value: enclosure_score,
            optimal_range: (0.5, 0.9),
            status: if enclosure_score >= 0.5 {
                AssessmentStatus::Optimal
            } else if enclosure_score >= 0.3 {
                AssessmentStatus::Acceptable
            } else {
                AssessmentStatus::Problematic
            },
            explanation: format!(
                "Pocket enclosure {:.0}% (well-defined binding cavity)",
                enclosure_score * 100.0
            ),
        };

        let total = ds.total.clamp(0.0, 1.0);

        DruggabilityFactors {
            total,
            volume: volume_score,
            hydrophobicity: hydro_score,
            enclosure: enclosure_score,
            h_bond_capacity: 0.5, // Would calculate from residue composition
            shape: volume_score * enclosure_score,
            classification: DruggabilityClass::from_score(total),
            assessments: vec![volume_assessment, hydro_assessment, enclosure_assessment],
        }
    }

    fn build_detection_signals(&self, pocket: &Pocket) -> Vec<DetectionSignal> {
        let mut signals = Vec::new();

        signals.push(DetectionSignal {
            source: "Geometric".to_string(),
            strength: pocket.druggability_score.total,
            weight: 0.4,
            description: "Alpha-sphere cavity detection".to_string(),
        });

        signals.push(DetectionSignal {
            source: "Shape".to_string(),
            strength: pocket.druggability_score.components.enclosure,
            weight: 0.2,
            description: "Pocket enclosure and depth".to_string(),
        });

        signals.push(DetectionSignal {
            source: "Chemistry".to_string(),
            strength: pocket.druggability_score.components.hydro,
            weight: 0.2,
            description: "Hydrophobic/polar balance".to_string(),
        });

        signals
    }

    fn generate_summary(
        &self,
        pocket: &Pocket,
        pocket_id: usize,
        confidence: &ConfidenceBreakdown,
        druggability: &DruggabilityFactors,
    ) -> String {
        format!(
            "Pocket {} is {} with {:.0}% confidence. \
             Volume: {:.0} Å³. Classification: {}. \
             Primary signal: {}.",
            pocket_id,
            druggability.classification.as_str().to_lowercase(),
            confidence.weighted_average() * 100.0,
            pocket.volume,
            druggability.classification.as_str(),
            confidence.dominant_signal(),
        )
    }

    fn generate_reasoning(
        &self,
        pocket: &Pocket,
        residues: &[ResidueContribution],
        druggability: &DruggabilityFactors,
    ) -> Vec<String> {
        let mut reasoning = Vec::new();

        // Volume reasoning
        if pocket.volume >= 300.0 && pocket.volume <= 1000.0 {
            reasoning.push(format!(
                "Optimal pocket volume ({:.0} Å³) suitable for small molecule binding.",
                pocket.volume
            ));
        } else if pocket.volume < 300.0 {
            reasoning.push(format!(
                "Small pocket volume ({:.0} Å³) may limit ligand diversity.",
                pocket.volume
            ));
        } else {
            reasoning.push(format!(
                "Large pocket volume ({:.0} Å³) may require fragment-based approaches.",
                pocket.volume
            ));
        }

        // Druggability reasoning
        match druggability.classification {
            DruggabilityClass::HighlyDruggable => {
                reasoning.push(
                    "Excellent druggability profile with balanced hydrophobicity and enclosure."
                        .to_string(),
                );
            }
            DruggabilityClass::Druggable => {
                reasoning.push("Good druggability profile suitable for standard drug design.".to_string());
            }
            DruggabilityClass::DifficultTarget => {
                reasoning.push(
                    "Challenging target may require specialized compound design.".to_string(),
                );
            }
            DruggabilityClass::Undruggable => {
                reasoning.push(
                    "Low druggability suggests allosteric or PPI-type targeting strategies."
                        .to_string(),
                );
            }
        }

        // Key residue reasoning
        let critical_residues: Vec<_> = residues.iter().filter(|r| r.is_critical).collect();
        if !critical_residues.is_empty() {
            let names: Vec<_> = critical_residues
                .iter()
                .take(3)
                .map(|r| format!("{}{}", r.residue_name, r.residue_seq))
                .collect();
            reasoning.push(format!(
                "Key residues for ligand interaction: {}.",
                names.join(", ")
            ));
        }

        // Role distribution
        let key_interaction_count = residues
            .iter()
            .filter(|r| r.role == ResidueRole::KeyInteraction)
            .count();
        if key_interaction_count >= 3 {
            reasoning.push(format!(
                "{} residues with strong H-bond potential for specific binding.",
                key_interaction_count
            ));
        }

        reasoning
    }

    /// Generate JSON-serializable explanation
    pub fn explain_json(&self, pocket: &Pocket, atoms: &[Atom], pocket_id: usize) -> serde_json::Value {
        let explanation = self.explain(pocket, atoms, pocket_id);
        serde_json::to_value(explanation).unwrap_or(serde_json::Value::Null)
    }

    /// Generate human-readable text report
    pub fn explain_text(&self, pocket: &Pocket, atoms: &[Atom], pocket_id: usize) -> String {
        let exp = self.explain(pocket, atoms, pocket_id);

        let mut report = String::new();

        report.push_str(&format!(
            "\n╔═══════════════════════════════════════════════════════════════╗\n"
        ));
        report.push_str(&format!(
            "║           POCKET {} EXPLANATION REPORT                        ║\n",
            exp.pocket_id
        ));
        report.push_str(&format!(
            "╠═══════════════════════════════════════════════════════════════╣\n"
        ));
        report.push_str(&format!(
            "║  Confidence: {:>5.1}%   Classification: {:<20}  ║\n",
            exp.confidence * 100.0,
            exp.druggability_factors.classification.as_str()
        ));
        report.push_str(&format!(
            "╠═══════════════════════════════════════════════════════════════╣\n"
        ));

        // Summary
        report.push_str(&format!("║  SUMMARY: {:<52} ║\n", &exp.summary[..exp.summary.len().min(52)]));

        // Confidence breakdown
        report.push_str(&format!(
            "╠═══════════════════════════════════════════════════════════════╣\n"
        ));
        report.push_str(&format!("║  CONFIDENCE BREAKDOWN:                                        ║\n"));
        report.push_str(&format!(
            "║    Geometric: {:>5.1}%   Cryptic: {:>5.1}%   Allosteric: {:>5.1}%    ║\n",
            exp.confidence_breakdown.geometric * 100.0,
            exp.confidence_breakdown.cryptic * 100.0,
            exp.confidence_breakdown.allosteric * 100.0
        ));

        // Druggability factors
        report.push_str(&format!(
            "╠═══════════════════════════════════════════════════════════════╣\n"
        ));
        report.push_str(&format!("║  DRUGGABILITY FACTORS:                                        ║\n"));
        report.push_str(&format!(
            "║    Volume: {:>5.1}%   Hydrophobicity: {:>5.1}%   Enclosure: {:>5.1}%  ║\n",
            exp.druggability_factors.volume * 100.0,
            exp.druggability_factors.hydrophobicity * 100.0,
            exp.druggability_factors.enclosure * 100.0
        ));

        // Top residues
        report.push_str(&format!(
            "╠═══════════════════════════════════════════════════════════════╣\n"
        ));
        report.push_str(&format!("║  KEY RESIDUES:                                                ║\n"));
        for res in exp.residue_contributions.iter().take(5) {
            let critical_str = if res.is_critical { "*CRITICAL*" } else { "" };
            report.push_str(&format!(
                "║    {:<3}{:>4} {:>12}: {:>5.1}%  {:<15}           ║\n",
                res.residue_name,
                res.residue_seq,
                res.role.as_str(),
                res.contribution_score * 100.0,
                critical_str
            ));
        }

        // Reasoning
        report.push_str(&format!(
            "╠═══════════════════════════════════════════════════════════════╣\n"
        ));
        report.push_str(&format!("║  REASONING:                                                   ║\n"));
        for reason in &exp.reasoning {
            let truncated = if reason.len() > 58 {
                format!("{}...", &reason[..55])
            } else {
                reason.clone()
            };
            report.push_str(&format!("║    - {:<56} ║\n", truncated));
        }

        report.push_str(&format!(
            "╚═══════════════════════════════════════════════════════════════╝\n"
        ));

        report
    }
}

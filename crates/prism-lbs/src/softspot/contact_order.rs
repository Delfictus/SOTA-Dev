//! Contact Order Analysis for cryptic site detection
//!
//! Implements relative contact order (RCO) and local contact order analysis
//! to identify regions that are more likely to undergo conformational rearrangement.
//!
//! ## Scientific Basis
//!
//! - Contact order measures the average sequence separation of contacting residues
//! - Low contact order regions have fewer long-range contacts and rearrange more easily
//! - Loop regions and domain linkers typically have low contact order
//! - These regions are often gates for cryptic binding sites
//!
//! ## References
//!
//! - Plaxco et al. (1998) - Contact order and protein folding rates
//! - Ivankov & Finkelstein (2004) - Prediction of protein folding rates

use crate::structure::Atom;
use std::collections::HashMap;

//=============================================================================
// CONSTANTS
//=============================================================================

/// Distance cutoff (Å) for defining a contact between residues
pub const CONTACT_DISTANCE: f64 = 8.0;

/// Minimum sequence separation to count as a contact
pub const MIN_SEQUENCE_SEPARATION: i32 = 3;

/// Window size for local contact order calculation
pub const LOCAL_WINDOW_SIZE: usize = 9;

/// Threshold for low contact order (potential rearrangement region)
pub const LOW_CONTACT_ORDER_THRESHOLD: f64 = 0.3;

/// Weight of contact order signal in combined scoring
pub const CONTACT_ORDER_WEIGHT: f64 = 0.15;

//=============================================================================
// TYPES
//=============================================================================

/// Per-residue contact order metrics
#[derive(Debug, Clone)]
pub struct ResidueContactOrder {
    pub residue_seq: i32,
    pub chain_id: char,
    /// Local contact order (averaged over window)
    pub local_contact_order: f64,
    /// Number of contacts this residue makes
    pub num_contacts: usize,
    /// Average sequence separation of contacts
    pub mean_sequence_separation: f64,
    /// Whether this is a low contact order region (potential flexibility)
    pub is_low_contact_order: bool,
}

/// Results from contact order analysis
#[derive(Debug, Clone)]
pub struct ContactOrderResult {
    pub residue_metrics: Vec<ResidueContactOrder>,
    /// Global relative contact order of the structure
    pub global_rco: f64,
    pub total_contacts: usize,
    pub total_residues: usize,
}

//=============================================================================
// CONTACT ORDER ANALYZER
//=============================================================================

/// Contact Order Analyzer for identifying flexible regions
pub struct ContactOrderAnalyzer {
    /// Distance cutoff for contacts
    pub contact_distance: f64,
    /// Minimum sequence separation
    pub min_separation: i32,
    /// Window size for local analysis
    pub window_size: usize,
    /// Threshold for flagging low contact order
    pub low_co_threshold: f64,
}

impl Default for ContactOrderAnalyzer {
    fn default() -> Self {
        Self {
            contact_distance: CONTACT_DISTANCE,
            min_separation: MIN_SEQUENCE_SEPARATION,
            window_size: LOCAL_WINDOW_SIZE,
            low_co_threshold: LOW_CONTACT_ORDER_THRESHOLD,
        }
    }
}

impl ContactOrderAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze protein structure for contact order
    pub fn analyze(&self, atoms: &[Atom]) -> ContactOrderResult {
        // Extract CA atoms for analysis
        let ca_atoms: Vec<&Atom> = atoms
            .iter()
            .filter(|a| a.name == "CA" && !a.is_hetatm)
            .collect();

        if ca_atoms.len() < 5 {
            return ContactOrderResult {
                residue_metrics: Vec::new(),
                global_rco: 0.0,
                total_contacts: 0,
                total_residues: ca_atoms.len(),
            };
        }

        log::debug!("[CONTACT_ORDER] Analyzing {} CA atoms", ca_atoms.len());

        // Build contact map
        let contacts = self.build_contact_map(&ca_atoms);

        // Calculate per-residue contact order
        let residue_metrics = self.calculate_residue_metrics(&ca_atoms, &contacts);

        // Calculate global relative contact order
        let global_rco = self.calculate_global_rco(&ca_atoms, &contacts);

        log::debug!(
            "[CONTACT_ORDER] Global RCO = {:.3}, {} contacts",
            global_rco,
            contacts.len()
        );

        ContactOrderResult {
            residue_metrics,
            global_rco,
            total_contacts: contacts.len(),
            total_residues: ca_atoms.len(),
        }
    }

    /// Build contact map: list of (i, j, distance) tuples
    fn build_contact_map(&self, ca_atoms: &[&Atom]) -> Vec<(usize, usize, f64)> {
        let mut contacts = Vec::new();
        let cutoff_sq = self.contact_distance * self.contact_distance;

        for i in 0..ca_atoms.len() {
            for j in (i + 1)..ca_atoms.len() {
                // Check sequence separation
                let seq_sep = (ca_atoms[j].residue_seq - ca_atoms[i].residue_seq).abs();
                if seq_sep < self.min_separation {
                    continue;
                }

                // Check distance
                let dx = ca_atoms[j].coord[0] - ca_atoms[i].coord[0];
                let dy = ca_atoms[j].coord[1] - ca_atoms[i].coord[1];
                let dz = ca_atoms[j].coord[2] - ca_atoms[i].coord[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < cutoff_sq {
                    contacts.push((i, j, dist_sq.sqrt()));
                }
            }
        }

        contacts
    }

    /// Calculate per-residue contact order metrics
    fn calculate_residue_metrics(
        &self,
        ca_atoms: &[&Atom],
        contacts: &[(usize, usize, f64)],
    ) -> Vec<ResidueContactOrder> {
        let n = ca_atoms.len();
        let mut residue_contacts: Vec<Vec<(usize, i32)>> = vec![Vec::new(); n];

        // Group contacts by residue
        for &(i, j, _) in contacts {
            let seq_sep = (ca_atoms[j].residue_seq - ca_atoms[i].residue_seq).abs();
            residue_contacts[i].push((j, seq_sep));
            residue_contacts[j].push((i, seq_sep));
        }

        // Calculate metrics per residue
        let mut metrics = Vec::with_capacity(n);
        let half_window = self.window_size / 2;

        for i in 0..n {
            let atom = ca_atoms[i];
            let num_contacts = residue_contacts[i].len();

            // Mean sequence separation of contacts
            let mean_sep = if num_contacts > 0 {
                residue_contacts[i].iter().map(|(_, sep)| *sep as f64).sum::<f64>()
                    / num_contacts as f64
            } else {
                0.0
            };

            // Local contact order: average contact order in a window around this residue
            let window_start = i.saturating_sub(half_window);
            let window_end = (i + half_window + 1).min(n);
            let window_len = window_end - window_start;

            let local_co = if window_len > 0 {
                let mut window_contacts = 0usize;
                let mut window_sep_sum = 0i64;

                for j in window_start..window_end {
                    for &(_, sep) in &residue_contacts[j] {
                        window_contacts += 1;
                        window_sep_sum += sep as i64;
                    }
                }

                if window_contacts > 0 {
                    (window_sep_sum as f64 / window_contacts as f64) / n as f64
                } else {
                    0.0
                }
            } else {
                0.0
            };

            let is_low_co = local_co < self.low_co_threshold && local_co > 0.0;

            metrics.push(ResidueContactOrder {
                residue_seq: atom.residue_seq,
                chain_id: atom.chain_id,
                local_contact_order: local_co,
                num_contacts,
                mean_sequence_separation: mean_sep,
                is_low_contact_order: is_low_co,
            });
        }

        metrics
    }

    /// Calculate global relative contact order
    fn calculate_global_rco(&self, ca_atoms: &[&Atom], contacts: &[(usize, usize, f64)]) -> f64 {
        if contacts.is_empty() || ca_atoms.len() < 2 {
            return 0.0;
        }

        let n = ca_atoms.len() as f64;
        let total_sep: i64 = contacts
            .iter()
            .map(|&(i, j, _)| (ca_atoms[j].residue_seq - ca_atoms[i].residue_seq).abs() as i64)
            .sum();

        total_sep as f64 / (contacts.len() as f64 * n)
    }

    /// Get residues with low contact order (potential flexibility points)
    pub fn get_low_contact_order_residues(&self, result: &ContactOrderResult) -> Vec<i32> {
        result
            .residue_metrics
            .iter()
            .filter(|m| m.is_low_contact_order)
            .map(|m| m.residue_seq)
            .collect()
    }

    /// Get contact order score for a specific residue (inverted: low CO = high score)
    pub fn get_residue_flexibility_score(
        &self,
        result: &ContactOrderResult,
        residue_seq: i32,
    ) -> Option<f64> {
        result
            .residue_metrics
            .iter()
            .find(|m| m.residue_seq == residue_seq)
            .map(|m| {
                // Invert: low contact order = high flexibility score
                if m.local_contact_order > 0.0 {
                    1.0 - (m.local_contact_order / result.global_rco).min(1.0)
                } else {
                    0.5 // Default for residues without contacts
                }
            })
    }
}

/// Convert contact order results to a residue -> flexibility score map
pub fn contact_order_to_score_map(result: &ContactOrderResult) -> HashMap<i32, f64> {
    let global_rco = result.global_rco.max(0.01);
    result
        .residue_metrics
        .iter()
        .map(|m| {
            let score = if m.local_contact_order > 0.0 {
                1.0 - (m.local_contact_order / global_rco).min(1.0)
            } else {
                0.5
            };
            (m.residue_seq, score)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ca_atom(serial: u32, residue_seq: i32, coord: [f64; 3]) -> Atom {
        Atom {
            serial,
            name: "CA".to_string(),
            residue_name: "ALA".to_string(),
            chain_id: 'A',
            residue_seq,
            insertion_code: None,
            coord,
            occupancy: 1.0,
            b_factor: 20.0,
            element: "C".to_string(),
            alt_loc: None,
            model: 1,
            is_hetatm: false,
            sasa: 0.0,
            hydrophobicity: 0.7,
            partial_charge: 0.0,
            is_surface: true,
            depth: 0.0,
            curvature: 0.0,
        }
    }

    #[test]
    fn test_analyzer_creation() {
        let analyzer = ContactOrderAnalyzer::new();
        assert_eq!(analyzer.contact_distance, CONTACT_DISTANCE);
    }

    #[test]
    fn test_linear_chain() {
        let analyzer = ContactOrderAnalyzer::new();

        // Linear chain - should have low contact order
        let atoms: Vec<Atom> = (0..20)
            .map(|i| make_ca_atom(i as u32, i as i32, [i as f64 * 3.8, 0.0, 0.0]))
            .collect();

        let result = analyzer.analyze(&atoms);
        assert_eq!(result.total_residues, 20);
        // Linear chain has no long-range contacts
        assert!(result.total_contacts < 20);
    }

    #[test]
    fn test_empty_input() {
        let analyzer = ContactOrderAnalyzer::new();
        let result = analyzer.analyze(&[]);
        assert_eq!(result.total_residues, 0);
    }
}

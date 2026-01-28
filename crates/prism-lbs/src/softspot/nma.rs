//! Normal Mode Analysis (NMA) for cryptic site detection
//!
//! Implements Anisotropic Network Model (ANM) to identify regions with high
//! conformational flexibility in low-frequency vibrational modes. These regions
//! are candidates for induced-fit cryptic binding sites.
//!
//! ## Scientific Basis
//!
//! - Low-frequency modes correspond to collective, functionally relevant motions
//! - Regions with high displacement in these modes can undergo conformational change
//! - This captures cryptic sites that have LOW B-factors but HIGH induced-fit potential
//!
//! ## References
//!
//! - Bahar et al. (1997) - Gaussian Network Model
//! - Atilgan et al. (2001) - Anisotropic Network Model
//! - Tobi & Bahar (2005) - Functional motions and binding sites

use crate::structure::Atom;
use super::lanczos::LanczosEigensolver;
use nalgebra::DMatrix;
use std::collections::HashMap;

//=============================================================================
// CONSTANTS
//=============================================================================

/// Cutoff distance (Å) for ANM spring connections
/// Typically 13-15 Å for coarse-grained models
pub const ANM_CUTOFF: f64 = 13.0;

/// Spring constant for ANM (arbitrary units, cancels in normalized analysis)
pub const SPRING_CONSTANT: f64 = 1.0;

/// Number of low-frequency modes to analyze (excluding 6 trivial modes)
pub const NUM_MODES_TO_ANALYZE: usize = 10;

/// Threshold for normalized mode displacement to flag as "mobile"
pub const MOBILITY_THRESHOLD: f64 = 1.5;

/// Weight of NMA signal in combined scoring
pub const NMA_WEIGHT: f64 = 0.25;

//=============================================================================
// TYPES
//=============================================================================

/// Per-residue NMA mobility score
#[derive(Debug, Clone)]
pub struct ResidueMobility {
    pub residue_seq: i32,
    pub chain_id: char,
    /// Mean squared displacement across low-frequency modes (normalized)
    pub mobility_score: f64,
    /// Collectivity: how distributed the motion is (0-1)
    pub collectivity: f64,
    /// Dominant mode index contributing to this residue's motion
    pub dominant_mode: usize,
}

/// Results from NMA analysis
#[derive(Debug, Clone)]
pub struct NmaResult {
    pub residue_mobilities: Vec<ResidueMobility>,
    pub num_modes_computed: usize,
    pub total_residues: usize,
}

//=============================================================================
// NMA ANALYZER
//=============================================================================

/// Normal Mode Analysis using Anisotropic Network Model
pub struct NmaAnalyzer {
    /// Cutoff distance for spring connections
    pub cutoff: f64,
    /// Number of modes to analyze
    pub num_modes: usize,
    /// Mobility threshold for flagging residues
    pub mobility_threshold: f64,
}

impl Default for NmaAnalyzer {
    fn default() -> Self {
        Self {
            cutoff: ANM_CUTOFF,
            num_modes: NUM_MODES_TO_ANALYZE,
            mobility_threshold: MOBILITY_THRESHOLD,
        }
    }
}

impl NmaAnalyzer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze protein structure for conformational mobility
    ///
    /// Returns mobility scores per residue based on low-frequency normal modes
    pub fn analyze(&self, atoms: &[Atom]) -> NmaResult {
        // Extract CA atoms for coarse-grained analysis
        let ca_atoms: Vec<&Atom> = atoms
            .iter()
            .filter(|a| a.name == "CA" && !a.is_hetatm)
            .collect();

        if ca_atoms.len() < 10 {
            log::warn!("[NMA] Too few CA atoms ({}) for reliable analysis", ca_atoms.len());
            return NmaResult {
                residue_mobilities: Vec::new(),
                num_modes_computed: 0,
                total_residues: ca_atoms.len(),
            };
        }

        log::debug!("[NMA] Analyzing {} CA atoms", ca_atoms.len());

        // Build Hessian matrix
        let hessian = self.build_hessian(&ca_atoms);

        // Compute eigenvalues and eigenvectors
        let (eigenvalues, eigenvectors) = self.compute_eigenmodes(&hessian, ca_atoms.len());

        // Calculate per-residue mobility from low-frequency modes
        let mobilities = self.calculate_mobilities(&ca_atoms, &eigenvalues, &eigenvectors);

        log::debug!("[NMA] Computed {} residue mobilities", mobilities.len());

        NmaResult {
            residue_mobilities: mobilities,
            num_modes_computed: eigenvalues.len().min(self.num_modes),
            total_residues: ca_atoms.len(),
        }
    }

    /// Build the ANM Hessian matrix
    ///
    /// The Hessian is a 3N x 3N matrix where N is the number of residues.
    /// For efficiency, we compute only what we need for the eigendecomposition.
    fn build_hessian(&self, ca_atoms: &[&Atom]) -> Vec<Vec<f64>> {
        let n = ca_atoms.len();
        let dim = 3 * n;
        let mut hessian = vec![vec![0.0; dim]; dim];
        let cutoff_sq = self.cutoff * self.cutoff;

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = ca_atoms[j].coord[0] - ca_atoms[i].coord[0];
                let dy = ca_atoms[j].coord[1] - ca_atoms[i].coord[1];
                let dz = ca_atoms[j].coord[2] - ca_atoms[i].coord[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                if dist_sq < cutoff_sq && dist_sq > 0.01 {
                    // Spring constant scaled by distance
                    let k = SPRING_CONSTANT / dist_sq;

                    // Off-diagonal 3x3 block
                    let dxyz = [dx, dy, dz];
                    for a in 0..3 {
                        for b in 0..3 {
                            let val = -k * dxyz[a] * dxyz[b] / dist_sq;
                            hessian[3 * i + a][3 * j + b] = val;
                            hessian[3 * j + b][3 * i + a] = val;
                            // Diagonal contributions
                            hessian[3 * i + a][3 * i + b] -= val;
                            hessian[3 * j + a][3 * j + b] -= val;
                        }
                    }
                }
            }
        }

        hessian
    }

    /// Compute eigenvalues and eigenvectors using Lanczos algorithm
    ///
    /// Uses the robust Lanczos eigensolver with full reorthogonalization
    /// for accurate computation of low-frequency modes.
    fn compute_eigenmodes(&self, hessian: &[Vec<f64>], n_residues: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
        let dim = 3 * n_residues;

        // Convert Vec<Vec<f64>> to nalgebra DMatrix
        let mut matrix_data = Vec::with_capacity(dim * dim);
        for row in hessian.iter().take(dim) {
            for j in 0..dim {
                matrix_data.push(*row.get(j).unwrap_or(&0.0));
            }
        }
        let hessian_matrix = DMatrix::from_row_slice(dim, dim, &matrix_data);

        // Request enough modes to skip 6 trivial modes + get num_modes non-trivial
        let k = (self.num_modes + 6).min(dim / 2).max(7);

        // Use Lanczos eigensolver with full reorthogonalization
        let solver = LanczosEigensolver {
            max_iter: 300,
            tol: 1e-8,
            num_lanczos_vectors: 50.min(dim),
            seed: Some(42),
        };

        let result = solver.compute_smallest(&hessian_matrix, k);

        // Filter out trivial modes (eigenvalue < 1e-6)
        let mut eigenvalues = Vec::with_capacity(self.num_modes);
        let mut eigenvectors = Vec::with_capacity(self.num_modes);

        for (eigenvalue, eigenvector) in result.eigenvalues.iter().zip(result.eigenvectors.iter()) {
            // Skip near-zero eigenvalues (trivial translation/rotation modes)
            if *eigenvalue > 1e-6 {
                eigenvalues.push(*eigenvalue);
                eigenvectors.push(eigenvector.iter().cloned().collect());

                if eigenvalues.len() >= self.num_modes {
                    break;
                }
            }
        }

        log::debug!("[NMA] Lanczos found {} non-trivial modes (converged: {})",
                   eigenvalues.len(), result.converged);

        (eigenvalues, eigenvectors)
    }

    /// Calculate per-residue mobility from eigenmodes
    fn calculate_mobilities(
        &self,
        ca_atoms: &[&Atom],
        eigenvalues: &[f64],
        eigenvectors: &[Vec<f64>],
    ) -> Vec<ResidueMobility> {
        let n = ca_atoms.len();
        let mut mobilities = Vec::with_capacity(n);

        // Calculate mean squared displacement for each residue
        for (i, atom) in ca_atoms.iter().enumerate() {
            let mut total_mobility = 0.0;
            let mut mode_contributions = Vec::new();

            for (mode_idx, (eigenvalue, eigenvector)) in eigenvalues.iter().zip(eigenvectors.iter()).enumerate() {
                if *eigenvalue <= 0.0 {
                    continue;
                }

                // Mean squared displacement contribution from this mode
                // MSD ~ 1/eigenvalue * |displacement|^2
                let dx = eigenvector.get(3 * i).copied().unwrap_or(0.0);
                let dy = eigenvector.get(3 * i + 1).copied().unwrap_or(0.0);
                let dz = eigenvector.get(3 * i + 2).copied().unwrap_or(0.0);
                let displacement_sq = dx * dx + dy * dy + dz * dz;

                let contribution = displacement_sq / eigenvalue;
                total_mobility += contribution;
                mode_contributions.push((mode_idx, contribution));
            }

            // Find dominant mode
            let dominant_mode = mode_contributions
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| *idx)
                .unwrap_or(0);

            // Calculate collectivity for this residue (how spread out the motion is)
            let collectivity = self.calculate_collectivity(i, eigenvectors, eigenvalues);

            mobilities.push(ResidueMobility {
                residue_seq: atom.residue_seq,
                chain_id: atom.chain_id,
                mobility_score: total_mobility,
                collectivity,
                dominant_mode,
            });
        }

        // Normalize mobility scores
        let max_mobility = mobilities
            .iter()
            .map(|m| m.mobility_score)
            .fold(0.0_f64, |a, b| a.max(b));

        if max_mobility > 0.0 {
            for m in &mut mobilities {
                m.mobility_score /= max_mobility;
            }
        }

        mobilities
    }

    /// Calculate collectivity - how distributed the motion is across residues
    fn calculate_collectivity(
        &self,
        _residue_idx: usize,
        eigenvectors: &[Vec<f64>],
        eigenvalues: &[f64],
    ) -> f64 {
        if eigenvectors.is_empty() || eigenvalues.is_empty() {
            return 0.0;
        }

        // Use the first non-trivial mode for collectivity
        let first_mode = &eigenvectors[0];
        let n_residues = first_mode.len() / 3;

        if n_residues == 0 {
            return 0.0;
        }

        // Calculate participation ratio
        let mut sum_sq = 0.0;
        let mut sum_fourth = 0.0;

        for i in 0..n_residues {
            let dx = first_mode.get(3 * i).copied().unwrap_or(0.0);
            let dy = first_mode.get(3 * i + 1).copied().unwrap_or(0.0);
            let dz = first_mode.get(3 * i + 2).copied().unwrap_or(0.0);
            let disp_sq = dx * dx + dy * dy + dz * dz;
            sum_sq += disp_sq;
            sum_fourth += disp_sq * disp_sq;
        }

        if sum_fourth > 0.0 {
            (sum_sq * sum_sq) / (n_residues as f64 * sum_fourth)
        } else {
            0.0
        }
    }

    /// Get residues with high mobility (potential cryptic site residues)
    pub fn get_mobile_residues(&self, result: &NmaResult) -> Vec<i32> {
        result
            .residue_mobilities
            .iter()
            .filter(|m| m.mobility_score > self.mobility_threshold / 3.0) // Normalized threshold
            .map(|m| m.residue_seq)
            .collect()
    }

    /// Get mobility score for a specific residue
    pub fn get_residue_mobility(&self, result: &NmaResult, residue_seq: i32) -> Option<f64> {
        result
            .residue_mobilities
            .iter()
            .find(|m| m.residue_seq == residue_seq)
            .map(|m| m.mobility_score)
    }
}

/// Convert NMA results to a residue -> score map for integration
pub fn nma_to_score_map(result: &NmaResult) -> HashMap<i32, f64> {
    result
        .residue_mobilities
        .iter()
        .map(|m| (m.residue_seq, m.mobility_score))
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
    fn test_nma_analyzer_creation() {
        let analyzer = NmaAnalyzer::new();
        assert_eq!(analyzer.cutoff, ANM_CUTOFF);
        assert_eq!(analyzer.num_modes, NUM_MODES_TO_ANALYZE);
    }

    #[test]
    fn test_nma_small_chain() {
        let analyzer = NmaAnalyzer::new();

        // Create a small linear chain
        let atoms: Vec<Atom> = (0..15)
            .map(|i| make_ca_atom(i as u32, i as i32, [i as f64 * 3.8, 0.0, 0.0]))
            .collect();

        let result = analyzer.analyze(&atoms);

        assert!(result.total_residues > 0);
        // Terminal residues should have higher mobility in a linear chain
    }

    #[test]
    fn test_empty_input() {
        let analyzer = NmaAnalyzer::new();
        let result = analyzer.analyze(&[]);
        assert_eq!(result.total_residues, 0);
        assert!(result.residue_mobilities.is_empty());
    }
}

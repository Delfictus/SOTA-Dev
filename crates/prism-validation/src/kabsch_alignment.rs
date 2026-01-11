//! Kabsch Alignment for Accurate Displacement Computation
//!
//! This module provides structural alignment and displacement computation
//! for the PRISM-Delta Blind Validation Pipeline (Phase 4).
//!
//! # Key Functions
//!
//! - `kabsch_align`: Align mobile structure onto reference using optimal rotation
//! - `align_and_compute_displacement`: Align and compute per-atom displacements
//! - `compute_rmsf`: Compute RMSF from ensemble of displacements
//! - `compute_rmsd`: Compute RMSD between two structures
//!
//! # Why Kabsch Alignment?
//!
//! When comparing conformations from molecular dynamics or HMC sampling,
//! we must remove rigid-body motion (translation + rotation) to accurately
//! measure internal conformational changes. The Kabsch algorithm finds the
//! optimal rotation matrix that minimizes RMSD between two structures.
//!
//! # Usage
//!
//! ```rust,ignore
//! use prism_validation::kabsch_alignment::{align_and_compute_displacement, compute_rmsf};
//!
//! // Align conformation to reference and get displacements
//! let (aligned, displacement) = align_and_compute_displacement(&reference, &conformation);
//!
//! // Compute RMSF from ensemble of displacements
//! let rmsf = compute_rmsf(&all_displacements);
//! ```

/// Result of alignment with computed displacement
#[derive(Debug, Clone)]
pub struct AlignmentResult {
    /// Aligned coordinates (in reference frame)
    pub aligned_coords: Vec<[f32; 3]>,
    /// Per-atom displacement vectors (aligned - reference)
    pub displacements: Vec<[f32; 3]>,
    /// RMSD after alignment
    pub rmsd: f64,
    /// Rotation matrix applied (3x3)
    pub rotation: [[f64; 3]; 3],
    /// Translation vector (reference centroid)
    pub translation: [f64; 3],
}

/// Compute centroid (center of mass) of a set of coordinates
pub fn compute_centroid(coords: &[[f32; 3]]) -> [f64; 3] {
    if coords.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    let n = coords.len() as f64;
    let mut centroid = [0.0f64; 3];

    for pos in coords {
        centroid[0] += pos[0] as f64;
        centroid[1] += pos[1] as f64;
        centroid[2] += pos[2] as f64;
    }

    centroid[0] /= n;
    centroid[1] /= n;
    centroid[2] /= n;

    centroid
}

/// Compute RMSD between two structures (must be same size, pre-aligned)
pub fn compute_rmsd(coords1: &[[f32; 3]], coords2: &[[f32; 3]]) -> f64 {
    if coords1.len() != coords2.len() || coords1.is_empty() {
        return f64::INFINITY;
    }

    let n = coords1.len() as f64;
    let sum_sq: f64 = coords1.iter().zip(coords2.iter())
        .map(|(a, b)| {
            let dx = a[0] as f64 - b[0] as f64;
            let dy = a[1] as f64 - b[1] as f64;
            let dz = a[2] as f64 - b[2] as f64;
            dx * dx + dy * dy + dz * dz
        })
        .sum();

    (sum_sq / n).sqrt()
}

/// Kabsch algorithm: compute optimal rotation matrix to align mobile onto reference
///
/// Uses the SVD-based method for robustness:
/// 1. Compute correlation matrix H = mobile^T * reference
/// 2. SVD: H = U * S * V^T
/// 3. R = V * U^T
/// 4. Handle reflection case (det(R) = -1)
fn compute_rotation_matrix(h: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    use nalgebra::Matrix3;

    // Convert to nalgebra matrix
    let h_mat = Matrix3::new(
        h[0][0], h[0][1], h[0][2],
        h[1][0], h[1][1], h[1][2],
        h[2][0], h[2][1], h[2][2],
    );

    // Compute SVD
    let svd = h_mat.svd(true, true);

    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();

    // Compute R = V * U^T
    let v = v_t.transpose();
    let u_t = u.transpose();
    let mut r = v * u_t;

    // Handle reflection case: if det(R) = -1, flip the sign of the last column of V
    let det = r.determinant();
    if det < 0.0 {
        // Flip the sign of the column of V corresponding to the smallest singular value
        let mut v_corrected = v;
        for i in 0..3 {
            v_corrected[(i, 2)] = -v_corrected[(i, 2)];
        }
        r = v_corrected * u_t;
    }

    // Convert back to array
    [
        [r[(0, 0)], r[(0, 1)], r[(0, 2)]],
        [r[(1, 0)], r[(1, 1)], r[(1, 2)]],
        [r[(2, 0)], r[(2, 1)], r[(2, 2)]],
    ]
}

/// Apply rotation matrix to a point
fn rotate_point(point: &[f64; 3], rotation: &[[f64; 3]; 3]) -> [f64; 3] {
    [
        rotation[0][0] * point[0] + rotation[0][1] * point[1] + rotation[0][2] * point[2],
        rotation[1][0] * point[0] + rotation[1][1] * point[1] + rotation[1][2] * point[2],
        rotation[2][0] * point[0] + rotation[2][1] * point[1] + rotation[2][2] * point[2],
    ]
}

/// Align mobile structure onto reference using Kabsch algorithm
///
/// # Arguments
/// * `reference` - Target structure (stays fixed)
/// * `mobile` - Structure to align (will be rotated/translated)
///
/// # Returns
/// Aligned coordinates in reference frame (as f64 for precision)
pub fn kabsch_align(reference: &[[f32; 3]], mobile: &[[f32; 3]]) -> Vec<[f64; 3]> {
    if mobile.len() != reference.len() || mobile.is_empty() {
        return mobile.iter()
            .map(|p| [p[0] as f64, p[1] as f64, p[2] as f64])
            .collect();
    }

    let n = mobile.len();

    // Step 1: Center both structures
    let mobile_center = compute_centroid(mobile);
    let ref_center = compute_centroid(reference);

    // Center the structures
    let mobile_centered: Vec<[f64; 3]> = mobile.iter()
        .map(|p| [
            p[0] as f64 - mobile_center[0],
            p[1] as f64 - mobile_center[1],
            p[2] as f64 - mobile_center[2],
        ])
        .collect();

    let ref_centered: Vec<[f64; 3]> = reference.iter()
        .map(|p| [
            p[0] as f64 - ref_center[0],
            p[1] as f64 - ref_center[1],
            p[2] as f64 - ref_center[2],
        ])
        .collect();

    // Step 2: Compute correlation matrix H = mobile^T * reference
    let mut h = [[0.0f64; 3]; 3];
    for i in 0..n {
        for j in 0..3 {
            for k in 0..3 {
                h[j][k] += mobile_centered[i][j] * ref_centered[i][k];
            }
        }
    }

    // Step 3: Compute optimal rotation matrix
    let rotation = compute_rotation_matrix(&h);

    // Step 4: Apply rotation and translate to reference frame
    mobile_centered.iter()
        .map(|p| {
            let rotated = rotate_point(p, &rotation);
            [
                rotated[0] + ref_center[0],
                rotated[1] + ref_center[1],
                rotated[2] + ref_center[2],
            ]
        })
        .collect()
}

/// Align conformation to reference and compute per-atom displacements
///
/// This is the primary function for the blind validation pipeline.
/// It performs Kabsch alignment and returns both aligned coordinates
/// and the displacement vectors from reference.
///
/// # Arguments
/// * `reference` - Relaxed apo structure (from AMBER minimization)
/// * `conformation` - Sampled conformation (from PRISM-NOVA ensemble)
///
/// # Returns
/// Tuple of (aligned coordinates, displacement vectors)
pub fn align_and_compute_displacement(
    reference: &[[f32; 3]],
    conformation: &[[f32; 3]],
) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    if conformation.len() != reference.len() || conformation.is_empty() {
        return (conformation.to_vec(), vec![[0.0, 0.0, 0.0]; conformation.len()]);
    }

    // Perform Kabsch alignment (returns f64 for precision)
    let aligned_f64 = kabsch_align(reference, conformation);

    // Compute displacements and convert back to f32
    let mut aligned = Vec::with_capacity(conformation.len());
    let mut displacement = Vec::with_capacity(conformation.len());

    for (i, aligned_pos) in aligned_f64.iter().enumerate() {
        let ref_pos = &reference[i];

        aligned.push([
            aligned_pos[0] as f32,
            aligned_pos[1] as f32,
            aligned_pos[2] as f32,
        ]);

        displacement.push([
            (aligned_pos[0] - ref_pos[0] as f64) as f32,
            (aligned_pos[1] - ref_pos[1] as f64) as f32,
            (aligned_pos[2] - ref_pos[2] as f64) as f32,
        ]);
    }

    (aligned, displacement)
}

/// Full alignment result with all metadata
pub fn align_with_full_result(
    reference: &[[f32; 3]],
    conformation: &[[f32; 3]],
) -> AlignmentResult {
    let (aligned, displacements) = align_and_compute_displacement(reference, conformation);

    // Compute RMSD
    let rmsd = if !aligned.is_empty() {
        let n = aligned.len() as f64;
        let sum_sq: f64 = displacements.iter()
            .map(|d| (d[0] as f64).powi(2) + (d[1] as f64).powi(2) + (d[2] as f64).powi(2))
            .sum();
        (sum_sq / n).sqrt()
    } else {
        0.0
    };

    // Recompute rotation and translation for metadata
    let ref_center = compute_centroid(reference);
    let mobile_center = compute_centroid(conformation);

    let mobile_centered: Vec<[f64; 3]> = conformation.iter()
        .map(|p| [
            p[0] as f64 - mobile_center[0],
            p[1] as f64 - mobile_center[1],
            p[2] as f64 - mobile_center[2],
        ])
        .collect();

    let ref_centered: Vec<[f64; 3]> = reference.iter()
        .map(|p| [
            p[0] as f64 - ref_center[0],
            p[1] as f64 - ref_center[1],
            p[2] as f64 - ref_center[2],
        ])
        .collect();

    let mut h = [[0.0f64; 3]; 3];
    for i in 0..conformation.len() {
        for j in 0..3 {
            for k in 0..3 {
                h[j][k] += mobile_centered[i][j] * ref_centered[i][k];
            }
        }
    }

    let rotation = compute_rotation_matrix(&h);

    AlignmentResult {
        aligned_coords: aligned,
        displacements,
        rmsd,
        rotation,
        translation: ref_center,
    }
}

/// Compute RMSF from an ensemble of displacement vectors
///
/// RMSF (Root Mean Square Fluctuation) measures the average deviation
/// of each atom from its mean position across the ensemble.
///
/// For each atom i:
///   RMSF[i] = sqrt(mean(|displacement|^2))
///
/// # Arguments
/// * `displacements` - Vector of displacement vectors per conformation
///                     Shape: [n_conformations][n_atoms][3]
///
/// # Returns
/// Per-atom RMSF values in Angstroms
pub fn compute_rmsf(displacements: &[Vec<[f32; 3]>]) -> Vec<f64> {
    if displacements.is_empty() || displacements[0].is_empty() {
        return Vec::new();
    }

    let n_atoms = displacements[0].len();
    let n_conf = displacements.len() as f64;

    (0..n_atoms).map(|i| {
        let mean_sq: f64 = displacements.iter()
            .map(|d| {
                if i < d.len() {
                    let dx = d[i][0] as f64;
                    let dy = d[i][1] as f64;
                    let dz = d[i][2] as f64;
                    dx*dx + dy*dy + dz*dz
                } else {
                    0.0
                }
            })
            .sum::<f64>() / n_conf;
        mean_sq.sqrt()
    }).collect()
}

/// Compute RMSF from f64 displacements (for higher precision)
pub fn compute_rmsf_f64(displacements: &[Vec<[f64; 3]>]) -> Vec<f64> {
    if displacements.is_empty() || displacements[0].is_empty() {
        return Vec::new();
    }

    let n_atoms = displacements[0].len();
    let n_conf = displacements.len() as f64;

    (0..n_atoms).map(|i| {
        let mean_sq: f64 = displacements.iter()
            .map(|d| {
                if i < d.len() {
                    d[i][0]*d[i][0] + d[i][1]*d[i][1] + d[i][2]*d[i][2]
                } else {
                    0.0
                }
            })
            .sum::<f64>() / n_conf;
        mean_sq.sqrt()
    }).collect()
}

/// Compute mean structure from ensemble
///
/// Useful for finding the average conformation after alignment.
pub fn compute_mean_structure(aligned_ensemble: &[Vec<[f32; 3]>]) -> Vec<[f32; 3]> {
    if aligned_ensemble.is_empty() || aligned_ensemble[0].is_empty() {
        return Vec::new();
    }

    let n_atoms = aligned_ensemble[0].len();
    let n_conf = aligned_ensemble.len() as f32;

    let mut mean = vec![[0.0f32; 3]; n_atoms];

    for conf in aligned_ensemble {
        for (i, pos) in conf.iter().enumerate() {
            if i < n_atoms {
                mean[i][0] += pos[0];
                mean[i][1] += pos[1];
                mean[i][2] += pos[2];
            }
        }
    }

    for pos in &mut mean {
        pos[0] /= n_conf;
        pos[1] /= n_conf;
        pos[2] /= n_conf;
    }

    mean
}

/// Compute displacement variance per atom (useful for B-factor estimation)
///
/// Returns both RMSF and variance for each atom.
pub fn compute_displacement_statistics(
    displacements: &[Vec<[f32; 3]>],
) -> (Vec<f64>, Vec<f64>) {
    let rmsf = compute_rmsf(displacements);
    let variance: Vec<f64> = rmsf.iter().map(|r| r * r).collect();
    (rmsf, variance)
}

/// Batch align an entire ensemble to a reference structure
///
/// # Arguments
/// * `reference` - Reference structure for alignment
/// * `ensemble` - Vector of conformations to align
///
/// # Returns
/// Tuple of (aligned ensemble, all displacements)
pub fn align_ensemble(
    reference: &[[f32; 3]],
    ensemble: &[Vec<[f32; 3]>],
) -> (Vec<Vec<[f32; 3]>>, Vec<Vec<[f32; 3]>>) {
    let mut aligned_ensemble = Vec::with_capacity(ensemble.len());
    let mut all_displacements = Vec::with_capacity(ensemble.len());

    for conf in ensemble {
        let (aligned, displacement) = align_and_compute_displacement(reference, conf);
        aligned_ensemble.push(aligned);
        all_displacements.push(displacement);
    }

    (aligned_ensemble, all_displacements)
}

/// Compute pairwise RMSD matrix for an ensemble
///
/// Useful for clustering analysis of conformations.
pub fn compute_pairwise_rmsd(ensemble: &[Vec<[f32; 3]>]) -> Vec<Vec<f64>> {
    let n = ensemble.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in i+1..n {
            // Align j to i and compute RMSD
            let aligned = kabsch_align(&ensemble[i], &ensemble[j]);

            let rmsd: f64 = if !aligned.is_empty() {
                let sum_sq: f64 = aligned.iter().enumerate()
                    .map(|(k, pos)| {
                        let ref_pos = &ensemble[i][k];
                        let dx = pos[0] - ref_pos[0] as f64;
                        let dy = pos[1] - ref_pos[1] as f64;
                        let dz = pos[2] - ref_pos[2] as f64;
                        dx*dx + dy*dy + dz*dz
                    })
                    .sum();
                (sum_sq / aligned.len() as f64).sqrt()
            } else {
                0.0
            };

            matrix[i][j] = rmsd;
            matrix[j][i] = rmsd;
        }
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_centroid() {
        let coords = vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ];
        let centroid = compute_centroid(&coords);
        assert!((centroid[0] - 0.6667).abs() < 0.01);
        assert!((centroid[1] - 0.6667).abs() < 0.01);
        assert!((centroid[2] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_identity_alignment() {
        // Same structure should have zero RMSD after alignment
        let coords = vec![
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
        ];
        let (aligned, displacement) = align_and_compute_displacement(&coords, &coords);

        // Displacements should be near zero
        for d in &displacement {
            assert!((d[0]).abs() < 1e-5);
            assert!((d[1]).abs() < 1e-5);
            assert!((d[2]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_translated_alignment() {
        // Translated structure should align perfectly
        let reference = vec![
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [7.6, 0.0, 0.0],
        ];
        let translated = vec![
            [10.0, 10.0, 10.0],
            [13.8, 10.0, 10.0],
            [17.6, 10.0, 10.0],
        ];

        let (aligned, displacement) = align_and_compute_displacement(&reference, &translated);

        // After alignment, displacements should be near zero
        for d in &displacement {
            assert!((d[0]).abs() < 1e-4, "dx = {}", d[0]);
            assert!((d[1]).abs() < 1e-4, "dy = {}", d[1]);
            assert!((d[2]).abs() < 1e-4, "dz = {}", d[2]);
        }
    }

    #[test]
    fn test_rmsf_calculation() {
        // Two conformations with known displacement
        let displacements = vec![
            vec![[0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]],
            vec![[-0.5, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.5, 0.0, 0.0]],
        ];

        let rmsf = compute_rmsf(&displacements);

        // RMSF should be sqrt(mean(x^2)) = sqrt((0.5^2 + 0.5^2)/2) = 0.5
        assert!((rmsf[0] - 0.5).abs() < 0.01, "RMSF[0] = {}", rmsf[0]);
        assert!((rmsf[1] - 1.0).abs() < 0.01, "RMSF[1] = {}", rmsf[1]);
        assert!((rmsf[2] - 1.5).abs() < 0.01, "RMSF[2] = {}", rmsf[2]);
    }

    #[test]
    fn test_rotated_alignment() {
        // 90-degree rotation around Z axis
        let reference = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        // Rotated 90 degrees: (x,y) -> (-y, x)
        let rotated = vec![
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ];

        let result = align_with_full_result(&reference, &rotated);

        // RMSD should be near zero after alignment
        assert!(result.rmsd < 0.01, "RMSD = {}", result.rmsd);
    }

    #[test]
    fn test_align_ensemble() {
        // Test that align_ensemble produces correct output structure
        // and that RMSF reflects residual motion after alignment
        let reference = vec![
            [0.0, 0.0, 0.0],
            [3.8, 0.0, 0.0],
            [1.9, 3.3, 0.0],  // Roughly equilateral triangle
        ];

        // Ensemble with small perturbations (simulating thermal motion)
        let ensemble = vec![
            vec![
                [0.1, 0.05, 0.0],    // Small displacement
                [3.75, 0.1, 0.0],
                [1.85, 3.35, 0.1],
            ],
            vec![
                [-0.1, -0.05, 0.0],  // Opposite displacement
                [3.85, -0.1, 0.0],
                [1.95, 3.25, -0.1],
            ],
        ];

        let (aligned, displacements) = align_ensemble(&reference, &ensemble);

        // Verify output structure
        assert_eq!(aligned.len(), 2, "Should return 2 aligned conformations");
        assert_eq!(displacements.len(), 2, "Should return 2 displacement sets");
        assert_eq!(aligned[0].len(), 3, "Each aligned conformation should have 3 atoms");
        assert_eq!(displacements[0].len(), 3, "Each displacement set should have 3 atoms");

        // Compute RMSF
        let rmsf = compute_rmsf(&displacements);
        assert_eq!(rmsf.len(), 3, "RMSF should have 3 values");

        // All RMSF values should be small (< 0.5 Å) after alignment
        // since the perturbations were small
        for (i, &r) in rmsf.iter().enumerate() {
            assert!(r < 0.5, "RMSF[{}] = {} should be < 0.5 Å for small perturbations", i, r);
            assert!(r >= 0.0, "RMSF[{}] = {} should be non-negative", i, r);
        }

        // Verify mean RMSF is reasonable (not zero, not huge)
        let mean_rmsf: f64 = rmsf.iter().sum::<f64>() / rmsf.len() as f64;
        assert!(mean_rmsf > 0.01, "Mean RMSF = {} should be > 0.01 (real motion)", mean_rmsf);
        assert!(mean_rmsf < 0.3, "Mean RMSF = {} should be < 0.3 (small motion)", mean_rmsf);
    }
}

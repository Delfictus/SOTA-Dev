//! Metric computation utilities for validation benchmarks
//!
//! Provides functions for computing:
//! - RMSD and related structural metrics
//! - RMSF and dynamics metrics
//! - Topological metrics (TDA)
//! - Statistical comparisons

use nalgebra::{Matrix3, Vector3};

/// Compute RMSD between two sets of coordinates
pub fn compute_rmsd(coords1: &[f32], coords2: &[f32]) -> f32 {
    if coords1.len() != coords2.len() || coords1.is_empty() {
        return f32::NAN;
    }

    let n_atoms = coords1.len() / 3;
    let mut sum_sq = 0.0f64;

    for i in 0..n_atoms {
        let dx = (coords1[i * 3] - coords2[i * 3]) as f64;
        let dy = (coords1[i * 3 + 1] - coords2[i * 3 + 1]) as f64;
        let dz = (coords1[i * 3 + 2] - coords2[i * 3 + 2]) as f64;
        sum_sq += dx * dx + dy * dy + dz * dz;
    }

    (sum_sq / n_atoms as f64).sqrt() as f32
}

/// Compute RMSD after optimal superposition (Kabsch algorithm)
pub fn compute_rmsd_aligned(coords1: &[f32], coords2: &[f32]) -> f32 {
    if coords1.len() != coords2.len() || coords1.is_empty() {
        return f32::NAN;
    }

    let n_atoms = coords1.len() / 3;

    // Convert to nalgebra vectors
    let points1: Vec<Vector3<f64>> = (0..n_atoms)
        .map(|i| {
            Vector3::new(
                coords1[i * 3] as f64,
                coords1[i * 3 + 1] as f64,
                coords1[i * 3 + 2] as f64,
            )
        })
        .collect();

    let points2: Vec<Vector3<f64>> = (0..n_atoms)
        .map(|i| {
            Vector3::new(
                coords2[i * 3] as f64,
                coords2[i * 3 + 1] as f64,
                coords2[i * 3 + 2] as f64,
            )
        })
        .collect();

    // Compute centroids
    let centroid1: Vector3<f64> =
        points1.iter().fold(Vector3::zeros(), |a, b| a + b) / n_atoms as f64;
    let centroid2: Vector3<f64> =
        points2.iter().fold(Vector3::zeros(), |a, b| a + b) / n_atoms as f64;

    // Center the points
    let centered1: Vec<Vector3<f64>> = points1.iter().map(|p| p - centroid1).collect();
    let centered2: Vec<Vector3<f64>> = points2.iter().map(|p| p - centroid2).collect();

    // Compute covariance matrix H
    let mut h = Matrix3::zeros();
    for i in 0..n_atoms {
        h += centered1[i] * centered2[i].transpose();
    }

    // SVD of H
    let svd = h.svd(true, true);
    let u = svd.u.unwrap();
    let v_t = svd.v_t.unwrap();

    // Compute rotation matrix R = V * U^T
    let mut r = v_t.transpose() * u.transpose();

    // Handle reflection case
    if r.determinant() < 0.0 {
        let mut v_t_corrected = v_t;
        v_t_corrected.row_mut(2).scale_mut(-1.0);
        r = v_t_corrected.transpose() * u.transpose();
    }

    // Apply rotation and compute RMSD
    let mut sum_sq = 0.0;
    for i in 0..n_atoms {
        let rotated = r * centered2[i];
        let diff = centered1[i] - rotated;
        sum_sq += diff.norm_squared();
    }

    (sum_sq / n_atoms as f64).sqrt() as f32
}

/// Compute RMSF (Root Mean Square Fluctuation) from trajectory
pub fn compute_rmsf(trajectory: &[Vec<f32>]) -> Vec<f32> {
    if trajectory.is_empty() || trajectory[0].is_empty() {
        return Vec::new();
    }

    let n_frames = trajectory.len();
    let n_coords = trajectory[0].len();
    let n_atoms = n_coords / 3;

    // Compute mean positions
    let mut mean_pos = vec![0.0f64; n_coords];
    for frame in trajectory {
        for (i, &coord) in frame.iter().enumerate() {
            mean_pos[i] += coord as f64;
        }
    }
    for coord in &mut mean_pos {
        *coord /= n_frames as f64;
    }

    // Compute fluctuations
    let mut rmsf = vec![0.0f64; n_atoms];
    for frame in trajectory {
        for i in 0..n_atoms {
            let dx = frame[i * 3] as f64 - mean_pos[i * 3];
            let dy = frame[i * 3 + 1] as f64 - mean_pos[i * 3 + 1];
            let dz = frame[i * 3 + 2] as f64 - mean_pos[i * 3 + 2];
            rmsf[i] += dx * dx + dy * dy + dz * dz;
        }
    }

    rmsf.iter()
        .map(|&v| ((v / n_frames as f64).sqrt()) as f32)
        .collect()
}

/// Compute pairwise RMSD matrix from trajectory
pub fn compute_pairwise_rmsd(trajectory: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n_frames = trajectory.len();
    let mut matrix = vec![vec![0.0f32; n_frames]; n_frames];

    for i in 0..n_frames {
        for j in (i + 1)..n_frames {
            let rmsd = compute_rmsd_aligned(&trajectory[i], &trajectory[j]);
            matrix[i][j] = rmsd;
            matrix[j][i] = rmsd;
        }
    }

    matrix
}

/// Compute mean and std of pairwise RMSD distribution
pub fn pairwise_rmsd_stats(matrix: &[Vec<f32>]) -> (f32, f32) {
    let n = matrix.len();
    if n < 2 {
        return (0.0, 0.0);
    }

    let mut values = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            values.push(matrix[i][j]);
        }
    }

    if values.is_empty() {
        return (0.0, 0.0);
    }

    let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
    let variance: f32 =
        values.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / (values.len() - 1) as f32;

    (mean, variance.sqrt())
}

/// Compute Pearson correlation coefficient
pub fn pearson_correlation(x: &[f32], y: &[f32]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x: f64 = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mean_y: f64 = y.iter().map(|&v| v as f64).sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] as f64 - mean_x;
        let dy = y[i] as f64 - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-10 || var_y < 1e-10 {
        return 0.0;
    }

    cov / (var_x.sqrt() * var_y.sqrt())
}

/// Compute Solvent Accessible Surface Area (SASA) using shrake-rupley algorithm (simplified)
pub fn compute_sasa(coords: &[f32], radii: &[f32], probe_radius: f32) -> f32 {
    let n_atoms = coords.len() / 3;
    if n_atoms != radii.len() {
        return 0.0;
    }

    // Simplified SASA calculation
    // In production, use proper Shrake-Rupley with sphere points
    let mut total_sasa = 0.0f32;
    let n_points = 92; // Golden spiral points

    for i in 0..n_atoms {
        let r_i = radii[i] + probe_radius;
        let pos_i = Vector3::new(
            coords[i * 3] as f64,
            coords[i * 3 + 1] as f64,
            coords[i * 3 + 2] as f64,
        );

        // Generate sphere points
        let mut accessible = 0;
        for k in 0..n_points {
            let theta = std::f64::consts::PI * (1.0 + 5.0_f64.sqrt()) * k as f64;
            let z = 1.0 - 2.0 * k as f64 / (n_points - 1) as f64;
            let r_xy = (1.0 - z * z).sqrt();

            let point = pos_i
                + r_i as f64 * Vector3::new(r_xy * theta.cos(), r_xy * theta.sin(), z);

            // Check if point is buried
            let mut buried = false;
            for j in 0..n_atoms {
                if i == j {
                    continue;
                }
                let pos_j = Vector3::new(
                    coords[j * 3] as f64,
                    coords[j * 3 + 1] as f64,
                    coords[j * 3 + 2] as f64,
                );
                let r_j = radii[j] + probe_radius;
                if (point - pos_j).norm() < r_j as f64 {
                    buried = true;
                    break;
                }
            }

            if !buried {
                accessible += 1;
            }
        }

        let atom_sasa =
            4.0 * std::f32::consts::PI * r_i * r_i * (accessible as f32 / n_points as f32);
        total_sasa += atom_sasa;
    }

    total_sasa
}

/// Compute SASA for a subset of residues (pocket)
pub fn compute_pocket_sasa(
    coords: &[f32],
    radii: &[f32],
    residue_indices: &[usize],
    residue_to_atoms: &[Vec<usize>],
    probe_radius: f32,
) -> f32 {
    // Get atoms belonging to pocket residues
    let pocket_atoms: Vec<usize> = residue_indices
        .iter()
        .filter(|&&r| r < residue_to_atoms.len())
        .flat_map(|&r| residue_to_atoms[r].clone())
        .collect();

    // Compute SASA for pocket atoms only
    let mut pocket_sasa = 0.0f32;
    let n_atoms = coords.len() / 3;
    let n_points = 92;

    for &i in &pocket_atoms {
        if i >= n_atoms {
            continue;
        }

        let r_i = radii[i] + probe_radius;
        let pos_i = Vector3::new(
            coords[i * 3] as f64,
            coords[i * 3 + 1] as f64,
            coords[i * 3 + 2] as f64,
        );

        let mut accessible = 0;
        for k in 0..n_points {
            let theta = std::f64::consts::PI * (1.0 + 5.0_f64.sqrt()) * k as f64;
            let z = 1.0 - 2.0 * k as f64 / (n_points - 1) as f64;
            let r_xy = (1.0 - z * z).sqrt();

            let point = pos_i
                + r_i as f64 * Vector3::new(r_xy * theta.cos(), r_xy * theta.sin(), z);

            let mut buried = false;
            for j in 0..n_atoms {
                if i == j {
                    continue;
                }
                let pos_j = Vector3::new(
                    coords[j * 3] as f64,
                    coords[j * 3 + 1] as f64,
                    coords[j * 3 + 2] as f64,
                );
                let r_j = radii[j] + probe_radius;
                if (point - pos_j).norm() < r_j as f64 {
                    buried = true;
                    break;
                }
            }

            if !buried {
                accessible += 1;
            }
        }

        let atom_sasa =
            4.0 * std::f32::consts::PI * r_i * r_i * (accessible as f32 / n_points as f32);
        pocket_sasa += atom_sasa;
    }

    pocket_sasa
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_rmsd_identical() {
        let coords = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let rmsd = compute_rmsd(&coords, &coords);
        assert!((rmsd - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_rmsd_different() {
        let coords1 = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let coords2 = vec![1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        let rmsd = compute_rmsd(&coords1, &coords2);
        assert!((rmsd - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pearson_correlation_perfect() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_pearson_correlation_negative() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let r = pearson_correlation(&x, &y);
        assert!((r - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_compute_rmsf() {
        // Three frames with atom moving in x
        let trajectory = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![2.0, 0.0, 0.0],
        ];
        let rmsf = compute_rmsf(&trajectory);
        assert_eq!(rmsf.len(), 1);
        // Mean position is (1, 0, 0), fluctuations are (-1, 0, 1)
        // RMSF = sqrt((1 + 0 + 1)/3) = sqrt(2/3) ≈ 0.816
        assert!((rmsf[0] - 0.816).abs() < 0.01);
    }
}

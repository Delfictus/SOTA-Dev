//! DVO (Discretized Volume Overlap) metric
//!
//! Standard metric for pocket spatial overlap accuracy.
//! DVO > 0.2 is considered successful prediction.
//! Uses 3D grid discretization for volume calculation.

use std::collections::HashSet;

/// Grid point for volume discretization
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
struct GridPoint {
    x: i32,
    y: i32,
    z: i32,
}

/// DVO calculation result
#[derive(Debug, Clone, Default)]
pub struct DvoResult {
    /// Jaccard index (0.0 to 1.0)
    pub jaccard: f64,
    /// Dice coefficient (0.0 to 1.0)
    pub dice: f64,
    /// Intersection volume in Å³
    pub intersection_volume: f64,
    /// Union volume in Å³
    pub union_volume: f64,
    /// Success flag (jaccard >= threshold)
    pub success: bool,
}

/// Standard DVO thresholds
pub const DVO_SUCCESS: f64 = 0.2;
pub const DVO_GOOD: f64 = 0.4;
pub const DVO_EXCELLENT: f64 = 0.6;

/// Calculate DVO using alpha sphere-based volume representation
///
/// # Arguments
/// * `predicted_spheres` - List of (center, radius) for predicted pocket
/// * `ground_truth_spheres` - List of (center, radius) for ground truth
/// * `grid_spacing` - Grid resolution in Angstroms (default 1.0)
///
/// # Returns
/// DvoResult with Jaccard and Dice coefficients
pub fn calculate_dvo(
    predicted_spheres: &[([f64; 3], f64)],
    ground_truth_spheres: &[([f64; 3], f64)],
    grid_spacing: f64,
) -> DvoResult {
    let predicted_grid = spheres_to_grid(predicted_spheres, grid_spacing);
    let ground_truth_grid = spheres_to_grid(ground_truth_spheres, grid_spacing);

    let (jaccard, dice, intersection, union) =
        jaccard_and_dice(&predicted_grid, &ground_truth_grid);

    // Convert grid points to volume
    let voxel_volume = grid_spacing.powi(3);
    let intersection_volume = intersection as f64 * voxel_volume;
    let union_volume = union as f64 * voxel_volume;

    DvoResult {
        jaccard,
        dice,
        intersection_volume,
        union_volume,
        success: jaccard >= DVO_SUCCESS,
    }
}

/// Calculate DVO from atom positions
/// Uses specified radius around each atom for volume
pub fn calculate_dvo_from_atoms(
    predicted_atoms: &[[f64; 3]],
    ground_truth_atoms: &[[f64; 3]],
    atom_radius: f64,
    grid_spacing: f64,
) -> DvoResult {
    let pred_spheres: Vec<_> = predicted_atoms.iter().map(|&pos| (pos, atom_radius)).collect();
    let gt_spheres: Vec<_> = ground_truth_atoms.iter().map(|&pos| (pos, atom_radius)).collect();

    calculate_dvo(&pred_spheres, &gt_spheres, grid_spacing)
}

/// Calculate DVO from residue lists using CA atom positions
pub fn calculate_dvo_from_residues(
    predicted_residues: &[i32],
    ground_truth_residues: &[i32],
    residue_positions: &std::collections::HashMap<i32, [f64; 3]>,
    atom_radius: f64,
    grid_spacing: f64,
) -> DvoResult {
    let pred_atoms: Vec<_> = predicted_residues
        .iter()
        .filter_map(|r| residue_positions.get(r).copied())
        .collect();
    let gt_atoms: Vec<_> = ground_truth_residues
        .iter()
        .filter_map(|r| residue_positions.get(r).copied())
        .collect();

    calculate_dvo_from_atoms(&pred_atoms, &gt_atoms, atom_radius, grid_spacing)
}

/// Calculate DVO from pocket centroid and volume (simplified)
///
/// Approximates pocket as sphere and calculates overlap
pub fn calculate_dvo_simple(
    pred_centroid: &[f64; 3],
    pred_volume: f64,
    gt_centroid: &[f64; 3],
    gt_volume: f64,
) -> DvoResult {
    // Calculate radii from volumes (assuming spheres)
    let pred_radius = (3.0 * pred_volume / (4.0 * std::f64::consts::PI)).cbrt();
    let gt_radius = (3.0 * gt_volume / (4.0 * std::f64::consts::PI)).cbrt();

    let pred_spheres = vec![(*pred_centroid, pred_radius)];
    let gt_spheres = vec![(*gt_centroid, gt_radius)];

    calculate_dvo(&pred_spheres, &gt_spheres, 1.0)
}

/// Convert spheres to grid points
fn spheres_to_grid(spheres: &[([f64; 3], f64)], spacing: f64) -> HashSet<GridPoint> {
    let mut grid = HashSet::new();

    for &(center, radius) in spheres {
        // Calculate bounding box for this sphere
        let min_x = ((center[0] - radius) / spacing).floor() as i32;
        let max_x = ((center[0] + radius) / spacing).ceil() as i32;
        let min_y = ((center[1] - radius) / spacing).floor() as i32;
        let max_y = ((center[1] + radius) / spacing).ceil() as i32;
        let min_z = ((center[2] - radius) / spacing).floor() as i32;
        let max_z = ((center[2] + radius) / spacing).ceil() as i32;

        for x in min_x..=max_x {
            for y in min_y..=max_y {
                for z in min_z..=max_z {
                    let point_center = [
                        x as f64 * spacing,
                        y as f64 * spacing,
                        z as f64 * spacing,
                    ];

                    let dist = euclidean_distance(&point_center, &center);
                    if dist <= radius {
                        grid.insert(GridPoint { x, y, z });
                    }
                }
            }
        }
    }

    grid
}

fn euclidean_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Calculate Jaccard and Dice indices
fn jaccard_and_dice(
    set_a: &HashSet<GridPoint>,
    set_b: &HashSet<GridPoint>,
) -> (f64, f64, usize, usize) {
    if set_a.is_empty() && set_b.is_empty() {
        return (0.0, 0.0, 0, 0);
    }

    let intersection = set_a.intersection(set_b).count();
    let union = set_a.union(set_b).count();

    let jaccard = if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    };

    let dice = if set_a.len() + set_b.len() == 0 {
        0.0
    } else {
        2.0 * intersection as f64 / (set_a.len() + set_b.len()) as f64
    };

    (jaccard, dice, intersection, union)
}

/// Check if DVO indicates successful prediction
pub fn is_dvo_success(dvo: f64, threshold: f64) -> bool {
    dvo >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dvo_identical() {
        let spheres = vec![([0.0, 0.0, 0.0], 3.0)];
        let result = calculate_dvo(&spheres, &spheres, 1.0);
        assert!((result.jaccard - 1.0).abs() < 0.01);
        assert!((result.dice - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_dvo_no_overlap() {
        let a = vec![([0.0, 0.0, 0.0], 1.0)];
        let b = vec![([100.0, 100.0, 100.0], 1.0)];
        let result = calculate_dvo(&a, &b, 1.0);
        assert!((result.jaccard - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_dvo_partial_overlap() {
        let a = vec![([0.0, 0.0, 0.0], 3.0)];
        let b = vec![([3.0, 0.0, 0.0], 3.0)]; // Centers 3Å apart
        let result = calculate_dvo(&a, &b, 1.0);
        assert!(result.jaccard > 0.0);
        assert!(result.jaccard < 1.0);
    }

    #[test]
    fn test_dvo_simple() {
        let result = calculate_dvo_simple(
            &[0.0, 0.0, 0.0],
            500.0,
            &[2.0, 0.0, 0.0],
            500.0,
        );
        assert!(result.jaccard > 0.5); // Should have significant overlap
    }
}

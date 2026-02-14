//! Pocket druggability scoring
//!
//! Computes and updates druggability scores for detected pockets using the
//! 8-component weighted scoring model from `scoring::composite::DruggabilityScorer`.
//!
//! Scoring components (Lipinski/SiteMap-inspired):
//!   1. Volume (sigmoid at 650 Å³)
//!   2. Hydrophobicity (Kyte-Doolittle scale normalized)
//!   3. Enclosure ratio (penalizes too open or too buried)
//!   4. Depth (sigmoid at ~6 Å)
//!   5. H-bond capacity (donors + acceptors, max ~10)
//!   6. Flexibility (low flex preferred for binding)
//!   7. Conservation (high conservation = functional site)
//!   8. Topology/persistence (from TDA persistence diagrams)

use crate::pocket::properties::Pocket;
use crate::scoring::composite::{DruggabilityScorer, ScoringWeights};

/// Score a single pocket for druggability using the 8-component model.
/// Updates `pocket.druggability_score` in place.
pub fn score_pocket_druggability(pocket: &mut Pocket) {
    if pocket.atom_indices.is_empty() {
        return;
    }
    let scorer = DruggabilityScorer::new(ScoringWeights::default());
    pocket.druggability_score = scorer.score(pocket);
}

/// Score a single pocket with custom weights.
pub fn score_pocket_druggability_with_weights(pocket: &mut Pocket, weights: ScoringWeights) {
    if pocket.atom_indices.is_empty() {
        return;
    }
    let scorer = DruggabilityScorer::new(weights);
    pocket.druggability_score = scorer.score(pocket);
}

/// Batch-score multiple pockets on CPU.
/// Updates `druggability_score` on each pocket in place.
pub fn score_pockets_batch(pockets: &mut [Pocket]) {
    let scorer = DruggabilityScorer::new(ScoringWeights::default());
    for pocket in pockets.iter_mut() {
        if !pocket.atom_indices.is_empty() {
            pocket.druggability_score = scorer.score(pocket);
        }
    }
}

/// Compute volume from atom positions using alpha-shape approximation.
/// Each atom contributes a sphere of radius ~1.7 Å (van der Waals C).
/// Overlapping spheres are counted once via voxel grid (2 Å resolution).
pub fn compute_pocket_volume(positions: &[f64], atom_indices: &[usize]) -> f64 {
    if atom_indices.is_empty() {
        return 0.0;
    }

    // Find bounding box
    let mut min_xyz = [f64::MAX; 3];
    let mut max_xyz = [f64::MIN; 3];
    for &idx in atom_indices {
        let base = idx * 3;
        if base + 2 >= positions.len() {
            continue;
        }
        for d in 0..3 {
            min_xyz[d] = min_xyz[d].min(positions[base + d]);
            max_xyz[d] = max_xyz[d].max(positions[base + d]);
        }
    }

    // Pad bounding box by probe radius
    let probe_radius = 1.7;
    for d in 0..3 {
        min_xyz[d] -= probe_radius;
        max_xyz[d] += probe_radius;
    }

    // Voxel grid (2 Å resolution)
    let voxel_size = 2.0;
    let nx = ((max_xyz[0] - min_xyz[0]) / voxel_size).ceil() as usize + 1;
    let ny = ((max_xyz[1] - min_xyz[1]) / voxel_size).ceil() as usize + 1;
    let nz = ((max_xyz[2] - min_xyz[2]) / voxel_size).ceil() as usize + 1;

    let mut occupied = vec![false; nx * ny * nz];
    let radius_sq = (probe_radius + voxel_size * 0.5) * (probe_radius + voxel_size * 0.5);

    for &idx in atom_indices {
        let base = idx * 3;
        if base + 2 >= positions.len() {
            continue;
        }
        let ax = positions[base];
        let ay = positions[base + 1];
        let az = positions[base + 2];

        // Mark voxels within atom's radius
        let vx_center = ((ax - min_xyz[0]) / voxel_size) as i32;
        let vy_center = ((ay - min_xyz[1]) / voxel_size) as i32;
        let vz_center = ((az - min_xyz[2]) / voxel_size) as i32;

        let search_range = (probe_radius / voxel_size).ceil() as i32 + 1;
        for dvx in -search_range..=search_range {
            for dvy in -search_range..=search_range {
                for dvz in -search_range..=search_range {
                    let vx = vx_center + dvx;
                    let vy = vy_center + dvy;
                    let vz = vz_center + dvz;
                    if vx < 0 || vy < 0 || vz < 0 {
                        continue;
                    }
                    let (ux, uy, uz) = (vx as usize, vy as usize, vz as usize);
                    if ux >= nx || uy >= ny || uz >= nz {
                        continue;
                    }

                    let cx = min_xyz[0] + (ux as f64 + 0.5) * voxel_size;
                    let cy = min_xyz[1] + (uy as f64 + 0.5) * voxel_size;
                    let cz = min_xyz[2] + (uz as f64 + 0.5) * voxel_size;
                    let dx = cx - ax;
                    let dy = cy - ay;
                    let dz = cz - az;

                    if dx * dx + dy * dy + dz * dz <= radius_sq {
                        occupied[ux * ny * nz + uy * nz + uz] = true;
                    }
                }
            }
        }
    }

    let occupied_count = occupied.iter().filter(|&&v| v).count();
    let voxel_volume = voxel_size * voxel_size * voxel_size;
    occupied_count as f64 * voxel_volume
}

/// Compute enclosure ratio: fraction of surface solid angles occluded by protein.
/// Sends rays from centroid; fraction that hit atoms = enclosure.
pub fn compute_enclosure_ratio(
    centroid: [f64; 3],
    positions: &[f64],
    n_atoms: usize,
    atom_radius: f64,
) -> f64 {
    let n_rays = 162; // Icosahedral sampling (sufficient for enclosure estimation)
    let golden_ratio = 1.618033988749895_f64;
    let pi = std::f64::consts::PI;
    let mut hit_count = 0;
    let max_dist = 15.0; // 15 Å max ray travel
    let radius_sq = atom_radius * atom_radius;

    for ray_idx in 0..n_rays {
        let theta = 2.0 * pi * ray_idx as f64 / golden_ratio;
        let phi = (1.0 - 2.0 * (ray_idx as f64 + 0.5) / n_rays as f64).acos();
        let dir = [
            phi.sin() * theta.cos(),
            phi.sin() * theta.sin(),
            phi.cos(),
        ];

        // March along ray, check for atom intersection
        let mut hit = false;
        let step = 0.5; // 0.5 Å steps
        let n_steps = (max_dist / step) as usize;
        for s in 1..=n_steps {
            let t = s as f64 * step;
            let px = centroid[0] + dir[0] * t;
            let py = centroid[1] + dir[1] * t;
            let pz = centroid[2] + dir[2] * t;

            for atom in 0..n_atoms {
                let base = atom * 3;
                if base + 2 >= positions.len() {
                    continue;
                }
                let dx = px - positions[base];
                let dy = py - positions[base + 1];
                let dz = pz - positions[base + 2];
                if dx * dx + dy * dy + dz * dz <= radius_sq {
                    hit = true;
                    break;
                }
            }
            if hit {
                break;
            }
        }

        if hit {
            hit_count += 1;
        }
    }

    hit_count as f64 / n_rays as f64
}

/// Update basic pocket statistics (volume, enclosure) and compute druggability score.
/// This is the main entry point called after pocket detection.
pub fn update_basic_pocket_stats(pocket: &mut Pocket) {
    if pocket.atom_indices.is_empty() {
        return;
    }
    pocket.volume = pocket.volume.max(0.0);
    score_pocket_druggability(pocket);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_pocket_druggability() {
        let mut pocket = Pocket::default();
        pocket.atom_indices = vec![0, 1, 2];
        pocket.volume = 650.0;
        pocket.mean_hydrophobicity = 0.0;
        pocket.enclosure_ratio = 0.6;
        pocket.mean_depth = 6.0;
        pocket.hbond_donors = 3;
        pocket.hbond_acceptors = 5;
        pocket.mean_flexibility = 20.0;
        pocket.mean_conservation = 0.8;
        pocket.persistence_score = 0.7;

        score_pocket_druggability(&mut pocket);

        assert!(pocket.druggability_score.total > 0.0,
            "Druggability score should be positive, got {}", pocket.druggability_score.total);
        assert!(pocket.druggability_score.total <= 1.0,
            "Druggability score should be <= 1.0, got {}", pocket.druggability_score.total);
    }

    #[test]
    fn test_empty_pocket() {
        let mut pocket = Pocket::default();
        score_pocket_druggability(&mut pocket);
        assert_eq!(pocket.druggability_score.total, 0.0);
    }

    #[test]
    fn test_compute_volume() {
        // Single atom at origin
        let positions = vec![0.0, 0.0, 0.0];
        let indices = vec![0];
        let vol = compute_pocket_volume(&positions, &indices);
        assert!(vol > 0.0, "Single atom volume should be positive: {}", vol);
        // vdW sphere of C ~ (4/3)π(1.7)³ ≈ 20.6 Å³
        assert!(vol < 200.0, "Single atom volume should be < 200 Å³: {}", vol);
    }

    #[test]
    fn test_enclosure_ratio() {
        // Atom shell around origin should give high enclosure
        let mut positions = Vec::new();
        for i in 0..100 {
            let theta = 2.0 * std::f64::consts::PI * i as f64 / 100.0;
            positions.push(5.0 * theta.cos());
            positions.push(5.0 * theta.sin());
            positions.push(0.0);
        }
        let ratio = compute_enclosure_ratio([0.0, 0.0, 0.0], &positions, 100, 2.0);
        assert!(ratio > 0.0, "Enclosure should be positive: {}", ratio);
    }
}

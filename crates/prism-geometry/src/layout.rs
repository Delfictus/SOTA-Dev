//! Graph layout generation for geometry stress analysis.
//!
//! Provides simple 2D graph layouts (spring-electrical, circular, random)
//! for use with geometry sensor layer when explicit vertex positions are not available.

use rand::Rng;
use std::f32::consts::PI;

/// Generate random 2D positions for vertices
///
/// Positions are uniformly distributed in [0, 1] x [0, 1].
/// Useful as fallback when no better layout is available.
///
/// # Arguments
/// * `num_vertices` - Number of vertices to position
/// * `seed` - Random seed for reproducibility
///
/// # Returns
/// Position array: [x0, y0, x1, y1, ..., x_n, y_n]
pub fn generate_random_layout(num_vertices: usize, seed: u64) -> Vec<f32> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut positions = Vec::with_capacity(num_vertices * 2);

    for _ in 0..num_vertices {
        positions.push(rng.gen::<f32>());
        positions.push(rng.gen::<f32>());
    }

    positions
}

/// Generate circular layout for vertices
///
/// Vertices are placed evenly around a unit circle.
/// Produces clean, symmetric layouts good for initial exploration.
///
/// # Arguments
/// * `num_vertices` - Number of vertices to position
///
/// # Returns
/// Position array: [x0, y0, x1, y1, ..., x_n, y_n]
pub fn generate_circular_layout(num_vertices: usize) -> Vec<f32> {
    let mut positions = Vec::with_capacity(num_vertices * 2);
    let radius = 0.5f32;
    let center_x = 0.5f32;
    let center_y = 0.5f32;

    for i in 0..num_vertices {
        let angle = 2.0 * PI * (i as f32) / (num_vertices as f32);
        let x = center_x + radius * angle.cos();
        let y = center_y + radius * angle.sin();
        positions.push(x);
        positions.push(y);
    }

    positions
}

/// Generate spring-electrical layout (simplified Fruchterman-Reingold)
///
/// Uses force-directed layout algorithm to position vertices.
/// Repulsive forces between all vertex pairs, attractive forces along edges.
///
/// # Arguments
/// * `num_vertices` - Number of vertices
/// * `adjacency` - Graph adjacency list
/// * `iterations` - Number of layout iterations (default: 50)
/// * `seed` - Random seed for initial positions
///
/// # Returns
/// Position array: [x0, y0, x1, y1, ..., x_n, y_n]
pub fn generate_spring_layout(
    num_vertices: usize,
    adjacency: &[Vec<usize>],
    iterations: usize,
    seed: u64,
) -> Vec<f32> {
    // Start with random positions
    let mut positions = generate_random_layout(num_vertices, seed);

    let k = (1.0 / (num_vertices as f32)).sqrt(); // Optimal edge length
    let _area = 1.0; // Unit square
    let mut temperature = 0.1f32; // Initial temperature

    for _ in 0..iterations {
        let mut forces = vec![0.0f32; num_vertices * 2];

        // Repulsive forces between all pairs
        for i in 0..num_vertices {
            for j in (i + 1)..num_vertices {
                let dx = positions[j * 2] - positions[i * 2];
                let dy = positions[j * 2 + 1] - positions[i * 2 + 1];
                let dist = (dx * dx + dy * dy).sqrt().max(0.01);

                let repulsion = k * k / dist;
                let fx = (dx / dist) * repulsion;
                let fy = (dy / dist) * repulsion;

                forces[i * 2] -= fx;
                forces[i * 2 + 1] -= fy;
                forces[j * 2] += fx;
                forces[j * 2 + 1] += fy;
            }
        }

        // Attractive forces along edges
        for (i, neighbors) in adjacency.iter().enumerate() {
            for &j in neighbors {
                if i >= j {
                    continue;
                } // Avoid double-counting

                let dx = positions[j * 2] - positions[i * 2];
                let dy = positions[j * 2 + 1] - positions[i * 2 + 1];
                let dist = (dx * dx + dy * dy).sqrt().max(0.01);

                let attraction = dist * dist / k;
                let fx = (dx / dist) * attraction;
                let fy = (dy / dist) * attraction;

                forces[i * 2] += fx;
                forces[i * 2 + 1] += fy;
                forces[j * 2] -= fx;
                forces[j * 2 + 1] -= fy;
            }
        }

        // Apply forces with temperature cooling
        for i in 0..num_vertices {
            let force_mag = (forces[i * 2] * forces[i * 2] + forces[i * 2 + 1] * forces[i * 2 + 1])
                .sqrt()
                .max(0.01);
            let displacement = force_mag.min(temperature);

            positions[i * 2] += (forces[i * 2] / force_mag) * displacement;
            positions[i * 2 + 1] += (forces[i * 2 + 1] / force_mag) * displacement;

            // Clamp to unit square
            positions[i * 2] = positions[i * 2].clamp(0.0, 1.0);
            positions[i * 2 + 1] = positions[i * 2 + 1].clamp(0.0, 1.0);
        }

        // Cool temperature
        temperature *= 0.95;
    }

    positions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_layout() {
        let positions = generate_random_layout(10, 42);
        assert_eq!(positions.len(), 20);

        // Check all positions in [0, 1]
        for &pos in &positions {
            assert!(pos >= 0.0 && pos <= 1.0);
        }
    }

    #[test]
    fn test_circular_layout() {
        let positions = generate_circular_layout(4);
        assert_eq!(positions.len(), 8);

        // Check positions form rough circle
        let center_x = 0.5;
        let center_y = 0.5;
        for i in 0..4 {
            let x = positions[i * 2];
            let y = positions[i * 2 + 1];
            let dist = ((x - center_x).powi(2) + (y - center_y).powi(2)).sqrt();
            assert!((dist - 0.5).abs() < 0.01);
        }
    }

    #[test]
    fn test_spring_layout() {
        // Triangle graph
        let adjacency = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
        let positions = generate_spring_layout(3, &adjacency, 50, 42);
        assert_eq!(positions.len(), 6);

        // Check all positions in [0, 1]
        for &pos in &positions {
            assert!(pos >= 0.0 && pos <= 1.0);
        }
    }
}

//! Distance matrix utilities

/// Simple distance matrix representation
pub type DistanceMatrix = Vec<Vec<f64>>;

/// Create a dense distance matrix from 3D coordinates
pub fn build_distance_matrix(points: &[[f64; 3]]) -> DistanceMatrix {
    let n = points.len();
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = points[i][0] - points[j][0];
            let dy = points[i][1] - points[j][1];
            let dz = points[i][2] - points[j][2];
            let d = (dx * dx + dy * dy + dz * dz).sqrt();
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    dist
}

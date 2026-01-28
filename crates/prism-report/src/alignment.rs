//! CA Kabsch Alignment Module
//!
//! Aligns apo and holo structures using CA atoms for voxel-ligand correlation.
//! Uses quaternion-based Kabsch algorithm for robust 3D alignment.

use anyhow::{bail, Context, Result};
use std::path::Path;

// =============================================================================
// ATOM DATA
// =============================================================================

/// Simple atom representation
#[derive(Debug, Clone)]
pub struct Atom {
    /// Atom serial number
    pub serial: u32,
    /// Atom name (e.g., "CA", "N", "C")
    pub name: String,
    /// Residue name (e.g., "ALA", "GLY")
    pub res_name: String,
    /// Chain ID
    pub chain: char,
    /// Residue number
    pub res_num: i32,
    /// Position (Å)
    pub pos: [f64; 3],
}

/// Parse PDB file and extract atoms
pub fn parse_pdb(path: impl AsRef<Path>) -> Result<Vec<Atom>> {
    let content = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("Failed to read PDB: {}", path.as_ref().display()))?;

    let mut atoms = Vec::new();

    for line in content.lines() {
        if !line.starts_with("ATOM") && !line.starts_with("HETATM") {
            continue;
        }

        if line.len() < 54 {
            continue;
        }

        let serial: u32 = line[6..11].trim().parse().unwrap_or(0);
        let name = line[12..16].trim().to_string();
        let res_name = line[17..20].trim().to_string();
        let chain = line.chars().nth(21).unwrap_or('A');
        let res_num: i32 = line[22..26].trim().parse().unwrap_or(0);
        let x: f64 = line[30..38].trim().parse().unwrap_or(0.0);
        let y: f64 = line[38..46].trim().parse().unwrap_or(0.0);
        let z: f64 = line[46..54].trim().parse().unwrap_or(0.0);

        atoms.push(Atom {
            serial,
            name,
            res_name,
            chain,
            res_num,
            pos: [x, y, z],
        });
    }

    Ok(atoms)
}

/// Extract CA atoms
pub fn extract_ca_atoms(atoms: &[Atom]) -> Vec<&Atom> {
    atoms.iter().filter(|a| a.name == "CA").collect()
}

/// Extract ligand atoms (HETATM, not water/ions)
pub fn extract_ligand_atoms(path: impl AsRef<Path>) -> Result<Vec<Atom>> {
    let content = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("Failed to read PDB: {}", path.as_ref().display()))?;

    let mut atoms = Vec::new();
    let exclude_residues = ["HOH", "WAT", "NA", "CL", "MG", "CA", "ZN", "FE", "K"];

    for line in content.lines() {
        if !line.starts_with("HETATM") {
            continue;
        }

        if line.len() < 54 {
            continue;
        }

        let res_name = line[17..20].trim();
        if exclude_residues.contains(&res_name) {
            continue;
        }

        let serial: u32 = line[6..11].trim().parse().unwrap_or(0);
        let name = line[12..16].trim().to_string();
        let chain = line.chars().nth(21).unwrap_or('A');
        let res_num: i32 = line[22..26].trim().parse().unwrap_or(0);
        let x: f64 = line[30..38].trim().parse().unwrap_or(0.0);
        let y: f64 = line[38..46].trim().parse().unwrap_or(0.0);
        let z: f64 = line[46..54].trim().parse().unwrap_or(0.0);

        atoms.push(Atom {
            serial,
            name,
            res_name: res_name.to_string(),
            chain,
            res_num,
            pos: [x, y, z],
        });
    }

    Ok(atoms)
}

// =============================================================================
// KABSCH ALIGNMENT (Quaternion-based)
// =============================================================================

/// 3x3 matrix (row-major)
type Mat3 = [[f64; 3]; 3];

/// Compute centroid of positions
fn centroid(positions: &[[f64; 3]]) -> [f64; 3] {
    if positions.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    let n = positions.len() as f64;
    let mut c = [0.0f64; 3];
    for p in positions {
        c[0] += p[0];
        c[1] += p[1];
        c[2] += p[2];
    }
    [c[0] / n, c[1] / n, c[2] / n]
}

/// Center positions around origin
fn center_positions(positions: &[[f64; 3]], cent: [f64; 3]) -> Vec<[f64; 3]> {
    positions
        .iter()
        .map(|p| [p[0] - cent[0], p[1] - cent[1], p[2] - cent[2]])
        .collect()
}

/// Compute cross-covariance matrix H = Q'^T * P'
/// Where P' and Q' are centered point sets (P = reference, Q = mobile)
/// H[i][j] = sum_k Q'_k[i] * P'_k[j]
/// This gives the matrix for finding R such that R*Q' ≈ P'
fn correlation_matrix(p: &[[f64; 3]], q: &[[f64; 3]]) -> Mat3 {
    let mut h = [[0.0f64; 3]; 3];

    for (pi, qi) in p.iter().zip(q.iter()) {
        for i in 0..3 {
            for j in 0..3 {
                // H = Q^T * P, so H[i][j] = sum_k Q_k[i] * P_k[j]
                h[i][j] += qi[i] * pi[j];
            }
        }
    }

    h
}

/// Matrix determinant
fn det3(m: &Mat3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Matrix-vector multiplication
fn matvec(m: &Mat3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Build the 4x4 K matrix for quaternion extraction from correlation matrix H
/// K = | H[0][0]+H[1][1]+H[2][2]   H[1][2]-H[2][1]          H[2][0]-H[0][2]          H[0][1]-H[1][0]       |
///     | H[1][2]-H[2][1]          H[0][0]-H[1][1]-H[2][2]   H[0][1]+H[1][0]          H[0][2]+H[2][0]       |
///     | H[2][0]-H[0][2]          H[0][1]+H[1][0]         -H[0][0]+H[1][1]-H[2][2]   H[1][2]+H[2][1]       |
///     | H[0][1]-H[1][0]          H[0][2]+H[2][0]          H[1][2]+H[2][1]         -H[0][0]-H[1][1]+H[2][2]|
fn build_k_matrix(h: &Mat3) -> [[f64; 4]; 4] {
    let trace = h[0][0] + h[1][1] + h[2][2];

    [
        [
            trace,
            h[1][2] - h[2][1],
            h[2][0] - h[0][2],
            h[0][1] - h[1][0],
        ],
        [
            h[1][2] - h[2][1],
            h[0][0] - h[1][1] - h[2][2],
            h[0][1] + h[1][0],
            h[0][2] + h[2][0],
        ],
        [
            h[2][0] - h[0][2],
            h[0][1] + h[1][0],
            -h[0][0] + h[1][1] - h[2][2],
            h[1][2] + h[2][1],
        ],
        [
            h[0][1] - h[1][0],
            h[0][2] + h[2][0],
            h[1][2] + h[2][1],
            -h[0][0] - h[1][1] + h[2][2],
        ],
    ]
}

/// Jacobi eigenvalue solver for 4x4 symmetric matrices
/// Returns (eigenvalues, eigenvectors) where eigenvectors are columns of the matrix
fn jacobi_eigen_4x4(k: &[[f64; 4]; 4]) -> ([f64; 4], [[f64; 4]; 4]) {
    let mut a = *k; // Working copy of matrix
    let mut v = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]; // Eigenvector matrix (starts as identity)

    const MAX_ITER: usize = 50;
    const EPSILON: f64 = 1e-15;

    for _ in 0..MAX_ITER {
        // Find largest off-diagonal element
        let mut max_val = 0.0f64;
        let mut p = 0usize;
        let mut q = 1usize;

        for i in 0..4 {
            for j in (i + 1)..4 {
                let abs_val = a[i][j].abs();
                if abs_val > max_val {
                    max_val = abs_val;
                    p = i;
                    q = j;
                }
            }
        }

        // Check convergence
        if max_val < EPSILON {
            break;
        }

        // Compute rotation angle
        let diff = a[q][q] - a[p][p];
        let theta = if diff.abs() < EPSILON {
            std::f64::consts::FRAC_PI_4 * a[p][q].signum()
        } else {
            0.5 * (2.0 * a[p][q] / diff).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply Jacobi rotation to A: A' = J^T * A * J
        // This zeros out a[p][q] and a[q][p]

        // Store values we'll need
        let app = a[p][p];
        let aqq = a[q][q];
        let apq = a[p][q];

        // Update diagonal elements
        a[p][p] = c * c * app - 2.0 * c * s * apq + s * s * aqq;
        a[q][q] = s * s * app + 2.0 * c * s * apq + c * c * aqq;
        a[p][q] = 0.0;
        a[q][p] = 0.0;

        // Update off-diagonal elements in rows/columns p and q
        for r in 0..4 {
            if r != p && r != q {
                let arp = a[r][p];
                let arq = a[r][q];
                a[r][p] = c * arp - s * arq;
                a[p][r] = a[r][p];
                a[r][q] = s * arp + c * arq;
                a[q][r] = a[r][q];
            }
        }

        // Accumulate eigenvectors: V' = V * J
        for r in 0..4 {
            let vrp = v[r][p];
            let vrq = v[r][q];
            v[r][p] = c * vrp - s * vrq;
            v[r][q] = s * vrp + c * vrq;
        }
    }

    // Eigenvalues are now on the diagonal
    let eigenvalues = [a[0][0], a[1][1], a[2][2], a[3][3]];

    (eigenvalues, v)
}

/// Find the eigenvector corresponding to the largest eigenvalue of a 4x4 symmetric matrix
fn largest_eigenvector_4x4(k: &[[f64; 4]; 4]) -> [f64; 4] {
    let (eigenvalues, eigenvectors) = jacobi_eigen_4x4(k);

    // Find index of largest eigenvalue
    let mut max_idx = 0;
    let mut max_val = eigenvalues[0];
    for i in 1..4 {
        if eigenvalues[i] > max_val {
            max_val = eigenvalues[i];
            max_idx = i;
        }
    }

    // Extract eigenvector (column max_idx of eigenvectors matrix)
    let mut v = [0.0f64; 4];
    for i in 0..4 {
        v[i] = eigenvectors[i][max_idx];
    }

    // Normalize to unit quaternion
    let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
    if norm > 1e-15 {
        for i in 0..4 {
            v[i] /= norm;
        }
    }

    // Ensure w >= 0 for canonical quaternion representation
    if v[0] < 0.0 {
        for i in 0..4 {
            v[i] = -v[i];
        }
    }

    v
}

/// Convert quaternion [w, x, y, z] to rotation matrix
fn quaternion_to_matrix(q: [f64; 4]) -> Mat3 {
    let w = q[0];
    let x = q[1];
    let y = q[2];
    let z = q[3];

    let ww = w * w;
    let xx = x * x;
    let yy = y * y;
    let zz = z * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;

    [
        [ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz],
    ]
}

/// Compute optimal rotation matrix using quaternion-based Kabsch algorithm
///
/// Given two sets of points P (reference) and Q (mobile), finds rotation R
/// that minimizes sum of squared distances when Q is transformed to match P.
///
/// The algorithm:
/// 1. Center both point sets at origin
/// 2. Compute correlation matrix H = P'^T * Q'
/// 3. Build 4x4 K matrix from H
/// 4. Find largest eigenvector of K (the optimal quaternion)
/// 5. Convert quaternion to rotation matrix
/// 6. Handle reflection if det(R) < 0
pub fn kabsch_rotation(p_centered: &[[f64; 3]], q_centered: &[[f64; 3]]) -> Result<Mat3> {
    if p_centered.len() != q_centered.len() {
        bail!(
            "Point sets must have same length: {} vs {}",
            p_centered.len(),
            q_centered.len()
        );
    }

    if p_centered.len() < 3 {
        bail!("Need at least 3 points for Kabsch alignment");
    }

    // Compute correlation matrix H = P'^T * Q'
    let h = correlation_matrix(p_centered, q_centered);

    // Build K matrix
    let k = build_k_matrix(&h);

    // Find largest eigenvector (quaternion)
    let q = largest_eigenvector_4x4(&k);

    // Convert to rotation matrix
    let mut r = quaternion_to_matrix(q);

    // Check for reflection (det(R) should be +1 for proper rotation)
    let d = det3(&r);
    if d < 0.0 {
        // Flip sign of last column to fix reflection
        r[0][2] = -r[0][2];
        r[1][2] = -r[1][2];
        r[2][2] = -r[2][2];
    }

    Ok(r)
}

/// Alignment transformation
#[derive(Debug, Clone)]
pub struct Alignment {
    /// Rotation matrix (3x3)
    pub rotation: Mat3,
    /// Centroid of reference (P)
    pub centroid_ref: [f64; 3],
    /// Centroid of mobile (Q)
    pub centroid_mob: [f64; 3],
    /// RMSD after alignment
    pub rmsd: f64,
    /// Number of points aligned
    pub n_points: usize,
}

impl Alignment {
    /// Apply alignment to a point (transforms mobile to reference frame)
    /// Q_aligned = R * (Q - centroid_mob) + centroid_ref
    pub fn transform(&self, point: [f64; 3]) -> [f64; 3] {
        // Center around mobile centroid
        let centered = [
            point[0] - self.centroid_mob[0],
            point[1] - self.centroid_mob[1],
            point[2] - self.centroid_mob[2],
        ];
        // Rotate
        let rotated = matvec(&self.rotation, centered);
        // Translate to reference frame
        [
            rotated[0] + self.centroid_ref[0],
            rotated[1] + self.centroid_ref[1],
            rotated[2] + self.centroid_ref[2],
        ]
    }

    /// Transform multiple points
    pub fn transform_points(&self, points: &[[f64; 3]]) -> Vec<[f64; 3]> {
        points.iter().map(|&p| self.transform(p)).collect()
    }

    /// Transform a point with f32 coordinates (convenience method)
    pub fn transform_f32(&self, point: [f32; 3]) -> [f32; 3] {
        let result = self.transform([point[0] as f64, point[1] as f64, point[2] as f64]);
        [result[0] as f32, result[1] as f32, result[2] as f32]
    }
}

/// Compute full Kabsch alignment (rotation + translation)
///
/// Aligns mobile points Q to reference points P by finding the optimal
/// rotation R such that: Q_aligned = R * (Q - centroid_Q) + centroid_P
/// minimizes RMSD.
pub fn kabsch_align(reference: &[[f64; 3]], mobile: &[[f64; 3]]) -> Result<Alignment> {
    if reference.len() != mobile.len() {
        bail!(
            "Point sets must have same length: {} vs {}",
            reference.len(),
            mobile.len()
        );
    }

    if reference.len() < 3 {
        bail!("Need at least 3 points for Kabsch alignment");
    }

    // Compute centroids
    let cent_ref = centroid(reference);
    let cent_mob = centroid(mobile);

    // Center both point sets
    let ref_centered = center_positions(reference, cent_ref);
    let mob_centered = center_positions(mobile, cent_mob);

    // Compute optimal rotation
    let rotation = kabsch_rotation(&ref_centered, &mob_centered)?;

    // Compute RMSD
    // RMSD = sqrt(mean(||R * mob_centered[i] - ref_centered[i]||^2))
    let mut sum_sq = 0.0f64;
    for (r, m) in ref_centered.iter().zip(mob_centered.iter()) {
        let rotated = matvec(&rotation, *m);
        let dx = rotated[0] - r[0];
        let dy = rotated[1] - r[1];
        let dz = rotated[2] - r[2];
        sum_sq += dx * dx + dy * dy + dz * dz;
    }
    let rmsd = (sum_sq / reference.len() as f64).sqrt();

    Ok(Alignment {
        rotation,
        centroid_ref: cent_ref,
        centroid_mob: cent_mob,
        rmsd,
        n_points: reference.len(),
    })
}

/// Align holo to apo using CA atoms
pub fn align_structures(apo_pdb: impl AsRef<Path>, holo_pdb: impl AsRef<Path>) -> Result<Alignment> {
    let apo_atoms = parse_pdb(apo_pdb)?;
    let holo_atoms = parse_pdb(holo_pdb)?;

    let apo_ca = extract_ca_atoms(&apo_atoms);
    let holo_ca = extract_ca_atoms(&holo_atoms);

    // Match CA atoms by residue number
    let mut apo_positions = Vec::new();
    let mut holo_positions = Vec::new();

    for apo_atom in &apo_ca {
        for holo_atom in &holo_ca {
            if apo_atom.chain == holo_atom.chain && apo_atom.res_num == holo_atom.res_num {
                apo_positions.push(apo_atom.pos);
                holo_positions.push(holo_atom.pos);
                break;
            }
        }
    }

    if apo_positions.len() < 3 {
        bail!(
            "Need at least 3 matching CA atoms for alignment, found {}",
            apo_positions.len()
        );
    }

    kabsch_align(&apo_positions, &holo_positions)
}

// =============================================================================
// VOXEL-LIGAND OVERLAP
// =============================================================================

use crate::voxelize::VoxelGrid;

/// Voxel-ligand overlap metrics
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct VoxelLigandOverlap {
    /// Fraction of ligand atoms within occupied voxels
    pub ligand_coverage: f32,

    /// Fraction of occupied voxels containing ligand
    pub voxel_hit_rate: f32,

    /// Mean density at ligand positions
    pub mean_density_at_ligand: f32,

    /// Number of ligand atoms
    pub n_ligand_atoms: usize,

    /// Number of ligand atoms in occupied voxels
    pub n_covered_atoms: usize,

    /// CA RMSD of alignment (Å)
    pub alignment_rmsd: f32,
}

/// Compute voxel-ligand overlap after Kabsch alignment
pub fn compute_voxel_ligand_overlap(
    grid: &VoxelGrid,
    ligand_atoms: &[Atom],
    alignment: &Alignment,
    threshold: f32,
) -> VoxelLigandOverlap {
    if ligand_atoms.is_empty() {
        return VoxelLigandOverlap {
            alignment_rmsd: alignment.rmsd as f32,
            ..Default::default()
        };
    }

    // Transform ligand atoms to apo frame
    let transformed: Vec<[f32; 3]> = ligand_atoms
        .iter()
        .map(|a| alignment.transform_f32([a.pos[0] as f32, a.pos[1] as f32, a.pos[2] as f32]))
        .collect();

    // Check which ligand atoms are in occupied voxels
    let mut covered = 0;
    let mut total_density = 0.0f32;

    for pos in &transformed {
        let voxel = grid.world_to_voxel([pos[0], pos[1], pos[2]]);
        let x = voxel[0].round() as i32;
        let y = voxel[1].round() as i32;
        let z = voxel[2].round() as i32;

        if x >= 0
            && x < grid.dims[0] as i32
            && y >= 0
            && y < grid.dims[1] as i32
            && z >= 0
            && z < grid.dims[2] as i32
        {
            let density = grid.get(x as usize, y as usize, z as usize);
            total_density += density;
            if density > threshold {
                covered += 1;
            }
        }
    }

    // Count occupied voxels
    let occupied_voxels: Vec<(usize, usize, usize)> = (0..grid.dims[2])
        .flat_map(|z| {
            (0..grid.dims[1])
                .flat_map(move |y| (0..grid.dims[0]).map(move |x| (x, y, z)))
        })
        .filter(|&(x, y, z)| grid.get(x, y, z) > threshold)
        .collect();

    // Check which occupied voxels contain ligand
    let mut voxels_with_ligand = std::collections::HashSet::new();
    for pos in &transformed {
        let voxel = grid.world_to_voxel([pos[0], pos[1], pos[2]]);
        let x = voxel[0].round() as usize;
        let y = voxel[1].round() as usize;
        let z = voxel[2].round() as usize;

        if x < grid.dims[0] && y < grid.dims[1] && z < grid.dims[2] {
            if grid.get(x, y, z) > threshold {
                voxels_with_ligand.insert((x, y, z));
            }
        }
    }

    let voxel_hit_rate = if occupied_voxels.is_empty() {
        0.0
    } else {
        voxels_with_ligand.len() as f32 / occupied_voxels.len() as f32
    };

    VoxelLigandOverlap {
        ligand_coverage: covered as f32 / ligand_atoms.len() as f32,
        voxel_hit_rate,
        mean_density_at_ligand: total_density / ligand_atoms.len() as f32,
        n_ligand_atoms: ligand_atoms.len(),
        n_covered_atoms: covered,
        alignment_rmsd: alignment.rmsd as f32,
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn mat_approx_identity(m: &Mat3, tol: f64) -> bool {
        approx_eq(m[0][0], 1.0, tol) && approx_eq(m[0][1], 0.0, tol) && approx_eq(m[0][2], 0.0, tol) &&
        approx_eq(m[1][0], 0.0, tol) && approx_eq(m[1][1], 1.0, tol) && approx_eq(m[1][2], 0.0, tol) &&
        approx_eq(m[2][0], 0.0, tol) && approx_eq(m[2][1], 0.0, tol) && approx_eq(m[2][2], 1.0, tol)
    }

    fn vec_approx_eq(a: [f64; 3], b: [f64; 3], tol: f64) -> bool {
        approx_eq(a[0], b[0], tol) && approx_eq(a[1], b[1], tol) && approx_eq(a[2], b[2], tol)
    }

    #[test]
    fn test_centroid() {
        let points = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]];
        let c = centroid(&points);
        assert!(approx_eq(c[0], 2.0 / 3.0, 1e-10));
        assert!(approx_eq(c[1], 2.0 / 3.0, 1e-10));
        assert!(approx_eq(c[2], 0.0, 1e-10));
    }

    #[test]
    fn test_kabsch_identity() {
        // Identical point sets should give identity rotation and zero RMSD
        let points = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];

        let alignment = kabsch_align(&points, &points).unwrap();

        // RMSD should be essentially zero
        assert!(alignment.rmsd < 1e-10, "RMSD should be ~0, got {}", alignment.rmsd);

        // Rotation should be identity
        assert!(mat_approx_identity(&alignment.rotation, 1e-10),
            "Rotation should be identity, got {:?}", alignment.rotation);

        // Centroids should be equal
        assert!(vec_approx_eq(alignment.centroid_ref, alignment.centroid_mob, 1e-10),
            "Centroids should match");

        // Transform should return same points
        for &p in &points {
            let transformed = alignment.transform(p);
            assert!(vec_approx_eq(transformed, p, 1e-10),
                "Transform should preserve points, got {:?} vs {:?}", transformed, p);
        }
    }

    #[test]
    fn test_kabsch_translation() {
        // Pure translation: mobile is reference shifted by (5, 5, 5)
        let reference = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let translation = [5.0, 5.0, 5.0];
        let mobile: Vec<[f64; 3]> = reference
            .iter()
            .map(|p| [p[0] + translation[0], p[1] + translation[1], p[2] + translation[2]])
            .collect();

        let alignment = kabsch_align(&reference, &mobile).unwrap();

        // RMSD should be essentially zero (pure translation)
        assert!(alignment.rmsd < 1e-10, "RMSD should be ~0 for pure translation, got {}", alignment.rmsd);

        // Rotation should be identity
        assert!(mat_approx_identity(&alignment.rotation, 1e-10),
            "Rotation should be identity for pure translation, got {:?}", alignment.rotation);

        // Translation = centroid_ref - centroid_mob should equal -translation
        let computed_translation = [
            alignment.centroid_ref[0] - alignment.centroid_mob[0],
            alignment.centroid_ref[1] - alignment.centroid_mob[1],
            alignment.centroid_ref[2] - alignment.centroid_mob[2],
        ];
        let expected_translation = [-translation[0], -translation[1], -translation[2]];
        assert!(vec_approx_eq(computed_translation, expected_translation, 1e-10),
            "Translation should be {:?}, got {:?}", expected_translation, computed_translation);

        // Transform mobile points back to reference
        for (i, &m) in mobile.iter().enumerate() {
            let transformed = alignment.transform(m);
            assert!(vec_approx_eq(transformed, reference[i], 1e-10),
                "Point {} mismatch: {:?} vs {:?}", i, transformed, reference[i]);
        }
    }

    #[test]
    fn test_kabsch_90deg_rotation() {
        // 90 degree rotation around Z axis
        let reference = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ];
        // After 90 deg CCW rotation around Z: (x,y,z) -> (-y,x,z)
        let mobile = [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
        ];

        let alignment = kabsch_align(&reference, &mobile).unwrap();

        // RMSD should be ~0
        assert!(alignment.rmsd < 1e-8, "RMSD should be ~0, got {}", alignment.rmsd);

        // Transform mobile back to reference
        for (i, &m) in mobile.iter().enumerate() {
            let transformed = alignment.transform(m);
            assert!(vec_approx_eq(transformed, reference[i], 1e-8),
                "Point {} mismatch after rotation: {:?} vs {:?}", i, transformed, reference[i]);
        }
    }

    #[test]
    fn test_quaternion_to_matrix_identity() {
        // Identity quaternion: [1, 0, 0, 0]
        let q = [1.0, 0.0, 0.0, 0.0];
        let r = quaternion_to_matrix(q);
        assert!(mat_approx_identity(&r, 1e-15));
    }

    #[test]
    fn test_determinant_positive() {
        // Test that all computed rotations have det = +1
        let reference = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ];

        // Various rotated versions
        let test_cases = vec![
            // Pure translation
            reference.iter().map(|p| [p[0] + 10.0, p[1] - 5.0, p[2] + 3.0]).collect::<Vec<_>>(),
            // Identity
            reference.to_vec(),
        ];

        for mobile in test_cases {
            let alignment = kabsch_align(&reference, &mobile).unwrap();
            let d = det3(&alignment.rotation);
            assert!((d - 1.0).abs() < 1e-10, "Determinant should be +1, got {}", d);
        }
    }
}

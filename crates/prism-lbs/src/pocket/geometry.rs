//! Pocket geometry utilities (volume, enclosure)

use crate::structure::{ProteinStructure, VDW_RADII};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const DEFAULT_PROBE_RADIUS: f64 = 1.4;
const DEFAULT_VOXEL_RESOLUTION: f64 = 0.75;
const DEFAULT_ALPHA_SHAPE_SHRINK: f64 = 0.5;

/// Geometry computation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryConfig {
    /// Voxel resolution (Angstrom) for volume estimation
    pub voxel_resolution: f64,
    /// Probe radius used in voxel volume (Angstrom)
    pub probe_radius: f64,
    /// Whether to use voxel volume (otherwise bbox fallback)
    pub use_voxel_volume: bool,
    /// Whether to use a convex hull volume estimator
    pub use_convex_hull_volume: bool,
    /// Epsilon for hull face visibility
    pub convex_hull_epsilon: f64,
    /// Whether to use alpha-shape style volume (shrunk spheres)
    pub use_alpha_shape_volume: bool,
    /// Shrink factor applied to probe radius for alpha-shape
    pub alpha_shape_shrink: f64,
    /// Resolution for alpha-shape voxelization
    pub alpha_shape_resolution: f64,
    /// Whether to compute enclosure using boundary atoms rather than surface flags
    pub use_boundary_enclosure: bool,
    /// Whether to compute enclosure using neighbor distance cutoff
    pub use_neighbor_enclosure: bool,
    /// Cutoff for neighbor-based enclosure (Angstrom)
    pub boundary_cutoff: f64,
    /// Whether to compute flood-fill cavity volume (voids)
    pub use_flood_fill_cavity: bool,
    /// Resolution for cavity flood fill (Angstrom)
    pub cavity_resolution: f64,
}

impl Default for GeometryConfig {
    fn default() -> Self {
        Self {
            voxel_resolution: DEFAULT_VOXEL_RESOLUTION,
            probe_radius: DEFAULT_PROBE_RADIUS,
            use_voxel_volume: true,
            use_convex_hull_volume: true,
            convex_hull_epsilon: 1e-6,
            use_alpha_shape_volume: true,
            alpha_shape_shrink: DEFAULT_ALPHA_SHAPE_SHRINK,
            alpha_shape_resolution: 0.6,
            use_boundary_enclosure: true,
            use_neighbor_enclosure: false,
            boundary_cutoff: 4.5,
            use_flood_fill_cavity: true,
            cavity_resolution: 0.8,
        }
    }
}

/// Axis-aligned bounding box volume for a set of atom indices (fallback estimator)
pub fn bounding_box_volume(structure: &ProteinStructure, atom_indices: &[usize]) -> f64 {
    if atom_indices.is_empty() {
        return 0.0;
    }
    let mut min_corner = [f64::INFINITY; 3];
    let mut max_corner = [f64::NEG_INFINITY; 3];
    for &idx in atom_indices {
        if let Some(atom) = structure.atoms.get(idx) {
            for i in 0..3 {
                min_corner[i] = min_corner[i].min(atom.coord[i]);
                max_corner[i] = max_corner[i].max(atom.coord[i]);
            }
        }
    }
    let dx = (max_corner[0] - min_corner[0]).max(0.0);
    let dy = (max_corner[1] - min_corner[1]).max(0.0);
    let dz = (max_corner[2] - min_corner[2]).max(0.0);
    dx * dy * dz
}

/// Voxelized volume estimation using union of atom spheres (vdW + probe radius)
pub fn voxel_volume(
    structure: &ProteinStructure,
    atom_indices: &[usize],
    voxel_resolution: Option<f64>,
    probe_radius: Option<f64>,
) -> f64 {
    if atom_indices.is_empty() {
        return 0.0;
    }
    let res = voxel_resolution
        .unwrap_or(DEFAULT_VOXEL_RESOLUTION)
        .max(0.25);
    let probe = probe_radius.unwrap_or(DEFAULT_PROBE_RADIUS);

    let mut min_corner = [f64::INFINITY; 3];
    let mut max_corner = [f64::NEG_INFINITY; 3];
    for &idx in atom_indices {
        if let Some(atom) = structure.atoms.get(idx) {
            for i in 0..3 {
                min_corner[i] = min_corner[i].min(atom.coord[i]);
                max_corner[i] = max_corner[i].max(atom.coord[i]);
            }
        }
    }
    // Pad by max radius + probe to ensure coverage
    let pad = 2.5;
    for i in 0..3 {
        min_corner[i] -= pad;
        max_corner[i] += pad;
    }

    let nx = ((max_corner[0] - min_corner[0]) / res).ceil() as usize;
    let ny = ((max_corner[1] - min_corner[1]) / res).ceil() as usize;
    let nz = ((max_corner[2] - min_corner[2]) / res).ceil() as usize;

    let mut occupied = 0usize;
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let cx = min_corner[0] + ix as f64 * res;
                let cy = min_corner[1] + iy as f64 * res;
                let cz = min_corner[2] + iz as f64 * res;
                if voxel_inside_any_atom([cx, cy, cz], structure, atom_indices, probe) {
                    occupied += 1;
                }
            }
        }
    }
    occupied as f64 * res * res * res
}

fn voxel_inside_any_atom(
    center: [f64; 3],
    structure: &ProteinStructure,
    atom_indices: &[usize],
    probe_radius: f64,
) -> bool {
    for &idx in atom_indices {
        if let Some(atom) = structure.atoms.get(idx) {
            let r = VDW_RADII.get(&atom.element).copied().unwrap_or(1.5) + probe_radius;
            let dx = center[0] - atom.coord[0];
            let dy = center[1] - atom.coord[1];
            let dz = center[2] - atom.coord[2];
            if dx * dx + dy * dy + dz * dz <= r * r {
                return true;
            }
        }
    }
    false
}

/// Alpha-shape-inspired volume via voxelization with reduced probe radius
pub fn alpha_shape_volume(
    structure: &ProteinStructure,
    atom_indices: &[usize],
    resolution: f64,
    shrink: f64,
) -> f64 {
    let adjusted_probe = (DEFAULT_PROBE_RADIUS - shrink).max(0.0);
    voxel_volume(
        structure,
        atom_indices,
        Some(resolution),
        Some(adjusted_probe),
    )
}

/// Convex hull volume using a lightweight QuickHull implementation
pub fn convex_hull_volume(
    structure: &ProteinStructure,
    atom_indices: &[usize],
    epsilon: f64,
) -> f64 {
    let points: Vec<[f64; 3]> = atom_indices
        .iter()
        .filter_map(|&i| structure.atoms.get(i).map(|a| a.coord))
        .collect();
    if points.len() < 4 {
        return 0.0;
    }

    // Initial tetrahedron selection
    let (p0, p1) = farthest_pair(&points);
    let p2 = farthest_point_from_line(&points, p0, p1, epsilon);
    if p2.is_none() {
        return 0.0;
    }
    let p2 = p2.unwrap();
    let p3 = farthest_point_from_plane(&points, p0, p1, p2, epsilon);
    if p3.is_none() {
        return 0.0;
    }
    let p3 = p3.unwrap();

    let mut faces = initial_faces(p0, p1, p2, p3, &points);
    let centroid = mean_point(&points);
    orient_faces_outward(&mut faces, &points, &centroid);

    for (idx, _) in points.iter().enumerate() {
        if [p0, p1, p2, p3].contains(&idx) {
            continue;
        }
        let visible: Vec<usize> = faces
            .iter()
            .enumerate()
            .filter(|(_, f)| signed_distance(&points[idx], f) > epsilon)
            .map(|(i, _)| i)
            .collect();
        if visible.is_empty() {
            continue;
        }

        let mut horizon: HashMap<(usize, usize), usize> = HashMap::new();
        for &fi in &visible {
            let f = &faces[fi].verts;
            let edges = [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])];
            for &(a, b) in &edges {
                let key = if a < b { (a, b) } else { (b, a) };
                *horizon.entry(key).or_default() += 1;
            }
        }

        // Remove visible faces
        faces = faces
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !visible.contains(i))
            .map(|(_, f)| f)
            .collect();

        // Add new faces along the horizon
        for ((a, b), count) in horizon {
            if count != 1 {
                continue; // not part of horizon
            }
            let mut verts = [a, b, idx];
            let mut face = Face::new(verts, &points);
            if signed_distance(&centroid, &face) > 0.0 {
                verts.swap(0, 1);
                face = Face::new(verts, &points);
            }
            if face.normal_magnitude > epsilon {
                faces.push(face);
            }
        }
    }

    // Volume from signed tetrahedra with origin
    let mut volume = 0.0;
    for f in &faces {
        let a = points[f.verts[0]];
        let b = points[f.verts[1]];
        let c = points[f.verts[2]];
        let cross = cross_raw(&b, &c);
        let v = dot_raw(&a, &cross) / 6.0;
        volume += v;
    }
    volume.abs()
}

#[derive(Clone, Debug)]
struct Face {
    verts: [usize; 3],
    normal: [f64; 3],
    offset: f64,
    normal_magnitude: f64,
}

impl Face {
    fn new(verts: [usize; 3], points: &[[f64; 3]]) -> Self {
        let a = points[verts[0]];
        let b = points[verts[1]];
        let c = points[verts[2]];
        let n_raw = cross_vec(&b, &c, &a);
        let norm = (n_raw[0] * n_raw[0] + n_raw[1] * n_raw[1] + n_raw[2] * n_raw[2]).sqrt();
        let normal = if norm > 0.0 {
            [n_raw[0] / norm, n_raw[1] / norm, n_raw[2] / norm]
        } else {
            [0.0, 0.0, 0.0]
        };
        let offset = -dot(&normal, &a);
        Self {
            verts,
            normal,
            offset,
            normal_magnitude: norm,
        }
    }
}

fn signed_distance(p: &[f64; 3], face: &Face) -> f64 {
    dot(&face.normal, p) + face.offset
}

fn farthest_pair(points: &[[f64; 3]]) -> (usize, usize) {
    let mut max_d2 = -1.0;
    let mut pair = (0, 1);
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let d2 = distance_squared(&points[i], &points[j]);
            if d2 > max_d2 {
                max_d2 = d2;
                pair = (i, j);
            }
        }
    }
    pair
}

fn farthest_point_from_line(
    points: &[[f64; 3]],
    a: usize,
    b: usize,
    epsilon: f64,
) -> Option<usize> {
    let mut max_area = epsilon;
    let mut idx = None;
    let ab = [
        points[b][0] - points[a][0],
        points[b][1] - points[a][1],
        points[b][2] - points[a][2],
    ];
    for (i, p) in points.iter().enumerate() {
        if i == a || i == b {
            continue;
        }
        let ap = [
            p[0] - points[a][0],
            p[1] - points[a][1],
            p[2] - points[a][2],
        ];
        let cross = [
            ab[1] * ap[2] - ab[2] * ap[1],
            ab[2] * ap[0] - ab[0] * ap[2],
            ab[0] * ap[1] - ab[1] * ap[0],
        ];
        let area = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
        if area > max_area {
            max_area = area;
            idx = Some(i);
        }
    }
    idx
}

fn farthest_point_from_plane(
    points: &[[f64; 3]],
    a: usize,
    b: usize,
    c: usize,
    epsilon: f64,
) -> Option<usize> {
    let normal = cross_vec(&points[b], &points[c], &points[a]);
    let norm = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if norm <= epsilon {
        return None;
    }
    let normal = [normal[0] / norm, normal[1] / norm, normal[2] / norm];
    let offset = -dot(&normal, &points[a]);
    let mut max_dist = epsilon;
    let mut idx = None;
    for (i, p) in points.iter().enumerate() {
        if i == a || i == b || i == c {
            continue;
        }
        let dist = (dot(&normal, p) + offset).abs();
        if dist > max_dist {
            max_dist = dist;
            idx = Some(i);
        }
    }
    idx
}

fn initial_faces(a: usize, b: usize, c: usize, d: usize, points: &[[f64; 3]]) -> Vec<Face> {
    let faces_idx = [[a, b, c], [a, b, d], [a, c, d], [b, c, d]];
    faces_idx
        .iter()
        .map(|&verts| Face::new(verts, points))
        .collect()
}

fn orient_faces_outward(faces: &mut [Face], points: &[[f64; 3]], centroid: &[f64; 3]) {
    for face in faces.iter_mut() {
        if signed_distance(centroid, face) > 0.0 {
            face.verts.swap(0, 1);
            *face = Face::new(face.verts, points);
        }
    }
}

fn mean_point(points: &[[f64; 3]]) -> [f64; 3] {
    let mut sum = [0.0, 0.0, 0.0];
    for p in points {
        sum[0] += p[0];
        sum[1] += p[1];
        sum[2] += p[2];
    }
    let n = points.len() as f64;
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

fn cross_vec(b: &[f64; 3], c: &[f64; 3], a: &[f64; 3]) -> [f64; 3] {
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

fn cross_raw(u: &[f64; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

fn dot(u: &[f64; 3], v: &[f64; 3]) -> f64 {
    u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
}

fn dot_raw(u: &[f64; 3], v: &[f64; 3]) -> f64 {
    dot(u, v)
}

fn distance_squared(p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    let dx = p1[0] - p2[0];
    let dy = p1[1] - p2[1];
    let dz = p1[2] - p2[2];
    dx * dx + dy * dy + dz * dz
}

/// Simple enclosure ratio: surface atoms in pocket vs total pocket atoms
pub fn enclosure_ratio(structure: &ProteinStructure, atom_indices: &[usize]) -> f64 {
    if atom_indices.is_empty() {
        return 0.0;
    }
    let surface_count = atom_indices
        .iter()
        .filter(|&&idx| structure.atoms.get(idx).map_or(false, |a| a.is_surface))
        .count();
    1.0 - (surface_count as f64 / atom_indices.len() as f64)
}

/// Enclosure ratio derived from boundary atoms (1 - boundary/total)
pub fn boundary_enclosure(atom_indices: &[usize], boundary_atoms: &[usize]) -> f64 {
    if atom_indices.is_empty() {
        return 0.0;
    }
    let boundary_count = boundary_atoms.len();
    1.0 - (boundary_count as f64 / atom_indices.len() as f64)
}

/// Neighbor-based enclosure: counts atoms whose nearest non-pocket atom is within cutoff as boundary
pub fn neighbor_enclosure(
    structure: &ProteinStructure,
    atom_indices: &[usize],
    cutoff: f64,
) -> f64 {
    if atom_indices.is_empty() {
        return 0.0;
    }
    let cutoff2 = cutoff * cutoff;
    let mut boundary_count = 0usize;
    for &idx in atom_indices {
        if let Some(atom) = structure.atoms.get(idx) {
            let mut is_boundary = false;
            for (j, other) in structure.atoms.iter().enumerate() {
                if atom_indices.contains(&j) {
                    continue;
                }
                let dx = atom.coord[0] - other.coord[0];
                let dy = atom.coord[1] - other.coord[1];
                let dz = atom.coord[2] - other.coord[2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 <= cutoff2 {
                    is_boundary = true;
                    break;
                }
            }
            if is_boundary {
                boundary_count += 1;
            }
        }
    }
    1.0 - (boundary_count as f64 / atom_indices.len() as f64)
}

/// Flood-fill cavity volume inside bounding box (voxels not reachable from box boundary)
pub fn flood_fill_cavity_volume(
    structure: &ProteinStructure,
    atom_indices: &[usize],
    resolution: f64,
    probe_radius: f64,
) -> f64 {
    if atom_indices.is_empty() {
        return 0.0;
    }
    let res = resolution.max(0.25);
    let mut min_corner = [f64::INFINITY; 3];
    let mut max_corner = [f64::NEG_INFINITY; 3];
    for &idx in atom_indices {
        if let Some(atom) = structure.atoms.get(idx) {
            for i in 0..3 {
                min_corner[i] = min_corner[i].min(atom.coord[i]);
                max_corner[i] = max_corner[i].max(atom.coord[i]);
            }
        }
    }
    let pad = 3.0;
    for i in 0..3 {
        min_corner[i] -= pad;
        max_corner[i] += pad;
    }

    let nx = ((max_corner[0] - min_corner[0]) / res).ceil() as usize;
    let ny = ((max_corner[1] - min_corner[1]) / res).ceil() as usize;
    let nz = ((max_corner[2] - min_corner[2]) / res).ceil() as usize;
    let total = nx * ny * nz;
    if total == 0 {
        return 0.0;
    }

    let mut occupied = vec![false; total];
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let cx = min_corner[0] + ix as f64 * res;
                let cy = min_corner[1] + iy as f64 * res;
                let cz = min_corner[2] + iz as f64 * res;
                let idx_lin = ix * ny * nz + iy * nz + iz;
                occupied[idx_lin] =
                    voxel_inside_any_atom([cx, cy, cz], structure, atom_indices, probe_radius);
            }
        }
    }

    let mut visited = vec![false; total];
    let mut queue = std::collections::VecDeque::new();
    // seed from boundary voxels
    for ix in [0, nx.saturating_sub(1)].into_iter() {
        for iy in 0..ny {
            for iz in 0..nz {
                let idx_lin = ix * ny * nz + iy * nz + iz;
                if !occupied[idx_lin] && !visited[idx_lin] {
                    visited[idx_lin] = true;
                    queue.push_back((ix, iy, iz));
                }
            }
        }
    }
    for iy in [0, ny.saturating_sub(1)].into_iter() {
        for ix in 0..nx {
            for iz in 0..nz {
                let idx_lin = ix * ny * nz + iy * nz + iz;
                if !occupied[idx_lin] && !visited[idx_lin] {
                    visited[idx_lin] = true;
                    queue.push_back((ix, iy, iz));
                }
            }
        }
    }
    for iz in [0, nz.saturating_sub(1)].into_iter() {
        for ix in 0..nx {
            for iy in 0..ny {
                let idx_lin = ix * ny * nz + iy * nz + iz;
                if !occupied[idx_lin] && !visited[idx_lin] {
                    visited[idx_lin] = true;
                    queue.push_back((ix, iy, iz));
                }
            }
        }
    }

    let neighbors = [
        (1isize, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    ];

    while let Some((x, y, z)) = queue.pop_front() {
        for (dx, dy, dz) in neighbors {
            let nxp = x as isize + dx;
            let nyp = y as isize + dy;
            let nzp = z as isize + dz;
            if nxp < 0 || nyp < 0 || nzp < 0 {
                continue;
            }
            let (nxp, nyp, nzp) = (nxp as usize, nyp as usize, nzp as usize);
            if nxp >= nx || nyp >= ny || nzp >= nz {
                continue;
            }
            let idx_lin = nxp * ny * nz + nyp * nz + nzp;
            if occupied[idx_lin] || visited[idx_lin] {
                continue;
            }
            visited[idx_lin] = true;
            queue.push_back((nxp, nyp, nzp));
        }
    }

    let mut cavity_voxels = 0usize;
    for ix in 0..nx {
        for iy in 0..ny {
            for iz in 0..nz {
                let idx_lin = ix * ny * nz + iy * nz + iz;
                if !occupied[idx_lin] && !visited[idx_lin] {
                    cavity_voxels += 1;
                }
            }
        }
    }

    cavity_voxels as f64 * res * res * res
}

//! Delaunay Tessellation-based Alpha Sphere Detection
//!
//! Implements the true fpocket algorithm for detecting alpha spheres using
//! Delaunay tessellation. Alpha spheres are spheres tangent to 4 atoms,
//! representing the Voronoi vertices in the dual of the Delaunay tessellation.
//!
//! ## Algorithm
//!
//! 1. Lift 3D points to 4D paraboloid: (x,y,z) → (x,y,z,x²+y²+z²)
//! 2. Compute lower convex hull in 4D (gives Delaunay tetrahedra)
//! 3. For each tetrahedron, compute circumsphere
//! 4. Filter by radius and empty space constraints
//!
//! ## References
//!
//! - Le Guilloux, V. et al. (2009) Fpocket: An open source platform for ligand pocket detection
//! - Edelsbrunner, H. & Mücke, E.P. (1994) Three-dimensional alpha shapes

use crate::structure::Atom;
use nalgebra::Vector3;
use std::collections::{HashMap, HashSet};

/// Alpha sphere detected via Delaunay tessellation
#[derive(Debug, Clone)]
pub struct DelaunayAlphaSphere {
    /// Center coordinates [x, y, z]
    pub center: [f64; 3],
    /// Sphere radius
    pub radius: f64,
    /// Indices of 4 defining atoms (tetrahedron vertices)
    pub vertices: [usize; 4],
    /// Whether this sphere is valid (in empty space, correct size)
    pub is_valid: bool,
    /// Polarity score (fraction of polar defining atoms)
    pub polarity: f64,
}

impl Default for DelaunayAlphaSphere {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0, 0.0],
            radius: 0.0,
            vertices: [0, 0, 0, 0],
            is_valid: false,
            polarity: 0.0,
        }
    }
}

/// Delaunay-based alpha sphere detector (fpocket method)
pub struct DelaunayAlphaSphereDetector {
    /// Minimum alpha sphere radius (Å)
    pub min_radius: f64,
    /// Maximum alpha sphere radius (Å)
    pub max_radius: f64,
    /// Probe radius for solvent exclusion (Å)
    pub probe_radius: f64,
    /// Tolerance for degenerate cases
    pub tolerance: f64,
}

impl Default for DelaunayAlphaSphereDetector {
    fn default() -> Self {
        Self {
            min_radius: 2.8,   // Slightly larger than water
            max_radius: 10.0,  // Max pocket depth
            probe_radius: 1.4, // Water probe
            tolerance: 1e-10,
        }
    }
}

impl DelaunayAlphaSphereDetector {
    /// Create detector with custom radius bounds
    pub fn new(min_radius: f64, max_radius: f64) -> Self {
        Self {
            min_radius,
            max_radius,
            ..Default::default()
        }
    }

    /// Detect alpha spheres using 3D Delaunay tessellation
    pub fn detect(&self, atoms: &[Atom]) -> Vec<DelaunayAlphaSphere> {
        if atoms.len() < 4 {
            return Vec::new();
        }

        // Step 1: Build 3D Delaunay tessellation
        let tetrahedra = self.compute_delaunay_3d(atoms);

        // Step 2: For each tetrahedron, compute circumsphere (Voronoi vertex)
        let mut alpha_spheres = Vec::with_capacity(tetrahedra.len());

        for tet in &tetrahedra {
            if let Some(mut sphere) = self.compute_circumsphere(atoms, tet) {
                // Step 3: Filter by radius constraints
                if sphere.radius >= self.min_radius && sphere.radius <= self.max_radius {
                    // Step 4: Check if sphere center is outside all atoms (empty space)
                    if self.is_in_empty_space(atoms, &sphere) {
                        // Calculate polarity
                        sphere.polarity = self.calculate_polarity(atoms, &sphere);
                        sphere.is_valid = true;
                        alpha_spheres.push(sphere);
                    }
                }
            }
        }

        // Step 5: Remove redundant spheres (same Voronoi vertex from multiple tets)
        self.deduplicate_spheres(alpha_spheres)
    }

    /// Compute 3D Delaunay tessellation using convex hull lifting
    ///
    /// The key insight is that Delaunay triangulation in d dimensions equals
    /// the lower convex hull in d+1 dimensions when points are lifted to
    /// a paraboloid.
    fn compute_delaunay_3d(&self, atoms: &[Atom]) -> Vec<[usize; 4]> {
        let n = atoms.len();

        // Lift 3D points to 4D paraboloid: (x,y,z) -> (x,y,z, x²+y²+z²)
        let mut points_4d: Vec<[f64; 4]> = Vec::with_capacity(n);

        for atom in atoms {
            let [x, y, z] = atom.coord;
            let w = x * x + y * y + z * z;
            points_4d.push([x, y, z, w]);
        }

        // Compute 4D convex hull (lower hull gives Delaunay)
        self.convex_hull_4d_lower(&points_4d)
    }

    /// Incremental convex hull in 4D (lower facets only)
    fn convex_hull_4d_lower(&self, points: &[[f64; 4]]) -> Vec<[usize; 4]> {
        let n = points.len();

        // Handle small cases
        if n < 4 {
            return Vec::new();
        }

        if n == 4 {
            // Check if points are coplanar in 3D
            if self.are_coplanar(&points[0], &points[1], &points[2], &points[3]) {
                return Vec::new();
            }
            return vec![[0, 1, 2, 3]];
        }

        let mut tetrahedra = Vec::new();

        // Find initial simplex (5 non-coplanar points)
        let initial = match self.find_initial_simplex(points) {
            Some(s) => s,
            None => return Vec::new(),
        };

        // Build initial hull from 5 points (creates 5 facets)
        let mut hull_facets: Vec<([usize; 4], [f64; 5])> = Vec::new();

        // Create facets of the initial 4-simplex
        for i in 0..5 {
            let mut facet = [0usize; 4];
            let mut idx = 0;
            for j in 0..5 {
                if i != j {
                    facet[idx] = initial[j];
                    idx += 1;
                }
            }
            facet.sort();

            // Compute outward normal
            let normal = self.facet_normal_4d(points, &facet, initial[i]);

            // Keep only lower facets (normal[3] < 0 means facing "down" in w)
            if normal[3] < -self.tolerance {
                hull_facets.push((facet, normal));
            }
        }

        // Track which points are in the hull
        let mut in_hull: HashSet<usize> = initial.iter().cloned().collect();

        // Incrementally add remaining points
        for i in 0..n {
            if in_hull.contains(&i) {
                continue;
            }

            let p = &points[i];

            // Find visible facets (point is "above" facet in 4D)
            let mut visible: Vec<usize> = Vec::new();
            for (idx, (facet, normal)) in hull_facets.iter().enumerate() {
                let d = self.signed_distance_4d(p, &points[facet[0]], normal);
                if d > self.tolerance {
                    visible.push(idx);
                }
            }

            if visible.is_empty() {
                continue; // Point inside hull
            }

            // Find horizon edges (ridges shared by exactly one visible facet)
            let horizon = self.find_horizon_4d(&hull_facets, &visible);

            // Remove visible facets and add new ones
            let mut new_facets: Vec<([usize; 4], [f64; 5])> = hull_facets
                .iter()
                .enumerate()
                .filter(|(idx, _)| !visible.contains(idx))
                .map(|(_, f)| f.clone())
                .collect();

            // Add new facets connecting point to horizon
            for edge in horizon {
                let mut new_facet = [edge[0], edge[1], edge[2], i];
                new_facet.sort();

                // Find a point inside the hull for orientation
                let interior_point = initial[0];
                let normal = self.facet_normal_4d(points, &new_facet, interior_point);

                // Keep only lower facets
                if normal[3] < -self.tolerance {
                    new_facets.push((new_facet, normal));
                }
            }

            hull_facets = new_facets;
            in_hull.insert(i);
        }

        // Extract tetrahedra from lower hull facets
        for (facet, _) in hull_facets {
            tetrahedra.push(facet);
        }

        tetrahedra
    }

    /// Check if 4 points are coplanar in 3D projection
    fn are_coplanar(&self, a: &[f64; 4], b: &[f64; 4], c: &[f64; 4], d: &[f64; 4]) -> bool {
        // Use first 3 coordinates only
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let ad = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];

        // Scalar triple product (volume of parallelepiped)
        let vol = ab[0] * (ac[1] * ad[2] - ac[2] * ad[1])
            - ab[1] * (ac[0] * ad[2] - ac[2] * ad[0])
            + ab[2] * (ac[0] * ad[1] - ac[1] * ad[0]);

        vol.abs() < self.tolerance
    }

    /// Compute circumsphere of tetrahedron (Voronoi vertex)
    fn compute_circumsphere(
        &self,
        atoms: &[Atom],
        tet: &[usize; 4],
    ) -> Option<DelaunayAlphaSphere> {
        let p0 = Vector3::new(atoms[tet[0]].coord[0], atoms[tet[0]].coord[1], atoms[tet[0]].coord[2]);
        let p1 = Vector3::new(atoms[tet[1]].coord[0], atoms[tet[1]].coord[1], atoms[tet[1]].coord[2]);
        let p2 = Vector3::new(atoms[tet[2]].coord[0], atoms[tet[2]].coord[1], atoms[tet[2]].coord[2]);
        let p3 = Vector3::new(atoms[tet[3]].coord[0], atoms[tet[3]].coord[1], atoms[tet[3]].coord[2]);

        // Translate to p0 at origin
        let a = p1 - p0;
        let b = p2 - p0;
        let c = p3 - p0;

        // Solve for circumcenter using determinant formulas
        let a_sq = a.dot(&a);
        let b_sq = b.dot(&b);
        let c_sq = c.dot(&c);

        // Cross products for determinant
        let b_cross_c = b.cross(&c);
        let c_cross_a = c.cross(&a);
        let a_cross_b = a.cross(&b);

        let denom = 2.0 * a.dot(&b_cross_c);

        if denom.abs() < self.tolerance {
            return None; // Degenerate tetrahedron (flat or very thin)
        }

        let center_local = (a_sq * b_cross_c + b_sq * c_cross_a + c_sq * a_cross_b) / denom;
        let center = p0 + center_local;
        let radius = center_local.norm();

        Some(DelaunayAlphaSphere {
            center: [center.x, center.y, center.z],
            radius,
            vertices: *tet,
            is_valid: false,
            polarity: 0.0,
        })
    }

    /// Check if sphere center is in empty space (not inside any atom)
    fn is_in_empty_space(&self, atoms: &[Atom], sphere: &DelaunayAlphaSphere) -> bool {
        let center = Vector3::new(sphere.center[0], sphere.center[1], sphere.center[2]);

        for atom in atoms {
            let atom_pos = Vector3::new(atom.coord[0], atom.coord[1], atom.coord[2]);
            let dist = (center - atom_pos).norm();
            let atom_radius = atom.vdw_radius();

            // Sphere center must be outside atom + probe radius
            // Allow small tolerance for numerical precision
            if dist < atom_radius + self.probe_radius - 0.3 {
                return false;
            }
        }

        true
    }

    /// Calculate polarity of alpha sphere based on defining atoms
    fn calculate_polarity(&self, atoms: &[Atom], sphere: &DelaunayAlphaSphere) -> f64 {
        let polar_count = sphere
            .vertices
            .iter()
            .filter(|&&idx| self.is_polar_atom(&atoms[idx]))
            .count();

        polar_count as f64 / 4.0
    }

    /// Check if atom is polar (N, O, S)
    fn is_polar_atom(&self, atom: &Atom) -> bool {
        let elem = if !atom.element.is_empty() {
            atom.element.trim().to_uppercase()
        } else {
            atom.name.trim().chars().take(1).collect::<String>().to_uppercase()
        };

        matches!(elem.as_str(), "N" | "O" | "S")
    }

    /// Remove duplicate spheres (same Voronoi vertex from numerical precision)
    fn deduplicate_spheres(
        &self,
        mut spheres: Vec<DelaunayAlphaSphere>,
    ) -> Vec<DelaunayAlphaSphere> {
        if spheres.len() < 2 {
            return spheres;
        }

        // Sort by center coordinates for efficient dedup
        spheres.sort_by(|a, b| {
            a.center[0]
                .partial_cmp(&b.center[0])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    a.center[1]
                        .partial_cmp(&b.center[1])
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
                .then(
                    a.center[2]
                        .partial_cmp(&b.center[2])
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });

        let mut unique = Vec::with_capacity(spheres.len());
        unique.push(spheres[0].clone());

        for sphere in spheres.into_iter().skip(1) {
            let last = unique.last().unwrap();
            let dist_sq = (sphere.center[0] - last.center[0]).powi(2)
                + (sphere.center[1] - last.center[1]).powi(2)
                + (sphere.center[2] - last.center[2]).powi(2);

            if dist_sq > 0.01 {
                // > 0.1Å apart
                unique.push(sphere);
            }
        }

        unique
    }

    /// Find 5 affinely independent points for initial simplex
    fn find_initial_simplex(&self, points: &[[f64; 4]]) -> Option<[usize; 5]> {
        let n = points.len();
        if n < 5 {
            return None;
        }

        // Start with first point
        let mut simplex = vec![0usize];

        // Find point furthest from first
        let mut max_dist = 0.0;
        let mut best = 1;
        for i in 1..n {
            let d = self.dist_4d(&points[0], &points[i]);
            if d > max_dist {
                max_dist = d;
                best = i;
            }
        }
        if max_dist < self.tolerance {
            return None;
        }
        simplex.push(best);

        // Find point furthest from line
        max_dist = 0.0;
        best = 0;
        for i in 0..n {
            if simplex.contains(&i) {
                continue;
            }
            let d = self.dist_to_line_4d(&points[i], &points[simplex[0]], &points[simplex[1]]);
            if d > max_dist {
                max_dist = d;
                best = i;
            }
        }
        if max_dist < self.tolerance {
            return None;
        }
        simplex.push(best);

        // Find point furthest from plane (3 points)
        max_dist = 0.0;
        best = 0;
        for i in 0..n {
            if simplex.contains(&i) {
                continue;
            }
            let d = self.dist_to_plane_4d(
                &points[i],
                &points[simplex[0]],
                &points[simplex[1]],
                &points[simplex[2]],
            );
            if d > max_dist {
                max_dist = d;
                best = i;
            }
        }
        if max_dist < self.tolerance {
            return None;
        }
        simplex.push(best);

        // Find point furthest from hyperplane (4 points)
        max_dist = 0.0;
        best = 0;
        for i in 0..n {
            if simplex.contains(&i) {
                continue;
            }
            let d = self.dist_to_hyperplane_4d(
                &points[i],
                &points[simplex[0]],
                &points[simplex[1]],
                &points[simplex[2]],
                &points[simplex[3]],
            );
            if d > max_dist {
                max_dist = d;
                best = i;
            }
        }
        if max_dist < self.tolerance {
            return None;
        }
        simplex.push(best);

        Some([simplex[0], simplex[1], simplex[2], simplex[3], simplex[4]])
    }

    fn dist_4d(&self, a: &[f64; 4], b: &[f64; 4]) -> f64 {
        ((a[0] - b[0]).powi(2)
            + (a[1] - b[1]).powi(2)
            + (a[2] - b[2]).powi(2)
            + (a[3] - b[3]).powi(2))
        .sqrt()
    }

    fn dist_to_line_4d(&self, p: &[f64; 4], a: &[f64; 4], b: &[f64; 4]) -> f64 {
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2], b[3] - a[3]];
        let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2], p[3] - a[3]];

        let ab_len_sq = ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2] + ab[3] * ab[3];
        if ab_len_sq < self.tolerance {
            return self.dist_4d(p, a);
        }

        let t = (ap[0] * ab[0] + ap[1] * ab[1] + ap[2] * ab[2] + ap[3] * ab[3]) / ab_len_sq;
        let closest = [
            a[0] + t * ab[0],
            a[1] + t * ab[1],
            a[2] + t * ab[2],
            a[3] + t * ab[3],
        ];

        self.dist_4d(p, &closest)
    }

    fn dist_to_plane_4d(&self, p: &[f64; 4], a: &[f64; 4], b: &[f64; 4], c: &[f64; 4]) -> f64 {
        // Use Gram-Schmidt orthogonalization
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2], b[3] - a[3]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2], c[3] - a[3]];
        let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2], p[3] - a[3]];

        // Orthonormalize ab
        let ab_len = (ab[0] * ab[0] + ab[1] * ab[1] + ab[2] * ab[2] + ab[3] * ab[3]).sqrt();
        if ab_len < self.tolerance {
            return self.dist_4d(p, a);
        }
        let e1 = [ab[0] / ab_len, ab[1] / ab_len, ab[2] / ab_len, ab[3] / ab_len];

        // ac orthogonal to e1
        let ac_dot_e1 = ac[0] * e1[0] + ac[1] * e1[1] + ac[2] * e1[2] + ac[3] * e1[3];
        let ac_orth = [
            ac[0] - ac_dot_e1 * e1[0],
            ac[1] - ac_dot_e1 * e1[1],
            ac[2] - ac_dot_e1 * e1[2],
            ac[3] - ac_dot_e1 * e1[3],
        ];
        let ac_orth_len = (ac_orth[0] * ac_orth[0]
            + ac_orth[1] * ac_orth[1]
            + ac_orth[2] * ac_orth[2]
            + ac_orth[3] * ac_orth[3])
        .sqrt();
        if ac_orth_len < self.tolerance {
            return self.dist_to_line_4d(p, a, b);
        }
        let e2 = [
            ac_orth[0] / ac_orth_len,
            ac_orth[1] / ac_orth_len,
            ac_orth[2] / ac_orth_len,
            ac_orth[3] / ac_orth_len,
        ];

        // Project ap onto span of e1, e2
        let proj1 = ap[0] * e1[0] + ap[1] * e1[1] + ap[2] * e1[2] + ap[3] * e1[3];
        let proj2 = ap[0] * e2[0] + ap[1] * e2[1] + ap[2] * e2[2] + ap[3] * e2[3];
        let proj = [
            proj1 * e1[0] + proj2 * e2[0],
            proj1 * e1[1] + proj2 * e2[1],
            proj1 * e1[2] + proj2 * e2[2],
            proj1 * e1[3] + proj2 * e2[3],
        ];

        let diff = [
            ap[0] - proj[0],
            ap[1] - proj[1],
            ap[2] - proj[2],
            ap[3] - proj[3],
        ];
        (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] + diff[3] * diff[3]).sqrt()
    }

    fn dist_to_hyperplane_4d(
        &self,
        p: &[f64; 4],
        a: &[f64; 4],
        b: &[f64; 4],
        c: &[f64; 4],
        d: &[f64; 4],
    ) -> f64 {
        let normal = self.hyperplane_normal_4d(a, b, c, d);
        let ap = [p[0] - a[0], p[1] - a[1], p[2] - a[2], p[3] - a[3]];

        let norm_len = (normal[0] * normal[0]
            + normal[1] * normal[1]
            + normal[2] * normal[2]
            + normal[3] * normal[3])
        .sqrt();

        if norm_len < self.tolerance {
            return 0.0;
        }

        (ap[0] * normal[0] + ap[1] * normal[1] + ap[2] * normal[2] + ap[3] * normal[3]).abs()
            / norm_len
    }

    fn hyperplane_normal_4d(
        &self,
        a: &[f64; 4],
        b: &[f64; 4],
        c: &[f64; 4],
        d: &[f64; 4],
    ) -> [f64; 4] {
        let v1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2], b[3] - a[3]];
        let v2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2], c[3] - a[3]];
        let v3 = [d[0] - a[0], d[1] - a[1], d[2] - a[2], d[3] - a[3]];

        // 4D cross product using cofactor expansion
        let n0 = v1[1] * (v2[2] * v3[3] - v2[3] * v3[2])
            - v1[2] * (v2[1] * v3[3] - v2[3] * v3[1])
            + v1[3] * (v2[1] * v3[2] - v2[2] * v3[1]);

        let n1 = -(v1[0] * (v2[2] * v3[3] - v2[3] * v3[2])
            - v1[2] * (v2[0] * v3[3] - v2[3] * v3[0])
            + v1[3] * (v2[0] * v3[2] - v2[2] * v3[0]));

        let n2 = v1[0] * (v2[1] * v3[3] - v2[3] * v3[1])
            - v1[1] * (v2[0] * v3[3] - v2[3] * v3[0])
            + v1[3] * (v2[0] * v3[1] - v2[1] * v3[0]);

        let n3 = -(v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
            - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
            + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]));

        [n0, n1, n2, n3]
    }

    fn facet_normal_4d(
        &self,
        points: &[[f64; 4]],
        facet: &[usize; 4],
        interior_point: usize,
    ) -> [f64; 5] {
        let a = &points[facet[0]];
        let b = &points[facet[1]];
        let c = &points[facet[2]];
        let d = &points[facet[3]];

        let mut normal = self.hyperplane_normal_4d(a, b, c, d);

        // Orient normal to point away from interior
        let int_pt = &points[interior_point];
        let to_interior = [
            int_pt[0] - a[0],
            int_pt[1] - a[1],
            int_pt[2] - a[2],
            int_pt[3] - a[3],
        ];

        let dot = normal[0] * to_interior[0]
            + normal[1] * to_interior[1]
            + normal[2] * to_interior[2]
            + normal[3] * to_interior[3];

        if dot > 0.0 {
            normal = [-normal[0], -normal[1], -normal[2], -normal[3]];
        }

        let offset = -(normal[0] * a[0] + normal[1] * a[1] + normal[2] * a[2] + normal[3] * a[3]);

        [normal[0], normal[1], normal[2], normal[3], offset]
    }

    fn signed_distance_4d(&self, p: &[f64; 4], _origin: &[f64; 4], normal: &[f64; 5]) -> f64 {
        let d =
            p[0] * normal[0] + p[1] * normal[1] + p[2] * normal[2] + p[3] * normal[3] + normal[4];
        let norm_len = (normal[0] * normal[0]
            + normal[1] * normal[1]
            + normal[2] * normal[2]
            + normal[3] * normal[3])
        .sqrt();

        if norm_len < self.tolerance {
            return 0.0;
        }

        d / norm_len
    }

    fn find_horizon_4d(
        &self,
        facets: &[([usize; 4], [f64; 5])],
        visible: &[usize],
    ) -> Vec<[usize; 3]> {
        // Count how many times each ridge appears in visible facets
        let mut ridge_count: HashMap<[usize; 3], usize> = HashMap::new();

        for &vis_idx in visible {
            let facet = &facets[vis_idx].0;

            // Each 4-facet has 4 ridges (3-faces)
            for i in 0..4 {
                let mut ridge = [0usize; 3];
                let mut idx = 0;
                for j in 0..4 {
                    if i != j {
                        ridge[idx] = facet[j];
                        idx += 1;
                    }
                }
                ridge.sort();
                *ridge_count.entry(ridge).or_insert(0) += 1;
            }
        }

        // Horizon = ridges with count == 1 (on boundary between visible/invisible)
        ridge_count
            .into_iter()
            .filter(|(_, count)| *count == 1)
            .map(|(ridge, _)| ridge)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_atom(x: f64, y: f64, z: f64, element: &str) -> Atom {
        Atom::new(
            1,
            element.to_string(),
            "ALA".to_string(),
            'A',
            1,
            [x, y, z],
            1.0,
            20.0,
            element.to_string(),
        )
    }

    #[test]
    fn test_circumsphere_regular_tetrahedron() {
        let detector = DelaunayAlphaSphereDetector::default();

        // Regular tetrahedron centered at origin
        let a = 2.0_f64.sqrt();
        let atoms = vec![
            make_atom(1.0, 1.0, 1.0, "C"),
            make_atom(1.0, -1.0, -1.0, "C"),
            make_atom(-1.0, 1.0, -1.0, "C"),
            make_atom(-1.0, -1.0, 1.0, "C"),
        ];

        let sphere = detector.compute_circumsphere(&atoms, &[0, 1, 2, 3]).unwrap();

        // Center should be at origin
        assert!(sphere.center[0].abs() < 0.01, "x = {}", sphere.center[0]);
        assert!(sphere.center[1].abs() < 0.01, "y = {}", sphere.center[1]);
        assert!(sphere.center[2].abs() < 0.01, "z = {}", sphere.center[2]);

        // Radius should be sqrt(3) ≈ 1.732
        assert!(
            (sphere.radius - 3.0_f64.sqrt()).abs() < 0.01,
            "radius = {}",
            sphere.radius
        );
    }

    #[test]
    fn test_degenerate_tetrahedron() {
        let detector = DelaunayAlphaSphereDetector::default();

        // Coplanar points (degenerate)
        let atoms = vec![
            make_atom(0.0, 0.0, 0.0, "C"),
            make_atom(1.0, 0.0, 0.0, "C"),
            make_atom(0.0, 1.0, 0.0, "C"),
            make_atom(1.0, 1.0, 0.0, "C"),
        ];

        let sphere = detector.compute_circumsphere(&atoms, &[0, 1, 2, 3]);
        assert!(sphere.is_none(), "Degenerate tetrahedron should return None");
    }

    #[test]
    fn test_alpha_sphere_filtering() {
        let detector = DelaunayAlphaSphereDetector {
            min_radius: 3.0,
            max_radius: 8.0,
            ..Default::default()
        };

        // Create a simple pocket
        let atoms = vec![
            make_atom(0.0, 0.0, 0.0, "C"),
            make_atom(5.0, 0.0, 0.0, "C"),
            make_atom(2.5, 4.33, 0.0, "C"),
            make_atom(2.5, 1.44, 4.08, "C"),
            make_atom(10.0, 0.0, 0.0, "C"), // Far atom
        ];

        let spheres = detector.detect(&atoms);

        // All returned spheres should be in range
        for sphere in &spheres {
            assert!(sphere.radius >= detector.min_radius);
            assert!(sphere.radius <= detector.max_radius);
            assert!(sphere.is_valid);
        }
    }

    #[test]
    fn test_polarity_calculation() {
        let detector = DelaunayAlphaSphereDetector::default();

        // Mix of polar and non-polar atoms
        let atoms = vec![
            make_atom(0.0, 0.0, 0.0, "C"),
            make_atom(3.0, 0.0, 0.0, "N"),
            make_atom(1.5, 2.6, 0.0, "O"),
            make_atom(1.5, 0.87, 2.45, "C"),
        ];

        let sphere = DelaunayAlphaSphere {
            center: [1.5, 1.0, 1.0],
            radius: 2.0,
            vertices: [0, 1, 2, 3],
            is_valid: true,
            polarity: 0.0,
        };

        let polarity = detector.calculate_polarity(&atoms, &sphere);

        // 2 polar atoms (N, O) out of 4
        assert!((polarity - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_empty_input() {
        let detector = DelaunayAlphaSphereDetector::default();
        let spheres = detector.detect(&[]);
        assert!(spheres.is_empty());
    }

    #[test]
    fn test_insufficient_points() {
        let detector = DelaunayAlphaSphereDetector::default();

        let atoms = vec![
            make_atom(0.0, 0.0, 0.0, "C"),
            make_atom(1.0, 0.0, 0.0, "C"),
            make_atom(0.0, 1.0, 0.0, "C"),
        ];

        let spheres = detector.detect(&atoms);
        assert!(spheres.is_empty());
    }
}

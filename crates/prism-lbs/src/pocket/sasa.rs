//! Shrake-Rupley Solvent Accessible Surface Area Calculation
//!
//! Implements the classic Shrake-Rupley algorithm for computing the solvent
//! accessible surface area (SASA) of atoms in a protein structure.
//!
//! The algorithm works by distributing test points uniformly on a sphere
//! around each atom and counting how many points are not occluded by
//! neighboring atoms.
//!
//! ## Algorithm
//!
//! 1. Generate N test points uniformly distributed on a unit sphere (Fibonacci lattice)
//! 2. For each atom i:
//!    a. Scale test points to atom surface (radius = vdW + probe)
//!    b. Count points not inside any neighbor's sphere
//!    c. SASA_i = (accessible/N) × 4πr²
//!
//! ## References
//!
//! - Shrake, A. & Rupley, J.A. (1973) J Mol Biol 79, 351-371
//! - Lee, B. & Richards, F.M. (1971) J Mol Biol 55, 379-400

use crate::structure::Atom;
use std::collections::HashMap;
use std::f64::consts::PI;

/// Van der Waals radii (Å) for common elements
pub fn get_vdw_radius(element: &str) -> f64 {
    match element.trim().to_uppercase().as_str() {
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "S" => 1.80,
        "P" => 1.80,
        "H" => 1.20,
        "F" => 1.47,
        "CL" => 1.75,
        "BR" => 1.85,
        "I" => 1.98,
        "FE" => 1.80,
        "ZN" => 1.39,
        "MG" => 1.73,
        "CA" => 2.31,
        "NA" => 2.27,
        "K" => 2.75,
        _ => 1.70, // Default to carbon
    }
}

/// Result of SASA calculation
#[derive(Debug, Clone)]
pub struct SasaResult {
    /// Per-atom SASA values in Å²
    pub atom_sasa: Vec<f64>,
    /// Total SASA of the structure in Å²
    pub total_sasa: f64,
    /// Per-residue SASA values in Å²
    pub residue_sasa: HashMap<i32, f64>,
    /// Indices of surface atoms (SASA > threshold)
    pub surface_atoms: Vec<usize>,
    /// Indices of buried atoms (SASA ≤ threshold)
    pub buried_atoms: Vec<usize>,
}

/// Shrake-Rupley solvent accessible surface area calculator
pub struct ShrakeRupleySASA {
    /// Number of test points per atom (92 = fibonacci sphere, 252 = higher precision)
    pub n_points: usize,
    /// Probe radius (water = 1.4 Å)
    pub probe_radius: f64,
    /// Threshold for surface/buried classification (Å²)
    pub surface_threshold: f64,
    /// Precomputed sphere points
    sphere_points: Vec<[f64; 3]>,
}

impl Default for ShrakeRupleySASA {
    fn default() -> Self {
        let mut sasa = Self {
            n_points: 92,
            probe_radius: 1.4,
            surface_threshold: 5.0,
            sphere_points: Vec::new(),
        };
        sasa.sphere_points = sasa.generate_sphere_points();
        sasa
    }
}

impl ShrakeRupleySASA {
    /// Create a new SASA calculator with custom parameters
    pub fn new(n_points: usize, probe_radius: f64) -> Self {
        let mut sasa = Self {
            n_points,
            probe_radius,
            surface_threshold: 5.0,
            sphere_points: Vec::new(),
        };
        sasa.sphere_points = sasa.generate_sphere_points();
        sasa
    }

    /// Create high-precision calculator (252 points)
    pub fn high_precision() -> Self {
        Self::new(252, 1.4)
    }

    /// Generate points on unit sphere using Fibonacci lattice
    ///
    /// The Fibonacci lattice provides a nearly uniform distribution of points
    /// on the sphere surface with O(N) complexity.
    fn generate_sphere_points(&self) -> Vec<[f64; 3]> {
        let mut points = Vec::with_capacity(self.n_points);

        let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let angle_increment = 2.0 * PI / golden_ratio;

        for i in 0..self.n_points {
            // t ranges from 0 to 1
            let t = (i as f64 + 0.5) / self.n_points as f64;

            // Inclination (polar angle from z-axis)
            let inclination = (1.0 - 2.0 * t).acos();

            // Azimuth (longitude)
            let azimuth = angle_increment * i as f64;

            // Convert to Cartesian
            let sin_inc = inclination.sin();
            let x = sin_inc * azimuth.cos();
            let y = sin_inc * azimuth.sin();
            let z = inclination.cos();

            points.push([x, y, z]);
        }

        points
    }

    /// Calculate SASA for all atoms
    pub fn calculate(&self, atoms: &[Atom]) -> SasaResult {
        let n = atoms.len();
        if n == 0 {
            return SasaResult {
                atom_sasa: Vec::new(),
                total_sasa: 0.0,
                residue_sasa: HashMap::new(),
                surface_atoms: Vec::new(),
                buried_atoms: Vec::new(),
            };
        }

        // Build neighbor list for efficient collision detection
        let neighbors = self.build_neighbor_list(atoms);

        // Calculate SASA for each atom
        let mut atom_sasa = vec![0.0; n];
        let mut residue_sasa: HashMap<i32, f64> = HashMap::new();

        for i in 0..n {
            let atom_i = &atoms[i];
            let radius_i = self.get_atom_radius(atom_i) + self.probe_radius;
            let center_i = atom_i.coord;

            let mut accessible_count = 0;

            // Check each test point
            for point in &self.sphere_points {
                // Scale and translate point to atom surface
                let test_point = [
                    center_i[0] + radius_i * point[0],
                    center_i[1] + radius_i * point[1],
                    center_i[2] + radius_i * point[2],
                ];

                // Check if point is buried by any neighbor
                let mut is_accessible = true;

                for &j in &neighbors[i] {
                    let atom_j = &atoms[j];
                    let radius_j = self.get_atom_radius(atom_j) + self.probe_radius;
                    let center_j = atom_j.coord;

                    // Distance squared from test point to neighbor center
                    let dist_sq = (test_point[0] - center_j[0]).powi(2)
                        + (test_point[1] - center_j[1]).powi(2)
                        + (test_point[2] - center_j[2]).powi(2);

                    // If inside neighbor's sphere, point is not accessible
                    if dist_sq < radius_j * radius_j {
                        is_accessible = false;
                        break;
                    }
                }

                if is_accessible {
                    accessible_count += 1;
                }
            }

            // SASA = fraction_accessible × sphere_surface_area
            let fraction = accessible_count as f64 / self.n_points as f64;
            let sasa = fraction * 4.0 * PI * radius_i * radius_i;
            atom_sasa[i] = sasa;

            // Accumulate per-residue SASA
            *residue_sasa.entry(atom_i.residue_seq).or_insert(0.0) += sasa;
        }

        // Calculate total and classify atoms
        let total_sasa = atom_sasa.iter().sum();

        let mut surface_atoms = Vec::new();
        let mut buried_atoms = Vec::new();

        for (i, &sasa) in atom_sasa.iter().enumerate() {
            if sasa > self.surface_threshold {
                surface_atoms.push(i);
            } else {
                buried_atoms.push(i);
            }
        }

        SasaResult {
            atom_sasa,
            total_sasa,
            residue_sasa,
            surface_atoms,
            buried_atoms,
        }
    }

    /// Identify surface atoms (SASA > threshold)
    pub fn identify_surface_atoms(&self, atoms: &[Atom], threshold: f64) -> Vec<usize> {
        let result = self.calculate(atoms);

        result
            .atom_sasa
            .iter()
            .enumerate()
            .filter(|(_, &s)| s > threshold)
            .map(|(i, _)| i)
            .collect()
    }

    /// Calculate relative SASA (% of max possible for residue type)
    pub fn calculate_relative_sasa(&self, atoms: &[Atom]) -> HashMap<i32, f64> {
        let result = self.calculate(atoms);

        // Max SASA values for extended tripeptide (Gly-X-Gly) from Miller et al.
        let max_sasa: HashMap<&str, f64> = [
            ("ALA", 113.0),
            ("ARG", 241.0),
            ("ASN", 158.0),
            ("ASP", 151.0),
            ("CYS", 140.0),
            ("GLN", 189.0),
            ("GLU", 183.0),
            ("GLY", 85.0),
            ("HIS", 194.0),
            ("ILE", 182.0),
            ("LEU", 180.0),
            ("LYS", 211.0),
            ("MET", 204.0),
            ("PHE", 218.0),
            ("PRO", 143.0),
            ("SER", 122.0),
            ("THR", 146.0),
            ("TRP", 259.0),
            ("TYR", 229.0),
            ("VAL", 160.0),
        ]
        .iter()
        .cloned()
        .collect();

        // Get residue names
        let mut residue_names: HashMap<i32, String> = HashMap::new();
        for atom in atoms {
            residue_names
                .entry(atom.residue_seq)
                .or_insert_with(|| atom.residue_name.trim().to_uppercase());
        }

        // Calculate relative SASA
        let mut relative_sasa = HashMap::new();
        for (&res_seq, &sasa) in &result.residue_sasa {
            if let Some(res_name) = residue_names.get(&res_seq) {
                let max = max_sasa.get(res_name.as_str()).unwrap_or(&150.0);
                relative_sasa.insert(res_seq, (sasa / max).min(1.0));
            }
        }

        relative_sasa
    }

    /// Get VdW radius for an atom
    fn get_atom_radius(&self, atom: &Atom) -> f64 {
        // Use the Atom's built-in vdw_radius method
        atom.vdw_radius()
    }

    /// Build neighbor list for efficient collision detection using spatial hash
    fn build_neighbor_list(&self, atoms: &[Atom]) -> Vec<Vec<usize>> {
        let n = atoms.len();

        // Maximum interaction distance: 2 × (max_vdw + probe)
        let max_radius = 3.0 + 2.0 * self.probe_radius; // ~5.8 Å
        let cutoff = 2.0 * max_radius;
        let cutoff_sq = cutoff * cutoff;

        let mut neighbors = vec![Vec::new(); n];

        // Use spatial hash for O(n) neighbor finding
        let cell_size = cutoff;
        let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();

        // Insert atoms into grid
        for (i, atom) in atoms.iter().enumerate() {
            let cell = (
                (atom.coord[0] / cell_size).floor() as i32,
                (atom.coord[1] / cell_size).floor() as i32,
                (atom.coord[2] / cell_size).floor() as i32,
            );
            grid.entry(cell).or_insert_with(Vec::new).push(i);
        }

        // Find neighbors
        for i in 0..n {
            let atom_i = &atoms[i];
            let cell = (
                (atom_i.coord[0] / cell_size).floor() as i32,
                (atom_i.coord[1] / cell_size).floor() as i32,
                (atom_i.coord[2] / cell_size).floor() as i32,
            );

            // Check 27 neighboring cells
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        let neighbor_cell = (cell.0 + dx, cell.1 + dy, cell.2 + dz);

                        if let Some(cell_atoms) = grid.get(&neighbor_cell) {
                            for &j in cell_atoms {
                                if i == j {
                                    continue;
                                }

                                let atom_j = &atoms[j];
                                let dist_sq = (atom_i.coord[0] - atom_j.coord[0]).powi(2)
                                    + (atom_i.coord[1] - atom_j.coord[1]).powi(2)
                                    + (atom_i.coord[2] - atom_j.coord[2]).powi(2);

                                if dist_sq < cutoff_sq {
                                    neighbors[i].push(j);
                                }
                            }
                        }
                    }
                }
            }
        }

        neighbors
    }
}

/// Calculate SASA for a subset of atoms (e.g., pocket lining residues)
pub fn calculate_pocket_sasa(atoms: &[Atom], pocket_atoms: &[usize]) -> f64 {
    let sasa_calc = ShrakeRupleySASA::default();
    let result = sasa_calc.calculate(atoms);

    pocket_atoms
        .iter()
        .filter_map(|&i| result.atom_sasa.get(i))
        .sum()
}

/// Calculate burial fraction for a set of residues
pub fn calculate_burial_fraction(atoms: &[Atom], residues: &[i32]) -> f64 {
    let sasa_calc = ShrakeRupleySASA::default();
    let relative_sasa = sasa_calc.calculate_relative_sasa(atoms);

    let total: f64 = residues
        .iter()
        .filter_map(|r| relative_sasa.get(r))
        .sum();

    let count = residues
        .iter()
        .filter(|r| relative_sasa.contains_key(r))
        .count();

    if count > 0 {
        1.0 - (total / count as f64)
    } else {
        0.0
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
    fn test_isolated_atom_sasa() {
        let sasa_calc = ShrakeRupleySASA::default();

        // Single carbon atom at origin
        let atoms = vec![make_atom(0.0, 0.0, 0.0, "C")];

        let result = sasa_calc.calculate(&atoms);

        // SASA should be full sphere: 4π(1.7 + 1.4)² ≈ 120.8 Å²
        let expected = 4.0 * PI * (1.7 + 1.4_f64).powi(2);
        let relative_error = (result.atom_sasa[0] - expected).abs() / expected;

        assert!(
            relative_error < 0.05,
            "Expected SASA ≈ {:.1}, got {:.1} (error: {:.1}%)",
            expected,
            result.atom_sasa[0],
            relative_error * 100.0
        );
    }

    #[test]
    fn test_buried_atom_sasa() {
        let sasa_calc = ShrakeRupleySASA::default();

        // Central atom surrounded by 6 atoms in octahedral arrangement
        let d = 2.5; // Distance where spheres overlap significantly
        let atoms = vec![
            make_atom(0.0, 0.0, 0.0, "C"),
            make_atom(d, 0.0, 0.0, "C"),
            make_atom(-d, 0.0, 0.0, "C"),
            make_atom(0.0, d, 0.0, "C"),
            make_atom(0.0, -d, 0.0, "C"),
            make_atom(0.0, 0.0, d, "C"),
            make_atom(0.0, 0.0, -d, "C"),
        ];

        let result = sasa_calc.calculate(&atoms);

        // Central atom should be significantly buried
        assert!(
            result.atom_sasa[0] < 50.0,
            "Central atom should be mostly buried, got SASA = {:.1}",
            result.atom_sasa[0]
        );
    }

    #[test]
    fn test_two_touching_spheres() {
        let sasa_calc = ShrakeRupleySASA::default();

        // Two carbon atoms at close distance (overlapping probe spheres)
        // When atoms are 2*(vdW+probe) apart, spheres just touch - use closer distance
        let r = 1.7 + 1.4; // vdW + probe
        let atoms = vec![
            make_atom(0.0, 0.0, 0.0, "C"),
            make_atom(r * 1.5, 0.0, 0.0, "C"),  // Closer than 2r so probes overlap
        ];

        let result = sasa_calc.calculate(&atoms);

        // Both should have similar SASA (symmetric)
        let diff = (result.atom_sasa[0] - result.atom_sasa[1]).abs();
        assert!(diff < 5.0, "Symmetric atoms should have similar SASA, diff={}", diff);

        // Both should have some SASA
        assert!(result.atom_sasa[0] > 0.0, "Atom 0 should have SASA");
        assert!(result.atom_sasa[1] > 0.0, "Atom 1 should have SASA");
    }

    #[test]
    fn test_total_sasa() {
        let sasa_calc = ShrakeRupleySASA::default();

        let atoms = vec![
            make_atom(0.0, 0.0, 0.0, "C"),
            make_atom(5.0, 0.0, 0.0, "N"),
            make_atom(10.0, 0.0, 0.0, "O"),
        ];

        let result = sasa_calc.calculate(&atoms);

        // Total should equal sum of individual
        let sum: f64 = result.atom_sasa.iter().sum();
        assert!(
            (result.total_sasa - sum).abs() < 0.01,
            "Total SASA should equal sum of individual"
        );
    }

    #[test]
    fn test_sphere_points_coverage() {
        let sasa = ShrakeRupleySASA::default();
        let points = &sasa.sphere_points;

        // Check we have correct number of points
        assert_eq!(points.len(), 92);

        // All points should be on unit sphere
        for p in points {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!(
                (r - 1.0).abs() < 1e-10,
                "Point should be on unit sphere, r = {}",
                r
            );
        }

        // Check roughly uniform distribution: z values should span [-1, 1]
        let z_min = points.iter().map(|p| p[2]).fold(f64::INFINITY, f64::min);
        let z_max = points.iter().map(|p| p[2]).fold(f64::NEG_INFINITY, f64::max);

        assert!(z_min < -0.95, "Points should cover south pole");
        assert!(z_max > 0.95, "Points should cover north pole");
    }

    #[test]
    fn test_neighbor_list_performance() {
        let sasa_calc = ShrakeRupleySASA::default();

        // Create a line of atoms with spacing beyond cutoff
        // Cutoff is ~11.6 Å (2 * (3.0 + 2*1.4)), so atoms 4Å apart will be neighbors
        let atoms: Vec<Atom> = (0..100)
            .map(|i| make_atom(i as f64 * 4.0, 0.0, 0.0, "C"))
            .collect();

        let neighbors = sasa_calc.build_neighbor_list(&atoms);

        // With 4Å spacing and ~11.6Å cutoff, each atom has ~2-3 neighbors on each side
        // End atoms should have some neighbors
        assert!(!neighbors[0].is_empty(), "End atoms should have neighbors");
        assert!(neighbors[50].len() >= 1, "Middle atoms should have neighbors");
    }

    #[test]
    fn test_surface_vs_buried_classification() {
        let sasa_calc = ShrakeRupleySASA {
            surface_threshold: 10.0,
            ..Default::default()
        };

        // Isolated atom should be surface
        let atoms = vec![make_atom(0.0, 0.0, 0.0, "C")];
        let result = sasa_calc.calculate(&atoms);

        assert!(
            result.surface_atoms.contains(&0),
            "Isolated atom should be classified as surface"
        );
        assert!(
            result.buried_atoms.is_empty(),
            "No atoms should be buried"
        );
    }
}

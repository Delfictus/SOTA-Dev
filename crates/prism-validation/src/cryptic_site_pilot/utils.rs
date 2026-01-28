//! Utility functions for Cryptic Site Pilot
//!
//! Provides standalone implementations of PDB parsing and SASA calculation
//! that don't depend on feature-gated modules.

use std::f64::consts::PI;

/// Simple atom structure for PDB parsing
#[derive(Debug, Clone)]
pub struct SimpleAtom {
    /// Atom serial number
    pub serial: i32,
    /// Atom name (e.g., "CA", "N", "C")
    pub name: String,
    /// Residue name (e.g., "ALA", "GLY")
    pub residue_name: String,
    /// Chain identifier
    pub chain_id: char,
    /// Residue sequence number
    pub residue_seq: i32,
    /// 3D coordinates [x, y, z]
    pub coord: [f64; 3],
    /// B-factor (temperature factor)
    pub bfactor: f64,
    /// Element symbol
    pub element: String,
    /// Is this a HETATM record?
    pub is_hetatm: bool,
}

/// Parse a PDB file content into SimpleAtom structures
pub fn parse_pdb_simple(content: &str) -> Vec<SimpleAtom> {
    let mut atoms = Vec::new();

    for line in content.lines() {
        let record_type = if line.len() >= 6 { &line[0..6] } else { continue };

        let is_hetatm = match record_type {
            "ATOM  " => false,
            "HETATM" => true,
            _ => continue,
        };

        // Skip if line is too short
        if line.len() < 54 {
            continue;
        }

        // Parse PDB columns (fixed width format)
        let serial = line[6..11].trim().parse::<i32>().unwrap_or(0);
        let name = line[12..16].trim().to_string();
        let residue_name = line[17..20].trim().to_string();
        let chain_id = line[21..22].chars().next().unwrap_or('A');
        let residue_seq = line[22..26].trim().parse::<i32>().unwrap_or(0);

        let x = line[30..38].trim().parse::<f64>().unwrap_or(0.0);
        let y = line[38..46].trim().parse::<f64>().unwrap_or(0.0);
        let z = line[46..54].trim().parse::<f64>().unwrap_or(0.0);

        let bfactor = if line.len() >= 66 {
            line[60..66].trim().parse::<f64>().unwrap_or(0.0)
        } else {
            0.0
        };

        let element = if line.len() >= 78 {
            line[76..78].trim().to_string()
        } else {
            // Derive from atom name
            name.chars().filter(|c| c.is_alphabetic()).take(1).collect()
        };

        atoms.push(SimpleAtom {
            serial,
            name,
            residue_name,
            chain_id,
            residue_seq,
            coord: [x, y, z],
            bfactor,
            element,
            is_hetatm,
        });
    }

    atoms
}

/// Shrake-Rupley SASA calculation result
#[derive(Debug, Clone, Default)]
pub struct SASAResult {
    /// Total SASA in Å²
    pub total_sasa: f64,
    /// Per-atom SASA values
    pub per_atom_sasa: Vec<f64>,
}

/// Shrake-Rupley SASA calculator with configurable parameters
#[derive(Clone)]
pub struct ShrakeRupleySASA {
    /// Number of test points per atom
    n_points: usize,
    /// Probe radius in Å
    probe_radius: f64,
    /// Pre-computed unit sphere points
    sphere_points: Vec<[f64; 3]>,
}

impl Default for ShrakeRupleySASA {
    fn default() -> Self {
        Self::new(92, 1.4)
    }
}

impl ShrakeRupleySASA {
    /// Create a new SASA calculator
    ///
    /// # Arguments
    /// * `n_points` - Number of test points per atom (92 Fibonacci points recommended)
    /// * `probe_radius` - Solvent probe radius in Å (typically 1.4 for water)
    pub fn new(n_points: usize, probe_radius: f64) -> Self {
        let sphere_points = Self::generate_fibonacci_sphere(n_points);
        Self {
            n_points,
            probe_radius,
            sphere_points,
        }
    }

    /// Generate uniformly distributed points on a unit sphere using Fibonacci spiral
    fn generate_fibonacci_sphere(n: usize) -> Vec<[f64; 3]> {
        let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let golden_angle = 2.0 * PI / golden_ratio;

        (0..n).map(|i| {
            let theta = golden_angle * i as f64;
            let z = 1.0 - (2.0 * i as f64 + 1.0) / n as f64;
            let r = (1.0 - z * z).sqrt();
            [r * theta.cos(), r * theta.sin(), z]
        }).collect()
    }

    /// Get van der Waals radius for an element
    fn vdw_radius(element: &str) -> f64 {
        match element.trim().to_uppercase().as_str() {
            "C" => 1.70,
            "N" => 1.55,
            "O" => 1.52,
            "S" => 1.80,
            "P" => 1.80,
            "H" => 1.20,
            "FE" | "ZN" | "MG" | "CA" => 1.40,
            _ => 1.70, // Default to carbon
        }
    }

    /// Calculate SASA for a set of atoms
    pub fn calculate(&self, atoms: &[SimpleAtom]) -> SASAResult {
        if atoms.is_empty() {
            return SASAResult::default();
        }

        let n_atoms = atoms.len();
        let mut per_atom_sasa = vec![0.0f64; n_atoms];

        // Pre-compute atom radii (vdW + probe)
        let radii: Vec<f64> = atoms.iter()
            .map(|a| Self::vdw_radius(&a.element) + self.probe_radius)
            .collect();

        // Build neighbor list for efficiency
        let neighbors = self.build_neighbor_list(atoms, &radii);

        // For each atom, count exposed surface points
        for i in 0..n_atoms {
            let atom = &atoms[i];
            let radius = radii[i];
            let mut exposed = 0;

            // Test each sphere point
            'point: for point in &self.sphere_points {
                // Scale and translate point to atom surface
                let test_point = [
                    atom.coord[0] + radius * point[0],
                    atom.coord[1] + radius * point[1],
                    atom.coord[2] + radius * point[2],
                ];

                // Check if point is inside any neighbor atom
                for &j in &neighbors[i] {
                    let other = &atoms[j];
                    let other_radius = radii[j];

                    let dx = test_point[0] - other.coord[0];
                    let dy = test_point[1] - other.coord[1];
                    let dz = test_point[2] - other.coord[2];
                    let dist_sq = dx * dx + dy * dy + dz * dz;

                    if dist_sq < other_radius * other_radius {
                        // Point is buried
                        continue 'point;
                    }
                }

                // Point is exposed
                exposed += 1;
            }

            // Calculate SASA contribution
            // Surface area = 4πr² × (exposed fraction)
            let fraction = exposed as f64 / self.n_points as f64;
            per_atom_sasa[i] = 4.0 * PI * radius * radius * fraction;
        }

        let total_sasa = per_atom_sasa.iter().sum();

        SASAResult {
            total_sasa,
            per_atom_sasa,
        }
    }

    /// Build neighbor list for each atom
    fn build_neighbor_list(&self, atoms: &[SimpleAtom], radii: &[f64]) -> Vec<Vec<usize>> {
        let n = atoms.len();
        let mut neighbors = vec![Vec::new(); n];

        // Find maximum radius for cutoff
        let max_radius = radii.iter().cloned().fold(0.0f64, f64::max);
        let cutoff = 2.0 * max_radius;
        let cutoff_sq = cutoff * cutoff;

        // O(N²) but simple - could use spatial hashing for larger systems
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = atoms[i].coord[0] - atoms[j].coord[0];
                let dy = atoms[i].coord[1] - atoms[j].coord[1];
                let dz = atoms[i].coord[2] - atoms[j].coord[2];
                let dist_sq = dx * dx + dy * dy + dz * dz;

                // Check if atoms can possibly overlap
                let sum_radii = radii[i] + radii[j];
                if dist_sq < (sum_radii + cutoff) * (sum_radii + cutoff) {
                    neighbors[i].push(j);
                    neighbors[j].push(i);
                }
            }
        }

        neighbors
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pdb_simple() {
        let pdb = r#"ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00 20.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00 20.00           C
END"#;

        let atoms = parse_pdb_simple(pdb);
        assert_eq!(atoms.len(), 3);
        assert_eq!(atoms[0].name, "N");
        assert_eq!(atoms[1].name, "CA");
        assert_eq!(atoms[2].name, "C");
        assert_eq!(atoms[0].residue_name, "ALA");
        assert_eq!(atoms[0].chain_id, 'A');
    }

    #[test]
    fn test_fibonacci_sphere() {
        let points = ShrakeRupleySASA::generate_fibonacci_sphere(92);
        assert_eq!(points.len(), 92);

        // All points should be on unit sphere
        for p in &points {
            let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((r - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sasa_single_atom() {
        let sasa = ShrakeRupleySASA::default();

        let atoms = vec![SimpleAtom {
            serial: 1,
            name: "CA".to_string(),
            residue_name: "ALA".to_string(),
            chain_id: 'A',
            residue_seq: 1,
            coord: [0.0, 0.0, 0.0],
            bfactor: 20.0,
            element: "C".to_string(),
            is_hetatm: false,
        }];

        let result = sasa.calculate(&atoms);

        // Single exposed atom should have SASA ≈ 4πr² where r = vdW + probe
        // For carbon: r = 1.70 + 1.40 = 3.10 Å
        // Expected SASA ≈ 4 × π × 3.10² ≈ 120.8 Å²
        let expected = 4.0 * PI * (1.70 + 1.4) * (1.70 + 1.4);
        assert!((result.total_sasa - expected).abs() < 5.0);
    }
}

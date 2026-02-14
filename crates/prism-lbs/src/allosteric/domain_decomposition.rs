//! Stage 1A: Domain Decomposition via Spectral Clustering
//!
//! Uses graph Laplacian eigendecomposition to identify structural domains.
//! The number of near-zero eigenvalues indicates the natural domain count.
//!
//! Algorithm:
//! 1. Build residue contact matrix (Cα-Cα distances with Gaussian weights)
//! 2. Compute normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
//! 3. Eigendecomposition to find k smallest eigenvectors
//! 4. Eigengap heuristic to determine optimal k
//! 5. K-means clustering on eigenvector embedding

use crate::structure::Atom;
use super::types::*;
use std::collections::{HashMap, HashSet};

/// Domain decomposer using spectral graph clustering
pub struct DomainDecomposer {
    /// Contact distance cutoff (Å)
    pub contact_cutoff: f64,
    /// Minimum domain size (residues)
    pub min_domain_size: usize,
    /// Number of eigenvectors for clustering
    pub n_eigenvectors: usize,
    /// Gaussian sigma for contact weighting
    pub sigma: f64,
}

impl Default for DomainDecomposer {
    fn default() -> Self {
        Self {
            contact_cutoff: 10.0,
            min_domain_size: 30,
            n_eigenvectors: 5,
            sigma: 6.0,
        }
    }
}

impl DomainDecomposer {
    pub fn new(contact_cutoff: f64, min_domain_size: usize) -> Self {
        Self {
            contact_cutoff,
            min_domain_size,
            ..Default::default()
        }
    }

    /// Decompose structure into domains using spectral clustering
    pub fn decompose(&self, atoms: &[Atom]) -> Vec<Domain> {
        // Get unique residues and their Cα atoms
        let ca_atoms = self.get_ca_atoms(atoms);
        let residues: Vec<i32> = ca_atoms.keys().copied().collect();
        let n = residues.len();

        if n < self.min_domain_size {
            // Structure too small for domain decomposition
            return vec![self.create_single_domain(atoms, &residues)];
        }

        // Step 1: Build contact matrix with Gaussian weights
        let contact_matrix = self.build_contact_matrix(&ca_atoms, &residues);

        // Step 2: Compute normalized Laplacian
        let laplacian = self.compute_normalized_laplacian(&contact_matrix, n);

        // Step 3: Eigendecomposition
        let (eigenvalues, eigenvectors) = self.eigen_decomposition(&laplacian, n);

        // Step 4: Determine optimal number of domains via eigengap
        let n_domains = self.find_eigengap(&eigenvalues);

        // Step 5: K-means clustering on eigenvector embedding
        let assignments = self.spectral_kmeans(&eigenvectors, n, n_domains);

        // Step 6: Build domain objects
        self.build_domains(atoms, &residues, &assignments, n_domains)
    }

    fn get_ca_atoms(&self, atoms: &[Atom]) -> HashMap<i32, [f64; 3]> {
        atoms
            .iter()
            .filter(|a| a.name.trim() == "CA")
            .map(|a| (a.residue_seq, a.coord))
            .collect()
    }

    fn build_contact_matrix(
        &self,
        ca_atoms: &HashMap<i32, [f64; 3]>,
        residues: &[i32],
    ) -> Vec<f64> {
        let n = residues.len();
        let mut matrix = vec![0.0; n * n];
        let cutoff_sq = self.contact_cutoff * self.contact_cutoff;
        let sigma_sq_2 = 2.0 * self.sigma * self.sigma;

        for (i, &res_i) in residues.iter().enumerate() {
            if let Some(&coord_i) = ca_atoms.get(&res_i) {
                for (j, &res_j) in residues.iter().enumerate().skip(i + 1) {
                    if let Some(&coord_j) = ca_atoms.get(&res_j) {
                        let dist_sq = (coord_i[0] - coord_j[0]).powi(2)
                            + (coord_i[1] - coord_j[1]).powi(2)
                            + (coord_i[2] - coord_j[2]).powi(2);

                        if dist_sq < cutoff_sq {
                            // Gaussian contact strength
                            let strength = (-dist_sq / sigma_sq_2).exp();
                            matrix[i * n + j] = strength;
                            matrix[j * n + i] = strength;
                        }
                    }
                }
            }
        }

        matrix
    }

    fn compute_normalized_laplacian(&self, adjacency: &[f64], n: usize) -> Vec<f64> {
        // Compute degree for each node
        let mut degrees = vec![0.0; n];
        for i in 0..n {
            degrees[i] = (0..n).map(|j| adjacency[i * n + j]).sum();
        }

        // Compute D^(-1/2)
        let inv_sqrt_deg: Vec<f64> = degrees
            .iter()
            .map(|&d| if d > 1e-10 { 1.0 / d.sqrt() } else { 0.0 })
            .collect();

        // Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
        let mut laplacian = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    laplacian[i * n + j] = 1.0;
                } else if adjacency[i * n + j] > 0.0 {
                    laplacian[i * n + j] =
                        -adjacency[i * n + j] * inv_sqrt_deg[i] * inv_sqrt_deg[j];
                }
            }
        }

        laplacian
    }

    /// Power iteration-based eigendecomposition for smallest eigenvalues
    fn eigen_decomposition(&self, laplacian: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
        let k = self.n_eigenvectors.min(n - 1).max(2);
        let mut eigenvalues = vec![0.0; k];
        let mut eigenvectors = vec![0.0; n * k];

        // We need smallest eigenvalues, so shift: B = I - L
        // Then largest eigenvalues of B correspond to smallest of L
        let mut shifted = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                shifted[i * n + j] = if i == j {
                    1.0 - laplacian[i * n + j]
                } else {
                    -laplacian[i * n + j]
                };
            }
        }

        // Power iteration with deflation
        let mut deflated = shifted.clone();

        for ev_idx in 0..k {
            // Initialize random vector
            let mut v: Vec<f64> = (0..n).map(|i| ((i * 7 + 13) % 100) as f64 / 100.0).collect();
            normalize(&mut v);

            // Power iteration
            for _ in 0..100 {
                let mut v_new = vec![0.0; n];

                // Matrix-vector multiplication
                for i in 0..n {
                    for j in 0..n {
                        v_new[i] += deflated[i * n + j] * v[j];
                    }
                }

                // Rayleigh quotient (eigenvalue estimate)
                let lambda: f64 = v.iter().zip(v_new.iter()).map(|(a, b)| a * b).sum();

                normalize(&mut v_new);

                // Check convergence
                let diff: f64 = v.iter().zip(v_new.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                v = v_new;

                if diff < 1e-10 {
                    break;
                }
            }

            // Store eigenvalue (convert back from shifted)
            let lambda: f64 = (0..n)
                .map(|i| {
                    let av: f64 = (0..n).map(|j| deflated[i * n + j] * v[j]).sum();
                    v[i] * av
                })
                .sum();
            eigenvalues[ev_idx] = 1.0 - lambda;

            // Store eigenvector
            for (i, &val) in v.iter().enumerate() {
                eigenvectors[i * k + ev_idx] = val;
            }

            // Deflate matrix: B' = B - lambda * v * v^T
            for i in 0..n {
                for j in 0..n {
                    deflated[i * n + j] -= lambda * v[i] * v[j];
                }
            }
        }

        // Sort by eigenvalue (smallest first)
        let mut indices: Vec<usize> = (0..k).collect();
        indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

        let sorted_eigenvalues: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
        let mut sorted_eigenvectors = vec![0.0; n * k];

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            for i in 0..n {
                sorted_eigenvectors[i * k + new_idx] = eigenvectors[i * k + old_idx];
            }
        }

        (sorted_eigenvalues, sorted_eigenvectors)
    }

    /// Find optimal number of clusters using eigengap heuristic
    fn find_eigengap(&self, eigenvalues: &[f64]) -> usize {
        if eigenvalues.len() < 2 {
            return 1;
        }

        let mut max_gap = 0.0;
        let mut n_clusters = 2;

        // Look for largest gap (indicates natural cluster boundary)
        for i in 1..eigenvalues.len().min(8) {
            let gap = eigenvalues[i] - eigenvalues[i - 1];
            if gap > max_gap {
                max_gap = gap;
                n_clusters = i;
            }
        }

        // Reasonable bounds: 2-8 domains
        n_clusters.max(2).min(8)
    }

    /// K-means clustering on eigenvector embedding
    fn spectral_kmeans(&self, eigenvectors: &[f64], n: usize, k: usize) -> Vec<usize> {
        let n_features = self.n_eigenvectors.min(k);
        let mut assignments = vec![0; n];
        let mut centroids = vec![vec![0.0; n_features]; k];

        // Initialize centroids using k-means++
        self.kmeans_pp_init(eigenvectors, n, n_features, k, &mut centroids);

        // K-means iterations
        for _ in 0..50 {
            // Assignment step
            let mut changed = false;
            for i in 0..n {
                let point: Vec<f64> = (0..n_features)
                    .map(|f| eigenvectors[i * self.n_eigenvectors.min(n - 1).max(2) + f])
                    .collect();

                let mut best_cluster = 0;
                let mut best_dist = f64::INFINITY;

                for (c, centroid) in centroids.iter().enumerate() {
                    let dist: f64 = point
                        .iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();

                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }

                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update step
            let mut counts = vec![0; k];
            for centroid in &mut centroids {
                centroid.fill(0.0);
            }

            for i in 0..n {
                let c = assignments[i];
                counts[c] += 1;
                for f in 0..n_features {
                    centroids[c][f] +=
                        eigenvectors[i * self.n_eigenvectors.min(n - 1).max(2) + f];
                }
            }

            for (c, centroid) in centroids.iter_mut().enumerate() {
                if counts[c] > 0 {
                    for val in centroid.iter_mut() {
                        *val /= counts[c] as f64;
                    }
                }
            }
        }

        assignments
    }

    fn kmeans_pp_init(
        &self,
        eigenvectors: &[f64],
        n: usize,
        n_features: usize,
        k: usize,
        centroids: &mut [Vec<f64>],
    ) {
        let ev_cols = self.n_eigenvectors.min(n - 1).max(2);

        // First centroid: random point
        let first_idx = 0;
        for f in 0..n_features {
            centroids[0][f] = eigenvectors[first_idx * ev_cols + f];
        }

        // Remaining centroids: weighted by distance to existing centroids
        for c in 1..k {
            let mut best_idx = 0;
            let mut best_min_dist = 0.0;

            for i in 0..n {
                let point: Vec<f64> = (0..n_features)
                    .map(|f| eigenvectors[i * ev_cols + f])
                    .collect();

                // Distance to nearest existing centroid
                let min_dist: f64 = (0..c)
                    .map(|cc| {
                        point
                            .iter()
                            .zip(centroids[cc].iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>()
                    })
                    .fold(f64::INFINITY, f64::min);

                if min_dist > best_min_dist {
                    best_min_dist = min_dist;
                    best_idx = i;
                }
            }

            for f in 0..n_features {
                centroids[c][f] = eigenvectors[best_idx * ev_cols + f];
            }
        }
    }

    fn build_domains(
        &self,
        atoms: &[Atom],
        residues: &[i32],
        assignments: &[usize],
        n_domains: usize,
    ) -> Vec<Domain> {
        let mut domains = Vec::new();

        for domain_id in 0..n_domains {
            let domain_residues: Vec<i32> = residues
                .iter()
                .enumerate()
                .filter(|(i, _)| assignments[*i] == domain_id)
                .map(|(_, &r)| r)
                .collect();

            if domain_residues.len() < self.min_domain_size {
                continue;
            }

            // Calculate domain properties
            let centroid = calculate_residue_centroid(atoms, &domain_residues);
            let radius_of_gyration = calculate_radius_of_gyration(atoms, &domain_residues, &centroid);
            let internal_contacts = self.count_internal_contacts(atoms, &domain_residues);
            let ss_composition = self.calculate_ss_composition(atoms, &domain_residues);

            domains.push(Domain {
                id: domains.len(),
                residues: domain_residues,
                centroid,
                radius_of_gyration,
                internal_contacts,
                ss_composition,
            });
        }

        // If no valid domains, create single domain
        if domains.is_empty() {
            domains.push(self.create_single_domain(atoms, residues));
        }

        domains
    }

    fn create_single_domain(&self, atoms: &[Atom], residues: &[i32]) -> Domain {
        let centroid = calculate_residue_centroid(atoms, residues);
        let radius_of_gyration = calculate_radius_of_gyration(atoms, residues, &centroid);

        Domain {
            id: 0,
            residues: residues.to_vec(),
            centroid,
            radius_of_gyration,
            internal_contacts: 0,
            ss_composition: SecondaryStructureComposition::default(),
        }
    }

    fn count_internal_contacts(&self, atoms: &[Atom], residues: &[i32]) -> usize {
        let residue_set: HashSet<i32> = residues.iter().copied().collect();
        let cutoff_sq = self.contact_cutoff * self.contact_cutoff;
        let mut contacts = 0;

        let ca_atoms: Vec<_> = atoms
            .iter()
            .filter(|a| a.name.trim() == "CA" && residue_set.contains(&a.residue_seq))
            .collect();

        for (i, ca_i) in ca_atoms.iter().enumerate() {
            for ca_j in ca_atoms.iter().skip(i + 1) {
                let dist_sq = (ca_i.coord[0] - ca_j.coord[0]).powi(2)
                    + (ca_i.coord[1] - ca_j.coord[1]).powi(2)
                    + (ca_i.coord[2] - ca_j.coord[2]).powi(2);

                if dist_sq < cutoff_sq && (ca_i.residue_seq - ca_j.residue_seq).abs() > 3 {
                    contacts += 1;
                }
            }
        }

        contacts
    }

    fn calculate_ss_composition(&self, atoms: &[Atom], residues: &[i32]) -> SecondaryStructureComposition {
        // Simplified: would integrate with actual DSSP in production
        let _residue_set: HashSet<i32> = residues.iter().copied().collect();

        // For now return default - full implementation would use secondary structure assignment
        SecondaryStructureComposition {
            helix_fraction: 0.35,
            strand_fraction: 0.25,
            coil_fraction: 0.40,
        }
    }

    /// Find interfaces between domains
    pub fn find_domain_interfaces(
        &self,
        atoms: &[Atom],
        domains: &[Domain],
    ) -> Vec<DomainInterface> {
        let mut interfaces = Vec::new();
        let interface_cutoff = 8.0;

        for (i, domain_a) in domains.iter().enumerate() {
            for domain_b in domains.iter().skip(i + 1) {
                let interface_a = self.find_interface_residues(
                    atoms,
                    &domain_a.residues,
                    &domain_b.residues,
                    interface_cutoff,
                );

                let interface_b = self.find_interface_residues(
                    atoms,
                    &domain_b.residues,
                    &domain_a.residues,
                    interface_cutoff,
                );

                if interface_a.len() >= 3 && interface_b.len() >= 3 {
                    let mut all_interface: Vec<i32> = interface_a
                        .iter()
                        .chain(interface_b.iter())
                        .copied()
                        .collect();
                    all_interface.sort();
                    all_interface.dedup();

                    let centroid = calculate_residue_centroid(atoms, &all_interface);
                    let buried_sasa = self.estimate_interface_burial(atoms, &all_interface);
                    let shape_comp = self.estimate_shape_complementarity(&interface_a, &interface_b);

                    interfaces.push(DomainInterface {
                        domain_a_id: domain_a.id,
                        domain_b_id: domain_b.id,
                        residues: all_interface,
                        centroid,
                        buried_sasa,
                        shape_complementarity: shape_comp,
                        hydrogen_bonds: 0, // Would calculate from structure
                        salt_bridges: 0,
                    });
                }
            }
        }

        interfaces
    }

    fn find_interface_residues(
        &self,
        atoms: &[Atom],
        domain_residues: &[i32],
        other_residues: &[i32],
        cutoff: f64,
    ) -> Vec<i32> {
        let domain_set: HashSet<i32> = domain_residues.iter().copied().collect();
        let other_set: HashSet<i32> = other_residues.iter().copied().collect();
        let cutoff_sq = cutoff * cutoff;

        let mut interface = HashSet::new();

        for atom_a in atoms.iter().filter(|a| domain_set.contains(&a.residue_seq)) {
            for atom_b in atoms.iter().filter(|a| other_set.contains(&a.residue_seq)) {
                let dist_sq = (atom_a.coord[0] - atom_b.coord[0]).powi(2)
                    + (atom_a.coord[1] - atom_b.coord[1]).powi(2)
                    + (atom_a.coord[2] - atom_b.coord[2]).powi(2);

                if dist_sq < cutoff_sq {
                    interface.insert(atom_a.residue_seq);
                    break;
                }
            }
        }

        interface.into_iter().collect()
    }

    fn estimate_interface_burial(&self, atoms: &[Atom], interface_residues: &[i32]) -> f64 {
        // Simplified: estimate based on residue count and average SASA contribution
        let avg_sasa_per_residue = 80.0; // Approximate Å² per residue
        interface_residues.len() as f64 * avg_sasa_per_residue * 0.5
    }

    fn estimate_shape_complementarity(&self, _interface_a: &[i32], _interface_b: &[i32]) -> f64 {
        // Simplified: would calculate actual shape complementarity in production
        0.65 // Default moderate complementarity
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn normalize(v: &mut [f64]) {
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

pub fn calculate_residue_centroid(atoms: &[Atom], residues: &[i32]) -> [f64; 3] {
    let residue_set: HashSet<i32> = residues.iter().copied().collect();

    let ca_atoms: Vec<_> = atoms
        .iter()
        .filter(|a| a.name.trim() == "CA" && residue_set.contains(&a.residue_seq))
        .collect();

    if ca_atoms.is_empty() {
        return [0.0, 0.0, 0.0];
    }

    let mut centroid = [0.0, 0.0, 0.0];
    for ca in &ca_atoms {
        centroid[0] += ca.coord[0];
        centroid[1] += ca.coord[1];
        centroid[2] += ca.coord[2];
    }

    let n = ca_atoms.len() as f64;
    [centroid[0] / n, centroid[1] / n, centroid[2] / n]
}

fn calculate_radius_of_gyration(atoms: &[Atom], residues: &[i32], centroid: &[f64; 3]) -> f64 {
    let residue_set: HashSet<i32> = residues.iter().copied().collect();

    let ca_atoms: Vec<_> = atoms
        .iter()
        .filter(|a| a.name.trim() == "CA" && residue_set.contains(&a.residue_seq))
        .collect();

    if ca_atoms.is_empty() {
        return 0.0;
    }

    let sum_sq: f64 = ca_atoms
        .iter()
        .map(|ca| {
            (ca.coord[0] - centroid[0]).powi(2)
                + (ca.coord[1] - centroid[1]).powi(2)
                + (ca.coord[2] - centroid[2]).powi(2)
        })
        .sum();

    (sum_sq / ca_atoms.len() as f64).sqrt()
}

pub fn euclidean_distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

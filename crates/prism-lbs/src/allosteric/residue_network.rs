//! Stage 3A: Residue Interaction Network
//!
//! Builds a weighted graph of residue-residue interactions to model
//! allosteric communication pathways. Uses Floyd-Warshall algorithm
//! for all-pairs shortest paths (GPU-accelerated when available).
//!
//! Edge weights represent interaction strength (Gaussian decay or inverse distance).
//! Paths through the network model signal propagation during allosteric transitions.

use crate::structure::Atom;
use super::gpu_apsp::GpuFloydWarshall;
use super::types::*;
use std::collections::{HashMap, HashSet};

/// Residue network analyzer for allosteric coupling
pub struct ResidueNetworkAnalyzer {
    /// Contact cutoff for network edges (Å)
    pub contact_cutoff: f64,
    /// Edge weight scheme
    pub weight_scheme: EdgeWeightScheme,
    /// Use Cβ atoms (Cα for Gly) instead of Cα only
    pub use_cb_atoms: bool,
    /// GPU-accelerated Floyd-Warshall solver (optional)
    pub gpu_solver: Option<GpuFloydWarshall>,
}

impl Default for ResidueNetworkAnalyzer {
    fn default() -> Self {
        Self {
            contact_cutoff: 10.0,
            weight_scheme: EdgeWeightScheme::DistanceBased { sigma: 6.0 },
            use_cb_atoms: true,
            gpu_solver: Some(GpuFloydWarshall::new()),
        }
    }
}

impl ResidueNetworkAnalyzer {
    pub fn new(contact_cutoff: f64, weight_scheme: EdgeWeightScheme) -> Self {
        Self {
            contact_cutoff,
            weight_scheme,
            use_cb_atoms: true,
            gpu_solver: Some(GpuFloydWarshall::new()),
        }
    }

    /// Create analyzer with GPU acceleration
    #[cfg(feature = "cuda")]
    pub fn with_gpu(mut self) -> Self {
        let mut solver = GpuFloydWarshall::new();
        if let Err(e) = solver.init_cuda() {
            log::warn!("GPU Floyd-Warshall initialization failed: {}", e);
        }
        self.gpu_solver = Some(solver);
        self
    }

    /// Disable GPU acceleration (use CPU-only)
    pub fn cpu_only(mut self) -> Self {
        self.gpu_solver = None;
        self
    }

    /// Build residue interaction network from structure
    pub fn build_network(&self, atoms: &[Atom]) -> ResidueNetwork {
        // Get representative coordinates (Cβ or Cα)
        let coords = self.get_representative_coordinates(atoms);
        let residues: Vec<i32> = coords.keys().copied().collect();
        let n = residues.len();

        let mut network = ResidueNetwork::new(n);

        // Build mappings
        for (i, &res) in residues.iter().enumerate() {
            network.residue_to_idx.insert(res, i);
            network.idx_to_residue.push(res);
        }

        // Build adjacency matrix
        let cutoff_sq = self.contact_cutoff * self.contact_cutoff;

        for (i, &res_i) in residues.iter().enumerate() {
            if let Some(coord_i) = coords.get(&res_i) {
                for (j, &res_j) in residues.iter().enumerate().skip(i + 1) {
                    if let Some(coord_j) = coords.get(&res_j) {
                        let dist_sq = (coord_i[0] - coord_j[0]).powi(2)
                            + (coord_i[1] - coord_j[1]).powi(2)
                            + (coord_i[2] - coord_j[2]).powi(2);

                        if dist_sq < cutoff_sq {
                            let dist = dist_sq.sqrt();
                            let weight = self.calculate_edge_weight(dist);

                            network.set(i, j, weight);
                            network.set(j, i, weight);
                        }
                    }
                }
            }
        }

        network
    }

    fn get_representative_coordinates(&self, atoms: &[Atom]) -> HashMap<i32, [f64; 3]> {
        let mut coords: HashMap<i32, [f64; 3]> = HashMap::new();
        let mut seen_cb: HashSet<i32> = HashSet::new();

        if self.use_cb_atoms {
            // First pass: collect Cβ atoms
            for atom in atoms {
                let name = atom.name.trim();
                if name == "CB" {
                    coords.insert(atom.residue_seq, atom.coord);
                    seen_cb.insert(atom.residue_seq);
                }
            }

            // Second pass: use Cα for glycine (no Cβ)
            for atom in atoms {
                let name = atom.name.trim();
                if name == "CA" && !seen_cb.contains(&atom.residue_seq) {
                    coords.insert(atom.residue_seq, atom.coord);
                }
            }
        } else {
            // Just use Cα
            for atom in atoms {
                if atom.name.trim() == "CA" {
                    coords.insert(atom.residue_seq, atom.coord);
                }
            }
        }

        coords
    }

    fn calculate_edge_weight(&self, distance: f64) -> f64 {
        match self.weight_scheme {
            EdgeWeightScheme::Binary => 1.0,
            EdgeWeightScheme::DistanceBased { sigma } => {
                (-distance.powi(2) / (2.0 * sigma.powi(2))).exp()
            }
            EdgeWeightScheme::InverseDistance => {
                if distance > 0.1 {
                    1.0 / distance
                } else {
                    10.0
                }
            }
        }
    }

    /// Calculate shortest paths using Floyd-Warshall algorithm
    /// Returns (distance matrix, predecessor matrix for path reconstruction)
    /// Uses GPU acceleration when available, falls back to CPU
    pub fn floyd_warshall(&self, network: &ResidueNetwork) -> (Vec<f64>, Vec<Option<usize>>) {
        let n = network.size;

        // Predecessor matrix for path reconstruction (always built on CPU)
        let mut next: Vec<Option<usize>> = vec![None; n * n];

        // Initialize next pointers from adjacency
        for i in 0..n {
            for j in 0..n {
                let weight = network.get(i, j);
                if weight > 0.0 && i != j {
                    next[i * n + j] = Some(j);
                }
            }
        }

        // Try GPU-accelerated computation
        if let Some(ref gpu_solver) = self.gpu_solver {
            // Convert to f32 adjacency matrix for GPU
            let mut adjacency_f32 = vec![0.0f32; n * n];
            for i in 0..n {
                for j in 0..n {
                    adjacency_f32[i * n + j] = network.get(i, j) as f32;
                }
            }

            log::debug!("[Floyd-Warshall] Using GPU solver for {} x {} matrix", n, n);
            let dist_f32 = gpu_solver.compute(&adjacency_f32, n);

            // Convert distances back to f64
            let dist: Vec<f64> = dist_f32.into_iter().map(|d| d as f64).collect();

            // Update next matrix from computed distances
            // (GPU gives us distances but not predecessors, so we recompute next)
            self.update_predecessors_from_distances(n, &dist, &mut next, network);

            return (dist, next);
        }

        // CPU fallback
        log::debug!("[Floyd-Warshall] Using CPU solver for {} x {} matrix", n, n);
        self.cpu_floyd_warshall(network)
    }

    /// CPU implementation of Floyd-Warshall
    fn cpu_floyd_warshall(&self, network: &ResidueNetwork) -> (Vec<f64>, Vec<Option<usize>>) {
        let n = network.size;

        // Distance matrix (initialized to infinity for no edge)
        let mut dist = vec![f64::INFINITY; n * n];

        // Predecessor matrix for path reconstruction
        let mut next: Vec<Option<usize>> = vec![None; n * n];

        // Initialize from adjacency
        for i in 0..n {
            dist[i * n + i] = 0.0;

            for j in 0..n {
                let weight = network.get(i, j);
                if weight > 0.0 && i != j {
                    // Convert weight to distance (stronger contact = shorter path)
                    dist[i * n + j] = 1.0 / weight;
                    next[i * n + j] = Some(j);
                }
            }
        }

        // Floyd-Warshall iterations
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let through_k = dist[i * n + k] + dist[k * n + j];
                    if through_k < dist[i * n + j] {
                        dist[i * n + j] = through_k;
                        next[i * n + j] = next[i * n + k];
                    }
                }
            }
        }

        (dist, next)
    }

    /// Update predecessor matrix from computed distances
    fn update_predecessors_from_distances(
        &self,
        n: usize,
        dist: &[f64],
        next: &mut [Option<usize>],
        network: &ResidueNetwork,
    ) {
        // For each pair (i,j), find the correct predecessor
        // This is a reconstruction step needed after GPU computation
        for i in 0..n {
            for j in 0..n {
                if i == j || dist[i * n + j].is_infinite() {
                    next[i * n + j] = None;
                    continue;
                }

                // Check all neighbors of i to find the one on shortest path
                for k in 0..n {
                    let weight = network.get(i, k);
                    if weight > 0.0 && k != i {
                        // If going through k gives us the shortest path
                        let path_via_k = dist[i * n + k] + dist[k * n + j];
                        if (path_via_k - dist[i * n + j]).abs() < 1e-6 {
                            // k is on the shortest path from i to j
                            next[i * n + j] = Some(k);
                            break;
                        }
                    }
                }

                // If no predecessor found through neighbors, check direct edge
                if next[i * n + j].is_none() && network.get(i, j) > 0.0 {
                    next[i * n + j] = Some(j);
                }
            }
        }
    }

    /// Reconstruct path from predecessor matrix
    pub fn reconstruct_path(
        &self,
        next: &[Option<usize>],
        n: usize,
        from: usize,
        to: usize,
        idx_to_residue: &[i32],
    ) -> Vec<i32> {
        if next[from * n + to].is_none() {
            return Vec::new();
        }

        let mut path = vec![idx_to_residue[from]];
        let mut current = from;

        while current != to {
            match next[current * n + to] {
                Some(next_node) => {
                    path.push(idx_to_residue[next_node]);
                    current = next_node;
                }
                None => break,
            }
        }

        path
    }

    /// Calculate allosteric coupling between two site regions
    pub fn calculate_allosteric_coupling(
        &self,
        network: &ResidueNetwork,
        allosteric_residues: &[i32],
        active_site_residues: &[i32],
    ) -> AllostericCoupling {
        // Run Floyd-Warshall
        let (distances, next) = self.floyd_warshall(network);
        let n = network.size;

        // Convert residues to indices
        let allo_indices: Vec<usize> = allosteric_residues
            .iter()
            .filter_map(|r| network.residue_to_idx.get(r).copied())
            .collect();

        let active_indices: Vec<usize> = active_site_residues
            .iter()
            .filter_map(|r| network.residue_to_idx.get(r).copied())
            .collect();

        if allo_indices.is_empty() || active_indices.is_empty() {
            return AllostericCoupling {
                coupling_strength: 0.0,
                shortest_path_length: f64::INFINITY,
                path_residues: Vec::new(),
                allosteric_residues: allosteric_residues.to_vec(),
                active_site_residues: active_site_residues.to_vec(),
                signal_attenuation: 1.0,
            };
        }

        // Find shortest path and average coupling
        let mut total_coupling = 0.0;
        let mut path_count = 0;
        let mut min_path_length = f64::INFINITY;
        let mut best_path: Vec<i32> = Vec::new();
        let mut best_from = 0;
        let mut best_to = 0;

        for &allo_idx in &allo_indices {
            for &active_idx in &active_indices {
                let path_length = distances[allo_idx * n + active_idx];

                if path_length < f64::INFINITY {
                    // Coupling strength decreases exponentially with path length
                    let coupling = (-path_length / 5.0).exp();
                    total_coupling += coupling;
                    path_count += 1;

                    if path_length < min_path_length {
                        min_path_length = path_length;
                        best_from = allo_idx;
                        best_to = active_idx;
                    }
                }
            }
        }

        // Reconstruct best path
        if min_path_length < f64::INFINITY {
            best_path = self.reconstruct_path(&next, n, best_from, best_to, &network.idx_to_residue);
        }

        let avg_coupling = if path_count > 0 {
            total_coupling / path_count as f64
        } else {
            0.0
        };

        // Signal attenuation estimate (per step)
        let n_steps = best_path.len().saturating_sub(1);
        let signal_attenuation = 0.9_f64.powi(n_steps as i32);

        AllostericCoupling {
            coupling_strength: avg_coupling,
            shortest_path_length: if min_path_length.is_finite() {
                min_path_length
            } else {
                0.0
            },
            path_residues: best_path,
            allosteric_residues: allosteric_residues.to_vec(),
            active_site_residues: active_site_residues.to_vec(),
            signal_attenuation,
        }
    }

    /// Find all communication pathways between sites
    pub fn find_communication_pathways(
        &self,
        network: &ResidueNetwork,
        source_residues: &[i32],
        target_residues: &[i32],
        max_paths: usize,
    ) -> Vec<CommunicationPathway> {
        let (distances, next) = self.floyd_warshall(network);
        let n = network.size;

        let mut pathways: Vec<(f64, i32, i32)> = Vec::new();

        for &source in source_residues {
            if let Some(&src_idx) = network.residue_to_idx.get(&source) {
                for &target in target_residues {
                    if let Some(&tgt_idx) = network.residue_to_idx.get(&target) {
                        let length = distances[src_idx * n + tgt_idx];
                        if length < f64::INFINITY {
                            pathways.push((length, source, target));
                        }
                    }
                }
            }
        }

        // Sort by path length
        pathways.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        pathways
            .into_iter()
            .take(max_paths)
            .map(|(length, source, target)| {
                let src_idx = network.residue_to_idx[&source];
                let tgt_idx = network.residue_to_idx[&target];

                let path = self.reconstruct_path(&next, n, src_idx, tgt_idx, &network.idx_to_residue);

                CommunicationPathway {
                    source,
                    target,
                    path,
                    length,
                    bottleneck: None, // Will be filled by centrality analysis
                }
            })
            .collect()
    }

    /// Calculate network properties for a set of residues
    pub fn calculate_subgraph_properties(
        &self,
        network: &ResidueNetwork,
        residues: &[i32],
    ) -> SubgraphProperties {
        let indices: Vec<usize> = residues
            .iter()
            .filter_map(|r| network.residue_to_idx.get(r).copied())
            .collect();

        if indices.is_empty() {
            return SubgraphProperties::default();
        }

        // Count internal edges
        let mut internal_edges = 0;
        let mut total_weight = 0.0;

        for (i, &idx_i) in indices.iter().enumerate() {
            for &idx_j in indices.iter().skip(i + 1) {
                let weight = network.get(idx_i, idx_j);
                if weight > 0.0 {
                    internal_edges += 1;
                    total_weight += weight;
                }
            }
        }

        // Calculate density
        let max_edges = indices.len() * (indices.len() - 1) / 2;
        let density = if max_edges > 0 {
            internal_edges as f64 / max_edges as f64
        } else {
            0.0
        };

        // Count external connections
        let index_set: HashSet<usize> = indices.iter().copied().collect();
        let mut external_connections = 0;

        for &idx in &indices {
            for j in 0..network.size {
                if !index_set.contains(&j) && network.get(idx, j) > 0.0 {
                    external_connections += 1;
                }
            }
        }

        SubgraphProperties {
            n_residues: indices.len(),
            internal_edges,
            external_connections,
            density,
            mean_edge_weight: if internal_edges > 0 {
                total_weight / internal_edges as f64
            } else {
                0.0
            },
        }
    }
}

/// Properties of a subgraph (pocket region)
#[derive(Debug, Clone, Default)]
pub struct SubgraphProperties {
    pub n_residues: usize,
    pub internal_edges: usize,
    pub external_connections: usize,
    pub density: f64,
    pub mean_edge_weight: f64,
}

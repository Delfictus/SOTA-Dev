//! Stage 3B: Allosteric Coupling Analysis
//!
//! Analyzes allosteric communication using network centrality measures:
//! - Betweenness centrality (Brandes algorithm) - identifies communication hubs
//! - Closeness centrality - measures how quickly info spreads
//! - Eigenvector centrality - identifies influential residues
//!
//! High-centrality residues are critical for allosteric signal transmission.

use super::types::*;
use std::collections::{HashMap, VecDeque};

/// Allosteric coupling analyzer using network centrality
pub struct AllostericCouplingAnalyzer {
    /// Normalize centrality scores
    pub normalize: bool,
    /// Minimum centrality to report
    pub min_centrality: f64,
}

impl Default for AllostericCouplingAnalyzer {
    fn default() -> Self {
        Self {
            normalize: true,
            min_centrality: 0.1,
        }
    }
}

impl AllostericCouplingAnalyzer {
    pub fn new(normalize: bool) -> Self {
        Self {
            normalize,
            ..Default::default()
        }
    }

    /// Calculate betweenness centrality for all residues (Brandes algorithm)
    ///
    /// Betweenness centrality measures how often a node lies on shortest paths
    /// between other nodes. High betweenness = critical communication hub.
    pub fn calculate_betweenness_centrality(
        &self,
        network: &ResidueNetwork,
    ) -> HashMap<i32, f64> {
        let n = network.size;
        let mut centrality = vec![0.0; n];

        // Brandes algorithm
        for s in 0..n {
            // BFS from source s
            let mut stack: Vec<usize> = Vec::new();
            let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
            let mut sigma = vec![0.0; n]; // Number of shortest paths
            let mut dist = vec![-1i32; n]; // BFS distance

            sigma[s] = 1.0;
            dist[s] = 0;

            let mut queue = VecDeque::new();
            queue.push_back(s);

            while let Some(v) = queue.pop_front() {
                stack.push(v);

                // Check all neighbors
                for w in 0..n {
                    if network.get(v, w) > 0.0 {
                        // w found for first time?
                        if dist[w] < 0 {
                            dist[w] = dist[v] + 1;
                            queue.push_back(w);
                        }

                        // Shortest path to w via v?
                        if dist[w] == dist[v] + 1 {
                            sigma[w] += sigma[v];
                            pred[w].push(v);
                        }
                    }
                }
            }

            // Accumulation
            let mut delta = vec![0.0; n];

            while let Some(w) = stack.pop() {
                for &v in &pred[w] {
                    let contrib = (sigma[v] / sigma[w]) * (1.0 + delta[w]);
                    delta[v] += contrib;
                }

                if w != s {
                    centrality[w] += delta[w];
                }
            }
        }

        // Normalize
        if self.normalize && n > 2 {
            let norm_factor = 2.0 / ((n - 1) * (n - 2)) as f64;
            for c in &mut centrality {
                *c *= norm_factor;
            }
        }

        // Map to residue numbers
        network
            .idx_to_residue
            .iter()
            .enumerate()
            .map(|(i, &res)| (res, centrality[i]))
            .collect()
    }

    /// Calculate closeness centrality for all residues
    ///
    /// Closeness = 1 / (sum of shortest paths to all other nodes)
    /// High closeness = information spreads quickly from this node
    pub fn calculate_closeness_centrality(
        &self,
        network: &ResidueNetwork,
    ) -> HashMap<i32, f64> {
        let n = network.size;
        let mut centrality = vec![0.0; n];

        for s in 0..n {
            // BFS from source
            let mut dist = vec![f64::INFINITY; n];
            dist[s] = 0.0;

            let mut queue = VecDeque::new();
            queue.push_back(s);

            while let Some(v) = queue.pop_front() {
                for w in 0..n {
                    if network.get(v, w) > 0.0 && dist[w] == f64::INFINITY {
                        dist[w] = dist[v] + 1.0;
                        queue.push_back(w);
                    }
                }
            }

            // Sum of finite distances
            let total_dist: f64 = dist.iter().filter(|&&d| d.is_finite() && d > 0.0).sum();

            if total_dist > 0.0 {
                // Count reachable nodes
                let reachable = dist.iter().filter(|&&d| d.is_finite()).count();
                centrality[s] = (reachable - 1) as f64 / total_dist;
            }
        }

        // Normalize
        if self.normalize && n > 1 {
            let max_closeness = centrality.iter().cloned().fold(0.0, f64::max);
            if max_closeness > 0.0 {
                for c in &mut centrality {
                    *c /= max_closeness;
                }
            }
        }

        network
            .idx_to_residue
            .iter()
            .enumerate()
            .map(|(i, &res)| (res, centrality[i]))
            .collect()
    }

    /// Calculate degree centrality for all residues
    ///
    /// Degree = number of edges / (n-1)
    /// Simple measure of local connectivity
    pub fn calculate_degree_centrality(
        &self,
        network: &ResidueNetwork,
    ) -> HashMap<i32, f64> {
        let n = network.size;
        let mut centrality = vec![0.0; n];

        for i in 0..n {
            let degree: f64 = (0..n)
                .filter(|&j| network.get(i, j) > 0.0)
                .count() as f64;

            centrality[i] = if n > 1 {
                degree / (n - 1) as f64
            } else {
                0.0
            };
        }

        network
            .idx_to_residue
            .iter()
            .enumerate()
            .map(|(i, &res)| (res, centrality[i]))
            .collect()
    }

    /// Calculate eigenvector centrality (power iteration)
    ///
    /// Nodes connected to high-centrality nodes get higher scores.
    /// Identifies influential network hubs.
    pub fn calculate_eigenvector_centrality(
        &self,
        network: &ResidueNetwork,
    ) -> HashMap<i32, f64> {
        let n = network.size;
        if n == 0 {
            return HashMap::new();
        }

        // Initialize uniformly
        let mut centrality: Vec<f64> = vec![1.0 / n as f64; n];

        // Power iteration
        for _ in 0..100 {
            let mut new_centrality = vec![0.0; n];

            // Matrix-vector multiplication: c' = A * c
            for i in 0..n {
                for j in 0..n {
                    new_centrality[i] += network.get(i, j) * centrality[j];
                }
            }

            // Normalize
            let norm: f64 = new_centrality.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                for c in &mut new_centrality {
                    *c /= norm;
                }
            }

            // Check convergence
            let diff: f64 = centrality
                .iter()
                .zip(new_centrality.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            centrality = new_centrality;

            if diff < 1e-8 {
                break;
            }
        }

        // Normalize to [0, 1]
        if self.normalize {
            let max_val = centrality.iter().cloned().fold(0.0, f64::max);
            if max_val > 0.0 {
                for c in &mut centrality {
                    *c /= max_val;
                }
            }
        }

        network
            .idx_to_residue
            .iter()
            .enumerate()
            .map(|(i, &res)| (res, centrality[i]))
            .collect()
    }

    /// Find allosteric hotspots (high centrality residues)
    pub fn find_allosteric_hotspots(
        &self,
        network: &ResidueNetwork,
        threshold: f64,
    ) -> Vec<AllostericHotspot> {
        let betweenness = self.calculate_betweenness_centrality(network);
        let closeness = self.calculate_closeness_centrality(network);
        let degree = self.calculate_degree_centrality(network);
        let eigenvector = self.calculate_eigenvector_centrality(network);

        let mut hotspots: Vec<AllostericHotspot> = network
            .idx_to_residue
            .iter()
            .filter_map(|&res| {
                let bc = betweenness.get(&res).copied().unwrap_or(0.0);
                let cc = closeness.get(&res).copied().unwrap_or(0.0);
                let dc = degree.get(&res).copied().unwrap_or(0.0);
                let ec = eigenvector.get(&res).copied().unwrap_or(0.0);

                // Combined score (weighted average)
                let combined = 0.4 * bc + 0.3 * cc + 0.1 * dc + 0.2 * ec;

                if combined >= threshold {
                    Some(AllostericHotspot {
                        residue: res,
                        betweenness_centrality: bc,
                        closeness_centrality: cc,
                        degree_centrality: dc,
                        eigenvector_centrality: ec,
                        combined_score: combined,
                    })
                } else {
                    None
                }
            })
            .collect();

        // Sort by combined score (descending)
        hotspots.sort_by(|a, b| b.combined_score.partial_cmp(&a.combined_score).unwrap());

        hotspots
    }

    /// Annotate communication pathways with bottleneck residues
    pub fn annotate_pathway_bottlenecks(
        &self,
        network: &ResidueNetwork,
        pathways: &mut [CommunicationPathway],
    ) {
        let betweenness = self.calculate_betweenness_centrality(network);

        for pathway in pathways {
            // Find residue with highest betweenness on the path
            let bottleneck = pathway
                .path
                .iter()
                .skip(1) // Skip source
                .take(pathway.path.len().saturating_sub(2)) // Skip target
                .max_by(|&a, &b| {
                    let bc_a = betweenness.get(a).copied().unwrap_or(0.0);
                    let bc_b = betweenness.get(b).copied().unwrap_or(0.0);
                    bc_a.partial_cmp(&bc_b).unwrap()
                })
                .copied();

            pathway.bottleneck = bottleneck;
        }
    }

    /// Calculate communication efficiency between regions
    pub fn calculate_communication_efficiency(
        &self,
        network: &ResidueNetwork,
        region_a: &[i32],
        region_b: &[i32],
    ) -> f64 {
        let n = network.size;

        // Get distances (simple BFS-based)
        let mut total_efficiency = 0.0;
        let mut pair_count = 0;

        for &res_a in region_a {
            if let Some(&idx_a) = network.residue_to_idx.get(&res_a) {
                // BFS from res_a
                let mut dist = vec![f64::INFINITY; n];
                dist[idx_a] = 0.0;

                let mut queue = VecDeque::new();
                queue.push_back(idx_a);

                while let Some(v) = queue.pop_front() {
                    for w in 0..n {
                        if network.get(v, w) > 0.0 && dist[w] == f64::INFINITY {
                            dist[w] = dist[v] + 1.0;
                            queue.push_back(w);
                        }
                    }
                }

                // Calculate efficiency to region_b
                for &res_b in region_b {
                    if let Some(&idx_b) = network.residue_to_idx.get(&res_b) {
                        if dist[idx_b].is_finite() && dist[idx_b] > 0.0 {
                            total_efficiency += 1.0 / dist[idx_b];
                            pair_count += 1;
                        }
                    }
                }
            }
        }

        if pair_count > 0 {
            total_efficiency / pair_count as f64
        } else {
            0.0
        }
    }

    /// Identify critical residues for allosteric communication
    /// (residues whose removal would most disrupt communication)
    pub fn identify_critical_residues(
        &self,
        network: &ResidueNetwork,
        source_residues: &[i32],
        target_residues: &[i32],
        top_k: usize,
    ) -> Vec<CriticalResidue> {
        // Calculate baseline efficiency
        let baseline = self.calculate_communication_efficiency(network, source_residues, target_residues);

        if baseline <= 0.0 {
            return Vec::new();
        }

        let betweenness = self.calculate_betweenness_centrality(network);

        // For each intermediate residue, estimate impact of removal
        let source_set: std::collections::HashSet<i32> = source_residues.iter().copied().collect();
        let target_set: std::collections::HashSet<i32> = target_residues.iter().copied().collect();

        let mut critical: Vec<CriticalResidue> = network
            .idx_to_residue
            .iter()
            .filter(|&&res| !source_set.contains(&res) && !target_set.contains(&res))
            .filter_map(|&res| {
                let bc = betweenness.get(&res).copied().unwrap_or(0.0);

                // Higher betweenness = more critical
                // Estimate disruption proportional to betweenness
                let estimated_disruption = bc * baseline;

                if bc > self.min_centrality {
                    Some(CriticalResidue {
                        residue: res,
                        betweenness: bc,
                        estimated_disruption,
                        on_shortest_path: bc > 0.1, // Rough estimate
                    })
                } else {
                    None
                }
            })
            .collect();

        critical.sort_by(|a, b| b.estimated_disruption.partial_cmp(&a.estimated_disruption).unwrap());
        critical.truncate(top_k);

        critical
    }
}

/// An allosteric hotspot residue with centrality scores
#[derive(Debug, Clone)]
pub struct AllostericHotspot {
    pub residue: i32,
    pub betweenness_centrality: f64,
    pub closeness_centrality: f64,
    pub degree_centrality: f64,
    pub eigenvector_centrality: f64,
    pub combined_score: f64,
}

/// A residue critical for allosteric communication
#[derive(Debug, Clone)]
pub struct CriticalResidue {
    pub residue: i32,
    pub betweenness: f64,
    pub estimated_disruption: f64,
    pub on_shortest_path: bool,
}

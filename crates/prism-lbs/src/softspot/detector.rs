//! Soft-spot detector for cryptic binding site identification
//!
//! Implements the core algorithm for detecting cryptic (hidden) binding sites
//! using B-factor flexibility, local packing density, and hydrophobicity signals.

use crate::softspot::constants::*;
use crate::softspot::types::*;
use crate::structure::Atom;
use std::collections::HashMap;

/// Configuration for the soft-spot detector
#[derive(Debug, Clone)]
pub struct SoftSpotConfig {
    /// Apply boundary boost for surface-adjacent clusters
    pub use_boundary_boost: bool,

    /// B-factor z-score threshold for flexibility
    pub bfactor_threshold: f64,

    /// Minimum residues to form a candidate
    pub min_cluster_size: usize,

    /// Maximum residues per candidate
    pub max_cluster_size: usize,

    /// Minimum score to report
    pub min_score: f64,
}

impl Default for SoftSpotConfig {
    fn default() -> Self {
        Self {
            use_boundary_boost: true,
            bfactor_threshold: BFACTOR_ZSCORE_THRESHOLD,
            min_cluster_size: MIN_CLUSTER_SIZE,
            max_cluster_size: MAX_CLUSTER_SIZE,
            min_score: MIN_CRYPTIC_SCORE,
        }
    }
}

/// Soft-spot detector for cryptic binding sites
///
/// Identifies regions that are not currently open cavities but have the
/// biophysical characteristics indicating they could become druggable
/// pockets upon conformational change.
pub struct SoftSpotDetector {
    pub config: SoftSpotConfig,
}

impl Default for SoftSpotDetector {
    fn default() -> Self {
        Self {
            config: SoftSpotConfig::default(),
        }
    }
}

impl SoftSpotDetector {
    /// Create a new detector with default configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a detector with custom configuration
    pub fn with_config(config: SoftSpotConfig) -> Self {
        Self { config }
    }

    /// Main entry point: detect cryptic sites from atoms
    ///
    /// # Arguments
    /// * `atoms` - Slice of protein atoms (should include all heavy atoms)
    ///
    /// # Returns
    /// Vector of cryptic site candidates, sorted by score (highest first)
    pub fn detect(&self, atoms: &[Atom]) -> Vec<CrypticCandidate> {
        if atoms.is_empty() {
            log::warn!("[SOFTSPOT] No atoms provided for analysis");
            return Vec::new();
        }

        log::info!(
            "[SOFTSPOT] Analyzing {} atoms for cryptic sites",
            atoms.len()
        );

        // Step 1: Analyze flexibility (B-factors)
        let flexibility_map = self.analyze_flexibility(atoms);
        log::debug!(
            "[SOFTSPOT] Flexibility analysis: {} residues scored",
            flexibility_map.len()
        );

        // Step 2: Analyze packing density
        let packing_map = self.analyze_packing(atoms);
        log::debug!(
            "[SOFTSPOT] Packing analysis: {} residues scored",
            packing_map.len()
        );

        // Step 3: Analyze hydrophobicity
        let hydrophobicity_map = self.analyze_hydrophobicity(atoms);
        log::debug!(
            "[SOFTSPOT] Hydrophobicity analysis: {} residues scored",
            hydrophobicity_map.len()
        );

        // Step 4: Combine signals into scored residues
        let scored_residues = self.combine_signals(atoms, &flexibility_map, &packing_map, &hydrophobicity_map);
        log::debug!(
            "[SOFTSPOT] {} flexible residues identified",
            scored_residues.len()
        );

        // Step 5: Cluster flexible residues spatially
        let clusters = self.cluster_residues(&scored_residues);
        log::debug!("[SOFTSPOT] {} clusters formed", clusters.len());

        // Step 6: Score clusters and filter
        let candidates = self.score_candidates(clusters);

        log::info!(
            "[SOFTSPOT] Found {} cryptic site candidates",
            candidates.len()
        );
        candidates
    }

    /// Calculate B-factor z-scores per residue
    ///
    /// Returns a map of residue_seq -> z-score
    fn analyze_flexibility(&self, atoms: &[Atom]) -> HashMap<i32, f64> {
        // Collect all B-factors
        let bfactors: Vec<f64> = atoms.iter().map(|a| a.b_factor).collect();

        if bfactors.is_empty() {
            return HashMap::new();
        }

        // Calculate mean and standard deviation
        let mean = bfactors.iter().sum::<f64>() / bfactors.len() as f64;
        let variance = bfactors.iter().map(|b| (b - mean).powi(2)).sum::<f64>() / bfactors.len() as f64;
        // Guard against zero variance (NMR structures, uniform B-factors)
        let std = variance.sqrt().max(0.1);

        // Group B-factors by residue
        let mut residue_bfactors: HashMap<i32, Vec<f64>> = HashMap::new();
        for atom in atoms {
            residue_bfactors
                .entry(atom.residue_seq)
                .or_default()
                .push(atom.b_factor);
        }

        // Calculate average B-factor per residue, convert to z-score
        residue_bfactors
            .into_iter()
            .map(|(res_seq, bfs)| {
                let avg_bf = bfs.iter().sum::<f64>() / bfs.len() as f64;
                let zscore = (avg_bf - mean) / std;
                (res_seq, zscore)
            })
            .collect()
    }

    /// Calculate local packing density per residue
    ///
    /// Returns a map of residue_seq -> density_ratio (actual/expected)
    fn analyze_packing(&self, atoms: &[Atom]) -> HashMap<i32, f64> {
        // Calculate centroid for each residue
        let mut residue_centroids: HashMap<i32, [f64; 3]> = HashMap::new();
        let mut residue_counts: HashMap<i32, usize> = HashMap::new();

        for atom in atoms {
            let entry = residue_centroids
                .entry(atom.residue_seq)
                .or_insert([0.0, 0.0, 0.0]);
            entry[0] += atom.coord[0];
            entry[1] += atom.coord[1];
            entry[2] += atom.coord[2];
            *residue_counts.entry(atom.residue_seq).or_insert(0) += 1;
        }

        // Normalize centroids
        for (res_seq, centroid) in residue_centroids.iter_mut() {
            let n = residue_counts[res_seq] as f64;
            centroid[0] /= n;
            centroid[1] /= n;
            centroid[2] /= n;
        }

        // Calculate packing density for each residue
        let mut packing_map = HashMap::new();
        let sphere_volume = (4.0 / 3.0) * std::f64::consts::PI * PACKING_RADIUS.powi(3);
        let radius_sq = PACKING_RADIUS * PACKING_RADIUS;

        for (&res_seq, centroid) in &residue_centroids {
            // Count atoms from OTHER residues within radius
            let nearby_count = atoms
                .iter()
                .filter(|a| a.residue_seq != res_seq)
                .filter(|a| {
                    let dx = a.coord[0] - centroid[0];
                    let dy = a.coord[1] - centroid[1];
                    let dz = a.coord[2] - centroid[2];
                    (dx * dx + dy * dy + dz * dz) < radius_sq
                })
                .count();

            let density = nearby_count as f64 / sphere_volume;
            let density_ratio = density / EXPECTED_PACKING_DENSITY;
            packing_map.insert(res_seq, density_ratio);
        }

        packing_map
    }

    /// Calculate hydrophobicity per residue
    ///
    /// Returns a map of residue_seq -> hydrophobicity (0-1 normalized)
    fn analyze_hydrophobicity(&self, atoms: &[Atom]) -> HashMap<i32, f64> {
        // Kyte-Doolittle scale normalized to 0-1
        // Original scale: -4.5 (Arg) to +4.5 (Ile)
        // Normalized: 0.0 (Arg) to 1.0 (Ile)
        let scale: HashMap<&str, f64> = [
            ("ALA", 0.700),
            ("ARG", 0.000),
            ("ASN", 0.111),
            ("ASP", 0.111),
            ("CYS", 0.778),
            ("GLN", 0.111),
            ("GLU", 0.111),
            ("GLY", 0.456),
            ("HIS", 0.144),
            ("ILE", 1.000),
            ("LEU", 0.922),
            ("LYS", 0.067),
            ("MET", 0.711),
            ("PHE", 0.811),
            ("PRO", 0.322),
            ("SER", 0.411),
            ("THR", 0.422),
            ("TRP", 0.400),
            ("TYR", 0.356),
            ("VAL", 0.967),
        ]
        .into_iter()
        .collect();

        // Get residue name for each residue
        let mut residue_names: HashMap<i32, String> = HashMap::new();
        for atom in atoms {
            residue_names
                .entry(atom.residue_seq)
                .or_insert_with(|| atom.residue_name.clone());
        }

        // Look up hydrophobicity
        residue_names
            .into_iter()
            .map(|(res_seq, name)| {
                let hydro = scale.get(name.as_str()).copied().unwrap_or(0.5);
                (res_seq, hydro)
            })
            .collect()
    }

    /// Combine flexibility, packing, and hydrophobicity signals
    ///
    /// Filters residues that qualify as "flexible" based on combined criteria
    fn combine_signals(
        &self,
        atoms: &[Atom],
        flexibility: &HashMap<i32, f64>,
        packing: &HashMap<i32, f64>,
        hydrophobicity: &HashMap<i32, f64>,
    ) -> Vec<FlexibleResidue> {
        // Build residue data with centroids
        let mut residue_data: HashMap<i32, (Vec<&Atom>, [f64; 3])> = HashMap::new();
        for atom in atoms {
            let entry = residue_data
                .entry(atom.residue_seq)
                .or_insert_with(|| (Vec::new(), [0.0, 0.0, 0.0]));
            entry.0.push(atom);
        }

        // Calculate centroids
        for (_, (atom_list, centroid)) in residue_data.iter_mut() {
            let n = atom_list.len() as f64;
            centroid[0] = atom_list.iter().map(|a| a.coord[0]).sum::<f64>() / n;
            centroid[1] = atom_list.iter().map(|a| a.coord[1]).sum::<f64>() / n;
            centroid[2] = atom_list.iter().map(|a| a.coord[2]).sum::<f64>() / n;
        }

        let mut results = Vec::new();

        for (&res_seq, (atom_list, centroid)) in &residue_data {
            let flex = flexibility.get(&res_seq).copied().unwrap_or(0.0);
            let pack = packing.get(&res_seq).copied().unwrap_or(1.0);
            let hydro = hydrophobicity.get(&res_seq).copied().unwrap_or(0.5);

            // Must have some flexibility signal
            if flex < BFACTOR_ZSCORE_MINIMUM {
                continue;
            }

            // Qualification: Flexible AND (loosely packed OR hydrophobic)
            // This captures regions that have potential to open AND
            // would be favorable for ligand binding
            let qualifies = flex > self.config.bfactor_threshold
                && (pack < PACKING_DEFICIT_THRESHOLD || hydro > HYDROPHOBICITY_THRESHOLD);

            if qualifies {
                results.push(FlexibleResidue {
                    chain_id: atom_list[0].chain_id,
                    residue_seq: res_seq,
                    residue_name: atom_list[0].residue_name.clone(),
                    bfactor_zscore: flex,
                    packing_density: pack,
                    hydrophobicity: hydro,
                    centroid: *centroid,
                });
            }
        }

        results
    }

    /// Cluster flexible residues spatially using single-linkage clustering
    fn cluster_residues(&self, residues: &[FlexibleResidue]) -> Vec<Vec<FlexibleResidue>> {
        if residues.is_empty() {
            return Vec::new();
        }

        let mut clusters: Vec<Vec<FlexibleResidue>> = Vec::new();
        let mut assigned = vec![false; residues.len()];
        let cluster_dist_sq = CLUSTER_DISTANCE * CLUSTER_DISTANCE;

        for i in 0..residues.len() {
            if assigned[i] {
                continue;
            }

            // Start new cluster
            let mut cluster = vec![residues[i].clone()];
            assigned[i] = true;

            // Grow cluster using single-linkage
            let mut changed = true;
            while changed {
                changed = false;
                for j in 0..residues.len() {
                    if assigned[j] {
                        continue;
                    }

                    // Check if residue j is close to any member of cluster
                    let is_close = cluster.iter().any(|c| {
                        let dx = c.centroid[0] - residues[j].centroid[0];
                        let dy = c.centroid[1] - residues[j].centroid[1];
                        let dz = c.centroid[2] - residues[j].centroid[2];
                        (dx * dx + dy * dy + dz * dz) < cluster_dist_sq
                    });

                    if is_close {
                        cluster.push(residues[j].clone());
                        assigned[j] = true;
                        changed = true;
                    }
                }
            }

            // Filter by size constraints
            if cluster.len() >= self.config.min_cluster_size
                && cluster.len() <= self.config.max_cluster_size
            {
                clusters.push(cluster);
            }
        }

        clusters
    }

    /// Score clusters and convert to candidates
    fn score_candidates(&self, clusters: Vec<Vec<FlexibleResidue>>) -> Vec<CrypticCandidate> {
        let mut candidates: Vec<CrypticCandidate> = clusters
            .into_iter()
            .enumerate()
            .filter_map(|(id, cluster)| self.score_cluster(id, cluster))
            .collect();

        // Sort by cryptic score (highest first)
        candidates.sort_by(|a, b| {
            b.cryptic_score
                .partial_cmp(&a.cryptic_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Re-assign IDs after sorting
        for (i, candidate) in candidates.iter_mut().enumerate() {
            candidate.id = i + 1;
        }

        candidates
    }

    /// Score a single cluster and convert to candidate
    fn score_cluster(&self, id: usize, cluster: Vec<FlexibleResidue>) -> Option<CrypticCandidate> {
        if cluster.is_empty() {
            return None;
        }

        let n = cluster.len() as f64;

        // Average flexibility (normalized to 0-1)
        // z-score of 3.0 maps to 1.0
        let avg_flex = cluster
            .iter()
            .map(|r| (r.bfactor_zscore / 3.0).clamp(0.0, 1.0))
            .sum::<f64>()
            / n;

        // Average packing deficit (1 - density_ratio, clamped)
        let avg_packing_deficit = cluster
            .iter()
            .map(|r| (1.0 - r.packing_density).clamp(0.0, 1.0))
            .sum::<f64>()
            / n;

        // Average hydrophobicity
        let avg_hydro = cluster.iter().map(|r| r.hydrophobicity).sum::<f64>() / n;

        // Cluster centroid
        let centroid = [
            cluster.iter().map(|r| r.centroid[0]).sum::<f64>() / n,
            cluster.iter().map(|r| r.centroid[1]).sum::<f64>() / n,
            cluster.iter().map(|r| r.centroid[2]).sum::<f64>() / n,
        ];

        // Spatial coherence: compact clusters score higher
        let max_dist = cluster
            .iter()
            .map(|r| {
                let dx = r.centroid[0] - centroid[0];
                let dy = r.centroid[1] - centroid[1];
                let dz = r.centroid[2] - centroid[2];
                (dx * dx + dy * dy + dz * dz).sqrt()
            })
            .fold(0.0_f64, |a, b| a.max(b));

        // Expected span for n residues (roughly sqrt(n) * 5 Angstroms)
        let expected_span = n.sqrt() * 5.0;
        let coherence = (expected_span / max_dist.max(1.0)).clamp(0.0, 1.0);

        // Combined cryptic score
        let cryptic_score = FLEXIBILITY_WEIGHT * avg_flex
            + PACKING_WEIGHT * avg_packing_deficit
            + HYDROPHOBICITY_WEIGHT * avg_hydro
            + COHERENCE_WEIGHT * coherence;

        // Filter by minimum score
        if cryptic_score < self.config.min_score {
            return None;
        }

        // Estimate volume
        let estimated_volume = cluster.len() as f64 * VOLUME_PER_RESIDUE;

        // Predict druggability
        let volume_factor = (estimated_volume / OPTIMAL_POCKET_VOLUME).clamp(0.0, 1.0);
        let predicted_druggability = (0.3 * avg_hydro
            + 0.3 * volume_factor
            + 0.2 * avg_packing_deficit
            + 0.2 * coherence)
            .clamp(0.0, 1.0);

        // Generate rationale
        let avg_zscore = cluster.iter().map(|r| r.bfactor_zscore).sum::<f64>() / n;
        let rationale = format!(
            "flex_z={:.2}, packing_deficit={:.0}%, hydrophobicity={:.0}%",
            avg_zscore,
            avg_packing_deficit * 100.0,
            avg_hydro * 100.0
        );

        Some(CrypticCandidate {
            id,
            residue_indices: cluster.iter().map(|r| r.residue_seq).collect(),
            centroid,
            estimated_volume,
            flexibility_score: avg_flex,
            packing_deficit: avg_packing_deficit,
            hydrophobic_score: avg_hydro,
            cryptic_score,
            predicted_druggability,
            confidence: CrypticConfidence::from_score(cryptic_score),
            rationale,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_atom(serial: u32, residue_seq: i32, coord: [f64; 3], b_factor: f64) -> Atom {
        Atom {
            serial,
            name: "CA".to_string(),
            residue_name: "ALA".to_string(),
            chain_id: 'A',
            residue_seq,
            insertion_code: None,
            coord,
            occupancy: 1.0,
            b_factor,
            element: "C".to_string(),
            alt_loc: None,
            model: 1,
            is_hetatm: false,
            sasa: 0.0,
            hydrophobicity: 0.7,
            partial_charge: 0.0,
            is_surface: true,
            depth: 0.0,
            curvature: 0.0,
        }
    }

    #[test]
    fn test_empty_input() {
        let detector = SoftSpotDetector::new();
        let result = detector.detect(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_flexibility_analysis() {
        let detector = SoftSpotDetector::new();
        let atoms = vec![
            make_test_atom(1, 1, [0.0, 0.0, 0.0], 10.0),
            make_test_atom(2, 2, [5.0, 0.0, 0.0], 50.0), // High B-factor
            make_test_atom(3, 3, [10.0, 0.0, 0.0], 20.0),
        ];

        let flexibility = detector.analyze_flexibility(&atoms);
        assert_eq!(flexibility.len(), 3);

        // Residue 2 should have highest z-score
        assert!(flexibility[&2] > flexibility[&1]);
        assert!(flexibility[&2] > flexibility[&3]);
    }

    #[test]
    fn test_clustering() {
        let detector = SoftSpotDetector::with_config(SoftSpotConfig {
            min_cluster_size: 2,
            max_cluster_size: 10,
            ..Default::default()
        });

        let residues = vec![
            FlexibleResidue {
                chain_id: 'A',
                residue_seq: 1,
                residue_name: "ALA".to_string(),
                bfactor_zscore: 1.0,
                packing_density: 0.5,
                hydrophobicity: 0.7,
                centroid: [0.0, 0.0, 0.0],
            },
            FlexibleResidue {
                chain_id: 'A',
                residue_seq: 2,
                residue_name: "VAL".to_string(),
                bfactor_zscore: 1.2,
                packing_density: 0.6,
                hydrophobicity: 0.8,
                centroid: [5.0, 0.0, 0.0], // Within CLUSTER_DISTANCE
            },
            FlexibleResidue {
                chain_id: 'A',
                residue_seq: 10,
                residue_name: "LEU".to_string(),
                bfactor_zscore: 0.9,
                packing_density: 0.4,
                hydrophobicity: 0.9,
                centroid: [50.0, 0.0, 0.0], // Far away
            },
        ];

        let clusters = detector.cluster_residues(&residues);

        // Should form 2 clusters: (1,2) and (10)
        // But (10) is alone, so only 1 cluster if min_cluster_size=2
        assert!(clusters.len() >= 1);
        assert!(clusters[0].len() == 2);
    }
}

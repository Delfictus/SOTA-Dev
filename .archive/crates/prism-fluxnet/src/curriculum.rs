//! Curriculum Q-Table Bank for warmstart initialization.
//!
//! Implements PRISM GPU Plan §6.4: Curriculum Q-Table Bank.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────┐
//! │         CurriculumBank                      │
//! │  ┌────────────────────────────────────────┐ │
//! │  │  Easy    → [Q-table, metadata]        │ │
//! │  │  Medium  → [Q-table, metadata]        │ │
//! │  │  Hard    → [Q-table, metadata]        │ │
//! │  │  VeryHard → [Q-table, metadata]       │ │
//! │  └────────────────────────────────────────┘ │
//! └─────────────────────────────────────────────┘
//!           ▲                          │
//!           │  Load catalog            │  Select best match
//!           │                          ▼
//! ┌─────────────────────────────────────────────┐
//! │  GraphStats (density, avg_degree, ...)      │
//! │  → DifficultyProfile classification         │
//! └─────────────────────────────────────────────┘
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use prism_fluxnet::curriculum::{CurriculumBank, GraphStats};
//!
//! // Load curriculum catalog
//! let bank = CurriculumBank::load("profiles/curriculum/catalog.json")?;
//!
//! // Classify graph difficulty
//! let stats = GraphStats::from_graph(&graph);
//! let profile = stats.classify_profile();
//!
//! // Select best-matching Q-table
//! if let Some(entry) = bank.select_best_match(profile) {
//!     rl_controller.initialize_from_curriculum(&entry.q_table)?;
//! }
//! ```

use prism_core::Graph;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

// ============================================================================
// Graph Statistics
// ============================================================================

/// Graph statistics for difficulty profiling.
///
/// Computes structural metrics to classify graph complexity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphStats {
    /// Number of vertices
    pub num_vertices: usize,

    /// Number of edges
    pub num_edges: usize,

    /// Graph density: 2 * |E| / (|V| * (|V| - 1))
    /// Range: [0, 1]
    pub density: f64,

    /// Average vertex degree: 2 * |E| / |V|
    pub avg_degree: f64,

    /// Clustering coefficient (optional, expensive to compute)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub clustering_coefficient: Option<f64>,

    /// Maximum degree
    pub max_degree: usize,

    /// Degree variance (spread of degree distribution)
    pub degree_variance: f64,
}

impl GraphStats {
    /// Computes statistics from a graph.
    ///
    /// ## Algorithm
    /// - Density = 2 * edges / (vertices * (vertices - 1))
    /// - Avg degree = 2 * edges / vertices
    /// - Max degree = max(degrees)
    /// - Degree variance = Var(degrees)
    ///
    /// Clustering coefficient is NOT computed by default (expensive).
    /// Call `with_clustering()` separately if needed.
    pub fn from_graph(graph: &Graph) -> Self {
        let n = graph.num_vertices;
        let m = graph.num_edges;

        let density = if n > 1 {
            (2.0 * m as f64) / (n * (n - 1)) as f64
        } else {
            0.0
        };

        let avg_degree = if n > 0 {
            (2.0 * m as f64) / n as f64
        } else {
            0.0
        };

        // Compute degree statistics
        let degrees: Vec<usize> = (0..n).map(|v| graph.degree(v)).collect();
        let max_degree = degrees.iter().max().copied().unwrap_or(0);

        let mean_degree = if !degrees.is_empty() {
            degrees.iter().sum::<usize>() as f64 / degrees.len() as f64
        } else {
            0.0
        };

        let degree_variance = if !degrees.is_empty() {
            degrees
                .iter()
                .map(|&d| {
                    let diff = d as f64 - mean_degree;
                    diff * diff
                })
                .sum::<f64>()
                / degrees.len() as f64
        } else {
            0.0
        };

        Self {
            num_vertices: n,
            num_edges: m,
            density,
            avg_degree,
            clustering_coefficient: None,
            max_degree,
            degree_variance,
        }
    }

    /// Computes clustering coefficient (expensive O(V * d^2) operation).
    ///
    /// Clustering coefficient measures local graph density:
    /// C(v) = 2 * |triangles(v)| / (degree(v) * (degree(v) - 1))
    ///
    /// Returns mean clustering coefficient across all vertices.
    pub fn compute_clustering_coefficient(&mut self, graph: &Graph) {
        let n = graph.num_vertices;
        if n == 0 {
            self.clustering_coefficient = Some(0.0);
            return;
        }

        let mut total_clustering = 0.0;
        let mut valid_vertices = 0;

        for v in 0..n {
            let neighbors = &graph.adjacency[v];
            let degree = neighbors.len();

            if degree < 2 {
                continue; // No triangles possible
            }

            // Count triangles: edges between neighbors of v
            let mut triangles = 0;
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let u = neighbors[i];
                    let w = neighbors[j];
                    if graph.adjacency[u].contains(&w) {
                        triangles += 1;
                    }
                }
            }

            let possible_edges = degree * (degree - 1) / 2;
            let clustering = if possible_edges > 0 {
                triangles as f64 / possible_edges as f64
            } else {
                0.0
            };

            total_clustering += clustering;
            valid_vertices += 1;
        }

        let mean_clustering = if valid_vertices > 0 {
            total_clustering / valid_vertices as f64
        } else {
            0.0
        };

        self.clustering_coefficient = Some(mean_clustering);
    }

    /// Classifies the graph into a difficulty profile.
    ///
    /// ## Classification Algorithm
    ///
    /// Uses decision tree based on density and average degree:
    ///
    /// ```text
    /// if density < 0.1 AND avg_degree < 10:
    ///     → Easy (sparse, low connectivity)
    /// elif density < 0.3 AND avg_degree < 50:
    ///     → Medium (moderate density)
    /// elif density < 0.6 AND avg_degree < 100:
    ///     → Hard (high density)
    /// else:
    ///     → VeryHard (dense, high connectivity)
    /// ```
    ///
    /// Refs: PRISM GPU Plan §6.4
    pub fn classify_profile(&self) -> DifficultyProfile {
        match (self.density, self.avg_degree) {
            _ if self.density < 0.1 && self.avg_degree < 10.0 => DifficultyProfile::Easy,
            _ if self.density < 0.3 && self.avg_degree < 50.0 => DifficultyProfile::Medium,
            _ if self.density < 0.6 && self.avg_degree < 100.0 => DifficultyProfile::Hard,
            _ => DifficultyProfile::VeryHard,
        }
    }
}

// ============================================================================
// Difficulty Profile
// ============================================================================

/// Graph difficulty classification.
///
/// Determines which pre-trained Q-table to use from the curriculum bank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum DifficultyProfile {
    /// Sparse graphs (density < 0.1, avg_degree < 10)
    Easy,

    /// Moderate density graphs (density < 0.3, avg_degree < 50)
    Medium,

    /// High density graphs (density < 0.6, avg_degree < 100)
    Hard,

    /// Very dense graphs (density >= 0.6 or avg_degree >= 100)
    VeryHard,
}

impl DifficultyProfile {
    /// Returns all difficulty profiles in order.
    pub fn all() -> Vec<Self> {
        vec![
            DifficultyProfile::Easy,
            DifficultyProfile::Medium,
            DifficultyProfile::Hard,
            DifficultyProfile::VeryHard,
        ]
    }

    /// Returns a human-readable description.
    pub fn description(&self) -> &str {
        match self {
            DifficultyProfile::Easy => "Sparse graphs with low connectivity",
            DifficultyProfile::Medium => "Moderate density graphs",
            DifficultyProfile::Hard => "High density graphs",
            DifficultyProfile::VeryHard => "Very dense graphs with high connectivity",
        }
    }
}

// ============================================================================
// Curriculum Entry
// ============================================================================

/// Single curriculum entry with pre-trained Q-table.
///
/// Associates a difficulty profile with a trained Q-table and metadata
/// tracking the training process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumEntry {
    /// Difficulty profile this Q-table is trained for
    pub profile: DifficultyProfile,

    /// Pre-trained Q-table: state_hash -> action_idx -> Q-value
    /// State hash is computed by discretizing UniversalRLState
    pub q_table: HashMap<u64, HashMap<usize, f32>>,

    /// Training metadata
    pub metadata: CurriculumMetadata,
}

impl CurriculumEntry {
    /// Creates a new curriculum entry.
    pub fn new(
        profile: DifficultyProfile,
        q_table: HashMap<u64, HashMap<usize, f32>>,
        metadata: CurriculumMetadata,
    ) -> Self {
        Self {
            profile,
            q_table,
            metadata,
        }
    }

    /// Returns the total number of state-action pairs in the Q-table.
    pub fn num_entries(&self) -> usize {
        self.q_table.values().map(|actions| actions.len()).sum()
    }

    /// Returns statistics about Q-values.
    pub fn q_value_stats(&self) -> (f32, f32, f32) {
        let mut sum = 0.0;
        let mut count = 0;
        let mut max_q = f32::MIN;
        let mut min_q = f32::MAX;

        for actions in self.q_table.values() {
            for &q in actions.values() {
                sum += q;
                count += 1;
                max_q = max_q.max(q);
                min_q = min_q.min(q);
            }
        }

        let mean = if count > 0 { sum / count as f32 } else { 0.0 };
        (mean, min_q, max_q)
    }
}

/// Metadata about curriculum entry training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumMetadata {
    /// Graph class used for training (e.g., "DSJC125", "random_sparse")
    pub graph_class: String,

    /// Number of training episodes
    pub training_episodes: usize,

    /// Average reward over last 100 episodes
    pub average_reward: f32,

    /// Episode at which training converged
    pub convergence_epoch: usize,

    /// Training timestamp (ISO 8601 format)
    pub timestamp: String,

    /// Optional: Training hyperparameters
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hyperparameters: Option<HashMap<String, f64>>,
}

// ============================================================================
// Curriculum Bank
// ============================================================================

/// Catalog of pre-trained Q-tables indexed by difficulty profile.
///
/// Stores multiple curriculum entries per profile, allowing selection
/// of the best-matching Q-table based on graph characteristics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumBank {
    /// Catalog version
    pub version: String,

    /// Curriculum entries indexed by difficulty profile
    pub entries: HashMap<DifficultyProfile, Vec<CurriculumEntry>>,
}

impl CurriculumBank {
    /// Creates an empty curriculum bank.
    pub fn new() -> Self {
        Self {
            version: "1.0".to_string(),
            entries: HashMap::new(),
        }
    }

    /// Adds a curriculum entry to the bank.
    pub fn add_entry(&mut self, entry: CurriculumEntry) {
        self.entries.entry(entry.profile).or_default().push(entry);
    }

    /// Loads curriculum bank from JSON file.
    ///
    /// ## File Format
    /// ```json
    /// {
    ///   "version": "1.0",
    ///   "entries": [
    ///     {
    ///       "profile": "Easy",
    ///       "q_table": { "12345": {"0": 0.5} },
    ///       "metadata": { ... }
    ///     }
    ///   ]
    /// }
    /// ```
    pub fn load(path: impl AsRef<Path>) -> Result<Self, Box<dyn std::error::Error>> {
        let path = path.as_ref();
        let contents = std::fs::read_to_string(path)?;

        // Deserialize from JSON with flexible format
        #[derive(Deserialize)]
        struct CatalogFile {
            version: String,
            entries: Vec<CurriculumEntry>,
        }

        let catalog: CatalogFile = serde_json::from_str(&contents)?;

        // Build bank from flat entry list
        let mut bank = CurriculumBank::new();
        bank.version = catalog.version;

        for entry in catalog.entries {
            bank.add_entry(entry);
        }

        log::info!(
            "Loaded curriculum bank from {}: version {}, {} profiles",
            path.display(),
            bank.version,
            bank.entries.len()
        );

        Ok(bank)
    }

    /// Saves curriculum bank to JSON file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn std::error::Error>> {
        let path = path.as_ref();

        // Flatten entries for serialization
        #[derive(Serialize)]
        struct CatalogFile {
            version: String,
            entries: Vec<CurriculumEntry>,
        }

        let mut all_entries = Vec::new();
        for entries in self.entries.values() {
            all_entries.extend(entries.iter().cloned());
        }

        let catalog = CatalogFile {
            version: self.version.clone(),
            entries: all_entries,
        };

        let json = serde_json::to_string_pretty(&catalog)?;

        // Create parent directory if needed
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        std::fs::write(path, json)?;

        log::info!(
            "Saved curriculum bank to {}: {} entries",
            path.display(),
            catalog.entries.len()
        );

        Ok(())
    }

    /// Selects the best-matching curriculum entry for a difficulty profile.
    ///
    /// ## Selection Algorithm
    /// 1. Find all entries matching the target profile
    /// 2. If no exact match, try adjacent profiles (fallback)
    /// 3. Select entry with highest average_reward
    /// 4. Return None if no entries available
    ///
    /// Refs: PRISM GPU Plan §6.4
    pub fn select_best_match(&self, profile: DifficultyProfile) -> Option<&CurriculumEntry> {
        // Try exact match first
        if let Some(entries) = self.entries.get(&profile) {
            if !entries.is_empty() {
                return Some(self.select_best_from_list(entries));
            }
        }

        // Fallback: Try adjacent profiles
        let fallback_profiles = self.get_fallback_profiles(profile);
        for fallback in fallback_profiles {
            if let Some(entries) = self.entries.get(&fallback) {
                if !entries.is_empty() {
                    log::debug!(
                        "No exact match for {:?}, using fallback {:?}",
                        profile,
                        fallback
                    );
                    return Some(self.select_best_from_list(entries));
                }
            }
        }

        None
    }

    /// Selects the best entry from a list based on average reward.
    fn select_best_from_list<'a>(&self, entries: &'a [CurriculumEntry]) -> &'a CurriculumEntry {
        entries
            .iter()
            .max_by(|a, b| {
                a.metadata
                    .average_reward
                    .partial_cmp(&b.metadata.average_reward)
                    .unwrap()
            })
            .unwrap()
    }

    /// Returns fallback profiles in order of preference.
    fn get_fallback_profiles(&self, profile: DifficultyProfile) -> Vec<DifficultyProfile> {
        match profile {
            DifficultyProfile::Easy => vec![DifficultyProfile::Medium],
            DifficultyProfile::Medium => vec![DifficultyProfile::Easy, DifficultyProfile::Hard],
            DifficultyProfile::Hard => vec![DifficultyProfile::Medium, DifficultyProfile::VeryHard],
            DifficultyProfile::VeryHard => vec![DifficultyProfile::Hard],
        }
    }

    /// Returns the total number of curriculum entries.
    pub fn num_entries(&self) -> usize {
        self.entries.values().map(|v| v.len()).sum()
    }

    /// Returns statistics about the bank.
    pub fn stats(&self) -> CurriculumBankStats {
        let mut total_entries = 0;
        let mut total_states = 0;
        let mut total_actions = 0;

        for entries in self.entries.values() {
            total_entries += entries.len();
            for entry in entries {
                total_states += entry.q_table.len();
                total_actions += entry.num_entries();
            }
        }

        CurriculumBankStats {
            version: self.version.clone(),
            num_profiles: self.entries.len(),
            total_entries,
            total_states,
            total_actions,
        }
    }
}

impl Default for CurriculumBank {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about curriculum bank.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumBankStats {
    pub version: String,
    pub num_profiles: usize,
    pub total_entries: usize,
    pub total_states: usize,
    pub total_actions: usize,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_stats_from_graph() {
        // Create a simple graph
        let mut graph = Graph::new(5);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);

        let stats = GraphStats::from_graph(&graph);

        assert_eq!(stats.num_vertices, 5);
        assert_eq!(stats.num_edges, 4);

        // density = 2 * 4 / (5 * 4) = 0.4
        assert!((stats.density - 0.4).abs() < 0.01);

        // avg_degree = 2 * 4 / 5 = 1.6
        assert!((stats.avg_degree - 1.6).abs() < 0.01);

        assert_eq!(stats.max_degree, 2);
    }

    #[test]
    fn test_clustering_coefficient() {
        // Create a triangle graph
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 0);

        let mut stats = GraphStats::from_graph(&graph);
        stats.compute_clustering_coefficient(&graph);

        // All vertices form a triangle, so clustering = 1.0
        assert!(stats.clustering_coefficient.is_some());
        let clustering = stats.clustering_coefficient.unwrap();
        assert!((clustering - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_profile_classification_easy() {
        let stats = GraphStats {
            num_vertices: 100,
            num_edges: 50,
            density: 0.05,
            avg_degree: 5.0,
            clustering_coefficient: None,
            max_degree: 10,
            degree_variance: 2.0,
        };

        assert_eq!(stats.classify_profile(), DifficultyProfile::Easy);
    }

    #[test]
    fn test_profile_classification_medium() {
        let stats = GraphStats {
            num_vertices: 100,
            num_edges: 500,
            density: 0.2,
            avg_degree: 30.0,
            clustering_coefficient: None,
            max_degree: 50,
            degree_variance: 10.0,
        };

        assert_eq!(stats.classify_profile(), DifficultyProfile::Medium);
    }

    #[test]
    fn test_profile_classification_hard() {
        let stats = GraphStats {
            num_vertices: 100,
            num_edges: 1500,
            density: 0.5,
            avg_degree: 80.0,
            clustering_coefficient: None,
            max_degree: 90,
            degree_variance: 20.0,
        };

        assert_eq!(stats.classify_profile(), DifficultyProfile::Hard);
    }

    #[test]
    fn test_profile_classification_very_hard() {
        let stats = GraphStats {
            num_vertices: 100,
            num_edges: 3000,
            density: 0.8,
            avg_degree: 120.0,
            clustering_coefficient: None,
            max_degree: 99,
            degree_variance: 5.0,
        };

        assert_eq!(stats.classify_profile(), DifficultyProfile::VeryHard);
    }

    #[test]
    fn test_curriculum_entry_creation() {
        let mut q_table = HashMap::new();
        let mut actions = HashMap::new();
        actions.insert(0, 0.5);
        actions.insert(1, 0.3);
        q_table.insert(12345, actions);

        let metadata = CurriculumMetadata {
            graph_class: "test".to_string(),
            training_episodes: 1000,
            average_reward: 0.8,
            convergence_epoch: 500,
            timestamp: "2025-01-15T10:00:00Z".to_string(),
            hyperparameters: None,
        };

        let entry = CurriculumEntry::new(DifficultyProfile::Easy, q_table, metadata);

        assert_eq!(entry.profile, DifficultyProfile::Easy);
        assert_eq!(entry.num_entries(), 2);

        let (mean, min, max) = entry.q_value_stats();
        assert!((mean - 0.4).abs() < 0.01);
        assert!((min - 0.3).abs() < 0.01);
        assert!((max - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_curriculum_bank_add_entry() {
        let mut bank = CurriculumBank::new();

        let entry1 = create_test_entry(DifficultyProfile::Easy, 0.8);
        let entry2 = create_test_entry(DifficultyProfile::Easy, 0.9);
        let entry3 = create_test_entry(DifficultyProfile::Medium, 0.7);

        bank.add_entry(entry1);
        bank.add_entry(entry2);
        bank.add_entry(entry3);

        assert_eq!(bank.num_entries(), 3);
        assert_eq!(bank.entries.get(&DifficultyProfile::Easy).unwrap().len(), 2);
        assert_eq!(
            bank.entries.get(&DifficultyProfile::Medium).unwrap().len(),
            1
        );
    }

    #[test]
    fn test_curriculum_bank_select_best_match() {
        let mut bank = CurriculumBank::new();

        let entry1 = create_test_entry(DifficultyProfile::Easy, 0.8);
        let entry2 = create_test_entry(DifficultyProfile::Easy, 0.9); // Best
        let entry3 = create_test_entry(DifficultyProfile::Medium, 0.7);

        bank.add_entry(entry1);
        bank.add_entry(entry2);
        bank.add_entry(entry3);

        // Should select entry2 (highest reward)
        let best = bank.select_best_match(DifficultyProfile::Easy).unwrap();
        assert_eq!(best.metadata.average_reward, 0.9);
    }

    #[test]
    fn test_curriculum_bank_fallback() {
        let mut bank = CurriculumBank::new();

        // Only have Medium entry
        let entry = create_test_entry(DifficultyProfile::Medium, 0.7);
        bank.add_entry(entry);

        // Request Easy, should fallback to Medium
        let fallback = bank.select_best_match(DifficultyProfile::Easy);
        assert!(fallback.is_some());
        assert_eq!(fallback.unwrap().profile, DifficultyProfile::Medium);
    }

    #[test]
    fn test_curriculum_bank_no_match() {
        let bank = CurriculumBank::new();

        // Empty bank
        let result = bank.select_best_match(DifficultyProfile::Easy);
        assert!(result.is_none());
    }

    #[test]
    fn test_curriculum_bank_save_load_roundtrip() {
        let mut bank = CurriculumBank::new();
        bank.add_entry(create_test_entry(DifficultyProfile::Easy, 0.8));
        bank.add_entry(create_test_entry(DifficultyProfile::Medium, 0.7));

        // Save to temp file
        let temp_dir = std::env::temp_dir();
        let temp_path = temp_dir.join("test_curriculum_roundtrip.json");

        bank.save(&temp_path).unwrap();

        // Load back
        let loaded_bank = CurriculumBank::load(&temp_path).unwrap();

        assert_eq!(loaded_bank.version, bank.version);
        assert_eq!(loaded_bank.num_entries(), 2);

        // Verify entries
        let easy_entry = loaded_bank
            .select_best_match(DifficultyProfile::Easy)
            .unwrap();
        assert_eq!(easy_entry.metadata.average_reward, 0.8);

        let medium_entry = loaded_bank
            .select_best_match(DifficultyProfile::Medium)
            .unwrap();
        assert_eq!(medium_entry.metadata.average_reward, 0.7);

        // Cleanup
        std::fs::remove_file(&temp_path).ok();
    }

    #[test]
    fn test_curriculum_bank_stats() {
        let mut bank = CurriculumBank::new();
        bank.add_entry(create_test_entry(DifficultyProfile::Easy, 0.8));
        bank.add_entry(create_test_entry(DifficultyProfile::Medium, 0.7));

        let stats = bank.stats();
        assert_eq!(stats.version, "1.0");
        assert_eq!(stats.num_profiles, 2);
        assert_eq!(stats.total_entries, 2);
        assert!(stats.total_states > 0);
        assert!(stats.total_actions > 0);
    }

    // Helper function to create test curriculum entries
    fn create_test_entry(profile: DifficultyProfile, reward: f32) -> CurriculumEntry {
        let mut q_table = HashMap::new();
        let mut actions = HashMap::new();
        actions.insert(0, 0.5);
        actions.insert(1, 0.3);
        q_table.insert(12345, actions);

        let metadata = CurriculumMetadata {
            graph_class: format!("test_{:?}", profile),
            training_episodes: 1000,
            average_reward: reward,
            convergence_epoch: 500,
            timestamp: "2025-01-15T10:00:00Z".to_string(),
            hyperparameters: None,
        };

        CurriculumEntry::new(profile, q_table, metadata)
    }
}

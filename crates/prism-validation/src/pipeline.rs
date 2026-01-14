//! Validation Pipeline
//!
//! End-to-end pipeline connecting curated PDB data to PRISM-NOVA benchmarks.
//! This module orchestrates the full validation workflow:
//!
//! 1. Load curated manifest with provenance
//! 2. Parse PDB files into simulation-ready coordinates
//! 3. Run benchmarks with goal-directed sampling
//! 4. Compare against ground truth (holo structures)
//! 5. Generate publication-ready reports

use crate::data_curation::{CurationManifest, CuratedTarget, AtomicMetadata};
use chrono::Datelike;
use crate::targets::{Target, TargetStructures, PocketDefinition, PocketType, DrugInfo, DrugStatus, Difficulty, ValidationType};
use crate::benchmarks::{AtlasBenchmark, ApoHoloBenchmark, RetrospectiveBenchmark, NovelCrypticBenchmark};
use crate::{ValidationBenchmark, ValidationConfig, BenchmarkResult, ValidationSummary, BenchmarkSummary};
use crate::reports::ValidationReport;
use anyhow::{Result, Context};
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use chrono::Utc;

/// Atomic coordinates extracted from PDB for simulation
#[derive(Debug, Clone)]
pub struct SimulationStructure {
    /// Target name
    pub name: String,
    /// PDB ID
    pub pdb_id: String,
    /// BLAKE3 hash for verification
    pub blake3_hash: String,
    /// CA atom positions (N x 3)
    pub ca_positions: Vec<[f32; 3]>,
    /// All atom positions (M x 3)
    pub all_positions: Vec<[f32; 3]>,
    /// Atom elements
    pub elements: Vec<String>,
    /// Residue indices for each atom
    pub residue_indices: Vec<usize>,
    /// Residue names
    pub residue_names: Vec<String>,
    /// Chain IDs
    pub chain_ids: Vec<String>,
    /// B-factors (temperature factors)
    pub b_factors: Vec<f32>,
    /// Atom names (e.g., "CA", "N", "O", "CB")
    pub atom_names: Vec<String>,
    /// Residue sequence numbers (from PDB)
    pub residue_seqs: Vec<i32>,
    /// Number of residues
    pub n_residues: usize,
    /// Number of atoms
    pub n_atoms: usize,
    /// Pocket residue indices (if defined)
    pub pocket_residues: Option<Vec<i32>>,
}

impl SimulationStructure {
    /// Extract simulation structure from atomic metadata
    pub fn from_metadata(metadata: &AtomicMetadata, pocket_residues: Option<Vec<i32>>) -> Self {
        let mut ca_positions = Vec::new();
        let mut all_positions = Vec::new();
        let mut elements = Vec::new();
        let mut residue_indices = Vec::new();
        let mut residue_names = Vec::new();
        let mut chain_ids = Vec::new();
        let mut b_factors = Vec::new();
        let mut atom_names = Vec::new();
        let mut residue_seqs = Vec::new();

        // Build residue index map
        let mut res_to_idx: HashMap<String, usize> = HashMap::new();
        for (idx, res) in metadata.residues.iter().enumerate() {
            let key = format!("{}:{}", res.chain_id, res.res_seq);
            res_to_idx.insert(key, idx);
        }

        for atom in &metadata.atoms {
            // Skip HETATM unless it's a standard residue
            if atom.is_hetatm {
                continue;
            }

            all_positions.push([atom.x, atom.y, atom.z]);
            elements.push(atom.element.clone());
            b_factors.push(atom.b_factor);
            chain_ids.push(atom.chain_id.clone());
            residue_names.push(atom.res_name.clone());
            atom_names.push(atom.name.clone());
            residue_seqs.push(atom.res_seq);

            let res_key = format!("{}:{}", atom.chain_id, atom.res_seq);
            let res_idx = res_to_idx.get(&res_key).copied().unwrap_or(0);
            residue_indices.push(res_idx);

            // Track CA atoms
            if atom.name == "CA" {
                ca_positions.push([atom.x, atom.y, atom.z]);
            }
        }

        Self {
            name: metadata.pdb_id.clone(),
            pdb_id: metadata.pdb_id.clone(),
            blake3_hash: metadata.blake3_hash.clone(),
            ca_positions,
            all_positions,
            elements,
            residue_indices,
            residue_names,
            chain_ids,
            b_factors,
            atom_names,
            residue_seqs,
            n_residues: metadata.residues.len(),
            n_atoms: metadata.atoms.len(),
            pocket_residues,
        }
    }

    /// Get pocket atom indices
    pub fn get_pocket_atom_indices(&self) -> Vec<usize> {
        let mut indices = Vec::new();
        if let Some(ref pocket_res) = self.pocket_residues {
            for (atom_idx, &res_idx) in self.residue_indices.iter().enumerate() {
                if pocket_res.contains(&(res_idx as i32)) {
                    indices.push(atom_idx);
                }
            }
        }
        indices
    }

    /// Compute center of mass
    pub fn center_of_mass(&self) -> [f32; 3] {
        if self.all_positions.is_empty() {
            return [0.0, 0.0, 0.0];
        }

        let n = self.all_positions.len() as f32;
        let sum: [f32; 3] = self.all_positions.iter().fold([0.0, 0.0, 0.0], |acc, pos| {
            [acc[0] + pos[0], acc[1] + pos[1], acc[2] + pos[2]]
        });

        [sum[0] / n, sum[1] / n, sum[2] / n]
    }

    /// Compute radius of gyration
    pub fn radius_of_gyration(&self) -> f32 {
        let com = self.center_of_mass();
        let n = self.all_positions.len() as f32;

        let sum_sq: f32 = self.all_positions.iter().map(|pos| {
            let dx = pos[0] - com[0];
            let dy = pos[1] - com[1];
            let dz = pos[2] - com[2];
            dx * dx + dy * dy + dz * dz
        }).sum();

        (sum_sq / n).sqrt()
    }
}

/// Validation pipeline orchestrator
pub struct ValidationPipeline {
    /// Configuration
    config: ValidationConfig,
    /// Curated manifest
    manifest: CurationManifest,
    /// Loaded structures (apo)
    apo_structures: HashMap<String, SimulationStructure>,
    /// Loaded structures (holo - ground truth)
    holo_structures: HashMap<String, SimulationStructure>,
    /// Benchmarks to run
    benchmarks: Vec<Box<dyn ValidationBenchmark>>,
}

impl ValidationPipeline {
    /// Load pipeline from curated manifest
    pub fn from_manifest(manifest_path: &Path, config: ValidationConfig) -> Result<Self> {
        log::info!("Loading curated manifest from {:?}", manifest_path);

        let content = std::fs::read_to_string(manifest_path)
            .context("Failed to read manifest")?;
        let manifest: CurationManifest = serde_json::from_str(&content)
            .context("Failed to parse manifest")?;

        log::info!("Loaded {} curated targets", manifest.targets.len());
        log::info!("Manifest BLAKE3: {}", manifest.manifest_hash.as_ref().unwrap_or(&"N/A".to_string()));

        // Verify manifest integrity
        if manifest.stats.valid_for_blind < manifest.stats.total_targets {
            log::warn!(
                "Only {}/{} targets valid for blind validation",
                manifest.stats.valid_for_blind,
                manifest.stats.total_targets
            );
        }

        // Load structures
        let mut apo_structures = HashMap::new();
        let mut holo_structures = HashMap::new();

        for target in &manifest.targets {
            log::info!("Loading structures for {}", target.name);

            // Load APO structure
            let apo = SimulationStructure::from_metadata(
                &target.apo_metadata,
                Some(target.pocket_residues.clone()),
            );
            apo_structures.insert(target.name.clone(), apo);

            // Load HOLO structure (ground truth)
            let holo = SimulationStructure::from_metadata(
                &target.holo_metadata,
                Some(target.pocket_residues.clone()),
            );
            holo_structures.insert(target.name.clone(), holo);
        }

        // Initialize benchmarks
        let benchmarks: Vec<Box<dyn ValidationBenchmark>> = vec![
            Box::new(AtlasBenchmark::new(&config)?),
            Box::new(ApoHoloBenchmark::new(&config)?),
            Box::new(RetrospectiveBenchmark::new(&config)?),
            Box::new(NovelCrypticBenchmark::new(&config)?),
        ];

        Ok(Self {
            config,
            manifest,
            apo_structures,
            holo_structures,
            benchmarks,
        })
    }

    /// Convert curated target to benchmark target
    fn to_benchmark_target(&self, curated: &CuratedTarget) -> Target {
        Target {
            name: curated.name.clone(),
            family: "kinase".to_string(), // Default, could be derived
            therapeutic_area: Some(curated.therapeutic_area.clone()),
            description: format!("{} target for {}", curated.name, curated.drug_name),
            structures: TargetStructures {
                apo_pdb: Some(curated.apo_provenance.local_path.clone()),
                holo_pdb: Some(curated.holo_provenance.local_path.clone()),
                ensemble_pdb: None,
                af3_pdb: None,
            },
            pocket: Some(PocketDefinition {
                residues: curated.pocket_residues.iter().map(|&r| r as usize).collect(),
                core_residues: curated.pocket_residues.iter().take(6).map(|&r| r as usize).collect(),
                expected_sasa_gain: 150.0,
                pocket_type: PocketType::Cryptic,
                is_cryptic: true,
                mechanism: None,
            }),
            drug_info: Some(DrugInfo {
                name: curated.drug_name.clone(),
                status: DrugStatus::Approved,
                approval_year: Some(curated.drug_date.year() as u32 + 5), // Rough estimate
                mechanism: "Pocket binding".to_string(),
                affinity_nm: None,
                drug_bound_pdb: None,
                smiles: None,
            }),
            difficulty: Difficulty::Hard,
            validation_type: ValidationType::RetrospectiveBlind,
            experimental: None,
        }
    }

    /// Run full validation pipeline
    pub fn run(&self) -> Result<ValidationSummary> {
        let started = Utc::now();
        let mut all_results: Vec<BenchmarkResult> = Vec::new();
        let mut benchmark_summaries: Vec<BenchmarkSummary> = Vec::new();

        log::info!("Starting validation pipeline with {} targets", self.manifest.targets.len());

        for benchmark in &self.benchmarks {
            log::info!("Running {} benchmark", benchmark.name());

            let mut results = Vec::new();
            let mut scores = Vec::new();
            let mut passed = 0;
            let mut best_score = 0.0f64;
            let mut worst_score = 100.0f64;
            let mut best_target = String::new();
            let mut worst_target = String::new();

            for curated_target in &self.manifest.targets {
                // Skip targets not valid for blind validation
                if !curated_target.valid_for_blind {
                    log::warn!("Skipping {} - not valid for blind validation", curated_target.name);
                    continue;
                }

                let target = self.to_benchmark_target(curated_target);

                log::info!("  Running on {} (APO: {}, HOLO: {})",
                    target.name,
                    curated_target.apo_provenance.pdb_id,
                    curated_target.holo_provenance.pdb_id
                );

                match benchmark.run(&target) {
                    Ok(result) => {
                        let score = benchmark.score(&result);

                        if result.passed {
                            passed += 1;
                        }

                        if score.overall > best_score {
                            best_score = score.overall;
                            best_target = target.name.clone();
                        }
                        if score.overall < worst_score {
                            worst_score = score.overall;
                            worst_target = target.name.clone();
                        }

                        scores.push(score.overall);
                        results.push(result);
                    }
                    Err(e) => {
                        log::error!("  Failed: {}", e);
                    }
                }
            }

            let mean_score = if scores.is_empty() {
                0.0
            } else {
                scores.iter().sum::<f64>() / scores.len() as f64
            };

            let std_score = if scores.len() > 1 {
                let variance: f64 = scores.iter()
                    .map(|s| (s - mean_score).powi(2))
                    .sum::<f64>() / (scores.len() - 1) as f64;
                variance.sqrt()
            } else {
                0.0
            };

            benchmark_summaries.push(BenchmarkSummary {
                benchmark: benchmark.name().to_string(),
                targets_run: results.len(),
                targets_passed: passed,
                pass_rate: if results.is_empty() { 0.0 } else { passed as f64 / results.len() as f64 },
                mean_score,
                std_score,
                best_target,
                worst_target,
            });

            all_results.extend(results);
        }

        let finished = Utc::now();

        let overall_pass_rate = if all_results.is_empty() {
            0.0
        } else {
            all_results.iter().filter(|r| r.passed).count() as f64 / all_results.len() as f64
        };

        let overall_score = if benchmark_summaries.is_empty() {
            0.0
        } else {
            benchmark_summaries.iter().map(|s| s.mean_score).sum::<f64>()
                / benchmark_summaries.len() as f64
        };

        Ok(ValidationSummary {
            started,
            finished,
            config: self.config.clone(),
            benchmark_summaries,
            overall_pass_rate,
            overall_score,
            af3_summary: None,
        })
    }

    /// Get structure for simulation
    pub fn get_apo_structure(&self, target_name: &str) -> Option<&SimulationStructure> {
        self.apo_structures.get(target_name)
    }

    /// Get ground truth structure
    pub fn get_holo_structure(&self, target_name: &str) -> Option<&SimulationStructure> {
        self.holo_structures.get(target_name)
    }

    /// Get all target names
    pub fn target_names(&self) -> Vec<String> {
        self.manifest.targets.iter().map(|t| t.name.clone()).collect()
    }

    /// Compute RMSD between two structures (CA atoms only)
    pub fn compute_ca_rmsd(struct1: &SimulationStructure, struct2: &SimulationStructure) -> Option<f32> {
        let n = struct1.ca_positions.len().min(struct2.ca_positions.len());
        if n == 0 {
            return None;
        }

        let sum_sq: f32 = struct1.ca_positions.iter()
            .zip(struct2.ca_positions.iter())
            .take(n)
            .map(|(p1, p2)| {
                let dx = p1[0] - p2[0];
                let dy = p1[1] - p2[1];
                let dz = p1[2] - p2[2];
                dx * dx + dy * dy + dz * dz
            })
            .sum();

        Some((sum_sq / n as f32).sqrt())
    }

    /// Compute pocket RMSD between apo and holo
    pub fn compute_pocket_rmsd(&self, target_name: &str) -> Option<f32> {
        let apo = self.apo_structures.get(target_name)?;
        let holo = self.holo_structures.get(target_name)?;

        // Get pocket atom positions
        let apo_pocket_idx = apo.get_pocket_atom_indices();
        let holo_pocket_idx = holo.get_pocket_atom_indices();

        let n = apo_pocket_idx.len().min(holo_pocket_idx.len());
        if n == 0 {
            return None;
        }

        let sum_sq: f32 = apo_pocket_idx.iter()
            .zip(holo_pocket_idx.iter())
            .take(n)
            .map(|(&i1, &i2)| {
                let p1 = &apo.all_positions[i1];
                let p2 = &holo.all_positions[i2];
                let dx = p1[0] - p2[0];
                let dy = p1[1] - p2[1];
                let dz = p1[2] - p2[2];
                dx * dx + dy * dy + dz * dz
            })
            .sum();

        Some((sum_sq / n as f32).sqrt())
    }

    /// Print pipeline status
    pub fn print_status(&self) {
        println!("\n╔══════════════════════════════════════════════════════════════════╗");
        println!("║              PRISM-4D Validation Pipeline Status                 ║");
        println!("╠══════════════════════════════════════════════════════════════════╣");
        println!("║  Curated targets: {:>3}                                           ║", self.manifest.targets.len());
        println!("║  Valid for blind: {:>3}                                           ║", self.manifest.stats.valid_for_blind);
        println!("║  APO structures:  {:>3}                                           ║", self.apo_structures.len());
        println!("║  HOLO structures: {:>3}                                           ║", self.holo_structures.len());
        println!("║  Benchmarks:      {:>3}                                           ║", self.benchmarks.len());
        println!("╚══════════════════════════════════════════════════════════════════╝");

        println!("\n📊 Target Summary:");
        println!("┌────────────────┬────────────────┬───────────┬───────────┬──────────┐");
        println!("│ Target         │ Drug           │ APO Atoms │ Holo Atoms│ Pocket   │");
        println!("├────────────────┼────────────────┼───────────┼───────────┼──────────┤");

        for target in &self.manifest.targets {
            let apo_atoms = self.apo_structures.get(&target.name)
                .map(|s| s.n_atoms)
                .unwrap_or(0);
            let holo_atoms = self.holo_structures.get(&target.name)
                .map(|s| s.n_atoms)
                .unwrap_or(0);
            let pocket_size = target.pocket_residues.len();

            println!("│ {:<14} │ {:<14} │ {:>9} │ {:>9} │ {:>8} │",
                &target.name[..target.name.len().min(14)],
                &target.drug_name[..target.drug_name.len().min(14)],
                apo_atoms,
                holo_atoms,
                pocket_size
            );
        }
        println!("└────────────────┴────────────────┴───────────┴───────────┴──────────┘");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_structure_center_of_mass() {
        let structure = SimulationStructure {
            name: "test".to_string(),
            pdb_id: "TEST".to_string(),
            blake3_hash: "test".to_string(),
            ca_positions: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            all_positions: vec![[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            elements: vec!["C".to_string(), "C".to_string()],
            residue_indices: vec![0, 1],
            residue_names: vec!["ALA".to_string(), "GLY".to_string()],
            chain_ids: vec!["A".to_string(), "A".to_string()],
            b_factors: vec![20.0, 20.0],
            atom_names: vec!["CA".to_string(), "CA".to_string()],
            residue_seqs: vec![1, 2],
            n_residues: 2,
            n_atoms: 2,
            pocket_residues: None,
        };

        let com = structure.center_of_mass();
        assert!((com[0] - 1.0).abs() < 0.001);
        assert!((com[1] - 0.0).abs() < 0.001);
        assert!((com[2] - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_ca_rmsd() {
        let struct1 = SimulationStructure {
            name: "test1".to_string(),
            pdb_id: "TEST1".to_string(),
            blake3_hash: "test1".to_string(),
            ca_positions: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            all_positions: vec![],
            elements: vec![],
            residue_indices: vec![],
            residue_names: vec![],
            chain_ids: vec![],
            b_factors: vec![],
            atom_names: vec![],
            residue_seqs: vec![],
            n_residues: 2,
            n_atoms: 2,
            pocket_residues: None,
        };

        let struct2 = SimulationStructure {
            name: "test2".to_string(),
            pdb_id: "TEST2".to_string(),
            blake3_hash: "test2".to_string(),
            ca_positions: vec![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            all_positions: vec![],
            elements: vec![],
            residue_indices: vec![],
            residue_names: vec![],
            chain_ids: vec![],
            b_factors: vec![],
            atom_names: vec![],
            residue_seqs: vec![],
            n_residues: 2,
            n_atoms: 2,
            pocket_residues: None,
        };

        let rmsd = ValidationPipeline::compute_ca_rmsd(&struct1, &struct2);
        assert!(rmsd.is_some());
        assert!((rmsd.unwrap() - 1.0).abs() < 0.001);
    }
}

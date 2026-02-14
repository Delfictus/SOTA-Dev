//! Cryptic Site Detection Pipeline
//!
//! Main orchestrator for the pilot-ready cryptic binding site detection system.
//!
//! # Pipeline Steps
//!
//! 1. **Load & Validate**: Parse PDB, sanitize, validate
//! 2. **Sample Conformations**: Generate ensemble via AMBER/NOVA backends
//! 3. **Track Volumes**: Detect pockets per frame, track over trajectory
//! 4. **Classify Sites**: Identify cryptic sites by volume variance
//! 5. **Score Druggability**: Physics-based druggability assessment
//! 6. **Generate Outputs**: PDB, CSV, HTML report
//!
//! # GPU SASA Acceleration
//!
//! When compiled with `cryptic-gpu` feature, uses GPU-accelerated LCPO SASA
//! calculation for ~10-100× speedup on trajectory analysis.

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::Path;

use super::utils::{parse_pdb_simple, SimpleAtom, ShrakeRupleySASA};
use crate::kabsch_alignment::{align_ensemble, compute_rmsf};
use super::config::CrypticPilotConfig;
use super::druggability::{DruggabilityScore, DruggabilityScorer};
use super::volume_tracker::{VolumeFrame, VolumeTimeSeries, VolumeTracker};
use super::outputs::html_report::{CrypticSiteReport, ReportGenerator, SimulationMetadata};
use super::outputs::pdb_writer::MultiModelPdbWriter;
use super::outputs::csv_outputs::{
    write_rmsf_csv, write_contacts_csv, extract_contacts, ContactResidue
};

// GPU SASA support (conditional compilation)
#[cfg(feature = "cryptic-gpu")]
use prism_gpu::LcpoSasaGpu;

/// Result of cryptic site detection pipeline
#[derive(Debug, Clone)]
pub struct CrypticPilotResult {
    /// PDB identifier
    pub pdb_id: String,
    /// Input structure hash (BLAKE3)
    pub input_hash: String,
    /// Number of residues
    pub n_residues: usize,
    /// Number of frames generated
    pub n_frames: usize,
    /// Total computation time (seconds)
    pub computation_time_secs: f64,

    /// Conformational ensemble (Cα coordinates)
    /// Shape: [n_frames][n_residues][3]
    pub conformations: Vec<Vec<[f32; 3]>>,

    /// Per-residue RMSF values
    pub rmsf: Vec<f64>,

    /// Mean RMSD from reference
    pub mean_rmsd: f64,
    /// RMSD standard deviation
    pub rmsd_std: f64,

    /// Detected cryptic sites
    pub cryptic_sites: Vec<CrypticSiteData>,

    /// All tracked pockets (including non-cryptic)
    pub all_pockets: Vec<VolumeTimeSeries>,

    /// Residue metadata
    pub residue_names: Vec<String>,
    pub residue_ids: Vec<i32>,
    pub chain_ids: Vec<char>,
}

/// Data for a single cryptic site
#[derive(Debug, Clone)]
pub struct CrypticSiteData {
    /// Site identifier
    pub site_id: String,
    /// Rank by druggability
    pub rank: usize,
    /// Residues defining the pocket
    pub residues: Vec<i32>,
    /// Pocket centroid
    pub centroid: [f64; 3],
    /// Volume statistics
    pub volume_series: VolumeTimeSeries,
    /// Druggability assessment
    pub druggability: DruggabilityScore,
    /// Contact residues for docking
    pub contacts: Vec<ContactResidue>,
    /// Representative open frame index
    pub representative_frame: usize,
    /// Top 5 open frame indices
    pub top_open_frames: Vec<usize>,
}

impl CrypticPilotResult {
    /// Write all outputs to a directory
    pub fn write_all_outputs(&self, output_dir: &Path) -> Result<()> {
        std::fs::create_dir_all(output_dir)?;

        let prefix = &self.pdb_id;

        // 1. Write trajectory PDB
        self.write_trajectory_pdb(output_dir, prefix)?;

        // 2. Write per-site PDB files
        self.write_site_pdbs(output_dir, prefix)?;

        // 3. Write RMSF CSV
        self.write_rmsf_csv(output_dir, prefix)?;

        // 4. Write volume CSV
        self.write_volume_csv(output_dir, prefix)?;

        // 5. Write contacts CSV (per site)
        self.write_contacts_csv(output_dir, prefix)?;

        // 6. Write HTML report
        self.write_html_report(output_dir, prefix)?;

        log::info!(
            "[PILOT] Wrote all outputs to {:?}",
            output_dir
        );

        Ok(())
    }

    fn write_trajectory_pdb(&self, output_dir: &Path, prefix: &str) -> Result<()> {
        let path = output_dir.join(format!("{}_trajectory.pdb", prefix));
        let mut file = std::fs::File::create(&path)?;

        let mut writer = MultiModelPdbWriter::new(&format!("{} PRISM-4D Trajectory", prefix));
        writer.set_ca_metadata(self.residue_names.clone(), self.chain_ids.first().copied().unwrap_or('A'));
        writer.rmsf_as_bfactor = true;

        writer.write(&self.conformations, &mut file, Some(&self.rmsf))?;

        log::info!("[PILOT] Wrote trajectory: {:?} ({} frames)", path, self.n_frames);
        Ok(())
    }

    fn write_site_pdbs(&self, output_dir: &Path, prefix: &str) -> Result<()> {
        for site in &self.cryptic_sites {
            let path = output_dir.join(format!("{}_{}_open.pdb", prefix, site.site_id.replace(" ", "_")));
            let mut file = std::fs::File::create(&path)?;

            let mut writer = MultiModelPdbWriter::new(&format!("{} {} Open Conformations", prefix, site.site_id));
            writer.set_ca_metadata(self.residue_names.clone(), self.chain_ids.first().copied().unwrap_or('A'));

            writer.write_selected(&self.conformations, &site.top_open_frames, &mut file, Some(&self.rmsf))?;

            log::info!("[PILOT] Wrote site PDB: {:?}", path);
        }
        Ok(())
    }

    fn write_rmsf_csv(&self, output_dir: &Path, prefix: &str) -> Result<()> {
        let path = output_dir.join(format!("{}_rmsf.csv", prefix));
        let mut file = std::fs::File::create(&path)?;

        write_rmsf_csv(
            &mut file,
            &self.residue_ids,
            &self.residue_names,
            &self.chain_ids,
            &self.rmsf,
            None,
        )?;

        log::info!("[PILOT] Wrote RMSF: {:?}", path);
        Ok(())
    }

    fn write_volume_csv(&self, output_dir: &Path, prefix: &str) -> Result<()> {
        let path = output_dir.join(format!("{}_volumes.csv", prefix));
        let mut tracker = VolumeTracker::default();

        // Reconstruct tracker from results
        for pocket in &self.all_pockets {
            for frame in &pocket.frames {
                tracker.add_pocket_observation(
                    frame.frame,
                    frame.time_ps,
                    frame.centroid,
                    frame.volume,
                    frame.sasa,
                    &pocket.defining_residues,
                    frame.druggability,
                );
            }
        }

        let csv = tracker.to_csv();
        std::fs::write(&path, csv)?;

        // Also write summary
        let summary_path = output_dir.join(format!("{}_volumes_summary.csv", prefix));
        let summary = tracker.summary_to_csv();
        std::fs::write(&summary_path, summary)?;

        log::info!("[PILOT] Wrote volumes: {:?}", path);
        Ok(())
    }

    fn write_contacts_csv(&self, output_dir: &Path, prefix: &str) -> Result<()> {
        for site in &self.cryptic_sites {
            let path = output_dir.join(format!("{}_{}_contacts.csv", prefix, site.site_id.replace(" ", "_")));
            let mut file = std::fs::File::create(&path)?;

            write_contacts_csv(&mut file, &site.site_id, &site.contacts)?;

            log::info!("[PILOT] Wrote contacts: {:?}", path);
        }
        Ok(())
    }

    fn write_html_report(&self, output_dir: &Path, prefix: &str) -> Result<()> {
        let path = output_dir.join(format!("{}_report.html", prefix));
        let mut file = std::fs::File::create(&path)?;

        let generator = ReportGenerator::new(&format!("PRISM-4D Cryptic Site Analysis: {}", prefix));

        let metadata = SimulationMetadata {
            pdb_id: self.pdb_id.clone(),
            n_residues: self.n_residues,
            n_atoms: self.n_residues * 8, // Approximate
            n_frames: self.n_frames,
            temperature_k: 310.0,
            duration_ns: (self.n_frames as f64) * 0.05, // 50 ps per frame
            mean_rmsd: self.mean_rmsd,
            rmsd_std: self.rmsd_std,
        };

        let site_reports: Vec<CrypticSiteReport> = self.cryptic_sites.iter().map(|site| {
            CrypticSiteReport {
                site_id: site.site_id.clone(),
                rank: site.rank,
                residues: site.residues.clone(),
                centroid: site.centroid,
                volume_stats: site.volume_series.stats.clone(),
                druggability: site.druggability.clone(),
                representative_frame: site.representative_frame,
            }
        }).collect();

        generator.generate(&mut file, &metadata, &site_reports)?;

        log::info!("[PILOT] Wrote HTML report: {:?}", path);
        Ok(())
    }
}

/// Main cryptic site detection pipeline
pub struct CrypticPilotPipeline {
    config: CrypticPilotConfig,
    druggability_scorer: DruggabilityScorer,
    sasa_calculator: ShrakeRupleySASA,
    /// GPU-accelerated LCPO SASA calculator (when cryptic-gpu feature enabled)
    #[cfg(feature = "cryptic-gpu")]
    gpu_sasa: Option<LcpoSasaGpu>,
}

impl CrypticPilotPipeline {
    /// Create a new pipeline with given configuration
    pub fn new(config: CrypticPilotConfig) -> Result<Self> {
        config.validate().map_err(|e| anyhow::anyhow!("{}", e))?;

        let druggability_scorer = DruggabilityScorer::with_weights(
            config.druggability_hydrophobic_weight,
            config.druggability_enclosure_weight,
            config.druggability_hbond_weight,
        );

        // Initialize GPU SASA calculator if cryptic-gpu feature enabled
        #[cfg(feature = "cryptic-gpu")]
        let gpu_sasa = {
            // CudaContext::new() returns Arc<CudaContext>
            match cudarc::driver::CudaContext::new(0) {
                Ok(context) => {
                    match LcpoSasaGpu::new(context) {
                        Ok(sasa) => {
                            log::info!("[PILOT] GPU LCPO SASA calculator initialized");
                            Some(sasa)
                        }
                        Err(e) => {
                            log::warn!("[PILOT] Failed to init GPU SASA, falling back to CPU: {}", e);
                            None
                        }
                    }
                }
                Err(e) => {
                    log::warn!("[PILOT] No CUDA device available, using CPU SASA: {}", e);
                    None
                }
            }
        };

        Ok(Self {
            config,
            druggability_scorer,
            sasa_calculator: ShrakeRupleySASA::default(),
            #[cfg(feature = "cryptic-gpu")]
            gpu_sasa,
        })
    }

    /// Run the complete pipeline on a prism-prep topology JSON file
    ///
    /// This is the RECOMMENDED method for production use. The topology
    /// has been properly sanitized and validated by prism-prep.
    ///
    /// When compiled with `cryptic-gpu` feature, uses GPU-accelerated LCPO SASA
    /// for ~10-100× speedup on per-frame SASA calculation.
    pub fn analyze_topology(&self, topology_path: &str) -> Result<CrypticPilotResult> {
        use super::topology_loader::PrismTopology;
        use std::path::Path;

        let start_time = std::time::Instant::now();

        log::info!("[PILOT] Starting cryptic site analysis from topology: {}", topology_path);

        // 1. Load prism-prep topology
        let topology = PrismTopology::load(Path::new(topology_path))
            .with_context(|| format!("Failed to load topology: {}", topology_path))?;

        let pdb_id = topology.get_pdb_id();
        let input_hash = blake3::hash(std::fs::read(topology_path)?.as_slice()).to_hex()[..16].to_string();

        // Extract Cα data from topology
        let reference_coords = topology.get_ca_coordinates();
        let residue_names = topology.get_ca_residue_names();
        let residue_ids = topology.get_ca_residue_ids();
        let chain_ids = topology.get_ca_chain_ids();
        let n_residues = reference_coords.len();

        if n_residues < 10 {
            anyhow::bail!("Too few residues in topology: {}", n_residues);
        }

        log::info!("[PILOT] Loaded {} residues ({} atoms) from sanitized topology: {}",
            n_residues, topology.n_atoms, pdb_id);

        // Continue with common analysis pipeline, passing topology for GPU SASA
        self.run_analysis_pipeline_with_topology(
            start_time,
            pdb_id,
            input_hash,
            reference_coords,
            residue_names,
            residue_ids,
            chain_ids,
            &topology,
        )
    }

    /// Run the complete pipeline on a raw PDB file (NOT RECOMMENDED for production)
    ///
    /// For production use, preprocess with prism-prep first and use analyze_topology().
    #[allow(deprecated)]
    pub fn analyze_raw_pdb(&self, pdb_path: &str) -> Result<CrypticPilotResult> {
        let start_time = std::time::Instant::now();

        log::warn!("[PILOT] Using raw PDB input - for production, use analyze_topology() with prism-prep output");
        log::info!("[PILOT] Starting cryptic site analysis for: {}", pdb_path);

        // 1. Load and parse structure
        let pdb_content = std::fs::read_to_string(pdb_path)
            .with_context(|| format!("Failed to read PDB file: {}", pdb_path))?;

        let atoms = parse_pdb_simple(&pdb_content);
        if atoms.is_empty() {
            anyhow::bail!("No atoms found in PDB file");
        }

        // Extract PDB ID from filename
        let pdb_id = Path::new(pdb_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Compute input hash
        let input_hash = blake3::hash(pdb_content.as_bytes()).to_hex()[..16].to_string();

        // Extract Cα atoms
        let ca_atoms: Vec<&SimpleAtom> = atoms.iter()
            .filter(|a| a.name == "CA" && !a.is_hetatm)
            .collect();

        if ca_atoms.len() < 10 {
            anyhow::bail!("Too few Cα atoms: {}", ca_atoms.len());
        }

        let n_residues = ca_atoms.len();
        log::info!("[PILOT] Loaded {} residues from {}", n_residues, pdb_id);

        // Extract metadata
        let residue_names: Vec<String> = ca_atoms.iter().map(|a| a.residue_name.clone()).collect();
        let residue_ids: Vec<i32> = ca_atoms.iter().map(|a| a.residue_seq).collect();
        let chain_ids: Vec<char> = ca_atoms.iter().map(|a| a.chain_id).collect();
        let reference_coords: Vec<[f32; 3]> = ca_atoms.iter()
            .map(|a| [a.coord[0] as f32, a.coord[1] as f32, a.coord[2] as f32])
            .collect();

        // Continue with common pipeline
        self.run_analysis_pipeline(
            start_time,
            pdb_id,
            input_hash,
            reference_coords,
            residue_names,
            residue_ids,
            chain_ids,
            Some(&atoms),
        )
    }

    /// Internal: Analysis pipeline with topology for GPU SASA
    fn run_analysis_pipeline_with_topology(
        &self,
        start_time: std::time::Instant,
        pdb_id: String,
        input_hash: String,
        reference_coords: Vec<[f32; 3]>,
        residue_names: Vec<String>,
        residue_ids: Vec<i32>,
        chain_ids: Vec<char>,
        topology: &super::topology_loader::PrismTopology,
    ) -> Result<CrypticPilotResult> {
        let n_residues = reference_coords.len();

        // 2. Generate conformational ensemble
        let conformations = self.generate_ensemble(&reference_coords)?;
        let n_frames = conformations.len();
        log::info!("[PILOT] Generated {} conformations", n_frames);

        // 3. Align ensemble and compute RMSF
        let (aligned_ensemble, displacements) = align_ensemble(&reference_coords, &conformations);
        let rmsf = compute_rmsf(&displacements);

        // Compute RMSD statistics
        let rmsds: Vec<f64> = aligned_ensemble.iter().map(|conf| {
            crate::kabsch_alignment::compute_rmsd(&reference_coords, conf)
        }).collect();

        let mean_rmsd = rmsds.iter().sum::<f64>() / rmsds.len() as f64;
        let rmsd_variance = rmsds.iter().map(|r| (r - mean_rmsd).powi(2)).sum::<f64>()
            / (rmsds.len() - 1).max(1) as f64;
        let rmsd_std = rmsd_variance.sqrt();

        // Pre-compute SASA for all atoms using GPU if available
        #[cfg(feature = "cryptic-gpu")]
        let per_residue_sasa = self.compute_residue_sasa_gpu(topology);

        #[cfg(not(feature = "cryptic-gpu"))]
        let per_residue_sasa: Option<Vec<f64>> = None;

        // 4. Track pocket volumes across trajectory
        let mut volume_tracker = VolumeTracker::new(
            self.config.min_pocket_volume,
            8.0, // centroid match distance
        );

        for (frame_idx, conf) in aligned_ensemble.iter().enumerate() {
            let pockets = self.detect_pockets_in_frame(conf, &residue_names);

            for (centroid, volume, residues) in pockets {
                // Compute SASA using pre-computed per-residue values or estimate
                let sasa = if let Some(ref res_sasa) = per_residue_sasa {
                    // Sum SASA for pocket residues
                    let unique_res_ids: Vec<i32> = {
                        let mut ids = topology.get_ca_residue_ids();
                        ids.sort();
                        ids.dedup();
                        ids
                    };
                    residues.iter()
                        .filter_map(|&res_id| {
                            unique_res_ids.iter().position(|&id| id == res_id)
                                .and_then(|idx| res_sasa.get(idx))
                        })
                        .sum::<f64>()
                } else {
                    // Estimate SASA from volume: SASA ≈ volume^(2/3) * 4.84
                    volume.powf(2.0 / 3.0) * 4.84
                };

                // Compute druggability
                let drug_score = self.score_pocket_druggability(&residues, &residue_names, volume);

                volume_tracker.add_pocket_observation(
                    frame_idx,
                    Some(frame_idx as f64 * 50.0), // 50 ps per frame
                    centroid,
                    volume,
                    sasa,
                    &residues,
                    Some(drug_score.score),
                );
            }
        }

        volume_tracker.finalize();

        // 5. Classify cryptic sites
        let cryptic_pockets = volume_tracker.get_cryptic_pockets();
        let all_pockets: Vec<VolumeTimeSeries> = volume_tracker.get_all_pockets()
            .into_iter()
            .cloned()
            .collect();

        log::info!("[PILOT] Found {} cryptic sites out of {} total pockets",
            cryptic_pockets.len(), all_pockets.len());

        // 6. Build cryptic site data
        let mut cryptic_sites: Vec<CrypticSiteData> = cryptic_pockets.iter().enumerate().map(|(rank, pocket)| {
            let rep_frame = pocket.stats.max_volume_frame;
            let top_frames: Vec<usize> = pocket.get_top_open_frames(self.config.n_representative_structures)
                .iter()
                .map(|f| f.frame)
                .collect();

            let centroid = self.compute_centroid(&aligned_ensemble[rep_frame], &pocket.defining_residues, &residue_ids);

            let pocket_residue_names: Vec<String> = pocket.defining_residues.iter()
                .filter_map(|&res_id| {
                    residue_ids.iter().position(|&r| r == res_id)
                        .and_then(|idx| residue_names.get(idx).cloned())
                })
                .collect();

            let druggability = self.druggability_scorer.score_simple(
                &pocket_residue_names,
                pocket.stats.mean_volume,
            );

            let coords_f64: Vec<[f64; 3]> = aligned_ensemble[rep_frame].iter()
                .map(|c| [c[0] as f64, c[1] as f64, c[2] as f64])
                .collect();

            let contacts = extract_contacts(
                centroid,
                &coords_f64,
                &residue_ids,
                &residue_names,
                &chain_ids,
                12.0,
            );

            CrypticSiteData {
                site_id: format!("Site {}", rank + 1),
                rank: rank + 1,
                residues: pocket.defining_residues.clone(),
                centroid,
                volume_series: (*pocket).clone(),
                druggability,
                contacts,
                representative_frame: rep_frame,
                top_open_frames: top_frames,
            }
        }).collect();

        // Sort by druggability score
        cryptic_sites.sort_by(|a, b| b.druggability.score.partial_cmp(&a.druggability.score)
            .unwrap_or(std::cmp::Ordering::Equal));

        for (i, site) in cryptic_sites.iter_mut().enumerate() {
            site.rank = i + 1;
        }

        let computation_time = start_time.elapsed().as_secs_f64();

        log::info!("[PILOT] Analysis complete in {:.2}s", computation_time);
        log::info!("[PILOT] Detected {} cryptic sites", cryptic_sites.len());

        for site in &cryptic_sites {
            log::info!(
                "[PILOT]   Site {}: {} residues, volume={:.0}Å³, open={:.0}%, druggability={:.2} ({})",
                site.rank,
                site.residues.len(),
                site.volume_series.stats.mean_volume,
                site.volume_series.stats.open_frequency * 100.0,
                site.druggability.score,
                site.druggability.classification.name(),
            );
        }

        Ok(CrypticPilotResult {
            pdb_id,
            input_hash,
            n_residues,
            n_frames,
            computation_time_secs: computation_time,
            conformations: aligned_ensemble,
            rmsf,
            mean_rmsd,
            rmsd_std,
            cryptic_sites,
            all_pockets,
            residue_names,
            residue_ids,
            chain_ids,
        })
    }

    /// Compute per-residue SASA using GPU LCPO
    #[cfg(feature = "cryptic-gpu")]
    fn compute_residue_sasa_gpu(
        &self,
        topology: &super::topology_loader::PrismTopology,
    ) -> Option<Vec<f64>> {
        let gpu_sasa = self.gpu_sasa.as_ref()?;

        let positions = topology.get_positions_flat_f32();
        let atom_types = topology.get_sasa_atom_types();
        let radii = topology.get_vdw_radii();
        let residue_map = topology.get_atom_to_residue_map();

        match gpu_sasa.compute(&positions, &atom_types, Some(&radii)) {
            Ok(result) => {
                // Aggregate to per-residue
                match gpu_sasa.aggregate_to_residues(&result.per_atom, &residue_map, topology.n_residues) {
                    Ok(residue_sasa) => {
                        log::info!("[PILOT] GPU SASA: total={:.1} Å², per-residue computed for {} residues",
                            result.total, residue_sasa.len());
                        Some(residue_sasa.into_iter().map(|s| s as f64).collect())
                    }
                    Err(e) => {
                        log::warn!("[PILOT] GPU SASA aggregation failed: {}", e);
                        None
                    }
                }
            }
            Err(e) => {
                log::warn!("[PILOT] GPU SASA computation failed: {}", e);
                None
            }
        }
    }

    /// Internal: Common analysis pipeline shared by both analyze methods
    fn run_analysis_pipeline(
        &self,
        start_time: std::time::Instant,
        pdb_id: String,
        input_hash: String,
        reference_coords: Vec<[f32; 3]>,
        residue_names: Vec<String>,
        residue_ids: Vec<i32>,
        chain_ids: Vec<char>,
        atoms: Option<&Vec<SimpleAtom>>,
    ) -> Result<CrypticPilotResult> {
        let n_residues = reference_coords.len();

        // 2. Generate conformational ensemble
        let conformations = self.generate_ensemble(&reference_coords)?;
        let n_frames = conformations.len();
        log::info!("[PILOT] Generated {} conformations", n_frames);

        // 3. Align ensemble and compute RMSF
        let (aligned_ensemble, displacements) = align_ensemble(&reference_coords, &conformations);
        let rmsf = compute_rmsf(&displacements);

        // Compute RMSD statistics
        let rmsds: Vec<f64> = aligned_ensemble.iter().map(|conf| {
            crate::kabsch_alignment::compute_rmsd(&reference_coords, conf)
        }).collect();

        let mean_rmsd = rmsds.iter().sum::<f64>() / rmsds.len() as f64;
        let rmsd_variance = rmsds.iter().map(|r| (r - mean_rmsd).powi(2)).sum::<f64>()
            / (rmsds.len() - 1).max(1) as f64;
        let rmsd_std = rmsd_variance.sqrt();

        // 4. Track pocket volumes across trajectory
        let mut volume_tracker = VolumeTracker::new(
            self.config.min_pocket_volume,
            8.0, // centroid match distance
        );

        for (frame_idx, conf) in aligned_ensemble.iter().enumerate() {
            let pockets = self.detect_pockets_in_frame(conf, &residue_names);

            for (centroid, volume, residues) in pockets {
                // Compute SASA - use atoms if available, otherwise estimate from volume
                let sasa = if let Some(atoms) = atoms {
                    self.compute_pocket_sasa(atoms, &residues)
                } else {
                    // Estimate SASA from volume: SASA ≈ volume^(2/3) * 4.84
                    // (based on sphere surface area / volume relationship)
                    volume.powf(2.0 / 3.0) * 4.84
                };

                // Compute druggability
                let drug_score = self.score_pocket_druggability(&residues, &residue_names, volume);

                volume_tracker.add_pocket_observation(
                    frame_idx,
                    Some(frame_idx as f64 * 50.0), // 50 ps per frame
                    centroid,
                    volume,
                    sasa,
                    &residues,
                    Some(drug_score.score),
                );
            }
        }

        volume_tracker.finalize();

        // 5. Classify cryptic sites
        let cryptic_pockets = volume_tracker.get_cryptic_pockets();
        let all_pockets: Vec<VolumeTimeSeries> = volume_tracker.get_all_pockets()
            .into_iter()
            .cloned()
            .collect();

        log::info!("[PILOT] Found {} cryptic sites out of {} total pockets",
            cryptic_pockets.len(), all_pockets.len());

        // 6. Build cryptic site data
        let mut cryptic_sites: Vec<CrypticSiteData> = cryptic_pockets.iter().enumerate().map(|(rank, pocket)| {
            // Get representative frame
            let rep_frame = pocket.stats.max_volume_frame;
            let top_frames: Vec<usize> = pocket.get_top_open_frames(self.config.n_representative_structures)
                .iter()
                .map(|f| f.frame)
                .collect();

            // Compute centroid from defining residues
            let centroid = self.compute_centroid(&aligned_ensemble[rep_frame], &pocket.defining_residues, &residue_ids);

            // Get residue names for druggability
            let pocket_residue_names: Vec<String> = pocket.defining_residues.iter()
                .filter_map(|&res_id| {
                    residue_ids.iter().position(|&r| r == res_id)
                        .and_then(|idx| residue_names.get(idx).cloned())
                })
                .collect();

            let druggability = self.druggability_scorer.score_simple(
                &pocket_residue_names,
                pocket.stats.mean_volume,
            );

            // Extract contacts
            let coords_f64: Vec<[f64; 3]> = aligned_ensemble[rep_frame].iter()
                .map(|c| [c[0] as f64, c[1] as f64, c[2] as f64])
                .collect();

            let contacts = extract_contacts(
                centroid,
                &coords_f64,
                &residue_ids,
                &residue_names,
                &chain_ids,
                12.0, // contact cutoff
            );

            CrypticSiteData {
                site_id: format!("Site {}", rank + 1),
                rank: rank + 1,
                residues: pocket.defining_residues.clone(),
                centroid,
                volume_series: (*pocket).clone(),
                druggability,
                contacts,
                representative_frame: rep_frame,
                top_open_frames: top_frames,
            }
        }).collect();

        // Sort by druggability score
        cryptic_sites.sort_by(|a, b| b.druggability.score.partial_cmp(&a.druggability.score)
            .unwrap_or(std::cmp::Ordering::Equal));

        // Re-rank
        for (i, site) in cryptic_sites.iter_mut().enumerate() {
            site.rank = i + 1;
        }

        let computation_time = start_time.elapsed().as_secs_f64();

        log::info!("[PILOT] Analysis complete in {:.2}s", computation_time);
        log::info!("[PILOT] Detected {} cryptic sites", cryptic_sites.len());

        for site in &cryptic_sites {
            log::info!(
                "[PILOT]   Site {}: {} residues, volume={:.0}Å³, open={:.0}%, druggability={:.2} ({})",
                site.rank,
                site.residues.len(),
                site.volume_series.stats.mean_volume,
                site.volume_series.stats.open_frequency * 100.0,
                site.druggability.score,
                site.druggability.classification.name(),
            );
        }

        Ok(CrypticPilotResult {
            pdb_id,
            input_hash,
            n_residues,
            n_frames,
            computation_time_secs: computation_time,
            conformations: aligned_ensemble,
            rmsf,
            mean_rmsd,
            rmsd_std,
            cryptic_sites,
            all_pockets,
            residue_names,
            residue_ids,
            chain_ids,
        })
    }

    /// Generate conformational ensemble using ANM-based sampling
    ///
    /// In production, this would use the SamplingBackend (NOVA/AMBER).
    /// For now, uses ANM for rapid prototyping.
    fn generate_ensemble(&self, reference: &[[f32; 3]]) -> Result<Vec<Vec<[f32; 3]>>> {
        use crate::anm_ensemble_v2::{AnmEnsembleGeneratorV2, AnmEnsembleConfigV2};

        let n_modes = 20.min(reference.len() - 1);
        let temperature = self.config.temperature_k;

        log::info!("[PILOT] Generating ensemble: {} frames, {} modes, T={:.1}K",
            self.config.n_frames, n_modes, temperature);

        let config = AnmEnsembleConfigV2 {
            n_conformations: self.config.n_frames,
            n_modes,
            temperature: temperature as f64,
            seed: Some(self.config.seed),
            ..Default::default()
        };

        let mut generator = AnmEnsembleGeneratorV2::new(config);
        let ensemble = generator.generate_ensemble(reference)
            .context("Failed to generate ANM ensemble")?;

        Ok(ensemble.conformations)
    }

    /// Detect pockets in a single frame using grid-based method
    fn detect_pockets_in_frame(
        &self,
        coords: &[[f32; 3]],
        _residue_names: &[String],
    ) -> Vec<([f64; 3], f64, Vec<i32>)> {
        // Simplified pocket detection based on local concavity
        // In production, would use alpha-spheres or fpocket-like algorithm

        let mut pockets = Vec::new();
        let n = coords.len();

        // Find local concave regions
        let neighbor_cutoff = 10.0f32;
        let neighbor_cutoff_sq = neighbor_cutoff * neighbor_cutoff;

        for i in 0..n {
            let center = coords[i];

            // Find neighbors
            let mut neighbors: Vec<usize> = Vec::new();
            for j in 0..n {
                if i == j { continue; }
                let dx = coords[j][0] - center[0];
                let dy = coords[j][1] - center[1];
                let dz = coords[j][2] - center[2];
                if dx * dx + dy * dy + dz * dz < neighbor_cutoff_sq {
                    neighbors.push(j);
                }
            }

            // Check for concavity (moderately buried but not fully buried)
            if neighbors.len() >= 6 && neighbors.len() <= 15 {
                // Potential pocket region
                let mut pocket_residues: Vec<i32> = vec![i as i32 + 1];
                pocket_residues.extend(neighbors.iter().take(10).map(|&j| j as i32 + 1));

                let centroid = [
                    center[0] as f64,
                    center[1] as f64,
                    center[2] as f64,
                ];

                // Estimate volume from neighbor count
                let volume = (neighbors.len() as f64) * 25.0; // ~25 Å³ per contact

                if volume >= self.config.min_pocket_volume && volume <= self.config.max_pocket_volume {
                    pockets.push((centroid, volume, pocket_residues));
                }
            }
        }

        // Merge overlapping pockets
        let merged = self.merge_overlapping_pockets(pockets);

        merged
    }

    /// Merge pockets with overlapping residues
    fn merge_overlapping_pockets(
        &self,
        pockets: Vec<([f64; 3], f64, Vec<i32>)>,
    ) -> Vec<([f64; 3], f64, Vec<i32>)> {
        if pockets.is_empty() {
            return pockets;
        }

        let mut merged: Vec<([f64; 3], f64, Vec<i32>)> = Vec::new();
        let mut used = vec![false; pockets.len()];

        for i in 0..pockets.len() {
            if used[i] { continue; }

            let mut current = pockets[i].clone();
            used[i] = true;

            // Find and merge overlapping pockets
            for j in (i + 1)..pockets.len() {
                if used[j] { continue; }

                // Check overlap
                let overlap: usize = current.2.iter()
                    .filter(|r| pockets[j].2.contains(r))
                    .count();

                if overlap >= 3 {
                    // Merge
                    let (c2, v2, r2) = &pockets[j];

                    // Update centroid
                    let n1 = current.2.len() as f64;
                    let n2 = r2.len() as f64;
                    current.0[0] = (current.0[0] * n1 + c2[0] * n2) / (n1 + n2);
                    current.0[1] = (current.0[1] * n1 + c2[1] * n2) / (n1 + n2);
                    current.0[2] = (current.0[2] * n1 + c2[2] * n2) / (n1 + n2);

                    // Update volume (avoid double-counting)
                    current.1 = current.1.max(*v2);

                    // Merge residues
                    for &res in r2 {
                        if !current.2.contains(&res) {
                            current.2.push(res);
                        }
                    }

                    used[j] = true;
                }
            }

            merged.push(current);
        }

        merged
    }

    /// Compute SASA for pocket residues
    fn compute_pocket_sasa(&self, atoms: &[SimpleAtom], pocket_residues: &[i32]) -> f64 {
        let pocket_atoms: Vec<SimpleAtom> = atoms.iter()
            .filter(|a| pocket_residues.contains(&a.residue_seq) && !a.is_hetatm)
            .cloned()
            .collect();

        if pocket_atoms.is_empty() {
            return 0.0;
        }

        let sasa_result = self.sasa_calculator.calculate(&pocket_atoms);
        sasa_result.total_sasa
    }

    /// Score pocket druggability
    fn score_pocket_druggability(
        &self,
        _residue_ids: &[i32],
        residue_names: &[String],
        volume: f64,
    ) -> DruggabilityScore {
        self.druggability_scorer.score_simple(residue_names, volume)
    }

    /// Compute centroid from residue IDs
    fn compute_centroid(
        &self,
        coords: &[[f32; 3]],
        residue_ids: &[i32],
        all_residue_ids: &[i32],
    ) -> [f64; 3] {
        let mut sum = [0.0f64; 3];
        let mut count = 0;

        for &res_id in residue_ids {
            if let Some(idx) = all_residue_ids.iter().position(|&r| r == res_id) {
                sum[0] += coords[idx][0] as f64;
                sum[1] += coords[idx][1] as f64;
                sum[2] += coords[idx][2] as f64;
                count += 1;
            }
        }

        if count > 0 {
            [sum[0] / count as f64, sum[1] / count as f64, sum[2] / count as f64]
        } else {
            [0.0, 0.0, 0.0]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_validation() {
        let config = CrypticPilotConfig::default();
        assert!(config.validate().is_ok());

        let mut bad_config = config.clone();
        bad_config.n_frames = 10;
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_pipeline_creation() {
        let config = CrypticPilotConfig::quick();
        let pipeline = CrypticPilotPipeline::new(config);
        assert!(pipeline.is_ok());
    }
}

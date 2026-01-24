//! PDB Data Curation Binary
//!
//! Downloads and curates PDB structures for retrospective blind validation.
//! Ensures scientific integrity through cryptographic provenance tracking.
//!
//! ## Usage
//!
//! ```bash
//! prism-curate --output-dir data/validation/curated
//! ```

use anyhow::{Context, Result};
use chrono::Utc;
use indicatif::{ProgressBar, ProgressStyle};
use prism_validation::data_curation::{
    AtomicMetadata, CuratedTarget, CurationManifest, CurationStats, DataCurator,
    PdbProvenance, TargetDefinition, get_validation_targets,
};
use std::collections::HashMap;
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .format_timestamp_secs()
        .init();

    let output_dir = std::env::args()
        .skip_while(|arg| arg != "--output-dir")
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/validation/curated"));

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     PRISM-4D PDB Data Curation with Cryptographic Provenance     ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  • BLAKE3 hashing for immutable audit trail                      ║");
    println!("║  • Full atomic metadata extraction                               ║");
    println!("║  • Temporal validation (no data leakage)                         ║");
    println!("║  • Scientific defensibility for retrospective blind validation   ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    // Create output directories
    std::fs::create_dir_all(&output_dir)?;
    let pdb_dir = output_dir.join("pdb");
    let apo_dir = pdb_dir.join("apo");
    let holo_dir = pdb_dir.join("holo");
    std::fs::create_dir_all(&apo_dir)?;
    std::fs::create_dir_all(&holo_dir)?;

    log::info!("Output directory: {:?}", output_dir);

    // Get validation targets
    let targets = get_validation_targets();
    log::info!("Found {} validation targets to curate", targets.len());

    // Create curators for apo and holo
    let mut apo_curator = DataCurator::new(apo_dir.clone());
    let mut holo_curator = DataCurator::new(holo_dir.clone());

    // Collect unique PDB IDs
    let mut apo_pdbs: Vec<(&TargetDefinition, String)> = Vec::new();
    let mut holo_pdbs: Vec<(&TargetDefinition, String)> = Vec::new();

    for target in &targets {
        apo_pdbs.push((target, target.apo_pdb.clone()));
        holo_pdbs.push((target, target.holo_pdb.clone()));
    }

    println!("\n📥 Phase 1: Downloading APO structures (blind input)...\n");
    println!("   These structures MUST predate drug discovery for valid blind testing.\n");

    let pb = ProgressBar::new(apo_pdbs.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
        .unwrap()
        .progress_chars("█▓▒░"));

    let mut apo_provenance: HashMap<String, (PdbProvenance, AtomicMetadata)> = HashMap::new();

    for (target, pdb_id) in &apo_pdbs {
        pb.set_message(format!("{} ({})", pdb_id, target.name));

        match download_and_process(&mut apo_curator, pdb_id, target).await {
            Ok((mut prov, metadata)) => {
                // Validate temporal integrity
                DataCurator::validate_temporal(&mut prov, target.drug_discovery_date);

                if prov.validation.temporal_valid {
                    log::info!(
                        "✓ {} [{}]: Valid - deposited {} days BEFORE drug discovery",
                        pdb_id,
                        target.name,
                        prov.validation.days_before_drug.unwrap_or(0)
                    );
                } else {
                    log::warn!(
                        "⚠ {} [{}]: TEMPORAL ISSUE - {}",
                        pdb_id,
                        target.name,
                        prov.validation.warnings.join("; ")
                    );
                }

                apo_provenance.insert(target.name.clone(), (prov, metadata));
            }
            Err(e) => {
                log::error!("✗ Failed to download {}: {}", pdb_id, e);
            }
        }

        pb.inc(1);
    }
    pb.finish_with_message("APO structures downloaded");

    println!("\n📥 Phase 2: Downloading HOLO structures (ground truth)...\n");
    println!("   These are for EVALUATION ONLY - not used during simulation!\n");

    let pb = ProgressBar::new(holo_pdbs.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta}) {msg}")
        .unwrap()
        .progress_chars("█▓▒░"));

    let mut holo_provenance: HashMap<String, (PdbProvenance, AtomicMetadata)> = HashMap::new();

    for (target, pdb_id) in &holo_pdbs {
        pb.set_message(format!("{} ({})", pdb_id, target.name));

        match download_and_process(&mut holo_curator, pdb_id, target).await {
            Ok((prov, metadata)) => {
                log::info!(
                    "✓ {} [{}]: {} atoms, {} residues",
                    pdb_id,
                    target.name,
                    prov.n_atoms,
                    prov.n_residues
                );
                holo_provenance.insert(target.name.clone(), (prov, metadata));
            }
            Err(e) => {
                log::error!("✗ Failed to download {}: {}", pdb_id, e);
            }
        }

        pb.inc(1);
    }
    pb.finish_with_message("HOLO structures downloaded");

    println!("\n🔍 Phase 3: Building curated target database...\n");

    let mut curated_targets: Vec<CuratedTarget> = Vec::new();
    let mut total_atoms = 0;
    let mut total_residues = 0;
    let mut area_counts: HashMap<String, usize> = HashMap::new();
    let mut valid_count = 0;

    for target in &targets {
        if let (Some((apo_prov, apo_meta)), Some((holo_prov, holo_meta))) =
            (apo_provenance.get(&target.name), holo_provenance.get(&target.name))
        {
            let valid_for_blind = apo_prov.validation.safe_for_blind;
            if valid_for_blind {
                valid_count += 1;
            }

            total_atoms += apo_meta.atoms.len() + holo_meta.atoms.len();
            total_residues += apo_meta.residues.len() + holo_meta.residues.len();
            *area_counts.entry(target.therapeutic_area.clone()).or_insert(0) += 1;

            let mut notes = target.notes.clone();
            notes.extend(apo_prov.validation.warnings.clone());

            curated_targets.push(CuratedTarget {
                name: target.name.clone(),
                therapeutic_area: target.therapeutic_area.clone(),
                drug_name: target.drug_name.clone(),
                drug_date: target.drug_discovery_date,
                apo_provenance: apo_prov.clone(),
                apo_metadata: apo_meta.clone(),
                holo_provenance: holo_prov.clone(),
                holo_metadata: holo_meta.clone(),
                pocket_residues: target.pocket_residues.clone(),
                valid_for_blind,
                notes,
            });

            println!(
                "   {} {}: {} → {} | APO: {} atoms | HOLO: {} atoms | Blind: {}",
                if valid_for_blind { "✓" } else { "⚠" },
                target.name,
                target.apo_pdb,
                target.holo_pdb,
                apo_meta.atoms.len(),
                holo_meta.atoms.len(),
                if valid_for_blind { "VALID" } else { "CAUTION" }
            );
        } else {
            log::warn!("Missing data for target {}", target.name);
        }
    }

    println!("\n📊 Phase 4: Generating provenance manifest...\n");

    let mut manifest = CurationManifest {
        version: "1.0.0".to_string(),
        created_at: Utc::now(),
        manifest_hash: None,
        targets: curated_targets.clone(),
        stats: CurationStats {
            total_targets: curated_targets.len(),
            valid_for_blind: valid_count,
            total_atoms,
            total_residues,
            therapeutic_areas: area_counts.clone(),
        },
    };

    let manifest_path = output_dir.join("curation_manifest.json");
    apo_curator.save_manifest(&mut manifest, &manifest_path)?;

    // Generate human-readable report
    let report_path = output_dir.join("CURATION_REPORT.md");
    generate_report(&manifest, &report_path)?;

    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                    CURATION COMPLETE                             ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Targets curated: {:>3}                                           ║", manifest.stats.total_targets);
    println!("║  Valid for blind: {:>3} ({:.0}%)                                      ║",
             valid_count,
             (valid_count as f64 / manifest.stats.total_targets.max(1) as f64) * 100.0);
    println!("║  Total atoms:     {:>7}                                       ║", total_atoms);
    println!("║  Total residues:  {:>7}                                       ║", total_residues);
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║  Manifest:  {:?}", manifest_path);
    println!("║  Report:    {:?}", report_path);
    println!("║  Manifest BLAKE3: {}...", &manifest.manifest_hash.as_ref().unwrap()[..16]);
    println!("╚══════════════════════════════════════════════════════════════════╝");

    // Print therapeutic area breakdown
    println!("\n📈 Therapeutic Area Coverage:");
    for (area, count) in &area_counts {
        println!("   • {}: {} targets", area, count);
    }

    // Print temporal validation summary
    println!("\n⏰ Temporal Integrity Summary:");
    for target in &curated_targets {
        let status = if target.valid_for_blind {
            format!(
                "✓ {} days before drug",
                target.apo_provenance.validation.days_before_drug.unwrap_or(0)
            )
        } else {
            format!(
                "⚠ {}",
                target.apo_provenance.validation.warnings.first()
                    .map(|s| s.as_str())
                    .unwrap_or("Unknown issue")
            )
        };
        println!("   • {} ({}): {}", target.name, target.drug_name, status);
    }

    Ok(())
}

async fn download_and_process(
    curator: &mut DataCurator,
    pdb_id: &str,
    target: &TargetDefinition,
) -> Result<(PdbProvenance, AtomicMetadata)> {
    // Download PDB
    let provenance = curator
        .download_pdb(pdb_id)
        .await
        .with_context(|| format!("Failed to download PDB {}", pdb_id))?;

    // Read file content for metadata extraction
    let content = std::fs::read_to_string(&provenance.local_path)
        .with_context(|| format!("Failed to read downloaded PDB {}", pdb_id))?;

    // Extract atomic metadata
    let metadata = DataCurator::extract_atomic_metadata(&content, pdb_id, &provenance.blake3_hash);

    log::debug!(
        "Extracted metadata for {}: {} atoms, {} residues, {} chains",
        pdb_id,
        metadata.atoms.len(),
        metadata.residues.len(),
        metadata.chains.len()
    );

    Ok((provenance, metadata))
}

fn generate_report(manifest: &CurationManifest, path: &PathBuf) -> Result<()> {
    let mut report = String::new();

    report.push_str("# PRISM-4D Data Curation Report\n\n");
    report.push_str(&format!("**Generated**: {}\n\n", manifest.created_at));
    report.push_str(&format!("**Manifest BLAKE3**: `{}`\n\n", manifest.manifest_hash.as_ref().unwrap_or(&"N/A".to_string())));

    report.push_str("## Executive Summary\n\n");
    report.push_str(&format!(
        "This document provides cryptographic provenance for {} validation targets \
        used in PRISM-4D retrospective blind validation. {} targets ({:.0}%) meet \
        temporal integrity requirements for scientifically defensible blind testing.\n\n",
        manifest.stats.total_targets,
        manifest.stats.valid_for_blind,
        (manifest.stats.valid_for_blind as f64 / manifest.stats.total_targets.max(1) as f64) * 100.0
    ));

    report.push_str("## Data Leakage Prevention\n\n");
    report.push_str("For retrospective blind validation to be scientifically defensible:\n\n");
    report.push_str("1. **APO structures must predate drug discovery** - We only use the APO structure \n");
    report.push_str("   (closed/inactive state) as input, and verify it was deposited before the drug was discovered.\n");
    report.push_str("2. **HOLO structures are ground truth only** - The HOLO structure (open/drug-bound state) \n");
    report.push_str("   is NEVER used during simulation, only for evaluation after the fact.\n");
    report.push_str("3. **No binding site information encoded** - The pocket residues are not provided to \n");
    report.push_str("   PRISM-NOVA during simulation; they're only used to evaluate if the correct site was found.\n\n");

    report.push_str("## Therapeutic Area Coverage\n\n");
    report.push_str("| Area | Targets |\n");
    report.push_str("|------|--------:|\n");
    for (area, count) in &manifest.stats.therapeutic_areas {
        report.push_str(&format!("| {} | {} |\n", area, count));
    }
    report.push_str("\n");

    report.push_str("## Target Details\n\n");

    for target in &manifest.targets {
        report.push_str(&format!("### {} ({})\n\n", target.name, target.therapeutic_area));
        report.push_str(&format!("**Drug**: {} (discovered ~{})\n\n", target.drug_name, target.drug_date));

        report.push_str("#### APO Structure (Blind Input)\n\n");
        report.push_str(&format!("- **PDB ID**: {}\n", target.apo_provenance.pdb_id));
        report.push_str(&format!("- **BLAKE3**: `{}`\n", target.apo_provenance.blake3_hash));
        report.push_str(&format!(
            "- **Deposition Date**: {}\n",
            target.apo_provenance.deposition_date
                .map(|d| d.to_string())
                .unwrap_or("Unknown".to_string())
        ));
        report.push_str(&format!("- **Atoms**: {}\n", target.apo_provenance.n_atoms));
        report.push_str(&format!("- **Residues**: {}\n", target.apo_provenance.n_residues));
        report.push_str(&format!(
            "- **Temporal Validity**: {}\n",
            if target.apo_provenance.validation.temporal_valid {
                format!(
                    "✓ VALID ({} days before drug discovery)",
                    target.apo_provenance.validation.days_before_drug.unwrap_or(0)
                )
            } else {
                format!(
                    "⚠ CAUTION - {}",
                    target.apo_provenance.validation.warnings.first()
                        .map(|s| s.as_str())
                        .unwrap_or("Unknown")
                )
            }
        ));
        report.push_str("\n");

        report.push_str("#### HOLO Structure (Ground Truth - Evaluation Only)\n\n");
        report.push_str(&format!("- **PDB ID**: {}\n", target.holo_provenance.pdb_id));
        report.push_str(&format!("- **BLAKE3**: `{}`\n", target.holo_provenance.blake3_hash));
        report.push_str(&format!("- **Atoms**: {}\n", target.holo_provenance.n_atoms));
        report.push_str(&format!("- **Residues**: {}\n", target.holo_provenance.n_residues));
        report.push_str("\n");

        if !target.notes.is_empty() {
            report.push_str("#### Notes\n\n");
            for note in &target.notes {
                report.push_str(&format!("- {}\n", note));
            }
            report.push_str("\n");
        }

        report.push_str("---\n\n");
    }

    report.push_str("## Provenance Verification\n\n");
    report.push_str("To verify the integrity of this dataset:\n\n");
    report.push_str("```bash\n");
    report.push_str("# Verify manifest hash\n");
    report.push_str("blake3sum curation_manifest.json\n");
    report.push_str("\n");
    report.push_str("# Verify individual PDB files\n");
    report.push_str("for pdb in pdb/apo/*.pdb pdb/holo/*.pdb; do\n");
    report.push_str("    echo \"$pdb: $(blake3sum $pdb | cut -d' ' -f1)\"\n");
    report.push_str("done\n");
    report.push_str("```\n\n");

    report.push_str("## Legal & Ethical Statement\n\n");
    report.push_str("All PDB structures are obtained from the RCSB Protein Data Bank, a publicly \n");
    report.push_str("available resource. The temporal validation ensures that our retrospective \n");
    report.push_str("blind testing is scientifically valid and does not constitute \"p-hacking\" \n");
    report.push_str("or cherry-picking of favorable results.\n");

    std::fs::write(path, report)?;
    log::info!("Generated curation report: {:?}", path);

    Ok(())
}

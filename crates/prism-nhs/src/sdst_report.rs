//! PRISM-Therm output pipeline: JSON reports, druggability PDB, ranked pocket table.
//!
//! Converts `PrismThermAnalysis` into publication-ready output files and
//! a human-readable ranked pocket summary for the log.

use serde::Serialize;
use std::io::Write;
use std::path::Path;

use crate::input::PrismPrepTopology;
use crate::sdst_bridge::{PrismThermAnalysis, PrismThermSiteResult, TideResidueResult};

// ---------------------------------------------------------------------------
// ResidueRole classification
// ---------------------------------------------------------------------------

/// Functional role of a TIDE-identified residue with respect to a pocket.
#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
pub enum ResidueRole {
    /// High TE + high Fisher: causally drives pocket AND high leverage for perturbation.
    /// Prime drug design target.
    Trigger,
    /// High TE + low Fisher: causally connected but not a leverage point.
    /// Conformationally important but harder to exploit.
    Stabilizer,
    /// High KL divergence: large heating/cooling asymmetry.
    /// Gatekeeper residue controlling pocket access.
    Gateway,
    /// Low everything: minimal causal influence on pocket.
    Spectator,
}

impl std::fmt::Display for ResidueRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResidueRole::Trigger    => write!(f, "TRIGGER"),
            ResidueRole::Stabilizer => write!(f, "STABIL."),
            ResidueRole::Gateway    => write!(f, "GATEWAY"),
            ResidueRole::Spectator  => write!(f, "SPECTAT"),
        }
    }
}

/// Classify a TIDE residue based on transfer entropy, Fisher info, and KL divergence.
fn classify_residue(r: &TideResidueResult) -> ResidueRole {
    let high_te = r.transfer_entropy > 0.005;
    let high_fisher = r.fisher_info > 0.002;
    let high_kl = r.kl_divergence > 0.1;

    if high_te && high_fisher {
        ResidueRole::Trigger
    } else if high_te {
        ResidueRole::Stabilizer
    } else if high_kl {
        ResidueRole::Gateway
    } else {
        ResidueRole::Spectator
    }
}

// ---------------------------------------------------------------------------
// PocketReport for standalone JSON
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ResidueContribution {
    pub residue_id: u32,
    pub residue_name: String,
    pub transfer_entropy: f32,
    pub causal_dg: f32,
    pub fisher_info: f32,
    pub kl_divergence: f32,
    pub role: String,
    pub n_causal_spikes: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct PocketReport {
    pub pocket_id: i32,
    pub centroid: [f32; 3],
    pub ccns_tau: f32,
    pub ccns_class: String,
    pub druggability_score: f32,
    pub hysteresis_asymmetry: f32,
    pub is_cryptic: bool,
    pub top_residues: Vec<ResidueContribution>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PrismThermReport {
    pub structure: String,
    pub sdst_event_count: u32,
    pub total_pockets: usize,
    pub cryptic_pockets: usize,
    pub tide_residues_mapped: usize,
    pub pockets: Vec<PocketReport>,
}

// ---------------------------------------------------------------------------
// Output functions
// ---------------------------------------------------------------------------

/// Build the standalone PRISM-Therm report from analysis results.
pub fn build_report(
    analysis: &PrismThermAnalysis,
    topology: &PrismPrepTopology,
    structure_name: &str,
    site_centroids: &[([f32; 3], i32)],  // (centroid, site_id) pairs
) -> PrismThermReport {
    let mut pockets: Vec<PocketReport> = Vec::new();

    for site in &analysis.sites {
        // Find centroid for this site
        let centroid = site_centroids.iter()
            .find(|(_, id)| *id == site.site_id)
            .map(|(c, _)| *c)
            .unwrap_or([0.0, 0.0, 0.0]);

        let top_residues: Vec<ResidueContribution> = site.tide_decomposition.iter()
            .map(|r| {
                let role = classify_residue(r);
                let res_name = residue_label(topology, r.residue_id);
                ResidueContribution {
                    residue_id: r.residue_id,
                    residue_name: res_name,
                    transfer_entropy: r.transfer_entropy,
                    causal_dg: r.causal_dg,
                    fisher_info: r.fisher_info,
                    kl_divergence: r.kl_divergence,
                    role: role.to_string(),
                    n_causal_spikes: r.n_causal_spikes,
                }
            })
            .collect();

        pockets.push(PocketReport {
            pocket_id: site.site_id,
            centroid,
            ccns_tau: site.tau,
            ccns_class: site.ccns_classification.clone(),
            druggability_score: if site.druggability.is_finite() { site.druggability } else { 0.0 },
            hysteresis_asymmetry: site.asymmetry_score,
            is_cryptic: site.is_hysteretic || site.asymmetry_score > 0.15,
            top_residues,
        });
    }

    // Sort by druggability descending
    pockets.sort_by(|a, b| b.druggability_score.partial_cmp(&a.druggability_score)
        .unwrap_or(std::cmp::Ordering::Equal));

    let cryptic_count = pockets.iter().filter(|p| p.is_cryptic).count();

    PrismThermReport {
        structure: structure_name.to_string(),
        sdst_event_count: analysis.sdst_event_count,
        total_pockets: pockets.len(),
        cryptic_pockets: cryptic_count,
        tide_residues_mapped: analysis.tide_residues_mapped,
        pockets,
    }
}

/// Write the standalone `.prism_therm.json` file.
pub fn write_json(report: &PrismThermReport, output_dir: &Path, base_name: &str) -> std::io::Result<()> {
    let path = output_dir.join(format!("{}.prism_therm.json", base_name));
    let json = serde_json::to_string_pretty(report)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(&path, json)?;
    log::info!("  PRISM-Therm JSON: {}", path.display());
    Ok(())
}

/// Write a druggability PDB where B-factor = per-residue druggability score.
///
/// The druggability score for each residue is computed as:
/// - For residues in the TIDE top-20 for any site: max(TE across sites) × 100
/// - For other residues: 0.0
///
/// This produces a heatmap viewable in PyMOL/ChimeraX/3Dmol.js:
///   `spectrum b, blue_white_red, minimum=0, maximum=50`
pub fn write_druggability_pdb(
    report: &PrismThermReport,
    topology: &PrismPrepTopology,
    output_dir: &Path,
    base_name: &str,
) -> std::io::Result<()> {
    // Build per-residue druggability: max TE × 100 across all pockets
    let n_residues = topology.n_residues;
    let mut res_score = vec![0.0f32; n_residues];

    for pocket in &report.pockets {
        for r in &pocket.top_residues {
            let rid = r.residue_id as usize;
            if rid < n_residues {
                let score = r.transfer_entropy * 100.0;
                if score > res_score[rid] {
                    res_score[rid] = score;
                }
            }
        }
    }

    let path = output_dir.join(format!("{}.druggability.pdb", base_name));
    let mut f = std::fs::File::create(&path)?;

    writeln!(f, "REMARK   PRISM-Therm druggability PDB")?;
    writeln!(f, "REMARK   B-factor = Transfer Entropy × 100 (per-residue max across pockets)")?;
    writeln!(f, "REMARK   Visualize: spectrum b, blue_white_red, minimum=0, maximum=50")?;

    for i in 0..topology.n_atoms {
        let x = topology.positions[i * 3];
        let y = topology.positions[i * 3 + 1];
        let z = topology.positions[i * 3 + 2];
        let res_id = topology.residue_ids[i];
        let b_factor = if res_id < n_residues { res_score[res_id] } else { 0.0 };

        let atom_name = &topology.atom_names[i];
        let res_name = if res_id < topology.residue_names.len() {
            &topology.residue_names[res_id]
        } else {
            "UNK"
        };
        let chain_id = if i < topology.chain_ids.len() {
            topology.chain_ids[i].chars().next().unwrap_or('A')
        } else {
            'A'
        };

        // PDB ATOM record format (fixed-width columns)
        write!(f, "ATOM  {:>5} {:>4} {:>3} {}{:>4}    {:>8.3}{:>8.3}{:>8.3}{:>6.2}{:>6.2}\n",
            (i + 1) % 100000,
            format_atom_name(atom_name),
            res_name,
            chain_id,
            (res_id + 1) % 10000,
            x, y, z,
            1.00,  // occupancy
            b_factor.min(99.99),
        )?;
    }
    writeln!(f, "END")?;

    log::info!("  PRISM-Therm druggability PDB: {}", path.display());
    Ok(())
}

/// Print the ranked pocket summary table to the log.
pub fn print_summary_table(report: &PrismThermReport) {
    log::info!("");
    log::info!("╔═════╦══════════════════════╦═══════╦════════╦══════════╦═════════╗");
    log::info!("║  #  ║ Centroid (Å)         ║  tau  ║ Drug.  ║ Asymm.   ║ Type    ║");
    log::info!("╠═════╬══════════════════════╬═══════╬════════╬══════════╬═════════╣");

    for (i, p) in report.pockets.iter().enumerate() {
        let ptype = if p.is_cryptic { "CRYPTIC" } else { "static " };
        log::info!(
            "║ {:>3} ║ ({:>5.1},{:>5.1},{:>5.1}) ║ {:>5.2} ║ {:>5.2}  ║ {:>7.3}  ║ {} ║",
            i + 1,
            p.centroid[0], p.centroid[1], p.centroid[2],
            p.ccns_tau,
            p.druggability_score,
            p.hysteresis_asymmetry,
            ptype,
        );
    }

    log::info!("╚═════╩══════════════════════╩═══════╩════════╩══════════╩═════════╝");

    // Print top causal residues for CRYPTIC pockets
    for p in &report.pockets {
        if !p.is_cryptic || p.top_residues.is_empty() { continue; }

        log::info!("");
        log::info!("  TIDE — Pocket {} (CRYPTIC, tau={:.2}, drug={:.2}):",
            p.pocket_id, p.ccns_tau, p.druggability_score);
        log::info!("    {:>4}  {:>6}  {:>8}  {:>8}  {:>8}  {:>7}  {:>5}",
            "Res", "Name", "TE", "dG", "Fisher", "KL", "Role");
        log::info!("    {}",  "─".repeat(60));

        for r in p.top_residues.iter().take(10) {
            log::info!("    {:>4}  {:>6}  {:>8.5}  {:>8.5}  {:>8.5}  {:>7.4}  {:>7}",
                r.residue_id,
                r.residue_name,
                r.transfer_entropy,
                r.causal_dg,
                r.fisher_info,
                r.kl_divergence,
                r.role,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Get a human-readable residue label like "ALA42" from topology.
fn residue_label(topology: &PrismPrepTopology, res_id: u32) -> String {
    let rid = res_id as usize;
    if rid < topology.residue_names.len() {
        format!("{}{}", topology.residue_names[rid], rid + 1)
    } else {
        format!("?{}", rid + 1)
    }
}

/// Format atom name for PDB (right-justify in 4-char field, with leading space for 1-3 char names).
fn format_atom_name(name: &str) -> String {
    if name.len() >= 4 {
        name[..4].to_string()
    } else {
        format!(" {:<3}", name)
    }
}

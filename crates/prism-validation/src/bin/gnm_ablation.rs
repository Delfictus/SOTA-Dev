//! GNM Enhancement Ablation Study
//!
//! Tests each enhancement individually to identify which ones help vs hurt.

use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use serde::Deserialize;

use prism_physics::gnm_enhanced::{EnhancedGnm, EnhancedGnmConfig};

#[derive(Debug, Clone, Deserialize)]
struct AtlasTarget {
    pdb_id: String,
    #[serde(default)]
    chain: String,
    n_residues: usize,
    md_rmsf: Vec<f64>,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info")
    ).init();

    let data_dir = PathBuf::from("data/atlas_alphaflow");
    let targets_path = data_dir.join("atlas_targets.json");
    let content = fs::read_to_string(&targets_path)?;
    let targets: Vec<AtlasTarget> = serde_json::from_str(&content)?;
    let pdb_dir = data_dir.join("pdb");

    // Define test configurations
    let configs = vec![
        ("Plain GNM (baseline)", EnhancedGnmConfig::plain()),
        ("+ Distance Weighting only", EnhancedGnmConfig {
            use_distance_weighting: true,
            use_multi_cutoff: false,
            use_secondary_structure: false,
            use_sidechain_factors: false,
            use_sasa_modulation: false,
            ..EnhancedGnmConfig::plain()
        }),
        ("+ Multi-Cutoff only", EnhancedGnmConfig {
            use_distance_weighting: false,
            use_multi_cutoff: true,
            use_secondary_structure: false,
            use_sidechain_factors: false,
            use_sasa_modulation: false,
            ensemble_cutoffs: vec![6.0, 7.0, 8.0, 10.0],
            ensemble_weights: vec![0.15, 0.35, 0.35, 0.15],
            ..EnhancedGnmConfig::plain()
        }),
        ("+ MC + DW (combined)", EnhancedGnmConfig {
            use_distance_weighting: true,
            use_multi_cutoff: true,
            use_secondary_structure: false,
            use_sidechain_factors: false,
            use_sasa_modulation: false,
            ensemble_cutoffs: vec![6.0, 7.0, 8.0, 10.0],
            ensemble_weights: vec![0.15, 0.35, 0.35, 0.15],
            ..EnhancedGnmConfig::plain()
        }),
        ("+ Secondary Structure", EnhancedGnmConfig {
            use_distance_weighting: false,
            use_multi_cutoff: false,
            use_secondary_structure: true,
            use_sidechain_factors: false,
            use_sasa_modulation: false,
            ..EnhancedGnmConfig::plain()
        }),
        ("+ Sidechain Factors", EnhancedGnmConfig {
            use_distance_weighting: false,
            use_multi_cutoff: false,
            use_secondary_structure: false,
            use_sidechain_factors: true,
            use_sasa_modulation: false,
            ..EnhancedGnmConfig::plain()
        }),
        ("Optimized Default", EnhancedGnmConfig::default()),
    ];

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║       GNM Enhancement Ablation Study                                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    for (name, config) in &configs {
        let mut correlations = Vec::new();
        let mut gnm = EnhancedGnm::with_config(config.clone());
        gnm.set_cutoff(9.0); // Use optimal cutoff

        for target in &targets {
            // Try both naming conventions
            let pdb_path_with_chain = pdb_dir.join(format!("{}_{}.pdb", target.pdb_id.to_lowercase(), target.chain));
            let pdb_path_no_chain = pdb_dir.join(format!("{}.pdb", target.pdb_id.to_lowercase()));

            let pdb_path = if pdb_path_with_chain.exists() {
                pdb_path_with_chain
            } else if pdb_path_no_chain.exists() {
                pdb_path_no_chain
            } else {
                continue;
            };

            let target_chain = if target.chain.is_empty() { None } else { Some(target.chain.as_str()) };

            if let Ok((ca_pos, res_names, kept_indices)) = parse_pdb_ca_chain_with_indices(&pdb_path, target_chain) {
                if ca_pos.len() < 3 { continue; }

                let res_refs: Vec<&str> = res_names.iter().map(|s| s.as_str()).collect();
                let result = gnm.compute_rmsf(&ca_pos, Some(&res_refs));

                // Align ground truth
                let aligned_exp: Vec<f64> = kept_indices.iter()
                    .filter_map(|&idx| target.md_rmsf.get(idx).copied())
                    .collect();

                if aligned_exp.len() == result.rmsf.len() && !aligned_exp.is_empty() {
                    let corr = pearson_correlation(&result.rmsf, &aligned_exp);
                    if corr.is_finite() {
                        correlations.push(corr);
                    }
                }
            }
        }

        if !correlations.is_empty() {
            let mean = correlations.iter().sum::<f64>() / correlations.len() as f64;
            let pass_rate = correlations.iter().filter(|&&c| c > 0.3).count() as f64 / correlations.len() as f64;
            println!("  {:30} ρ = {:.3}  pass = {:.1}%  (n={})", name, mean, pass_rate * 100.0, correlations.len());
        }
    }

    Ok(())
}

fn parse_pdb_ca_chain_with_indices(path: &PathBuf, target_chain: Option<&str>) -> Result<(Vec<[f32; 3]>, Vec<String>, Vec<usize>)> {
    let content = fs::read_to_string(path)?;
    let mut positions = Vec::new();
    let mut names = Vec::new();
    let mut kept_indices = Vec::new();
    let mut last_res_key = String::new();
    let mut target_chain_ca_index = 0usize;

    for line in content.lines() {
        if !line.starts_with("ATOM") { continue; }
        let atom_name = line.get(12..16).unwrap_or("").trim();
        if atom_name != "CA" { continue; }

        let chain_id = line.get(21..22).unwrap_or(" ");
        if let Some(target) = target_chain {
            if chain_id != target { continue; }
        }

        let current_index = target_chain_ca_index;
        target_chain_ca_index += 1;

        let alt_loc = line.get(16..17).unwrap_or(" ");
        if alt_loc != " " && alt_loc != "A" { continue; }

        let res_num = line.get(22..27).unwrap_or("0").trim();
        let res_key = format!("{}{}", chain_id, res_num);
        if res_key == last_res_key { continue; }
        last_res_key = res_key;

        let x: f32 = line.get(30..38).unwrap_or("0").trim().parse().unwrap_or(0.0);
        let y: f32 = line.get(38..46).unwrap_or("0").trim().parse().unwrap_or(0.0);
        let z: f32 = line.get(46..54).unwrap_or("0").trim().parse().unwrap_or(0.0);
        let res_name = line.get(17..20).unwrap_or("ALA").trim().to_string();

        positions.push([x, y, z]);
        names.push(res_name);
        kept_indices.push(current_index);
    }

    Ok((positions, names, kept_indices))
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() { return 0.0; }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;

    for (xi, yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    if var_x < 1e-10 || var_y < 1e-10 { return 0.0; }
    cov / (var_x.sqrt() * var_y.sqrt())
}

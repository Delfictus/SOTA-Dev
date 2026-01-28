//! GNM Breakthrough Experiments
//!
//! Testing physics-based improvements to break through ρ ≥ 0.70

use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use serde::Deserialize;
use nalgebra::{DMatrix, SymmetricEigen};

#[derive(Debug, Clone, Deserialize)]
struct AtlasTarget {
    pdb_id: String,
    #[serde(default)]
    chain: String,
    md_rmsf: Vec<f64>,
}

fn main() -> Result<()> {
    let data_dir = PathBuf::from("data/atlas_alphaflow");
    let targets_path = data_dir.join("atlas_targets.json");
    let content = fs::read_to_string(&targets_path)?;
    let targets: Vec<AtlasTarget> = serde_json::from_str(&content)?;
    let pdb_dir = data_dir.join("pdb");

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║       GNM BREAKTHROUGH EXPERIMENTS - Targeting ρ ≥ 0.70                    ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Test configurations
    let experiments = vec![
        ("Baseline (σ=5.0)", 5.0, 0, 1.0, false),
        ("σ=3.0", 3.0, 0, 1.0, false),
        ("σ=4.0", 4.0, 0, 1.0, false),
        ("σ=6.0", 6.0, 0, 1.0, false),
        ("σ=7.0", 7.0, 0, 1.0, false),
        ("Terminus boost 1.3x", 5.0, 5, 1.3, false),
        ("Terminus boost 1.5x", 5.0, 5, 1.5, false),
        ("Terminus boost 2.0x", 5.0, 5, 2.0, false),
        ("Top 20 modes only", 5.0, 0, 1.0, true),
        ("σ=4.0 + terminus 1.5x", 4.0, 5, 1.5, false),
    ];

    for (name, sigma, terminus_residues, terminus_boost, mode_limit) in &experiments {
        let mut correlations = Vec::new();

        for target in &targets {
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

            if let Ok((ca_pos, kept_indices)) = parse_pdb_ca_chain_with_indices(&pdb_path, target_chain) {
                if ca_pos.len() < 10 { continue; }

                // Compute RMSF with experimental settings
                let rmsf = compute_rmsf_experimental(
                    &ca_pos,
                    9.0,  // cutoff
                    *sigma,
                    *terminus_residues,
                    *terminus_boost,
                    *mode_limit,
                );

                // Align ground truth
                let aligned_exp: Vec<f64> = kept_indices.iter()
                    .filter_map(|&idx| target.md_rmsf.get(idx).copied())
                    .collect();

                if aligned_exp.len() == rmsf.len() && !aligned_exp.is_empty() {
                    let corr = pearson_correlation(&rmsf, &aligned_exp);
                    if corr.is_finite() {
                        correlations.push(corr);
                    }
                }
            }
        }

        if !correlations.is_empty() {
            let mean = correlations.iter().sum::<f64>() / correlations.len() as f64;
            let above_07 = correlations.iter().filter(|&&c| c >= 0.7).count();
            let pass_rate = correlations.iter().filter(|&&c| c > 0.3).count() as f64 / correlations.len() as f64;

            let delta = mean - 0.615;  // vs baseline
            let marker = if mean >= 0.70 { "🎯" } else if delta > 0.005 { "✅" } else if delta < -0.005 { "❌" } else { "  " };

            println!("  {} {:30} ρ = {:.3}  Δ={:+.3}  ≥0.7: {:2}/{}  pass={:.1}%",
                     marker, name, mean, delta, above_07, correlations.len(), pass_rate * 100.0);
        }
    }

    println!();
    println!("  Legend: 🎯 = ρ≥0.70, ✅ = improvement, ❌ = regression");
    Ok(())
}

fn compute_rmsf_experimental(
    ca_positions: &[[f32; 3]],
    cutoff: f64,
    sigma: f64,
    terminus_residues: usize,
    terminus_boost: f64,
    use_mode_limit: bool,
) -> Vec<f64> {
    let n = ca_positions.len();
    let cutoff_sq = cutoff * cutoff;
    let sigma_sq = sigma * sigma;

    // Build distance-weighted Kirchhoff matrix
    let mut kirchhoff = DMatrix::zeros(n, n);

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = (ca_positions[j][0] - ca_positions[i][0]) as f64;
            let dy = (ca_positions[j][1] - ca_positions[i][1]) as f64;
            let dz = (ca_positions[j][2] - ca_positions[i][2]) as f64;
            let dist_sq = dx * dx + dy * dy + dz * dz;

            if dist_sq < cutoff_sq {
                let w = (-dist_sq / (2.0 * sigma_sq)).exp();
                kirchhoff[(i, j)] = -w;
                kirchhoff[(j, i)] = -w;
            }
        }
    }

    // Set diagonal
    for i in 0..n {
        let row_sum: f64 = kirchhoff.row(i).iter().sum();
        kirchhoff[(i, i)] = -row_sum;
    }

    // Eigendecomposition
    let eigen = SymmetricEigen::new(kirchhoff);
    let eigenvalues = eigen.eigenvalues;
    let eigenvectors = eigen.eigenvectors;

    // Compute RMSF from inverse eigenvalues
    let mut rmsf = vec![0.0; n];
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

    // Skip first trivial mode (zero eigenvalue), optionally limit modes
    let mode_count = if use_mode_limit { 20.min(n - 1) } else { n - 1 };

    for k in 1..=mode_count {
        let idx = sorted_indices[k];
        let lambda = eigenvalues[idx];
        if lambda.abs() < 1e-6 { continue; }

        for i in 0..n {
            let v = eigenvectors[(i, idx)];
            rmsf[i] += v * v / lambda;
        }
    }

    // Normalize
    let max_rmsf = rmsf.iter().cloned().fold(0.0, f64::max);
    if max_rmsf > 1e-10 {
        for r in rmsf.iter_mut() {
            *r = (*r / max_rmsf).sqrt();
        }
    }

    // Apply terminus boosting
    if terminus_residues > 0 && terminus_boost > 1.0 {
        for i in 0..terminus_residues.min(n) {
            rmsf[i] *= terminus_boost;
        }
        for i in (n.saturating_sub(terminus_residues))..n {
            rmsf[i] *= terminus_boost;
        }

        // Re-normalize after boosting
        let max_rmsf = rmsf.iter().cloned().fold(0.0, f64::max);
        if max_rmsf > 1e-10 {
            for r in rmsf.iter_mut() {
                *r /= max_rmsf;
            }
        }
    }

    rmsf
}

fn parse_pdb_ca_chain_with_indices(path: &PathBuf, target_chain: Option<&str>) -> Result<(Vec<[f32; 3]>, Vec<usize>)> {
    let content = fs::read_to_string(path)?;
    let mut positions = Vec::new();
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

        positions.push([x, y, z]);
        kept_indices.push(current_index);
    }

    Ok((positions, kept_indices))
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

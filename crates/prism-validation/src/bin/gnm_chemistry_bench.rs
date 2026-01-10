//! Chemistry-Aware GNM Benchmark
//!
//! Tests the Chemistry-Aware GNM (CA-GNM) on ATLAS AlphaFlow-82 benchmark.
//! Compares against baseline GNM and Enhanced GNM.

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use serde::Deserialize;

// Import GNM implementations
use prism_physics::gnm::GaussianNetworkModel;
use prism_physics::gnm_chemistry::{ChemistryGnm, ChemistryGnmConfig};

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
    println!("║     CHEMISTRY-AWARE GNM BENCHMARK - Targeting ρ ≥ 0.70                    ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Run benchmarks: Plain GNM, Enhanced GNM (distance-weighted), Chemistry-Aware GNM
    // (name, use_dw, use_chemistry, optimized_config)
    let configs = vec![
        ("Plain GNM (baseline)", false, false, false),
        ("Distance-Weighted GNM", true, false, false),
        ("CA-GNM (full w/ salt)", false, true, false),
        ("CA-GNM (optimized)", false, true, true),       // Default config (no salt bridges)
        ("CA-GNM + DW (BEST)", true, true, true),        // Best combination
    ];

    for (name, use_dw, use_chemistry, use_optimized) in configs {
        let start = Instant::now();
        let mut correlations = Vec::new();
        let mut hbond_counts = Vec::new();
        let mut salt_bridge_counts = Vec::new();

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

            if let Ok((ca_pos, residue_names, kept_indices)) = parse_pdb_with_residues(&pdb_path, target_chain) {
                if ca_pos.len() < 10 { continue; }

                // Compute RMSF with the selected method
                let (rmsf, n_hbonds, n_salt) = if use_chemistry {
                    let config = if use_optimized {
                        // Optimized config: AA stiffness + burial only
                        let mut c = ChemistryGnmConfig::default();
                        c.cutoff = 9.0;
                        if !use_dw {
                            c.sigma = 100.0; // Large sigma = minimal distance decay
                        }
                        c
                    } else {
                        // Full experimental config
                        let mut c = ChemistryGnmConfig::full_experimental();
                        c.cutoff = 9.0;
                        if !use_dw {
                            c.sigma = 100.0;
                        }
                        c
                    };

                    let gnm = ChemistryGnm::with_config(config);
                    let res_refs: Vec<&str> = residue_names.iter().map(|s| s.as_str()).collect();
                    let result = gnm.compute_rmsf(&ca_pos, &res_refs);
                    (result.rmsf, result.hbonds.len(), result.salt_bridges.len())
                } else if use_dw {
                    // Distance-weighted GNM (current Enhanced GNM mode)
                    let rmsf = compute_distance_weighted_rmsf(&ca_pos, 9.0, 5.0);
                    (rmsf, 0, 0)
                } else {
                    // Plain GNM
                    let gnm = GaussianNetworkModel::with_cutoff(9.0);
                    let result = gnm.compute_rmsf(&ca_pos);
                    (result.rmsf, 0, 0)
                };

                hbond_counts.push(n_hbonds);
                salt_bridge_counts.push(n_salt);

                // Align with ground truth
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

        let elapsed = start.elapsed();

        if !correlations.is_empty() {
            let mean = correlations.iter().sum::<f64>() / correlations.len() as f64;
            let above_07 = correlations.iter().filter(|&&c| c >= 0.7).count();
            let above_06 = correlations.iter().filter(|&&c| c >= 0.6).count();
            let pass_rate = correlations.iter().filter(|&&c| c > 0.3).count() as f64 / correlations.len() as f64;

            let total_hbonds: usize = hbond_counts.iter().sum();
            let total_salt: usize = salt_bridge_counts.iter().sum();

            let marker = if mean >= 0.70 { "🎯" } else if mean >= 0.65 { "⭐" } else if mean >= 0.62 { "✅" } else { "  " };

            println!("{} {:30} ρ = {:.4}  ≥0.7: {:2}/{}  ≥0.6: {:2}/{}  pass={:.1}%  time={:.2}s",
                     marker, name, mean, above_07, correlations.len(), above_06, correlations.len(),
                     pass_rate * 100.0, elapsed.as_secs_f64());

            if total_hbonds > 0 || total_salt > 0 {
                println!("     └─ H-bonds detected: {}   Salt bridges: {}", total_hbonds, total_salt);
            }
        }
    }

    println!();
    println!("  Legend: 🎯 = ρ≥0.70, ⭐ = ρ≥0.65, ✅ = ρ≥0.62");
    println!();

    // Run ablation study on Chemistry-Aware components
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!("  CA-GNM ABLATION STUDY");
    println!("═══════════════════════════════════════════════════════════════════════════════");

    let ablation_configs = vec![
        ("Full CA-GNM", true, true, true, true, true),
        ("- AA Stiffness", false, true, true, true, true),
        ("- Burial Weighting", true, false, true, true, true),
        ("- H-bond Detection", true, true, false, true, true),
        ("- Salt Bridges", true, true, true, false, true),
        ("- Contact Types", true, true, true, true, false),
        ("AA Stiffness only", true, false, false, false, false),
        ("Burial only", false, true, false, false, false),
        ("H-bonds only", false, false, true, false, false),
    ];

    for (name, aa, burial, hbond, salt, contact) in ablation_configs {
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

            if let Ok((ca_pos, residue_names, kept_indices)) = parse_pdb_with_residues(&pdb_path, target_chain) {
                if ca_pos.len() < 10 { continue; }

                let config = ChemistryGnmConfig {
                    cutoff: 9.0,
                    sigma: 5.0,
                    use_aa_stiffness: aa,
                    use_burial_weighting: burial,
                    use_hbond_detection: hbond,
                    use_salt_bridges: salt,
                    use_contact_types: contact,
                    burial_radius: 10.0,
                };

                let gnm = ChemistryGnm::with_config(config);
                let res_refs: Vec<&str> = residue_names.iter().map(|s| s.as_str()).collect();
                let result = gnm.compute_rmsf(&ca_pos, &res_refs);

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
            let delta = mean - 0.615; // vs Enhanced GNM baseline
            let marker = if delta > 0.005 { "✅" } else if delta < -0.005 { "❌" } else { "  " };

            println!("  {} {:25} ρ = {:.4}  Δ = {:+.4}", marker, name, mean, delta);
        }
    }

    Ok(())
}

/// Compute distance-weighted RMSF (matches Enhanced GNM mode)
fn compute_distance_weighted_rmsf(ca_positions: &[[f32; 3]], cutoff: f64, sigma: f64) -> Vec<f64> {
    use nalgebra::{DMatrix, SymmetricEigen};

    let n = ca_positions.len();
    let cutoff_sq = cutoff * cutoff;
    let sigma_sq = sigma * sigma;

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

    for i in 0..n {
        let row_sum: f64 = kirchhoff.row(i).iter().sum();
        kirchhoff[(i, i)] = -row_sum;
    }

    let eigen = SymmetricEigen::new(kirchhoff);
    let eigenvalues = eigen.eigenvalues;
    let eigenvectors = eigen.eigenvectors;

    let mut rmsf = vec![0.0; n];
    let mut sorted_indices: Vec<usize> = (0..n).collect();
    sorted_indices.sort_by(|&a, &b| eigenvalues[a].partial_cmp(&eigenvalues[b]).unwrap());

    for k in 1..n {
        let idx = sorted_indices[k];
        let lambda = eigenvalues[idx];
        if lambda.abs() < 1e-6 { continue; }

        for i in 0..n {
            let v = eigenvectors[(i, idx)];
            rmsf[i] += v * v / lambda;
        }
    }

    let max_rmsf = rmsf.iter().cloned().fold(0.0, f64::max);
    if max_rmsf > 1e-10 {
        for r in rmsf.iter_mut() {
            *r = (*r / max_rmsf).sqrt();
        }
    }

    rmsf
}

/// Parse PDB file and extract Cα positions with residue names
fn parse_pdb_with_residues(path: &PathBuf, target_chain: Option<&str>) -> Result<(Vec<[f32; 3]>, Vec<String>, Vec<usize>)> {
    let content = fs::read_to_string(path)?;
    let mut positions = Vec::new();
    let mut residue_names = Vec::new();
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

        let res_name = line.get(17..20).unwrap_or("UNK").trim().to_string();

        let x: f32 = line.get(30..38).unwrap_or("0").trim().parse().unwrap_or(0.0);
        let y: f32 = line.get(38..46).unwrap_or("0").trim().parse().unwrap_or(0.0);
        let z: f32 = line.get(46..54).unwrap_or("0").trim().parse().unwrap_or(0.0);

        positions.push([x, y, z]);
        residue_names.push(res_name);
        kept_indices.push(current_index);
    }

    Ok((positions, residue_names, kept_indices))
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

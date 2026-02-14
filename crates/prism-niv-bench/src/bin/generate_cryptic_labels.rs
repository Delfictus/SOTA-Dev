use anyhow::Result;
use prism_niv_bench::structure_types::{NivBenchDataset, CrypticSite};
use prism_niv_bench::pdb_fetcher;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

// Simple SASA approximation based on neighbor count
// Real SASA requires full algorithm, this is a heuristic for bootstrapping
fn is_buried(residue_idx: usize, coords: &[(f32, f32, f32)]) -> bool {
    let center = coords[residue_idx];
    let mut neighbors = 0;
    for (i, &other) in coords.iter().enumerate() {
        if i == residue_idx { continue; }
        let dx = center.0 - other.0;
        let dy = center.1 - other.1;
        let dz = center.2 - other.2;
        let dist_sq = dx*dx + dy*dy + dz*dz;
        if dist_sq < 100.0 { // 10 Angstrom radius
            neighbors += 1;
        }
    }
    neighbors > 20 // High neighbor count = buried
}

fn main() -> Result<()> {
    println!("Generating Silver Standard Cryptic Labels...");

    // 1. Load IEDB Data
    let json_path = "data/niv_bench_dataset.json";
    let file = File::open(json_path)?;
    let reader = BufReader::new(file);
    let mut dataset: NivBenchDataset = serde_json::from_reader(reader)?;

    // 2. Load Structure
    let pdb_path = "data/niv_structures/8XPS.pdb";
    let structure = pdb_fetcher::parse_pdb(pdb_path)?;
    let coords = structure.ca_coords();

    println!("Loaded 8XPS: {} residues", coords.len());

    let mut cryptic_count = 0;
    let mut cryptic_sites = Vec::new();

    // 3. Analyze Epitopes
    if let Some(epitopes) = dataset.epitopes.get("8XPS") {
        for epitope in epitopes {
            let mut buried_residues = Vec::new();
            
            for &seq_num in &epitope.interface_residues {
                // Map sequence number to index (assuming 1-based, contiguous for now)
                // Real mapping needs seq alignment, but 8XPS is clean
                let idx = (seq_num as usize).saturating_sub(1);
                
                if idx < coords.len() {
                    if is_buried(idx, &coords) {
                        buried_residues.push(seq_num);
                    }
                }
            }

            if !buried_residues.is_empty() {
                // This epitope has buried parts! It is partially cryptic.
                cryptic_count += 1;
                cryptic_sites.push(CrypticSite {
                    residues: buried_residues,
                    sasa_apo: vec![0.0], // Placeholder
                    sasa_holo: vec![100.0], // Placeholder
                    p_rmsd: 0.0,
                    description: format!("Cryptic part of {}", epitope.name),
                });
            }
        }
    }

    println!("Found {} potentially cryptic epitopes.", cryptic_count);

    // 4. Save Updates
    dataset.cryptic_sites.insert("8XPS".to_string(), cryptic_sites);
    
    let outfile = File::create(json_path)?;
    let writer = BufWriter::new(outfile);
    serde_json::to_writer_pretty(writer, &dataset)?;

    println!("Updated {} with calculated cryptic sites.", json_path);
    Ok(())
}

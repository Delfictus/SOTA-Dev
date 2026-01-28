//! PDB structure fetching and parsing

use crate::{Result, structure_types::*};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::collections::HashMap;
use anyhow::Context;

/// Download PDB structure from RCSB
pub async fn download_pdb(pdb_id: &str, output_path: &str) -> Result<()> {
    // TODO: Implement PDB downloading if needed here, but main.rs handles it for now
    todo!("PDB downloading implementation")
}

/// Parse PDB file into ParamyxoStructure
pub fn parse_pdb(pdb_path: &str) -> Result<ParamyxoStructure> {
    let file = File::open(pdb_path).with_context(|| format!("Failed to open PDB file: {}", pdb_path))?;
    let reader = BufReader::new(file);

    let file_stem = std::path::Path::new(pdb_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("UNKNOWN")
        .to_uppercase();

    let mut atoms = Vec::new();
    let mut residues_map: HashMap<(String, u32), Residue> = HashMap::new();
    let mut chains_set = HashMap::new();
    let mut resolution = None;
    let mut experimental_method = None;

    for line in reader.lines() {
        let line = line?;
        if line.starts_with("REMARK   2 RESOLUTION.") {
             // Basic resolution parsing "REMARK   2 RESOLUTION.    2.35 ANGSTROMS."
             if let Some(res_str) = line.split_whitespace().nth(3) {
                 resolution = res_str.parse::<f32>().ok();
             }
        } else if line.starts_with("EXPDTA") {
            experimental_method = Some(line[10..].trim().to_string());
        } else if line.starts_with("ATOM  ") || line.starts_with("HETATM") {
            // Standard PDB column format
            // 1-6 Record name
            // 7-11 Serial number
            // 13-16 Atom name
            // 17 Alternate location
            // 18-20 Residue name
            // 22 Chain ID
            // 23-26 Residue sequence number
            // 31-38 X
            // 39-46 Y
            // 47-54 Z
            
            if line.len() < 54 { continue; }
            
            let serial = line[6..11].trim().parse().unwrap_or(0);
            let name = line[12..16].trim().to_string();
            let res_name = line[17..20].trim().to_string();
            let chain_id = line[21..22].to_string();
            let res_seq = line[22..26].trim().parse().unwrap_or(0);
            let x = line[30..38].trim().parse().unwrap_or(0.0);
            let y = line[38..46].trim().parse().unwrap_or(0.0);
            let z = line[46..54].trim().parse().unwrap_or(0.0);
            
            let atom = Atom {
                id: serial,
                name: name.clone(),
                element: name.chars().next().unwrap_or('X').to_string(), // Simplified element inference
                x, y, z,
                residue_id: res_seq,
                chain_id: chain_id.clone(),
            };
            
            atoms.push(atom.clone());
            
            let key = (chain_id.clone(), res_seq);
            let residue = residues_map.entry(key).or_insert_with(|| Residue {
                id: res_seq,
                name: res_name.clone(),
                chain_id: chain_id.clone(),
                sequence_number: res_seq,
                ca_coords: (0.0, 0.0, 0.0), // Updated later
                atoms: Vec::new(),
            });
            
            residue.atoms.push(atom);
            if name == "CA" {
                residue.ca_coords = (x, y, z);
            }
            
            chains_set.insert(chain_id.clone(), (res_seq, res_seq)); // Will update min/max
        }
    }
    
    // Sort residues
    let mut residues: Vec<Residue> = residues_map.into_values().collect();
    residues.sort_by(|a, b| {
        a.chain_id.cmp(&b.chain_id).then(a.sequence_number.cmp(&b.sequence_number))
    });
    
    // Build chain info
    let chains: Vec<ChainInfo> = chains_set.into_iter().map(|(id, (start, end))| {
        // Need to calculate actual start/end from residues
        let chain_residues: Vec<&Residue> = residues.iter().filter(|r| r.chain_id == id).collect();
        let min_res = chain_residues.iter().map(|r| r.sequence_number).min().unwrap_or(start);
        let max_res = chain_residues.iter().map(|r| r.sequence_number).max().unwrap_or(end);
        
        ChainInfo {
            id,
            protein_type: ProteinType::GProtein, // Default for 8XPS context
            start_residue: min_res,
            end_residue: max_res,
        }
    }).collect();

    Ok(ParamyxoStructure {
        pdb_id: file_stem,
        virus: VirusType::Nipah, // Default
        protein: ProteinType::GProtein, // Default
        chains,
        atoms,
        residues,
        sequence: String::new(), // Not parsing SEQRES for now
        resolution,
        experimental_method,
    })
}

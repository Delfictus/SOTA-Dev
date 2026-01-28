//! Ground truth data extraction and validation

use crate::structure_types::*;
use crate::glycan_dynamics::calculate_effective_accessibility;
use anyhow::{Result, Context};
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;

/// IEDB epitope record structure
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct IedbEpitope {
    pub structure_id: u32,
    #[serde(default)]
    pub linear_sequence: Option<String>,
    #[serde(default)]
    pub linear_sequence_length: Option<usize>,
    pub curated_source_antigens: Vec<CuratedSourceAntigen>,
    #[serde(default)]
    pub qualitative_measures: Vec<String>,
    #[serde(default)]
    pub mhc_classes: Option<Vec<String>>,
    #[serde(default)]
    pub assay_names: Vec<String>,
    #[serde(default)]
    pub source_organism_names: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct CuratedSourceAntigen {
    #[serde(default)]
    pub accession: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub starting_position: Option<u32>,
    #[serde(default)]
    pub ending_position: Option<u32>,
    #[serde(default)]
    pub source_organism_name: Option<String>,
}

/// Load and parse IEDB epitope data from JSON file
pub fn load_iedb_epitopes(path: &str) -> Result<Vec<IedbEpitope>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open IEDB data file: {}", path))?;
    let reader = BufReader::new(file);
    let epitopes: Vec<IedbEpitope> = serde_json::from_reader(reader)
        .map_err(|e| anyhow::anyhow!("Failed to parse IEDB JSON: {} - check field types match structure", e))?;

    println!("Loaded {} IEDB epitope records", epitopes.len());
    Ok(epitopes)
}

/// Extract ground truth epitope data from PDB structures
pub fn extract_ground_truth(structures: &[ParamyxoStructure]) -> Result<NivBenchDataset> {
    let mut cryptic_sites = HashMap::new();
    let known_escape_mutations = Vec::new();

    // Load IEDB experimental epitope data
    let iedb_path = "data/iedb_nipah_raw.json";
    let iedb_epitopes = load_iedb_epitopes(iedb_path)?;

    println!("\nMapping IEDB epitopes to PDB structures...");
    let epitopes = map_iedb_to_structures(structures, &iedb_epitopes)?;

    println!("Mapped {} epitope definitions to structures", epitopes.len());

    // Cryptic site extraction from apo/holo pairs
    let apo_opt = structures.iter().find(|s| s.pdb_id == "8XPS");
    let holo_opt = structures.iter().find(|s| s.pdb_id == "8XQ3");

    if let (Some(apo), Some(holo)) = (apo_opt, holo_opt) {
        println!("\nAnalyzing cryptic sites between {} (Apo) and {} (Holo)...", apo.pdb_id, holo.pdb_id);
        let accessibility = calculate_effective_accessibility(apo)?;
        let sites = define_cryptic_residues(apo, holo, &accessibility)?;
        cryptic_sites.insert("NiV_G_Dimeric_Interface".to_string(), sites);
    }

    // Split structures
    let (train, test, val) = create_splits();

    Ok(NivBenchDataset {
        structures: structures.to_vec(),
        epitopes,
        cryptic_sites,
        known_escape_mutations,
        train_structures: train,
        test_structures: test,
        validation_structures: val,
    })
}

/// Map IEDB epitopes to PDB structure residues
fn map_iedb_to_structures(
    structures: &[ParamyxoStructure],
    iedb_epitopes: &[IedbEpitope],
) -> Result<HashMap<String, Vec<EpitopeDefinition>>> {
    let mut epitope_map: HashMap<String, Vec<EpitopeDefinition>> = HashMap::new();

    for structure in structures {
        let mut structure_epitopes = Vec::new();

        // Match IEDB epitopes to this structure based on protein type
        for iedb_ep in iedb_epitopes {
            if let Some(antigen) = iedb_ep.curated_source_antigens.first() {
                // Check if this epitope matches the structure's protein
                let protein_name = match &antigen.name {
                    Some(n) => n.as_str(),
                    None => continue,
                };

                let matches = match (&structure.virus, &structure.protein) {
                    (VirusType::Nipah, ProteinType::GProtein) => {
                        protein_name.contains("Glycoprotein G") ||
                        protein_name.contains("attachment glycoprotein")
                    },
                    (VirusType::Nipah, ProteinType::FProtein) => {
                        protein_name.contains("Fusion") ||
                        protein_name.contains("glycoprotein F")
                    },
                    (VirusType::Hendra, ProteinType::GProtein) => {
                        protein_name.contains("Glycoprotein G")
                    },
                    _ => false,
                };

                if matches {
                    // Skip if positions are null
                    let start = match antigen.starting_position {
                        Some(s) => s as usize,
                        None => continue,
                    };
                    let end = match antigen.ending_position {
                        Some(e) => e as usize,
                        None => continue,
                    };

                    if start > 0 && end <= structure.residues.len() {
                        // Extract PDB sequence for this range (convert to 0-indexed)
                        let pdb_seq: String = structure.residues[(start-1)..end]
                            .iter()
                            .map(|r| {
                                match r.name.as_str() {
                                    "ALA" => 'A', "CYS" => 'C', "ASP" => 'D', "GLU" => 'E',
                                    "PHE" => 'F', "GLY" => 'G', "HIS" => 'H', "ILE" => 'I',
                                    "LYS" => 'K', "LEU" => 'L', "MET" => 'M', "ASN" => 'N',
                                    "PRO" => 'P', "GLN" => 'Q', "ARG" => 'R', "SER" => 'S',
                                    "THR" => 'T', "VAL" => 'V', "TRP" => 'W', "TYR" => 'Y',
                                    _ => 'X',
                                }
                            })
                            .collect();

                        // Validate sequence matches IEDB (skip if linear_sequence is null)
                        let iedb_sequence = match &iedb_ep.linear_sequence {
                            Some(seq) => seq.as_str(),
                            None => continue,
                        };

                        if pdb_seq == iedb_sequence {
                            let is_positive = iedb_ep.qualitative_measures.iter()
                                .any(|m| m.contains("Positive"));

                            structure_epitopes.push(EpitopeDefinition {
                                name: format!("IEDB:{}", iedb_ep.structure_id),
                                antibody_pdb: "experimental".to_string(),
                                interface_residues: (start as u32..=end as u32).collect(),
                                is_cryptic: false, // T-cell epitopes not typically cryptic
                                is_neutralizing: is_positive,
                                cross_reactive: iedb_ep.source_organism_names.len() > 1,
                                ic50_niv: None, // Not available in IEDB T-cell data
                                ic50_hev: None,
                            });
                        }
                    }
                }
            }
        }

        if !structure_epitopes.is_empty() {
            println!("  {} epitopes mapped to {}", structure_epitopes.len(), structure.pdb_id);
            epitope_map.insert(structure.pdb_id.clone(), structure_epitopes);
        }
    }

    Ok(epitope_map)
}

/// Determine epitope class from assay type
fn determine_epitope_class(assay_names: &[String]) -> String {
    if assay_names.iter().any(|a| a.contains("MHC class I")) {
        "T-cell (MHC-I)".to_string()
    } else if assay_names.iter().any(|a| a.contains("MHC class II")) {
        "T-cell (MHC-II)".to_string()
    } else if assay_names.iter().any(|a| a.contains("B cell")) {
        "B-cell".to_string()
    } else {
        "T-cell".to_string() // Default from ELISPOT assays
    }
}

/// Define cryptic residues based on accessibility changes and pRMSD
pub fn define_cryptic_residues(
    apo: &ParamyxoStructure, 
    holo: &ParamyxoStructure, 
    effective_accessibility: &[f32]
) -> Result<Vec<CrypticSite>> {
    let mut sites = Vec::new();
    let mut current_site_residues = Vec::new();
    let mut current_sasa_apo = Vec::new();
    let mut current_sasa_holo = Vec::new();
    let mut total_prmsd = 0.0;
    
    // Compute Holo SASA (approximate)
    let holo_sasa = crate::glycan_dynamics::calculate_effective_accessibility(holo)?;

    // Align sequences/residues (Assume 1-to-1 mapping by sequence number for now)
    // This is brittle if numbering differs, but standard for same protein different PDBs
    
    for (i, res_apo) in apo.residues.iter().enumerate() {
        // Find corresponding residue in holo
        if let Some(res_holo) = holo.get_residue(res_apo.sequence_number) {
            let idx_holo = holo.residues.iter().position(|r| r.sequence_number == res_apo.sequence_number).unwrap_or(0);
            
            let acc_apo = effective_accessibility[i];
            let acc_holo = holo_sasa[idx_holo]; // Use effective here too, or raw? Prompt says SASA_holo > 30.
            
            // Cryptic definition: Hidden in Apo (< 10 A^2), Exposed in Holo (> 30 A^2)
            // Note: Usually cryptic means exposed in Apo but hidden in Holo (interface), OR hidden in Apo (closed) and exposed in Holo (open).
            // Prompt says: "Cryptic = effective_accessibility < 10 Å² AND SASA_holo > 30 Å²" -> Hidden in Apo, Exposed in Holo.
            
            let is_cryptic = acc_apo < 10.0 && acc_holo > 30.0;
            
            if is_cryptic {
                // Calculate pRMSD for this residue
                let dx = res_apo.ca_coords.0 - res_holo.ca_coords.0;
                let dy = res_apo.ca_coords.1 - res_holo.ca_coords.1;
                let dz = res_apo.ca_coords.2 - res_holo.ca_coords.2;
                let dist = (dx*dx + dy*dy + dz*dz).sqrt();

                current_site_residues.push(res_apo.sequence_number);
                current_sasa_apo.push(acc_apo);
                current_sasa_holo.push(acc_holo);
                total_prmsd += dist;
            } else if !current_site_residues.is_empty() {
                // End of a contiguous segment? 
                // Grouping by spatial proximity is better, but sequence grouping is a start.
                // For now, we just collect all individual residues or small patches.
                // Let's just collect them all into one "Site" for the interface? 
                // Or make single-residue sites?
                // Structure definition has `residues: Vec<u32>`.
            }
        }
    }

    if !current_site_residues.is_empty() {
         sites.push(CrypticSite {
            residues: current_site_residues.clone(),
            sasa_apo: current_sasa_apo,
            sasa_holo: current_sasa_holo,
            p_rmsd: total_prmsd / current_site_residues.len() as f32,
            description: "Predicted cryptic site based on Apo/Holo comparison".to_string(),
        });
    }

    Ok(sites)
}

fn create_splits() -> (Vec<String>, Vec<String>, Vec<String>) {
    let train = vec![
        "8XPS".to_string(), "8XQ3".to_string(), "7TY0".to_string(), // NiV G
        "2X9M".to_string(), "6CMG".to_string()  // HeV G
    ];
    
    let test = vec![
        "7UPK".to_string(), "7UPD".to_string(), "7UPB".to_string(), // NiV F
        "7UPH".to_string() // Cross-reactive
    ];
    
    let val = vec![]; // Use subset of train or specific validation set
    
    (train, test, val)
}

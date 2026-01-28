use anyhow::{Result, Context};
use std::fs::{self, File};
use std::io::{Read, Write, BufReader, BufRead};
use std::path::Path;
use crate::structure_types::{ParamyxoStructure, VirusType, ProteinType, ChainInfo, Atom, Residue};

pub struct DataLoader {
    pub data_dir: String,
}

impl DataLoader {
    pub fn new(data_dir: &str) -> Self {
        Self {
            data_dir: data_dir.to_string(),
        }
    }

    /// Download all required NiV/HeV PDB structures
    pub async fn download_paramyxo_structures(&self) -> Result<()> {
        let pdb_ids = [
            // NiV
            "8XPS", "8XQ3", "7UPK", "7UPD", "7UPB", "8ZPV", "7SKT", "7TY0",
            // HeV
            "2X9M", "5EJB", "6CMG", "7UPH"
        ];

        fs::create_dir_all(&self.data_dir)?;

        for pdb_id in pdb_ids {
            let file_path = Path::new(&self.data_dir).join(format!("{}.pdb", pdb_id));
            if !file_path.exists() {
                println!("Downloading {}...", pdb_id);
                let url = format!("https://files.rcsb.org/download/{}.pdb", pdb_id);
                let response = reqwest::get(&url).await?;
                if !response.status().is_success() {
                    eprintln!("Failed to download {}: {}", pdb_id, response.status());
                    continue;
                }
                let content = response.text().await?;
                let mut file = File::create(&file_path)?;
                file.write_all(content.as_bytes())?;
            } else {
                println!("{} already exists.", pdb_id);
            }
        }
        Ok(())
    }

    /// Simple PDB parser (can be replaced with a more robust library like pdbtbx if needed)
    pub fn parse_pdb(&self, pdb_id: &str) -> Result<ParamyxoStructure> {
        let file_path = Path::new(&self.data_dir).join(format!("{}.pdb", pdb_id));
        let file = File::open(&file_path).context(format!("Failed to open PDB file: {:?}", file_path))?;
        let reader = BufReader::new(file);

        let mut atoms = Vec::new();
        let mut residues = Vec::new();
        let mut current_residue_num = 0;
        let mut current_residue_atoms = Vec::new();
        let mut current_residue_name = String::new();
        let mut current_chain_id = String::new();

        // Determine virus and protein type based on ID (simplified logic)
        let (virus, protein) = match pdb_id {
            "8XPS" | "8XQ3" | "7TY0" => (VirusType::Nipah, ProteinType::GProtein),
            "7UPK" | "7UPD" | "7UPB" => (VirusType::Nipah, ProteinType::FProtein),
            "8ZPV" => (VirusType::Nipah, ProteinType::Polymerase),
            "7SKT" => (VirusType::Nipah, ProteinType::MatrixProtein),
            "2X9M" | "6CMG" => (VirusType::Hendra, ProteinType::GProtein),
            "5EJB" => (VirusType::Hendra, ProteinType::FProtein),
            "7UPH" => (VirusType::Hendra, ProteinType::FProtein), // Cross-reactive complex
            _ => (VirusType::Nipah, ProteinType::GProtein), // Default
        };

        for line in reader.lines() {
            let line = line?;
            if line.starts_with("ATOM") {
                // Fixed width parsing based on PDB format
                // ATOM      1  N   ASP A   1      12.922  24.331  10.276  1.00 12.00           N
                let serial: u32 = line[6..11].trim().parse().unwrap_or(0);
                let name = line[12..16].trim().to_string();
                let res_name = line[17..20].trim().to_string();
                let chain = line[21..22].trim().to_string();
                let res_seq: u32 = line[22..26].trim().parse().unwrap_or(0);
                let x: f32 = line[30..38].trim().parse().unwrap_or(0.0);
                let y: f32 = line[38..46].trim().parse().unwrap_or(0.0);
                let z: f32 = line[46..54].trim().parse().unwrap_or(0.0);
                let element = line[76..78].trim().to_string();

                let atom = Atom {
                    id: serial,
                    name: name.clone(),
                    element,
                    x,
                    y,
                    z,
                    residue_id: res_seq,
                    chain_id: chain.clone(),
                };
                atoms.push(atom.clone());

                if res_seq != current_residue_num && current_residue_num != 0 {
                     // Finish previous residue
                     // Need CA coords
                     let ca = current_residue_atoms.iter().find(|a: &&Atom| a.name == "CA");
                     let ca_coords = if let Some(a) = ca {
                         (a.x, a.y, a.z)
                     } else {
                         // Fallback to first atom if CA missing (unlikely for standard residues)
                         (current_residue_atoms[0].x, current_residue_atoms[0].y, current_residue_atoms[0].z)
                     };

                     residues.push(Residue {
                         id: current_residue_num,
                         name: current_residue_name.clone(),
                         chain_id: current_chain_id.clone(),
                         sequence_number: current_residue_num,
                         ca_coords,
                         atoms: current_residue_atoms.clone(),
                     });
                     current_residue_atoms.clear();
                }

                current_residue_num = res_seq;
                current_residue_name = res_name;
                current_chain_id = chain;
                current_residue_atoms.push(atom);
            }
        }
        // Push last residue
        if !current_residue_atoms.is_empty() {
             let ca = current_residue_atoms.iter().find(|a: &&Atom| a.name == "CA");
             let ca_coords = if let Some(a) = ca {
                 (a.x, a.y, a.z)
             } else {
                 (current_residue_atoms[0].x, current_residue_atoms[0].y, current_residue_atoms[0].z)
             };

             residues.push(Residue {
                 id: current_residue_num,
                 name: current_residue_name.clone(),
                 chain_id: current_chain_id.clone(),
                 sequence_number: current_residue_num,
                 ca_coords,
                 atoms: current_residue_atoms,
             });
        }

        Ok(ParamyxoStructure {
            pdb_id: pdb_id.to_string(),
            virus,
            protein,
            chains: vec![], // TODO: Populate chains
            atoms,
            residues,
            sequence: "".to_string(), // Optional
            resolution: None,
            experimental_method: None,
        })
    }

    /// Parse antibody interface residues (< 5Å from antibody chain)
    pub fn parse_antibody_interface(&self, pdb_id: &str, antibody_chain: &str, antigen_chain: &str) -> Result<Vec<u32>> {
        let structure = self.parse_pdb(pdb_id)?;
        let mut interface_residues = Vec::new();

        let ag_atoms: Vec<&Atom> = structure.atoms.iter()
            .filter(|a| a.chain_id == antigen_chain)
            .collect();
        
        let ab_atoms: Vec<&Atom> = structure.atoms.iter()
            .filter(|a| a.chain_id == antibody_chain)
            .collect();

        for res in structure.residues.iter().filter(|r| r.chain_id == antigen_chain) {
            let mut is_interface = false;
            for atom_ag in &res.atoms {
                for atom_ab in &ab_atoms {
                    let dx = atom_ag.x - atom_ab.x;
                    let dy = atom_ag.y - atom_ab.y;
                    let dz = atom_ag.z - atom_ab.z;
                    let dist_sq = dx*dx + dy*dy + dz*dz;
                    if dist_sq < 25.0 { // 5.0 * 5.0
                        is_interface = true;
                        break;
                    }
                }
                if is_interface { break; }
            }
            if is_interface {
                interface_residues.push(res.sequence_number);
            }
        }
        interface_residues.sort();
        interface_residues.dedup();
        Ok(interface_residues)
    }
}

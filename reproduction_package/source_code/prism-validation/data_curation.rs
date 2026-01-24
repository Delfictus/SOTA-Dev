//! Data Curation with Cryptographic Provenance
//!
//! Ensures scientific integrity through:
//! - BLAKE3 hashing of all source files
//! - Full atomic metadata extraction and mapping
//! - Temporal validation (no data leakage)
//! - Immutable provenance chain
//!
//! ## Data Leakage Prevention
//!
//! For retrospective blind validation, we MUST ensure:
//! 1. Apo structures deposited BEFORE drug discovery
//! 2. No holo information used during simulation
//! 3. Drug binding site NOT encoded in input
//!
//! ## Provenance Chain
//!
//! Each structure gets a provenance record:
//! - Source URL (RCSB PDB)
//! - Download timestamp
//! - BLAKE3 hash of raw file
//! - Deposition date from PDB header
//! - All atomic metadata mapped

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use chrono::{DateTime, Utc, NaiveDate};

/// BLAKE3 hash (32 bytes, hex-encoded = 64 chars)
pub type Blake3Hash = String;

/// Provenance record for a single PDB file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PdbProvenance {
    /// PDB ID (e.g., "1M47")
    pub pdb_id: String,

    /// Source URL
    pub source_url: String,

    /// Download timestamp (UTC)
    pub downloaded_at: DateTime<Utc>,

    /// BLAKE3 hash of raw PDB file
    pub blake3_hash: Blake3Hash,

    /// File size in bytes
    pub file_size: u64,

    /// PDB deposition date (from header)
    pub deposition_date: Option<NaiveDate>,

    /// PDB release date
    pub release_date: Option<NaiveDate>,

    /// Resolution in Angstroms (if X-ray)
    pub resolution: Option<f32>,

    /// Experimental method
    pub method: Option<String>,

    /// Title from PDB header
    pub title: Option<String>,

    /// Number of atoms
    pub n_atoms: usize,

    /// Number of residues
    pub n_residues: usize,

    /// Chain IDs
    pub chains: Vec<String>,

    /// Local file path
    pub local_path: PathBuf,

    /// Validation status
    pub validation: ProvenanceValidation,
}

/// Validation status for provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceValidation {
    /// Is temporal constraint satisfied? (deposited before drug)
    pub temporal_valid: bool,

    /// Drug discovery date (for comparison)
    pub drug_discovery_date: Option<NaiveDate>,

    /// Days between deposition and drug discovery
    pub days_before_drug: Option<i64>,

    /// Any warnings
    pub warnings: Vec<String>,

    /// Is this file safe to use for blind validation?
    pub safe_for_blind: bool,
}

/// Full atomic metadata for a structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicMetadata {
    /// PDB ID
    pub pdb_id: String,

    /// BLAKE3 hash (links to provenance)
    pub blake3_hash: Blake3Hash,

    /// Per-atom metadata
    pub atoms: Vec<AtomRecord>,

    /// Per-residue metadata
    pub residues: Vec<ResidueRecord>,

    /// Per-chain metadata
    pub chains: Vec<ChainRecord>,

    /// Connectivity (bonds)
    pub bonds: Vec<BondRecord>,

    /// Secondary structure assignments
    pub secondary_structure: Vec<SecondaryStructureRecord>,

    /// Crystallographic metadata (if applicable)
    pub crystal: Option<CrystalMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomRecord {
    /// Atom serial number
    pub serial: u32,

    /// Atom name (e.g., "CA", "N", "O")
    pub name: String,

    /// Alternate location indicator
    pub alt_loc: Option<char>,

    /// Residue name (e.g., "ALA", "GLY")
    pub res_name: String,

    /// Chain ID
    pub chain_id: String,

    /// Residue sequence number
    pub res_seq: i32,

    /// Insertion code
    pub i_code: Option<char>,

    /// X coordinate (Angstroms)
    pub x: f32,

    /// Y coordinate
    pub y: f32,

    /// Z coordinate
    pub z: f32,

    /// Occupancy
    pub occupancy: f32,

    /// B-factor (temperature factor)
    pub b_factor: f32,

    /// Element symbol
    pub element: String,

    /// Formal charge
    pub charge: Option<i8>,

    /// Is this a HETATM?
    pub is_hetatm: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidueRecord {
    /// Chain ID
    pub chain_id: String,

    /// Residue sequence number
    pub res_seq: i32,

    /// Residue name
    pub res_name: String,

    /// Insertion code
    pub i_code: Option<char>,

    /// Atom indices belonging to this residue
    pub atom_indices: Vec<usize>,

    /// CA atom index (for proteins)
    pub ca_index: Option<usize>,

    /// Is this a standard amino acid?
    pub is_standard_aa: bool,

    /// Secondary structure type (H=helix, E=sheet, C=coil)
    pub ss_type: Option<char>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainRecord {
    /// Chain ID
    pub chain_id: String,

    /// Residue indices in this chain
    pub residue_indices: Vec<usize>,

    /// Sequence (one-letter codes)
    pub sequence: String,

    /// Is this a protein chain?
    pub is_protein: bool,

    /// Number of residues
    pub n_residues: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BondRecord {
    /// First atom index
    pub atom1: usize,

    /// Second atom index
    pub atom2: usize,

    /// Bond order (1=single, 2=double, etc.)
    pub order: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecondaryStructureRecord {
    /// Type (HELIX or SHEET)
    pub ss_type: String,

    /// Start residue chain
    pub start_chain: String,

    /// Start residue number
    pub start_res: i32,

    /// End residue chain
    pub end_chain: String,

    /// End residue number
    pub end_res: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrystalMetadata {
    /// Space group
    pub space_group: String,

    /// Unit cell dimensions (a, b, c in Angstroms)
    pub cell_dimensions: [f32; 3],

    /// Unit cell angles (alpha, beta, gamma in degrees)
    pub cell_angles: [f32; 3],

    /// Z value
    pub z_value: Option<u32>,
}

/// Validation target with full provenance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuratedTarget {
    /// Target name
    pub name: String,

    /// Therapeutic area
    pub therapeutic_area: String,

    /// Drug name
    pub drug_name: String,

    /// Drug approval/discovery date
    pub drug_date: NaiveDate,

    /// Apo structure provenance
    pub apo_provenance: PdbProvenance,

    /// Apo atomic metadata
    pub apo_metadata: AtomicMetadata,

    /// Holo structure provenance (ground truth - NOT used in simulation)
    pub holo_provenance: PdbProvenance,

    /// Holo atomic metadata
    pub holo_metadata: AtomicMetadata,

    /// Pocket residue indices (in apo structure)
    pub pocket_residues: Vec<i32>,

    /// Is this target valid for blind validation?
    pub valid_for_blind: bool,

    /// Validation notes
    pub notes: Vec<String>,
}

/// Manifest of all curated data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurationManifest {
    /// Version
    pub version: String,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// BLAKE3 hash of this manifest (excluding this field)
    pub manifest_hash: Option<Blake3Hash>,

    /// All curated targets
    pub targets: Vec<CuratedTarget>,

    /// Global statistics
    pub stats: CurationStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurationStats {
    pub total_targets: usize,
    pub valid_for_blind: usize,
    pub total_atoms: usize,
    pub total_residues: usize,
    pub therapeutic_areas: HashMap<String, usize>,
}

/// Data curator for validation targets
pub struct DataCurator {
    /// Output directory
    output_dir: PathBuf,

    /// Downloaded files
    downloaded: HashMap<String, PdbProvenance>,
}

impl DataCurator {
    pub fn new(output_dir: PathBuf) -> Self {
        Self {
            output_dir,
            downloaded: HashMap::new(),
        }
    }

    /// Compute BLAKE3 hash of file contents
    pub fn compute_blake3(data: &[u8]) -> Blake3Hash {
        let hash = blake3::hash(data);
        hash.to_hex().to_string()
    }

    /// Download PDB file from RCSB
    pub async fn download_pdb(&mut self, pdb_id: &str) -> anyhow::Result<PdbProvenance> {
        let pdb_id = pdb_id.to_uppercase();
        let url = format!("https://files.rcsb.org/download/{}.pdb", pdb_id);

        log::info!("Downloading {} from {}", pdb_id, url);

        // Download
        let response = reqwest::get(&url).await?;
        let content = response.bytes().await?;

        // Compute BLAKE3 hash
        let blake3_hash = Self::compute_blake3(&content);

        // Save to file
        let local_path = self.output_dir.join(format!("{}.pdb", pdb_id));
        std::fs::write(&local_path, &content)?;

        // Parse metadata
        let content_str = String::from_utf8_lossy(&content);
        let (deposition_date, release_date, resolution, method, title) =
            Self::parse_pdb_header(&content_str);
        let (n_atoms, n_residues, chains) = Self::count_atoms_residues(&content_str);

        let provenance = PdbProvenance {
            pdb_id: pdb_id.clone(),
            source_url: url,
            downloaded_at: Utc::now(),
            blake3_hash,
            file_size: content.len() as u64,
            deposition_date,
            release_date,
            resolution,
            method,
            title,
            n_atoms,
            n_residues,
            chains,
            local_path,
            validation: ProvenanceValidation {
                temporal_valid: false, // Will be set later
                drug_discovery_date: None,
                days_before_drug: None,
                warnings: Vec::new(),
                safe_for_blind: false,
            },
        };

        self.downloaded.insert(pdb_id, provenance.clone());

        Ok(provenance)
    }

    /// Parse PDB header for metadata
    fn parse_pdb_header(content: &str) -> (Option<NaiveDate>, Option<NaiveDate>, Option<f32>, Option<String>, Option<String>) {
        let mut deposition_date = None;
        let mut release_date = None;
        let mut resolution = None;
        let mut method = None;
        let mut title: Option<String> = None;

        for line in content.lines() {
            if line.starts_with("HEADER") && line.len() >= 66 {
                // Deposition date at columns 51-59 (DD-MMM-YY)
                let date_str = line.get(50..59).map(|s| s.trim());
                if let Some(ds) = date_str {
                    deposition_date = Self::parse_pdb_date(ds);
                }
            }

            if line.starts_with("REVDAT   1") && line.len() >= 22 {
                // Release date
                let date_str = line.get(13..22).map(|s| s.trim());
                if let Some(ds) = date_str {
                    release_date = Self::parse_pdb_date(ds);
                }
            }

            if line.starts_with("REMARK   2 RESOLUTION.") {
                // Resolution
                if let Some(res_str) = line.get(23..30) {
                    resolution = res_str.trim().parse().ok();
                }
            }

            if line.starts_with("EXPDTA") {
                method = Some(line.get(6..).unwrap_or("").trim().to_string());
            }

            if line.starts_with("TITLE") {
                let title_part = line.get(10..).unwrap_or("").trim();
                if let Some(ref mut t) = title {
                    t.push(' ');
                    t.push_str(title_part);
                } else {
                    title = Some(title_part.to_string());
                }
            }
        }

        (deposition_date, release_date, resolution, method, title)
    }

    /// Parse PDB date format (DD-MMM-YY or DD-MMM-YYYY)
    fn parse_pdb_date(s: &str) -> Option<NaiveDate> {
        let months = [
            ("JAN", 1), ("FEB", 2), ("MAR", 3), ("APR", 4),
            ("MAY", 5), ("JUN", 6), ("JUL", 7), ("AUG", 8),
            ("SEP", 9), ("OCT", 10), ("NOV", 11), ("DEC", 12),
        ];

        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 3 {
            return None;
        }

        let day: u32 = parts[0].parse().ok()?;
        let month = months.iter()
            .find(|(m, _)| *m == parts[1].to_uppercase())
            .map(|(_, n)| *n)?;
        let year: i32 = parts[2].parse().ok()?;

        // Handle 2-digit years
        let year = if year < 100 {
            if year > 50 { 1900 + year } else { 2000 + year }
        } else {
            year
        };

        NaiveDate::from_ymd_opt(year, month, day)
    }

    /// Count atoms and residues in PDB
    fn count_atoms_residues(content: &str) -> (usize, usize, Vec<String>) {
        let mut n_atoms = 0;
        let mut residues = std::collections::HashSet::new();
        let mut chains = std::collections::HashSet::new();

        for line in content.lines() {
            if line.starts_with("ATOM") || line.starts_with("HETATM") {
                n_atoms += 1;

                if line.len() >= 27 {
                    let chain = line.get(21..22).unwrap_or(" ").to_string();
                    let res_seq = line.get(22..26).unwrap_or("").trim();
                    let i_code = line.get(26..27).unwrap_or(" ");

                    chains.insert(chain.clone());
                    residues.insert(format!("{}:{}:{}", chain, res_seq, i_code));
                }
            }
        }

        let mut chain_vec: Vec<String> = chains.into_iter().collect();
        chain_vec.sort();

        (n_atoms, residues.len(), chain_vec)
    }

    /// Extract full atomic metadata from PDB
    pub fn extract_atomic_metadata(content: &str, pdb_id: &str, blake3_hash: &str) -> AtomicMetadata {
        let mut atoms = Vec::new();
        let mut residue_map: HashMap<String, Vec<usize>> = HashMap::new();
        let mut ss_records = Vec::new();

        for line in content.lines() {
            // Parse ATOM/HETATM records
            if (line.starts_with("ATOM") || line.starts_with("HETATM")) && line.len() >= 54 {
                let is_hetatm = line.starts_with("HETATM");

                let serial: u32 = line.get(6..11)
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
                let name = line.get(12..16).unwrap_or("").trim().to_string();
                let alt_loc = line.chars().nth(16).filter(|&c| c != ' ');
                let res_name = line.get(17..20).unwrap_or("").trim().to_string();
                let chain_id = line.get(21..22).unwrap_or(" ").to_string();
                let res_seq: i32 = line.get(22..26)
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
                let i_code = line.chars().nth(26).filter(|&c| c != ' ');

                let x: f32 = line.get(30..38)
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0.0);
                let y: f32 = line.get(38..46)
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0.0);
                let z: f32 = line.get(46..54)
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0.0);

                let occupancy: f32 = line.get(54..60)
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(1.0);
                let b_factor: f32 = line.get(60..66)
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0.0);

                let element = line.get(76..78)
                    .map(|s| s.trim().to_string())
                    .unwrap_or_else(|| name.chars().next().map(|c| c.to_string()).unwrap_or_default());

                let charge: Option<i8> = line.get(78..80)
                    .and_then(|s| {
                        let s = s.trim();
                        if s.is_empty() { return None; }
                        // Handle formats like "2+" or "+2"
                        let (num, sign) = if s.ends_with('+') || s.ends_with('-') {
                            (&s[..s.len()-1], s.chars().last())
                        } else if s.starts_with('+') || s.starts_with('-') {
                            (&s[1..], s.chars().next())
                        } else {
                            return None;
                        };
                        let n: i8 = num.parse().unwrap_or(1);
                        match sign {
                            Some('+') => Some(n),
                            Some('-') => Some(-n),
                            _ => None,
                        }
                    });

                let atom_idx = atoms.len();

                // Track residue membership
                let res_key = format!("{}:{}:{}", chain_id, res_seq, i_code.unwrap_or(' '));
                residue_map.entry(res_key).or_default().push(atom_idx);

                atoms.push(AtomRecord {
                    serial,
                    name,
                    alt_loc,
                    res_name,
                    chain_id,
                    res_seq,
                    i_code,
                    x,
                    y,
                    z,
                    occupancy,
                    b_factor,
                    element,
                    charge,
                    is_hetatm,
                });
            }

            // Parse HELIX records
            if line.starts_with("HELIX") && line.len() >= 38 {
                let start_chain = line.get(19..20).unwrap_or(" ").to_string();
                let start_res: i32 = line.get(21..25)
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
                let end_chain = line.get(31..32).unwrap_or(" ").to_string();
                let end_res: i32 = line.get(33..37)
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);

                ss_records.push(SecondaryStructureRecord {
                    ss_type: "HELIX".to_string(),
                    start_chain,
                    start_res,
                    end_chain,
                    end_res,
                });
            }

            // Parse SHEET records
            if line.starts_with("SHEET") && line.len() >= 38 {
                let start_chain = line.get(21..22).unwrap_or(" ").to_string();
                let start_res: i32 = line.get(22..26)
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);
                let end_chain = line.get(32..33).unwrap_or(" ").to_string();
                let end_res: i32 = line.get(33..37)
                    .and_then(|s| s.trim().parse().ok())
                    .unwrap_or(0);

                ss_records.push(SecondaryStructureRecord {
                    ss_type: "SHEET".to_string(),
                    start_chain,
                    start_res,
                    end_chain,
                    end_res,
                });
            }
        }

        // Build residue records
        let standard_aa = [
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
        ];

        let mut residues: Vec<ResidueRecord> = residue_map
            .iter()
            .map(|(key, indices)| {
                let parts: Vec<&str> = key.split(':').collect();
                let chain_id = parts.get(0).unwrap_or(&" ").to_string();
                let res_seq: i32 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
                let i_code = parts.get(2).and_then(|s| s.chars().next()).filter(|&c| c != ' ');

                let first_atom = indices.first().map(|&i| &atoms[i]);
                let res_name = first_atom.map(|a| a.res_name.clone()).unwrap_or_default();
                let is_standard_aa = standard_aa.contains(&res_name.as_str());

                let ca_index = indices.iter().find(|&&i| atoms[i].name == "CA").copied();

                ResidueRecord {
                    chain_id,
                    res_seq,
                    res_name,
                    i_code,
                    atom_indices: indices.clone(),
                    ca_index,
                    is_standard_aa,
                    ss_type: None, // Will be filled in
                }
            })
            .collect();

        // Sort residues by chain and sequence
        residues.sort_by(|a, b| {
            a.chain_id.cmp(&b.chain_id)
                .then(a.res_seq.cmp(&b.res_seq))
        });

        // Build chain records
        let mut chain_map: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, res) in residues.iter().enumerate() {
            chain_map.entry(res.chain_id.clone()).or_default().push(idx);
        }

        let aa_to_one = |three: &str| -> char {
            match three {
                "ALA" => 'A', "ARG" => 'R', "ASN" => 'N', "ASP" => 'D', "CYS" => 'C',
                "GLN" => 'Q', "GLU" => 'E', "GLY" => 'G', "HIS" => 'H', "ILE" => 'I',
                "LEU" => 'L', "LYS" => 'K', "MET" => 'M', "PHE" => 'F', "PRO" => 'P',
                "SER" => 'S', "THR" => 'T', "TRP" => 'W', "TYR" => 'Y', "VAL" => 'V',
                _ => 'X',
            }
        };

        let chains: Vec<ChainRecord> = chain_map
            .iter()
            .map(|(chain_id, res_indices)| {
                let sequence: String = res_indices
                    .iter()
                    .filter(|&&i| residues[i].is_standard_aa)
                    .map(|&i| aa_to_one(&residues[i].res_name))
                    .collect();

                let is_protein = res_indices.iter().any(|&i| residues[i].is_standard_aa);

                ChainRecord {
                    chain_id: chain_id.clone(),
                    residue_indices: res_indices.clone(),
                    sequence,
                    is_protein,
                    n_residues: res_indices.len(),
                }
            })
            .collect();

        AtomicMetadata {
            pdb_id: pdb_id.to_string(),
            blake3_hash: blake3_hash.to_string(),
            atoms,
            residues,
            chains,
            bonds: Vec::new(), // CONECT records could be parsed if needed
            secondary_structure: ss_records,
            crystal: None, // CRYST1 could be parsed if needed
        }
    }

    /// Validate temporal integrity (apo deposited before drug)
    pub fn validate_temporal(
        provenance: &mut PdbProvenance,
        drug_discovery_date: NaiveDate,
    ) {
        provenance.validation.drug_discovery_date = Some(drug_discovery_date);

        if let Some(deposition_date) = provenance.deposition_date {
            let days_diff = (drug_discovery_date - deposition_date).num_days();
            provenance.validation.days_before_drug = Some(days_diff);

            if days_diff > 0 {
                // Structure deposited BEFORE drug discovery - VALID
                provenance.validation.temporal_valid = true;
                provenance.validation.safe_for_blind = true;
            } else {
                // Structure deposited AFTER drug discovery - POTENTIAL LEAKAGE
                provenance.validation.temporal_valid = false;
                provenance.validation.safe_for_blind = false;
                provenance.validation.warnings.push(format!(
                    "CRITICAL: Structure deposited {} days AFTER drug discovery. Potential data leakage!",
                    -days_diff
                ));
            }
        } else {
            provenance.validation.warnings.push(
                "WARNING: Could not determine deposition date. Temporal validation not possible.".to_string()
            );
        }
    }

    /// Save manifest with integrity hash
    pub fn save_manifest(&self, manifest: &mut CurationManifest, path: &Path) -> anyhow::Result<()> {
        // Compute hash of manifest (excluding the hash field itself)
        manifest.manifest_hash = None;
        let temp_json = serde_json::to_string(manifest)?;
        manifest.manifest_hash = Some(Self::compute_blake3(temp_json.as_bytes()));

        // Save
        let json = serde_json::to_string_pretty(manifest)?;
        std::fs::write(path, json)?;

        log::info!("Saved manifest to {:?} with hash {}", path, manifest.manifest_hash.as_ref().unwrap());

        Ok(())
    }
}

/// Validation target definitions with temporal data
pub fn get_validation_targets() -> Vec<TargetDefinition> {
    vec![
        // === ONCOLOGY ===
        TargetDefinition {
            name: "KRAS_G12C".to_string(),
            therapeutic_area: "Oncology".to_string(),
            drug_name: "Sotorasib".to_string(),
            // Sotorasib approval: May 2021, discovery ~2013
            drug_discovery_date: NaiveDate::from_ymd_opt(2013, 1, 1).unwrap(),
            // 3GFT: Wild-type KRAS-GDP complex, deposited 2009-02-19 - BEFORE discovery ✓
            apo_pdb: "3GFT".to_string(),
            holo_pdb: "6OIM".to_string(),
            pocket_residues: vec![60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72],
            notes: vec!["Switch-II pocket".to_string()],
        },
        TargetDefinition {
            name: "BTK".to_string(),
            therapeutic_area: "Oncology".to_string(),
            drug_name: "Ibrutinib".to_string(),
            // Ibrutinib approval: 2013, discovery ~2007
            drug_discovery_date: NaiveDate::from_ymd_opt(2007, 1, 1).unwrap(),
            apo_pdb: "1K2P".to_string(), // Deposited 2001 - BEFORE discovery ✓
            holo_pdb: "5P9J".to_string(),
            pocket_residues: vec![408, 409, 410, 411, 412, 413, 414, 474, 475, 476, 477, 478, 481],
            notes: vec!["C481 covalent site".to_string()],
        },
        TargetDefinition {
            name: "BRAF_V600E".to_string(),
            therapeutic_area: "Oncology".to_string(),
            drug_name: "Vemurafenib".to_string(),
            // Vemurafenib approval: 2011, discovery ~2006
            drug_discovery_date: NaiveDate::from_ymd_opt(2006, 1, 1).unwrap(),
            // 1UWH: BRAF kinase domain, deposited 2003-12-19 - BEFORE discovery ✓
            apo_pdb: "1UWH".to_string(),
            // 3OG7: BRAF V600E with Vemurafenib
            holo_pdb: "3OG7".to_string(),
            pocket_residues: vec![464, 466, 468, 471, 505, 508, 529, 530, 531, 532, 535, 593, 594, 595],
            notes: vec!["DFG-out/αC-helix-out pocket".to_string()],
        },

        // === METABOLIC ===
        TargetDefinition {
            name: "PTP1B".to_string(),
            therapeutic_area: "Metabolic".to_string(),
            drug_name: "Trodusquemine".to_string(),
            // Trodusquemine discovery ~2010
            drug_discovery_date: NaiveDate::from_ymd_opt(2010, 1, 1).unwrap(),
            apo_pdb: "2HNP".to_string(), // Deposited 2006 - BEFORE ✓
            holo_pdb: "1T49".to_string(),
            pocket_residues: vec![280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290],
            notes: vec!["Allosteric C-terminal site".to_string()],
        },

        // === INFECTIOUS ===
        TargetDefinition {
            name: "HIV_RT".to_string(),
            therapeutic_area: "Infectious".to_string(),
            drug_name: "Rilpivirine".to_string(),
            // Rilpivirine approval: 2011, discovery ~2004
            drug_discovery_date: NaiveDate::from_ymd_opt(2004, 1, 1).unwrap(),
            apo_pdb: "1DLO".to_string(), // Deposited 1999 - BEFORE ✓
            holo_pdb: "4G1Q".to_string(),
            pocket_residues: vec![100, 101, 103, 106, 181, 188, 190, 227, 229, 230, 318],
            notes: vec!["NNRTI allosteric pocket".to_string()],
        },
        TargetDefinition {
            name: "HCV_NS3".to_string(),
            therapeutic_area: "Infectious".to_string(),
            drug_name: "Glecaprevir".to_string(),
            // Glecaprevir approval: 2017, discovery ~2012
            drug_discovery_date: NaiveDate::from_ymd_opt(2012, 1, 1).unwrap(),
            apo_pdb: "1A1R".to_string(), // Deposited 1998 - BEFORE ✓
            holo_pdb: "4NWL".to_string(),
            pocket_residues: vec![41, 42, 43, 55, 57, 132, 135, 136, 137, 139, 155, 156, 168],
            notes: vec!["NS3/4A protease site".to_string()],
        },
    ]
}

#[derive(Debug, Clone)]
pub struct TargetDefinition {
    pub name: String,
    pub therapeutic_area: String,
    pub drug_name: String,
    pub drug_discovery_date: NaiveDate,
    pub apo_pdb: String,
    pub holo_pdb: String,
    pub pocket_residues: Vec<i32>,
    pub notes: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blake3_hash() {
        let data = b"test data for hashing";
        let hash = DataCurator::compute_blake3(data);
        assert_eq!(hash.len(), 64); // BLAKE3 produces 32 bytes = 64 hex chars
    }

    #[test]
    fn test_parse_pdb_date() {
        let date = DataCurator::parse_pdb_date("15-JAN-98");
        assert_eq!(date, NaiveDate::from_ymd_opt(1998, 1, 15));

        let date2 = DataCurator::parse_pdb_date("01-DEC-05");
        assert_eq!(date2, NaiveDate::from_ymd_opt(2005, 12, 1));
    }

    #[test]
    fn test_temporal_validation_valid() {
        let mut prov = PdbProvenance {
            pdb_id: "TEST".to_string(),
            source_url: String::new(),
            downloaded_at: Utc::now(),
            blake3_hash: String::new(),
            file_size: 0,
            deposition_date: NaiveDate::from_ymd_opt(2000, 1, 1),
            release_date: None,
            resolution: None,
            method: None,
            title: None,
            n_atoms: 0,
            n_residues: 0,
            chains: Vec::new(),
            local_path: PathBuf::new(),
            validation: ProvenanceValidation {
                temporal_valid: false,
                drug_discovery_date: None,
                days_before_drug: None,
                warnings: Vec::new(),
                safe_for_blind: false,
            },
        };

        // Drug discovered in 2010, structure from 2000 - VALID
        DataCurator::validate_temporal(
            &mut prov,
            NaiveDate::from_ymd_opt(2010, 1, 1).unwrap(),
        );

        assert!(prov.validation.temporal_valid);
        assert!(prov.validation.safe_for_blind);
    }

    #[test]
    fn test_temporal_validation_leakage() {
        let mut prov = PdbProvenance {
            pdb_id: "TEST".to_string(),
            source_url: String::new(),
            downloaded_at: Utc::now(),
            blake3_hash: String::new(),
            file_size: 0,
            deposition_date: NaiveDate::from_ymd_opt(2015, 1, 1),
            release_date: None,
            resolution: None,
            method: None,
            title: None,
            n_atoms: 0,
            n_residues: 0,
            chains: Vec::new(),
            local_path: PathBuf::new(),
            validation: ProvenanceValidation {
                temporal_valid: false,
                drug_discovery_date: None,
                days_before_drug: None,
                warnings: Vec::new(),
                safe_for_blind: false,
            },
        };

        // Drug discovered in 2010, structure from 2015 - LEAKAGE!
        DataCurator::validate_temporal(
            &mut prov,
            NaiveDate::from_ymd_opt(2010, 1, 1).unwrap(),
        );

        assert!(!prov.validation.temporal_valid);
        assert!(!prov.validation.safe_for_blind);
        assert!(!prov.validation.warnings.is_empty());
    }
}

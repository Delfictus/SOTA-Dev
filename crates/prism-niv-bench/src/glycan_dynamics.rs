use crate::structure_types::{ParamyxoStructure, Residue};
use anyhow::Result;

/// Calculate effective accessibility accounting for dynamic glycan shielding
pub fn calculate_effective_accessibility(structure: &ParamyxoStructure) -> Result<Vec<f32>> {
    // 1. Calculate base SASA (Approximation)
    let base_sasa = estimate_sasa(structure);

    // 2. Identify Glycan sites (N-X-S/T)
    let glycan_sites = find_sequons(structure);

    // 3. Apply dynamic shielding
    let mut effective_sasa = Vec::new();
    
    for (i, _res) in structure.residues.iter().enumerate() {
        let sasa = base_sasa[i];
        let shielding = calculate_shielding_factor(structure, i, &glycan_sites);
        effective_sasa.push(sasa * (1.0 - shielding));
    }

    Ok(effective_sasa)
}

fn find_sequons(structure: &ParamyxoStructure) -> Vec<usize> {
    let mut sites = Vec::new();
    let residues = &structure.residues;
    
    for i in 0..residues.len().saturating_sub(2) {
        let r1 = &residues[i];
        let r2 = &residues[i+1];
        let r3 = &residues[i+2];

        // Check for sequence continuity (simplification: just check indices or chain)
        if r1.chain_id != r2.chain_id || r2.chain_id != r3.chain_id {
            continue;
        }

        // N-X-S/T (Proline is not allowed in X or at +1 usually, but strict pattern is N-{P}-[ST]-{P})
        // Simplified: N-X-S/T
        if r1.name == "ASN" {
            if r2.name != "PRO" && (r3.name == "SER" || r3.name == "THR") {
                sites.push(i);
            }
        }
    }
    sites
}

fn calculate_shielding_factor(structure: &ParamyxoStructure, target_idx: usize, glycan_sites: &[usize]) -> f32 {
    let target_res = &structure.residues[target_idx];
    let mut max_shielding = 0.0;

    for &site_idx in glycan_sites {
        if site_idx == target_idx { continue; } // Self doesn't shield self in this context? Or does it?
        
        let glycan_root = &structure.residues[site_idx];
        
        // Distance between target CA and glycan root CA
        let dx = target_res.ca_coords.0 - glycan_root.ca_coords.0;
        let dy = target_res.ca_coords.1 - glycan_root.ca_coords.1;
        let dz = target_res.ca_coords.2 - glycan_root.ca_coords.2;
        let dist = (dx*dx + dy*dy + dz*dz).sqrt();

        // Model glycan as a probabilistic cloud
        // Glycans can extend 10-20 Angstroms.
        // Simple model: Probability of occlusion decays with distance from root
        // But also, the target must be "under" the umbrella. 
        // For isotropic cloud:
        
        if dist < 15.0 {
            // Linear decay 1.0 at 0 to 0.0 at 15.0
            let shielding = 1.0 - (dist / 15.0);
            if shielding > max_shielding {
                max_shielding = shielding;
            }
        }
    }
    
    // Cap shielding at some reasonable max (e.g., 0.9) - a residue is rarely 100% occluded by glycans alone
    max_shielding * 0.9
}

/// Estimate SASA (Solvent Accessible Surface Area)
/// Using a simplified neighbor count approximation: SASA ~ MaxSASA * (1 - neighbors/MaxNeighbors)
fn estimate_sasa(structure: &ParamyxoStructure) -> Vec<f32> {
    let mut sasa_values = Vec::new();
    let residues = &structure.residues;

    for (i, res) in residues.iter().enumerate() {
        let mut neighbors = 0;
        for (j, other) in residues.iter().enumerate() {
            if i == j { continue; }
            let dx = res.ca_coords.0 - other.ca_coords.0;
            let dy = res.ca_coords.1 - other.ca_coords.1;
            let dz = res.ca_coords.2 - other.ca_coords.2;
            let dist_sq = dx*dx + dy*dy + dz*dz;
            
            // 10A neighborhood
            if dist_sq < 100.0 {
                neighbors += 1;
            }
        }

        // Approximate SASA mapping
        // Exposed ~ 0 neighbors, Buried ~ 20+ neighbors
        // Max SASA for a residue is roughly 200 A^2 (varies by type)
        let max_sasa = get_max_sasa(&res.name);
        let exposure_factor = 1.0 - (neighbors as f32 / 20.0).min(1.0);
        sasa_values.push(max_sasa * exposure_factor);
    }
    sasa_values
}

fn get_max_sasa(res_name: &str) -> f32 {
    // Approximate max SASA values (Tripeptide Gly-X-Gly)
    match res_name {
        "ALA" => 129.0,
        "ARG" => 274.0,
        "ASN" => 195.0,
        "ASP" => 193.0,
        "CYS" => 167.0,
        "GLU" => 223.0,
        "GLN" => 225.0,
        "GLY" => 104.0,
        "HIS" => 224.0,
        "ILE" => 197.0,
        "LEU" => 201.0,
        "LYS" => 236.0,
        "MET" => 224.0,
        "PHE" => 240.0,
        "PRO" => 159.0,
        "SER" => 155.0,
        "THR" => 172.0,
        "TRP" => 285.0,
        "TYR" => 263.0,
        "VAL" => 174.0,
        _ => 180.0,
    }
}

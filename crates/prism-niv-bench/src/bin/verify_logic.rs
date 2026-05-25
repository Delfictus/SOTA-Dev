use anyhow::Result;
use prism_niv_bench::structure_types::NivBenchDataset;
use std::fs::File;
use std::io::BufReader;

#[derive(Clone, Copy)]
enum CrypticAction {
    PredictCryptic,
    PredictExposed,
}

fn compute_cryptic_reward(
    action: CrypticAction,
    is_cryptic: bool,
    is_epitope: bool,
    confidence: f32,
) -> f32 {
    match (action, is_cryptic, is_epitope) {
        (CrypticAction::PredictCryptic, true, true) => 2.0 * confidence,
        (CrypticAction::PredictCryptic, true, false) => 1.0 * confidence,
        (CrypticAction::PredictCryptic, false, _) => -0.5 * confidence,
        (CrypticAction::PredictExposed, true, _) => -1.0 * confidence,
        (CrypticAction::PredictExposed, false, _) => 0.5 * confidence,
    }
}

fn is_cryptic(dataset: &NivBenchDataset, pdb_id: &str, residue_idx: u32) -> bool {
    dataset
        .cryptic_sites
        .get(pdb_id)
        .map(|sites| {
            sites
                .iter()
                .any(|site| site.residues.iter().any(|&residue| residue == residue_idx))
        })
        .unwrap_or(false)
}

fn is_epitope(dataset: &NivBenchDataset, pdb_id: &str, residue_idx: u32) -> bool {
    dataset
        .epitopes
        .get(pdb_id)
        .map(|epitopes| {
            epitopes
                .iter()
                .any(|epitope| epitope.interface_residues.iter().any(|&residue| residue == residue_idx))
        })
        .unwrap_or(false)
}

fn main() -> Result<()> {
    println!("--- Verifying Logic: Data -> GroundTruth -> Reward ---");

    // 1. Load Data
    let path = "data/niv_bench_dataset.json";
    println!("Loading {}", path);
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let dataset: NivBenchDataset = serde_json::from_reader(reader)?;

    // 2. Define Test Cases based on Synthetic Data
    // 8XPS:
    // Epitope 1: 100, 101, 102 (is_cryptic=true)
    // Cryptic Site 1: 200, 201 (defined in cryptic_sites)

    let test_cases = vec![
        (100, "Residue 100 (Cryptic Epitope)", true, true),
        (200, "Residue 200 (Cryptic Site)", true, false), // defined in cryptic_sites but not epitope list? Wait, let's check JSON
        (999, "Residue 999 (Noise)", false, false),
    ];

    for (residue_idx, desc, expected_cryptic, expected_epitope) in test_cases {
        println!("\nTesting: {}", desc);

        let is_cryptic = is_cryptic(&dataset, "8XPS", residue_idx);
        let is_epitope = is_epitope(&dataset, "8XPS", residue_idx);

        println!(
            "  Ground Truth: Cryptic={}, Epitope={}",
            is_cryptic, is_epitope
        );

        if is_cryptic != expected_cryptic {
            println!(
                "  [FAIL] Cryptic status mismatch! Expected {}",
                expected_cryptic
            );
        }
        if is_epitope != expected_epitope {
            println!(
                "  [FAIL] Epitope status mismatch! Expected {}",
                expected_epitope
            );
        }

        // Test Reward Function
        let reward_cryptic =
            compute_cryptic_reward(CrypticAction::PredictCryptic, is_cryptic, is_epitope, 1.0);
        let reward_exposed =
            compute_cryptic_reward(CrypticAction::PredictExposed, is_cryptic, is_epitope, 1.0);

        println!("  Reward (PredictCryptic): {:.2}", reward_cryptic);
        println!("  Reward (PredictExposed): {:.2}", reward_exposed);

        if is_cryptic && reward_cryptic < 1.0 {
            println!("  [FAIL] Reward logic broken for Cryptic site!");
        }
        if is_cryptic && reward_exposed > -0.5 {
            println!("  [FAIL] Penalty logic broken for Missed Cryptic site!");
        }
    }

    Ok(())
}

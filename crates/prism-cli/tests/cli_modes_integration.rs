use std::fs;
use std::path::Path;
/// Integration tests for PRISM CLI modes
///
/// This test suite verifies that all CLI modes (coloring, biomolecular, materials, mec-only)
/// can be executed successfully and produce valid telemetry output.
///
/// Tests:
/// - test_mec_only_mode: Runs MEC-only diagnostics and validates telemetry
/// - test_coloring_mode: Runs graph coloring with GNN and validates telemetry
/// - test_biomolecular_mode: Runs protein structure prediction and validates telemetry
/// - test_materials_mode: Runs materials discovery and validates telemetry
use std::process::Command;

#[test]
fn test_mec_only_mode() {
    // Clean up previous telemetry
    let telemetry_path = "telemetry_mec_only_test.jsonl";
    let _ = fs::remove_file(telemetry_path);

    // Run CLI in mec-only mode
    let output = Command::new("./target/release/prism-cli")
        .arg("--mode")
        .arg("mec-only")
        .output()
        .expect("Failed to execute prism-cli");

    // Check that command succeeded
    assert!(
        output.status.success(),
        "mec-only mode failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify telemetry file was created
    assert!(
        Path::new("telemetry_mec_only.jsonl").exists(),
        "telemetry_mec_only.jsonl not created"
    );

    // Read and validate telemetry
    let telemetry_content =
        fs::read_to_string("telemetry_mec_only.jsonl").expect("Failed to read telemetry file");

    let last_line = telemetry_content.lines().last().unwrap();
    let telemetry: serde_json::Value =
        serde_json::from_str(last_line).expect("Failed to parse telemetry JSON");

    // Validate required fields
    assert_eq!(telemetry["mode"], "mec-only");
    assert!(telemetry["num_molecules"].as_u64().unwrap() > 0);
    assert!(telemetry["results"]["free_energy"].as_f64().is_some());
    assert!(telemetry["results"]["entropy"].as_f64().is_some());
    assert!(telemetry["config"]["time_step"].as_f64().is_some());
    assert!(telemetry["config"]["iterations"].as_u64().is_some());
}

#[test]
fn test_coloring_mode() {
    // Skip if DSJC125.1.col doesn't exist
    let input_path = "benchmarks/dimacs/DSJC125.1.col";
    if !Path::new(input_path).exists() {
        eprintln!("Skipping test_coloring_mode: {} not found", input_path);
        return;
    }

    let config_path = "configs/gnn_test.toml";
    if !Path::new(config_path).exists() {
        eprintln!("Skipping test_coloring_mode: {} not found", config_path);
        return;
    }

    // Clean up previous telemetry
    let telemetry_path = "telemetry_gnn.jsonl";
    let initial_lines = if Path::new(telemetry_path).exists() {
        fs::read_to_string(telemetry_path).unwrap().lines().count()
    } else {
        0
    };

    // Run CLI in coloring mode with GNN
    let output = Command::new("./target/release/prism-cli")
        .arg("--mode")
        .arg("coloring")
        .arg("--input")
        .arg(input_path)
        .arg("--config")
        .arg(config_path)
        .arg("--attempts")
        .arg("1")
        .output()
        .expect("Failed to execute prism-cli");

    // Check that command succeeded
    assert!(
        output.status.success(),
        "coloring mode failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify telemetry file was created/appended
    assert!(
        Path::new(telemetry_path).exists(),
        "telemetry_gnn.jsonl not created"
    );

    // Read and validate telemetry
    let telemetry_content =
        fs::read_to_string(telemetry_path).expect("Failed to read telemetry file");

    let lines: Vec<&str> = telemetry_content.lines().collect();
    assert!(lines.len() > initial_lines, "No new telemetry entry added");

    let last_line = lines.last().unwrap();
    let telemetry: serde_json::Value =
        serde_json::from_str(last_line).expect("Failed to parse telemetry JSON");

    // Validate required fields
    assert_eq!(telemetry["mode"], "gnn");
    assert!(telemetry["graph_vertices"].as_u64().unwrap() > 0);
    assert!(telemetry["graph_edges"].as_u64().unwrap() > 0);
    assert!(telemetry["predicted_chromatic"].as_u64().unwrap() > 0);
    assert!(telemetry["actual_chromatic"].as_u64().is_some());
    assert!(telemetry["confidence"].as_f64().is_some());
    assert!(telemetry["manifold_features"]["dimension"]
        .as_f64()
        .is_some());
    assert!(telemetry["manifold_features"]["curvature"]
        .as_f64()
        .is_some());
}

#[test]
fn test_biomolecular_mode() {
    // Skip if input files don't exist
    let sequence_path = "benchmarks/biomolecular/nipah_glycoprotein.fasta";
    if !Path::new(sequence_path).exists() {
        eprintln!(
            "Skipping test_biomolecular_mode: {} not found",
            sequence_path
        );
        return;
    }

    // Clean up previous telemetry
    let telemetry_path = "telemetry_biomolecular.jsonl";
    let initial_lines = if Path::new(telemetry_path).exists() {
        fs::read_to_string(telemetry_path).unwrap().lines().count()
    } else {
        0
    };

    // Run CLI in biomolecular mode
    let output = Command::new("./target/release/prism-cli")
        .arg("--mode")
        .arg("biomolecular")
        .arg("--sequence")
        .arg(sequence_path)
        .arg("--ligand")
        .arg("CC(C)Oc1ccc(C(=O)N[C@H](C(=O)N[C@@H](Cc2ccccc2)C(=O)O)C(C)C)cc1")
        .output()
        .expect("Failed to execute prism-cli");

    // Check that command succeeded
    assert!(
        output.status.success(),
        "biomolecular mode failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify telemetry file was created/appended
    assert!(
        Path::new(telemetry_path).exists(),
        "telemetry_biomolecular.jsonl not created"
    );

    // Read and validate telemetry
    let telemetry_content =
        fs::read_to_string(telemetry_path).expect("Failed to read telemetry file");

    let lines: Vec<&str> = telemetry_content.lines().collect();
    assert!(lines.len() > initial_lines, "No new telemetry entry added");

    let last_line = lines.last().unwrap();
    let telemetry: serde_json::Value =
        serde_json::from_str(last_line).expect("Failed to parse telemetry JSON");

    // Validate required fields
    assert_eq!(telemetry["mode"], "biomolecular");
    assert!(telemetry["results"]["residues"].as_u64().unwrap() > 0);
    assert!(telemetry["results"]["binding_sites"].as_u64().unwrap() > 0);
    assert!(telemetry["results"]["best_affinity"].as_f64().is_some());
    assert!(telemetry["results"]["confidence"].as_f64().is_some());
    assert!(telemetry["results"]["rmsd"].as_f64().is_some());
    assert_eq!(telemetry["sequence_path"], sequence_path);
}

#[test]
fn test_materials_mode() {
    // Clean up previous telemetry
    let telemetry_path = "telemetry_materials.jsonl";
    let initial_lines = if Path::new(telemetry_path).exists() {
        fs::read_to_string(telemetry_path).unwrap().lines().count()
    } else {
        0
    };

    // Run CLI in materials mode
    let output = Command::new("./target/release/prism-cli")
        .arg("--mode")
        .arg("materials")
        .output()
        .expect("Failed to execute prism-cli");

    // Check that command succeeded
    assert!(
        output.status.success(),
        "materials mode failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify telemetry file was created/appended
    assert!(
        Path::new(telemetry_path).exists(),
        "telemetry_materials.jsonl not created"
    );

    // Read and validate telemetry
    let telemetry_content =
        fs::read_to_string(telemetry_path).expect("Failed to read telemetry file");

    let lines: Vec<&str> = telemetry_content.lines().collect();
    assert!(lines.len() > initial_lines, "No new telemetry entry added");

    let last_line = lines.last().unwrap();
    let telemetry: serde_json::Value =
        serde_json::from_str(last_line).expect("Failed to parse telemetry JSON");

    // Validate required fields
    assert_eq!(telemetry["mode"], "materials");
    assert!(telemetry["results"]["num_candidates"].as_u64().unwrap() > 0);
    assert!(telemetry["results"]["best_composition"].as_str().is_some());
    assert!(telemetry["results"]["best_confidence"].as_f64().is_some());
    assert!(telemetry["results"]["best_band_gap"].as_f64().is_some());
    assert!(telemetry["results"]["best_formation_energy"]
        .as_f64()
        .is_some());
    assert!(telemetry["results"]["best_stability"].as_f64().is_some());
    assert!(telemetry["target"]["band_gap_range"].as_array().is_some());
}

#[test]
fn test_invalid_mode() {
    // Run CLI with invalid mode
    let output = Command::new("./target/release/prism-cli")
        .arg("--mode")
        .arg("invalid_mode")
        .output()
        .expect("Failed to execute prism-cli");

    // Check that command failed
    assert!(
        !output.status.success(),
        "Expected failure for invalid mode"
    );

    // Check error message
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("Unknown mode") || stderr.contains("invalid_mode"),
        "Expected error message about unknown mode"
    );
}

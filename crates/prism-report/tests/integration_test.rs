//! Integration tests for prism-report
//!
//! Verifies the output contract is satisfied using FinalizeStage with events.jsonl.

use prism_report::config::{ReportConfig, OutputFormats};
use prism_report::event_cloud::{AblationPhase, EventWriter, PocketEvent, TempPhase};
use prism_report::finalize::FinalizeStage;
use std::path::PathBuf;
use tempfile::TempDir;

/// Create test events for integration tests
///
/// Events are created with dense frame coverage within each (phase, replicate_id) run
/// to ensure persistence thresholds are met. frame_idx is the local step within each run.
fn create_test_events() -> Vec<PocketEvent> {
    let mut events = Vec::new();

    // Create events at a consistent location (single cluster) with high persistence
    // across multiple (phase, replicate_id) runs
    let base_xyz = [10.0, 15.0, 20.0];

    // Create dense events for each (phase, replicate_id) combination
    for phase in [AblationPhase::Baseline, AblationPhase::CryoOnly, AblationPhase::CryoUv] {
        for replicate_id in 0..3 {
            // Create 5 consecutive frames (0-4) for each run = 100% persistence
            for frame_idx in 0..5 {
                let temp_phase = match phase {
                    AblationPhase::Baseline => TempPhase::Warm,
                    _ => match frame_idx % 3 {
                        0 => TempPhase::Cold,
                        1 => TempPhase::Ramp,
                        _ => TempPhase::Warm,
                    },
                };

                // Small random offset to keep events in same cluster
                let offset = (frame_idx as f32 + replicate_id as f32) * 0.1;
                events.push(PocketEvent {
                    center_xyz: [
                        base_xyz[0] + offset,
                        base_xyz[1] + offset * 0.5,
                        base_xyz[2] + offset * 0.3,
                    ],
                    volume_a3: 150.0 + frame_idx as f32 * 10.0,
                    spike_count: 3 + frame_idx,
                    phase,
                    temp_phase,
                    replicate_id,
                    frame_idx, // Local step within this (phase, replicate_id) run
                    residues: vec![1, 2, 3],
                    confidence: 0.7 + (frame_idx as f32) * 0.05,
                    wavelength_nm: if phase == AblationPhase::CryoUv {
                        Some(280.0)
                    } else {
                        None
                    },
                });
            }
        }
    }

    // Add a second distinct cluster at a different location
    let base_xyz2 = [50.0, 50.0, 50.0];
    for phase in [AblationPhase::CryoOnly, AblationPhase::CryoUv] {
        for replicate_id in 0..2 {
            for frame_idx in 0..3 {
                let temp_phase = match frame_idx % 3 {
                    0 => TempPhase::Cold,
                    1 => TempPhase::Ramp,
                    _ => TempPhase::Warm,
                };

                let offset = (frame_idx as f32 + replicate_id as f32) * 0.15;
                events.push(PocketEvent {
                    center_xyz: [
                        base_xyz2[0] + offset,
                        base_xyz2[1] + offset * 0.5,
                        base_xyz2[2] + offset * 0.3,
                    ],
                    volume_a3: 200.0 + frame_idx as f32 * 15.0,
                    spike_count: 5 + frame_idx,
                    phase,
                    temp_phase,
                    replicate_id,
                    frame_idx,
                    residues: vec![10, 11, 12],
                    confidence: 0.8,
                    wavelength_nm: if phase == AblationPhase::CryoUv {
                        Some(280.0)
                    } else {
                        None
                    },
                });
            }
        }
    }

    events
}

/// Write events to a JSONL file
fn write_events_to_file(events: &[PocketEvent], path: &std::path::Path) -> anyhow::Result<()> {
    let mut writer = EventWriter::new(path)?;
    for event in events {
        writer.write_event(event)?;
    }
    writer.flush()?;
    Ok(())
}

/// Create mock topology JSON that overlaps with test event coordinates
fn create_mock_topology() -> &'static str {
    r#"{
        "n_atoms": 20,
        "positions": [
            10.0, 15.0, 20.0,
            11.0, 16.0, 21.0,
            12.0, 17.0, 22.0,
            13.0, 18.0, 23.0,
            14.0, 19.0, 24.0,
            50.0, 50.0, 50.0,
            51.0, 51.0, 51.0,
            52.0, 52.0, 52.0,
            53.0, 53.0, 53.0,
            54.0, 54.0, 54.0,
            10.5, 15.5, 20.5,
            11.5, 16.5, 21.5,
            12.5, 17.5, 22.5,
            13.5, 18.5, 23.5,
            14.5, 19.5, 24.5,
            50.5, 50.5, 50.5,
            51.5, 51.5, 51.5,
            52.5, 52.5, 52.5,
            53.5, 53.5, 53.5,
            54.5, 54.5, 54.5
        ],
        "residue_ids": [1, 1, 2, 2, 3, 10, 10, 11, 11, 12, 1, 1, 2, 2, 3, 10, 10, 11, 11, 12],
        "residue_names": ["ALA", "ALA", "LEU", "LEU", "VAL", "PHE", "PHE", "ILE", "ILE", "MET", "ALA", "ALA", "LEU", "LEU", "VAL", "PHE", "PHE", "ILE", "ILE", "MET"],
        "chain_ids": ["A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A", "A"]
    }"#
}

/// Test that FinalizeStage produces all required outputs
#[test]
fn test_output_contract() {
    let tmp = TempDir::new().expect("Failed to create temp dir");
    let output_dir = tmp.path().join("results");
    let events_path = tmp.path().join("events.jsonl");
    let topology_path = tmp.path().join("topology.json");

    // Create and write test events
    let events = create_test_events();
    write_events_to_file(&events, &events_path).unwrap();

    // Create mock topology that overlaps with event coordinates
    std::fs::write(&topology_path, create_mock_topology()).unwrap();

    // Create minimal test PDB (for visualization only)
    let test_pdb = tmp.path().join("test.pdb");
    std::fs::write(
        &test_pdb,
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n",
    ).unwrap();

    let config = ReportConfig {
        input_pdb: test_pdb,
        output_dir: output_dir.clone(),
        replicates: 3,
        wavelengths: vec![258.0, 274.0, 280.0],
        output_formats: OutputFormats {
            html: true,
            pdf: false, // Skip PDF in tests (requires external tool)
            json: true,
            csv: true,
            pymol: false, // Skip in tests
            chimerax: false, // Skip in tests
            figures: true,
            mrc_volumes: false,
        },
        ..Default::default()
    };

    let stage = FinalizeStage::new_with_topology(config, events_path, topology_path, 42).expect("Failed to create FinalizeStage");
    let result = stage.run().expect("FinalizeStage failed");

    // Verify required files exist
    assert!(output_dir.join("report.html").exists(), "report.html missing");
    assert!(output_dir.join("summary.json").exists(), "summary.json missing");
    assert!(output_dir.join("correlation.csv").exists(), "correlation.csv missing");

    // Verify required directories exist
    assert!(output_dir.join("sites").is_dir(), "sites/ missing");
    assert!(output_dir.join("volumes").is_dir(), "volumes/ missing");
    assert!(output_dir.join("trajectories").is_dir(), "trajectories/ missing");
    assert!(output_dir.join("provenance").is_dir(), "provenance/ missing");

    // Verify provenance files
    assert!(output_dir.join("provenance/manifest.json").exists(), "manifest.json missing");
    assert!(output_dir.join("provenance/versions.json").exists(), "versions.json missing");
    assert!(output_dir.join("provenance/seeds.json").exists(), "seeds.json missing");
    assert!(output_dir.join("provenance/params.json").exists(), "params.json missing");

    // Verify at least one site was detected
    assert!(result.n_sites > 0, "No sites detected");

    // Verify site directory structure
    let site_dirs: Vec<_> = std::fs::read_dir(output_dir.join("sites"))
        .expect("Failed to read sites dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();

    assert!(!site_dirs.is_empty(), "No site directories created");

    for site_dir in site_dirs {
        let path = site_dir.path();
        let site_id = path.file_name().unwrap().to_str().unwrap();

        assert!(path.join("site.pdb").exists(), "{}/site.pdb missing", site_id);
        assert!(path.join("site.mol2").exists(), "{}/site.mol2 missing", site_id);
        assert!(path.join("residues.txt").exists(), "{}/residues.txt missing", site_id);
        assert!(path.join("correlation.json").exists(), "{}/correlation.json missing", site_id);
        assert!(path.join("figures").is_dir(), "{}/figures/ missing", site_id);
    }

    println!("Output contract verified successfully!");
    println!("Sites detected: {}", result.n_sites);
    println!("Druggable sites: {}", result.n_druggable);
}

/// Test summary.json structure
#[test]
fn test_summary_json_structure() {
    let tmp = TempDir::new().expect("Failed to create temp dir");
    let output_dir = tmp.path().join("results");
    let events_path = tmp.path().join("events.jsonl");
    let topology_path = tmp.path().join("topology.json");

    // Create and write test events
    let events = create_test_events();
    write_events_to_file(&events, &events_path).unwrap();

    // Create mock topology
    std::fs::write(&topology_path, create_mock_topology()).unwrap();

    let test_pdb = tmp.path().join("test.pdb");
    std::fs::write(
        &test_pdb,
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n",
    ).unwrap();

    let config = ReportConfig {
        input_pdb: test_pdb,
        output_dir: output_dir.clone(),
        output_formats: OutputFormats {
            html: false,
            pdf: false,
            json: true,
            csv: false,
            pymol: false,
            chimerax: false,
            figures: false,
            mrc_volumes: false,
        },
        ..Default::default()
    };

    let stage = FinalizeStage::new_with_topology(config, events_path, topology_path, 42).expect("Failed to create FinalizeStage");
    stage.run().expect("FinalizeStage failed");

    // Read and parse summary.json
    let summary_content = std::fs::read_to_string(output_dir.join("summary.json"))
        .expect("Failed to read summary.json");
    let summary: serde_json::Value = serde_json::from_str(&summary_content)
        .expect("Failed to parse summary.json");

    // Verify required fields
    assert!(summary.get("version").is_some(), "Missing version field");
    assert!(summary.get("timestamp").is_some(), "Missing timestamp field");
    assert!(summary.get("input").is_some(), "Missing input field");
    assert!(summary.get("sites").is_some(), "Missing sites field");
    assert!(summary.get("ablation").is_some(), "Missing ablation field");
    assert!(summary.get("ranking_weights").is_some(), "Missing ranking_weights field");
    assert!(summary.get("statistics").is_some(), "Missing statistics field");

    // Verify ablation structure
    let ablation = summary.get("ablation").unwrap();
    assert!(ablation.get("baseline_spikes").is_some(), "Missing baseline_spikes");
    assert!(ablation.get("cryo_only_spikes").is_some(), "Missing cryo_only_spikes");
    assert!(ablation.get("cryo_uv_spikes").is_some(), "Missing cryo_uv_spikes");
    assert!(ablation.get("cryo_contrast_significant").is_some(), "Missing cryo_contrast_significant");
    assert!(ablation.get("uv_response_significant").is_some(), "Missing uv_response_significant");
    assert!(ablation.get("interpretation").is_some(), "Missing interpretation");

    println!("summary.json structure verified!");
}

/// Test ablation is mandatory
#[test]
fn test_ablation_mandatory() {
    use prism_report::config::AblationConfig;

    // Default config should be valid (all modes enabled)
    let default_ablation = AblationConfig::default();
    assert!(default_ablation.validate().is_ok(), "Default ablation should be valid");

    // Disabled mode should fail validation
    let invalid_ablation = AblationConfig {
        run_baseline: false, // Invalid!
        run_cryo_only: true,
        run_cryo_uv: true,
    };
    assert!(invalid_ablation.validate().is_err(), "Should reject disabled ablation mode");
}

/// Test correlation CSV format
#[test]
fn test_correlation_csv_format() {
    let tmp = TempDir::new().expect("Failed to create temp dir");
    let output_dir = tmp.path().join("results");
    let events_path = tmp.path().join("events.jsonl");
    let topology_path = tmp.path().join("topology.json");

    // Create and write test events
    let events = create_test_events();
    write_events_to_file(&events, &events_path).unwrap();

    // Create mock topology
    std::fs::write(&topology_path, create_mock_topology()).unwrap();

    let test_pdb = tmp.path().join("test.pdb");
    std::fs::write(
        &test_pdb,
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n",
    ).unwrap();

    let config = ReportConfig {
        input_pdb: test_pdb,
        output_dir: output_dir.clone(),
        output_formats: OutputFormats {
            html: false,
            pdf: false,
            json: false,
            csv: true,
            pymol: false,
            chimerax: false,
            figures: false,
            mrc_volumes: false,
        },
        ..Default::default()
    };

    let stage = FinalizeStage::new_with_topology(config, events_path, topology_path, 42).expect("Failed to create FinalizeStage");
    stage.run().expect("FinalizeStage failed");

    // Read and verify CSV
    let csv_content = std::fs::read_to_string(output_dir.join("correlation.csv"))
        .expect("Failed to read correlation.csv");

    // Check header
    let lines: Vec<&str> = csv_content.lines().collect();
    assert!(!lines.is_empty(), "CSV should have header");

    let header = lines[0];
    assert!(header.contains("site_id"), "Missing site_id column");
    assert!(header.contains("rank"), "Missing rank column");

    println!("correlation.csv format verified!");
}

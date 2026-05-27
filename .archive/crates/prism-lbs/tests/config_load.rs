use prism_lbs::LbsConfig;
use std::path::PathBuf;

#[test]
fn loads_default_config_file() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("configs/default.toml");
    let config = LbsConfig::from_file(&path).expect("load default config");

    assert!((config.graph.distance_threshold - 5.0).abs() < f64::EPSILON);
    assert!(config.geometry.use_convex_hull_volume);
    assert!(config.geometry.use_alpha_shape_volume);
    assert!(config.geometry.use_flood_fill_cavity);
    assert!(config.geometry.use_boundary_enclosure);
    assert!((config.phase1.temperature - 1.0).abs() < f64::EPSILON);
    assert!((config.phase2.conflict_penalty - 1.0).abs() < f64::EPSILON);
    assert_eq!(config.phase4.top_centrality, 10);
    assert_eq!(config.phase6.max_features, 32);
    assert!((config.phase1.learning_rate - 0.12).abs() < f64::EPSILON);
    assert_eq!(config.phase1.max_pockets, 12);
    assert_eq!(config.output.formats.len(), 2);
    assert_eq!(config.top_n, 10);
}

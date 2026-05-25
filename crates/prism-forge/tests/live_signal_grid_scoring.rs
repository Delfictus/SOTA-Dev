use prism_forge::scoring::{
    score_molecule, LoadedSignalGrid, VoxelField, W_CLASH_POCKET, W_COMPLEMENT,
};
use std::collections::{HashMap, HashSet};

fn voxel(
    complement: f64,
    clash: f64,
    stable: bool,
    activated: bool,
    consensus_bonus: f64,
) -> VoxelField {
    VoxelField {
        complement,
        clash,
        raw_clash: clash,
        cryptic_bonus: 0.0,
        stable_occupied: stable,
        thermally_destabilized: false,
        thermally_activated: activated,
        thermally_released: false,
        coherence_score: 1.0,
        coherence_factor: 1.0,
        coherence_missing: false,
        primary_residue_idx: None,
        cold_mean: clash,
        warm_mean: complement,
        delta: complement - clash,
        consensus_complement_bonus: consensus_bonus,
        on_activation_pathway: false,
    }
}

#[test]
fn live_signal_grid_map_to_voxel_uses_xyz_order() {
    let grid = LoadedSignalGrid {
        field: HashMap::new(),
        origin: [0.0, 0.0, 0.0],
        spacing: 1.0,
        dims: [10, 10, 10],
        condition_id: "test".to_owned(),
    };

    assert_eq!(grid.map_to_voxel(2.2, 3.1, 4.9), Some(432));
    assert_eq!(grid.map_to_voxel(-0.1, 0.0, 0.0), None);
}

#[test]
fn live_signal_grid_scores_complement_clash_lock_and_shear() {
    let mut field = HashMap::new();
    field.insert(0, voxel(2.0, 0.0, false, true, 0.5));
    field.insert(1, voxel(0.0, 1.0, true, false, 0.0));
    let grid = LoadedSignalGrid {
        field,
        origin: [0.0, 0.0, 0.0],
        spacing: 1.0,
        dims: [10, 10, 10],
        condition_id: "test".to_owned(),
    };
    let lock_mask = HashSet::from([1_u64]);
    let shear = HashMap::from([(0_u64, 0.25), (1_u64, 0.75)]);

    let score = score_molecule(&[(0.1, 0.1, 0.1), (1.1, 0.1, 0.1)], &grid, &lock_mask, &shear);

    assert_eq!(score.pi_complement, 2.0);
    assert_eq!(score.pi_clash_pocket, 0.0);
    assert_eq!(score.pi_clash_lock, 1.0);
    assert_eq!(score.lock_atom_count, 1);
    assert_eq!(score.consensus_bonus, 0.5);
    assert_eq!(score.sigma_shear, 1.0);
    assert!(score.reward < W_COMPLEMENT * 2.0 + 1.0 + 0.5);
    assert!(score.reward > -W_CLASH_POCKET);
}

use prism_lbs::pocket::geometry::{
    alpha_shape_volume, bounding_box_volume, convex_hull_volume, enclosure_ratio,
    flood_fill_cavity_volume, voxel_volume,
};
use prism_lbs::structure::ProteinStructure;

fn cube_pdb() -> String {
    [
        "ATOM      1  C1  ALA A   1       0.000   0.000   0.000  1.00 10.00           C",
        "ATOM      2  C2  ALA A   1       2.000   0.000   0.000  1.00 10.00           C",
        "ATOM      3  C3  ALA A   1       0.000   2.000   0.000  1.00 10.00           C",
        "ATOM      4  C4  ALA A   1       0.000   0.000   2.000  1.00 10.00           C",
        "END",
    ]
    .join("\n")
}

#[test]
fn volume_and_enclosure_estimates() {
    let pdb = cube_pdb();
    let mut structure = ProteinStructure::from_pdb_str(&pdb).expect("parse pdb");
    for atom in &mut structure.atoms {
        atom.is_surface = true;
        atom.sasa = 5.0;
    }
    let indices: Vec<usize> = (0..structure.atoms.len()).collect();

    let bbox_vol = bounding_box_volume(&structure, &indices);
    assert!(
        bbox_vol > 7.0 && bbox_vol < 10.0,
        "bbox volume={}",
        bbox_vol
    );

    let voxel_vol = voxel_volume(&structure, &indices, Some(0.5), Some(0.0));
    assert!(voxel_vol > 7.0, "voxel volume={}", voxel_vol);

    let alpha_vol = alpha_shape_volume(&structure, &indices, 0.4, 0.3);
    assert!(alpha_vol > 0.5, "alpha volume={}", alpha_vol);

    let hull_vol = convex_hull_volume(&structure, &indices, 1e-6);
    assert!(hull_vol > 1.0, "hull volume={}", hull_vol);

    let cavity_vol = flood_fill_cavity_volume(&structure, &indices, 0.5, 0.0);
    assert!(cavity_vol >= 0.0, "cavity volume={}", cavity_vol);

    let enclosure = enclosure_ratio(&structure, &indices);
    assert!(
        enclosure < 1.0 && enclosure >= 0.0,
        "enclosure={}",
        enclosure
    );
}

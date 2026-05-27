use prism_lbs::{
    graph::ProteinGraphBuilder, phases::SurfaceReservoirPhase, structure::ProteinStructure,
};

fn simple_pdb() -> String {
    let mut lines = Vec::new();
    lines.push(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00 10.00           N"
            .to_string(),
    );
    lines.push(
        "ATOM      2  CA  ALA A   1       1.500   0.000   0.000  1.00 10.00           C"
            .to_string(),
    );
    lines.push(
        "ATOM      3  C   ALA A   1       1.500   1.500   0.000  1.00 10.00           C"
            .to_string(),
    );
    lines.push(
        "ATOM      4  O   ALA A   1       0.000   1.500   0.000  1.00 10.00           O"
            .to_string(),
    );
    lines.push(
        "ATOM      5  CB  ALA A   1       1.500  -0.750  -1.200  1.00 10.00           C"
            .to_string(),
    );
    lines.push(
        "ATOM      6  N   ALA A   2       3.000   0.000   0.000  1.00 10.00           N"
            .to_string(),
    );
    lines.push("END".to_string());
    lines.join("\n")
}

#[test]
fn pocket_stats_smoke() {
    let pdb = simple_pdb();
    let mut structure = ProteinStructure::from_pdb_str(&pdb).expect("parse pdb");
    // Mark all atoms as surface to allow selection
    for atom in &mut structure.atoms {
        atom.is_surface = true;
        atom.sasa = 10.0;
        atom.depth = 2.0;
        atom.hydrophobicity = 1.0;
    }
    structure.refresh_residue_properties();

    let graph_builder = ProteinGraphBuilder::new(Default::default());
    let graph = graph_builder.build(&structure).expect("graph");

    let reservoir = SurfaceReservoirPhase::new(Default::default()).execute(&graph);
    assert_eq!(reservoir.activation_state.len(), graph.adjacency.len());
}

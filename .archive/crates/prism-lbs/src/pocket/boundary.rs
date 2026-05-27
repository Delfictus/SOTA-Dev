//! Pocket boundary helper functions

use crate::graph::ProteinGraph;

/// Identify boundary vertices where neighbors have different coloring
pub fn boundary_vertices(coloring: &[usize], graph: &ProteinGraph) -> Vec<usize> {
    coloring
        .iter()
        .enumerate()
        .filter(|(i, &c)| {
            graph.adjacency[*i]
                .iter()
                .any(|&j| coloring.get(j).map_or(false, |col| *col != c))
        })
        .map(|(i, _)| i)
        .collect()
}
